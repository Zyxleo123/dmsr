#!/usr/bin/env python
"""The last offline check: splice a tile for real and re-run the halo finder.

Every other proxy metric is measured against *labels*, which are themselves
per-tile decompositions of a whole-box catalog. This one is measured against a
whole-box catalog produced from a field in which the tile really was swapped.
The difference between the two is the cross-tile interaction that per-tile
credit assumes away, and it is the only thing here that can catch a proxy which
ranks tile summaries perfectly and still predicts the wrong sign for the change
an actor would actually make.

Two stages:

``select`` (CPU, seconds)
    Choose twelve ``(box, tile, donor)`` splices from the held-out boxes and
    write the plan. Stratified, and spread over at least three boxes.
``run`` (CPU, hours -- one array task per splice)
    Assemble ``splice_tiles(frozen_field, donor_field, [tile])``, run Rockstar
    on the complete periodic box, and append the measured ``dR`` next to the
    predicted one.

Why the strata
--------------
Twelve splices chosen as "the twelve the proxy is most confident are good" would
measure the proxy exactly where it is strongest and nowhere else -- a 100% pass
there is consistent with it being wrong about everything it is unsure of, which
is most of what an actor will hand it. So the plan predeclares three strata:

``predicted_positive``
    The rows the proxy would select. This is the decision it is used for.
``high_uncertainty``
    The rows the ensemble disagrees about. These are where the actor's
    ``mean - beta*std`` bound is doing its work, and if the sign is wrong here
    the bound is not protecting anything.
``random``
    A control drawn without reference to the proxy at all. Without it, a
    selected-only pass rate cannot be compared to anything.

Missing results never count as a pass: :func:`gate_catalog_proxy.splice_verification`
reads what has landed and reports the criterion unmet if it is short.

    python scripts/reward/splice_verify.py --stage select --arm a
    python scripts/reward/splice_verify.py --stage run --arm a --index 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from _proxy_data import (  # noqa: E402
    ARMS, as_arrays, build_row_context, ensemble_delta, load_rows,
    true_delta_rewards,
)
from _sr2_direct import (  # noqa: E402
    actor_config_of, add_direct_args, append_jsonl, banner, bins_of, boxes_of,
    candidate_tag, dataset_of, direct_root, load_direct_config,
    load_reward_models, run_dir, tile_grid_of, write_json_atomic,
)

from cosmo_sr.eval.rockstar import run_rockstar_on_field  # noqa: E402
from cosmo_sr.reward.catalog_proxy import ProxyEnsemble  # noqa: E402
from cosmo_sr.reward.tiles import direct_full_box_stats, splice_tiles  # noqa: E402
from cosmo_sr.reward.torch_reward import TorchSummary  # noqa: E402

STRATA = ("predicted_positive", "high_uncertainty", "random")


def candidate_dir(box: str, tag: str) -> Path:
    return direct_root("candidates", f"{box}__{tag}")


def plan_path(run: Path, arm: str) -> Path:
    return run / f"splice_plan_{arm}.json"


def results_path(run: Path, arm: str) -> Path:
    return run / f"splice_verification_{arm}.jsonl"


def _box_summary(counts: Dict[str, List[float]], volume: float) -> TorchSummary:
    def col(k):
        return torch.as_tensor(np.asarray(counts[k], dtype=np.float64).reshape(1, -1))
    return TorchSummary(n_sub=col("n_sub"), n_host=col("n_host"),
                        occ_numerator=col("occ_numerator"),
                        volume_mpc3=torch.tensor([float(volume)], dtype=torch.float64))


# --------------------------------------------------------------------------- #
# select
# --------------------------------------------------------------------------- #
def _pick_spread_over_boxes(order: List[int], boxes: np.ndarray, k: int,
                            taken: set) -> List[int]:
    """Take ``k`` rows in preference order, round-robin over boxes.

    Straight top-k concentrates in whichever box happens to hold the extreme
    predictions, and the criterion this feeds requires at least three boxes.
    Round-robin gives the same ranking within a box while forcing the spread.
    """
    by_box: Dict[str, List[int]] = {}
    for i in order:
        if i in taken:
            continue
        by_box.setdefault(str(boxes[i]), []).append(i)
    picked: List[int] = []
    while len(picked) < k and any(by_box.values()):
        for b in sorted(by_box):
            if len(picked) >= k:
                break
            if by_box[b]:
                picked.append(by_box[b].pop(0))
    return picked


def stage_select(cfg, args) -> int:
    gate = dict(cfg.get("proxy_gate", {}))
    acfg = actor_config_of(cfg)
    _, reward_t = load_reward_models(cfg)
    arm = args.arm
    run = run_dir(args.run_name, create=True)
    proxy_dir = run / f"proxy_{arm}"
    if not proxy_dir.is_dir():
        print(f">>> MISSING INPUT: {proxy_dir}")
        print(">>> produced by: scripts/reward/train_catalog_proxy.py")
        return 0
    ens = ProxyEnsemble.load(proxy_dir).freeze()

    table = direct_root("proxy_data") / "rows.jsonl"
    gate_boxes = set(boxes_of(cfg, "proxy_gate"))
    rows_all = load_rows(table, require_complete=not args.allow_incomplete)
    rows = [r for r in rows_all if r["box"] in gate_boxes]
    if not rows:
        print(f">>> GATE FAILED: no held-out rows from {sorted(gate_boxes)}")
        return 0

    arrays = as_arrays(rows, arm)
    ctx = build_row_context(rows)
    w_joint, w_occ = float(acfg.w_joint_reward), float(acfg.w_occ_reward)
    true = true_delta_rewards(ctx, reward_t, w_joint=w_joint, w_occ=w_occ)
    mean, std = ensemble_delta(
        ens.members, torch.as_tensor(arrays["features"], dtype=torch.float64),
        ctx, reward_t, w_joint=w_joint, w_occ=w_occ)

    # The base of every splice is the box's own frozen candidate at the base
    # seed -- the same field the labels' dR is measured against, so a measured
    # change here means the same thing a predicted one does.
    seeds = [int(s) for s in dataset_of(cfg).get("frozen_seeds", [0])]
    frozen_tag = candidate_tag("frozen", seed=(seeds[0] if seeds else 0))
    # A frozen row spliced into the frozen box is the identity; and a donor whose
    # field was dropped cannot be spliced at all.
    usable = np.asarray([
        bool(np.isfinite(mean[i]) and np.isfinite(true[i])
             and rows[i]["tag"] != frozen_tag
             and (candidate_dir(rows[i]["box"], rows[i]["tag"]) / "field.npy").is_file()
             and (candidate_dir(rows[i]["box"], frozen_tag) / "field.npy").is_file())
        for i in range(len(rows))])
    idx_usable = np.nonzero(usable)[0]
    if idx_usable.size == 0:
        print(">>> GATE FAILED: no held-out candidate has both its own field and")
        print(f">>> its box's {frozen_tag} field on disk, so nothing can be spliced.")
        print(">>> Regenerate the held-out candidates without --drop-field.")
        return 0

    n_total = int(gate.get("n_splice_verifications", 12))
    strata = [str(s) for s in gate.get("splice_strata", STRATA)]
    per = max(1, n_total // max(len(strata), 1))
    rng = np.random.default_rng(int(args.seed))
    taken: set = set()
    plan: List[Dict] = []
    for k, name in enumerate(strata):
        want = per if k < len(strata) - 1 else n_total - per * (len(strata) - 1)
        if name == "predicted_positive":
            order = [int(i) for i in idx_usable[np.argsort(-mean[idx_usable])]]
        elif name == "high_uncertainty":
            order = [int(i) for i in idx_usable[np.argsort(-std[idx_usable])]]
        else:
            order = [int(i) for i in rng.permutation(idx_usable)]
        for i in _pick_spread_over_boxes(order, arrays["box"], want, taken):
            taken.add(i)
            r = rows[i]
            plan.append({
                "stratum": name, "row_index": int(i),
                "box": r["box"], "tile_id": int(r["tile_id"]),
                "donor_tag": r["tag"], "source": r["source"],
                "alpha": r.get("alpha"), "mode": r.get("mode", "both"),
                "base_tag": frozen_tag,
                "predicted_dR": float(mean[i]),
                "predicted_std": float(std[i]),
                "label_dR": float(true[i]),
            })

    doc = {
        "arm": arm, "run_name": args.run_name, "n_planned": len(plan),
        "boxes": sorted({p["box"] for p in plan}),
        "strata": {s: sum(1 for p in plan if p["stratum"] == s) for s in strata},
        "frozen_tag": frozen_tag, "splices": plan,
    }
    write_json_atomic(plan_path(run, arm), doc)
    banner(json.dumps({k: v for k, v in doc.items() if k != "splices"}, indent=2))
    if len(doc["boxes"]) < int(gate.get("min_splice_boxes", 3)):
        print(f">>> WARNING: the plan spans {len(doc['boxes'])} boxes, below the "
              f">>> configured minimum {gate.get('min_splice_boxes', 3)}. The gate "
              ">>> will report n_splice_boxes as unmet.")
    print(f"  submit one array task per splice: --stage run --index 0..{len(plan) - 1}")
    return 0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def stage_run(cfg, args) -> int:
    acfg = actor_config_of(cfg)
    _, reward_t = load_reward_models(cfg)
    bins = bins_of(cfg["_reward"])
    d = cfg["_reward"]["data"]
    box_l = float(d.get("boxsize_mpc_h", 100.0))
    grid = tile_grid_of(cfg)
    arm = args.arm
    run = run_dir(args.run_name, create=True)

    p = plan_path(run, arm)
    if not p.is_file():
        print(f">>> MISSING INPUT: {p}")
        print(">>> produced by: splice_verify.py --stage select")
        return 0
    plan = json.loads(p.read_text())["splices"]
    if args.index >= len(plan):
        print(f">>> array index {args.index} is beyond the {len(plan)}-splice plan; "
              "nothing to do.")
        return 0
    item = plan[int(args.index)]

    out = results_path(run, arm)
    if out.is_file() and args.reuse:
        from _sr2_direct import read_jsonl
        done = {(r["box"], int(r["tile_id"]), r["donor_tag"])
                for r in read_jsonl(out)}
        if (item["box"], int(item["tile_id"]), item["donor_tag"]) in done:
            banner(f"splice {args.index} already verified; nothing to do.")
            return 0

    base_dir = candidate_dir(item["box"], item["base_tag"])
    donor_dir = candidate_dir(item["box"], item["donor_tag"])
    for dpath in (base_dir, donor_dir):
        if not (dpath / "field.npy").is_file():
            print(f">>> MISSING INPUT: {dpath / 'field.npy'}")
            print(">>> produced by: collect_catalog_proxy_data.py --stage generate")
            return 0
    frozen_report = base_dir / "label_report.json"
    if not frozen_report.is_file():
        print(f">>> MISSING INPUT: {frozen_report}")
        print(">>> produced by: collect_catalog_proxy_data.py --stage label")
        return 0
    before_counts = json.loads(frozen_report.read_text())["full_box"]

    t0 = time.time()
    banner(f"splice {args.index}: {item['box']} tile {item['tile_id']} <- "
           f"{item['donor_tag']} ({item['stratum']})")
    base = np.load(base_dir / "field.npy", mmap_mode="r")
    donor = np.load(donor_dir / "field.npy", mmap_mode="r")
    field = splice_tiles(np.asarray(base), np.asarray(donor),
                         [int(item["tile_id"])], grid)
    del base, donor

    work = run / "splices" / f"{arm}_{args.index:02d}_{item['box']}_t{item['tile_id']}"
    cat = run_rockstar_on_field(
        field, work, tag=f"splice{args.index:02d}",
        boxsize_kpc_h=box_l * 1000.0, redshift=float(d.get("redshift", 0.0)),
        overwrite=not args.reuse)
    del field
    for g in Path(work).glob("*.gadget2"):
        g.unlink(missing_ok=True)

    after = direct_full_box_stats(cat, bins)
    vol = box_l ** 3
    w_joint, w_occ = float(acfg.w_joint_reward), float(acfg.w_occ_reward)
    with torch.no_grad():
        s_before = _box_summary(before_counts, vol)
        s_after = _box_summary({k: after[k].tolist()
                                for k in ("n_sub", "n_host", "occ_numerator")}, vol)
        before_scores = reward_t.scores(s_before)
        after_scores = reward_t.scores(s_after)
        measured = {f"dR_{k[2:]}": float(after_scores[k][0] - before_scores[k][0])
                    for k in after_scores}
        measured["dR_combined"] = float(
            reward_t.combined(s_after, w_joint=w_joint, w_occ=w_occ)[0]
            - reward_t.combined(s_before, w_joint=w_joint, w_occ=w_occ)[0])

    row = dict(item)
    row.update({
        "index": int(args.index), "arm": arm,
        "measured_dR": measured["dR_combined"],
        "measured": measured,
        "sign_agrees": bool(np.sign(item["predicted_dR"])
                            == np.sign(measured["dR_combined"])),
        "n_objects_after": int(cat.n),
        "full_box_after": {k: after[k].tolist() for k in
                           ("n_sub", "n_host", "occ_numerator", "occupation")},
        "full_box_before": before_counts,
        "catalog_path": str(getattr(cat, "path", "")),
        "wall_min": (time.time() - t0) / 60.0,
    })
    append_jsonl(out, row)
    banner(f"predicted {item['predicted_dR']:+.5g} vs measured "
           f"{measured['dR_combined']:+.5g}  -> "
           f"{'AGREE' if row['sign_agrees'] else 'DISAGREE'}   {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--stage", required=True, choices=("select", "run"))
    ap.add_argument("--arm", default="a", choices=list(ARMS))
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse", action="store_true", default=True)
    ap.add_argument("--overwrite", dest="reuse", action="store_false")
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    if args.stage == "select":
        return stage_select(cfg, args)
    return stage_run(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
