#!/usr/bin/env python
"""The one immutable record: both arms, every metric, and the decision.

This is where the milestone ends. It reads the two per-arm verdicts, adds the
arm-versus-arm comparison neither of them can make on its own, applies the
advancement rule, and writes ``proxy_benchmark.json`` **once**. Re-running it
without ``--overwrite`` refuses rather than silently replacing the record: a
benchmark that can be rewritten after the fact is a draft, and the point of this
file is that it was written before anyone knew what to wish for.

The comparison is bootstrapped over BOXES
-----------------------------------------
The table has ~50,000 rows and twelve boxes. The rows are not 50,000 independent
examples: tiles of one box share its large-scale modes, its LR realisation and
its frozen SR2 draw. Twelve is the sample size. Every interval here therefore
resamples *boxes* with replacement and recomputes the metric on the resampled
rows, which is the only version of "is B better than A" that is not mostly a
statement about how many tiles fit in a box.

The advancement rule
--------------------
An arm advances only on its FULL per-arm verdict: the offline held-out screen
AND the actor-like Rockstar verification (a probe_only fine-tune's own
generated tiles, spliced into held-out boxes and re-run through the real halo
finder). An arm that passed offline but has no actor verification is recorded
as *awaiting* and refused -- only real Rockstar results can authorize direct
SR2 fine-tuning.

* no arm passes in full -- do not fine-tune SR2. The proxy or its data is the
  thing to fix, not the number of unfrozen parameters;
* one or more arms pass -- advance them to the same small fine-tuning screen.
  Proxy accuracy alone does not establish that the *gradient* is useful, which
  is a different question and a later one;
* among the advanced arms, prefer the simplest (a, then b, then c) unless a
  more complex one is ahead by more than the box-bootstrap interval;
* B or C passing where A fails, especially on the velocity-only interventions
  -- the phase-space information is justified, and the velocity-only slice is
  what makes that attributable rather than incidental.

    python scripts/reward/proxy_benchmark.py --run-name direct_a
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from _proxy_data import (  # noqa: E402
    ARMS, as_arrays, build_row_context, load_rows, make_arm_features, rank_metrics,
    slice_rows, stream_ensemble_delta, true_delta_rewards, unit_ids_of,
)
from _sr2_direct import (  # noqa: E402
    actor_config_of, add_direct_args, banner, boxes_of, code_commit, direct_root,
    file_sha, labels_complete_path, load_direct_config, load_reward_models,
    region_width_of, rockstar_provenance, run_dir, write_json_atomic,
)

from cosmo_sr.reward.catalog_proxy import ProxyEnsemble  # noqa: E402


def box_bootstrap(pred: Dict[str, np.ndarray], true: np.ndarray,
                  box: np.ndarray, tile: np.ndarray, *, n: int = 2000,
                  seed: int = 0, metric: str = "within_tile_spearman") -> Dict:
    """Resample BOXES with replacement; report each arm and their difference.

    The difference is computed *within* each bootstrap replicate, not as the
    difference of two independently bootstrapped distributions: the arms are
    evaluated on the same rows, so their errors are strongly correlated and
    treating the intervals as independent would inflate the spread of the
    difference by roughly the square root of two and hide a real gap.
    """
    boxes = sorted({str(b) for b in box})
    idx_by_box = {b: np.nonzero(box == b)[0] for b in boxes}
    rng = np.random.default_rng(int(seed))
    draws: Dict[str, List[float]] = {a: [] for a in pred}
    arms = sorted(pred)
    arm_pairs = [(x, y) for i, x in enumerate(arms) for y in arms[i + 1:]]
    diffs: Dict[str, List[float]] = {f"{y}-{x}": [] for x, y in arm_pairs}
    for _ in range(int(n)):
        picked = rng.choice(boxes, size=len(boxes), replace=True)
        idx = np.concatenate([idx_by_box[b] for b in picked])
        vals = {}
        for a in arms:
            vals[a] = rank_metrics(pred[a][idx], true[idx], box[idx],
                                   tile[idx])[metric]
            draws[a].append(vals[a])
        # Differences are computed WITHIN the replicate (see the docstring).
        for x, y in arm_pairs:
            diffs[f"{y}-{x}"].append(vals[y] - vals[x])

    def ci(v: Sequence[float]) -> Dict[str, float]:
        arr = np.asarray(v, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"median": float("nan"), "lo90": float("nan"),
                    "hi90": float("nan")}
        return {"median": float(np.median(arr)),
                "lo90": float(np.percentile(arr, 5)),
                "hi90": float(np.percentile(arr, 95))}

    def verdict_of(d: Dict[str, float]) -> str:
        return ("second_better" if d["lo90"] > 0 else
                "first_better" if d["hi90"] < 0 else "equivalent")

    out: Dict = {"metric": metric, "n_boxes": len(boxes), "n_draws": int(n),
                 "per_arm": {a: ci(draws[a]) for a in arms}}
    pairwise: Dict[str, Dict] = {}
    for x, y in arm_pairs:
        key = f"{y}-{x}"
        d = ci(diffs[key])
        d["arms"] = f"{y} - {x}"
        finite = np.asarray([v for v in diffs[key] if np.isfinite(v)])
        d["fraction_positive"] = (float(np.mean(finite > 0)) if finite.size
                                  else float("nan"))
        d["verdict"] = verdict_of(d)
        pairwise[key] = d
    out["pairwise"] = pairwise
    if len(arms) == 2:
        # The historical two-arm keys, kept so a two-arm run's record reads the
        # same as it always did.
        out["difference"] = pairwise[f"{arms[1]}-{arms[0]}"]
        out["verdict"] = out["difference"]["verdict"]
    return out


def _decide(verdicts: Dict[str, Dict], comparison: Dict,
            vel_comparison: Dict) -> Dict:
    """The advancement rule, as a function rather than as a paragraph.

    An arm advances only on its FULL verdict: the offline screen and the
    actor-like Rockstar verification together (``passed`` in the per-arm gate
    report). An arm that cleared the offline screen but whose actor
    verification has not landed is named explicitly, and the decision refuses
    to advance it: only real Rockstar results can authorize fine-tuning, and a
    benchmark that advanced an arm on a promise would be the promise, recorded.

    Among the arms that advance, the PREFERRED one is the simplest (a, then b,
    then c) unless a more complex arm beats it by more than the box-bootstrap
    interval on their difference -- complexity has to buy a distinguishable
    improvement or it is not bought.
    """
    passed = {a: bool(v.get("passed", False)) for a, v in verdicts.items()}
    offline = {a: bool(v.get("offline_passed", v.get("passed", False)))
               for a, v in verdicts.items()}
    advance = [a for a in verdicts if passed[a]]
    awaiting = [a for a in verdicts if offline[a] and not passed[a]]
    pairwise = comparison.get("pairwise", {})

    if not advance:
        if awaiting:
            return {
                "decision": "actor_gate_incomplete_do_not_finetune",
                "advance": [],
                "awaiting_actor_gate": sorted(awaiting),
                "rationale": (
                    f"Arm(s) {sorted(awaiting)} cleared the offline screen but "
                    "their actor-like Rockstar verification is missing or "
                    "failing. Run the probe_only fine-tune and "
                    "actor_rockstar_verify.py, re-run the per-arm gate, then "
                    "re-run this benchmark. Nothing may fine-tune SR2 on "
                    "offline evidence alone."),
            }
        return {
            "decision": "do_not_finetune",
            "advance": [],
            "rationale": "No arm cleared its predeclared criteria. Fix the "
                         "proxy or its data, or move to the true-catalog oracle "
                         "renderer. Unfreezing more of SR2 is not a response to "
                         "a proxy that cannot rank.",
        }

    # Preference among the advanced arms: simplest first, displaced only by a
    # distinguishable win.
    order = [a for a in ARMS if a in advance]
    preferred = order[0]
    for rival in order[1:]:
        key = f"{rival}-{preferred}" if rival > preferred else f"{preferred}-{rival}"
        d = pairwise.get(key, {})
        v = d.get("verdict", "equivalent")
        rival_is_second = key.startswith(rival)
        if (v == "second_better" and rival_is_second) or \
                (v == "first_better" and not rival_is_second):
            preferred = rival

    vel = vel_comparison.get("pairwise", {})
    notes = []
    for rich, base in [(r, "a") for r in ("b", "c", "d", "e", "f")]:
        if rich in advance:
            v = vel.get(f"{rich}-{base}", {}).get("verdict", "")
            if v == "second_better":
                notes.append(
                    f"arm {rich.upper()} wins the velocity-only intervention "
                    "slice, which the density-only arm cannot see by "
                    "construction -- its advantage is attributable to the "
                    "phase-space information.")
            elif v:
                notes.append(
                    f"arm {rich.upper()} does NOT separate from arm A on the "
                    "velocity-only slice; treat any advantage's attribution "
                    "as open.")

    doc = {
        "decision": "advance_" + "_".join(sorted(advance)),
        "advance": sorted(advance),
        "preferred": preferred,
        "rationale": (
            f"Arm(s) {sorted(advance)} cleared both the offline screen and the "
            f"actor-like Rockstar verification. Preferred: {preferred.upper()} "
            "-- the simplest passing arm, displaced only by a rival ahead by "
            "more than the box-bootstrap interval. "
            + " ".join(notes)).strip(),
    }
    if awaiting:
        doc["awaiting_actor_gate"] = sorted(awaiting)
        doc["rationale"] += (
            f" Arm(s) {sorted(awaiting)} are offline-eligible but await their "
            "actor verification and do NOT advance.")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing proxy_benchmark.json (it is meant "
                         "to be written once)")
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    acfg = actor_config_of(cfg)
    _, reward_t = load_reward_models(cfg)
    run = run_dir(args.run_name, create=True)
    out_path = run / "proxy_benchmark.json"
    if out_path.is_file() and not args.overwrite:
        print(f">>> {out_path} already exists and is meant to be immutable.")
        print(">>> Re-run with --overwrite only if you are deliberately replacing")
        print(">>> a record written before the numbers were known.")
        return 0

    arms = [a for a in args.arms if a in ARMS]
    verdicts: Dict[str, Dict] = {}
    missing: List[str] = []
    for arm in arms:
        p = run / f"proxy_gate_{arm}.json"
        if p.is_file():
            verdicts[arm] = json.loads(p.read_text())
        else:
            missing.append(str(p))
    if missing:
        print(">>> MISSING INPUT: " + ", ".join(missing))
        print(">>> produced by: scripts/reward/gate_catalog_proxy.py --arm <arm>")
        print(">>> A benchmark written without every arm's verdict would record a")
        print(">>> comparison that was never made.")
        return 0

    table = direct_root("proxy_data") / "rows.jsonl"
    rows_all = load_rows(table, require_complete=not args.allow_incomplete)
    gate_boxes = set(boxes_of(cfg, "proxy_gate"))
    rows = [r for r in rows_all if r["box"] in gate_boxes]
    if not rows:
        print(f">>> GATE FAILED: no held-out rows from {sorted(gate_boxes)}")
        return 0

    w_joint, w_occ = float(acfg.w_joint_reward), float(acfg.w_occ_reward)
    device = torch.device(args.device if (args.device != "cuda"
                                          or torch.cuda.is_available()) else "cpu")
    ctx_cpu = build_row_context(rows)
    true = true_delta_rewards(ctx_cpu, reward_t, w_joint=w_joint, w_occ=w_occ)
    reward_t = reward_t.to(device)
    ctx = ctx_cpu.to(device)
    common = as_arrays(rows, "a")
    unit = unit_ids_of(common["tile_id"], width=region_width_of(cfg))
    chunk = {"a": len(rows), "b": len(rows), "c": 4096, "d": 4096, "e": 512, "f": 48}
    preds: Dict[str, np.ndarray] = {}
    for arm in arms:
        d = run / f"proxy_{arm}"
        if not d.is_dir():
            print(f">>> MISSING INPUT: {d}")
            return 0
        ens = ProxyEnsemble.load(d).freeze()
        provider = make_arm_features(arm, rows, table_dir=table.parent, cfg=cfg)
        members = [m.to(device) for m in ens.members]
        preds[arm], _ = stream_ensemble_delta(
            provider, members, ctx, reward_t, w_joint=w_joint, w_occ=w_occ,
            chunk_rows=chunk.get(arm, 512), device=device)

    comparison = box_bootstrap(preds, true, common["box"], unit,
                               n=int(args.n_bootstrap), seed=int(args.seed))
    slices = dict(cfg.get("proxy_report_slices", {}))
    slice_cmp: Dict[str, Dict] = {}
    for name, spec in slices.items():
        idx = slice_rows(common, spec or {})
        if idx.size < 10 or len({str(b) for b in common["box"][idx]}) < 2:
            slice_cmp[name] = {"n_rows": int(idx.size),
                               "note": "too few rows or boxes to bootstrap"}
            continue
        slice_cmp[name] = box_bootstrap(
            {a: preds[a][idx] for a in arms}, true[idx], common["box"][idx],
            unit[idx], n=int(args.n_bootstrap), seed=int(args.seed))
        slice_cmp[name]["n_rows"] = int(idx.size)

    decision = _decide(verdicts, comparison,
                       slice_cmp.get("interventions_vel", {}))

    train_report = run / "train_report.json"
    marker = labels_complete_path()
    index_report = direct_root("proxy_data") / "index_report.json"
    provenance = {
        "code_commit": code_commit(),
        "config_path": str(args.config),
        "config_sha": file_sha(args.config) if Path(args.config).is_file() else "",
        "reward_config_path": cfg.get("_reward_config_path", ""),
        "table": str(table),
        "table_sha": file_sha(table) if table.is_file() else "",
        "labels_complete": (json.loads(marker.read_text()) if marker.is_file()
                            else None),
        "index_report": (json.loads(index_report.read_text())
                         if index_report.is_file() else None),
        "train_report": (json.loads(train_report.read_text())
                         if train_report.is_file() else None),
        "proxy_dirs": {a: str(run / f"proxy_{a}") for a in arms},
    }
    provenance.update(rockstar_provenance(cfg))

    doc = {
        "run_name": args.run_name,
        "arms": arms,
        "n_gate_rows": len(rows),
        "gate_boxes": sorted(gate_boxes),
        "passed": {a: bool(verdicts[a].get("passed", False)) for a in arms},
        "offline_passed": {a: bool(verdicts[a].get("offline_passed",
                                                   verdicts[a].get("passed", False)))
                           for a in arms},
        "actor_passed": {a: bool(verdicts[a].get("actor_passed", False))
                         for a in arms},
        "failures": {a: verdicts[a].get("failures", []) for a in arms},
        "checks": {a: verdicts[a].get("checks", []) for a in arms},
        "per_arm_verdict": verdicts,
        "arm_comparison": comparison,
        "arm_comparison_by_slice": slice_cmp,
        "decision": decision,
        "provenance": provenance,
        "allow_incomplete": bool(args.allow_incomplete),
    }
    body = json.dumps(doc, indent=2, sort_keys=True)
    doc["content_sha256"] = hashlib.sha256(body.encode()).hexdigest()[:32]
    write_json_atomic(out_path, doc)

    banner(json.dumps({
        "passed": doc["passed"],
        "arm_comparison": comparison.get("difference", comparison),
        "decision": decision,
    }, indent=2))
    for a in arms:
        for f in verdicts[a].get("failures", []):
            print(f"   arm {a} !! {f}")
    print(f"  benchmark -> {out_path}", flush=True)
    if decision["decision"] == "do_not_finetune":
        print("  !! The milestone ends here. No SR2 parameter may move on this "
              "evidence.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
