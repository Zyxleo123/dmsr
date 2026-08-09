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

The advancement rule (plan section 7)
-------------------------------------
* neither arm passes -- do not fine-tune SR2. The proxy or its data is the thing
  to fix, not the number of unfrozen parameters;
* one arm passes -- advance only that arm;
* both pass -- advance both to the same small fine-tuning screen. Proxy accuracy
  alone does not establish that the *gradient* is useful, which is a different
  question and a later one;
* B passes where A fails, especially on the velocity-only interventions --
  phase-space summaries are justified, and the velocity-only slice is what makes
  that attributable rather than incidental;
* the two are equivalent -- prefer A, because it is the simpler model, unless B
  wins the later equal-budget real-Rockstar screen.

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
    ARMS, as_arrays, build_row_context, ensemble_delta, load_rows, rank_metrics,
    slice_rows, true_delta_rewards,
)
from _sr2_direct import (  # noqa: E402
    actor_config_of, add_direct_args, banner, boxes_of, code_commit, direct_root,
    file_sha, labels_complete_path, load_direct_config, load_reward_models,
    rockstar_provenance, run_dir, write_json_atomic,
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
    diff: List[float] = []
    arms = sorted(pred)
    for _ in range(int(n)):
        picked = rng.choice(boxes, size=len(boxes), replace=True)
        idx = np.concatenate([idx_by_box[b] for b in picked])
        vals = {}
        for a in arms:
            vals[a] = rank_metrics(pred[a][idx], true[idx], box[idx],
                                   tile[idx])[metric]
            draws[a].append(vals[a])
        if len(arms) == 2:
            diff.append(vals[arms[1]] - vals[arms[0]])

    def ci(v: Sequence[float]) -> Dict[str, float]:
        arr = np.asarray(v, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"median": float("nan"), "lo90": float("nan"),
                    "hi90": float("nan")}
        return {"median": float(np.median(arr)),
                "lo90": float(np.percentile(arr, 5)),
                "hi90": float(np.percentile(arr, 95))}

    out: Dict = {"metric": metric, "n_boxes": len(boxes), "n_draws": int(n),
                 "per_arm": {a: ci(draws[a]) for a in arms}}
    if len(arms) == 2:
        d = ci(diff)
        d["arms"] = f"{arms[1]} - {arms[0]}"
        finite = np.asarray([x for x in diff if np.isfinite(x)])
        d["fraction_positive"] = (float(np.mean(finite > 0)) if finite.size
                                  else float("nan"))
        out["difference"] = d
        out["verdict"] = ("second_better" if d["lo90"] > 0 else
                          "first_better" if d["hi90"] < 0 else "equivalent")
    return out


def _decide(verdicts: Dict[str, Dict], comparison: Dict,
            vel_comparison: Dict) -> Dict:
    """Plan section 7, as a function rather than as a paragraph someone reads."""
    passed = {a: bool(v.get("passed", False)) for a, v in verdicts.items()}
    a_ok, b_ok = passed.get("a", False), passed.get("b", False)
    equiv = comparison.get("verdict", "equivalent")

    if not a_ok and not b_ok:
        return {
            "decision": "do_not_finetune",
            "advance": [],
            "rationale": "Neither arm cleared its predeclared criteria. Fix the "
                         "proxy or its data, or move to the true-catalog oracle "
                         "renderer. Unfreezing more of SR2 is not a response to "
                         "a proxy that cannot rank.",
        }
    if a_ok and not b_ok:
        return {"decision": "advance_a_only", "advance": ["a"],
                "rationale": "Only the density-only arm passed. The phase-space "
                             "summaries are not justified by this evidence."}
    if b_ok and not a_ok:
        vel = vel_comparison.get("verdict", "equivalent")
        return {
            "decision": "advance_b_only", "advance": ["b"],
            "phase_space_justified": True,
            "velocity_slice_verdict": vel,
            "rationale": (
                "Only the phase-space arm passed"
                + (", and it also wins the velocity-only intervention slice, "
                   "which arm A cannot see by construction -- the win is "
                   "attributable to the extra coordinates."
                   if vel == "second_better" else
                   ". Note that it does NOT separate from arm A on the "
                   "velocity-only slice, so the advantage is not clearly "
                   "attributable to the phase-space coordinates; treat the "
                   "attribution as open.")),
        }
    if equiv == "equivalent":
        return {
            "decision": "advance_both_prefer_a", "advance": ["a", "b"],
            "preferred": "a",
            "rationale": "Both arms passed and the box-bootstrap interval on "
                         "their difference contains zero. Prefer A for "
                         "simplicity; B may still be chosen if it produces "
                         "better real-Rockstar improvement in the later "
                         "equal-budget fine-tuning screen.",
        }
    better = "b" if equiv == "second_better" else "a"
    return {
        "decision": "advance_both", "advance": ["a", "b"], "preferred": better,
        "rationale": f"Both arms passed and arm {better.upper()} is ahead by more "
                     "than the box-bootstrap interval. Both still go to the same "
                     "small fine-tuning screen: proxy accuracy alone does not "
                     "establish that the gradient is useful.",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
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
    ctx = build_row_context(rows)
    true = true_delta_rewards(ctx, reward_t, w_joint=w_joint, w_occ=w_occ)
    common = as_arrays(rows, arms[0])
    preds: Dict[str, np.ndarray] = {}
    for arm in arms:
        d = run / f"proxy_{arm}"
        if not d.is_dir():
            print(f">>> MISSING INPUT: {d}")
            return 0
        ens = ProxyEnsemble.load(d).freeze()
        feats = torch.as_tensor(as_arrays(rows, arm)["features"], dtype=torch.float64)
        preds[arm], _ = ensemble_delta(ens.members, feats, ctx, reward_t,
                                       w_joint=w_joint, w_occ=w_occ)

    comparison = box_bootstrap(preds, true, common["box"], common["tile_id"],
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
            common["tile_id"][idx], n=int(args.n_bootstrap), seed=int(args.seed))
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
