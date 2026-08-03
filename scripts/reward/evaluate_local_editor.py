#!/usr/bin/env python
"""Final comparison: six arms at an equal Rockstar-call budget.

    1. frozen SR2                       (the anchor)
    2. random analytic actions          (does search do anything?)
    3. CEM                              (what search found)
    4. Gaussian-mixture distillation    (the mandatory baseline)
    5. conditional action flow          (the model the plan is for)
    6. the masked-HR oracle             (an unattainable upper bound)

Arm 6 is read from the reward line's Experiment-1 rows, not rerun: it is an
upper bound precisely because it uses the answer, so it belongs in the table as
a ceiling and nowhere else.

Equal budget is enforced, not assumed: every arm is truncated to the same number
of scored candidate boxes, and the truncation is reported. Comparing a 28-box
CEM arm against an 8-box flow arm would flatter whichever one got more Rockstar
calls, which is the only currency this pipeline spends.

Primary result
--------------
Improvement in **occupation** in at least two reliable host bins including one
upper bin, as the plan defines it. Everything else -- object realisation rate,
host preservation, artifacts, feasibility, catalog diversity across seeds -- is
reported alongside, because a table with only the primary number in it cannot be
audited.

    python scripts/reward/evaluate_local_editor.py --run-name le_a
    python scripts/reward/evaluate_local_editor.py --run-name le_a --final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from _local_common import (  # noqa: E402
    PIPELINE, add_local_args, assert_no_final_boxes, banner, load_local_config,
    read_jsonl, rows_path, run_dir, write_json,
)

from cosmo_sr.reward import paths  # noqa: E402
from cosmo_sr.reward.local_reward import (  # noqa: E402
    ProposalOutcome, gate1_verdict, is_scientific_success,
)

ARM_ORDER = ("frozen", "random", "cem", "gmm", "flow", "oracle_hr")


def arm_of(row: Dict) -> str:
    ctl = row.get("control", "none")
    return row["arm"] if ctl in ("none", "") else f"{row['arm']}:{ctl}"


def summarize_arm(rows: Sequence[Dict], *, budget: Optional[int] = None) -> Dict:
    """Everything one arm contributes to the table, from its candidate rows."""
    rows = sorted(rows, key=lambda r: str(r.get("candidate_id", "")))
    truncated = False
    if budget is not None and len(rows) > budget:
        rows, truncated = rows[:budget], True

    outs: List[ProposalOutcome] = []
    occ, feas, dmg, art = [], [], [], []
    for r in rows:
        feas.append(bool(r.get("feasible_field", False)))
        if r.get("occupation") is not None:
            occ.append(np.asarray(r["occupation"], dtype=np.float64))
        for o in r.get("outcomes", []):
            oo = ProposalOutcome.from_dict(o)
            outs.append(oo)
            dmg.append(float(oo.r_host_damage))
            art.append(float(oo.n_artifacts))
    n = len(outs)
    succ = [o for o in outs if is_scientific_success(o)]
    inert = [o for o in outs if o.n_active_particles == 0]
    return {
        "n_candidate_boxes": len(rows), "budget_truncated": truncated,
        "n_proposals": n,
        # Proposals that moved no particles at all. They consumed Rockstar budget
        # without testing anything, so a high value makes every other rate in
        # this row a rate over a smaller effective sample than it looks.
        "inert_fraction": (len(inert) / n) if n else float("nan"),
        "object_realization_rate": (len(succ) / n) if n else float("nan"),
        "n_successes": len(succ),
        "reward_mean": float(np.mean([o.reward for o in outs])) if n else float("nan"),
        "reward_max": float(np.max([o.reward for o in outs])) if n else float("nan"),
        "host_preserved_rate": (float(np.mean([o.host_matched for o in outs]))
                                if n else float("nan")),
        "host_damage_mean": float(np.mean(dmg)) if dmg else float("nan"),
        "artifacts_per_proposal": float(np.mean(art)) if art else float("nan"),
        "field_feasible_rate": float(np.mean(feas)) if feas else float("nan"),
        "occupation_mean": (np.nanmean(np.stack(occ), axis=0).tolist() if occ else None),
        # Catalog diversity across seeds: the spread of the occupation curve
        # across this arm's candidates. A policy whose samples all produce the
        # same catalog has no distribution to speak of, whatever its action
        # diversity says.
        "occupation_spread": (
            float(np.nanmean(np.nanstd(np.stack(occ), axis=0)
                             / np.maximum(np.abs(np.nanmean(np.stack(occ), axis=0)), 1e-9)))
            if len(occ) > 1 else float("nan")),
        "hosts": sorted({int(o.base_host_id) for o in succ}),
        "boxes": sorted({str(r.get("box", "")) for r in rows}),
    }


def oracle_arm(boxes: Sequence[str]) -> Dict:
    """The masked-HR oracle, as an upper bound only."""
    rows: List[Dict] = []
    for box in boxes:
        p = paths.subdir("oracle_hr", box) / "interventions.jsonl"
        if p.is_file():
            rows.extend([r for r in read_jsonl(p)
                         if r.get("kind") == "targeted" and float(r.get("alpha", 0)) > 0])
    if not rows:
        return {"n_candidate_boxes": 0, "note": "no Experiment-1 rows found"}
    rec = [float(r.get("recovery_rate", np.nan)) for r in rows]
    occ = [np.asarray(r["occupation"], dtype=np.float64) for r in rows
           if r.get("occupation") is not None]
    return {
        "n_candidate_boxes": len(rows),
        "object_realization_rate": float(np.nanmax(rec)) if rec else float("nan"),
        "object_realization_rate_at_alpha": {
            f"{r.get('mode')}@{r.get('alpha')}": float(r.get("recovery_rate", np.nan))
            for r in rows},
        "occupation_mean": (np.nanmean(np.stack(occ), axis=0).tolist() if occ else None),
        "note": ("UNATTAINABLE upper bound: this arm injects the true HR "
                 "correction and is not a deployable method."),
    }


def occupation_verdict(cfg: Dict, base_occ, arm_occ, hr_occ) -> Dict:
    """The plan's primary result: improvement toward HR in the right bins.

    "Improvement" is a strict reduction of ``|occ - occ_HR|`` per host bin, not a
    rise in occupation: overshooting a bin is not progress, and a criterion that
    rewarded any increase would be satisfied by an editor that shatters hosts.
    """
    rc = cfg.get("reward", {})
    reliable = [int(i) for i in rc.get("reliable_host_bins", [0, 1, 2, 3])]
    upper = [int(i) for i in rc.get("upper_reliable_host_bins", [2, 3])]
    if base_occ is None or arm_occ is None or hr_occ is None:
        return {"verdict": "not evaluable",
                "reason": "missing frozen, arm or HR occupation curve"}
    b = np.asarray(base_occ, float)
    a = np.asarray(arm_occ, float)
    h = np.asarray(hr_occ, float)
    improved = []
    per_bin = {}
    for i in reliable:
        if i >= b.size or not np.isfinite(b[i]) or not np.isfinite(a[i]) \
                or not np.isfinite(h[i]):
            per_bin[str(i)] = None
            continue
        d0, d1 = abs(b[i] - h[i]), abs(a[i] - h[i])
        per_bin[str(i)] = {"base_gap": float(d0), "arm_gap": float(d1),
                           "improved": bool(d1 < d0)}
        if d1 < d0:
            improved.append(i)
    n_upper = len([i for i in improved if i in upper])
    ok = (len(improved) >= int(rc.get("min_improved_reliable_bins", 2))
          and n_upper >= int(rc.get("min_improved_upper_bins", 1)))
    return {"verdict": "pass" if ok else "fail",
            "improved_bins": improved, "n_improved_upper": n_upper,
            "per_bin": per_bin,
            "requirement": ">= 2 reliable bins improved, >= 1 of them upper "
                           "(host bins 2 or 3)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_local_args(ap)
    ap.add_argument("--run-name", default="le_a")
    ap.add_argument("--boxes", default="")
    ap.add_argument("--budget", type=int, default=0,
                    help="candidate boxes per arm; 0 = the smallest arm's count, "
                         "which is what makes the comparison equal-budget")
    ap.add_argument("--final", action="store_true",
                    help="allow the final-eval boxes. Use ONCE, at the end")
    ap.add_argument("--hr-occupation", default="",
                    help="JSON with the HR occupation curve; without it the "
                         "primary verdict is reported as not evaluable")
    args = ap.parse_args(argv)

    cfg = load_local_config(args)
    boxes = ([b.strip() for b in args.boxes.split(",") if b.strip()]
             or list(cfg.get("search_boxes", [])))
    if not args.final:
        assert_no_final_boxes(cfg, boxes, script="evaluate_local_editor.py")
    else:
        banner("--final: the held-out boxes are being opened. This is a one-shot "
               "action; every number from here is the reported result.")

    rows: List[Dict] = []
    for box in boxes:
        rows.extend([r for r in read_jsonl(rows_path(args.run_name, box))
                     if r.get("scored")])
    if not rows:
        print(f">>> GATE: no scored rows for {args.run_name} in {boxes}.")
        print(">>> produced by: local_editor_candidates_cpu.sbatch")
        print(">>> exiting 0 so dependents report the same rather than stranding.")
        return 0

    by_arm: Dict[str, List[Dict]] = {}
    for r in rows:
        by_arm.setdefault(arm_of(r), []).append(r)

    main_arms = [a for a in ("random", "cem", "gmm", "flow") if a in by_arm]
    budget = int(args.budget) or (min(len(by_arm[a]) for a in main_arms)
                                  if main_arms else 0)

    table: Dict[str, Dict] = {}
    for arm, rs in sorted(by_arm.items()):
        b = budget if arm in main_arms and budget > 0 else None
        table[arm] = summarize_arm(rs, budget=b)
    table["oracle_hr"] = oracle_arm(boxes)

    hr_occ = None
    if args.hr_occupation and Path(args.hr_occupation).is_file():
        hr_occ = json.loads(Path(args.hr_occupation).read_text()).get("occupation")
    base_occ = table.get("frozen", {}).get("occupation_mean")
    verdicts = {a: occupation_verdict(cfg, base_occ,
                                      table[a].get("occupation_mean"), hr_occ)
                for a in main_arms}

    flow_vs_gmm = None
    if "flow" in table and "gmm" in table:
        f, g = table["flow"], table["gmm"]
        better_reward = float(f["reward_mean"]) > float(g["reward_mean"])
        better_spread = float(f.get("occupation_spread", np.nan) or np.nan) > \
            float(g.get("occupation_spread", np.nan) or np.nan)
        flow_vs_gmm = {
            "flow_reward_mean": f["reward_mean"], "gmm_reward_mean": g["reward_mean"],
            "flow_occupation_spread": f.get("occupation_spread"),
            "gmm_occupation_spread": g.get("occupation_spread"),
            "flow_justified": bool(better_reward or better_spread),
            "criterion": ("the flow is justified only if it beats the mixture on "
                          "realised reward or on catalog diversity at equal "
                          "Rockstar budget; otherwise report the mixture."),
        }

    report = {
        "pipeline": PIPELINE, "run_name": args.run_name, "boxes": boxes,
        "final_boxes_opened": bool(args.final),
        "equal_budget_candidate_boxes": budget,
        "arms": table,
        "occupation_verdicts": verdicts,
        "flow_vs_gaussian_mixture": flow_vs_gmm,
        "gate1": gate1_verdict(
            rows, forbidden_boxes=([] if args.final
                                   else cfg.get("split", {}).get("final_eval_boxes", []))),
        "arm_order": list(ARM_ORDER),
    }
    out = write_json(run_dir(args.run_name) / "final_comparison.json", report)

    banner(f"final comparison ({budget} candidate boxes per arm) -> {out}")
    hdr = (f"    {'arm':<24}{'boxes':>6}{'props':>7}{'realized':>10}"
           f"{'r_mean':>9}{'host_ok':>9}{'artif':>8}{'feasible':>10}{'inert':>8}")
    print(hdr, flush=True)
    for arm in list(ARM_ORDER) + [a for a in sorted(table) if a not in ARM_ORDER]:
        t = table.get(arm)
        if not t:
            continue
        print(f"    {arm:<24}{t.get('n_candidate_boxes', 0):>6}"
              f"{t.get('n_proposals', 0):>7}"
              f"{t.get('object_realization_rate', float('nan')):>10.3f}"
              f"{t.get('reward_mean', float('nan')):>9.3f}"
              f"{t.get('host_preserved_rate', float('nan')):>9.3f}"
              f"{t.get('artifacts_per_proposal', float('nan')):>8.2f}"
              f"{t.get('field_feasible_rate', float('nan')):>10.3f}"
              f"{t.get('inert_fraction', float('nan')):>8.3f}", flush=True)
    for a, v in verdicts.items():
        print(f"    occupation verdict [{a}]: {v['verdict']} "
              f"(improved {v.get('improved_bins')})", flush=True)
    if flow_vs_gmm is not None:
        print(f"    flow justified over the mixture: "
              f"{flow_vs_gmm['flow_justified']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
