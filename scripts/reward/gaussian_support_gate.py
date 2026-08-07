#!/usr/bin/env python
"""Phase 2.3 (CPU): does the reward have usable support before a long run?

Samples from the untrained / reference Gaussian policy are already scored by the
existing CPU stage; this reads those rows and answers one question: **is there
anything for reward-weighted likelihood to learn from?**

Measured, per amplitude arm and pooled:

* fraction of candidates that are feasible;
* fraction with positive ``dR_occ = R_occ_reliable(B + delta) - R_occ_reliable(B)``
  against the SAME frozen SR2 realisation;
* variation in occupation and in subhalo count across candidates;
* ``tanh`` saturation;
* effective sample size ``ESS = (sum w_i)^2 / sum w_i^2`` over the elite weights;
* whether at least two reliable occupation bins improve, one of them upper.

``dR_occ`` is a *paired difference against the candidate's own base*, never a
comparison to a different realisation: SR2's seed-to-seed scatter is larger than
the effect being looked for, so an unpaired version would measure the seed.

Thresholds are **report thresholds, not scientific truths**. They say whether
this pipeline has enough signal to justify a long training run. They are in the
config, they are printed with the verdict, and a failure prints the four
follow-ups (amplitude sweep, multiscale variance reallocation, latent search,
shaped diagnostics) rather than suggesting that diffusion will fix it.

    python scripts/reward/gaussian_support_gate.py --run-name gauss_ref_k16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from _common import (add_common_args, banner, load_reward_config,
                     write_json)

from cosmo_sr.reward import paths
from cosmo_sr.reward.catalog import read_summaries, pool
from cosmo_sr.reward.replay import elite_weights
from cosmo_sr.reward.reward import RewardModel


def effective_sample_size(w: Sequence[float]) -> float:
    """``(sum w)^2 / sum w^2``: how many examples the weighted loss really has.

    An ESS of 1 means one candidate owns the objective and the "training set" is
    a single crop wearing a distribution's clothes.
    """
    a = np.asarray(list(w), dtype=np.float64)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size == 0:
        return 0.0
    return float(a.sum() ** 2 / max((a ** 2).sum(), 1e-30))


def relative_spread(values: Sequence[float]) -> float:
    """Std / |mean| across candidates; NaN for fewer than two finite values."""
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    return float(a.std(ddof=1) / max(abs(a.mean()), 1e-30))


def load_scored(run_dir: Path) -> List[Dict]:
    rows = []
    for p in sorted((run_dir / "scored").glob("*.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except ValueError:
            print(f"  ! unreadable {p.name}", flush=True)
    return rows


def occupation_of(row: Dict) -> Optional[tuple]:
    """``(occupation curve, pooled summary)`` from a scored row's whole-box file.

    The whole-box summary, not the chunk file: the chunk attribution exists for
    credit assignment and drops hosts in proportion to their mass, which is
    exactly the population the occupation reward is about.
    """
    p = row.get("full_box_summary")
    if not p or not Path(p).is_file():
        return None
    ens = pool(read_summaries(p))
    return np.asarray(ens.occupation(), dtype=np.float64), ens


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--model-config", default="configs/reward/gaussian_policy.yaml")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--reward-model", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_reward_config(args)
    from cosmo_sr.utils.config import load_config
    mc = load_config(args.model_config)
    gcfg = dict(mc.get("support_gate", {}))
    rcfg = dict(cfg.get("replay", {}))
    occ_cfg = cfg.get("reward", {}).get("occupation", {})
    reliable = list(occ_cfg.get("reliable_host_bins", [0, 1, 2, 3]))
    upper = list(occ_cfg.get("upper_reliable_host_bins", [2, 3]))

    thresholds = {
        "min_feasible_positive": int(gcfg.get("min_feasible_positive", 5)),
        "min_ess": float(gcfg.get("min_ess", 8.0)),
        "min_improved_reliable_bins": int(gcfg.get(
            "min_improved_reliable_bins",
            occ_cfg.get("min_improved_reliable_bins", 2))),
        "min_improved_upper_bins": int(gcfg.get(
            "min_improved_upper_bins", occ_cfg.get("min_improved_upper_bins", 1))),
        "max_tanh_saturation": float(gcfg.get("max_tanh_saturation", 0.05)),
        "min_occupation_spread": float(gcfg.get("min_occupation_spread", 0.02)),
        "min_subhalo_count_spread": float(gcfg.get("min_subhalo_count_spread", 0.02)),
        # Positive candidates carrying NO `critical` constraint breach. A
        # critical breach does not make a candidate infeasible (see
        # constraints.severity in reward.yaml), so without this the gate could
        # pass on a set of improvements that are all density collapse. 0 disables.
        "min_clean_positive": int(gcfg.get("min_clean_positive", 1)),
    }

    run_dir = paths.ORACLE(args.run_name)
    man_path = run_dir / "candidates.json"
    if not man_path.is_file():
        raise SystemExit(f"no {man_path}; run sample_gaussian_candidates.py first")
    manifest = json.loads(man_path.read_text())
    rows = load_scored(run_dir)
    if not rows:
        raise SystemExit(
            f"no scored rows under {run_dir}/scored; run the scoring array "
            f"(scripts/slurm/gaussian_score_cpu.sbatch) first")

    rm_path = Path(args.reward_model) if args.reward_model else \
        paths.subdir("reward_model") / "reward_model.json"
    if not Path(rm_path).is_file():
        raise SystemExit(
            f"no reward model at {rm_path}; the gate is defined on R_occ_reliable "
            f"and cannot be evaluated without it. Run "
            f"scripts/reward/fit_reward_model.py.")
    model = RewardModel.from_dict(json.loads(Path(rm_path).read_text()))

    # Amplitude / seed metadata lives in the manifest, not in the scored rows
    # (score_oracle.py is shared with the supervised line and knows nothing about
    # this arm), so join them on (box, seed).
    meta = {(c["box"], int(c["seed"])): c for c in manifest.get("candidates", [])}

    baselines: Dict[str, Dict] = {}
    cands: List[Dict] = []
    for r in rows:
        if r.get("seed") is None:
            baselines[str(r["box"])] = r
        else:
            cands.append(r)
    if not baselines:
        raise SystemExit(
            "no per-box baseline rows: dR_occ is a PAIRED difference against the "
            "candidate's own frozen SR2 realisation, so the a=0 baselines must be "
            "scored first (BASELINES=1 in the scoring array)")

    base_scores: Dict[str, Dict] = {}
    base_occ: Dict[str, np.ndarray] = {}
    for box, r in baselines.items():
        got = occupation_of(r)
        if got is None:
            continue
        occ, ens = got
        base_occ[box] = occ
        base_scores[box] = model.scores(ens, reliable_host_bins=reliable)

    per_candidate: List[Dict] = []
    for r in cands:
        box = str(r["box"])
        if box not in base_scores:
            continue
        got = occupation_of(r)
        if got is None:
            continue
        occ, ens = got
        sc = model.scores(ens, reliable_host_bins=reliable)
        m = meta.get((box, int(r["seed"])), {})
        d_occ = occ - base_occ[box]
        improved = [int(b) for b in reliable
                    if b < d_occ.size and np.isfinite(d_occ[b]) and d_occ[b] > 0]
        per_candidate.append({
            "box": box, "seed": int(r["seed"]),
            "amplitude": float(m.get("amplitude", float("nan"))),
            "projection_mode": m.get("projection_mode", ""),
            "alpha_disp": m.get("alpha_disp"), "alpha_vel": m.get("alpha_vel"),
            "feasible": bool(r.get("feasible_field", False)),
            "violations": r.get("violations", []),
            # Non-blocking breaches, carried through from score_oracle. A
            # `critical` one does not make the candidate infeasible, so it would
            # otherwise be invisible in a gate that only reads `feasible`.
            "constraint_warnings": r.get("constraint_warnings", []),
            "constraint_critical": r.get("constraint_critical", []),
            "R_occ_reliable": float(sc.get("R_occ_reliable", float("nan"))),
            "R_occ": float(sc.get("R_occ", float("nan"))),
            "R_cat": float(sc.get("R_cat", float("nan"))),
            "R_abund": float(sc.get("R_abund", float("nan"))),
            "dR_occ_reliable": float(
                sc.get("R_occ_reliable", float("nan"))
                - base_scores[box].get("R_occ_reliable", float("nan"))),
            "dR_cat": float(sc.get("R_cat", float("nan"))
                            - base_scores[box].get("R_cat", float("nan"))),
            "dR_abund": float(sc.get("R_abund", float("nan"))
                              - base_scores[box].get("R_abund", float("nan"))),
            "n_subs_full_box": int(r.get("n_subs_full_box", 0)),
            "n_hosts_full_box": int(r.get("n_hosts_full_box", 0)),
            "occupation_per_bin": occ.tolist(),
            "d_occupation_per_bin": d_occ.tolist(),
            "improved_reliable_bins": improved,
            "improved_upper_bins": [b for b in improved if b in upper],
            "tanh_saturated_fraction": float(m.get("tanh_saturated_fraction", np.nan)),
            "coarse_fraction": float(m.get("coarse_fraction", np.nan)),
            "low_k_change": float(r.get("constraints", {}).get("low_k_change", np.nan)),
        })

    if not per_candidate:
        raise SystemExit("no candidate could be paired with a scored baseline")

    arms = sorted({c["amplitude"] for c in per_candidate if np.isfinite(c["amplitude"])})
    report = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "reward_model": str(rm_path),
        "untrained_reference_policy": bool(
            manifest.get("untrained_reference_policy", False)),
        "policy_hash": manifest.get("policy_hash", ""),
        "correction": manifest.get("correction", {}),
        "reliable_host_bins": reliable,
        "upper_reliable_host_bins": upper,
        "thresholds": thresholds,
        "thresholds_note": (
            "Report thresholds, not scientific truths: they decide whether a long "
            "training run is justified, not whether the method works."
        ),
        "n_candidates": len(per_candidate),
        "n_boxes": len(base_scores),
        "amplitude_arms": arms,
        "per_arm": {},
        "candidates": per_candidate,
    }

    def summarize(sel: List[Dict]) -> Dict:
        feas = [c for c in sel if c["feasible"]]
        pos = [c for c in feas if np.isfinite(c["dR_occ_reliable"])
               and c["dR_occ_reliable"] > 0]
        w = elite_weights([c["dR_occ_reliable"] for c in pos],
                          tau=float(rcfg.get("tau", 1.0)),
                          a_max=float(rcfg.get("a_max", 5.0)),
                          w_max=float(rcfg.get("w_max", 10.0)),
                          normalize=str(rcfg.get("weight_normalize", "mean"))) \
            if pos else np.zeros(0)
        occ_bins = np.array([c["occupation_per_bin"] for c in sel], dtype=np.float64) \
            if sel else np.zeros((0, 0))
        spreads = [relative_spread(occ_bins[:, j]) for j in range(occ_bins.shape[1])] \
            if occ_bins.size else []
        improved_counts: Dict[int, int] = {}
        for c in pos:
            for b in c["improved_reliable_bins"]:
                improved_counts[b] = improved_counts.get(b, 0) + 1
        best = max(pos, key=lambda c: c["dR_occ_reliable"], default=None)
        crit_pos = [c for c in pos if c["constraint_critical"]]
        return {
            "n": len(sel),
            "n_feasible": len(feas),
            "feasible_fraction": len(feas) / max(len(sel), 1),
            "n_feasible_positive": len(pos),
            "positive_fraction_of_feasible": len(pos) / max(len(feas), 1),
            # Of the candidates this gate would train on, how many breach a
            # non-blocking bound. `n_critical_positive` is the one that matters:
            # an improvement that only exists alongside a critical breach is not
            # an improvement this pipeline can keep.
            "n_critical": len([c for c in sel if c["constraint_critical"]]),
            "n_critical_positive": len(crit_pos),
            "critical_fraction_of_positive": len(crit_pos) / max(len(pos), 1),
            "n_warned": len([c for c in sel if c["constraint_warnings"]]),
            "critical_breaches": sorted({t for c in sel
                                         for t in c["constraint_critical"]})[:8],
            "ess": effective_sample_size(w),
            "dR_occ_reliable": {
                "min": float(np.nanmin([c["dR_occ_reliable"] for c in sel])) if sel else np.nan,
                "median": float(np.nanmedian([c["dR_occ_reliable"] for c in sel])) if sel else np.nan,
                "max": float(np.nanmax([c["dR_occ_reliable"] for c in sel])) if sel else np.nan,
            },
            "dR_abund_median": float(
                np.nanmedian([c["dR_abund"] for c in sel])) if sel else np.nan,
            "occupation_spread_per_bin": spreads,
            "occupation_spread_mean": float(np.nanmean(spreads)) if spreads else np.nan,
            "subhalo_count_spread": relative_spread(
                [c["n_subs_full_box"] for c in sel]),
            "host_count_spread": relative_spread([c["n_hosts_full_box"] for c in sel]),
            "tanh_saturation_max": float(
                np.nanmax([c["tanh_saturated_fraction"] for c in sel])) if sel else np.nan,
            "improved_bin_counts": improved_counts,
            "best_candidate": None if best is None else {
                k: best[k] for k in ("box", "seed", "amplitude", "dR_occ_reliable",
                                     "improved_reliable_bins", "improved_upper_bins")},
        }

    for a in arms:
        report["per_arm"][f"amplitude_{a:g}"] = summarize(
            [c for c in per_candidate if c["amplitude"] == a])
    pooled = summarize(per_candidate)
    report["pooled"] = pooled

    # ------------------------------------------------------------- criteria
    best_bins = max((set(c["improved_reliable_bins"])
                     for c in per_candidate
                     if c["feasible"] and c["dR_occ_reliable"] > 0), default=set())
    best_upper = best_bins & set(upper)
    criteria = {
        "feasible_positive_examples": {
            "value": pooled["n_feasible_positive"],
            "threshold": thresholds["min_feasible_positive"],
            "passed": pooled["n_feasible_positive"] >= thresholds["min_feasible_positive"],
            "detail": (f"{pooled['n_feasible_positive']} feasible candidates with "
                       f"dR_occ_reliable > 0 (need "
                       f"{thresholds['min_feasible_positive']})"),
        },
        "effective_sample_size": {
            "value": pooled["ess"],
            "threshold": thresholds["min_ess"],
            "passed": pooled["ess"] >= thresholds["min_ess"],
            "detail": (f"ESS = {pooled['ess']:.2f} over the positive elite weights "
                       f"(need {thresholds['min_ess']:.1f}); below this the "
                       f"weighted loss is a handful of crops"),
        },
        "improved_reliable_bins": {
            "value": sorted(best_bins),
            "threshold": thresholds["min_improved_reliable_bins"],
            "passed": len(best_bins) >= thresholds["min_improved_reliable_bins"]
                      and len(best_upper) >= thresholds["min_improved_upper_bins"],
            "detail": (f"best candidate improves reliable bins {sorted(best_bins)} "
                       f"(upper: {sorted(best_upper)}); need "
                       f"{thresholds['min_improved_reliable_bins']} including "
                       f"{thresholds['min_improved_upper_bins']} upper"),
        },
        "catalog_responds": {
            "value": pooled["occupation_spread_mean"],
            "threshold": thresholds["min_occupation_spread"],
            "passed": bool(np.isfinite(pooled["occupation_spread_mean"])
                           and pooled["occupation_spread_mean"]
                           >= thresholds["min_occupation_spread"]),
            "detail": (f"occupation spread {pooled['occupation_spread_mean']:.4g}, "
                       f"subhalo-count spread {pooled['subhalo_count_spread']:.4g}. "
                       f"Below the floor the catalog does not respond to the action "
                       f"at all -- SR2's own failure reproduced in the policy"),
        },
        "improvement_is_not_a_critical_breach": {
            "value": pooled["n_feasible_positive"] - pooled["n_critical_positive"],
            "threshold": thresholds["min_clean_positive"],
            "passed": (thresholds["min_clean_positive"] <= 0
                       or (pooled["n_feasible_positive"]
                           - pooled["n_critical_positive"])
                       >= thresholds["min_clean_positive"]),
            "detail": (
                f"{pooled['n_feasible_positive'] - pooled['n_critical_positive']} of "
                f"{pooled['n_feasible_positive']} positive candidates carry no "
                f"`critical` breach (need {thresholds['min_clean_positive']}); "
                f"critical breaches seen: "
                f"{', '.join(pooled['critical_breaches']) or 'none'}. A catalog "
                f"improvement that only ever appears alongside a critical field "
                f"breach is the density-collapse mode, not a result"),
        },
        "bound_is_not_the_model": {
            "value": pooled["tanh_saturation_max"],
            "threshold": thresholds["max_tanh_saturation"],
            "passed": not np.isfinite(pooled["tanh_saturation_max"])
                      or pooled["tanh_saturation_max"] <= thresholds["max_tanh_saturation"],
            "detail": (f"max tanh saturation {pooled['tanh_saturation_max']:.4g} "
                       f"(limit {thresholds['max_tanh_saturation']}); above it the "
                       f"amplitude bound, not the network, is setting the edit"),
        },
    }
    failed = [k for k, v in criteria.items() if not v["passed"]]
    report["criteria"] = criteria
    report["failed"] = failed
    report["passed"] = not failed
    report["verdict"] = "support_usable" if not failed else "support_insufficient"
    if failed:
        report["next_steps"] = [
            "sweep the amplitude and the multiscale variance allocation "
            "(model.sigma_init per scale, sampling.amplitude_curriculum)",
            "try latent optimisation / search over the coefficient fields a_s "
            "before training on them",
            "add shaped phase-space or density-collapse diagnostics to see whether "
            "the edit does anything physical at all",
            "do NOT assume diffusion will fix zero reward variation: a richer "
            "sampler over the same action space with the same reward has the same "
            "support",
        ]

    out = Path(args.out) if args.out else run_dir / "support_gate.json"
    write_json(out, report)
    _write_markdown(run_dir / "support_gate.md", report)

    banner(f"support gate: {report['verdict'].upper()}")
    for name, c in criteria.items():
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name}: {c['detail']}")
    print(f"\n  per-amplitude feasible/positive/ESS:")
    for k, v in report["per_arm"].items():
        print(f"    {k:16s} n={v['n']:3d} feasible={v['n_feasible']:3d} "
              f"positive={v['n_feasible_positive']:3d} ESS={v['ess']:.2f} "
              f"dR_occ_max={v['dR_occ_reliable']['max']:.4g} "
              f"critical={v['n_critical']:3d} warn={v['n_warned']:3d}")
    if pooled["n_critical"]:
        print(f"\n  !! {pooled['n_critical']} of {pooled['n']} candidates breach a "
              f"CRITICAL bound ({pooled['n_critical_positive']} of them among the "
              f"{pooled['n_feasible_positive']} positives). These are feasible by "
              f"policy -- see constraints.severity in reward.yaml -- not by "
              f"measurement:")
        for t in pooled["critical_breaches"]:
            print(f"       {t}")
    if failed:
        print("\n  next steps (in order):")
        for i, s in enumerate(report["next_steps"], 1):
            print(f"    {i}. {s}")
    print(f"\n  -> {out}")


def _write_markdown(path: Path, rep: Dict) -> None:
    L: List[str] = []
    L.append(f"# Support gate -- `{rep['run_name']}`\n")
    L.append(f"**{rep['verdict']}** "
             f"({'all criteria passed' if rep['passed'] else 'failed: ' + ', '.join(rep['failed'])})\n")
    L.append(f"{rep['n_candidates']} candidates over {rep['n_boxes']} boxes; "
             f"policy `{rep.get('policy_hash', '')[:12]}`"
             + (" (untrained reference)" if rep.get("untrained_reference_policy") else "")
             + ".\n")
    L.append(f"> {rep['thresholds_note']}\n")

    L.append("## Criteria\n")
    L.append("| criterion | value | threshold | result |")
    L.append("|---|---|---|---|")
    for k, c in rep["criteria"].items():
        v = c["value"]
        vs = f"{v:.4g}" if isinstance(v, float) else str(v)
        L.append(f"| `{k}` | {vs} | {c['threshold']} | "
                 f"{'PASS' if c['passed'] else '**FAIL**'} |")
    L.append("")

    L.append("## Amplitude arms\n")
    L.append("| arm | n | feasible | positive dR_occ | ESS | max dR_occ "
             "| occ spread | sat | critical | warn |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, v in rep["per_arm"].items():
        L.append(f"| `{k}` | {v['n']} | {v['n_feasible']} | {v['n_feasible_positive']} "
                 f"| {v['ess']:.2f} | {v['dR_occ_reliable']['max']:.4g} "
                 f"| {v['occupation_spread_mean']:.4g} "
                 f"| {v['tanh_saturation_max']:.4g} "
                 f"| {v.get('n_critical', 0)} | {v.get('n_warned', 0)} |")
    L.append("")

    pooled = rep.get("pooled", {})
    if pooled.get("n_critical"):
        L.append("## Critical breaches\n")
        L.append(f"{pooled['n_critical']} of {pooled['n']} candidates breach a "
                 f"constraint marked `critical`, including "
                 f"{pooled['n_critical_positive']} of the "
                 f"{pooled['n_feasible_positive']} positive ones. They count as "
                 f"feasible by *policy* (`constraints.severity` in `reward.yaml`), "
                 f"not because the field statistic passed.\n")
        for t in pooled.get("critical_breaches", []):
            L.append(f"* `{t}`")
        L.append("")

    L.append("## Reading this\n")
    L.append("* `dR_occ_reliable` is a **paired** difference against the "
             "candidate's own frozen SR2 realisation. SR2's seed-to-seed scatter "
             "is larger than the effect, so an unpaired version measures the seed.")
    L.append("* `R_abund` is reported but never sufficient: abundance-only "
             "improvement does not count as success.")
    L.append("* ESS is over the positive candidates' elite weights. A low ESS with "
             "many positives means one candidate owns the objective.\n")

    if not rep["passed"]:
        L.append("## Next steps\n")
        for i, s in enumerate(rep.get("next_steps", []), 1):
            L.append(f"{i}. {s}")
        L.append("")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
