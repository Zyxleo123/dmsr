#!/usr/bin/env python
"""Turn projection-oracle rows into JSON + CSV + Markdown, and a recommendation.

Reads every ``rows/*.json`` written by ``audit_projection_oracle.py`` and applies
the decision rule of :mod:`cosmo_sr.reward.projection`:

1. reject the hard null projection if ``alpha = 0`` significantly damages
   occupation, host recovery or density relative to ``alpha = 1``;
2. otherwise recommend the **smallest** coarse allowance that is statistically
   indistinguishable from ``alpha = 1`` on every primary metric;
3. choose displacement and velocity separately, from their own sweeps.

Uncertainty is a box bootstrap with seeds averaged inside a box first, and every
comparison against ``alpha = 1`` is paired on the box.

Nothing here reruns a field or a halo finder: it reads rows and writes summaries,
so it belongs on the CPU partition and can be re-rendered as often as wanted.

    python scripts/reward/projection_oracle_report.py --run-name proj0
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from _common import add_common_args, banner, load_reward_config, write_json

from cosmo_sr.reward import paths
from cosmo_sr.reward.projection import (DEFAULT_ALPHAS, PRIMARY_METRICS, MetricSpec,
                                        arm_plan, bootstrap_ci, choose_alpha,
                                        compare_to_reference, per_box_means,
                                        reference_arm_name)

#: Everything tabulated per arm. The decision rule reads only PRIMARY_METRICS;
#: the rest are reported because a recommendation nobody can audit is a guess.
REPORTED: Sequence[MetricSpec] = (
    *PRIMARY_METRICS,
    MetricSpec("R_occ", True, "occupation reward (all active bins)"),
    MetricSpec("R_cat", True, "joint catalog reward"),
    MetricSpec("R_abund", True, "abundance reward"),
    MetricSpec("n_subs_full_box", True, "subhalo count, whole box"),
    MetricSpec("n_hosts_full_box", True, "host count, whole box"),
    MetricSpec("subhalo_ratio_vs_hr", True, "subhalo count / HR subhalo count"),
    MetricSpec("displacement_power_error", False, "mean |T(k) - 1|, displacement"),
    MetricSpec("velocity_power_error", False, "mean |T(k) - 1|, velocity"),
    MetricSpec("low_k_change", False, "||A(X) - A(B)|| / ||A(B)||"),
    MetricSpec("coarse_mismatch_vs_hr", False, "||A(X) - A(HR)|| / ||A(HR)||"),
    MetricSpec("coarse_mismatch_vs_lr", False, "||A(X) - y_lr|| / ||y_lr||"),
    MetricSpec("density_sigma_ratio", True, "sigma(delta_X) / sigma(delta_HR)"),
    MetricSpec("density_pdf_l1", False, "L1 distance of the density PDF"),
    MetricSpec("host_match_score_median", False, "median host match score"),
)


def load_rows(run_dir: Path) -> List[Dict]:
    rows = []
    for p in sorted((run_dir / "rows").glob("*.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except ValueError:
            print(f"  ! unreadable row {p.name}, skipped", flush=True)
    return rows


def occupation_table(rows: Sequence[Dict], arm: str) -> Dict:
    """Per-host-bin occupation, averaged over seeds then boxes, with a box CI."""
    sel = [r for r in rows if str(r.get("arm")) == arm and r.get("occupation_per_bin")]
    if not sel:
        return {}
    n_bins = len(sel[0]["occupation_per_bin"])
    out: Dict[str, Dict] = {}
    for j in range(n_bins):
        expanded = [{**r, f"_occ{j}": (r["occupation_per_bin"][j]
                                       if j < len(r["occupation_per_bin"]) else np.nan)}
                    for r in sel]
        out[f"host_bin_{j}"] = bootstrap_ci(per_box_means(expanded, f"_occ{j}"))
    return out


def abundance_table(rows: Sequence[Dict], arm: str) -> Dict:
    sel = [r for r in rows if str(r.get("arm")) == arm and r.get("n_sub_per_bin")]
    if not sel:
        return {}
    n_bins = len(sel[0]["n_sub_per_bin"])
    out: Dict[str, Dict] = {}
    for j in range(n_bins):
        expanded = [{**r, f"_ab{j}": (r["n_sub_per_bin"][j]
                                      if j < len(r["n_sub_per_bin"]) else np.nan)}
                    for r in sel]
        out[f"sub_bin_{j}"] = bootstrap_ci(per_box_means(expanded, f"_ab{j}"))
    return out


def _fmt(v, nd=4) -> str:
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "nan"
    return f"{f:.{nd}g}"


def _ci(d: Dict, nd=4) -> str:
    if not d or not np.isfinite(d.get("mean", np.nan)):
        return "-"
    if not np.isfinite(d.get("lo", np.nan)):
        return f"{_fmt(d['mean'], nd)} (n={d.get('n_boxes', 0)})"
    return f"{_fmt(d['mean'], nd)} [{_fmt(d['lo'], nd)}, {_fmt(d['hi'], nd)}]"


VERDICT_MARK = {"improved": "better", "indistinguishable": "same",
                "damaged": "WORSE", "undetermined": "?"}


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--run-name", default="projection_oracle")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--alphas", default=None)
    ap.add_argument("--sweeps", default="joint,disp_only,vel_only")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--out-dir", default=None,
                    help="where the report lands (default: the run dir)")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    run_dir = Path(args.run_dir) if args.run_dir else \
        paths.AUDITS(f"projection_oracle/{args.run_name}")
    if not run_dir.is_dir():
        raise SystemExit(f"no projection-oracle run at {run_dir}")
    rows = load_rows(run_dir)
    if not rows:
        raise SystemExit(f"no rows under {run_dir}/rows; has the audit job run?")

    po = cfg.get("projection_oracle", {})
    alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else \
        [float(a) for a in po.get("alphas", DEFAULT_ALPHAS)]
    sweeps = [s.strip() for s in args.sweeps.split(",") if s.strip()]
    arms = arm_plan(alphas, sweeps)
    ref = reference_arm_name(arms)
    present = {str(r.get("arm")) for r in rows}
    boot = dict(n_boot=int(args.n_boot), seed=int(args.bootstrap_seed), ci=float(args.ci))

    # ---------------------------------------------------------------- tables
    per_arm: Dict[str, Dict] = {}
    for arm in arms:
        if arm.name not in present:
            continue
        sel = [r for r in rows if str(r.get("arm")) == arm.name]
        entry: Dict = {
            **arm.to_dict(),
            "n_rows": len(sel),
            "boxes": sorted({str(r["box"]) for r in sel}),
            "seeds": sorted({r.get("seed") for r in sel if r.get("seed") is not None}),
            "metrics": {},
            "occupation_per_host_bin": occupation_table(rows, arm.name),
            "abundance_per_sub_bin": abundance_table(rows, arm.name),
        }
        for m in REPORTED:
            entry["metrics"][m.name] = bootstrap_ci(per_box_means(sel, m.name), **boot)
        if arm.name not in ("sr2", "hr", ref):
            entry["vs_reference"] = {
                m.name: compare_to_reference(rows, m, arm.name, ref, **boot)
                for m in REPORTED
            }
        recon = [float(r["reconstruction_rel_rms_vs_hr"]) for r in sel
                 if r.get("reconstruction_rel_rms_vs_hr") is not None]
        if recon:
            entry["reconstruction_rel_rms_vs_hr_max"] = float(np.max(recon))
        per_arm[arm.name] = entry

    # ------------------------------------------------------------ decisions
    decisions = {}
    for sweep in sweeps:
        if any(a.sweep == sweep and a.name in present for a in arms):
            decisions[sweep] = choose_alpha(rows, arms, sweep=sweep, **boot)

    recommendation = {
        "alpha_disp": float(
            decisions.get("disp_only", decisions.get("joint", {}))
            .get("recommended", {}).get("alpha", 1.0)),
        "alpha_vel": float(
            decisions.get("vel_only", decisions.get("joint", {}))
            .get("recommended", {}).get("alpha", 1.0)),
        "hard_null_rejected": any(d.get("hard_null_rejected") for d in decisions.values()),
        "source_sweeps": {
            "alpha_disp": "disp_only" if "disp_only" in decisions else "joint",
            "alpha_vel": "vel_only" if "vel_only" in decisions else "joint",
        },
    }
    recommendation["correction_mode"] = (
        "block_null"
        if recommendation["alpha_disp"] == 0.0 and recommendation["alpha_vel"] == 0.0
        else "block_leaky"
    )

    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "n_rows": len(rows),
        "arms_present": sorted(present),
        "arms_missing": [a.name for a in arms if a.name not in present],
        "reference_arm": ref,
        "bootstrap": {**boot, "unit": "box (seeds averaged within a box first)",
                      "paired": "comparisons against the reference are paired on the box"},
        "primary_metrics": [m.name for m in PRIMARY_METRICS],
        "per_arm": per_arm,
        "decisions": decisions,
        "recommendation": recommendation,
        "scope": (
            "This report chooses a constraint (alpha_disp, alpha_vel) only. The "
            "fields it summarises were scored and discarded; no training example "
            "was produced from them."
        ),
    }
    write_json(out_dir / "projection_oracle_report.json", report)
    _write_csv(out_dir / "projection_oracle_arms.csv", per_arm, ref)
    _write_rows_csv(out_dir / "projection_oracle_rows.csv", rows)
    _write_markdown(out_dir / "projection_oracle_report.md", report)

    banner("projection oracle report")
    print(f"  arms present : {len(present)} ({len(rows)} rows)")
    for sweep, d in decisions.items():
        rec = d["recommended"]
        print(f"  {sweep:10s} -> alpha={rec['alpha']:g} ({rec['arm']})"
              + ("  [hard null REJECTED]" if d["hard_null_rejected"] else ""))
    print(f"  recommendation: alpha_disp={recommendation['alpha_disp']:g} "
          f"alpha_vel={recommendation['alpha_vel']:g} "
          f"mode={recommendation['correction_mode']}")
    print(f"  -> {out_dir}/projection_oracle_report.{{json,md}}, *.csv")


# --------------------------------------------------------------------------- #
def _write_csv(path: Path, per_arm: Dict, ref: str) -> None:
    """One row per arm: mean, CI and the verdict against the reference."""
    metric_names = [m.name for m in REPORTED]
    header = ["arm", "sweep", "alpha_disp", "alpha_vel", "n_rows", "n_boxes"]
    for m in metric_names:
        header += [f"{m}", f"{m}_lo", f"{m}_hi", f"{m}_vs_ref", f"{m}_verdict"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for name, e in per_arm.items():
            first = e["metrics"].get(metric_names[0], {})
            row = [name, e["sweep"], e["alpha_disp"], e["alpha_vel"], e["n_rows"],
                   first.get("n_boxes", 0)]
            for m in metric_names:
                d = e["metrics"].get(m, {})
                cmp = e.get("vs_reference", {}).get(m, {})
                row += [d.get("mean"), d.get("lo"), d.get("hi"),
                        cmp.get("diff", {}).get("mean"), cmp.get("verdict", "")]
            w.writerow(row)


def _write_rows_csv(path: Path, rows: Sequence[Dict]) -> None:
    """Every scored row, flat, so a plot can be redrawn without recomputation."""
    keep = ["arm", "sweep", "box", "seed", "alpha_disp", "alpha_vel",
            "R_occ_reliable", "R_occ", "R_cat", "R_abund",
            "n_hosts_full_box", "n_subs_full_box", "host_recovery_fraction",
            "subhalo_ratio_vs_hr", "host_match_score_median",
            "low_k_change", "lr_consistency_error", "coarse_mismatch_vs_hr",
            "coarse_mismatch_vs_hr_disp", "coarse_mismatch_vs_hr_vel",
            "coarse_mismatch_vs_lr", "displacement_power_error",
            "velocity_power_error", "density_power_error", "density_sigma_ratio",
            "density_pdf_l1"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keep, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keep})


def _write_markdown(path: Path, rep: Dict) -> None:
    R = rep["recommendation"]
    L: List[str] = []
    L.append(f"# Projection oracle -- `{rep['run_name']}`\n")
    L.append(f"{rep['n_rows']} scored rows over {len(rep['arms_present'])} arms. "
             f"Reference arm: `{rep['reference_arm']}` (alpha = 1, i.e. Psi_HR).\n")
    L.append(f"Uncertainty: {rep['bootstrap']['unit']}, "
             f"{rep['bootstrap']['n_boot']} resamples, "
             f"{100 * rep['bootstrap']['ci']:.0f}% CI; "
             f"{rep['bootstrap']['paired']}.\n")
    L.append("> " + rep["scope"] + "\n")

    L.append("## Recommendation\n")
    L.append(f"* **alpha_disp = {R['alpha_disp']:g}** "
             f"(from the `{R['source_sweeps']['alpha_disp']}` sweep)")
    L.append(f"* **alpha_vel  = {R['alpha_vel']:g}** "
             f"(from the `{R['source_sweeps']['alpha_vel']}` sweep)")
    L.append(f"* correction mode: `{R['correction_mode']}`")
    L.append(f"* hard null projection rejected: "
             f"**{'yes' if R['hard_null_rejected'] else 'no'}**\n")

    for sweep, d in rep["decisions"].items():
        L.append(f"### Sweep `{sweep}` (varying `{d['varying']}`)\n")
        if d["hard_null_rejected"]:
            L.append(f"`alpha = 0` is **damaged** on "
                     f"{', '.join('`' + m + '`' for m in d['hard_null_damaged_metrics'])}"
                     f" relative to `alpha = 1`, so the hard null projection is "
                     f"rejected for this sweep.\n")
        else:
            L.append("`alpha = 0` was not shown to damage any primary metric "
                     "(which is not the same as showing it is harmless -- check "
                     "`n_paired_boxes` below).\n")
        prim = d["metrics"]
        L.append("| alpha | arm | " + " | ".join(f"`{m}`" for m in prim) + " |")
        L.append("|---|---|" + "---|" * len(prim))
        for name, a in sorted(d["arms"].items(), key=lambda kv: kv[1]["alpha"]):
            cells = []
            for m in prim:
                c = a["metrics"][m]
                cells.append(f"{_fmt(c['diff']['mean'])} "
                             f"({VERDICT_MARK.get(c['verdict'], c['verdict'])})")
            L.append(f"| {a['alpha']:g} | `{name}` | " + " | ".join(cells) + " |")
        L.append("\nCells are `mean(arm - reference)` with the paired-bootstrap "
                 "verdict. `WORSE` means the CI excludes zero on the bad side.\n")
        rec = d["recommended"]
        L.append(f"Chosen: **alpha = {rec['alpha']:g}** (`{rec['arm']}`)"
                 + (f" -- {rec['note']}" if rec.get("note") else "") + "\n")

    L.append("## Per-arm summary\n")
    cols = ["R_occ_reliable", "host_recovery_fraction", "n_subs_full_box",
            "n_hosts_full_box", "density_power_error", "coarse_mismatch_vs_hr",
            "low_k_change"]
    L.append("| arm | a_dis | a_vel | " + " | ".join(f"`{c}`" for c in cols) + " |")
    L.append("|---|---|---|" + "---|" * len(cols))
    for name, e in rep["per_arm"].items():
        L.append(f"| `{name}` | {_fmt(e['alpha_disp'], 3)} | {_fmt(e['alpha_vel'], 3)} | "
                 + " | ".join(_ci(e["metrics"].get(c, {})) for c in cols) + " |")
    L.append("")

    ref_arm = rep["per_arm"].get(rep["reference_arm"], {})
    if "reconstruction_rel_rms_vs_hr_max" in ref_arm:
        L.append(f"Self-consistency: the `alpha = 1` arm reproduces `Psi_HR` to "
                 f"{_fmt(ref_arm['reconstruction_rel_rms_vs_hr_max'], 3)} relative "
                 f"RMS (it should be float32 round-off; anything larger means the "
                 f"projection identity `P_N + P_R = I` is not holding on real "
                 f"fields and no number above is trustworthy).\n")

    L.append("## Occupation by host bin\n")
    L.append("Mean over boxes with a box-bootstrap CI, per arm.\n")
    any_occ = next((e for e in rep["per_arm"].values() if e["occupation_per_host_bin"]),
                   None)
    if any_occ:
        bins = list(any_occ["occupation_per_host_bin"])
        L.append("| arm | " + " | ".join(f"`{b}`" for b in bins) + " |")
        L.append("|---|" + "---|" * len(bins))
        for name, e in rep["per_arm"].items():
            t = e["occupation_per_host_bin"]
            if t:
                L.append(f"| `{name}` | " + " | ".join(_ci(t.get(b, {})) for b in bins) + " |")
    else:
        L.append("_No occupation curves in the rows (halo finding was skipped)._")
    L.append("")

    if rep["arms_missing"]:
        L.append(f"## Missing arms\n\n`{'`, `'.join(rep['arms_missing'])}` produced no "
                 f"rows. Every conclusion above is conditional on the arms that "
                 f"did run.\n")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
