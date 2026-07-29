#!/usr/bin/env python
"""Stage-1 summary: box-bootstrap CIs + matched-host visualisations.

Reads ``field_rows.jsonl``, ``halo_rows.jsonl``, ``match_rows.jsonl`` from a
Stage-1 run directory. Seeds are *not* treated as independent simulations —
CIs resample boxes only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _bootstrap_mean(per_box: np.ndarray, n_boot: int = 5000, seed: int = 0):
    from cosmo_sr.tts.bootstrap import bootstrap_ci
    return bootstrap_ci(
        per_box, n_boot=n_boot, alpha=0.05, rng=np.random.default_rng(seed)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1", default=str(ROOT / "runs/sr2_baseline/stage1"))
    ap.add_argument("--out", default=None, help="default: <stage1>/analyze")
    ap.add_argument("--seed0-only", action="store_true",
                    help="summarise seed=0 only (baseline failure, not TTS)")
    args = ap.parse_args()

    stage1 = Path(args.stage1)
    out = Path(args.out or stage1 / "analyze")
    out.mkdir(parents=True, exist_ok=True)

    field_rows = _load_jsonl(stage1 / "field_rows.jsonl")
    halo_rows = _load_jsonl(stage1 / "halo_rows.jsonl")
    match_rows = _load_jsonl(stage1 / "match_rows.jsonl")
    if args.seed0_only:
        field_rows = [r for r in field_rows if int(r.get("seed", -1)) == 0]
        match_rows = [r for r in match_rows if int(r.get("seed", -1)) == 0]
        halo_rows = [r for r in halo_rows
                     if r.get("tag") == "hr" or int(r.get("seed", -1)) == 0]

    # --- field controls (seed-averaged per box, then box bootstrap) ---------- #
    field_by_box: Dict[str, List[dict]] = defaultdict(list)
    for r in field_rows:
        field_by_box[r["box"]].append(r)
    field_keys = [
        "density_Pk_ratio_high", "density_Tk_high", "density_rk_high",
        "density_sigma_ratio", "density_pdf_l1", "bispectrum_eq_rel",
        "disp_Pk_ratio_high", "vel_Pk_ratio_high",
    ]
    field_summary = {}
    for key in field_keys:
        per_box = []
        for box, rows in sorted(field_by_box.items()):
            vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
            if vals:
                per_box.append(float(np.mean(vals)))
        if per_box:
            field_summary[key] = _bootstrap_mean(np.asarray(per_box))

    # --- halo counts -------------------------------------------------------- #
    hr_by_box = {r["box"]: r for r in halo_rows if r.get("tag") == "hr"}
    sr_by_box_seed = {(r["box"], int(r["seed"])): r
                     for r in halo_rows if r.get("tag") == "sr"}
    nsub_ratio = []
    for box, hr in hr_by_box.items():
        srs = [sr_by_box_seed[k] for k in sr_by_box_seed if k[0] == box]
        if not srs or hr["n_subs"] == 0:
            continue
        nsub_ratio.append(float(np.mean([s["n_subs"] for s in srs]) / hr["n_subs"]))
    halo_summary = {
        "nsub_sr_over_hr": _bootstrap_mean(np.asarray(nsub_ratio)) if nsub_ratio else None,
        "n_hr_boxes": len(hr_by_box),
    }

    # --- match classification ---------------------------------------------- #
    class_frac_by_box: Dict[str, Dict[str, float]] = {}
    for r in match_rows:
        n = max(int(r.get("n_hr_subs_classified", 0)), 1)
        cc = r.get("class_counts", {})
        fr = {k: cc.get(k, 0) / n for k in
              ("recovered", "spatially_shifted", "recovered_biased",
               "merged_into_host", "missing")}
        class_frac_by_box.setdefault(r["box"], []).append(fr)
    class_summary = {}
    for label in ("recovered", "spatially_shifted", "recovered_biased",
                  "merged_into_host", "missing"):
        per_box = []
        for box, frs in class_frac_by_box.items():
            per_box.append(float(np.mean([f[label] for f in frs])))
        if per_box:
            class_summary[label] = _bootstrap_mean(np.asarray(per_box))

    # Dominant failure mode from seed-0 (or mean) missing+merged vs shifted
    def _m(label):
        return class_summary.get(label, {}).get("mean", 0.0)

    modes = {
        "failure_to_form_peaks": _m("missing"),
        "excessive_merging": _m("merged_into_host"),
        "inaccurate_positions": _m("spatially_shifted"),
        "biased_mass_vmax": _m("recovered_biased"),
    }
    dominant = max(modes, key=modes.get) if modes else "unknown"

    report = {
        "stage1": str(stage1.resolve()),
        "seed0_only": bool(args.seed0_only),
        "field": field_summary,
        "halo": halo_summary,
        "classification": class_summary,
        "failure_modes": modes,
        "dominant_failure_mode": dominant,
        "note": (
            "CIs are box-level bootstrap. Seeds are averaged within a box before "
            "resampling. z=2 deferred."
        ),
    }
    with open(out / "stage1_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    # Markdown
    lines = [
        "# Stage-1 subhalo failure report (z=0)",
        "",
        f"Dominant failure mode: **{dominant}**",
        "",
        "## Field controls (box bootstrap mean [95% CI])",
        "",
    ]
    for k, v in field_summary.items():
        lines.append(f"- `{k}`: {v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] (n={v['n_boxes']})")
    lines += ["", "## Subhalo classification fractions", ""]
    for k, v in class_summary.items():
        lines.append(f"- `{k}`: {v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")
    if halo_summary.get("nsub_sr_over_hr"):
        v = halo_summary["nsub_sr_over_hr"]
        lines += ["", f"N_sub(SR)/N_sub(HR): {v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"]
    (out / "stage1_report.md").write_text("\n".join(lines) + "\n")

    # Optional host viz from first match row with records
    _maybe_plot_hosts(match_rows, out)

    print("\n".join(lines))
    print(f"\nWrote {out}/stage1_report.{{json,md}}")


def _maybe_plot_hosts(match_rows: List[dict], out: Path, n_hosts: int = 4):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    # Pick the first box/seed and show class pie + dx vs mass for missed/recovered
    if not match_rows:
        return
    row = match_rows[0]
    recs = row.get("records", [])
    if not recs:
        return
    counts = Counter(r["class"] for r in recs)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels, sizes = zip(*sorted(counts.items())) if counts else ([], [])
    if sizes:
        axes[0].pie(sizes, labels=labels, autopct="%1.0f%%")
    axes[0].set_title(f"{row['box']} seed={row['seed']}: HR subhalo classes")
    xs = [r.get("r_rvir", np.nan) for r in recs]
    ys = [r.get("mass_ratio", np.nan) for r in recs]
    cols = [{"recovered": "C0", "spatially_shifted": "C1", "recovered_biased": "C2",
             "merged_into_host": "C3", "missing": "C4"}.get(r["class"], "k") for r in recs]
    axes[1].scatter(xs, ys, c=cols, s=12, alpha=0.8)
    axes[1].axhline(1, color="k", lw=0.6)
    axes[1].set_xlabel(r"$r / R_{\rm vir}$ (HR)")
    axes[1].set_ylabel(r"$M_{\rm SR}/M_{\rm HR}$ (matched)")
    axes[1].set_title("matched subs (colour = class)")
    fig.tight_layout()
    fig.savefig(out / "class_overview.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
