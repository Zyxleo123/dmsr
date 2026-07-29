#!/usr/bin/env python
"""CPU audit of the paired residual targets, before any GPU training.

The point is to catch normalization and periodic-coordinate mistakes while they
are still cheap. A residual whose power is mostly *below* the LR Nyquist means
the baseline is misaligned or mis-normalised, not that the residual is
interesting -- SR2 already reproduces those scales.

Reports per box: per-channel mean/std, residual RMS relative to HR and to SR2,
power spectra of HR / SR2 / residual, residual power in the low, transition and
high k bands, the fraction of residual power below the LR Nyquist, the
distribution of the maximum residual magnitude, and (when several redshifts are
available) the same table per redshift.

    python scripts/reward/audit_residual_targets.py --boxes set0,set1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _common import (add_common_args, banner, hr_path, load_reward_config,
                     parse_boxes, write_json)

from cosmo_sr.reward import paths
from cosmo_sr.reward.targets import residual_stats, resolve_boxes


def _verdict(rows) -> dict:
    """Blunt pass/fail read of the audit, so nobody has to eyeball the JSON."""
    flags = []
    fracs, rel = [], []
    for r in rows:
        for ch, sp in r["stats"]["spectra"].items():
            fracs.append(sp["frac_residual_power_below_lr_nyquist"])
        rel.extend(r["stats"]["residual_rms_over_hr"][0:3])
    f = float(np.mean(fracs)) if fracs else float("nan")
    rr = float(np.mean(rel)) if rel else float("nan")
    if not np.isfinite(f):
        flags.append("no spectra computed")
    elif f > 0.5:
        flags.append(
            f"{100 * f:.0f}% of residual power is below the LR Nyquist -- the "
            "residual is dominated by scales SR2 already reproduces; suspect a "
            "normalization or alignment bug, not a modelling opportunity"
        )
    if np.isfinite(rr) and rr > 0.5:
        flags.append(
            f"residual RMS is {100 * rr:.0f}% of HR RMS -- far too large for a "
            "correction on top of a working baseline"
        )
    return {
        "mean_frac_residual_power_below_lr_nyquist": f,
        "mean_residual_rms_over_hr_disp": rr,
        "flags": flags,
        "pass": len(flags) == 0,
    }


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--boxes", default=None, help="comma list; default = train split")
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "dev", "final"])
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--channels", default="0,1,2", help="channels to spectrum")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_reward_config(args)
    boxes = parse_boxes(args.boxes, cfg, args.split)
    out = Path(args.out) if args.out else paths.AUDITS("residual_target_audit", create=True)
    out.mkdir(parents=True, exist_ok=True)
    channels = tuple(int(c) for c in args.channels.split(","))
    scale = int(cfg["data"]["scale_factor"])
    z = float(cfg["data"].get("redshift", 0.0))

    bps = resolve_boxes(boxes, cfg["data"]["root"], base_dir=args.base_dir,
                        base_seed=args.base_seed)
    missing = [b.box for b in bps if b.base is None]
    if missing:
        raise SystemExit(
            f"no cached SR2 base for {missing}. Run:\n"
            f"  sbatch scripts/slurm/cache_sr2_base.sbatch"
        )

    banner(f"residual-target audit: {len(bps)} boxes -> {out}")
    rows = []
    for bp in bps:
        t0 = time.time()
        hr = np.load(bp.hr, mmap_mode="r")
        base = np.load(bp.base, mmap_mode="r")
        st = residual_stats(hr, base, scale_factor=scale, n_bins=args.n_bins,
                            channels=channels)
        rows.append({"box": bp.box, "redshift": z, "base": str(bp.base),
                     "seconds": time.time() - t0, "stats": st})
        d = st["spectra"][f"ch{channels[0]}"]
        print(
            f"[{bp.box}] rms/HR={st['residual_rms_over_hr'][0]:.4f} "
            f"below-LR-Nyq={d['frac_residual_power_below_lr_nyquist']:.3f} "
            f"P_high/P_low={d['residual_power_high_k'] / max(d['residual_power_low_k'], 1e-30):.3g} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    report = {
        "boxes": boxes,
        "base_seed": int(args.base_seed),
        "scale_factor": scale,
        "redshifts_present": sorted({r["redshift"] for r in rows}),
        "per_box": rows,
        "verdict": _verdict(rows),
    }
    write_json(out / "residual_target_audit.json", report)

    # Compact per-redshift comparison table.
    by_z = {}
    for r in rows:
        by_z.setdefault(r["redshift"], []).append(r)
    lines = ["# Residual target audit", "",
             "| z | boxes | rms/HR (disp) | rms/SR2 (disp) | frac P below LR Nyq | |dPsi|max p99 |",
             "|---|---|---|---|---|---|"]
    for zz, rs in sorted(by_z.items()):
        rr = float(np.mean([np.mean(r["stats"]["residual_rms_over_hr"][0:3]) for r in rs]))
        rb = float(np.mean([np.mean(r["stats"]["residual_rms_over_base"][0:3]) for r in rs]))
        fb = float(np.mean([
            r["stats"]["spectra"][f"ch{channels[0]}"]["frac_residual_power_below_lr_nyquist"]
            for r in rs
        ]))
        mx = float(np.mean([
            r["stats"]["residual_absmax_slab_distribution"]["p99"] for r in rs
        ]))
        lines.append(f"| {zz} | {len(rs)} | {rr:.4f} | {rb:.4f} | {fb:.3f} | {mx:.4g} |")
    lines += ["", "## Verdict", ""]
    v = report["verdict"]
    lines.append(f"**{'PASS' if v['pass'] else 'FAIL'}**")
    for f in v["flags"]:
        lines.append(f"* {f}")
    (out / "residual_target_audit.md").write_text("\n".join(lines) + "\n")

    # sigma_res for the diffusion model: per-channel residual std over all boxes.
    sigma = np.mean([r["stats"]["residual_std"] for r in rows], axis=0)
    write_json(out / "sigma_res.json", {
        "sigma_res": sigma.tolist(),
        "boxes": boxes,
        "note": "per-channel residual std; whitening scale for the diffusion model",
    })
    banner(f"verdict: {'PASS' if report['verdict']['pass'] else 'FAIL'}")
    for f in report["verdict"]["flags"]:
        print(f"  ! {f}", flush=True)


if __name__ == "__main__":
    main()
