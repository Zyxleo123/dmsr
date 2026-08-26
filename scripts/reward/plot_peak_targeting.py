#!/usr/bin/env python
"""Redraw the peak-targeting diagnostic from its JSONL rows -- no recompute.

Per box: whether a DoG density peak marks each missing subhalo (proximity vs
threshold), the peak-vs-subhalo distance, and whether peak-only centres recover
the deployable score.

    python scripts/reward/plot_peak_targeting.py --boxes set8,set9
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import paths, read_jsonl  # noqa: E402


def plot_box(box: str, prox: List[Dict], cond: List[Dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    thrs = sorted({r["threshold"] for r in prox})

    # 1. proximity vs threshold (with peak count on a twin axis)
    fw = [np.mean([r["peak_within_rref"] for r in prox if r["threshold"] == t]) for t in thrs]
    fh = [np.mean([r["peak_within_half_rref"] for r in prox if r["threshold"] == t]) for t in thrs]
    fc = [np.mean([r["peak_closer_than_sub"] for r in prox if r["threshold"] == t]) for t in thrs]
    npk = [prox_n(prox, t) for t in thrs]
    axes[0].plot(thrs, fw, "-o", color="C0", label="peak within $R_{ref}$")
    axes[0].plot(thrs, fh, "-o", color="C2", label="within $0.5 R_{ref}$")
    axes[0].plot(thrs, fc, "-o", color="C3", label="closer than nearest sub")
    axes[0].set_xlabel("DoG residual threshold")
    axes[0].set_ylabel("fraction of missing subhalos")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title(f"{box}: does a peak mark the missing subhalo?")
    axes[0].legend(fontsize=8, loc="center right")
    axes[0].grid(alpha=0.3)
    ax2 = axes[0].twinx()
    ax2.plot(thrs, npk, ":", color="grey")
    ax2.set_ylabel("n peaks (grey, dotted)", color="grey")
    ax2.set_yscale("log")

    # 2. nearest-peak vs nearest-sub distance, per threshold
    dp = [np.median([r["nearest_peak_mpc_h"] for r in prox if r["threshold"] == t]) for t in thrs]
    ds = [np.median([r["nearest_sub_mpc_h"] for r in prox if r["threshold"] == t]) for t in thrs]
    rref = np.median([r["r_ref_mpc_h"] for r in prox])
    axes[1].plot(thrs, dp, "-o", color="C0", label="nearest DoG peak")
    axes[1].plot(thrs, ds, "-s", color="C1", label="nearest frozen-SR2 sub")
    axes[1].axhline(rref, ls="--", color="k", lw=1, label="median $R_{ref}$")
    axes[1].set_xlabel("DoG residual threshold")
    axes[1].set_ylabel("median distance to missing subhalo [Mpc/h]")
    axes[1].set_title("peak vs existing subhalo proximity")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # 3. scoring recovery: A oracle vs C peak-only
    conds = ["A_oracle", "C_peak"]
    labels = ["A: oracle", "C: peak-only"]
    x = np.arange(len(conds))
    for i, (key, color) in enumerate((("hr_gt_sr2_bind", "C3"),
                                      ("bound_t1", "C0"))):
        vals = [np.mean([r[key] for r in cond if r["condition"] == c])
                if any(r["condition"] == c for r in cond) else 0.0 for c in conds]
        axes[2].bar(x + (i - 0.5) * 0.35, vals, 0.35,
                    label={"hr_gt_sr2_bind": "HR>SR2 (bind)",
                           "bound_t1": "bound @HR"}[key], color=color)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("fraction of objects")
    axes[2].set_title("does a peak centre recover the signal?")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def prox_n(prox: List[Dict], t: float) -> int:
    for r in prox:
        if r["threshold"] == t:
            return int(r["n_peaks"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes", default="set8,set9")
    ap.add_argument("--out-name", default="peak_targeting")
    args = ap.parse_args(argv)
    root = paths.subdir("audits", args.out_name)
    for box in [b.strip() for b in args.boxes.split(",") if b.strip()]:
        d = root / box
        pp, cp = d / "proximity.jsonl", d / "conditions.jsonl"
        if pp.is_file() and cp.is_file():
            plot_box(box, read_jsonl(pp), read_jsonl(cp), d / "peak_targeting.png")
            print(f"wrote {d / 'peak_targeting.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
