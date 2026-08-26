#!/usr/bin/env python
"""Redraw the deployable-partition diagnostic from its JSONL rows -- no recompute.

Per box: how the formation signal degrades as the oracle is removed
(A oracle -> B contaminated bag -> C deployable centre), and whether the score
still rises where the deployable centre actually lands near real structure.

    python scripts/reward/plot_deployable_partition.py --boxes set8,set9
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

CONDS = ["A_oracle", "B_contam_bag", "C_deployable"]
LABELS = ["A: oracle", "B: contam bag", "C: deployable"]


def plot_box(box: str, rows: List[Dict], out: Path) -> None:
    kind = "missing_target"
    sel = [r for r in rows if r["kind"] == kind]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. degradation of the pass metrics across conditions
    metrics = [("hr_gt_sr2_bind", "HR>SR2 (bind)", "C3"),
               ("bound_t1", "bound @HR", "C0"),
               ("hr_gt_sr2_dens", "HR>SR2 (dens)", "C2")]
    x = np.arange(len(CONDS))
    w = 0.25
    for i, (key, label, color) in enumerate(metrics):
        vals = [np.mean([r[key] for r in sel if r["condition"] == c]) if
                any(r["condition"] == c for r in sel) else 0.0 for c in CONDS]
        axes[0].bar(x + (i - 1) * w, vals, w, label=label, color=color)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(LABELS, fontsize=8)
    axes[0].set_ylabel("fraction of objects")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"{box}: does the signal survive removing the oracle?")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].grid(alpha=0.3, axis="y")

    # 2. scrambled-velocity rejection across conditions (median ratio; <1 good)
    sr = [np.median([r["scram_ratio"] for r in sel if r["condition"] == c])
          if any(r["condition"] == c for r in sel) else np.nan for c in CONDS]
    axes[1].bar(x, sr, 0.5, color="C4")
    axes[1].axhline(1.0, ls="--", color="k", lw=1, label="no discount")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LABELS, fontsize=8)
    axes[1].set_ylabel("median scram / real score  @HR")
    axes[1].set_title("hot-clump rejection under deployment")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    # 3. condition C: does the score rise where the centre lands near structure?
    c = [r for r in sel if r["condition"] == "C_deployable"]
    if c:
        off = np.array([r["center_offset_mpc_h"] for r in c])
        rise = np.array([r["score_bind_t1"] - r["score_bind_t0"] for r in c])
        up = rise > 0
        axes[2].scatter(off[up], rise[up], s=24, color="C0", label="HR>SR2")
        axes[2].scatter(off[~up], rise[~up], s=24, color="C3", label="HR<SR2")
        axes[2].axhline(0, ls="--", color="k", lw=1)
    axes[2].set_xlabel("deployable centre offset from true [Mpc/h]")
    axes[2].set_ylabel("score(HR) - score(SR2), binding")
    axes[2].set_title("C: signal vs centre offset")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes", default="set8,set9")
    ap.add_argument("--out-name", default="deployable_partition")
    args = ap.parse_args(argv)
    root = paths.subdir("audits", args.out_name)
    for box in [b.strip() for b in args.boxes.split(",") if b.strip()]:
        d = root / box
        rp = d / "conditions.jsonl"
        if rp.is_file():
            plot_box(box, read_jsonl(rp), d / "deployable.png")
            print(f"wrote {d / 'deployable.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
