#!/usr/bin/env python
"""Redraw the progress-signal diagnostic from its JSONL rows -- no recompute.

Per box: the normalised score path (does the reward climb steadily or only jump
at the end?), the binding-margin trajectory (does the object become bound?), and
the scrambled-velocity control (is a hot clump discounted?).

    python scripts/reward/plot_progress_signal.py --boxes set8,set9
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


def _norm_paths(rows: List[Dict], kind: str, key: str):
    """Per-object path normalised to its own [min, max] over the path.

    Normalised by the *range*, not the endpoint delta: the straight-line morph
    overshoots (peaks mid-path), so dividing by ``y[-1]-y[0]`` blows up when the
    endpoints nearly coincide. Range-normalising shows the overshoot honestly --
    a curve that reaches 1 before ``t=1`` and comes back down.
    """
    ids = sorted({r["hr_sub_id"] for r in rows if r["kind"] == kind})
    ts = sorted({r["t"] for r in rows if r["kind"] == kind})
    curves = []
    for sid in ids:
        pr = sorted([r for r in rows if r["kind"] == kind and r["hr_sub_id"] == sid],
                    key=lambda r: r["t"])
        y = np.array([r[key] for r in pr])
        rng = y.max() - y.min()
        curves.append((y - y.min()) / rng if rng > 1e-12 else y * 0.0)
    if not curves:
        return np.array(ts), None, None, None
    C = np.vstack(curves)
    return (np.array(ts), np.median(C, 0),
            np.quantile(C, 0.25, 0), np.quantile(C, 0.75, 0))


def plot_box(box: str, path_rows: List[Dict], sum_rows: List[Dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    kind = "missing_target"

    # 1. normalised score path: steady climb (diagonal) vs late jump (hockey stick)
    for key, color, label in (("score_bind", "C3", "multiscale + binding"),
                              ("score_narrow", "C7", "single narrow scale")):
        t, med, lo, hi = _norm_paths(path_rows, kind, key)
        if med is None:
            continue
        axes[0].plot(t, med, "-o", color=color, ms=4, label=label)
        axes[0].fill_between(t, lo, hi, color=color, alpha=0.15)
    axes[0].plot([0, 1], [0, 1], ":", color="k", lw=1, label="steady climb")
    axes[0].set_xlabel("path $t$  (frozen SR2 $\\to$ HR)")
    axes[0].set_ylabel("score, normalised to path range")
    axes[0].set_title(f"{box}: does the reward guide the whole way?")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(alpha=0.3)

    # 2. virial-ratio trajectory (the gated quantity): O(1) is bound, >> 1 unbound
    ids = sorted({r["hr_sub_id"] for r in path_rows if r["kind"] == kind})
    ts = sorted({r["t"] for r in path_rows if r["kind"] == kind})
    V = []
    for sid in ids:
        pr = sorted([r for r in path_rows if r["kind"] == kind and r["hr_sub_id"] == sid],
                    key=lambda r: r["t"])
        V.append([r["virial_ratio"] for r in pr])
    if V:
        V = np.array(V)
        axes[1].plot(ts, np.median(V, 0), "-o", color="C0", ms=4)
        axes[1].fill_between(ts, np.quantile(V, 0.25, 0), np.quantile(V, 0.75, 0),
                             color="C0", alpha=0.15)
    axes[1].axhline(2.5, ls="--", color="k", lw=1, label="bound threshold")
    axes[1].set_xlabel("path $t$")
    axes[1].set_ylabel("virial ratio  $\\sigma_v^2 / (GM/R)$")
    axes[1].set_title("does the object become bound?")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # 3. scrambled-velocity control
    real = np.array([s["score_real_t1"] for s in sum_rows if s["kind"] == kind])
    scram = np.array([s["score_scram_t1"] for s in sum_rows if s["kind"] == kind])
    if real.size:
        axes[2].scatter(real, scram, s=20, color="C3")
        lim = float(max(real.max(), scram.max())) * 1.05
        axes[2].plot([0, lim], [0, lim], ":", color="k", lw=1, label="no discount")
        axes[2].set_xlim(0, lim)
        axes[2].set_ylim(0, lim)
    axes[2].set_xlabel("score, real HR velocities")
    axes[2].set_ylabel("score, scrambled (hot) velocities")
    axes[2].set_title("is a dense-but-unbound clump rejected?")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes", default="set8,set9")
    ap.add_argument("--out-name", default="progress_signal")
    args = ap.parse_args(argv)
    root = paths.subdir("audits", args.out_name)
    for box in [b.strip() for b in args.boxes.split(",") if b.strip()]:
        d = root / box
        pth, smy = d / "path.jsonl", d / "object_summary.jsonl"
        if pth.is_file() and smy.is_file():
            plot_box(box, read_jsonl(pth), read_jsonl(smy), d / "progress.png")
            print(f"wrote {d / 'progress.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
