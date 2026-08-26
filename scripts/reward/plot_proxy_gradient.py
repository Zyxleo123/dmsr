#!/usr/bin/env python
"""Render the proxy-gradient-vs-SR2-output diagnostic. CPU, redrawable.

Reads ``gradient/proxy_grad_<arm>_<tag>.{npz,json}`` written by
``proxy_gradient_diagnostic.py`` and draws three panels per figure:

  1. ``g = dQ/d(cand)`` vs the SR2 output value, one subplot per output channel,
     as a 2-D density (many voxels overlap) with the pooled OLS fit drawn on top
     and its fixed point ``c0 = -a/b`` marked. A downward line crossing zero is
     an ATTRACTOR: gradient ascent on Q pushes every voxel of that channel toward
     one value c0 -- points converging. A flat cloud at g~=0 is a DEAD gradient
     (the proxy gives the field nothing to follow).
  2. The per-unit fixed points c0 across every (tile, seed), one strip per
     channel. If they collapse onto a single c0 the proxy pulls DIFFERENT
     candidates to the SAME point -- convergence/mode-collapse -- which is the
     candidate-collapse story seen from the actor's side.
  3. Dead-voxel fraction and |g| per channel: how much of the field the gradient
     actually touches.

    python scripts/reward/plot_proxy_gradient.py --run-name direct_a --arm c
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _sr2_direct import run_dir  # noqa: E402

# Okabe-Ito: the standard CVD-safe categorical order. disp_{x,y,z} then vel_{x,y,z}.
OKABE = ["#0072B2", "#56B4E9", "#009E73", "#D55E00", "#E69F00", "#CC79A7"]
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--arm", default="c")
    ap.add_argument("--tag", default="frozen", help="frozen | rung_<x> | overfit ...")
    ap.add_argument("--config", default="configs/reward/sr2_direct_finetune.yaml")
    args = ap.parse_args(argv)

    run = run_dir(args.run_name)
    gdir = run / "gradient"
    npz_p = gdir / f"proxy_grad_{args.arm}_{args.tag}.npz"
    json_p = gdir / f"proxy_grad_{args.arm}_{args.tag}.json"
    if not npz_p.is_file():
        print(f">>> MISSING INPUT: {npz_p}")
        print(">>> produced by: scripts/reward/proxy_gradient_diagnostic.py")
        return 0
    d = np.load(npz_p, allow_pickle=True)
    summ = json.loads(json_p.read_text())
    chan_names = [str(c) for c in d["chan_names"]]
    cand, grad, chan = d["cand"], d["grad"], d["chan"]
    agg = summ["channel_aggregate"]
    fits = summ["fits"]

    _style()

    # ---- Figure 1: g vs cand, per channel, with the OLS attractor line -----
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ci, (ax, cname) in enumerate(zip(axes.ravel(), chan_names)):
        m = chan == ci
        x, y = cand[m], grad[m]
        if x.size:
            # robust axis limits so a few outliers do not flatten the cloud
            xl = np.percentile(x, [0.5, 99.5])
            yl = np.percentile(y, [0.5, 99.5])
            hb = ax.hexbin(x, y, gridsize=45, cmap="Blues", mincnt=1,
                           extent=(xl[0], xl[1], yl[0], yl[1]), linewidths=0)
            a = agg[cname]
            b, a0 = a["median_slope"], None
            # pooled OLS on the subsample for the drawn line (matches per-unit sign)
            if np.std(x) > 0:
                bb, aa = np.polyfit(x.astype(np.float64), y.astype(np.float64), 1)
                xs = np.linspace(xl[0], xl[1], 100)
                ax.plot(xs, bb * xs + aa, color=OKABE[3], lw=2,
                        label=f"slope {bb:+.2e}")
                c0 = -aa / bb if abs(bb) > 1e-30 else np.nan
                if xl[0] <= c0 <= xl[1]:
                    ax.axvline(c0, color=OKABE[4], lw=1.5, ls="--")
                    ax.annotate(f"c0={c0:+.2f}", (c0, yl[1]), color=OKABE[4],
                                fontsize=8, ha="center", va="top")
            ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(f"{cname}   dead {agg[cname]['median_dead_frac']:.0%}",
                     color=INK, fontsize=10)
        if ci >= 3:
            ax.set_xlabel("SR2 output value")
        if ci % 3 == 0:
            ax.set_ylabel(r"$\partial Q/\partial\,$cand")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(
        f"Proxy gradient into the SR2 output  |  run {args.run_name}  arm {args.arm}"
        f"  ({args.tag})\n"
        f"down-sloping line crossing 0 = attractor (voxels pulled to c0); "
        f"flat cloud at 0 = dead gradient", fontsize=12, color=INK)
    f1 = gdir / f"proxy_grad_{args.arm}_{args.tag}_scatter.png"
    fig.savefig(f1, dpi=130)
    plt.close(fig)

    # ---- Figure 2 + 3: fixed-point clustering and gradient reach -----------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    for ci, cname in enumerate(chan_names):
        fps = np.array([r["fixed_point"] for r in fits
                        if r["channel"] == cname], dtype=np.float64)
        fps = fps[np.isfinite(fps)]
        if fps.size:
            jit = (np.random.default_rng(ci).random(fps.size) - 0.5) * 0.6
            axL.scatter(fps, np.full(fps.size, ci) + jit, s=14, alpha=0.6,
                        color=OKABE[ci], edgecolors="none")
            axL.scatter([np.median(fps)], [ci], marker="|", s=400,
                        color=INK, linewidths=2, zorder=5)
    axL.set_yticks(range(len(chan_names)))
    axL.set_yticklabels(chan_names)
    axL.set_xlabel("per-unit attractor fixed point $c_0=-a/b$")
    axL.set_title("Do different candidates converge to one point?\n"
                  "tight column = collapse to a shared c0", fontsize=10)
    xall = np.array([r["fixed_point"] for r in fits], dtype=np.float64)
    xall = xall[np.isfinite(xall)]
    if xall.size:
        axL.set_xlim(*np.percentile(xall, [2, 98]))

    # gradient reach: dead fraction (bar) with |g| annotated
    dead = [agg[c]["median_dead_frac"] for c in chan_names]
    gmag = [agg[c]["median_grad_abs_mean"] for c in chan_names]
    ypos = np.arange(len(chan_names))
    axR.barh(ypos, dead, color=[OKABE[i] for i in range(len(chan_names))],
             height=0.6)
    for i, (dfrac, gm) in enumerate(zip(dead, gmag)):
        axR.annotate(f"|g|={gm:.1e}", (min(dfrac + 0.02, 0.98), i),
                     va="center", ha="left", fontsize=8, color=MUTED)
    axR.set_yticks(ypos)
    axR.set_yticklabels(chan_names)
    axR.set_xlim(0, 1)
    axR.set_xlabel("fraction of voxels with |g| < 1% of channel max (dead)")
    axR.set_title("How much of the field the gradient touches", fontsize=10)
    axR.invert_yaxis()

    neg = np.mean([agg[c]["frac_negative_slope"] for c in chan_names])
    fig.suptitle(
        f"run {args.run_name}  arm {args.arm} ({args.tag})  |  "
        f"mean frac attractor(neg slope)={neg:.2f}  |  n_units={summ['n_units']}",
        fontsize=12, color=INK)
    f2 = gdir / f"proxy_grad_{args.arm}_{args.tag}_convergence.png"
    fig.savefig(f2, dpi=130)
    plt.close(fig)

    print(f"wrote {f1}")
    print(f"wrote {f2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
