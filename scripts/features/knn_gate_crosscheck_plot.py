#!/usr/bin/env python
"""Figure for knn_gate_crosscheck.py. Reads its JSON only.

Top: every gated arm's region 50-200p subhalos, split into HR-matched (solid) and
unmatched (light), with fragment candidates marked -- is a count gain HR's
objects or something else? Sorted by the raw count; HR and frozen outlined.

Bottom: across the arms that have kNN terms, each kNN term's change vs frozen
against each gate metric's change vs frozen, with Spearman rho. A kNN term is a
LOSS: tracking a good metric (matched, recall) shows as rho < 0, tracking a bad
one (fragments, lost hosts) as rho > 0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3dc"
SOLID, LIGHT, MARK = "#2a78d6", "#9ec5f4", "#eb6834"
KNN = ["knn_class_cond[sub_50_200]", "knn_class_cond[host]", "knn_class_cond_equal",
       "knn_paired"]
GATE = ["sub_50_200", "sub_50_200_matched", "hr_recall_50_200",
        "fragment_candidates", "lost_ge200"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.labelcolor": INK, "savefig.facecolor": "white"})
    r = json.loads(Path(a.json).read_text())
    rows = r["rows"]
    frozen = r["frozen_tag"]
    fig = plt.figure(figsize=(22, 6 + 3.2 * len(KNN)), layout="constrained")
    gs = fig.add_gridspec(1 + len(KNN), len(GATE), height_ratios=[1.6] + [1] * len(KNN))

    ax = fig.add_subplot(gs[0, :])
    order = sorted(rows, key=lambda x: x["sub_50_200"])
    xs = np.arange(len(order))
    m = np.array([x["sub_50_200_matched"] for x in order])
    u = np.array([x["sub_50_200"] - x["sub_50_200_matched"] for x in order])
    ax.bar(xs, m, color=SOLID, width=0.8, label="matched to an HR halo")
    ax.bar(xs, u, bottom=m, color=LIGHT, width=0.8, label="unmatched")
    ax.plot(xs, [x["fragment_candidates"] for x in order], "D", color=MARK, ms=6,
            label="fragment candidates (unmatched, inside a lost frozen >=200p halo)")
    for i, x in enumerate(order):
        if x["tag"] in ("hr", frozen):
            ax.bar(i, x["sub_50_200"], fill=False, edgecolor=INK, lw=2, width=0.8)
    ax.set_xticks(xs, [x["label"] for x in order], rotation=60, ha="right")
    ax.set(ylabel="region subhalos, 50-200p",
           title=f"Gated set9 arms: whose subhalos are HR's?  (region = {len(r['region_tiles'])} "
                 f"spliced tiles + {r['dilate']} Mpc/h; match = max(r_vir,0.15) & num_p x{r['mass_factor']})")
    ax.legend(frameon=False)
    ax.grid(axis="y", color=GRID, lw=0.6)

    base = next((x for x in rows if x["tag"] == frozen), None)
    have = [x for x in rows if x["tag"] not in (frozen, "hr") and KNN[0] in x]
    for i, kk in enumerate(KNN):
        for j, gk in enumerate(GATE):
            ax = fig.add_subplot(gs[1 + i, j])
            if base is None or kk not in base or not have:
                ax.axis("off")
                continue
            xv = np.array([x[kk] - base[kk] for x in have])
            yv = np.array([x[gk] - base[gk] for x in have], dtype=float)
            ax.axhline(0, color=GRID, lw=1)
            ax.axvline(0, color=GRID, lw=1)
            ax.plot(xv, yv, "o", color=SOLID, ms=7)
            for x0, y0, x in zip(xv, yv, have):
                ax.annotate(x["label"].replace("all_blocks_", ""), (x0, y0),
                            fontsize=6, color=INK2, xytext=(3, 2),
                            textcoords="offset points")
            c = r["spearman"].get(kk, {}).get(gk)
            ax.set_title(f"rho={c['rho']:+.2f} (p={c['p']:.2g}, n={c['n']})" if c else "n/a",
                         fontsize=9)
            if i == len(KNN) - 1:
                ax.set_xlabel(f"Δ {kk.replace('knn_', '')}")
            if j == 0:
                ax.set_ylabel(f"{kk.replace('knn_', '')}\nΔ gate metric")
            if i == 0:
                ax.text(0.5, 1.18, f"Δ {gk}", transform=ax.transAxes, ha="center",
                        fontsize=10, weight="bold")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"=== wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
