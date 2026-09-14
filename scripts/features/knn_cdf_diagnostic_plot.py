#!/usr/bin/env python
"""Figures for ``knn_cdf_diagnostic.py``. Reads its JSON only; no recomputation.

Row 1, one panel per ``k``: each field's ratio to HR of ``r_k``, quantile bin by
quantile bin. Left edge = the most clumped particles, where the signal lives;
1.0 is HR; the grey band is the loss's dead zone (charged outside it). Above 1 =
neighbours further away than HR = under-clumped.

Row 2: the boundary experiment -- incomplete neighbour lists and p95 overestimate
of ``r_kmax`` against the buffer, with the dead zone marked, so the buffer the
trainer should use can be read off -- and a table of the numbers a verdict quotes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Reference categorical palette (dataviz palette.md, light), fixed order, by
# entity: a field keeps its colour whichever fields are present.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
         "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3dc"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True, help="png path")
    ap.add_argument("--order", nargs="*", default=None,
                    help="field labels in colour-slot order (default: JSON order)")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK,
                         "savefig.facecolor": "white"})

    res = json.loads(Path(a.json).read_text())
    ks = res["ks"]
    dz = float(res["dead_zone"])
    fields = [f for f in (a.order or list(res["fields"]))
              if f in res["fields"] and "ratio_bins" in res["fields"][f]]
    colour = {f: SLOTS[i % len(SLOTS)] for i, f in enumerate(fields)}

    fig = plt.figure(figsize=(4.2 * len(ks), 9.5), layout="constrained")
    gs = fig.add_gridspec(2, len(ks))
    for j, k in enumerate(ks):
        ax = fig.add_subplot(gs[0, j])
        ax.axhspan(1 / (1 + dz), 1 + dz, color=GRID, zorder=0, lw=0)
        ax.axhline(1.0, color=INK2, lw=1, zorder=1)
        for f in fields:
            rb = np.asarray(res["fields"][f]["ratio_bins"])[:, j]
            x = (np.arange(len(rb)) + 0.5) / len(rb)
            ax.plot(x, rb, color=colour[f], lw=2, label=f, zorder=2)
        ax.set_yscale("log")
        from matplotlib.ticker import FormatStrFormatter, LogLocator
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=np.arange(1, 10) * 0.1,
                                              numticks=20))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_minor_formatter(FormatStrFormatter(""))
        ax.set(xlabel="quantile of $r_k$ (0 = most clumped)",
               title=f"k = {k}", xlim=(0, 1))
        if j == 0:
            ax.set_ylabel("$r_k$ / $r_k^{HR}$  (>1 under-clumped)")
        ax.grid(axis="y", color=GRID, lw=0.6)
    fig.axes[0].legend(frameon=False, fontsize=9, loc="upper right")

    b = res.get("boundary")
    ax = fig.add_subplot(gs[1, 0])
    if b:
        recs = [r for r in b["records"] if r.get("n_queries")]
        bx = [r["buffer_cells"] for r in recs]
        ax.plot(bx, [100 * r["rel_err_p95"] for r in recs], "o-", color=SLOTS[0],
                lw=2, ms=8, label="p95 overestimate of $r_{kmax}$")
        ax.plot(bx, [100 * r["incomplete_frac"] for r in recs], "s-",
                color=SLOTS[1], lw=2, ms=8, label="queries missing a neighbour")
        ax.axhline(100 * dz, color=INK2, ls="--", lw=1)
        ax.text(bx[-1], 100 * dz, " dead zone", va="bottom", ha="right",
                color=INK2, fontsize=9)
        ax.set(xlabel="buffer [cells]", ylabel="percent",
               title=f"Tile boundary, HR, k_max={b['k_max']}, {b['n_tiles']} tiles")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6)

    tab = fig.add_subplot(gs[1, 1:])
    tab.axis("off")
    rows = []
    for f in fields:
        r = res["fields"][f]
        rows.append([f, *[f"{v:.3f}" for v in r["ratio_q10"]],
                     *[f"{v:.2f}" for v in r["clumped_recall"]],
                     f"{r['loss_two_sided']:.4f}"])
    cols = (["field"] + [f"q10 ratio\nk={k}" for k in ks]
            + [f"recall\nk={k}" for k in ks] + ["loss"])
    if rows:
        t = tab.table(cellText=rows, colLabels=cols, loc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1, 2.0)
    tab.set_title("q10 ratio: bottom-decile $r_k$ vs HR (1 = matched).\n"
                  "recall: share of HR's bottom-decile particles also bottom-decile "
                  "in the field (chance = 0.10, identical = 1.00).", fontsize=10)

    fig.suptitle(f"kNN distance distributions vs HR — {res['n_sample']:,} "
                 f"box-wide particles, full-box search", fontsize=13)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"=== wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
