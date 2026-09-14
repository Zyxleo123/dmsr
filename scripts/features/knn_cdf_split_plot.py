#!/usr/bin/env python
"""Figure for knn_cdf_split_diagnostic.py. Reads its JSON only.

Top row, one panel per HR-owner class: each field's paired median r_k ratio to
HR (same Lagrangian ids) against k. 1.0 is HR; grey band is the loss's dead
zone; >1 means those particles' neighbours are further away than in HR. The
sub_50_200 panel is the stop rule.

Bottom row: tile-local estimator error against the full box, per mode -- p95 of
|r_kmax/r_true - 1| and of |r_8/r_true - 1| -- and the candidate multiplier the
neighbour modes cost.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
         "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3dc"
CLASSES = ("sub_50_200", "sub_ge200", "sub_lt50", "host", "unbound")
MODES = ("hr_lagrangian", "hr_centroid", "hr_centroid_nbr", "tuned_centroid_nbr",
         "tuned_centroid_frozen")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.labelcolor": INK, "savefig.facecolor": "white"})
    r = json.loads(Path(a.json).read_text())
    ks, dz = r["ks"], float(r["dead_zone"])
    fields = [f for f in r["membership"] if f != "hr"]
    colour = {f: SLOTS[i % len(SLOTS)] for i, f in enumerate(fields)}
    lfs = r.get("loss_forms", {})
    fig = plt.figure(figsize=(20, 14 if lfs else 9.5), layout="constrained")
    gs = fig.add_gridspec(3 if lfs else 2, len(CLASSES))
    for j, c in enumerate(CLASSES):
        ax = fig.add_subplot(gs[0, j])
        ax.axhspan(1 / (1 + dz), 1 + dz, color=GRID, lw=0)
        ax.axhline(1.0, color=INK2, lw=1)
        for f in fields:
            e = r["membership"][f]["classes"].get(c)
            if e and "paired_ratio_med" in e:
                ax.plot(ks, e["paired_ratio_med"], "o-", color=colour[f], lw=2,
                        ms=8, label=f)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        n = r["class_sample_counts"].get(c, 0)
        ax.set(title=f"{c}  (n={n:,})", xlabel="k")
        if j == 0:
            ax.set_ylabel("paired median $r_k$ / HR")
        ax.grid(axis="y", color=GRID, lw=0.6)
    fig.axes[0].legend(frameon=False, fontsize=9)

    b = r.get("boundary", {})
    mcol = {m: SLOTS[i] for i, m in enumerate(MODES)}
    for j, (key, title) in enumerate((("rel_kmax_p95", f"p95 |error| of r_{max(ks)}"),
                                      ("rel_small_p95", "p95 |error| of r_8"))):
        ax = fig.add_subplot(gs[1, j:j + 2] if j == 0 else gs[1, 2:4])
        for m in MODES:
            if m not in b:
                continue
            xs = [x["size_cells"] for x in b[m]]
            ax.plot(xs, [100 * x[key] for x in b[m]], "o-", color=mcol[m], lw=2,
                    ms=8, label=f"{m} ({b[m][0]['size_kind']})")
        ax.axhline(100 * dz, color=INK2, ls="--", lw=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set(xlabel="buffer or neighbour margin [cells]", ylabel="percent",
               title=title)
        ax.grid(axis="y", color=GRID, lw=0.6)
        if j == 0:
            ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(gs[1, 4])
    for m in MODES:
        if m in b and ("nbr" in m or "frozen" in m):
            ax.plot([x["size_cells"] for x in b[m]],
                    [x["candidate_multiplier"] for x in b[m]], "o-", color=mcol[m],
                    lw=2, ms=8)
    ax.set(xlabel="neighbour margin [cells]", ylabel="candidates / tile particles",
           title="search cost")
    ax.grid(axis="y", color=GRID, lw=0.6)
    if lfs:
        forms = ("unpaired", "class_cond_mass", "class_cond_equal", "paired")
        base = "frozen" if "frozen" in lfs else None
        ax = fig.add_subplot(gs[2, 0:3])
        arms = [f for f in fields if f in lfs and f != base]
        width = 0.8 / max(len(arms), 1)
        for i, f in enumerate(arms):
            vals = [100 * (lfs[f][f"{k}_total"] / lfs[base][f"{k}_total"] - 1)
                    if base and lfs[base][f"{k}_total"] > 0 else np.nan
                    for k in forms]
            ax.bar(np.arange(len(forms)) + (i - (len(arms) - 1) / 2) * width, vals,
                   width=width * 0.9, color=colour[f], label=f)
        ax.axhline(0, color=INK2, lw=1)
        # Symlog: the unpaired form's frozen value is ~1e-4, so its relative
        # changes run to 1e5 % and a linear axis flattens every other form.
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xticks(range(len(forms)), forms)
        ax.set(ylabel="loss change vs frozen [%]  (<0 = closer to HR)",
               title="Which loss form moves with the arms?")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax = fig.add_subplot(gs[2, 3:5])
        ax.axis("off")
        if base:
            rows = [[k] + [f"{lfs[f][f'{k}_total']:.4g}" for f in [base] + arms]
                    for k in forms]
            t = ax.table(cellText=rows, colLabels=["form"] + [base] + arms,
                         loc="center")
            t.auto_set_font_size(False)
            t.set_fontsize(9)
            t.scale(1, 1.8)
        region = r.get("region_tiles")
        ax.set_title("absolute loss (dead-zoned squared hinge, mean over k)"
                     + (f"\nregion: {len(region)} spliced tiles" if region else
                        "\nbox-wide sample"), fontsize=10)
    fig.suptitle("kNN distances by HR membership, and tile-local estimator error"
                 + ("  (spliced region only)" if r.get("region_tiles") else ""),
                 fontsize=13)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=140)
    print(f"=== wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
