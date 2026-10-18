#!/usr/bin/env python
"""Draw the bump figures from bump_cutouts.py's saved npz/json (no recomputation).

1. ``bump_null.png``     observed bumps vs the angular-shuffle null, per contrast band
2. ``bump_context.png``  one host's slab: HR / frozen / fine-tuned, bumps marked
                         (filled = became a Rockstar object, hollow = did not)
3. ``bump_gallery.png``  projected particles of example bumps, band x (field, found?)
4. ``bump_phase.png``    the same bumps in phase space: distance vs |v - v_core|

Colours: the validated categorical slots 1-3 (dataviz reference palette, light):
HR blue, frozen orange, fine-tuned aqua; density images use one hue light->dark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COLOR = {"hr": "#2a78d6", "frozen": "#eb6834", "coarse_soft": "#1baf7a"}
LABEL = {"hr": "HR", "frozen": "frozen SR2", "coarse_soft": "fine-tuned"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
BANDS = ["3-10x", "10-30x", "30-100x", ">100x"]


def style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
        "font.size": 9, "axes.titlesize": 9, "axes.spines.top": False,
        "axes.spines.right": False})


def cutouts(z):
    off = z["cut_offsets"]
    for k, m in enumerate(z["cut_meta"]):
        sl = slice(int(off[k]), int(off[k + 1]))
        yield m, z["cut_pos"][sl], z["cut_vel"][sl]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--fields", default="hr,frozen,coarse_soft")
    ap.add_argument("--half", type=float, default=0.15, help="gallery half-width, Mpc/h")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    style(plt)
    D = Path(a.dir)
    names = [n for n in a.fields.split(",") if (D / f"{n}_bumps.npz").is_file()]
    Z = {n: np.load(D / f"{n}_bumps.npz") for n in names}
    null = json.loads((D / "bump_null.json").read_text())

    # 1. null ------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        r = null[n]
        xs = np.arange(4)
        ax.bar(xs - 0.2, r["observed"], 0.38, color=COLOR[n], label="observed")
        ax.bar(xs + 0.2, r["null"], 0.38, color=SURFACE, edgecolor=INK2, hatch="///",
               linewidth=0.8, label="shuffled null")
        for x, o, nn in zip(xs, r["observed"], r["null"]):
            ax.text(x, max(o, nn) * 1.03 + 5, f"{(1 - nn / o) * 100:.0f}% real" if o else "",
                    ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xticks(xs, BANDS)
        ax.set_title(f"{LABEL[n]}: bumps at {r['x_range'][0]}-{r['x_range'][1]} R_vir, "
                     f"{r['null_hosts']} hosts", loc="left")
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("bumps (local k=32 density max)")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(D / "bump_null.png", dpi=170)
    plt.close(fig)

    # 2. context -------------------------------------------------------------
    fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 4.2))
    axes = np.atleast_1d(axes)
    vmax = max(float(Z[n]["ctx_img"].max()) for n in names)
    for ax, n in zip(axes, names):
        z = Z[n]
        ext, rv = float(z["ctx_extent"][0]), float(z["ctx_rvir"][0])
        ax.imshow(z["ctx_img"].T + 0.5, origin="lower", extent=[-ext / rv, ext / rv] * 2,
                  cmap="Greys", norm=LogNorm(vmin=0.5, vmax=vmax))
        for rr in (0.5, 1.0):
            ax.add_patch(plt.Circle((0, 0), rr, fill=False, color=INK2, lw=0.6, ls="--"))
        bp = z["ctx_bumps"] / rv
        hit = z["ctx_bump_hit"].astype(bool)
        size = 6 + 9 * z["ctx_bump_band"]
        ax.scatter(bp[hit, 0], bp[hit, 1], s=size[hit], color=COLOR[n], edgecolor=SURFACE,
                   linewidth=0.5, label=f"bump, became an object ({hit.sum()})")
        ax.scatter(bp[~hit, 0], bp[~hit, 1], s=size[~hit], facecolor="none",
                   edgecolor=COLOR[n], linewidth=0.9, label=f"bump, no object ({(~hit).sum()})")
        ax.set_title(f"{LABEL[n]}: host {int(z['ctx_host_id'][0])} slab", loc="left")
        ax.set_xlabel("x / R_vir")
        ax.set_aspect("equal")
        ax.legend(frameon=False, loc="lower left", fontsize=7)
    axes[0].set_ylabel("y / R_vir")
    fig.tight_layout()
    fig.savefig(D / "bump_context.png", dpi=170)
    plt.close(fig)

    # 3 + 4. galleries: rows = band, columns = (field, found / not found) -------
    cols = [(n, h) for n in names for h in (1, 0)]
    picks = {}
    for n in names:
        for m, p, v in cutouts(Z[n]):
            key = (n, int(m[2]), int(m[1]))
            picks.setdefault(key, (m, p, v))
    for kind in ("gallery", "phase"):
        fig, axes = plt.subplots(4, len(cols), figsize=(1.9 * len(cols), 1.95 * 4),
                                 squeeze=False)
        for bi in range(4):
            for ci_, (n, h) in enumerate(cols):
                ax = axes[bi, ci_]
                got = picks.get((n, h, bi))
                if bi == 0:
                    ax.set_title(f"{LABEL[n]}\n{'object' if h else 'no object'}", fontsize=8)
                if got is None:
                    ax.axis("off")
                    continue
                m, p, v = got
                if kind == "gallery":
                    sl = np.abs(p[:, 2]) < a.half
                    ax.hist2d(p[sl, 0], p[sl, 1], bins=60, range=[[-a.half, a.half]] * 2,
                              cmap="Greys", norm=LogNorm(vmin=0.9))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect("equal")
                else:
                    r = np.linalg.norm(p, axis=1)
                    s = r < a.half
                    ax.scatter(r[s] * 1000.0, np.linalg.norm(v[s], axis=1), s=1.5,
                               color=COLOR[n], alpha=0.45, linewidth=0, rasterized=True)
                    ax.set_xlim(0, a.half * 1000)
                    ax.set_ylim(0, 1500)
                    if ci_ == 0:
                        ax.set_ylabel("|v - v_core| km/s", fontsize=7)
                    if bi == 3:
                        ax.set_xlabel("r kpc/h", fontsize=7)
                    ax.tick_params(labelsize=6)
                note = f"{m[3]:.0f}x  x={m[4]:.2f}"
                if h:
                    note += f"  {int(m[5])}p"
                ax.text(0.03, 0.97, note, transform=ax.transAxes, fontsize=6.5, va="top",
                        color=INK2)
            axes[bi, 0].annotate(BANDS[bi], xy=(-0.32, 0.5), xycoords="axes fraction",
                                 rotation=90, va="center", ha="center", fontsize=8)
        fig.suptitle(("Example bumps at 0.5-1 R_vir, projected particles (+-150 kpc/h)"
                      if kind == "gallery" else
                      "Same bumps in phase space: a bound clump is a cold spot at small r"),
                     x=0.01, ha="left", fontsize=9)
        fig.tight_layout()
        fig.savefig(D / f"bump_{kind}.png", dpi=170)
        plt.close(fig)
    print("wrote", ", ".join(str(D / f) for f in ("bump_null.png", "bump_context.png",
                                                   "bump_gallery.png", "bump_phase.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
