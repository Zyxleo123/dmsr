#!/usr/bin/env python
"""Render the held-out gather figures from the extracted data. CPU, seconds.

Reads only the ``.npz`` + ``.json`` that ``gather_holdout_figures_data.py``
wrote, so a colour, label or layout change re-renders without re-reading a
3.2 GB field -- the redrawable-figure rule.

Three figures:
  fig1_host_density  frozen | tuned | HR, the example host's edited material as
                     a density projection with every Rockstar subhalo marked.
                     The 20 -> 366 -> 369 as a picture.
  fig2_mass_function subhalo counts within R_vir per mass bin, HR/base/frozen/
                     tuned -- the same numbers, quantitative.
  fig3_local_excess  hosts>=200p and subs>=50p RESTRICTED to the 32 spliced
                     tiles: does tuned exceed HR where it actually edited?
  fig4_box_damage    hosts>=200p near the host, base vs tuned, lost hosts marked
                     -- where the -343 sits relative to the edited tiles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

C = {"hr": "#111111", "base": "#888888", "frozen": "#1f77b4",
     "self": "#d62728", "tuned": "#d62728", "nocentre": "#2ca02c",
     "full": "#9467bd", "radial": "#ff7f0e"}


def marker_sizes(num_p: np.ndarray) -> np.ndarray:
    """Marker area scaled by log particle count, floored so 50p is visible."""
    return 6.0 + 22.0 * (np.log10(np.maximum(num_p, 1)) - np.log10(50))


def fig_host_density(z, S, out: Path, zoom_rvir: float = 2.6):
    """Zoomed to a few R_vir so the individual subhalos are resolved.

    The whole-block view (33 Mpc/h) buries all ~370 subhalos inside the 1.55
    Mpc/h R_vir circle -- a single dot. The abundance win only becomes legible at
    the halo's own scale, so the window is ``zoom_rvir`` search radii on a side.
    HR uses the hi-res host-region density if the data job wrote one, else its
    subhalo markers on a neutral ground.
    """
    cen, rvir = z["host_centre"], float(z["host_rvir_mpc"][0])
    half = zoom_rvir * rvir
    xlim = (cen[0] - half, cen[0] + half)
    zlim = (cen[2] - half, cen[2] + half)
    hxr, hzr = z["host_density_xr"], z["host_density_zr"]
    ext = [hxr[0], hxr[1], hzr[0], hzr[1]]
    dens = {k: (z[f"{k}_host_density"] if f"{k}_host_density" in z.files else None)
            for k in ("frozen", "self", "hr")}

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 6.0), constrained_layout=True)

    def panel(a, key, title, color):
        if dens[key] is not None:
            a.imshow(np.log10(dens[key] + 1.0), origin="lower", extent=ext,
                     cmap="magma", aspect="equal",
                     vmax=np.log10(np.nanmax(dens[key]) + 1.0))
        else:
            a.set_facecolor("#f4f0ea")
        pos = z[f"overlay_{key}_pos"]; npart = z[f"overlay_{key}_np"]
        if pos.shape[0]:
            a.scatter(pos[:, 0], pos[:, 2], s=marker_sizes(npart) * 2.2,
                      facecolors="none", edgecolors="cyan" if dens[key] is not None
                      else color, linewidths=1.3)
        a.add_patch(Circle((cen[0], cen[2]), rvir, fill=False,
                           edgecolor="cyan", lw=1.6, ls="--"))
        a.plot(cen[0], cen[2], "+", color="cyan", ms=14, mew=2)
        a.set_title(f"{title}\n{pos.shape[0]} subhalos in $R_{{vir}}$", fontsize=12)
        a.set_xlabel("x  [Mpc/h]"); a.set_xlim(xlim); a.set_ylim(zlim)

    panel(ax[0], "frozen", "frozen SR2 (control)", C["frozen"])
    panel(ax[1], "self", "tuned (self)", C["self"])
    panel(ax[2], "hr", "HR (truth)", C["hr"])
    ax[0].set_ylabel("z  [Mpc/h]")
    fig.suptitle(f"Held-out host {S['host_id']} ({S['box']}), zoomed to "
                 f"{2*zoom_rvir:.0f} $R_{{vir}}$: subhalos as circles "
                 f"(size ~ mass), cyan dashed = host $R_{{vir}}$ "
                 f"{S['host_rvir_mpc']:.2f} Mpc/h", fontsize=13)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_mass_function(S, out: Path):
    labels = S["mass_bin_labels"]
    order = [k for k in ("hr", "base", "frozen", "self") if k in S["mass_function"]]
    x = np.arange(len(labels)); w = 0.8 / len(order)
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    for i, k in enumerate(order):
        vals = S["mass_function"][k]
        ax.bar(x + (i - (len(order) - 1) / 2) * w, vals, w,
               label=f"{k} (tot {sum(vals)})", color=C.get(k, "#555"))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("subhalos within $R_{vir}$")
    ax.set_title(f"Host {S['host_id']} subhalo mass function "
                 f"(held-out {S['box']}): base 20 -> tuned "
                 f"{sum(S['mass_function']['self'])} vs HR "
                 f"{sum(S['mass_function']['hr'])}")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")


def fig_local_excess(S, out: Path):
    rc = S["region_counts"]
    order = [k for k in ("hr", "base", "frozen", "self", "nocentre",
                         "full", "radial") if k in rc]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    for a, key, title in ((ax[0], "hosts_ge_200p", "hosts >= 200p"),
                          (ax[1], "subs_ge_50p", "subhalos >= 50p")):
        vals = [rc[k]["in_tiles"][key] for k in order]
        bars = a.bar(order, vals, color=[C.get(k, "#555") for k in order])
        a.bar_label(bars, fontsize=9)
        a.axhline(rc["hr"]["in_tiles"][key], color=C["hr"], ls="--", lw=1,
                  label="HR (truth)")
        a.set_title(f"{title}  in the {S['n_spliced_tiles']} spliced tiles "
                    f"({100*S['spliced_volume_fraction']:.1f}% of box)")
        a.tick_params(axis="x", rotation=20); a.legend()
    fig.suptitle("Counts restricted to the edited region -- the like-for-like "
                 "HR reference the box-wide numbers cannot give", fontsize=12)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")


def fig_box_damage(z, S, out: Path):
    cen = z["host_centre"]
    slab = 12.5  # Mpc/h in y around the host, one tile thick
    def in_slab(pos):
        dy = np.abs(pos[:, 1] - cen[1]); dy = np.minimum(dy, 100 - dy)
        return dy <= slab
    fig, ax = plt.subplots(figsize=(8.4, 8.0), constrained_layout=True)
    # the spliced tile footprints in this slab
    for tb in z["tile_boxes"]:
        (ylo, yhi) = tb[1]
        dy = min(abs(ylo - cen[1]), abs(yhi - cen[1]), 100 - abs(ylo - cen[1]))
        if dy > slab + 12.5:
            continue
        ax.add_patch(Rectangle((tb[0, 0], tb[2, 0]), tb[0, 1] - tb[0, 0],
                               tb[2, 1] - tb[2, 0], fill=False,
                               edgecolor="orange", lw=1.0, alpha=0.7))
    bp, bin_, blost = z["base_hosts_pos"], z["base_hosts_in"], z["base_hosts_lost"]
    sp, sgain = z["self_hosts_pos"], z["self_hosts_gain"]
    ms = in_slab(bp)
    ax.scatter(bp[ms & ~blost, 0], bp[ms & ~blost, 2], s=10, c="#888",
               label="host (base, kept)")
    ax.scatter(bp[ms & blost, 0], bp[ms & blost, 2], s=70, marker="x",
               c="#d62728", label="host lost by tuning", zorder=5)
    ss = in_slab(sp)
    ax.scatter(sp[ss & sgain, 0], sp[ss & sgain, 2], s=42, marker="+",
               c="#2ca02c", label="host gained by tuning", zorder=5)
    ax.plot(cen[0], cen[2], "*", c="cyan", ms=16, mec="k", label="example host")
    ax.set_xlabel("x  [Mpc/h]"); ax.set_ylabel("z  [Mpc/h]")
    ax.set_title(f"hosts>=200p in a {2*slab:.0f} Mpc/h y-slab through host "
                 f"{S['host_id']}\norange = spliced tiles; "
                 f"lost {S['host_damage']['base_only_lost']}, "
                 f"gained {S['host_damage']['self_only_gained']} in-region")
    ax.legend(loc="upper right", fontsize=9); ax.set_aspect("equal")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")


def fig_cosmic_web(z, S, out: Path):
    """Full-box density slab: SR2 before | tuned | HR, the web at large scale.

    The edited tiles are outlined in orange. 93.75% of each panel is identical
    frozen SR2 by construction, so the comparison is really "what changed in the
    outlined tiles" -- read it there.
    """
    before = str(z["web_before"])
    panels = [(before, f"{before} SR2 (before)"), ("self", "tuned (self)"),
              ("hr", "HR (truth)")]
    panels = [(k, t) for k, t in panels if f"{k}_box_density" in z.files]
    n = len(panels)
    slab = float(z["box_slab_mpc"][0])
    cen = z["host_centre"]
    # a shared colour scale from the HR panel (or the last available) so damage
    # reads as a real deficit, not a per-panel renormalisation
    ref = z[f"{panels[-1][0]}_box_density"]
    vmax = np.log10(np.percentile(ref[ref > 0], 99.7) + 1.0)

    fig, ax = plt.subplots(1, n, figsize=(6.0 * n, 6.4), constrained_layout=True)
    if n == 1:
        ax = [ax]
    for a, (key, title) in zip(ax, panels):
        d = z[f"{key}_box_density"]
        a.imshow(np.log10(d + 1.0), origin="lower", extent=[0, 100, 0, 100],
                 cmap="magma", aspect="equal", vmax=vmax)
        for tb in z["tile_boxes"]:
            ylo, yhi = tb[1]
            dy = min(abs(ylo - cen[1]), abs(yhi - cen[1]), 100 - abs(ylo - cen[1]))
            if dy <= slab + 12.5:
                a.add_patch(Rectangle((tb[0, 0], tb[2, 0]), tb[0, 1] - tb[0, 0],
                                      tb[2, 1] - tb[2, 0], fill=False,
                                      edgecolor="#39ff14", lw=1.1, alpha=0.9))
        a.plot(cen[0], cen[2], "+", color="cyan", ms=13, mew=2)
        a.set_title(title, fontsize=13)
        a.set_xlabel("x  [Mpc/h]")
    ax[0].set_ylabel("z  [Mpc/h]")
    fig.suptitle(f"Cosmic web, {2*slab:.0f} Mpc/h slab through host "
                 f"{S['host_id']} ({S['box']}); green = the 32 edited tiles "
                 f"(6.25% of box), same colour scale across panels", fontsize=13)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the .npz/.json stem")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    stem = Path(args.data)
    if stem.suffix in (".npz", ".json"):
        stem = stem.with_suffix("")
    z = np.load(stem.with_suffix(".npz"))
    S = json.loads(stem.with_suffix(".json").read_text())
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    fig_host_density(z, S, out / "fig1_host_density.png")
    fig_mass_function(S, out / "fig2_mass_function.png")
    fig_local_excess(S, out / "fig3_local_excess.png")
    fig_cosmic_web(z, S, out / "fig4_cosmic_web.png")
    fig_box_damage(z, S, out / "fig5_host_scatter.png")
    print(f"\nfigures under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
