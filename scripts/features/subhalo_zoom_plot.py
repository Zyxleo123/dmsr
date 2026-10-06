#!/usr/bin/env python
"""Render the per-subhalo slab figures. CPU, seconds.

Reads only the ``.npz`` + ``.json`` from :mod:`subhalo_zoom_data`, so a colour,
layout or scaling change re-renders without touching a field -- the redrawable
figure rule.

Three figures, in the order they should be read:

``fig1_subhalo_grid``   one row per subhalo, columns ``tuned | frozen | HR``, at
                        the *thinnest* slab. This is the picture the host panels
                        could not give: if the tuned arm's subhalos are real,
                        their knots appear here and are absent from the frozen
                        column at the same coordinates.
``fig2_depth_ladder``   one row per subhalo, columns running from the thinnest
                        slab to the full column, in the tuned field. Reading
                        left to right *is* the burial: it shows the object
                        disappearing into the projection, which is the claim
                        being tested.
``fig3_contrast``       excess ratio and signal-to-noise against slab depth,
                        all subhalos, all three fields -- the projection story
                        as numbers, because a colour map can be tuned into or
                        out of showing anything.
``fig4_aperture``       **the figure to quote.** The same subhalos measured four
                        ways: full R_vir vs a core aperture, and a local
                        background vs the host's own profile at that radius. It
                        exists because the first run showed the naive choice
                        (full R_vir, local annulus) scores HR's own subhalos at
                        0.4-1.2x -- so any conclusion drawn from that column is
                        a conclusion about the aperture.

All panels in one row share a colour scale, and the scale is built from the
three arms together. Without that the eye compares three independently
normalised images and always sees a knot in each.

    python scripts/features/subhalo_zoom_plot.py \\
        --data $MG/all_blocks_self/holdout_set9/figures/subzoom_self \\
        --out-dir $MG/all_blocks_self/holdout_set9/figures/subzoom
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gather_holdout_add_density import stem_path, with_ext  # noqa: E402

ARM_ORDER = ("tuned", "frozen", "hr")
ARM_LABEL = {"tuned": "tuned", "frozen": "frozen SR2", "hr": "HR (truth)"}
C = {"tuned": "#d62728", "frozen": "#1f77b4", "hr": "#111111"}


def depth_label(d: float) -> str:
    return "full column" if d <= 0 else f"±{d:g} R$_{{vir}}$"


def as_overdensity(counts: np.ndarray) -> np.ndarray:
    """Counts -> 1 + delta, against the panel's own robust background.

    The median of the non-empty bins, not the mean: the mean of a panel that
    contains a real subhalo is pulled up by the subhalo itself, which would
    divide the signal out of the picture.
    """
    c = np.asarray(counts, dtype=np.float64)
    nz = c[c > 0]
    bg = float(np.median(nz)) if nz.size else 1.0
    return c / max(bg, 1e-6)


def row_scale(panels: List[np.ndarray]) -> tuple:
    """A single (vmin, vmax) in log10 for every panel of one row."""
    stack = np.concatenate([np.log10(as_overdensity(p) + 0.1).ravel()
                            for p in panels])
    return float(np.quantile(stack, 0.02)), float(np.quantile(stack, 0.995))


def _panel(ax, counts: np.ndarray, half: float, vmin: float, vmax: float,
           r: float) -> None:
    ax.imshow(np.log10(as_overdensity(counts) + 0.1).T, origin="lower",
              extent=[-half, half, -half, half], cmap="magma",
              aspect="equal", vmin=vmin, vmax=vmax)
    # The reported R_vir, so the eye judges the knot against the aperture the
    # contrast numbers actually used rather than against the panel.
    ax.add_patch(Circle((0.0, 0.0), r, fill=False, lw=1.0, ls="--",
                        ec="#7fffd4", alpha=0.9))
    ax.set_xticks([]); ax.set_yticks([])


def fig_grid(z, meta: Dict, out: Path, depth_index: int = 0) -> None:
    """Rows = subhalos, columns = the three fields, at one slab depth."""
    subs = meta["subhalos"]
    n = len(subs)
    depths = meta["depths_rvir"]
    di = min(int(depth_index), len(depths) - 1)
    fig, axes = plt.subplots(n, 3, figsize=(9.6, 3.2 * n),
                             constrained_layout=True, squeeze=False)
    for k, s in enumerate(subs):
        panels = [z[f"sub{k:02d}_{a}"][di] for a in ARM_ORDER]
        vmin, vmax = row_scale(panels)
        half = float(s["half_xy_mpc_h"])
        for j, arm in enumerate(ARM_ORDER):
            ax = axes[k][j]
            _panel(ax, panels[j], half, vmin, vmax, float(s["rvir_mpc_h"]))
            lad = s["arms"][arm]["ladder"][di]
            ax.set_title(f"{ARM_LABEL[arm]}   {lad['ratio']:.2f}× "
                         f"(snr {lad['snr']:.1f})", fontsize=9.5,
                         color=C[arm])
        flag = " (window floored)" if s.get("window_floored") else ""
        axes[k][0].set_ylabel(
            f"id {s['halo_id']}\n{s['num_p']}p, R$_{{vir}}$={s['rvir_mpc_h']:.3f}\n"
            f"{s['dist_over_host_rvir']:.2f} R$_{{vir,host}}$ out\n"
            f"±{half:.2f} Mpc/h{flag}", fontsize=8.0)
    fig.suptitle(
        f"{meta['tag']} · host {meta['host_id']} · slab {depth_label(depths[di])} "
        f"· window ±{meta['half_xy_rvir']:g} R$_{{vir,sub}}$ · "
        f"particles: {meta['particle_source']}", fontsize=11)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}", flush=True)


def fig_depth(z, meta: Dict, out: Path, arm: str = "tuned") -> None:
    """Rows = subhalos, columns = slab depth. Reading right is the burial."""
    subs = meta["subhalos"]
    depths = meta["depths_rvir"]
    n, m = len(subs), len(depths)
    fig, axes = plt.subplots(n, m, figsize=(3.1 * m, 3.2 * n),
                             constrained_layout=True, squeeze=False)
    for k, s in enumerate(subs):
        panels = [z[f"sub{k:02d}_{arm}"][d] for d in range(m)]
        # One scale per row again, but here it spans the depths, so a panel
        # fading out is the object being diluted and not a rescaling artefact.
        vmin, vmax = row_scale(panels)
        half = float(s["half_xy_mpc_h"])
        for j in range(m):
            ax = axes[k][j]
            _panel(ax, panels[j], half, vmin, vmax, float(s["rvir_mpc_h"]))
            lad = s["arms"][arm]["ladder"][j]
            ax.set_title(f"{depth_label(depths[j])}  {lad['ratio']:.2f}×",
                         fontsize=9.5)
        axes[k][0].set_ylabel(f"id {s['halo_id']}\n{s['num_p']}p", fontsize=8.5)
    fig.suptitle(f"{meta['tag']} · {ARM_LABEL.get(arm, arm)} · host "
                 f"{meta['host_id']} · the same subhalo at five slab depths",
                 fontsize=11)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}", flush=True)


def fig_contrast(meta: Dict, out: Path) -> None:
    """The numbers: excess ratio and snr against slab depth, per field."""
    subs = meta["subhalos"]
    depths = np.asarray(meta["depths_rvir"], dtype=float)
    # Put the full column at the right-hand end of a log-ish axis.
    x = np.where(depths > 0, depths, np.nanmax(depths[depths > 0]) * 3.0
                 if (depths > 0).any() else 1.0)
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), constrained_layout=True,
                             sharex=True)
    for j, arm in enumerate(ARM_ORDER):
        for s in subs:
            lad = s["arms"][arm]["ladder"]
            ratio = [c["ratio"] for c in lad]
            snr = [c["snr"] for c in lad]
            w = 0.6 + 1.6 * (np.log10(max(s["num_p"], 1)) - np.log10(50))
            axes[0][j].plot(x, ratio, "-o", ms=3.0, lw=w, alpha=0.75,
                            color=C[arm])
            axes[1][j].plot(x, snr, "-o", ms=3.0, lw=w, alpha=0.75, color=C[arm])
        axes[0][j].axhline(1.0, color="#999999", lw=0.9, ls=":")
        axes[1][j].axhline(3.0, color="#999999", lw=0.9, ls=":")
        axes[0][j].set_yscale("log")
        axes[0][j].set_title(ARM_LABEL[arm], color=C[arm])
        axes[1][j].set_xscale("log")
        axes[1][j].set_xlabel("slab half-depth / R$_{vir,sub}$  "
                              "(rightmost = full column)")
    axes[0][0].set_ylabel("overdensity ratio\nrho(<r) / rho(3r..5r)")
    axes[1][0].set_ylabel("excess / Poisson noise (snr)\ndotted line: snr = 3")
    fig.suptitle(
        f"{meta['tag']} · host {meta['host_id']} · {len(subs)} subhalos · "
        "a real object decays towards 1 as the slab opens; a phantom is at 1 "
        "even in the thinnest slab", fontsize=11)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}", flush=True)



def fig_aperture(meta: Dict, out: Path) -> None:
    """Ratio against particle count, for all four aperture/background choices.

    Laid out so the two corrections separate: moving down a column changes the
    APERTURE (full R_vir -> core), moving across a row changes the BACKGROUND
    (local annulus -> the host's profile at the same radius). A point below the
    dotted line is an object the measurement calls absent.
    """
    subs = meta["subhalos"]
    keys = [("contrast_3d", "full R$_{vir}$ · local annulus"),
            ("host_shell", "full R$_{vir}$ · host profile"),
            ("contrast_3d_core", f"core ({meta.get('core_frac', 0.3):g} R$_{{vir}}$) · local annulus"),
            ("host_shell_core", f"core ({meta.get('core_frac', 0.3):g} R$_{{vir}}$) · host profile")]
    present = [(k, t) for k, t in keys
               if subs and k in subs[0].get("arms", {}).get(ARM_ORDER[0], {})]
    if not present:
        print("  fig4: this data file predates the aperture/background work, "
              "skipped", flush=True)
        return

    n = len(present)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True,
                             squeeze=False, sharex=True, sharey=True)
    flat = [axes[i // 2][i % 2] for i in range(4)]
    for ax, (key, title) in zip(flat, present):
        for arm in ARM_ORDER:
            x = [s["num_p"] for s in subs]
            y = [s["arms"][arm][key]["ratio"] for s in subs]
            ax.plot(x, y, "o", ms=6.5, alpha=0.8, color=C[arm],
                    label=ARM_LABEL[arm])
        ax.axhline(1.0, color="#999999", lw=0.9, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(title, fontsize=10.5)
    for ax in flat[len(present):]:
        ax.set_visible(False)
    flat[0].legend(fontsize=9, loc="upper left")
    for i in (2, 3):
        flat[i].set_xlabel("bound particles")
    for i in (0, 2):
        flat[i].set_ylabel("overdensity ratio")
    fig.suptitle(
        f"{meta['tag']} · host {meta['host_id']} · why the aperture and the "
        "background both matter\n"
        "left→right removes the host's density gradient; top→bottom stops "
        "averaging the core over the tidal envelope", fontsize=11)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="stem written by subhalo_zoom_data.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--depth-index", type=int, default=0,
                    help="which slab depth fig1 draws; 0 is the thinnest")
    ap.add_argument("--arm", default="tuned", choices=list(ARM_ORDER))
    ap.add_argument("--figures", default="fig1,fig2,fig3,fig4")
    args = ap.parse_args(argv)

    stem = stem_path(args.data)
    meta = json.loads(with_ext(stem, ".json").read_text())
    z = np.load(with_ext(stem, ".npz"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    want = {w.strip() for w in args.figures.split(",")}

    print(f"=== {len(meta['subhalos'])} subhalos, source {meta['particle_source']}",
          flush=True)
    if "fig1" in want:
        fig_grid(z, meta, out_dir / "fig1_subhalo_grid.png", args.depth_index)
    if "fig2" in want:
        fig_depth(z, meta, out_dir / "fig2_depth_ladder.png", args.arm)
    if "fig3" in want:
        fig_contrast(meta, out_dir / "fig3_contrast.png")
    if "fig4" in want:
        fig_aperture(meta, out_dir / "fig4_aperture.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
