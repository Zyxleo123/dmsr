#!/usr/bin/env python
"""Draw (and redraw) the gather fine-tune's eval panels. CPU, no generator.

``finetune_host_gather.py`` writes one ``eval/step*.npz`` per eval step holding
the three valid-centre density grids -- frozen SR2, the fine-tuned candidate and
HR -- for every training tile, plus the true HR subhalo centres the loss is
supervised on. Everything here reads only those files, so a figure can be
restyled or re-rendered at any time without touching a GPU, which is the
project's rule for anything that produces a picture.

Each figure is one tile at one step: three panels on a **shared** colour scale,
because the whole question is whether the middle panel moved toward the right
one. The panels are a *max projection* of ``1 + delta`` through a slab, not a
single slice: a 366-particle subhalo is under one cell across, so a slice through
the wrong plane misses it entirely and would make an improvement look like
nothing happened. Circles mark the true HR subhalos whose centres fall inside the
slab, at the loss kernel's own radius -- so what the loss asked for and what the
field did are visible in the same frame.

    python scripts/features/render_gather_slices.py --run-dir <.../set8_h271800_fine>
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1a1a1a"
MUTED = "#6b6b6b"
RING = "#D55E00"


def _projection(vol: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """``log10`` of the max of ``1 + delta`` through the slab ``[lo, hi)``."""
    slab = np.asarray(vol)[:, :, lo:hi]
    return np.log10(np.maximum(1.0 + slab.max(axis=2), 1e-3))


def render_npz(npz_path, out_dir, slab: int = 4) -> list:
    """One PNG per tile. Returns the paths written."""
    z = np.load(str(npz_path), allow_pickle=False)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = int(z["step"])
    tiles = z["tiles"]
    cell = float(z["cellsize_mpc_h"]) if "cellsize_mpc_h" in z else 0.1953
    box = str(z["box"]) if "box" in z else ""
    host = int(z["host_id"]) if "host_id" in z else -1

    written = []
    for i, tile in enumerate(tiles):
        hr = z["delta_hr"][i]
        g = hr.shape[-1]
        live = np.flatnonzero(z["mask"][i] > 0)
        centre = z["centre"][i]
        # Centre the slab where the supervision is: the mass-weighted mean plane
        # of this tile's targets. An arbitrary mid-plane can miss every one.
        if live.size:
            w = np.asarray(z["hr_compact"][i])[live]
            zc = float(np.sum(centre[live, 2] * w) / max(w.sum(), 1e-9))
        else:
            zc = 0.5 * g
        lo = int(max(0, round(zc) - slab))
        hi = int(min(g, round(zc) + slab + 1))

        panels = [("frozen SR2", z["delta_frozen"][i]),
                  (f"fine-tuned (step {step})", z["delta_out"][i]),
                  ("HR (truth)", hr)]
        imgs = [(name, _projection(v, lo, hi)) for name, v in panels]
        vmax = float(max(im.max() for _, im in imgs))
        vmin = 0.0
        extent = (0.0, g * cell, 0.0, g * cell)

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), constrained_layout=True)
        for ax, (name, im) in zip(axes, imgs):
            h = ax.imshow(im.T, origin="lower", extent=extent, vmin=vmin, vmax=vmax,
                          cmap="magma", interpolation="nearest")
            ax.set_title(name, color=INK, fontsize=12)
            ax.set_xlabel("Mpc/h", color=MUTED, fontsize=9)
            for j in live:
                if not lo <= centre[j, 2] < hi:
                    continue
                r = max(float(z["sigma"][i][j]), 1.0) * cell
                ax.add_patch(plt.Circle((centre[j, 0] * cell, centre[j, 1] * cell),
                                        r, fill=False, lw=0.8, ec=RING, alpha=0.85))
        n_ring = int(sum(1 for j in live if lo <= centre[j, 2] < hi))
        cb = fig.colorbar(h, ax=axes, fraction=0.025, pad=0.01)
        cb.set_label(r"$\log_{10}\max(1+\delta)$ through the slab", fontsize=9)
        fig.suptitle(
            f"{box} host {host}, tile {int(tile)} -- slab z = [{lo}, {hi}) cells, "
            f"{n_ring} of {live.size} true HR subhalos ringed",
            color=INK, fontsize=12)
        p = out_dir / f"step{step:06d}_tile{int(tile):03d}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True,
                    help="a host_gather run directory (holds eval/step*.npz)")
    ap.add_argument("--slab", type=int, default=4)
    ap.add_argument("--last-only", action="store_true",
                    help="redraw only the final eval step")
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    files = sorted(glob.glob(str(run / "eval" / "step*.npz")))
    if not files:
        raise SystemExit(f"no eval/step*.npz under {run}")
    if args.last_only:
        files = files[-1:]
    n = 0
    for f in files:
        for p in render_npz(f, run / "slices", slab=args.slab):
            n += 1
            print(f"  wrote {p}")
    print(f"{n} figures from {len(files)} eval steps -> {run / 'slices'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
