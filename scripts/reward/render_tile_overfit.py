#!/usr/bin/env python
"""Render the tile-overfit process to a GIF + a before/after still. CPU.

Reads tile_overfit/tile_overfit_<tag>.npz written by overfit_tile_to_proxy.py.
No model, no GPU -- redrawable from the saved snapshots. ffmpeg is not available,
so the animation is written with matplotlib's pillow (GIF) writer.

    python scripts/reward/render_tile_overfit.py --run-name direct_a --tag set0_t489_c
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.animation as animation  # noqa: E402

from _sr2_direct import run_dir  # noqa: E402

INK = "#1a1a1a"
MUTED = "#6b6b6b"
ACCENT = "#D55E00"
REAL_C = "#009E73"      # real (Rockstar) measurements, distinct from proxy ACCENT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--tag", required=True, help="e.g. set0_t489_c")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args(argv)

    d = run_dir(args.run_name) / "tile_overfit"
    npz_p = d / f"tile_overfit_{args.tag}.npz"
    if not npz_p.is_file():
        print(f">>> MISSING INPUT: {npz_p}")
        print(">>> produced by: scripts/reward/overfit_tile_to_proxy.py")
        return 0
    z = np.load(npz_p)
    frames = z["frames"]                 # (F, R, R) log1p density mid-slice
    fsteps = z["frame_steps"]
    q_hist, iters = z["q_hist"], z["iters"]
    dro = z["dr_occ_hist"]
    summ = json.loads((d / f"tile_overfit_{args.tag}.json").read_text())

    # Real Rockstar rows, if the monitor has run. Same reward (dR_occ) as the
    # proxy, so they overlay on one axis; reliable host bins headline the counts.
    reliable = [2, 3]
    mon = d / f"rockstar_monitor_{args.tag}"
    rock = sorted((json.loads(p.read_text()) for p in mon.glob("iter_*.json")),
                  key=lambda r: r["iter"]) if mon.is_dir() else []

    def rel_sum(r, key):
        return float(np.sum([np.asarray(r[key], float)[b] for b in reliable
                             if b < len(r[key])]))
    if rock:
        rk_iter = np.array([r["iter"] for r in rock])
        rk_docc = np.array([r["measured_dR_occ"] for r in rock], float)
        rk_nh = np.array([rel_sum(r, "n_host_after") for r in rock])
        rk_nh0 = rel_sum(rock[0], "n_host_before")

    vmax = float(np.percentile(frames, 99.5))
    vmin = float(frames.min())

    # ---- static before/after -------------------------------------------
    d0, d1 = z["density_initial"], z["density_final"]
    mid = d0.shape[-1] // 2
    s0 = np.log1p(np.clip(d0[:, :, mid], 0, None))
    s1 = np.log1p(np.clip(d1[:, :, mid], 0, None))
    diff = s1 - s0
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    vm = float(np.percentile(np.concatenate([s0, s1]), 99.5))
    ax[0].imshow(s0, cmap="magma", vmin=0, vmax=vm); ax[0].set_title(
        f"input  (SR2 tile)   Q={summ['q_start']:+.3f}", color=INK)
    ax[1].imshow(s1, cmap="magma", vmin=0, vmax=vm); ax[1].set_title(
        f"output (after {summ['iters_run']} steps)   Q={summ['q_end']:+.3f}", color=INK)
    dm = float(np.percentile(np.abs(diff), 99)) or 1.0
    im = ax[2].imshow(diff, cmap="RdBu_r", vmin=-dm, vmax=dm)
    ax[2].set_title("output - input  (log density)", color=INK)
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    title = (f"Tile {summ['box']}/t{summ['tile']} overfit to proxy arm {summ['arm']}  |  "
             f"PROXY dR_occ {summ['dR_occ_start']:+.3f} -> {summ['dR_occ_end']:+.3f}  |  "
             f"low_k_change {summ['low_k_change_end']:.1e} (guarded)")
    if rock:
        title += (f"\nREAL (Rockstar) dR_occ {rk_docc[0]:+.3f} -> {rk_docc[-1]:+.3f}"
                  f"   |   real n_host bins{reliable} {rk_nh0:.1f} -> {rk_nh[-1]:.1f}"
                  f"   (iter0 real {rk_docc[0]:+.2g}, splice=frozen)")
    fig.suptitle(title, color=INK, fontsize=11)
    still = d / f"tile_overfit_{args.tag}_beforeafter.png"
    fig.savefig(still, dpi=130); plt.close(fig)

    # ---- animation: density slice (left) + reward curve (right) ---------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    imL = axL.imshow(frames[0], cmap="magma", vmin=vmin, vmax=vmax, animated=True)
    axL.set_xticks([]); axL.set_yticks([])
    ttl = axL.set_title("", color=INK)
    fig.colorbar(imL, ax=axL, fraction=0.046, label="log(1+density)")

    axR.plot(iters, q_hist, color=ACCENT, lw=1.8, label="proxy Q_safe")
    axR.plot(iters, dro, color=MUTED, lw=1.2, label="proxy dR_occ")
    if rock:
        # same reward as proxy dR_occ, measured -- the gap is the self-deception
        axR.plot(rk_iter, rk_docc, "s-", color=REAL_C, lw=1.4, ms=7,
                 label="REAL dR_occ (Rockstar)")
    axR.set_xlabel("iteration"); axR.set_ylabel("reward (dR_occ / Q_safe)")
    axR.legend(loc="lower right", frameon=False, fontsize=9)
    axR.grid(True, color="#e6e6e6")
    marker = axR.axvline(0, color=INK, lw=1.2, ls="--")
    dot, = axR.plot([iters[0]], [q_hist[0]], "o", color=ACCENT, ms=7)

    def upd(i):
        imL.set_data(frames[i])
        step = int(fsteps[i])
        ttl.set_text(f"step {step}   Q_safe {np.interp(step, iters, q_hist):+.4f}")
        marker.set_xdata([step, step])
        dot.set_data([step], [np.interp(step, iters, q_hist)])
        return imL, ttl, marker, dot

    anim = animation.FuncAnimation(fig, upd, frames=len(frames),
                                   interval=1000 // max(1, args.fps), blit=False)
    gif = d / f"tile_overfit_{args.tag}.gif"
    anim.save(gif, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)

    print(f"wrote {still}")
    print(f"wrote {gif}  ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
