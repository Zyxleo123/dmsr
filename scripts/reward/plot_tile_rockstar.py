#!/usr/bin/env python
"""Overlay the REAL (Rockstar) reward on the proxy's during a tile overfit. CPU.

Reads tile_overfit/rockstar_monitor_<tag>/iter_*.json (real, sparse) and
tile_overfit/metrics_<tag>.jsonl (proxy, dense), and draws:
  1. proxy dR_occ (dense line) vs measured dR_occ (markers) -- SAME reward, one
     axis, so the gap between them is exactly how much the proxy is fooling
     itself. iter 0 measured ~0 is the sanity check.
  2. real reliable-bin host + occ_numerator counts vs iter -- did REAL occupancy
     move, or only the proxy's opinion?

    python scripts/reward/plot_tile_rockstar.py --run-name direct_a --tag set0_t486_c
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _sr2_direct import load_direct_config, run_dir  # noqa: E402

INK, MUTED = "#1a1a1a", "#6b6b6b"
PROXY_C, REAL_C, ALT_C = "#0072B2", "#D55E00", "#009E73"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", default="configs/reward/sr2_direct_finetune.yaml")
    args = ap.parse_args(argv)

    of = run_dir(args.run_name) / "tile_overfit"
    mon = of / f"rockstar_monitor_{args.tag}"
    rows = sorted((json.loads(p.read_text()) for p in mon.glob("iter_*.json")),
                  key=lambda r: r["iter"])
    if not rows:
        print(f">>> MISSING INPUT: {mon}/iter_*.json")
        print(">>> produced by: scripts/reward/rockstar_monitor_tile.py")
        return 0
    proxy = [json.loads(l) for l in
             (of / f"metrics_{args.tag}.jsonl").read_text().splitlines() if l.strip()]

    # reliable host bins to headline: the config's own upper-reliable set (2,3).
    cfg = load_direct_config(args)
    reliable = [int(b) for b in
                cfg.get("overfit", {}).get("upper_reliable_host_bins", [2, 3])]

    r_iter = np.array([r["iter"] for r in rows])
    r_docc = np.array([r["measured_dR_occ"] for r in rows], dtype=float)
    p_iter = np.array([r["iter"] for r in proxy])
    p_docc = np.array([r["dR_occ"] for r in proxy], dtype=float)

    def reliable_sum(r, key):
        return float(np.sum([np.asarray(r[key], float)[b] for b in reliable
                             if b < len(r[key])]))
    nh_before = reliable_sum(rows[0], "n_host_before")
    nh = np.array([reliable_sum(r, "n_host_after") for r in rows])
    occ = np.array([reliable_sum(r, "occ_numerator_after") for r in rows])
    occ_before = reliable_sum(rows[0], "occ_numerator_before")

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "font.size": 10, "axes.grid": True, "grid.color": "#e8e8e8"})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # panel 1: proxy vs real dR_occ (same reward, same axis)
    axL.plot(p_iter, p_docc, color=PROXY_C, lw=1.6, label="proxy dR_occ (surrogate)")
    axL.plot(r_iter, r_docc, "o-", color=REAL_C, lw=1.6, ms=7,
             label="real dR_occ (Rockstar)")
    axL.axhline(0, color=MUTED, lw=1)
    axL.set_xlabel("iteration"); axL.set_ylabel("dR_occ  (occupation reward change)")
    axL.set_title("Proxy reward vs the real thing\n"
                  "gap = how much the surrogate fools itself", color=INK)
    axL.legend(frameon=False, fontsize=9)
    # annotate the sanity check + final divergence
    if r_iter[0] == 0:
        axL.annotate(f"iter0 real {r_docc[0]:+.2g}\n(splice=frozen, ~0 expected)",
                     (r_iter[0], r_docc[0]), fontsize=8, color=REAL_C,
                     xytext=(8, 12), textcoords="offset points")

    # panel 2: real reliable-bin counts vs iter
    axR.axhline(nh_before, color=REAL_C, ls=":", lw=1.2,
                label=f"n_host frozen ({nh_before:.1f})")
    axR.plot(r_iter, nh, "o-", color=REAL_C, lw=1.6, ms=7, label="n_host (real)")
    axR.axhline(occ_before, color=ALT_C, ls=":", lw=1.2,
                label=f"occ_numerator frozen ({occ_before:.1f})")
    axR.plot(r_iter, occ, "s-", color=ALT_C, lw=1.6, ms=6,
             label="occ_numerator=hosted subs (real)")
    axR.set_xlabel("iteration")
    axR.set_ylabel(f"real count in reliable host bins {reliable}")
    axR.set_title("Did REAL occupancy move?", color=INK)
    axR.legend(frameon=False, fontsize=8)

    fig.suptitle(f"Tile overfit reality check  |  run {args.run_name}  {args.tag}  "
                 f"|  {len(rows)} Rockstar snapshots", color=INK, fontsize=12)
    out = mon / f"tile_rockstar_{args.tag}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"wrote {out}")
    # a compact verdict line for the log
    print(f"  proxy dR_occ end {p_docc[-1]:+.4g}  vs  real dR_occ end "
          f"{r_docc[-1]:+.4g}   (iter0 real {r_docc[0]:+.3g})")
    print(f"  real reliable n_host {nh_before:.1f} -> {nh[-1]:.1f}   "
          f"occ_numerator {occ_before:.1f} -> {occ[-1]:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
