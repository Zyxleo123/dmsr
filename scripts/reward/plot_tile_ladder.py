#!/usr/bin/env python
"""Compare the tile-overfit control ladder: where does dR_occ go non-positive?

Reads each rung's proxy metrics + real Rockstar rows and draws, per rung
(ordered as given = increasing control):
  1. final proxy dR_occ vs final real dR_occ, with y=0 -- the crossover where
     control finally holds the reward non-positive.
  2. real reliable-bin host + occ_numerator CHANGE (final - frozen) -- whether
     any rung actually moved real occupancy (spoiler expected: no).

    python scripts/reward/plot_tile_ladder.py --run-name direct_a \
        --labels ctlA,ctlB,ctlC,ctlD --box set0 --tile 486 --arm c
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

INK, MUTED = "#1a1a1a", "#6b6b6b"
PROXY_C, REAL_C, ALT_C = "#0072B2", "#D55E00", "#009E73"
RELIABLE = [2, 3]


def rel_sum(vec):
    v = np.asarray(vec, float)
    return float(np.sum([v[b] for b in RELIABLE if b < len(v)]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--labels", required=True, help="comma list, low->high control")
    ap.add_argument("--box", default="set0")
    ap.add_argument("--tile", type=int, default=486)
    ap.add_argument("--arm", default="c")
    args = ap.parse_args(argv)

    of = run_dir(args.run_name) / "tile_overfit"
    labels = [s for s in args.labels.split(",") if s]
    rows = []
    for lab in labels:
        tag = f"{args.box}_t{args.tile}_{args.arm}__{lab}"
        mp = (of / f"metrics_{tag}.jsonl")
        mon = of / f"rockstar_monitor_{tag}"
        if not mp.is_file():
            print(f">>> missing proxy metrics for {tag}; skipping")
            continue
        proxy = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
        rk = sorted((json.loads(p.read_text()) for p in mon.glob("iter_*.json")),
                    key=lambda r: r["iter"]) if mon.is_dir() else []
        rows.append({
            "label": lab, "tag": tag,
            "proxy_docc_end": proxy[-1]["dR_occ"] if proxy else np.nan,
            "real_docc_end": rk[-1]["measured_dR_occ"] if rk else np.nan,
            "real_docc0": rk[0]["measured_dR_occ"] if rk else np.nan,
            "d_nhost": (rel_sum(rk[-1]["n_host_after"]) - rel_sum(rk[0]["n_host_before"]))
                       if rk else np.nan,
            "d_occ": (rel_sum(rk[-1]["occ_numerator_after"])
                      - rel_sum(rk[0]["occ_numerator_before"])) if rk else np.nan,
            "has_real": bool(rk),
        })
    if not rows:
        print(">>> no rungs found; run the ladder first")
        return 0

    x = np.arange(len(rows))
    labs = [r["label"] for r in rows]
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "font.size": 10, "axes.grid": True, "grid.color": "#e8e8e8"})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # panel 1: proxy vs real final dR_occ per rung
    p_end = np.array([r["proxy_docc_end"] for r in rows])
    r_end = np.array([r["real_docc_end"] for r in rows])
    axL.plot(x, p_end, "o-", color=PROXY_C, lw=1.8, ms=8, label="proxy dR_occ (end)")
    if np.any(np.isfinite(r_end)):
        axL.plot(x, r_end, "s-", color=REAL_C, lw=1.8, ms=8,
                 label="real dR_occ (end, Rockstar)")
    axL.axhline(0, color=INK, lw=1.2)
    axL.fill_between(x, 0, axL.get_ylim()[1], color="#c0392b", alpha=0.05)
    axL.set_xticks(x); axL.set_xticklabels(labs)
    axL.set_xlabel("control rung (increasing ->)")
    axL.set_ylabel("final dR_occ")
    axL.set_title("Crossover: where control holds dR_occ <= 0\n"
                  "(shaded = reward still running away)", color=INK)
    axL.legend(frameon=False, fontsize=9)
    for xi, v in zip(x, p_end):
        if np.isfinite(v):
            axL.annotate(f"{v:+.0f}", (xi, v), fontsize=8, color=PROXY_C,
                         xytext=(0, 6), textcoords="offset points", ha="center")

    # panel 2: real occupancy CHANGE per rung (did anything real move?)
    dnh = np.array([r["d_nhost"] for r in rows])
    docc = np.array([r["d_occ"] for r in rows])
    w = 0.38
    axR.bar(x - w / 2, dnh, w, color=REAL_C, label=f"Δ n_host bins{RELIABLE}")
    axR.bar(x + w / 2, docc, w, color=ALT_C, label="Δ occ_numerator (hosted subs)")
    axR.axhline(0, color=INK, lw=1)
    axR.set_xticks(x); axR.set_xticklabels(labs)
    axR.set_xlabel("control rung (increasing ->)")
    axR.set_ylabel("real count change (final - frozen)")
    axR.set_title("Did REAL occupancy move on ANY rung?", color=INK)
    axR.legend(frameon=False, fontsize=9)

    fig.suptitle(f"Tile-overfit control ladder  |  {args.run_name}  "
                 f"{args.box}/t{args.tile} arm {args.arm}", color=INK, fontsize=12)
    out = of / f"tile_ladder_{args.box}_t{args.tile}_{args.arm}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"wrote {out}")
    for r in rows:
        print(f"  {r['label']:6s} proxy dR_occ_end {r['proxy_docc_end']:+8.2f}  "
              f"real dR_occ_end {r['real_docc_end']:+8.2f}  "
              f"Δn_host {r['d_nhost']:+.1f}  Δocc {r['d_occ']:+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
