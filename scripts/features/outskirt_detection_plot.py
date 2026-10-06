#!/usr/bin/env python
"""Redraw the outskirt-detection figures from the probe's JSON. CPU, seconds.

Three panels, and the order is the argument:

``fig1_radial``
    hit rate against host-centric radius, one line per arm, with the median
    ``num_p`` of each bin annotated. The annotation is the control: the gap is
    a radial result only while the mass is flat across the bins.
``fig2_address``
    the decomposition -- ``P(addressed)`` and ``P(hit | addressed)`` against
    radius. Whichever of the two carries the trend is the defect to fix, and
    the panel exists so that cannot be read off a single pooled number.
``fig3_contrast``
    hit vs miss on every per-set axis, as a ratio, so the axes that separate
    the two are visible against the ones that do not.

Reads only ``<probe_out>/<arm>.json``; no catalog, no field.

    python scripts/features/outskirt_detection_plot.py \\
        --probe-dir /.../outskirt_probe/set9 --arms all_blocks_self ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reward"))
from _common import banner  # noqa: E402

CONTRAST_AXES = (
    ("median_offset_mpc_h", "|centroid - target|"),
    ("median_r_rms_over_rvir_hr", "r_rms / Rvir_HR"),
    ("median_virial", "2T/|W|"),
    ("median_bound_hard", "bound_hard"),
    ("median_sigma_v", "sigma_v"),
    ("median_num_p", "num_p"),
)


def _binned(prof: List[Dict], key: str):
    rows = [r for r in prof if not r["all_radii"]]
    x = [0.5 * (r["r_lo"] + min(r["r_hi"], 2.0)) for r in rows]
    return np.array(x), np.array([r[key] for r in rows], dtype=float), rows


def fig_radial(recs: Dict[str, Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for arm, rec in recs.items():
        x, y, rows = _binned(rec["radial_profile"], "hit_rate")
        ax.plot(x, 100 * y, "o-", label=f"{arm} ({rec['searched_at']})")
    _, _, rows = _binned(next(iter(recs.values()))["radial_profile"], "hit_rate")
    for r, xx in zip(rows, _binned(next(iter(recs.values()))["radial_profile"],
                                   "hit_rate")[0]):
        ax.annotate(f"{r['median_num_p']:.0f}p\nn={r['n']}", (xx, 0),
                    xytext=(0, -34), textcoords="offset points",
                    ha="center", fontsize=7, color="0.4")
    ax.set_xlabel("host-centric radius  r / R_vir  (own parent)")
    ax.set_ylabel("supervised targets recovered [%]")
    ax.set_title("Rockstar recovers the inner supervised sets, not the outskirt ones\n"
                 "(median num_p per bin annotated: the mass control)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_address(recs: Dict[str, Dict], out: Path, tol: float = 0.30) -> None:
    have = {a: r for a, r in recs.items() if r["address_decomposition"]}
    if not have:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    for arm, rec in have.items():
        rows = [r for r in rec["address_decomposition"]
                if abs(r["tol_mpc_h"] - tol) < 1e-9 and not r["all_radii"]]
        x = [0.5 * (r["r_lo"] + min(r["r_hi"], 2.0)) for r in rows]
        axes[0].plot(x, [100 * r["p_addressed"] for r in rows], "o-", label=arm)
        axes[1].plot(x, [100 * r["hit_rate_given_addressed"] for r in rows],
                     "o-", label=arm)
        axes[1].plot(x, [100 * r["hit_rate_given_unaddressed"] for r in rows],
                     "x--", alpha=0.5)
    axes[0].set_ylabel(f"P(centroid within {tol:g} Mpc/h of target) [%]")
    axes[0].set_title("placement: is the material where HR's subhalo is?")
    axes[1].set_ylabel("recovered [%]")
    axes[1].set_title("construction: given it IS there, does Rockstar see it?\n"
                      "solid = addressed, dashed x = not addressed")
    for a in axes:
        a.set_xlabel("r / R_vir (own parent)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_contrast(recs: Dict[str, Dict], out: Path) -> None:
    labels = [lab for _, lab in CONTRAST_AXES]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    w = 0.8 / max(len(recs), 1)
    for i, (arm, rec) in enumerate(recs.items()):
        con = rec["hit_miss_contrast"]
        vals = []
        for key, _ in CONTRAST_AXES:
            h, m = con["hit"].get(key, np.nan), con["miss"].get(key, np.nan)
            vals.append(h / m if (np.isfinite(h) and np.isfinite(m) and m) else np.nan)
        ax.bar(np.arange(len(labels)) + i * w, vals, width=w, label=arm)
    ax.axhline(1.0, color="k", lw=1)
    ax.set_xticks(np.arange(len(labels)) + 0.4 - w / 2)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("median(hit) / median(miss)")
    ax.set_title("which axis separates a recovered target from a lost one\n"
                 "(1.0 = no separation)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--address-tol", type=float, default=0.30)
    args = ap.parse_args(argv)

    d = Path(args.probe_dir)
    names = args.arms or [p.stem for p in sorted(d.glob("*.json"))
                          if p.stem != "summary"]
    recs: Dict[str, Dict] = {}
    for a in names:
        p = d / f"{a}.json"
        if p.is_file():
            recs[a] = json.loads(p.read_text())
        else:
            print(f"  {a}: no {p}, skipped")
    if not recs:
        raise SystemExit(f"no arm JSON under {d} -- run the probe first")

    banner(f"outskirt detection figures -- {len(recs)} arm(s)")
    fig_radial(recs, d / "fig1_radial.png")
    fig_address(recs, d / "fig2_address.png", tol=args.address_tol)
    fig_contrast(recs, d / "fig3_contrast.png")
    for f in ("fig1_radial.png", "fig2_address.png", "fig3_contrast.png"):
        if (d / f).is_file():
            print(f"  wrote {d / f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
