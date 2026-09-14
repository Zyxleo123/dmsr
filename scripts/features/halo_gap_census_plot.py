#!/usr/bin/env python
"""Figures for ``halo_gap_census.py``, redrawn from ``census.json`` alone.

No catalog is reparsed and no field is read: the census writes every count it
measured, so a plotting change costs seconds rather than a resubmission.

Three panels, in the order the decision is made:

  1. the top-level mass function, HR vs base vs any arms -- the counts.
  2. ``base/HR`` per mass bin, for top-level and for subhalos -- the SHAPE that
     separates a small-halo deficit from a size-independent one, which is the
     verdict. A flat line means a mass window buys nothing.
  3. the top-level gap split by environment -- whether the missing halos sit
     where a cluster's Lagrangian tiles could reach them, which decides whether
     the answer is a wider POPULATION or a wider TILING.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# base is the reference the gap is measured from, HR is truth, arms are extra.
COLORS = {"hr": "#111111", "base": "#c1121f", "self": "#0077b6"}
FALLBACK = ["#2a9d8f", "#e07a5f", "#7209b7", "#606c38"]


def _color(kind: str, i: int) -> str:
    return COLORS.get(kind, FALLBACK[i % len(FALLBACK)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scope", default="",
                    help="'region' or 'box'; default prefers region when the "
                         "census measured one, because for an ARM the box is "
                         "93.75%% untouched frozen SR2")
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir)
    cen = json.loads((out_dir / "census.json").read_text())
    scope = a.scope or ("region" if "region" in cen["scopes"] else "box")
    sc = cen["scopes"][scope]
    labels = cen["mass_labels"]
    x = np.arange(len(labels))
    kinds = cen["kinds"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    for i, k in enumerate(kinds):
        ax.plot(x, sc["counts"][k]["top"]["by_mass"], "o-", label=k,
                color=_color(k, i), lw=2)
    ax.set_yscale("log")
    ax.set_title(f"top-level halo mass function ({scope})")
    ax.set_ylabel("count")

    ax = axes[1]
    for i, k in enumerate(kinds):
        if k == "hr":
            continue
        for pop, ls in (("top", "-"), ("sub", "--")):
            r = [np.nan if v is None else v for v in sc["gaps"][k][pop]["ratio"]]
            ax.plot(x, r, ls, marker="o", color=_color(k, i),
                    label=f"{k} {pop}")
    ax.axhline(1.0, color="#111111", lw=1, ls=":")
    ax.set_ylim(0, 1.6)
    # A FLAT line is the whole point of the panel: it means the deficit is
    # size-independent and a num_p window buys nothing.
    ax.set_title("candidate / HR per mass bin\n(flat = size-independent deficit)")
    ax.set_ylabel("count ratio")

    ax = axes[2]
    envs = cen["env_labels"]
    w = 0.8 / len(envs)
    for j, e in enumerate(envs):
        ax.bar(x + j * w - 0.4 + w / 2,
               sc["gaps"]["base"]["top_by_env"][e]["gap"], width=w, label=e)
    ax.axhline(0.0, color="#111111", lw=1)
    ax.set_title("top-level gap HR - base, by environment\n"
                 "(distance to nearest bigger host, in its R_vir)")
    ax.set_ylabel("missing halos")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("num_p")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    # The verdict is the deliverable, so it goes on the figure in full rather
    # than truncated -- a reader who has the png has the decision.
    import textwrap
    fig.suptitle("\n".join(textwrap.wrap(
        f"{cen['box']} halo-count census ({scope}) -- {sc['verdict']['text']}",
        width=170)), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / f"halo_gap_census_{scope}.png"
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
