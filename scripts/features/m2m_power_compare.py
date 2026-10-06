#!/usr/bin/env python
"""Overlay several arms' P(k) ratio curves on one axis, per field.

``m2m_power_plot.py`` draws ONE checkpoint against frozen SR2 and HR, which
answers "where does this candidate stop tracking". It cannot answer the question
an A/B actually asks -- does arm X hold more power than arm Y, and at which k --
because the two live in different figures and the eye cannot difference two log
plots across a page. This reads several npz and puts the arms on one axis.

CPU only, no generator, no checkpoint: the curves were written to disk precisely
so a figure costs nothing to redraw.

Colours are the dataviz skill's validated categorical order, taken unchanged and
assigned in fixed order to ARMS (identity, not rank), so an arm keeps its hue
whatever else is plotted. Frozen SR2 is drawn as a grey dashed REFERENCE rather
than a categorical slot: it is the target to beat, not another arm, and giving it
a series hue would read as a fourth candidate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#: dataviz categorical theme, light mode, slots 1-3. Fixed order, never cycled.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
REF_GREY = "0.35"
INK = "#0b0b0b"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", action="append", required=True, metavar="LABEL=PATH",
                    help="repeatable; the label is what the legend shows")
    ap.add_argument("--fields", default="displacement,density",
                    help="comma separated (default: displacement,density)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args(argv)

    fields = [f.strip() for f in a.fields.split(",") if f.strip()]
    arms = []
    for spec in a.npz:
        if "=" not in spec:
            raise SystemExit(f"--npz wants LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        if not Path(path).is_file():
            print(f"  skip (missing): {path}")
            continue
        arms.append((label, np.load(path, allow_pickle=True)))
    if not arms:
        raise SystemExit("no npz could be read")

    kny = float(arms[0][1]["lr_nyquist"])
    box = str(arms[0][1]["box"])

    fig, axes = plt.subplots(1, len(fields), figsize=(6.0 * len(fields), 4.8),
                             squeeze=False)
    for j, name in enumerate(fields):
        ax = axes[0, j]
        # frozen SR2 first so the arms draw over it
        z0 = arms[0][1]
        k0 = z0[f"{name}_k"]
        m0 = k0 > 0
        den0 = np.clip(z0[f"{name}_pk_hr"], 1e-30, None)
        ax.loglog(k0[m0], (z0[f"{name}_pk_frozen"] / den0)[m0],
                  color=REF_GREY, lw=1.8, ls="--", label="frozen SR2 (target)")
        for i, (label, z) in enumerate(arms):
            k = z[f"{name}_k"]
            m = k > 0
            den = np.clip(z[f"{name}_pk_hr"], 1e-30, None)
            ax.loglog(k[m], (z[f"{name}_pk_sr"] / den)[m],
                      color=SERIES[i % len(SERIES)], lw=2.0, label=label)
        ax.axhline(1.0, color=INK, lw=1.0, alpha=0.5)
        ax.axvline(kny, color=REF_GREY, ls=":", lw=1.2)
        # Anchored to the BOTTOM of the panel, not to y=1: the density curves
        # sit on 1.0 and a label there lands on top of the data.
        ax.annotate(f"LR Nyquist k={kny:g}", xy=(kny, 2e-5), xytext=(5, 2),
                    textcoords="offset points", fontsize=8, color=REF_GREY)
        ax.set_title(name, fontsize=12, color=INK)
        ax.set_xlabel("k [cycles per box]", color=INK)
        if j == 0:
            ax.set_ylabel("P / P_HR   (1.0 = matches HR)", color=INK)
        ax.set_ylim(1e-5, 1e2)
        ax.grid(alpha=0.22, which="both")
        # A legend on every panel repeats identical entries; one is enough, and
        # it goes where the curves are lowest so it covers no data.
        if j == 0:
            ax.legend(fontsize=9, loc="lower left", framealpha=0.9)

    fig.suptitle(a.title or f"P(k) ratio to HR on {box}", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(a.out, dpi=140)
    print(f"wrote {Path(a.out).resolve()}")
    for label, z in arms:
        for name in fields:
            k, r = z[f"{name}_k"], z[f"{name}_pk_sr"] / np.clip(z[f"{name}_pk_hr"], 1e-30, None)
            ok = np.where(r >= 0.5)[0]
            kh = float(k[ok[-1]]) if ok.size else float("nan")
            print(f"  {label:22s} {name:13s} k_half = {kh:7.1f}  ({kh / kny:.2f} x Nyquist)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
