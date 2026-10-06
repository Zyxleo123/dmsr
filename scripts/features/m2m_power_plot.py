#!/usr/bin/env python
"""Plot the P(k) curves ``m2m_power_curves.py`` saved. Reads the npz, nothing else.

CPU only and no generator: the point of writing curves to disk was that the
figure can be redrawn without regenerating a 512^3 box, so this must never need
a GPU or a checkpoint.

What the figure has to make legible is one thing: WHERE the candidate stops
tracking HR. The LR input is 64^3 and so carries nothing above k = 32, and the
measured falloff sits exactly there -- displacement drops from 0.148 at k=32 to
9.6e-04 at k=64, three orders of magnitude in one octave, while frozen SR2 holds
0.94 across the same boundary. So the Nyquist line is drawn on every panel and
the ratio panel is the one that carries the argument; the raw spectra are kept
above it because a ratio alone cannot show that P_HR is well-determined up there
(it is: 7e-03 at k=128), which is what rules out a near-empty-bin artefact in
the reduction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIELDS = ("displacement", "velocity", "density")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="from m2m_power_curves.py")
    ap.add_argument("--out", default="", help="default: alongside the npz")
    ap.add_argument("--title", default="")
    a = ap.parse_args(argv)

    z = np.load(a.npz, allow_pickle=True)
    kny = float(z["lr_nyquist"])
    box = str(z["box"])
    ckpt = Path(str(z["ckpt"])).name

    fig, axes = plt.subplots(2, len(FIELDS), figsize=(5.0 * len(FIELDS), 7.4),
                             sharex=True)
    for j, name in enumerate(FIELDS):
        k = z[f"{name}_k"]
        p_sr, p_fz, p_hr = (z[f"{name}_pk_sr"], z[f"{name}_pk_frozen"],
                            z[f"{name}_pk_hr"])
        m = k > 0

        ax = axes[0, j]
        ax.loglog(k[m], p_hr[m], color="k", lw=2.0, label="HR (target)")
        ax.loglog(k[m], p_fz[m], color="tab:blue", lw=1.6, label="frozen SR2")
        ax.loglog(k[m], p_sr[m], color="tab:red", lw=1.6, label="candidate")
        ax.axvline(kny, color="0.4", ls="--", lw=1.2)
        ax.set_title(name, fontsize=12)
        ax.set_ylabel("P(k)" if j == 0 else "")
        ax.grid(alpha=0.25, which="both")
        if j == 0:
            ax.legend(fontsize=9, loc="lower left")

        ax = axes[1, j]
        den = np.clip(p_hr, 1e-30, None)
        ax.loglog(k[m], (p_fz / den)[m], color="tab:blue", lw=1.6)
        ax.loglog(k[m], (p_sr / den)[m], color="tab:red", lw=1.6)
        ax.axhline(1.0, color="k", lw=1.0, alpha=0.6)
        ax.axvline(kny, color="0.4", ls="--", lw=1.2)
        # The label goes on one panel only; repeated on three it is clutter.
        if j == 0:
            ax.set_ylabel("P / P_HR")
            ax.annotate(f"LR Nyquist\nk = {kny:g}", xy=(kny, 1.0),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color="0.3")
        ax.set_xlabel("k [cycles per box]")
        ax.set_ylim(1e-7, 1e2)
        ax.grid(alpha=0.25, which="both")

    fig.suptitle(a.title or f"{ckpt} on {box}: the candidate stops at the LR Nyquist",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(a.out) if a.out else Path(a.npz).with_suffix(".png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
