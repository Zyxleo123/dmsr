#!/usr/bin/env python
"""Two figures from the dumped slices: one for displacement, one for velocity.

Three columns -- candidate (our reproduction), frozen SR2, HR -- and two rows:
the full 512-voxel slice, and a zoom. The zoom is not decoration. All three
fields agree below k=32 by construction, so at full-box scale they look alike;
the missing octave only becomes visible once the panel is small enough for
single-voxel texture to resolve.

**One shared colour scale per row**, taken from HR. Per-panel autoscaling would
be actively misleading here: a field with a fraction of HR's small-scale power
would be stretched to fill the same colour range and could look MORE textured
than the target. The scale is symmetric about zero because these are signed
displacement/velocity components.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLS = (("cand", "reproduction"), ("frozen", "frozen SR2"), ("hr", "HR (target)"))


def one_figure(z, name: str, out: Path, cmap: str = "RdBu_r") -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.8))
    for r, kind in enumerate(("full", "zoom")):
        # HR sets the scale for the row; see the module docstring.
        ref = z[f"{name}_hr_{kind}"]
        v = float(np.percentile(np.abs(ref), 99.5))
        for c, (key, title) in enumerate(COLS):
            ax = axes[r, c]
            im = ax.imshow(z[f"{name}_{key}_{kind}"], origin="lower",
                           cmap=cmap, vmin=-v, vmax=v, interpolation="nearest")
            if r == 0:
                ax.set_title(title, fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                side = z[f"{name}_hr_{kind}"].shape[0]
                ax.set_ylabel(f"{kind}  ({side}$^2$ voxels)", fontsize=10)
        fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.020, pad=0.01)

    ck = Path(str(z["ckpt"])).name
    fig.suptitle(f"{name} (one component) -- {ck}, epoch {int(z['epoch'])}, "
                 f"{str(z['box'])};  shared colour scale from HR", fontsize=13)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="from m2m_slice_dump.py")
    ap.add_argument("--outdir", default="")
    a = ap.parse_args(argv)

    z = np.load(a.npz, allow_pickle=True)
    outdir = Path(a.outdir) if a.outdir else Path(a.npz).parent
    stem = Path(a.npz).stem
    for name in ("displacement", "velocity"):
        p = one_figure(z, name, outdir / f"{stem}_{name}.png")
        print(f"wrote {p.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
