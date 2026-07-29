#!/usr/bin/env python
"""How big must a training crop be for its OWN particles to give a valid density?

Stage 2 established that scoring a 64^3 Eulerian region needs particles from a
64 HR-cell Lagrangian buffer around it. That is fatal for crop-level density
supervision as currently built: a 64^3 HR crop (``crop_lr=8``) cannot supply its
own buffer, so the density critic can never see a correct field.

But most of that 64 cells is **bulk translation**. The median |Psi| on set14 is 36
HR cells while the differential motion that actually builds haloes is much
smaller, and a coherent translation of a whole crop moves matter without changing
its clustering. Depositing at ``q + (Psi - <Psi>_crop)`` removes that mode, which
should shrink the buffer a lot.

This script measures the buffer requirement with and without the bulk mode, which
sets the minimum crop size for Branch A::

    crop_hr >= region_hr + 2 * buffer_required

Both variants are scored against the SAME exact reference (the true, absolutely
positioned, fully buffered density of the region), because the question is not
"does bulk subtraction converge to something" but "does it converge to the truth".
Bulk subtraction shifts the whole region, so the comparison is made against the
truth field shifted by the same amount -- i.e. we ask whether the region's
INTERNAL structure is recovered, which is what a high-pass density critic sees.

Usage
-----
    python scripts/dmsr_bulkflow_buffer.py \
        --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set14.npy \
        --regions 32 64 --out runs/dmsr/stage2b_bulkflow
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dmsr_cic_buffer_audit import (  # noqa: E402
    cic_block_into_region,
    cic_into_region,
    region_metrics,
)


def block_disp(hr, lo, side, ngrid, scale):
    """``(3, side, side, side)`` displacement in HR cells, periodic, plane-by-plane."""
    i1 = np.mod(np.arange(lo[1], lo[1] + side), ngrid)
    i2 = np.mod(np.arange(lo[2], lo[2] + side), ngrid)
    d = np.empty((3, side, side, side), dtype=np.float32)
    for s in range(side):
        p = int((lo[0] + s) % ngrid)
        d[:, s] = np.asarray(hr[0:3, p], dtype=np.float32)[:, i1][:, :, i2]
    d *= scale
    return d


def deposit(disp, lo, origin, R, ngrid, subtract_bulk):
    """CIC the block into the scored cube; optionally remove its mean displacement."""
    side = disp.shape[-1]
    d = disp
    bulk = np.zeros(3)
    if subtract_bulk:
        # Round to whole cells before subtracting: a fractional shift would leave
        # the scored cube misaligned by up to half a cell against the reference and
        # decorrelate the comparison for reasons that have nothing to do with the
        # buffer. Whole cells remove the bulk mode to within 0.5 cell, which is all
        # that is needed to shrink the buffer.
        bulk = np.round(d.reshape(3, -1).mean(axis=1))
        d = d - bulk[:, None, None, None]
    q = [np.arange(lo[i], lo[i] + side, dtype=np.float64) + 0.5 for i in range(3)]
    pos = np.empty((3, side, side, side), dtype=np.float64)
    pos[0] = d[0] + q[0][:, None, None]
    pos[1] = d[1] + q[1][None, :, None]
    pos[2] = d[2] + q[2][None, None, :]
    # Positions moved by -bulk, so the cube holding the same matter moves by -bulk.
    org_i = [int(origin[i] - bulk[i]) for i in range(3)]
    return cic_into_region(pos.reshape(3, -1), org_i, R, ngrid), bulk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr", required=True)
    ap.add_argument("--regions", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--buffers", type=int, nargs="+", default=[0, 4, 8, 16, 24, 32, 48, 64])
    ap.add_argument("--origin", type=int, nargs=3, default=None)
    ap.add_argument("--boxsize", type=float, default=100000.0)
    ap.add_argument("--dis-norm", type=float, default=6000.0)
    ap.add_argument("--out", default="runs/dmsr/stage2b_bulkflow")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    hr = np.load(args.hr, mmap_mode="r")
    ngrid = hr.shape[-1]
    cellsize = args.boxsize / ngrid
    scale = args.dis_norm / cellsize

    results = {}
    for R in args.regions:
        origin = args.origin or [(ngrid - R) // 2] * 3
        print(f"\n=== scored region {R}^3 at {origin} ===")

        # Exact reference: absolute positions, buffer wide enough to be provably
        # complete. Uses the slab-chunked deposit -- materialising the 296^3 block
        # and its float64 positions at once exceeds this session's ~1 GiB cap.
        m_ref, _ = cic_block_into_region(hr, origin, R, 116, ngrid,
                                         args.dis_norm, cellsize)
        ref_rec, ref_delta, _ = region_metrics(m_ref, R ** 3)
        print(f"  reference: sigma={ref_rec['sigma']:.4f} peaks>10={ref_rec['n_peaks_gt10']}")

        print(f"  {'buf':>5} {'crop_hr':>8} | {'ABSOLUTE':>28} | {'BULK-SUBTRACTED':>28}")
        print(f"  {'':>5} {'':>8} | {'relRMS':>9}{'corr':>9}{'sig/ref':>10} | "
              f"{'relRMS':>9}{'corr':>9}{'sig/ref':>10}")
        for b in args.buffers:
            lo = [origin[i] - b for i in range(3)]
            side = R + 2 * b
            d = block_disp(hr, lo, side, ngrid, scale)
            row = {"buffer": b, "crop_hr": side}
            for tag, sub in (("abs", False), ("bulk", True)):
                m, bulk = deposit(d, lo, origin, R, ngrid, subtract_bulk=sub)
                rec, delta, _ = region_metrics(m, R ** 3)
                rel = float(np.sqrt(np.mean((delta - ref_delta) ** 2))
                            / np.sqrt(np.mean(ref_delta ** 2)))
                corr = float(np.corrcoef(delta.ravel(), ref_delta.ravel())[0, 1])
                row[tag] = {"rel_rms": rel, "corr": corr,
                            "sigma_ratio": rec["sigma"] / max(ref_rec["sigma"], 1e-12),
                            "sigma": rec["sigma"], "peaks_gt10": rec["n_peaks_gt10"],
                            "mass_frac": rec["mass_total"] / R ** 3,
                            "bulk_cells": [float(x) for x in bulk]}
            del d
            results[f"R{R}_b{b}"] = row
            print(f"  {b:>5} {side:>8} | {row['abs']['rel_rms']:>9.4f}"
                  f"{row['abs']['corr']:>9.4f}{row['abs']['sigma_ratio']:>10.4f} | "
                  f"{row['bulk']['rel_rms']:>9.4f}{row['bulk']['corr']:>9.4f}"
                  f"{row['bulk']['sigma_ratio']:>10.4f}", flush=True)
        results[f"R{R}_reference"] = ref_rec

    with open(out / "bulkflow_buffer.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}/bulkflow_buffer.json")


if __name__ == "__main__":
    main()
