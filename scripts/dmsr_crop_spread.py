#!/usr/bin/env python
"""What Eulerian region can a Lagrangian crop of side C score correctly?

Stage 2 measured that scoring a *fixed* 64^3 Eulerian region needs particles from
a 64 HR-cell Lagrangian buffer around it. That number is dominated by bulk flow
(median |Psi| = 36 cells) and is the wrong number for designing a training crop,
because we are not obliged to score the region sitting at the crop's own
coordinates.

Particles landing in Eulerian region ``E`` come from Lagrangian ``q`` with
``q + Psi(q) in E``. Writing ``Psi = <Psi>_B + dPsi`` for a crop ``B``, the crop's
particles fill the Eulerian cube centred at ``centre(B) + <Psi>_B`` and shrunk by
the internal spread of ``dPsi``. So for a crop of side ``C``::

    R_valid = C - 2 * spread,     spread = max |Psi - <Psi>_B| over the crop

and the scored region must be *offset* by ``<Psi>_B``. A rigid translation is a
symmetry of the deposit, which is why subtracting the bulk while also moving the
target changes nothing at all -- the only thing that buys back region is the fact
that ``dPsi`` is much smaller than ``Psi``.

This script measures ``spread`` versus crop size, which sets the minimum crop for
Branch A, and then verifies the resulting ``R_valid`` by actually depositing.

Usage
-----
    python scripts/dmsr_crop_spread.py \
        --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set14.npy \
        --crops 64 96 128 192 --out runs/dmsr/stage2c_crop_spread
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dmsr_bulkflow_buffer import block_disp  # noqa: E402
from dmsr_cic_buffer_audit import (  # noqa: E402
    cic_block_into_region,
    cic_into_region,
    region_metrics,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr", required=True)
    ap.add_argument("--crops", type=int, nargs="+", default=[64, 96, 128, 192])
    ap.add_argument("--samples", type=int, default=8, help="random crops per size")
    ap.add_argument("--boxsize", type=float, default=100000.0)
    ap.add_argument("--dis-norm", type=float, default=6000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r-sweep", type=int, nargs="*", default=None,
                    help="scored region sizes to sweep per crop (HR cells)")
    ap.add_argument("--verify", action="store_true",
                    help="also deposit and score the implied valid region")
    ap.add_argument("--out", default="runs/dmsr/stage2c_crop_spread")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    hr = np.load(args.hr, mmap_mode="r")
    ngrid = hr.shape[-1]
    cellsize = args.boxsize / ngrid
    scale = args.dis_norm / cellsize
    rng = np.random.default_rng(args.seed)

    results = {}
    print(f"{'crop_hr':>8} {'crop_lr':>8} {'spread_max':>11} {'spread_p999':>12} "
          f"{'spread_rms':>11} {'R_valid':>8} {'R/C':>6}")
    for C in args.crops:
        spreads_max, spreads_999, spreads_rms, bulks = [], [], [], []
        for _ in range(args.samples):
            lo = [int(x) for x in rng.integers(0, ngrid, size=3)]
            d = block_disp(hr, lo, C, ngrid, scale)          # HR cells
            mean = d.reshape(3, -1).mean(axis=1)
            dd = d - mean[:, None, None, None]
            per_axis_max = float(np.abs(dd).max())
            spreads_max.append(per_axis_max)
            spreads_999.append(float(np.quantile(np.abs(dd), 0.999)))
            spreads_rms.append(float(np.sqrt((dd ** 2).mean())))
            bulks.append(float(np.abs(mean).max()))
            del d, dd
        smax = float(np.mean(spreads_max))
        R_valid = max(int(C - 2 * np.ceil(smax)), 0)
        results[f"C{C}"] = {
            "crop_hr": C, "crop_lr": C // 8,
            "spread_max_mean": smax, "spread_max_worst": float(np.max(spreads_max)),
            "spread_p999_mean": float(np.mean(spreads_999)),
            "spread_rms_mean": float(np.mean(spreads_rms)),
            "bulk_max_mean": float(np.mean(bulks)),
            "R_valid": R_valid,
        }
        print(f"{C:>8} {C//8:>8} {smax:>11.2f} {np.mean(spreads_999):>12.2f} "
              f"{np.mean(spreads_rms):>11.2f} {R_valid:>8} {R_valid/C:>6.2f}")

    if args.r_sweep:
        # The R_valid above uses the worst-case single particle, which is very
        # conservative: the rms internal deviation is ~4x smaller than the max.
        # This sweeps the scored region directly and reports accuracy, so the crop
        # size can be chosen against a tolerance instead of a hard bound.
        print("\nR sweep: crop's own particles -> offset region, vs exact reference")
        print(f"{'crop_hr':>8} {'R':>6} {'R/C':>6} {'relRMS':>9} {'corr':>8} "
              f"{'sig/ref':>9} {'mass':>8}")
        for C in args.crops:
            lo = [int(x) for x in rng.integers(0, ngrid, size=3)]
            d = block_disp(hr, lo, C, ngrid, scale)
            mean = np.round(d.reshape(3, -1).mean(axis=1))
            q = [np.arange(lo[i], lo[i] + C, dtype=np.float64) + 0.5 for i in range(3)]
            pos = np.empty((3, C, C, C), dtype=np.float64)
            pos[0] = d[0] + q[0][:, None, None]
            pos[1] = d[1] + q[1][None, :, None]
            pos[2] = d[2] + q[2][None, None, :]
            pos = pos.reshape(3, -1)
            del d
            rows = []
            for R in [r for r in args.r_sweep if r <= C]:
                org = [int(lo[i] + (C - R) // 2 + mean[i]) for i in range(3)]
                m_crop = cic_into_region(pos, org, R, ngrid)
                rec, delta, _ = region_metrics(m_crop, R ** 3)
                m_ref, _ = cic_block_into_region(hr, org, R, 116, ngrid,
                                                 args.dis_norm, cellsize)
                ref_rec, ref_delta, _ = region_metrics(m_ref, R ** 3)
                rel = float(np.sqrt(np.mean((delta - ref_delta) ** 2))
                            / np.sqrt(np.mean(ref_delta ** 2)))
                corr = float(np.corrcoef(delta.ravel(), ref_delta.ravel())[0, 1])
                rows.append({"R": R, "rel_rms": rel, "corr": corr,
                             "sigma_ratio": rec["sigma"] / max(ref_rec["sigma"], 1e-12),
                             "mass_frac": float(m_crop.sum() / max(m_ref.sum(), 1e-12))})
                print(f"{C:>8} {R:>6} {R/C:>6.2f} {rel:>9.4f} {corr:>8.4f} "
                      f"{rows[-1]['sigma_ratio']:>9.4f} {rows[-1]['mass_frac']:>8.4f}",
                      flush=True)
            results[f"C{C}"]["r_sweep"] = rows
            del pos

    if args.verify:
        print("\nverify: deposit crop's own particles into the offset valid region")
        print(f"{'crop_hr':>8} {'R':>6} {'relRMS':>9} {'corr':>8} {'sig/ref':>9} {'mass':>8}")
        for C in args.crops:
            R = results[f"C{C}"]["R_valid"]
            if R < 8:
                print(f"{C:>8} {R:>6}   (too small to score)")
                continue
            lo = [int(x) for x in rng.integers(0, ngrid, size=3)]
            d = block_disp(hr, lo, C, ngrid, scale)
            mean = np.round(d.reshape(3, -1).mean(axis=1))
            # Eulerian cube the crop can fill: centred at crop centre + bulk
            org = [int(lo[i] + (C - R) // 2 + mean[i]) for i in range(3)]
            q = [np.arange(lo[i], lo[i] + C, dtype=np.float64) + 0.5 for i in range(3)]
            pos = np.empty((3, C, C, C), dtype=np.float64)
            pos[0] = d[0] + q[0][:, None, None]
            pos[1] = d[1] + q[1][None, :, None]
            pos[2] = d[2] + q[2][None, None, :]
            m_crop = cic_into_region(pos.reshape(3, -1), org, R, ngrid)
            del pos, d
            rec, delta, _ = region_metrics(m_crop, R ** 3)
            m_ref, _ = cic_block_into_region(hr, org, R, 116, ngrid,
                                             args.dis_norm, cellsize)
            ref_rec, ref_delta, _ = region_metrics(m_ref, R ** 3)
            rel = float(np.sqrt(np.mean((delta - ref_delta) ** 2))
                        / np.sqrt(np.mean(ref_delta ** 2)))
            corr = float(np.corrcoef(delta.ravel(), ref_delta.ravel())[0, 1])
            results[f"C{C}"]["verify"] = {
                "rel_rms": rel, "corr": corr,
                "sigma_ratio": rec["sigma"] / max(ref_rec["sigma"], 1e-12),
                "mass_frac": rec["mass_total"] / max(m_ref.sum(), 1e-12)}
            print(f"{C:>8} {R:>6} {rel:>9.4f} {corr:>8.4f} "
                  f"{rec['sigma'] / max(ref_rec['sigma'], 1e-12):>9.4f} "
                  f"{rec['mass_total'] / max(m_ref.sum(), 1e-12):>8.4f}", flush=True)

    with open(out / "crop_spread.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}/crop_spread.json")


if __name__ == "__main__":
    main()
