#!/usr/bin/env python
"""Stage 2: does crop-level CIC density converge as the particle buffer grows?

Motivation
----------
``cosmo_sr.eval.density.cic_density`` deposits a crop's particles with ``% ng``,
i.e. it wraps *inside the crop*. Its docstring calls that "an approximation
(particles that leave the crop reappear on the far side)". On this data it is not
a mild approximation: the median |Psi| on set14 is ~36 HR cells and the max is
~127, against a 64^3 HR training crop. Most particles leave the crop entirely,
and wrapping them back scrambles the Eulerian field.

The correct construction deposits every particle at its ABSOLUTE periodic box
position ``q + Psi`` and scores only a fixed Eulerian sub-cube, including
particles whose Lagrangian coordinate lies OUTSIDE the scored cube but which
travel into it. This script measures how wide the Lagrangian buffer must be
before the scored central density stops changing.

Definitions
-----------
scored region
    A fixed **Eulerian** cube of side ``region`` HR cells at ``origin`` (in HR
    cell units). Density is only ever reported inside it.
buffer ``b``
    Particles are taken from the **Lagrangian** block
    ``[origin - b, origin + region + b)`` (periodic). ``b`` grows until the
    scored density converges.
reference
    ``b_ref = ceil(max |Psi|) + 2`` is provably exact: no particle outside that
    block can reach the scored region. Verified against a true full-box deposit
    with ``--check-fullbox``.

Usage
-----
    python scripts/dmsr_cic_buffer_audit.py \
        --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set14.npy \
        --region 64 --out runs/dmsr/stage2_cic_buffer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# CIC deposition into a fixed Eulerian sub-cube
# --------------------------------------------------------------------------- #
def cic_into_region(pos, origin, region, ngrid):
    """CIC-deposit particles into a fixed Eulerian cube, periodic in the full box.

    Parameters
    ----------
    pos:
        ``(3, M)`` absolute particle positions in HR *cell units*, any real value
        (wrapped internally into ``[0, ngrid)``).
    origin:
        Length-3 lower corner of the scored cube, in HR cell units.
    region:
        Side of the scored cube, in HR cells.
    ngrid:
        Side of the full periodic box, in HR cells.

    Returns
    -------
    ``(region, region, region)`` float64 mass grid. Only weight landing inside
    the cube is accumulated; everything else is discarded. This is a *partial*
    deposit by construction -- correctness comes from feeding it enough
    particles, which is exactly what this script measures.
    """
    R = int(region)
    ng = int(ngrid)
    # Each CIC corner is resolved as an ABSOLUTE periodic cell first, and only then
    # tested against the scored cube. Testing before wrapping (the obvious version)
    # silently drops the ``floor(u)+1`` corner of any particle sitting in the
    # one-cell shell just below a face, which costs a thin sheet of mass on the
    # three lower faces -- a ~3/R surface bias that the regression tests catch as
    # non-conservation under a rigid translation.
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0

    grid = np.zeros(R * R * R, dtype=np.float64)
    rel = [None, None, None]
    for ox in (0, 1):
        wx = frac[0] if ox else 1.0 - frac[0]
        rel[0] = np.mod(i0[0] + ox - int(origin[0]), ng)
        mx = rel[0] < R
        if not mx.any():
            continue
        for oy in (0, 1):
            wy = frac[1] if oy else 1.0 - frac[1]
            rel[1] = np.mod(i0[1] + oy - int(origin[1]), ng)
            my = mx & (rel[1] < R)
            if not my.any():
                continue
            for oz in (0, 1):
                wz = frac[2] if oz else 1.0 - frac[2]
                rel[2] = np.mod(i0[2] + oz - int(origin[2]), ng)
                m = my & (rel[2] < R)
                if not m.any():
                    continue
                w = wx[m] * wy[m] * wz[m]
                idx = (rel[0][m] * R + rel[1][m]) * R + rel[2][m]
                grid += np.bincount(idx, weights=w, minlength=R * R * R)
    return grid.reshape(R, R, R)


def iter_block_positions(hr, origin, region, buf, ngrid, dis_norm, cellsize, slab=32):
    """Yield ``(3, M)`` absolute positions for slabs of a padded Lagrangian block.

    ``hr`` is the ``(6, N, N, N)`` normalized field (memmap ok); channels 0:3 are
    displacement. The Lagrangian coordinate of cell ``(i,j,k)`` is ``i + 0.5``
    etc., so a particle's position is ``q + Psi`` with ``Psi`` converted from the
    on-disk normalization into HR cell units.

    Slabbing along axis 0 keeps peak memory at ``slab/side`` of the full block --
    the widest buffers here cover 294^3 = 25M particles, and materialising those
    as float64 positions plus the index/weight temporaries in one shot is what
    made the unchunked version die partway through the sweep.
    """
    lo = [int(origin[d] - buf) for d in range(3)]
    side = int(region + 2 * buf)
    scale = dis_norm / cellsize
    idx1 = np.mod(np.arange(lo[1], lo[1] + side), ngrid)
    idx2 = np.mod(np.arange(lo[2], lo[2] + side), ngrid)
    q1 = np.arange(lo[1], lo[1] + side, dtype=np.float64) + 0.5
    q2 = np.arange(lo[2], lo[2] + side, dtype=np.float64) + 0.5

    for s in range(0, side, slab):
        e = min(s + slab, side)
        idx0 = np.mod(np.arange(lo[0] + s, lo[0] + e), ngrid)
        disp = np.asarray(hr[0:3][:, idx0][:, :, idx1][:, :, :, idx2], dtype=np.float32)
        q0 = np.arange(lo[0] + s, lo[0] + e, dtype=np.float64) + 0.5
        pos = np.empty((3, e - s, side, side), dtype=np.float64)
        pos[0] = disp[0] * scale + q0[:, None, None]
        pos[1] = disp[1] * scale + q1[None, :, None]
        pos[2] = disp[2] * scale + q2[None, None, :]
        del disp
        yield pos.reshape(3, -1)


def cic_block_into_region(hr, origin, region, buf, ngrid, dis_norm, cellsize, slab=32):
    """Slab-chunked CIC of a padded Lagrangian block into the scored cube."""
    R = int(region)
    mass = np.zeros((R, R, R), dtype=np.float64)
    n = 0
    for pos in iter_block_positions(hr, origin, region, buf, ngrid,
                                    dis_norm, cellsize, slab=slab):
        mass += cic_into_region(pos, origin, R, ngrid)
        n += pos.shape[1]
        del pos
    return mass, n


# --------------------------------------------------------------------------- #
# metrics on the scored cube
# --------------------------------------------------------------------------- #
def _power_spectrum(delta):
    n = delta.shape[-1]
    fk = np.fft.rfftn(delta)
    p = (np.abs(fk) ** 2) / n ** 3
    kx = np.fft.fftfreq(n) * n
    kz = np.fft.rfftfreq(n) * n
    kmag = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)
    kmag = np.where(kmag > 0, kmag, -1.0)
    nb = n // 2
    edges = np.linspace(0, nb, nb + 1)
    which = np.digitize(kmag.ravel(), edges) - 1
    ok = (which >= 0) & (which < nb)
    sums = np.bincount(which[ok], weights=p.ravel()[ok], minlength=nb)
    cnts = np.bincount(which[ok], minlength=nb)
    return 0.5 * (edges[:-1] + edges[1:]), sums / np.maximum(cnts, 1)


def region_metrics(mass, npart_expected):
    """Overdensity metrics for one scored cube.

    ``mass`` is the raw CIC mass grid. It is normalized by the *expected* mean
    (total particles / total cells for a uniform universe), NOT by its own mean:
    self-normalizing would hide exactly the mass-deficit that an undersized
    buffer produces.
    """
    R = mass.shape[-1]
    mean_expected = npart_expected / R ** 3
    delta = mass / mean_expected - 1.0
    k, pk = _power_spectrum(delta)
    hi = int(len(pk) * 2 / 3)
    return {
        "mass_total": float(mass.sum()),
        "mass_per_cell": float(mass.mean()),
        "sigma": float(delta.std()),
        "delta_max": float(delta.max()),
        "n_peaks_gt10": int((delta > 10).sum()),
        "n_peaks_gt50": int((delta > 50).sum()),
        "n_peaks_gt100": int((delta > 100).sum()),
        "pk_highk": float(np.mean(pk[hi:])),
        "pk_lowk": float(np.mean(pk[: len(pk) // 3])),
    }, delta, (k, pk)


def multi_crop_check(hr, region, nsample, ngrid, dis_norm, cellsize, buf, seed, out):
    """Is the crop-wrapped density even *rank*-faithful across many crops?

    One crop can be unlucky. This repeats the wrapped-vs-converged comparison over
    ``nsample`` random crops and reports the correlation of the voxel fields and
    the rank correlation of ``sigma``. If the wrapped construction cannot even
    order crops by clumpiness, then a density critic fed wrapped crops is being
    trained against a target that does not track the physical one.
    """
    rng = np.random.default_rng(seed)
    R = int(region)
    dscale = dis_norm / cellsize
    i_ = np.arange(R)
    q = np.arange(R, dtype=np.float64) + 0.5
    rows = []
    for i in range(int(nsample)):
        origin = [int(x) for x in rng.integers(0, ngrid, size=3)]
        mass_ref, _ = cic_block_into_region(hr, origin, R, buf, ngrid, dis_norm, cellsize)
        ref, ref_delta, _ = region_metrics(mass_ref, R ** 3)

        # the crop the current evaluator would see, wrapped inside itself
        w1 = np.mod(origin[1] + i_, ngrid)
        w2 = np.mod(origin[2] + i_, ngrid)
        disp_c = np.empty((3, R, R, R), dtype=np.float32)
        for s in range(R):
            p = int((origin[0] + s) % ngrid)
            disp_c[:, s] = np.asarray(hr[0:3, p], dtype=np.float32)[:, w1][:, :, w2]
        disp_c *= dscale
        pos = np.empty((3, R, R, R))
        pos[0] = disp_c[0] + q[:, None, None]
        pos[1] = disp_c[1] + q[None, :, None]
        pos[2] = disp_c[2] + q[None, None, :]
        mass_w = cic_into_region(np.mod(pos.reshape(3, -1), R), [0, 0, 0], R, R)
        wr, w_delta, _ = region_metrics(mass_w, R ** 3)
        del disp_c, pos

        corr = float(np.corrcoef(w_delta.ravel(), ref_delta.ravel())[0, 1])
        rows.append({"origin": origin, "corr": corr,
                     "sigma_wrapped": wr["sigma"], "sigma_true": ref["sigma"],
                     "peaks_wrapped": wr["n_peaks_gt10"], "peaks_true": ref["n_peaks_gt10"]})
        print(f"  crop {i:>2} at {origin}: corr={corr:+.4f}  "
              f"sigma wrapped={wr['sigma']:.3f} true={ref['sigma']:.3f}", flush=True)

    sw = np.array([r["sigma_wrapped"] for r in rows])
    st = np.array([r["sigma_true"] for r in rows])
    # rank correlation without scipy -- importing scipy.stats costs ~2 min on this
    # NFS mount, which is longer than the whole measurement
    def _rank(x):
        return np.argsort(np.argsort(x)).astype(float) + 1.0
    rho = float(np.corrcoef(_rank(sw), _rank(st))[0, 1])
    summary = {
        "n_crops": len(rows),
        "corr_field_mean": float(np.mean([r["corr"] for r in rows])),
        "corr_field_max": float(np.max([r["corr"] for r in rows])),
        "sigma_ratio_mean": float(np.mean(sw / st)),
        "spearman_sigma_wrapped_vs_true": rho,
        "pearson_sigma_wrapped_vs_true": float(np.corrcoef(sw, st)[0, 1]),
    }
    print("\n=== multi-crop summary ===")
    for k, v in summary.items():
        print(f"  {k:>34}: {v}")
    with open(Path(out) / "multicrop_check.json", "w") as f:
        json.dump({"summary": summary, "crops": rows}, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr", required=True)
    ap.add_argument("--multi-crop", type=int, default=0,
                    help="instead of the buffer sweep, compare wrapped vs converged "
                         "density over this many random crops")
    ap.add_argument("--region", type=int, default=64, help="scored cube side, HR cells")
    ap.add_argument("--origin", type=int, nargs=3, default=None,
                    help="scored cube lower corner (default: box center)")
    ap.add_argument("--buffers", type=int, nargs="+",
                    default=[0, 8, 16, 32, 48, 64, 96, 128, 160],
                    help="Lagrangian buffer widths in HR cells")
    ap.add_argument("--boxsize", type=float, default=100000.0)
    ap.add_argument("--dis-norm", type=float, default=6000.0)
    ap.add_argument("--check-fullbox", action="store_true",
                    help="also deposit ALL box particles as an independent reference")
    ap.add_argument("--ref-buffer", type=int, default=0,
                    help="use this buffer as the converged reference instead of "
                         "ceil(max|Psi|)+2 (verify convergence first)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/dmsr/stage2_cic_buffer")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    hr = np.load(args.hr, mmap_mode="r")
    ngrid = hr.shape[-1]
    cellsize = args.boxsize / ngrid
    scale = args.dis_norm / cellsize
    R = int(args.region)
    origin = args.origin or [(ngrid - R) // 2] * 3
    print(f"box {ngrid}^3  cellsize {cellsize:.2f} kpc/h  scored region {R}^3 at {origin}")

    # --- max displacement -> provably-sufficient buffer ---------------------- #
    mx = 0.0
    for i in range(0, ngrid, 64):
        blk = np.asarray(hr[0:3, i:i + 64], dtype=np.float32) * scale
        mx = max(mx, float(np.abs(blk).max()))
        del blk
    b_exact = int(np.ceil(mx)) + 2
    print(f"max per-axis |Psi| = {mx:.2f} HR cells  ->  provably exact buffer b_ref = {b_exact}")

    # ``b_exact`` is provably sufficient but also the most expensive deposit. Once a
    # cheaper buffer has been shown to reproduce it exactly, --ref-buffer lets the
    # sweep use that instead so a re-run fits in a short job.
    if args.ref_buffer:
        b_exact = int(args.ref_buffer)
        print(f"using --ref-buffer {b_exact} as the converged reference")

    if args.multi_crop:
        multi_crop_check(hr, R, args.multi_crop, ngrid, args.dis_norm, cellsize,
                         b_exact, args.seed, out)
        return

    buffers = sorted(set([b for b in args.buffers if b <= ngrid // 2] + [b_exact]))
    npart_region = R ** 3  # uniform-universe expectation for the scored cube

    results = {}
    deltas = {}
    spectra = {}

    # --- current evaluator (wrap-inside-crop), for reference ----------------- #
    disp_c = np.asarray(
        hr[0:3, origin[0]:origin[0] + R, origin[1]:origin[1] + R, origin[2]:origin[2] + R],
        dtype=np.float32) * scale
    q = np.arange(R, dtype=np.float64) + 0.5
    pos_c = np.empty((3, R, R, R))
    pos_c[0] = disp_c[0] + q[:, None, None]
    pos_c[1] = disp_c[1] + q[None, :, None]
    pos_c[2] = disp_c[2] + q[None, None, :]
    pos_c = np.mod(pos_c.reshape(3, -1), R)          # <-- the % ng wrap in cic_density
    m_wrap = cic_into_region(pos_c, [0, 0, 0], R, R)
    results["wrap_in_crop"], deltas["wrap_in_crop"], spectra["wrap_in_crop"] = \
        region_metrics(m_wrap, npart_region)
    del disp_c, pos_c
    print(f"  [wrap_in_crop  ] sigma={results['wrap_in_crop']['sigma']:.3f} "
          f"mass={results['wrap_in_crop']['mass_total']:.0f}")

    # --- buffered absolute deposits ------------------------------------------ #
    for b in buffers:
        mass, n_in = cic_block_into_region(hr, origin, R, b, ngrid, args.dis_norm, cellsize)
        key = f"buf{b}"
        results[key], deltas[key], spectra[key] = region_metrics(mass, npart_region)
        results[key]["n_particles_deposited"] = int(n_in)
        results[key]["buffer"] = int(b)
        print(f"  [buf {b:>4}      ] sigma={results[key]['sigma']:.4f} "
              f"mass={results[key]['mass_total']:.0f} "
              f"({results[key]['mass_total'] / npart_region:.4f} of expected)  "
              f"peaks>10={results[key]['n_peaks_gt10']}", flush=True)

    ref_key = f"buf{b_exact}"
    ref = deltas[ref_key]

    # --- mass flux bookkeeping: what fraction of the Lagrangian block leaves? - #
    n_stay = n_tot = 0
    for pos in iter_block_positions(hr, origin, R, 0, ngrid, args.dis_norm, cellsize):
        u = np.stack([np.mod(pos[d] - origin[d], ngrid) for d in range(3)])
        n_stay += int(np.all((u >= 0) & (u < R), axis=0).sum())
        n_tot += pos.shape[1]
        del pos, u
    frac_stay = n_stay / max(n_tot, 1)
    mass_ref = results[ref_key]["mass_total"]
    print(f"\nmass bookkeeping: {frac_stay:.4f} of the region's own particles stay inside; "
          f"converged region holds {mass_ref / npart_region:.4f} of a uniform share")

    # --- convergence vs reference -------------------------------------------- #
    for key in list(results):
        d = deltas[key]
        num = float(np.sqrt(np.mean((d - ref) ** 2)))
        den = float(np.sqrt(np.mean(ref ** 2)))
        results[key]["rel_rms_vs_ref"] = num / max(den, 1e-12)
        results[key]["sigma_ratio_vs_ref"] = results[key]["sigma"] / max(
            results[ref_key]["sigma"], 1e-12)
        results[key]["pk_highk_ratio_vs_ref"] = results[key]["pk_highk"] / max(
            results[ref_key]["pk_highk"], 1e-30)
        cc = np.corrcoef(d.ravel(), ref.ravel())[0, 1]
        results[key]["corr_vs_ref"] = float(cc)

    print(f"\n{'mode':>14} {'relRMS':>9} {'sig/ref':>9} {'Phi/ref':>9} {'corr':>8} {'mass/exp':>9}")
    for key in ["wrap_in_crop"] + [f"buf{b}" for b in buffers]:
        r = results[key]
        print(f"{key:>14} {r['rel_rms_vs_ref']:>9.4f} {r['sigma_ratio_vs_ref']:>9.4f} "
              f"{r['pk_highk_ratio_vs_ref']:>9.4f} {r['corr_vs_ref']:>8.4f} "
              f"{r['mass_total'] / npart_region:>9.4f}")

    meta = {
        "hr": args.hr, "ngrid": int(ngrid), "region": R, "origin": list(map(int, origin)),
        "max_disp_hr_cells": mx, "b_exact": b_exact, "ref_key": ref_key,
        "frac_own_particles_staying": frac_stay,
    }
    with open(out / "buffer_audit.json", "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    np.savez_compressed(out / "spectra.npz",
                        **{f"k_{k}": v[0] for k, v in spectra.items()},
                        **{f"Pk_{k}": v[1] for k, v in spectra.items()})

    # --- plot ----------------------------------------------------------------- #
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bs = buffers
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].semilogy(bs, [results[f"buf{b}"]["rel_rms_vs_ref"] for b in bs], "o-")
        ax[0].axhline(results["wrap_in_crop"]["rel_rms_vs_ref"], color="r", ls="--",
                      label="current evaluator (wrap in crop)")
        ax[0].set_xlabel("Lagrangian buffer [HR cells]")
        ax[0].set_ylabel("rel RMS vs converged"); ax[0].legend(fontsize=8)
        ax[0].set_title("central density convergence")
        ax[1].plot(bs, [results[f"buf{b}"]["sigma_ratio_vs_ref"] for b in bs], "o-")
        ax[1].axhline(1, color="k", lw=0.8)
        ax[1].axhline(results["wrap_in_crop"]["sigma_ratio_vs_ref"], color="r", ls="--")
        ax[1].set_xlabel("buffer [HR cells]"); ax[1].set_title("sigma / sigma_converged")
        ax[2].plot(bs, [results[f"buf{b}"]["mass_total"] / npart_region for b in bs], "o-")
        ax[2].axhline(1, color="k", lw=0.8)
        ax[2].set_xlabel("buffer [HR cells]"); ax[2].set_title("mass captured / uniform share")
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "buffer_convergence.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"(plot skipped: {e})")

    print(f"\nWrote {out}/buffer_audit.json, spectra.npz, buffer_convergence.png")


if __name__ == "__main__":
    main()
