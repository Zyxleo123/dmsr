#!/usr/bin/env python
"""Stage 1: does the learned displacement residual destroy phase-coherent collapse?

Given a generated full-box displacement ``Psi_ours`` and the trilinear floor
``Psi_tri``, define the learned correction and dial it in linearly::

    dPsi     = Psi_ours - Psi_tri
    Psi_alpha = Psi_tri + alpha * dPsi

and score density, deformation and folding against ``alpha``. If density degrades
monotonically while displacement power *rises*, the correction is adding
small-scale displacement power in a way that does not build haloes -- direct
evidence that the residual is not phase-coherent with collapse.

Helmholtz split
---------------
On a periodic box the correction separates uniquely in Fourier space into

    dPsi_long(k)  = khat (khat . dPsi(k))      compressive / curl-free
    dPsi_trans(k) = dPsi(k) - dPsi_long(k)     divergence-free / rotational

Only the longitudinal part changes ``div Psi`` to first order, so it is the part
that can create or destroy collapse; the transverse part reshuffles particles
along equidensity flows. Running ``alpha`` separately on each answers "is the
damage compressive or rotational?" directly rather than by inference.

Inputs are the cached ``.npy`` fields written by
``compare_flow_baseline.py --save-fields`` -- regenerating a full-box draw costs
about 5 GPU-hours, and this script needs ~19 scorings of it.

Usage
-----
    python scripts/dmsr_residual_alpha.py \
        --fields runs/dmsr/fields_set14 --label unconstrained \
        --alphas 0 0.1 0.25 0.5 0.75 1.0 1.25 \
        --out runs/dmsr/stage1_residual_alpha
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------- #
def helmholtz_split(d, device):
    """Longitudinal / transverse split of a ``(3, N, N, N)`` periodic vector field.

    Returns ``(d_long, d_trans)`` as float32 CPU arrays. The ``k = 0`` mode is a
    uniform translation: it has no divergence and no curl, so it is assigned
    wholly to the transverse part (any split is arbitrary there, and putting it in
    the longitudinal part would make ``alpha`` shift the whole box bodily).

    Nyquist handling
    ----------------
    ``fftfreq`` reports the Nyquist bin of an even-length axis as ``-n/2``, while
    its mirror under ``k -> -k`` is the same bin. For a mode that is at Nyquist in
    one axis but not another, the reflected wavevector is therefore not ``+-k``,
    so ``P(k) = khat khat^T`` stops being even, the projected spectrum stops being
    Hermitian, and ``irfftn`` silently drops the imaginary part -- measured at 7%
    of the longitudinal energy on white noise, which also broke the
    ``|L|^2 + |T|^2 = |d|^2`` identity (0.954 instead of 1). The Nyquist bin has no
    well-defined direction on a real grid, so its wavevector component is set to
    zero; the identity then holds to float32 round-off.
    """
    n = d.shape[-1]
    dt = torch.as_tensor(d, dtype=torch.float32, device=device)
    fk = torch.fft.rfftn(dt, dim=(-3, -2, -1))
    del dt

    kx = torch.fft.fftfreq(n, device=device) * n
    kz = torch.fft.rfftfreq(n, device=device) * n
    if n % 2 == 0:
        kx = kx.clone(); kx[n // 2] = 0.0
        kz = kz.clone(); kz[-1] = 0.0
    k2 = (kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)
    k2 = k2.clamp_min(1e-20)

    kdotf = (kx[:, None, None] * fk[0] + kx[None, :, None] * fk[1]
             + kz[None, None, :] * fk[2]) / k2
    fl = torch.empty_like(fk)
    fl[0] = kx[:, None, None] * kdotf
    fl[1] = kx[None, :, None] * kdotf
    fl[2] = kz[None, None, :] * kdotf
    del kdotf
    fl[:, 0, 0, 0] = 0.0                      # k=0 -> transverse by convention

    long_ = torch.fft.irfftn(fl, s=(n, n, n), dim=(-3, -2, -1)).cpu().numpy()
    trans = torch.fft.irfftn(fk - fl, s=(n, n, n), dim=(-3, -2, -1)).cpu().numpy()
    del fk, fl
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return long_.astype(np.float32), trans.astype(np.float32)


def band_power(field, device, nb=3):
    """Mean power of a ``(3,N,N,N)`` field in ``nb`` radial k-bands (low -> high)."""
    n = field.shape[-1]
    ft = torch.as_tensor(field, dtype=torch.float32, device=device)
    p = torch.zeros(nb, device=device)
    cnt = torch.zeros(nb, device=device)
    kx = torch.fft.fftfreq(n, device=device) * n
    kz = torch.fft.rfftfreq(n, device=device) * n
    km = (kx[:, None, None] ** 2 + kx[None, :, None] ** 2
          + kz[None, None, :] ** 2).sqrt()
    idx = (km / (n / 2 / nb)).long().clamp(0, nb - 1)
    for c in range(3):
        fk = torch.fft.rfftn(ft[c], dim=(-3, -2, -1))
        pw = (fk.real ** 2 + fk.imag ** 2) / n ** 3
        p.index_add_(0, idx.reshape(-1), pw.reshape(-1))
        cnt.index_add_(0, idx.reshape(-1), torch.ones_like(pw.reshape(-1)))
        del fk, pw
    return (p / cnt.clamp_min(1)).cpu().numpy()


def cic_full_box(disp, cellsize, dis_norm, device, grid_mult=1):
    """Exact full-box CIC. Periodic wrap here is the true box wrap, so this is
    the one place a plain ``% ng`` deposit is correct (see Stage 2)."""
    from cosmo_sr.eval.density import cic_density
    d = torch.as_tensor(disp, dtype=torch.float32, device=device)[None]
    out = cic_density(d, cellsize, dis_norm, grid_mult=grid_mult)
    del d
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out[0, 0]


def density_stats(delta, ref=None, nb=256):
    n = delta.shape[-1]
    fk = torch.fft.rfftn(delta, dim=(-3, -2, -1))
    pw = (fk.real ** 2 + fk.imag ** 2) / n ** 3
    kx = torch.fft.fftfreq(n, device=delta.device) * n
    kz = torch.fft.rfftfreq(n, device=delta.device) * n
    km = (kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2).sqrt()
    idx = (km / (n / 2 / nb)).long().clamp(0, nb - 1)
    sums = torch.zeros(nb, device=delta.device).index_add_(0, idx.reshape(-1), pw.reshape(-1))
    cnts = torch.zeros(nb, device=delta.device).index_add_(
        0, idx.reshape(-1), torch.ones_like(pw.reshape(-1)))
    pk = (sums / cnts.clamp_min(1))
    hi = int(nb * 2 / 3)
    rec = {
        "sigma": float(delta.std()),
        "pk_highk": float(pk[hi:].mean()),
        "delta_max": float(delta.max()),
        "n_peaks_gt10": int((delta > 10).sum()),
        "n_peaks_gt50": int((delta > 50).sum()),
        "n_peaks_gt100": int((delta > 100).sum()),
        "frac_void_lt_m0p8": float((delta < -0.8).float().mean()),
    }
    if ref is not None:
        fr = torch.fft.rfftn(ref, dim=(-3, -2, -1))
        pr = (fr.real ** 2 + fr.imag ** 2) / n ** 3
        cx = (fk.real * fr.real + fk.imag * fr.imag) / n ** 3
        sr_ = torch.zeros(nb, device=delta.device).index_add_(0, idx.reshape(-1), pr.reshape(-1))
        sc = torch.zeros(nb, device=delta.device).index_add_(0, idx.reshape(-1), cx.reshape(-1))
        pkr = sr_ / cnts.clamp_min(1)
        pkc = sc / cnts.clamp_min(1)
        rk = pkc / (pk * pkr).clamp_min(1e-30).sqrt()
        rec["pk_ratio_highk"] = float((pk[hi:] / pkr[hi:].clamp_min(1e-30)).mean())
        rec["rk_highk"] = float(rk[hi:].mean())
        rec["rk_mean"] = float(rk.mean())
        rec["sigma_ratio"] = float(delta.std() / ref.std().clamp_min(1e-12))
        del fr, pr, cx
    return rec, pk.cpu().numpy()


def deformation_stats(disp_cells, device, slab=32):
    """det J, divergence and folding fraction, computed slab-wise.

    ``J = I + dPsi/dq`` on the Lagrangian lattice (unit spacing in HR cells).
    Materialising all nine gradient components of a 512^3 field at once is 4.8 GiB;
    slabs with a one-cell halo keep it bounded while staying exactly periodic.
    """
    n = disp_cells.shape[-1]
    d = torch.as_tensor(disp_cells, dtype=torch.float32, device=device)
    n_neg = 0
    n_small = 0
    det_sum = 0.0
    det_sq = 0.0
    div_sq = 0.0
    det_min = float("inf")
    hist = torch.zeros(200, device=device)
    edges = torch.linspace(-2.0, 6.0, 201, device=device)
    for s in range(0, n, slab):
        e = min(s + slab, n)
        sl = torch.arange(s - 1, e + 1, device=device) % n
        blk = d[:, sl]                                   # (3, e-s+2, n, n)
        g = torch.empty((3, 3) + (e - s, n, n), device=device)
        for i in range(3):
            g[i, 0] = 0.5 * (blk[i, 2:] - blk[i, :-2])
            g[i, 1] = 0.5 * (torch.roll(blk[i, 1:-1], -1, 1) - torch.roll(blk[i, 1:-1], 1, 1))
            g[i, 2] = 0.5 * (torch.roll(blk[i, 1:-1], -1, 2) - torch.roll(blk[i, 1:-1], 1, 2))
        del blk
        J = g.clone()
        for i in range(3):
            J[i, i] += 1.0
        det = (J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
               - J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0])
               + J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]))
        div = g[0, 0] + g[1, 1] + g[2, 2]
        del g, J
        n_neg += int((det < 0).sum())
        n_small += int((det < 0.1).sum())
        det_sum += float(det.sum())
        det_sq += float((det ** 2).sum())
        div_sq += float((div ** 2).sum())
        det_min = min(det_min, float(det.min()))
        hist += torch.histogram(det.flatten().cpu(), bins=edges.cpu()).hist.to(device)
        del det, div
    tot = n ** 3
    mean = det_sum / tot
    return {
        "detJ_mean": mean,
        "detJ_std": float(np.sqrt(max(det_sq / tot - mean ** 2, 0.0))),
        "detJ_min": det_min,
        "frac_detJ_negative": n_neg / tot,
        "frac_detJ_lt_0p1": n_small / tot,
        "div_rms": float(np.sqrt(div_sq / tot)),
    }, hist.cpu().numpy()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True,
                    help="dir of cached .npy fields from compare_flow_baseline --save-fields")
    ap.add_argument("--label", default="unconstrained", help="which model field to dial in")
    ap.add_argument("--base", default="trilinear")
    ap.add_argument("--truth", default="hr_truth")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25])
    ap.add_argument("--components", nargs="+", default=["full", "long", "trans"],
                    choices=["full", "long", "trans"])
    ap.add_argument("--boxsize", type=float, default=100000.0)
    ap.add_argument("--dis-norm", type=float, default=6000.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/dmsr/stage1_residual_alpha")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fd = Path(args.fields)
    print(f"device={device}  fields={fd}")

    ours = np.load(fd / f"{args.label}.npy")[0:3].astype(np.float32)
    tri = np.load(fd / f"{args.base}.npy")[0:3].astype(np.float32)
    truth = np.load(fd / f"{args.truth}.npy")[0:3].astype(np.float32)
    n = ours.shape[-1]
    cellsize = args.boxsize / n
    dscale = args.dis_norm / cellsize
    print(f"fields {ours.shape}, cellsize {cellsize:.2f} kpc/h")

    d_full = ours - tri
    print("Helmholtz split ...", flush=True)
    d_long, d_trans = helmholtz_split(d_full, device)
    resid = float(np.sqrt(((d_full - d_long - d_trans) ** 2).mean())
                  / np.sqrt((d_full ** 2).mean()))
    print(f"  split residual (should be ~0): {resid:.3e}")
    comps = {"full": d_full, "long": d_long, "trans": d_trans}
    for kk, vv in comps.items():
        print(f"  |{kk:>5}| rms = {float(np.sqrt((vv**2).mean())):.6f}")

    delta_truth = cic_full_box(truth, cellsize, args.dis_norm, device)
    truth_rec, pk_truth = density_stats(delta_truth)
    truth_def, _ = deformation_stats(truth * dscale, device)
    print(f"\nTRUTH: sigma={truth_rec['sigma']:.4f} peaks>10={truth_rec['n_peaks_gt10']} "
          f"detJ<0={truth_def['frac_detJ_negative']:.5f} div_rms={truth_def['div_rms']:.4f}")

    results = {"truth": {"density": truth_rec, "deformation": truth_def,
                         "band_power": band_power(truth, device).tolist()},
               "helmholtz_residual": resid,
               "component_rms": {k: float(np.sqrt((v ** 2).mean())) for k, v in comps.items()}}

    print(f"\n{'comp':>6} {'alpha':>6} {'sigma':>8} {'sig/tru':>8} {'pkhi/tru':>9} "
          f"{'rk_hi':>7} {'peaks>10':>9} {'detJ<0':>9} {'div_rms':>8} {'P_hi(disp)':>11}")
    for comp in args.components:
        dd = comps[comp]
        for a in args.alphas:
            psi = tri + a * dd
            delta = cic_full_box(psi, cellsize, args.dis_norm, device)
            drec, pk = density_stats(delta, ref=delta_truth)
            defrec, hist = deformation_stats(psi * dscale, device)
            bp = band_power(psi, device)
            key = f"{comp}_a{a:g}"
            results[key] = {"component": comp, "alpha": a, "density": drec,
                            "deformation": defrec, "band_power": bp.tolist()}
            np.save(out / f"pk_density_{key}.npy", pk)
            print(f"{comp:>6} {a:>6.2f} {drec['sigma']:>8.4f} "
                  f"{drec['sigma_ratio']:>8.4f} {drec['pk_ratio_highk']:>9.4f} "
                  f"{drec['rk_highk']:>7.4f} {drec['n_peaks_gt10']:>9} "
                  f"{defrec['frac_detJ_negative']:>9.5f} {defrec['div_rms']:>8.4f} "
                  f"{bp[-1]:>11.4g}", flush=True)
            del psi, delta
            if device.type == "cuda":
                torch.cuda.empty_cache()

    with open(out / "residual_alpha.json", "w") as f:
        json.dump(results, f, indent=2)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
        for comp in args.components:
            al = [a for a in args.alphas]
            g = lambda f: [f(results[f"{comp}_a{a:g}"]) for a in al]  # noqa: E731
            ax[0].plot(al, g(lambda r: r["density"]["sigma_ratio"]), "o-", label=comp)
            ax[1].plot(al, g(lambda r: r["density"]["pk_ratio_highk"]), "o-", label=comp)
            ax[2].plot(al, g(lambda r: r["deformation"]["frac_detJ_negative"]), "o-", label=comp)
            ax[3].plot(al, g(lambda r: r["band_power"][-1]), "o-", label=comp)
        ax[0].axhline(1, color="k", lw=0.8); ax[0].set_title("density sigma / truth")
        ax[1].axhline(1, color="k", lw=0.8); ax[1].set_title("density P(k) high-k / truth")
        ax[2].axhline(truth_def["frac_detJ_negative"], color="k", ls="--")
        ax[2].set_title("folding fraction det J < 0")
        ax[3].axhline(results["truth"]["band_power"][-1], color="k", ls="--")
        ax[3].set_title("displacement high-k power")
        for a_ in ax:
            a_.set_xlabel("alpha"); a_.grid(alpha=0.3); a_.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "residual_alpha.png", dpi=120); plt.close(fig)
    except Exception as e:
        print(f"(plot skipped: {e})")

    print(f"\nWrote {out}/residual_alpha.json, residual_alpha.png")


if __name__ == "__main__":
    main()
