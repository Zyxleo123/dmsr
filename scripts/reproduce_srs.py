#!/usr/bin/env python
"""Formal reproduction of the SR2 / SRS-map2map paper result (Ni et al. 2021).

This does ONE thing: take the authors' *released* generator ``SRmodel/G_z0.pt``
(z=0) and confirm that, applied fresh to our held-out test box, it reproduces the
paper's signature claim -- that the SR field recovers the small-scale statistics
of the true HR field: the DISPLACEMENT, VELOCITY, and (most importantly) the
EULERIAN DENSITY power spectra are restored up to the small scales, with high
phase correlation.

It is deliberately SRS-only (plus a trilinear floor + HR truth for reference), so
the artifact is a clean "we can reproduce their model's behaviour" statement,
uncontaminated by our own DMSR arms. For the head-to-head DMSR-vs-SRS bake-off use
``compare_flow_baseline.py`` instead.

For each of the three fields we report, over the top 1/3 of k-bins (the small
scales that mean-collapse destroys):
  * transfer  T(k) = sqrt(P_SR / P_HR)   -- amplitude recovery (target 1)
  * ratio     P_SR / P_HR                -- power recovery      (target 1)
  * r(k)                                 -- phase/cross-corr    (target 1)
and for density additionally sigma(delta)_SR / sigma(delta)_HR (clumpiness).

The SR field is stochastic; ``--seed`` fixes the generator noise. Full-box CIC
density runs on GPU (device-aware); FFT spectra stay on CPU (numpy) so the metric
definitions match the rest of the eval suite byte-for-byte.

Example (GPU node):
  python scripts/reproduce_srs.py --root /zfsauton/scratch/yixiz/DMSR/paired_catnorm \
      --sets set14 --nsplit 8 --pad 3 --out runs/dmsr/reproduce_srs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Fields we score, and the channel that represents each for the P(k)/T(k)/r(k)
# panels. Displacement/velocity are per-component (x); density is derived from the
# full 3-vector displacement via CIC. Order fixes the figure row order.
FIELDS = ("displacement", "velocity", "density")
FIELD_CHANNEL = {"displacement": 0, "velocity": 3}  # density handled separately


def _highk(arr: np.ndarray, frac: float = 1.0 / 3.0) -> float:
    """Mean of ``arr`` over the top ``frac`` of k-bins (the small-scale band)."""
    lo = int(len(arr) * (1.0 - frac))
    return float(np.mean(arr[lo:]))


def _field_spectra(sr_c, hr_c, tri_c) -> Dict[str, np.ndarray]:
    """P(k), transfer, ratio, r(k) for one scalar cube of SR / tri vs HR."""
    from cosmo_sr.eval.spectra import power_spectrum, cross_correlation_coefficient

    k, pk_sr = power_spectrum(sr_c)
    _, pk_hr = power_spectrum(hr_c)
    _, pk_tri = power_spectrum(tri_c)
    _, rk_sr = cross_correlation_coefficient(sr_c, hr_c)
    _, rk_tri = cross_correlation_coefficient(tri_c, hr_c)
    denom = np.clip(pk_hr, 1e-30, None)
    return {
        "k": k, "pk_hr": pk_hr, "pk_sr": pk_sr, "pk_tri": pk_tri,
        "ratio_sr": pk_sr / denom, "ratio_tri": pk_tri / denom,
        "transfer_sr": np.sqrt(pk_sr / denom), "transfer_tri": np.sqrt(pk_tri / denom),
        "rk_sr": rk_sr, "rk_tri": rk_tri,
    }


def _density_cube(field, cellsize, dis_norm, device):
    """Eulerian overdensity delta (N,N,N) from the 3-vector displacement (0:3)."""
    from cosmo_sr.eval.density import cic_density
    d = cic_density(torch.as_tensor(field[0:3]).float()[None].to(device),
                    cellsize, dis_norm)
    return d[0, 0].cpu().numpy(), float(d.std())


def _score_box(name, lr, hr, srs_G, scale, nsplit, pad, seed,
               cellsize, dis_norm, device):
    """Run SRS + trilinear on one box; return {field: spectra} and scalar metrics."""
    from cosmo_sr.eval.baseline_srs import super_resolve_srs

    sr = np.asarray(super_resolve_srs(srs_G, lr, scale_factor=scale, nsplit=nsplit,
                                      pad=pad, seed=seed, device=device),
                    dtype=np.float32)
    tri = F.interpolate(torch.from_numpy(lr)[None].float(), scale_factor=scale,
                        mode="trilinear", align_corners=False)[0].numpy()

    spectra: Dict[str, Dict[str, np.ndarray]] = {}
    metrics: Dict[str, float] = {}

    for f in ("displacement", "velocity"):
        c = FIELD_CHANNEL[f]
        sp = _field_spectra(sr[c], hr[c], tri[c])
        spectra[f] = sp
        metrics[f"{f}_transfer_highk"] = _highk(sp["transfer_sr"])
        metrics[f"{f}_ratio_highk"] = _highk(sp["ratio_sr"])
        metrics[f"{f}_rk_highk"] = _highk(sp["rk_sr"])

    # density (the headline field): CIC delta from displacement, then same spectra
    d_sr, sig_sr = _density_cube(sr, cellsize, dis_norm, device)
    d_hr, sig_hr = _density_cube(hr, cellsize, dis_norm, device)
    d_tri, _ = _density_cube(tri, cellsize, dis_norm, device)
    sp = _field_spectra(d_sr, d_hr, d_tri)
    spectra["density"] = sp
    metrics["density_transfer_highk"] = _highk(sp["transfer_sr"])
    metrics["density_ratio_highk"] = _highk(sp["ratio_sr"])
    metrics["density_rk_highk"] = _highk(sp["rk_sr"])
    metrics["density_sigma_ratio"] = sig_sr / max(sig_hr, 1e-12)

    print(f"  [{name}] "
          f"disp T_hi={metrics['displacement_transfer_highk']:.3f} "
          f"r_hi={metrics['displacement_rk_highk']:.3f} | "
          f"vel T_hi={metrics['velocity_transfer_highk']:.3f} | "
          f"DENS P/P_HR_hi={metrics['density_ratio_highk']:.3f} "
          f"T_hi={metrics['density_transfer_highk']:.3f} "
          f"r_hi={metrics['density_rk_highk']:.3f} "
          f"sig={metrics['density_sigma_ratio']:.3f}", flush=True)
    return spectra, metrics


def _make_figure(agg_spectra, out: Path):
    """3 rows (disp/vel/density) x 3 cols (P(k), transfer, r(k))."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(figure skipped: {e})")
        return
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for row, f in enumerate(FIELDS):
        sp = agg_spectra[f]
        k = sp["k"]
        a0, a1, a2 = axes[row]
        a0.loglog(k, sp["pk_hr"], "k-", lw=2, label="HR truth")
        a0.loglog(k, sp["pk_sr"], "C0--", lw=1.6, label="SRS (G_z0)")
        a0.loglog(k, sp["pk_tri"], "C3:", lw=1.2, label="trilinear")
        a0.set_ylabel(f"{f}\nP(k)")
        if row == 0:
            a0.set_title("power spectrum P(k)")
            a0.legend(fontsize=8)
        a1.semilogx(k, sp["transfer_sr"], "C0-", lw=1.6, label="SRS")
        a1.semilogx(k, sp["transfer_tri"], "C3:", lw=1.2, label="trilinear")
        a1.axhline(1, color="k", lw=0.8)
        a1.set_ylim(0, 1.3)
        if row == 0:
            a1.set_title("transfer T(k)=sqrt(P_SR/P_HR)  [target 1]")
        a2.semilogx(k, sp["rk_sr"], "C0-", lw=1.6, label="SRS")
        a2.semilogx(k, sp["rk_tri"], "C3:", lw=1.2, label="trilinear")
        a2.axhline(1, color="k", lw=0.8)
        a2.set_ylim(0, 1.02)
        if row == 0:
            a2.set_title("cross-correlation r(k)  [target 1]")
        for a in (a0, a1, a2):
            a.set_xlabel("k [mode]")
    fig.suptitle("SR2 / SRS-map2map G_z0 reproduction: SR vs HR truth", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "reproduce_srs.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/zfsauton/scratch/yixiz/DMSR/paired_catnorm",
                    help="dir with lr/<set>.npy and hr/<set>.npy")
    ap.add_argument("--sets", default="set14",
                    help="comma-separated held-out box name(s) to average over")
    ap.add_argument("--model", default=None,
                    help="G_z0.pt path (default: external/SRS-map2map/SRmodel/G_z0.pt)")
    ap.add_argument("--nsplit", type=int, default=8, help="SRS tiling (must divide Ng_lr)")
    ap.add_argument("--pad", type=int, default=3, help="periodic LR pad per tile (SRS uses 3)")
    ap.add_argument("--boxsize", type=float, default=100000.0, help="full-box size kpc/h")
    ap.add_argument("--dis-norm", type=float, default=6000.0,
                    help="kpc/h per normalized disp unit (6000*D(z); z=0 -> 6000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/dmsr/reproduce_srs")
    args = ap.parse_args()

    # our (map2map-free) modules FIRST, before baseline_srs prepends the SRS fork
    from cosmo_sr.data.field_io import load_field
    from cosmo_sr.eval.baseline_srs import load_srs_generator

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  cuda={torch.cuda.is_available()}"
          + (f"  GPU={torch.cuda.get_device_name(0)}" if torch.cuda.is_available()
             else "  (CPU-ONLY -- full-box CIC will be slow)"), flush=True)

    model_path = args.model or str(Path(__file__).resolve().parents[1] /
                                   "external" / "SRS-map2map" / "SRmodel" / "G_z0.pt")
    root = Path(args.root)
    sets = [s for s in args.sets.split(",") if s]

    # peek at one box to get the scale + cellsize
    lr0 = load_field(str(root / "lr" / f"{sets[0]}.npy")).astype(np.float32)
    hr0 = np.asarray(load_field(str(root / "hr" / f"{sets[0]}.npy"), mmap=True),
                     dtype=np.float32)
    Ng, Nhr = lr0.shape[1], hr0.shape[1]
    scale = Nhr // Ng
    cellsize = args.boxsize / Nhr
    print(f"Ng_lr={Ng}  Ng_hr={Nhr}  scale={scale}  cellsize={cellsize:.2f} kpc/h  "
          f"sets={sets}", flush=True)

    srs_G = load_srs_generator(model_path, scale_factor=scale, device=device)

    all_spectra: List[Dict[str, Dict[str, np.ndarray]]] = []
    all_metrics: List[Dict[str, float]] = []
    for s in sets:
        lr = load_field(str(root / "lr" / f"{s}.npy")).astype(np.float32)
        hr = np.asarray(load_field(str(root / "hr" / f"{s}.npy"), mmap=True),
                        dtype=np.float32)
        hr = np.ascontiguousarray(hr)
        sp, mt = _score_box(s, lr, hr, srs_G, scale, args.nsplit, args.pad,
                            args.seed, cellsize, args.dis_norm, device)
        all_spectra.append(sp)
        all_metrics.append(mt)

    # aggregate: spectra averaged across boxes (same k-grid), metrics mean+/-std
    agg_spectra: Dict[str, Dict[str, np.ndarray]] = {}
    for f in FIELDS:
        keys = all_spectra[0][f].keys()
        agg_spectra[f] = {kk: np.mean([sp[f][kk] for sp in all_spectra], axis=0)
                          for kk in keys}
    agg_metrics = {}
    for kk in all_metrics[0]:
        vals = np.array([m[kk] for m in all_metrics], dtype=np.float64)
        agg_metrics[kk] = {"mean": float(vals.mean()), "std": float(vals.std())}

    _make_figure(agg_spectra, out)

    npz = {}
    for f in FIELDS:
        for kk, v in agg_spectra[f].items():
            npz[f"{f}__{kk}"] = v
    np.savez(out / "reproduce_srs_spectra.npz", **npz)
    with open(out / "reproduce_srs_metrics.json", "w") as fh:
        json.dump({"sets": sets, "model": model_path, "nsplit": args.nsplit,
                   "pad": args.pad, "seed": args.seed,
                   "metrics_mean_std": agg_metrics,
                   "per_box": all_metrics}, fh, indent=2)

    # --- verdict ---
    def mv(key):
        return agg_metrics[key]["mean"], agg_metrics[key]["std"]
    print("\n=== SR2 / SRS-map2map G_z0 reproduction (mean over "
          f"{len(sets)} box{'es' if len(sets) != 1 else ''}) ===")
    print(f"{'field':>13} {'P/P_HR(hi-k)':>13} {'T(k)(hi-k)':>11} {'r(k)(hi-k)':>11}")
    for f in FIELDS:
        pr, _ = mv(f"{f}_ratio_highk")
        tr, _ = mv(f"{f}_transfer_highk")
        rk, _ = mv(f"{f}_rk_highk")
        print(f"{f:>13} {pr:>13.3f} {tr:>11.3f} {rk:>11.3f}")
    ds, dss = mv("density_sigma_ratio")
    dpr, _ = mv("density_ratio_highk")
    print(f"\ndensity sigma-ratio (clumpiness, target 1) : {ds:.3f} +/- {dss:.3f}")
    ok = abs(dpr - 1.0) <= 0.15 and ds >= 0.9
    print("PAPER CLAIM (small-scale density power recovered to ~1): "
          + ("REPRODUCED" if ok else "NOT reproduced")
          + f"  [density P/P_HR(hi-k)={dpr:.3f}, sigma-ratio={ds:.3f}]")
    print(f"\nWrote {out}/reproduce_srs.png, reproduce_srs_spectra.npz, "
          "reproduce_srs_metrics.json")


if __name__ == "__main__":
    main()
