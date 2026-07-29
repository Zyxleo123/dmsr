#!/usr/bin/env python
"""Fair comparison of the residual null-space FLOW SR model against baselines.

Unlike compare_baseline.py (which only scores the deterministic SimpleSRGenerator),
this scores the *flow-cascade* model (ODE sampler) alongside:
  * trilinear : naive trilinear upsample of LR                (floor)
  * <flow>    : one or more trained flow checkpoints          (ours / pairedonly)
  * srs       : pretrained SRS-map2map SR-GAN, SRmodel/G_z0.pt (full-paired SOTA ceiling)

All methods super-resolve the SAME center crop of the REAL LR (not an HR-derived
LR) and are scored against the matching HR crop. Because the flow and SRS are
stochastic, the headline metrics are DISTRIBUTIONAL:
  * P(k)/P_HR(k)           -- power recovery (mean-collapse shows as <1 at high k)
  * r(k)                   -- cross-correlation with HR
  * density sigma-ratio    -- Eulerian clumpiness vs HR (target 1.0)
HR MSE is reported but must NOT be the headline: it rewards mean-collapse and
penalises adversarial realism, so it flatters exactly the failure we are fixing.

Example (GPU node), once the two runs have checkpoints:
  python scripts/compare_flow_baseline.py \
    --lr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr/set15.npy \
    --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set15.npy \
    --flow ours:configs/sr_flow_strongband.yaml:runs/sr_flow_strongband/ckpt_last.pt \
    --flow pairedonly:configs/sr_flow_pairedonly.yaml:runs/sr_flow_pairedonly/ckpt_last.pt \
    --lr-crop 32 --n-steps 20 --diversity 3 --out runs/compare_flow_set15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _highk_ratio(pk: np.ndarray, pk_hr: np.ndarray, frac: float = 1 / 3) -> float:
    """Mean P/P_HR over the top ``frac`` of k-bins (the small-scale deficit)."""
    n = len(pk)
    lo = int(n * (1 - frac))
    denom = np.clip(pk_hr[lo:], 1e-30, None)
    return float(np.mean(pk[lo:] / denom))


def _score(name, sr, lr_c, hr_c, degrader, ch, cellsize, dis_norm, device=None):
    """Metrics + spectra for one SR field ``sr`` (C,N,N,N) vs HR crop.

    ``cic_density`` is device-aware, so the 512^3 CIC scatter runs on ``device``
    (GPU) when given — CPU-side it dominated a full-box run. FFT-based spectra
    stay on CPU (numpy) to keep the metric definitions identical to the SRS-only
    run; per-stage timings are printed so the real bottleneck is visible.
    """
    import time
    from cosmo_sr.eval.metrics import compute_metrics
    from cosmo_sr.eval.spectra import power_spectrum, cross_correlation_coefficient
    from cosmo_sr.eval.density import cic_density

    def _highk_mean(arr):  # mean over top 1/3 of k-bins (matches _highk_ratio band)
        lo = int(len(arr) * (1 - 1 / 3))
        return float(np.mean(arr[lo:]))
    dev = device or torch.device("cpu")

    t = time.time()
    m = compute_metrics(degrader, sr, lr_c, x_hr=hr_c)
    t_cm = time.time() - t; t = time.time()
    k, pk = power_spectrum(sr[ch])
    _, pk_hr = power_spectrum(hr_c[ch])
    _, rk = cross_correlation_coefficient(sr[ch], hr_c[ch])
    t_fft = time.time() - t; t = time.time()

    rec = {
        "hr_mse": float(m.get("hr_mse", float("nan"))),
        "hr_relative_mse": float(m.get("hr_relative_mse", float("nan"))),
        "lr_recon_mse": float(m["lr_recon_mse"]),
        "cross_corr_mean": float(np.mean(m.get("hr_cross_corr_mean", [np.nan]))),
        "highk_power_ratio": _highk_ratio(pk, pk_hr),
    }
    t_dens = float("nan")
    rkd_full = None
    try:  # density (displacement channels 0:3); bonus, never fatal
        d = cic_density(torch.as_tensor(sr[0:3]).float()[None].to(dev), cellsize, dis_norm)
        dt = cic_density(torch.as_tensor(hr_c[0:3]).float()[None].to(dev), cellsize, dis_norm)
        dnp, dtnp = d[0, 0].cpu().numpy(), dt[0, 0].cpu().numpy()
        kd, pkd = power_spectrum(dnp)
        _, pkt = power_spectrum(dtnp)
        _, rkd = cross_correlation_coefficient(dnp, dtnp)  # density phase vs true
        rec["density_sigma_ratio"] = float(d.std() / dt.std().clamp_min(1e-12))
        rec["density_highk_pk_ratio"] = _highk_ratio(pkd, pkt)
        rec["density_rk_highk"] = _highk_mean(rkd)          # <-- density r(k), high-k
        rec["density_rk_mean"] = float(np.mean(rkd))
        rkd_full = rkd
        t_dens = time.time() - t
    except Exception as e:  # pragma: no cover
        rec["density_error"] = repr(e)
    print(f"  [{name:>12}] timing s: compute_metrics={t_cm:.1f} fft={t_fft:.1f} density={t_dens:.1f}")

    print(f"  [{name:>12}] relMSE={rec['hr_relative_mse']:.4g}  "
          f"LRrecon={rec['lr_recon_mse']:.4g}  <r>={rec['cross_corr_mean']:.4f}  "
          f"P/P_HR(hi-k)={rec['highk_power_ratio']:.3f}  "
          f"dens_sig={rec.get('density_sigma_ratio', float('nan')):.3f}  "
          f"dens_rk(hi-k)={rec.get('density_rk_highk', float('nan')):.3f}")
    return rec, (k, pk, rk, rkd_full)


def _flow_sr(label, cfgp, ckpt, lr_crop_t, device, n_steps, diversity, seed):
    """Sample the flow cascade on the LR crop; return (mean_over_draws, one_sample)."""
    from cosmo_sr.utils.config import load_config
    from cosmo_sr.operators.multiscale import MultiScaleOperators
    from cosmo_sr.train.train_flow import _build_base_upscaler, _build_flow_model
    from cosmo_sr.inference.flow_sample import super_resolve_cascade

    cfg = load_config(cfgp)
    channels = int(cfg.get("model", {}).get("channels", 6))
    factor = int(cfg.get("factor", 2))
    resolutions = tuple(cfg.get("resolutions", [64, 128, 256]))
    top = 2 * max(resolutions)

    ops = MultiScaleOperators(factor=factor).to(device)
    base_upscaler = _build_base_upscaler(cfg, channels, factor, device)
    model = _build_flow_model(cfg.get("model", {}), channels, factor).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    if "extra" in state and "base_upscaler" in state["extra"]:
        base_upscaler.load_state_dict(state["extra"]["base_upscaler"])
    model.eval()

    draws = []
    for i in range(max(1, diversity)):
        out = super_resolve_cascade(model, ops, base_upscaler, lr_crop_t,
                                    resolutions=resolutions, n_steps=n_steps, seed=seed + i)
        draws.append(out[top][0].detach().cpu().numpy().astype(np.float32))
    draws = np.stack(draws)  # (D, C, N, N, N)
    return draws.mean(0), draws[0]


def _load_dmsr(cfgp, ckpt, device):
    """Load a *current DMSR* model (NullSpaceFlow / MeanInnovationFlow).

    Unlike ``_flow_sr`` (which targets the retired flow-cascade architecture),
    this loads the post-pivot checkpoints via ``dmsr_eval.load_flow`` so SRS,
    trilinear and our models can all be scored on the SAME metrics. These models
    are displacement-only; returns ``(model, use_channels)``.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts"))
    from cosmo_sr.utils.config import load_config
    from dmsr_eval import load_flow  # noqa: E402  (script-dir import)

    cfg = load_config(cfgp)
    dcfg = cfg.get("data", {})
    uc = dcfg.get("use_channels") or [0, 1, 2]
    model = load_flow(cfg, len(uc), ckpt, device, use_ema=True)
    return model, uc


def _dmsr_generate_tiled(model, y_full, scale, tile, halo, n_steps, seed):
    """Full-box generation by periodic-halo tiling + center stitch.

    A 512^3 field blows the ~2^31 per-CUDA-kernel element limit, so the box is
    covered by ``tile``-sized LR tiles, each padded with ``halo`` LR voxels of
    *periodic* context (the full box IS periodic) and generated at
    ``scale*(tile+2*halo)``; only the central ``scale*tile`` is kept.
    ``y_full``: (1,C,Ng,Ng,Ng) on device.

    NOT equivalent to a full-box forward
    ------------------------------------
    This function used to claim that ``halo*scale >= 64`` makes each core
    "identical to the full-box forward" because the model has only local ops. That
    is false. ``UNetResidualFlowModel`` is built with ``norm="group"``, and
    ``nn.GroupNorm`` reduces over ``(C/g, D, H, W)`` -- every output voxel depends
    on the mean and variance of the WHOLE window. The dependency is therefore
    window-global no matter how large the halo is, and two different tile sizes
    renormalise the same physical region differently.

    Measured on ``t13_unconstrained_s0`` (LR encoder pathway, set14): the
    conditioning features of a fixed 8^3 LR region computed in an 8^3 window
    correlate only r = 0.47 with the same region computed in the full 64^3 box,
    and 93% of that discrepancy is a per-channel affine rescale, i.e. GroupNorm
    statistics. Since training used ``crop_lr=8`` (8^3 LR windows) and this
    function is normally called with ``tile=16, halo=8`` (32^3 LR windows), the
    numbers it produces carry a train/eval normalisation shift on top of whatever
    the model actually gets wrong. Record ``tile``/``halo`` alongside any result
    and do not compare across different values of them.
    """
    C, Ng = y_full.shape[1], y_full.shape[2]
    assert Ng % tile == 0, f"tile {tile} must divide Ng {Ng}"
    Nhr = Ng * scale
    yp = F.pad(y_full, (halo,) * 6, mode="circular")  # (1,C,Ng+2h,...)
    out = np.zeros((C, Nhr, Nhr, Nhr), dtype=np.float32)
    nt = Ng // tile
    h, t = halo * scale, tile * scale
    for ix in range(nt):
        for iy in range(nt):
            for iz in range(nt):
                a, b, c = ix * tile, iy * tile, iz * tile
                tin = yp[:, :, a:a + tile + 2 * halo,
                              b:b + tile + 2 * halo,
                              c:c + tile + 2 * halo]
                torch.manual_seed(seed + (ix * nt + iy) * nt + iz)
                with torch.no_grad():
                    tout = model.generate(tin, n_steps=n_steps)
                core = tout[0, :, h:h + t, h:h + t, h:h + t].cpu().numpy()
                out[:, ix * t:ix * t + t, iy * t:iy * t + t, iz * t:iz * t + t] = core
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", required=True, help="REAL LR catnorm .npy (Ng=64)")
    ap.add_argument("--hr", required=True, help="HR catnorm .npy (Ng=512)")
    ap.add_argument("--flow", action="append", default=[],
                    help="[retired arch] LABEL:CONFIG:CHECKPOINT (repeatable)")
    ap.add_argument("--dmsr", action="append", default=[],
                    help="current DMSR model LABEL:CONFIG:CHECKPOINT (repeatable)")
    ap.add_argument("--dmsr-tile", type=int, default=0,
                    help="LR tile size for full-box tiled generation (0=single-shot). "
                         "Must divide the LR crop; a 512^3 one-shot exceeds the CUDA "
                         "per-kernel element limit, so full-box needs tiling.")
    ap.add_argument("--dmsr-halo", type=int, default=8,
                    help="periodic LR halo per tile (halo*scale must exceed the model "
                         "receptive field, ~64 HR voxels; default 8 -> 64 HR)")
    ap.add_argument("--lr-crop", type=int, default=32,
                    help="LR-grid center-crop size; HR crop = lr_crop*scale")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--diversity", type=int, default=3, help="flow draws to average for MSE")
    ap.add_argument("--no-srs", action="store_true")
    ap.add_argument("--srs-model", default=None)
    ap.add_argument("--srs-nsplit", type=int, default=4)
    ap.add_argument("--srs-pad", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channel", type=int, default=0, help="channel for P(k)/r(k) plots")
    ap.add_argument("--boxsize", type=float, default=100000.0, help="full-box kpc/h")
    ap.add_argument("--dis-norm", type=float, default=6000.0)
    ap.add_argument("--out", default="runs/compare_flow")
    ap.add_argument("--save-fields", default=None,
                    help="directory to cache every scored SR field as <label>.npy. "
                         "A full-box DMSR draw costs ~5 GPU-hours here; Stage 1 "
                         "(residual-alpha) and Stage 6 (best-of-K) both need the "
                         "same fields, so caching them turns a re-run into minutes.")
    args = ap.parse_args()

    # our (map2map-free) modules FIRST, before the SRS fork touches sys.path
    from cosmo_sr.data.field_io import load_field
    from cosmo_sr.operators.degrader import FixedDegrader
    from cosmo_sr.eval.spectra import power_spectrum

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fields_dir = Path(args.save_fields) if args.save_fields else None
    if fields_dir is not None:
        fields_dir.mkdir(parents=True, exist_ok=True)

    def cache(label, arr):
        if fields_dir is None:
            return
        p = fields_dir / f"{label}.npy"
        np.save(p, np.asarray(arr, dtype=np.float32))
        print(f"  [{label:>12}] cached field -> {p}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  cuda_available={torch.cuda.is_available()}  "
          f"{'GPU=' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-ONLY (slow!)'}",
          flush=True)
    ch = args.channel

    lr = load_field(args.lr).astype(np.float32)          # (6,64,64,64)
    hr = np.asarray(load_field(args.hr, mmap=True), dtype=np.float32)  # (6,512,...)
    Ng, Nhr = lr.shape[1], hr.shape[1]
    scale = Nhr // Ng
    lc = args.lr_crop
    hc = lc * scale
    s_lr = (Ng - lc) // 2
    s_hr = s_lr * scale
    lr_c = np.ascontiguousarray(lr[:, s_lr:s_lr + lc, s_lr:s_lr + lc, s_lr:s_lr + lc])
    hr_c = np.ascontiguousarray(hr[:, s_hr:s_hr + hc, s_hr:s_hr + hc, s_hr:s_hr + hc])
    print(f"scale={scale}  LR crop {lr_c.shape}  ->  HR crop {hr_c.shape}")

    degrader = FixedDegrader(scale).to(device)
    cellsize = args.boxsize / Nhr  # physical grid spacing (same for any sub-box)

    results: Dict[str, dict] = {}
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    k_hr, pk_hr = power_spectrum(hr_c[ch])

    # --- trilinear floor ---
    tri = F.interpolate(torch.from_numpy(lr_c)[None], scale_factor=scale,
                        mode="trilinear", align_corners=False)[0].numpy()
    results["trilinear"], spectra["trilinear"] = _score(
        "trilinear", tri, lr_c, hr_c, degrader, ch, cellsize, args.dis_norm, device)
    cache("trilinear", tri)
    cache("hr_truth", hr_c)

    # --- flow checkpoints ---
    lr_crop_t = torch.from_numpy(lr_c)[None].to(device)
    for spec in args.flow:
        label, cfgp, ckpt = spec.split(":")
        if not Path(ckpt).exists():
            print(f"  [{label}] SKIP -- checkpoint not found: {ckpt}")
            continue
        sr_mean, sr_one = _flow_sr(label, cfgp, ckpt, lr_crop_t, device,
                                   args.n_steps, args.diversity, args.seed)
        # distributional metrics on a single physical sample; MSE on the mean
        rec, sp = _score(label, sr_one, lr_c, hr_c, degrader, ch, cellsize, args.dis_norm)
        rec["hr_mse_mean_of_draws"] = float(np.mean((sr_mean - hr_c) ** 2))
        results[label], spectra[label] = rec, sp

    # --- current DMSR models (displacement-only; tiled for full-box) ---
    for spec in args.dmsr:
        label, cfgp, ckpt = spec.split(":")
        if not Path(ckpt).exists():
            print(f"  [{label}] SKIP -- checkpoint not found: {ckpt}")
            continue
        model, uc = _load_dmsr(cfgp, ckpt, device)
        lr_cm = np.ascontiguousarray(lr_c[uc])
        hr_cm = np.ascontiguousarray(hr_c[uc])
        lr_cm_t = torch.from_numpy(lr_cm)[None].to(device)
        import time as _time
        draws = []
        for i in range(max(1, args.diversity)):
            t0 = _time.time()
            if args.dmsr_tile > 0:
                x = _dmsr_generate_tiled(model, lr_cm_t, scale, args.dmsr_tile,
                                         args.dmsr_halo, args.n_steps, args.seed + i)
            else:
                torch.manual_seed(args.seed + i)
                with torch.no_grad():
                    x = model.generate(lr_cm_t, n_steps=args.n_steps)[0].cpu().numpy()
            print(f"  [{label:>12}] generate draw {i} in {_time.time()-t0:.1f}s")
            draws.append(x.astype(np.float32))
        draws = np.stack(draws)
        sr_mean, sr_one = draws.mean(0), draws[0]
        del model, draws
        if device.type == "cuda":
            torch.cuda.empty_cache()
        rec, sp = _score(label, sr_one, lr_cm, hr_cm, degrader,
                         min(ch, len(uc) - 1), cellsize, args.dis_norm, device)
        rec["hr_mse_mean_of_draws"] = float(np.mean((sr_mean - hr_cm) ** 2))
        rec["dmsr_tile"] = int(args.dmsr_tile)
        rec["dmsr_halo"] = int(args.dmsr_halo)
        results[label], spectra[label] = rec, sp
        cache(label, sr_one)

    # --- SRS SOTA ceiling ---
    if not args.no_srs:
        try:
            from cosmo_sr.eval.baseline_srs import load_srs_generator, super_resolve_srs
            srs_path = args.srs_model or str(Path(__file__).resolve().parents[1] /
                                             "external" / "SRS-map2map" / "SRmodel" / "G_z0.pt")
            srs_G = load_srs_generator(srs_path, scale_factor=scale, device=device)
            srs = super_resolve_srs(srs_G, lr_c, scale_factor=scale, nsplit=args.srs_nsplit,
                                    pad=args.srs_pad, seed=args.seed, device=device)
            srs = np.asarray(srs, dtype=np.float32)
            results["srs"], spectra["srs"] = _score(
                "srs", srs, lr_c, hr_c, degrader, ch, cellsize, args.dis_norm, device)
            cache("srs", srs)
        except Exception as e:
            print(f"  [srs] SKIP -- {e!r}")

    # --- write ---
    with open(out / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    npz = {"k_hr": k_hr, "Pk_hr": pk_hr}
    for name, (k, pk, rk, rkd) in spectra.items():
        npz[f"k_{name}"] = k; npz[f"Pk_{name}"] = pk; npz[f"rk_{name}"] = rk
        if rkd is not None:
            npz[f"rkd_{name}"] = rkd
    np.savez(out / "spectra.npz", **npz)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = (["trilinear"]
                 + [s.split(":")[0] for s in args.flow]
                 + [s.split(":")[0] for s in args.dmsr]
                 + (["srs"] if not args.no_srs else []))
        order = [n for n in order if n in spectra]
        # P(k) overlay + P/P_HR ratio + r(k)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        axes[0].loglog(k_hr, pk_hr, "k-", lw=2, label="HR truth")
        for n in order:
            k, pk, rk, rkd = spectra[n]
            axes[0].loglog(k, pk, "--", label=n)
            axes[1].semilogx(k, pk / np.clip(pk_hr, 1e-30, None), "-", label=n)
            axes[2].semilogx(k, rk, "-", label=n)
        axes[0].set_title(f"P(k) ch{ch}"); axes[0].set_xlabel("k"); axes[0].legend(fontsize=8)
        axes[1].axhline(1, color="k", lw=0.8); axes[1].set_ylim(0, 1.4)
        axes[1].set_title("P(k)/P_HR (target 1)"); axes[1].set_xlabel("k")
        axes[2].axhline(1, color="k", lw=0.8); axes[2].set_ylim(0, 1.02)
        axes[2].set_title("cross-corr r(k)"); axes[2].set_xlabel("k")
        fig.tight_layout(); fig.savefig(out / "spectra.png", dpi=120); plt.close(fig)
    except Exception as e:
        print(f"(plot skipped: {e})")

    print("\n=== summary (headline = distributional) ===")
    print(f"{'method':>12} {'relMSE':>9} {'LRrecon':>9} {'<r>':>7} {'P/PHR_hik':>10} {'dens_sig':>9}")
    for n, r in results.items():
        print(f"{n:>12} {r['hr_relative_mse']:>9.4g} {r['lr_recon_mse']:>9.4g} "
              f"{r['cross_corr_mean']:>7.4f} {r['highk_power_ratio']:>10.3f} "
              f"{r.get('density_sigma_ratio', float('nan')):>9.3f}")
    print(f"\nWrote {out}/comparison.json, spectra.npz, spectra.png")


if __name__ == "__main__":
    main()
