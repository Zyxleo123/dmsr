#!/usr/bin/env python
"""Compare our SR model against the paper baseline (pretrained SRS-map2map GAN).

Scores three methods on the SAME held-out paired set with the SAME metrics:
  * trilinear : naive trilinear upsample of LR (reference floor)
  * srs       : pretrained SRS-map2map SR-GAN (Ni et al.), SRmodel/G_z0.pt
  * ours      : our trained generator (from --our-config / --our-checkpoint)

Metrics per method: HR MSE, HR relative MSE, LR-reconstruction MSE (mse(A(SR),LR)),
per-channel cross-correlation with HR, and isotropic power spectra (overlaid).

Outputs (in --out): comparison.json, spectra.npz, spectra.png, slices.png.

Example (GPU node):
  python scripts/compare_baseline.py \
    --our-config configs/mixed_real.yaml \
    --our-checkpoint runs/mixed_real/ckpt_last.pt \
    --lr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr/set15.npy \
    --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set15.npy \
    --nsplit 8 --out runs/compare_set15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def _metrics_for(name, sr, lr, hr, degrader):
    from cosmo_sr.eval.metrics import compute_metrics
    m = compute_metrics(degrader, sr, lr, x_hr=hr)
    print(f"[{name}] HR MSE={m.get('hr_mse'):.5g}  relMSE={m.get('hr_relative_mse'):.5g}  "
          f"LR-recon MSE={m['lr_recon_mse']:.5g}  <r>={np.mean(m.get('hr_cross_corr_mean', [np.nan])):.4f}")
    return m


def _spectra(sr, channels=(0,)):
    from cosmo_sr.eval.spectra import power_spectrum
    out = {}
    for c in channels:
        k, pk = power_spectrum(sr[c])
        out[c] = (k, pk)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--our-config", required=True)
    ap.add_argument("--our-checkpoint", required=True)
    ap.add_argument("--lr", required=True, help="LR catnorm .npy (Ng)")
    ap.add_argument("--hr", required=True, help="HR catnorm .npy (Ng*scale)")
    ap.add_argument("--srs-model", default=None, help="SRS G checkpoint (default external/SRS-map2map/SRmodel/G_z0.pt)")
    ap.add_argument("--nsplit", type=int, default=8)
    ap.add_argument("--our-pad-lr", type=int, default=2)
    ap.add_argument("--srs-pad", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channel", type=int, default=0, help="channel for spectra/slice plots")
    ap.add_argument("--out", default="runs/compare")
    args = ap.parse_args()

    # Import our (map2map-free) modules FIRST, before the SRS fork is imported.
    from cosmo_sr.data.field_io import load_field
    from cosmo_sr.operators.degrader import FixedDegrader
    from cosmo_sr.models.wrappers import build_generator, model_config_kwargs
    from cosmo_sr.inference.tile_sr import super_resolve_full_box
    from cosmo_sr.utils.config import load_config

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.our_config)
    scale = int(cfg.get("data", {}).get("scale_factor", 8))

    lr = load_field(args.lr).astype(np.float32)
    hr = load_field(args.hr, mmap=True)
    hr = np.asarray(hr, dtype=np.float32)
    Ng = lr.shape[1]
    assert hr.shape[1] == Ng * scale, f"HR/LR mismatch: {hr.shape} vs LR {lr.shape} scale {scale}"

    degrader = FixedDegrader(scale).to(device)

    results: Dict[str, dict] = {}
    spectra_store: Dict[str, np.ndarray] = {}

    # k, Pk of HR + LR references
    from cosmo_sr.eval.spectra import power_spectrum
    k_hr, pk_hr = power_spectrum(hr[args.channel])
    spectra_store["k_hr"] = k_hr; spectra_store["Pk_hr"] = pk_hr

    # --- trilinear baseline ---
    tri = F.interpolate(torch.from_numpy(lr).unsqueeze(0), scale_factor=scale,
                        mode="trilinear", align_corners=False).squeeze(0).numpy()
    results["trilinear"] = _metrics_for("trilinear", tri, lr, hr, degrader)
    k, pk = power_spectrum(tri[args.channel])
    spectra_store["k_trilinear"], spectra_store["Pk_trilinear"] = k, pk

    # --- ours ---
    model_cfg = dict(cfg.get("model", {"name": "SimpleSRGenerator"}))
    kwargs = model_config_kwargs(model_cfg); kwargs.setdefault("scale_factor", scale)
    our_G = build_generator(model_cfg.get("name", "SimpleSRGenerator"), **kwargs).to(device)
    state = torch.load(args.our_checkpoint, map_location=device)
    our_G.load_state_dict(state["model"]); our_G.eval()
    ours = super_resolve_full_box(our_G, lr, scale, nsplit=args.nsplit, pad_lr=args.our_pad_lr)
    ours = np.asarray(ours, dtype=np.float32)
    results["ours"] = _metrics_for("ours", ours, lr, hr, degrader)
    k, pk = power_spectrum(ours[args.channel])
    spectra_store["k_ours"], spectra_store["Pk_ours"] = k, pk

    # --- SRS paper baseline (imports the SRS map2map fork) ---
    from cosmo_sr.eval.baseline_srs import load_srs_generator, super_resolve_srs
    srs_path = args.srs_model or str(Path(__file__).resolve().parents[1] / "external" /
                                     "SRS-map2map" / "SRmodel" / "G_z0.pt")
    srs_G = load_srs_generator(srs_path, scale_factor=scale, device=device)
    srs = super_resolve_srs(srs_G, lr, scale_factor=scale, nsplit=args.nsplit,
                            pad=args.srs_pad, seed=args.seed, device=device)
    results["srs"] = _metrics_for("srs", srs, lr, hr, degrader)
    k, pk = power_spectrum(srs[args.channel])
    spectra_store["k_srs"], spectra_store["Pk_srs"] = k, pk

    # --- write outputs ---
    with open(out / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(out / "spectra.npz", **spectra_store)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # spectra overlay
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(k_hr, pk_hr, "k-", label="HR (truth)")
    for name, style in [("trilinear", "C2--"), ("srs", "C1-"), ("ours", "C0-")]:
        ax.loglog(spectra_store[f"k_{name}"], spectra_store[f"Pk_{name}"], style, label=name)
    ax.set_xlabel("k [modes/box]"); ax.set_ylabel("P(k)")
    ax.set_title(f"Power spectrum (channel {args.channel})"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "spectra.png", dpi=120); plt.close(fig)

    # slice comparison
    c, N = args.channel, hr.shape[1]
    panels = [("LR (up)", tri[c, N // 2]), ("SRS", srs[c, N // 2]),
              ("Ours", ours[c, N // 2]), ("HR truth", hr[c, N // 2])]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    vmin, vmax = hr[c].min(), hr[c].max()
    for axp, (title, img) in zip(axes, panels):
        im = axp.imshow(img, origin="lower", vmin=vmin, vmax=vmax)
        axp.set_title(title); axp.set_xticks([]); axp.set_yticks([])
        fig.colorbar(im, ax=axp, fraction=0.046)
    fig.tight_layout(); fig.savefig(out / "slices.png", dpi=120); plt.close(fig)

    print(f"\nWrote {out}/comparison.json, spectra.npz, spectra.png, slices.png")


if __name__ == "__main__":
    main()
