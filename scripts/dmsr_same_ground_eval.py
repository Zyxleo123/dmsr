#!/usr/bin/env python
"""Same-ground comparison of DMSR flows trained at DIFFERENT crop sizes.

Why this script exists
----------------------
The per-crop evaluator (`scripts/dmsr_eval.py`) computes every spectral statistic
on a crop whose grid size equals the model's training crop. Two models trained at
`crop_lr=8` (64^3 HR) and `crop_lr=16` (128^3 HR) therefore report `rk`/`Tk` on
DIFFERENT physical k-grids with DIFFERENT shell counts and noise floors -- an
8^3-LR crop spans 12.5 Mpc/h, a 16^3-LR crop spans 25 Mpc/h, so mode k=1 is a 2x
different physical scale. Comparing those numbers directly is confounded.

This script removes the confound by evaluating every model on ONE identical
held-out region:

  * A fixed region of the held-out box (``region_lr`` LR voxels -> ``region_lr *
    factor`` HR voxels) is chosen once.
  * Each model tiles that region into NON-OVERLAPPING blocks of its OWN native
    crop size, generates each tile, and stitches them into the region's HR grid.
    Tiles are aligned to the LR grid, so each tile stays exactly consistent
    (``A(x_hat_tile) = y_tile``) and the stitched field is consistent everywhere.
  * Hard (non-overlapping) tiling is deliberate: overlap-blending would HIDE the
    very boundary/seam effect that motivates larger crops. Each model runs in its
    trained receptive-field regime; the smaller-crop model simply has 8x more seam
    area per unit volume, which is the effect under test.
  * All statistics are then computed on the SAME region grid (same ``n`` => same
    ``k_lr`` => same bins => same physical scales for every model).

A ``seam_ratio`` diagnostic quantifies the boundary artifact directly: the mean
squared field jump across tile-boundary planes divided by the mean squared jump
across all interior planes (~1 = no seam signature, >1 = visible seams).

Usage
-----
    python scripts/dmsr_same_ground_eval.py \
        --model crop8:configs/dmsr/stage_a_paired_flow.yaml:runs/dmsr/stage_a/ckpt_best.pt \
        --model crop16:configs/dmsr/stage_a_crop16.yaml:runs/dmsr/stage_a_crop16/ckpt_best.pt \
        --split val --region-lr 32 --n-steps 20 --seed 0 \
        --out runs/dmsr/same_ground

The region size (``--region-lr``) trades coverage against memory: 32 -> 256^3 HR
(~0.4 GB/field, the default) is large enough to expose seams and large-scale
coherence while running on a modest GPU; 64 -> full 512^3 box is available for a
final figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # to import dmsr_eval

from cosmo_sr.data.crops import periodic_crop  # noqa: E402
from cosmo_sr.data.field_io import load_field  # noqa: E402
from cosmo_sr.dmsr.data import resolve_split  # noqa: E402
from cosmo_sr.dmsr.density import HighPassDensity, cellsizes  # noqa: E402
from cosmo_sr.dmsr.evaluate import (  # noqa: E402
    equilateral_bispectrum,
    pdf_error,
    power_error,
    rk_tk_summary,
    squeezed_cross_bispectrum,
)
from cosmo_sr.utils.config import apply_overrides, load_config  # noqa: E402
from dmsr_eval import load_flow  # noqa: E402


# --------------------------------------------------------------------------- #
# Region loading + tiled stitching
# --------------------------------------------------------------------------- #
def _load_region(
    lr_path: str, hr_path: str, factor: int, region_lr: int, origin_lr: Tuple[int, int, int],
    use_channels,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a matched (LR, HR) region as (1, C, n, n, n) tensors.

    The region is a `region_lr`-voxel LR block at `origin_lr` (periodic) and the
    aligned `region_lr*factor` HR block, sliced to `use_channels` exactly as the
    training dataset does.
    """
    lr = load_field(lr_path, mmap=True)
    hr = load_field(hr_path, mmap=True)
    if use_channels is not None:
        lr = np.ascontiguousarray(lr[use_channels])
        hr = np.ascontiguousarray(hr[use_channels])
    lr_reg = periodic_crop(lr, origin_lr, region_lr, pad=0)
    hr_origin = tuple(int(o) * factor for o in origin_lr)
    hr_reg = periodic_crop(hr, hr_origin, region_lr * factor, pad=0)
    lr_t = torch.from_numpy(np.ascontiguousarray(lr_reg)).float().unsqueeze(0)
    hr_t = torch.from_numpy(np.ascontiguousarray(hr_reg)).float().unsqueeze(0)
    return lr_t, hr_t


@torch.no_grad()
def stitch_region(
    flow, lr_region: torch.Tensor, tile_lr: int, factor: int, n_steps: int, device,
) -> torch.Tensor:
    """Non-overlapping tiled generation over an LR region -> stitched HR field.

    `lr_region` is (1, C, R, R, R) with R = region_lr. Returns (1, C, R*factor, ...).
    """
    _, c, R, _, _ = lr_region.shape
    if R % tile_lr != 0:
        raise ValueError(f"region_lr={R} not a multiple of tile_lr={tile_lr}")
    n_hr = R * factor
    out = torch.zeros(1, c, n_hr, n_hr, n_hr, device=device)
    for i in range(0, R, tile_lr):
        for j in range(0, R, tile_lr):
            for k in range(0, R, tile_lr):
                y = lr_region[:, :, i:i + tile_lr, j:j + tile_lr, k:k + tile_lr].to(device)
                x_hat = flow.generate(y, n_steps=n_steps)
                hi, hj, hk = i * factor, j * factor, k * factor
                t = tile_lr * factor
                out[:, :, hi:hi + t, hj:hj + t, hk:hk + t] = x_hat
    return out


# --------------------------------------------------------------------------- #
# Seam diagnostic
# --------------------------------------------------------------------------- #
def seam_ratio(field: torch.Tensor, tile_hr: int, factor: int) -> float:
    """Excess squared jump at tile-boundary planes vs OTHER block-boundary planes.

    `field` is (1, C, N, N, N); jumps are averaged over channels and the three
    axes. The reference is the set of planes at multiples of `factor` (the operator
    block size, where `A_plus(y)` already has a block-constant step) that are NOT
    tile boundaries. This controls for the block structure both tile sizes share,
    so the ratio isolates the EXTRA discontinuity introduced at tile seams: ~1.0
    means seams look like any other block boundary; >1.0 quantifies a visible seam.
    """
    n = field.shape[-1]
    seam_sq, seam_cnt, ref_sq, ref_cnt = 0.0, 0, 0.0, 0
    for axis in (-3, -2, -1):
        d = (field.index_select(axis, torch.arange(1, n, device=field.device))
             - field.index_select(axis, torch.arange(0, n - 1, device=field.device)))
        d2 = d.pow(2).mean(dim=1)  # over channels -> (1, ...) with axis length n-1
        # d index p corresponds to the plane between HR voxels p and p+1, i.e. b=p+1.
        bidx = torch.arange(1, n, device=field.device)
        is_seam = (bidx % tile_hr == 0)
        is_ref = (bidx % factor == 0) & ~is_seam
        d2_flat = d2.movedim(axis, 0).reshape(d2.shape[axis], -1)  # (n-1, rest)
        seam_sq += float(d2_flat[is_seam].sum()); seam_cnt += int(is_seam.sum()) * d2_flat.shape[1]
        ref_sq += float(d2_flat[is_ref].sum()); ref_cnt += int(is_ref.sum()) * d2_flat.shape[1]
    ref_mean = ref_sq / max(ref_cnt, 1)
    seam_mean = seam_sq / max(seam_cnt, 1)
    return float(seam_mean / ref_mean) if ref_mean > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Stats on the shared grid
# --------------------------------------------------------------------------- #
def region_metrics(
    x_hat: torch.Tensor, x_hr: torch.Tensor, factor: int, highpass: HighPassDensity,
    tile_hr: int, n_bins: int = 24,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["mse"] = float((x_hat - x_hr).pow(2).mean())
    out.update(rk_tk_summary(x_hat, x_hr, factor, n_bins=n_bins))

    rho_hat, rho_true = highpass.density(x_hat), highpass.density(x_hr)
    out["density_power_error"] = power_error(rho_hat, rho_true, n_bins)
    out["density_pdf_error"] = pdf_error(rho_hat, rho_true)
    out.update(rk_tk_summary(rho_hat, rho_true, factor, n_bins=n_bins, prefix="density_"))

    n = x_hat.shape[-1]
    k_lr = n / (2.0 * factor)
    width = max(2.0, k_lr / 4.0)
    ks = [0.5 * k_lr, k_lr, 1.5 * k_lr]
    b_hat = equilateral_bispectrum(rho_hat, ks, width)
    b_true = equilateral_bispectrum(rho_true, ks, width)
    out["bispectrum_error"] = float(
        ((b_hat - b_true).abs() / b_true.abs().clamp_min(1e-20)).mean())
    sq_hat = squeezed_cross_bispectrum(rho_hat, rho_true, 0.3 * k_lr, 1.5 * k_lr, width)
    sq_true = squeezed_cross_bispectrum(rho_true, rho_true, 0.3 * k_lr, 1.5 * k_lr, width)
    out["squeezed_cross_bispectrum_error"] = float(
        (sq_hat - sq_true).abs() / sq_true.abs().clamp_min(1e-20))

    out["seam_ratio_density"] = seam_ratio(rho_hat, tile_hr, factor)
    out["seam_ratio_field"] = seam_ratio(x_hat, tile_hr, factor)
    return out


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def save_slice_figure(base_up, model_fields: Dict[str, torch.Tensor], x_hr, highpass, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("LR upsampled (A+y)", base_up)] + list(model_fields.items()) + [("true HR", x_hr)]
    z = x_hr.shape[-1] // 2
    dens = [(name, highpass.density(f)[0, 0, :, :, z].cpu().numpy()) for name, f in panels]
    vmax = float(np.percentile(dens[-1][1], 99.5))
    fig, axes = plt.subplots(1, len(dens), figsize=(3.2 * len(dens), 3.4))
    for ax, (name, img) in zip(np.atleast_1d(axes), dens):
        ax.imshow(img, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(name, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Same-ground density: LR -> stitched SR -> true HR (z-slice)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True,
                    help="name:config.yaml:ckpt.pt  (repeatable)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--box-index", type=int, default=0, help="which box within the split")
    ap.add_argument("--region-lr", type=int, default=32, help="LR voxels of the eval region")
    ap.add_argument("--origin", type=int, nargs=3, default=[0, 0, 0])
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--set", nargs="*", default=None, help="dotted overrides for ALL configs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    specs = []
    for m in args.model:
        name, cfg_path, ckpt = m.split(":", 2)
        specs.append((name, cfg_path, ckpt))

    # Geometry + split come from the FIRST model's config; assert the rest agree on
    # the things that must be shared for the comparison to be fair.
    cfg0 = apply_overrides(load_config(specs[0][1]), args.set)
    factor = int(cfg0.get("factor", 8))
    dcfg0 = cfg0.get("data", {})
    use_channels = dcfg0.get("use_channels")
    split = resolve_split(dcfg0)
    lr_paths = getattr(split, f"{args.split}_lr")
    hr_paths = getattr(split, f"{args.split}_hr")
    if not lr_paths:
        raise ValueError(f"no {args.split} boxes in split")
    lr_path, hr_path = lr_paths[args.box_index], hr_paths[args.box_index]
    print(f"[region] box={Path(hr_path).stem} split={args.split} "
          f"region_lr={args.region_lr} -> HR {args.region_lr * factor}^3 origin={args.origin}")

    lr_region, hr_region = _load_region(
        lr_path, hr_path, factor, args.region_lr, tuple(args.origin), use_channels)
    hr_region = hr_region.to(device)

    hr_cellsize, _ = cellsizes(dcfg0, factor)
    highpass = HighPassDensity(
        factor=factor, lowpass=cfg0.get("critic", {}).get("lowpass", "blockavg"),
        disp_channels=(0, 1, 2), cellsize=hr_cellsize,
        dis_norm=float(dcfg0.get("dis_norm", 6000.0))).to(device)

    channels = len(use_channels) if use_channels is not None else int(dcfg0.get("channels", 6))

    # base = A+(y) upsampled, shared reference (uses the first operator; factor is
    # identical across models so any model's A_plus is the same map).
    results: Dict[str, Dict[str, float]] = {}
    model_fields: Dict[str, torch.Tensor] = {}
    base_up = None
    for name, cfg_path, ckpt in specs:
        cfg = apply_overrides(load_config(cfg_path), args.set)
        if int(cfg.get("factor", 8)) != factor:
            raise ValueError(f"{name}: factor mismatch")
        tile_lr = int(cfg.get("data", {}).get("crop_lr", 8))
        tile_hr = tile_lr * factor
        flow = load_flow(cfg, channels, ckpt, device, use_ema=not args.no_ema)
        torch.manual_seed(args.seed)  # reproducible, matched across models
        x_hat = stitch_region(flow, lr_region, tile_lr, factor, args.n_steps, device)
        if base_up is None:
            base_up = flow.operator.A_plus(lr_region.to(device))
        abs_err, rel_err = flow.operator.consistency_error(x_hat, lr_region.to(device))
        m = region_metrics(x_hat, hr_region, factor, highpass, tile_hr, n_bins=args.n_bins)
        m["exact_consistency_rel"] = rel_err
        m["tile_lr"] = tile_lr
        results[name] = m
        model_fields[name] = x_hat
        print(f"[{name}] tile_lr={tile_lr} rk_trans={m['rk_transition']:.4f} "
              f"rk_high={m['rk_high']:.4f} bispec={m['bispectrum_error']:.4f} "
              f"seam(dens)={m['seam_ratio_density']:.3f} cons_rel={rel_err:.2e}")
        del flow
        torch.cuda.empty_cache() if device.type == "cuda" else None

    with open(out_dir / "same_ground.json", "w") as f:
        json.dump({"config": {"region_lr": args.region_lr, "factor": factor,
                              "box": Path(hr_path).stem, "split": args.split,
                              "n_steps": args.n_steps, "seed": args.seed},
                   "results": results}, f, indent=2)
    save_slice_figure(base_up, model_fields, hr_region, highpass,
                      out_dir / "same_ground_density.png")

    # Comparison table on the shared grid.
    keys = ["rk_low", "rk_transition", "rk_high", "Tk_error_transition", "Tk_error_high",
            "density_power_error", "bispectrum_error", "squeezed_cross_bispectrum_error",
            "density_pdf_error", "seam_ratio_density", "seam_ratio_field"]
    names = list(results)
    print("\n=== SAME-GROUND comparison (identical region + k-grid) ===")
    print(f"{'metric':<34} " + "  ".join(f"{n:>12}" for n in names))
    for k in keys:
        row = "  ".join(f"{results[n].get(k, float('nan')):>12.4f}" for n in names)
        print(f"{k:<34} {row}")
    print(f"\nWrote {out_dir/'same_ground.json'} and {out_dir/'same_ground_density.png'}")


if __name__ == "__main__":
    main()
