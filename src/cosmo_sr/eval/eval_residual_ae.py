"""Evaluate a trained residual autoencoder on a real HR field (Experiment 1).

Compares AE reconstruction to base-only (identity) upsampling and reports
per-octave consistency, residual/field MSE, residual power ratio and high-k
recovery. Writes ``metrics.json``.

Usage::

    python -m cosmo_sr.eval.eval_residual_ae \
        --config configs/poc_residual_ae_real3.yaml \
        --checkpoint runs/poc_residual_ae_real3/ckpt_last.pt \
        --hr .../set15.npy --crop 128 --out runs/poc_residual_ae_real3/eval_set15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from ..data.field_io import load_field
from ..data.crops import periodic_crop
from ..operators.multiscale import MultiScaleOperators
from ..operators.base_upscaler import consistent_base
from ..train.common import load_checkpoint, save_slice_png
from ..train.train_residual_ae import build_ae, build_base_upscaler
from ..eval.flow_eval import highk_power_ratio
from ..utils.config import load_config


def build_pyramid(crop: torch.Tensor, full_res: int, n_levels: int) -> Dict[int, torch.Tensor]:
    levels = [crop]
    for _ in range(n_levels - 1):
        levels.append(F.avg_pool3d(levels[-1], kernel_size=2, stride=2))
    return {full_res // (2 ** lvl): t for lvl, t in enumerate(levels)}


@torch.no_grad()
def run(cfg: Dict, checkpoint: str, hr_path: str, crop_size: int, out_dir: str,
        n_crops: int = 4, device: str = "cpu") -> Dict[str, float]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    factor = int(cfg.get("factor", 2))
    channels = int(cfg.get("model", {}).get("channels", 6))
    full_res = int(cfg.get("data", {}).get("full_res", 512))
    n_levels = int(cfg.get("data", {}).get("n_levels", 4))
    resolutions: List[int] = list(cfg.get("resolutions", [64, 128, 256]))
    n_bands = int(cfg.get("loss", {}).get("n_bands", 8))

    ops = MultiScaleOperators(factor).to(dev)
    base_upscaler = build_base_upscaler(cfg, channels, factor, dev)
    ae = build_ae(cfg).to(dev)
    state = load_checkpoint(checkpoint, ae, map_location=dev)
    if isinstance(state.get("extra"), dict) and "base_upscaler" in state["extra"]:
        try:
            base_upscaler.load_state_dict(state["extra"]["base_upscaler"])
        except Exception:
            pass
    ae.eval()

    field = load_field(hr_path, mmap=True)
    Ng = field.shape[1]
    rng = np.random.default_rng(0)

    agg: Dict[str, List[float]] = {}

    def add(key, val):
        agg.setdefault(key, []).append(float(val))

    slice_saved = False
    for ci in range(n_crops):
        start = tuple(int(rng.integers(0, Ng)) for _ in range(3))
        crop_np = periodic_crop(field, start, crop_size, pad=0)
        crop = torch.from_numpy(np.ascontiguousarray(crop_np)).float().unsqueeze(0).to(dev)
        pyr = build_pyramid(crop, full_res, n_levels)
        for R in resolutions:
            if R not in pyr or 2 * R not in pyr:
                continue
            x_R = pyr[R]
            x_2R = pyr[2 * R]
            base = consistent_base(base_upscaler, ops, x_R)
            r_star = ops.P_null(x_2R - base)
            z = ae.encode(r_star)
            r_recon = ops.P_null(ae.decode(z))
            x_recon = base + r_recon
            x_base = base  # base-only reconstruction

            denom = float(torch.mean(x_R ** 2)) or 1.0
            add(f"val/ae/consistency_rel_R{R}", float(torch.mean((ops.A(x_recon) - x_R) ** 2)) / denom)
            add(f"val/ae/recon_res_mse_R{R}", float(F.mse_loss(r_recon, r_star)))
            vt = float(torch.var(r_star)) or 1.0
            add(f"val/ae/res_power_ratio_R{R}", float(torch.var(r_recon)) / vt)
            add(f"val/ae/finite_frac_R{R}", float(torch.isfinite(x_recon).float().mean()))

            ae_x_mse = float(F.mse_loss(x_recon, x_2R))
            base_x_mse = float(F.mse_loss(x_base, x_2R))
            add(f"val/ae_x_mse_R{R}", ae_x_mse)
            add(f"val/base_x_mse_R{R}", base_x_mse)
            add(f"val/ae_improvement_x_mse_R{R}", base_x_mse / (ae_x_mse if ae_x_mse > 0 else 1.0))
            ae_hk = float(highk_power_ratio(x_recon, x_2R)["highk_power_ratio"])
            base_hk = float(highk_power_ratio(x_base, x_2R)["highk_power_ratio"])
            add(f"val/ae/highk_R{R}", ae_hk)
            add(f"val/ae_highk_R{R}", ae_hk)
            add(f"val/base_highk_R{R}", base_hk)

            if not slice_saved and R == max(resolutions):
                save_slice_png(out / f"recon_R{R}.png", x_recon[0].cpu().numpy(), title=f"AE recon R{R}")
                save_slice_png(out / f"true_R{R}.png", x_2R[0].cpu().numpy(), title=f"true 2R (R{R})")
                save_slice_png(out / f"base_R{R}.png", x_base[0].cpu().numpy(), title=f"base R{R}")
                slice_saved = True

    metrics = {k: float(np.mean(v)) for k, v in agg.items()}
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval_residual_ae] wrote {out/'metrics.json'}")
    for k in sorted(metrics):
        print(f"  {k}: {metrics[k]:.4g}")
    return metrics


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate residual autoencoder (Experiment 1)")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--hr", required=True, type=str)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n-crops", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg, args.checkpoint, args.hr, args.crop, args.out,
        n_crops=args.n_crops, device=args.device)


if __name__ == "__main__":
    main()
