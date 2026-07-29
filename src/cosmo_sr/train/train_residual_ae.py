"""Residual autoencoder training (stage 1 of the latent residual flow).

For each adjacent octave pair ``(x_R, x_2R)`` the target is the null-space
residual ``r_star = P_null_R(x_2R - B_cons_R(x_R))``. The AE compresses and
reconstructs it; the reconstruction is re-projected through ``P_null`` and added
to the consistent base so that ``A_R(x_recon) = x_R`` holds exactly::

    z            = ae.encode(r_star)
    r_recon      = P_null(ae.decode(z))
    x_recon      = B_cons(x_R) + r_recon

Losses: reconstruction MSE on the residual and the field, plus a band-power
statistics term. ``loss_A_metric`` is logged but not optimised (hard consistency
already makes it ~0).

Usage::

    python -m cosmo_sr.train.train_residual_ae --config configs/residual_ae.yaml --smoke
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from ..data.datasets import PyramidCropDataset, SyntheticPyramidDataset, infinite_loader, list_fields
from ..losses.flow import band_power, band_statistics_loss
from ..models.residual_autoencoder import ResidualAutoencoder
from ..models.unet_baseline import SimpleSRGenerator
from ..operators.base_upscaler import BackboneUpscaler, IdentityUpscaler, consistent_base
from ..operators.multiscale import MultiScaleOperators
from ..utils.seed import seed_everything
from . import common


def build_ae(cfg: Dict[str, Any]) -> ResidualAutoencoder:
    m = cfg.get("model", {})
    return ResidualAutoencoder(
        channels=int(m.get("channels", 6)),
        width=int(m.get("width", 32)),
        ch_mults=tuple(m.get("ch_mults", (1, 2, 2))),
        latent_channels=int(m.get("latent_channels", 16)),
        n_res=int(m.get("n_res", 1)),
    )


def build_base_upscaler(cfg: Dict[str, Any], channels: int, factor: int, device):
    kind = str(cfg.get("base_upscaler", {}).get("kind", "identity")).lower()
    if kind == "identity":
        return IdentityUpscaler(factor=factor).to(device)
    if kind in ("backbone", "sr2"):
        bcfg = cfg.get("base_upscaler", {})
        backbone = SimpleSRGenerator(
            in_channels=channels, out_channels=channels, scale_factor=factor,
            width=int(bcfg.get("width", 32)), depth=int(bcfg.get("depth", 2)),
        )
        return BackboneUpscaler(backbone, factor=factor).to(device)
    raise ValueError(f"Unknown base_upscaler.kind {kind!r}")


def _build_datasets(cfg: Dict[str, Any], smoke: bool):
    data = cfg.get("data", {})
    crop_hr = int(data.get("crop_hr", 64))
    n_levels = int(data.get("n_levels", 4))
    full_res = int(data.get("full_res", 512))
    mmap = bool(data.get("mmap", False))
    hr_paths = list_fields(data.get("paired_hr_glob"))
    synthetic = smoke or not hr_paths
    if synthetic:
        crop_hr = min(crop_hr, 32)
        paired_ds = SyntheticPyramidDataset(num_samples=8, crop_hr=crop_hr, n_levels=n_levels,
                                            full_res=full_res, seed=0)
        val_ds = SyntheticPyramidDataset(num_samples=4, crop_hr=crop_hr, n_levels=n_levels,
                                         full_res=full_res, seed=999)
    else:
        paired_ds = PyramidCropDataset(hr_paths, crop_hr=crop_hr, n_levels=n_levels,
                                       full_res=full_res, seed=0, mmap=mmap)
        val_hr = list_fields(data.get("val_hr_glob"))
        val_ds = (PyramidCropDataset(val_hr, crop_hr=crop_hr, n_levels=n_levels,
                                     full_res=full_res, seed=123, mmap=mmap)
                  if val_hr else None)
    return paired_ds, val_ds, crop_hr, n_levels, full_res


def ae_step_metrics(ae, ops, base_upscaler, x_R, x_2R, n_bands):
    """Compute AE forward + all loss terms/metrics for one octave pair."""
    base = consistent_base(base_upscaler, ops, x_R)
    r_star = ops.P_null(x_2R - base)
    z = ae.encode(r_star)
    r_recon = ops.P_null(ae.decode(z))
    x_recon = base + r_recon
    loss_r = F.mse_loss(r_recon, r_star)
    loss_x = F.mse_loss(x_recon, x_2R)
    tgt_band = band_power(r_star, n_bands=n_bands, log=True)
    loss_band = band_statistics_loss(r_recon, tgt_band, n_bands=n_bands)
    with torch.no_grad():
        a_err = float(torch.mean((ops.A(x_recon) - x_R) ** 2))
        denom = float(torch.mean(x_R ** 2)) or 1.0
        vt = float(torch.var(r_star)) or 1.0
        stats = {
            "recon_res_mse": float(loss_r.detach()),
            "recon_x_mse": float(loss_x.detach()),
            "band_loss": float(loss_band.detach()),
            "loss_A_metric": a_err,
            "consistency_rel": a_err / denom,
            "res_power_ratio": float(torch.var(r_recon)) / vt,
            "latent_mean": float(z.mean()),
            "latent_std": float(z.std()),
            "latent_abs_mean": float(z.abs().mean()),
            "latent_l2": float(z.pow(2).mean().sqrt()),
        }
    return loss_r, loss_x, loss_band, stats


@torch.no_grad()
def _val_ae_metrics(ae, ops, base_upscaler, val_iter, resolutions, device, n_bands,
                    n_batches: int = 2):
    if val_iter is None:
        return {}
    ae.eval()
    agg: Dict[str, float] = {}
    count = 0
    for _ in range(n_batches):
        vb = common.to_device_batch(next(val_iter), device)
        for R in resolutions:
            x_R = vb[f"r{R}"]
            x_2R = vb[f"r{2 * R}"]
            _, _, _, s = ae_step_metrics(ae, ops, base_upscaler, x_R, x_2R, n_bands)
            agg[f"val/ae/recon_res_mse_R{R}"] = agg.get(f"val/ae/recon_res_mse_R{R}", 0.0) + s["recon_res_mse"]
            agg[f"val/ae/recon_x_mse_R{R}"] = agg.get(f"val/ae/recon_x_mse_R{R}", 0.0) + s["recon_x_mse"]
            agg[f"val/ae/consistency_rel_R{R}"] = agg.get(f"val/ae/consistency_rel_R{R}", 0.0) + s["consistency_rel"]
            agg[f"val/ae/res_power_ratio_R{R}"] = agg.get(f"val/ae/res_power_ratio_R{R}", 0.0) + s["res_power_ratio"]
        count += 1
    ae.train()
    return {k: v / max(count, 1) for k, v in agg.items()}


def train(cfg: Dict[str, Any], smoke: bool = False) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})
    model_cfg = cfg.get("model", {})
    seed = int(train_cfg.get("seed", 0))
    seed_everything(seed)

    device = common.select_device("cpu" if smoke else train_cfg.get("device"))
    channels = int(model_cfg.get("channels", 6))
    factor = int(cfg.get("factor", 2))
    resolutions: List[int] = list(cfg.get("resolutions", [64, 128, 256]))
    n_bands = int(loss_cfg.get("n_bands", 8))
    lambda_r = float(loss_cfg.get("lambda_r", 1.0))
    lambda_x = float(loss_cfg.get("lambda_x", 1.0))
    lambda_band = float(loss_cfg.get("lambda_band", 1.0))
    amp_enabled = bool(train_cfg.get("amp", False))

    steps = int(train_cfg.get("steps", 20000))
    if smoke:
        steps = min(steps, 12)
    lr = float(train_cfg.get("lr", 1e-4))
    bs = int(train_cfg.get("batch_size", 1))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 1000))
    eval_every = int(train_cfg.get("eval_every", max(log_every, 500)))
    if smoke:
        eval_every = min(eval_every, 6)

    paired_ds, val_ds, crop_hr, n_levels, full_res = _build_datasets(cfg, smoke)

    ops = MultiScaleOperators(factor=factor).to(device)
    base_upscaler = build_base_upscaler(cfg, channels, factor, device)
    ae = build_ae(cfg).to(device)

    params = list(ae.parameters()) + list(base_upscaler.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    autocast, scaler = common.amp_components(amp_enabled, device)

    run_dir = Path(cfg.get("output", {}).get("run_dir", "runs/residual_ae"))
    common.init_run_dir(run_dir, cfg)
    use_wandb = (not smoke) and common.maybe_init_wandb(cfg, run_dir, "residual_ae")
    logger = common.CSVLogger(run_dir, use_wandb=use_wandb)

    paired_iter = infinite_loader(paired_ds, bs, seed=seed)
    val_iter = infinite_loader(val_ds, bs, seed=seed + 99) if val_ds is not None else None
    rng = np.random.default_rng(seed)

    ae.train()
    first: Dict[str, float] = {}
    last: Dict[str, float] = {}
    for step in range(1, steps + 1):
        t_step = time.perf_counter()
        do_eval = val_iter is not None and (step % eval_every == 0 or step == steps)
        will_log = step % log_every == 0 or step == 1 or step == steps or do_eval

        pb = common.to_device_batch(next(paired_iter), device)
        R = resolutions[int(rng.integers(0, len(resolutions)))]
        x_R = pb[f"r{R}"]
        x_2R = pb[f"r{2 * R}"]
        with autocast():
            loss_r, loss_x, loss_band, stats = ae_step_metrics(
                ae, ops, base_upscaler, x_R, x_2R, n_bands
            )
            loss = lambda_r * loss_r + lambda_x * loss_x + lambda_band * loss_band

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        logs: Dict[str, float] = {}
        if will_log:
            scaler.unscale_(optimizer)
            logs["grad_norm"] = common.grad_global_norm(params)
        scaler.step(optimizer)
        scaler.update()

        logs["loss"] = float(loss.detach())
        logs["octave_R"] = float(R)
        logs["ae/loss"] = float(loss.detach())
        logs["ae/recon_res_mse"] = stats["recon_res_mse"]
        logs["ae/recon_x_mse"] = stats["recon_x_mse"]
        logs["ae/band_loss"] = stats["band_loss"]
        logs["ae/consistency_rel"] = stats["consistency_rel"]
        logs["ae/res_power_ratio"] = stats["res_power_ratio"]
        logs["ae/latent_mean"] = stats["latent_mean"]
        logs["ae/latent_std"] = stats["latent_std"]
        logs["ae/latent_abs_mean"] = stats["latent_abs_mean"]
        logs["ae/latent_l2"] = stats["latent_l2"]
        logs["ae/loss_A_metric"] = stats["loss_A_metric"]
        last = logs
        if not first:
            first = dict(logs)

        if will_log:
            logs.update(common.system_metrics(device, time.perf_counter() - t_step))
            row = {**logs, "lr": lr}
            if do_eval:
                row.update(_val_ae_metrics(ae, ops, base_upscaler, val_iter, resolutions,
                                           device, n_bands))
            logger.log(step, row)
        if save_every > 0 and (step % save_every == 0):
            common.save_checkpoint(run_dir / f"ckpt_{step}.pt", ae, optimizer, step,
                                   extra={"base_upscaler": base_upscaler.state_dict()})

    common.save_checkpoint(run_dir / "ckpt_last.pt", ae, optimizer, steps,
                           extra={"base_upscaler": base_upscaler.state_dict()})
    logger.close()
    common.finish_wandb()
    return {"run_dir": str(run_dir), "first": first, "last": last, "steps": steps,
            "checkpoint": str(run_dir / "ckpt_last.pt")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Residual autoencoder training")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--smoke", action="store_true", help="short CPU smoke run")
    parser.add_argument("--set", nargs="*", default=None,
                        help="dotted config overrides, e.g. train.steps=30")
    args = parser.parse_args(argv)
    from ..utils.config import apply_overrides, load_config

    cfg = apply_overrides(load_config(args.config), args.set)
    result = train(cfg, smoke=args.smoke)
    msg = " ".join(f"{k}={v:.4g}" for k, v in result["last"].items())
    print(f"[residual_ae] done: steps={result['steps']} last[{msg}] run_dir={result['run_dir']}")


if __name__ == "__main__":
    main()
