"""Our method: ambient LR-consistency + scarce paired HR training.

Forward::

    x_hat_unpaired = G(y_lr_unpaired);  loss_ambient = mse(A(x_hat_unpaired), y_lr_unpaired)
    (optional) x_hat_paired = G(y_lr_paired);  loss_pair = mse(x_hat_paired, x_hr_paired)
    loss = lambda_ambient*loss_ambient + lambda_pair*loss_pair + lambda_reg*loss_reg

Usage::

    python -m cosmo_sr.train.train_ambient --config configs/ambient_smoke.yaml --smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..data.datasets import (
    FieldCropDataset,
    SyntheticSRDataset,
    infinite_loader,
    list_fields,
)
from ..losses import compute_losses
from ..losses.ambient import compute_ambient
from ..losses.supervised import supervised_loss
from ..models.wrappers import build_generator, model_config_kwargs
from ..operators.degrader import FixedDegrader
from ..utils.seed import seed_everything
from . import common


def _build_datasets(cfg: Dict[str, Any], smoke: bool, use_ambient: bool, use_pair: bool):
    data = cfg.get("data", {})
    crop_lr = int(data.get("crop_lr", 16))
    scale = int(data.get("scale_factor", 8))
    mmap = bool(data.get("mmap", False))

    unpaired_lr = list_fields(data.get("unpaired_lr_glob"))
    paired_lr = list_fields(data.get("paired_lr_glob"))
    paired_hr = list_fields(data.get("paired_hr_glob"))
    val_lr = list_fields(data.get("val_lr_glob"))
    val_hr = list_fields(data.get("val_hr_glob"))

    synthetic = smoke or not (unpaired_lr or (paired_lr and paired_hr))
    if synthetic:
        crop_lr = min(crop_lr, 8)

    unpaired_ds = None
    if use_ambient:
        if synthetic or not unpaired_lr:
            unpaired_ds = SyntheticSRDataset(
                num_samples=8, crop_lr=crop_lr, scale_factor=scale, seed=0
            )
        else:
            unpaired_ds = FieldCropDataset(
                unpaired_lr, None, crop_lr=crop_lr, scale_factor=scale, seed=0, mmap=mmap
            )

    paired_ds = None
    if use_pair:
        if synthetic or not (paired_lr and paired_hr):
            paired_ds = SyntheticSRDataset(
                num_samples=4, crop_lr=crop_lr, scale_factor=scale, seed=500
            )
        else:
            paired_ds = FieldCropDataset(
                paired_lr, paired_hr, crop_lr=crop_lr, scale_factor=scale, seed=1, mmap=mmap
            )

    val_ds = None
    if val_lr and val_hr and not synthetic:
        val_ds = FieldCropDataset(val_lr, val_hr, crop_lr=crop_lr, scale_factor=scale, seed=2, mmap=mmap)
    elif use_pair:
        val_ds = SyntheticSRDataset(
            num_samples=4, crop_lr=crop_lr, scale_factor=scale, seed=900
        )
    return unpaired_ds, paired_ds, val_ds, crop_lr, scale


@torch.no_grad()
def _val_metrics(model, degrader, val_iter, device, n_batches: int = 2):
    if val_iter is None:
        return {}
    model.eval()
    amb = 0.0
    hr = 0.0
    have_hr = False
    for _ in range(n_batches):
        batch = common.to_device_batch(next(val_iter), device)
        loss_amb, x_hat, _ = compute_ambient(model, degrader, batch["lr"])
        amb += loss_amb.item()
        if "hr" in batch:
            hr += supervised_loss(x_hat, batch["hr"]).item()
            have_hr = True
    model.train()
    out = {"val_ambient": amb / n_batches}
    if have_hr:
        out["val_hr_mse"] = hr / n_batches
    return out


def train(cfg: Dict[str, Any], smoke: bool = False) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})
    seed = int(train_cfg.get("seed", 0))
    seed_everything(seed)

    device = common.select_device("cpu" if smoke else train_cfg.get("device"))

    lambda_ambient = float(loss_cfg.get("lambda_ambient", 1.0))
    lambda_pair = float(loss_cfg.get("lambda_pair", 1.0))
    lambda_tv = float(loss_cfg.get("lambda_tv", 0.0))
    recon = loss_cfg.get("recon", "mse")
    huber_delta = float(loss_cfg.get("huber_delta", 1.0))
    reg_cfg = {"lambda_tv": lambda_tv,
               "lambda_finite": float(loss_cfg.get("lambda_finite", 0.0)),
               "lambda_meanstd": float(loss_cfg.get("lambda_meanstd", 0.0))}

    use_ambient = lambda_ambient != 0.0
    use_pair = lambda_pair != 0.0

    steps = int(train_cfg.get("steps", 10000))
    if smoke:
        steps = min(steps, 20)
    lr = float(train_cfg.get("lr", 1e-4))
    bs_unpaired = int(train_cfg.get("batch_size_unpaired", 1))
    bs_paired = int(train_cfg.get("batch_size_paired", 1))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 1000))

    unpaired_ds, paired_ds, val_ds, crop_lr, scale = _build_datasets(
        cfg, smoke, use_ambient, use_pair
    )

    degrader = FixedDegrader(scale, mode=cfg.get("degrader", {}).get("mode", "average")).to(device)

    model_cfg = dict(cfg.get("model", {"name": "SimpleSRGenerator"}))
    kwargs = model_config_kwargs(model_cfg)
    kwargs.setdefault("scale_factor", scale)
    model = build_generator(model_cfg.get("name", "SimpleSRGenerator"), **kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    run_dir = Path(cfg.get("output", {}).get("run_dir", "runs/ambient"))
    common.init_run_dir(run_dir, cfg)
    use_wandb = (not smoke) and common.maybe_init_wandb(cfg, run_dir, "ambient")
    logger = common.CSVLogger(run_dir, use_wandb=use_wandb)

    unpaired_iter = (
        infinite_loader(unpaired_ds, bs_unpaired, seed=seed) if unpaired_ds is not None else None
    )
    paired_iter = (
        infinite_loader(paired_ds, bs_paired, seed=seed + 7) if paired_ds is not None else None
    )
    val_iter = infinite_loader(val_ds, 1, seed=seed + 11) if val_ds is not None else None

    import time

    model.train()
    first = {}
    last = {}
    for step in range(1, steps + 1):
        t_step = time.perf_counter()
        will_log = step % log_every == 0 or step == 1 or step == steps
        kw: Dict[str, Any] = {}
        if unpaired_iter is not None:
            ub = common.to_device_batch(next(unpaired_iter), device)
            kw["y_lr_unpaired"] = ub["lr"]
        if paired_iter is not None:
            pb = common.to_device_batch(next(paired_iter), device)
            kw["y_lr_paired"] = pb["lr"]
            kw["x_hr_paired"] = pb["hr"]

        losses = compute_losses(
            model, degrader,
            lambda_ambient=lambda_ambient, lambda_pair=lambda_pair, lambda_reg=1.0,
            reg_cfg=reg_cfg, raise_on_nonfinite=True, recon=recon, huber_delta=huber_delta, **kw,
        )
        loss = losses["loss"]

        optimizer.zero_grad()
        loss.backward()
        grad_norm = common.grad_global_norm(model.parameters()) if will_log else None
        optimizer.step()

        scalars = {k: v.item() for k, v in losses.items()}
        last = scalars
        if not first:
            first = dict(scalars)

        if will_log:
            metrics = {**scalars, "lr": lr}
            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm
            metrics.update(common.system_metrics(device, time.perf_counter() - t_step))
            metrics.update(_val_metrics(model, degrader, val_iter, device))
            logger.log(step, metrics)

        if save_every > 0 and (step % save_every == 0):
            common.save_checkpoint(run_dir / f"ckpt_{step}.pt", model, optimizer, step)

    common.save_checkpoint(run_dir / "ckpt_last.pt", model, optimizer, steps)
    logger.close()
    common.finish_wandb()
    return {
        "run_dir": str(run_dir),
        "first": first,
        "last": last,
        "steps": steps,
        "checkpoint": str(run_dir / "ckpt_last.pt"),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Ambient + scarce-paired SR training")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--smoke", action="store_true", help="short CPU smoke run")
    args = parser.parse_args(argv)

    from ..utils.config import load_config

    cfg = load_config(args.config)
    result = train(cfg, smoke=args.smoke)
    msg = " ".join(f"{k}={v:.4g}" for k, v in result["last"].items())
    print(f"[ambient] done: steps={result['steps']} last[{msg}] run_dir={result['run_dir']}")


if __name__ == "__main__":
    main()
