"""Supervised baseline training: ``G(y_lr) -> x_hr`` on paired data only.

Usage::

    python -m cosmo_sr.train.train_supervised --config configs/supervised_baseline.yaml
    python -m cosmo_sr.train.train_supervised --config configs/supervised_baseline.yaml --smoke
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
from ..losses.regularizers import regularization_loss, regularizers_enabled, assert_finite
from ..losses.supervised import supervised_loss
from ..models.wrappers import build_generator, model_config_kwargs
from ..utils.seed import seed_everything
from . import common


def _build_datasets(cfg: Dict[str, Any], smoke: bool):
    data = cfg.get("data", {})
    crop_lr = int(data.get("crop_lr", 16))
    scale = int(data.get("scale_factor", 8))
    mmap = bool(data.get("mmap", False))

    train_lr = list_fields(data.get("train_lr_glob"))
    train_hr = list_fields(data.get("train_hr_glob"))
    val_lr = list_fields(data.get("val_lr_glob"))
    val_hr = list_fields(data.get("val_hr_glob"))

    if smoke or not (train_lr and train_hr):
        crop_lr = min(crop_lr, 8)
        train_ds = SyntheticSRDataset(
            num_samples=8, crop_lr=crop_lr, scale_factor=scale, seed=0
        )
        val_ds = SyntheticSRDataset(
            num_samples=4, crop_lr=crop_lr, scale_factor=scale, seed=1000
        )
        return train_ds, val_ds, crop_lr, scale

    train_ds = FieldCropDataset(train_lr, train_hr, crop_lr=crop_lr, scale_factor=scale, seed=0, mmap=mmap)
    val_ds = None
    if val_lr and val_hr:
        val_ds = FieldCropDataset(val_lr, val_hr, crop_lr=crop_lr, scale_factor=scale, seed=1, mmap=mmap)
    return train_ds, val_ds, crop_lr, scale


@torch.no_grad()
def _val_loss(model, loader_iter, device, n_batches: int = 2) -> Optional[float]:
    if loader_iter is None:
        return None
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        batch = common.to_device_batch(next(loader_iter), device)
        x_hat = model(batch["lr"])
        total += supervised_loss(x_hat, batch["hr"]).item()
    model.train()
    return total / n_batches


def train(cfg: Dict[str, Any], smoke: bool = False) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {})
    seed = int(train_cfg.get("seed", 0))
    seed_everything(seed)

    device = common.select_device("cpu" if smoke else train_cfg.get("device"))

    steps = int(train_cfg.get("steps", 10000))
    if smoke:
        steps = min(steps, 20)
    lr = float(train_cfg.get("lr", 1e-4))
    batch_size = int(train_cfg.get("batch_size", 1))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 1000))

    train_ds, val_ds, crop_lr, scale = _build_datasets(cfg, smoke)

    model_cfg = dict(cfg.get("model", {"name": "SimpleSRGenerator"}))
    kwargs = model_config_kwargs(model_cfg)
    kwargs.setdefault("scale_factor", scale)
    model = build_generator(model_cfg.get("name", "SimpleSRGenerator"), **kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    run_dir = Path(cfg.get("output", {}).get("run_dir", "runs/supervised_baseline"))
    common.init_run_dir(run_dir, cfg)
    use_wandb = (not smoke) and common.maybe_init_wandb(cfg, run_dir, "supervised")
    logger = common.CSVLogger(run_dir, use_wandb=use_wandb)

    reg_cfg = cfg.get("loss", {})
    use_reg = regularizers_enabled(reg_cfg)
    recon = reg_cfg.get("recon", "mse")
    huber_delta = float(reg_cfg.get("huber_delta", 1.0))

    train_iter = infinite_loader(train_ds, batch_size, seed=seed)
    val_iter = infinite_loader(val_ds, batch_size, seed=seed + 1) if val_ds is not None else None

    import time

    model.train()
    first_loss = None
    last_loss = None
    for step in range(1, steps + 1):
        t_step = time.perf_counter()
        will_log = step % log_every == 0 or step == 1 or step == steps
        batch = common.to_device_batch(next(train_iter), device)
        x_hat = model(batch["lr"])
        loss = supervised_loss(x_hat, batch["hr"], kind=recon, huber_delta=huber_delta)
        if use_reg:
            loss_reg, _ = regularization_loss(x_hat, reg_cfg, reference=batch["hr"])
            loss = loss + loss_reg
        assert_finite(loss, "train_loss")

        optimizer.zero_grad()
        loss.backward()
        grad_norm = common.grad_global_norm(model.parameters()) if will_log else None
        optimizer.step()

        last_loss = loss.item()
        if first_loss is None:
            first_loss = last_loss

        if will_log:
            vloss = _val_loss(model, val_iter, device)
            metrics = {"train_loss": last_loss, "lr": lr}
            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm
            metrics.update(common.system_metrics(device, time.perf_counter() - t_step))
            if vloss is not None:
                metrics["val_loss"] = vloss
            logger.log(step, metrics)

        if save_every > 0 and (step % save_every == 0):
            common.save_checkpoint(run_dir / f"ckpt_{step}.pt", model, optimizer, step)

    common.save_checkpoint(run_dir / "ckpt_last.pt", model, optimizer, steps)
    logger.close()
    common.finish_wandb()
    return {
        "run_dir": str(run_dir),
        "first_loss": first_loss,
        "last_loss": last_loss,
        "steps": steps,
        "checkpoint": str(run_dir / "ckpt_last.pt"),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Supervised SR baseline training")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--smoke", action="store_true", help="short CPU smoke run")
    args = parser.parse_args(argv)

    from ..utils.config import load_config

    cfg = load_config(args.config)
    result = train(cfg, smoke=args.smoke)
    print(
        f"[supervised] done: steps={result['steps']} "
        f"first_loss={result['first_loss']:.6g} last_loss={result['last_loss']:.6g} "
        f"run_dir={result['run_dir']}"
    )


if __name__ == "__main__":
    main()
