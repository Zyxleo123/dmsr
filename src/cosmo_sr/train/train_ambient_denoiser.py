"""Controlled-study trainer for the operator-conditioned HR denoiser (Phase C).

One shared denoiser ``D_psi`` trained with a clean HR branch and/or an
operator-conditioned ambient branch. The ambient branch's ``(y, g)`` pairs come
from clean HR via one of three constructions (all clean HR is hidden from the
ambient loss -- it only sees the measurement):

    branches.ambient:  none | fixed(C1) | true_shift(C2) | virtual_shift(C3)

Named runs (same model/optim/data/seed, only the branch knobs differ):

    P0 clean-only            clean=true  ambient=none
    P1/P2/P3 ambient-only    clean=false ambient=fixed/true_shift/virtual_shift
    P4/P5/P6 mixed           clean=true  ambient=fixed/true_shift/virtual_shift

Decisive comparison: P0 vs P4 vs P6 (and P5 = the C2 upper bound). Gate 2 =
P6(virtual) beating P4(fixed) on held-out clean-HR metrics across seeds.

    python -m cosmo_sr.train.train_ambient_denoiser --config configs/prior_mixed_virtual_shift.yaml [--smoke]
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.datasets import FieldCropDataset, infinite_loader, list_fields
from ..eval.denoise_eval import clean_denoise_metrics
from ..losses.ambient_denoise import (
    ambient_denoise_loss,
    build_ambient_target,
    clean_denoise_loss,
)
from ..models.operator_denoiser import (
    CosineSchedule,
    ModelEMA,
    OperatorConditionedDenoiser,
)
from ..operators.shifted_operator import ShiftedDownsampleOperator
from ..utils.seed import seed_everything
from . import common

AMBIENT_MODES = {"none", "fixed", "true_shift", "virtual_shift"}


class _SyntheticCrops(Dataset):
    """Random HR-like crops for CPU smoke runs (keyed ``"lr"`` like FieldCropDataset)."""

    def __init__(self, channels: int, crop: int, n: int = 16, seed: int = 0):
        self.c, self.crop, self.n = channels, crop, n
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"lr": torch.randn(self.c, self.crop, self.crop, self.crop, generator=self.g)}


def _build_crop_stream(paths, crop, channels, use_channels, augment, mmap, seed):
    return FieldCropDataset(
        paths, None, crop_lr=crop, scale_factor=1, channels=channels,
        use_channels=use_channels, augment=augment, mmap=mmap, seed=seed,
    )


def train(cfg: Dict[str, Any], smoke: bool = False) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    br = cfg.get("branches", {})
    seed = int(train_cfg.get("seed", 0))
    seed_everything(seed)

    device = common.select_device("cpu" if smoke else train_cfg.get("device"))
    factor = int(cfg.get("factor", 2))
    use_channels = data_cfg.get("use_channels")  # e.g. [0,1,2] displacement-only
    channels = len(use_channels) if use_channels else int(model_cfg.get("channels", 6))
    crop_hr = int(data_cfg.get("crop_hr", 64))

    use_clean = bool(br.get("clean", True))
    ambient_mode = str(br.get("ambient", "none"))
    if ambient_mode not in AMBIENT_MODES:
        raise ValueError(f"branches.ambient must be one of {AMBIENT_MODES}, got {ambient_mode!r}")
    lambda_ambient = float(br.get("lambda_ambient", 1.0))
    use_ambient = ambient_mode != "none" and lambda_ambient != 0.0
    if not (use_clean or use_ambient):
        raise ValueError("at least one of the clean / ambient branches must be active")

    steps = min(int(train_cfg.get("steps", 20000)), 8 if smoke else 1 << 30)
    lr = float(train_cfg.get("lr", 1e-4))
    bs = int(train_cfg.get("batch_size", 1))
    ema_decay = float(train_cfg.get("ema_decay", 0.999))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 2000))
    eval_every = min(int(train_cfg.get("eval_every", 500)), 4 if smoke else 1 << 30)
    amp_enabled = bool(train_cfg.get("amp", False))

    # ---- data ----
    hr_paths = list_fields(data_cfg.get("hr_glob"))
    # Boxes the ambient branch may see. These stand in for LR-only boxes, so keeping them
    # disjoint from hr_glob is what makes the few-HR/many-LR claim testable; defaults to
    # hr_glob (both branches share one pool) when unset.
    ambient_paths = list_fields(data_cfg.get("ambient_glob")) or hr_paths
    val_paths = list_fields(data_cfg.get("val_hr_glob"))
    augment = bool(data_cfg.get("augment", True))
    mmap = bool(data_cfg.get("mmap", False))
    if smoke or not hr_paths:
        crop_hr = min(crop_hr, 16)
        clean_ds = _SyntheticCrops(channels, crop_hr, seed=seed)
        ambient_ds = _SyntheticCrops(channels, crop_hr, seed=seed + 1)
        val_ds: Optional[Dataset] = _SyntheticCrops(channels, crop_hr, n=4, seed=seed + 2)
    else:
        clean_ds = _build_crop_stream(hr_paths, crop_hr, 6, use_channels, augment, mmap, seed)
        ambient_ds = _build_crop_stream(ambient_paths, crop_hr, 6, use_channels, augment, mmap, seed + 1)
        val_ds = (_build_crop_stream(val_paths, crop_hr, 6, use_channels, False, mmap, seed + 2)
                  if val_paths else None)

    clean_iter = infinite_loader(clean_ds, bs, seed=seed)
    ambient_iter = infinite_loader(ambient_ds, bs, seed=seed + 100)
    val_iter = infinite_loader(val_ds, bs, seed=seed + 200) if val_ds is not None else None

    # ---- model / operator / schedule ----
    operator = ShiftedDownsampleOperator(factor).to(device)
    schedule = CosineSchedule()
    model = OperatorConditionedDenoiser(
        channels=channels,
        width=int(model_cfg.get("width", 64)),
        num_levels=int(model_cfg.get("num_levels", 3)),
        blocks_per_level=int(model_cfg.get("blocks_per_level", 1)),
        embed_dim=int(model_cfg.get("embed_dim", 128)),
        factor=factor,
        use_resblocks=bool(model_cfg.get("use_resblocks", True)),
        use_attention=bool(model_cfg.get("use_attention", True)),
        attention_heads=int(model_cfg.get("attention_heads", 4)),
        use_checkpoint=bool(model_cfg.get("grad_checkpoint", False)),
    ).to(device)
    ema = ModelEMA(model, decay=ema_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    autocast, scaler = common.amp_components(amp_enabled, device)

    run_dir = Path(cfg.get("output", {}).get("run_dir", "runs/prior_denoiser"))
    common.init_run_dir(run_dir, cfg)
    use_wandb = (not smoke) and common.maybe_init_wandb(cfg, run_dir, "prior")
    logger = common.CSVLogger(run_dir, use_wandb=use_wandb)
    manifest = {"train_boxes": [Path(p).stem for p in hr_paths],
                "ambient_boxes": [Path(p).stem for p in ambient_paths],
                "val_boxes": [Path(p).stem for p in val_paths],
                "clean": use_clean, "ambient": ambient_mode,
                "lambda_ambient": lambda_ambient, "channels": channels,
                "factor": factor, "n_params": sum(p.numel() for p in model.parameters())}
    (run_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[prior] split manifest: {manifest}")
    rng = np.random.default_rng(seed)

    model.train()
    first: Dict[str, float] = {}
    last: Dict[str, float] = {}
    for step in range(1, steps + 1):
        t_step = time.perf_counter()
        do_eval = val_iter is not None and (step % eval_every == 0 or step == steps)
        will_log = step % log_every == 0 or step == 1 or step == steps or do_eval
        logs: Dict[str, float] = {}

        optimizer.zero_grad()
        total = torch.zeros((), device=device)
        with autocast():
            if use_clean:
                x = common.to_device_batch(next(clean_iter), device)["lr"]
                l_clean, d = clean_denoise_loss(model, x, schedule)
                total = total + l_clean
                logs.update(d)
            if use_ambient:
                x2 = common.to_device_batch(next(ambient_iter), device)["lr"]
                y, g, kind = build_ambient_target(x2, operator, ambient_mode, rng)
                l_amb, d = ambient_denoise_loss(model, operator, y, g, schedule, kind=kind)
                total = total + lambda_ambient * l_amb
                logs.update(d)

        scaler.scale(total).backward()
        if will_log:
            scaler.unscale_(optimizer)
            logs["grad_norm"] = common.grad_global_norm(model.parameters())
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        logs["loss"] = float(total.detach())
        last = logs
        if not first:
            first = dict(logs)

        if will_log:
            logs.update(common.system_metrics(device, time.perf_counter() - t_step))
            row = {**logs, "lr": lr}
            if do_eval:
                row.update(clean_denoise_metrics(ema.module, val_iter, device, schedule,
                                                 n_batches=1 if smoke else 2))
            logger.log(step, row)
        if save_every > 0 and step % save_every == 0:
            common.save_checkpoint(run_dir / f"ckpt_{step}.pt", model, optimizer, step,
                                   extra={"ema": ema.module.state_dict(), **manifest})

    common.save_checkpoint(run_dir / "ckpt_last.pt", model, optimizer, steps,
                           extra={"ema": ema.module.state_dict(), **manifest})
    logger.close()
    common.finish_wandb()
    return {"run_dir": str(run_dir), "first": first, "last": last, "steps": steps,
            "checkpoint": str(run_dir / "ckpt_last.pt")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Operator-conditioned HR denoiser (Phase C)")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--smoke", action="store_true", help="short CPU smoke run")
    parser.add_argument("--set", nargs="*", default=None,
                        help="dotted config overrides, e.g. train.steps=30")
    args = parser.parse_args(argv)

    from ..utils.config import apply_overrides, load_config

    cfg = apply_overrides(load_config(args.config), args.set)
    result = train(cfg, smoke=args.smoke)
    msg = " ".join(f"{k}={v:.4g}" for k, v in result["last"].items() if isinstance(v, (int, float)))
    print(f"[prior] done: steps={result['steps']} last[{msg}] run_dir={result['run_dir']}")


if __name__ == "__main__":
    main()
