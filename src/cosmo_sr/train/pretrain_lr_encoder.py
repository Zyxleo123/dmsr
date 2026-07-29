"""Stage-B prerequisite: masked-reconstruction pretraining of the LR encoder.

Trains :class:`~cosmo_sr.dmsr.encoder.LRMaskedAutoencoder` on **all training LR
boxes** -- the paired ones plus the ~350 LR-only ones -- and saves the encoder
weights separately so ``train_dmsr.py`` can load them with
``model.condition_encoder_init: lr_pretrained``.

Validation and test boxes are excluded by :func:`resolve_split`, which raises if
``lr_only_glob`` matches any held-out box. The resolved manifest is written to
``<run_dir>/split.json`` so the exclusion is auditable after the fact.

Usage::

    python -m cosmo_sr.train.pretrain_lr_encoder --config configs/dmsr/lr_ssl.yaml
    python -m cosmo_sr.train.pretrain_lr_encoder --config configs/dmsr/lr_ssl.yaml --smoke
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import torch

from ..data.datasets import FieldCropDataset, infinite_loader
from ..dmsr.data import resolve_split
from ..dmsr.encoder import LRMaskedAutoencoder
from ..dmsr.ssl import masked_reconstruction_loss
from ..utils.config import apply_overrides, load_config
from ..utils.seed import seed_everything
from . import common


def build_dataset(cfg: Dict[str, Any], split) -> FieldCropDataset:
    """All training LR boxes: paired-train LR + the unpaired pool."""
    dcfg = cfg.get("data", {})
    paths = list(split.train_lr) + list(split.lr_only)
    if not paths:
        raise ValueError("no LR boxes available for pretraining")
    use_channels = dcfg.get("use_channels")
    return FieldCropDataset(
        lr_paths=paths,
        hr_paths=None,
        crop_lr=int(dcfg.get("crop_lr", 16)),
        scale_factor=int(cfg.get("factor", 8)),
        seed=int(cfg.get("train", {}).get("seed", 0)),
        augment=False,   # SSL applies its own rotations/translations
        channels=int(dcfg.get("channels", 6)),
        use_channels=use_channels,
        mmap=bool(dcfg.get("mmap", True)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=None, help="dotted.key=value overrides")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic run")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    tcfg = dict(cfg.get("train", {}))
    scfg = dict(cfg.get("ssl", {}))
    if args.smoke:
        tcfg["steps"] = int(tcfg.get("smoke_steps", 5))
        tcfg["batch_size"] = 2
        cfg.setdefault("wandb", {})["mode"] = "disabled"

    seed_everything(int(tcfg.get("seed", 0)))
    device = common.select_device(cfg.get("device"))
    run_dir = common.init_run_dir(cfg.get("output", {}).get("run_dir", "runs/dmsr_lr_ssl"), cfg)

    split = resolve_split(cfg.get("data", {}))
    split.save(Path(run_dir) / "split.json")
    n_boxes = len(split.train_lr) + len(split.lr_only)
    print(f"[ssl] pretraining on {n_boxes} LR boxes "
          f"({len(split.train_lr)} paired-train + {len(split.lr_only)} LR-only); "
          f"excluded {len(split.val_hr)} val + {len(split.test_hr)} test boxes")

    dcfg = cfg.get("data", {})
    use_channels = dcfg.get("use_channels")
    channels = len(use_channels) if use_channels else int(dcfg.get("channels", 6))

    model = LRMaskedAutoencoder(
        in_channels=channels,
        width=int(cfg.get("model", {}).get("encoder_width", 64)),
        cond_channels=int(cfg.get("model", {}).get("cond_channels", 32)),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg.get("lr", 2e-4)))

    dataset = build_dataset(cfg, split)
    batch_size = int(tcfg.get("batch_size", 8))
    loader = infinite_loader(dataset, batch_size, seed=int(tcfg.get("seed", 0)),
                             num_workers=int(tcfg.get("num_workers", 0)))

    use_wandb = common.maybe_init_wandb(cfg, run_dir, job_type="pretrain")
    logger = common.CSVLogger(run_dir, use_wandb=use_wandb)
    gen = torch.Generator().manual_seed(int(tcfg.get("seed", 0)))

    steps = int(tcfg.get("steps", 20000))
    log_every = int(tcfg.get("log_every", 50))
    save_every = int(tcfg.get("save_every", 2000))
    t0 = time.time()

    for step in range(1, steps + 1):
        batch = common.to_device_batch(next(loader), device)
        loss, metrics = masked_reconstruction_loss(
            model,
            batch["lr"],
            block_size=int(scfg.get("block_size", 2)),
            mask_ratio=float(scfg.get("mask_ratio", 0.5)),
            channel_mask_p=float(scfg.get("channel_mask_p", 0.15)),
            translate=bool(scfg.get("translate", True)),
            rotate=bool(scfg.get("rotate", True)),
            lambda_fourier=float(scfg.get("lambda_fourier", 0.0)),
            generator=gen,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = common.grad_global_norm(model.parameters())
        opt.step()

        if step % log_every == 0 or step == 1:
            row = {"grad_norm": grad_norm, **metrics,
                   **common.system_metrics(device, (time.time() - t0) / step)}
            logger.log(step, row)
            print(f"[ssl] step {step}/{steps} loss={metrics['ssl_loss']:.5f}")

        if step % save_every == 0 or step == steps:
            model.save_encoder(Path(run_dir) / "encoder.pt")
            common.save_checkpoint(Path(run_dir) / "ckpt_last.pt", model, opt, step=step)

    model.save_encoder(Path(run_dir) / "encoder.pt")
    print(f"[ssl] wrote {Path(run_dir) / 'encoder.pt'}")
    common.finish_wandb()


if __name__ == "__main__":
    main()
