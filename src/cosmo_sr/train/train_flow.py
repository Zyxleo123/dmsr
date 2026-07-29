"""Scale-shared stochastic null-space residual flow training.

Combines two objectives on a single shared velocity network ``v_phi``:

* **Paired flow matching** over adjacent octaves ``(x_R, x_2R)`` for
  ``R in resolutions`` (built as an average-pool pyramid of the HR boxes)::

      base   = B_cons_R(x_R)
      r_star = P_null_R(x_2R - base)
      L_FM   = mse(v_phi(r_t, t, x_R, R), r_star - z)

* **LR-only** (``y_R`` without ``x_2R``): sample a residual with the flow, form
  ``x_hat = B_cons_R(y_R) + P_null_R(r)`` and apply::

      L_deg  = mse(A_R(x_hat), y_R)                 (near-zero; consistency safety)
      L_band = mse(bandpower(r_gen), target_band_R) (matches real residual bands)

Usage::

    python -m cosmo_sr.train.train_flow --config configs/flow_cascade.yaml --smoke
"""
from __future__ import annotations

import os

# Must be set before the CUDA allocator initialises. Reduces fragmentation for
# the large, variably-sized 3D volumes used across octaves (helps avoid
# "reserved but unallocated" OOMs on long-running jobs). No-op if already set.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..data.datasets import (
    FieldCropDataset,
    PyramidCropDataset,
    SyntheticPyramidDataset,
    infinite_loader,
    list_fields,
)
from ..losses.flow import (
    band_power,
    band_statistics_loss,
    degrade_consistency_loss,
    flow_matching_loss,
)
from ..models.flow_unet import UNetResidualFlowModel
from ..models.hblock_flow import HBlockResidualFlowModel
from ..models.residual_flow import ResidualFlowModel
from ..models.unet_baseline import SimpleSRGenerator
from ..operators.base_upscaler import (
    BackboneUpscaler,
    IdentityUpscaler,
    consistent_base,
)
from ..operators.multiscale import MultiScaleOperators
from ..inference.flow_sample import integrate_residual, sample_step
from ..eval.flow_eval import highk_power_ratio, z_diversity
from ..utils.seed import seed_everything
from . import common

import time


def _build_base_upscaler(cfg: Dict[str, Any], channels: int, factor: int, device):
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
    raise ValueError(f"Unknown base_upscaler.kind {kind!r} (use 'identity' or 'backbone')")


def _build_flow_model(model_cfg: Dict[str, Any], channels: int, factor: int) -> torch.nn.Module:
    """Construct the velocity network from ``model:`` config (default keeps old configs working)."""
    name = str(model_cfg.get("name", "ResidualFlowModel"))
    if name == "ResidualFlowModel":
        return ResidualFlowModel(
            channels=channels,
            width=int(model_cfg.get("width", 64)),
            depth=int(model_cfg.get("depth", 4)),
            embed_dim=int(model_cfg.get("embed_dim", 128)),
            context_channels=int(model_cfg.get("context_channels", 0)),
            factor=factor,
            use_checkpoint=bool(model_cfg.get("grad_checkpoint", False)),
        )
    if name == "UNetResidualFlowModel":
        return UNetResidualFlowModel(
            channels=channels,
            width=int(model_cfg.get("width", 64)),
            num_levels=int(model_cfg.get("num_levels", 2)),
            blocks_per_level=int(model_cfg.get("blocks_per_level", 1)),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            padding=str(model_cfg.get("padding", "same")),
            norm=str(model_cfg.get("norm", "group")),
            num_groups=int(model_cfg.get("num_groups", 8)),
            activation=str(model_cfg.get("activation", "silu")),
            use_resblocks=bool(model_cfg.get("use_resblocks", False)),
            use_film=bool(model_cfg.get("use_film", False)),
            embed_dim=int(model_cfg.get("embed_dim", 128)),
            use_attention=bool(model_cfg.get("use_attention", False)),
            attention_heads=int(model_cfg.get("attention_heads", 4)),
            zero_init_tail=bool(model_cfg.get("zero_init_tail", False)),
            use_checkpoint=bool(
                model_cfg.get("use_checkpoint", model_cfg.get("grad_checkpoint", False))
            ),
            context_channels=int(model_cfg.get("context_channels", 0)),
            factor=factor,
            map2map_compat=bool(model_cfg.get("map2map_compat", False)),
        )
    if name == "HBlockResidualFlowModel":
        return HBlockResidualFlowModel(
            channels=channels,
            width=int(model_cfg.get("width", 64)),
            num_levels=int(model_cfg.get("num_levels", 3)),
            embed_dim=int(model_cfg.get("embed_dim", 128)),
            context_channels=int(model_cfg.get("context_channels", 0)),
            factor=factor,
            groups=int(model_cfg.get("groups", 8)),
            zero_init_tail=bool(model_cfg.get("zero_init_tail", True)),
            use_checkpoint=bool(
                model_cfg.get("use_checkpoint", model_cfg.get("grad_checkpoint", False))
            ),
        )
    raise ValueError(
        f"Unknown model.name {name!r} (use 'ResidualFlowModel', 'UNetResidualFlowModel', "
        "or 'HBlockResidualFlowModel')"
    )


def _resolve_lr_streams(cfg: Dict[str, Any], crop_hr: int, n_levels: int) -> List[Dict[str, Any]]:
    """Normalise LR-only source config into a list of stream specs.

    Each stream is ``{"glob", "res", "crop", "pool"}``. Two config styles:

    * new: ``data.lr_only: [{glob, res, crop, pool}, ...]`` (multi-octave), or
    * legacy: ``data.unpaired_lr_glob`` (+ ``loss.lr_only_res``/``data.lr_crop``)
      -> a single stream.
    """
    data = cfg.get("data", {})
    streams = data.get("lr_only")
    if streams:
        out = []
        for s in streams:
            crop = int(s.get("crop", crop_hr // (2 ** (n_levels - 1))))
            out.append({
                "glob": s["glob"],
                "res": int(s["res"]),
                "crop": crop,
                "pool": int(s.get("pool", 1)),
            })
        return out
    legacy_glob = data.get("unpaired_lr_glob")
    if legacy_glob:
        res = int(cfg.get("loss", {}).get("lr_only_res", 64))
        crop = int(data.get("lr_crop", crop_hr // (2 ** (n_levels - 1))))
        return [{"glob": legacy_glob, "res": res, "crop": crop, "pool": 1}]
    return []


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
        paired_ds = SyntheticPyramidDataset(
            num_samples=8, crop_hr=crop_hr, n_levels=n_levels, full_res=full_res, seed=0
        )
        val_ds = SyntheticPyramidDataset(
            num_samples=4, crop_hr=crop_hr, n_levels=n_levels, full_res=full_res, seed=999
        )
    else:
        paired_ds = PyramidCropDataset(
            hr_paths, crop_hr=crop_hr, n_levels=n_levels, full_res=full_res,
            seed=0, mmap=mmap,
        )
        val_hr = list_fields(data.get("val_hr_glob"))
        val_ds = (
            PyramidCropDataset(
                val_hr, crop_hr=crop_hr, n_levels=n_levels, full_res=full_res,
                seed=123, mmap=mmap,
            )
            if val_hr else None
        )

    lr_streams: List[Dict[str, Any]] = []
    if not smoke:
        for i, spec in enumerate(_resolve_lr_streams(cfg, crop_hr, n_levels)):
            paths = list_fields(spec["glob"])
            if not paths:
                continue
            ds = FieldCropDataset(
                paths, None, crop_lr=spec["crop"], scale_factor=1, seed=7 + i, mmap=mmap
            )
            lr_streams.append({"ds": ds, "res": spec["res"], "pool": spec["pool"]})
    return paired_ds, lr_streams, val_ds, crop_hr, n_levels, full_res


@torch.no_grad()
def _val_flow_metrics(model, ops, base_upscaler, val_iter, resolutions,
                      n_steps: int, device, n_batches: int = 2,
                      diversity_samples: int = 2, autocast=None):
    """Per-octave validation for the stochastic cascade.

    Logs, per octave ``R``: flow-matching loss, sampled ``A``-consistency,
    high-k and all-k power ratios, residual-power ratio (gen/true amplitude), and
    ``z``-diversity (the key stochastic-SR metric). Also aggregate means.
    """
    from contextlib import nullcontext

    from ..losses.flow import flow_matching_loss

    if val_iter is None:
        return {}
    if autocast is None:
        autocast = nullcontext
    model.eval()
    agg: Dict[str, float] = {}
    count = 0
    for _ in range(n_batches):
        vb = common.to_device_batch(next(val_iter), device)
        for R in resolutions:
            x_R = vb[f"r{R}"]
            x_2R = vb[f"r{2 * R}"]
            R_t = torch.full((x_R.shape[0],), float(R), device=device)
            with autocast():
                fm, _ = flow_matching_loss(model, ops, base_upscaler, x_R, x_2R, R_t)
                x_hat = sample_step(model, ops, base_upscaler, x_R, float(R), n_steps=n_steps)
            denom = float(torch.mean(x_R ** 2))
            cons = float(torch.mean((ops.A(x_hat) - x_R) ** 2)) / (denom if denom > 0 else 1.0)
            pr = highk_power_ratio(x_hat, x_2R)
            base = consistent_base(base_upscaler, ops, x_R)
            res_gen = ops.P_null(x_hat - base)
            res_true = ops.P_null(x_2R - base)
            vt = float(torch.var(res_true))
            respow = float(torch.var(res_gen)) / (vt if vt > 0 else 1.0)
            with autocast():
                zdiv = z_diversity(model, ops, base_upscaler, x_R, float(R),
                                   n_samples=diversity_samples, n_steps=n_steps)["z_rel_diversity"]
            agg[f"val_fm_R{R}"] = agg.get(f"val_fm_R{R}", 0.0) + float(fm)
            agg[f"val_cons_R{R}"] = agg.get(f"val_cons_R{R}", 0.0) + cons
            agg[f"val_highk_R{R}"] = agg.get(f"val_highk_R{R}", 0.0) + float(pr["highk_power_ratio"])
            agg[f"val_allk_R{R}"] = agg.get(f"val_allk_R{R}", 0.0) + float(pr["allk_power_ratio"])
            agg[f"val_respow_R{R}"] = agg.get(f"val_respow_R{R}", 0.0) + respow
            agg[f"val_zdiv_R{R}"] = agg.get(f"val_zdiv_R{R}", 0.0) + float(zdiv)
        count += 1
    model.train()
    out = {k: v / max(count, 1) for k, v in agg.items()}
    if out:
        for pref, name in [("val_fm_R", "val_fm"), ("val_cons_R", "val_consistency_rel"),
                           ("val_highk_R", "val_highk"), ("val_zdiv_R", "val_zdiv"),
                           ("val_respow_R", "val_respow")]:
            vals = [v for k, v in out.items() if k.startswith(pref)]
            if vals:
                out[name] = float(np.mean(vals))
    return out


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
    lambda_deg = float(loss_cfg.get("lambda_deg", 1.0))
    lambda_band = float(loss_cfg.get("lambda_band", 1.0))
    band_ema = float(loss_cfg.get("band_ema", 0.9))
    n_lr_steps = int(loss_cfg.get("lr_sample_steps", 4))
    amp_enabled = bool(train_cfg.get("amp", False))

    steps = int(train_cfg.get("steps", 20000))
    if smoke:
        steps = min(steps, 12)
    lr = float(train_cfg.get("lr", 1e-4))
    bs = int(train_cfg.get("batch_size", 1))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 1000))
    eval_every = int(train_cfg.get("eval_every", max(log_every, 500)))
    eval_steps = int(loss_cfg.get("eval_sample_steps", 10))
    if smoke:
        eval_every = min(eval_every, 6)

    paired_ds, lr_streams, val_ds, crop_hr, n_levels, full_res = _build_datasets(cfg, smoke)

    ops = MultiScaleOperators(factor=factor).to(device)
    base_upscaler = _build_base_upscaler(cfg, channels, factor, device)
    model = _build_flow_model(model_cfg, channels, factor).to(device)

    params = list(model.parameters()) + list(base_upscaler.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    autocast, scaler = common.amp_components(amp_enabled, device)
    if amp_enabled and device.type != "cuda":
        print("[flow] amp=true ignored on non-CUDA device")

    run_dir = Path(cfg.get("output", {}).get("run_dir", "runs/flow_cascade"))
    common.init_run_dir(run_dir, cfg)
    use_wandb = (not smoke) and common.maybe_init_wandb(cfg, run_dir, "flow")
    logger = common.CSVLogger(run_dir, use_wandb=use_wandb)

    paired_iter = infinite_loader(paired_ds, bs, seed=seed)
    val_iter = infinite_loader(val_ds, bs, seed=seed + 99) if val_ds is not None else None
    for i, s in enumerate(lr_streams):
        s["iter"] = infinite_loader(s["ds"], bs, seed=seed + 5 + i)

    # per-octave EMA of real residual band power (targets for LR-only band loss)
    band_targets: Dict[int, torch.Tensor] = {}
    rng = np.random.default_rng(seed)

    model.train()
    first: Dict[str, float] = {}
    last: Dict[str, float] = {}
    for step in range(1, steps + 1):
        t_step = time.perf_counter()
        do_eval = val_iter is not None and (step % eval_every == 0 or step == steps)
        will_log = step % log_every == 0 or step == 1 or step == steps or do_eval
        logs: Dict[str, float] = {}

        # ---- paired flow matching (random octave per step) ----
        pb = common.to_device_batch(next(paired_iter), device)
        oct_idx = int(rng.integers(0, len(resolutions)))
        R = resolutions[oct_idx]
        # map octave label R to the pyramid keys; finest crop is at full_res
        x_R = pb[f"r{R}"]
        x_2R = pb[f"r{2 * R}"]
        R_tensor = torch.full((x_R.shape[0],), float(R), device=device)
        with autocast():
            loss_fm, r_star = flow_matching_loss(model, ops, base_upscaler, x_R, x_2R, R_tensor)
        with torch.no_grad():
            tgt = band_power(r_star, n_bands=n_bands, log=True)
            if R in band_targets:
                band_targets[R] = band_ema * band_targets[R] + (1 - band_ema) * tgt
            else:
                band_targets[R] = tgt
        loss = loss_fm
        logs["loss_fm"] = float(loss_fm.detach())
        logs[f"loss_fm_R{R}"] = float(loss_fm.detach())  # per-octave train series

        # ---- LR-only (degrade consistency + band statistics), multi-octave ----
        # pick a random LR stream whose octave already has a band target
        ready = [s for s in lr_streams if s["res"] in band_targets]
        if ready:
            s = ready[int(rng.integers(0, len(ready)))]
            Rlr = s["res"]
            lb = common.to_device_batch(next(s["iter"]), device)
            y = lb["lr"]
            if s["pool"] > 1:
                y = torch.nn.functional.avg_pool3d(y, kernel_size=s["pool"], stride=s["pool"])
            with autocast():
                r_gen = integrate_residual(model, ops, y, float(Rlr), n_steps=n_lr_steps)
                x_hat = consistent_base(base_upscaler, ops, y) + r_gen
                loss_deg = degrade_consistency_loss(ops, x_hat, y)
                loss_band = band_statistics_loss(r_gen, band_targets[Rlr], n_bands=n_bands)
            loss = loss + lambda_deg * loss_deg + lambda_band * loss_band
            logs["loss_deg"] = float(loss_deg.detach())
            logs["loss_band"] = float(loss_band.detach())
            logs["lr_only_R"] = float(Rlr)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if will_log:
            scaler.unscale_(optimizer)
            logs["grad_norm"] = common.grad_global_norm(params)
        scaler.step(optimizer)
        scaler.update()

        logs["loss"] = float(loss.detach())
        logs["octave_R"] = float(R)
        last = logs
        if not first:
            first = dict(logs)

        if will_log:
            logs.update(common.system_metrics(device, time.perf_counter() - t_step))
            row = {**logs, "lr": lr}
            if do_eval:
                row.update(_val_flow_metrics(
                    model, ops, base_upscaler, val_iter, resolutions, eval_steps, device,
                    autocast=autocast,
                ))
            logger.log(step, row)
        if save_every > 0 and (step % save_every == 0):
            common.save_checkpoint(
                run_dir / f"ckpt_{step}.pt", model, optimizer, step,
                extra={"base_upscaler": base_upscaler.state_dict()},
            )

    common.save_checkpoint(
        run_dir / "ckpt_last.pt", model, optimizer, steps,
        extra={"base_upscaler": base_upscaler.state_dict(),
               "band_targets": {k: v.cpu() for k, v in band_targets.items()}},
    )
    logger.close()
    common.finish_wandb()
    return {"run_dir": str(run_dir), "first": first, "last": last, "steps": steps,
            "checkpoint": str(run_dir / "ckpt_last.pt")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Residual null-space flow training")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--smoke", action="store_true", help="short CPU smoke run")
    args = parser.parse_args(argv)

    from ..utils.config import load_config

    cfg = load_config(args.config)
    result = train(cfg, smoke=args.smoke)
    msg = " ".join(f"{k}={v:.4g}" for k, v in result["last"].items())
    print(f"[flow] done: steps={result['steps']} last[{msg}] run_dir={result['run_dir']}")


if __name__ == "__main__":
    main()
