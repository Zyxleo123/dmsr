"""Latent conditional flow training (stage 3 of the latent residual flow).

Trains a latent velocity model on AE latents of the null-space HR residuals,
with a *frozen* residual autoencoder and a *frozen* learned degrader. Hard LR
consistency is guaranteed by construction (``P_null`` + consistent base); the
degrader is used only as an auxiliary physical-degrader consistency term.

Per adjacent octave pair::

    base   = B_cons_R(x_R)
    r_star = P_null_R(x_2R - base)
    z1     = ae.encode(r_star)                 # frozen AE, no grad
    z0     = N(0, I); t ~ U(0,1)
    z_t    = (1 - t) z0 + t z1;   v_target = z1 - z0

Classifier-free guidance: with probability ``p_uncond`` the condition ``x_R`` is
replaced by a zero (null) condition.

Losses::

    loss_fm      = mse(v_pred, v_target)
    loss_clean_z = mse(z_t + (1-t) v_pred, z1)
    r_pred       = P_null(ae.decode(z1_pred))
    x_pred       = base + r_pred
    loss_D       = mse(D_phi(x_pred, R), x_R)   # frozen D_phi
    loss_x       = mse(x_pred, x_2R)
    loss_band    = band_statistics_loss(r_pred, bandpower(r_star))
    loss_A_metric= mse(A(x_pred), x_R)          # metric only (lambda_A defaults 0)

Usage::

    python -m cosmo_sr.train.train_latent_flow --config configs/latent_flow.yaml --smoke
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..data.datasets import (
    FieldCropDataset,
    PyramidCropDataset,
    SyntheticPyramidDataset,
    infinite_loader,
    list_fields,
)
from ..losses.flow import band_power, band_statistics_loss
from ..models.residual_autoencoder import ResidualAutoencoder
from ..models.learned_degrader import LearnedDegrader
from ..models.latent_flow import LatentFlowModel
from ..models.unet_baseline import SimpleSRGenerator
from ..operators.base_upscaler import BackboneUpscaler, IdentityUpscaler, consistent_base
from ..operators.multiscale import MultiScaleOperators
from ..inference.latent_flow_sample import (
    latent_shape_from_cond,
    sample_latent_step,
)
from ..eval.flow_eval import highk_power_ratio, sr2_power_summary
from ..utils.seed import seed_everything
from . import common


# --------------------------------------------------------------------------- #
# Builders (shared with eval)
# --------------------------------------------------------------------------- #
def build_ae_from(section: Dict[str, Any]) -> ResidualAutoencoder:
    return ResidualAutoencoder(
        channels=int(section.get("channels", 6)),
        width=int(section.get("width", 32)),
        ch_mults=tuple(section.get("ch_mults", (1, 2, 2))),
        latent_channels=int(section.get("latent_channels", 16)),
        n_res=int(section.get("n_res", 1)),
    )


def build_degrader_from(section: Dict[str, Any], factor: int) -> LearnedDegrader:
    return LearnedDegrader(
        channels=int(section.get("channels", 6)),
        width=int(section.get("width", 32)),
        depth=int(section.get("depth", 2)),
        factor=factor,
        use_res_embed=bool(section.get("use_res_embed", True)),
        embed_dim=int(section.get("embed_dim", 64)),
    )


def build_latent_flow(cfg: Dict[str, Any], latent_channels: int, channels: int) -> LatentFlowModel:
    m = cfg.get("model", {})
    return LatentFlowModel(
        latent_channels=int(m.get("latent_channels", latent_channels)),
        cond_channels=int(m.get("cond_channels", channels)),
        width=int(m.get("width", 64)),
        depth=int(m.get("depth", 4)),
        embed_dim=int(m.get("embed_dim", 128)),
        cond_mode=str(m.get("cond_mode", "trilinear")),
        use_checkpoint=bool(m.get("grad_checkpoint", False)),
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


def load_frozen_ae(cfg: Dict[str, Any], device) -> ResidualAutoencoder:
    ae = build_ae_from(cfg.get("ae", {})).to(device)
    ckpt = cfg.get("ae_checkpoint")
    if ckpt and Path(ckpt).exists():
        common.load_checkpoint(ckpt, ae, map_location=device)
    else:
        if ckpt:
            print(f"[latent_flow] AE checkpoint {ckpt} not found; using fresh (untrained) AE")
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def load_frozen_degrader(cfg: Dict[str, Any], factor: int, device) -> LearnedDegrader:
    deg = build_degrader_from(cfg.get("degrader", {}), factor).to(device)
    ckpt = cfg.get("degrader_checkpoint")
    if ckpt and Path(ckpt).exists():
        common.load_checkpoint(ckpt, deg, map_location=device)
    else:
        if ckpt:
            print(f"[latent_flow] degrader checkpoint {ckpt} not found; using fresh degrader")
    deg.eval()
    for p in deg.parameters():
        p.requires_grad_(False)
    return deg


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def _resolve_lr_streams(cfg: Dict[str, Any], crop_hr: int, n_levels: int):
    streams = cfg.get("data", {}).get("lr_only") or cfg.get("lr_only")
    out = []
    if streams:
        for s in streams:
            crop = int(s.get("crop", crop_hr // (2 ** (n_levels - 1))))
            out.append({"glob": s["glob"], "res": int(s["res"]), "crop": crop,
                        "pool": int(s.get("pool", 1))})
    return out


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
    lr_streams = []
    if not smoke:
        for i, spec in enumerate(_resolve_lr_streams(cfg, crop_hr, n_levels)):
            paths = list_fields(spec["glob"])
            if not paths:
                continue
            ds = FieldCropDataset(paths, None, crop_lr=spec["crop"], scale_factor=1,
                                  seed=7 + i, mmap=mmap)
            lr_streams.append({"ds": ds, "res": spec["res"], "pool": spec["pool"]})
    return paired_ds, lr_streams, val_ds, crop_hr, n_levels, full_res


# --------------------------------------------------------------------------- #
# Grad-enabled conditional latent integration (LR-only branch)
# --------------------------------------------------------------------------- #
def _integrate_latent_cond(model, cond, R, latent_shape, n_steps, device, dtype,
                           bp_steps: Optional[int] = None):
    z = torch.randn(*latent_shape, device=device, dtype=dtype)
    b = z.shape[0]
    dt = 1.0 / n_steps
    cutoff = 0 if bp_steps is None else max(n_steps - bp_steps, 0)
    for i in range(n_steps):
        t = torch.full((b,), i * dt, device=device, dtype=dtype)
        if i < cutoff:
            with torch.no_grad():
                z = z + dt * model(z, t, cond, R)
        else:
            z = z + dt * model(z, t, cond, R)
    return z


# --------------------------------------------------------------------------- #
# Validation (sampled)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _val_latent_metrics(model, ae, deg, ops, base_upscaler, val_iter, resolutions,
                        device, n_steps, cfg_scale, n_batches=2, diversity_samples=2):
    if val_iter is None:
        return {}
    model.eval()
    agg: Dict[str, float] = {}
    count = 0
    for _ in range(n_batches):
        vb = common.to_device_batch(next(val_iter), device)
        for R in resolutions:
            x_R = vb[f"r{R}"]
            x_2R = vb[f"r{2 * R}"]
            x_hat = sample_latent_step(model, ae, ops, base_upscaler, x_R, float(R),
                                       n_steps=n_steps, cfg_scale=cfg_scale)
            denom = float(torch.mean(x_R ** 2)) or 1.0
            cons = float(torch.mean((ops.A(x_hat) - x_R) ** 2)) / denom
            d_out = deg(x_hat, torch.full((x_R.shape[0],), float(R), device=device))
            d_cons = float(torch.mean((d_out - x_R) ** 2)) / denom
            deg_delta = float((d_out - ops.A(x_hat)).pow(2).mean().sqrt())
            x_mse = float(F.mse_loss(x_hat, x_2R))
            pr = highk_power_ratio(x_hat, x_2R)
            base = consistent_base(base_upscaler, ops, x_R)
            res_gen = ops.P_null(x_hat - base)
            res_true = ops.P_null(x_2R - base)
            vt = float(torch.var(res_true)) or 1.0
            respow = float(torch.var(res_gen)) / vt
            cc = sr2_power_summary(x_hat, x_2R)["cross_corr_mean_per_channel"]
            cc = float(np.nanmean(cc)) if len(cc) else float("nan")
            finite = float(torch.isfinite(x_hat).float().mean())
            # z-diversity across seeds
            samples = [sample_latent_step(model, ae, ops, base_upscaler, x_R, float(R),
                                          n_steps=n_steps, cfg_scale=cfg_scale)
                       for _ in range(diversity_samples)]
            stack = torch.stack(samples, dim=0)
            per_voxel_std = stack.std(dim=0)
            signal_std = stack.mean(dim=0).std().clamp_min(1e-12)
            zdiv = float(per_voxel_std.mean() / signal_std)

            def acc(key, val):
                agg[key] = agg.get(key, 0.0) + val

            acc(f"val/consistency_rel_R{R}", cons)
            acc(f"val/D_consistency_rel_R{R}", d_cons)
            acc(f"val/x_mse_R{R}", x_mse)
            acc(f"val/highk_R{R}", float(pr["highk_power_ratio"]))
            acc(f"val/allk_R{R}", float(pr["allk_power_ratio"]))
            acc(f"val/respow_R{R}", respow)
            acc(f"val/zdiv_R{R}", zdiv)
            acc(f"val/cross_corr_R{R}", cc)
            acc(f"val/finite_frac_R{R}", finite)
            acc(f"val/D_gap_R{R}", d_cons - cons)
            acc(f"val/degrader_delta_on_gen_R{R}", deg_delta)
        count += 1
    model.train()
    return {k: v / max(count, 1) for k, v in agg.items()}


def train(cfg: Dict[str, Any], smoke: bool = False) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})
    seed = int(train_cfg.get("seed", 0))
    seed_everything(seed)

    device = common.select_device("cpu" if smoke else train_cfg.get("device"))
    factor = int(cfg.get("factor", 2))
    channels = int(cfg.get("ae", {}).get("channels", 6))
    resolutions: List[int] = list(cfg.get("resolutions", [64, 128, 256]))

    n_bands = int(loss_cfg.get("n_bands", 8))
    lambda_clean = float(loss_cfg.get("lambda_clean", 0.1))
    lambda_D = float(loss_cfg.get("lambda_D", 0.01))
    lambda_x = float(loss_cfg.get("lambda_x", 0.1))
    lambda_band = float(loss_cfg.get("lambda_band", 0.1))
    lambda_A = float(loss_cfg.get("lambda_A", 0.0))
    band_ema = float(loss_cfg.get("band_ema", 0.9))
    p_uncond = float(loss_cfg.get("p_uncond", 0.1))
    n_lr_steps = int(loss_cfg.get("lr_sample_steps", 4))
    lambda_lr_band = float(loss_cfg.get("lambda_lr_band", 0.0))
    lambda_lr_deg = float(loss_cfg.get("lambda_lr_deg", 0.0))
    bp_steps = loss_cfg.get("bp_steps")
    bp_steps = int(bp_steps) if bp_steps is not None else None
    eval_steps = int(loss_cfg.get("eval_sample_steps", 10))
    eval_cfg_scale = float(loss_cfg.get("eval_cfg_scale", 1.0))
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

    paired_ds, lr_streams, val_ds, crop_hr, n_levels, full_res = _build_datasets(cfg, smoke)

    ops = MultiScaleOperators(factor=factor).to(device)
    base_upscaler = build_base_upscaler(cfg, channels, factor, device)
    ae = load_frozen_ae(cfg, device)
    deg = load_frozen_degrader(cfg, factor, device)
    model = build_latent_flow(cfg, ae.latent_channels, channels).to(device)

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    autocast, scaler = common.amp_components(amp_enabled, device)

    run_dir = Path(cfg.get("output", {}).get("run_dir", "runs/latent_flow"))
    common.init_run_dir(run_dir, cfg)
    use_wandb = (not smoke) and common.maybe_init_wandb(cfg, run_dir, "latent_flow")
    logger = common.CSVLogger(run_dir, use_wandb=use_wandb)

    paired_iter = infinite_loader(paired_ds, bs, seed=seed)
    val_iter = infinite_loader(val_ds, bs, seed=seed + 99) if val_ds is not None else None
    for i, s in enumerate(lr_streams):
        s["iter"] = infinite_loader(s["ds"], bs, seed=seed + 5 + i)

    band_targets: Dict[int, torch.Tensor] = {}
    rng = np.random.default_rng(seed)
    drop_total = 0.0
    drop_count = 0.0

    model.train()
    first: Dict[str, float] = {}
    last: Dict[str, float] = {}
    for step in range(1, steps + 1):
        t_step = time.perf_counter()
        do_eval = val_iter is not None and (step % eval_every == 0 or step == steps)
        will_log = step % log_every == 0 or step == 1 or step == steps or do_eval
        logs: Dict[str, float] = {}

        pb = common.to_device_batch(next(paired_iter), device)
        R = resolutions[int(rng.integers(0, len(resolutions)))]
        x_R = pb[f"r{R}"]
        x_2R = pb[f"r{2 * R}"]
        R_t = torch.full((x_R.shape[0],), float(R), device=device)

        base = consistent_base(base_upscaler, ops, x_R)
        r_star = ops.P_null(x_2R - base)
        with torch.no_grad():
            z1 = ae.encode(r_star)
            tgt_band = band_power(r_star, n_bands=n_bands, log=True)
            band_targets[R] = (band_ema * band_targets[R] + (1 - band_ema) * tgt_band
                               if R in band_targets else tgt_band)

        b = z1.shape[0]
        z0 = torch.randn_like(z1)
        t = torch.rand(b, device=device, dtype=z1.dtype)
        t_b = t.view(b, *([1] * (z1.dim() - 1)))
        z_t = (1 - t_b) * z0 + t_b * z1
        v_target = z1 - z0

        # condition dropout for CFG
        drop = (torch.rand(b, device=device) < p_uncond)
        cond = x_R.clone()
        if drop.any():
            cond[drop] = 0.0
        drop_total += float(drop.float().sum())
        drop_count += float(b)

        with autocast():
            v_pred = model(z_t, t, cond, R_t)
            loss_fm = F.mse_loss(v_pred, v_target)
            z1_pred = z_t + (1 - t_b) * v_pred
            loss_clean = F.mse_loss(z1_pred, z1)
            r_pred = ops.P_null(ae.decode(z1_pred))
            x_pred = base + r_pred
            loss_D = F.mse_loss(deg(x_pred, R_t), x_R)
            loss_x = F.mse_loss(x_pred, x_2R)
            loss_band = band_statistics_loss(r_pred, tgt_band, n_bands=n_bands)
            loss_A = F.mse_loss(ops.A(x_pred), x_R)
            loss = (loss_fm + lambda_clean * loss_clean + lambda_D * loss_D
                    + lambda_x * loss_x + lambda_band * loss_band + lambda_A * loss_A)

        logs["flow/loss_fm"] = float(loss_fm.detach())
        logs[f"flow/loss_fm_R{R}"] = float(loss_fm.detach())
        logs["flow/loss_clean_z"] = float(loss_clean.detach())
        logs["flow/loss_D"] = float(loss_D.detach())
        logs["flow/loss_x"] = float(loss_x.detach())
        logs["flow/loss_band"] = float(loss_band.detach())
        logs["flow/loss_A_metric"] = float(loss_A.detach())
        logs["flow/v_pred_norm"] = float(v_pred.detach().pow(2).mean().sqrt())
        logs["flow/v_target_norm"] = float(v_target.detach().pow(2).mean().sqrt())
        logs["flow/z0_std"] = float(z0.std())
        logs["flow/z1_mean"] = float(z1.mean())
        logs["flow/z1_std"] = float(z1.std())
        logs["flow/z1_pred_mse"] = float(loss_clean.detach())
        logs["flow/cond_drop_rate"] = drop_total / max(drop_count, 1.0)

        # ---- LR-only branch (optional) ----
        ready = [s for s in lr_streams if s["res"] in band_targets]
        if ready and (lambda_lr_band > 0 or lambda_lr_deg > 0):
            s = ready[int(rng.integers(0, len(ready)))]
            Rlr = s["res"]
            lb = common.to_device_batch(next(s["iter"]), device)
            y = lb["lr"]
            if s["pool"] > 1:
                y = F.avg_pool3d(y, kernel_size=s["pool"], stride=s["pool"])
            R_lr_t = torch.full((y.shape[0],), float(Rlr), device=device)
            with autocast():
                ls = latent_shape_from_cond(ae, ops, y)
                z1_lr = _integrate_latent_cond(model, y, R_lr_t, ls, n_lr_steps,
                                               device, y.dtype, bp_steps=bp_steps)
                r_gen = ops.P_null(ae.decode(z1_lr))
                x_hat_lr = consistent_base(base_upscaler, ops, y) + r_gen
                loss_band_lr = band_statistics_loss(r_gen, band_targets[Rlr], n_bands=n_bands)
                loss_deg_lr = F.mse_loss(deg(x_hat_lr, R_lr_t), y)
                loss = loss + lambda_lr_band * loss_band_lr + lambda_lr_deg * loss_deg_lr
            logs["lr_only/loss_band"] = float(loss_band_lr.detach())
            logs["lr_only/loss_deg_metric"] = float(loss_deg_lr.detach())
            logs["lr_only/R"] = float(Rlr)
            with torch.no_grad():
                logs[f"lr_only/respow_gen_R{Rlr}"] = float(torch.var(r_gen))

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
                row.update(_val_latent_metrics(
                    model, ae, deg, ops, base_upscaler, val_iter, resolutions,
                    device, eval_steps, eval_cfg_scale,
                ))
            logger.log(step, row)
        if save_every > 0 and (step % save_every == 0):
            common.save_checkpoint(run_dir / f"ckpt_{step}.pt", model, optimizer, step,
                                   extra={"base_upscaler": base_upscaler.state_dict()})

    common.save_checkpoint(run_dir / "ckpt_last.pt", model, optimizer, steps,
                           extra={"base_upscaler": base_upscaler.state_dict(),
                                  "band_targets": {k: v.cpu() for k, v in band_targets.items()}})
    logger.close()
    common.finish_wandb()
    return {"run_dir": str(run_dir), "first": first, "last": last, "steps": steps,
            "checkpoint": str(run_dir / "ckpt_last.pt"), "ae": ae, "degrader": deg}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Latent conditional flow training")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--smoke", action="store_true", help="short CPU smoke run")
    parser.add_argument("--set", nargs="*", default=None,
                        help="dotted config overrides, e.g. train.steps=30")
    args = parser.parse_args(argv)
    from ..utils.config import apply_overrides, load_config

    cfg = apply_overrides(load_config(args.config), args.set)
    result = train(cfg, smoke=args.smoke)
    msg = " ".join(f"{k}={v:.4g}" for k, v in result["last"].items()
                   if isinstance(v, (int, float)))
    print(f"[latent_flow] done: steps={result['steps']} last[{msg}] run_dir={result['run_dir']}")


if __name__ == "__main__":
    main()
