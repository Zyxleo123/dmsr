"""Learned degrader training (stage 2 of the latent residual flow).

Trains ``D_phi(x_2R, R) = A_R(x_2R) + Delta_phi(x_2R, R)`` to predict the coarse
field ``x_R`` from ``x_2R``. Because the training pairs here are built by exact
average pooling (``x_R = A_R(x_2R)``), ``A_R`` is already the optimal target and
``D_phi`` should stay close to it (``delta_rms`` small). This is expected and
acceptable -- the purpose is to prove the machinery is trainable, loadable and
freezeable, not to beat ``A`` on pyramid-pooled data.

    y_pred     = D_phi(x_2R, R)
    delta      = y_pred - A_R(x_2R)
    loss_mse   = mse(y_pred, x_R)
    loss_delta = mean(delta ** 2)
    loss       = loss_mse + lambda_delta * loss_delta

Usage::

    python -m cosmo_sr.train.train_degrader --config configs/learned_degrader.yaml --smoke
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

from ..data.datasets import (
    FieldCropDataset,
    GridCropDataset,
    PyramidCropDataset,
    SyntheticPyramidDataset,
    SyntheticSRDataset,
    finite_loader,
    infinite_loader,
    list_fields,
)
from ..eval.density import density_metrics
from ..models.learned_degrader import LearnedDegrader
from ..operators.multiscale import MultiScaleOperators
from ..utils.seed import seed_everything
from . import common


def build_degrader(cfg: Dict[str, Any]) -> LearnedDegrader:
    m = cfg.get("model", {})
    return LearnedDegrader(
        channels=int(m.get("channels", 6)),
        width=int(m.get("width", 32)),
        depth=int(m.get("depth", 2)),
        factor=int(cfg.get("factor", 2)),
        use_res_embed=bool(m.get("use_res_embed", True)),
        embed_dim=int(m.get("embed_dim", 64)),
    )


def _build_datasets(cfg: Dict[str, Any], smoke: bool):
    """Build train/val datasets for degrader training.

    ``data.mode: pyramid`` (default) derives every "LR" level from the HR field
    via exact ``avg_pool3d`` (``PyramidCropDataset``) -- fine for exercising the
    training machinery, but the target ``A_R`` is then trivially achievable
    since it's the same operator the data was built with.

    ``data.mode: real_pair`` instead loads genuinely independent LR/HR fields
    (separate N-body runs, not one derived from the other) via
    ``FieldCropDataset``, so ``D_phi`` has to learn an actual, non-trivial
    forward/degradation operator rather than reproduce ``avg_pool3d``.
    """
    data = cfg.get("data", {})
    mode = str(data.get("mode", "pyramid"))
    if mode == "real_pair":
        return _build_real_pair_datasets(cfg, smoke)

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


def _build_real_pair_datasets(cfg: Dict[str, Any], smoke: bool):
    data = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    # `field_channels` is what's on disk (6); `use_channels` picks the subset to
    # train on -- [0, 1, 2] for a displacement-only run. model.channels must
    # match the selection.
    channels = int(data.get("field_channels", 6))
    use_channels = data.get("use_channels")
    n_model_ch = len(use_channels) if use_channels else channels
    if int(model_cfg.get("channels", 6)) != n_model_ch:
        raise ValueError(
            f"model.channels={model_cfg.get('channels')} does not match the "
            f"{n_model_ch} channel(s) selected by data.use_channels={use_channels}"
        )
    factor = int(cfg.get("factor", 2))
    crop_lr = int(data.get("crop_lr", 16))
    mmap = bool(data.get("mmap", False))

    lr_paths = list_fields(data.get("paired_lr_glob"))
    hr_paths = list_fields(data.get("paired_hr_glob"))
    if not smoke and (not lr_paths or not hr_paths):
        # Do NOT quietly fall through to the synthetic fixture here. That fixture
        # has lr == A(hr) *exactly*, so a typo'd glob would train against a
        # zero-residual target, converge beautifully, and mean nothing -- the
        # exact failure this whole investigation was about. Fail loudly instead;
        # --smoke is the only way to ask for the synthetic path.
        raise FileNotFoundError(
            "data.mode=real_pair matched no files on disk:\n"
            f"  paired_lr_glob={data.get('paired_lr_glob')!r} -> {len(lr_paths)} file(s)\n"
            f"  paired_hr_glob={data.get('paired_hr_glob')!r} -> {len(hr_paths)} file(s)\n"
            "Pass --smoke to use the synthetic fixture on purpose."
        )
    if smoke:
        # Smoke-only fallback to exercise the real_pair code path without real
        # data on disk; ``lr == A(hr)`` here so it does NOT test non-trivial
        # degradation learning, only that the plumbing runs.
        crop_lr = min(crop_lr, 8)
        paired_ds = SyntheticSRDataset(num_samples=8, crop_lr=crop_lr, scale_factor=factor,
                                       channels=n_model_ch, seed=0)
        val_ds = SyntheticSRDataset(num_samples=4, crop_lr=crop_lr, scale_factor=factor,
                                    channels=n_model_ch, seed=999)
        return paired_ds, val_ds, crop_lr, None, None

    augment = bool(data.get("augment", False))
    fixed_crops = data.get("fixed_crops")
    fixed_crops = int(fixed_crops) if fixed_crops is not None else None
    paired_ds = FieldCropDataset(
        lr_paths, hr_paths, crop_lr=crop_lr, scale_factor=factor,
        seed=0, channels=channels, mmap=mmap, augment=augment,
        fixed_crops=fixed_crops, use_channels=use_channels,
    )
    val_lr = list_fields(data.get("val_lr_glob"))
    val_hr = list_fields(data.get("val_hr_glob"))
    if not (val_lr and val_hr):
        return paired_ds, None, crop_lr, None, None

    if fixed_crops is not None:
        # Memorization check: val must draw the *same* closed crop set as train
        # (seed 0, aug off) so both losses are expected to collapse together.
        val_ds = FieldCropDataset(
            val_lr, val_hr, crop_lr=crop_lr, scale_factor=factor,
            seed=0, channels=channels, mmap=mmap, fixed_crops=fixed_crops,
            use_channels=use_channels,
        )
    else:
        # Deterministic, exhaustive crop grid -- never random crops. See
        # `_val_degrader_metrics_real_pair` for why the old 2-random-crop
        # estimate made val useless.
        val_ds = GridCropDataset(
            val_lr, val_hr, crop_lr=crop_lr, scale_factor=factor,
            channels=channels, mmap=mmap,
            stride=data.get("val_stride"),
            max_crops=data.get("val_max_crops"),
            use_channels=use_channels,
        )
    return paired_ds, val_ds, crop_lr, None, None


def density_cfg(cfg: Dict[str, Any]):
    """``(cellsize, dis_norm, n_bands)`` for the Eulerian density eval, or None.

    Enabled by ``eval.density: true``. Requires the first 3 channels to be
    displacement (always true for our layout).
    """
    ev = cfg.get("eval", {})
    if not bool(ev.get("density", False)):
        return None
    data = cfg.get("data", {})
    boxsize = float(ev.get("boxsize", 100000.0))          # kpc/h
    lr_grid = float(data.get("lr_grid_res", 64))
    return (boxsize / lr_grid,                            # Lagrangian cellsize
            float(ev.get("dis_norm", 6000.0)),            # kpc/h per norm unit
            int(ev.get("n_bands", 6)))


# Canonical channel layout is disp[0:3] + vel[3:6]. The two groups behave
# completely differently against the A baseline -- on the real paired boxes
# avg-pool already explains 99.2% of the displacement variance but only ~65% of
# the velocity variance, so velocity carries ~99% of the raw MSE and a single
# summed number tells you nothing about displacement. Always report both.
def channel_groups(channels: int):
    if channels == 6:
        return (("disp", slice(0, 3)), ("vel", slice(3, 6)))
    if channels == 3:
        return (("disp", slice(0, 3)),)   # displacement-only runs
    return ()


def degrader_metrics(deg, ops, x_R, x_2R, R_tensor, channel_weights=None):
    """Losses + diagnostics for one batch.

    ``channel_weights`` (a ``(C,)`` tensor, or ``None``) reweights the MSE per
    channel. Unweighted, the loss is ~99% velocity and the model is effectively
    not trained on displacement at all.

    The headline diagnostic is ``skill = 1 - mse / mse_vs_A_baseline``: the
    fraction of the A-baseline error the model actually removes. Raw ``mse`` is
    dominated by which crop was drawn (per-crop baseline difficulty spans 0.05
    to 0.53 on this data), so raw-loss curves compare crop sequences, not
    models -- ``skill`` divides that shared difficulty term out.
    """
    y_pred = deg(x_2R, R_tensor)
    a_base = ops.A(x_2R)
    delta = y_pred - a_base
    se = (y_pred - x_R) ** 2
    if channel_weights is None:
        loss_mse = se.mean()
    else:
        w = channel_weights.to(se.device).view(1, -1, 1, 1, 1)
        loss_mse = (se * w).mean()
    loss_delta = torch.mean(delta ** 2)

    with torch.no_grad():
        se_base = (a_base - x_R) ** 2
        mse = float(se.mean())
        mse_base = float(se_base.mean())
        a_rms = float(a_base.pow(2).mean().sqrt()) or 1.0
        stats = {
            "mse": mse,
            "mse_vs_A_baseline": mse_base,
            "skill": 1.0 - mse / mse_base if mse_base > 0 else 0.0,
            "delta_rms": float(delta.pow(2).mean().sqrt()),
            "delta_to_A_ratio": float(delta.pow(2).mean().sqrt()) / a_rms,
            "output_mean": float(y_pred.mean()),
            "output_std": float(y_pred.std()),
        }
        for name, sl in channel_groups(x_R.shape[1]):
            g_mse = float(se[:, sl].mean())
            g_base = float(se_base[:, sl].mean())
            stats[f"mse_{name}"] = g_mse
            stats[f"mse_vs_A_baseline_{name}"] = g_base
            stats[f"skill_{name}"] = 1.0 - g_mse / g_base if g_base > 0 else 0.0
            # std ratio: an MSE-trained model shrinks toward the conditional
            # mean, so this drifts below 1 exactly when the output is
            # over-smoothed relative to the true LR field.
            stats[f"std_ratio_{name}"] = (
                float(y_pred[:, sl].std()) / (float(x_R[:, sl].std()) or 1.0)
            )
    return loss_mse, loss_delta, stats


@torch.no_grad()
def _val_degrader_metrics(deg, ops, val_iter, resolutions, device, n_batches: int = 2):
    if val_iter is None:
        return {}
    deg.eval()
    agg: Dict[str, float] = {}
    count = 0
    for _ in range(n_batches):
        vb = common.to_device_batch(next(val_iter), device)
        for R in resolutions:
            x_R = vb[f"r{R}"]
            x_2R = vb[f"r{2 * R}"]
            R_t = torch.full((x_R.shape[0],), float(R), device=device)
            _, _, s = degrader_metrics(deg, ops, x_R, x_2R, R_t)
            agg[f"val/degrader/mse_R{R}"] = agg.get(f"val/degrader/mse_R{R}", 0.0) + s["mse"]
            agg[f"val/degrader/mse_vs_A_baseline_R{R}"] = agg.get(f"val/degrader/mse_vs_A_baseline_R{R}", 0.0) + s["mse_vs_A_baseline"]
            agg[f"val/degrader/delta_rms_R{R}"] = agg.get(f"val/degrader/delta_rms_R{R}", 0.0) + s["delta_rms"]
            agg[f"val/degrader/delta_to_A_ratio_R{R}"] = agg.get(f"val/degrader/delta_to_A_ratio_R{R}", 0.0) + s["delta_to_A_ratio"]
        count += 1
    deg.train()
    return {k: v / max(count, 1) for k, v in agg.items()}


@torch.no_grad()
def _val_degrader_metrics_real_pair(deg, ops, val_loader, R, device, dens=None):
    """Full deterministic pass over the val crop grid.

    This used to average 2 random crops at batch_size 1. Per-crop MSE varies by
    ~10x here, so that estimate was almost entirely sampling noise -- three
    different model sizes all reported best_val ~0.0347 because they had each
    drawn the same lucky crop, not because they performed the same. We now sum
    squared error over *every* crop of a fixed grid and divide once, so the
    number is an exact dataset MSE and is comparable across steps and runs.
    """
    if val_loader is None:
        return {}
    deg.eval()
    groups = None
    se_sum = 0.0
    base_sum = 0.0
    delta_sq_sum = 0.0
    a_sq_sum = 0.0
    g_se: Dict[str, float] = {}
    g_base: Dict[str, float] = {}
    n_vox = 0
    dens_acc: Dict[str, float] = {}
    dens_n = 0
    for vb in val_loader:
        vb = common.to_device_batch(vb, device)
        x_R, x_2R = vb["lr"], vb["hr"]
        if groups is None:
            groups = channel_groups(x_R.shape[1])
        R_t = torch.full((x_R.shape[0],), float(R), device=device)
        y_pred = deg(x_2R, R_t)
        a_base = ops.A(x_2R)
        se = (y_pred - x_R) ** 2
        se_base = (a_base - x_R) ** 2

        if dens is not None:
            cellsize, dis_norm, n_bands = dens
            dm = density_metrics(y_pred[:, 0:3], x_R[:, 0:3], a_base[:, 0:3],
                                 cellsize, dis_norm, n_bands, prefix="val/density/")
            for k, v in dm.items():
                dens_acc[k] = dens_acc.get(k, 0.0) + v
            dens_n += 1
        # accumulate sums (not means-of-means): the last batch may be short
        k = se.numel()
        n_vox += k
        se_sum += float(se.sum())
        base_sum += float(se_base.sum())
        delta_sq_sum += float((y_pred - a_base).pow(2).sum())
        a_sq_sum += float(a_base.pow(2).sum())
        for name, sl in groups:
            g_se[name] = g_se.get(name, 0.0) + float(se[:, sl].sum())
            g_base[name] = g_base.get(name, 0.0) + float(se_base[:, sl].sum())
    deg.train()
    if n_vox == 0:
        return {}
    mse = se_sum / n_vox
    mse_base = base_sum / n_vox
    out = {
        "val/degrader/mse": mse,
        "val/degrader/mse_vs_A_baseline": mse_base,
        "val/degrader/skill": 1.0 - mse / mse_base if mse_base > 0 else 0.0,
        "val/degrader/delta_rms": (delta_sq_sum / n_vox) ** 0.5,
        "val/degrader/delta_to_A_ratio": (
            (delta_sq_sum / n_vox) ** 0.5 / (((a_sq_sum / n_vox) ** 0.5) or 1.0)
        ),
    }
    for name, _sl in groups or ():
        # each group is 3 of C channels -> same voxel count share
        share = n_vox * 3 // (x_R.shape[1])
        m, b = g_se[name] / share, g_base[name] / share
        out[f"val/degrader/mse_{name}"] = m
        out[f"val/degrader/mse_vs_A_baseline_{name}"] = b
        out[f"val/degrader/skill_{name}"] = 1.0 - m / b if b > 0 else 0.0
    for k, v in dens_acc.items():
        out[k] = v / max(dens_n, 1)
    return out


def train(cfg: Dict[str, Any], smoke: bool = False) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})
    model_cfg = cfg.get("model", {})
    seed = int(train_cfg.get("seed", 0))
    seed_everything(seed)

    device = common.select_device("cpu" if smoke else train_cfg.get("device"))
    factor = int(cfg.get("factor", 2))
    data_cfg = cfg.get("data", {})
    mode = str(data_cfg.get("mode", "pyramid"))
    resolutions: List[int] = list(cfg.get("resolutions", [64, 128, 256]))
    real_pair_R = float(data_cfg.get("lr_grid_res", resolutions[0] if resolutions else 64))
    lambda_delta = float(loss_cfg.get("lambda_delta", 1e-3))
    amp_enabled = bool(train_cfg.get("amp", False))

    steps = int(train_cfg.get("steps", 20000))
    if smoke:
        steps = min(steps, 12)
    lr = float(train_cfg.get("lr", 1e-4))
    bs = int(train_cfg.get("batch_size", 1))
    accum_steps = max(1, int(train_cfg.get("accum_steps", 1)))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    warmup = int(train_cfg.get("warmup", 0))
    min_lr_frac = float(train_cfg.get("min_lr_frac", 1.0))  # 1.0 => constant LR
    normalize_channels = bool(loss_cfg.get("normalize_channels", False))
    num_workers = int(train_cfg.get("num_workers", 0))
    if smoke:
        num_workers = 0
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 1000))
    eval_every = int(train_cfg.get("eval_every", max(log_every, 500)))
    if smoke:
        eval_every = min(eval_every, 6)
        accum_steps = min(accum_steps, 2)
    early_stop_patience = int(train_cfg.get("early_stop_patience", 0))

    paired_ds, val_ds, crop_hr, n_levels, full_res = _build_datasets(cfg, smoke)
    dens = density_cfg(cfg) if mode == "real_pair" else None
    ops = MultiScaleOperators(factor=factor).to(device)
    deg = build_degrader(cfg).to(device)

    params = list(deg.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    autocast, scaler = common.amp_components(amp_enabled, device)
    sched = common.build_lr_schedule(optimizer, lr, steps, warmup, min_lr_frac)

    run_dir = Path(cfg.get("output", {}).get("run_dir", "runs/learned_degrader"))
    common.init_run_dir(run_dir, cfg)
    use_wandb = (not smoke) and common.maybe_init_wandb(cfg, run_dir, "degrader")
    logger = common.CSVLogger(run_dir, use_wandb=use_wandb)

    paired_iter = infinite_loader(paired_ds, bs, seed=seed, num_workers=num_workers)
    val_loader = (finite_loader(val_ds, bs, num_workers=num_workers)
                  if val_ds is not None else None)
    val_iter = (infinite_loader(val_ds, bs, seed=seed + 99)
                if val_ds is not None and mode != "real_pair" else None)
    rng = np.random.default_rng(seed)

    # Per-channel loss weights. Unweighted, ~99% of the MSE is velocity and the
    # displacement channels are effectively untrained; weighting by the inverse
    # residual variance makes every channel contribute comparably.
    channel_weights = None
    if normalize_channels and mode == "real_pair":
        res_std = common.estimate_residual_std(paired_iter, ops, device,
                                               n_batches=2 if smoke else 32)
        channel_weights = (1.0 / res_std.pow(2))
        channel_weights = channel_weights / channel_weights.mean()  # mean weight 1
        print(f"[degrader] residual std per channel: "
              f"{[round(float(v), 4) for v in res_std]}")
        print(f"[degrader] channel weights:          "
              f"{[round(float(v), 3) for v in channel_weights]}")

    deg.train()
    first: Dict[str, float] = {}
    last: Dict[str, float] = {}
    best_val = float("inf")
    best_step = -1
    evals_since_improve = 0
    final_step = 0
    skill_ema = None
    for step in range(1, steps + 1):
        final_step = step
        t_step = time.perf_counter()
        do_eval = val_ds is not None and (step % eval_every == 0 or step == steps)
        will_log = step % log_every == 0 or step == 1 or step == steps or do_eval

        # Gradient accumulation: the per-crop loss varies ~10x here, so a single
        # crop's gradient is dominated by whatever rare dense crop came up.
        # accum_steps * batch_size is the effective batch.
        optimizer.zero_grad(set_to_none=True)
        acc_stats: Dict[str, float] = {}
        acc_loss = 0.0
        for _ in range(accum_steps):
            pb = common.to_device_batch(next(paired_iter), device)
            if mode == "real_pair":
                x_R, x_2R = pb["lr"], pb["hr"]
                R = real_pair_R
            else:
                R = resolutions[int(rng.integers(0, len(resolutions)))]
                x_R = pb[f"r{R}"]
                x_2R = pb[f"r{2 * R}"]
            R_t = torch.full((x_R.shape[0],), float(R), device=device)
            with autocast():
                loss_mse, loss_delta, stats = degrader_metrics(
                    deg, ops, x_R, x_2R, R_t, channel_weights
                )
                loss = (loss_mse + lambda_delta * loss_delta) / accum_steps
            scaler.scale(loss).backward()
            acc_loss += float(loss.detach()) * accum_steps
            for k, v in stats.items():
                acc_stats[k] = acc_stats.get(k, 0.0) + v / accum_steps

        scaler.unscale_(optimizer)
        grad_norm = common.grad_global_norm(params)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if sched is not None:
            sched.step()

        stats = acc_stats
        # Skill (1 - mse/A_baseline) is the only model-dependent number in the
        # per-step logs; EMA it so the curve is readable instead of tracking
        # per-crop difficulty. Compare THIS across model sizes, not `loss`.
        skill_ema = (stats["skill"] if skill_ema is None
                     else 0.98 * skill_ema + 0.02 * stats["skill"])

        logs: Dict[str, float] = {"grad_norm": grad_norm}
        logs["loss"] = acc_loss
        logs["octave_R"] = float(R)
        logs["degrader/loss"] = acc_loss
        logs["degrader/skill_ema"] = skill_ema
        for k, v in stats.items():
            logs[f"degrader/{k}"] = v
        last = logs
        if not first:
            first = dict(logs)

        val_loss = None
        if will_log:
            logs.update(common.system_metrics(device, time.perf_counter() - t_step))
            row = {**logs, "lr": optimizer.param_groups[0]["lr"]}
            if do_eval:
                if mode == "real_pair":
                    val_metrics = _val_degrader_metrics_real_pair(
                        deg, ops, val_loader, real_pair_R, device, dens
                    )
                    val_loss = val_metrics.get("val/degrader/mse")
                else:
                    val_metrics = _val_degrader_metrics(deg, ops, val_iter, resolutions, device)
                    r_losses = [val_metrics[k] for k in
                                (f"val/degrader/mse_R{R}" for R in resolutions) if k in val_metrics]
                    val_loss = sum(r_losses) / len(r_losses) if r_losses else None
                row.update(val_metrics)
                if val_loss is not None:
                    if val_loss < best_val:
                        best_val, best_step, evals_since_improve = val_loss, step, 0
                        common.save_checkpoint(run_dir / "ckpt_best.pt", deg, optimizer, step,
                                               extra={"val_loss": best_val})
                    else:
                        evals_since_improve += 1
                    row["val/best_loss"] = best_val
            logger.log(step, row)
            # Console line -> slurm-*.out. Note `loss` is NOT the thing to watch
            # (it is ~95% per-crop difficulty); skill is.
            msg = (f"[degrader] step {step:>6}/{steps}  loss {acc_loss:.4f}  "
                   f"skill_ema {skill_ema:+.4f}  lr {optimizer.param_groups[0]['lr']:.2e}  "
                   f"gnorm {grad_norm:.3f}")
            if do_eval and val_loss is not None:
                msg += (f"  | val_mse {val_loss:.5f} "
                        f"(A {row['val/degrader/mse_vs_A_baseline']:.5f}) "
                        f"val_skill {row['val/degrader/skill']:+.4f}")
                for g in ("disp", "vel"):
                    k = f"val/degrader/skill_{g}"
                    if k in row:
                        msg += f" {g} {row[k]:+.4f}"
                sr = row.get("val/density/sigma_ratio")
                if sr is not None:
                    msg += (f"  | sigma_ratio {sr:.3f} "
                            f"(A: {row['val/density/sigma_ratio_A']:.3f})")
            print(msg, flush=True)
        if save_every > 0 and (step % save_every == 0):
            common.save_checkpoint(run_dir / f"ckpt_{step}.pt", deg, optimizer, step)
        if (early_stop_patience > 0 and val_loss is not None
                and evals_since_improve >= early_stop_patience):
            print(f"[degrader] early stop at step {step}: no val/degrader/mse improvement in "
                  f"{early_stop_patience} eval windows (best={best_val:.6g} @ step {best_step})")
            break

    common.save_checkpoint(run_dir / "ckpt_last.pt", deg, optimizer, final_step)
    logger.close()
    common.finish_wandb()
    return {"run_dir": str(run_dir), "first": first, "last": last, "steps": final_step,
            "best_val": best_val if best_step >= 0 else None, "best_step": best_step,
            "checkpoint": str(run_dir / "ckpt_last.pt"),
            "best_checkpoint": str(run_dir / "ckpt_best.pt") if best_step >= 0 else None}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Learned degrader training")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--smoke", action="store_true", help="short CPU smoke run")
    parser.add_argument("--set", nargs="*", default=None,
                        help="dotted config overrides, e.g. train.steps=30")
    args = parser.parse_args(argv)
    from ..utils.config import apply_overrides, load_config

    cfg = apply_overrides(load_config(args.config), args.set)
    result = train(cfg, smoke=args.smoke)
    msg = " ".join(f"{k}={v:.4g}" for k, v in result["last"].items())
    print(f"[degrader] done: steps={result['steps']} last[{msg}] run_dir={result['run_dir']}")


if __name__ == "__main__":
    main()
