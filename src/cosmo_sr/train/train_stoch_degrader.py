"""Stochastic degrader training: conditional flow matching for ``p(x_R | x_2R)``.

The MSE-trained :class:`~cosmo_sr.models.learned_degrader.LearnedDegrader` can
only fit ``E[x_R | x_2R]``, and on real paired N-body runs that conditional mean
is nearly ``A`` itself -- so its loss plateaus at the conditional *variance*, and
its output is systematically short of velocity power. See
``models/stochastic_degrader.py`` for the measurements.

Here we fit the distribution instead, with the same linear-interpolant flow
matching objective already used by the residual flow (``losses/flow.py``):

    r      = (x_R - A_R(x_2R)) / sigma
    r_t    = (1 - t) z + t r,      z ~ N(0, I),  t ~ U[0, 1]
    loss   = mse( v_theta(r_t, t, x_2R, R),  r - z )

The number to watch is **not** the training loss (flow-matching loss is not
comparable across models in absolute terms). Watch:

  * ``val/stoch/pk_ratio_vel_b*`` -- sampled velocity power / true power, per
    k-band. This is the whole point: it should sit near 1 where the A baseline
    (``pk_ratio_A_vel_b*``) sits well below 1 at high k.
  * ``val/stoch/mse_mean_K`` -- MSE of the K-sample conditional mean. Should
    land near the deterministic degrader's MSE, confirming we lost nothing.
  * ``val/stoch/mse_sample`` -- MSE of a *single* sample. Expected to be roughly
    2x the conditional-mean MSE (you pay the conditional variance twice); that
    is correct behaviour for a calibrated sampler, not a regression.

Usage::

    python -m cosmo_sr.train.train_stoch_degrader --config configs/stoch_degrader_real.yaml --smoke
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..data.datasets import finite_loader, infinite_loader
from ..eval.density import density_metrics
from ..losses.flow import band_power, sample_flow_pair
from ..models.stochastic_degrader import StochasticDegrader
from ..operators.multiscale import MultiScaleOperators
from ..utils.seed import seed_everything
from . import common
from .train_degrader import _build_real_pair_datasets, channel_groups, density_cfg


def build_stoch_degrader(cfg: Dict[str, Any]) -> StochasticDegrader:
    m = cfg.get("model", {})
    return StochasticDegrader(
        channels=int(m.get("channels", 6)),
        width=int(m.get("width", 64)),
        depth=int(m.get("depth", 4)),
        factor=int(cfg.get("factor", 8)),
        use_res_embed=bool(m.get("use_res_embed", True)),
        embed_dim=int(m.get("embed_dim", 64)),
    )


def flow_loss(model: StochasticDegrader, x_R: torch.Tensor, x_2R: torch.Tensor, R
              ) -> torch.Tensor:
    """Conditional flow-matching loss on the whitened degradation residual."""
    r_star = model.target_residual(x_R, x_2R)
    r_t, v_star, t, _z = sample_flow_pair(r_star)
    v = model(r_t, t, x_2R, R)
    return F.mse_loss(v, v_star)


@torch.no_grad()
def _val_stoch(model, ops, val_loader, R, device, n_steps: int, n_mean: int,
               n_bands: int, do_sampling: bool, dens=None) -> Dict[str, float]:
    """Deterministic full pass over the val crop grid."""
    if val_loader is None:
        return {}
    model.eval()
    groups = ()
    fm_sum, n_batch = 0.0, 0
    se_sample = se_mean = se_base = 0.0
    n_vox = 0
    bp: Dict[str, torch.Tensor] = {}
    bp_n = 0.0
    dens_acc: Dict[str, float] = {}
    dens_n = 0
    for vb in val_loader:
        vb = common.to_device_batch(vb, device)
        x_R, x_2R = vb["lr"], vb["hr"]
        groups = channel_groups(x_R.shape[1])
        R_t = torch.full((x_R.shape[0],), float(R), device=device)

        fm_sum += float(flow_loss(model, x_R, x_2R, R_t))
        n_batch += 1
        if not do_sampling:
            continue

        a_base = ops.A(x_2R)
        one = model.sample(x_2R, R_t, n_steps=n_steps)
        mean_k = model.conditional_mean(x_2R, R_t, n_samples=n_mean, n_steps=n_steps)

        n_vox += x_R.numel()
        se_sample += float((one - x_R).pow(2).sum())
        se_mean += float((mean_k - x_R).pow(2).sum())
        se_base += float((a_base - x_R).pow(2).sum())

        w = float(x_R.shape[0])
        bp_n += w
        for name, sl in groups:
            for key, field in (("pred", one), ("true", x_R), ("A", a_base)):
                b = band_power(field[:, sl], n_bands=n_bands, log=False) * w
                k = f"{key}_{name}"
                bp[k] = b if k not in bp else bp[k] + b

        # The headline metric: does a SAMPLE make the right universe? Field-space
        # MSE and displacement P(k) both hide the fact that A over-clumps by 13%.
        if dens is not None:
            cellsize, dis_norm, nb_d = dens
            dm = density_metrics(one[:, 0:3], x_R[:, 0:3], a_base[:, 0:3],
                                 cellsize, dis_norm, nb_d, prefix="val/density/")
            for k, v in dm.items():
                dens_acc[k] = dens_acc.get(k, 0.0) + v
            dens_n += 1

    model.train()
    if n_batch == 0:
        return {}
    out: Dict[str, float] = {"val/stoch/flow_loss": fm_sum / n_batch}
    if not do_sampling or n_vox == 0:
        return out

    mse_base = se_base / n_vox
    out["val/stoch/mse_sample"] = se_sample / n_vox
    out["val/stoch/mse_mean_K"] = se_mean / n_vox
    out["val/stoch/mse_vs_A_baseline"] = mse_base
    # Skill of the conditional mean -- directly comparable to the deterministic
    # degrader's val/degrader/skill.
    out["val/stoch/skill_mean_K"] = (
        1.0 - (se_mean / n_vox) / mse_base if mse_base > 0 else 0.0
    )
    for name, _sl in groups:
        true_b = (bp[f"true_{name}"] / bp_n).clamp_min(1e-12)
        pred_r = (bp[f"pred_{name}"] / bp_n) / true_b
        base_r = (bp[f"A_{name}"] / bp_n) / true_b
        # log-space |bias| averaged over bands: 0 == spectrum matches exactly.
        out[f"val/stoch/pk_logbias_{name}"] = float(
            torch.log(pred_r.clamp_min(1e-12)).abs().mean()
        )
        out[f"val/stoch/pk_logbias_A_{name}"] = float(
            torch.log(base_r.clamp_min(1e-12)).abs().mean()
        )
        for b in range(pred_r.numel()):
            out[f"val/stoch/pk_ratio_{name}_b{b}"] = float(pred_r[b])
            out[f"val/stoch/pk_ratio_A_{name}_b{b}"] = float(base_r[b])
    for k, v in dens_acc.items():
        out[k] = v / max(dens_n, 1)
    return out


def train(cfg: Dict[str, Any], smoke: bool = False) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})
    seed = int(train_cfg.get("seed", 0))
    seed_everything(seed)

    device = common.select_device("cpu" if smoke else train_cfg.get("device"))
    factor = int(cfg.get("factor", 8))
    R = float(data_cfg.get("lr_grid_res", 64))
    amp_enabled = bool(train_cfg.get("amp", False))

    steps = int(train_cfg.get("steps", 40000))
    if smoke:
        steps = min(steps, 12)
    lr = float(train_cfg.get("lr", 2e-4))
    bs = int(train_cfg.get("batch_size", 4))
    accum_steps = max(1, int(train_cfg.get("accum_steps", 1)))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    warmup = int(train_cfg.get("warmup", 500))
    min_lr_frac = float(train_cfg.get("min_lr_frac", 0.05))
    num_workers = int(train_cfg.get("num_workers", 0))
    if smoke:
        num_workers = 0
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 2000))
    eval_every = int(train_cfg.get("eval_every", 1000))
    # Sampling is ~2*n_steps forward passes per crop, so do it less often than
    # the (cheap) flow-loss eval.
    sample_every = int(train_cfg.get("sample_every", 4 * eval_every))
    n_steps = int(eval_cfg.get("ode_steps", 20))
    n_mean = int(eval_cfg.get("mean_samples", 8))
    n_bands = int(eval_cfg.get("n_bands", 6))
    if smoke:
        eval_every = min(eval_every, 6)
        sample_every = min(sample_every, 6)
        accum_steps, n_steps, n_mean = min(accum_steps, 2), 2, 2

    if str(data_cfg.get("mode", "real_pair")) != "real_pair":
        raise ValueError("train_stoch_degrader only supports data.mode: real_pair")
    paired_ds, val_ds, _crop, _nl, _fr = _build_real_pair_datasets(cfg, smoke)
    dens = density_cfg(cfg)

    ops = MultiScaleOperators(factor=factor).to(device)
    model = build_stoch_degrader(cfg).to(device)

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    autocast, scaler = common.amp_components(amp_enabled, device)
    sched = common.build_lr_schedule(optimizer, lr, steps, warmup, min_lr_frac)

    run_dir = Path(cfg.get("output", {}).get("run_dir", "runs/stoch_degrader"))
    common.init_run_dir(run_dir, cfg)
    use_wandb = (not smoke) and common.maybe_init_wandb(cfg, run_dir, "stoch_degrader")
    logger = common.CSVLogger(run_dir, use_wandb=use_wandb)

    paired_iter = infinite_loader(paired_ds, bs, seed=seed, num_workers=num_workers)
    val_loader = (finite_loader(val_ds, bs, num_workers=num_workers)
                  if val_ds is not None else None)

    # Whiten the flow-matching target: the disp and vel residuals differ ~9x in
    # scale, and flow matching interpolates against a unit Gaussian.
    res_std = common.estimate_residual_std(paired_iter, ops, device,
                                           n_batches=2 if smoke else 32)
    model.set_residual_std(res_std)
    print(f"[stoch] residual std per channel: {[round(float(v), 4) for v in res_std]}")

    model.train()
    first: Dict[str, float] = {}
    last: Dict[str, float] = {}
    best_val = float("inf")
    best_step = -1
    final_step = 0
    loss_ema: Optional[float] = None
    for step in range(1, steps + 1):
        final_step = step
        t_step = time.perf_counter()
        do_eval = val_loader is not None and (step % eval_every == 0 or step == steps)
        do_sampling = do_eval and (step % sample_every == 0 or step == steps)
        will_log = step % log_every == 0 or step == 1 or step == steps or do_eval

        optimizer.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _ in range(accum_steps):
            pb = common.to_device_batch(next(paired_iter), device)
            x_R, x_2R = pb["lr"], pb["hr"]
            R_t = torch.full((x_R.shape[0],), float(R), device=device)
            with autocast():
                loss = flow_loss(model, x_R, x_2R, R_t) / accum_steps
            scaler.scale(loss).backward()
            acc_loss += float(loss.detach()) * accum_steps

        scaler.unscale_(optimizer)
        grad_norm = common.grad_global_norm(params)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if sched is not None:
            sched.step()

        loss_ema = acc_loss if loss_ema is None else 0.98 * loss_ema + 0.02 * acc_loss
        logs = {
            "loss": acc_loss,
            "stoch/flow_loss": acc_loss,
            "stoch/flow_loss_ema": loss_ema,
            "grad_norm": grad_norm,
        }
        last = logs
        if not first:
            first = dict(logs)

        if will_log:
            logs.update(common.system_metrics(device, time.perf_counter() - t_step))
            row = {**logs, "lr": optimizer.param_groups[0]["lr"]}
            if do_eval:
                val_metrics = _val_stoch(model, ops, val_loader, R, device,
                                         n_steps, n_mean, n_bands, do_sampling, dens)
                row.update(val_metrics)
                v = val_metrics.get("val/stoch/flow_loss")
                if v is not None and v < best_val:
                    best_val, best_step = v, step
                    common.save_checkpoint(run_dir / "ckpt_best.pt", model, optimizer,
                                           step, extra={"val_loss": best_val})
                row["val/best_loss"] = best_val
            logger.log(step, row)
            # Console line -> slurm-*.out. metrics.csv/tb/wandb have everything,
            # but a 40k-step job that prints nothing is impossible to babysit.
            msg = (f"[stoch] step {step:>6}/{steps}  fm_loss {acc_loss:.4f} "
                   f"(ema {loss_ema:.4f})  lr {optimizer.param_groups[0]['lr']:.2e}  "
                   f"gnorm {grad_norm:.3f}")
            if do_eval:
                vfm = row.get("val/stoch/flow_loss")
                if vfm is not None:
                    msg += f"  | val_fm {vfm:.4f}"
                sr, sra = row.get("val/density/sigma_ratio"), row.get("val/density/sigma_ratio_A")
                if sr is not None:
                    # THE metric: 1.0 = degraded field clusters like the real LR
                    # field. sigma_ratio_A is the avg-pool control (over-clumps).
                    msg += f"  | sigma_ratio {sr:.3f} (A: {sra:.3f})"
                mk = row.get("val/stoch/mse_mean_K")
                if mk is not None:
                    msg += (f"  | mse_sample {row['val/stoch/mse_sample']:.4f} "
                            f"mse_mean{n_mean} {mk:.4f} "
                            f"A {row['val/stoch/mse_vs_A_baseline']:.4f}")
            print(msg, flush=True)
        if save_every > 0 and (step % save_every == 0):
            common.save_checkpoint(run_dir / f"ckpt_{step}.pt", model, optimizer, step)

    common.save_checkpoint(run_dir / "ckpt_last.pt", model, optimizer, final_step)
    logger.close()
    common.finish_wandb()
    return {"run_dir": str(run_dir), "first": first, "last": last, "steps": final_step,
            "best_val": best_val if best_step >= 0 else None, "best_step": best_step,
            "checkpoint": str(run_dir / "ckpt_last.pt")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stochastic (flow-matching) degrader training")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--smoke", action="store_true", help="short CPU smoke run")
    parser.add_argument("--set", nargs="*", default=None,
                        help="dotted config overrides, e.g. train.steps=30")
    args = parser.parse_args(argv)
    from ..utils.config import apply_overrides, load_config

    cfg = apply_overrides(load_config(args.config), args.set)
    result = train(cfg, smoke=args.smoke)
    msg = " ".join(f"{k}={v:.4g}" for k, v in result["last"].items())
    print(f"[stoch] done: steps={result['steps']} last[{msg}] run_dir={result['run_dir']}")


if __name__ == "__main__":
    main()
