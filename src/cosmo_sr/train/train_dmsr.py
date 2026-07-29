"""DMSR stage trainer: Stages A-E of the null-space flow + HR critic program.

Stages (selected by ``stage:`` in the config, or ``--stage``)::

    det deterministic paired null-space regression            (no-stochasticity control)
    a   paired stochastic null-space flow                     (supervised baseline)
    b   = a, but the condition encoder starts from LR SSL      (encoder ablation)
    c   = b + HR critic, fakes from **paired LR only**         (critic control)
    d   = b + HR critic, fakes from **balanced all-LR**        (OUR MAIN MODEL)
    e   = d + cubic-equivariance regularization                (optional)

The experiment this file exists to run is **C vs D**. They share architecture,
initialization, optimizer settings, crop sizes, step count, critic update count
and ``lambda_adv`` schedule. The *only* intended difference is where the critic's
fake samples come from:

    Stage C: the second generator batch repeats **paired** LR crops
    Stage D: the second generator batch draws **environment-balanced LR-only** crops

Compute is matched by construction: both stages run the identical cycle of
``1 paired generator batch + 1 second-stream generator batch + n_critic critic
updates`` per step. Stage C repeating paired crops (rather than skipping the
second batch) is what makes the comparison about *data source* instead of about
*number of adversarial updates*. ``--audit-compute`` prints the resulting counts.

Consistency note: ``A(x_hat) = y`` holds by construction and is logged as
``train/exact_consistency_rel``. It is never added to any loss.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch

from ..data.datasets import (
    SyntheticSRDataset,
    finite_loader,
    infinite_loader,
)
from ..dmsr.critic import HRCritic, LazyR1, hinge_d_loss, hinge_g_loss
from ..dmsr.cubic import sample_cubic_rotation
from ..dmsr.data import (
    BalancedLRDataset,
    LRCropPool,
    build_balanced_sampler,
    build_paired_dataset,
    build_val_dataset,
    resolve_split,
)
from ..dmsr.density import (
    CriticInputNormalizer,
    HighPassDensity,
    cellsizes,
    critic_input,
    density_channels,
)
from ..dmsr.evaluate import BandEdges, evaluate_batch, sample_diversity, condition_shuffle_gap
from ..dmsr.flow import (
    a_free_flow_loss,
    build_flow,
    deterministic_free_loss,
    deterministic_regression_loss,
    load_pretrained_encoder,
    null_space_flow_loss,
    unconstrained_flow_loss,
)
from ..dmsr.fourier_diag import BandDiagnosticAccumulator
from ..dmsr.mean_innovation import (
    MeanInnovationFlow,
    build_mean_innovation,
    innovation_diagnostics,
    innovation_flow_loss,
    mean_reconstruction_loss,
)
from ..models.operator_denoiser import ModelEMA
from ..utils.config import apply_overrides, load_config
from ..utils.seed import seed_everything
from . import common

STAGES = ("det", "a", "b", "c", "d", "e")
ADVERSARIAL_STAGES = ("c", "d", "e")
ALL_LR_STAGES = ("d", "e")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def lambda_adv_at(step: int, cfg: Dict[str, Any]) -> float:
    """Linear ramp of ``lambda_adv`` from 0 after the critic warmup.

    Ramped (never switched on at full strength) so the adversarial gradient stays
    well below the paired flow gradient early on -- see ``adv.target_grad_ratio``.
    """
    a = cfg.get("adv", {})
    start = int(a.get("start_step", 0))
    warm = max(1, int(a.get("ramp_steps", 1)))
    lam = float(a.get("lambda_adv", 0.1))
    if step < start:
        return 0.0
    return lam * min(1.0, (step - start) / warm)


def _flow_grad_norm(flow) -> float:
    return common.grad_global_norm(list(flow.parameters()))


def equivariance_loss(flow, ema_flow, y: torch.Tensor, r_t: torch.Tensor,
                      t: torch.Tensor) -> torch.Tensor:
    """Stage E: normalised cubic-equivariance loss against the EMA teacher.

    ``v_theta(g(r_t), t, g(y)) ~ g(v_teacher(r_t, t, y))``. Both the voxel axes and
    the vector components are transformed by :class:`~cosmo_sr.dmsr.cubic.CubicRotation`.
    """
    g = sample_cubic_rotation()
    with torch.no_grad():
        v_teacher = ema_flow.module.velocity(r_t, t, y)
    v_rot = flow.velocity(g.apply(r_t), t, g.apply(y))
    target = g.apply(v_teacher)
    denom = v_teacher.pow(2).mean() + 1e-8
    return (v_rot - target).pow(2).mean() / denom


class _Stream:
    """Named infinite batch stream, so logs can attribute a batch to its source."""

    def __init__(self, name: str, loader: Iterator, device):
        self.name = name
        self.loader = loader
        self.device = device
        self.count = 0

    def next(self) -> Dict[str, torch.Tensor]:
        self.count += 1
        return common.to_device_batch(next(self.loader), self.device)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_streams(cfg, split, stage, device, channels, run_dir):
    """Paired stream, second stream, val loader, and the balancing report."""
    dcfg = cfg.get("data", {})
    tcfg = cfg.get("train", {})
    factor = int(cfg.get("factor", 8))
    crop_lr = int(dcfg.get("crop_lr", 8))
    batch_size = int(tcfg.get("batch_size", 1))
    seed = int(tcfg.get("seed", 0))
    use_channels = dcfg.get("use_channels")
    workers = int(tcfg.get("num_workers", 0))
    balance_report = None
    _, lr_cellsize = cellsizes(dcfg, factor)   # descriptors act on LR crops

    if cfg.get("_synthetic"):
        ds = SyntheticSRDataset(num_samples=64, channels=channels,
                                crop_lr=crop_lr, scale_factor=factor, seed=seed)
        paired = _Stream("paired", infinite_loader(ds, batch_size, seed=seed), device)
        second = _Stream("paired_repeat" if stage == "c" else "lr_only",
                         infinite_loader(ds, batch_size, seed=seed + 1), device)
        val_loader = finite_loader(ds, batch_size)
        return paired, second, val_loader, balance_report

    paired_ds = build_paired_dataset(
        split, crop_lr=crop_lr, scale_factor=factor, seed=seed,
        augment=bool(dcfg.get("augment", True)),
        channels=int(dcfg.get("channels", 6)), use_channels=use_channels,
        mmap=bool(dcfg.get("mmap", True)),
    )
    paired = _Stream("paired", infinite_loader(paired_ds, batch_size, seed=seed,
                                               num_workers=workers), device)

    if stage in ALL_LR_STAGES:
        # Balanced LR-only stream, matched to the paired environment distribution.
        pool_kwargs = dict(
            crop_lr=crop_lr, channels=int(dcfg.get("channels", 6)),
            use_channels=use_channels, mmap=bool(dcfg.get("mmap", True)),
            descriptor_kwargs={**cfg.get("env", {}).get("descriptor", {}),
                               "cellsize": lr_cellsize},
        )
        n_pool = int(cfg.get("env", {}).get("pool_size", 2048))
        paired_pool = LRCropPool(split.train_lr, n_crops=n_pool, seed=seed, **pool_kwargs)
        unpaired_pool = LRCropPool(split.lr_only, n_crops=n_pool * 2, seed=seed + 1, **pool_kwargs)
        sampler, standardizer = build_balanced_sampler(
            paired_pool, unpaired_pool,
            n_dims=int(cfg.get("env", {}).get("n_dims", 2)),
            n_bins=int(cfg.get("env", {}).get("n_bins", 8)),
            seed=seed,
        )
        rep = sampler.report()
        balance_report = {"balance": rep.to_dict(), "standardizer": standardizer.to_dict()}
        with open(Path(run_dir) / "env_balance.json", "w") as f:
            json.dump(balance_report, f, indent=2)
        print(f"[env] source-classifier AUC before={rep.auc_before:.3f} "
              f"after={rep.auc_after:.3f} (target <= 0.60); "
              f"{rep.n_in_support}/{rep.n_unpaired} crops in paired support; "
              f"descriptors kept={standardizer.kept_names} dropped={standardizer.dropped_names}")
        thresh = float(cfg.get("env", {}).get("max_auc", 0.60))
        if rep.auc_after > thresh and not bool(cfg.get("env", {}).get("allow_auc_fail", False)):
            raise RuntimeError(
                f"source_classifier_auc={rep.auc_after:.3f} > {thresh}: the balanced "
                "LR-only pool is still distinguishable from the paired pool. Improve "
                "matching (more bins/dims), reduce descriptor dimensionality, or "
                "restrict the support before running adversarial training. Set "
                "env.allow_auc_fail=true only to inspect a deliberately unbalanced run."
            )
        second_ds = BalancedLRDataset(unpaired_pool, sampler,
                                      length=int(dcfg.get("epoch_length", 4096)), seed=seed)
        second = _Stream("lr_only", infinite_loader(second_ds, batch_size, seed=seed + 1), device)
    else:
        # Stage C control: repeat PAIRED crops so update counts match Stage D.
        second_ds = build_paired_dataset(
            split, crop_lr=crop_lr, scale_factor=factor, seed=seed + 1,
            augment=bool(dcfg.get("augment", True)),
            channels=int(dcfg.get("channels", 6)), use_channels=use_channels,
            mmap=bool(dcfg.get("mmap", True)),
        )
        second = _Stream("paired_repeat", infinite_loader(second_ds, batch_size, seed=seed + 1,
                                                         num_workers=workers), device)

    val_ds = build_val_dataset(
        split, crop_lr=crop_lr, scale_factor=factor,
        channels=int(dcfg.get("channels", 6)), use_channels=use_channels,
        mmap=bool(dcfg.get("mmap", True)),
        max_crops=int(cfg.get("eval", {}).get("max_val_crops", 16)),
    )
    val_loader = finite_loader(val_ds, batch_size)
    return paired, second, val_loader, balance_report


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(flow, val_loader, cfg, device, factor, highpass) -> Dict[str, float]:
    ecfg = cfg.get("eval", {})
    n_steps = int(ecfg.get("n_steps", 20))
    bands = BandEdges(float(ecfg.get("low_frac", 0.5)), float(ecfg.get("high_frac", 1.5)))
    acc: Dict[str, list] = {}
    n_batches = int(ecfg.get("max_val_batches", 4))
    first: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        y = batch["lr"].to(device)
        x = batch["hr"].to(device)
        if first is None:
            first = (y, x)
        m = evaluate_batch(flow, y, x, factor, highpass, n_steps=n_steps, bands=bands)
        for k, v in m.items():
            acc.setdefault(k, []).append(v)

    out = {f"val_{k}": float(np.nanmean(v)) for k, v in acc.items()}
    if first is not None:
        y, x = first
        out.update({f"val_{k}": v for k, v in
                    sample_diversity(flow, y, n_samples=int(ecfg.get("diversity_samples", 3)),
                                     n_steps=n_steps).items()})
        out.update({f"val_{k}": v for k, v in
                    condition_shuffle_gap(flow, y, x, factor, n_steps=n_steps).items()})
        if isinstance(flow, MeanInnovationFlow):        # Part 2 innovation diagnostics
            out.update({f"val_{k}": v for k, v in
                        innovation_diagnostics(flow, y, x, factor,
                                               n_samples=int(ecfg.get("diversity_samples", 3)),
                                               n_steps=n_steps).items()})
    return out


def _post_train_viz(run_dir: Path) -> None:
    """Auto-render the per-run visualisation (dmsr_run_viz.py) after training.

    Best-effort and non-fatal: a viz failure must never fail a training run.
    Spawned as a subprocess so it gets a clean CUDA context / memory and reuses
    the standalone CLI. Repo root (= cwd for the script's ``src`` import path and
    relative run paths) is derived from this file's location.
    """
    import subprocess
    repo_root = Path(__file__).resolve().parents[3]      # .../cosmo_sr_project
    script = repo_root / "scripts" / "dmsr_run_viz.py"
    if not script.exists():
        print(f"[viz] skipped: {script} not found")
        return
    try:
        print(f"[viz] rendering per-run figures for {run_dir} ...")
        subprocess.run(
            [sys.executable, str(script), "--run", str(Path(run_dir).resolve())],
            cwd=str(repo_root), check=True, timeout=3600,
        )
    except Exception as e:  # noqa: BLE001 -- never fail training on a viz error
        print(f"[viz] skipped ({type(e).__name__}: {e})")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None, choices=STAGES)
    ap.add_argument("--set", nargs="*", default=None, help="dotted.key=value overrides")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic run")
    ap.add_argument("--audit-compute", action="store_true",
                    help="print the update-count budget and exit")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    stage = str(args.stage or cfg.get("stage", "a")).lower()
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    cfg["stage"] = stage

    tcfg = dict(cfg.get("train", {}))
    acfg = dict(cfg.get("adv", {}))
    n_critic = int(acfg.get("n_critic", 1))
    steps = int(tcfg.get("steps", 20000))

    if args.audit_compute:
        adv = stage in ADVERSARIAL_STAGES
        print(json.dumps({
            "stage": stage, "steps": steps,
            "generator_updates": steps * (2 if adv else 1),
            "paired_generator_updates": steps,
            "second_stream_generator_updates": steps if adv else 0,
            "second_stream_source": ("lr_only" if stage in ALL_LR_STAGES
                                     else ("paired_repeat" if adv else None)),
            "critic_updates": steps * n_critic if adv else 0,
            "batch_size": int(tcfg.get("batch_size", 1)),
            "crop_lr": int(cfg.get("data", {}).get("crop_lr", 8)),
        }, indent=2))
        return

    if args.smoke:
        tcfg["steps"] = steps = int(tcfg.get("smoke_steps", 4))
        tcfg["batch_size"] = 2
        cfg["_synthetic"] = True
        cfg.setdefault("wandb", {})["mode"] = "disabled"
        cfg.setdefault("eval", {})["n_steps"] = 2
        cfg["eval"]["max_val_batches"] = 1
        cfg["train"] = tcfg
        # Redirect the run directory. A smoke run started from a real stage config
        # inherits that config's `output.run_dir`, and `init_run_dir` then rewrites
        # `config.yaml` / `env.json` in the directory of a possibly *still-running*
        # job -- silently replacing that run's manifest with `steps: 4,
        # _synthetic: true`. This happened once here to a live Stage A run: only the
        # manifest was corrupted (metrics and checkpoints are rewritten by the live
        # process from its own state), but the manifest is the audit trail, so a
        # smoke run must never be able to touch it.
        out = cfg.setdefault("output", {})
        base_dir = str(out.get("run_dir", f"runs/dmsr/stage_{stage}")).rstrip("/")
        out["run_dir"] = f"{base_dir}_smoke_{stage}"

    seed = int(tcfg.get("seed", 0))
    seed_everything(seed)
    device = common.select_device(cfg.get("device"))
    run_dir = Path(common.init_run_dir(
        cfg.get("output", {}).get("run_dir", f"runs/dmsr_stage_{stage}"), cfg))

    dcfg = cfg.get("data", {})
    use_channels = dcfg.get("use_channels")
    channels = len(use_channels) if use_channels else int(dcfg.get("channels", 6))
    factor = int(cfg.get("factor", 8))

    split = None
    if not cfg.get("_synthetic"):
        split = resolve_split(dcfg)
        split.save(run_dir / "split.json")
        print(f"[split] train={len(split.train_hr)} val={len(split.val_hr)} "
              f"test={len(split.test_hr)} lr_only={len(split.lr_only)} boxes")

    # -- model ------------------------------------------------------------- #
    # Experiment E (mean + innovation) is Stage C mechanics with the generator
    # decomposed into a frozen deterministic mean + a stochastic innovation flow.
    # Disabled by default: with `mean_innovation.enabled=false` this whole branch
    # is skipped and the model / loss / logging are byte-identical to Stage C.
    mi_cfg = dict(cfg.get("mean_innovation", {}))
    use_mean_innov = bool(mi_cfg.get("enabled", False))
    # Ablation (b): unconstrained generator (projection removed). Mutually
    # exclusive with mean_innovation. `cons_lambda` weights the soft consistency
    # penalty (0 = fully free; consistency is logged regardless).
    use_unconstrained = bool(cfg.get("model", {}).get("unconstrained", False))
    # Ablation: remove the operator entirely (no A_plus base, no projection).
    # `det` -> one-shot MSE regression of the full field; `a`/`c` -> full-field flow.
    use_a_free = bool(cfg.get("model", {}).get("a_free", False))
    cons_lambda = float(cfg.get("soft_consistency", {}).get("lambda", 0.0))
    if use_unconstrained and use_mean_innov:
        raise ValueError("model.unconstrained is incompatible with mean_innovation.enabled")
    if use_a_free and use_mean_innov:
        raise ValueError("model.a_free is incompatible with mean_innovation.enabled")
    if use_a_free and use_unconstrained:
        raise ValueError("model.a_free and model.unconstrained are mutually exclusive")
    if use_mean_innov:
        if stage == "det":
            raise ValueError("mean_innovation.enabled is incompatible with stage 'det'")
        flow = build_mean_innovation(cfg, channels, device,
                                     load_ckpts=not cfg.get("_synthetic"))
        flow.deterministic = False
        inner = flow.innovation          # encoder init / init_from target
        print(f"[mean_innov] frozen deterministic mean + innovation flow "
              f"(mean_ckpt={mi_cfg.get('mean_ckpt')}, freeze_mean={mi_cfg.get('freeze_mean', True)})")
    else:
        flow = build_flow(cfg, channels).to(device)
        # The `det` baseline is a one-shot regressor, not a flow -- evaluating it by ODE
        # integration would score a trajectory it was never trained on.
        flow.deterministic = (stage == "det")
        inner = flow

    enc_init = str(cfg.get("model", {}).get("condition_encoder_init", "random")).lower()
    if enc_init == "lr_pretrained":
        ckpt = cfg.get("model", {}).get("encoder_ckpt")
        if not ckpt:
            raise ValueError("condition_encoder_init=lr_pretrained requires model.encoder_ckpt")
        info = load_pretrained_encoder(inner, ckpt)
        print(f"[encoder] initialised from {info['encoder_ckpt']}")
    elif enc_init != "random":
        raise ValueError(f"condition_encoder_init must be 'random' or 'lr_pretrained', got {enc_init!r}")

    if cfg.get("model", {}).get("init_from"):
        state = torch.load(cfg["model"]["init_from"], map_location=device, weights_only=False)
        inner.load_state_dict(state["model"])
        print(f"[init] generator{' innovation flow' if use_mean_innov else ''} "
              f"initialised from {cfg['model']['init_from']}")

    freeze_enc = bool(cfg.get("model", {}).get("freeze_encoder", False))
    flow.set_encoder_trainable(not freeze_enc)

    base_lr = float(tcfg.get("lr", 1e-4))
    enc_lr_scale = float(cfg.get("model", {}).get("encoder_lr_scale", 0.1))
    groups = [{"params": list(flow.flow_parameters()), "lr": base_lr}]
    if not freeze_enc:
        groups.append({
            "params": list(flow.encoder_parameters()),
            "lr": base_lr * (enc_lr_scale if enc_init == "lr_pretrained" else 1.0),
        })
    # Optional joint fine-tuning (DISABLED by default; run only after the frozen-mean
    # experiment). The mean gets a lower LR and is trained ONLY by its own
    # reconstruction loss (the flow loss keeps the mean detached), per spec.
    joint_ft = use_mean_innov and bool(mi_cfg.get("joint_finetune", False))
    if joint_ft:
        flow.unfreeze_mean()
        mean_lr = base_lr * float(mi_cfg.get("mean_lr_scale", 0.1))
        groups.append({"params": list(flow.mean_parameters()), "lr": mean_lr})
        print(f"[mean_innov] JOINT fine-tuning: mean lr = {mean_lr:.2e} (< flow lr {base_lr:.2e})")
    opt_g = torch.optim.Adam(groups, lr=base_lr, betas=tuple(tcfg.get("betas", (0.9, 0.99))))
    ema = ModelEMA(flow, decay=float(tcfg.get("ema_decay", 0.999)))

    hr_cellsize, lr_cellsize = cellsizes(dcfg, factor)
    highpass = HighPassDensity(
        factor=factor,
        lowpass=str(cfg.get("critic", {}).get("lowpass", "blockavg")),
        kcut_frac=float(cfg.get("critic", {}).get("kcut_frac", 1.0)),
        cellsize=hr_cellsize,
        dis_norm=float(dcfg.get("dis_norm", 6000.0)),
        # 0 = legacy wrapped deposit (byte-identical). A positive value scores
        # only that central Eulerian cube, offset by the crop's own bulk
        # displacement. Measured on set14 with a 64^3 crop: the wrapped field is
        # correlated r=0.08 with the truth, valid_center=64 gives 0.955 and
        # valid_center=32 gives 1.000 (rel RMS 0.005).
        valid_center=int(cfg.get("critic", {}).get("valid_center", 0)),
    ).to(device)
    print(f"[geom] boxsize={dcfg.get('boxsize', 100000.0)} kpc/h  "
          f"HR cellsize={hr_cellsize:.4f}  LR cellsize={lr_cellsize:.4f}")

    adversarial = stage in ADVERSARIAL_STAGES
    # Critic-input modes (defaults reproduce the original concat(P_A(x), rho_high)):
    #   residual_mode: 'nullspace' (P_A) | 'full' (whole displacement, SR2-style)
    #   density_mode:  'highpass' | 'full' (whole delta) | 'off'
    #   global_pool:   local PatchGAN (False) | global single-score D (True, SR2-style)
    ccfg_modes = cfg.get("critic", {})
    res_mode = str(ccfg_modes.get("residual_mode", "nullspace"))
    den_mode = str(ccfg_modes.get("density_mode", "highpass"))
    n_density_ch = density_channels(den_mode)
    critic = opt_d = lazy_r1 = None
    if adversarial:
        ccfg = cfg.get("critic", {})
        critic = HRCritic(
            in_channels=channels + n_density_ch,
            width=int(ccfg.get("width", 64)),
            n_layers=int(ccfg.get("n_layers", 3)),
            global_pool=bool(ccfg.get("global_pool", False)),
        ).to(device)
        opt_d = torch.optim.Adam(critic.parameters(), lr=float(ccfg.get("lr", 2e-4)),
                                 betas=tuple(ccfg.get("betas", (0.0, 0.99))))
        lazy_r1 = LazyR1(gamma=float(ccfg.get("r1_gamma", 10.0)),
                         interval=int(ccfg.get("r1_interval", 16)))

    paired, second, val_loader, balance_report = build_streams(
        cfg, split, stage, device, channels, run_dir)

    # Fixed critic-input scales, estimated from REAL paired HR crops only and then
    # applied identically to real and fake (see CriticInputNormalizer).
    critic_norm = None
    if adversarial:
        n_fit = int(cfg.get("critic", {}).get("norm_fit_batches", 16))
        fit_batches = [paired.next()["hr"] for _ in range(n_fit)]
        critic_norm = CriticInputNormalizer.fit(
            fit_batches, flow.operator, highpass,
            residual_mode=res_mode, density_mode=den_mode).to(device)
        with open(run_dir / "critic_norm.json", "w") as f:
            json.dump(critic_norm.to_dict(), f, indent=2)
        print(f"[critic] input scales (from {n_fit} real batches): {critic_norm.to_dict()}")

    use_wandb = common.maybe_init_wandb(cfg, run_dir, job_type=f"stage_{stage}")
    logger = common.CSVLogger(run_dir, use_wandb=use_wandb)

    adv_n_steps = int(acfg.get("gen_ode_steps", 4))
    bp_steps = acfg.get("bp_steps", 1)
    warmup_d = int(acfg.get("critic_warmup_steps", 0))
    log_every = int(tcfg.get("log_every", 50))
    eval_every = int(tcfg.get("eval_every", 500))
    save_every = int(tcfg.get("save_every", 2000))
    grad_log_every = int(tcfg.get("grad_log_every", 200))
    # In-training Fourier-band diagnostic (Part 3). 0 = OFF (default) => byte-identical
    # to Stage C. When > 0 it reuses the flow loss's own (v_pred, v_target) via
    # return_fields, so enabling it draws NO extra RNG and does not perturb training.
    fourier_diag_every = int(cfg.get("diagnostics", {}).get("fourier_band_every", 0))
    best_metric_name = str(cfg.get("eval", {}).get("best_metric", "val_rk_transition"))
    best_mode = str(cfg.get("eval", {}).get("best_mode", "max"))
    best_value = -np.inf if best_mode == "max" else np.inf

    clip = float(tcfg.get("grad_clip", 0.0))
    counters = {"gen_paired": 0, "gen_second": 0, "critic": 0}
    t0 = time.time()

    for step in range(1, steps + 1):
        lam = lambda_adv_at(step, cfg) if adversarial else 0.0
        row: Dict[str, float] = {"step": step, "lambda_adv": lam}

        # ---------------- critic warmup (generator frozen) ----------------- #
        in_warmup = adversarial and step <= warmup_d

        # ---------------- generator: paired batch -------------------------- #
        batch = paired.next()
        y_p, x_p = batch["lr"], batch["hr"]
        if not in_warmup:
            want_fields = fourier_diag_every > 0 and (step % fourier_diag_every == 0)
            fields = None
            if stage == "det":
                # A_plus-base-but-unprojected (unconstrained) or operator-free (a_free)
                # deterministic regression; else the projected null-space regression.
                if use_unconstrained or use_a_free:
                    loss_flow, m_flow = deterministic_free_loss(flow, y_p, x_p)
                else:
                    loss_flow, m_flow = deterministic_regression_loss(flow, y_p, x_p)
            elif use_mean_innov:
                out = innovation_flow_loss(flow, y_p, x_p, return_fields=want_fields)
                loss_flow, m_flow = out[0], out[1]
                fields = out[2] if want_fields else None
            elif use_a_free:
                out = a_free_flow_loss(flow, y_p, x_p, return_fields=want_fields)
                loss_flow, m_flow = out[0], out[1]
                fields = out[2] if want_fields else None
            elif use_unconstrained:
                out = unconstrained_flow_loss(flow, y_p, x_p, lambda_cons=cons_lambda,
                                              return_fields=want_fields)
                loss_flow, m_flow = out[0], out[1]
                fields = out[2] if want_fields else None
            else:
                out = null_space_flow_loss(flow, y_p, x_p, return_fields=want_fields)
                loss_flow, m_flow = out[0], out[1]
                fields = out[2] if want_fields else None
            row.update(m_flow)
            if fields is not None:                       # cheap in-training band monitor
                _acc = BandDiagnosticAccumulator(factor=factor)
                _acc.add(fields["v_pred"], fields["v_target"], flow.operator)
                _bt = _acc.result().band_totals()
                for _band in ("low", "transition", "high"):
                    row[f"flowdiag_loss_frac_{_band}"] = _bt[_band]["loss_fraction"]
                    row[f"flowdiag_rel_err_{_band}"] = _bt[_band]["relative_shell_error"]

            loss_adv_p = None
            if adversarial and lam > 0:
                x_fake = flow.generate(y_p, n_steps=adv_n_steps, bp_steps=bp_steps)
                loss_adv_p = hinge_g_loss(critic(critic_input(
                    x_fake, flow.operator, highpass, normalizer=critic_norm,
                    residual_mode=res_mode, density_mode=den_mode)))
                row["loss_G_adv"] = float(loss_adv_p.detach())

            # Periodically measure the two gradient norms separately so
            # lambda_adv can be calibrated (target ratio 0.1-0.3).
            if loss_adv_p is not None and step % grad_log_every == 0:
                opt_g.zero_grad(set_to_none=True)
                loss_flow.backward(retain_graph=True)
                row["grad_norm_flow"] = _flow_grad_norm(flow)
                opt_g.zero_grad(set_to_none=True)
                (lam * loss_adv_p).backward(retain_graph=True)
                row["grad_norm_adv"] = _flow_grad_norm(flow)
                row["grad_ratio_adv_flow"] = row["grad_norm_adv"] / max(row["grad_norm_flow"], 1e-12)
                opt_g.zero_grad(set_to_none=True)

            loss_g = loss_flow + (lam * loss_adv_p if loss_adv_p is not None else 0.0)

            if joint_ft:   # mean trained ONLY by its own recon loss (flow keeps it detached)
                loss_mean, m_mean = mean_reconstruction_loss(flow, y_p, x_p)
                loss_g = loss_g + float(mi_cfg.get("lambda_mean_recon", 1.0)) * loss_mean
                row.update(m_mean)

            if stage == "e" and float(cfg.get("equivariance", {}).get("lambda_eq", 0.0)) > 0:
                if np.random.rand() < float(cfg.get("equivariance", {}).get("fraction", 0.25)):
                    r_target = flow.operator.P_A(x_p)
                    t_e = torch.rand(y_p.shape[0], device=device, dtype=y_p.dtype)
                    z_null = flow.operator.P_A(torch.randn_like(x_p))
                    tb = t_e.view(-1, *([1] * (x_p.dim() - 1)))
                    r_t = (1 - tb) * z_null + tb * r_target
                    l_eq = equivariance_loss(flow, ema, y_p, r_t, t_e)
                    loss_g = loss_g + float(cfg["equivariance"]["lambda_eq"]) * l_eq
                    row["loss_eq"] = float(l_eq.detach())

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            if "grad_norm_flow" not in row:
                row["grad_norm_flow"] = _flow_grad_norm(flow)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(flow.parameters(), clip)
            opt_g.step()
            counters["gen_paired"] += 1

            with torch.no_grad():
                x_hat_diag = flow.operator.combine(y_p, flow.operator.P_A(x_p))
                row["exact_consistency_rel"] = flow.operator.consistency_error(x_hat_diag, y_p)[1]

        # ------- generator: second stream (paired-repeat in C, LR-only in D) ---- #
        batch2 = second.next()
        y_2 = batch2["lr"]
        if adversarial and lam > 0 and not in_warmup:
            x_fake2 = flow.generate(y_2, n_steps=adv_n_steps, bp_steps=bp_steps)
            loss_adv2 = hinge_g_loss(critic(critic_input(
                x_fake2, flow.operator, highpass, normalizer=critic_norm,
                residual_mode=res_mode, density_mode=den_mode)))
            # No HR regression / pseudo-target loss here, by design: the second
            # stream has no HR ground truth and inventing one would be exactly
            # the pseudo-pair shortcut this stage excludes.
            opt_g.zero_grad(set_to_none=True)
            (lam * loss_adv2).backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(flow.parameters(), clip)
            opt_g.step()
            counters["gen_second"] += 1
            row[f"loss_G_adv_{second.name}"] = float(loss_adv2.detach())

        ema.update(flow)

        # ---------------- critic updates ----------------------------------- #
        if adversarial:
            for ci in range(n_critic):
                y_src, tag = ((y_p, "paired") if ci % 2 == 0 else (y_2, second.name))
                with torch.no_grad():
                    x_fake_d = flow.generate(y_src, n_steps=adv_n_steps)
                real_in = critic_input(x_p, flow.operator, highpass, normalizer=critic_norm,
                                       residual_mode=res_mode, density_mode=den_mode).detach()
                fake_in = critic_input(x_fake_d, flow.operator, highpass, normalizer=critic_norm,
                                       residual_mode=res_mode, density_mode=den_mode).detach()
                s_real, s_fake = critic(real_in), critic(fake_in)
                loss_d = hinge_d_loss(s_real, s_fake)
                pen, r1m = lazy_r1(critic, real_in)
                if pen is not None:
                    loss_d = loss_d + pen
                    row.update(r1m)
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
                counters["critic"] += 1
                row["loss_D"] = float(loss_d.detach())
                row["critic_real_score"] = float(s_real.detach().mean())
                key = ("critic_fake_paired_score" if tag == "paired"
                       else "critic_fake_unpaired_score")
                row[key] = float(s_fake.detach().mean())

        # ---------------- logging / eval / checkpoints --------------------- #
        if step % log_every == 0 or step == 1:
            row.update(common.system_metrics(device, (time.time() - t0) / step))
            row.update({f"count_{k}": v for k, v in counters.items()})
            if balance_report:
                row["source_classifier_auc"] = balance_report["balance"]["auc_after"]
            logger.log(step, {k: v for k, v in row.items() if k != "step"})
            msg = " ".join(f"{k}={row[k]:.4g}" for k in
                           ("loss_flow", "loss_G_adv", "loss_D") if k in row)
            print(f"[{stage}] step {step}/{steps} {msg}")

        if eval_every and (step % eval_every == 0 or step == steps):
            flow.eval()
            vm = validate(ema.module, val_loader, cfg, device, factor, highpass)
            flow.train()
            logger.log(step, vm)
            print(f"[{stage}] step {step} " +
                  " ".join(f"{k}={v:.4g}" for k, v in list(vm.items())[:6]))
            cur = vm.get(best_metric_name)
            if cur is not None and np.isfinite(cur):
                better = cur > best_value if best_mode == "max" else cur < best_value
                if better:
                    best_value = cur
                    common.save_checkpoint(run_dir / "ckpt_best.pt", flow, opt_g, step=step,
                                           extra={"ema": ema.module.state_dict(),
                                                  best_metric_name: cur, "stage": stage})

        if save_every and (step % save_every == 0 or step == steps):
            common.save_checkpoint(run_dir / "ckpt_last.pt", flow, opt_g, step=step,
                                   extra={"ema": ema.module.state_dict(), "stage": stage})
            if critic is not None:
                common.save_checkpoint(run_dir / "critic_last.pt", critic, opt_d, step=step)

    with open(run_dir / "compute_audit.json", "w") as f:
        json.dump({"stage": stage, "steps": steps, **counters,
                   "second_stream_source": second.name, "n_critic": n_critic}, f, indent=2)
    print(f"[{stage}] done. counters={counters} second_stream={second.name}")
    common.finish_wandb()

    # Auto-render per-run figures (skip synthetic smoke runs). Non-fatal.
    if not cfg.get("_synthetic") and (run_dir / "ckpt_best.pt").exists():
        _post_train_viz(run_dir)


if __name__ == "__main__":
    main()
