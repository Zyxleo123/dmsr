"""Training loop for the residual diffusion prior and its reward-weighted round.

One loop serves both stages because they differ only in which batches are drawn
and which extra loss terms are active:

* **prior** (stage 3): ``L_sup`` alone, on real paired residuals.
* **distill** (stage 8): ``L = L_sup + lambda_elite L_elite + lambda_ref L_ref``,
  initialised from the prior, with the teacher frozen and ``lambda_elite`` ramped
  from zero. Paired supervision is *kept throughout* -- dropping it is how a
  reward-weighted round quietly turns into reward hacking.

The abort conditions in ``distill.abort_on`` are checked at every validation and
stop the run rather than letting it finish and be discovered later.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.crops import periodic_crop
from ..models.operator_denoiser import ModelEMA
from ..train.common import (CSVLogger, amp_components, build_lr_schedule,
                            grad_global_norm, init_run_dir, maybe_init_wandb,
                            save_checkpoint, select_device, system_metrics)
from ..utils.seed import seed_everything
from .diffusion import DiffusionConfig, denoising_loss, unwhiten, whiten
from .model import ResidualDenoiser, build_residual_denoiser
from .replay import ReplayEntry, elite_weights, read_replay
from .targets import PairedResidualCrops, resolve_boxes

__all__ = ["ReplayCropDataset", "TrainState", "run_training"]


class ReplayCropDataset(Dataset):
    """Crops of generated elite residuals, with their reward weights.

    Conditioning tensors are **not** duplicated: each entry references a box, an
    HR origin and the shared SR2 base cache, and the crop is cut from the
    memory-mapped originals.

    The residual is returned **scaled by the entry's own ``residual_scale``**.
    What earned the reward is ``Psi_base + a * dPsi``, so ``a * dPsi`` is the
    target; returning the raw ``dPsi`` made a residual-scale sweep select on
    ``a`` and then train on something else, and the discrepancy was invisible
    because ``a = 1`` in every default config.

    ``context_margin`` matches :class:`PairedResidualCrops`: elite crops must
    carry the same real context as paired ones or the two loss terms are
    computed on different operators.
    """

    def __init__(
        self,
        entries: Sequence[ReplayEntry],
        box_paths: Dict[str, Dict[str, Path]],
        *,
        crop_hr: int = 64,
        scale_factor: int = 8,
        length: int = 100000,
        seed: int = 0,
        redshift: float = 0.0,
        context_margin: int = 0,
    ):
        if not entries:
            raise ValueError("replay dataset is empty")
        if crop_hr % scale_factor:
            raise ValueError(f"crop_hr={crop_hr} must be a multiple of {scale_factor}")
        if int(context_margin) % int(scale_factor):
            raise ValueError(
                f"context_margin={context_margin} must be a multiple of "
                f"{scale_factor}"
            )
        self.entries = list(entries)
        self.box_paths = box_paths
        self.core_hr = int(crop_hr)
        self.context_margin = int(context_margin)
        self.crop_hr = int(crop_hr)
        self.scale_factor = int(scale_factor)
        self.length = int(length)
        self.seed = int(seed)
        self.redshift = float(redshift)
        self._open: Dict[str, Dict[str, np.ndarray]] = {}
        self._resid: Dict[str, np.ndarray] = {}
        self.weights = np.ones(len(self.entries), dtype=np.float64)

    def set_weights(self, w: Sequence[float]) -> None:
        w = np.asarray(w, dtype=np.float64)
        if w.shape[0] != len(self.entries):
            raise ValueError(f"{w.shape[0]} weights for {len(self.entries)} entries")
        self.weights = w

    def __len__(self) -> int:
        return self.length

    def _box(self, box: str) -> Dict[str, np.ndarray]:
        if box not in self._open:
            p = self.box_paths[box]
            self._open[box] = {
                "lr": np.load(p["lr"], mmap_mode="r"),
                "base": np.load(p["base"], mmap_mode="r"),
            }
        return self._open[box]

    def _residual(self, e: ReplayEntry) -> np.ndarray:
        if e.residual_path not in self._resid:
            self._resid[e.residual_path] = np.load(e.residual_path, mmap_mode="r")
        return self._resid[e.residual_path]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed * 7_919 + i)
        j = int(rng.integers(len(self.entries)))
        e = self.entries[j]
        f = self._box(e.box)
        # The residual on disk is the full box (one file shared by every chunk of
        # that realisation); the entry's chunk is the sub-cube that earned the
        # credit, so the crop must stay inside it.
        arr = self._residual(e)
        chunk = int(e.chunk_hr)
        full_box = int(arr.shape[-1]) > chunk
        span = chunk - self.core_hr
        off = tuple(
            int(rng.integers(span + 1)) // self.scale_factor * self.scale_factor
            for _ in range(3)
        ) if span > 0 else (0, 0, 0)

        m_hr = self.context_margin
        m_lr = m_hr // self.scale_factor
        # The core origin, in the box's own coordinates. The residual field on
        # disk is periodic in the box, so the context is taken periodically too
        # -- the same neighbourhood the sampler used when it produced it.
        hr_start = tuple(int(e.hr_origin[d]) + off[d] for d in range(3))
        if full_box:
            resid = periodic_crop(np.asarray(arr), hr_start, self.core_hr, pad=m_hr)
        else:
            # A per-chunk residual file has no context outside the chunk; wrap
            # inside it rather than reading past the end.
            resid = periodic_crop(np.asarray(arr), off, self.core_hr, pad=m_hr)
        # What earned the reward is Psi_base + a * dPsi, so a * dPsi is the target.
        a = float(getattr(e, "residual_scale", 1.0) or 1.0)
        if a != 1.0:
            resid = np.asarray(resid, dtype=np.float32) * np.float32(a)

        lr_start = tuple(s // self.scale_factor for s in hr_start)
        base = periodic_crop(np.asarray(f["base"]), hr_start, self.core_hr, pad=m_hr)
        y_lr = periodic_crop(np.asarray(f["lr"]), lr_start,
                             self.core_hr // self.scale_factor, pad=m_lr)
        return {
            "residual": torch.from_numpy(np.ascontiguousarray(resid, dtype=np.float32)),
            "psi_base": torch.from_numpy(np.ascontiguousarray(base, dtype=np.float32)),
            "y_lr": torch.from_numpy(np.ascontiguousarray(y_lr, dtype=np.float32)),
            "weight": torch.tensor(float(self.weights[j]), dtype=torch.float32),
            "redshift": torch.tensor(self.redshift, dtype=torch.float32),
            "core_hr": torch.tensor(self.core_hr, dtype=torch.long),
        }


@dataclass
class TrainState:
    step: int = 0
    best_val: float = float("inf")


def _infinite(loader: DataLoader):
    while True:
        for b in loader:
            yield b


def _band_metrics(pred: torch.Tensor, true: torch.Tensor, factor: int,
                  prefix: str) -> Dict[str, float]:
    from ..dmsr.evaluate import rk_tk_summary

    try:
        return rk_tk_summary(pred, true, factor=factor, prefix=prefix)
    except Exception:      # small crops can fall below the binning minimum
        return {}


@torch.no_grad()
def _diagnostics(
    model: ResidualDenoiser,
    batch: Dict[str, torch.Tensor],
    diff_cfg: DiffusionConfig,
    *,
    device,
    scale_factor: int,
    residual_scale: float,
    n_samples: int = 2,
    boxsize_mpc_h: float = 100.0,
    dis_norm_kpc_h: float = 6000.0,
    core: Optional[int] = None,
) -> Dict[str, float]:
    """Composed-field diagnostics on one validation batch.

    Density is deposited with :func:`cic_density_valid_center`, not the plain
    wrap-in-crop ``cic_density``: on a crop the wrap is a scrambling, not an
    approximation (r=0.08 with the true crop density -- see
    ``eval/density.py::cic_density_valid_center``), which would make
    ``diag_dens_*`` track a corrupted field instead of the model. Scoring the
    central half of the crop (``region = n_hr // 2``) is the same ratio
    validated in ``configs/dmsr/t13_fix_vc32.yaml`` (corr~1.0000, rel RMS~0.005
    at crop=64/region=32). Absolute density fidelity is still cross-checked on
    full boxes by the evaluation stage.
    """
    from ..eval.density import cic_density_valid_center

    from .base import _ddim_from_noise
    from .diffusion import valid_core

    model.eval()
    base_full = batch["psi_base"].to(device)
    y = batch["y_lr"].to(device)
    resid_true_full = batch["residual"].to(device)

    samples = []
    clip_stats: Dict[str, float] = {}
    for s in range(int(n_samples)):
        gen = torch.Generator(device="cpu").manual_seed(1234 + s)
        u = torch.randn(tuple(base_full.shape), generator=gen,
                        dtype=torch.float32).to(device)
        st: Dict = {}
        u = _ddim_from_noise(model, u, diff_cfg,
                             cond={"y_lr": y, "psi_base": base_full, "redshift": 0.0},
                             seed=1234 + s, stats=st)
        clip_stats["diag_x0_clip_fraction_max"] = max(
            clip_stats.get("diag_x0_clip_fraction_max", 0.0),
            float(st.get("x0_clip_fraction_max", 0.0)),
        )
        clip_stats["diag_x0_clip_fraction_last"] = max(
            clip_stats.get("diag_x0_clip_fraction_last", 0.0),
            float(st.get("x0_clip_fraction_last", 0.0)),
        )
        samples.append(unwhiten(u, model.sigma_res))
    # Sampling needs the context; every reported number is on the valid core,
    # so the diagnostics measure what full-box tiling will actually write.
    stack = valid_core(torch.stack(samples), core)
    base = valid_core(base_full, core)
    resid_true = valid_core(resid_true_full, core)
    hr = base + resid_true
    y = valid_core(y, None if core is None else max(int(core) // scale_factor, 1))
    pred_resid = stack[0]
    composed = base + float(residual_scale) * pred_resid

    scale_rms = stack.pow(2).mean().sqrt().clamp_min(1e-12)
    out: Dict[str, float] = {
        "diag_residual_rms_pred": float(pred_resid.pow(2).mean().sqrt()),
        "diag_residual_rms_true": float(resid_true.pow(2).mean().sqrt()),
        "diag_sample_diversity": float(
            stack.var(dim=0, unbiased=True).mean().sqrt() / scale_rms
        ) if n_samples > 1 else float("nan"),
    }
    out.update(clip_stats)
    out.update(_band_metrics(composed[:, 0:3], hr[:, 0:3], scale_factor, "diag_disp_"))
    out.update(_band_metrics(base[:, 0:3], hr[:, 0:3], scale_factor, "diag_base_"))

    n_hr = base.shape[-1]
    cell = boxsize_mpc_h * 1000.0 / (n_hr * (base.shape[-1] and 1))
    # crop cellsize == box cellsize: the crop is cut from a 512^3 box
    cell = boxsize_mpc_h * 1000.0 / 512.0
    try:
        region = n_hr // 2
        d_pred = cic_density_valid_center(composed[:, 0:3], cell, dis_norm_kpc_h, region)
        d_true = cic_density_valid_center(hr[:, 0:3], cell, dis_norm_kpc_h, region)
        d_base = cic_density_valid_center(base[:, 0:3], cell, dis_norm_kpc_h, region)
        out["diag_density_sigma_ratio"] = float(d_pred.std() / d_true.std().clamp_min(1e-12))
        out["diag_density_sigma_ratio_base"] = float(d_base.std() / d_true.std().clamp_min(1e-12))
        out.update(_band_metrics(d_pred, d_true, 1, "diag_dens_"))
    except Exception:
        pass

    # LR consistency of the composed field, relative to the frozen baseline.
    from ..operators.multiscale import block_average

    a_hat = block_average(composed, scale_factor)
    a_base = block_average(base, scale_factor)
    out["diag_low_k_change"] = float(
        (a_hat - a_base).pow(2).mean().sqrt() / a_base.pow(2).mean().sqrt().clamp_min(1e-12)
    )
    out["diag_lr_consistency"] = float(
        (a_hat - y).pow(2).mean().sqrt() / y.pow(2).mean().sqrt().clamp_min(1e-12)
    )
    model.train()
    return out


def run_training(cfg: Dict, *, mode: str = "prior", smoke: bool = False,
                 resume: Optional[str] = None, run_dir: Optional[str] = None) -> Path:
    """Train the residual denoiser. ``mode`` is ``"prior"`` or ``"distill"``."""
    if mode not in ("prior", "distill"):
        raise ValueError(f"mode must be 'prior' or 'distill', got {mode!r}")
    tcfg = dict(cfg.get("train", {}))
    dcfg = dict(cfg.get("data", {}))
    mcfg = dict(cfg.get("model", {}))
    diff = DiffusionConfig(**{k: v for k, v in dict(cfg.get("diffusion", {})).items()
                              if k in DiffusionConfig.__dataclass_fields__})
    distill_cfg = dict(cfg.get("distill", {}))

    if smoke:
        tcfg.update(steps=6, batch_size=1, grad_accum=1, num_workers=0,
                    val_every=3, ckpt_every=1000, log_every=1, val_batches=1,
                    amp=False, diag_samples=2)
        dcfg["crop_hr"] = int(dcfg.get("smoke_crop_hr", 32))
        mcfg = {**mcfg, "width": 8, "num_levels": 2, "blocks_per_level": 1,
                "use_checkpoint": False}
        diff = DiffusionConfig(n_steps=3, loss_weighting=diff.loss_weighting)

    out_dir = Path(run_dir or cfg.get("output", {}).get("run_dir") or "runs/reward_tmp")
    init_run_dir(out_dir, cfg)
    seed_everything(int(tcfg.get("seed", 0)))
    device = select_device(tcfg.get("device"))

    scale = int(dcfg.get("scale_factor", 8))
    crop_hr = int(dcfg.get("crop_hr", 64))
    base_seed = int(dcfg.get("base_seed", 0))
    # Context margin: how much real neighbourhood is supplied around the scored
    # core, in HR cells. It must be at least the model's receptive-field
    # half-width, and should equal the margin the full-box sampler uses
    # (TILE_MARGIN), or training and inference are different operations. See
    # PairedResidualCrops for the argument.
    context_margin = int(dcfg.get("context_margin", 0))
    if smoke:
        # The smoke crop is tiny and the smoke model is 1 block deep; a 48-cell
        # margin around a 32-cell core would make the run a memory test.
        context_margin = int(dcfg.get("smoke_context_margin", 0))

    # ---------------------------------------------------------------- data --
    split = cfg.get("split", {})
    train_boxes = list(split.get("train_boxes", []))
    val_boxes = list(split.get("val_boxes", []))
    overlap = set(train_boxes) & set(val_boxes)
    if overlap:
        raise RuntimeError(f"train/val box overlap: {sorted(overlap)}")

    replay_entries: Optional[List[ReplayEntry]] = None
    if mode == "distill" and distill_cfg.get("replay_manifest"):
        # Read and leak-check before anything touches the data root, so a leak is
        # reported as a leak rather than as a missing-file error further down.
        replay_entries = read_replay(distill_cfg["replay_manifest"], verify=not smoke)
        leaked = sorted({e.box for e in replay_entries} - set(train_boxes))
        if leaked:
            raise RuntimeError(
                f"replay buffer contains non-training boxes {leaked}; "
                "validation/test data must never enter the elite set"
            )

    if smoke and not Path(dcfg.get("root", "")).is_dir():
        train_ds = _SyntheticResidualCrops(crop_hr, scale, length=64, seed=0)
        val_ds = _SyntheticResidualCrops(crop_hr, scale, length=8, seed=1)
    else:
        tb = resolve_boxes(train_boxes, dcfg["root"], base_seed=base_seed)
        vb = resolve_boxes(val_boxes, dcfg["root"], base_seed=base_seed)
        # Rung 3 only: restrict training crops to a handful of fixed massive-host
        # chunks. Validation stays on the whole box, so the run reports both
        # "reward rose on the hosts" and "what it cost everywhere else".
        train_ds = PairedResidualCrops(
            tb, crop_hr=crop_hr, scale_factor=scale,
            length=int(dcfg.get("train_length", 200000)),
            seed=int(tcfg.get("seed", 0)), redshift=float(dcfg.get("redshift", 0.0)),
            host_chunks=dcfg.get("fixed_host_chunks"),
            chunk_hr=int(cfg.get("geometry", {}).get("chunk_hr", 128)),
            context_margin=context_margin,
        )
        # Validation carries the SAME context margin: diagnostics measured on
        # wrapped crops would hide exactly the discrepancy the margin exists to
        # remove, which is how a contaminated prior passes Gate A.
        val_ds = PairedResidualCrops(
            vb, crop_hr=crop_hr, scale_factor=scale,
            length=int(dcfg.get("val_length", 512)),
            seed=int(tcfg.get("seed", 0)) + 991,
            redshift=float(dcfg.get("redshift", 0.0)),
            context_margin=context_margin,
        )
        print(f"crops: core {crop_hr}^3 + {context_margin} cells of context "
              f"= {crop_hr + 2 * context_margin}^3 input", flush=True)

    nw = int(tcfg.get("num_workers", 0))
    bs = int(tcfg.get("batch_size", 4))
    train_loader = DataLoader(train_ds, batch_size=bs, num_workers=nw, drop_last=True,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=bs, num_workers=0, drop_last=True)
    train_it = _infinite(train_loader)

    replay_it = None
    replay_ds = None
    if mode == "distill":
        manifest = distill_cfg.get("replay_manifest")
        if not manifest and not smoke:
            raise ValueError("distill mode needs distill.replay_manifest")
        if not manifest:
            # The GPU smoke test exists to prove the two-loss step runs, which it
            # can do on synthetic elites; it must not require a full oracle round.
            replay_ds = _SyntheticResidualCrops(crop_hr, scale, length=32, seed=2)
            replay_it = _infinite(DataLoader(replay_ds, batch_size=bs, drop_last=True))
            print("smoke: synthetic replay batches (no manifest given)", flush=True)
        else:
            entries = replay_entries
            # Resolve each box's base field from the ENTRY, not from the config:
            # the elite was composed against one specific cached SR2 field, and
            # conditioning on a different one makes the recorded residual the
            # wrong target. A mismatch is an error, not a fallback.
            seeds = {e.base_seed for e in entries}
            if len(seeds) > 1:
                raise RuntimeError(
                    f"replay entries span base seeds {sorted(seeds)}; one round "
                    f"must be composed against a single frozen realisation"
                )
            entry_seed = int(next(iter(seeds))) if seeds else base_seed
            if entry_seed != base_seed:
                print(f"replay: base_seed {entry_seed} from the manifest overrides "
                      f"data.base_seed={base_seed}", flush=True)
            box_paths = {}
            for b in resolve_boxes(sorted({e.box for e in entries}), dcfg["root"],
                                   base_seed=entry_seed):
                recorded = {e.base_id for e in entries
                            if e.box == b.box and e.base_id}
                if recorded:
                    if len(recorded) > 1:
                        raise RuntimeError(
                            f"{b.box}: replay entries name {len(recorded)} "
                            f"different base fields {sorted(recorded)}"
                        )
                    p = Path(next(iter(recorded)))
                    if not p.is_file():
                        raise FileNotFoundError(
                            f"{b.box}: replay names base field {p}, which is gone"
                        )
                    if b.base is not None and Path(b.base) != p:
                        print(f"replay: {b.box} base <- {p} (manifest), not "
                              f"{b.base} (cache glob)", flush=True)
                    box_paths[b.box] = {"lr": b.lr, "base": p}
                else:
                    box_paths[b.box] = {"lr": b.lr, "base": b.base}
            replay_ds = ReplayCropDataset(
                entries, box_paths, crop_hr=crop_hr, scale_factor=scale,
                length=int(dcfg.get("train_length", 200000)),
                seed=int(tcfg.get("seed", 0)) + 17,
                redshift=float(dcfg.get("redshift", 0.0)),
                context_margin=context_margin,
            )
            replay_ds.set_weights(elite_weights(
                [e.marginal_contribution for e in entries],
                tau=float(distill_cfg.get("tau", 1.0)),
                a_max=float(distill_cfg.get("a_max", 5.0)),
                w_max=float(distill_cfg.get("w_max", 10.0)),
                normalize=str(distill_cfg.get("weight_normalize", "mean")),
            ))
            # ``paired_fraction`` sets the paired:elite ratio *within* a step.
            # Both terms are present at every step regardless -- dropping paired
            # batches for part of training is how the supervised anchor silently
            # disappears.
            pf = min(max(float(distill_cfg.get("paired_fraction", 0.5)), 1e-3), 0.999)
            elite_bs = max(1, int(round(bs * (1.0 - pf) / pf)))
            replay_it = _infinite(DataLoader(replay_ds, batch_size=elite_bs,
                                             num_workers=nw, drop_last=True))
            print(f"replay: {len(entries)} elite chunks, batch {bs} paired + "
                  f"{elite_bs} elite", flush=True)

    # --------------------------------------------------------------- model --
    sigma = mcfg.pop("sigma_res", None)
    if sigma is None:
        sigma = _sigma_from_batch(train_ds)
    model = build_residual_denoiser({**mcfg, "sigma_res": sigma,
                                     "scale_factor": scale}).to(device)
    teacher = None
    if mode == "distill" and float(distill_cfg.get("lambda_ref", 0.0)) > 0:
        teacher = build_residual_denoiser({**mcfg, "sigma_res": sigma,
                                           "scale_factor": scale}).to(device).eval()
        tp = distill_cfg.get("teacher_checkpoint") or distill_cfg.get("init_from")
        if tp:
            teacher.load_state_dict(_prior_weights(tp, prefer_ema=True))
        for p in teacher.parameters():
            p.requires_grad_(False)
    if mode == "distill" and distill_cfg.get("init_from"):
        model.load_state_dict(_prior_weights(distill_cfg["init_from"],
                                             prefer_ema=True))

    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 1e-4)),
                            weight_decay=float(tcfg.get("weight_decay", 0.0)))
    steps = int(tcfg.get("steps", 1000))
    sched = build_lr_schedule(opt, float(tcfg.get("lr", 1e-4)), steps,
                              warmup=int(tcfg.get("warmup", 0)),
                              min_lr_frac=float(tcfg.get("min_lr_frac", 1.0)))
    ema = ModelEMA(model, decay=float(tcfg.get("ema_decay", 0.999)))
    autocast, scaler = amp_components(bool(tcfg.get("amp", False)), device)

    st = TrainState()
    if resume and Path(resume).is_file():
        ck = torch.load(resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if ck.get("extra", {}).get("ema"):
            ema.module.load_state_dict(ck["extra"]["ema"])
        st.step = int(ck.get("step", 0))
        st.best_val = float(ck.get("extra", {}).get("best_val", st.best_val))
        if sched is not None:
            # Without this the LR restarts at the warmup floor and re-runs the
            # whole cosine decay from scratch, so a job that hits its time limit
            # comes back with a *different* schedule than the one it was
            # training under -- and a resumed run is not the run it resumed.
            sd = ck.get("extra", {}).get("scheduler")
            if sd:
                sched.load_state_dict(sd)
            else:
                for _ in range(st.step):
                    sched.step()
                print(f"! {resume} has no scheduler state; replayed {st.step} "
                      f"steps to reconstruct the LR", flush=True)
        print(f"resumed from {resume} at step {st.step} "
              f"(lr={opt.param_groups[0]['lr']:.3g})", flush=True)

    use_wandb = maybe_init_wandb(cfg, out_dir, job_type=f"reward_{mode}")
    logger = CSVLogger(out_dir, use_wandb=use_wandb)
    accum = max(1, int(tcfg.get("grad_accum", 1)))
    lam_ref = float(distill_cfg.get("lambda_ref", 0.0))
    lam_elite_target = float(distill_cfg.get("lambda_elite", 1.0))
    ramp = max(1, int(distill_cfg.get("lambda_elite_warmup_steps", 1)))
    paired_fraction = float(distill_cfg.get("paired_fraction", 0.5))
    abort = dict(distill_cfg.get("abort_on", {}))

    model.train()
    t_start = time.time()
    while st.step < steps:
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        agg: Dict[str, float] = {}
        lam_elite = lam_elite_target * min(1.0, st.step / ramp) if mode == "distill" else 0.0

        for _ in range(accum):
            b = next(train_it)
            with autocast():
                l_sup, aux = _sup_loss(model, b, diff, device)
            total = l_sup
            agg["loss_sup"] = agg.get("loss_sup", 0.0) + float(l_sup.detach())

            if mode == "distill":
                rb = next(replay_it)
                w = rb.get("weight")
                with autocast():
                    l_el, aux_el = _sup_loss(
                        model, rb, diff, device,
                        weight=None if w is None else w.to(device),
                    )
                total = total + lam_elite * l_el
                agg["loss_elite"] = agg.get("loss_elite", 0.0) + float(l_el.detach())
                if w is not None:
                    agg["elite_weight_mean"] = agg.get("elite_weight_mean", 0.0) + \
                        float(w.mean())
                    agg["elite_weight_max"] = max(
                        agg.get("elite_weight_max", 0.0), float(w.max())
                    )
                if teacher is not None and lam_ref > 0:
                    with torch.no_grad():
                        ref = teacher(aux["u_t"], aux["t"], **aux["cond"])
                    l_ref = F.mse_loss(aux["eps_hat"], ref)
                    total = total + lam_ref * l_ref
                    agg["loss_ref"] = agg.get("loss_ref", 0.0) + float(l_ref.detach())

            scaled = total / accum
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

        if scaler is not None:
            scaler.unscale_(opt)
        gn = grad_global_norm(model.parameters())
        if float(tcfg.get("grad_clip", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        if sched is not None:
            sched.step()
        ema.update(model)
        st.step += 1

        if st.step % int(tcfg.get("log_every", 50)) == 0 or st.step == 1:
            row = {k: v / accum if k.startswith("loss") else v for k, v in agg.items()}
            cur_lr = float(sched.get_last_lr()[0]) if sched is not None \
                else float(opt.param_groups[0]["lr"])
            row.update(lr=cur_lr, grad_norm=gn, lambda_elite=lam_elite)
            row.update(system_metrics(device, time.time() - t0))
            logger.log(st.step, row)

        if st.step % int(tcfg.get("val_every", 1000)) == 0 or st.step == steps:
            vrow = _validate(ema.module, val_loader, diff, device,
                             n_batches=int(tcfg.get("val_batches", 8)))
            vrow.update(_diagnostics(
                ema.module, next(iter(val_loader)), diff, device=device,
                scale_factor=scale,
                residual_scale=float(tcfg.get("residual_scale", 1.0)),
                n_samples=int(tcfg.get("diag_samples", 2)),
                boxsize_mpc_h=float(dcfg.get("boxsize_mpc_h", 100.0)),
                dis_norm_kpc_h=float(dcfg.get("dis_norm_kpc_h", 6000.0)),
                core=crop_hr if context_margin else None,
            ))
            logger.log(st.step, vrow)
            if mode == "distill":
                bad = _check_abort(vrow, agg, abort)
                if bad:
                    (out_dir / "ABORTED.json").write_text(
                        json.dumps({"step": st.step, "reasons": bad, "row": vrow},
                                   indent=2, default=float)
                    )
                    raise RuntimeError(f"distillation aborted at step {st.step}: {bad}")
            if vrow.get("val_loss", float("inf")) < st.best_val:
                st.best_val = float(vrow["val_loss"])
                save_checkpoint(out_dir / "ckpt_best.pt", model, opt, st.step,
                                extra={"ema": ema.module.state_dict(),
                                       "scheduler": sched.state_dict()
                                       if sched is not None else None,
                                       "best_val": st.best_val,
                                       "sigma_res": list(np.asarray(sigma).ravel()),
                                       "mode": mode, "val": vrow})

        if st.step % int(tcfg.get("ckpt_every", 2000)) == 0 or st.step == steps:
            save_checkpoint(out_dir / "ckpt_last.pt", model, opt, st.step,
                            extra={"ema": ema.module.state_dict(),
                                   "scheduler": sched.state_dict()
                                   if sched is not None else None,
                                   "best_val": st.best_val,
                                   "sigma_res": list(np.asarray(sigma).ravel()),
                                   "mode": mode})

    save_checkpoint(out_dir / "ckpt_last.pt", model, opt, st.step,
                    extra={"ema": ema.module.state_dict(),
                           "scheduler": sched.state_dict() if sched is not None else None,
                           "best_val": st.best_val,
                           "sigma_res": list(np.asarray(sigma).ravel()), "mode": mode})
    logger.close()
    print(f"finished {st.step} steps in {time.time() - t_start:.0f}s -> {out_dir}",
          flush=True)
    return out_dir


def _core_of(batch) -> Optional[int]:
    """The valid-core size of a batch, if the loader reported one."""
    c = batch.get("core_hr")
    if c is None:
        return None
    c = int(np.asarray(c).ravel()[0])
    return c if c > 0 else None


def _sup_loss(model, batch, diff: DiffusionConfig, device, weight=None):
    resid = batch["residual"].to(device, non_blocking=True)
    base = batch["psi_base"].to(device, non_blocking=True)
    y = batch["y_lr"].to(device, non_blocking=True)
    z = float(batch["redshift"][0]) if "redshift" in batch else 0.0
    u0 = whiten(resid, model.sigma_res)
    cond = {"y_lr": y, "psi_base": base, "redshift": z}
    loss, aux = denoising_loss(model, u0, cond, diff, per_sample_weight=weight,
                               core=_core_of(batch))
    aux["cond"] = cond
    return loss, aux


@torch.no_grad()
def _validate(model, loader, diff: DiffusionConfig, device, n_batches: int = 8) -> Dict[str, float]:
    model.eval()
    losses, rms = [], []
    for i, b in enumerate(loader):
        if i >= n_batches:
            break
        # Fixed timestep grid so validation is comparable across steps rather
        # than dominated by which t happened to be drawn.
        torch.manual_seed(1000 + i)
        loss, _ = _sup_loss(model, b, diff, device)
        losses.append(float(loss))
        rms.append(float(b["residual"].pow(2).mean().sqrt()))
    model.train()
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_target_residual_rms": float(np.mean(rms)) if rms else float("nan"),
    }


def _check_abort(vrow: Dict[str, float], agg: Dict[str, float], abort: Dict) -> List[str]:
    bad: List[str] = []
    if not abort:
        return bad
    rr = vrow.get("diag_residual_rms_pred", float("nan"))
    rt = vrow.get("diag_residual_rms_true", float("nan"))
    if np.isfinite(rr) and np.isfinite(rt) and rt > 0:
        ratio = rr / rt
        if ratio > float(abort.get("residual_rms_ratio_max", np.inf)):
            bad.append(f"residual RMS ratio {ratio:.2f} exceeds "
                       f"{abort['residual_rms_ratio_max']}")
    lk = vrow.get("diag_low_k_change", float("nan"))
    if np.isfinite(lk) and lk > float(abort.get("low_k_change_max", np.inf)):
        bad.append(f"low-k change {lk:.4f} exceeds {abort['low_k_change_max']}")
    dv = vrow.get("diag_sample_diversity", float("nan"))
    if np.isfinite(dv) and dv < float(abort.get("diversity_min", -np.inf)):
        bad.append(f"sample diversity {dv:.4f} below {abort['diversity_min']}")
    wmax = agg.get("elite_weight_max", 0.0)
    wmean = agg.get("elite_weight_mean", 0.0)
    limit = float(abort.get("elite_weight_ratio_max", np.inf))
    if wmean > 0 and np.isfinite(limit) and wmax / wmean > limit:
        bad.append(
            f"largest elite weight is {wmax / wmean:.1f}x the batch mean "
            f"(limit {limit}): the elite loss is a handful of crops"
        )
    return bad


def _prior_weights(path: str | Path, *, prefer_ema: bool = True) -> Dict:
    """The state dict a checkpoint should be *used* with.

    ``save_checkpoint`` stores the EMA weights under ``extra.ema``, not at the
    top level, so ``state.get("ema") or state["model"]`` silently fell through
    to the raw weights every single time. That matters here specifically because
    the oracle samples from ``extra.ema``: initialising the student from the raw
    weights, or referencing a raw-weight teacher, means distilling toward
    elites that a *different* model produced.
    """
    st = torch.load(str(path), map_location="cpu")
    ema = st.get("extra", {}).get("ema") or st.get("ema")
    if prefer_ema and ema:
        print(f"loaded EMA weights from {path}", flush=True)
        return ema
    if prefer_ema:
        print(f"! {path} has no EMA weights; using raw weights. The oracle "
              f"samples from EMA, so the elites came from a different model.",
              flush=True)
    return st["model"]


def _sigma_from_batch(ds, n: int = 8) -> List[float]:
    """Fallback whitening scale when the audit's sigma_res was not supplied."""
    acc = None
    for i in range(int(n)):
        r = ds[i]["residual"].numpy()
        s = (r.astype(np.float64) ** 2).mean(axis=(1, 2, 3))
        acc = s if acc is None else acc + s
    return list(np.sqrt(np.maximum(acc / float(n), 1e-24)))


class _SyntheticResidualCrops(Dataset):
    """Synthetic paired crops so the CPU/GPU smoke path needs no data on disk."""

    def __init__(self, crop_hr: int, scale_factor: int, length: int = 64, seed: int = 0):
        self.crop_hr = int(crop_hr)
        self.scale_factor = int(scale_factor)
        self.length = int(length)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed * 1013 + i)
        n, m = self.crop_hr, self.crop_hr // self.scale_factor
        return {
            "residual": 0.01 * torch.randn(6, n, n, n, generator=g),
            "psi_base": torch.randn(6, n, n, n, generator=g),
            "y_lr": torch.randn(6, m, m, m, generator=g),
            "redshift": torch.tensor(0.0),
        }
