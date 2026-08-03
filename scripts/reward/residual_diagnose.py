#!/usr/bin/env python
"""Decompose the residual prior's failure: wrong size, wrong direction, or both.

Gate A says the composed field is destroyed (density sigma ratio 0.10, low_k
change 1.09). That single verdict cannot say *why*, and the three candidate
explanations need different fixes:

* **amplitude** -- the residual has the right shape and too much of it. Fixed by
  ``residual_scale``, no retraining. Diagnosed by the alpha sweep: compose
  ``Psi_base + a * dPsi_hat`` for several ``a`` and see whether the field
  metrics recover at some ``a < 1``.
* **direction** -- the residual is incoherent with the true one, so shrinking it
  only makes it harmless (at ``a = 0`` you recover SR2 exactly, which is not an
  improvement). Diagnosed by ``r(k)`` between ``dPsi_hat`` and the *true*
  ``dPsi = Psi_HR - Psi_base``. This is the measurement nothing in the pipeline
  currently makes: every existing diagnostic compares the *composed field* to
  HR, where the base dominates and a useless residual still scores well.
* **conditioning** -- the model ignores ``Psi_base`` / ``y_lr`` and has learned
  generic denoising. Diagnosed by re-sampling with the conditioning taken from a
  *different* crop: if ``r(k)`` and the field metrics barely move, the
  conditioning is decorative.

Every row also carries the **oracle row** ``a = 1`` composed with the TRUE
residual. Without it there is no way to read the other numbers: it turns out the
true residual itself scores ``low_k_change ~ 0.3``, well above the calibrated
ceiling of 0.14, so "fails the low-k constraint" does not by itself mean "the
sample is bad". A reference the constraints were never checked against.

The per-step clip profile is recorded in full. ``train.py`` logs only the max
and the last step, which cannot distinguish "clipped once at t_max, harmless"
from "clipped throughout, the bound is setting the amplitude".

    python scripts/reward/residual_diagnose.py \
        --checkpoint $ZFS/DMSR/dmsr_reward/checkpoints/residual_prior/ckpt_best.pt \
        --n-crops 8 --alphas 0,0.1,0.2,0.33,0.5,1.0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from _common import (add_common_args, append_jsonl, banner, load_reward_config,
                     select_device, split_boxes, write_json)

from cosmo_sr.utils.config import load_config

from cosmo_sr.dmsr.evaluate import power_error, rk_tk_summary
from cosmo_sr.eval.density import cic_density_valid_center
from cosmo_sr.operators.multiscale import block_average
from cosmo_sr.reward import paths
from cosmo_sr.reward.diffusion import DiffusionConfig, ddim_sample, valid_core
from cosmo_sr.reward.targets import PairedResidualCrops, resolve_boxes

from sample_oracle_candidates import ckpt_identity, load_model


def _field_metrics(composed: torch.Tensor, base: torch.Tensor, hr: torch.Tensor,
                   y: torch.Tensor, *, scale_factor: int, boxsize_mpc_h: float,
                   dis_norm_kpc_h: float, prefix: str = "") -> Dict[str, float]:
    """The Gate-A field criteria, on one already-cored crop.

    Deliberately the same formulas as ``train.py::_diagnostics`` so a number here
    is comparable to the training curve rather than a second opinion computed a
    slightly different way.
    """
    out: Dict[str, float] = {}
    a_hat = block_average(composed, scale_factor)
    a_base = block_average(base, scale_factor)
    out[f"{prefix}low_k_change"] = float(
        (a_hat - a_base).pow(2).mean().sqrt()
        / a_base.pow(2).mean().sqrt().clamp_min(1e-12))
    out[f"{prefix}lr_consistency"] = float(
        (a_hat - y).pow(2).mean().sqrt() / y.pow(2).mean().sqrt().clamp_min(1e-12))
    out.update(rk_tk_summary(composed[:, 0:3], hr[:, 0:3], factor=scale_factor,
                             prefix=f"{prefix}disp_"))
    out[f"{prefix}displacement_power_error"] = power_error(composed[:, 0:3],
                                                           hr[:, 0:3])
    cell = boxsize_mpc_h * 1000.0 / 512.0
    region = int(base.shape[-1]) // 2
    try:
        d_pred = cic_density_valid_center(composed[:, 0:3], cell, dis_norm_kpc_h, region)
        d_true = cic_density_valid_center(hr[:, 0:3], cell, dis_norm_kpc_h, region)
        out[f"{prefix}density_sigma_ratio"] = float(
            d_pred.std() / d_true.std().clamp_min(1e-12))
        out[f"{prefix}density_power_error"] = power_error(d_pred, d_true)
        out.update(rk_tk_summary(d_pred, d_true, factor=1, prefix=f"{prefix}dens_"))
    except Exception as exc:                      # crops below the binning minimum
        out[f"{prefix}density_error"] = repr(exc)
    return out


def _coherence(pred: torch.Tensor, true: torch.Tensor, *, scale_factor: int,
               prefix: str = "coh_") -> Dict[str, float]:
    """Is the predicted residual pointing the same way as the true one?

    ``r(k)`` between the two RESIDUALS -- not between composed fields. A residual
    that is pure noise of the right amplitude still gives a composed field whose
    ``r(k)`` against HR is dominated by ``Psi_base``; only this comparison can
    see it. ``r = 0`` means the residual carries no information about the
    correction at that scale, whatever its amplitude.
    """
    out = rk_tk_summary(pred[:, 0:3], true[:, 0:3], factor=scale_factor,
                        prefix=f"{prefix}disp_")
    p = pred.flatten().double()
    t = true.flatten().double()
    out[f"{prefix}cosine"] = float(
        (p @ t) / (p.norm() * t.norm()).clamp_min(1e-30))
    # Per-channel amplitude ratio: the "how much too big" number, kept per
    # channel because displacement and velocity differ by 8x in these units and
    # a channel-mixed RMS blends them into something uninterpretable.
    rp = pred.double().pow(2).mean(dim=(0, 2, 3, 4)).sqrt()
    rt = true.double().pow(2).mean(dim=(0, 2, 3, 4)).sqrt()
    out[f"{prefix}amp_ratio_per_channel"] = [
        float(a / b) if b > 0 else float("nan") for a, b in zip(rp, rt)]
    out[f"{prefix}pred_rms_per_channel"] = [float(v) for v in rp]
    out[f"{prefix}true_rms_per_channel"] = [float(v) for v in rt]
    return out


@torch.no_grad()
def _sample(model, base: torch.Tensor, y: torch.Tensor, z: float,
            diff: DiffusionConfig, *, seed: int, device) -> tuple:
    """One residual draw, plus the FULL per-step clip profile.

    The generator lives on the sampling device because ``ddim_sample`` draws the
    starting noise itself, with ``device=device``, and torch refuses a CPU
    generator there. Same seed + same device = same start, which is what the
    matched/shuffled comparison needs: the two arms must differ only in their
    conditioning, never in the noise they started from.
    """
    from cosmo_sr.reward.diffusion import unwhiten

    gen = torch.Generator(device=device).manual_seed(int(seed))
    clip_log: List[Dict] = []
    u = ddim_sample(model, tuple(base.shape),
                    {"y_lr": y, "psi_base": base, "redshift": z},
                    diff, device=device, generator=gen, clip_log=clip_log)
    return unwhiten(u, model.sigma_res), clip_log


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", default="configs/reward/residual_prior.yaml")
    ap.add_argument("--boxes", default=None, help="default = val split")
    ap.add_argument("--n-crops", type=int, default=8)
    ap.add_argument("--alphas", default="0,0.1,0.2,0.33,0.5,1.0")
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--t-max", type=float, default=None,
                    help="override the sampler t_max; 0.95-0.98 removes the "
                         "endpoint clip, see the x0 division by alpha(t_max)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--shuffle-cond", action="store_true",
                    help="also sample with psi_base/y_lr taken from ANOTHER crop")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    # SMOKE ONLY. The measured numbers depend on both: a core below the
    # receptive-field diameter is contaminated by its own padding, and a margin
    # below 48 is not the neighbourhood full-box tiling supplies. Use them to
    # check the code path on a CPU in seconds, never to produce a verdict.
    ap.add_argument("--crop-hr", type=int, default=None,
                    help="override the core size (smoke runs only)")
    ap.add_argument("--context-margin", type=int, default=None,
                    help="override the context margin (smoke runs only)")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    # load_config, not yaml.safe_load: residual_prior.yaml carries
    # `base: reward.yaml` and only states what differs, so a raw parse returns a
    # config with no data.scale_factor and load_model dies on the KeyError.
    # Assembled exactly as sample_oracle_candidates.py does, so the model this
    # job measures is byte-identical to the one the oracle samples.
    mcfg = load_config(args.model_config)
    cfg_model = {**cfg, "model": mcfg.get("model", {}),
                 "diffusion": mcfg.get("diffusion", {})}
    d = cfg["data"]
    scale = int(d["scale_factor"])
    boxes = [b.strip() for b in args.boxes.split(",")] if args.boxes \
        else split_boxes(cfg, "val")
    device = select_device(args.device)

    out_dir = Path(args.out) if args.out else paths.AUDITS("residual_diagnose",
                                                           create=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "residual_diagnose.jsonl"
    if rows_path.exists():
        rows_path.unlink()

    model = load_model(args.checkpoint, cfg_model, device, use_ema=not args.no_ema)
    # crop_hr / context_margin live in the MODEL config's data block (the merged
    # child overrides reward.yaml's data), so read them from mcfg, not from cfg.
    dcfg = mcfg.get("data", {})
    crop_hr = int(args.crop_hr or dcfg.get("crop_hr", 64))
    margin = int(dcfg.get("context_margin", 48) if args.context_margin is None
                 else args.context_margin)
    smoke_geom = (args.crop_hr is not None or args.context_margin is not None)
    if smoke_geom:
        print(f"  !! SMOKE GEOMETRY core={crop_hr} margin={margin}, not the "
              f"trained {dcfg.get('crop_hr')}/{dcfg.get('context_margin')}. "
              f"Code-path check only -- these numbers are not a verdict.",
              flush=True)
    diff = DiffusionConfig(**{k: v for k, v in cfg_model.get("diffusion", {}).items()
                              if k in DiffusionConfig.__dataclass_fields__})
    if args.n_steps:
        diff.n_steps = int(args.n_steps)
    if args.t_max:
        diff.t_max = float(args.t_max)

    banner(f"residual diagnosis on {boxes}: {args.n_crops} crops, "
           f"core {crop_hr}^3 + {margin} context, {diff.n_steps} steps, "
           f"t_max={diff.t_max}")
    print(f"  checkpoint: {args.checkpoint} (ema={not args.no_ema})")
    print(f"  sigma_res:  {[round(float(v), 4) for v in model.sigma_res]}")

    bx = resolve_boxes(boxes, d["root"], base_seed=args.base_seed)
    ds = PairedResidualCrops(bx, crop_hr=crop_hr, scale_factor=scale,
                             length=max(int(args.n_crops), 2),
                             seed=args.seed + 991,
                             redshift=float(d.get("redshift", 0.0)),
                             context_margin=margin)
    alphas = [float(a) for a in args.alphas.split(",")]
    z = float(d.get("redshift", 0.0))
    kw = dict(scale_factor=scale, boxsize_mpc_h=float(d.get("boxsize_mpc_h", 100.0)),
              dis_norm_kpc_h=float(d.get("dis_norm_kpc_h", 6000.0)))

    t0 = time.time()
    for i in range(int(args.n_crops)):
        item = ds[i]
        base = item["psi_base"][None].to(device)
        y = item["y_lr"][None].to(device)
        rtrue = item["residual"][None].to(device)
        core = int(np.asarray(item["core_hr"]).ravel()[0]) if "core_hr" in item \
            else crop_hr

        resid, clip_log = _sample(model, base, y, z, diff, seed=args.seed + i,
                                  device=device)

        cb, ct, cr, cy = (valid_core(base, core), valid_core(rtrue, core),
                          valid_core(resid, core),
                          valid_core(y, max(core // scale, 1)))
        hr = cb + ct

        common = {
            "crop": i, "boxes": boxes, "checkpoint": str(args.checkpoint),
            "ckpt": ckpt_identity(args.checkpoint), "ema": not args.no_ema,
            "n_steps": diff.n_steps, "t_max": diff.t_max, "seed": args.seed + i,
            "core_hr": core, "context_margin": margin,
            "clip_profile": [round(float(r["clip_fraction"]), 5) for r in clip_log],
            "clip_t": [round(float(r["t"]), 4) for r in clip_log],
            "clip_rms_sigma": [round(float(r["rms_sigma"]), 4) for r in clip_log],
        }

        # --- the reference row: what the TRUE residual scores ---------------
        row = {**common, "arm": "oracle_true_residual", "alpha": 1.0}
        row.update(_field_metrics(cb + ct, cb, hr, cy, prefix="", **kw))
        append_jsonl(rows_path, row)

        # --- the amplitude sweep -------------------------------------------
        coh = _coherence(cr, ct, scale_factor=scale)
        for a in alphas:
            row = {**common, "arm": "sampled", "alpha": a, **coh}
            row.update(_field_metrics(cb + a * cr, cb, hr, cy, prefix="", **kw))
            append_jsonl(rows_path, row)

        print(f"[crop {i}] clip@0={common['clip_profile'][0]:.3f} "
              f"clip@last={common['clip_profile'][-1]:.4f}  "
              f"coh_rk_high={coh.get('coh_disp_rk_high', float('nan')):.4f} "
              f"cosine={coh['coh_cosine']:+.4f}  "
              f"amp_ratio={[round(v, 2) for v in coh['coh_amp_ratio_per_channel']]}",
              flush=True)

        # --- the conditioning test -----------------------------------------
        if args.shuffle_cond:
            other = ds[(i + 1) % max(int(args.n_crops), 2)]
            base_s = other["psi_base"][None].to(device)
            y_s = other["y_lr"][None].to(device)
            resid_s, clip_s = _sample(model, base_s, y_s, z, diff,
                                      seed=args.seed + i, device=device)
            # Compare the SHUFFLED-conditioning sample against THIS crop's true
            # residual. If the model uses its conditioning, this must collapse
            # relative to the matched arm above.
            coh_s = _coherence(valid_core(resid_s, core), ct, scale_factor=scale)
            row = {**common, "arm": "shuffled_cond", "alpha": 1.0, **coh_s,
                   "clip_profile": [round(float(r["clip_fraction"]), 5)
                                    for r in clip_s]}
            # and how much the SAMPLE ITSELF changed when the conditioning did
            row["delta_vs_matched_cosine"] = float(
                (valid_core(resid_s, core).flatten().double()
                 @ cr.flatten().double())
                / (valid_core(resid_s, core).norm().double()
                   * cr.norm().double()).clamp_min(1e-30))
            append_jsonl(rows_path, row)
            print(f"          shuffled: coh_rk_high="
                  f"{coh_s.get('coh_disp_rk_high', float('nan')):.4f} "
                  f"cosine={coh_s['coh_cosine']:+.4f}  "
                  f"sample-vs-matched cosine="
                  f"{row['delta_vs_matched_cosine']:+.4f}", flush=True)

    write_json(out_dir / "residual_diagnose_meta.json", {
        "alphas": alphas, "boxes": boxes, "n_crops": int(args.n_crops),
        "checkpoint": str(args.checkpoint), "ckpt": ckpt_identity(args.checkpoint),
        "ema": not args.no_ema, "n_steps": diff.n_steps, "t_max": diff.t_max,
        "crop_hr": crop_hr, "context_margin": margin,
        "sigma_res": [float(v) for v in model.sigma_res],
        "shuffle_cond": bool(args.shuffle_cond),
        "smoke_geometry": bool(smoke_geom),
        "seconds": time.time() - t0,
        "rows": str(rows_path),
    })
    banner(f"rows -> {rows_path}")
    print("=== report with: sbatch scripts/slurm/residual_diagnose_report_cpu.sbatch")


if __name__ == "__main__":
    main()
