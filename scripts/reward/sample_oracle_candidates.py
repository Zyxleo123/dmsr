#!/usr/bin/env python
"""Stage 6a (GPU): generate K residual realisations per validation box.

The GPU job only *generates and saves*. Halo finding, constraint evaluation and
reward computation are separate CPU stages, so a GPU allocation is never held
while Rockstar runs for a quarter of an hour.

Saves the **residual** rather than the composed field: the frozen SR2 base is
already cached and shared, so storing ``dPsi`` keeps one 3.2 GB array per
candidate instead of two, and the replay buffer needs exactly this array.

    python scripts/reward/sample_oracle_candidates.py \
        --checkpoint $ZFS/dmsr_reward/checkpoints/residual_prior/ckpt_best.pt \
        --boxes set8,set9 --k 8 --run-name prior_k8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from _common import (add_common_args, banner, load_reward_config, lr_path,
                     parse_boxes, select_device, split_boxes, write_json)

from cosmo_sr.reward import paths
from cosmo_sr.reward.diffusion import DiffusionConfig
from cosmo_sr.reward.model import build_residual_denoiser
from cosmo_sr.reward.sampling import (TileSpec, measure_receptive_field,
                                      sample_residual_box, tile_margin_for)


def load_model(ckpt_path: str, cfg: dict, device, use_ema: bool = True):
    state = torch.load(ckpt_path, map_location="cpu")
    mcfg = dict(cfg.get("model", {}))
    mcfg.pop("sigma_res", None)
    sigma = state.get("extra", {}).get("sigma_res")
    model = build_residual_denoiser({
        **mcfg, "sigma_res": sigma,
        "scale_factor": int(cfg["data"]["scale_factor"]),
    })
    sd = state["extra"]["ema"] if (use_ema and state.get("extra", {}).get("ema")) \
        else state["model"]
    model.load_state_dict(sd)
    return model.to(device).eval()


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--model-config", default="configs/reward/residual_prior.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--boxes", default=None, help="comma list; default = val split")
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "dev", "final"])
    ap.add_argument("--k", type=int, default=None, help="candidates per box")
    ap.add_argument("--seed0", type=int, default=0, help="first residual seed")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument("--run-name", default="oracle_round0")
    ap.add_argument("--tile-core", type=int, default=128)
    ap.add_argument("--tile-margin", type=int, default=48)
    ap.add_argument("--tile-batch", type=int, default=1)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    from cosmo_sr.utils.config import load_config
    mcfg = load_config(args.model_config)
    cfg_model = {**cfg, "model": mcfg.get("model", {}),
                 "diffusion": mcfg.get("diffusion", {})}

    boxes = parse_boxes(args.boxes, cfg, args.split)
    k = int(args.k or cfg.get("reward", {}).get("num_ensembles_K", 8))
    device = select_device(args.device)
    out = paths.ORACLE(args.run_name, create=True)
    fields = out / "fields"
    fields.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, cfg_model, device, use_ema=not args.no_ema)
    diff = DiffusionConfig(**{
        k2: v for k2, v in dict(cfg_model.get("diffusion", {})).items()
        if k2 in DiffusionConfig.__dataclass_fields__
    })
    if args.n_steps:
        diff.n_steps = int(args.n_steps)

    ng_hr = int(cfg["data"]["ng_hr"])
    scale = int(cfg["data"]["scale_factor"])
    spec = TileSpec(ng_hr, core=args.tile_core, margin=args.tile_margin,
                    scale_factor=scale)
    # Measured once here, on the CPU-cheap structural probe, and then skipped per
    # sample (verify_margin=False). Too small a margin is a silent seam, so this is
    # a hard exit, not a warning.
    rf = measure_receptive_field(
        model, channels=int(cfg_model["model"].get("channels", 6)),
        scale_factor=scale, device=device,
    )
    need = tile_margin_for(rf, scale)
    banner(f"tile core={spec.core} margin={spec.margin}; receptive field {rf} "
           f"-> minimum margin {need}")
    if spec.margin < need:
        raise SystemExit(
            f"receptive-field half-width {rf} needs --tile-margin {need}, got "
            f"{spec.margin}; valid-core tiling would leak tile padding into the "
            f"written core"
        )

    rows = []
    for box in boxes:
        lr = np.load(lr_path(cfg, box))
        hits = sorted(paths.SR2_BASE_CACHE().glob(f"{box}_seed{args.base_seed}_*.npy"))
        if not hits:
            raise SystemExit(f"no cached SR2 base for {box}")
        base = np.load(hits[0], mmap_mode="r")
        for j in range(k):
            seed = int(args.seed0) + j
            path = fields / f"{box}_resid_seed{seed}.npy"
            if path.is_file() and not args.overwrite:
                print(f"[{box} k={j}] exists", flush=True)
                rows.append({"box": box, "seed": seed, "residual": str(path),
                             "regenerated": False})
                continue
            t0 = time.time()
            clip_log: list = []
            resid = sample_residual_box(
                model, np.asarray(base), lr, seed=seed, cfg=diff, spec=spec,
                device=device, redshift=float(cfg["data"].get("redshift", 0.0)),
                tile_batch=int(args.tile_batch), verify_margin=False,
                clip_log=clip_log,
            )
            # Clipping at x0_clip is what keeps the early DDIM steps finite; a
            # large fraction means the bound, not the model, set the amplitude.
            clip_max = max((c["clip_fraction"] for c in clip_log), default=0.0)
            clip_last = clip_log[-1]["clip_fraction"] if clip_log else 0.0
            if args.dtype == "float16":
                resid = resid.astype(np.float16)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, resid)
            tmp.replace(path)
            dt = time.time() - t0
            rms = float(np.sqrt(np.mean(resid[0:3].astype(np.float64) ** 2)))
            print(f"[{box} k={j}] seed={seed} rms={rms:.4g} "
                  f"clip max={100 * clip_max:.3f}% last={100 * clip_last:.3f}% "
                  f"({dt:.0f}s)", flush=True)
            rows.append({
                "box": box, "seed": seed, "residual": str(path),
                "residual_rms_disp": rms, "seconds": dt, "regenerated": True,
                "base": str(hits[0]),
                "x0_clip": float(diff.x0_clip),
                "clip_fraction_max": clip_max,
                "clip_fraction_final_step": clip_last,
                "clip_log": clip_log,
            })

    write_json(out / "candidates.json", {
        "run_name": args.run_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "use_ema": not args.no_ema,
        "model_config": str(Path(args.model_config).resolve()),
        "boxes": boxes,
        "K": k,
        "seed0": int(args.seed0),
        "base_seed": int(args.base_seed),
        "residual_scale": float(args.residual_scale),
        "diffusion": diff.__dict__,
        "tile": {"core": spec.core, "margin": spec.margin,
                 "receptive_field_halfwidth": int(rf)},
        "dtype": args.dtype,
        "candidates": rows,
    })
    banner(f"{len(rows)} candidates -> {out}")


if __name__ == "__main__":
    main()
