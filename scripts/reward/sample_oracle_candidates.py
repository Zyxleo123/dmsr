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
import hashlib
import json
import time
from pathlib import Path

from typing import Dict, Optional

import numpy as np
import torch

from _common import (add_common_args, banner, load_reward_config, lr_path,
                     parse_boxes, select_device, split_boxes, write_json)

from cosmo_sr.reward import paths
from cosmo_sr.reward.base import find_base_field
from cosmo_sr.reward.diffusion import DiffusionConfig
from cosmo_sr.reward.model import build_residual_denoiser
from cosmo_sr.reward.sampling import (TileSpec, measure_receptive_field,
                                      sample_residual_box, tile_margin_for)


def load_model(ckpt_path: str, cfg: dict, device, use_ema: bool = True):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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


def field_fingerprint(**parts) -> str:
    """Hash of everything a saved field depends on.

    A run name is not provenance. Reusing one with a different checkpoint,
    sampler, tiling or dtype used to keep the old ``.npy`` and write a new
    manifest describing it, and nothing downstream could tell -- the evaluation
    stage then aggregated whatever JSON happened to be in the directory. The
    fingerprint is written next to each field and compared before any reuse, so a
    changed input regenerates instead of being relabelled.

    The checkpoint enters by size and mtime rather than by content hash: these
    are multi-hundred-MB files read from scratch, and a checkpoint that is
    overwritten in place always changes at least its mtime.
    """
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def ckpt_identity(path) -> Dict:
    p = Path(path).resolve()
    st = p.stat()
    return {"path": str(p), "size": int(st.st_size), "mtime": int(st.st_mtime)}


def read_fingerprint(path: Path) -> Optional[str]:
    p = path.with_suffix(".fingerprint.json")
    if not p.is_file():
        return None
    try:
        return str(json.loads(p.read_text()).get("fingerprint"))
    except (ValueError, OSError):
        return None


def write_fingerprint(path: Path, fp: str, parts: Dict) -> None:
    path.with_suffix(".fingerprint.json").write_text(
        json.dumps({"fingerprint": fp, "inputs": parts}, indent=2,
                   sort_keys=True, default=str)
    )


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

    common = {
        "checkpoint": ckpt_identity(args.checkpoint),
        "use_ema": not args.no_ema,
        "model_config": hashlib.sha256(
            Path(args.model_config).read_bytes()).hexdigest()[:32],
        "diffusion": diff.__dict__,
        "tile": {"core": spec.core, "margin": spec.margin},
        "dtype": args.dtype,
        "redshift": float(cfg["data"].get("redshift", 0.0)),
    }

    rows = []
    for box in boxes:
        lr = np.load(lr_path(cfg, box))
        base_path = find_base_field(box, args.base_seed)
        if base_path is None:
            raise SystemExit(f"no cached SR2 base for {box}")
        base = np.load(base_path, mmap_mode="r")
        for j in range(k):
            seed = int(args.seed0) + j
            path = fields / f"{box}_resid_seed{seed}.npy"
            parts = {**common, "box": box, "seed": seed,
                     "base": str(Path(base_path).resolve())}
            fp = field_fingerprint(**parts)
            if path.is_file() and not args.overwrite:
                old = read_fingerprint(path)
                if old == fp:
                    print(f"[{box} k={j}] exists", flush=True)
                    rows.append({"box": box, "seed": seed, "residual": str(path),
                                 "fingerprint": fp, "regenerated": False})
                    continue
                # Same filename, different inputs. Keeping it would put a field
                # from another checkpoint/sampler under this run's manifest.
                print(f"[{box} k={j}] REGENERATING: {path.name} was produced "
                      f"with different inputs (fingerprint "
                      f"{old or 'missing'} != {fp})", flush=True)
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
            write_fingerprint(path, fp, parts)
            dt = time.time() - t0
            rms = float(np.sqrt(np.mean(resid[0:3].astype(np.float64) ** 2)))
            print(f"[{box} k={j}] seed={seed} rms={rms:.4g} "
                  f"clip max={100 * clip_max:.3f}% last={100 * clip_last:.3f}% "
                  f"({dt:.0f}s)", flush=True)
            rows.append({
                "box": box, "seed": seed, "residual": str(path),
                "fingerprint": fp,
                "residual_rms_disp": rms, "seconds": dt, "regenerated": True,
                "base": str(base_path),
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
