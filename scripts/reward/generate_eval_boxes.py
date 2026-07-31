#!/usr/bin/env python
"""Stage 9a (GPU): full-box samples for every arm of the final comparison.

Arms:

``sr2``      the frozen baseline (already cached; only linked, never resampled);
``prior``    ``n_samples`` ordinary random draws from the supervised prior;
``distill``  ``n_samples`` ordinary random draws from the distilled student;
``bestofk``  the oracle's selected sample, copied in by the CPU stage.

The distilled arm gets exactly the same number of random samples as the prior
and **no selection**: the headline claim is about average samples, so the code
never offers a "pick the best" switch here.

    python scripts/reward/generate_eval_boxes.py --run-name final \
        --prior-ckpt .../residual_prior/ckpt_best.pt \
        --distill-ckpt .../distill_round0/ckpt_best.pt \
        --boxes set12,set13,set14,set15 --n-samples 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from _common import (add_common_args, banner, load_reward_config, lr_path,
                     parse_boxes, select_device, write_json)
from sample_oracle_candidates import load_model

from cosmo_sr.reward import paths
from cosmo_sr.reward.base import find_base_field
from cosmo_sr.reward.diffusion import DiffusionConfig
from cosmo_sr.reward.sampling import (TileSpec, measure_receptive_field,
                                      sample_residual_box, tile_margin_for)


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--run-name", default="final")
    ap.add_argument("--model-config", default="configs/reward/residual_prior.yaml")
    ap.add_argument("--prior-ckpt", default=None)
    ap.add_argument("--distill-ckpt", default=None)
    ap.add_argument("--boxes", default=None, help="default = test split")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "dev", "final"])
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument("--tile-core", type=int, default=128)
    ap.add_argument("--tile-margin", type=int, default=48)
    ap.add_argument("--tile-batch", type=int, default=1)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    from cosmo_sr.utils.config import load_config
    mc = load_config(args.model_config)
    cfg_model = {**cfg, "model": mc.get("model", {}), "diffusion": mc.get("diffusion", {})}

    boxes = parse_boxes(args.boxes, cfg, args.split)
    device = select_device(args.device)
    out = paths.EVAL(args.run_name, create=True)
    fields = out / "fields"
    fields.mkdir(parents=True, exist_ok=True)

    diff = DiffusionConfig(**{
        k: v for k, v in dict(cfg_model.get("diffusion", {})).items()
        if k in DiffusionConfig.__dataclass_fields__
    })
    if args.n_steps:
        diff.n_steps = int(args.n_steps)

    ng_hr = int(cfg["data"]["ng_hr"])
    sf = int(cfg["data"]["scale_factor"])
    spec = TileSpec(ng_hr, core=args.tile_core, margin=args.tile_margin, scale_factor=sf)

    arms = {}
    if args.prior_ckpt:
        arms["prior"] = load_model(args.prior_ckpt, cfg_model, device,
                                   use_ema=not args.no_ema)
    if args.distill_ckpt:
        arms["distill"] = load_model(args.distill_ckpt, cfg_model, device,
                                     use_ema=not args.no_ema)
    if not arms:
        raise SystemExit("nothing to generate: pass --prior-ckpt and/or --distill-ckpt")

    for name, m in arms.items():
        rf = measure_receptive_field(
            m, channels=int(cfg_model["model"].get("channels", 6)),
            scale_factor=sf, device=device)
        need = tile_margin_for(rf, sf)
        if spec.margin < need:
            raise SystemExit(
                f"[{name}] receptive-field half-width {rf} needs --tile-margin "
                f"{need}, got {spec.margin}"
            )
        banner(f"{name}: receptive field {rf}, minimum margin {need} "
               f"<= {spec.margin}")

    rows = []
    for box in boxes:
        lr = np.load(lr_path(cfg, box))
        base_path = find_base_field(box, args.base_seed)
        if base_path is None:
            raise SystemExit(f"no cached SR2 base for {box}")
        base = np.load(base_path, mmap_mode="r")
        rows.append({"arm": "sr2", "box": box, "sample": 0, "seed": args.base_seed,
                     "field": str(base_path), "residual": None})

        for name, model in arms.items():
            for s in range(int(args.n_samples)):
                seed = int(args.seed0) + s
                rp = fields / f"{box}_{name}_resid_seed{seed}.npy"
                if rp.is_file() and not args.overwrite:
                    rows.append({"arm": name, "box": box, "sample": s, "seed": seed,
                                 "residual": str(rp), "base": str(base_path),
                                 "regenerated": False})
                    print(f"[{box} {name} s={s}] exists", flush=True)
                    continue
                t0 = time.time()
                resid = sample_residual_box(
                    model, np.asarray(base), lr, seed=seed, cfg=diff, spec=spec,
                    device=device, redshift=float(cfg["data"].get("redshift", 0.0)),
                    tile_batch=int(args.tile_batch), verify_margin=False,
                )
                tmp = rp.with_suffix(".tmp.npy")
                np.save(tmp, resid)
                tmp.replace(rp)
                rms = float(np.sqrt(np.mean(resid[0:3].astype(np.float64) ** 2)))
                print(f"[{box} {name} s={s}] seed={seed} rms={rms:.4g} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                rows.append({"arm": name, "box": box, "sample": s, "seed": seed,
                             "residual": str(rp), "base": str(base_path),
                             "residual_rms_disp": rms, "regenerated": True})

    write_json(out / "eval_fields.json", {
        "run_name": args.run_name,
        "boxes": boxes, "split": args.split,
        "n_samples": int(args.n_samples),
        "residual_scale": float(args.residual_scale),
        "prior_ckpt": args.prior_ckpt, "distill_ckpt": args.distill_ckpt,
        "use_ema": not args.no_ema,
        "diffusion": diff.__dict__,
        "tile": {"core": spec.core, "margin": spec.margin},
        "selection": "none -- these are ordinary random samples for every arm",
        "fields": rows,
    })
    banner(f"{len(rows)} entries -> {out / 'eval_fields.json'}")


if __name__ == "__main__":
    main()
