#!/usr/bin/env python
"""Rung 2 (GPU): one CEM generation over the diffusion noise.

Gate B asks whether good residuals appear *by chance* in best-of-K sampling.
This asks the different question of whether they **exist** in the action space
at all, by searching the noise instead of resampling it independently: score a
population, keep the elites, and draw the next population near them.

One invocation is one iteration's generation step. Iteration 0 samples the
population from seeds (identical mechanics to best-of-K); later iterations read
the previous iteration's ``elites.npz`` and perturb those noise vectors. The
noise that produced each candidate is saved next to the residual, because the
selection step has to hand the winners forward.

Candidates keep the ``seed`` key that ``score_oracle.py`` indexes on -- here it
is a unique candidate id (``iteration * 1000 + j``), not an RNG seed -- so the
scoring and reporting stages run unmodified.

    python scripts/reward/cem_search.py --iteration 0 --run-name cem_a \
        --checkpoint $ZFS/dmsr_reward/checkpoints/residual_prior/ckpt_best.pt
"""
from __future__ import annotations

import argparse
import json
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


def run_dir(run_name: str, iteration: int, create: bool = False) -> Path:
    return paths.ORACLE(f"{run_name}_it{int(iteration)}", create=create)


def candidate_id(iteration: int, j: int) -> int:
    return int(iteration) * 1000 + int(j)


def perturb(elite: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """A child noise vector near ``elite``.

    Renormalised to the unit-variance shell the sampler's own draw lives on: the
    DDIM schedule assumes ``u ~ N(0, I)`` at ``t_max``, and an unnormalised
    ``elite + sigma * xi`` has variance ``1 + sigma^2``, which would inflate the
    residual amplitude with every iteration and confound "search found a better
    field" with "search turned the amplitude up".
    """
    child = elite.astype(np.float32) + np.float32(sigma) * rng.standard_normal(
        elite.shape, dtype=np.float32
    )
    return child / np.float32(np.sqrt(1.0 + float(sigma) ** 2))


def load_elites(run_name: str, iteration: int, box: str):
    """Elite noise vectors of the previous iteration, best first."""
    prev = run_dir(run_name, iteration - 1)
    p = prev / "elites.npz"
    if not p.is_file():
        raise SystemExit(
            f"iteration {iteration} needs {p}; the previous iteration's "
            f"selection step did not run"
        )
    with np.load(p) as z:
        keys = sorted(k for k in z.files if k.startswith(f"{box}__noise"))
        if not keys:
            raise SystemExit(f"no elites for box {box} in {p}")
        return [np.asarray(z[k], dtype=np.float32) for k in keys]


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--model-config", default="configs/reward/residual_prior.yaml")
    ap.add_argument("--cem-config", default="configs/reward/cem.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run-name", required=True, help="base name; per-iteration dirs get _it{i}")
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--boxes", default=None, help="comma list; default = cem config")
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "dev", "final"])
    ap.add_argument("--population", type=int, default=None)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--residual-scale", type=float, default=1.0)
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
    ccfg = dict(load_config(args.cem_config).get("cem", {}))
    cfg_model = {**cfg, "model": mcfg.get("model", {}),
                 "diffusion": mcfg.get("diffusion", {})}

    it = int(args.iteration)
    P = int(args.population or ccfg.get("population", 8))
    E = int(ccfg.get("elites", 2))
    sigma = float(ccfg.get("sigma0", 0.3)) * float(ccfg.get("decay", 0.7)) ** max(0, it - 1)
    boxes = parse_boxes(args.boxes, cfg, args.split) if args.boxes \
        else list(ccfg.get("boxes") or parse_boxes(None, cfg, args.split))

    device = select_device(args.device)
    out = run_dir(args.run_name, it, create=True)
    fields = out / "fields"
    noise_dir = out / "noise"
    fields.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, cfg_model, device, use_ema=not args.no_ema)
    diff = DiffusionConfig(**{
        k2: v for k2, v in dict(cfg_model.get("diffusion", {})).items()
        if k2 in DiffusionConfig.__dataclass_fields__
    })
    if args.n_steps:
        diff.n_steps = int(args.n_steps)
    if "churn" in ccfg:
        diff.churn = float(ccfg["churn"])
    # An explicit noise vector only determines the sample when nothing else is
    # drawn during sampling; with churn the perturbation would not be the thing
    # under search. Refuse rather than silently overriding the config.
    if diff.churn > 0:
        raise SystemExit(
            f"CEM requires churn = 0 (got {diff.churn}); an explicit init_noise "
            f"does not determine the sample when the sampler keeps drawing"
        )

    ng_hr = int(cfg["data"]["ng_hr"])
    scale = int(cfg["data"]["scale_factor"])
    spec = TileSpec(ng_hr, core=args.tile_core, margin=args.tile_margin,
                    scale_factor=scale)
    rf = measure_receptive_field(
        model, channels=int(cfg_model["model"].get("channels", 6)),
        scale_factor=scale, device=device,
    )
    need = tile_margin_for(rf, scale)
    banner(f"CEM iter {it}: P={P} E={E} sigma={sigma:.4g} boxes={boxes}")
    banner(f"tile core={spec.core} margin={spec.margin}; receptive field {rf} "
           f"-> minimum margin {need}")
    if spec.margin < need:
        raise SystemExit(
            f"receptive-field half-width {rf} needs --tile-margin {need}, got "
            f"{spec.margin}; valid-core tiling would leak tile padding into the "
            f"written core"
        )

    rng = np.random.default_rng(int(args.seed0) + 9973 * it)
    rows = []
    for box in boxes:
        lr = np.load(lr_path(cfg, box))
        base_path = find_base_field(box, args.base_seed)
        if base_path is None:
            raise SystemExit(f"no cached SR2 base for {box}")
        base = np.load(base_path, mmap_mode="r")
        elites = load_elites(args.run_name, it, box) if it > 0 else []

        for j in range(P):
            cid = candidate_id(it, j)
            path = fields / f"{box}_resid_seed{cid}.npy"
            npath = noise_dir / f"{box}_noise_seed{cid}.npy"
            if path.is_file() and npath.is_file() and not args.overwrite:
                print(f"[{box} j={j}] exists", flush=True)
                rows.append({"box": box, "seed": cid, "residual": str(path),
                             "noise": str(npath), "cem_iter": it,
                             "regenerated": False})
                continue

            if it == 0:
                parent = -1
                init = None
                seed = int(args.seed0) + j
            else:
                parent = j % max(1, len(elites))
                init = perturb(elites[parent], sigma, rng)
                seed = cid

            t0 = time.time()
            clip_log: list = []
            if init is None:
                # Reproduce the sampler's own draw so the winning vector can be
                # perturbed later; iteration 0 must record noise like the rest.
                gen = torch.Generator(device="cpu").manual_seed(int(seed))
                init = torch.randn(tuple(np.asarray(base).shape), generator=gen,
                                   dtype=torch.float32).numpy()
            resid = sample_residual_box(
                model, np.asarray(base), lr, seed=seed, cfg=diff, spec=spec,
                device=device, redshift=float(cfg["data"].get("redshift", 0.0)),
                tile_batch=int(args.tile_batch), verify_margin=False,
                clip_log=clip_log, init_noise=init,
            )
            clip_max = max((c["clip_fraction"] for c in clip_log), default=0.0)
            clip_last = clip_log[-1]["clip_fraction"] if clip_log else 0.0
            if args.dtype == "float16":
                resid = resid.astype(np.float16)
            for arr, dst in ((resid, path), (init.astype(np.float16), npath)):
                tmp = dst.with_suffix(".tmp.npy")
                np.save(tmp, arr)
                tmp.replace(dst)
            dt = time.time() - t0
            rms = float(np.sqrt(np.mean(resid[0:3].astype(np.float64) ** 2)))
            print(f"[{box} j={j}] cid={cid} parent={parent} rms={rms:.4g} "
                  f"clip max={100 * clip_max:.3f}% last={100 * clip_last:.3f}% "
                  f"({dt:.0f}s)", flush=True)
            rows.append({
                "box": box, "seed": cid, "residual": str(path),
                "noise": str(npath), "cem_iter": it, "parent_elite_rank": parent,
                "residual_rms_disp": rms, "seconds": dt, "regenerated": True,
                "base": str(base_path),
                "x0_clip": float(diff.x0_clip),
                "clip_fraction_max": clip_max,
                "clip_fraction_final_step": clip_last,
                "clip_log": clip_log,
            })

    write_json(out / "candidates.json", {
        "run_name": f"{args.run_name}_it{it}",
        "cem_run": args.run_name,
        "cem_iter": it,
        "cem": {"population": P, "elites": E, "sigma": sigma,
                "sigma0": float(ccfg.get("sigma0", 0.3)),
                "decay": float(ccfg.get("decay", 0.7))},
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "use_ema": not args.no_ema,
        "model_config": str(Path(args.model_config).resolve()),
        "boxes": boxes,
        "K": P,
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
