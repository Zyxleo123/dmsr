#!/usr/bin/env python
"""Stage 9a (GPU): full-box samples for every arm of the final comparison.

Arms:

``sr2``       the frozen baseline (already cached; only linked, never resampled);
``prior``     ``n_samples`` ordinary random draws from the supervised prior;
``distill``   ``n_samples`` ordinary random draws from the distilled student;
``bestofk``   ``k`` extra draws from the PRIOR, from which the CPU stage selects
              one by reward -- the control that says whether distillation beat
              simply sampling more from the model it started from;
``samecomp``  ordinary draws from a supervised-only checkpoint trained for the
              same number of steps as the distilled one, so "distill is better
              than prior" cannot be explained by extra optimisation alone;
``abundance`` draws from a checkpoint distilled against an abundance-only
              reward, which separates "the reward moved occupation" from "any
              catalog reward moves the catalog".

The ``prior`` and ``distill`` arms get the same number of random samples and
**no selection**: the headline claim is about average samples, so the code never
offers a "pick the best" switch for them. ``bestofk`` is selected, deliberately
and separately, and is reported as its own arm.

The default split is ``final`` (set13-15). It is NOT ``test``: ``test`` includes
set12, which the SR2 subhalo study and the reward sanity check have both already
looked at, so a number computed on it is development data wearing a held-out
label.

    python scripts/reward/generate_eval_boxes.py --run-name final \
        --prior-ckpt .../residual_prior/ckpt_best.pt \
        --distill-ckpt .../distill_round0/ckpt_best.pt \
        --same-compute-ckpt .../prior_long/ckpt_best.pt \
        --abundance-ckpt .../distill_abundance/ckpt_best.pt \
        --best-of-k 8 --n-samples 4
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from _common import (add_common_args, banner, load_reward_config, lr_path,
                     parse_boxes, select_device, write_json)
from sample_oracle_candidates import (ckpt_identity, field_fingerprint, load_model,
                                      read_fingerprint, write_fingerprint)

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
    ap.add_argument("--same-compute-ckpt", default=None,
                    help="supervised-only checkpoint trained for as many steps "
                         "as the distilled one (the same-compute control)")
    ap.add_argument("--abundance-ckpt", default=None,
                    help="checkpoint distilled against an abundance-only reward "
                         "(the abundance-only control)")
    ap.add_argument("--boxes", default=None, help="default = final split")
    # 'final' (set13-15), not 'test': test includes set12, which is declared dev
    # data in configs/reward/reward.yaml because the subhalo study already used
    # it. The sbatch launcher defaults to final too; these two used to disagree.
    ap.add_argument("--split", default="final",
                    choices=["train", "val", "test", "dev", "final"])
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--best-of-k", type=int, default=0,
                    help="extra PRIOR draws forming the best-of-K arm; the CPU "
                         "stage selects one of them by reward. 0 disables it")
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
    model_cfg_hash = hashlib.sha256(
        Path(args.model_config).read_bytes()).hexdigest()[:32]
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

    # arm -> (checkpoint, model, n_samples, seed0). best-of-K draws from the
    # prior with its own seed block so its samples are disjoint from the prior
    # arm's -- selecting from the same draws the prior arm is averaged over would
    # make the two arms share candidates.
    ckpts = {
        "prior": args.prior_ckpt,
        "distill": args.distill_ckpt,
        "samecomp": args.same_compute_ckpt,
        "abundance": args.abundance_ckpt,
    }
    arms = {}
    for name, ck in ckpts.items():
        if ck:
            arms[name] = load_model(ck, cfg_model, device, use_ema=not args.no_ema)
    if not arms:
        raise SystemExit(
            "nothing to generate: pass at least one of --prior-ckpt, "
            "--distill-ckpt, --same-compute-ckpt, --abundance-ckpt"
        )
    plan = [(n, arms[n], int(args.n_samples), int(args.seed0)) for n in arms]
    if int(args.best_of_k) > 0:
        if "prior" not in arms:
            raise SystemExit("--best-of-k needs --prior-ckpt: it is a control on "
                             "sampling the prior harder, not a separate model")
        plan.append(("bestofk", arms["prior"], int(args.best_of_k),
                     int(args.seed0) + 10000))
    missing = [n for n in ("distill", "samecomp", "abundance") if not ckpts.get(n)] \
        + ([] if int(args.best_of_k) > 0 else ["bestofk"])
    if missing:
        print(f"  ! arms NOT generated: {missing}. The final comparison is "
              f"incomplete without them -- 'distill beats prior' has "
              f"same-compute and abundance-only explanations until those arms "
              f"exist.", flush=True)

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

        for name, model, n_samples, seed_base in plan:
            for s in range(int(n_samples)):
                seed = int(seed_base) + s
                rp = fields / f"{box}_{name}_resid_seed{seed}.npy"
                parts = {
                    "arm": name,
                    "checkpoint": ckpt_identity(
                        ckpts["prior" if name == "bestofk" else name]),
                    "use_ema": not args.no_ema,
                    "model_config": model_cfg_hash,
                    "diffusion": diff.__dict__,
                    "tile": {"core": spec.core, "margin": spec.margin},
                    "box": box, "seed": seed,
                    "base": str(Path(base_path).resolve()),
                    "redshift": float(cfg["data"].get("redshift", 0.0)),
                }
                fp = field_fingerprint(**parts)
                row = {"arm": name, "box": box, "sample": s, "seed": seed,
                       "residual": str(rp), "base": str(base_path),
                       "fingerprint": fp, "selected": name == "bestofk"}
                if rp.is_file() and not args.overwrite:
                    if read_fingerprint(rp) == fp:
                        rows.append({**row, "regenerated": False})
                        print(f"[{box} {name} s={s}] exists", flush=True)
                        continue
                    print(f"[{box} {name} s={s}] REGENERATING: {rp.name} was "
                          f"produced with different inputs", flush=True)
                t0 = time.time()
                resid = sample_residual_box(
                    model, np.asarray(base), lr, seed=seed, cfg=diff, spec=spec,
                    device=device, redshift=float(cfg["data"].get("redshift", 0.0)),
                    tile_batch=int(args.tile_batch), verify_margin=False,
                )
                tmp = rp.with_suffix(".tmp.npy")
                np.save(tmp, resid)
                tmp.replace(rp)
                write_fingerprint(rp, fp, parts)
                rms = float(np.sqrt(np.mean(resid[0:3].astype(np.float64) ** 2)))
                print(f"[{box} {name} s={s}] seed={seed} rms={rms:.4g} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                rows.append({**row, "residual_rms_disp": rms, "regenerated": True})

    write_json(out / "eval_fields.json", {
        "run_name": args.run_name,
        "boxes": boxes, "split": args.split,
        "n_samples": int(args.n_samples),
        "best_of_k": int(args.best_of_k),
        "residual_scale": float(args.residual_scale),
        "checkpoints": {k: v for k, v in ckpts.items() if v},
        "arms": [n for n, _, _, _ in plan] + ["sr2"],
        "arms_missing": missing,
        "use_ema": not args.no_ema,
        "diffusion": diff.__dict__,
        "tile": {"core": spec.core, "margin": spec.margin},
        "selection": (
            "none for sr2/prior/distill/samecomp/abundance -- ordinary random "
            "samples. The bestofk arm is a disjoint block of prior draws that "
            "the CPU stage selects ONE of, by reward."
        ),
        "fields": rows,
    })
    banner(f"{len(rows)} entries -> {out / 'eval_fields.json'}")


if __name__ == "__main__":
    main()
