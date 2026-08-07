#!/usr/bin/env python
"""Phase 2.2 (GPU): full-box candidates from the Gaussian residual policy.

One forward pass per tile -- there is no sampler loop -- so this is far cheaper
than the diffusion oracle it parallels. It writes the **residual** ``delta``, not
the composed field: the frozen SR2 base is cached and shared, so one 3.2 GB array
per candidate instead of two.

The manifest is deliberately written in the format
``scripts/reward/score_oracle.py`` already reads, so halo finding, constraint
evaluation, chunk attribution, credit assignment and the replay builder are
**reused unchanged**. The Gaussian-specific provenance (the latent action) goes
in a sidecar under ``actions/``, so nothing in the supervised line has to know
this arm exists.

What identifies a stored action
-------------------------------
``a_s = mu_s + sigma_s * eps_s`` and ``eps_s`` is a *global coordinate-aligned*
field determined by ``action_seed`` alone, so ``(policy checkpoint, base field,
action seed)`` reconstructs the action bit-exactly. The record therefore stores
those, plus the behaviour log-probability as a checksum, and only the scales
listed in ``sampling.store_action_scales`` are dumped as arrays. Regeneration is
exact; a 1.6 GB fine-scale dump per candidate is not worth avoiding it.

    python scripts/reward/sample_gaussian_candidates.py \\
        --checkpoint .../gaussian/ckpt.pt --run-name gauss_k16 --k 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from _common import (add_common_args, banner, load_reward_config, lr_path,
                     parse_boxes, select_device, write_json)
from sample_oracle_candidates import (ckpt_identity, field_fingerprint,
                                      read_fingerprint, write_fingerprint)

from cosmo_sr.reward import paths
from cosmo_sr.reward.base import find_base_field
from cosmo_sr.reward.correction import CorrectionConfig, require_calibrated_scales
from cosmo_sr.reward.gaussian_policy import (GaussianPolicyConfig,
                                             MultiScaleGaussianPolicy,
                                             policy_receptive_field,
                                             sample_policy_box)
from cosmo_sr.reward.sampling import TileSpec, tile_margin_for
from cosmo_sr.utils.config import load_config


def build_policy(cfg: Dict, correction_overrides: Optional[Dict] = None
                 ) -> MultiScaleGaussianPolicy:
    mcfg = GaussianPolicyConfig.from_dict(cfg.get("model", {}))
    corr = dict(cfg.get("correction", {}))
    corr.update(correction_overrides or {})
    corr.setdefault("scale_factor", mcfg.scale_factor)
    corr.setdefault("channels", mcfg.channels)
    ccfg = CorrectionConfig.from_dict(corr)
    why = require_calibrated_scales(ccfg.scales)
    if why:
        raise SystemExit(why)
    return MultiScaleGaussianPolicy(mcfg, ccfg)


def load_policy(ckpt_path: Optional[str], cfg: Dict, device,
                correction_overrides: Optional[Dict] = None, use_ema: bool = True):
    """Load a trained policy, or build the reference (untrained) one.

    ``--checkpoint`` is optional on purpose: the support gate (Phase 2.3) has to
    sample the *reference* policy, before any training, and forcing a checkpoint
    for that would mean writing one out just to read it back.
    """
    policy = build_policy(cfg, correction_overrides)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = state["extra"]["ema"] if (use_ema and state.get("extra", {}).get("ema")) \
            else state["model"]
        policy.load_state_dict(sd)
    return policy.to(device).eval()


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--model-config", default="configs/reward/gaussian_policy.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="trained policy; omit to sample the untrained reference "
                         "policy (the support gate's arm)")
    ap.add_argument("--run-name", default="gauss_k16")
    ap.add_argument("--boxes", default=None)
    ap.add_argument("--split", default=None,
                    choices=["train", "val", "test", "dev", "final"])
    ap.add_argument("--k", type=int, default=None, help="candidates per box per amplitude")
    ap.add_argument("--seed0", type=int, default=None)
    ap.add_argument("--base-seed", type=int, default=None)
    ap.add_argument("--amplitudes", default=None,
                    help="comma list; default = sampling.amplitude_curriculum")
    ap.add_argument("--alpha-disp", type=float, default=None)
    ap.add_argument("--alpha-vel", type=float, default=None)
    ap.add_argument("--mode", default=None, choices=["none", "block_null",
                                                     "block_leaky", "split"])
    ap.add_argument("--scales-path", default=None)
    ap.add_argument("--residual-scale", type=float, default=None)
    ap.add_argument("--tile-core", type=int, default=None)
    ap.add_argument("--tile-margin", type=int, default=None)
    ap.add_argument("--tile-batch", type=int, default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    mc = load_config(args.model_config)
    full = {**cfg, "model": mc.get("model", {}), "correction": mc.get("correction", {}),
            "sampling": mc.get("sampling", {})}
    scfg = full["sampling"]

    overrides = {}
    if args.alpha_disp is not None:
        overrides["alpha_disp"] = float(args.alpha_disp)
    if args.alpha_vel is not None:
        overrides["alpha_vel"] = float(args.alpha_vel)
    if args.mode:
        overrides["mode"] = args.mode
    if args.scales_path:
        overrides["scales_path"] = args.scales_path

    boxes = parse_boxes(args.boxes or ",".join(scfg.get("boxes", [])) or None,
                        cfg, args.split or scfg.get("split", "val"))
    k = int(args.k if args.k is not None else scfg.get("k", 16))
    seed0 = int(args.seed0 if args.seed0 is not None else scfg.get("seed0", 0))
    base_seed = int(args.base_seed if args.base_seed is not None
                    else scfg.get("base_seed", 0))
    amplitudes = [float(a) for a in args.amplitudes.split(",")] if args.amplitudes \
        else [float(a) for a in scfg.get("amplitude_curriculum", [1.0])]
    residual_scale = float(args.residual_scale if args.residual_scale is not None
                           else scfg.get("residual_scale", 1.0))
    store_scales = list(scfg.get("store_action_scales", []))

    device = select_device(args.device)
    out = paths.ORACLE(args.run_name, create=True)
    fields = out / "fields"
    actions_dir = out / "actions"
    fields.mkdir(parents=True, exist_ok=True)
    actions_dir.mkdir(parents=True, exist_ok=True)

    # One policy per amplitude: the amplitude lives in the correction transform,
    # and building it per arm keeps the curriculum honest -- a candidate's
    # amplitude is a property of the object that produced it, not a label.
    policies = {
        a: load_policy(args.checkpoint, full, device,
                       {**overrides, "amplitude": a}, use_ema=not args.no_ema)
        for a in amplitudes
    }
    any_policy = policies[amplitudes[0]]
    policy_hash = any_policy.parameter_hash()

    ng_hr = int(cfg["data"]["ng_hr"])
    sf = int(cfg["data"]["scale_factor"])
    core = int(args.tile_core if args.tile_core is not None else scfg.get("tile_core", 128))
    margin = int(args.tile_margin if args.tile_margin is not None
                 else scfg.get("tile_margin", 48))
    tile_batch = int(args.tile_batch if args.tile_batch is not None
                     else scfg.get("tile_batch", 1))
    spec = TileSpec(ng_hr, core=core, margin=margin, scale_factor=sf)

    rf = policy_receptive_field(any_policy, size=max(64, 2 * margin + 16), device=device)
    need = tile_margin_for(rf, sf)
    banner(f"tile core={spec.core} margin={spec.margin}; conditioning receptive "
           f"field {rf} -> minimum margin {need}")
    if spec.margin < need:
        raise SystemExit(
            f"receptive-field half-width {rf} needs --tile-margin {need}, got "
            f"{spec.margin}; valid-core tiling would leak tile padding into the "
            f"written core")

    common = {
        "checkpoint": ckpt_identity(args.checkpoint) if args.checkpoint else "untrained",
        "policy_hash": policy_hash,
        "use_ema": not args.no_ema,
        "model_config": hashlib.sha256(
            Path(args.model_config).read_bytes()).hexdigest()[:32],
        "tile": {"core": spec.core, "margin": spec.margin},
        "dtype": args.dtype,
        "arch": "gaussian_residual_unet",
    }

    rows: List[Dict] = []
    for box in boxes:
        lr = np.load(lr_path(cfg, box))
        base_path = find_base_field(box, base_seed)
        if base_path is None:
            raise SystemExit(f"no cached SR2 base for {box} seed {base_seed}")
        base = np.load(base_path, mmap_mode="r")

        for amp in amplitudes:
            policy = policies[amp]
            ccfg = policy.correction.cfg
            for j in range(k):
                seed = seed0 + j
                name = f"{box}_amp{amp:g}_seed{seed}".replace(".", "p")
                path = fields / f"{name}.npy"
                parts = {**common, "box": box, "action_seed": seed, "amplitude": amp,
                         "mode": ccfg.mode, "alpha_disp": ccfg.alpha_disp,
                         "alpha_vel": ccfg.alpha_vel,
                         "base": str(Path(base_path).resolve())}
                fp = field_fingerprint(**parts)
                rec_path = actions_dir / f"{name}.json"
                if path.is_file() and not args.overwrite and read_fingerprint(path) == fp:
                    print(f"[{name}] exists", flush=True)
                    rows.append(_row(box, seed, path, base_path, fp, amp, ccfg,
                                     json.loads(rec_path.read_text()), rec_path,
                                     regenerated=False))
                    continue

                t0 = time.time()
                delta, stats = sample_policy_box(
                    policy, np.asarray(base), lr, seed=seed, spec=spec, device=device,
                    tile_batch=tile_batch, verify_margin=False,
                    return_actions=tuple(store_scales),
                )
                dt = time.time() - t0

                stored = {}
                for scale, arr in dict(stats.pop("actions", {})).items():
                    p = actions_dir / f"{name}__a_{scale}.npy"
                    np.save(p.with_suffix(".tmp.npy"), arr.astype(np.float16))
                    p.with_suffix(".tmp.npy").replace(p)
                    stored[scale] = str(p)

                record = {
                    "box": box, "action_seed": seed, "base_seed": base_seed,
                    "base_field": str(base_path),
                    "policy_checkpoint": str(args.checkpoint) if args.checkpoint else "",
                    "policy_hash": policy_hash,
                    "use_ema": not args.no_ema,
                    "model_config": str(Path(args.model_config).resolve()),
                    "amplitude": float(amp),
                    "projection_mode": ccfg.mode,
                    "alpha_disp": float(ccfg.alpha_disp),
                    "alpha_vel": float(ccfg.alpha_vel),
                    "correction_scales": ccfg.scales.to_dict(),
                    "behavior_log_prob_sum": float(stats.get("log_prob_sum", float("nan"))),
                    "behavior_log_prob_per_element": float(
                        stats.get("log_prob_per_element", float("nan"))),
                    "stored_action_scales": stored,
                    "tile": {"core": spec.core, "margin": spec.margin},
                    "stats": {k2: v for k2, v in stats.items()
                              if isinstance(v, (int, float, str, bool))},
                    "note": (
                        "a_s = mu_s + sigma_s * eps_s with eps from the GLOBAL "
                        "coordinate-aligned lattice at action_seed, so "
                        "(policy_hash, base_field, action_seed) reconstructs the "
                        "action exactly; behavior_log_prob_sum is the checksum."
                    ),
                }
                write_json(rec_path, record)

                if args.dtype == "float16":
                    delta = delta.astype(np.float16)
                tmp = path.with_suffix(".tmp.npy")
                np.save(tmp, delta)
                tmp.replace(path)
                write_fingerprint(path, fp, parts)

                rms = float(np.sqrt(np.mean(delta[0:3].astype(np.float64) ** 2)))
                print(f"[{name}] rms={rms:.4g} "
                      f"sat={100 * stats.get('tanh_saturated_fraction', 0.0):.2f}% "
                      f"coarse_frac={stats.get('coarse_fraction', float('nan')):.3f} "
                      f"({dt:.0f}s)", flush=True)
                rows.append({**_row(box, seed, path, base_path, fp, amp, ccfg,
                                    record, rec_path, regenerated=True),
                             "residual_rms_disp": rms, "seconds": dt,
                             "tanh_saturated_fraction": float(
                                 stats.get("tanh_saturated_fraction", float("nan"))),
                             "coarse_fraction": float(
                                 stats.get("coarse_fraction", float("nan")))})

    # Written in score_oracle.py's format on purpose: that stage, oracle_report.py
    # and build_replay.py then work on this arm with no changes at all.
    write_json(out / "candidates.json", {
        "run_name": args.run_name,
        "arch": "gaussian_residual_unet",
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else "",
        "untrained_reference_policy": not bool(args.checkpoint),
        "policy_hash": policy_hash,
        "use_ema": not args.no_ema,
        "model_config": str(Path(args.model_config).resolve()),
        "boxes": boxes,
        "K": k,
        "seed0": seed0,
        "base_seed": base_seed,
        "residual_scale": residual_scale,
        "amplitudes": amplitudes,
        "correction": policies[amplitudes[0]].correction.cfg.to_dict(),
        "tile": {"core": spec.core, "margin": spec.margin,
                 "receptive_field_halfwidth": int(rf)},
        "dtype": args.dtype,
        "candidates": rows,
    })
    banner(f"{len(rows)} candidates -> {out}")
    print("next:")
    print(f"  sbatch scripts/slurm/gaussian_score_cpu.sbatch RUN_NAME={args.run_name} "
          f"BASELINES=1   # then the candidate array, then AGGREGATE=1")
    print(f"  sbatch scripts/slurm/gaussian_support_gate_cpu.sbatch "
          f"RUN_NAME={args.run_name}")


def _row(box, seed, path, base_path, fp, amp, ccfg, record, record_path, *,
         regenerated) -> Dict:
    return {
        # score_oracle.py reads exactly these four; do not rename them.
        "box": box, "seed": int(seed), "residual": str(path), "base": str(base_path),
        # everything below is this arm's own provenance, ignored by that stage
        "fingerprint": fp,
        "amplitude": float(amp),
        "projection_mode": ccfg.mode,
        "alpha_disp": float(ccfg.alpha_disp),
        "alpha_vel": float(ccfg.alpha_vel),
        "action_record": str(record_path),
        "behavior_log_prob_sum": float(record.get("behavior_log_prob_sum", float("nan"))),
        "policy_hash": record.get("policy_hash", ""),
        "regenerated": bool(regenerated),
    }


if __name__ == "__main__":
    main()
