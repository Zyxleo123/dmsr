#!/usr/bin/env python
"""Generate full periodic boxes from a checkpoint and measure the field guards.

GPU, and deliberately separate from scoring: this job produces evidence (fields,
per-(box, seed) field metrics) and takes no decisions. ``score_sr2_direct.py``
reads what this writes and decides, on a CPU node.

Evaluation uses **complete periodic boxes**, not crops: ``cic_density_valid_center``
is the right tool inside a training step and a whole box is the right tool for a
verdict, because on a whole box the CIC deposit is exact rather than an
approximation with a measured error budget.

    python scripts/reward/evaluate_sr2_direct.py --run-name direct_a \
        --checkpoint .../ema_generator.pt --boxes set8,set9 --seeds 0,1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from _sr2_direct import (  # noqa: E402
    add_direct_args, append_jsonl, array_sha, assert_not_sealed, banner,
    boxes_of, direct_root, file_sha, geometry_of, load_direct_config, load_hr,
    load_lr, model_path_of, run_dir, soft_config_of,
)

from cosmo_sr.reward import fields as F  # noqa: E402
from cosmo_sr.reward.soft_structure import structural_diversity  # noqa: E402
from cosmo_sr.tts.sampling import super_resolve_srs_seeded  # noqa: E402
from cosmo_sr.tts.srs_noise import load_controlled_generator  # noqa: E402


def field_metrics(cfg, cand: np.ndarray, frozen: np.ndarray, hr: np.ndarray,
                  lr: np.ndarray) -> Dict[str, float]:
    """The whole-box guards: density power vs HR, and LR-visible change vs frozen."""
    g = cfg["geometry"]
    d = cfg["_reward"]["data"]
    sf = int(g.get("scale_factor", cfg["model"]["scale_factor"]))

    a_c = F.block_average(cand, sf)
    a_0 = F.block_average(frozen, sf)
    out = {
        "low_k_change": F.rel_rms(a_c, a_0),
        "lr_consistency_error": F.rel_rms(a_c, lr),
    }
    del a_c, a_0

    kw = dict(boxsize_mpc_h=float(g.get("boxsize_mpc_h", 100.0)),
              dis_norm_kpc_h=float(g.get("dis_norm_kpc_h", 6000.0)),
              redshift=float(d.get("redshift", 0.0)))
    d_c = F.cic_density_box(cand[0:3], **kw)
    d_h = F.cic_density_box(np.asarray(hr[0:3]), **kw)
    _, p_c, p_h, _ = F.cross_power(d_c, d_h, 24)
    out["density_power_error"] = float(np.mean(np.abs(np.log(
        np.maximum(p_c, 1e-30) / np.maximum(p_h, 1e-30)))))
    out["density_sigma_ratio"] = float(d_c.std() / max(d_h.std(), 1e-30))
    del d_h
    d_0 = F.cic_density_box(np.asarray(frozen[0:3]), **kw)
    _, p_0, _, _ = F.cross_power(d_0, d_0, 24)
    out["density_sigma_ratio_vs_frozen"] = float(d_c.std() / max(d_0.std(), 1e-30))
    del d_c, d_0

    t_err = []
    for ch in (0, 1, 2):
        _, ph, phr, _ = F.cross_power(np.asarray(cand[ch]), np.asarray(hr[ch]), 24)
        t_err.append(float(np.mean(np.abs(
            np.sqrt(np.maximum(ph, 0.0) / np.maximum(phr, 1e-30)) - 1.0))))
    out["displacement_power_error"] = float(np.mean(t_err))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--checkpoint", default="",
                    help="EMA generator checkpoint; empty means the FROZEN "
                         "generator, which is how the paired baseline is made")
    ap.add_argument("--boxes", default="")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--keep-fields", action="store_true")
    ap.add_argument("--diversity-tile", type=int, default=0,
                    help="tile id used for the structural-diversity measurement")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    geom = geometry_of(cfg)
    scfg = soft_config_of(cfg)
    boxes = ([b.strip() for b in args.boxes.split(",") if b.strip()]
             if args.boxes else boxes_of(cfg, "actor_eval"))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    assert_not_sealed(cfg, boxes)

    is_frozen = not args.checkpoint
    path = Path(args.checkpoint) if args.checkpoint else model_path_of(cfg)
    tag = args.tag or ("frozen" if is_frozen else Path(path).parent.name)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    run = run_dir(args.run_name, create=True)
    rows_path = run / f"field_metrics_{tag}.jsonl"
    banner(f"evaluating {tag} ({path.name}) on {boxes} x seeds {seeds}")

    gen = load_controlled_generator(path, scale_factor=geom.scale_factor,
                                    device=device, eval_mode=True)
    frozen_gen = gen if is_frozen else load_controlled_generator(
        model_path_of(cfg), scale_factor=geom.scale_factor, device=device,
        eval_mode=True)

    done = {(r["box"], r["seed"]) for r in
            (__import__("json").loads(l) for l in rows_path.read_text().splitlines()
             if l.strip())} if rows_path.is_file() else set()

    out_dir = direct_root("eval", args.run_name, tag, create=True)
    for box in boxes:
        lr = np.asarray(load_lr(cfg, box), dtype=np.float32)
        hr = load_hr(cfg, box)
        for seed in seeds:
            if (box, seed) in done:
                print(f"  {box}/seed{seed}: already measured, skipping", flush=True)
                continue
            t0 = time.time()
            cand = super_resolve_srs_seeded(
                gen, lr, seed, scale_factor=geom.scale_factor, nsplit=geom.nsplit,
                pad=geom.pad, device=device, noise_mode="per_tile")
            frozen = cand if is_frozen else super_resolve_srs_seeded(
                frozen_gen, lr, seed, scale_factor=geom.scale_factor,
                nsplit=geom.nsplit, pad=geom.pad, device=device,
                noise_mode="per_tile")

            m = field_metrics(cfg, cand, frozen, hr, lr)
            # Structural diversity needs two seeds of the SAME tile, so it is
            # measured on one tile rather than on the whole box: a full second
            # box per seed pair would cost more than it tells.
            grid = geom.tile_grid()
            sl = grid.slices(int(args.diversity_tile))
            other = super_resolve_srs_seeded(
                gen, lr, seed + 991, scale_factor=geom.scale_factor,
                nsplit=geom.nsplit, pad=geom.pad, device=device,
                noise_mode="per_tile")
            draws = torch.from_numpy(np.stack([
                cand[:, sl[0], sl[1], sl[2]], other[:, sl[0], sl[1], sl[2]]
            ])[None]).float()
            del other
            with torch.no_grad():
                div = structural_diversity(draws, scfg)
            m.update({f"d_{k}": float(v[0]) for k, v in div.items()})
            m["d_struct"] = float(div["d_struct"][0])

            if args.keep_fields:
                np.save(out_dir / f"{box}_seed{seed}.npy", cand.astype(np.float32))
            row = {"box": box, "seed": int(seed), "tag": tag,
                   "checkpoint": str(path), "checkpoint_sha": file_sha(path),
                   "frozen_baseline": bool(is_frozen), "lr_sha": array_sha(lr),
                   "seconds": round(time.time() - t0, 1), **m}
            append_jsonl(rows_path, row)
            print(f"  {box}/seed{seed}: density_power_error={m['density_power_error']:.5f} "
                  f"low_k={m['low_k_change']:.4f} d_struct={m['d_struct']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            del cand, frozen
    print(f"  rows -> {rows_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
