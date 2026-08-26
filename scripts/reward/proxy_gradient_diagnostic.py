#!/usr/bin/env python
"""Measure the proxy's gradient into the SR2 output field.

The actor's whole objective is ``-Q_safe(proxy(features(G_theta(Y,z))))``, and
the only thing that moves SR2's weights is ``dQ/d theta``, which factors through
``g = dQ/d(cand)`` -- the gradient of the proxy's scalar reward with respect to
every voxel of the generated tile. If the proxy has collapsed to the per-tile
mean (the candidate-collapse finding), that field is degenerate: near-zero, or a
single attractor that pulls every candidate to one point. This job measures it
directly, at the SAME differentiable extractor + frozen proxy the trainer uses,
so the number is the actor's, not a lookalike.

For each (box, tile, seed) it computes ``cand = G(Y, z)``, then
``g = d[sum_b Q_safe_b] / d cand`` by autograd, and saves, per output channel:
  * a voxel subsample of ``(cand_value, g_value)`` pairs -- the scatter the plot
    job renders;
  * the OLS fit ``g ~= a + b*cand`` per (tile, seed, channel): a NEGATIVE slope
    ``b`` is an attractor and ``c0 = -a/b`` is the value it pulls voxels toward,
    so whether the c0's across tiles/seeds cluster answers "does it converge
    points"; a flat/zero slope is a dead gradient;
  * ``|g|`` magnitude stats and the dead-voxel fraction.

It does NOT fine-tune anything and writes only under the run's ``gradient/``
directory. Uses the frozen G_z0 by default; pass ``--checkpoint`` to measure the
gradient at a fine-tuned checkpoint instead.

    python scripts/reward/proxy_gradient_diagnostic.py --run-name direct_a --arm c
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from _sr2_direct import (  # noqa: E402
    actor_config_of, add_direct_args, arm_f_valid_center, banner, boxes_of,
    geometry_of, load_direct_config, load_hr, load_lr, load_reward_models,
    model_path_of, phase_space_config_of, run_dir, soft_config_of,
    soft_rockstar_config_of, write_json,
)
from train_sr2_direct import (  # noqa: E402
    frozen_field_for, frozen_summaries_for, host_rich_tiles,
)

from cosmo_sr.reward.catalog_proxy import ProxyEnsemble  # noqa: E402
from cosmo_sr.reward.torch_reward import summary_from_tiles  # noqa: E402
from cosmo_sr.train.sr2_finetune_data import SR2TileDataset, collate_tiles  # noqa: E402
from cosmo_sr.train.train_sr2_direct import DirectFinetuneTrainer, attach_summaries  # noqa: E402
from cosmo_sr.tts.srs_noise import load_controlled_generator  # noqa: E402


CHAN_NAMES = ["disp_x", "disp_y", "disp_z", "vel_x", "vel_y", "vel_z"]


def ols(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """``y ~= a + b x``. Returns slope, intercept, corr and the fixed point."""
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    if x.size < 8 or np.std(x) < 1e-30:
        return {"slope": 0.0, "intercept": float(np.mean(y)) if y.size else 0.0,
                "corr": 0.0, "fixed_point": float("nan"), "n": int(x.size)}
    b, a = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 1e-30 else 0.0
    fp = float(-a / b) if abs(b) > 1e-30 else float("nan")
    return {"slope": float(b), "intercept": float(a), "corr": r,
            "fixed_point": fp, "n": int(x.size)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--arm", default="c")
    ap.add_argument("--boxes", default="", help="default: first two actor-train boxes")
    ap.add_argument("--n-tiles", type=int, default=6,
                    help="host-rich tiles per box (where occupation can move)")
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="noise seeds per tile (draws); draw 0 is the base seed")
    ap.add_argument("--subsample", type=int, default=4000,
                    help="voxels kept per (tile, seed, channel) for the scatter")
    ap.add_argument("--dead-eps-frac", type=float, default=0.01,
                    help="a voxel is 'dead' if |g| < eps_frac * max|g| for its channel")
    ap.add_argument("--checkpoint", default="",
                    help="EMA generator to measure at; default = frozen G_z0")
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    run = run_dir(args.run_name, create=True)
    geom = geometry_of(cfg)
    scfg = soft_config_of(cfg)
    acfg = actor_config_of(cfg)
    pscfg = phase_space_config_of(cfg)
    rcfg = soft_rockstar_config_of(cfg)
    rng = np.random.default_rng(int(args.seed))

    proxy_dir = run / f"proxy_{args.arm}"
    if not proxy_dir.is_dir():
        print(f">>> MISSING INPUT: {proxy_dir}")
        print(">>> produced by: scripts/reward/train_catalog_proxy.py")
        return 0
    proxies = ProxyEnsemble.load(proxy_dir)
    _, reward_t = load_reward_models(cfg)

    boxes = ([b.strip() for b in args.boxes.split(",") if b.strip()]
             if args.boxes else boxes_of(cfg, "actor_train")[:2])
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # ---- data (same setup as train_sr2_direct.main) ----------------------
    base_seed = 0
    lr_fields, hr_fields, frozen_fields, summaries = {}, {}, {}, {}
    for b in boxes:
        lr_fields[b] = np.asarray(load_lr(cfg, b), dtype=np.float32)
        hr_fields[b] = load_hr(cfg, b)
        f = frozen_field_for(cfg, b, base_seed)
        if f is not None:
            frozen_fields[b] = f
        summaries[b] = frozen_summaries_for(cfg, b)

    # Host-rich tiles only: a tile with no resolved host cannot move occupation,
    # so its gradient would measure the guards, not the reward.
    upper = [2, 3]
    tile_ids = sorted({t for b in boxes
                       for t in host_rich_tiles(summaries[b], int(args.n_tiles), upper)})
    banner(f"arm {args.arm}: {len(boxes)} boxes x {len(tile_ids)} host-rich tiles "
           f"x {args.n_seeds} seeds")

    frozen_gen = load_controlled_generator(
        model_path_of(cfg), scale_factor=geom.scale_factor,
        device=torch.device("cpu"), eval_mode=True)
    ds = SR2TileDataset(
        boxes=boxes, lr_fields=lr_fields, hr_fields=hr_fields, geom=geom,
        frozen_fields=frozen_fields, frozen_summaries=summaries,
        frozen_generator=frozen_gen, tile_ids=tile_ids, base_seed=base_seed,
        noise_draws=int(args.n_seeds))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=collate_tiles,
        num_workers=0, drop_last=False)

    box_summaries = {b: summary_from_tiles(list(summaries[b].values())) for b in boxes}
    frozen_tiles = {(b, t): summary_from_tiles([summaries[b][t]])
                    for b in boxes for t in summaries[b]}

    # ---- trainer (the exact extractor + frozen proxy the actor uses) ------
    acfg.amp = False  # a clean float32 gradient, not the AMP training path
    trainer = DirectFinetuneTrainer(
        model_path_of(cfg), proxies, reward_t, cfg=acfg, geom=geom,
        soft_cfg=scfg, device=device, arm=str(args.arm), phase_cfg=pscfg,
        rockstar_cfg=rcfg, f_valid_center=arm_f_valid_center(cfg))
    trainer.use_amp = False
    if args.checkpoint:
        blob = trainer.load(args.checkpoint)
        banner(f"measuring gradient at {args.checkpoint} (step "
               f"{blob.get('step','?')})")

    n_draws = int(args.n_seeds)
    beta = float(acfg.beta_uncertainty)
    w_joint, w_occ = float(acfg.w_joint_reward), float(acfg.w_occ_reward)

    # per-(unit, channel) scatter samples and per-unit fit rows
    scatter: Dict[str, List[np.ndarray]] = {"cand": [], "grad": [], "chan": [],
                                            "unit": []}
    fit_rows: List[Dict] = []
    unit_index: List[Dict] = []

    for batch in loader:
        b_id = str(batch["box"][0])
        t_id = int(batch["tile_id"][0].item())
        batch = attach_summaries(batch, box_summaries, frozen_tiles)

        lr = batch["lr"].to(device)
        noise = {k: v.to(device) for k, v in batch["noise"].items()}
        # fold (example, draw) -> row, exactly like the trainer.
        from cosmo_sr.train.sr2_finetune_data import fold_draws
        lr_f, noise_f, nb, nd = fold_draws({"lr": lr, "noise": noise})
        from cosmo_sr.train.sr2_finetune_data import trim_to_tile
        with torch.no_grad():
            base_all = trim_to_tile(trainer.frozen(lr_f, noise=noise_f), geom).float()
        # candidate output WITH grad tracked on the field itself
        cand_all = trim_to_tile(trainer.actor(lr_f, noise=noise_f), geom).float()

        box = batch["box_summary"].to(device)
        frozen_tile = batch["frozen_tile_summary"].to(device)

        # One seed (draw) at a time: draw d is row d for this single-example batch.
        for s in range(nd):
            cand = cand_all[s:s + 1].detach().clone().requires_grad_(True)
            base = base_all[s:s + 1]
            lr_row = lr_f[s:s + 1]
            feats = trainer._extract(cand, base, lr=lr_row)
            all_dr = trainer.proxies.delta_rewards_all(
                trainer.reward, feats.to(torch.float64), box, frozen_tile,
                w_joint=w_joint, w_occ=w_occ)
            q = ProxyEnsemble.q_safe(all_dr["dR_combined"], beta=beta)
            trainer.actor.zero_grad(set_to_none=True)
            if cand.grad is not None:
                cand.grad = None
            q["q_safe"].sum().backward()
            g = cand.grad.detach().float().cpu().numpy()[0]      # (6, T, T, T)
            c = cand.detach().float().cpu().numpy()[0]            # (6, T, T, T)

            uid = f"{b_id}/t{t_id}/s{s}"
            unit_index.append({"unit": uid, "box": b_id, "tile": t_id, "seed": s,
                               "q_safe": float(q["q_safe"].mean()),
                               "dR_occ": float(all_dr["dR_occ"].mean()),
                               "dR_combined": float(all_dr["dR_combined"].mean())})
            for ci, cname in enumerate(CHAN_NAMES):
                gc = g[ci].ravel()
                cc = c[ci].ravel()
                gmax = float(np.max(np.abs(gc))) if gc.size else 0.0
                dead = (float(np.mean(np.abs(gc) < args.dead_eps_frac * gmax))
                        if gmax > 0 else 1.0)
                fit = ols(cc, gc)
                fit.update({"unit": uid, "box": b_id, "tile": t_id, "seed": s,
                            "channel": cname,
                            "grad_abs_mean": float(np.mean(np.abs(gc))),
                            "grad_abs_max": gmax,
                            "grad_std": float(np.std(gc)),
                            "dead_frac": dead})
                fit_rows.append(fit)
                # subsample voxels for the scatter
                k = min(int(args.subsample), gc.size)
                idx = rng.choice(gc.size, size=k, replace=False)
                scatter["cand"].append(cc[idx].astype(np.float32))
                scatter["grad"].append(gc[idx].astype(np.float32))
                scatter["chan"].append(np.full(k, ci, dtype=np.int16))
                scatter["unit"].append(np.full(k, len(unit_index) - 1, dtype=np.int32))
        print(f"  {b_id} tile {t_id}: {nd} seeds done", flush=True)

    out_dir = run / "gradient"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (Path(args.checkpoint).parent.name if args.checkpoint else "frozen")
    npz = out_dir / f"proxy_grad_{args.arm}_{tag}.npz"
    np.savez_compressed(
        npz,
        cand=np.concatenate(scatter["cand"]),
        grad=np.concatenate(scatter["grad"]),
        chan=np.concatenate(scatter["chan"]),
        unit=np.concatenate(scatter["unit"]),
        chan_names=np.array(CHAN_NAMES),
        unit_labels=np.array([u["unit"] for u in unit_index]),
    )

    # per-channel aggregate: do the fixed points cluster (convergence)?
    agg: Dict[str, Dict] = {}
    for ci, cname in enumerate(CHAN_NAMES):
        rows = [r for r in fit_rows if r["channel"] == cname]
        slopes = np.array([r["slope"] for r in rows], dtype=np.float64)
        fps = np.array([r["fixed_point"] for r in rows], dtype=np.float64)
        fps = fps[np.isfinite(fps)]
        neg = float(np.mean(slopes < 0)) if slopes.size else 0.0
        agg[cname] = {
            "n_units": len(rows),
            "median_slope": float(np.median(slopes)) if slopes.size else 0.0,
            "frac_negative_slope": neg,
            "median_dead_frac": float(np.median([r["dead_frac"] for r in rows])),
            "median_grad_abs_mean": float(np.median([r["grad_abs_mean"] for r in rows])),
            "fixed_point_median": float(np.median(fps)) if fps.size else float("nan"),
            "fixed_point_iqr": (float(np.subtract(*np.percentile(fps, [75, 25])))
                                if fps.size else float("nan")),
        }

    summary = {
        "run_name": args.run_name, "arm": str(args.arm), "checkpoint": args.checkpoint,
        "tag": tag, "boxes": boxes, "tile_ids": tile_ids, "n_seeds": n_draws,
        "n_units": len(unit_index), "beta": beta,
        "npz": str(npz), "channel_aggregate": agg,
        "units": unit_index, "fits": fit_rows,
    }
    write_json(out_dir / f"proxy_grad_{args.arm}_{tag}.json", summary)
    banner(f"wrote {npz}")
    print("  per-channel: median slope / frac attractor(neg slope) / dead frac", flush=True)
    for cname, a in agg.items():
        print(f"    {cname:7s}  slope {a['median_slope']:+.3e}  "
              f"neg {a['frac_negative_slope']:.2f}  dead {a['median_dead_frac']:.2f}  "
              f"c0 {a['fixed_point_median']:+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
