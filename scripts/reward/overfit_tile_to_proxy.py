#!/usr/bin/env python
"""Overfit ONE SR2 tile to the proxy: field-space gradient ascent to convergence.

Not a fine-tune of the generator -- the tile itself is the free variable. Start
from a frozen-SR2 tile ``cand0 = G(Y, z)``, then optimise the field ``x`` to
maximise the proxy reward ``Q_safe(x)`` under the actor's own field guards, until
the reward plateaus:

    x* = argmax_x  Q_safe(x)  -  lambda_P L_Pdensity(x, HR)
                              -  lambda_low L_lowk(x, cand0)
                              -  lambda_prox ||x - cand0||^2

The low-k guard pins the LR-visible scales to the starting tile, so the result is
"the SAME tile, refined" -- only the high-k structure the reward cares about is
free to move. Every few steps it snapshots the tile's density mid-slice and the
reward, so a CPU job can render the process to a GIF.

This is deliberately the surrogate's fixed point, NOT a physical one: a field
optimised directly against a proxy will exploit it, and the guards bound that but
do not remove it. The demo shows what the gradient DOES, not that x* is a better
catalog. Reads the same frozen-but-differentiable proxy the actor uses.

    python scripts/reward/overfit_tile_to_proxy.py --run-name direct_a --arm c \
        --box set0 --tile 489 --iters 800
"""
from __future__ import annotations

import argparse
import json
import time
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
from train_sr2_direct import frozen_field_for, frozen_summaries_for, host_rich_tiles

from cosmo_sr.reward.catalog_proxy import ProxyEnsemble  # noqa: E402
from cosmo_sr.reward.soft_structure import density_from_disp  # noqa: E402
from cosmo_sr.reward.torch_reward import summary_from_tiles  # noqa: E402
from cosmo_sr.train.sr2_finetune_data import (  # noqa: E402
    SR2TileDataset, collate_tiles, fold_draws, trim_to_tile,
)
from cosmo_sr.train.train_sr2_direct import DirectFinetuneTrainer, attach_summaries  # noqa: E402
from cosmo_sr.tts.srs_noise import load_controlled_generator  # noqa: E402


def density_slice(field: torch.Tensor, scfg) -> np.ndarray:
    """log1p density mid-slice of a (1, 6, N,N,N) tile, for the video."""
    with torch.no_grad():
        d = density_from_disp(field, scfg)[0, 0]        # (R, R, R)
    mid = d.shape[-1] // 2
    return torch.log1p(d[:, :, mid].clamp_min(0)).float().cpu().numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--arm", default="c")
    ap.add_argument("--box", default="", help="default: first actor-train box")
    ap.add_argument("--tile", type=int, default=-1,
                    help="tile id; default: the most host-rich tile of the box")
    ap.add_argument("--seed", type=int, default=0, help="noise seed for cand0")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--field-lr", type=float, default=1e-3,
                    help="Adam step on the field. Lower = slower/gentler descent")
    ap.add_argument("--save-every", type=int, default=5, help="GIF frame cadence")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="save the full 6-ch field every N iters (0=off) for the "
                         "Rockstar monitor to splice + re-run the halo finder")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    # convergence: stop when best Q has not improved by --tol over --patience saves
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    # objective weights; default to the actor config's
    ap.add_argument("--w-reward", type=float, default=-1.0)
    ap.add_argument("--w-density", type=float, default=-1.0)
    ap.add_argument("--w-lowk", type=float, default=-1.0)
    ap.add_argument("--w-prox", type=float, default=-1.0)
    # host-halo control: pin the density of the starting tile's densest cells so
    # the reward cannot buy occupation by moving/erasing the host. 0 = off.
    ap.add_argument("--w-host", type=float, default=0.0)
    ap.add_argument("--host-quantile", type=float, default=0.99,
                    help="cells above this density quantile of cand0 are 'host'")
    ap.add_argument("--label", default="",
                    help="suffix appended to the output tag as '__<label>', so a "
                         "new experiment does not overwrite an old one")
    ap.add_argument("--device", default="")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    run = run_dir(args.run_name, create=True)
    geom = geometry_of(cfg)
    scfg = soft_config_of(cfg)
    acfg = actor_config_of(cfg)
    pscfg = phase_space_config_of(cfg)
    rcfg = soft_rockstar_config_of(cfg)

    w_reward = acfg.lambda_reward if args.w_reward < 0 else args.w_reward
    w_dens = acfg.lambda_density_power if args.w_density < 0 else args.w_density
    w_lowk = acfg.lambda_low_k if args.w_lowk < 0 else args.w_lowk
    w_prox = acfg.lambda_prox if args.w_prox < 0 else args.w_prox
    w_host = float(args.w_host)
    beta = float(acfg.beta_uncertainty)
    w_joint, w_occ = float(acfg.w_joint_reward), float(acfg.w_occ_reward)

    proxy_dir = run / f"proxy_{args.arm}"
    if not proxy_dir.is_dir():
        print(f">>> MISSING INPUT: {proxy_dir}")
        return 0
    proxies = ProxyEnsemble.load(proxy_dir)
    _, reward_t = load_reward_models(cfg)

    box = args.box or boxes_of(cfg, "actor_train")[0]
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # ---- one tile of one box ---------------------------------------------
    lr_fields = {box: np.asarray(load_lr(cfg, box), dtype=np.float32)}
    hr_fields = {box: load_hr(cfg, box)}
    summaries = {box: frozen_summaries_for(cfg, box)}
    f = frozen_field_for(cfg, box, args.seed)
    frozen_fields = {box: f} if f is not None else {}
    tile = (int(args.tile) if args.tile >= 0
            else host_rich_tiles(summaries[box], 1, [2, 3])[0])
    banner(f"overfit tile {box}/t{tile} (arm {args.arm}) on {device}")

    frozen_gen = load_controlled_generator(
        model_path_of(cfg), scale_factor=geom.scale_factor,
        device=torch.device("cpu"), eval_mode=True)
    ds = SR2TileDataset(
        boxes=[box], lr_fields=lr_fields, hr_fields=hr_fields, geom=geom,
        frozen_fields=frozen_fields, frozen_summaries=summaries,
        frozen_generator=frozen_gen, tile_ids=[tile], base_seed=args.seed,
        noise_draws=1)
    batch = attach_summaries(
        collate_tiles([ds[0]]),
        {box: summary_from_tiles(list(summaries[box].values()))},
        {(box, t): summary_from_tiles([summaries[box][t]]) for t in summaries[box]})

    trainer = DirectFinetuneTrainer(
        model_path_of(cfg), proxies, reward_t, cfg=acfg, geom=geom,
        soft_cfg=scfg, device=device, arm=str(args.arm), phase_cfg=pscfg,
        rockstar_cfg=rcfg, f_valid_center=arm_f_valid_center(cfg))
    trainer.use_amp = False

    lr = batch["lr"].to(device)
    noise = {k: v.to(device) for k, v in batch["noise"].items()}
    hr = batch["hr"].to(device)
    lr_f, noise_f, _, _ = fold_draws({"lr": lr, "noise": noise})
    with torch.no_grad():
        cand0 = trim_to_tile(trainer.frozen(lr_f, noise=noise_f), geom).float()
    box_sum = batch["box_summary"].to(device)
    frozen_tile = batch["frozen_tile_summary"].to(device)

    # ---- the free field ---------------------------------------------------
    field = cand0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([field], lr=float(args.field_lr))

    # host-halo mask: the densest cells of the STARTING tile. Pinning their
    # density holds the host fixed while the reward works on everything else.
    with torch.no_grad():
        d0_host = density_from_disp(cand0, scfg)                 # (1,1,R,R,R)
        thr = torch.quantile(d0_host.flatten(), float(args.host_quantile))
        host_mask = (d0_host >= thr).float()
        host_norm = d0_host.pow(2).mul(host_mask).mean().clamp_min(1e-12)

    frames: List[np.ndarray] = [density_slice(cand0, scfg)]
    frame_steps: List[int] = [0]
    hist: List[Dict] = []
    log = (run / "tile_overfit")
    log.mkdir(parents=True, exist_ok=True)
    tag = f"{box}_t{tile}_{args.arm}" + (f"__{args.label}" if args.label else "")
    snap_dir = log / f"snaps_{tag}"
    snap_iters: List[int] = []
    if int(args.ckpt_every) > 0:
        snap_dir.mkdir(parents=True, exist_ok=True)
        np.save(snap_dir / "field_it0000.npy", cand0[0].detach().cpu().numpy())
        snap_iters.append(0)
    jl = log / f"metrics_{tag}.jsonl"
    jl.write_text("")

    best_q = -1e30
    best_iter = 0
    t0 = time.time()
    for it in range(int(args.iters) + 1):
        feats = trainer._extract(field, cand0, lr=lr_f)
        all_dr = trainer.proxies.delta_rewards_all(
            trainer.reward, feats.to(torch.float64), box_sum, frozen_tile,
            w_joint=w_joint, w_occ=w_occ)
        q = ProxyEnsemble.q_safe(all_dr["dR_combined"], beta=beta)
        reward_term = -q["q_safe"].mean().to(torch.float32)
        Lp = trainer._density_power_loss(field, hr)
        Ll = trainer._low_k_loss(field, cand0)
        Lprox = (field - cand0).pow(2).mean()
        if w_host > 0:
            d_now = density_from_disp(field, scfg)
            Lhost = ((d_now - d0_host).pow(2) * host_mask).mean() / host_norm
        else:
            Lhost = torch.zeros((), device=device)
        loss = (w_reward * reward_term + w_dens * Lp + w_lowk * Ll
                + w_prox * Lprox + w_host * Lhost)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_([field], float(args.grad_clip))
        opt.step()

        qv = float(q["q_safe"].mean())
        row = {"iter": it, "q_safe": qv,
               "dR_occ": float(all_dr["dR_occ"].mean()),
               "dR_combined": float(all_dr["dR_combined"].mean()),
               "loss": float(loss.detach()),
               "L_density": float(Lp.detach()), "L_lowk": float(Ll.detach()),
               "low_k_change": trainer.low_k_change(float(Ll.detach())),
               "L_prox": float(Lprox.detach()), "L_host": float(Lhost.detach()),
               "field_grad_norm": float(gnorm),
               "elapsed_s": round(time.time() - t0, 1)}
        hist.append(row)
        with jl.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        if it % int(args.save_every) == 0:
            frames.append(density_slice(field, scfg))
            frame_steps.append(it)
        if int(args.ckpt_every) > 0 and it > 0 and it % int(args.ckpt_every) == 0:
            np.save(snap_dir / f"field_it{it:04d}.npy",
                    field[0].detach().cpu().numpy())
            snap_iters.append(it)
        if it % 50 == 0:
            print(f"  it {it:4d}  q_safe {qv:+.4f}  dR_occ {row['dR_occ']:+.4f}  "
                  f"low_k {row['low_k_change']:.2e}  Ldens {row['L_density']:.2e}  "
                  f"Lhost {row['L_host']:.2e}", flush=True)

        # tol <= 0 disables the plateau early-stop: the run uses its full ITERS
        # budget, so "the end" is a fixed iteration count (what a slow, strongly
        # guarded run wants -- it may still be gently descending at the end).
        if float(args.tol) > 0:
            if qv > best_q + float(args.tol):
                best_q, best_iter = qv, it
            elif it - best_iter >= int(args.patience) * int(args.save_every):
                banner(f"converged: q_safe plateaued at {best_q:+.4f} "
                       f"(no +{args.tol} gain for {it - best_iter} iters)")
                break

    frames.append(density_slice(field, scfg))
    frame_steps.append(hist[-1]["iter"])

    with torch.no_grad():
        d0 = density_from_disp(cand0, scfg)[0, 0].cpu().numpy()
        d1 = density_from_disp(field, scfg)[0, 0].cpu().numpy()
    npz = log / f"tile_overfit_{tag}.npz"
    np.savez_compressed(
        npz,
        frames=np.stack(frames).astype(np.float32),
        frame_steps=np.array(frame_steps),
        q_hist=np.array([r["q_safe"] for r in hist]),
        dr_occ_hist=np.array([r["dR_occ"] for r in hist]),
        lowk_hist=np.array([r["low_k_change"] for r in hist]),
        iters=np.array([r["iter"] for r in hist]),
        density_initial=d0.astype(np.float32),
        density_final=d1.astype(np.float32),
    )
    np.save(log / f"field_initial_{tag}.npy", cand0[0].detach().cpu().numpy())
    np.save(log / f"field_final_{tag}.npy", field[0].detach().cpu().numpy())
    if int(args.ckpt_every) > 0 and hist[-1]["iter"] not in snap_iters:
        np.save(snap_dir / f"field_it{hist[-1]['iter']:04d}.npy",
                field[0].detach().cpu().numpy())
        snap_iters.append(hist[-1]["iter"])
    write_json(log / f"tile_overfit_{tag}.json", {
        "run_name": args.run_name, "arm": args.arm, "box": box, "tile": tile,
        "seed": args.seed, "tag": tag, "iters_run": hist[-1]["iter"],
        "q_start": hist[0]["q_safe"], "q_end": hist[-1]["q_safe"],
        "dR_occ_start": hist[0]["dR_occ"], "dR_occ_end": hist[-1]["dR_occ"],
        "low_k_change_end": hist[-1]["low_k_change"],
        "weights": {"reward": w_reward, "density": w_dens, "lowk": w_lowk,
                    "prox": w_prox, "host": w_host, "beta": beta,
                    "field_lr": args.field_lr},
        "host_quantile": float(args.host_quantile),
        "npz": str(npz), "n_frames": len(frames),
        "snap_dir": str(snap_dir) if snap_iters else "",
        "snap_iters": snap_iters,
    })
    banner(f"q_safe {hist[0]['q_safe']:+.4f} -> {hist[-1]['q_safe']:+.4f}  "
           f"dR_occ {hist[0]['dR_occ']:+.4f} -> {hist[-1]['dR_occ']:+.4f}  "
           f"low_k_change {hist[-1]['low_k_change']:.2e}")
    print(f"  wrote {npz}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
