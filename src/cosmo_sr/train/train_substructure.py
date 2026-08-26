"""Train the moment-constrained substructure module (pilot step 5, Option B).

``docs/sr2_substructure_module.md`` section 4 (native Lagrangian tiles) with the
high-pass replaced by the moment target of ``docs/sr2_moment_constraint.md``.
The frozen SR2 field is left untouched; a conditional flow-matching network learns
to emit the projected residual ``d`` per ``64^3`` tile, conditioned on the SR2
tile and the LR host channels. Inference (``scripts/reward/sample_substructure_
field.py``) assembles the tiles, applies ``Pi`` once, adds to SR2, and Rockstar
gates the reassembled box.

What this run is and is not: it is the cheapest decisive test of "does the
mechanism move the subhalo ratio at all," trained in-sample on set8 (the one box
with both host features and a moment target). It samples a conditional
distribution and only *adds* to a frozen SR2, so in-sample gaming is limited, but
generalization stays the honest follow-up if it passes.

    python -m cosmo_sr.train.train_substructure --config configs/substructure_set8.yaml
    python -m cosmo_sr.train.train_substructure --config configs/substructure_set8.yaml --smoke
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from ..models.flow_unet import UNetResidualFlowModel
from ..utils.config import load_config
from ..utils.seed import seed_everything
from . import common
from . import substructure_data as sd


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def _reward_root() -> Path:
    return Path(os.environ.get(
        "DMSR_REWARD_ROOT", "/zfsauton/scratch/yixiz/DMSR/dmsr_reward"))


def _build_model(model_cfg: Dict[str, Any], device) -> UNetResidualFlowModel:
    """The velocity net: 6-channel I/O, SR2 conditioning + host context, FiLM t."""
    return UNetResidualFlowModel(
        channels=6,
        width=int(model_cfg.get("width", 64)),
        num_levels=int(model_cfg.get("num_levels", 3)),
        blocks_per_level=int(model_cfg.get("blocks_per_level", 1)),
        norm=str(model_cfg.get("norm", "group")),
        num_groups=int(model_cfg.get("num_groups", 8)),
        activation=str(model_cfg.get("activation", "silu")),
        use_resblocks=bool(model_cfg.get("use_resblocks", True)),
        use_film=True,                       # flow time t must reach the net
        embed_dim=int(model_cfg.get("embed_dim", 128)),
        zero_init_tail=bool(model_cfg.get("zero_init_tail", True)),
        use_checkpoint=bool(model_cfg.get("use_checkpoint", False)),
        context_channels=sd.HOST_CHANNELS,
        factor=1,                            # conditioning and target share the grid
        padding="same",
    ).to(device)


def _load_boxes(cfg: Dict[str, Any], device, force_sr2: bool):
    """Frozen SR2 box + moment target + host features + HR field for a box.

    Returns ``(boxes, feat, hr)`` -- ``feat`` and the (memory-mapped) HR field
    are kept for the in-loop Rockstar eval; only cubes of HR are ever read.
    """
    from ..features.lagrangian_host import LagrangianHostFeatures
    from ..tts.srs_noise import load_controlled_generator

    import sys
    proj = Path(__file__).resolve().parents[3]
    for p in (proj / "src", proj / "scripts" / "reward"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from _sr2_direct import (  # noqa: E402
        geometry_of, load_direct_config, load_hr, load_lr, model_path_of)

    box = str(cfg["box"])
    direct_cfg = load_direct_config(
        Namespace(config=cfg.get("direct_config",
                                 "configs/reward/sr2_direct_finetune.yaml"),
                  overrides=[]))
    geom = geometry_of(direct_cfg)
    seed = int(cfg.get("base_seed", 0))

    mt_dir = _reward_root() / "moment_target" / box
    target_path = mt_dir / f"{box}_moment_target.npy"
    if not target_path.is_file():
        raise SystemExit(
            f"no moment target at {target_path}; run "
            "scripts/slurm/submit_moment_target.sh first")
    feat_path = (_reward_root() / "lagrangian_host" / box
                 / f"{box}_lagrangian_host.npz")
    if not feat_path.is_file():
        raise SystemExit(f"no host features at {feat_path}")

    print(f"[setup] generating/loading frozen SR2 box for {box} ...", flush=True)
    model = load_controlled_generator(
        model_path_of(direct_cfg),
        in_chan=int(direct_cfg.get("model", {}).get("in_chan", 6)),
        out_chan=int(direct_cfg.get("model", {}).get("out_chan", 6)),
        scale_factor=geom.scale_factor, device=device, eval_mode=True)
    for p in model.parameters():
        p.requires_grad_(False)
    lr = load_lr(direct_cfg, box)
    sr2_box = sd.load_or_make_sr2_box(
        model, lr, geom, device, seed, int(cfg.get("sr2_batch", 8)),
        cache_path=mt_dir / f"{box}_sr2_box.npy", force=force_sr2)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    target_box = np.asarray(np.load(target_path)).astype(np.float32)
    feat = LagrangianHostFeatures.from_npz(str(feat_path))
    boxes = sd.SubstructureBoxes.build(
        sr2_box, target_box, feat,
        k=int(cfg.get("scale_k", 3)), eps=float(cfg.get("scale_eps", 1e-3)))
    top = np.argsort(-boxes.weights)[:5]
    print(f"[setup] tile weights: top-5 {[int(t) for t in top]} "
          f"= {boxes.weights[top].round(4).tolist()}; "
          f"{int(np.sum(boxes.weights > 2.0 / sd.N_TILES))} tiles above uniform",
          flush=True)
    hr = load_hr(direct_cfg, box)          # memory-mapped; eval reads only cubes
    return boxes, feat, hr


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _conditional_spread(model, boxes: sd.SubstructureBoxes, tile_id: int,
                        device, n_steps: int) -> Dict[str, float]:
    """Seed-to-seed spread of the generated ``d`` on one fixed tile (item 5).

    The collapse alarm: two samples on identical conditioning. ``rel_spread`` near
    0 means the flow is deterministic (the failure that produced the whole
    substructure document); a healthy conditional is broad at subhalo scale.
    """
    tt = boxes.tile_tensors(tile_id, device)
    x_in = tt["x_in"][None]
    ctx = tt["context"][None]
    g1 = torch.Generator(device=device.type).manual_seed(1)
    g2 = torch.Generator(device=device.type).manual_seed(2)
    d1 = sd.integrate_tile(model, x_in, ctx, n_steps, generator=g1)
    d2 = sd.integrate_tile(model, x_in, ctx, n_steps, generator=g2)
    diff = float(torch.sqrt(torch.mean((d1 - d2) ** 2)))
    mean = float(torch.sqrt(torch.mean((0.5 * (d1 + d2)) ** 2)))
    return {"cond_spread_abs": diff,
            "cond_spread_rel": diff / (mean + 1e-8),
            "cond_sample_rms": mean}


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def _smoke_batch(batch: int, device) -> Dict[str, torch.Tensor]:
    """Random tile-shaped tensors for a CPU smoke run (no 512^3 machinery)."""
    n = 16                                    # divisible by 2**num_levels
    shape = (batch, 6, n, n, n)
    return {
        "x_in": torch.randn(shape, device=device),
        "x1": torch.randn(shape, device=device),
        "context": torch.randn((batch, sd.HOST_CHANNELS, n, n, n), device=device),
    }


def train(cfg: Dict[str, Any], smoke: bool = False) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {})
    seed = int(train_cfg.get("seed", 0))
    seed_everything(seed)
    device = common.select_device("cpu" if smoke else train_cfg.get("device"))

    steps = int(train_cfg.get("steps", 20000))
    bs = int(train_cfg.get("batch_size", 8))
    lr = float(train_cfg.get("lr", 2e-4))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 2000))
    eval_every = int(train_cfg.get("eval_every", 1000))
    eval_steps = int(train_cfg.get("eval_sample_steps", 20))
    amp_enabled = bool(train_cfg.get("amp", False))
    if smoke:
        steps, bs, log_every, eval_every, save_every = 8, 2, 1, 4, 0

    if smoke:
        model_cfg = dict(cfg.get("model", {}))
        model_cfg.setdefault("num_levels", 2)     # 16^3 tiles -> <=2 levels
        model = _build_model(model_cfg, device)
        boxes = feat = hr = None
    else:
        model = _build_model(cfg.get("model", {}), device)
        boxes, feat, hr = _load_boxes(
            cfg, device, force_sr2=bool(cfg.get("force_sr2", False)))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    autocast, scaler = common.amp_components(amp_enabled, device)

    run_dir = Path(os.environ.get(
        "SUBSTRUCTURE_RUN_DIR",
        cfg.get("output", {}).get("run_dir", "runs/substructure_set8")))
    common.init_run_dir(run_dir, cfg)
    use_wandb = (not smoke) and common.maybe_init_wandb(cfg, run_dir, "substructure")
    logger = common.CSVLogger(run_dir, use_wandb=use_wandb)

    rng = np.random.default_rng(seed)
    spread_tile = int(cfg.get("spread_tile", -1))
    if boxes is not None and spread_tile < 0:
        spread_tile = int(np.argmax(boxes.weights))   # a host-rich tile

    # In-loop Rockstar eval on the cluster host (opt-in; the physical signal).
    rs_cfg = dict(cfg.get("eval_rockstar", {}))
    rs_on = (not smoke) and boxes is not None and bool(rs_cfg.get("enabled", True))
    rs_every = int(rs_cfg.get("every", 4000))
    rs_cache: Dict[str, Any] = {}
    rs_row = None
    if rs_on:
        from . import substructure_eval as se
        rs_row = se.host_row_of(feat, rs_cfg.get("host_id"))
        rs_work = run_dir / "rockstar_eval"
        print(f"[setup] rockstar eval on host row {rs_row} "
              f"(id {int(feat.table.host_id[rs_row])}, "
              f"logM {np.log10(feat.table.mvir[rs_row]):.2f}) every {rs_every} steps",
              flush=True)

    model.train()
    first: Dict[str, float] = {}
    last: Dict[str, float] = {}
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        batch = (_smoke_batch(bs, device) if smoke
                 else boxes.sample_batch(rng, bs, device))
        with autocast():
            loss = sd.cfm_loss(model, batch["x_in"], batch["x1"], batch["context"])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        will_log = step % log_every == 0 or step == 1 or step == steps
        if will_log:
            scaler.unscale_(optimizer)
            gnorm = common.grad_global_norm(model.parameters())
        scaler.step(optimizer)
        scaler.update()

        logs = {"loss": float(loss.detach())}
        last = logs
        if not first:
            first = dict(logs)
        if will_log:
            logs["grad_norm"] = gnorm
            logs.update(common.system_metrics(device, time.perf_counter() - t0))
            do_eval = (not smoke) and boxes is not None and (
                step % eval_every == 0 or step == steps)
            if do_eval:
                model.eval()
                logs.update(_conditional_spread(
                    model, boxes, spread_tile, device, eval_steps))
                model.train()
            if rs_on and (step % rs_every == 0 or step == steps):
                model.eval()
                t_rs = time.perf_counter()
                logs.update(se.region_rockstar_eval(
                    model, boxes, feat, hr, rs_row, work_dir=rs_work,
                    cache=rs_cache,
                    region_sites=int(rs_cfg.get("region_sites", 192)),
                    n_steps=int(rs_cfg.get("n_steps", eval_steps)), device=device))
                logs["rs_eval_s"] = time.perf_counter() - t_rs
                model.train()
            logger.log(step, logs)
        if save_every > 0 and step % save_every == 0:
            common.save_checkpoint(run_dir / f"ckpt_{step}.pt", model, optimizer, step)

    common.save_checkpoint(run_dir / "ckpt_last.pt", model, optimizer, steps)
    logger.close()
    common.finish_wandb()
    return {"run_dir": str(run_dir), "first": first, "last": last, "steps": steps,
            "checkpoint": str(run_dir / "ckpt_last.pt")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Substructure module flow training")
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="short CPU smoke run")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    result = train(cfg, smoke=args.smoke)
    msg = " ".join(f"{k}={v:.4g}" for k, v in result["last"].items())
    print(f"[substructure] done: steps={result['steps']} last[{msg}] "
          f"run_dir={result['run_dir']}")


if __name__ == "__main__":
    main()
