#!/usr/bin/env python
"""Sample a full-box catnorm field from the trained substructure module.

Inference rule of ``docs/sr2_moment_constraint.md`` section 5.1 and
``docs/sr2_substructure_module.md`` section 4: regenerate the frozen SR2 box,
sample the projected residual ``d`` tile by tile from the flow, un-normalize it
by the same local scale ``s`` training used, assemble the whole-box ``d``, apply
the per-host affine-moment projector ``Pi`` once (making ``d in range(Pi)``
exact), and write ``Psi_final = Psi_SR2 + Pi(d)`` as a ``(6, 512, 512, 512)``
catnorm ``.npy`` -- the exact format ``scripts/reward/catalog_summaries.py
--source candidate --field`` consumes, so the existing flow->Rockstar catalog and
compare stages run unchanged.

    python scripts/reward/sample_substructure_field.py \
        --config configs/substructure_set8.yaml \
        --checkpoint <run>/ckpt_last.pt \
        --box set8 --n-steps 20 --seed 0 --out <field>.npy
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _sr2_direct import (  # noqa: E402
    geometry_of, load_direct_config, load_lr, model_path_of)
from cosmo_sr.features.lagrangian_host import LagrangianHostFeatures  # noqa: E402
from cosmo_sr.features.moment_constraint import from_features  # noqa: E402
from cosmo_sr.train import substructure_data as sd  # noqa: E402
from cosmo_sr.train.train_substructure import _build_model, _reward_root  # noqa: E402
from cosmo_sr.tts.srs_noise import load_controlled_generator  # noqa: E402
from cosmo_sr.utils.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--box", default=None)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.is_file() and not args.overwrite:
        print(f"cached -> {out_path}", flush=True)
        return

    cfg = load_config(args.config)
    box = str(args.box or cfg["box"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    direct_cfg = load_direct_config(Namespace(
        config=cfg.get("direct_config", "configs/reward/sr2_direct_finetune.yaml"),
        overrides=[]))
    geom = geometry_of(direct_cfg)
    base_seed = int(cfg.get("base_seed", 0))

    mt_dir = _reward_root() / "moment_target" / box
    feat_path = (_reward_root() / "lagrangian_host" / box
                 / f"{box}_lagrangian_host.npz")
    if not feat_path.is_file():
        raise SystemExit(f"no host features at {feat_path}")

    # --- frozen SR2 box (cached from training) + local scale ------------------
    print(f"[sample] frozen SR2 box for {box} ...", flush=True)
    gen_model = load_controlled_generator(
        model_path_of(direct_cfg),
        in_chan=int(direct_cfg.get("model", {}).get("in_chan", 6)),
        out_chan=int(direct_cfg.get("model", {}).get("out_chan", 6)),
        scale_factor=geom.scale_factor, device=device, eval_mode=True)
    for p in gen_model.parameters():
        p.requires_grad_(False)
    lr = load_lr(direct_cfg, box)
    sr2_box = sd.load_or_make_sr2_box(
        gen_model, lr, geom, device, base_seed, int(cfg.get("sr2_batch", 8)),
        cache_path=mt_dir / f"{box}_sr2_box.npy")
    del gen_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    s_disp, s_vel = sd.scale_fields(
        sr2_box, k=int(cfg.get("scale_k", 3)), eps=float(cfg.get("scale_eps", 1e-3)))
    feat = LagrangianHostFeatures.from_npz(str(feat_path))
    host = sd.host_context_stack(feat)

    # --- flow model ----------------------------------------------------------
    model = _build_model(cfg.get("model", {}), device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    # --- sample d tile by tile ----------------------------------------------
    d_box = np.empty((6, sd.NG_HR, sd.NG_HR, sd.NG_HR), dtype=np.float32)
    t0 = time.time()
    for t in range(sd.N_TILES):
        ix, iy, iz = sd.tile_coord(t)
        hx, hy, hz = sd.hr_block(ix, iy, iz)
        lx, ly, lz = sd.lr_block(ix, iy, iz)
        sr2_t = torch.from_numpy(
            np.ascontiguousarray(sr2_box[:, hx, hy, hz])).to(device)
        sdt = torch.from_numpy(s_disp[hx, hy, hz][None]).to(device)
        svt = torch.from_numpy(s_vel[hx, hy, hz][None]).to(device)
        crop = host[:, lx, ly, lz]
        f = sd.UPSAMPLE
        ctx = torch.from_numpy(np.ascontiguousarray(
            crop.repeat(f, axis=1).repeat(f, axis=2).repeat(f, axis=3))).to(device)
        x_in = sd.apply_scale(sr2_t, sdt, svt, undo=False)[None]
        g = torch.Generator(device=device.type).manual_seed(base_seed * 1_000_003 + t)
        d_norm = sd.integrate_tile(model, x_in, ctx[None], args.n_steps, generator=g)[0]
        d = sd.apply_scale(d_norm, sdt, svt, undo=True)
        d_box[:, hx, hy, hz] = d.float().cpu().numpy()
        if t % 64 == 0:
            print(f"    tiles {t}/{sd.N_TILES} ({time.time() - t0:.0f}s)", flush=True)

    # --- single final projection, then add to SR2 ----------------------------
    print("[sample] building Pi and projecting the assembled d ...", flush=True)
    proj = from_features(feat, mode=str(cfg.get("moment_mode", "affine")))
    d_proj = proj.apply(d_box)
    final = (sr2_box + d_proj).astype(np.float32)
    if final.shape != (6, sd.NG_HR, sd.NG_HR, sd.NG_HR):
        raise SystemExit(f"unexpected field shape {final.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npy")
    np.save(tmp, final)
    tmp.replace(out_path)
    rms_d = float(np.sqrt(np.mean(d_proj[0:3] ** 2)))
    print(f"[sample] wrote {out_path} shape={final.shape} "
          f"n_steps={args.n_steps} seed={args.seed} rms|Pi(d)_disp|={rms_d:.4f} "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
