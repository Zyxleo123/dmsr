"""Evaluate a trained latent flow via the 64->512 cascade (Experiments 3-5).

Runs the latent-flow cascade at several CFG scales and diversity seeds on a real
HR field, compares against base-only and AE-oracle baselines, verifies hard LR
consistency, and writes ``metrics.json``, ``spectra.npz`` and diagnostic PNGs.

Usage::

    python -m cosmo_sr.eval.eval_latent_flow \
        --config configs/poc_latent_flow_real3_paired_only.yaml \
        --checkpoint runs/poc_latent_flow_real3_paired_only/ckpt_last.pt \
        --hr .../set15.npy --crop 128 --n-steps 20 --diversity 3 \
        --cfg-scales 0.0 1.0 2.0 --out runs/.../eval_set15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from ..data.field_io import load_field
from ..data.crops import periodic_crop
from ..operators.multiscale import MultiScaleOperators
from ..operators.base_upscaler import IdentityUpscaler, consistent_base
from ..inference.latent_flow_sample import super_resolve_latent_cascade, sample_latent_step
from ..train.common import load_checkpoint, save_slice_png
from ..train.train_latent_flow import (
    build_latent_flow,
    build_base_upscaler,
    load_frozen_ae,
    load_frozen_degrader,
)
from ..eval.flow_eval import highk_power_ratio, mean_power_spectrum, sr2_power_summary
from ..utils.config import load_config


def build_pyramid(crop: torch.Tensor, full_res: int, n_levels: int) -> Dict[int, torch.Tensor]:
    levels = [crop]
    for _ in range(n_levels - 1):
        levels.append(F.avg_pool3d(levels[-1], kernel_size=2, stride=2))
    return {full_res // (2 ** lvl): t for lvl, t in enumerate(levels)}


@torch.no_grad()
def run(cfg: Dict, checkpoint: str, hr_path: str, crop_size: int, out_dir: str,
        n_steps: int = 20, diversity: int = 3, cfg_scales=(0.0, 1.0, 2.0),
        device: str = "cpu") -> Dict[str, float]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    factor = int(cfg.get("factor", 2))
    channels = int(cfg.get("ae", {}).get("channels", 6))
    full_res = int(cfg.get("data", {}).get("full_res", 512))
    n_levels = int(cfg.get("data", {}).get("n_levels", 4))
    resolutions: List[int] = list(cfg.get("resolutions", [64, 128, 256]))

    ops = MultiScaleOperators(factor).to(dev)
    base_upscaler = build_base_upscaler(cfg, channels, factor, dev)
    ae = load_frozen_ae(cfg, dev)
    deg = load_frozen_degrader(cfg, factor, dev)
    model = build_latent_flow(cfg, ae.latent_channels, channels).to(dev)
    state = load_checkpoint(checkpoint, model, map_location=dev)
    if isinstance(state.get("extra"), dict) and "base_upscaler" in state["extra"]:
        try:
            base_upscaler.load_state_dict(state["extra"]["base_upscaler"])
        except Exception:
            pass
    model.eval()

    field = load_field(hr_path, mmap=True)
    Ng = field.shape[1]
    rng = np.random.default_rng(0)
    start = tuple(int(rng.integers(0, Ng)) for _ in range(3))
    crop_np = periodic_crop(field, start, crop_size, pad=0)
    crop = torch.from_numpy(np.ascontiguousarray(crop_np)).float().unsqueeze(0).to(dev)
    pyr = build_pyramid(crop, full_res, n_levels)
    res_present = [r for r in resolutions if r in pyr and 2 * r in pyr]
    identity = IdentityUpscaler(factor).to(dev)

    metrics: Dict[str, float] = {}

    # ---- per-octave baselines (true coarse) ----
    for R in res_present:
        x_R = pyr[R]
        x_2R = pyr[2 * R]
        base = consistent_base(identity, ops, x_R)
        base_x_mse = float(F.mse_loss(base, x_2R))
        z_true = ae.encode(ops.P_null(x_2R - base))
        x_ae = base + ops.P_null(ae.decode(z_true))
        ae_x_mse = float(F.mse_loss(x_ae, x_2R))
        metrics[f"val/base_x_mse_R{R}"] = base_x_mse
        metrics[f"val/ae_oracle_x_mse_R{R}"] = ae_x_mse
        metrics[f"val/base_highk_R{R}"] = float(highk_power_ratio(base, x_2R)["highk_power_ratio"])
        metrics[f"val/ae_oracle_highk_R{R}"] = float(highk_power_ratio(x_ae, x_2R)["highk_power_ratio"])

    # ---- cascade per CFG scale ----
    coarsest = min(res_present)
    slices_done = False
    spectra: Dict[str, np.ndarray] = {}
    for s in cfg_scales:
        casc = super_resolve_latent_cascade(
            model, ae, ops, base_upscaler, pyr[coarsest], tuple(res_present),
            n_steps=n_steps, cfg_scale=float(s), seed=0,
        )
        coarse_chain = {coarsest * 2: pyr[coarsest]}
        prev = pyr[coarsest]
        for R in res_present:
            coarse_chain[2 * R] = prev
            prev = casc[2 * R]
        for R in res_present:
            lvl = 2 * R
            x_hat = casc[lvl]
            x_true = pyr[lvl]
            coarse_in = coarse_chain[lvl]
            denom = float(torch.mean(coarse_in ** 2)) or 1.0
            cons = float(torch.mean((ops.A(x_hat) - coarse_in) ** 2)) / denom
            x_mse = float(F.mse_loss(x_hat, x_true))
            pr = highk_power_ratio(x_hat, x_true)
            base = consistent_base(identity, ops, pyr[R])
            res_gen = ops.P_null(x_hat - consistent_base(base_upscaler, ops, coarse_in))
            res_true = ops.P_null(x_true - base)
            vt = float(torch.var(res_true)) or 1.0
            respow = float(torch.var(res_gen)) / vt
            finite = float(torch.isfinite(x_hat).float().mean())
            cc = sr2_power_summary(x_hat, x_true)["cross_corr_mean_per_channel"]
            cc = float(np.nanmean(cc)) if len(cc) else float("nan")
            d_out = deg(x_hat, torch.full((x_hat.shape[0],), float(R), device=dev))
            d_cons = float(torch.mean((d_out - coarse_in) ** 2)) / denom
            tag = f"cfg{s}"
            metrics[f"val/{tag}/consistency_rel_R{R}"] = cons
            metrics[f"val/{tag}/D_consistency_rel_R{R}"] = d_cons
            metrics[f"val/{tag}/x_mse_R{R}"] = x_mse
            metrics[f"val/{tag}/highk_R{R}"] = float(pr["highk_power_ratio"])
            metrics[f"val/{tag}/allk_R{R}"] = float(pr["allk_power_ratio"])
            metrics[f"val/{tag}/respow_R{R}"] = respow
            metrics[f"val/{tag}/cross_corr_R{R}"] = cc
            metrics[f"val/{tag}/finite_frac_R{R}"] = finite
            metrics[f"val/{tag}/flow_x_mse_R{R}"] = x_mse
            metrics[f"val/{tag}/flow_highk_R{R}"] = float(pr["highk_power_ratio"])
            base_x_mse = metrics.get(f"val/base_x_mse_R{R}", float("nan"))
            metrics[f"val/{tag}/flow_vs_base_improvement_R{R}"] = (
                base_x_mse / (x_mse if x_mse > 0 else 1.0)
            )
            ae_x_mse = metrics.get(f"val/ae_oracle_x_mse_R{R}", float("nan"))
            metrics[f"val/{tag}/flow_vs_ae_oracle_gap_R{R}"] = x_mse - ae_x_mse

        # slices/spectra for the top level
        top = 2 * max(res_present)
        if not slices_done or s in (1.0, 2.0):
            if s in (1.0, 2.0):
                save_slice_png(out / f"central_slice_sample_cfg{int(s)}.png",
                               casc[top][0].cpu().numpy(), title=f"sample cfg={s}")
        if not slices_done:
            base_top = consistent_base(identity, ops, pyr[max(res_present)])
            save_slice_png(out / "central_slice_base.png", base_top[0].cpu().numpy(), title="base")
            save_slice_png(out / "central_slice_hr.png", pyr[top][0].cpu().numpy(), title="HR")
            kb, pb = mean_power_spectrum(base_top[0].cpu())
            kf, pf = mean_power_spectrum(casc[top][0].cpu())
            kh, ph = mean_power_spectrum(pyr[top][0].cpu())
            spectra.update({"k": kh, "power_base": pb, "power_flow": pf, "power_hr": ph})
            slices_done = True

    # ---- diversity across seeds (top level, cfg=1) ----
    top = 2 * max(res_present)
    div_scale = 1.0 if 1.0 in cfg_scales else float(cfg_scales[0])
    samples = []
    for d in range(max(diversity, 1)):
        casc = super_resolve_latent_cascade(
            model, ae, ops, base_upscaler, pyr[coarsest], tuple(res_present),
            n_steps=n_steps, cfg_scale=div_scale, seed=1000 + d,
        )
        samples.append(casc[top])
    stack = torch.stack(samples, dim=0)
    per_voxel_std = stack.std(dim=0)
    signal_std = stack.mean(dim=0).std().clamp_min(1e-12)
    metrics[f"final/zdiv_{top}"] = float(per_voxel_std.mean() / signal_std)
    save_slice_png(out / "diversity_slice_std.png", per_voxel_std[0].cpu().numpy(),
                   title="diversity std")

    # ---- final headline metrics (cfg=1 preferred) ----
    ftag = f"cfg{div_scale}"
    for lvl in (128, 256, 512):
        R = lvl // 2
        if f"val/{ftag}/consistency_rel_R{R}" in metrics:
            metrics[f"final/consistency_rel_{lvl}"] = metrics[f"val/{ftag}/consistency_rel_R{R}"]
    top_R = top // 2
    for name in ("highk", "allk", "respow", "x_mse", "finite_frac"):
        key = f"val/{ftag}/{name}_R{top_R}"
        if key in metrics:
            metrics[f"final/{name}_{top}"] = metrics[key]
    if f"val/base_x_mse_R{top_R}" in metrics:
        metrics[f"final/base_x_mse_{top}"] = metrics[f"val/base_x_mse_R{top_R}"]
        fm = metrics.get(f"final/x_mse_{top}", float("nan"))
        metrics[f"final/flow_vs_base_improvement_{top}"] = (
            metrics[f"final/base_x_mse_{top}"] / (fm if fm and fm > 0 else 1.0)
        )

    # ---- plots ----
    _save_power_plots(out, spectra)

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if spectra:
        np.savez(out / "spectra.npz", **{k: np.asarray(v) for k, v in spectra.items()})
    print(f"[eval_latent_flow] wrote {out/'metrics.json'}")
    for k in sorted(metrics):
        if k.startswith("final/"):
            print(f"  {k}: {metrics[k]:.4g}")
    return metrics


def _save_power_plots(out: Path, spectra: Dict[str, np.ndarray]) -> None:
    if not spectra:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = spectra.get("k")
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, label in (("power_base", "base"), ("power_flow", "flow"), ("power_hr", "HR")):
        p = spectra.get(name)
        if p is not None and k is not None:
            n = min(len(k), len(p))
            ax.loglog(k[:n], np.clip(p[:n], 1e-30, None), label=label)
    ax.set_xlabel("k"); ax.set_ylabel("P(k)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "power.png", dpi=100); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    pb, pf, ph = spectra.get("power_base"), spectra.get("power_flow"), spectra.get("power_hr")
    if k is not None and ph is not None:
        n = min(len(k), len(ph))
        if pf is not None:
            ax.semilogx(k[:n], np.clip(pf[:n], 1e-30, None) / np.clip(ph[:n], 1e-30, None), label="flow/HR")
        if pb is not None:
            ax.semilogx(k[:n], np.clip(pb[:n], 1e-30, None) / np.clip(ph[:n], 1e-30, None), label="base/HR")
    ax.axhline(1.0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("k"); ax.set_ylabel("P/P_HR"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "residual_power.png", dpi=100); plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate latent flow cascade (Experiments 3-5)")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--hr", required=True, type=str)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--diversity", type=int, default=3)
    p.add_argument("--cfg-scales", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--set", nargs="*", default=None,
                   help="dotted config overrides, e.g. ae_checkpoint=path")
    args = p.parse_args(argv)
    from ..utils.config import apply_overrides

    cfg = apply_overrides(load_config(args.config), args.set)
    run(cfg, args.checkpoint, args.hr, args.crop, args.out,
        n_steps=args.n_steps, diversity=args.diversity,
        cfg_scales=tuple(args.cfg_scales), device=args.device)


if __name__ == "__main__":
    main()
