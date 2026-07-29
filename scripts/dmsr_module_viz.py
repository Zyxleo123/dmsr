"""Visualise every module's output on one held-out crop.

Panels (Eulerian density delta unless noted; middle z-slice):
  1. SR results:  A_plus(y) floor | MSE mean | Flow C | E (mean+innov) | true HR
  2. Residues (SR - true HR), diverging, shared symmetric scale
  3. E decomposition in FIELD space (disp0, additive): A_plus(y) | mean m | innovation P_A(u) | m+innov vs true P_A(x_hr)
  4. Radial density power P(k) and cross-correlation r(k) vs HR  -> quantifies how the innovation restores high-k power the MSE mean smooths away.

Runs on CPU by default (set CUDA_VISIBLE_DEVICES to use a GPU); one small crop, no OOM risk.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from cosmo_sr.utils.config import load_config                       # noqa: E402
from cosmo_sr.dmsr.data import build_val_dataset, resolve_split     # noqa: E402
from cosmo_sr.dmsr.density import HighPassDensity, cellsizes        # noqa: E402
from cosmo_sr.dmsr.evaluate import auto_cross_power                 # noqa: E402
from cosmo_sr.dmsr.operator import NullSpaceOperator                # noqa: E402

# reuse the (mean_innovation-aware) loader we patched into dmsr_eval
import importlib.util                                               # noqa: E402
_spec = importlib.util.spec_from_file_location("dev", "scripts/dmsr_eval.py")
_dev = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_dev)
load_flow = _dev.load_flow


def _slice(field: torch.Tensor, ch: int = 0) -> np.ndarray:
    """Middle z-slice of channel `ch` of a (1,C,N,N,N) tensor."""
    n = field.shape[-1]
    return field[0, ch, :, :, n // 2].detach().cpu().numpy()


def _dens_slice(dens: torch.Tensor) -> np.ndarray:
    n = dens.shape[-1]
    return dens[0, 0, :, :, n // 2].detach().cpu().numpy()


def _pk(field: torch.Tensor, n_bins=24):
    p, _, _, k = auto_cross_power(field, field, n_bins)
    return k.cpu().numpy(), p.cpu().numpy()


def _rk(pred: torch.Tensor, true: torch.Tensor, n_bins=24):
    p_pp, p_tt, p_pt, k = auto_cross_power(pred, true, n_bins)
    rk = (p_pt / (p_pp * p_tt).clamp_min(1e-30).sqrt()).cpu().numpy()
    return k.cpu().numpy(), rk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/dmsr/figures/module_viz")
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--crop-lr", type=int, default=0,
                    help="override LR crop side (0=config). e.g. 16 -> 128^3 HR, 32 -> 256^3 HR")
    ap.add_argument("--crop", type=int, default=-1, help="crop index; -1 = auto-pick most structured")
    ap.add_argument("--scan", type=int, default=40, help="crops to scan when auto-picking")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-innov", type=int, default=3, help="innovation samples to show diversity")
    ap.add_argument("--n-steps", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ---- configs / checkpoints for each module ----
    cfg_e = load_config("configs/dmsr/mean_innovation_e.yaml")
    cfg_c = load_config("configs/dmsr/stage_c_critic_pairedlr.yaml")
    cfg_m = load_config("configs/dmsr/paired_deterministic.yaml")
    dcfg = cfg_e.get("data", {})
    factor = int(cfg_e.get("factor", 8))
    uc = dcfg.get("use_channels")
    channels = len(uc) if uc else int(dcfg.get("channels", 6))

    op = NullSpaceOperator(factor=factor).to(device)
    hr_cs, lr_cs = cellsizes(dcfg, factor)
    _lp = str(cfg_e.get("critic", {}).get("lowpass", "blockavg"))
    _dn = float(cfg_e.get("critic", {}).get("dis_norm", 6000.0))
    highpass = HighPassDensity(factor=factor, lowpass=_lp, cellsize=hr_cs, dis_norm=_dn)
    highpass_lr = HighPassDensity(factor=factor, lowpass=_lp, cellsize=lr_cs, dis_norm=_dn)
    dens = lambda f: highpass.density(f)
    dens_lr = lambda f: highpass_lr.density(f)   # LR density uses the LR cellsize

    print("[viz] loading modules ...")
    mse = load_flow(cfg_m, channels, "runs/dmsr/paired_deterministic/ckpt_best.pt", device)
    flow_c = load_flow(cfg_c, channels, "runs/dmsr/stage_c_s0/ckpt_best.pt", device)
    flow_e = load_flow(cfg_e, channels, "runs/dmsr/mean_innovation_e_s0/ckpt_best.pt", device)

    # ---- pick a crop ----
    split = resolve_split(dcfg)
    crop_lr = args.crop_lr if args.crop_lr > 0 else int(dcfg.get("crop_lr", 8))
    print(f"[viz] crop_lr={crop_lr} -> HR {crop_lr*factor}^3")
    ds = build_val_dataset(split, crop_lr=crop_lr, scale_factor=factor,
                           channels=int(dcfg.get("channels", 6)), use_channels=uc,
                           mmap=bool(dcfg.get("mmap", True)),
                           max_crops=max(args.scan, args.crop + 1), which=args.split)
    if args.crop >= 0:
        idx = args.crop
    else:
        best, idx = -1.0, 0
        for i in range(min(args.scan, len(ds))):
            b = ds[i]
            s = float(dens(b["hr"].unsqueeze(0).to(device)).std())
            if s > best:
                best, idx = s, i
        print(f"[viz] auto-picked crop {idx} (density std {best:.3f})")
    sample = ds[idx]
    y = sample["lr"].unsqueeze(0).to(device)
    x_hr = sample["hr"].unsqueeze(0).to(device)

    # ---- module outputs (fixed z shared by C and E) ----
    z = torch.randn(x_hr.shape, device=device)
    with torch.no_grad():
        x_base = op.A_plus(y)
        x_mse = mse.generate(y)                              # A_plus + P_A(mean)
        x_c = flow_c.generate(y, n_steps=args.n_steps, z=z)
        m = flow_e.mean_residual(y).detach()                # P_A(mean)  (E's frozen mean)
        innov = [flow_e.sample_innovation(y, n_steps=args.n_steps,
                                          z=(z if k == 0 else torch.randn_like(z)))
                 for k in range(args.n_innov)]
        x_e = [op.A_plus(y) + op.P_A(m + e) for e in innov]  # full E samples
        r_true = op.P_A(x_hr)                                # true null-space residual (field)
        e_gt = op.P_A(r_true - m)                            # innovation ground truth (irreducible)

    # densities
    d_hr = dens(x_hr); d_base = dens(x_base); d_mse = dens(x_mse)
    d_c = dens(x_c); d_e = dens(x_e[0])

    # ---------- FIG 1: SR density results (raw LR + SR chain + HR) ----------
    d_lr = dens_lr(y)                                    # native LR-grid density
    cols = [("A+(y)=LR upsampled", d_base), ("MSE mean", d_mse), ("Flow C", d_c),
            ("E mean+innov", d_e), ("true HR", d_hr)]
    vmin, vmax = np.percentile(_dens_slice(d_hr), [1, 99])
    fig, ax = plt.subplots(1, 6, figsize=(22.5, 4.1))
    # raw LR gets its own scale (different grid + amplitude); nearest = honest coarseness
    lv0, lv1 = np.percentile(_dens_slice(d_lr), [1, 99])
    im0 = ax[0].imshow(_dens_slice(d_lr), vmin=lv0, vmax=lv1, cmap="magma",
                       origin="lower", interpolation="nearest")
    ax[0].set_title(f"LR input ({d_lr.shape[-1]}³, native)", fontsize=12)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.02)
    for a, (t, d) in zip(ax[1:], cols):                  # SR chain + HR share the HR scale
        im = a.imshow(_dens_slice(d), vmin=vmin, vmax=vmax, cmap="magma", origin="lower")
        a.set_title(t, fontsize=12); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=list(ax[1:]), fraction=0.012, pad=0.01, label="overdensity δ")
    fig.suptitle(f"Density: raw LR → SR chain → true HR (test crop {idx}, z-slice)", fontsize=13)
    fig.savefig(out / "1_sr_density.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---------- FIG 2: residues (SR - HR) in density ----------
    res = [("MSE − HR", _dens_slice(d_mse) - _dens_slice(d_hr)),
           ("Flow C − HR", _dens_slice(d_c) - _dens_slice(d_hr)),
           ("E − HR", _dens_slice(d_e) - _dens_slice(d_hr))]
    rlim = max(np.abs(r[1]).max() for r in res)
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for a, (t, r) in zip(ax, res):
        im = a.imshow(r, vmin=-rlim, vmax=rlim, cmap="RdBu_r", origin="lower")
        rms = float(np.sqrt((r ** 2).mean()))
        a.set_title(f"{t}\nRMS={rms:.3f}", fontsize=12); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.016, pad=0.01, label="density residual (SR − HR)")
    fig.suptitle(f"Residue = SR − true HR (lower/whiter = better; crop {idx})", fontsize=13)
    fig.savefig(out / "2_residue_density.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---------- FIG 3: E decomposition in FIELD space (disp0, additive) ----------
    parts = [("A+(y) base [disp0]", _slice(x_base)),
             ("+ mean m [disp0]", _slice(m)),
             ("+ innovation P_A(u) [disp0]", _slice(innov[0])),
             ("true residual P_A(x_hr) [disp0]", _slice(r_true))]
    flim = max(np.abs(p[1]).max() for p in parts[1:])   # share scale on the residual-space panels
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for a, (t, s) in zip(ax, parts):
        lim = np.abs(s).max() if t.startswith("A+") else flim
        im = a.imshow(s, vmin=-lim, vmax=lim, cmap="RdBu_r", origin="lower")
        a.set_title(t, fontsize=11); a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    fig.suptitle("E decomposition (field space, additive): mean + innovation ≈ true null-space residual",
                 fontsize=13)
    fig.savefig(out / "3_e_decomposition.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---------- FIG 4: FIELD-SPACE transfer T(k) and correlation r(k) ----------
    # This is where the mean-vs-flow tradeoff actually lives (rk_high / Tk_error_high
    # are field-space). Density P(k) is dominated by a few peaked halo cells and does
    # not isolate it. "mean only" == the deterministic paired_det reconstruction.
    def _pkf(f): return auto_cross_power(f, f, 24)[0]
    p_hr = _pkf(x_hr); k = auto_cross_power(x_hr, x_hr, 24)[3].cpu().numpy()
    k_lr = x_hr.shape[-1] / (2.0 * factor)
    series = [("A+(y) floor", x_base, "0.6"), ("mean only (det)", x_mse, "C0"),
              ("Flow C", x_c, "C1"), ("E (mean+innov)", x_e[0], "C2")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for lab, f, c in series:
        Tk = (_pkf(f) / p_hr.clamp_min(1e-30)).clamp_min(0).sqrt().cpu().numpy()
        a1.semilogx(k, Tk, label=lab, color=c, lw=(2.2 if lab.startswith("E") else 1.5))
    a1.axhline(1.0, ls="--", c="k", lw=0.8); a1.axvline(k_lr, ls=":", c="gray")
    a1.text(k_lr, 0.05, " k_LR", fontsize=9); a1.set_ylim(0, 1.35)
    a1.set_xlabel("k"); a1.set_ylabel("T(k) = √(P_SR/P_HR)")
    a1.set_title("Field-space power transfer (want T→1)"); a1.legend(fontsize=9, loc="lower left")
    for lab, f, c in series[1:]:
        _, rk = _rk(f, x_hr)
        a2.semilogx(k, rk, label=lab, color=c, lw=(2.2 if lab.startswith("E") else 1.5))
    a2.axvline(k_lr, ls=":", c="gray"); a2.set_ylim(0.4, 1.02)
    a2.set_xlabel("k"); a2.set_ylabel("r(k) vs HR")
    a2.set_title("Field-space correlation with HR"); a2.legend(fontsize=9, loc="lower left")
    fig.suptitle("How the innovation helps: it lifts high-k field power T(k) toward 1 that the "
                 "deterministic mean under-produces — at a small cost in r(k)", fontsize=11)
    fig.savefig(out / "4_transfer_rk.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---------- FIG 5: residual hardness (the simple test) ----------
    # Point being tested: the null-space residual is NOT more information than the
    # full field -- it is a lower-variance, high-k, partly-predictable SUBSET.
    p_full = _pkf(x_hr); p_res = _pkf(r_true); p_m = _pkf(m); p_e = _pkf(e_gt)
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for lab, p, c, lw in [("full field x_hr", p_full, "k", 2.2),
                          ("residual P_A(x_hr)", p_res, "C3", 1.6),
                          ("mean m (predictable)", p_m, "C0", 1.5),
                          ("innovation e_gt (irreducible)", p_e, "C2", 2.0)]:
        b1.loglog(k, p.cpu().numpy(), label=lab, color=c, lw=lw)
    b1.axvline(k_lr, ls=":", c="gray"); b1.text(k_lr, b1.get_ylim()[0], " k_LR", fontsize=9)
    b1.set_xlabel("k"); b1.set_ylabel("P(k) field")
    b1.set_title("Power ladder: residual is high-k; innovation = irreducible part")
    b1.legend(fontsize=8)
    _, rk_mp = _rk(m, r_true)
    b2.semilogx(k, rk_mp, color="C0", lw=2.2, label="r(k): deterministic mean vs true residual")
    b2.axvline(k_lr, ls=":", c="gray"); b2.set_ylim(0, 1.02)
    b2.set_xlabel("k"); b2.set_ylabel("predictable fraction r(k)")
    b2.set_title("How much of the residual the mean already explains\n(high = easy, low = irreducibly hard)")
    b2.legend(fontsize=9, loc="lower left")
    fig.suptitle("Residual hardness: lower total power than the full field, concentrated at high k, "
                 "partly predictable by the mean", fontsize=11)
    fig.savefig(out / "5_residual_hardness.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # decisive numbers (real-space power = mean square; exact, not binned)
    vf = float(x_hr.pow(2).mean()); vr = float(r_true.pow(2).mean())
    vm = float(m.pow(2).mean()); ve = float(e_gt.pow(2).mean())
    print("\n[hardness] field power (mean square), crop", idx)
    print(f"   full field x_hr     : {vf:.4f}")
    print(f"   residual P_A(x_hr)  : {vr:.4f}   ({100*vr/vf:.1f}% of the full field's power)")
    print(f"     - mean m          : {vm:.4f}   ({100*vm/vr:.1f}% of residual = predictable)")
    print(f"     - innovation e_gt : {ve:.4f}   ({100*ve/vr:.1f}% of residual = irreducible)")

    # ---------- numeric summary ----------
    def d_rms(a, b): return float(((dens(a) - dens(b)) ** 2).mean().sqrt())
    print("\n[viz] density-residual RMS (SR vs HR):")
    print(f"   A+(y) floor : {d_rms(x_base, x_hr):.4f}")
    print(f"   MSE mean    : {d_rms(x_mse, x_hr):.4f}")
    print(f"   Flow C      : {d_rms(x_c, x_hr):.4f}")
    print(f"   E           : {d_rms(x_e[0], x_hr):.4f}")
    print(f"\n[viz] wrote 4 figures to {out}/")


if __name__ == "__main__":
    main()
