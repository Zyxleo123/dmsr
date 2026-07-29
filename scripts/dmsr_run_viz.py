"""Per-run visualisation: THIS run's model vs the A+(y) floor and the true HR.

Generalises ``dmsr_module_viz.py`` (which compares three *fixed* modules) to a
single run directory. It loads the run's own ``config.yaml`` + ``ckpt_best.pt``,
evaluates one held-out crop, and writes figures **directly under the run dir**,
at two resolutions:

  <name>.png       64^3  HR  (crop_lr=8)
  <name>_128.png   128^3 HR  (crop_lr=16)

Always written (any run type -- critic / nullspace / unconstrained / det):
  1_sr_density        LR input | A+(y) floor | this SR | true HR   (Eulerian density)
  2_residue_density   A+(y) - HR | this SR - HR                    (density residue)
  4_transfer_rk       field-space T(k) and r(k): A+(y), this SR vs HR

Written only when the checkpoint carries a mean+innovation decomposition
(``mean_innovation.enabled`` -- the "MSE/inno" plots):
  3_e_decomposition   A+(y) | + mean m | + innovation | true residual P_A(x_hr)
  5_residual_hardness power ladder + the fraction of the residual the mean explains

Auto-invoked at the end of ``train_dmsr.py``; also runnable standalone:

  python scripts/dmsr_run_viz.py --run runs/dmsr/t13_critic_s0
  python scripts/dmsr_run_viz.py --run runs/dmsr/t13_critic_s0 --res 64   # single res

Runs on CPU by default (set CUDA_VISIBLE_DEVICES for a GPU); one small crop.
"""
from __future__ import annotations
import argparse
import importlib.util
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
_spec = importlib.util.spec_from_file_location("dev", "scripts/dmsr_eval.py")
_dev = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_dev)
load_flow = _dev.load_flow


# --------------------------------------------------------------------------- #
# small helpers (shared with dmsr_module_viz)
# --------------------------------------------------------------------------- #
def _slice(field: torch.Tensor, ch: int = 0) -> np.ndarray:
    n = field.shape[-1]
    return field[0, ch, :, :, n // 2].detach().cpu().numpy()


def _dens_slice(dens: torch.Tensor) -> np.ndarray:
    n = dens.shape[-1]
    return dens[0, 0, :, :, n // 2].detach().cpu().numpy()


def _rk(pred: torch.Tensor, true: torch.Tensor, n_bins=24):
    p_pp, p_tt, p_pt, k = auto_cross_power(pred, true, n_bins)
    rk = (p_pt / (p_pp * p_tt).clamp_min(1e-30).sqrt()).cpu().numpy()
    return k.cpu().numpy(), rk


def _pkf(f):
    return auto_cross_power(f, f, 24)[0]


# --------------------------------------------------------------------------- #
# reusable figure panels (shared by the per-run render and the map2map baseline)
# --------------------------------------------------------------------------- #
def fig_density_chain(out, label, y, x_base, x_sr, x_hr, dens, dens_lr, tag, suffix):
    """FIG 1: raw LR | A+(y) floor | SR | true HR, in Eulerian density."""
    d_hr = dens(x_hr); d_base = dens(x_base); d_sr = dens(x_sr); d_lr = dens_lr(y)
    cols = [("A+(y) = LR upsampled", d_base), (f"{label} (SR)", d_sr), ("true HR", d_hr)]
    vmin, vmax = np.percentile(_dens_slice(d_hr), [1, 99])
    fig, ax = plt.subplots(1, 4, figsize=(16.5, 4.1))
    lv0, lv1 = np.percentile(_dens_slice(d_lr), [1, 99])
    im0 = ax[0].imshow(_dens_slice(d_lr), vmin=lv0, vmax=lv1, cmap="magma",
                       origin="lower", interpolation="nearest")
    ax[0].set_title(f"LR input ({d_lr.shape[-1]}³, native)", fontsize=12)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.02)
    for a, (t, d) in zip(ax[1:], cols):
        im = a.imshow(_dens_slice(d), vmin=vmin, vmax=vmax, cmap="magma", origin="lower")
        a.set_title(t, fontsize=12); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=list(ax[1:]), fraction=0.016, pad=0.01, label="overdensity δ")
    fig.suptitle(f"[{label}] Density: raw LR → SR → true HR  ({tag}, z-slice)", fontsize=13)
    fig.savefig(out / f"1_sr_density{suffix}.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_density_residue(out, label, x_base, x_sr, x_hr, dens, tag, suffix):
    """FIG 2: density residue A+(y)-HR | SR-HR."""
    d_hr = dens(x_hr); d_base = dens(x_base); d_sr = dens(x_sr)
    res = [("A+(y) − HR", _dens_slice(d_base) - _dens_slice(d_hr)),
           (f"{label} − HR", _dens_slice(d_sr) - _dens_slice(d_hr))]
    rlim = max(np.abs(r[1]).max() for r in res)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for a, (t, r) in zip(ax, res):
        im = a.imshow(r, vmin=-rlim, vmax=rlim, cmap="RdBu_r", origin="lower")
        rms = float(np.sqrt((r ** 2).mean()))
        a.set_title(f"{t}\nRMS={rms:.3f}", fontsize=12); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.024, pad=0.01, label="density residual (SR − HR)")
    fig.suptitle(f"[{label}] Residue = SR − true HR (whiter = better; {tag})", fontsize=13)
    fig.savefig(out / f"2_residue_density{suffix}.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_transfer_rk(out, label, x_base, x_sr, x_hr, factor, suffix):
    """FIG 4: field-space power transfer T(k) and correlation r(k) vs HR."""
    p_hr = _pkf(x_hr); k = auto_cross_power(x_hr, x_hr, 24)[3].cpu().numpy()
    k_lr = x_hr.shape[-1] / (2.0 * factor)
    series = [("A+(y) floor", x_base, "0.6"), (label, x_sr, "C1")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for lab, f, c in series:
        Tk = (_pkf(f) / p_hr.clamp_min(1e-30)).clamp_min(0).sqrt().cpu().numpy()
        a1.semilogx(k, Tk, label=lab, color=c, lw=(2.2 if c == "C1" else 1.5))
    a1.axhline(1.0, ls="--", c="k", lw=0.8); a1.axvline(k_lr, ls=":", c="gray")
    a1.text(k_lr, 0.05, " k_LR", fontsize=9); a1.set_ylim(0, 1.35)
    a1.set_xlabel("k"); a1.set_ylabel("T(k) = √(P_SR/P_HR)")
    a1.set_title("Field-space power transfer (want T→1)"); a1.legend(fontsize=9, loc="lower left")
    for lab, f, c in series:
        _, rk = _rk(f, x_hr)
        a2.semilogx(k, rk, label=lab, color=c, lw=(2.2 if c == "C1" else 1.5))
    a2.axvline(k_lr, ls=":", c="gray"); a2.set_ylim(0.4, 1.02)
    a2.set_xlabel("k"); a2.set_ylabel("r(k) vs HR")
    a2.set_title("Field-space correlation with HR"); a2.legend(fontsize=9, loc="lower left")
    fig.suptitle(f"[{label}] Field-space power transfer T(k) and correlation r(k) vs HR", fontsize=12)
    fig.savefig(out / f"4_transfer_rk{suffix}.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_qpsi(out, label, x_sr, x_hr, dis_norm, tag, suffix):
    """FIG 6: q+Ψ displacement field in physical kpc/h. Rows Ψx/Ψy/Ψz, columns
    SR | GT | SR−GT. SR & GT share a symmetric scale; the difference column has
    its own (q cancels in the difference, so this is exactly Ψ_SR − Ψ_GT)."""
    comps = ["Ψx", "Ψy", "Ψz"]
    sr = [_slice(x_sr, ch) * dis_norm for ch in range(3)]     # -> physical kpc/h
    gt = [_slice(x_hr, ch) * dis_norm for ch in range(3)]
    diff = [s - g for s, g in zip(sr, gt)]
    vsg = max(max(np.abs(a).max() for a in sr), max(np.abs(a).max() for a in gt))
    vd = max(np.abs(d).max() for d in diff) or 1.0
    col_titles = [f"{label} (SR)", "true HR (GT)", "SR − GT"]
    fig, ax = plt.subplots(3, 3, figsize=(12, 11.5))
    im_sg = im_d = None
    for r in range(3):
        for c, (data, lim) in enumerate([(sr[r], vsg), (gt[r], vsg), (diff[r], vd)]):
            im = ax[r, c].imshow(data, vmin=-lim, vmax=lim, cmap="RdBu_r", origin="lower")
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(col_titles[c], fontsize=12)
            if c == 0:
                ax[r, c].set_ylabel(comps[r], fontsize=14)
            if c == 2:
                im_d = im
                rms = float(np.sqrt((diff[r] ** 2).mean()))
                ax[r, c].text(0.03, 0.94, f"RMS={rms:.1f}", transform=ax[r, c].transAxes,
                              fontsize=10, va="top", color="k",
                              bbox=dict(boxstyle="round", fc="white", alpha=0.7))
            else:
                im_sg = im
    fig.colorbar(im_sg, ax=ax[:, :2], fraction=0.020, pad=0.01, label="displacement (kpc/h)")
    fig.colorbar(im_d, ax=ax[:, 2], fraction=0.040, pad=0.02, label="SR − GT (kpc/h)")
    fig.suptitle(f"[{label}] q+Ψ displacement field (kpc/h): SR vs GT vs difference ({tag})",
                 fontsize=13)
    fig.savefig(out / f"6_qpsi_displacement{suffix}.png", dpi=130, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# render one resolution
# --------------------------------------------------------------------------- #
def render(run_dir, label, cfg, split, model, op, dens, dens_lr, factor,
           dis_norm, crop_lr, suffix, args, device, is_mean_innov):
    dcfg = cfg.get("data", {})
    uc = dcfg.get("use_channels")

    print(f"[viz] {label}: crop_lr={crop_lr} -> HR {crop_lr*factor}^3")
    # Memory-map the split box by default (mmap=True), matching how training
    # reads data (`data.mmap`): low host-RAM, so it will not OOM on small
    # allocations. Pass --no-mmap on a warm high-RAM node for one fast
    # sequential load instead of many random page-faults.
    ds = build_val_dataset(split, crop_lr=crop_lr, scale_factor=factor,
                           channels=int(dcfg.get("channels", 6)), use_channels=uc,
                           mmap=args.mmap,
                           max_crops=max(args.scan, args.crop + 1), which=args.split)
    if args.crop >= 0:
        idx = min(args.crop, len(ds) - 1)
    else:
        best, idx = -1.0, 0
        for i in range(min(args.scan, len(ds))):
            s = float(dens(ds[i]["hr"].unsqueeze(0).to(device)).std())
            if s > best:
                best, idx = s, i
        print(f"[viz]   auto-picked crop {idx} (density std {best:.3f})")
    sample = ds[idx]
    y = sample["lr"].unsqueeze(0).to(device)
    x_hr = sample["hr"].unsqueeze(0).to(device)

    z = torch.randn(x_hr.shape, device=device)
    with torch.no_grad():
        x_base = op.A_plus(y)
        x_sr = model.generate(y, n_steps=args.n_steps, z=z)
        r_true = op.P_A(x_hr)                                # true null-space residual (field)
        if is_mean_innov:
            m = model.mean_residual(y).detach()              # P_A(mean)
            innov0 = model.sample_innovation(y, n_steps=args.n_steps, z=z)
            e_gt = op.P_A(r_true - m)                         # irreducible innovation

    tag = f"test crop {idx}"
    fig_density_chain(run_dir, label, y, x_base, x_sr, x_hr, dens, dens_lr, tag, suffix)
    fig_density_residue(run_dir, label, x_base, x_sr, x_hr, dens, tag, suffix)
    fig_transfer_rk(run_dir, label, x_base, x_sr, x_hr, factor, suffix)
    fig_qpsi(run_dir, label, x_sr, x_hr, dis_norm, tag, suffix)

    # numeric summary (real-space power = mean square)
    def d_rms(a, b): return float(((dens(a) - dens(b)) ** 2).mean().sqrt())
    print(f"[viz]   density-residual RMS: A+(y) floor {d_rms(x_base, x_hr):.4f}  "
          f"{label} {d_rms(x_sr, x_hr):.4f}")

    if not is_mean_innov:
        return  # 3/5 need the mean+innovation split

    k = auto_cross_power(x_hr, x_hr, 24)[3].cpu().numpy()
    k_lr = x_hr.shape[-1] / (2.0 * factor)

    # ---------- FIG 3: E decomposition in FIELD space (disp0, additive) ----------
    parts = [("A+(y) base [disp0]", _slice(x_base)),
             ("+ mean m [disp0]", _slice(m)),
             ("+ innovation P_A(u) [disp0]", _slice(innov0)),
             ("true residual P_A(x_hr) [disp0]", _slice(r_true))]
    flim = max(np.abs(p[1]).max() for p in parts[1:])
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for a, (t, s) in zip(ax, parts):
        lim = np.abs(s).max() if t.startswith("A+") else flim
        im = a.imshow(s, vmin=-lim, vmax=lim, cmap="RdBu_r", origin="lower")
        a.set_title(t, fontsize=11); a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    fig.suptitle(f"[{label}] decomposition (field space, additive): "
                 "mean + innovation ≈ true null-space residual", fontsize=13)
    fig.savefig(run_dir / f"3_e_decomposition{suffix}.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---------- FIG 5: residual hardness (mean vs innovation) ----------
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
    b2.semilogx(k, rk_mp, color="C0", lw=2.2, label="r(k): mean vs true residual")
    b2.axvline(k_lr, ls=":", c="gray"); b2.set_ylim(0, 1.02)
    b2.set_xlabel("k"); b2.set_ylabel("predictable fraction r(k)")
    b2.set_title("How much of the residual the mean already explains\n(high = easy, low = irreducibly hard)")
    b2.legend(fontsize=9, loc="lower left")
    fig.suptitle(f"[{label}] Residual hardness: mean (predictable) vs innovation (irreducible)", fontsize=12)
    fig.savefig(run_dir / f"5_residual_hardness{suffix}.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    vf = float(x_hr.pow(2).mean()); vr = float(r_true.pow(2).mean())
    vm = float(m.pow(2).mean()); ve = float(e_gt.pow(2).mean())
    print(f"[viz]   hardness: residual {100*vr/vf:.1f}% of field | "
          f"mean {100*vm/vr:.1f}% predictable | innov {100*ve/vr:.1f}% irreducible")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (has config.yaml + ckpt_best.pt)")
    ap.add_argument("--ckpt", default="ckpt_best.pt", help="checkpoint filename within the run dir")
    ap.add_argument("--res", nargs="*", type=int, default=[64, 128],
                    help="HR resolutions to render (64 -> crop_lr 8, 128 -> crop_lr 16)")
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--crop", type=int, default=-1, help="crop index; -1 = auto-pick most structured")
    ap.add_argument("--scan", type=int, default=40, help="crops to scan when auto-picking")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--no-mmap", action="store_false", dest="mmap",
                    help="load the split box into RAM instead of memory-mapping it "
                         "(faster on warm high-RAM nodes; may OOM on small allocations)")
    ap.set_defaults(mmap=True)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg_path = run_dir / "config.yaml"
    ckpt_path = run_dir / args.ckpt
    if not cfg_path.exists() or not ckpt_path.exists():
        print(f"[viz] skip {run_dir}: missing "
              f"{'config.yaml' if not cfg_path.exists() else args.ckpt}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    cfg = load_config(str(cfg_path))
    dcfg = cfg.get("data", {})
    factor = int(cfg.get("factor", 8))
    uc = dcfg.get("use_channels")
    channels = len(uc) if uc else int(dcfg.get("channels", 6))
    is_mean_innov = bool(cfg.get("mean_innovation", {}).get("enabled"))
    label = run_dir.name

    op = NullSpaceOperator(factor=factor).to(device)
    hr_cs, lr_cs = cellsizes(dcfg, factor)
    _lp = str(cfg.get("critic", {}).get("lowpass", "blockavg"))
    _dn = float(cfg.get("critic", {}).get("dis_norm", dcfg.get("dis_norm", 6000.0)))
    highpass = HighPassDensity(factor=factor, lowpass=_lp, cellsize=hr_cs, dis_norm=_dn)
    highpass_lr = HighPassDensity(factor=factor, lowpass=_lp, cellsize=lr_cs, dis_norm=_dn)
    dens = lambda f: highpass.density(f)
    dens_lr = lambda f: highpass_lr.density(f)

    print(f"[viz] {label}: loading model ({'mean+innovation' if is_mean_innov else 'flow'}) ...")
    model = load_flow(cfg, channels, str(ckpt_path), device)
    split = resolve_split(dcfg)

    res_to_crop = {64: 8, 128: 16}
    for res in args.res:
        crop_lr = res_to_crop.get(res)
        if crop_lr is None:
            print(f"[viz] unknown res {res} (expected 64 or 128); skipping")
            continue
        suffix = "" if res == 64 else f"_{res}"
        try:
            render(run_dir, label, cfg, split, model, op, dens, dens_lr, factor,
                   _dn, crop_lr, suffix, args, device, is_mean_innov)
        except Exception as e:  # never let one resolution kill the other
            print(f"[viz] {label} res {res} failed: {type(e).__name__}: {e}")

    print(f"[viz] {label}: wrote figures to {run_dir}/")


if __name__ == "__main__":
    main()
