#!/usr/bin/env python
"""Visualize a trained learned-degrader checkpoint as projected halo density.

Raw displacement/velocity channels are smooth-looking and don't visually read
as cosmic structure. Instead, this reconstructs Eulerian particle positions
from the (normalized) displacement channels of each field -- baseline A(HR),
the learned D_phi(HR) prediction, and the true LR target -- and projects them
onto a 2D density image (particles binned by (x, y), summed over z), which is
where halos actually show up as bright clumps. Also writes the pixel-wise
difference image between the D_phi prediction and the true target.

Example (CPU is fine -- this is a single small-model forward pass, no GPU
needed even though training used slurm+GPU)::

    python scripts/visualize_degrader.py \
        --run-dir runs/degrader_real_512to64_w32_d2_resembed_lr1e-4_s20k_allboxes_aug \
        --lr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr/set14.npy \
        --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set14.npy \
        --out runs/degrader_real_512to64_w32_d2_resembed_lr1e-4_s20k_allboxes_aug/eval_viz/set14_halo.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def reconstruct_positions(field: np.ndarray, boxsize: float, redshift: float = 0.0) -> np.ndarray:
    """Undo cosmological disp-norm and add the Lagrangian lattice to get
    Eulerian positions ``(3, Ng, Ng, Ng)`` in the same units as ``boxsize``.

    No periodic wrap is applied: the field may be a crop (not itself a
    periodic box), and typical z=0 displacements are much smaller than a
    reasonably sized crop, so unwrapped positions are a fine approximation
    for a qualitative density image (a small fraction of particles near a
    crop's edge may fall slightly outside it and be dropped by the histogram
    binning below).
    """
    from cosmo_sr.data.preprocess_srs import growth_D

    Ng = field.shape[1]
    disp = field[0:3].astype(np.float64)
    disp_phys = disp * (6000.0 * growth_D(redshift))
    cellsize = boxsize / Ng
    lattice = (np.arange(Ng) + 0.5) * cellsize
    pos = np.empty_like(disp_phys)
    pos[0] = disp_phys[0] + lattice.reshape(-1, 1, 1)
    pos[1] = disp_phys[1] + lattice.reshape(-1, 1)
    pos[2] = disp_phys[2] + lattice
    return pos


def project_density(field: np.ndarray, boxsize: float, bins: int, redshift: float = 0.0) -> np.ndarray:
    """Project a ``(6, Ng, Ng, Ng)`` field to a 2D (x, y) particle-count image."""
    pos = reconstruct_positions(field, boxsize, redshift)
    x, y = pos[0].ravel(), pos[1].ravel()
    hist, _, _ = np.histogram2d(x, y, bins=bins, range=[[0.0, boxsize], [0.0, boxsize]])
    return hist.T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="training run dir (holds config.yaml + checkpoint)")
    ap.add_argument("--checkpoint", default=None, help="default: <run-dir>/ckpt_best.pt")
    ap.add_argument("--config", default=None, help="default: <run-dir>/config.yaml")
    ap.add_argument("--lr", required=True, help="true LR catnorm .npy (ground truth target)")
    ap.add_argument("--hr", required=True, help="HR catnorm .npy fed to the degrader")
    ap.add_argument("--boxsize", type=float, default=100000.0,
                     help="box size in kpc/h (default: SRS-map2map LR-sim paramfile.genic BoxSize)")
    ap.add_argument("--redshift", type=float, default=0.0)
    ap.add_argument("--crop-hr", type=int, default=192,
                     help="HR crop size fed to the degrader (0 = full box, slower)")
    ap.add_argument("--bins", type=int, default=384, help="2D projected-density histogram resolution")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from cosmo_sr.data.field_io import load_field
    from cosmo_sr.data.crops import periodic_crop
    from cosmo_sr.operators.multiscale import MultiScaleOperators
    from cosmo_sr.train.train_degrader import build_degrader
    from cosmo_sr.train import common
    from cosmo_sr.utils.config import load_config

    run_dir = Path(args.run_dir)
    cfg = load_config(args.config or run_dir / "config.yaml")
    checkpoint = args.checkpoint or str(run_dir / "ckpt_best.pt")
    factor = int(cfg.get("factor", 8))
    real_pair_R = float(cfg.get("data", {}).get("lr_grid_res", 64))

    deg = build_degrader(cfg)
    common.load_checkpoint(checkpoint, deg, map_location="cpu")
    deg.eval()
    ops = MultiScaleOperators(factor=factor)

    hr_full = load_field(args.hr, mmap=True)
    lr_full = load_field(args.lr, mmap=True)
    full_hr_Ng = hr_full.shape[1]
    if args.crop_hr > 0:
        crop_hr = args.crop_hr
        crop_lr = crop_hr // factor
        hr = periodic_crop(np.asarray(hr_full), (0, 0, 0), crop_hr, pad=0)
        lr_true = periodic_crop(np.asarray(lr_full), (0, 0, 0), crop_lr, pad=0)
    else:
        hr = np.asarray(hr_full)
        lr_true = np.asarray(lr_full)
    hr = np.ascontiguousarray(hr, dtype=np.float32)
    lr_true = np.ascontiguousarray(lr_true, dtype=np.float32)
    crop_boxsize = args.boxsize * (hr.shape[1] / full_hr_Ng)

    hr_t = torch.from_numpy(hr).unsqueeze(0)
    with torch.no_grad():
        y_pred = deg(hr_t, real_pair_R).squeeze(0).numpy()
        a_base = ops.A(hr_t).squeeze(0).numpy()

    print(f"[visualize_degrader] HR crop {hr.shape}, LR grids {lr_true.shape}, "
          f"boxsize(crop)={crop_boxsize:.1f} kpc/h")

    dens_base = project_density(a_base, crop_boxsize, args.bins, args.redshift)
    dens_pred = project_density(y_pred, crop_boxsize, args.bins, args.redshift)
    dens_true = project_density(lr_true, crop_boxsize, args.bins, args.redshift)
    diff_pred = dens_pred - dens_true
    diff_base = dens_base - dens_true

    mse_pred = float(np.mean(diff_pred ** 2))
    mse_base = float(np.mean(diff_base ** 2))
    print(f"[visualize_degrader] projected-density MSE: D_phi vs true = {mse_pred:.4g}, "
          f"A(HR) vs true = {mse_base:.4g}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_panels = [("Baseline A(HR)", dens_base), ("Learned D_phi(HR)", dens_pred),
                  ("True LR target", dens_true)]
    vmax = max(np.log1p(d).max() for _, d in log_panels)

    diff_panels = [("Diff: A(HR) - true", diff_base, mse_base), ("Diff: D_phi - true", diff_pred, mse_pred)]
    dmax = max(np.abs(d).max() for _, d, _ in diff_panels) or 1.0

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, (title, img) in zip(axes[:3], log_panels):
        im = ax.imshow(np.log1p(img), origin="lower", vmin=0, vmax=vmax, cmap="inferno")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label="log1p(particle count)")

    for ax, (title, img, mse) in zip(axes[3:], diff_panels):
        im = ax.imshow(img, origin="lower", vmin=-dmax, vmax=dmax, cmap="RdBu_r")
        ax.set_title(f"{title} (MSE={mse:.3g})")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label="particle count")

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[visualize_degrader] wrote {out_path}")


if __name__ == "__main__":
    main()
