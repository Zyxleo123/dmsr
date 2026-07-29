"""Per-candidate metrics for test-time scaling of SR2.

Every statistic here is computed on a **complete periodic box** (the natural unit
for these fields) rather than on isolated crops, which would break periodicity
and bias every Fourier and CIC estimate.

The density statistics are built from the CIC field reconstructed from **all
three displacement components** -- ``q + Psi`` deposited on the Lagrangian
lattice. Channel 0 of the displacement is a *coordinate*, not a density, and
scoring it as one is the single easiest way to get a plausible-looking but
meaningless answer.

Most estimators are reused from :mod:`cosmo_sr.dmsr.evaluate` so that a
candidate scored here is directly comparable to numbers already in the repo. The
one thing reimplemented is CIC deposition, because
:func:`cosmo_sr.eval.density.cic_density` materialises ``(B, 3, N, N, N)`` int64
index tensors -- 3.2 GB at ``N = 512`` before the accumulator -- so it is
replaced by a slab-wise version with bounded memory.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..dmsr.evaluate import (
    auto_cross_power,
    divergence_field,
    equilateral_bispectrum,
    pdf_error,
    power_error,
    rk_tk_summary,
    squeezed_cross_bispectrum,
)

__all__ = [
    "DensityGeometry",
    "MomentAccumulator",
    "boundary_discontinuity",
    "candidate_metrics",
    "cic_density_slabs",
    "lr_reconstruction",
]


class DensityGeometry:
    """Physical constants needed to turn normalized displacements into a density.

    ``cellsize`` is the Lagrangian cell of the grid the field lives on
    (``boxsize / N``) and ``dis_norm`` is kpc/h per normalized displacement unit
    (``6000 * D(z)``). Both defaults match ``configs/dmsr/_base.yaml``: a
    100 Mpc/h box on the 512^3 HR grid at z = 0.
    """

    def __init__(self, boxsize: float = 100000.0, ng: int = 512, dis_norm: float = 6000.0):
        self.boxsize = float(boxsize)
        self.ng = int(ng)
        self.dis_norm = float(dis_norm)

    @property
    def cellsize(self) -> float:
        return self.boxsize / self.ng

    def for_grid(self, ng: int) -> "DensityGeometry":
        return DensityGeometry(self.boxsize, int(ng), self.dis_norm)


# --------------------------------------------------------------------------- #
# CIC density
# --------------------------------------------------------------------------- #
def cic_density_slabs(
    disp: torch.Tensor,
    cellsize: float,
    dis_norm: float,
    slab: int = 32,
) -> torch.Tensor:
    """Overdensity ``delta = rho / rho_bar - 1`` from a ``(1, 3, N, N, N)`` displacement.

    Particles start on the Lagrangian lattice at ``q`` (cell centres), move to
    ``q + Psi`` and are cloud-in-cell deposited back onto the same ``N^3``
    periodic mesh. Identical maths to
    :func:`cosmo_sr.eval.density.cic_density`, evaluated ``slab`` planes at a
    time so peak memory stays ~``slab/N`` of the one-shot version.

    Deliberately **not** wrapped in ``no_grad``: Stage 4 backpropagates a density
    statistic to the injected noise. Gradients flow through the CIC weights (the
    cell indices are piecewise constant, as usual for CIC). Callers that only
    need numbers should run inside their own ``no_grad`` block.

    .. warning::
       On a **crop** rather than a full box this is only meaningful as a relative
       comparison. Particles leaving the crop wrap to the far side instead of
       being replaced by their real neighbours, and this repo has measured the
       damage: a crop-level CIC field correlates at r ~ 0.08 with the truth and
       overstates ``sigma`` by ~2.2x unless a ~64-cell buffer is included.
       Tile-level use here (Stage 4 refinement, Stage 5 tile scoring) compares
       candidates that share the same crop and therefore the same bias; do not
       read absolute densities off a tile.
    """
    if disp.dim() != 5 or disp.shape[0] != 1 or disp.shape[1] != 3:
        raise ValueError(f"cic_density_slabs expects (1, 3, N, N, N), got {tuple(disp.shape)}")
    n = disp.shape[-1]
    dev = disp.device
    dens = torch.zeros(n ** 3, device=dev, dtype=torch.float32)
    lat = torch.arange(n, device=dev, dtype=torch.float32) + 0.5
    scale = float(dis_norm) / float(cellsize)

    for x0 in range(0, n, slab):
        x1 = min(x0 + slab, n)
        qx = lat[x0:x1].view(-1, 1, 1)
        qy = lat.view(1, -1, 1)
        qz = lat.view(1, 1, -1)
        d = disp[0, :, x0:x1].float()
        gx = (d[0] * scale + qx) % n
        gy = (d[1] * scale + qy) % n
        gz = (d[2] * scale + qz) % n

        ix0, iy0, iz0 = torch.floor(gx).long(), torch.floor(gy).long(), torch.floor(gz).long()
        fx, fy, fz = gx - ix0.float(), gy - iy0.float(), gz - iz0.float()
        del gx, gy, gz, d
        for ox in (0, 1):
            wx = fx if ox else (1.0 - fx)
            jx = (ix0 + ox) % n
            for oy in (0, 1):
                wy = fy if oy else (1.0 - fy)
                jy = (iy0 + oy) % n
                for oz in (0, 1):
                    wz = fz if oz else (1.0 - fz)
                    jz = (iz0 + oz) % n
                    idx = ((jx * n + jy) * n + jz).reshape(-1)
                    dens.index_add_(0, idx, (wx * wy * wz).reshape(-1))
                    del idx
    dens = dens.view(1, 1, n, n, n)
    return dens / dens.mean().clamp_min(1e-12) - 1.0


# --------------------------------------------------------------------------- #
# Cheap field diagnostics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def lr_reconstruction(sr: torch.Tensor, lr: torch.Tensor, scale_factor: int) -> Dict[str, float]:
    """``|A(x_SR) - y|`` for the block-average operator ``A``.

    Reported per channel group and normalized. ``A = avg_pool`` is known to be a
    *misspecified* model of this dataset's LR simulations (block-averaging the HR
    field does not reproduce the independently run LR box), so a small value here
    is a consistency check, not a quality guarantee -- see
    ``cosmo_sr/dmsr/density.py`` for the measurement.
    """
    a = F.avg_pool3d(sr, kernel_size=int(scale_factor))
    out: Dict[str, float] = {}
    for name, sl in (("disp", slice(0, 3)), ("vel", slice(3, 6))):
        num = float((a[:, sl] - lr[:, sl]).pow(2).mean())
        den = float(lr[:, sl].pow(2).mean()) or 1.0
        out[f"lr_recon_mse_{name}"] = num
        out[f"lr_recon_rel_{name}"] = num / den
    out["lr_recon_mse"] = float((a - lr).pow(2).mean())
    return out


@torch.no_grad()
def boundary_discontinuity(
    field: torch.Tensor, tile_size: int, channels: Sequence[int] = (0, 1, 2)
) -> float:
    """Seam-plane jump relative to the typical neighbouring-voxel jump.

    For each axis, compare the mean squared first difference across the planes
    where two tiles meet (multiples of ``tile_size`` in HR voxels, including the
    periodic wrap at 0) with the same quantity over *all* planes. ``1.0`` means
    the seams look like ordinary interior structure; ``> 1`` means visible tiling.
    """
    x = field[:, list(channels)].float()
    n = x.shape[-1]
    if tile_size <= 0 or tile_size >= n:
        return float("nan")
    seam = torch.arange(0, n, tile_size, device=x.device)
    ratios = []
    for dim in (-3, -2, -1):
        d = (x - torch.roll(x, 1, dims=dim)).pow(2)
        all_mean = float(d.mean())
        seam_mean = float(d.index_select(dim, seam).mean())
        ratios.append(seam_mean / max(all_mean, 1e-30))
    return float(np.mean(ratios))


class MomentAccumulator:
    """Running mean / mean-square over candidates, for diversity without storage.

    Holding ``K`` full 512^3 realisations to measure their spread costs 3.2 GB
    each; two running moments cost two. The across-candidate variance gives the
    mean pairwise distance exactly, since for i.i.d. samples
    ``E|x_i - x_j|^2 = 2 * Var``.
    """

    def __init__(self):
        self.n = 0
        self._sum: Optional[torch.Tensor] = None
        self._sumsq: Optional[torch.Tensor] = None

    def add(self, x: torch.Tensor) -> None:
        x = x.float()
        if self._sum is None:
            self._sum = torch.zeros_like(x)
            self._sumsq = torch.zeros_like(x)
        self._sum += x
        self._sumsq += x * x
        self.n += 1

    def summary(self, prefix: str = "") -> Dict[str, float]:
        if self.n < 2:
            return {f"{prefix}diversity": float("nan"), f"{prefix}pairwise_rms": float("nan")}
        mean = self._sum / self.n
        var = (self._sumsq / self.n - mean * mean).clamp_min(0.0)
        # unbiased across-candidate variance
        var = var * (self.n / (self.n - 1.0))
        rms = float(self._sumsq.mean() / self.n) ** 0.5
        sd = float(var.mean()) ** 0.5
        return {
            f"{prefix}diversity": sd / max(rms, 1e-30),
            f"{prefix}pairwise_rms": (2.0 ** 0.5) * sd / max(rms, 1e-30),
            f"{prefix}rms": rms,
        }


# --------------------------------------------------------------------------- #
# Full per-candidate metric set
# --------------------------------------------------------------------------- #
@torch.no_grad()
def candidate_metrics(
    sr: torch.Tensor,
    hr: Optional[torch.Tensor],
    lr: Optional[torch.Tensor],
    factor: int = 8,
    geometry: Optional[DensityGeometry] = None,
    n_bins: int = 24,
    tile_size: int = 0,
    rho_sr: Optional[torch.Tensor] = None,
    rho_hr: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Every per-candidate statistic, for one ``(1, 6, N, N, N)`` realisation.

    ``hr`` may be ``None`` (then only test-time-computable quantities are
    returned). ``rho_sr`` / ``rho_hr`` let a caller pass densities it already
    computed, since CIC on a 512^3 box is the dominant cost.

    Keys are grouped by prefix: ``disp_``/``vel_``/``density_`` for spectral
    agreement, ``density_power_error``/``density_pdf_error`` for distributional
    agreement, ``bispectrum_*`` for higher order, ``lr_recon_*`` for operator
    consistency and ``boundary_ratio`` for tiling artefacts.
    """
    if sr.dim() != 5 or sr.shape[0] != 1:
        raise ValueError(f"candidate_metrics expects (1, C, N, N, N), got {tuple(sr.shape)}")
    n = sr.shape[-1]
    geometry = geometry or DensityGeometry(ng=n)
    geo = geometry.for_grid(n)
    out: Dict[str, float] = {}

    disp_sr, vel_sr = sr[:, 0:3], sr[:, 3:6]
    if rho_sr is None:
        rho_sr = cic_density_slabs(disp_sr, geo.cellsize, geo.dis_norm)

    # --- self-statistics (no HR needed) ---------------------------------- #
    out["density_sigma"] = float(rho_sr.std())
    out["density_skew"] = float(((rho_sr - rho_sr.mean()) ** 3).mean() / rho_sr.std().pow(3).clamp_min(1e-30))
    out["vel_rms"] = float(vel_sr.pow(2).mean().sqrt())
    out["disp_rms"] = float(disp_sr.pow(2).mean().sqrt())
    out["vel_div_rms"] = float(divergence_field(vel_sr).pow(2).mean().sqrt())
    if tile_size:
        out["boundary_ratio"] = boundary_discontinuity(sr, tile_size)

    if lr is not None:
        out.update(lr_reconstruction(sr, lr, factor))

    if hr is None:
        return out

    # --- HR-referenced statistics ---------------------------------------- #
    disp_hr, vel_hr = hr[:, 0:3], hr[:, 3:6]
    out.update(rk_tk_summary(disp_sr, disp_hr, factor, n_bins=n_bins, prefix="disp_"))
    out.update(rk_tk_summary(vel_sr, vel_hr, factor, n_bins=n_bins, prefix="vel_"))
    out["disp_mse"] = float((disp_sr - disp_hr).pow(2).mean())
    out["vel_mse"] = float((vel_sr - vel_hr).pow(2).mean())

    if rho_hr is None:
        rho_hr = cic_density_slabs(disp_hr, geo.cellsize, geo.dis_norm)
    out.update(rk_tk_summary(rho_sr, rho_hr, factor, n_bins=n_bins, prefix="density_"))
    out["density_power_error"] = power_error(rho_sr, rho_hr, n_bins)
    out["density_pdf_error"] = pdf_error(rho_sr, rho_hr)
    out["density_sigma_ratio"] = float(rho_sr.std() / rho_hr.std().clamp_min(1e-30))

    out["velocity_power_error"] = power_error(vel_sr, vel_hr, n_bins)
    out["velocity_divergence_pdf_error"] = pdf_error(
        divergence_field(vel_sr), divergence_field(vel_hr)
    )

    k_lr = n / (2.0 * factor)
    width = max(2.0, k_lr / 4.0)
    ks = [0.5 * k_lr, k_lr, 1.5 * k_lr]
    b_sr = equilateral_bispectrum(rho_sr, ks, width)
    b_hr = equilateral_bispectrum(rho_hr, ks, width)
    rel = (b_sr - b_hr).abs() / b_hr.abs().clamp_min(1e-20)
    out["bispectrum_equilateral_error"] = float(rel.mean())
    for i, k in enumerate(ks):
        out[f"bispectrum_equilateral_error_k{i}"] = float(rel[i])
    sq_sr = squeezed_cross_bispectrum(rho_sr, rho_hr, 0.3 * k_lr, 1.5 * k_lr, width)
    sq_hr = squeezed_cross_bispectrum(rho_hr, rho_hr, 0.3 * k_lr, 1.5 * k_lr, width)
    out["bispectrum_squeezed_error"] = float((sq_sr - sq_hr).abs() / sq_hr.abs().clamp_min(1e-20))
    return out


@torch.no_grad()
def power_spectrum_curve(field: torch.Tensor, n_bins: int = 24) -> np.ndarray:
    """Shell-averaged auto power of a ``(1, C, N, N, N)`` field, for ensemble plots."""
    p, _, _, _ = auto_cross_power(field, field, n_bins)
    return p.detach().cpu().numpy()


#: Fixed histogram support for ``log10(1 + delta)``. Fixed, not per-field, so
#: histograms from different candidates and boxes can be averaged and compared.
LOG_DENSITY_RANGE = (-1.5, 2.5)


@torch.no_grad()
def density_profile(
    rho: torch.Tensor, n_bins: int = 24, pdf_bins: int = 50
) -> Dict[str, np.ndarray]:
    """Density power spectrum and ``log10(1 + delta)`` PDF, as plain arrays.

    Stored per candidate so that *ensemble-level* power/PDF/bispectrum shifts --
    the signature of a selector biasing the output distribution rather than
    improving it -- can be checked afterwards without regenerating any box.
    """
    lo, hi = LOG_DENSITY_RANGE
    x = torch.log10((rho + 1.0).clamp_min(1e-6)).reshape(-1)
    hist = torch.histc(x.clamp(lo, hi), bins=pdf_bins, min=lo, max=hi)
    hist = hist / hist.sum().clamp_min(1.0)
    return {
        "density_pk": power_spectrum_curve(rho, n_bins).astype(np.float32),
        "log_density_pdf": hist.detach().cpu().numpy().astype(np.float32),
    }
