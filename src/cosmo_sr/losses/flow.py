"""Losses for the scale-shared stochastic null-space residual flow.

* :func:`flow_matching_loss` -- conditional flow matching (rectified/linear
  interpolant) on the null-space residual, for paired adjacent octaves.
* :func:`degrade_consistency_loss` -- ``mse(A_R(x_hat), y_R)`` (a near-zero
  monitor + safety term, since the parametrisation is consistent by construction).
* :func:`band_power` / :func:`band_statistics_loss` -- differentiable per-octave
  residual power statistics, used to make LR-only data inform the generator.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..operators.multiscale import MultiScaleOperators
from ..operators.base_upscaler import BaseUpscaler, consistent_base


# --------------------------------------------------------------------------- #
# Flow matching (paired)
# --------------------------------------------------------------------------- #
def sample_flow_pair(
    r_star: torch.Tensor, t: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear-interpolant flow-matching sample.

    ``r_t = (1 - t) z + t r_star``, target velocity ``v_star = r_star - z``.
    Returns ``(r_t, v_star, t, z)``. ``t`` is drawn per-sample if not given.
    """
    b = r_star.shape[0]
    z = torch.randn_like(r_star)
    if t is None:
        t = torch.rand(b, device=r_star.device, dtype=r_star.dtype)
    t_b = t.view(b, *([1] * (r_star.dim() - 1)))
    r_t = (1.0 - t_b) * z + t_b * r_star
    v_star = r_star - z
    return r_t, v_star, t, z


def flow_matching_loss(
    model: torch.nn.Module,
    ops: MultiScaleOperators,
    base_upscaler: BaseUpscaler,
    x_R: torch.Tensor,
    x_2R: torch.Tensor,
    R: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flow-matching loss for one adjacent octave pair ``(x_R, x_2R)``.

    Returns ``(loss, r_star)`` where ``r_star = P_null_R(x_2R - B_cons_R(x_R))``
    is the target residual (useful for band-statistics targets).
    """
    base = consistent_base(base_upscaler, ops, x_R)
    r_star = ops.P_null(x_2R - base)
    r_t, v_star, t, _z = sample_flow_pair(r_star)
    v = model(r_t, t, x_R, R)
    return F.mse_loss(v, v_star), r_star.detach()


# --------------------------------------------------------------------------- #
# Degrade consistency (LR-only monitor / safety)
# --------------------------------------------------------------------------- #
def degrade_consistency_loss(
    ops: MultiScaleOperators, x_hat_2R: torch.Tensor, y_R: torch.Tensor
) -> torch.Tensor:
    """``mse(A_R(x_hat), y_R)``."""
    return F.mse_loss(ops.A(x_hat_2R), y_R)


# --------------------------------------------------------------------------- #
# Band-power statistics (differentiable, torch)
# --------------------------------------------------------------------------- #
_KBIN_CACHE: Dict[Tuple[int, int, str], torch.Tensor] = {}


def _kbin_index(n: int, n_bands: int, device) -> torch.Tensor:
    """Per-mode band index for an ``rfftn`` grid of size ``n`` (log-spaced bands)."""
    key = (n, n_bands, str(device))
    cached = _KBIN_CACHE.get(key)
    if cached is not None:
        return cached
    kx = torch.fft.fftfreq(n, device=device) * n
    kz = torch.fft.rfftfreq(n, device=device) * n
    KX, KY, KZ = torch.meshgrid(kx, kx, kz, indexing="ij")
    kmag = torch.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    kmax = float(n // 2)
    # log-spaced edges over [1, kmax]; DC folded into first band
    edges = torch.logspace(0, torch.log10(torch.tensor(kmax)), n_bands + 1, device=device)
    idx = torch.bucketize(kmag.clamp_min(1.0), edges[1:-1].contiguous())
    idx = idx.clamp(0, n_bands - 1)
    _KBIN_CACHE[key] = idx
    return idx


def band_power(field: torch.Tensor, n_bands: int = 8, log: bool = True) -> torch.Tensor:
    """Mean power per radial band, averaged over batch and channels.

    Returns a ``(n_bands,)`` tensor. Differentiable (uses ``torch.fft``).
    """
    if field.dim() != 5:
        raise ValueError(f"band_power expects (B, C, N, N, N), got {tuple(field.shape)}")
    n = field.shape[-1]
    # FFT in fp32 regardless of an active autocast context: cuFFT half-precision
    # support is limited/unstable, and this op is cheap relative to the conv stack.
    field = field.float()
    fft = torch.fft.rfftn(field, dim=(-3, -2, -1))
    power = (fft.real ** 2 + fft.imag ** 2) / (n ** 3)  # (B, C, N, N, N//2+1)
    idx = _kbin_index(n, n_bands, field.device).reshape(-1)
    power_flat = power.reshape(power.shape[0], power.shape[1], -1)
    sums = torch.zeros(n_bands, device=field.device, dtype=power.dtype)
    counts = torch.zeros(n_bands, device=field.device, dtype=power.dtype)
    pmean = power_flat.mean(dim=(0, 1))  # average over batch & channel per mode
    sums.index_add_(0, idx, pmean)
    counts.index_add_(0, idx, torch.ones_like(pmean))
    band = sums / counts.clamp_min(1.0)
    if log:
        band = torch.log(band.clamp_min(1e-12))
    return band


def band_statistics_loss(
    r_gen: torch.Tensor, target_band: torch.Tensor, n_bands: int = 8
) -> torch.Tensor:
    """MSE between generated residual band power and a (detached) target."""
    gen_band = band_power(r_gen, n_bands=n_bands, log=True)
    return F.mse_loss(gen_band, target_band.detach())
