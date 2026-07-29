"""Part 1 -- what the null space of block-average-and-decimate actually is.

The single point of this module is to make one claim concrete and testable:

    ker(A) for the block-average/decimate operator A is **not** an ideal Fourier
    high-pass band.

An ideal high-pass keeps every mode with ``|k| > k_cut`` and drops the rest, so it
is a projector onto one contiguous frequency interval. ``P_A`` is a different
projector. A field lies in ``ker(A)`` iff it averages to zero over every
non-overlapping block -- i.e. it is a *per-block zero-mean* pattern. Written out
for the synthetic factor-two operator

    A(x)[j] = (x[2j] + x[2j+1]) / 2,

the null space is exactly the set

    r = [a0, -a0, a1, -a1, ...],           A(r)[j] = (a_j - a_j)/2 = 0,

for **any** envelope ``a`` -- constant, slowly varying, or arbitrary. In Fourier
space that is the Nyquist carrier ``(+1,-1,+1,-1,...)`` *modulated* by the envelope
``a``. Multiplication in real space is convolution in Fourier space, so the
spectrum of ``r`` is the spectrum of ``a`` **shifted to sit around Nyquist** and
smeared by the two-cell block window's sinc side-lobes. A slowly varying ``a``
therefore does not give a single Nyquist spike: it gives a *structured combination*
of modes clustered near Nyquist, created by the averaging window and by aliasing.
That is the difference from an ideal high-pass, and the tests in
``tests/dmsr/test_nullspace_spectral.py`` exhibit it numerically.

Everything here is a diagnostic. Nothing is a training loss, nothing back-props,
and the FFTs are used only to *describe* fields, never to move ``A``/``A_plus``/
``P_A`` into Fourier space (which the DMSR stage forbids -- all operators stay in
real space).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


# --------------------------------------------------------------------------- #
# Synthetic 1D factor-s block operator (for the aliasing demonstration)
# --------------------------------------------------------------------------- #
def block_average_1d(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """``A(x)[j] = mean(x[factor*j : factor*j+factor])`` along the last axis."""
    n = x.shape[-1]
    if n % factor != 0:
        raise ValueError(f"length {n} not divisible by factor {factor}")
    return x.reshape(*x.shape[:-1], n // factor, factor).mean(dim=-1)


def block_upsample_1d(y: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """``A_plus``: broadcast each coarse cell over ``factor`` fine cells."""
    return y.repeat_interleave(factor, dim=-1)


def null_projection_1d(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """``P_A(x) = x - A_plus(A(x))`` in 1D -- removes the per-block mean."""
    return x - block_upsample_1d(block_average_1d(x, factor), factor)


def alternating_null_vector(a: torch.Tensor) -> torch.Tensor:
    """Build the canonical factor-two null vector ``[a0,-a0,a1,-a1,...]``.

    ``a`` has shape ``(..., M)`` (the envelope, one value per block); the result
    has shape ``(..., 2M)``. By construction every adjacent pair sums to zero, so
    ``block_average_1d(result, 2) == 0`` for *any* ``a``.
    """
    signs = torch.tensor([1.0, -1.0], dtype=a.dtype, device=a.device)
    # (..., M, 1) * (2,) -> (..., M, 2) -> (..., 2M)
    return (a.unsqueeze(-1) * signs).reshape(*a.shape[:-1], a.shape[-1] * 2)


def block_zero_mean_vector(coeffs: torch.Tensor, factor: int) -> torch.Tensor:
    """General ``ker(A)`` element for factor ``s``: any per-block zero-mean pattern.

    ``coeffs`` has shape ``(..., M, s)``; each length-``s`` block is made zero-mean
    (its block mean subtracted), so the flattened ``(..., M*s)`` field is annihilated
    by the factor-``s`` block average. Generalises :func:`alternating_null_vector`
    (which is the ``s=2`` special case ``coeffs=[a, 0]`` up to a scale).
    """
    if coeffs.shape[-1] != factor:
        raise ValueError(f"coeffs last dim {coeffs.shape[-1]} != factor {factor}")
    centred = coeffs - coeffs.mean(dim=-1, keepdim=True)
    return centred.reshape(*coeffs.shape[:-2], coeffs.shape[-2] * factor)


def rfft_power(x: torch.Tensor) -> torch.Tensor:
    """One-sided power spectrum ``|rfft(x)|^2`` along the last axis (orthonormal)."""
    f = torch.fft.rfft(x.float(), dim=-1, norm="ortho")
    return f.real ** 2 + f.imag ** 2


# --------------------------------------------------------------------------- #
# 3D radial spectrum of a real field (diagnostic only)
# --------------------------------------------------------------------------- #
def _kmag_full(n: int, device) -> torch.Tensor:
    """Integer-mode ``|k|`` over the FULL (non-Hermitian) fftn cube."""
    k = torch.fft.fftfreq(n, device=device) * n
    KX, KY, KZ = torch.meshgrid(k, k, k, indexing="ij")
    return torch.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)


def radial_power_spectrum_3d(
    field: torch.Tensor, n_shells: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(shell_k, power_per_mode, n_modes)`` for a ``(B, C, N, N, N)`` real field.

    Uses the full complex ``fftn`` with ``norm='ortho'`` (Parseval-exact) and
    integer radial shells ``round(|k|)``. Power is averaged over batch and channels
    and over the modes inside each shell (so ``power_per_mode`` is comparable across
    shells regardless of how many modes each contains). Diagnostic only.
    """
    n = field.shape[-1]
    if n_shells <= 0:
        n_shells = n // 2 + 1
    f = torch.fft.fftn(field.float(), dim=(-3, -2, -1), norm="ortho")
    p = (f.real ** 2 + f.imag ** 2).mean(dim=(0, 1))          # (N, N, N)
    kmag = _kmag_full(n, field.device)
    idx = kmag.round().long().clamp(max=n_shells - 1).reshape(-1)
    counts = torch.zeros(n_shells, device=field.device).index_add_(
        0, idx, torch.ones_like(idx, dtype=torch.float32)
    )
    energy = torch.zeros(n_shells, device=field.device).index_add_(0, idx, p.reshape(-1))
    shell_k = torch.arange(n_shells, device=field.device, dtype=torch.float32)
    per_mode = energy / counts.clamp_min(1.0)
    return shell_k, per_mode, counts


@dataclass
class NullSpaceSpectrumReport:
    """Human/machine-readable summary of the ``P_A``-vs-ideal-highpass demonstration."""

    factor: int
    grid: int
    projected_spectrum_fraction_below_khalf: float   # power of P_A(x) below k_Ny/2
    projected_spectrum_fraction_above_khalf: float   # ... and above
    highpass_leaks_into_nullspace: float             # ||P_A(highpass(x)) - highpass(x)||/||.||
    single_mode_is_not_eigen_max_rel: float          # worst |P_A e_q - lambda e_q| residual

    def to_dict(self) -> Dict[str, float]:
        return {
            "factor": self.factor,
            "grid": self.grid,
            "projected_fraction_below_khalf": self.projected_spectrum_fraction_below_khalf,
            "projected_fraction_above_khalf": self.projected_spectrum_fraction_above_khalf,
            "highpass_leaks_into_nullspace": self.highpass_leaks_into_nullspace,
            "single_mode_not_eigen_max_rel": self.single_mode_is_not_eigen_max_rel,
        }
