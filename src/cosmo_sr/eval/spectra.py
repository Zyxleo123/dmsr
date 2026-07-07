"""Isotropic power spectra and cross-correlation for cubic scalar fields."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _to_numpy(x) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _check_scalar_cube(field: np.ndarray) -> int:
    field = np.asarray(field)
    if field.ndim != 3 or not (field.shape[0] == field.shape[1] == field.shape[2]):
        raise ValueError(
            f"power spectrum expects a cubic scalar field (N, N, N), got {field.shape}"
        )
    return field.shape[0]


def _k_grid(N: int) -> np.ndarray:
    kx = np.fft.fftfreq(N) * N
    kz = np.fft.rfftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij")
    return np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)


def _bin_power(
    kmag: np.ndarray, power: np.ndarray, N: int, include_dc: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    kmag = kmag.ravel()
    power = power.ravel()
    start = -0.5 if include_dc else 0.5
    kbin_edges = np.arange(start, N // 2 + 1.5, 1.0)
    which = np.digitize(kmag, kbin_edges)
    kcenters = []
    pmeans = []
    for b in range(1, len(kbin_edges)):
        mask = which == b
        if not np.any(mask):
            continue
        kcenters.append(kmag[mask].mean())
        pmeans.append(power[mask].mean())
    return np.asarray(kcenters), np.asarray(pmeans)


def power_spectrum(
    field, boxsize: Optional[float] = None, include_dc: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Isotropic (spherically averaged) power spectrum of a cubic scalar field.

    Returns ``(k, Pk)``. ``k`` is in mode units (cycles per box) unless
    ``boxsize`` is given, in which case ``k`` is scaled by ``2*pi/boxsize``.
    The DC (k=0) mode is excluded by default; for a constant field this yields an
    empty spectrum (all power is at k=0). Set ``include_dc=True`` to keep it.
    """
    field = _to_numpy(field).astype(np.float64)
    N = _check_scalar_cube(field)
    fft = np.fft.rfftn(field)
    power = (np.abs(fft) ** 2) / (N ** 3)
    kmag = _k_grid(N)

    if not include_dc:
        # push the DC mode below the first bin edge so it is excluded
        kmag = np.where(kmag > 0, kmag, -1.0)

    k, Pk = _bin_power(kmag, power, N, include_dc=include_dc)
    if boxsize is not None:
        k = k * (2 * np.pi / float(boxsize))
    return k, Pk


def cross_power_spectrum(
    field_a, field_b, boxsize: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Real part of the cross power spectrum of two cubic scalar fields."""
    a = _to_numpy(field_a).astype(np.float64)
    b = _to_numpy(field_b).astype(np.float64)
    N = _check_scalar_cube(a)
    if b.shape != a.shape:
        raise ValueError("cross_power_spectrum requires matching shapes")
    fa = np.fft.rfftn(a)
    fb = np.fft.rfftn(b)
    cross = np.real(fa * np.conj(fb)) / (N ** 3)
    kmag = _k_grid(N)
    kmag = np.where(kmag > 0, kmag, -1.0)
    k, Pk = _bin_power(kmag, cross, N, include_dc=False)
    if boxsize is not None:
        k = k * (2 * np.pi / float(boxsize))
    return k, Pk


def cross_correlation_coefficient(field_a, field_b) -> Tuple[np.ndarray, np.ndarray]:
    """Scale-dependent cross-correlation coefficient ``r(k) = Pab / sqrt(Pa Pb)``."""
    k, Pab = cross_power_spectrum(field_a, field_b)
    _, Pa = power_spectrum(field_a)
    _, Pb = power_spectrum(field_b)
    denom = np.sqrt(Pa * Pb)
    r = np.where(denom > 0, Pab / denom, 0.0)
    return k, r
