"""SR2-style higher-order / physical summary statistics.

These are the "stage-2" evaluation metrics from the SR papers. The tractable
ones (equilateral bispectrum, two-point correlation, velocity statistics) are
implemented here from the fields directly. Halo/subhalo abundance requires a
particle halo finder (FoF/Rockstar) run on reconstructed positions and is left
as a documented hook (:func:`halo_abundance`).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .spectra import _to_numpy, power_spectrum, _k_grid


def velocity_statistics(field, vel_slice: slice = slice(3, 6)) -> Dict[str, Any]:
    """Basic velocity-field statistics for channels ``vel_slice`` of ``(C,N,...)``."""
    field = _to_numpy(field).astype(np.float64)
    if field.ndim != 4:
        raise ValueError(f"velocity_statistics expects (C, N, N, N), got {field.shape}")
    v = field[vel_slice]
    speed = np.sqrt(np.sum(v ** 2, axis=0))
    return {
        "vel_mean_per_channel": v.mean(axis=(1, 2, 3)).tolist(),
        "vel_std_per_channel": v.std(axis=(1, 2, 3)).tolist(),
        "speed_mean": float(speed.mean()),
        "speed_rms": float(np.sqrt(np.mean(speed ** 2))),
    }


def two_point_correlation(field, boxsize: Optional[float] = None):
    """Isotropic two-point correlation ``xi(r)`` via FFT of the power spectrum."""
    field = _to_numpy(field).astype(np.float64)
    if field.ndim != 3:
        raise ValueError(f"two_point_correlation expects (N, N, N), got {field.shape}")
    N = field.shape[0]
    delta = field - field.mean()
    fft = np.fft.rfftn(delta)
    power = (np.abs(fft) ** 2) / (N ** 3)
    xi = np.fft.irfftn(power, s=(N, N, N), axes=(0, 1, 2))
    # radial average
    ix = np.fft.fftfreq(N) * N
    RX, RY, RZ = np.meshgrid(ix, ix, ix, indexing="ij")
    rmag = np.sqrt(RX ** 2 + RY ** 2 + RZ ** 2).ravel()
    xi_flat = xi.ravel()
    edges = np.arange(0.5, N // 2 + 1.5, 1.0)
    which = np.digitize(rmag, edges)
    rc, xic = [], []
    for b in range(1, len(edges)):
        m = which == b
        if np.any(m):
            rc.append(rmag[m].mean())
            xic.append(xi_flat[m].mean())
    r = np.asarray(rc)
    if boxsize is not None:
        r = r * (float(boxsize) / N)
    return r, np.asarray(xic)


def equilateral_bispectrum(
    field, n_bins: int = 8, kmin: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Equilateral reduced bispectrum ``B(k, k, k)`` via the shell method.

    For each k-shell we build the band-filtered field ``I_k(x) = IFFT(delta *
    shell)`` and estimate ``B_eq(k) ~ < I_k(x)^3 >`` with the mode-count
    normalisation ``N_tri(k)`` from the shell indicator. This is the standard
    Sefusatti/Scoccimarro shell estimator restricted to equilateral triangles.
    """
    field = _to_numpy(field).astype(np.float64)
    if field.ndim != 3:
        raise ValueError(f"equilateral_bispectrum expects (N, N, N), got {field.shape}")
    N = field.shape[0]
    delta = field - field.mean()
    dk = np.fft.rfftn(delta)
    kmag = _k_grid(N)  # (N, N, N//2+1)

    kmax = float(N // 2)
    edges = np.linspace(kmin, kmax, n_bins + 1)
    ks, bks = [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        shell = ((kmag >= lo) & (kmag < hi)).astype(np.float64)
        if shell.sum() == 0:
            continue
        Ik = np.fft.irfftn(dk * shell, s=(N, N, N), axes=(0, 1, 2))
        Nk = np.fft.irfftn(shell, s=(N, N, N), axes=(0, 1, 2))
        num = np.mean(Ik ** 3)
        den = np.mean(Nk ** 3)
        ks.append(0.5 * (lo + hi))
        bks.append(num / den if den != 0 else np.nan)
    return np.asarray(ks), np.asarray(bks)


def halo_abundance(*_args, **_kwargs):  # pragma: no cover - documented hook
    """Halo/subhalo abundance (mass function).

    Not implemented in-repo: this requires reconstructing particle positions from
    the displacement field and running an external halo finder (FoF / Rockstar).
    Wire your halo catalogs here to compare abundance vs the SR2 baseline.
    """
    raise NotImplementedError(
        "halo_abundance requires an external halo finder on reconstructed particles; "
        "see README 'SR2 metrics' for the intended pipeline."
    )
