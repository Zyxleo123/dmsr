"""Memory-bounded numpy field diagnostics for full 512^3 boxes.

The torch helpers in :mod:`cosmo_sr.eval.density` and :mod:`cosmo_sr.dmsr.evaluate`
are the right tools inside a training step, where crops are small and gradients
matter. The reward pipeline instead scores whole boxes on CPU nodes, one channel
at a time, so these numpy versions exist to keep peak memory near one channel
(``512^3 float32 = 537 MB``) rather than one six-channel box.

Conventions match the rest of the repo exactly: fields are ``(C, N, N, N)``
channel-first catnorm, ``k`` is in mode units (``1 .. N/2``), and the LR Nyquist
is ``N / (2 * scale_factor)``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "band_masks",
    "block_average",
    "cic_density_box",
    "cross_power",
    "density_pdf",
    "equilateral_bispectrum",
    "radial_power",
    "rel_rms",
]


def _kmag(n: int) -> np.ndarray:
    kx = np.fft.fftfreq(n) * n
    kz = np.fft.rfftfreq(n) * n
    return np.sqrt(
        kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2
    ).astype(np.float32)


def _binner(n: int, n_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    km = _kmag(n)
    edges = np.linspace(0.0, n / 2.0, int(n_bins) + 1)
    which = np.clip(np.digitize(km.ravel(), edges) - 1, 0, int(n_bins) - 1)
    counts = np.bincount(which, minlength=int(n_bins)).astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, which, counts


def radial_power(field: np.ndarray, n_bins: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Isotropic auto power of a cubic scalar field, ``k`` in mode units."""
    x = np.asarray(field, dtype=np.float32)
    n = x.shape[-1]
    f = np.fft.rfftn(x, axes=(-3, -2, -1))
    p = ((f.real ** 2 + f.imag ** 2) / float(n ** 3)).ravel()
    del f
    centers, which, counts = _binner(n, n_bins)
    sums = np.bincount(which, weights=p, minlength=int(n_bins))
    return centers, sums / np.maximum(counts, 1.0)


def cross_power(
    a: np.ndarray, b: np.ndarray, n_bins: int = 24
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(k, P_aa, P_bb, P_ab)`` for two cubic scalar fields."""
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
    n = x.shape[-1]
    fx = np.fft.rfftn(x, axes=(-3, -2, -1))
    fy = np.fft.rfftn(y, axes=(-3, -2, -1))
    norm = float(n ** 3)
    paa = ((fx.real ** 2 + fx.imag ** 2) / norm).ravel()
    pbb = ((fy.real ** 2 + fy.imag ** 2) / norm).ravel()
    pab = ((fx.real * fy.real + fx.imag * fy.imag) / norm).ravel()
    del fx, fy
    centers, which, counts = _binner(n, n_bins)
    out = []
    for p in (paa, pbb, pab):
        out.append(np.bincount(which, weights=p, minlength=int(n_bins)) / np.maximum(counts, 1.0))
    return centers, out[0], out[1], out[2]


def band_masks(k: np.ndarray, k_lr_nyquist: float) -> Dict[str, np.ndarray]:
    """Low / transition / high bands relative to the LR Nyquist frequency.

    Same split as ``cosmo_sr.dmsr.evaluate.BandEdges``: everything below half the
    LR Nyquist is "already right in the frozen baseline and must not move",
    everything above twice it is the small-scale regime the residual is for.
    """
    k = np.asarray(k, dtype=np.float64)
    kn = float(k_lr_nyquist)
    return {
        "low": k < 0.5 * kn,
        "transition": (k >= 0.5 * kn) & (k < 2.0 * kn),
        "high": k >= 2.0 * kn,
    }


def block_average(field: np.ndarray, factor: int) -> np.ndarray:
    """``A``: average non-overlapping ``factor^3`` blocks, ``(C, N, ...) -> (C, N/f, ...)``."""
    x = np.asarray(field, dtype=np.float32)
    c, n = x.shape[0], x.shape[1]
    if n % factor:
        raise ValueError(f"block_average: {n} not divisible by {factor}")
    m = n // factor
    return x.reshape(c, m, factor, m, factor, m, factor).mean(axis=(2, 4, 6))


def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    """``||a - b|| / ||b||`` over all elements."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    den = float(np.sqrt(np.mean(y ** 2)))
    return float(np.sqrt(np.mean((x - y) ** 2)) / max(den, 1e-30))


def cic_density_box(
    disp_norm: np.ndarray,
    *,
    boxsize_mpc_h: float = 100.0,
    dis_norm_kpc_h: float = 6000.0,
    redshift: float = 0.0,
    slab: int = 32,
) -> np.ndarray:
    """Full-box CIC overdensity from a ``(3, N, N, N)`` normalized displacement.

    Same maths as :func:`cosmo_sr.eval.density.cic_density` (periodic wrap on the
    complete box, so the wrap is exact rather than an approximation), evaluated
    in Lagrangian slabs so peak memory is ``slab/N`` of the one-shot version.
    """
    from ..data.preprocess_srs import growth_D

    x = np.asarray(disp_norm)
    if x.ndim != 4 or x.shape[0] < 3:
        raise ValueError(f"expected (3, N, N, N), got {x.shape}")
    n = int(x.shape[1])
    cellsize = boxsize_mpc_h * 1000.0 / n
    scale = float(dis_norm_kpc_h) * float(growth_D(redshift)) / cellsize
    dens = np.zeros(n ** 3, dtype=np.float32)
    lat = np.arange(n, dtype=np.float32) + 0.5

    for x0 in range(0, n, int(slab)):
        x1 = min(x0 + int(slab), n)
        d = np.asarray(x[0:3, x0:x1], dtype=np.float32)
        g0 = d[0] * scale + lat[x0:x1].reshape(-1, 1, 1)
        g1 = d[1] * scale + lat.reshape(1, -1, 1)
        g2 = d[2] * scale + lat.reshape(1, 1, -1)
        del d
        gs = [np.mod(g, n) for g in (g0, g1, g2)]
        del g0, g1, g2
        i0 = [np.floor(g).astype(np.int64) for g in gs]
        fr = [(g - i.astype(np.float32)) for g, i in zip(gs, i0)]
        del gs
        for ox in (0, 1):
            for oy in (0, 1):
                for oz in (0, 1):
                    w = (fr[0] if ox else 1.0 - fr[0]) * \
                        (fr[1] if oy else 1.0 - fr[1]) * \
                        (fr[2] if oz else 1.0 - fr[2])
                    ix = (i0[0] + ox) % n
                    iy = (i0[1] + oy) % n
                    iz = (i0[2] + oz) % n
                    idx = ((ix * n + iy) * n + iz).ravel()
                    dens += np.bincount(idx, weights=w.ravel(), minlength=n ** 3).astype(
                        np.float32
                    )
                    del ix, iy, iz, idx, w
        del i0, fr
    mean = float(dens.mean())
    return (dens / max(mean, 1e-30) - 1.0).reshape(n, n, n)


def equilateral_bispectrum(
    delta: np.ndarray, n_bins: int = 8, k_max_frac: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Equilateral bispectrum ``B(k, k, k)`` by the shell-filter estimator.

    For each ``k`` shell, ``I_k(x)`` is the field band-filtered to that shell and
    ``N_k(x)`` the same filter applied to unity; then

        ``B(k,k,k) = sum_x I_k(x)^3 / sum_x N_k(x)^3``,

    which counts exactly the closed triangles with all three sides in the shell.
    Only the equilateral configuration is computed: the full bispectrum is a
    three-index object and this diagnostic exists to detect a residual that adds
    small-scale power without the accompanying phase alignment, which the
    equilateral slice already shows.

    One shell at a time, so peak memory is a few ``N^3`` arrays.
    """
    x = np.asarray(delta, dtype=np.float32)
    n = int(x.shape[-1])
    f = np.fft.rfftn(x, axes=(-3, -2, -1))
    km = _kmag(n)
    edges = np.linspace(0.0, float(k_max_frac) * n, int(n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.full(int(n_bins), np.nan, dtype=np.float64)
    for b in range(int(n_bins)):
        shell = ((km >= edges[b]) & (km < edges[b + 1]))
        if not shell.any():
            continue
        ik = np.fft.irfftn(np.where(shell, f, 0.0), s=(n, n, n), axes=(-3, -2, -1))
        nk = np.fft.irfftn(shell.astype(np.float32), s=(n, n, n), axes=(-3, -2, -1))
        den = float(np.sum(nk.astype(np.float64) ** 3))
        if abs(den) > 0:
            out[b] = float(np.sum(ik.astype(np.float64) ** 3)) / den
        del ik, nk, shell
    return centers, out


def density_pdf(delta: np.ndarray, bins: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised histogram of ``log10(1 + delta)`` (clipped), the SR2 PDF diagnostic."""
    x = np.log10(np.clip(np.asarray(delta, dtype=np.float32) + 1.0, 1e-3, None))
    if bins is None:
        bins = np.linspace(-3.0, 3.0, 61)
    h, edges = np.histogram(x, bins=bins, density=False)
    h = h.astype(np.float64)
    total = h.sum()
    return 0.5 * (edges[:-1] + edges[1:]), h / max(total, 1.0)
