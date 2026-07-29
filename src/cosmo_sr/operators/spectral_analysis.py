"""Coverage / singular-spectrum analysis for stacked shifted operators.

Quantifies how much subcell-shift diversity expands the identifiable subspace of
``A`` (Gate 1). Everything is computed from the eigenvalues of
``G = sum_g H_g^T H_g``. Because axis-aligned box-averaging and axis-aligned
shifts are separable, the 3D eigenvalues are outer products of 1D eigenvalues, so
the full spectrum is exact without ever forming a large matrix.

See ``docs/gate1_operator_coverage.md`` for the interpretation and the measured
degradation-noise floors that turn nominal coverage into realizable coverage.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


def boxcar_shift_matrix_1d(n: int, w: int, g: int) -> np.ndarray:
    """Explicit ``(n//w, n)`` matrix of ``H_g`` in 1D: each row averages the
    width-``w`` window starting at ``j*w + g`` (mod ``n``)."""
    if n % w != 0:
        raise ValueError(f"n={n} not divisible by w={w}")
    m = n // w
    h = np.zeros((m, n))
    for j in range(m):
        for k in range(w):
            h[j, (j * w + g + k) % n] = 1.0 / w
    return h


def stacked_eigs_1d(n: int, w: int, shifts: Iterable[int]) -> np.ndarray:
    """Ascending eigenvalues of the 1D ``G = sum_g H_g^T H_g``."""
    g_mat = sum(
        boxcar_shift_matrix_1d(n, w, g).T @ boxcar_shift_matrix_1d(n, w, g)
        for g in shifts
    )
    return np.linalg.eigvalsh(g_mat)


def stacked_rank_1d(n: int, w: int, shifts: Iterable[int], tol: float = 1e-9) -> int:
    """Rank of the 1D stacked operator (``# eigenvalues > tol * max``)."""
    ev = stacked_eigs_1d(n, w, shifts)
    return int((ev > tol * ev.max()).sum())


def coverage_spectrum_3d(
    eigs_per_axis: Sequence[np.ndarray],
) -> np.ndarray:
    """Descending 3D eigenvalues = outer product of three 1D eigenvalue vectors."""
    a, b, c = eigs_per_axis
    lam = (a[:, None, None] * b[None, :, None] * c[None, None, :]).ravel()
    return np.sort(lam)[::-1]


def effective_rank(lam: np.ndarray) -> float:
    """Entropy-based effective rank ``exp(-sum p log p)``, ``p = lam / sum lam``."""
    lam = np.asarray(lam, dtype=np.float64)
    lam = lam[lam > 0]
    p = lam / lam.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def coverage_summary(
    lam: np.ndarray,
    thresholds: Sequence[float] = (1e-1, 1e-2, 1e-3, 1e-4),
    eta_frac: float | None = None,
    ref: float | None = None,
    tol: float = 1e-9,
) -> Dict[str, float]:
    """Summary of a (descending) eigenvalue spectrum.

    ``rank``/``nullity``/``eff_rank`` and the fraction of modes above each
    ``threshold * max`` (per-case max -- the plan's nominal table).

    If ``eta_frac`` (measured degradation-noise fraction) is given,
    ``eta_identifiable`` counts modes above the noise floor via the
    posterior-identifiability criterion ``lambda > eta_frac * ref``. Pass a
    **shared** ``ref`` (the fully-measured single-operator eigenvalue, e.g. the
    fixed-``A`` spectrum's max in the *same raw units*) to compare across
    operator sets -- the count is then monotone in operator diversity (the C2
    upper bound). ``ref`` defaults to this spectrum's own max (per-case).
    """
    lam = np.asarray(lam, dtype=np.float64)
    lmax = lam.max()
    lr = lam / lmax
    out: Dict[str, float] = {
        "dim": int(lam.size),
        "rank": int((lr > tol).sum()),
        "nullity": int((lr <= tol).sum()),
        "eff_rank": effective_rank(lam),
    }
    for t in thresholds:
        out[f"frac_above_{t:.0e}"] = float((lr > t).mean())
    if eta_frac is not None:
        floor = float(eta_frac) * (lmax if ref is None else float(ref))
        out["eta_frac"] = float(eta_frac)
        out["eta_identifiable"] = int((lam > floor).sum())
    return out
