"""Is ``p(HR | SR2)`` broad at subhalo scale? -- the arithmetic behind the test.

``docs/sr2_substructure_module.md`` section 9 step 2 is the pilot's critical
path: the whole "sample, do not regress" premise (skeleton item 2, section 6.1)
assumes the conditional distribution of an HR tile given its SR2 tile is *broad*
in the fine modes. If it turns out nearly determined, a regressor would do and
the flow-matching design is overbuilt. Step 3 -- re-choosing the high-pass on a
**measured** spectrum of ``Psi_HR - Psi_SR2`` rather than on
``docs/host_crop_learnability.md`` section 4's arithmetic -- is the same FFT and
lives here too.

Everything in this module is a pure function of arrays so that
``tests/features/test_cond_spread.py`` can pin it; the I/O, the box loading and
the site selection are in ``scripts/features/measure_conditional_spread.py``.

Two measurements, and what each can and cannot conclude
------------------------------------------------------
**The spectrum** (:func:`radial_cross_spectra`). ``r(k)`` between HR and SR2
displacement, and the power of their difference, in real ``h/Mpc``. Exact, no
model. ``1 - r(k)^2`` is the residual variance fraction left by the best
*isotropic linear* predictor of HR's mode ``k`` from SR2's mode ``k``, so it is
an exact statement about that class and nothing more.

**The local predictor** (:func:`ridge_fit`, :func:`r2_uncentred`). The best
linear -- optionally random-feature nonlinear -- map from SR2's field in a
``(2R+1)^3`` neighbourhood to HR's high-pass displacement at the centre site,
fitted on one box and scored on another. This is a *conditional mean*
estimator, which is exactly the functional section 6.1 says an L2 regressor
converges to.

The asymmetry matters and is the reason both are reported. Residual variance of
any predictor is ``>= Var(HR | SR2)``, so:

* a **small** residual proves the conditional is narrow -- it kills "must
  sample" outright;
* a **large** residual does not prove the conditional is broad. It proves this
  function class cannot find the mean, which is evidence and is what the design
  needs, but it is not a proof and must not be written up as one.

The low-pass control is what makes either reading trustworthy: the identical
pipeline applied to the *smoothed* HR field must recover it, because SR2 is
known to be right at those scales. A pipeline that fails the control is
measuring its own blind spot.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "RidgeSolver",
    "apply_random_features",
    "band_power_fraction",
    "gaussian_lowpass",
    "hann_window",
    "local_scale",
    "neighbourhood_matrix",
    "r2_uncentred",
    "radial_cross_spectra",
    "random_feature_map",
    "ridge_fit",
    "ridge_predict",
    "wavenumbers",
]


# --------------------------------------------------------------------------- #
# Fourier
# --------------------------------------------------------------------------- #
def wavenumbers(n: int, dx: float) -> np.ndarray:
    """``(n, n, n)`` ``|k|`` in ``h/Mpc`` for a cube of ``n`` cells of size ``dx``.

    ``2 pi`` convention, matching ``docs/host_crop_learnability.md`` section 4
    where a scale ``L`` is quoted as ``k = 2 pi / L``. Getting this wrong by
    ``2 pi`` would move the 8 h/Mpc cut by 0.8 decades, which is the whole
    argument of that section.
    """
    f = 2.0 * np.pi * np.fft.fftfreq(int(n), d=float(dx))
    return np.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2
                   + f[None, None, :] ** 2)


def hann_window(n: int) -> np.ndarray:
    """``(n, n, n)`` separable Hann window, for spectra of a NON-periodic cube.

    Required whenever a sub-cube is cut out of the box. An FFT treats its input
    as periodic, so a tile carved from a larger field has an artificial step at
    every face, and that step leaks power across **every** ``k``. The leaked
    component is the tile's coherent bulk flow, which HR and SR2 share almost
    exactly -- so it arrives in both fields nearly identically and inflates
    ``r(k)`` toward 1 at scales where the two are in fact uncorrelated.

    Measured on set8, four 64^3 tiles, above 7.3 h/Mpc:

    ==========  ================  ==================
    tile        ``r`` unwindowed  ``r`` with Hann
    ==========  ================  ==================
    398         0.831             -0.017
    462         0.835             -0.004
    100         0.921              0.010
    7           0.890              0.005
    ==========  ================  ==================

    An unwindowed tile spectrum said SR2's displacement tracks HR's to 83% at
    subhalo scale. It does not track it at all. The whole-box spectrum needs no
    window because the box genuinely is periodic; only sub-cubes do.
    """
    w = np.hanning(int(n) + 2)[1:-1].astype(np.float64)
    return (w[:, None, None] * w[None, :, None] * w[None, None, :])


def radial_cross_spectra(a: np.ndarray, b: np.ndarray, dx: float,
                         *, n_bins: int = 24,
                         window: Optional[np.ndarray] = None,
                         kmag: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Auto, cross and difference power of two vector fields, in radial ``k`` bins.

    ``window`` multiplies both fields before the transform and MUST be supplied
    when ``a`` and ``b`` are sub-cubes of a larger field -- see
    :func:`hann_window` for what happens when it is not. Every statistic
    reported here is a ratio of powers, so the window's normalisation cancels
    and no correction factor is applied.

    ``a`` and ``b`` are ``(C, n, n, n)`` -- the three displacement components of
    HR and of SR2 on the *same* Lagrangian sites, in Mpc/h. Power is summed over
    components, so ``P`` is the power of the vector field and ``r`` is its
    vector cross-correlation.

    Bins are logarithmic from the fundamental to Nyquist; ``k = 0`` is dropped,
    because a patch's mean displacement is bulk motion and carries no
    information about whether anything inside it collapsed.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 4:
        raise ValueError(f"expected matching (C, n, n, n); got {a.shape} {b.shape}")
    n = int(a.shape[-1])
    if kmag is None:
        kmag = wavenumbers(n, dx)
    if window is not None:
        w = np.asarray(window, dtype=np.float32)
        a, b = a * w, b * w

    fa = np.fft.fftn(a, axes=(-3, -2, -1))
    fb = np.fft.fftn(b, axes=(-3, -2, -1))
    # P = |F|^2 * V / N^2 with V = N dx^3, i.e. |F|^2 dx^3 / N.
    norm = float(dx) ** 3 / float(n ** 3)
    pa = (np.abs(fa) ** 2).sum(axis=0) * norm
    pb = (np.abs(fb) ** 2).sum(axis=0) * norm
    pd = (np.abs(fa - fb) ** 2).sum(axis=0) * norm
    px = np.real(fa * np.conj(fb)).sum(axis=0) * norm

    k_f = 2.0 * np.pi / (n * float(dx))
    k_ny = np.pi / float(dx)
    edges = np.logspace(np.log10(k_f * 0.999), np.log10(k_ny * 1.001), int(n_bins) + 1)
    which = np.digitize(kmag.reshape(-1), edges) - 1
    flat = kmag.reshape(-1)
    ok = (which >= 0) & (which < int(n_bins)) & (flat > 0)
    w = which[ok]
    cnt = np.bincount(w, minlength=int(n_bins)).astype(np.float64)
    out = {"k_edges": edges, "counts": cnt,
           "k": np.where(cnt > 0,
                         np.bincount(w, weights=flat[ok], minlength=int(n_bins))
                         / np.maximum(cnt, 1), np.nan)}
    for name, arr in (("P_a", pa), ("P_b", pb), ("P_diff", pd), ("P_cross", px)):
        s = np.bincount(w, weights=arr.reshape(-1)[ok], minlength=int(n_bins))
        out[name] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return out


def band_power_fraction(k: np.ndarray, p: np.ndarray, counts: np.ndarray,
                        k_split: float) -> Tuple[float, float]:
    """``(below, above)`` share of total power either side of ``k_split``.

    Mode-count weighted, because a radial bin's mean power is per mode and the
    shells are wildly unequal in size. This is the number
    ``docs/host_crop_learnability.md`` section 4 needs to stop being *derived*:
    how much of the residual the module must emit actually lives above the cut
    the design would filter it to.
    """
    k = np.asarray(k, dtype=np.float64)
    w = np.asarray(counts, dtype=np.float64) * np.asarray(p, dtype=np.float64)
    good = np.isfinite(w) & np.isfinite(k)
    tot = float(w[good].sum())
    if tot <= 0:
        return float("nan"), float("nan")
    lo = float(w[good & (k < float(k_split))].sum())
    return lo / tot, 1.0 - lo / tot


# --------------------------------------------------------------------------- #
# Real-space band split
# --------------------------------------------------------------------------- #
def gaussian_lowpass(x: np.ndarray, sigma_sites: float) -> np.ndarray:
    """Periodic Gaussian smoothing, per component, of a ``(C, n, n, n)`` field.

    A real-space split rather than a sharp Fourier one, for two reasons: it is
    separable and cheap on a whole 512^3 box, and it is *local*, so the high-pass
    residual at a site is a function of that site's neighbourhood and not of the
    whole box -- which is the property the local predictor below assumes.
    ``mode='wrap'`` is exact here: the Lagrangian lattice is periodic.

    The pass edge is ``k ~ 1 / (sigma * dx)``; at the HR spacing 0.1953 Mpc/h,
    sigma of 1, 2 and 4 sites are 5.1, 2.6 and 1.3 h/Mpc -- which brackets the
    2-7 h/Mpc band every resolvable subhalo sits in.
    """
    from scipy.ndimage import gaussian_filter

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"expected (C, n, n, n), got {x.shape}")
    if float(sigma_sites) <= 0:
        return np.zeros_like(x)
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        gaussian_filter(x[c], sigma=float(sigma_sites), mode="wrap", output=out[c])
    return out


def local_scale(psi: np.ndarray, sigma_sites: float = 3.0,
                floor: float = 1e-3) -> np.ndarray:
    """``s(q)``: smoothed rms ``|Psi|``, the pointwise normalisation of section 4.2.

    ``(n, n, n)`` from a ``(3, n, n, n)`` displacement in Mpc/h. Derived from
    SR2's own field and from no catalog, which is the property section 4.2
    argues for: it is defined everywhere, continuous across host boundaries and
    available at inference by construction.
    """
    from scipy.ndimage import gaussian_filter

    psi = np.asarray(psi, dtype=np.float32)
    sq = (psi ** 2).sum(axis=0)
    m = gaussian_filter(sq, sigma=float(sigma_sites), mode="wrap")
    return np.sqrt(np.maximum(m, 0.0) + float(floor) ** 2).astype(np.float32)


# --------------------------------------------------------------------------- #
# The local predictor
# --------------------------------------------------------------------------- #
def neighbourhood_matrix(field: np.ndarray, sites: np.ndarray, radius: int,
                         *, chunk: int = 20000) -> np.ndarray:
    """``(N, C * (2r+1)^3)`` periodic neighbourhoods of ``field`` around ``sites``.

    ``field`` is ``(C, n, n, n)`` and ``sites`` is ``(N, 3)`` integer lattice
    sites. The wrap is a modulo on the index arrays, so a site on the box face
    is not a special case and no padding copy of the box is made.
    """
    field = np.asarray(field)
    if field.ndim != 4:
        raise ValueError(f"expected (C, n, n, n), got {field.shape}")
    c, n = int(field.shape[0]), int(field.shape[-1])
    r = int(radius)
    w = 2 * r + 1
    s = np.asarray(sites, dtype=np.int64).reshape(-1, 3)
    off = np.arange(-r, r + 1, dtype=np.int64)
    out = np.empty((s.shape[0], c * w ** 3), dtype=np.float32)
    for lo in range(0, s.shape[0], int(chunk)):
        hi = min(lo + int(chunk), s.shape[0])
        b = s[lo:hi]
        ix = (b[:, 0, None] + off[None, :]) % n
        iy = (b[:, 1, None] + off[None, :]) % n
        iz = (b[:, 2, None] + off[None, :]) % n
        blk = field[:, ix[:, :, None, None], iy[:, None, :, None],
                    iz[:, None, None, :]]          # (C, m, w, w, w)
        out[lo:hi] = np.moveaxis(blk, 0, 1).reshape(hi - lo, -1)
    return out


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> Dict[str, np.ndarray]:
    """Closed-form ridge with an intercept, ``(N, F) -> (N, T)``.

    Solved on the normal equations because ``F`` here is a few thousand and
    ``N`` a few hundred thousand: ``X^T X`` is the small object. Features are
    standardised first so that a single ``alpha`` means the same thing for a
    displacement component and for a squared random feature.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{x.shape[0]} rows of x against {y.shape[0]} of y")
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    xs = (x - mu) / sd
    ybar = y.mean(axis=0)
    g = xs.T @ xs
    g.flat[:: g.shape[0] + 1] += float(alpha) * max(x.shape[0], 1)
    coef = np.linalg.solve(g, xs.T @ (y - ybar))
    return {"coef": coef, "mean": mu, "scale": sd, "intercept": ybar}


class RidgeSolver:
    """Ridge with the Gram matrix computed once and reused.

    The measurement fits the same feature matrix against many targets (three
    high-pass bands, their low-pass controls) at several ``alpha``. ``X^T X`` is
    the expensive object and depends on **none** of that, so computing it per
    fit would multiply a four-minute job by fifty for no change in any answer.

    Standardisation is folded in rather than materialising a second copy of
    ``X``: at a few hundred thousand rows and a few thousand features the
    float64 copy is larger than the box the features came from.
    """

    def __init__(self, x: np.ndarray, *, chunk: int = 5000) -> None:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"expected (N, F), got {x.shape}")
        self.x = x
        self.n, self.f = int(x.shape[0]), int(x.shape[1])
        self.chunk = int(chunk)
        self.mean = x.mean(axis=0, dtype=np.float64)
        sd = x.std(axis=0, dtype=np.float64)
        sd[sd < 1e-12] = 1.0
        self.scale = sd
        g = np.zeros((self.f, self.f), dtype=np.float64)
        for lo in range(0, self.n, self.chunk):
            z = self._z(lo, min(lo + self.chunk, self.n))
            g += z.T @ z
        self.gram = g

    def _z(self, lo: int, hi: int) -> np.ndarray:
        return (self.x[lo:hi].astype(np.float64) - self.mean) / self.scale

    def rhs(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(X_std^T (y - ybar), ybar)`` for a target block, in chunks."""
        y = np.asarray(y, dtype=np.float64)
        if y.shape[0] != self.n:
            raise ValueError(f"{y.shape[0]} target rows against {self.n} feature rows")
        ybar = y.mean(axis=0)
        out = np.zeros((self.f, y.shape[1]), dtype=np.float64)
        for lo in range(0, self.n, self.chunk):
            hi = min(lo + self.chunk, self.n)
            out += self._z(lo, hi).T @ (y[lo:hi] - ybar)
        return out, ybar

    def fit(self, y: np.ndarray, alpha: float) -> Dict[str, np.ndarray]:
        rhs, ybar = self.rhs(y)
        g = self.gram.copy()
        g.flat[:: self.f + 1] += float(alpha) * max(self.n, 1)
        return {"coef": np.linalg.solve(g, rhs), "mean": self.mean,
                "scale": self.scale, "intercept": ybar}


def ridge_predict(fit: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x - fit["mean"]) / fit["scale"]) @ fit["coef"] + fit["intercept"]


def r2_uncentred(y: np.ndarray, yhat: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> float:
    """``1 - E||y - yhat||^2 / E||y||^2``, optionally per-site weighted.

    Uncentred on purpose. The target is a high-pass field with mean zero by
    construction, and the reference a run has to beat is "predict nothing",
    not "predict the sample mean" -- an ``R^2`` that flattered a predictor for
    getting a mean right would hide exactly the failure being looked for.

    ``weights`` exists because of a contradiction the first real run produced.
    Targets are divided by the local scale ``s`` before fitting, per section
    4.2. That is right for *fitting*: it equalises the per-site gradient. It is
    wrong for *reporting*, because a pooled sum of ``|y/s|^2`` is dominated by
    the smallest-``s`` sites rather than equalised across them -- and ``s`` is
    the rms of the full displacement, i.e. bulk flow, while the target is the
    fine residual, a ratio section 1.1 says varies by orders of magnitude. The
    first run scored the identity predictor at -0.14 where the measured spectrum
    of the same box puts it at +0.63. Passing ``weights = s^2`` undoes the
    normalisation at scoring time and makes the number directly comparable to
    ``P_diff / P_HR`` from :func:`radial_cross_spectra`. Report both: they
    answer different questions and only their agreement establishes that either
    is measuring the field rather than a subpopulation of it.
    """
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    err = (y - yhat) ** 2
    sq = y ** 2
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
        err, sq = err * w, sq * w
    den = float(sq.sum())
    if den <= 0:
        return float("nan")
    return 1.0 - float(err.sum()) / den


def random_feature_map(n_features: int, n_out: int, *, gamma: float,
                       seed: int = 0) -> Dict[str, np.ndarray]:
    """Random Fourier features for the RBF kernel -- the cheapest nonlinearity.

    Present so that "a linear map cannot predict the fine modes" is not confused
    with "nothing can". If the nonlinear score matches the linear one, the
    statement is about the conditioning rather than about linearity.
    """
    rng = np.random.default_rng(int(seed))
    return {
        "w": rng.normal(scale=np.sqrt(2.0 * float(gamma)),
                        size=(int(n_features), int(n_out))).astype(np.float32),
        "b": rng.uniform(0.0, 2.0 * np.pi, size=int(n_out)).astype(np.float32),
    }


def apply_random_features(rf: Dict[str, np.ndarray], x: np.ndarray,
                          *, chunk: int = 20000) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n_out = int(rf["b"].size)
    out = np.empty((x.shape[0], n_out), dtype=np.float32)
    amp = np.float32(np.sqrt(2.0 / n_out))
    for lo in range(0, x.shape[0], int(chunk)):
        hi = min(lo + int(chunk), x.shape[0])
        out[lo:hi] = amp * np.cos(x[lo:hi] @ rf["w"] + rf["b"])
    return out
