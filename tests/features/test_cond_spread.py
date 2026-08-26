"""Pins for the conditional-spread arithmetic.

The measurement this module backs (``docs/sr2_substructure_module.md`` section 9
step 2) is decisive in one direction only -- a *small* residual would kill the
"must sample" premise -- so the estimator has to be trustworthy in that
direction. These tests are mostly about that: a construction whose answer is
known analytically has to come out right, and the two conventions that would
silently move every number (the ``2 pi`` in ``k`` and the periodic wrap) are
pinned against hand-computed values.
"""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.features import (
    apply_random_features, band_power_fraction, gaussian_lowpass, local_scale,
    neighbourhood_matrix, r2_uncentred, radial_cross_spectra,
    random_feature_map, ridge_fit, ridge_predict, wavenumbers,
)


# --------------------------------------------------------------------------- #
# Fourier conventions
# --------------------------------------------------------------------------- #
def test_wavenumber_convention_is_two_pi_over_wavelength():
    n, dx = 16, 0.25            # box side 4.0
    k = wavenumbers(n, dx)
    assert k[0, 0, 0] == 0.0
    # the fundamental along one axis is 2 pi / L, not 1 / L
    assert k[1, 0, 0] == pytest.approx(2.0 * np.pi / (n * dx))
    # Nyquist
    assert k[n // 2, 0, 0] == pytest.approx(np.pi / dx)


def test_a_single_sine_lands_in_the_bin_holding_its_wavenumber():
    n, dx = 32, 0.2
    m = 3                        # three wavelengths across the box
    q = np.arange(n) * dx
    wave = np.sin(2.0 * np.pi * m * q / (n * dx)).astype(np.float32)
    f = np.zeros((3, n, n, n), dtype=np.float32)
    f[0] = wave[:, None, None]

    s = radial_cross_spectra(f, f, dx, n_bins=12)
    k_true = 2.0 * np.pi * m / (n * dx)
    peak = int(np.nanargmax(np.nan_to_num(s["P_a"] * s["counts"])))
    assert s["k_edges"][peak] <= k_true <= s["k_edges"][peak + 1]


def test_identical_fields_correlate_perfectly_and_differ_by_nothing():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 16, 16, 16)).astype(np.float32)
    s = radial_cross_spectra(a, a, 0.2, n_bins=8)
    good = s["counts"] > 0
    r = s["P_cross"][good] / np.sqrt(s["P_a"][good] * s["P_b"][good])
    assert np.allclose(r, 1.0, atol=1e-5)
    assert np.allclose(np.nan_to_num(s["P_diff"][good]), 0.0, atol=1e-10)


def test_independent_fields_add_in_the_difference_and_do_not_correlate():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(3, 24, 24, 24)).astype(np.float32)
    b = rng.normal(size=(3, 24, 24, 24)).astype(np.float32)
    s = radial_cross_spectra(a, b, 0.2, n_bins=6)
    good = s["counts"] > 200
    # P_diff = P_a + P_b - 2 P_cross exactly, mode by mode and hence in the mean
    assert np.allclose(s["P_diff"][good],
                       s["P_a"][good] + s["P_b"][good] - 2 * s["P_cross"][good],
                       rtol=1e-5)
    r = s["P_cross"][good] / np.sqrt(s["P_a"][good] * s["P_b"][good])
    assert np.abs(r).max() < 0.2


def test_band_power_fraction_weights_bins_by_their_mode_count():
    # two bins of equal mean power but very unequal shell size: the big shell
    # must dominate, or a spectrum's "fraction above the cut" is meaningless.
    k = np.array([1.0, 10.0])
    p = np.array([1.0, 1.0])
    counts = np.array([1.0, 99.0])
    lo, hi = band_power_fraction(k, p, counts, k_split=5.0)
    assert lo == pytest.approx(0.01)
    assert hi == pytest.approx(0.99)
    assert lo + hi == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The real-space band split
# --------------------------------------------------------------------------- #
def test_lowpass_preserves_a_constant_and_wraps_at_the_face():
    x = np.full((1, 8, 8, 8), 2.5, dtype=np.float32)
    assert np.allclose(gaussian_lowpass(x, 1.5), 2.5, atol=1e-5)

    # a spike at site 0 must leak to site n-1: with mode='wrap' the lattice is
    # periodic, which is the same assumption every other module here makes.
    y = np.zeros((1, 16, 16, 16), dtype=np.float32)
    y[0, 0, 0, 0] = 1.0
    sm = gaussian_lowpass(y, 1.0)
    assert sm[0, -1, 0, 0] > 0
    assert sm[0, -1, 0, 0] == pytest.approx(sm[0, 1, 0, 0], rel=1e-4)


def test_local_scale_is_the_rms_of_a_uniform_displacement():
    psi = np.zeros((3, 8, 8, 8), dtype=np.float32)
    psi[0] = 3.0
    psi[1] = 4.0
    s = local_scale(psi, sigma_sites=1.0, floor=0.0)
    assert np.allclose(s, 5.0, rtol=1e-4)


# --------------------------------------------------------------------------- #
# Neighbourhoods
# --------------------------------------------------------------------------- #
def test_neighbourhood_centre_is_the_site_itself():
    rng = np.random.default_rng(2)
    f = rng.normal(size=(3, 12, 12, 12)).astype(np.float32)
    sites = np.array([[0, 0, 0], [5, 6, 7], [11, 11, 11]])
    r = 2
    w = 2 * r + 1
    m = neighbourhood_matrix(f, sites, r).reshape(len(sites), 3, w, w, w)
    for j, (a, b, c) in enumerate(sites):
        assert np.allclose(m[j, :, r, r, r], f[:, a, b, c])


def test_neighbourhood_wraps_periodically():
    f = np.arange(3 * 4 ** 3, dtype=np.float32).reshape(3, 4, 4, 4)
    m = neighbourhood_matrix(f, np.array([[0, 0, 0]]), 1).reshape(1, 3, 3, 3, 3)
    # the "-1" plane of a site at index 0 is the plane at index n-1
    assert np.allclose(m[0, :, 0, 1, 1], f[:, 3, 0, 0])
    assert np.allclose(m[0, :, 2, 1, 1], f[:, 1, 0, 0])


def test_neighbourhood_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(3)
    f = rng.normal(size=(2, 10, 10, 10)).astype(np.float32)
    sites = rng.integers(0, 10, size=(37, 3))
    a = neighbourhood_matrix(f, sites, 1, chunk=100)
    b = neighbourhood_matrix(f, sites, 1, chunk=5)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# The predictor, and the direction it can conclude in
# --------------------------------------------------------------------------- #
def test_ridge_recovers_an_exact_linear_map_and_scores_one():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(4000, 20))
    w = rng.normal(size=(20, 3))
    y = x @ w + 0.5
    fit = ridge_fit(x, y, alpha=1e-10)
    assert r2_uncentred(y, ridge_predict(fit, x)) == pytest.approx(1.0, abs=1e-6)


def test_ridge_scores_zero_on_a_target_its_features_know_nothing_about():
    """The failure mode the whole measurement turns on.

    If HR's fine modes were independent of SR2's neighbourhood, this is the
    number that would come back -- and it must come back ~0 on held-out data
    rather than being inflated by the fit, or "the conditional mean is empty"
    could not be distinguished from "the fit overfitted".
    """
    rng = np.random.default_rng(5)
    x_fit = rng.normal(size=(3000, 40))
    y_fit = rng.normal(size=(3000, 3))
    x_test = rng.normal(size=(3000, 40))
    y_test = rng.normal(size=(3000, 3))
    fit = ridge_fit(x_fit, y_fit, alpha=1e-3)
    assert r2_uncentred(y_fit, ridge_predict(fit, x_fit)) > 0.0        # in-sample
    assert abs(r2_uncentred(y_test, ridge_predict(fit, x_test))) < 0.05


def test_random_features_let_a_linear_fit_reach_a_nonlinear_target():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(6000, 4))
    y = (np.sin(2.0 * x[:, :1]) * x[:, 1:2]).astype(np.float64)
    lin = ridge_fit(x, y, alpha=1e-6)
    rf = random_feature_map(4, 400, gamma=0.5, seed=0)
    phi = apply_random_features(rf, x.astype(np.float32))
    nonlin = ridge_fit(phi, y, alpha=1e-6)
    assert r2_uncentred(y, ridge_predict(lin, x)) < 0.2
    assert r2_uncentred(y, ridge_predict(nonlin, phi)) > 0.7


# --------------------------------------------------------------------------- #
# The cached-Gram solver
# --------------------------------------------------------------------------- #
def test_ridge_solver_agrees_with_the_one_shot_fit():
    """Reusing the Gram must not change a single number.

    This is the only guarantee that makes the optimisation safe: the solver
    exists purely so the same feature matrix can be fitted against many targets,
    and a discrepancy here would show up as a physics result rather than as a
    numerical one.
    """
    from cosmo_sr.features import RidgeSolver

    rng = np.random.default_rng(7)
    x = rng.normal(size=(500, 12))
    y = rng.normal(size=(500, 3))
    solver = RidgeSolver(x, chunk=37)          # a chunk that does not divide 500
    for alpha in (1e-6, 1e-2, 1.0):
        a = ridge_predict(ridge_fit(x, y, alpha), x)
        b = ridge_predict(solver.fit(y, alpha), x)
        assert np.allclose(a, b, atol=1e-8)


def test_ridge_solver_rejects_a_target_of_the_wrong_length():
    from cosmo_sr.features import RidgeSolver

    rng = np.random.default_rng(8)
    solver = RidgeSolver(rng.normal(size=(40, 5)))
    with pytest.raises(ValueError):
        solver.fit(rng.normal(size=(39, 2)), 1e-3)


# --------------------------------------------------------------------------- #
# Spectral leakage: the bug that inverted a real result
# --------------------------------------------------------------------------- #
def _two_fields_sharing_a_bulk_flow(n, seed=0):
    """A big shared gradient plus INDEPENDENT fine structure in each field.

    The truth is unambiguous by construction: at high k the two fields know
    nothing about each other, so r(k) there must be ~0.
    """
    from cosmo_sr.features import hann_window  # noqa: F401  (import-time check)

    rng = np.random.default_rng(seed)
    q = np.arange(n, dtype=np.float32)
    bulk = np.zeros((3, n, n, n), dtype=np.float32)
    bulk[0] = q[:, None, None] * 3.0          # a ramp: huge, and shared
    a = bulk + rng.normal(scale=0.05, size=(3, n, n, n)).astype(np.float32)
    b = bulk + rng.normal(scale=0.05, size=(3, n, n, n)).astype(np.float32)
    return a, b


def test_an_unwindowed_subcube_fakes_correlation_at_high_k():
    """The measured failure, reproduced.

    Cutting a cube out of a larger field and FFT-ing it as periodic puts a step
    at every face. The step is the shared bulk flow, so its leakage arrives in
    both fields at once and reports agreement at wavenumbers where there is
    none. On set8 this read r = 0.83 where the true value is ~0.
    """
    from cosmo_sr.features import hann_window

    n = 32
    a, b = _two_fields_sharing_a_bulk_flow(n)
    dx = 0.1953
    k_hi = 0.6 * np.pi / dx

    raw = radial_cross_spectra(a, b, dx, n_bins=12)
    won = radial_cross_spectra(a, b, dx, n_bins=12, window=hann_window(n))

    def r_above(s):
        k, c = s["k"], s["counts"]
        m = np.isfinite(k) & (c > 0) & (k >= k_hi)
        return (np.sum((s["P_cross"] * c)[m])
                / np.sqrt(np.sum((s["P_a"] * c)[m]) * np.sum((s["P_b"] * c)[m])))

    # A Hann window suppresses leakage but does not annihilate it, and this
    # synthetic ramp is far harsher than any real field (bulk-to-fine amplitude
    # ~2000:1, against ~10:1 on set8, where the windowed value came out at
    # 0.00-0.01). So the pin is the suppression, not an absolute floor.
    assert r_above(raw) > 0.5                          # the artefact
    assert abs(r_above(won)) < 0.3 * r_above(raw)      # mostly removed


def test_the_window_cancels_out_of_a_ratio_statistic():
    """Windowing must not bias r where there is no leakage to remove."""
    from cosmo_sr.features import hann_window

    rng = np.random.default_rng(4)
    n = 24
    a = rng.normal(size=(3, n, n, n)).astype(np.float32)
    b = (0.7 * a + 0.3 * rng.normal(size=(3, n, n, n))).astype(np.float32)
    raw = radial_cross_spectra(a, b, 0.2, n_bins=6)
    won = radial_cross_spectra(a, b, 0.2, n_bins=6, window=hann_window(n))
    m = (raw["counts"] > 100) & (won["counts"] > 100)
    r_raw = raw["P_cross"][m] / np.sqrt(raw["P_a"][m] * raw["P_b"][m])
    r_won = won["P_cross"][m] / np.sqrt(won["P_a"][m] * won["P_b"][m])
    assert np.allclose(r_raw, r_won, atol=0.06)


# --------------------------------------------------------------------------- #
# The identity cross-check (script-level), reconciled to the site-space route
# --------------------------------------------------------------------------- #
def _load_measure_module():
    """Import the driver script as a module (it lives outside the package)."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for p in (root / "src", root / "scripts" / "reward"):
        if str(p) not in __import__("sys").path:
            __import__("sys").path.insert(0, str(p))
    path = root / "scripts" / "features" / "measure_conditional_spread.py"
    spec = importlib.util.spec_from_file_location("mcs_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_spectrum_identity_matches_the_real_space_high_pass_by_parseval():
    """The Gaussian-weighted spectrum route must equal the site-space identity.

    This is the exact quantity the gate compares. Before the fix the spectrum
    route used a sharp ``k >= 1/(sigma*dx)`` mask, which scores the identity
    predictor on a different band than the soft-Gaussian high-pass the site
    route applies -- the 0.42 gap. Parseval says the two are equal when the
    spectrum weights by the same transfer function, and this pins it.
    """
    mcs = _load_measure_module()
    rng = np.random.default_rng(3)
    n, dx, sigma = 40, 0.1953125, 0.7
    shared = rng.standard_normal((3, n, n, n))
    hr = shared + 0.8 * rng.standard_normal((3, n, n, n))
    sr2 = shared + 0.8 * rng.standard_normal((3, n, n, n))

    hp_hr = hr - gaussian_lowpass(hr.astype(np.float32), sigma)
    hp_sr = sr2 - gaussian_lowpass(sr2.astype(np.float32), sigma)
    r2_site = 1.0 - ((hp_hr - hp_sr) ** 2).sum() / (hp_hr ** 2).sum()

    s = radial_cross_spectra(hr.astype(np.float32), sr2.astype(np.float32),
                             dx, n_bins=48)
    strata = {"wb": {"k": s["k"], "counts": s["counts"], "P_diff": s["P_diff"],
                     "P_hr": s["P_a"], "n_tiles": 1}}
    r2_spec = mcs.spectrum_identity_r2(strata, sigma_sites=sigma, dx=dx)

    assert abs(r2_site - r2_spec) < 0.02
