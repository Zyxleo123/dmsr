"""End-to-end pins for the conditional-spread driver, on synthetic boxes.

The real run reads two 3.2 GiB boxes, so what is checked here is that the
driver's own arithmetic does what the docstring claims on fields whose answer is
constructed. Two constructions carry the weight:

* a box where HR **is** a deterministic local function of SR2 -- the score must
  come back high, or the measurement could never detect determinism and its
  "broad" reading would be vacuous;
* a box where HR's fine modes are an independent draw -- the score must come
  back at ~0 on the held-out box while the low-pass control still passes, which
  is the exact signature the pilot is looking for.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load():
    for p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    path = PROJECT_ROOT / "scripts" / "features" / "measure_conditional_spread.py"
    spec = importlib.util.spec_from_file_location("measure_conditional_spread", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()


class _Args:
    identity_gap_max = 0.25
    sigmas = [1.0]
    radius = 4
    scale_sigma = 3.0
    alphas = [1e-6, 1e-3, 1e-1]
    val_frac = 0.25
    nl_dim = 8
    rff_dim = 64
    n_bins = 8
    k_split = 8.0
    cluster_log_mvir = 14.0
    group_log_mvir = 13.0
    n_tiles = 8
    seed = 0
    determined_min = 0.30
    control_min = 0.80


def _smooth_noise(rng, n, sigma):
    from scipy.ndimage import gaussian_filter
    x = rng.normal(size=(3, n, n, n)).astype(np.float32)
    for c in range(3):
        gaussian_filter(x[c], sigma=sigma, mode="wrap", output=x[c])
    return x


def _sites(rng, n, count):
    return rng.integers(0, n, size=(count, 3)).astype(np.int64)


# --------------------------------------------------------------------------- #
# Tile geometry
# --------------------------------------------------------------------------- #
def test_tile_slices_partition_the_box_exactly_once():
    n = (M.NG_HR // M.TILE) ** 3
    seen = np.zeros((M.NG_HR,) * 3, dtype=np.int8)
    for t in range(n):
        seen[M.tile_slice(t)] += 1
    assert seen.min() == 1 and seen.max() == 1


def test_k_cross_interpolates_the_crossing_and_reports_inf_when_never():
    k = np.array([1.0, 2.0, 4.0])
    r = np.array([1.0, 0.75, 0.25])
    assert M._k_cross(k, r, 0.5) == pytest.approx(3.0)
    assert M._k_cross(k, r, 0.01) == float("inf")


# --------------------------------------------------------------------------- #
# The two constructions
# --------------------------------------------------------------------------- #
def _score(hr_of_sr2, *, n=32, count=6000, seed=0):
    """Run the predictor half of the driver on a synthetic pair of boxes."""
    args = _Args()
    out = {}
    ex = {}
    for box, sd in (("fit", seed), ("test", seed + 100)):
        rng = np.random.default_rng(sd)
        sr2 = _smooth_noise(rng, n, 1.5)
        hr = hr_of_sr2(sr2, rng)
        x, tgt, _ = M.build_examples(hr, sr2, _sites(rng, n, count), args)
        ex[box] = (x, tgt)
    key = f"sigma{args.sigmas[0]:g}"
    fitter = M.Fitter(ex["fit"][0], ex["test"][0], args.val_frac)
    for band in ("high", "low"):
        out[band] = fitter.score(ex["fit"][1][key][band],
                                 ex["test"][1][key][band], args.alphas)
    return out


def test_a_deterministic_local_map_is_recovered():
    """HR = a fixed linear function of SR2's neighbourhood, plus a shift."""
    def det(sr2, rng):
        return (0.8 * sr2 + 0.3 * np.roll(sr2, 1, axis=1)).astype(np.float32)

    r = _score(det)
    assert r["high"]["r2_heldout_box"] > 0.8
    assert r["low"]["r2_heldout_box"] > 0.8


def test_an_independent_fine_realisation_scores_zero_but_the_control_passes():
    """HR keeps SR2's smooth part and redraws its fine part independently.

    This is the shape of the answer the design predicts, and both halves are
    load-bearing: the ~0 on the high-pass target is the finding, and the high
    score on the low-pass target is the proof that the ~0 is the physics rather
    than the pipeline failing.

    Two constraints on the construction, both learned by getting them wrong.
    The smooth part is taken at a sigma well *below* the one the driver splits
    at, because a Gaussian pair that close does not separate: an earlier version
    smoothed both at sigma 2 and scored 0.62, with the "independent" target
    still carrying a predictable piece of SR2. And the smoothing has to stay
    inside the predictor's receptive field, or the low-pass control fails for
    want of radius and reports an estimator failure that is not one. Both
    failure modes exist in the real run too, which is why --radius defaults to
    >= 2.5x the largest --sigma.
    """
    from scipy.ndimage import gaussian_filter

    def redraw(sr2, rng):
        lo = np.empty_like(sr2)
        for c in range(3):
            gaussian_filter(sr2[c], sigma=3.0, mode="wrap", output=lo[c])
        fine = rng.normal(size=sr2.shape).astype(np.float32)
        fine *= 0.10 * float(np.std(sr2)) / float(np.std(fine))
        return (lo + fine).astype(np.float32)

    r = _score(redraw)
    assert abs(r["high"]["r2_heldout_box"]) < 0.15
    assert r["low"]["r2_heldout_box"] > 0.8


# --------------------------------------------------------------------------- #
# Spectra
# --------------------------------------------------------------------------- #
def test_spectra_stratify_by_tile_host_mass(monkeypatch):
    monkeypatch.setattr(M, "NG_HR", 32)
    monkeypatch.setattr(M, "TILE", 16)
    rng = np.random.default_rng(0)
    sr2 = _smooth_noise(rng, 32, 1.5)
    hr = sr2 + 0.1 * _smooth_noise(rng, 32, 0.5)
    logm = np.array([14.5] + [12.0] * 7)
    args = _Args()
    args.n_tiles = 8
    out = M.spectra_for_box(hr, sr2, logm, args)
    assert out["cluster"]["n_tiles"] == 1
    assert out["field"]["n_tiles"] == 7
    # HR is SR2 plus a small fine perturbation, so the residual must sit at
    # high k and r must fall with k.
    r = np.array(out["field"]["r"])
    good = np.isfinite(r)
    assert r[good][0] > r[good][-1]
    assert out["field"]["resid_power_above_split"] >= 0.0
    lo = out["field"]["resid_power_below_split"]
    assert lo + out["field"]["resid_power_above_split"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The verdict text is a decision, so it is pinned like one
# --------------------------------------------------------------------------- #
def _pred(high, nl, low, ident=0.5):
    return {"sigma1": {"host_sites": {
        "high": {"r2_heldout_box": high, "r2_identity_unweighted": ident},
        "high_nonlinear": {"r2_heldout_box": nl},
        "low": {"r2_heldout_box": low},
    }}}


def _spec(p_diff_over_hr):
    """One stratum whose spectrum implies identity R2 = 1 - p_diff_over_hr."""
    return {"box": {"s": {
        "n_tiles": 1,
        "k": [1.0, 20.0],
        "counts": [1.0, 1.0],
        "P_hr": [1.0, 1.0],
        "P_diff": [0.0, p_diff_over_hr],
    }}}


def test_verdict_calls_a_failed_control_inconclusive_rather_than_broad():
    args = _Args()
    v = M.verdict(_pred(0.01, 0.02, 0.10), args)
    assert not v["control_passed"]
    assert v["text"].startswith("INCONCLUSIVE")


def test_verdict_reports_determined_when_either_predictor_finds_the_mean():
    args = _Args()
    assert M.verdict(_pred(0.05, 0.55, 0.95), args)["text"].startswith("DETERMINED")
    assert M.verdict(_pred(0.02, 0.03, 0.95), args)["text"].startswith("BROAD")


# --------------------------------------------------------------------------- #
# Running without the second box
# --------------------------------------------------------------------------- #
def test_a_missing_owner_array_is_not_an_error():
    """set9's raw particle dumps were deleted, so it has no owner array.

    Requiring one would gate out the decisive test over an artifact that cannot
    be regenerated without re-running Rockstar. Missing means "unstratified",
    not "stop".
    """
    got = M._owner(Path("/nonexistent-root"), "setX", "hr")
    assert got is None


def test_spectra_pool_into_one_stratum_without_host_masses(monkeypatch):
    monkeypatch.setattr(M, "NG_HR", 32)
    monkeypatch.setattr(M, "TILE", 16)
    rng = np.random.default_rng(0)
    sr2 = _smooth_noise(rng, 32, 1.5)
    hr = sr2 + 0.1 * _smooth_noise(rng, 32, 0.5)
    out = M.spectra_for_box(hr, sr2, None, _Args())
    assert list(out) == ["unstratified"]
    assert out["unstratified"]["n_tiles"] == 8


def test_verdict_names_the_subset_it_read_and_falls_back_when_unstratified():
    args = _Args()
    pooled = {"sigma1": {"all": {
        "high": {"r2_heldout_box": 0.02, "r2_identity_unweighted": 0.5},
        "high_nonlinear": {"r2_heldout_box": 0.03},
        "low": {"r2_heldout_box": 0.95},
    }}}
    v = M.verdict(pooled, args)
    assert v["subset"] == "all"
    assert v["text"].startswith("BROAD")
    assert M.verdict(_pred(0.02, 0.03, 0.95), args)["subset"] == "host_sites"


# --------------------------------------------------------------------------- #
# The cross-check that the first real run failed
# --------------------------------------------------------------------------- #
def test_spectrum_identity_r2_reads_one_minus_pdiff_over_phr():
    got = M.spectrum_identity_r2(_spec(0.37)["box"], k_edge=7.3)
    assert got == pytest.approx(0.63)


def test_a_site_space_score_that_contradicts_the_spectrum_blocks_the_verdict():
    """The first real run scored identity at -0.14 in site space while the

    spectrum of the same box put it at +0.63. Both cannot describe the field.
    The verdict must refuse to report an R2 in that state rather than publish
    the more convenient of the two.
    """
    args = _Args()
    v = M.verdict(_pred(0.001, -0.01, 0.976, ident=-0.14), args, _spec(0.37))
    assert not v["consistent"]
    assert v["text"].startswith("INCONSISTENT")
    assert "r2_high_linear" in v            # the numbers are still recorded


def test_agreeing_routes_let_the_normal_reading_through():
    args = _Args()
    v = M.verdict(_pred(0.001, -0.01, 0.976, ident=0.60), args, _spec(0.37))
    assert v["consistent"]
    assert v["text"].startswith("BROAD")
    assert v["identity_from_spectrum"] == pytest.approx(0.63)
