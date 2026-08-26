"""Rule or address: the arithmetic that decides it.

``docs/sr2_member_gather.md`` section 6 measured that the centre term is worth
8/154 -> 72/154, and the 2026-08-23 pool measured that frozen SR2 sits a median
5.59 search radii from where the term wants each set. Those two facts together
are why the whole objective's learnability turns on one question -- is that
offset a systematic infall deficit (learnable) or isotropic scatter (not) -- and
these tests pin the estimator that answers it.

Two ways the answer could be wrong and neither is visible in the output: a
decomposition that reports anisotropy where there is none, and a rule scored on
the hosts it was fitted on. Both are constructed here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reward"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from centre_offset_decompose import (  # noqa: E402
    anisotropy, apply_rule, fit_rules, score,
)


def _rows(offsets, *, scale=0.15, d_host=1.0, num_p=500, log_mvir=14.4,
          rng=None):
    """Sets on random clustercentric directions carrying given offsets.

    ``offsets`` is a callable ``rhat -> o``, so a test says what the physics is
    and the geometry is generated around it.
    """
    rng = rng or np.random.default_rng(0)
    rows = []
    for _ in range(400):
        u = rng.normal(size=3)
        rhat = u / np.linalg.norm(u)
        o = np.asarray(offsets(rhat), dtype=np.float64)
        rows.append({
            "key": "setX:h1", "halo_id": 1, "num_p": num_p, "n_live": num_p,
            "scale": scale, "log_mvir": log_mvir, "d_host": d_host,
            "o": o.tolist(), "rhat": rhat.tolist(),
            "o_par": float(o @ rhat),
            "o_perp": float(np.linalg.norm(o - (o @ rhat) * rhat)),
            "o_abs": float(np.linalg.norm(o)),
        })
    return rows


# --------------------------------------------------------------------------- #
# The decomposition
# --------------------------------------------------------------------------- #
def test_a_pure_infall_deficit_reads_as_fully_radial():
    """Every set sitting 0.6 Mpc/h too far OUT: the learnable extreme."""
    rows = _rows(lambda rhat: 0.6 * rhat)
    a = anisotropy(rows)
    assert a["radial_variance_fraction"] == pytest.approx(1.0, abs=1e-9)
    assert a["median_o_par_mpc_h"] == pytest.approx(0.6, abs=1e-9)
    assert a["frac_o_par_positive"] == 1.0
    assert a["median_o_abs_radii"] == pytest.approx(4.0)      # 0.6 / 0.15


def test_isotropic_scatter_lands_on_the_one_third_null():
    """The unlearnable extreme, and the null it must reproduce."""
    rng = np.random.default_rng(11)
    rows = _rows(lambda rhat: rng.normal(scale=0.35, size=3), rng=rng)
    a = anisotropy(rows)
    assert a["radial_variance_fraction"] == pytest.approx(1.0 / 3.0, abs=0.05)
    # A random direction is as often inward as outward.
    assert a["frac_o_par_positive"] == pytest.approx(0.5, abs=0.08)


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #
def test_the_radial_rule_closes_a_pure_infall_deficit():
    rows = _rows(lambda rhat: 0.6 * rhat)
    fit = fit_rules(rows)
    assert fit["radial_a"] == pytest.approx(0.6, abs=1e-9)
    s = score(rows, fit)
    assert s["none"]["frac_within_1r"] == 0.0            # 4 radii out
    assert s["radial"]["frac_within_1r"] == 1.0          # fully corrected
    assert s["radial"]["median_radii"] < 1e-9


def test_no_rule_touches_isotropic_scatter():
    """The result that would say the centre term is an address."""
    rng = np.random.default_rng(5)
    rows = _rows(lambda rhat: rng.normal(scale=0.5, size=3), rng=rng)
    fit = fit_rules(rows)
    s = score(rows, fit)
    for rule in ("radial", "regressed"):
        # Fitting a direction to something that has none cannot help by more
        # than sampling noise, and must not be allowed to look like it did.
        assert s[rule]["median_radii"] > 0.85 * s["none"]["median_radii"]


def test_the_regressed_rule_picks_up_a_dependence_the_radial_one_cannot():
    """An infall deficit that grows with clustercentric distance."""
    rng = np.random.default_rng(2)
    rows = []
    for d in np.linspace(0.3, 3.0, 400):
        u = rng.normal(size=3)
        rhat = u / np.linalg.norm(u)
        o = (0.05 + 0.25 * d) * rhat
        rows.append({
            "key": "setX:h1", "halo_id": 1, "num_p": 500, "n_live": 500,
            "scale": 0.15, "log_mvir": 14.4, "d_host": float(d),
            "o": o.tolist(), "rhat": rhat.tolist(), "o_par": float(o @ rhat),
            "o_perp": 0.0, "o_abs": float(np.linalg.norm(o))})
    fit = fit_rules(rows)
    s = score(rows, fit)
    assert s["regressed"]["median_radii"] < 1e-6
    # The single-scalar rule cannot express a gradient in d_host, so the two
    # rules must separate -- otherwise the regression is not being fitted.
    assert s["radial"]["median_radii"] > 0.5


def test_a_rule_fitted_on_one_population_is_scored_honestly_on_another():
    """The split is the whole point: an in-sample fit proves nothing."""
    train = _rows(lambda rhat: 0.6 * rhat)
    hold = _rows(lambda rhat: -0.6 * rhat)     # the opposite sign: infall EXCESS
    fit = fit_rules(train)
    s_tr, s_ho = score(train, fit), score(hold, fit)
    assert s_tr["radial"]["frac_within_1r"] == 1.0
    # Applying the training rule to a population it does not describe must make
    # things worse, and the report must show that rather than refit.
    assert s_ho["radial"]["median_radii"] > s_ho["none"]["median_radii"]


def test_the_residual_is_measured_in_the_gate_s_search_radii():
    """One unit is one `compare_gather_catalog` search radius, or the number
    cannot be read against 72/154."""
    rows = _rows(lambda rhat: 0.30 * rhat, scale=0.15)
    fit = {"radial_a": 0.0, "regressed_beta": [0.0, 0.0, 0.0, 0.0]}
    assert score(rows, fit)["none"]["median_radii"] == pytest.approx(2.0)
    wide = _rows(lambda rhat: 0.30 * rhat, scale=0.60)
    assert score(wide, fit)["none"]["median_radii"] == pytest.approx(0.5)
    assert score(wide, fit)["none"]["frac_within_1r"] == 1.0


def test_the_none_rule_is_the_frozen_field_untouched():
    rows = _rows(lambda rhat: 0.42 * rhat)
    r = apply_rule(rows, "none", fit_rules(rows))
    assert np.allclose(r, [x["o_abs"] for x in rows])


def test_an_unknown_rule_is_refused_rather_than_silently_skipped():
    rows = _rows(lambda rhat: 0.1 * rhat)
    with pytest.raises(ValueError, match="unknown rule"):
        apply_rule(rows, "wishful", fit_rules(rows))


def test_an_empty_split_reports_nothing_rather_than_a_nan():
    assert score([], {"radial_a": 0.0}) == {}
    assert anisotropy([]) == {}


# --------------------------------------------------------------------------- #
# The verdict statistic
#
# It reads `explained_fraction` and NOT `frac_within_1r`, and the reason is a
# calibration trap: a search radius is max(r_vir, 0.15) Mpc/h, so an offset that
# is 80% systematic can still leave most sets outside one radius. The free field
# itself, holding every address, reached only 46.8% of targets. These pin that
# a large real effect is not reported as a null.
# --------------------------------------------------------------------------- #
def test_explained_fraction_is_one_for_a_rule_that_closes_the_offset():
    rows = _rows(lambda rhat: 0.6 * rhat)
    s = score(rows, fit_rules(rows))
    assert s["none"]["explained_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert s["radial"]["explained_fraction"] == pytest.approx(1.0, abs=1e-9)


def test_a_strongly_systematic_offset_is_not_reported_as_a_null():
    """The trap: 80% radial, yet almost nothing lands inside one search radius."""
    rng = np.random.default_rng(9)
    rows = _rows(lambda rhat: 0.55 * rhat + rng.normal(scale=0.18, size=3),
                 scale=0.15, rng=rng)
    s = score(rows, fit_rules(rows))
    assert s["radial"]["frac_within_1r"] < 0.35        # the misleading column
    assert s["radial"]["explained_fraction"] > 0.5     # the one the verdict reads


def test_explained_fraction_is_negative_when_a_rule_makes_things_worse():
    """A rule carried onto a population it does not describe must show as harm."""
    train = _rows(lambda rhat: 0.6 * rhat)
    hold = _rows(lambda rhat: -0.6 * rhat)
    s = score(hold, fit_rules(train))
    assert s["radial"]["explained_fraction"] < 0.0
