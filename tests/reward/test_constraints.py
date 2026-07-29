"""Section 5: field fidelity as a deterministic feasibility filter."""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.reward.constraints import (ConstraintSet, check_feasible, constraint_values,
                                         diversity_value, load_constraints)


SF = 4
NG = 32


@pytest.fixture
def triple():
    """A base field, a matching HR field, and the LR field they agree with."""
    rng = np.random.default_rng(0)
    hr = rng.normal(0, 0.05, size=(6, NG, NG, NG)).astype(np.float32)
    base = hr + rng.normal(0, 0.002, size=hr.shape).astype(np.float32)
    lr = base.reshape(6, NG // SF, SF, NG // SF, SF, NG // SF, SF).mean(axis=(2, 4, 6))
    return base, hr, lr.astype(np.float32)


def cons_all():
    return ConstraintSet(low_k_change_max=0.02, displacement_power_error_max=0.4,
                         density_power_error_max=0.5, lr_consistency_error_max=0.05,
                         diversity_min=0.05)


def test_zero_residual_passes_every_preservation_constraint(triple):
    base, hr, lr = triple
    v = constraint_values(base, base, lr, hr=hr, scale_factor=SF, n_bins=8,
                          compute_density=False, diversity=0.2)
    assert v["low_k_change"] == pytest.approx(0.0, abs=1e-12)
    assert v["lr_consistency_error"] == pytest.approx(0.0, abs=1e-6)
    ok, viol = check_feasible(
        v, ConstraintSet(low_k_change_max=0.02, displacement_power_error_max=None,
                         density_power_error_max=None, lr_consistency_error_max=0.05,
                         diversity_min=0.05))
    assert ok, viol


def test_a_large_scale_offset_fails_the_low_k_constraint(triple):
    base, hr, lr = triple
    hat = base.copy()
    hat[0:3] += 0.02          # a uniform bulk shift: pure low-k
    v = constraint_values(hat, base, lr, hr=hr, scale_factor=SF, n_bins=8,
                          compute_density=False, diversity=0.2)
    assert v["low_k_change"] > 0.02
    ok, viol = check_feasible(v, ConstraintSet(low_k_change_max=0.02,
                                               diversity_min=None))
    assert not ok
    assert any(s.startswith("low_k_change") for s in viol)


def test_small_scale_noise_leaves_low_k_alone(triple):
    # The complement of the previous test: the residual is *allowed* to add
    # small-scale power, and the low-k filter must not punish it for that.
    base, hr, lr = triple
    rng = np.random.default_rng(1)
    hat = base.copy()
    noise = rng.normal(0, 0.01, size=base[0:3].shape).astype(np.float32)
    noise -= noise.reshape(3, NG // SF, SF, NG // SF, SF, NG // SF, SF).mean(
        axis=(2, 4, 6)).repeat(SF, 1).repeat(SF, 2).repeat(SF, 3)
    hat[0:3] += noise
    v = constraint_values(hat, base, lr, hr=hr, scale_factor=SF, n_bins=8,
                          compute_density=False)
    assert v["low_k_change"] < 1e-5


def test_excessive_high_frequency_noise_fails_the_power_constraint(triple):
    base, hr, lr = triple
    rng = np.random.default_rng(2)
    hat = base.copy()
    hat[0:3] += rng.normal(0, 0.5, size=base[0:3].shape).astype(np.float32)
    v = constraint_values(hat, base, lr, hr=hr, scale_factor=SF, n_bins=8,
                          compute_density=False)
    assert v["displacement_power_error"] > 0.4
    ok, _ = check_feasible(v, ConstraintSet(low_k_change_max=None,
                                            displacement_power_error_max=0.4,
                                            diversity_min=None))
    assert not ok


def test_an_lr_inconsistent_field_fails_the_consistency_constraint(triple):
    base, hr, lr = triple
    hat = base * 1.5
    v = constraint_values(hat, base, lr, hr=hr, scale_factor=SF, n_bins=8,
                          compute_density=False)
    assert v["lr_consistency_error"] > v["lr_consistency_error_base"]
    ok, _ = check_feasible(v, ConstraintSet(low_k_change_max=None,
                                            lr_consistency_error_max=0.05,
                                            diversity_min=None))
    assert not ok


def test_classification_is_deterministic(triple):
    base, hr, lr = triple
    hat = base + 0.001
    a = constraint_values(hat, base, lr, hr=hr, scale_factor=SF, n_bins=8,
                          compute_density=False, diversity=0.1)
    b = constraint_values(hat, base, lr, hr=hr, scale_factor=SF, n_bins=8,
                          compute_density=False, diversity=0.1)
    assert a.keys() == b.keys()
    for k in a:
        if np.isnan(a[k]):
            assert np.isnan(b[k]), k
        else:
            assert a[k] == b[k], k
    assert check_feasible(a, cons_all()) == check_feasible(b, cons_all())


def test_a_nan_never_silently_passes():
    vals = {"low_k_change": float("nan"), "displacement_power_error": 0.0,
            "density_power_error": 0.0, "lr_consistency_error": 0.0, "diversity": 1.0}
    ok, viol = check_feasible(vals, cons_all())
    assert not ok
    assert "low_k_change=nan" in viol


def test_a_missing_value_is_treated_as_a_violation():
    ok, viol = check_feasible({"diversity": 1.0}, cons_all())
    assert not ok
    assert len(viol) == 4


def test_a_disabled_threshold_is_reported_but_not_enforced():
    vals = {"low_k_change": 99.0, "diversity": float("nan")}
    ok, viol = check_feasible(vals, ConstraintSet(low_k_change_max=None,
                                                  diversity_min=None))
    assert ok and viol == []


def test_diversity_detects_a_collapsed_sampler():
    a = np.random.default_rng(0).normal(0, 1, size=(3, 8, 8, 8)).astype(np.float32)
    assert diversity_value([a, a.copy(), a.copy()]) < 1e-6
    spread = [a + 0.5 * np.random.default_rng(i).normal(0, 1, a.shape).astype(np.float32)
              for i in range(3)]
    assert diversity_value(spread) > 0.05
    assert np.isnan(diversity_value([a]))


def test_thresholds_load_from_yaml_and_none_disables():
    c = load_constraints({"low_k_change_max": 0.03, "density_power_error_max": None})
    assert c.low_k_change_max == 0.03
    assert c.density_power_error_max is None
    assert c.to_dict()["low_k_change_max"] == 0.03


def test_missing_hr_makes_hr_referenced_constraints_infeasible_not_silent(triple):
    base, hr, lr = triple
    v = constraint_values(base, base, lr, hr=None, scale_factor=SF, n_bins=8,
                          compute_density=False, diversity=0.2)
    assert np.isnan(v["displacement_power_error"])
    ok, viol = check_feasible(v, cons_all())
    assert not ok
    assert "displacement_power_error=nan" in viol
