"""The direct line's checkpoint gate: paired, relative, and calibrated or refused."""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.reward.direct_gates import (
    RelativeDensityGate, calibrate_from_frozen_seeds, check_direct_gates,
    load_direct_gate, paired_degradation,
)


def frozen_rows(boxes=("set0", "set1"), seeds=(0, 1, 2), base=0.020, spread=0.002):
    rng = np.random.default_rng(0)
    return [
        {"box": b, "seed": s,
         "density_power_error": base + float(rng.normal(0.0, spread))}
        for b in boxes for s in seeds
    ]


def candidate_rows(frozen, delta, **extra):
    return [{**r, "density_power_error": r["density_power_error"] + delta, **extra}
            for r in frozen]


def test_calibration_needs_at_least_two_seeds_per_box():
    with pytest.raises(ValueError, match=">= 2 seeds per box"):
        calibrate_from_frozen_seeds(frozen_rows(seeds=(0,)))


def test_calibration_proposal_scales_with_the_frozen_spread():
    tight = calibrate_from_frozen_seeds(frozen_rows(spread=0.0005))
    loose = calibrate_from_frozen_seeds(frozen_rows(spread=0.005))
    assert loose["proposal"]["mean_degradation_max"] > \
        tight["proposal"]["mean_degradation_max"]
    assert loose["proposal"]["single_box_degradation_max"] > \
        tight["proposal"]["single_box_degradation_max"]
    assert tight["proposal"]["calibrated"] is True
    assert set(tight["per_box_std"]) == {"set0", "set1"}


def test_calibration_rejects_non_finite_measurements():
    rows = frozen_rows()
    rows[0]["density_power_error"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        calibrate_from_frozen_seeds(rows)


def test_pairing_is_on_box_and_seed():
    frozen = frozen_rows()
    cand = candidate_rows(frozen, 0.001)
    pairs = paired_degradation(cand, frozen)
    assert len(pairs) == len(frozen)
    for row in pairs:
        assert row["degradation"] == pytest.approx(0.001, abs=1e-12)


def test_a_candidate_without_its_own_frozen_baseline_is_an_error():
    frozen = frozen_rows(seeds=(0, 1))
    cand = candidate_rows(frozen_rows(seeds=(0, 1, 7)), 0.0)
    with pytest.raises(KeyError, match="seed7"):
        paired_degradation(cand, frozen)


def _gate(**kw):
    base = dict(mean_degradation_max=0.002, single_box_degradation_max=0.004,
                low_k_change_max=0.14, d_struct_min=0.05,
                density_power_error_max=0.03751, calibrated=True)
    base.update(kw)
    return RelativeDensityGate(**base)


def test_a_checkpoint_that_does_not_degrade_density_passes():
    frozen = frozen_rows()
    cand = candidate_rows(frozen, -0.0005, low_k_change=0.01, d_struct=0.2)
    r = check_direct_gates(cand, frozen, _gate())
    assert r.passed, r.violations
    assert r.values["mean_degradation"] < 0.0


def test_mean_degradation_beyond_the_tolerance_is_rejected():
    frozen = frozen_rows()
    cand = candidate_rows(frozen, 0.003, low_k_change=0.01, d_struct=0.2)
    r = check_direct_gates(cand, frozen, _gate())
    assert not r.passed
    assert any("mean density-power degradation" in v for v in r.violations)


def test_one_ruined_box_is_rejected_even_when_the_mean_is_fine():
    frozen = frozen_rows(boxes=tuple(f"set{i}" for i in range(8)))
    cand = candidate_rows(frozen, 0.0, low_k_change=0.01, d_struct=0.2)
    cand[0]["density_power_error"] += 0.02          # one box wrecked
    r = check_direct_gates(cand, frozen, _gate())
    assert r.values["mean_degradation"] < 0.002     # the average hides it
    assert not r.passed
    assert any("worst single-box" in v for v in r.violations)


def test_low_k_and_diversity_still_block():
    frozen = frozen_rows()
    r = check_direct_gates(
        candidate_rows(frozen, 0.0, low_k_change=0.5, d_struct=0.2), frozen, _gate())
    assert any("low-k" in v for v in r.violations)
    r = check_direct_gates(
        candidate_rows(frozen, 0.0, low_k_change=0.01, d_struct=0.001), frozen, _gate())
    assert any("structural diversity" in v for v in r.violations)


def test_nan_never_passes_an_enabled_bound():
    frozen = frozen_rows()
    cand = candidate_rows(frozen, 0.0, low_k_change=float("nan"), d_struct=0.2)
    r = check_direct_gates(cand, frozen, _gate())
    assert not r.passed
    assert any("nan" in v for v in r.violations)


def test_an_uncalibrated_gate_refuses_everything():
    frozen = frozen_rows()
    cand = candidate_rows(frozen, -0.01, low_k_change=0.0, d_struct=0.5)
    r = check_direct_gates(cand, frozen, _gate(calibrated=False))
    assert not r.passed
    assert any("calibrated is false" in v for v in r.violations)


def test_all_bounds_are_reported_not_short_circuited():
    frozen = frozen_rows()
    cand = candidate_rows(frozen, 0.01, low_k_change=0.9, d_struct=0.0)
    r = check_direct_gates(cand, frozen, _gate())
    # Every breach is listed, so one fix at a time is not needed to see them all.
    assert len(r.violations) >= 3


def test_the_absolute_backstop_is_stricter_than_nothing():
    frozen = [{"box": "set0", "seed": s, "density_power_error": 0.030}
              for s in (0, 1)]
    # Barely any degradation, but the absolute error is already over the bound.
    cand = [{**r, "density_power_error": 0.0376, "low_k_change": 0.01,
             "d_struct": 0.2} for r in frozen]
    r = check_direct_gates(cand, frozen, _gate(mean_degradation_max=0.01,
                                               single_box_degradation_max=0.02))
    assert any("absolute density-power error" in v for v in r.violations)


def test_load_direct_gate_from_config_block():
    g = load_direct_gate({"mean_degradation_max": 0.001, "calibrated": True,
                          "d_struct_min": None})
    assert g.mean_degradation_max == pytest.approx(0.001)
    assert g.d_struct_min is None
    assert g.calibrated is True
    # Anything unmentioned keeps its documented default rather than vanishing.
    assert g.low_k_change_max == pytest.approx(0.139595)


def test_result_serialises():
    frozen = frozen_rows()
    cand = candidate_rows(frozen, 0.0, low_k_change=0.01, d_struct=0.2)
    d = check_direct_gates(cand, frozen, _gate()).to_dict()
    assert set(d) == {"passed", "violations", "values", "per_box", "gate"}
    assert d["gate"]["calibrated"] is True
