"""Arm B's phase-space features: do they measure velocity, and only velocity?

The whole point of the arm comparison is that A and B differ in exactly one
thing. Two claims therefore have to hold mechanically rather than by inspection:
arm A must be bit-identical to the incumbent density-only features, and the
velocity block must move when the velocities move and not otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmo_sr.eval.density import cic_deposit_valid_center, cic_density_valid_center
from cosmo_sr.reward.phase_space import (
    ARMS, PhaseSpaceConfig, arm_features, arm_paired_feature_names,
    arm_paired_features, phase_space_feature_names, phase_space_features,
    validate_phase_space_features,
)
from cosmo_sr.reward.soft_structure import (
    SoftStructureConfig, feature_names, paired_features,
)


def _field(n=24, seed=0, scale=0.02):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 6, n, n, n, generator=g) * scale


def test_weighted_deposit_matches_the_density_it_generalises():
    """The mass channel of the generalised deposit IS cic_density_valid_center.

    They share one corner generator; if that ever stops being true, every
    velocity feature silently starts dividing a momentum grid by a mass grid
    that describes different cells.
    """
    disp = _field(20)[:, 0:3]
    values = torch.zeros(2, 1, 20, 20, 20)
    mass, _ = cic_deposit_valid_center(disp, values, 195.3125, 6000.0, region=10)
    ref = cic_density_valid_center(disp, 195.3125, 6000.0, region=10)
    assert torch.allclose(mass * 1.0 - 1.0, ref, atol=1e-6)


def test_weighted_deposit_is_the_mass_weighted_mean():
    """A constant per-particle value must deposit as that constant everywhere."""
    disp = _field(20)[:, 0:3]
    values = torch.full((2, 2, 20, 20, 20), 3.5)
    mass, acc = cic_deposit_valid_center(disp, values, 195.3125, 6000.0, region=10)
    occupied = mass > 1e-3
    ratio = (acc[:, 0:1] / mass.clamp_min(1e-12))[occupied]
    assert torch.allclose(ratio, torch.full_like(ratio, 3.5), atol=1e-3)


def test_arm_a_is_the_incumbent_feature_vector_exactly():
    cand, frozen = _field(24, 0), _field(24, 1)
    assert torch.equal(arm_paired_features(cand, frozen, "a"),
                       paired_features(cand, frozen))


def test_arm_b_extends_arm_a_without_disturbing_it():
    """B's first 13 coordinates are A's, so the shared block is literally shared."""
    cand = _field(24, 2)
    a = arm_features(cand, "a")
    b = arm_features(cand, "b")
    n_dens = len(feature_names(SoftStructureConfig()))
    assert a.shape[1] == n_dens
    assert b.shape[1] == n_dens + len(phase_space_feature_names())
    assert torch.equal(b[:, :n_dens], a)


def test_paired_dimensions_and_names_line_up():
    for arm, expect in (("a", 26), ("b", 44)):
        names = arm_paired_feature_names(arm)
        assert len(names) == expect
        assert len(set(names)) == expect
        cand, frozen = _field(24, 3), _field(24, 4)
        assert arm_paired_features(cand, frozen, arm).shape == (2, expect)


def test_velocity_features_ignore_the_velocity_of_a_three_channel_field():
    """Arm B on a displacement-only field is an error, not a silent zero.

    Scoring the missing velocity as zero would make arm B quietly equal arm A
    plus nine constants, and the comparison would report "no difference" for the
    wrong reason.
    """
    with pytest.raises(ValueError, match="6 channels"):
        arm_features(_field(20)[:, 0:3], "b")


def test_the_synthetic_ladders_hold():
    """Infall raises the coherence, speed raises the dispersion, density is inert."""
    d = validate_phase_space_features(n=24)
    assert d["infall_monotone_increasing"], d["infall_coherence"]
    assert d["vdisp_monotone_increasing"], d["vdisp_compact"]
    # The strong claim: NOTHING in the density block moved while only the
    # velocities changed. Exactly zero, because the two blocks share only the
    # displacement channels, which the ladder holds fixed.
    assert d["density_feature_max_abs_spread"] == 0.0
    assert d["ok"]


def test_features_are_differentiable_in_both_channel_groups():
    """The proxy is a gradient source; a feature with no gradient is decoration."""
    f = (_field(24, 5)).clone().requires_grad_(True)
    frozen = _field(24, 6)
    arm_paired_features(f, frozen, "b").sum().backward()
    assert torch.isfinite(f.grad).all()
    assert float(f.grad[:, 0:3].abs().sum()) > 0
    assert float(f.grad[:, 3:6].abs().sum()) > 0


def test_no_nan_on_an_empty_region():
    """A crop that deposits almost nothing must not produce NaN.

    Every velocity feature is a ratio, and the tiles at the edge of a collapsed
    region genuinely do have near-zero compact mass. A NaN there would poison a
    whole training batch through the loss.
    """
    disp = torch.zeros(1, 3, 20, 20, 20)
    vel = torch.zeros(1, 3, 20, 20, 20)
    out = phase_space_features(disp, vel, SoftStructureConfig(), PhaseSpaceConfig())
    assert torch.isfinite(out).all(), out


def test_vel_norm_is_part_of_the_configuration():
    """Rescaling the stored velocities and the norm together is a no-op.

    That is the property that makes vel_norm_km_s a unit conversion rather than
    a tuning knob: the features describe km/s, whatever the storage convention.
    """
    f = _field(24, 7)
    base = phase_space_features(f[:, 0:3], f[:, 3:6], None, PhaseSpaceConfig())
    scaled = phase_space_features(
        f[:, 0:3], f[:, 3:6] * 2.0, None,
        PhaseSpaceConfig(vel_norm_km_s=PhaseSpaceConfig().vel_norm_km_s / 2.0))
    assert torch.allclose(base, scaled, atol=1e-5)


def test_arms_registry_is_what_the_scripts_iterate():
    assert ARMS == ("a", "b")
