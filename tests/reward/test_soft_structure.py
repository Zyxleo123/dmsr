"""The soft features have to order structure, and they have to carry gradient.

Both are checkable without a generator, a box or a halo finder, so they are
checked here; the "gradient reaches SR2's weights" half lives in
``tests/train/test_sr2_direct_actor.py`` where a generator is available.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmo_sr.reward.soft_structure import (
    SoftStructureConfig, _synthetic_delta, density_from_disp, feature_names,
    paired_feature_names, paired_features, soft_structure_features,
    structural_diversity, validate_soft_features,
)


def test_feature_names_match_width():
    cfg = SoftStructureConfig()
    names = feature_names(cfg)
    f = soft_structure_features(_synthetic_delta(16, 1.0, 100.0), cfg)
    assert f.shape == (1, len(names))
    assert len(set(names)) == len(names)


def test_paired_feature_names_are_value_then_difference():
    cfg = SoftStructureConfig()
    base = feature_names(cfg)
    paired = paired_feature_names(cfg)
    assert paired[:len(base)] == base
    assert paired[len(base):] == [f"d_{n}" for n in base]


def test_compact_clump_scores_above_a_smeared_one_of_the_same_mass():
    """The gate the plan puts in front of the feature proxy."""
    report = validate_soft_features()
    assert report["compact_mass_monotone_decreasing"], report["compact_mass"]
    assert report["second_moment_monotone_increasing"], report["second_moment"]
    assert report["envelope_gains_from_smearing"], report["envelope_mass"]
    # "same mass" has to be true of the test, not just of the intent.
    assert report["total_mass_relative_spread"] < 1e-3
    assert report["ok"]


def test_mass_features_are_not_dominated_by_a_pedestal():
    """A uniform field must score near zero *relative to the signal range*.

    This is why the thresholds are applied to ``log1p(delta)``: with a sigmoid of
    width proportional to ``delta``, an empty cell scored 0.054 on
    ``compact_mass`` against a real signal of a few percent -- the constant was
    larger than the thing being measured. The check is a ratio, not an absolute
    bound, because "small" only means anything against the range the feature
    actually spans.
    """
    cfg = SoftStructureConfig()
    uniform = torch.zeros(1, 1, 16, 16, 16)
    f = soft_structure_features(uniform, cfg)[0]
    names = feature_names(cfg)
    report = validate_soft_features(cfg)
    for key in ("compact_mass", "envelope_mass"):
        pedestal = float(f[names.index(key)])
        span = max(report[key]) - min(report[key])
        assert span > 1e-3, f"{key} has no range to compare against"
        assert pedestal < 0.02 * span, f"{key}: pedestal {pedestal} vs span {span}"


def test_features_are_finite_on_extreme_inputs():
    cfg = SoftStructureConfig()
    for delta in (
        torch.full((1, 1, 8, 8, 8), -1.0),        # a completely empty region
        torch.full((1, 1, 8, 8, 8), 1e4),         # absurdly overdense
        _synthetic_delta(8, 0.5, 1e5),
    ):
        f = soft_structure_features(delta, cfg)
        assert torch.isfinite(f).all(), delta.flatten()[0]


def test_features_are_differentiable_in_the_density():
    cfg = SoftStructureConfig()
    d = _synthetic_delta(16, 1.5, 2000.0).clone().requires_grad_(True)
    soft_structure_features(d, cfg).sum().backward()
    assert d.grad is not None
    assert torch.isfinite(d.grad).all()
    assert float(d.grad.abs().sum()) > 0.0


def test_gradient_reaches_a_displacement_field_through_cic():
    cfg = SoftStructureConfig(ng_hr=512, region_fraction=0.5)
    torch.manual_seed(0)
    disp = (0.02 * torch.randn(1, 3, 32, 32, 32)).requires_grad_(True)
    delta = density_from_disp(disp, cfg)
    assert delta.shape == (1, 1, 16, 16, 16)
    soft_structure_features(delta, cfg).sum().backward()
    assert disp.grad is not None
    assert torch.isfinite(disp.grad).all()
    assert float(disp.grad.abs().sum()) > 0.0


def test_paired_features_difference_is_zero_against_itself():
    cfg = SoftStructureConfig(ng_hr=512)
    torch.manual_seed(1)
    disp = 0.02 * torch.randn(2, 6, 32, 32, 32)
    f = paired_features(disp, disp, cfg)
    n = len(feature_names(cfg))
    assert f.shape == (2, 2 * n)
    assert torch.allclose(f[:, n:], torch.zeros_like(f[:, n:]), atol=1e-6)


def test_paired_features_do_not_backpropagate_into_the_frozen_branch():
    cfg = SoftStructureConfig(ng_hr=512)
    torch.manual_seed(2)
    cand = (0.02 * torch.randn(1, 6, 32, 32, 32)).requires_grad_(True)
    frozen = (0.02 * torch.randn(1, 6, 32, 32, 32)).requires_grad_(True)
    paired_features(cand, frozen, cfg).sum().backward()
    assert cand.grad is not None and float(cand.grad.abs().sum()) > 0.0
    # The baseline is a reference; a gradient into it would be a graph artefact.
    assert frozen.grad is None


def test_structural_diversity_falls_to_zero_for_identical_draws():
    cfg = SoftStructureConfig(ng_hr=512)
    torch.manual_seed(3)
    one = 0.02 * torch.randn(1, 1, 6, 32, 32, 32)
    collapsed = one.expand(1, 2, 6, 32, 32, 32).contiguous()
    d = structural_diversity(collapsed, cfg)
    assert float(d["d_struct"][0]) == pytest.approx(0.0, abs=1e-6)

    varied = torch.cat([one, one + 0.01 * torch.randn_like(one)], dim=1)
    d2 = structural_diversity(varied, cfg)
    assert float(d2["d_struct"][0]) > 0.0
    # The combined figure is the MINIMUM, so one healthy channel cannot excuse a
    # collapsed one.
    assert float(d2["d_struct"][0]) == pytest.approx(
        min(float(d2["displacement"][0]), float(d2["density"][0]),
            float(d2["soft_peak"][0])), rel=1e-6)


def test_structural_diversity_needs_two_draws():
    cfg = SoftStructureConfig(ng_hr=512)
    with pytest.raises(ValueError, match="at least 2 noise draws"):
        structural_diversity(torch.zeros(1, 1, 6, 16, 16, 16), cfg)


def test_diversity_ignores_velocity_only_spread():
    """Six-channel spread would pass here; the structural measure must not."""
    cfg = SoftStructureConfig(ng_hr=512)
    torch.manual_seed(4)
    base = 0.02 * torch.randn(1, 1, 6, 32, 32, 32)
    other = base.clone()
    other[:, :, 3:6] += 10.0          # velocity moves a lot, displacement not at all
    draws = torch.cat([base, other], dim=1)
    d = structural_diversity(draws, cfg)
    assert float(d["displacement"][0]) == pytest.approx(0.0, abs=1e-6)
    assert float(d["d_struct"][0]) == pytest.approx(0.0, abs=1e-6)


def test_config_rejects_inconsistent_scales():
    with pytest.raises(ValueError, match="same length"):
        SoftStructureConfig(peak_scales=(1, 2), peak_deltas=(10.0,))
    with pytest.raises(ValueError, match="envelope_lo"):
        SoftStructureConfig(envelope_lo=100.0, envelope_hi=10.0)
