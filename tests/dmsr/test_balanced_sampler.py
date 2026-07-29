"""7.5 -- the environment-balanced sampler actually closes the distribution gap."""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.dmsr.env import (
    DESCRIPTOR_NAMES,
    DescriptorStandardizer,
    EnvironmentBalancedSampler,
    roc_auc,
    source_classifier_auc,
)

N_DESC = len(DESCRIPTOR_NAMES)


def _synthetic_pools(seed=0, n_paired=400, n_unpaired=3000, shift=2.0):
    """Deliberately mismatched pools: unpaired is shifted and over-dispersed."""
    rng = np.random.default_rng(seed)
    paired = np.zeros((n_paired, N_DESC))
    unpaired = np.zeros((n_unpaired, N_DESC))
    # Two informative descriptors (indices 2 and 5), the rest constant.
    paired[:, 2] = rng.normal(0.0, 1.0, n_paired)
    paired[:, 5] = rng.normal(0.0, 1.0, n_paired)
    unpaired[:, 2] = rng.normal(shift, 1.8, n_unpaired)
    unpaired[:, 5] = rng.normal(shift * 0.5, 1.8, n_unpaired)
    return paired, unpaired


def test_roc_auc_matches_known_cases():
    assert roc_auc(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    assert roc_auc(np.array([4.0, 3.0, 2.0, 1.0]), np.array([0, 0, 1, 1])) == pytest.approx(0.0)
    assert roc_auc(np.ones(4), np.array([0, 0, 1, 1])) == pytest.approx(0.5)


def test_ordinary_sampling_preserves_the_discrepancy():
    """Without balancing, a classifier separates the two pools easily."""
    paired, unpaired = _synthetic_pools()
    std = DescriptorStandardizer.fit(paired)
    auc = source_classifier_auc(std.transform(paired), std.transform(unpaired), seed=0)
    assert auc > 0.75, f"synthetic pools were not actually distinguishable (AUC={auc:.3f})"


def test_balanced_sampler_substantially_reduces_the_discrepancy():
    paired, unpaired = _synthetic_pools()
    std = DescriptorStandardizer.fit(paired)
    sampler = EnvironmentBalancedSampler(paired, unpaired, std, n_dims=2, n_bins=8, seed=0)

    assert sampler.auc_after < sampler.auc_before
    assert sampler.auc_after <= 0.60, (
        f"balanced AUC {sampler.auc_after:.3f} exceeds the 0.60 target "
        f"(before={sampler.auc_before:.3f})"
    )


def test_constant_descriptors_are_dropped():
    paired, _ = _synthetic_pools()
    std = DescriptorStandardizer.fit(paired)
    assert set(std.kept_names) == {DESCRIPTOR_NAMES[2], DESCRIPTOR_NAMES[5]}
    assert "redshift" in std.dropped_names
    assert std.transform(paired).shape[1] == 2


def test_out_of_support_crops_are_rejected():
    paired, unpaired = _synthetic_pools(shift=3.0)
    std = DescriptorStandardizer.fit(paired)
    sampler = EnvironmentBalancedSampler(paired, unpaired, std, n_dims=2, n_bins=8, seed=0)
    rep = sampler.report()
    assert rep.n_rejected > 0, "no unpaired crop was flagged outside paired support"
    assert rep.n_in_support + rep.n_rejected == rep.n_unpaired
    # Rejected crops must carry exactly zero sampling probability.
    assert float(sampler.weights[~sampler.in_support].sum()) == 0.0


def test_sampling_is_reproducible_with_a_fixed_seed():
    paired, unpaired = _synthetic_pools()
    std = DescriptorStandardizer.fit(paired)
    a = EnvironmentBalancedSampler(paired, unpaired, std, seed=7).sample(256)
    b = EnvironmentBalancedSampler(paired, unpaired, std, seed=7).sample(256)
    c = EnvironmentBalancedSampler(paired, unpaired, std, seed=8).sample(256)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_weights_form_a_probability_distribution():
    paired, unpaired = _synthetic_pools()
    std = DescriptorStandardizer.fit(paired)
    s = EnvironmentBalancedSampler(paired, unpaired, std, seed=0)
    assert s.weights.min() >= 0.0
    assert float(s.weights.sum()) == pytest.approx(1.0, abs=1e-9)


def test_disjoint_pools_raise():
    """No overlap at all is an error, not a silently empty sampler."""
    rng = np.random.default_rng(0)
    paired = np.zeros((200, N_DESC)); paired[:, 2] = rng.normal(0, 0.1, 200)
    unpaired = np.zeros((200, N_DESC)); unpaired[:, 2] = rng.normal(500, 0.1, 200)
    std = DescriptorStandardizer.fit(paired)
    with pytest.raises(ValueError, match="do not overlap|no unpaired crop"):
        EnvironmentBalancedSampler(paired, unpaired, std, seed=0)
