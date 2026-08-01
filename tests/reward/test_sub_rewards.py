"""R_occ and R_abund: the occupation-primary decomposition of the reward.

The joint 11-d Mahalanobis distance can fall because the abundance block moved
while <N_sub | M_host> stayed flat. Occupation is the primary scientific target,
so it gets its own score, and these tests pin down the properties Gate B relies
on:

* the sub-scores are real Mahalanobis distances (zero at mu, positive elsewhere);
* they are *marginal*, not slices of the joint precision, so a change confined
  to one block cannot move the other block's score;
* a joint improvement driven entirely by abundance leaves R_occ unchanged --
  which is exactly the case Gate B must not pass;
* empty host bins do not produce NaN scores;
* the reliable-bin variant ignores the sparse bins it excludes.
"""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.reward.catalog import EnsembleSummary, pool
from cosmo_sr.reward.reward import fit_reward_model


@pytest.fixture
def model(hr_chunks, bins):
    # method="bootstrap", covariance="full": these tests are about the
    # block-vs-joint-precision algebra, which needs off-diagonal structure to be
    # a real distinction. The production default (whole_box, covariance="auto")
    # diagonalizes with this fixture's 4 boxes and 5 active dims -- correctly,
    # since 4 boxes cannot identify a 5-d covariance -- but a diagonal covariance
    # makes every block trivially independent and collapses exactly the property
    # under test.
    return fit_reward_model(hr_chunks, bins, ensemble_size=8, n_draws=300, seed=0,
                            method="bootstrap", covariance="full")


def _ens(like, *, n_sub=None, occ_num=None, n_host=None):
    return EnsembleSummary(
        n_sub=np.asarray(n_sub if n_sub is not None else like.n_sub, dtype=float),
        n_host=np.asarray(n_host if n_host is not None else like.n_host, dtype=float),
        occ_numerator=np.asarray(
            occ_num if occ_num is not None else like.occ_numerator, dtype=float),
        volume_mpc3=float(like.volume_mpc3),
    )


def test_index_blocks_partition_the_summary_vector(model, bins):
    a = model.abundance_index
    o = model.occupation_index
    assert a.tolist() == list(range(bins.n_sub_bins))
    assert o.tolist() == list(range(bins.n_sub_bins, bins.dim))
    assert set(a.tolist()).isdisjoint(o.tolist())


def test_sub_scores_are_nonpositive_and_zero_at_the_hr_mean(model, hr_chunks):
    ens = pool(hr_chunks[:8])
    s = model.scores(ens)
    assert set(s) == {"R_cat", "R_occ", "R_abund"}
    for v in s.values():
        assert np.isfinite(v) and v <= 0.0

    # An ensemble sitting exactly at mu scores zero in every block.
    j = model.bins.n_sub_bins
    vol = float(ens.volume_mpc3)
    dens = 10.0 ** model.mu[:j] - model.bins.abundance_floor_halos / vol
    occ = 10.0 ** model.mu[j:] - model.bins.occupation_floor
    n_host = np.maximum(np.asarray(ens.n_host, dtype=float), 1.0)
    at_mu = EnsembleSummary(dens * vol, n_host, occ * n_host, vol)
    s0 = model.scores(at_mu)
    assert s0["R_cat"] == pytest.approx(0.0, abs=1e-8)
    assert s0["R_occ"] == pytest.approx(0.0, abs=1e-8)
    assert s0["R_abund"] == pytest.approx(0.0, abs=1e-8)


def test_the_blocks_are_marginal_so_one_block_cannot_move_the_other(model, hr_chunks):
    """The whole point of the split.

    A slice of the *joint* precision would leak the other block's residual in
    through the cross-covariance -- so perturbing abundance alone would change
    R_occ, and Gate B could be passed by an abundance-only sample. The marginal
    form must be immune to that.
    """
    ens = pool(hr_chunks[:8])
    base = model.scores(ens)

    abundance_moved = _ens(ens, n_sub=np.asarray(ens.n_sub, dtype=float) * 3.0)
    s = model.scores(abundance_moved)
    assert s["R_abund"] < base["R_abund"] - 1e-9      # abundance got worse
    assert s["R_occ"] == pytest.approx(base["R_occ"], rel=1e-9, abs=1e-9)

    occupation_moved = _ens(
        ens, occ_num=np.asarray(ens.occ_numerator, dtype=float) * 3.0)
    s = model.scores(occupation_moved)
    assert s["R_occ"] < base["R_occ"] - 1e-9
    assert s["R_abund"] == pytest.approx(base["R_abund"], rel=1e-9, abs=1e-9)


def test_an_abundance_only_fix_moves_R_abund_and_not_R_occ(model, hr_chunks):
    """The failure mode Gate B exists to catch, and why R_cat cannot decide it.

    Start from an ensemble that is wrong in both blocks and fix only abundance.
    R_abund improves and R_occ does not move at all -- so occupation is exactly
    as flat as before, which is the outcome Gate B must refuse to pass.

    The joint R_cat, meanwhile, is not even guaranteed to move in the *right*
    direction: with strongly correlated bins, being consistently wrong in both
    blocks costs less Mahalanobis distance than being wrong in one, so a genuine
    abundance fix can make R_cat worse. That is not a bug in the reward -- it is
    a correlated-Gaussian statement about a joint discrepancy -- but it is a
    decisive reason not to read occupation progress off R_cat.
    """
    ens = pool(hr_chunks[:8])
    bad = _ens(ens,
               n_sub=np.asarray(ens.n_sub, dtype=float) * 0.3,
               occ_num=np.asarray(ens.occ_numerator, dtype=float) * 0.3)
    fixed = _ens(bad, n_sub=np.asarray(ens.n_sub, dtype=float))

    r_bad, r_fix = model.scores(bad), model.scores(fixed)
    assert r_fix["R_abund"] > r_bad["R_abund"]                  # abundance fixed
    assert r_fix["R_occ"] == pytest.approx(r_bad["R_occ"], rel=1e-9, abs=1e-9)
    # R_cat moved, but its sign carries no information about occupation.
    assert r_fix["R_cat"] != pytest.approx(r_bad["R_cat"], rel=1e-6)


def test_occupation_gap_is_nan_for_empty_host_bins_not_zero(model, hr_chunks):
    ens = pool(hr_chunks[:8])
    empty = _ens(ens,
                 n_host=np.array([float(ens.n_host[0]), 0.0]),
                 occ_num=np.array([float(ens.occ_numerator[0]), 0.0]))
    gap = model.occupation_gap(empty)
    assert np.isfinite(gap[0])
    assert np.isnan(gap[1]), "an empty bin must not look like a perfect match"
    # and the scores stay finite regardless
    for v in model.scores(empty).values():
        assert np.isfinite(v)


def test_occupation_gap_shrinks_as_the_ensemble_approaches_hr(model, hr_chunks):
    ens = pool(hr_chunks[:8])
    gaps = []
    for f in (0.2, 0.5, 0.8, 1.0):
        num = np.asarray(ens.occ_numerator, dtype=float)
        target_occ = 10.0 ** model.mu[model.bins.n_sub_bins:] - model.bins.occupation_floor
        target_num = target_occ * np.asarray(ens.n_host, dtype=float)
        gaps.append(np.nanmean(
            model.occupation_gap(_ens(ens, occ_num=num + f * (target_num - num)))))
    assert all(b <= a + 1e-9 for a, b in zip(gaps, gaps[1:])), gaps


def test_reliable_bin_variant_ignores_the_excluded_sparse_bins(model, hr_chunks):
    ens = pool(hr_chunks[:8])
    reliable = [0]                       # exclude host bin 1 as "sparse"
    s = model.scores(ens, reliable)
    assert "R_occ_reliable" in s and np.isfinite(s["R_occ_reliable"])

    # Wrecking only the excluded bin must not change the reliable-bin score.
    wrecked = _ens(ens, occ_num=np.asarray(ens.occ_numerator, dtype=float)
                   * np.array([1.0, 8.0]))
    s2 = model.scores(wrecked, reliable)
    assert s2["R_occ_reliable"] == pytest.approx(s["R_occ_reliable"], rel=1e-9, abs=1e-9)
    assert s2["R_occ"] < s["R_occ"] - 1e-9      # but the full R_occ notices


def test_block_precision_matches_an_explicit_submatrix_inverse(model, hr_chunks):
    idx = model.occupation_index
    sub = model.cov_reg[np.ix_(idx, idx)]
    ens = pool(hr_chunks[:8])
    s, _ = model.vector(ens)
    d = (s - model.mu)[idx]
    expected = float(d @ np.linalg.inv(sub) @ d)
    assert model.block_mahalanobis2(ens, idx) == pytest.approx(expected, rel=1e-10)
    # And it is NOT the joint-precision slice, which would be the partial form.
    joint_slice = float(d @ model.precision[np.ix_(idx, idx)] @ d)
    assert abs(joint_slice - expected) > 1e-12


# --------------------------------------------------------------------------- #
# include_sparse_in_reward: an excluded bin must carry no weight at all
# --------------------------------------------------------------------------- #
def test_an_inactive_dimension_cannot_change_the_reward(hr_chunks, bins):
    """Dropping the sparse host bin must actually drop it from R_cat.

    ``include_sparse_in_reward: false`` was documentation only: bins_of built
    every host bin regardless, so the near-empty 1e14 bin stayed in the 11-d
    quadratic form -- the configuration audit_reward_covariance.py reports as
    "fail", with 96% of the baseline distance in that one bin.
    """
    import numpy as np

    from cosmo_sr.reward.catalog import summary_vector
    from cosmo_sr.reward.reward import fit_reward_model

    full = fit_reward_model(hr_chunks, bins, ensemble_size=8, n_draws=200, seed=0)
    last = bins.dim - 1
    active = [i for i in range(bins.dim) if i != last]
    trimmed = fit_reward_model(hr_chunks, bins, ensemble_size=8, n_draws=200,
                               seed=0, active_dims=active)

    assert trimmed.active_dim == bins.dim - 1
    assert full.active_dim == bins.dim

    # The excluded bin contributes exactly zero, whatever it holds.
    ens = _ens_from(hr_chunks)
    comp = trimmed.components(ens)
    names = list(bins.labels())
    assert comp[f"contrib_{names[last]}"] == 0.0
    assert comp["mahalanobis2"] == pytest.approx(-trimmed.reward(ens))
    # ...and it is still in mu/cov, so it remains reportable.
    assert trimmed.dim == bins.dim
    assert np.isfinite(trimmed.mu[last])


def test_the_excluded_bin_is_still_reported_in_the_occupation_gap(hr_chunks, bins):
    from cosmo_sr.reward.reward import fit_reward_model

    active = list(range(bins.dim - 1))
    m = fit_reward_model(hr_chunks, bins, ensemble_size=8, n_draws=200, seed=0,
                         active_dims=active)
    gap = m.occupation_gap(_ens_from(hr_chunks))
    # Indexed by HOST BIN, not by active dimension: the gate reads it that way.
    assert gap.shape[0] == bins.n_host_bins


def _ens_from(chunks):
    from cosmo_sr.reward.catalog import pool

    return pool(list(chunks)[:8])
