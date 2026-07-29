"""Section 4: the catalog summary, the pooling algebra, and the reward.

The ten checks the plan asks for, on synthetic catalogs with known answers.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import make_chunk, synthetic_catalog

from cosmo_sr.reward.catalog import (ChunkSummary, EnsembleSummary, load_bins, pool,
                                     read_summaries, summarize_catalog, summary_vector,
                                     write_summaries)
from cosmo_sr.reward.geometry import ChunkGrid, PurityGrid
from cosmo_sr.reward.reward import RewardModel, fit_reward_model


# --- 1. HR-identical summary receives the maximum reward --------------------
def test_hr_identical_summary_is_the_maximum_reward(hr_chunks, bins):
    model = fit_reward_model(hr_chunks, bins, ensemble_size=8, n_draws=200,
                             shrinkage=0.1, seed=0)
    # Construct an ensemble whose summary vector is exactly mu.
    ens = pool(hr_chunks[:8])
    r = model.reward(ens)
    assert r <= 0.0
    # No ensemble can beat the mean itself.
    exact = _ensemble_at(model.mu, bins, ens)
    assert model.reward(exact) == pytest.approx(0.0, abs=1e-8)
    assert model.reward(exact) >= r


def _ensemble_at(target_vec, bins, like: EnsembleSummary) -> EnsembleSummary:
    """Build an EnsembleSummary whose summary_vector is exactly ``target_vec``."""
    j = bins.n_sub_bins
    vol = float(like.volume_mpc3)
    floor_a = bins.abundance_floor_halos / vol
    dens = 10.0 ** np.asarray(target_vec[:j]) - floor_a
    n_sub = dens * vol
    occ = 10.0 ** np.asarray(target_vec[j:]) - bins.occupation_floor
    n_host = np.maximum(np.asarray(like.n_host, dtype=np.float64), 1.0)
    return EnsembleSummary(n_sub=n_sub, n_host=n_host,
                           occ_numerator=occ * n_host, volume_mpc3=vol)


# --- 2. monotone degradation -------------------------------------------------
def test_moving_one_bin_away_lowers_the_reward_monotonically(hr_chunks, bins):
    model = fit_reward_model(hr_chunks, bins, ensemble_size=8, n_draws=200, seed=0)
    ens = pool(hr_chunks[:8])
    rewards = []
    for f in (1.0, 1.5, 2.5, 5.0):
        e = EnsembleSummary(
            n_sub=np.asarray(ens.n_sub) * np.array([f, 1.0, 1.0]),
            n_host=np.asarray(ens.n_host),
            occ_numerator=np.asarray(ens.occ_numerator),
            volume_mpc3=ens.volume_mpc3,
        )
        rewards.append(model.reward(e))
    assert all(b < a for a, b in zip(rewards, rewards[1:])), rewards


# --- 3. correlated bins ------------------------------------------------------
def test_correlated_bins_are_penalised_less_along_the_correlated_direction(bins):
    """A shift along a strongly correlated direction must cost less than the
    same-size shift across it -- that is the entire point of using C^-1."""
    rng = np.random.default_rng(0)
    n = 300
    a = rng.normal(0, 1, n)
    v = np.stack([a, a + 0.01 * rng.normal(0, 1, n)], axis=1)   # near-degenerate
    cov = np.cov(v, rowvar=False)
    model = RewardModel(mu=v.mean(axis=0), cov=cov, lam=1e-4, bins=bins,
                        ensemble_size=8, n_draws=n)
    d_along = np.array([1.0, 1.0])
    d_across = np.array([1.0, -1.0])
    p = model.precision
    assert float(d_along @ p @ d_along) < float(d_across @ p @ d_across)


# --- 4. regularization prevents singular failures ---------------------------
def test_shrinkage_makes_a_singular_covariance_invertible(bins):
    d = bins.dim
    model = RewardModel(mu=np.zeros(d), cov=np.zeros((d, d)), lam=1e-6, bins=bins,
                        ensemble_size=8, n_draws=1)
    assert np.isfinite(model.precision).all()
    assert np.isfinite(model.condition_number)


def test_fit_never_produces_a_singular_covariance():
    from cosmo_sr.reward.catalog import CatalogBins

    b = CatalogBins(sub_mass_edges=(1e10, 1e11, 1e12), host_mass_edges=(1e12, 1e13))
    identical = [make_chunk(box=f"set{i}", cid=0, n_sub=(10, 5), n_host=(3,), occ=(9,))
                 for i in range(3)]
    model = fit_reward_model(identical, b, ensemble_size=2, n_draws=20, shrinkage=0.1)
    assert np.isfinite(model.precision).all()


# --- 5. empty host bins ------------------------------------------------------
def test_empty_host_bins_do_not_create_nans(bins, hr_chunks):
    model = fit_reward_model(hr_chunks, bins, ensemble_size=8, n_draws=100, seed=0)
    empty = EnsembleSummary(
        n_sub=np.array([10.0, 3.0, 1.0]),
        n_host=np.array([0.0, 0.0]),         # no hosts at all
        occ_numerator=np.array([0.0, 0.0]),
        volume_mpc3=900.0,
    )
    s, valid = summary_vector(empty, bins, empty_fill=model.mu)
    assert np.isfinite(s).all()
    assert not valid[bins.n_sub_bins:].any()
    assert np.isfinite(model.reward(empty))
    # An uninformative bin must contribute exactly zero, not a fabricated penalty.
    contrib = model.components(empty)
    assert contrib["n_valid_bins"] == float(bins.n_sub_bins)


# --- 6. pooling is associative ----------------------------------------------
def test_pooling_before_or_after_summarising_gives_the_same_answer(bins, hr_chunks):
    model = fit_reward_model(hr_chunks, bins, ensemble_size=8, n_draws=100, seed=0)
    a, b = hr_chunks[:4], hr_chunks[4:8]
    merged_last = pool([pool(a), pool(b)])
    merged_first = pool(a + b)
    assert np.allclose(merged_last.n_sub, merged_first.n_sub)
    assert np.allclose(merged_last.occ_numerator, merged_first.occ_numerator)
    assert model.reward(merged_last) == pytest.approx(model.reward(merged_first))


def test_occupation_pools_as_a_ratio_of_sums_not_a_mean_of_ratios(bins):
    # 10 subs / 1 host and 0 subs / 9 hosts pool to 1.0, not to 5.0.
    a = make_chunk(cid=0, n_sub=(10, 0, 0), n_host=(1, 0), occ=(10, 0))
    b = make_chunk(cid=1, n_sub=(0, 0, 0), n_host=(9, 0), occ=(0, 0))
    merged = pool([a, b])
    assert merged.occupation()[0] == pytest.approx(1.0)


# --- 7. chunk order is irrelevant -------------------------------------------
def test_chunk_order_does_not_change_the_reward(bins, hr_chunks):
    model = fit_reward_model(hr_chunks, bins, ensemble_size=8, n_draws=100, seed=0)
    sel = hr_chunks[:8]
    r1 = model.reward(pool(sel))
    r2 = model.reward(pool(list(reversed(sel))))
    rng = np.random.default_rng(3)
    order = list(rng.permutation(len(sel)))
    r3 = model.reward(pool([sel[i] for i in order]))
    assert r1 == pytest.approx(r2) == pytest.approx(r3)


# --- 8. boundary-excluded objects never enter a summary ---------------------
def test_boundary_contaminated_halos_are_excluded(bins):
    cat = synthetic_catalog(
        hosts=[((10.0, 10.0, 10.0), 5e12), ((60.0, 60.0, 60.0), 5e12)],
        subs_per_host=[3, 4],
    )
    assign = np.full(cat.n, -1, dtype=np.int64)
    # Assign only the first host and its subhalos; everything else is boundary.
    assign[0] = 0
    assign[2:5] = 0                      # the 3 subs of host 0
    out = summarize_catalog(cat, assign, bins, [1000.0, 1000.0],
                            box="set0", source="hr", chunk_ids=[0, 1])
    assert out[0].n_host.sum() == 1
    assert out[0].n_sub.sum() == 3
    assert out[1].n_host.sum() == 0      # the excluded host contributes nowhere
    assert out[1].n_sub.sum() == 0
    assert out[0].n_excluded_boundary > 0


def test_a_subhalo_whose_host_is_in_another_chunk_is_dropped(bins):
    cat = synthetic_catalog(hosts=[((10.0, 10.0, 10.0), 5e12)], subs_per_host=[2])
    assign = np.array([0, 1, 1], dtype=np.int64)      # host in 0, subs in 1
    out = summarize_catalog(cat, assign, bins, [1000.0, 1000.0],
                            box="set0", source="hr", chunk_ids=[0, 1])
    assert out[0].n_host.sum() == 1
    assert out[0].n_sub.sum() == 0
    assert out[1].n_sub.sum() == 0, "an occupation numerator must match its host's chunk"


def test_unresolved_objects_are_dropped_by_the_particle_cut(bins):
    cat = synthetic_catalog(hosts=[((10.0, 10.0, 10.0), 5e12)], subs_per_host=[3],
                            sub_num_p=5)      # below min_sub_particles = 20
    assign = np.zeros(cat.n, dtype=np.int64)
    out = summarize_catalog(cat, assign, bins, [1000.0], box="set0", source="hr",
                            chunk_ids=[0])
    assert out[0].n_sub.sum() == 0
    assert out[0].n_host.sum() == 1


# --- 9. unoptimized fields do not move the reward ---------------------------
def test_changing_an_unoptimized_field_does_not_change_the_reward(bins, hr_chunks):
    model = fit_reward_model(hr_chunks, bins, ensemble_size=8, n_draws=100, seed=0)
    cat = synthetic_catalog(hosts=[((10.0, 10.0, 10.0), 5e12)], subs_per_host=[4])
    assign = np.zeros(cat.n, dtype=np.int64)
    before = summarize_catalog(cat, assign, bins, [1000.0], box="set0",
                               source="hr", chunk_ids=[0])[0]
    # Velocities and radial positions are held-out statistics; the reward is
    # blind to them by construction.
    cat.vel[:] = cat.vel + 500.0
    cat.pos[:] = (cat.pos + 0.01) % 100.0
    after = summarize_catalog(cat, assign, bins, [1000.0], box="set0",
                              source="hr", chunk_ids=[0])[0]
    assert model.reward(pool([before])) == pytest.approx(model.reward(pool([after])))


# --- 10. serialisation round trip -------------------------------------------
def test_serialised_and_in_memory_summaries_give_identical_rewards(tmp_path, bins,
                                                                   hr_chunks):
    model = fit_reward_model(hr_chunks, bins, ensemble_size=8, n_draws=100, seed=0)
    p = write_summaries(tmp_path / "s.jsonl", hr_chunks[:8])
    back = read_summaries(p)
    assert model.reward(pool(back)) == pytest.approx(model.reward(pool(hr_chunks[:8])))


def test_reward_model_round_trips_through_json(tmp_path, bins, hr_chunks):
    model = fit_reward_model(hr_chunks, bins, ensemble_size=8, n_draws=100, seed=0)
    d = json.loads(json.dumps(model.to_dict()))
    back = RewardModel.from_dict(d)
    ens = pool(hr_chunks[:8])
    assert back.reward(ens) == pytest.approx(model.reward(ens))


# --- configuration and bookkeeping ------------------------------------------
def test_bins_come_from_one_yaml_block():
    b = load_bins({
        "sub_mass_bins": {"log10_min": 10.0, "log10_max": 13.0, "n_bins": 6},
        "host_mass_bins": {"log10_min": 12.0, "log10_max": 14.5, "n_bins": 5},
        "min_sub_particles": 30,
    })
    assert b.n_sub_bins == 6 and b.n_host_bins == 5 and b.dim == 11
    assert b.min_sub_particles == 30
    assert len(b.labels()) == b.dim


def test_raw_counts_are_kept_alongside_normalized_statistics(bins):
    c = make_chunk(n_sub=(7, 2, 0), n_host=(3, 1), occ=(6, 2), volume=500.0)
    d = c.to_dict()
    assert d["n_sub"] == [7, 2, 0] and d["n_host"] == [3, 1]
    ens = pool([c])
    assert np.allclose(ens.number_density(), np.array([7, 2, 0]) / 500.0)
    assert np.allclose(ens.occupation(), np.array([2.0, 2.0]))


def test_pooling_summaries_with_different_bins_is_refused(bins):
    a = make_chunk(n_sub=(1, 2, 3), n_host=(1, 1), occ=(1, 1))
    b = make_chunk(n_sub=(1, 2), n_host=(1, 1), occ=(1, 1))
    with pytest.raises(ValueError, match="different bins"):
        pool([a, b])


def test_condition_number_and_shrinkage_are_reported(hr_chunks, bins):
    model = fit_reward_model(hr_chunks, bins, ensemble_size=8, n_draws=200,
                             shrinkage=0.1, seed=0)
    d = model.to_dict()
    assert d["lam"] > 0
    assert np.isfinite(d["cov_reg_condition_number"])
    assert d["meta"]["shrinkage"] == 0.1
