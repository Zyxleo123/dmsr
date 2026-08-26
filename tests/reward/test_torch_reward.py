"""The torch reward must BE the numpy reward, not resemble it.

Every comparison here is an equality to 1e-5 or better on a model fitted the way
production fits one. The empty-host-bin case gets its own tests because it is
the only place the two implementations branch, and a disagreement there would be
invisible in an aggregate: an empty bin is filled with ``mu`` and contributes
exactly zero, so a wrong branch produces a *plausible* number.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmo_sr.reward.catalog import (
    CatalogBins, ChunkSummary, EnsembleSummary, pool, summary_vector,
)
from cosmo_sr.reward.reward import fit_reward_model
from cosmo_sr.reward.tiles import TileSummary
from cosmo_sr.reward.torch_reward import (
    TorchRewardModel, TorchSummary, summary_from_ensemble, summary_from_tiles,
)

TOL = 1e-5


@pytest.fixture
def wide_bins() -> CatalogBins:
    """The production shape: 6 subhalo-mass bins, 5 host-mass bins."""
    return CatalogBins(
        sub_mass_edges=tuple(np.logspace(10.1, 13.1, 7).tolist()),
        host_mass_edges=tuple(np.logspace(12.0, 14.5, 6).tolist()),
    )


@pytest.fixture
def population(wide_bins):
    rng = np.random.default_rng(0)
    out = []
    for b in range(6):
        off = float(rng.normal(0.0, 0.15))
        for c in range(8):
            s = float(np.exp(off + rng.normal(0.0, 0.05)))
            out.append(ChunkSummary(
                box=f"set{b}", chunk_id=c, source="hr",
                n_sub=np.round(np.array([400., 160., 60., 22., 8., 3.]) * s),
                n_host=np.round(np.array([160., 54., 16., 4., 1.]) * s) + 1,
                occ_numerator=np.round(np.array([200., 90., 40., 14., 5.]) * s),
                volume_mpc3=1562.5,
            ))
    return out


@pytest.fixture
def model(population, wide_bins):
    # active_dims drops the sparse 1e14 host bin, exactly as the production
    # config does (reward.occupation.include_sparse_in_reward: false).
    return fit_reward_model(population, wide_bins,
                            active_dims=[i for i in range(11) if i != 10])


@pytest.fixture
def tmodel(model):
    return TorchRewardModel.from_numpy(model)


def _ens(population, lo=0, hi=8):
    return pool(population[lo:hi])


def test_active_dims_and_blocks_are_carried_over(model, tmodel):
    assert tmodel.active_dims == model.active_dims
    assert list(tmodel.abundance_dims) == [int(i) for i in model.abundance_index]
    assert list(tmodel.occupation_dims) == [int(i) for i in model.occupation_index]
    assert 10 not in tmodel.active_dims


def test_cov_reg_matches_including_the_ridge(model, tmodel):
    assert np.allclose(tmodel.cov_reg.numpy(), model.cov_reg, rtol=0, atol=1e-12)


def test_summary_vector_parity(model, tmodel, population):
    ens = _ens(population)
    s_np, valid_np = summary_vector(ens, model.bins, empty_fill=model.mu)
    s_t, valid_t = tmodel.summary_vector(summary_from_ensemble(ens))
    assert np.allclose(s_t[0].numpy(), s_np, rtol=0, atol=TOL)
    assert np.array_equal(valid_t[0].numpy(), valid_np)


@pytest.mark.parametrize("lo,hi", [(0, 8), (8, 16), (0, 48), (16, 20)])
def test_reward_parity_nonempty(model, tmodel, population, lo, hi):
    ens = _ens(population, lo, hi)
    ts = summary_from_ensemble(ens)
    got = {k: float(v[0]) for k, v in tmodel.scores(ts).items()}
    assert got["R_cat"] == pytest.approx(model.reward(ens), abs=TOL)
    assert got["R_occ"] == pytest.approx(model.reward_occupation(ens), abs=TOL)
    assert got["R_abund"] == pytest.approx(model.reward_abundance(ens), abs=TOL)


def test_reward_parity_with_empty_host_bins(model, tmodel, population):
    """Two of five host bins emptied: the branch the two versions could differ on."""
    ens = _ens(population)
    empty = EnsembleSummary(
        n_sub=ens.n_sub,
        n_host=np.array([120.0, 40.0, 0.0, 0.0, 2.0]),
        occ_numerator=np.array([150.0, 60.0, 0.0, 0.0, 3.0]),
        volume_mpc3=ens.volume_mpc3,
    )
    ts = summary_from_ensemble(empty)
    assert np.any(empty.empty_host_bins())
    assert float(tmodel.reward(ts)[0]) == pytest.approx(model.reward(empty), abs=TOL)
    assert float(tmodel.reward_occupation(ts)[0]) == pytest.approx(
        model.reward_occupation(empty), abs=TOL)
    assert float(tmodel.reward_abundance(ts)[0]) == pytest.approx(
        model.reward_abundance(empty), abs=TOL)


def test_all_host_bins_empty_parity(model, tmodel, population):
    ens = _ens(population)
    empty = EnsembleSummary(ens.n_sub, np.zeros(5), np.zeros(5), ens.volume_mpc3)
    ts = summary_from_ensemble(empty)
    # Every occupation entry falls back to mu, so R_occ is exactly zero.
    assert float(tmodel.reward_occupation(ts)[0]) == pytest.approx(
        model.reward_occupation(empty), abs=TOL)
    assert float(tmodel.reward_occupation(ts)[0]) == pytest.approx(0.0, abs=1e-12)


def test_occupation_curve_parity_including_nan(model, tmodel, population):
    ens = _ens(population)
    mixed = EnsembleSummary(ens.n_sub, np.array([10.0, 0.0, 5.0, 0.0, 1.0]),
                            np.array([20.0, 3.0, 12.0, 0.0, 2.0]), ens.volume_mpc3)
    ts = summary_from_ensemble(mixed)
    a = model.occupation_curve(mixed)
    b = tmodel.occupation_curve(ts)[0].numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b))
    ok = ~np.isnan(a)
    assert np.allclose(a[ok], b[ok], rtol=0, atol=TOL)


def test_batched_agrees_with_one_at_a_time(model, tmodel, population):
    a, b = _ens(population, 0, 8), _ens(population, 8, 16)
    batch = TorchSummary(
        n_sub=torch.tensor(np.stack([a.n_sub, b.n_sub]), dtype=torch.float64),
        n_host=torch.tensor(np.stack([a.n_host, b.n_host]), dtype=torch.float64),
        occ_numerator=torch.tensor(np.stack([a.occ_numerator, b.occ_numerator]),
                                   dtype=torch.float64),
        volume_mpc3=torch.tensor([a.volume_mpc3, b.volume_mpc3], dtype=torch.float64),
    )
    r = tmodel.reward(batch)
    assert float(r[0]) == pytest.approx(model.reward(a), abs=TOL)
    assert float(r[1]) == pytest.approx(model.reward(b), abs=TOL)


def test_reward_sign_a_summary_closer_to_hr_scores_higher(model, tmodel):
    """The direction the whole optimisation depends on."""
    mu = model.mu
    bins = model.bins
    j = bins.n_sub_bins

    def summary_at(scale: float) -> EnsembleSummary:
        # Build counts whose transformed vector sits `scale` of the way from a
        # deliberately-wrong point back to mu.
        vol = 1562.5
        target_ab = 10.0 ** mu[:j] * vol
        target_occ = 10.0 ** mu[j:] - bins.occupation_floor
        wrong_ab = target_ab * 0.25
        wrong_occ = target_occ * 0.25
        n_sub = wrong_ab + scale * (target_ab - wrong_ab)
        occ = wrong_occ + scale * (target_occ - wrong_occ)
        n_host = np.full(bins.n_host_bins, 50.0)
        return EnsembleSummary(n_sub, n_host, occ * n_host, vol)

    far, near = summary_at(0.0), summary_at(0.9)
    assert model.reward(near) > model.reward(far)
    ts_far, ts_near = summary_from_ensemble(far), summary_from_ensemble(near)
    assert float(tmodel.reward(ts_near)) > float(tmodel.reward(ts_far))
    assert float(tmodel.reward_occupation(ts_near)) > float(
        tmodel.reward_occupation(ts_far))


def test_summary_from_tiles_pools_like_pool_tiles(tmodel, wide_bins):
    from cosmo_sr.reward.tiles import pool_tiles

    tiles = [
        TileSummary(box="setA", tile_id=t, source="frozen",
                    n_sub=np.arange(6, dtype=np.float64) + t,
                    n_host=np.arange(5, dtype=np.float64) + 1.0,
                    occ_numerator=np.arange(5, dtype=np.float64) * 2.0,
                    volume_mpc3=100.0)
        for t in range(4)
    ]
    ts = summary_from_tiles(tiles)
    ref = pool_tiles(tiles)
    assert np.allclose(ts.n_sub[0].numpy(), ref.n_sub)
    assert np.allclose(ts.n_host[0].numpy(), ref.n_host)
    assert np.allclose(ts.occ_numerator[0].numpy(), ref.occ_numerator)
    assert float(ts.volume_mpc3[0]) == pytest.approx(ref.volume_mpc3)


def test_swap_summary_keeps_volume_and_floors_at_zero(tmodel, population):
    box = summary_from_ensemble(_ens(population))
    frozen_tile = TorchSummary(
        n_sub=box.n_sub * 0.01, n_host=box.n_host * 0.01,
        occ_numerator=box.occ_numerator * 0.01, volume_mpc3=box.volume_mpc3,
    )
    # A wildly over-large prediction is legal input and must not produce a
    # negative host count.
    huge = TorchSummary(
        n_sub=box.n_sub * 5.0, n_host=box.n_host * 5.0,
        occ_numerator=box.occ_numerator * 5.0, volume_mpc3=box.volume_mpc3,
    )
    out = tmodel.swap_summary(box, frozen_tile, huge)
    assert torch.equal(out.volume_mpc3, box.volume_mpc3)
    assert bool((out.n_host >= 0).all())

    negative = TorchSummary(
        n_sub=torch.zeros_like(box.n_sub), n_host=torch.zeros_like(box.n_host),
        occ_numerator=torch.zeros_like(box.occ_numerator),
        volume_mpc3=box.volume_mpc3,
    )
    big_frozen = TorchSummary(
        n_sub=box.n_sub * 2.0, n_host=box.n_host * 2.0,
        occ_numerator=box.occ_numerator * 2.0, volume_mpc3=box.volume_mpc3,
    )
    clamped = tmodel.swap_clamped_fraction(box, big_frozen, negative)
    assert float(clamped[0]) > 0.0


def test_delta_reward_swap_is_zero_when_nothing_changes(tmodel, population):
    box = summary_from_ensemble(_ens(population))
    tile = TorchSummary(
        n_sub=box.n_sub * 0.02, n_host=box.n_host * 0.02,
        occ_numerator=box.occ_numerator * 0.02, volume_mpc3=box.volume_mpc3,
    )
    out = tmodel.delta_reward_swap(box, tile, tile)
    for key in ("dR_cat", "dR_occ", "dR_abund", "dR_combined", "dR_hosted_subs"):
        assert float(out[key][0]) == pytest.approx(0.0, abs=1e-10)


def test_reward_hosted_subs_is_log10_of_pooled_numerator(tmodel, population):
    """The hosted-subhalo target reads log10(sum occ_numerator), nothing else.

    This is the whole-box statistic reward_stability_scan selected; the swap
    form must reduce to it so the trained proxy and the scan agree on what the
    reward is.
    """
    box = summary_from_ensemble(_ens(population))
    import numpy as np
    ref = float(np.log10(max(float(box.occ_numerator.sum()), 0.5)))
    assert float(tmodel.reward_hosted_subs(box)[0]) == pytest.approx(ref, abs=1e-9)


def test_dR_hosted_subs_ignores_host_deletion(tmodel, population):
    """Deleting hosts (occ_numerator unchanged) does not raise the hosted target.

    This is the property the target was chosen for: the occupation *ratio* rose
    on the alpha ladder by shrinking the host denominator, and dR_hosted_subs
    must be immune to exactly that -- it depends on the numerator alone.
    """
    box = summary_from_ensemble(_ens(population))
    frozen_tile = TorchSummary(
        n_sub=box.n_sub * 0.05, n_host=box.n_host * 0.05,
        occ_numerator=box.occ_numerator * 0.05, volume_mpc3=box.volume_mpc3,
    )
    # Same subhalos-in-hosts, but half the hosts removed from this tile.
    fewer_hosts = TorchSummary(
        n_sub=frozen_tile.n_sub, n_host=frozen_tile.n_host * 0.5,
        occ_numerator=frozen_tile.occ_numerator, volume_mpc3=box.volume_mpc3,
    )
    out = tmodel.delta_reward_swap(box, frozen_tile, fewer_hosts)
    assert float(out["dR_hosted_subs"][0]) == pytest.approx(0.0, abs=1e-10)
    # The ratio target reacts to the same host-only edit; the hosted count does
    # not. (Sign is fixture-dependent -- what matters is that it is not immune.)
    assert abs(float(out["dR_occ"][0])) > 1e-6


def test_delta_reward_swap_matches_direct_recomputation(tmodel, population):
    box = summary_from_ensemble(_ens(population))
    frozen_tile = TorchSummary(
        n_sub=box.n_sub * 0.02, n_host=box.n_host * 0.02,
        occ_numerator=box.occ_numerator * 0.02, volume_mpc3=box.volume_mpc3,
    )
    better = TorchSummary(
        n_sub=frozen_tile.n_sub * 1.3, n_host=frozen_tile.n_host,
        occ_numerator=frozen_tile.occ_numerator * 1.5,
        volume_mpc3=box.volume_mpc3,
    )
    swapped = tmodel.swap_summary(box, frozen_tile, better)
    out = tmodel.delta_reward_swap(box, frozen_tile, better)
    assert float(out["dR_occ"][0]) == pytest.approx(
        float(tmodel.reward_occupation(swapped)[0]
              - tmodel.reward_occupation(box)[0]), abs=1e-10)


def test_gradient_flows_into_the_predicted_tile(tmodel, population):
    box = summary_from_ensemble(_ens(population))
    frozen_tile = TorchSummary(
        n_sub=box.n_sub * 0.02, n_host=box.n_host * 0.02,
        occ_numerator=box.occ_numerator * 0.02, volume_mpc3=box.volume_mpc3,
    )
    pred = TorchSummary(
        n_sub=(frozen_tile.n_sub * 1.1).requires_grad_(True),
        n_host=(frozen_tile.n_host * 1.0).requires_grad_(True),
        occ_numerator=(frozen_tile.occ_numerator * 1.2).requires_grad_(True),
        volume_mpc3=box.volume_mpc3,
    )
    tmodel.delta_reward_swap(box, frozen_tile, pred)["dR_combined"].sum().backward()
    for t in (pred.n_sub, pred.n_host, pred.occ_numerator):
        assert t.grad is not None
        assert torch.isfinite(t.grad).all()
        assert float(t.grad.abs().sum()) > 0.0


def test_combined_weights_occupation_primary(tmodel, population):
    ts = summary_from_ensemble(_ens(population))
    c = tmodel.combined(ts, w_joint=0.25, w_occ=1.0)
    expected = tmodel.reward_occupation(ts) + 0.25 * tmodel.reward(ts)
    assert float(c[0]) == pytest.approx(float(expected[0]), abs=1e-12)
