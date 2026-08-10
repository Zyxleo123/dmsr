"""Region aggregation must be an exact periodic partition, at every width.

Every identity here is one a region-scale label silently depends on: a partition
that duplicates or drops a tile, or a region swap that disagrees with a direct
recomputation, would put unattributable counts into the training target exactly
as a broken tile decomposition would.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmo_sr.reward.catalog import CatalogBins
from cosmo_sr.reward.regions import (
    REGION_WIDTHS, RegionGrid, aggregate_tile_counts, aggregate_tile_volume,
    stack_tile_bags, tile_bags,
)
from cosmo_sr.reward.reward import fit_reward_model
from cosmo_sr.reward.catalog import ChunkSummary, pool
from cosmo_sr.reward.tiles import TileGrid
from cosmo_sr.reward.torch_reward import TorchRewardModel, TorchSummary


NG = 16
TILE = 2          # 8 tiles per axis, the deployed ratio (512/64)
COUNTS = ("n_sub", "n_host", "occ_numerator")


@pytest.fixture
def grid() -> TileGrid:
    return TileGrid(ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=100.0)


@pytest.fixture
def counts(grid):
    rng = np.random.default_rng(0)
    return {
        "n_sub": rng.random((grid.n_tiles, 6)) * 10.0,
        "n_host": rng.random((grid.n_tiles, 5)) * 3.0,
        "occ_numerator": rng.random((grid.n_tiles, 5)) * 6.0,
        "volume_mpc3": np.full(grid.n_tiles, grid.tile_volume_mpc3),
    }


# --------------------------------------------------------------------------- #
# Partition geometry
# --------------------------------------------------------------------------- #
def test_widths_divide_the_grid_and_expose_the_right_counts(grid):
    expect = {1: (512, 1), 2: (64, 8), 4: (8, 64), 8: (1, 1)}
    for w in REGION_WIDTHS:
        rg = RegionGrid(grid, w)
        n_regions, n_offsets = expect[w]
        assert rg.n_regions == n_regions
        assert len(rg.valid_offsets()) == n_offsets
        assert rg.tiles_per_region == w ** 3


def test_width_that_does_not_divide_is_refused(grid):
    with pytest.raises(ValueError, match="does not divide"):
        RegionGrid(grid, 3)


def test_every_partition_covers_every_tile_exactly_once(grid):
    """Periodic wrapping neither duplicates nor omits a tile, at any offset."""
    for w in REGION_WIDTHS:
        rg = RegionGrid(grid, w)
        for off in rg.valid_offsets():
            part = rg.partition(off)              # __post_init__ calls check()
            seen = np.concatenate([part.tiles_of(r)
                                   for r in range(rg.n_regions)])
            assert seen.size == grid.n_tiles
            assert np.array_equal(np.sort(seen), np.arange(grid.n_tiles))
            # Region membership is a genuine partition: sizes all equal.
            sizes = [part.tiles_of(r).size for r in range(rg.n_regions)]
            assert set(sizes) == {rg.tiles_per_region}


def test_a_wrapped_offset_partition_is_still_a_cover(grid):
    """An offset that pushes a region across the periodic boundary is valid."""
    rg = RegionGrid(grid, 2)
    part = rg.partition((1, 1, 1))               # every region straddles a face
    counts = np.bincount(part.region_of_tile, minlength=rg.n_regions)
    assert np.all(counts == rg.tiles_per_region)


# --------------------------------------------------------------------------- #
# Aggregation identities
# --------------------------------------------------------------------------- #
def test_width_one_reproduces_the_tiles(grid, counts):
    rg = RegionGrid(grid, 1)
    part = rg.partition()
    agg = aggregate_tile_counts(part, {k: counts[k] for k in COUNTS})
    for k in COUNTS:
        assert np.allclose(agg[k], counts[k])
    assert np.allclose(aggregate_tile_volume(part, counts["volume_mpc3"]),
                       counts["volume_mpc3"])


def test_width_eight_is_one_whole_box_region(grid, counts):
    rg = RegionGrid(grid, 8)
    assert rg.n_regions == 1 and len(rg.valid_offsets()) == 1
    part = rg.partition()
    agg = aggregate_tile_counts(part, {k: counts[k] for k in COUNTS})
    for k in COUNTS:
        assert agg[k].shape == (1, counts[k].shape[1])
        assert np.allclose(agg[k][0], counts[k].sum(axis=0))


def test_regions_sum_to_the_whole_box_at_every_width_and_offset(grid, counts):
    whole = {k: counts[k].sum(axis=0) for k in COUNTS}
    vol = counts["volume_mpc3"].sum()
    for w in REGION_WIDTHS:
        rg = RegionGrid(grid, w)
        for off in rg.valid_offsets():
            part = rg.partition(off)
            agg = aggregate_tile_counts(part, {k: counts[k] for k in COUNTS})
            for k in COUNTS:
                assert np.abs(agg[k].sum(axis=0) - whole[k]).max() < 1e-6
            assert abs(aggregate_tile_volume(
                part, counts["volume_mpc3"]).sum() - vol) < 1e-6


def test_feature_bags_keep_individual_tiles(grid):
    rng = np.random.default_rng(1)
    feats = rng.random((grid.n_tiles, 4))
    rg = RegionGrid(grid, 2)
    part = rg.partition()
    bags = tile_bags(part, feats)
    assert len(bags) == rg.n_regions
    for r, bag in enumerate(bags):
        assert bag.shape == (rg.tiles_per_region, 4)
        assert np.array_equal(bag, feats[part.tiles_of(r)])   # not summed
    stacked = stack_tile_bags(part, feats)
    assert stacked.shape == (rg.n_regions, rg.tiles_per_region, 4)


# --------------------------------------------------------------------------- #
# Region reward: swap identity and mixed-baseline safety
# --------------------------------------------------------------------------- #
def _reward_model():
    bins = CatalogBins(sub_mass_edges=tuple(np.logspace(10.1, 13.1, 7).tolist()),
                       host_mass_edges=tuple(np.logspace(12.0, 14.5, 6).tolist()))
    rng = np.random.default_rng(0)
    chunks = []
    for b in range(6):
        off = float(rng.normal(0.0, 0.15))
        for c in range(8):
            s = float(np.exp(off + rng.normal(0.0, 0.05)))
            chunks.append(ChunkSummary(
                box=f"set{b}", chunk_id=c, source="hr",
                n_sub=np.round(np.array([400., 160., 60., 22., 8., 3.]) * s),
                n_host=np.round(np.array([160., 54., 16., 4., 1.]) * s) + 1,
                occ_numerator=np.round(np.array([200., 90., 40., 14., 5.]) * s),
                volume_mpc3=1562.5))
    return fit_reward_model(chunks, bins,
                            active_dims=[i for i in range(11) if i != 10])


def _region_summary(cts, vol):
    return TorchSummary(
        n_sub=torch.as_tensor(cts["n_sub"]),
        n_host=torch.as_tensor(cts["n_host"]),
        occ_numerator=torch.as_tensor(cts["occ_numerator"]),
        volume_mpc3=torch.as_tensor(vol))


def test_candidate_equal_to_baseline_gives_exactly_zero_regional_reward(grid, counts):
    reward = TorchRewardModel.from_numpy(_reward_model())
    rg = RegionGrid(grid, 2)
    part = rg.partition((1, 0, 1))
    base_r = aggregate_tile_counts(part, {k: counts[k] for k in COUNTS})
    vol = aggregate_tile_volume(part, counts["volume_mpc3"])
    n_r = rg.n_regions
    box = TorchSummary(
        n_sub=torch.as_tensor(base_r["n_sub"].sum(0, keepdims=True).repeat(n_r, 0)),
        n_host=torch.as_tensor(base_r["n_host"].sum(0, keepdims=True).repeat(n_r, 0)),
        occ_numerator=torch.as_tensor(
            base_r["occ_numerator"].sum(0, keepdims=True).repeat(n_r, 0)),
        volume_mpc3=torch.as_tensor(np.full(n_r, vol.sum())))
    frozen = _region_summary(base_r, vol)
    d = reward.delta_reward_swap(box, frozen, frozen)   # candidate == baseline
    assert torch.allclose(d["dR_combined"],
                          torch.zeros(n_r, dtype=torch.float64), atol=1e-9)


def test_regional_swap_reward_equals_direct_recomputation(grid, counts):
    """dR from the batched region swap == R(box - region + cand) - R(box), by hand."""
    reward = TorchRewardModel.from_numpy(_reward_model())
    rng = np.random.default_rng(3)
    cand_tiles = {k: counts[k] * rng.uniform(0.5, 1.5, size=counts[k].shape)
                  for k in COUNTS}
    rg = RegionGrid(grid, 4)
    part = rg.partition((2, 1, 0))
    n_r = rg.n_regions
    base_r = aggregate_tile_counts(part, {k: counts[k] for k in COUNTS})
    cand_r = aggregate_tile_counts(part, cand_tiles)
    vol = aggregate_tile_volume(part, counts["volume_mpc3"])

    box = TorchSummary(
        n_sub=torch.as_tensor(base_r["n_sub"].sum(0, keepdims=True).repeat(n_r, 0)),
        n_host=torch.as_tensor(base_r["n_host"].sum(0, keepdims=True).repeat(n_r, 0)),
        occ_numerator=torch.as_tensor(
            base_r["occ_numerator"].sum(0, keepdims=True).repeat(n_r, 0)),
        volume_mpc3=torch.as_tensor(np.full(n_r, vol.sum())))
    frozen = _region_summary(base_r, vol)
    cand = _region_summary(cand_r, vol)
    d = reward.delta_reward_swap(box, frozen, cand)["dR_combined"].numpy()

    # Direct: for each region, form the mixed box explicitly and difference R.
    box_np = {k: base_r[k].sum(0) for k in COUNTS}
    for g in range(n_r):
        mixed = {k: np.clip(box_np[k] - base_r[k][g] + cand_r[k][g], 0.0, None)
                 for k in COUNTS}
        s_mixed = TorchSummary(
            n_sub=torch.as_tensor(mixed["n_sub"]).reshape(1, -1),
            n_host=torch.as_tensor(mixed["n_host"]).reshape(1, -1),
            occ_numerator=torch.as_tensor(mixed["occ_numerator"]).reshape(1, -1),
            volume_mpc3=torch.as_tensor([float(vol.sum())]))
        s_box = TorchSummary(
            n_sub=torch.as_tensor(box_np["n_sub"]).reshape(1, -1),
            n_host=torch.as_tensor(box_np["n_host"]).reshape(1, -1),
            occ_numerator=torch.as_tensor(box_np["occ_numerator"]).reshape(1, -1),
            volume_mpc3=torch.as_tensor([float(vol.sum())]))
        direct = float(reward.combined(s_mixed) - reward.combined(s_box))
        assert d[g] == pytest.approx(direct, abs=1e-9)


def test_mixing_baselines_visibly_changes_the_reward(grid, counts):
    """C_r from one seed with c_{r,g} removed from another is a different number.

    The consistent counterfactual removes the region contribution of the *same*
    baseline the box was pooled from; substituting a different baseline's region
    changes dR, which is why the diagnostic must never do it.
    """
    reward = TorchRewardModel.from_numpy(_reward_model())
    rng = np.random.default_rng(7)
    other_tiles = {k: counts[k] * rng.uniform(0.3, 1.7, size=counts[k].shape)
                   for k in COUNTS}
    cand_tiles = {k: counts[k] * rng.uniform(0.5, 1.5, size=counts[k].shape)
                  for k in COUNTS}
    rg = RegionGrid(grid, 2)
    part = rg.partition()
    n_r = rg.n_regions
    base_r = aggregate_tile_counts(part, {k: counts[k] for k in COUNTS})
    other_r = aggregate_tile_counts(part, other_tiles)
    cand_r = aggregate_tile_counts(part, cand_tiles)
    vol = aggregate_tile_volume(part, counts["volume_mpc3"])

    box = TorchSummary(
        n_sub=torch.as_tensor(base_r["n_sub"].sum(0, keepdims=True).repeat(n_r, 0)),
        n_host=torch.as_tensor(base_r["n_host"].sum(0, keepdims=True).repeat(n_r, 0)),
        occ_numerator=torch.as_tensor(
            base_r["occ_numerator"].sum(0, keepdims=True).repeat(n_r, 0)),
        volume_mpc3=torch.as_tensor(np.full(n_r, vol.sum())))
    consistent = reward.delta_reward_swap(
        box, _region_summary(base_r, vol), _region_summary(cand_r, vol))["dR_combined"]
    mixed = reward.delta_reward_swap(
        box, _region_summary(other_r, vol), _region_summary(cand_r, vol))["dR_combined"]
    assert not torch.allclose(consistent, mixed, atol=1e-6)


# --------------------------------------------------------------------------- #
# The reported metric is the proxy gate's within-group metric
# --------------------------------------------------------------------------- #
def test_reported_metric_matches_the_gate_within_group_metric():
    """The diagnostic and the proxy gate must score ordering the same way.

    Both go through ``spearman`` and ``tie_aware_agreement``; this pins that the
    gate helper ``_proxy_data.rank_metrics`` returns exactly those on one group.
    """
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts" / "reward"))
    from _proxy_data import rank_metrics
    from cosmo_sr.reward.catalog_proxy import spearman, tie_aware_agreement

    pred = np.array([0.3, 0.1, 0.2, 0.2, 0.9])
    true = np.array([1.0, 0.0, 2.0, 3.0, 4.0])
    box = np.array(["b"] * 5)
    tile = np.zeros(5, dtype=np.int64)          # one group

    m = rank_metrics(pred, true, box, tile)
    assert m["within_tile_spearman"] == pytest.approx(spearman(pred, true))
    acc, npair = tie_aware_agreement(pred, true)
    assert m["pairwise_accuracy"] == pytest.approx(acc)
    assert m["n_pairs"] == npair


# --------------------------------------------------------------------------- #
# The diagnostic's core: offset_metrics on synthetic labels
# --------------------------------------------------------------------------- #
def _load_region_diagnostic():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for p in (root / "src", root / "scripts" / "reward"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import region_attribution_diagnostic as rad
    return rad


def _synthetic_labels(grid, seeds, n_cands, touched_tiles, rng):
    """{seed: {'fractional': {counts}}} and a candidate list sharing untouched tiles."""
    base_seed0 = {
        "n_sub": rng.random((grid.n_tiles, 6)) * 5.0,
        "n_host": rng.random((grid.n_tiles, 5)) * 2.0 + 0.5,
        "occ_numerator": rng.random((grid.n_tiles, 5)) * 4.0,
        "volume_mpc3": np.full(grid.n_tiles, grid.tile_volume_mpc3),
    }
    frozen = {}
    for s in seeds:
        jitter = 1.0 + 0.05 * rng.standard_normal((grid.n_tiles, 1))
        frozen[s] = {"fractional": {
            "n_sub": base_seed0["n_sub"] * jitter,
            "n_host": base_seed0["n_host"] * jitter,
            "occ_numerator": base_seed0["occ_numerator"] * jitter,
            "volume_mpc3": base_seed0["volume_mpc3"].copy()}}
    cands = []
    for c in range(n_cands):
        cc = {k: base_seed0[k].copy() for k in base_seed0}   # == frozen seed0 field
        edit = 1.0 + (c + 1) * 0.1                            # alpha-ladder-like
        for k in ("n_sub", "n_host", "occ_numerator"):
            cc[k] = cc[k].copy()
            cc[k][touched_tiles] *= edit
        cands.append({"fractional": cc})
    return frozen, cands


def test_offset_metrics_shape_and_untouched_regions_are_not_informative():
    rad = _load_region_diagnostic()
    reward = TorchRewardModel.from_numpy(_reward_model())
    grid = TileGrid(ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=100.0)
    rng = np.random.default_rng(0)
    touched = np.zeros(grid.n_tiles, dtype=bool)
    touched[[0, 1, 8, 9]] = True                              # a corner block
    frozen, cands = _synthetic_labels(grid, [0, 1, 2], 3, touched, rng)

    rg = RegionGrid(grid, 2)
    m = rad.offset_metrics(reward, frozen, cands, touched, rg.partition(),
                           "fractional", w_joint=0.25, w_occ=1.0)
    assert m["n_regions"] == rg.n_regions
    assert 0 < m["n_touched_regions"] <= rg.n_regions
    # Informative regions are a subset of touched regions.
    assert m["n_informative_regions"] <= m["n_touched_regions"]
    assert m["n_nontied_candidate_pairs"] > 0
    for k in ("baseline_context_stability_spearman",
              "baseline_context_stability_pairwise",
              "regional_signal_to_noise", "pooled_cancellation_fraction"):
        assert k in m


def test_identical_baselines_give_perfect_context_stability():
    """If every frozen seed is the same field, the ordering cannot depend on it."""
    rad = _load_region_diagnostic()
    reward = TorchRewardModel.from_numpy(_reward_model())
    grid = TileGrid(ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=100.0)
    rng = np.random.default_rng(2)
    touched = np.zeros(grid.n_tiles, dtype=bool)
    touched[[0, 1, 8, 9, 64, 65]] = True
    frozen, cands = _synthetic_labels(grid, [0], 4, touched, rng)
    # Duplicate the single seed into three identical baselines.
    frozen = {s: frozen[0] for s in (0, 1, 2)}

    rg = RegionGrid(grid, 2)
    m = rad.offset_metrics(reward, frozen, cands, touched, rg.partition(),
                           "fractional", w_joint=0.25, w_occ=1.0)
    assert m["baseline_context_stability_spearman"] == pytest.approx(1.0)
    assert m["baseline_context_stability_pairwise"] == pytest.approx(1.0)


def test_pooled_cancellation_rises_from_tile_to_box():
    """Width 1 cancels nothing; wider regions absorb signed churn -> more cancels."""
    rad = _load_region_diagnostic()
    reward = TorchRewardModel.from_numpy(_reward_model())
    grid = TileGrid(ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=100.0)
    rng = np.random.default_rng(5)
    touched = np.zeros(grid.n_tiles, dtype=bool)
    touched[np.arange(0, grid.n_tiles, 7)] = True
    frozen, cands = _synthetic_labels(grid, [0, 1, 2, 3], 3, touched, rng)

    cancel = {}
    for w in (1, 2, 4, 8):
        rg = RegionGrid(grid, w)
        m = rad.offset_metrics(reward, frozen, cands, touched, rg.partition(),
                               "fractional", w_joint=0.25, w_occ=1.0)
        cancel[w] = m["pooled_cancellation_fraction"]
    assert cancel[1] == pytest.approx(0.0, abs=1e-9)          # region == tile
    assert cancel[8] >= cancel[4] >= cancel[2] >= cancel[1] - 1e-9
