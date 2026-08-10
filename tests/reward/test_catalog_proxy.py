"""The proxy's contract: nonnegative counts, honest splits, usable gradients."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmo_sr.reward.catalog import CatalogBins, ChunkSummary, pool
from cosmo_sr.reward.catalog_proxy import (
    CatalogProxy, ProxyConfig, ProxyEnsemble, bin_weights_from_counts, count_loss,
    make_within_tile_pairs, pairwise_ranking_loss, spearman, split_indices_by_box,
    tie_aware_agreement,
)
from cosmo_sr.reward.reward import fit_reward_model
from cosmo_sr.reward.torch_reward import (
    TorchRewardModel, TorchSummary, summary_from_ensemble,
)


@pytest.fixture
def cfg() -> ProxyConfig:
    return ProxyConfig(n_features=8, n_sub_bins=6, n_host_bins=5, hidden=(16,))


@pytest.fixture
def proxy(cfg) -> CatalogProxy:
    return CatalogProxy(cfg, seed=0)


def test_output_shapes_and_nonnegativity(proxy, cfg):
    torch.manual_seed(0)
    x = torch.randn(7, cfg.n_features, dtype=torch.float64) * 10.0
    out = proxy(x)
    assert set(out) == {"n_sub", "n_host", "occ_numerator"}
    assert out["n_sub"].shape == (7, cfg.n_sub_bins)
    assert out["n_host"].shape == (7, cfg.n_host_bins)
    assert out["occ_numerator"].shape == (7, cfg.n_host_bins)
    for v in out.values():
        assert bool((v >= 0).all()), "softplus outputs must be nonnegative"
        assert torch.isfinite(v).all()


def test_extreme_inputs_stay_nonnegative_and_finite(proxy, cfg):
    x = torch.full((3, cfg.n_features), 1e6, dtype=torch.float64)
    x[1] = -1e6
    out = proxy(x)
    for v in out.values():
        assert bool((v >= 0).all())
        assert torch.isfinite(v).all()


def test_output_scale_sets_the_magnitude(cfg):
    scale = [100.0] * cfg.n_outputs
    a = CatalogProxy(cfg, seed=1)
    b = CatalogProxy(ProxyConfig(**{**cfg.to_dict(), "output_scale": scale,
                                    "hidden": tuple(cfg.hidden)}), seed=1)
    x = torch.zeros(2, cfg.n_features, dtype=torch.float64)
    assert torch.allclose(b(x)["n_sub"], 100.0 * a(x)["n_sub"], rtol=1e-9)


def test_standardizer_is_fit_and_travels_with_the_checkpoint(tmp_path, cfg):
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, size=(64, cfg.n_features))
    x[:, 0] = 7.0                                  # a constant column
    p = CatalogProxy(cfg, seed=2).fit_standardizer(x)
    assert float(p.feat_std[0]) == pytest.approx(1.0)     # not zero, not inf
    assert float(p.feat_mean[1]) == pytest.approx(x[:, 1].mean(), rel=1e-9)
    path = p.save(tmp_path / "m.pt")
    q = CatalogProxy.load(path)
    assert torch.allclose(q.feat_mean, p.feat_mean)
    assert torch.allclose(q.feat_std, p.feat_std)
    t = torch.as_tensor(x, dtype=torch.float64)
    assert torch.allclose(q(t)["n_host"], p(t)["n_host"])


# --------------------------------------------------------------------------- #
# Splits and pairs
# --------------------------------------------------------------------------- #
def test_box_grouped_split_is_disjoint():
    boxes = ["set0", "set0", "set1", "set8", "set9", "set12"]
    out = split_indices_by_box(boxes, ["set0", "set1"], ["set8", "set9"])
    assert out["train"].tolist() == [0, 1, 2]
    assert out["val"].tolist() == [3, 4]
    assert set(out["dropped_boxes"].tolist()) == {"set12"}
    assert not (set(out["train"].tolist()) & set(out["val"].tolist()))


def test_overlapping_split_is_refused():
    with pytest.raises(ValueError, match="both the train and validation split"):
        split_indices_by_box(["set0"], ["set0"], ["set0"])


def test_pairs_are_formed_only_within_box_and_tile():
    box = ["a", "a", "a", "b", "b"]
    tile = [0, 0, 1, 0, 0]
    target = [1.0, 2.0, 3.0, 0.5, 4.0]
    pairs = make_within_tile_pairs(box, tile, target, max_pairs_per_group=100)
    for i, j in pairs:
        assert (box[i], tile[i]) == (box[j], tile[j])
        assert target[i] > target[j], "pair[:, 0] must be the better row"
    # (a,0) gives one pair, (b,0) gives one, (a,1) is a singleton and gives none.
    assert len(pairs) == 2


def test_pairs_drop_ties_below_the_margin():
    box = ["a"] * 4
    tile = [0] * 4
    target = [1.0, 1.0001, 5.0, 5.0001]
    loose = make_within_tile_pairs(box, tile, target, max_pairs_per_group=100)
    tight = make_within_tile_pairs(box, tile, target, max_pairs_per_group=100,
                                   min_margin=0.5)
    assert len(loose) == 6
    assert len(tight) == 4          # the two near-ties are gone


def test_pairs_skip_non_finite_targets():
    pairs = make_within_tile_pairs(["a"] * 3, [0] * 3, [1.0, float("nan"), 2.0],
                                   max_pairs_per_group=100)
    assert len(pairs) == 1


def test_ranking_loss_prefers_higher_scores():
    better = torch.tensor([2.0, 3.0])
    worse = torch.tensor([1.0, 0.0])
    assert float(pairwise_ranking_loss(better, worse)) < float(
        pairwise_ranking_loss(worse, better))


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def test_bin_weights_lift_rare_bins_but_are_capped():
    targets = np.zeros((100, 3))
    targets[:, 0] = 1300.0        # abundant
    targets[:, 1] = 130.0
    targets[:, 2] = 3.0           # rare
    w = bin_weights_from_counts(targets, cap=8.0, floor=0.25)
    assert w[2] > w[1] > w[0]
    assert w.max() <= 8.0 and w.min() >= 0.25


def test_an_identically_zero_bin_gets_the_floor_not_the_cap():
    """Absent is not rare: a constant-zero target has nothing to protect."""
    targets = np.zeros((10, 2))
    targets[:, 0] = 5.0
    w = bin_weights_from_counts(targets, cap=4.0, floor=0.25)
    assert np.isfinite(w).all()
    assert w[1] == pytest.approx(0.25)


def test_count_loss_is_zero_on_a_perfect_prediction():
    t = {"n_sub": torch.rand(4, 6) * 10, "n_host": torch.rand(4, 5) * 3,
         "occ_numerator": torch.rand(4, 5) * 6}
    out = count_loss(t, t)
    assert float(out["loss"]) == pytest.approx(0.0, abs=1e-12)
    for k in ("loss_n_sub", "loss_n_host", "loss_occ_numerator"):
        assert float(out[k]) == pytest.approx(0.0, abs=1e-12)


def test_count_loss_grows_with_error_and_accepts_weights():
    t = {"n_sub": torch.full((2, 6), 4.0), "n_host": torch.full((2, 5), 2.0),
         "occ_numerator": torch.full((2, 5), 3.0)}
    p = {k: v * 3.0 for k, v in t.items()}
    small = count_loss({k: v * 1.1 for k, v in t.items()}, t)["loss"]
    big = count_loss(p, t)["loss"]
    assert float(big) > float(small)
    w = torch.ones(16)
    w[6] = 10.0
    assert torch.isfinite(count_loss(p, t, weights=w)["loss"])


def test_count_loss_rejects_mismatched_weights():
    t = {"n_sub": torch.ones(1, 6), "n_host": torch.ones(1, 5),
         "occ_numerator": torch.ones(1, 5)}
    with pytest.raises(ValueError, match="weights has"):
        count_loss(t, t, weights=torch.ones(3))


# --------------------------------------------------------------------------- #
# Ensemble
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
                            active_dims=[i for i in range(11) if i != 10]), pool(chunks)


def test_q_safe_penalises_disagreement():
    per_member = torch.tensor([[1.0, 1.0], [1.0, 3.0], [1.0, -1.0]])
    out = ProxyEnsemble.q_safe(per_member, beta=1.0)
    assert float(out["mean"][0]) == pytest.approx(1.0)
    assert float(out["std"][0]) == pytest.approx(0.0)
    assert float(out["q_safe"][0]) == pytest.approx(1.0)
    # Same mean, wide spread -> a strictly lower safe value.
    assert float(out["q_safe"][1]) < float(out["q_safe"][0])


def test_q_safe_beta_zero_is_the_plain_mean():
    per_member = torch.tensor([[0.0, 5.0], [4.0, -5.0]])
    out = ProxyEnsemble.q_safe(per_member, beta=0.0)
    assert torch.allclose(out["q_safe"], per_member.mean(dim=0))


def test_ensemble_is_frozen_but_still_differentiable(cfg):
    model, ens = _reward_model()
    reward = TorchRewardModel.from_numpy(model)
    proxies = ProxyEnsemble([CatalogProxy(cfg, seed=s) for s in range(3)]).freeze()
    assert all(not p.requires_grad for p in proxies.parameters())

    box = summary_from_ensemble(ens)
    frozen_tile = TorchSummary(
        n_sub=box.n_sub * 0.02, n_host=box.n_host * 0.02,
        occ_numerator=box.occ_numerator * 0.02, volume_mpc3=box.volume_mpc3)
    feats = torch.zeros(1, cfg.n_features, dtype=torch.float64, requires_grad=True)

    per_member = proxies.delta_rewards(reward, feats, box, frozen_tile)
    assert per_member.shape == (3, 1)
    q = ProxyEnsemble.q_safe(per_member, beta=1.0)["q_safe"]
    q.sum().backward()
    # Frozen parameters, live gradient: the configuration actor training needs.
    assert feats.grad is not None
    assert torch.isfinite(feats.grad).all()
    assert float(feats.grad.abs().sum()) > 0.0
    assert all(p.grad is None for p in proxies.parameters())


def test_delta_rewards_all_returns_every_reward(cfg):
    model, ens = _reward_model()
    reward = TorchRewardModel.from_numpy(model)
    proxies = ProxyEnsemble([CatalogProxy(cfg, seed=s) for s in range(2)]).freeze()
    box = summary_from_ensemble(ens)
    tile = TorchSummary(n_sub=box.n_sub * 0.02, n_host=box.n_host * 0.02,
                        occ_numerator=box.occ_numerator * 0.02,
                        volume_mpc3=box.volume_mpc3)
    out = proxies.delta_rewards_all(
        reward, torch.zeros(1, cfg.n_features, dtype=torch.float64), box, tile)
    for key in ("dR_cat", "dR_occ", "dR_abund", "dR_combined"):
        assert out[key].shape == (2, 1)


def test_ensemble_round_trip(tmp_path, cfg):
    proxies = ProxyEnsemble([CatalogProxy(cfg, seed=s) for s in range(3)])
    proxies.save(tmp_path / "ens")
    back = ProxyEnsemble.load(tmp_path / "ens")
    assert len(back) == 3
    x = torch.randn(2, cfg.n_features, dtype=torch.float64)
    for a, b in zip(proxies.members, back.members):
        assert torch.allclose(a(x)["n_sub"], b(x)["n_sub"])


def test_members_with_different_seeds_disagree(cfg):
    a, b = CatalogProxy(cfg, seed=0), CatalogProxy(cfg, seed=1)
    x = torch.randn(4, cfg.n_features, dtype=torch.float64)
    assert not torch.allclose(a(x)["n_sub"], b(x)["n_sub"])


def test_spearman_basics():
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1.0], [2.0]))


def test_spearman_averages_tied_ranks():
    """A tie must get the average rank, not an invented order from argsort."""
    # Two identical rankings with a tie block correlate perfectly.
    assert spearman([1, 1, 2, 2, 3], [1, 1, 2, 2, 3]) == pytest.approx(1.0)
    # Against SciPy's definition on tied data (computed by hand here):
    # a = [1, 2, 2, 3] -> avg ranks [0, 1.5, 1.5, 3]; b = [10, 5, 5, 1] reversed.
    assert spearman([1, 2, 2, 3], [10, 5, 5, 1]) == pytest.approx(-1.0)
    # The old double-argsort gave a spurious non-unit value for tied inputs; the
    # averaged-rank form is exactly +/-1 when the tie structure matches.


def test_spearman_is_nan_for_a_constant_input():
    """A constant vector has no ranking, so the correlation is undefined, not 0.

    The double-argsort implementation returned a finite number here, which let an
    unrankable (all-equal) prediction score as 'no worse than chance'.
    """
    assert np.isnan(spearman([5.0, 5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0]))
    assert np.isnan(spearman([1.0, 2.0, 3.0, 4.0], [7.0, 7.0, 7.0, 7.0]))


def test_pairs_drop_exact_ties_even_at_zero_margin():
    """min_margin=0 must still discard equal-target pairs.

    ``abs(gap) < 0`` is never true, so at zero margin an exact tie used to slip
    through and train the proxy to prefer one of two identical rewards at random.
    """
    box = ["a"] * 3
    tile = [0] * 3
    target = [1.0, 1.0, 2.0]          # the (1.0, 1.0) pair is an exact tie
    pairs = make_within_tile_pairs(box, tile, target, max_pairs_per_group=100,
                                   min_margin=0.0)
    assert len(pairs) == 2            # only the two distinct pairs survive
    for i, j in pairs:
        assert target[i] != target[j]


def test_tie_aware_agreement_scores_prediction_ties_as_half():
    # True order descending, prediction a tie: the naive == would call this
    # 'correct'. It is a coin flip, so it scores 0.5.
    acc, n = tie_aware_agreement([1.0, 1.0], [2.0, 1.0])
    assert n == 1 and acc == pytest.approx(0.5)
    # A correct strict ordering scores 1, a wrong one 0.
    assert tie_aware_agreement([2.0, 1.0], [2.0, 1.0]) == (1.0, 1)
    assert tie_aware_agreement([1.0, 2.0], [2.0, 1.0]) == (0.0, 1)
    # Equal-*true* pairs are not counted at all.
    acc2, n2 = tie_aware_agreement([1.0, 2.0], [3.0, 3.0])
    assert n2 == 0 and np.isnan(acc2)
