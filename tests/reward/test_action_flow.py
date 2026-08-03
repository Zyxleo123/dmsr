"""The conditional action policy: shapes, conditioning, diversity, and weights.

These are CPU-sized checks on the parts that can be wrong without training
failing: a sampler that ignores its conditioning, a policy that collapses to one
action, or reward weights that concentrate the effective batch on a single
sample. All three look like "it trained fine" from the loss curve alone.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cosmo_sr.reward.action_flow import (  # noqa: E402
    ActionFlow, GaussianMixturePolicy, action_diversity, flow_matching_loss,
    host_features, reference_velocity_penalty, reward_weights, token_features,
)

D_A, D_C = 8, 14


def net(seed=0):
    torch.manual_seed(seed)
    return ActionFlow(D_A, D_C, width=32, depth=2)


def cond(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, D_C, generator=g)


# ---------------------------------------------------------------------------
# Conditioning features
# ---------------------------------------------------------------------------


def test_host_features_are_all_measurable_on_the_frozen_box():
    h = host_features({"mvir": 1e13, "rvir_mpc": 0.5, "vmax": 300.0,
                       "n_sub_current": 4, "smooth_fraction": 0.8,
                       "n_members": 17000})
    assert h.shape == (6,) and np.isfinite(h).all()
    assert h[0] == pytest.approx(13.0)


def test_token_features_carry_the_direction():
    c = token_features({"log_mass_ratio": -2.0, "radius_rvir": 0.4,
                        "direction": (0.0, 0.0, 1.0)})
    assert c.shape == (5,)
    assert c[4] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------


def test_weights_are_bounded_and_average_to_one():
    r = np.array([0.0, 1.0, 2.0, 50.0, -3.0, 0.5])
    h = np.array([1, 1, 1, 2, 2, 2])
    w = reward_weights(r, h, tau=0.5, w_max=10.0)
    assert np.all(w > 0) and np.isfinite(w).all()
    assert w.mean() == pytest.approx(1.0)
    assert w.max() / w.min() <= 10.0 ** 2 + 1e-6


def test_an_enormous_reward_cannot_take_over_the_batch():
    """Bounding only the top weight is not enough: with one reward far above the
    baseline every other sample underflows to zero and the effective batch size
    collapses to one anyway. The exponent is bounded symmetrically, so no sample
    is ever dropped."""
    r = np.array([0.0, 0.0, 0.0, 1000.0])
    h = np.zeros(4, dtype=int)
    w = reward_weights(r, h, tau=0.1, w_max=5.0)
    assert w.min() > 0.0, "a sample was weighted out of the batch entirely"
    assert w.max() / w.min() <= 5.0 ** 2 + 1e-6
    # Kish effective sample size. Unbounded weights give exactly 1.0 here (three
    # samples underflow to zero); bounding the exponent keeps all four alive.
    ess = float(w.sum() ** 2 / np.sum(w ** 2))
    assert ess > 1.2, ess


def test_the_baseline_is_per_host():
    """A host that is simply easier must not dominate: an action that is average
    for its own host gets weight 1 whatever that host's absolute reward is."""
    r = np.array([5.0, 5.0, 0.0, 0.0])
    h = np.array([1, 1, 2, 2])
    w = reward_weights(r, h, tau=0.5)
    assert np.allclose(w, 1.0)


def test_a_single_sample_host_gets_weight_one():
    w = reward_weights([3.0], [7], tau=0.5)
    assert w[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------


def test_samples_have_the_action_shape():
    s = net().sample(cond(11), n_steps=8)
    assert s.shape == (11, D_A) and torch.isfinite(s).all()


def test_the_loss_is_finite_and_weightable():
    n = net()
    a = torch.randn(16, D_A)
    c = cond(16)
    l0, info = flow_matching_loss(n, a, c)
    l1, _ = flow_matching_loss(n, a, c, torch.full((16,), 2.0))
    assert torch.isfinite(l0) and torch.isfinite(l1)
    assert "unweighted_cfm" in info


def test_a_trained_flow_follows_its_conditioning():
    """Two conditioning groups whose target actions differ; the sampler must put
    them in different places. A flow that ignores ``cond`` still drives the loss
    down by learning the pooled marginal."""
    torch.manual_seed(0)
    n_per = 256
    c = torch.cat([torch.zeros(n_per, D_C), torch.ones(n_per, D_C)])
    a = torch.cat([torch.full((n_per, D_A), -2.0),
                   torch.full((n_per, D_A), 2.0)]) + 0.1 * torch.randn(2 * n_per, D_A)
    m = ActionFlow(D_A, D_C, width=64, depth=2)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(800):
        i = torch.randint(0, a.shape[0], (128,))
        loss, _ = flow_matching_loss(m, a[i], c[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    lo = m.sample(torch.zeros(64, D_C), n_steps=64).mean().item()
    hi = m.sample(torch.ones(64, D_C), n_steps=64).mean().item()
    assert lo < -1.0 < 1.0 < hi, (lo, hi)


def test_a_trained_flow_does_not_collapse_to_one_action():
    torch.manual_seed(1)
    a = torch.randn(512, D_A) * 1.5
    c = torch.zeros(512, D_C)
    m = ActionFlow(D_A, D_C, width=64, depth=2)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(600):
        i = torch.randint(0, 512, (128,))
        loss, _ = flow_matching_loss(m, a[i], c[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    s = m.sample(torch.zeros(256, D_C), n_steps=64).numpy()
    d = action_diversity(s)
    assert d["min_std"] > 0.3, d


def test_the_reference_penalty_is_zero_against_an_identical_copy():
    import copy
    n = net()
    ref = copy.deepcopy(n)
    p = reference_velocity_penalty(n, ref, torch.randn(16, D_A), cond(16))
    assert float(p.detach()) == pytest.approx(0.0, abs=1e-12)


def test_the_reference_penalty_is_positive_against_a_different_network():
    p = reference_velocity_penalty(net(0), net(1), torch.randn(16, D_A), cond(16))
    assert float(p.detach()) > 0


# ---------------------------------------------------------------------------
# The mandatory baseline
# ---------------------------------------------------------------------------


def test_the_mixture_samples_and_scores_with_the_right_shapes():
    torch.manual_seed(0)
    g = GaussianMixturePolicy(D_A, D_C, n_components=3, width=32)
    c = cond(9)
    assert g.sample(c).shape == (9, D_A)
    assert g.log_prob(torch.randn(9, D_A), c).shape == (9,)
    assert torch.isfinite(g.loss(torch.randn(9, D_A), c)).item()


def test_the_mixture_can_represent_two_modes():
    """The baseline has to be genuinely multimodal, or "the flow covers more
    modes" would be a comparison against a straw man."""
    torch.manual_seed(2)
    a = torch.cat([torch.full((256, D_A), -3.0), torch.full((256, D_A), 3.0)]) \
        + 0.2 * torch.randn(512, D_A)
    c = torch.zeros(512, D_C)
    g = GaussianMixturePolicy(D_A, D_C, n_components=4, width=64)
    opt = torch.optim.Adam(g.parameters(), lr=5e-3)
    for _ in range(800):
        i = torch.randint(0, 512, (128,))
        loss = g.loss(a[i], c[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    s = g.sample(torch.zeros(512, D_C)).numpy()[:, 0]
    assert (s < -1.5).sum() > 50 and (s > 1.5).sum() > 50, "one mode was dropped"


def test_diversity_detects_a_collapsed_coordinate():
    s = np.random.default_rng(0).normal(size=(64, D_A))
    s[:, 3] = 1.234                     # one coordinate frozen
    d = action_diversity(s)
    assert d["min_std"] == pytest.approx(0.0, abs=1e-9)
    assert d["mean_std"] > 0.5
