"""Arms D/E/F, the frozen-relative residual head, and the objective fixes.

Unit tests only -- no cluster, no real Rockstar. They pin the properties the
six-arm comparison rests on: the spatial arms respond to layout where the
DeepSets arm cannot, arm E's cache matches its live extractor, arm F has exactly
the SR2 critic's input/output shape, every arm's residual head starts at frozen,
and the pair/pooling fixes behave as claimed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "reward"))

from cosmo_sr.reward.catalog_proxy import (  # noqa: E402
    CatalogProxy, ProxyConfig, load_proxy, make_within_tile_pairs,
    PAIR_RANDOM,
)
from cosmo_sr.reward.soft_rockstar import (  # noqa: E402
    SoftRockstarProxy, SoftRockstarProxyConfig,
)
from cosmo_sr.reward.spatial_proxy import (  # noqa: E402
    FullGridProxy, FullGridProxyConfig, SR2DiscriminatorProxy,
    SR2DiscriminatorProxyConfig, SpatialTokenProxy, SpatialTokenProxyConfig,
)
from cosmo_sr.reward.torch_reward import TorchSummary  # noqa: E402

J, I = 6, 5


def _frozen(b: int) -> TorchSummary:
    g = torch.Generator().manual_seed(0)
    return TorchSummary(
        torch.rand(b, J, generator=g).double() * 5 + 0.5,
        torch.rand(b, I, generator=g).double() * 3 + 0.5,
        torch.rand(b, I, generator=g).double() * 2 + 0.2,
        torch.ones(b).double() * 1e5)


def _all_models():
    return {
        "a": (CatalogProxy(ProxyConfig(n_features=44), seed=0), torch.randn(3, 44).double()),
        "c": (SoftRockstarProxy(SoftRockstarProxyConfig(n_token_features=8, n_tokens=512), seed=0),
              torch.randn(3, 2, 512, 8).double()),
        "d": (SpatialTokenProxy(SpatialTokenProxyConfig(n_token_features=8, n_tokens=512), seed=0),
              torch.randn(3, 2, 512, 8).double()),
        "e": (FullGridProxy(FullGridProxyConfig(), seed=0),
              torch.randn(3, 2, 5, 32, 32, 32).double()),
        "f": (SR2DiscriminatorProxy(SR2DiscriminatorProxyConfig(), seed=0),
              torch.randn(3, 20, 64, 64, 64).double()),
    }


# --------------------------------------------------------------------------- #
# The residual head
# --------------------------------------------------------------------------- #
def test_zero_residual_reconstructs_frozen_for_every_arm():
    """A zero-initialised residual head predicts exactly the frozen summary."""
    frozen = _frozen(3)
    for arm, (m, x) in _all_models().items():
        m.eval()
        out = m.summary(x, frozen.volume_mpc3, frozen)
        assert torch.allclose(out.n_sub, frozen.n_sub, atol=1e-9), arm
        assert torch.allclose(out.n_host, frozen.n_host, atol=1e-9), arm
        assert torch.allclose(out.occ_numerator, frozen.occ_numerator, atol=1e-9), arm


def test_absolute_count_checkpoint_still_loads(tmp_path):
    """A pre-residual blob (no residual_head key) loads with the absolute head."""
    m = CatalogProxy(ProxyConfig(n_features=44, output_scale=[1.0] * 16), seed=1)
    p = tmp_path / "old.pt"
    m.save(p)
    blob = torch.load(p, weights_only=False)
    cfg = dict(blob["config"]); cfg.pop("residual_head")     # simulate old blob
    blob["config"] = cfg
    torch.save(blob, p)
    loaded = load_proxy(p)
    assert loaded.residual_head is False
    # The absolute head ignores frozen and uses softplus*scale -> finite counts.
    out = loaded.summary(torch.randn(2, 44).double(), torch.ones(2).double() * 1e5)
    assert torch.isfinite(out.n_sub).all()


def test_all_arms_save_load_roundtrip(tmp_path):
    frozen = _frozen(2)
    for arm, (m, x) in _all_models().items():
        p = tmp_path / f"{arm}.pt"
        m.save(p)
        m2 = load_proxy(p)
        assert type(m2).__name__ == type(m).__name__
        assert m2.residual_head is True
        m.eval(); m2.eval()
        a = m.summary(x[:2], frozen.volume_mpc3, frozen).n_sub
        b = m2.summary(x[:2], frozen.volume_mpc3, frozen).n_sub
        assert torch.allclose(a, b), arm


# --------------------------------------------------------------------------- #
# Layout: arm D sees it, arm C does not
# --------------------------------------------------------------------------- #
def test_d_responds_to_token_shuffle_while_c_is_invariant():
    """C pools permutation-invariantly; D convolves the ordered grid."""
    tokens = torch.randn(2, 2, 512, 8).double()
    perm = torch.randperm(512, generator=torch.Generator().manual_seed(1))
    shuffled = tokens[:, :, perm, :]
    frozen = _frozen(2)

    c = SoftRockstarProxy(SoftRockstarProxyConfig(n_token_features=8, n_tokens=512), seed=0)
    c.fit_standardizer(tokens.numpy()); c.eval()
    c0 = c.summary(tokens, frozen.volume_mpc3, frozen).n_sub
    c1 = c.summary(shuffled, frozen.volume_mpc3, frozen).n_sub
    assert torch.allclose(c0, c1, atol=1e-8), "arm C must be permutation-invariant"

    d = SpatialTokenProxy(SpatialTokenProxyConfig(n_token_features=8, n_tokens=512), seed=0)
    d.fit_standardizer(tokens.numpy())
    # Break the zero-init so the conv actually reads the layout.
    torch.nn.init.normal_(d.head.weight, std=0.2)
    d.eval()
    d0 = d.summary(tokens, frozen.volume_mpc3, frozen).n_sub
    d1 = d.summary(shuffled, frozen.volume_mpc3, frozen).n_sub
    assert not torch.allclose(d0, d1, atol=1e-6), "arm D must depend on token layout"


# --------------------------------------------------------------------------- #
# Arm E: cache matches live extractor
# --------------------------------------------------------------------------- #
def test_e_cached_float16_matches_live_extractor():
    from cosmo_sr.reward.phase_space import phase_space_paired_grid
    from cosmo_sr.reward.soft_structure import SoftStructureConfig
    from cosmo_sr.reward.phase_space import PhaseSpaceConfig

    torch.manual_seed(0)
    cand = torch.randn(2, 6, 64, 64, 64) * 0.1
    froz = torch.randn(2, 6, 64, 64, 64) * 0.1
    live = phase_space_paired_grid(cand, froz, SoftStructureConfig(), PhaseSpaceConfig())
    assert live.shape == (2, 2, 5, 32, 32, 32)
    cached = live.to(torch.float16).float()          # what the sidecar stores
    err = (live.float() - cached).abs().max().item()
    scale = live.abs().max().item()
    assert err <= 1e-3 * max(scale, 1.0) + 1e-3


# --------------------------------------------------------------------------- #
# Arm F: exact SR2 critic input/output shape
# --------------------------------------------------------------------------- #
def test_f_has_20_input_channels_and_16_outputs():
    m = SR2DiscriminatorProxy(SR2DiscriminatorProxyConfig())
    assert m.in_chan == 20
    assert m.cfg.n_outputs == 16
    frozen = _frozen(2)
    out = m.summary(torch.randn(2, 20, 64, 64, 64).double(), frozen.volume_mpc3, frozen)
    assert out.n_sub.shape[1] + out.n_host.shape[1] + out.occ_numerator.shape[1] == 16
    with pytest.raises(ValueError, match="20"):
        m(torch.randn(2, 12, 64, 64, 64).double(), frozen)


def test_f_matches_critic_input_contract():
    """Arm F consumes the exact 20-channel critic_input built from LR + field."""
    from cosmo_sr.reward.sr2_adversarial import critic_input

    lr = torch.randn(2, 6, 8, 8, 8) * 0.1
    field = torch.randn(2, 6, 64, 64, 64) * 0.1
    ci = critic_input(lr, field, cellsize_kpc_h=195.3, grid_mult=2)
    assert ci.shape == (2, 20, 64, 64, 64)
    m = SR2DiscriminatorProxy(SR2DiscriminatorProxyConfig())
    frozen = _frozen(2)
    out = m.summary(ci.double(), frozen.volume_mpc3, frozen)
    assert torch.isfinite(out.n_sub).all()


# --------------------------------------------------------------------------- #
# Gradients reach the candidate through D/E/F
# --------------------------------------------------------------------------- #
def test_candidate_gradients_finite_and_nonzero_for_spatial_arms():
    frozen = _frozen(2)
    cases = {
        "d": (SpatialTokenProxy(SpatialTokenProxyConfig(n_token_features=8, n_tokens=512), seed=0),
              torch.randn(2, 2, 512, 8).double()),
        "e": (FullGridProxy(FullGridProxyConfig(), seed=0),
              torch.randn(2, 2, 5, 32, 32, 32).double()),
        "f": (SR2DiscriminatorProxy(SR2DiscriminatorProxyConfig(), seed=0),
              torch.randn(2, 20, 64, 64, 64).double()),
    }
    for arm, (m, x) in cases.items():
        torch.nn.init.normal_(m.head.weight, std=0.2)    # nonzero residual
        x = x.clone().requires_grad_(True)
        out = m(x, frozen)
        (out["n_sub"].sum() + out["occ_numerator"].sum()).backward()
        assert torch.isfinite(x.grad).all(), arm
        assert float(x.grad.abs().sum()) > 0, arm


def test_e_extractor_field_gradient_is_finite():
    """The dense grid's per-cell sqrt must not send a NaN gradient to the field."""
    from cosmo_sr.reward.phase_space import phase_space_grid
    from cosmo_sr.reward.soft_structure import SoftStructureConfig
    from cosmo_sr.reward.phase_space import PhaseSpaceConfig

    torch.manual_seed(0)
    cand = (torch.randn(2, 6, 64, 64, 64) * 0.1).requires_grad_(True)
    g = phase_space_grid(cand[:, 0:3], cand[:, 3:6], SoftStructureConfig(),
                         PhaseSpaceConfig())
    g.sum().backward()
    assert torch.isfinite(cand.grad).all()
    assert float(cand.grad.abs().sum()) > 0


# --------------------------------------------------------------------------- #
# Objective fixes: pairs, weights, pooling
# --------------------------------------------------------------------------- #
def test_priority_pairs_are_built_before_random_fill():
    from cosmo_sr.reward.catalog_proxy import (
        PAIR_FROZEN_VS_INTERVENTION, PAIR_ADJACENT_ALPHA)

    # One (box, unit) group: a frozen tile and an alpha ladder, plus noise rows.
    box = np.array(["set0"] * 6)
    tile = np.zeros(6, dtype=int)
    source = np.array(["frozen", "intervention", "intervention", "intervention",
                       "frozen_seed", "hr"])
    alpha = np.array([np.nan, 0.25, 0.5, 1.0, np.nan, np.nan])
    mode = np.array(["both"] * 6)
    target = np.array([0.0, 0.1, 0.2, 0.4, 0.05, 0.9])

    pairs, kinds = make_within_tile_pairs(
        box, tile, target, source=source, alpha=alpha, mode=mode,
        max_pairs_per_group=32, rng=np.random.default_rng(0), return_kinds=True)
    # Priority kinds (frozen-vs-intervention, adjacent-alpha) must all precede any
    # random-fill pair.
    first_random = next((k for k, kind in enumerate(kinds) if kind == PAIR_RANDOM),
                        len(kinds))
    assert set(kinds[:first_random]) <= {PAIR_FROZEN_VS_INTERVENTION, PAIR_ADJACENT_ALPHA}
    # Frozen (row 0) is paired against each of the three interventions.
    fvi = [tuple(p) for p, kk in zip(pairs, kinds) if kk == PAIR_FROZEN_VS_INTERVENTION]
    assert len(fvi) == 3


def test_changed_pairs_retain_the_3x_weight():
    from _proxy_data import pair_weights

    pairs = np.array([[0, 1], [2, 3]])       # first pair changed, second not
    kinds = np.array([0, 0])
    mult = np.ones(4)
    changed = np.array([True, False, False, False])   # pair 0 touches a changed row
    w = pair_weights(pairs, kinds, mult, changed, type_weights={0: 1.0},
                     changed_weight=3.0)
    assert np.isclose(w[0], 3.0) and np.isclose(w[1], 1.0)


def test_pair_bootstrap_zero_excludes_the_pair():
    from _proxy_data import pair_weights

    pairs = np.array([[0, 1]])
    w = pair_weights(pairs, np.array([2]), mult=np.array([0.0, 0.0]),
                     changed=np.array([True, True]), type_weights={2: 1.0})
    assert float(w[0]) == 0.0


def test_per_candidate_pooling_prevents_cross_candidate_cancellation():
    from _proxy_data import pooled_count_error, pooled_count_error_by_candidate

    box = np.array(["set0"] * 4)
    tag = np.array(["A", "A", "B", "B"])
    true = {"n_sub": np.full((4, 1), 10.0), "n_host": np.full((4, 1), 5.0),
            "occ_numerator": np.full((4, 1), 2.0)}
    # A over-predicts to 15, B under to 5 -> the all-row sum cancels exactly.
    pred = {"n_sub": np.array([[15.], [15.], [5.], [5.]]),
            "n_host": np.full((4, 1), 5.0), "occ_numerator": np.full((4, 1), 2.0)}
    allrow = pooled_count_error(pred, true)
    percand = pooled_count_error_by_candidate(pred, true, box, tag)
    assert allrow["n_sub_log_error"][0] < 1e-9         # cancels
    assert percand["mean_log_error"] > 0.05            # does not
    assert percand["n_candidates"] == 2 and percand["n_boxes"] == 1
