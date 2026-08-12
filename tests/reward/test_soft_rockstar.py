"""Arm C: the soft-Rockstar token grid and the DeepSets proxy on top of it.

Two families of claims. The FEATURES must be a faithful, differentiable token
decomposition of the crop -- shared deposit with arm B, mass-weighted so empty
tokens cannot shout, and stable when the field is stable. The MODEL must be
permutation-invariant over tokens (that is the entire point of DeepSets; a
model that secretly reads the token index has learned the grid, not the
structure) and must round-trip through the same ensemble serialization arms A
and B use, because the training and gating scripts cannot special-case it.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmo_sr.reward.arms import ARMS, arm_storage, sidecar_arms
from cosmo_sr.reward.catalog_proxy import ProxyEnsemble, load_proxy
from cosmo_sr.reward.phase_space import PhaseSpaceConfig, deposit_phase_space
from cosmo_sr.reward.soft_rockstar import (
    SoftRockstarConfig, SoftRockstarProxy, SoftRockstarProxyConfig,
    paired_token_feature_names, soft_rockstar_paired_tokens,
    soft_rockstar_tokens, token_feature_names,
)
from cosmo_sr.reward.soft_structure import SoftStructureConfig


def _field(n=24, seed=0, scale=0.02):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 6, n, n, n, generator=g) * scale


SMALL = SoftRockstarConfig(tokens_per_axis=4)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def test_token_shapes_and_names_agree():
    f = _field(24, 1)
    tok = soft_rockstar_tokens(f, None, None, SMALL)
    names = token_feature_names(SMALL)
    assert tok.shape == (2, 4 ** 3, len(names))
    paired = soft_rockstar_paired_tokens(f, _field(24, 2), None, None, SMALL)
    assert paired.shape == (2, 2, 4 ** 3, len(names))
    assert paired_token_feature_names(SMALL) == list(names) + [
        f"d_{n}" for n in names]


def test_paired_second_block_is_the_difference():
    """Same slot convention as the flat arms: [candidate, candidate - frozen].

    And a candidate paired with ITSELF must have an exactly-zero difference
    block -- that is what makes a frozen row a ranking tie rather than noise.
    """
    cand, base = _field(24, 3), _field(24, 4)
    paired = soft_rockstar_paired_tokens(cand, base, None, None, SMALL)
    tc = soft_rockstar_tokens(cand, None, None, SMALL)
    tb = soft_rockstar_tokens(base, None, None, SMALL)
    assert torch.allclose(paired[:, 0], tc)
    assert torch.allclose(paired[:, 1], tc - tb)
    self_paired = soft_rockstar_paired_tokens(cand, cand, None, None, SMALL)
    assert torch.equal(self_paired[:, 1], torch.zeros_like(self_paired[:, 1]))


def test_tokens_share_the_deposit_with_arm_b():
    """The token mass column must sum to what ONE deposit put down.

    Arms B and C both derive from :func:`deposit_phase_space`; if C ran its own
    deposit the two arms could disagree about which particles landed where and
    the arm comparison would be confounded by the deposit, not the features.
    """
    f = _field(24, 5)
    m, _, _ = deposit_phase_space(f[:, 0:3], f[:, 3:6])
    tok = soft_rockstar_tokens(f, None, None, SMALL)
    # Column 0 is log1p of mean token mass; invert and compare against the
    # block means of the deposit's own mass grid.
    p = m.shape[-1] // 4
    block_mean = torch.nn.functional.avg_pool3d(m, p).flatten(1)
    got = torch.exp(tok[:, :, 0])          # column 0 is log(mean 1 + delta)
    assert torch.allclose(got, block_mean, rtol=1e-4, atol=1e-5)


def test_tokens_are_differentiable_and_finite_on_an_empty_crop():
    f = _field(24, 6).clone().requires_grad_(True)
    soft_rockstar_tokens(f, None, None, SMALL).sum().backward()
    assert torch.isfinite(f.grad).all()
    assert float(f.grad.abs().sum()) > 0

    empty = torch.zeros(1, 6, 20, 20, 20)
    tok = soft_rockstar_tokens(empty, None, None, SoftRockstarConfig(tokens_per_axis=2))
    assert torch.isfinite(tok).all(), tok


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _small_proxy(seed=0):
    cfg = SoftRockstarProxyConfig(
        n_token_features=len(token_feature_names(SMALL)), n_tokens=4 ** 3,
        token_hidden=(8,), embed_dim=4)
    m = SoftRockstarProxy(cfg, seed=seed)
    g = torch.Generator().manual_seed(99)
    tokens = torch.randn(16, 2, 4 ** 3, cfg.n_token_features, generator=g)
    m.fit_standardizer(tokens)
    return m, tokens


def test_proxy_is_permutation_invariant_over_tokens():
    m, tokens = _small_proxy()
    perm = torch.randperm(tokens.shape[2], generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        a = m(tokens)
        b = m(tokens[:, :, perm])
    for k in ("n_sub", "n_host", "occ_numerator"):
        assert torch.allclose(a[k], b[k], atol=1e-5), k


def test_proxy_round_trips_through_the_shared_serialization(tmp_path):
    """load_proxy must rebuild an arm-C member from the class name in the blob.

    The ensemble directory layout is identical across arms, so this is the one
    mechanism keeping a directory of arm-C members from silently loading as
    (shape-incompatible) CatalogProxy checkpoints.
    """
    m, tokens = _small_proxy(seed=3)
    p = m.save(tmp_path / "member_00.pt")
    again = load_proxy(p)
    assert isinstance(again, SoftRockstarProxy)
    with torch.no_grad():
        a, b = m(tokens), again(tokens)
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), k

    ens = ProxyEnsemble.load(tmp_path)
    assert len(ens) == 1
    assert isinstance(ens.members[0], SoftRockstarProxy)


def test_proxy_gradient_reaches_the_field_through_the_tokens():
    """The actor path: field -> tokens -> proxy -> scalar, end to end.

    The residual head is zero-initialised, so an UNTRAINED proxy outputs a
    constant (exactly frozen) and its input gradient is identically zero -- that
    is the correct prior, not a bug. The actor path uses a *trained* proxy, so the
    property that matters is that once the head is non-trivial the gradient
    reaches the field; break the zero-init to stand in for training.
    """
    m, _ = _small_proxy()
    torch.nn.init.normal_(m.head[-1].weight, std=0.2)
    m.eval()
    cand = _field(24, 8).clone().requires_grad_(True)
    tok = soft_rockstar_paired_tokens(cand, _field(24, 9), None, None, SMALL)
    out = m(tok)
    (out["n_sub"].sum() + out["occ_numerator"].sum()).backward()
    assert torch.isfinite(cand.grad).all()
    assert float(cand.grad.abs().sum()) > 0


def test_arm_registry_declares_c_as_sidecar():
    assert ARMS == ("a", "b", "c", "d", "e", "f")
    assert arm_storage("a") == arm_storage("b") == "inline"
    assert arm_storage("c") == arm_storage("d") == arm_storage("e") == "sidecar"
    assert arm_storage("f") == "field"
    assert sidecar_arms() == ("c", "d", "e")
