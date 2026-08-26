"""Pin the gather HR critic's input view, normaliser and adversarial ramp.

The critic is a REGULARISER added to the member-gather loss to charge for the
field-realism defects the held-out gate found (velocity power collapse,
misplaced small-scale power) that no moment term can see -- see
``docs/sr2_gather_critic.md``. These tests pin the three pieces this line owns:
the six-channel high-pass view, its real-only normaliser, and the warmup+ramp
schedule. The critic net and the hinge losses are the DMSR ones, tested there.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reward"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cosmo_sr.dmsr.critic import HRCritic, hinge_d_loss, hinge_g_loss
from cosmo_sr.features.gather_critic import (
    GatherCriticNorm, gather_critic_input, highpass_field,
)
from cosmo_sr.operators.multiscale import block_average

FACTOR = 4
N = 16          # divisible by FACTOR, small and fast
C = 6


def _field(seed: int, b: int = 2, n: int = N) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, C, n, n, n, generator=g)


# --------------------------------------------------------------------------- #
# The high-pass view
# --------------------------------------------------------------------------- #
def test_highpass_removes_the_lr_resolvable_component():
    """A(x_hp) == 0: the high-pass carries no block-averaged (LR) power.

    This is the whole reason the coarse field is withheld -- it is identical in
    real and fake because both share the LR tile, so a critic fed it would waste
    capacity on structure A_plus(y) reproduces byte-for-byte in both.
    """
    x = _field(0)
    hp = highpass_field(x, FACTOR)
    coarse = block_average(hp, FACTOR)
    assert coarse.abs().max() < 1e-5


def test_highpass_keeps_all_six_channels():
    x = _field(1)
    hp = highpass_field(x, FACTOR)
    assert hp.shape == x.shape
    # Velocity channels [3:6] carry real high-pass power -- the point of the
    # whole exercise; a view that dropped them could not police the collapse.
    assert hp[:, 3:6].pow(2).mean() > 0


def test_highpass_is_differentiable():
    x = _field(2).requires_grad_(True)
    highpass_field(x, FACTOR).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_highpass_rejects_wrong_rank():
    with pytest.raises(ValueError):
        highpass_field(torch.randn(2, C, N, N), FACTOR)


# --------------------------------------------------------------------------- #
# The normaliser: real-only, per-channel, fixed
# --------------------------------------------------------------------------- #
def test_norm_makes_each_channel_unit_rms_on_the_fit_data():
    hp = highpass_field(_field(3, b=4), FACTOR)
    norm = GatherCriticNorm.fit([hp])
    out = norm(hp)
    per_ch = out.reshape(out.shape[0], C, -1).pow(2).mean(dim=(0, 2)).sqrt()
    assert torch.allclose(per_ch, torch.ones(C), atol=1e-4)


def test_norm_applies_identically_to_real_and_fake():
    """The same fixed scales scale any tensor -- the anti-shortcut invariant."""
    norm = GatherCriticNorm.fit([highpass_field(_field(4), FACTOR)])
    a, b = highpass_field(_field(5), FACTOR), highpass_field(_field(6), FACTOR)
    # Ratio of the normalised outputs equals the raw ratio: no per-batch stats.
    assert torch.allclose(norm(a) * norm.scale.view(1, C, 1, 1, 1), a, atol=1e-5)
    assert torch.allclose(norm(b) * norm.scale.view(1, C, 1, 1, 1), b, atol=1e-5)


def test_norm_roundtrips_through_dict():
    norm = GatherCriticNorm.fit([highpass_field(_field(7), FACTOR)])
    d = norm.to_dict()
    assert len(d["scale"]) == C and all(s > 0 for s in d["scale"])


def test_norm_channel_mismatch_raises():
    norm = GatherCriticNorm.fit([highpass_field(_field(8), FACTOR)])
    with pytest.raises(ValueError):
        norm(torch.randn(1, 3, N, N, N))


def test_fit_needs_tiles():
    with pytest.raises(ValueError):
        GatherCriticNorm.fit([])


# --------------------------------------------------------------------------- #
# critic_input into the critic
# --------------------------------------------------------------------------- #
def test_critic_input_feeds_a_six_channel_critic():
    norm = GatherCriticNorm.fit([highpass_field(_field(9), FACTOR)])
    critic = HRCritic(in_channels=C, width=8, n_layers=2)
    x = _field(10)
    ci = gather_critic_input(x, FACTOR, normalizer=norm)
    assert ci.shape == x.shape
    score = critic(ci)
    assert score.dim() == 5 and score.shape[1] == 1


def test_generator_adv_gradient_reaches_the_input():
    """-D(high-pass(cand)) must be differentiable back to cand: the adversarial
    signal is useless if it does not reach the generator's output."""
    norm = GatherCriticNorm.fit([highpass_field(_field(11), FACTOR)])
    critic = HRCritic(in_channels=C, width=8, n_layers=2)
    cand = _field(12).requires_grad_(True)
    g = hinge_g_loss(critic(gather_critic_input(cand, FACTOR, normalizer=norm)))
    g.backward()
    assert cand.grad is not None and cand.grad.abs().sum() > 0


def test_hinge_d_loss_separates_when_real_scores_high():
    real = torch.full((2, 1, 2, 2, 2), 2.0)
    fake = torch.full((2, 1, 2, 2, 2), -2.0)
    # Both terms saturated to zero when real>=1 and fake<=-1.
    assert hinge_d_loss(real, fake).item() == pytest.approx(0.0)
    # A confused critic (fake scored high) pays.
    assert hinge_d_loss(real, torch.zeros_like(fake)).item() > 0


# --------------------------------------------------------------------------- #
# The warmup + ramp schedule
# --------------------------------------------------------------------------- #
def _args(**kw):
    d = dict(w_adv=0.5, adv_warmup_steps=100, adv_ramp_steps=200)
    d.update(kw)
    return argparse.Namespace(**d)


def test_adv_weight_off_when_disabled():
    from finetune_member_gather import adv_weight_at
    assert adv_weight_at(9999, _args(w_adv=0.0)) == 0.0


def test_adv_weight_zero_through_warmup():
    from finetune_member_gather import adv_weight_at
    a = _args()
    assert adv_weight_at(1, a) == 0.0
    assert adv_weight_at(100, a) == 0.0          # inclusive of the last warmup step


def test_adv_weight_ramps_then_caps():
    from finetune_member_gather import adv_weight_at
    a = _args()
    assert adv_weight_at(200, a) == pytest.approx(0.5 * (100 / 200))   # mid-ramp
    assert adv_weight_at(300, a) == pytest.approx(0.5)                 # ramp end
    assert adv_weight_at(5000, a) == pytest.approx(0.5)               # capped
