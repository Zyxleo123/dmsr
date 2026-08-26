"""The adversarial arm is off, and the code says why rather than pretending."""
from __future__ import annotations

import pytest
import torch

from cosmo_sr.reward.sr2_adversarial import (
    AdversarialConfig, SR2Critic, build_critic, critic_input,
    find_discriminator_checkpoint, gradient_penalty, wgan_critic_loss,
    wgan_generator_loss,
)


def test_no_discriminator_checkpoint_exists_in_this_checkout():
    """Documented finding, kept as a test so it is re-checked, not remembered."""
    assert find_discriminator_checkpoint() is None


def test_mainline_config_is_off():
    cfg = AdversarialConfig()
    assert cfg.weight == 0.0
    assert not cfg.enabled()
    assert cfg.allow_fresh_critic is False
    assert cfg.gp_lambda == 10.0 and cfg.gp_every == 16


def test_build_critic_refuses_to_invent_one():
    critic, prov = build_critic(AdversarialConfig())
    assert critic is None
    assert prov["source"] == "none"
    assert "not 'ordinary continuation'" in prov["reason"]


def test_a_fresh_critic_is_labelled_as_an_ablation():
    critic, prov = build_critic(AdversarialConfig(allow_fresh_critic=True,
                                                  critic_width=8, critic_depth=2))
    assert isinstance(critic, SR2Critic)
    assert prov["source"] == "fresh_random_init_ABLATION_ONLY"
    assert prov["critic_warmup_steps"] == 1000


def test_critic_input_is_twenty_channels():
    lr = torch.randn(2, 6, 4, 4, 4)
    field = torch.randn(2, 6, 16, 16, 16) * 0.01
    x = critic_input(lr, field, cellsize_kpc_h=195.3125, grid_mult=2)
    assert x.shape == (2, 20, 16, 16, 16)
    # 6 upsampled LR + 6 field + 8 inverse-pixel-shuffled density.
    assert torch.allclose(x[:, 6:12], field)
    assert torch.isfinite(x).all()


def test_critic_input_valid_center_scores_the_central_cube():
    """valid_center>0 emits the central R^3 cube on every channel, still 20-wide."""
    lr = torch.randn(2, 6, 4, 4, 4)
    field = torch.randn(2, 6, 16, 16, 16) * 0.01
    x = critic_input(lr, field, cellsize_kpc_h=195.3125, grid_mult=2,
                     valid_center=8)
    assert x.shape == (2, 20, 8, 8, 8)
    # the field channels are the centre-crop of the input field...
    assert torch.allclose(x[:, 6:12], field[:, :, 4:12, 4:12, 4:12])
    assert torch.isfinite(x).all()


def test_critic_input_valid_center_is_translation_invariant():
    """A rigid bulk shift of every particle leaves the valid-centre density put.

    The wrapped deposit fails this: a shift moves mass across the `% ng` seam.
    The offset deposit follows the bulk, so the density channel is unchanged --
    the property that makes it a real density rather than a scrambling.
    """
    torch.manual_seed(0)
    cellsize, dis_norm = 195.3125, 6000.0
    lr = torch.zeros(1, 6, 4, 4, 4)
    field = torch.randn(1, 6, 16, 16, 16) * 0.02
    shifted = field.clone()
    # + exactly one HR cell of displacement on every particle: an integer rigid
    # translation, which the rounded bulk offset absorbs with no sub-cell change.
    shifted[:, 0:3] += cellsize / dis_norm
    a = critic_input(lr, field, cellsize_kpc_h=cellsize, dis_norm_kpc_h=dis_norm,
                     grid_mult=1, valid_center=8)
    b = critic_input(lr, shifted, cellsize_kpc_h=cellsize, dis_norm_kpc_h=dis_norm,
                     grid_mult=1, valid_center=8)
    # density channel is the last grid_mult^3 == 1 channel here
    assert torch.allclose(a[:, 12:], b[:, 12:], atol=1e-5)


def test_critic_input_rejects_the_wrong_channel_count():
    with pytest.raises(ValueError, match="6 LR and 6 field"):
        critic_input(torch.randn(1, 3, 4, 4, 4), torch.randn(1, 6, 16, 16, 16),
                     cellsize_kpc_h=195.3125)


def test_inverse_pixel_shuffle_is_lossless():
    from cosmo_sr.reward.sr2_adversarial import _inverse_pixel_shuffle

    x = torch.arange(2 * 1 * 4 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4, 4)
    y = _inverse_pixel_shuffle(x, 2)
    assert y.shape == (2, 8, 2, 2, 2)
    assert torch.equal(torch.sort(y.flatten())[0], torch.sort(x.flatten())[0])


def test_critic_forward_and_channel_check():
    c = SR2Critic(width=4, depth=2)
    assert float(c(torch.randn(3, 20, 16, 16, 16)).shape[0]) == 3
    with pytest.raises(ValueError, match="expects 20 channels"):
        c(torch.randn(1, 12, 16, 16, 16))


def test_wgan_losses_have_the_right_signs():
    real = torch.tensor([2.0, 2.0])
    fake = torch.tensor([-1.0, -1.0])
    assert float(wgan_critic_loss(real, fake)) < 0.0     # critic is doing well
    assert float(wgan_generator_loss(fake)) > 0.0        # generator is not


def test_gradient_penalty_is_finite_and_positive():
    torch.manual_seed(0)
    c = SR2Critic(width=4, depth=2)
    real = torch.randn(2, 20, 16, 16, 16)
    fake = torch.randn(2, 20, 16, 16, 16)
    gp = gradient_penalty(c, real, fake, gp_lambda=10.0)
    assert torch.isfinite(gp) and float(gp.detach()) > 0.0
    gp.backward()
    assert any(p.grad is not None for p in c.parameters())


def test_gradient_penalty_rejects_mismatched_shapes():
    c = SR2Critic(width=4, depth=2)
    with pytest.raises(ValueError, match="real"):
        gradient_penalty(c, torch.randn(1, 20, 8, 8, 8), torch.randn(2, 20, 8, 8, 8))


def test_critic_has_no_batch_dependent_normalisation():
    """BatchNorm would invalidate the per-sample gradient penalty."""
    c = SR2Critic(width=4, depth=2)
    assert not any(isinstance(m, (torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm))
                   for m in c.modules())
    x = torch.randn(4, 20, 16, 16, 16)
    alone = c(x[:1])
    together = c(x)[:1]
    assert torch.allclose(alone, together, atol=1e-5)
