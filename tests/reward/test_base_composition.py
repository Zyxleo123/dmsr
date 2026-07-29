"""Section 1: the frozen baseline interface and residual composition."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmo_sr.reward.base import ResidualComposer, compose
from cosmo_sr.reward.diffusion import DiffusionConfig, unwhiten, whiten
from cosmo_sr.reward.model import build_residual_denoiser


def tiny_model(**kw):
    cfg = {"channels": 6, "scale_factor": 4, "width": 8, "num_levels": 2,
           "blocks_per_level": 1, "embed_dim": 16, "num_groups": 4,
           "sigma_res": [0.02] * 6}
    cfg.update(kw)
    return build_residual_denoiser(cfg)


class FrozenStub(torch.nn.Module):
    """Stands in for SR2: has parameters, must never receive a gradient."""

    def __init__(self, channels=6):
        super().__init__()
        self.conv = torch.nn.Conv3d(channels, channels, 1)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, y):
        return self.conv(y)


def test_residual_scale_zero_is_exactly_the_frozen_output():
    base = torch.randn(1, 6, 8, 8, 8)
    resid = torch.randn(1, 6, 8, 8, 8)
    out = compose(base, resid, residual_scale=0.0)
    assert out is base                       # short-circuit, not 0 * resid
    assert torch.equal(out, base)


def test_residual_scale_zero_survives_a_nan_residual():
    # "Residual disabled" must mean the frozen output bit for bit, even if the
    # residual model produced garbage.
    base = torch.randn(1, 6, 4, 4, 4)
    resid = torch.full_like(base, float("nan"))
    out = compose(base, resid, residual_scale=0.0)
    assert torch.isfinite(out).all()


def test_compose_is_a_plain_sum_with_no_null_space_projection():
    base = torch.randn(1, 6, 8, 8, 8)
    resid = torch.randn(1, 6, 8, 8, 8)
    out = compose(base, resid, residual_scale=0.5)
    assert torch.allclose(out, base + 0.5 * resid, atol=0, rtol=0)


def test_compose_channel_subset_leaves_other_channels_untouched():
    base = torch.randn(1, 6, 4, 4, 4)
    resid = torch.ones_like(base)
    out = compose(base, resid, 1.0, channels=(0, 1, 2))
    assert torch.equal(out[:, 3:], base[:, 3:])
    assert torch.allclose(out[:, :3], base[:, :3] + 1.0)


def test_frozen_parameters_receive_no_gradient():
    frozen = FrozenStub()
    resid = tiny_model()
    y = torch.randn(1, 6, 8, 8, 8)
    base = frozen(y)
    pred = resid(torch.randn(1, 6, 8, 8, 8), torch.full((1,), 0.5),
                 y_lr=torch.randn(1, 6, 2, 2, 2), psi_base=base)
    (base + pred).pow(2).mean().backward()
    assert all(p.grad is None for p in frozen.parameters())
    assert any(p.grad is not None for p in resid.parameters())


def test_composer_optimizer_never_sees_sr2_parameters():
    from cosmo_sr.reward.base import FrozenSR2Base

    resid = tiny_model()
    comp = ResidualComposer(residual=resid, base=None, residual_scale=1.0)
    ids = {id(p) for p in comp.trainable_parameters()}
    assert ids == {id(p) for p in resid.parameters()}
    assert not isinstance(comp.base, FrozenSR2Base)


def test_shapes_and_dtype_match_the_baseline_pipeline():
    m = tiny_model()
    u = torch.randn(2, 6, 8, 8, 8)
    out = m(u, torch.full((2,), 0.3), y_lr=torch.randn(2, 6, 2, 2, 2),
            psi_base=torch.randn(2, 6, 8, 8, 8))
    assert out.shape == u.shape
    assert out.dtype == torch.float32


def test_untrained_model_predicts_zero_so_composition_starts_at_sr2():
    m = tiny_model()
    out = m(torch.randn(1, 6, 8, 8, 8), torch.full((1,), 0.5),
            y_lr=torch.randn(1, 6, 2, 2, 2), psi_base=torch.randn(1, 6, 8, 8, 8))
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_identical_seeds_reproduce_identical_residual_samples():
    comp = ResidualComposer(residual=tiny_model(), residual_scale=1.0)
    base = torch.randn(1, 6, 8, 8, 8)
    y = torch.randn(1, 6, 2, 2, 2)
    cfg = DiffusionConfig(n_steps=3)
    a = comp.sample_residual(base, y, seed=7, cfg=cfg)
    b = comp.sample_residual(base, y, seed=7, cfg=cfg)
    assert torch.equal(a, b)


def test_different_seeds_produce_different_residual_samples():
    # A zero-init head predicts eps = 0, which makes DDIM a deterministic map of
    # the initial noise -- different seeds must still give different fields.
    comp = ResidualComposer(residual=tiny_model(), residual_scale=1.0)
    base = torch.randn(1, 6, 8, 8, 8)
    y = torch.randn(1, 6, 2, 2, 2)
    cfg = DiffusionConfig(n_steps=3)
    a = comp.sample_residual(base, y, seed=0, cfg=cfg)
    b = comp.sample_residual(base, y, seed=1, cfg=cfg)
    assert not torch.allclose(a, b)
    assert float((a - b).abs().mean()) > 0


def test_disabled_residual_composer_returns_zeros():
    comp = ResidualComposer(residual=None, residual_scale=1.0)
    base = torch.randn(1, 6, 4, 4, 4)
    assert torch.equal(
        comp.sample_residual(base, torch.randn(1, 6, 1, 1, 1), seed=0),
        torch.zeros_like(base),
    )
    assert not comp.enabled


def test_whiten_unwhiten_roundtrip_is_exact():
    sigma = torch.tensor([0.01, 0.02, 0.03, 0.1, 0.1, 0.1])
    x = torch.randn(2, 6, 4, 4, 4)
    assert torch.allclose(unwhiten(whiten(x, sigma), sigma), x, atol=1e-6)


def test_cic_accepts_the_composed_field_without_conversion():
    from cosmo_sr.eval.density import cic_density

    base = 0.01 * torch.randn(1, 6, 16, 16, 16)
    resid = 0.001 * torch.randn(1, 6, 16, 16, 16)
    hat = compose(base, resid, 1.0)
    d = cic_density(hat[:, 0:3], cellsize=100000.0 / 512.0, dis_norm=6000.0)
    assert torch.isfinite(d).all()
    assert d.shape[-1] == 16


def test_composition_preserves_normalization_scale():
    # The sum happens in catnorm units, so a small residual must not move the
    # field's scale by anything like its own magnitude.
    base = torch.randn(1, 6, 8, 8, 8)
    resid = 0.01 * torch.randn(1, 6, 8, 8, 8)
    hat = compose(base, resid, 1.0)
    rel = float((hat.std() - base.std()).abs() / base.std())
    assert rel < 0.01


def test_crop_size_must_match_the_unet_divisor():
    m = tiny_model()
    with pytest.raises(ValueError, match="multiple of"):
        m(torch.randn(1, 6, 6, 6, 6), torch.full((1,), 0.5),
          y_lr=torch.randn(1, 6, 2, 2, 2), psi_base=torch.randn(1, 6, 6, 6, 6))


def test_lr_window_must_cover_the_same_region():
    m = tiny_model()
    with pytest.raises(ValueError, match="same region"):
        m(torch.randn(1, 6, 8, 8, 8), torch.full((1,), 0.5),
          y_lr=torch.randn(1, 6, 4, 4, 4),      # 4 * 4 = 16 != 8
          psi_base=torch.randn(1, 6, 8, 8, 8))
