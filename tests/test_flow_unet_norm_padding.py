"""Tests for the ``norm="channel"`` and ``padding_mode="circular"`` options.

Both were added to address measured defects (see
``docs/density_collapse_investigation.md``):

* ``nn.GroupNorm`` reduces over ``(C/G, D, H, W)``, making every output voxel
  depend on whole-window statistics. That is why the same physical region gets
  different conditioning features at different crop sizes, and it accounted for
  92.7% of the measured context-8 error.
* the velocity U-Net padded with zeros while the boxes are periodic and the LR
  encoder already pads circularly.

Both default to the previous behaviour, so existing configs are unchanged.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cosmo_sr.models.flow_unet import (
    ChannelGroupNorm3d,
    UNetResidualFlowModel,
    _make_norm,
)


def test_make_norm_channel_returns_pointwise_layer():
    assert isinstance(_make_norm("channel", 32), ChannelGroupNorm3d)
    assert isinstance(_make_norm("group", 32), nn.GroupNorm)
    with pytest.raises(ValueError):
        _make_norm("nope", 32)


def test_channel_norm_is_spatially_local_but_groupnorm_is_not():
    """The whole point: a sub-volume must normalise identically in isolation."""
    x = torch.randn(1, 8, 8, 8, 8)
    sub = x[:, :, :4, :4, :4]

    gn = nn.GroupNorm(2, 8, affine=False)
    assert not torch.allclose(gn(sub), gn(x)[:, :, :4, :4, :4], atol=1e-5)

    cn = ChannelGroupNorm3d(2, 8)
    torch.nn.init.ones_(cn.weight)
    torch.nn.init.zeros_(cn.bias)
    torch.testing.assert_close(cn(sub), cn(x)[:, :, :4, :4, :4], atol=1e-6, rtol=1e-5)


def test_channel_norm_normalises_within_each_group_at_each_voxel():
    cn = ChannelGroupNorm3d(2, 6)
    torch.nn.init.ones_(cn.weight)
    torch.nn.init.zeros_(cn.bias)
    gen = torch.Generator().manual_seed(0)          # keep the tolerance meaningful
    y = cn(torch.randn(2, 6, 3, 3, 3, generator=gen))
    g = y.reshape(2, 2, 3, 3, 3, 3)
    torch.testing.assert_close(g.mean(dim=2), torch.zeros_like(g.mean(dim=2)),
                               atol=1e-5, rtol=0)
    # Only 3 channels per group, so a group whose variance happens to be small is
    # visibly shrunk by the eps floor (var -> var/(var+eps)). That is the intended
    # behaviour of the layer, not slack in the test, so the tolerance allows it.
    torch.testing.assert_close(g.var(dim=2, unbiased=False),
                               torch.ones_like(g.mean(dim=2)), atol=2e-2, rtol=2e-2)


def test_channel_norm_matches_groupnorm_parameter_count():
    gn = nn.GroupNorm(4, 16)
    cn = ChannelGroupNorm3d(4, 16)
    assert sum(p.numel() for p in gn.parameters()) == sum(p.numel() for p in cn.parameters())


def _model(**kw):
    return UNetResidualFlowModel(
        channels=3, width=8, num_levels=2, blocks_per_level=1,
        padding="same", activation="silu", use_film=True, embed_dim=16,
        context_channels=0, factor=8, **kw
    ).eval()


@pytest.mark.parametrize("norm", ["group", "channel"])
@pytest.mark.parametrize("padding_mode", ["zeros", "circular"])
def test_forward_shapes_preserved(norm, padding_mode):
    m = _model(norm=norm, padding_mode=padding_mode)
    r = torch.randn(1, 3, 16, 16, 16)
    y = torch.randn(1, 3, 2, 2, 2)
    with torch.no_grad():
        v = m(r, torch.tensor([0.5]), y, torch.tensor([2.0]))
    assert v.shape == r.shape


def test_circular_padding_is_translation_equivariant_on_a_periodic_input():
    """With circular padding a periodic roll of the input must roll the output.

    Zero padding cannot do this: it invents a vacuum at the faces, so a roll
    changes which content sits against the fake boundary. GroupNorm is replaced
    here too, since its global statistics are roll-invariant but its presence is
    not what is under test.
    """
    m = _model(norm="channel", padding_mode="circular")
    r = torch.randn(1, 3, 16, 16, 16)
    y = torch.randn(1, 3, 2, 2, 2)
    t, R = torch.tensor([0.5]), torch.tensor([2.0])
    with torch.no_grad():
        v = m(r, t, y, R)
        # roll by a multiple of 2**num_levels so the strided stages stay aligned
        v_rolled = m(torch.roll(r, 4, dims=2), t, torch.roll(y, 1, dims=2), R)
    torch.testing.assert_close(torch.roll(v, 4, dims=2), v_rolled, atol=1e-4, rtol=1e-3)


def test_zero_padding_is_not_translation_equivariant():
    m = _model(norm="channel", padding_mode="zeros")
    r = torch.randn(1, 3, 16, 16, 16)
    y = torch.randn(1, 3, 2, 2, 2)
    t, R = torch.tensor([0.5]), torch.tensor([2.0])
    with torch.no_grad():
        v = m(r, t, y, R)
        v_rolled = m(torch.roll(r, 4, dims=2), t, torch.roll(y, 1, dims=2), R)
    assert not torch.allclose(torch.roll(v, 4, dims=2), v_rolled, atol=1e-4)


def test_defaults_are_unchanged():
    """A model built with no new kwargs must still be GroupNorm + zero padding."""
    m = _model()
    norms = [mod for mod in m.modules() if isinstance(mod, nn.GroupNorm)]
    assert norms, "default build should still use nn.GroupNorm"
    assert not any(isinstance(mod, ChannelGroupNorm3d) for mod in m.modules())
    convs = [mod for mod in m.modules()
             if isinstance(mod, nn.Conv3d) and mod.padding != (0, 0, 0)]
    assert convs and all(c.padding_mode == "zeros" for c in convs)


def test_build_flow_reads_norm_and_padding_mode():
    from cosmo_sr.dmsr.flow import build_flow

    cfg = {"factor": 8, "model": {"width": 8, "num_levels": 2, "cond_channels": 4,
                                  "encoder_width": 8, "embed_dim": 16,
                                  "norm": "channel", "padding_mode": "circular"}}
    flow = build_flow(cfg, channels=3)
    assert any(isinstance(m, ChannelGroupNorm3d) for m in flow.velocity_net.modules())
    padded = [m for m in flow.velocity_net.modules()
              if isinstance(m, torch.nn.Conv3d) and m.padding != (0, 0, 0)]
    assert padded and all(m.padding_mode == "circular" for m in padded)


def test_build_flow_defaults_unchanged():
    from cosmo_sr.dmsr.flow import build_flow

    cfg = {"factor": 8, "model": {"width": 8, "num_levels": 2, "cond_channels": 4,
                                  "encoder_width": 8, "embed_dim": 16}}
    flow = build_flow(cfg, channels=3)
    assert any(isinstance(m, nn.GroupNorm) for m in flow.velocity_net.modules())
    assert not any(isinstance(m, ChannelGroupNorm3d) for m in flow.velocity_net.modules())
