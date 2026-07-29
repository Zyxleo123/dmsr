"""Smoke/behaviour tests for the operator-conditioned denoiser."""
import torch

from cosmo_sr.models.operator_denoiser import (
    OperatorConditionedDenoiser,
    CosineSchedule,
    ModelEMA,
)


def _tiny():
    return OperatorConditionedDenoiser(
        channels=6, width=8, num_levels=2, embed_dim=16,
        use_attention=False, use_resblocks=False,
    )


def test_forward_shape():
    net = _tiny()
    x = torch.randn(2, 6, 16, 16, 16)
    t = torch.rand(2)
    out = net(x, t, shift=(1, 0, 1), kind="shifted")
    assert out.shape == x.shape


def test_scalar_and_batched_conditioning():
    net = _tiny()
    x = torch.randn(3, 6, 16, 16, 16)
    # scalar t, tuple shift, string kind broadcast over the batch
    a = net(x, 0.5, shift=(1, 1, 0), kind="fixed")
    # per-sample t, per-sample shift, per-sample kind index
    t = torch.tensor([0.1, 0.5, 0.9])
    shift = torch.tensor([[0, 0, 0], [1, 0, 1], [1, 1, 1]])
    kind = torch.tensor([0, 2, 1])
    b = net(x, t, shift=shift, kind=kind)
    assert a.shape == b.shape == x.shape


def test_operator_context_changes_output():
    # the denoiser must actually use the operator context
    torch.manual_seed(0)
    net = _tiny().eval()
    x = torch.randn(1, 6, 16, 16, 16)
    t = torch.tensor([0.4])
    identity = net(x, t, shift=(0, 0, 0), kind="identity")
    shifted = net(x, t, shift=(1, 1, 1), kind="shifted")
    assert (identity - shifted).abs().max() > 1e-6


def test_cosine_schedule_variance_preserving():
    sch = CosineSchedule()
    t = torch.linspace(0, 1, 11)
    a, s = sch.alpha_sigma(t)
    assert torch.allclose(a ** 2 + s ** 2, torch.ones_like(t), atol=1e-6)
    a0, s0 = sch.alpha_sigma(torch.zeros(1))
    assert a0.item() > 0.999 and s0.item() < 1e-3  # t=0 is clean


def test_ema_tracks_and_lags():
    net = _tiny()
    ema = ModelEMA(net, decay=0.9)
    p0 = next(iter(ema.module.parameters())).clone()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    ema.update(net)
    p1 = next(iter(ema.module.parameters()))
    # moved toward the model but not all the way (0.9 lag)
    assert not torch.allclose(p1, p0)
    assert (p1 - p0).abs().max() < 1.0
