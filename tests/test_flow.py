import torch

from cosmo_sr.models.residual_flow import ResidualFlowModel
from cosmo_sr.operators.multiscale import MultiScaleOperators
from cosmo_sr.operators.base_upscaler import IdentityUpscaler
from cosmo_sr.losses.flow import (
    flow_matching_loss,
    band_power,
    band_statistics_loss,
    degrade_consistency_loss,
)
from cosmo_sr.inference.flow_sample import (
    integrate_residual,
    sample_step,
    super_resolve_cascade,
)


def _model():
    return ResidualFlowModel(channels=6, width=8, depth=2, embed_dim=32)


def test_grad_checkpoint_matches_no_checkpoint():
    torch.manual_seed(0)
    m_ckpt = ResidualFlowModel(channels=6, width=8, depth=2, embed_dim=32, use_checkpoint=True)
    m_plain = ResidualFlowModel(channels=6, width=8, depth=2, embed_dim=32, use_checkpoint=False)
    m_plain.load_state_dict(m_ckpt.state_dict())
    m_ckpt.train()
    m_plain.train()

    y = torch.randn(1, 6, 4, 4, 4)
    r_t = torch.randn(1, 6, 8, 8, 8)
    t = torch.rand(1)
    R = torch.tensor([64.0])

    v1 = m_ckpt(r_t.clone().requires_grad_(True), t, y, R)
    v1.pow(2).mean().backward()
    v2 = m_plain(r_t.clone().requires_grad_(True), t, y, R)
    v2.pow(2).mean().backward()

    assert torch.allclose(v1, v2, atol=1e-5)
    for (n1, p1), (n2, p2) in zip(m_ckpt.named_parameters(), m_plain.named_parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-4), n1

    # checkpointing only engages in training mode; eval should skip it cleanly
    m_ckpt.eval()
    with torch.no_grad():
        v_eval = m_ckpt(r_t, t, y, R)
    assert torch.isfinite(v_eval).all()


def test_model_velocity_shape():
    model = _model()
    y = torch.randn(2, 6, 4, 4, 4)
    r_t = torch.randn(2, 6, 8, 8, 8)
    t = torch.rand(2)
    R = torch.full((2,), 64.0)
    v = model(r_t, t, y, R)
    assert v.shape == r_t.shape


def test_model_scale_shared():
    # same weights accept different octaves / sizes
    model = _model()
    for n in (4, 8):
        y = torch.randn(1, 6, n, n, n)
        r_t = torch.randn(1, 6, 2 * n, 2 * n, 2 * n)
        v = model(r_t, torch.rand(1), y, torch.tensor([128.0]))
        assert v.shape == r_t.shape


def test_flow_matching_loss_and_null_target():
    model = _model()
    ops = MultiScaleOperators(2)
    B = IdentityUpscaler(2)
    x_R = torch.randn(2, 6, 4, 4, 4)
    x_2R = torch.randn(2, 6, 8, 8, 8)
    R = torch.full((2,), 64.0)
    loss, r_star = flow_matching_loss(model, ops, B, x_R, x_2R, R)
    assert loss.ndim == 0 and torch.isfinite(loss)
    # target residual lives in the null space of A
    assert ops.A(r_star).abs().max() < 1e-4


def test_band_power_and_loss():
    h = torch.randn(2, 6, 16, 16, 16, requires_grad=True)
    bp = band_power(h, n_bands=6)
    assert bp.shape == (6,)
    target = torch.zeros(6)
    loss = band_statistics_loss(h, target, n_bands=6)
    loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_sample_step_hard_consistency():
    model = _model()
    ops = MultiScaleOperators(2)
    B = IdentityUpscaler(2)
    y = torch.randn(1, 6, 4, 4, 4)
    with torch.no_grad():
        x = sample_step(model, ops, B, y, 64.0, n_steps=3)
    assert x.shape == (1, 6, 8, 8, 8)
    # A(x_2R) == y by construction
    assert torch.allclose(ops.A(x), y, atol=1e-4)


def test_integrate_residual_in_null_space():
    model = _model()
    ops = MultiScaleOperators(2)
    y = torch.randn(1, 6, 4, 4, 4)
    with torch.no_grad():
        r = integrate_residual(model, ops, y, 64.0, n_steps=3)
    assert ops.A(r).abs().max() < 1e-4


def test_cascade_shapes():
    model = _model()
    ops = MultiScaleOperators(2)
    B = IdentityUpscaler(2)
    y64 = torch.randn(1, 6, 4, 4, 4)
    out = super_resolve_cascade(model, ops, B, y64, resolutions=(64, 128, 256), n_steps=2)
    assert set(out.keys()) == {128, 256, 512}
    assert out[128].shape == (1, 6, 8, 8, 8)
    assert out[256].shape == (1, 6, 16, 16, 16)
    assert out[512].shape == (1, 6, 32, 32, 32)
    # consistency chained: A(x512) == x256, etc.
    assert torch.allclose(ops.A(out[512]), out[256], atol=1e-4)
