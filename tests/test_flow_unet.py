import torch
import torch.nn as nn

from cosmo_sr.models.flow_unet import Map2MapUNet3D, UNetResidualFlowModel
from cosmo_sr.operators.multiscale import MultiScaleOperators
from cosmo_sr.operators.base_upscaler import IdentityUpscaler
from cosmo_sr.losses.flow import flow_matching_loss
from cosmo_sr.inference.flow_sample import sample_step


def _inputs(b=2, c=6, n=16):
    y_R = torch.randn(b, c, n, n, n)
    r_t = torch.randn(b, c, 2 * n, 2 * n, 2 * n)
    t = torch.rand(b)
    R = torch.full((b,), 64.0)
    return r_t, t, y_R, R


def test_flow_shape_same_padding():
    model = UNetResidualFlowModel(
        channels=6, width=16, num_levels=2, padding="same",
        norm="group", activation="silu", use_film=True, factor=2,
    )
    r_t, t, y_R, R = _inputs()
    out = model(r_t, t, y_R, R)
    assert out.shape == r_t.shape
    assert torch.isfinite(out).all()


def test_gadgets_off_runs():
    model = UNetResidualFlowModel(
        channels=6, width=16, num_levels=2, padding="same",
        norm="batch", activation="leaky_relu",
        use_resblocks=False, use_film=False, use_attention=False,
        zero_init_tail=False, use_checkpoint=False, factor=2,
    )
    r_t, t, y_R, R = _inputs()
    out = model(r_t, t, y_R, R)
    assert out.shape == r_t.shape


def test_all_gadgets_on_runs():
    model = UNetResidualFlowModel(
        channels=6, width=16, num_levels=2, blocks_per_level=2, padding="same",
        norm="group", activation="silu",
        use_resblocks=True, use_film=True, use_attention=True, attention_heads=4,
        use_checkpoint=True, factor=2,
    )
    r_t, t, y_R, R = _inputs(n=8)
    out = model(r_t, t, y_R, R)
    out.pow(2).mean().backward()
    assert out.shape == r_t.shape


def test_film_changes_output():
    model = UNetResidualFlowModel(
        channels=6, width=16, num_levels=2, padding="same",
        norm="group", activation="silu", use_film=True, factor=2,
    )
    model.eval()
    r_t, _, y_R, R = _inputs()
    with torch.no_grad():
        out1 = model(r_t, torch.zeros(2), y_R, R)
        out2 = model(r_t, torch.ones(2), y_R, R)
    assert not torch.allclose(out1, out2)


def test_film_off_ignores_t_and_R():
    model = UNetResidualFlowModel(
        channels=6, width=16, num_levels=2, padding="same",
        norm="group", activation="silu", use_film=False, factor=2,
    )
    model.eval()
    r_t, _, y_R, R = _inputs()
    with torch.no_grad():
        out1 = model(r_t, torch.zeros(2), y_R, R)
        out2 = model(r_t, torch.ones(2), y_R, torch.full((2,), 256.0))
    assert torch.allclose(out1, out2)


def test_zero_init_tail():
    model = UNetResidualFlowModel(
        channels=6, width=16, num_levels=2, padding="same",
        norm="group", activation="silu", zero_init_tail=True, factor=2,
    )
    r_t, t, y_R, R = _inputs()
    out = model(r_t, t, y_R, R)
    assert out.abs().max() < 1e-5


def test_grad_checkpoint_matches_no_checkpoint():
    torch.manual_seed(0)
    kw = dict(channels=6, width=16, num_levels=2, padding="same",
              norm="group", activation="silu", use_film=True, factor=2)
    m_ckpt = UNetResidualFlowModel(use_checkpoint=True, **kw)
    m_plain = UNetResidualFlowModel(use_checkpoint=False, **kw)
    m_plain.load_state_dict(m_ckpt.state_dict())
    m_ckpt.train()
    m_plain.train()

    r_t, t, y_R, R = _inputs(b=1, n=8)
    v1 = m_ckpt(r_t, t, y_R, R)
    v1.pow(2).mean().backward()
    v2 = m_plain(r_t, t, y_R, R)
    v2.pow(2).mean().backward()

    assert torch.allclose(v1, v2, atol=1e-5)
    for (n1, p1), (_, p2) in zip(m_ckpt.named_parameters(), m_plain.named_parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-4), n1


def test_map2map_compat_mode():
    # compat forces the plain map2map-style U-Net regardless of other flags
    model = UNetResidualFlowModel(
        channels=6, width=16, num_levels=2, map2map_compat=True,
        norm="group", activation="silu", use_resblocks=True, use_film=True,
        zero_init_tail=True, factor=2,
    )
    mods = list(model.modules())
    assert any(isinstance(m, nn.BatchNorm3d) for m in mods)
    assert any(isinstance(m, nn.LeakyReLU) for m in mods)
    assert not any(isinstance(m, (nn.GroupNorm, nn.SiLU)) for m in mods)
    assert not model.use_film

    # valid padding shrinks the output but keeps it cubic
    r_t, t, y_R, R = _inputs(b=1, n=24)
    out = model(r_t, t, y_R, R)
    assert out.shape[:2] == (1, 6)
    nx, ny, nz = out.shape[2:]
    assert nx == ny == nz and 0 < nx < r_t.shape[-1]


def test_unet_core_global_bypass_auto():
    torch.manual_seed(0)
    net = Map2MapUNet3D(in_channels=3, out_channels=3, width=8, padding="same",
                        norm="none", zero_init_tail=True)
    assert net.global_bypass  # auto: in == out
    x = torch.randn(1, 3, 16, 16, 16)
    with torch.no_grad():
        out = net(x)
    assert torch.allclose(out, x, atol=1e-5)  # zero tail + bypass == identity
    assert not Map2MapUNet3D(in_channels=6, out_channels=3, width=8).global_bypass


def test_drop_in_with_flow_losses_and_sampler():
    # same interface as ResidualFlowModel: works with the existing loss + sampler
    model = UNetResidualFlowModel(
        channels=6, width=8, num_levels=2, padding="same",
        norm="group", activation="silu", use_film=True, factor=2,
    )
    ops = MultiScaleOperators(2)
    B = IdentityUpscaler(2)
    x_R = torch.randn(2, 6, 4, 4, 4)
    x_2R = torch.randn(2, 6, 8, 8, 8)
    R = torch.full((2,), 64.0)
    loss, r_star = flow_matching_loss(model, ops, B, x_R, x_2R, R)
    assert loss.ndim == 0 and torch.isfinite(loss)
    with torch.no_grad():
        x = sample_step(model, ops, B, x_R, 64.0, n_steps=2)
    assert x.shape == x_2R.shape
    assert torch.allclose(ops.A(x), x_R, atol=1e-4)
