import torch

from cosmo_sr.operators.multiscale import (
    MultiScaleOperators,
    block_average,
    block_upsample,
    null_projection,
)
from cosmo_sr.operators.base_upscaler import (
    IdentityUpscaler,
    BackboneUpscaler,
    consistent_base,
)
from cosmo_sr.models.unet_baseline import SimpleSRGenerator


def test_A_right_inverse_U():
    # A_R(U_R(y)) == y exactly
    y = torch.randn(2, 6, 8, 8, 8)
    assert torch.allclose(block_average(block_upsample(y)), y, atol=1e-5)


def test_P_null_in_kernel_of_A():
    # A_R(P_null_R(h)) == 0
    h = torch.randn(2, 6, 16, 16, 16)
    ah = block_average(null_projection(h))
    assert ah.abs().max() < 1e-5


def test_P_null_idempotent():
    h = torch.randn(1, 6, 16, 16, 16)
    p = null_projection(h)
    assert torch.allclose(null_projection(p), p, atol=1e-5)


def test_ops_shapes():
    ops = MultiScaleOperators(factor=2)
    y = torch.randn(1, 6, 8, 8, 8)
    assert ops.U(y).shape == (1, 6, 16, 16, 16)
    assert ops.A(ops.U(y)).shape == y.shape


def test_consistent_base_identity():
    ops = MultiScaleOperators(2)
    B = IdentityUpscaler(2)
    y = torch.randn(2, 6, 8, 8, 8)
    xc = consistent_base(B, ops, y)
    assert xc.shape == (2, 6, 16, 16, 16)
    assert torch.allclose(ops.A(xc), y, atol=1e-5)


def test_consistent_base_backbone():
    # even with an arbitrary learned backbone, A(B_cons(y)) == y
    ops = MultiScaleOperators(2)
    backbone = SimpleSRGenerator(6, 6, scale_factor=2, width=8, depth=1)
    B = BackboneUpscaler(backbone, 2)
    y = torch.randn(1, 6, 8, 8, 8)
    xc = consistent_base(B, ops, y)
    assert torch.allclose(ops.A(xc), y, atol=1e-4)
