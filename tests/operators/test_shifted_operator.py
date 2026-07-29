"""Algebra tests for the symmetry-shifted degradation operators ``H_g = A o T_g``.

Covers the plan's required operator checks:
  ||A A^+ y - y||,  ||H_g P_null x||,  ||x - (range + null)||,  <H_g x, y> = <x, H_g^T y>.
"""
import itertools

import pytest
import torch

from cosmo_sr.operators.multiscale import MultiScaleOperators
from cosmo_sr.operators.shifted_operator import ShiftedDownsampleOperator, OperatorContext

FACTORS = [2, 4]
SHIFTS2 = list(itertools.product(range(2), repeat=3))


def _rand_hr(b, c, n, factor, dtype=torch.float64):
    torch.manual_seed(0)
    return torch.randn(b, c, n * factor, n * factor, n * factor, dtype=dtype)


@pytest.mark.parametrize("g", SHIFTS2)
def test_shifted_operator_definition(g):
    # H_g x == A(roll(x, g))
    op = ShiftedDownsampleOperator(2).double()
    x = _rand_hr(2, 6, 8, 2)
    ref = MultiScaleOperators(2).A(torch.roll(x, shifts=g, dims=(-3, -2, -1)))
    assert torch.allclose(op.forward(x, g), ref)


@pytest.mark.parametrize("g", SHIFTS2)
def test_A_Aplus_identity(g):
    # H_g H_g^+ y == y   (exact right inverse)
    op = ShiftedDownsampleOperator(2).double()
    y = torch.randn(2, 6, 8, 8, 8, dtype=torch.float64)
    assert torch.allclose(op.forward(op.pseudoinverse(y, g), g), y, atol=1e-10)


test_shifted_pseudoinverse = test_A_Aplus_identity  # plan alias


@pytest.mark.parametrize("g", SHIFTS2)
def test_null_projection_annihilation(g):
    # H_g (x - H_g^+ H_g x) == 0
    op = ShiftedDownsampleOperator(2).double()
    x = _rand_hr(2, 6, 8, 2)
    assert op.forward(op.project_null(x, g), g).abs().max() < 1e-10


@pytest.mark.parametrize("g", SHIFTS2)
def test_range_plus_null_reconstruction(g):
    # x == project_range(x) + project_null(x)
    op = ShiftedDownsampleOperator(2).double()
    x = _rand_hr(1, 6, 8, 2)
    recon = op.project_range(x, g) + op.project_null(x, g)
    assert torch.allclose(recon, x, atol=1e-10)


@pytest.mark.parametrize("g", SHIFTS2)
def test_adjoint_inner_product(g):
    # <H_g x, y>_LR == <x, H_g^T y>_HR
    op = ShiftedDownsampleOperator(2).double()
    x = _rand_hr(2, 6, 8, 2)
    y = torch.randn(2, 6, 8, 8, 8, dtype=torch.float64)
    lhs = (op.forward(x, g) * y).sum()
    rhs = (x * op.adjoint(y, g)).sum()
    assert torch.allclose(lhs, rhs, atol=1e-9)


@pytest.mark.parametrize("factor", FACTORS)
def test_batch_and_crop_shapes(factor):
    op = ShiftedDownsampleOperator(factor)
    for b, n in [(1, 4), (3, 8)]:
        x = torch.randn(b, 6, n * factor, n * factor, n * factor)
        y = op.forward(x, (1, 0, factor - 1))
        assert y.shape == (b, 6, n, n, n)
        assert op.pseudoinverse(y, (1, 0, factor - 1)).shape == x.shape
        assert op.adjoint(y, (1, 0, factor - 1)).shape == x.shape


def test_fixed_operator_is_g_zero():
    # None / OperatorContext(kind="fixed") == plain A U identities
    op = ShiftedDownsampleOperator(2).double()
    y = torch.randn(2, 6, 8, 8, 8, dtype=torch.float64)
    for ctx in (None, (0, 0, 0), OperatorContext(shift=(0, 0, 0), kind="fixed")):
        assert torch.allclose(op.forward(op.pseudoinverse(y, ctx), ctx), y, atol=1e-10)


def test_periodic_boundary_behavior():
    # A commutes with a *coarse* shift by `factor`: A(roll(x, factor)) == roll(A(x), 1).
    ops = MultiScaleOperators(2).double()
    x = _rand_hr(1, 6, 8, 2)
    lhs = ops.A(torch.roll(x, shifts=(2, 0, 0), dims=(-3, -2, -1)))
    rhs = torch.roll(ops.A(x), shifts=(1, 0, 0), dims=(-3, -2, -1))
    assert torch.allclose(lhs, rhs, atol=1e-10)


def test_float32_and_float64_consistency():
    op32 = ShiftedDownsampleOperator(2)
    op64 = ShiftedDownsampleOperator(2).double()
    x = torch.randn(2, 6, 16, 16, 16)
    g = (1, 1, 0)
    y32 = op32.forward(x, g)
    y64 = op64.forward(x.double(), g)
    assert torch.allclose(y32.double(), y64, atol=1e-5)
