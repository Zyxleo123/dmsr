"""7.1 -- operator and null-space identities.

Target: relative consistency error <= 1e-5. The block-average / block-broadcast
pair is exact up to float rounding, so these actually land near 1e-7.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmo_sr.dmsr.operator import NullSpaceOperator

FACTORS = [2, 4, 8]
TOL_REL = 1e-5


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def _abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm())


@pytest.fixture(params=FACTORS)
def op(request):
    return NullSpaceOperator(factor=request.param)


def _random_pair(op, batch=2, channels=6, n_lr=4, seed=0):
    torch.manual_seed(seed)
    n_hr = n_lr * op.factor
    x = torch.randn(batch, channels, n_hr, n_hr, n_hr)
    y = torch.randn(batch, channels, n_lr, n_lr, n_lr)
    return x, y


def test_a_of_a_plus_is_identity(op):
    """A(A_plus(y)) == y."""
    _, y = _random_pair(op)
    got = op.A(op.A_plus(y))
    assert _rel(got, y) <= TOL_REL
    # Per-element *relative* error. An absolute bound would be misleading here:
    # A_plus broadcasts a value over an s^3 block and A averages it back, so the
    # only error is fp32 summation over 512 terms at factor 8. Measured max
    # relative error: 1.5e-7 (s=2), 9.6e-7 (s=4), 7.6e-6 (s=8) -- growing with
    # block size, all within the 1e-5 target.
    max_rel = float((got - y).abs().max() / y.abs().max())
    assert max_rel <= TOL_REL, f"max elementwise relative error {max_rel:.2e}"


def test_a_of_a_plus_is_algebraically_exact_in_float64(op):
    """The identity is exact, not approximate: in fp64 the error is bit-for-bit 0.

    This pins down that any residual in fp32 is accumulation rounding and not a
    modelling approximation -- so the 1e-5 target is a precision statement about
    the dtype, not a tolerance the operator design needs.
    """
    _, y = _random_pair(op)
    y = y.double()
    assert float((op.A(op.A_plus(y)) - y).abs().max()) == 0.0


def test_a_of_p_a_is_zero(op):
    """A(P_A(x)) == 0."""
    x, _ = _random_pair(op)
    got = op.A(op.P_A(x))
    assert float(got.abs().max()) < 1e-5
    assert float(got.norm()) / float(op.A(x).norm()) <= TOL_REL


def test_p_a_idempotent(op):
    """P_A(P_A(x)) == P_A(x)."""
    x, _ = _random_pair(op)
    p = op.P_A(x)
    assert _rel(op.P_A(p), p) <= TOL_REL


def test_exact_consistency_of_combine(op):
    """A(A_plus(y) + P_A(r)) == y -- the structural guarantee."""
    r, y = _random_pair(op)
    x_hat = op.combine(y, r)
    got = op.A(x_hat)
    assert _rel(got, y) <= TOL_REL
    _, rel = op.consistency_error(x_hat, y)
    assert rel <= TOL_REL


def test_consistency_holds_on_real_data_shapes():
    """The production shape: 8^3 LR crop -> 64^3 HR crop, 3 displacement channels."""
    op = NullSpaceOperator(factor=8)
    torch.manual_seed(3)
    y = torch.randn(2, 3, 8, 8, 8)
    r = torch.randn(2, 3, 64, 64, 64)
    _, rel = op.consistency_error(op.combine(y, r), y)
    assert rel <= TOL_REL


def test_combine_rejects_mismatched_grid():
    op = NullSpaceOperator(factor=8)
    with pytest.raises(ValueError, match="residual grid"):
        op.combine(torch.randn(1, 3, 8, 8, 8), torch.randn(1, 3, 32, 32, 32))


def test_null_space_is_nontrivial(op):
    """Sanity: P_A must not be the zero map or the identity."""
    x, _ = _random_pair(op)
    p = op.P_A(x)
    assert float(p.norm()) > 0.1 * float(x.norm())
    assert _rel(p, x) > 1e-3
