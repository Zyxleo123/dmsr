"""Clean + operator-conditioned ambient denoising branches (Phase C)."""
import numpy as np
import torch

from cosmo_sr.models.operator_denoiser import OperatorConditionedDenoiser, CosineSchedule
from cosmo_sr.operators.shifted_operator import ShiftedDownsampleOperator
from cosmo_sr.losses.ambient_denoise import (
    clean_denoise_loss,
    ambient_denoise_loss,
    build_ambient_target,
)


def _net():
    return OperatorConditionedDenoiser(
        channels=6, width=8, num_levels=2, embed_dim=16,
        use_attention=False, use_resblocks=False,
    )


def test_clean_branch_backprops():
    net = _net()
    x = torch.randn(2, 6, 16, 16, 16)
    loss, diag = clean_denoise_loss(net, x, CosineSchedule())
    loss.backward()
    assert np.isfinite(diag["loss_clean"])
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


def test_ambient_branch_backprops_and_diagnostics():
    net = _net()
    op = ShiftedDownsampleOperator(2)
    y = torch.randn(2, 6, 8, 8, 8)
    loss, diag = ambient_denoise_loss(net, op, y, (1, 0, 1), CosineSchedule())
    loss.backward()
    assert np.isfinite(diag["loss_ambient"])
    for k in ("range_energy", "null_energy", "null_frac"):
        assert k in diag
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


def test_target_construction_semantics():
    op = ShiftedDownsampleOperator(2)
    rng = np.random.default_rng(0)
    x = torch.randn(1, 6, 16, 16, 16)

    y, g, kind = build_ambient_target(x, op, "fixed", rng)
    assert g == (0, 0, 0) and kind == "fixed"
    assert torch.allclose(y, op.forward(x, (0, 0, 0)))

    y, g, kind = build_ambient_target(x, op, "true_shift", rng)
    assert kind == "shifted"
    assert torch.allclose(y, op.forward(x, g))          # C2: y really is H_g x

    # C3: measurement is A x (g=0) but the *assigned* operator is a random shift
    seen_nonzero = False
    for _ in range(20):
        y, g, kind = build_ambient_target(x, op, "virtual_shift", rng)
        assert torch.allclose(y, op.forward(x, (0, 0, 0)))  # measurement == A x, not H_g x
        seen_nonzero |= (g != (0, 0, 0))
    assert seen_nonzero  # shifts are actually sampled


def test_true_shift_perfect_prediction_zero_loss():
    # If the denoiser recovered x exactly, C2's measurement loss would vanish
    # (H_g x == y). Verify the target/operator algebra supports that.
    op = ShiftedDownsampleOperator(2).double()
    rng = np.random.default_rng(1)
    x = torch.randn(1, 6, 16, 16, 16, dtype=torch.float64)
    y, g, _ = build_ambient_target(x, op, "true_shift", rng)
    assert op.forward(x, g).sub(y).abs().max() < 1e-10
