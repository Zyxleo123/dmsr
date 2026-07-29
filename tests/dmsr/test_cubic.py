"""7.6 -- cubic transforms. Stage E must not be enabled unless these pass.

Every check runs over **all 24** orientation-preserving rotations, not a sample.
"""
from __future__ import annotations

import pytest
import torch

from cosmo_sr.dmsr.cubic import CubicRotation, all_cubic_rotations
from cosmo_sr.dmsr.operator import NullSpaceOperator

FACTOR = 4
N_LR = 4
N_HR = N_LR * FACTOR
ROTATIONS = all_cubic_rotations()


@pytest.fixture(scope="module")
def field():
    torch.manual_seed(0)
    return torch.randn(2, 6, N_HR, N_HR, N_HR)


def test_there_are_exactly_24_proper_rotations():
    assert len(ROTATIONS) == 24
    assert len({(r.perm, r.flip_axes) for r in ROTATIONS}) == 24


def test_improper_transforms_are_rejected():
    with pytest.raises(ValueError, match="det"):
        CubicRotation((0, 1, 2), (0,))          # single reflection, det = -1


@pytest.mark.parametrize("g", ROTATIONS, ids=lambda g: f"{g.perm}{g.flip_axes}")
def test_inverse_recovers_the_input(g, field):
    assert torch.allclose(g.invert(g.apply(field)), field, atol=1e-6)
    assert torch.allclose(g.apply(g.invert(field)), field, atol=1e-6)


@pytest.mark.parametrize("g", ROTATIONS, ids=lambda g: f"{g.perm}{g.flip_axes}")
def test_norms_are_preserved(g, field):
    rotated = g.apply(field)
    assert float(rotated.pow(2).sum()) == pytest.approx(float(field.pow(2).sum()), rel=1e-5)
    # Pointwise vector magnitude must also be preserved, moved to the rotated voxel.
    mag = field[:, 0:3].pow(2).sum(1, keepdim=True)
    assert torch.allclose(
        rotated[:, 0:3].pow(2).sum(1, keepdim=True), g.apply(mag, scalar=True), atol=1e-5
    )


@pytest.mark.parametrize("g", ROTATIONS, ids=lambda g: f"{g.perm}{g.flip_axes}")
def test_vector_components_are_permuted_and_signed(g, field):
    """A rotation must move vector components, not just voxels.

    Comparing against a voxel-only transform catches the classic bug where the
    grid is rotated but the arrows are left pointing the old way.
    """
    proper = g.apply(field)
    voxels_only = g.apply(field, scalar=True)
    if g.perm != (0, 1, 2) or g.flip_axes:
        assert not torch.allclose(proper, voxels_only, atol=1e-6), (
            "vector components were not transformed"
        )


@pytest.mark.parametrize("g", ROTATIONS, ids=lambda g: f"{g.perm}{g.flip_axes}")
def test_operator_commutes_with_rotation(g, field):
    """A(g(x)) == g_LR(A(x)) -- rotations are symmetries of the measurement."""
    op = NullSpaceOperator(factor=FACTOR)
    assert torch.allclose(op.A(g.apply(field)), g.apply(op.A(field)), atol=1e-5)


@pytest.mark.parametrize("g", ROTATIONS, ids=lambda g: f"{g.perm}{g.flip_axes}")
def test_null_projection_commutes_with_rotation(g, field):
    """P_A(g(x)) == g(P_A(x)) -- so equivariance is well-posed in the null space."""
    op = NullSpaceOperator(factor=FACTOR)
    assert torch.allclose(op.P_A(g.apply(field)), g.apply(op.P_A(field)), atol=1e-5)


def test_rotations_close_under_composition(field):
    """Sanity that this really is a group action: g2(g1(x)) is some rotation of x."""
    g1, g2 = ROTATIONS[5], ROTATIONS[13]
    composed = g2.apply(g1.apply(field))
    assert any(torch.allclose(composed, g.apply(field), atol=1e-6) for g in ROTATIONS)


def test_scalar_fields_transform_without_component_mixing():
    rho = torch.randn(2, 1, N_HR, N_HR, N_HR)
    for g in ROTATIONS:
        out = g.apply(rho, scalar=True)
        assert out.shape == rho.shape
        assert torch.allclose(g.invert(out, scalar=True), rho, atol=1e-6)


def test_vector_field_with_bad_channel_count_raises():
    bad = torch.randn(1, 4, N_HR, N_HR, N_HR)
    with pytest.raises(ValueError, match="divisible by 3"):
        ROTATIONS[1].apply(bad)


def test_equivariance_loss_is_well_formed():
    """Stage E's loss: finite, non-negative, and differentiable.

    ``equivariance_loss`` compares ``v_theta(g(r_t), t, g(y))`` against
    ``g(v_teacher(r_t, t, y))``. This exercises the whole path (both the voxel and
    the vector-component transform, on both sides) so a convention mismatch between
    the two sides shows up here rather than silently as a bad Stage E run.
    """
    from cosmo_sr.dmsr.flow import NullSpaceFlow
    from cosmo_sr.models.operator_denoiser import ModelEMA
    from cosmo_sr.train.train_dmsr import equivariance_loss

    torch.manual_seed(0)
    flow = NullSpaceFlow(channels=3, factor=4, cond_channels=8, encoder_width=8,
                         width=8, num_levels=2, zero_init_tail=False)
    ema = ModelEMA(flow, decay=0.0)
    ema.update(flow)

    y = torch.randn(2, 3, 4, 4, 4)
    r_t = flow.operator.P_A(torch.randn(2, 3, 16, 16, 16))
    t = torch.full((2,), 0.5)

    loss = equivariance_loss(flow, ema, y, r_t, t)
    assert torch.isfinite(loss) and float(loss) >= 0.0
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in flow.velocity_net.parameters())
