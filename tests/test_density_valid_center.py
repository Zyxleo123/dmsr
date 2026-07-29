"""Tests for the valid-center CIC deposit that replaces the wrapped crop deposit.

``cic_density`` wraps a crop's particles with ``% ng``. On a whole periodic box
that is exact; on a crop it scrambles the field -- measured on set14, a 64^3 crop
deposited that way correlates only r = 0.08 with the region's true density, with
sigma inflated 2.2x, because only 9.75% of a region's own particles stay inside
it. ``cic_density_valid_center`` instead offsets by the crop's bulk displacement
and scores the central cube the crop can actually fill.

See ``docs/density_collapse_investigation.md`` and
``scripts/dmsr_crop_spread.py``.
"""
from __future__ import annotations

import pytest
import torch

from cosmo_sr.eval.density import cic_density, cic_density_valid_center

CELLSIZE = 100000.0 / 512
DIS_NORM = 6000.0


def _f(disp, region, grid_mult=1):
    return cic_density_valid_center(disp, CELLSIZE, DIS_NORM, region, grid_mult=grid_mult)


def test_zero_displacement_is_exactly_uniform():
    d = _f(torch.zeros(1, 3, 32, 32, 32), region=16)
    assert d.shape == (1, 1, 16, 16, 16)
    torch.testing.assert_close(d, torch.zeros_like(d), atol=1e-6, rtol=0)


def test_rigid_translation_leaves_the_field_uniform():
    """The scored cube follows the crop's bulk motion, so a rigid shift is a no-op.

    This is the property the old wrapped deposit lacked: there a bulk shift of
    tens of cells recycled particles through the crop faces and manufactured
    structure that is not in the universe.
    """
    u = torch.zeros(1, 3, 32, 32, 32)
    u[0, 0] = 7.0 * CELLSIZE / DIS_NORM
    u[0, 1] = -4.0 * CELLSIZE / DIS_NORM
    u[0, 2] = 11.0 * CELLSIZE / DIS_NORM
    d = _f(u, region=16)
    assert d.abs().max() < 1e-4


def test_bulk_shift_changes_the_wrapped_deposit_but_not_the_valid_center_one():
    """The defining difference, on a structured field.

    A *uniform* lattice stays uniform under any translation, so the contrast only
    shows once there is structure to move. Adding a bulk shift on top of a fixed
    small-scale pattern must leave the physics alone: ``valid_center`` follows the
    shift, while the wrapped deposit recycles particles through the crop faces and
    returns a different field.
    """
    g = torch.Generator().manual_seed(3)
    structure = torch.randn(1, 3, 32, 32, 32, generator=g) * (1.5 * CELLSIZE / DIS_NORM)
    shifted = structure.clone()
    shifted[0, 0] += 9.0 * CELLSIZE / DIS_NORM
    shifted[0, 1] -= 6.0 * CELLSIZE / DIS_NORM

    a, b = _f(structure, region=8), _f(shifted, region=8)
    torch.testing.assert_close(a, b, atol=2e-3, rtol=1e-2)

    la, lb = cic_density(structure, CELLSIZE, DIS_NORM), cic_density(shifted, CELLSIZE, DIS_NORM)
    assert (la - lb).abs().max() > 0.5, "legacy deposit should have moved"


def test_mass_is_conserved_under_zero_and_rigid_displacement():
    """With no internal motion, every scored cell must hold exactly one particle.

    Random displacement is deliberately excluded here: particles then genuinely
    cross the region boundary in both directions, so the region's total mass is a
    fluctuating quantity and exact conservation is the wrong expectation.
    """
    R = 8
    for u in (torch.zeros(1, 3, 32, 32, 32), None):
        if u is None:
            u = torch.zeros(1, 3, 32, 32, 32)
            u[0, 0] = 5.0 * CELLSIZE / DIS_NORM
            u[0, 2] = -3.0 * CELLSIZE / DIS_NORM
        mass = (_f(u, region=R) + 1.0).sum().item()
        assert mass == pytest.approx(R ** 3, rel=1e-5)


def test_oversized_region_loses_mass_rather_than_wrapping():
    """A region as large as the crop must show a deficit, never a silent wrap.

    The legacy path keeps every particle by construction (it wraps them back in),
    which is exactly what makes its error invisible.
    """
    g = torch.Generator().manual_seed(1)
    disp = torch.randn(1, 3, 32, 32, 32, generator=g) * (4.0 * CELLSIZE / DIS_NORM)
    full = (_f(disp, region=32) + 1.0).sum().item()
    assert full < 32 ** 3 * 0.99, "expected mass loss at the faces"
    legacy_mass = (cic_density(disp, CELLSIZE, DIS_NORM) + 1.0).sum().item()
    assert legacy_mass == pytest.approx(32 ** 3, rel=1e-4)


def test_gradients_reach_the_displacement():
    g = torch.Generator().manual_seed(2)
    disp = (torch.randn(1, 3, 16, 16, 16, generator=g)
            * (0.5 * CELLSIZE / DIS_NORM)).requires_grad_(True)
    _f(disp, region=8).pow(2).sum().backward()
    assert torch.isfinite(disp.grad).all()
    assert disp.grad.abs().sum() > 0


def test_grid_mult_shape_and_lattice_imprint():
    """``grid_mult`` refines the mesh; an unperturbed lattice then shows through.

    That is the true CIC response to a lattice (particles land exactly on fine
    cell boundaries), not a bug -- real displacements wash it out.
    """
    d = _f(torch.zeros(1, 3, 16, 16, 16), region=8, grid_mult=2)
    assert d.shape == (1, 1, 16, 16, 16)
    assert d.min().item() == pytest.approx(-1.0, abs=1e-4)
    assert d.max().item() == pytest.approx(7.0, abs=1e-3)


def test_region_larger_than_crop_is_rejected():
    with pytest.raises(ValueError):
        _f(torch.zeros(1, 3, 8, 8, 8), region=16)


def test_highpass_valid_center_is_off_by_default():
    from cosmo_sr.dmsr.density import HighPassDensity

    hp = HighPassDensity(factor=8, cellsize=CELLSIZE, dis_norm=DIS_NORM)
    assert hp.valid_center == 0
    x = torch.zeros(1, 3, 16, 16, 16)
    torch.testing.assert_close(hp.density(x), cic_density(x, CELLSIZE, DIS_NORM))


def test_critic_input_centre_crops_residual_to_match_density():
    import types

    from cosmo_sr.dmsr.density import HighPassDensity, critic_input

    hp = HighPassDensity(factor=8, cellsize=CELLSIZE, dis_norm=DIS_NORM, valid_center=8)
    x = torch.zeros(1, 3, 16, 16, 16)
    op = types.SimpleNamespace(P_A=lambda t: t)
    out = critic_input(x, op, hp, residual_mode="full", density_mode="highpass")
    assert out.shape[-1] == 8, "residual should be centre-cropped to the density size"
    assert out.shape[1] == 4, "3 residual channels + 1 high-pass density channel"
