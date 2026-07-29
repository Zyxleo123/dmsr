"""Tests for periodic subcell translations ``T_g``."""
import numpy as np
import pytest
import torch

from cosmo_sr.operators.symmetry import SubcellShift, as_shift


def test_shift_inverse():
    sym = SubcellShift(2)
    x = torch.randn(2, 6, 16, 16, 16)
    for g in [(0, 0, 0), (1, 0, 1), (1, 1, 1)]:
        assert torch.allclose(sym.invert_field(sym.apply_field(x, g), g), x)


def test_apply_noise_equals_apply_field():
    # translations act identically on fields and noise
    sym = SubcellShift(4)
    z = torch.randn(1, 6, 8, 8, 8)
    g = (2, 3, 1)
    assert torch.equal(sym.apply_noise(z, g), sym.apply_field(z, g))


def test_is_subcell():
    sym = SubcellShift(8)
    assert sym.is_subcell((1, 0, 0))
    assert sym.is_subcell((0, 0, 7))
    assert not sym.is_subcell((0, 0, 0))
    assert not sym.is_subcell((8, 16, 0))  # coarse multiples of factor


def test_all_shifts_count_and_range():
    sym = SubcellShift(2)
    shifts = sym.all_shifts()
    assert len(shifts) == 8
    assert all(0 <= v < 2 for g in shifts for v in g)


def test_sample_shift_in_range():
    sym = SubcellShift(8)
    rng = np.random.default_rng(0)
    for _ in range(50):
        g = sym.sample_shift(rng)
        assert all(0 <= v < 8 for v in g)


def test_as_shift_validation():
    assert as_shift(None) == (0, 0, 0)
    assert as_shift((1, 2, 3)) == (1, 2, 3)
    with pytest.raises(ValueError):
        as_shift((1, 2))
    with pytest.raises(ValueError):
        as_shift((0, 0, 5), factor=4)  # out of subcell range
