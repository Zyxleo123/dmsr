"""Regression tests for buffered CIC deposition into a fixed Eulerian region.

Stage 2 of the density investigation measured that ``cosmo_sr.eval.density.cic_density``
applied to a 64^3 crop -- which wraps particles back inside the crop -- produces a
field correlated only r = 0.08 with the true density of that region on set14, with
sigma inflated 2.2x. The correct construction deposits particles at their absolute
periodic positions from a padded Lagrangian block and scores only the central cube.

These tests pin the two properties that make that construction trustworthy:

1. once the buffer covers the maximum displacement, widening it further changes
   nothing (the answer has converged and is exact);
2. an undersized buffer loses mass monotonically, so a convergence failure is
   always visible as a mass deficit rather than as a silently wrong field.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dmsr_cic_buffer_audit import (  # noqa: E402
    cic_block_into_region,
    cic_into_region,
)


class _FakeHR:
    """``(6, N, N, N)``-shaped stand-in exposing the slicing the audit code uses."""

    def __init__(self, disp):
        self.disp = disp
        n = disp.shape[-1]
        self.shape = (6, n, n, n)

    def __getitem__(self, key):
        return self.disp[key]


def _uniform_shift_field(n, shift_cells, dis_norm, cellsize):
    """Every particle displaced by the same vector -> density stays uniform."""
    disp = np.zeros((3, n, n, n), dtype=np.float32)
    for d in range(3):
        disp[d] = shift_cells[d] * cellsize / dis_norm
    return _FakeHR(disp)


def _random_field(n, amp_cells, dis_norm, cellsize, seed=0):
    rng = np.random.default_rng(seed)
    disp = rng.standard_normal((3, n, n, n)).astype(np.float32)
    return _FakeHR(disp * (amp_cells * cellsize / dis_norm))


CELLSIZE = 100000.0 / 512
DIS_NORM = 6000.0


def test_converged_buffer_is_invariant_to_further_widening():
    """Beyond the max displacement, extra buffer must change nothing at all."""
    n, R = 64, 16
    amp = 3.0
    hr = _random_field(n, amp, DIS_NORM, CELLSIZE, seed=1)
    max_disp = float(np.abs(hr.disp * (DIS_NORM / CELLSIZE)).max())
    b_exact = int(np.ceil(max_disp)) + 2
    origin = [24, 24, 24]

    ref, _ = cic_block_into_region(hr, origin, R, b_exact, n, DIS_NORM, CELLSIZE)
    for extra in (2, 6, 12):
        wider, _ = cic_block_into_region(hr, origin, R, b_exact + extra, n,
                                         DIS_NORM, CELLSIZE)
        np.testing.assert_allclose(wider, ref, rtol=0, atol=1e-9)


def test_undersized_buffer_loses_mass_monotonically():
    n, R = 64, 16
    hr = _random_field(n, 4.0, DIS_NORM, CELLSIZE, seed=2)
    origin = [24, 24, 24]
    masses = [cic_block_into_region(hr, origin, R, b, n, DIS_NORM, CELLSIZE)[0].sum()
              for b in (0, 4, 8, 16, 24)]
    assert all(a <= b + 1e-9 for a, b in zip(masses, masses[1:])), masses
    assert masses[0] < masses[-1]


def test_uniform_displacement_gives_uniform_density():
    """A rigid translation must leave the scored region exactly uniform."""
    n, R = 64, 16
    shift = (5.0, -3.0, 2.0)
    hr = _uniform_shift_field(n, shift, DIS_NORM, CELLSIZE)
    b = int(np.ceil(max(abs(s) for s in shift))) + 2
    mass, _ = cic_block_into_region(hr, [20, 20, 20], R, b, n, DIS_NORM, CELLSIZE)
    np.testing.assert_allclose(mass, np.ones_like(mass), rtol=0, atol=1e-6)


def test_zero_displacement_recovers_one_particle_per_cell():
    n, R = 32, 8
    hr = _FakeHR(np.zeros((3, n, n, n), dtype=np.float32))
    mass, npart = cic_block_into_region(hr, [8, 8, 8], R, 2, n, DIS_NORM, CELLSIZE)
    np.testing.assert_allclose(mass, np.ones_like(mass), rtol=0, atol=1e-6)
    assert npart == (R + 4) ** 3


def test_region_deposit_conserves_mass_across_the_periodic_seam():
    """A region straddling the box edge must behave like any interior region."""
    n, R = 32, 8
    hr = _random_field(n, 2.0, DIS_NORM, CELLSIZE, seed=3)
    b = int(np.ceil(float(np.abs(hr.disp * (DIS_NORM / CELLSIZE)).max()))) + 2
    interior, _ = cic_block_into_region(hr, [12, 12, 12], R, b, n, DIS_NORM, CELLSIZE)
    seam, _ = cic_block_into_region(hr, [n - 4, n - 4, n - 4], R, b, n, DIS_NORM, CELLSIZE)
    for m in (interior, seam):
        assert m.min() >= 0.0
    # Scoring the WHOLE box must conserve mass exactly. Buffer 0 is the right call
    # here: the padded block would wrap over the box and deposit each particle more
    # than once, which is correct behaviour but not a mass-conservation check.
    total, _ = cic_block_into_region(hr, [0, 0, 0], n, 0, n, DIS_NORM, CELLSIZE)
    assert total.sum() == pytest.approx(n ** 3, rel=1e-6)


def test_wrapped_crop_disagrees_with_buffered_truth():
    """Pins the defect itself: wrapping inside the crop is NOT a mild approximation.

    Guards against someone "simplifying" the buffered path back to a plain
    ``cic_density`` call on the crop.
    """
    n, R = 64, 16
    hr = _random_field(n, 6.0, DIS_NORM, CELLSIZE, seed=4)
    origin = [24, 24, 24]
    b = int(np.ceil(float(np.abs(hr.disp * (DIS_NORM / CELLSIZE)).max()))) + 2
    ref, _ = cic_block_into_region(hr, origin, R, b, n, DIS_NORM, CELLSIZE)

    disp_c = hr.disp[:, origin[0]:origin[0] + R, origin[1]:origin[1] + R,
                     origin[2]:origin[2] + R] * (DIS_NORM / CELLSIZE)
    q = np.arange(R, dtype=np.float64) + 0.5
    pos = np.empty((3, R, R, R))
    pos[0] = disp_c[0] + q[:, None, None]
    pos[1] = disp_c[1] + q[None, :, None]
    pos[2] = disp_c[2] + q[None, None, :]
    wrapped = cic_into_region(np.mod(pos.reshape(3, -1), R), [0, 0, 0], R, R)

    assert wrapped.sum() == pytest.approx(R ** 3, rel=1e-6)   # keeps every particle
    assert ref.sum() != pytest.approx(R ** 3, rel=1e-3)       # truth need not
    corr = np.corrcoef(wrapped.ravel(), ref.ravel())[0, 1]
    assert corr < 0.5, f"wrapped crop unexpectedly agrees with truth (r={corr:.3f})"
