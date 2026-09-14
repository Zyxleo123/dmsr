"""Numerics of the LR-host-channel fidelity job.

Every assertion here guards a way this job could return a confident wrong
answer rather than fail: a grid convention that puts a half-cell offset into
every distance, a periodic match that misses across the boundary, or an R^2
that is really in-sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts" / "reward", ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

lhf = pytest.importorskip("lr_host_fidelity")

BOX = 100.0


def test_periodic_delta_wraps_the_short_way():
    a = np.array([[99.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    d = lhf.periodic_delta(a, b, BOX)
    assert np.allclose(np.abs(d[0, 0]), 2.0)


def test_nearest_is_periodic_and_matches_brute_force():
    rng = np.random.default_rng(0)
    q = rng.uniform(0, BOX, size=(37, 3))
    r = rng.uniform(0, BOX, size=(11, 3))
    i, d = lhf.nearest(q, r, BOX, chunk=5)          # chunking must not matter
    for k in range(q.shape[0]):
        dd = np.sqrt((lhf.periodic_delta(q[k], r, BOX) ** 2).sum(axis=1))
        assert i[k] == int(np.argmin(dd))
        assert d[k] == pytest.approx(dd.min())


def test_nearest_finds_a_neighbour_across_the_boundary():
    q = np.array([[99.5, 50.0, 50.0]])
    r = np.array([[50.0, 50.0, 50.0], [0.5, 50.0, 50.0]])
    i, d = lhf.nearest(q, r, BOX)
    assert i[0] == 1 and d[0] == pytest.approx(1.0)


def test_nearest_handles_an_empty_reference():
    i, d = lhf.nearest(np.zeros((3, 3)), np.zeros((0, 3)), BOX)
    assert (i == -1).all() and np.isinf(d).all()


def test_cic_and_sample_share_the_cell_centre_convention():
    """A point AT a cell centre must sample that cell's own value back.

    This is the half-cell bug: deposit on corners, read on centres, and every
    potential distance is wrong by 0.78 Mpc/h at ng=64 -- larger than the
    subhalo radii the verdict compares against.
    """
    ng = 8
    cell = BOX / ng
    centres = (np.array([[2, 3, 4]]) + 0.5) * cell
    grid = np.zeros((ng, ng, ng))
    grid[2, 3, 4] = 1.0
    assert lhf.sample_grid(grid, centres, BOX)[0] == pytest.approx(1.0)

    # and a single particle at that centre deposits its whole weight there
    d = lhf.cic_overdensity(centres, ng, BOX)
    peak = np.unravel_index(np.argmax(d), d.shape)
    assert peak == (2, 3, 4)


def test_cic_overdensity_is_mean_zero_and_conserves_mass():
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, BOX, size=(4000, 3))
    d = lhf.cic_overdensity(pos, 8, BOX)
    assert d.mean() == pytest.approx(0.0, abs=1e-12)
    assert d.min() >= -1.0 - 1e-12


def test_potential_is_negative_at_an_overdensity():
    """phi = -delta/k^2 must have its MINIMUM where the mass is.

    The sign is the whole channel: a potential that peaks at a cluster would
    hand the network the opposite feature.
    """
    ng = 16
    delta = np.full((ng, ng, ng), -1.0 / (ng ** 3 - 1))
    delta[8, 8, 8] = 1.0
    delta -= delta.mean()
    phi = lhf.potential_grids(delta, [0.0], BOX)[0.0]
    assert np.unravel_index(np.argmin(phi), phi.shape) == (8, 8, 8)


def test_potential_grids_are_standardised_per_scale():
    rng = np.random.default_rng(2)
    delta = rng.normal(size=(16, 16, 16))
    delta -= delta.mean()
    out = lhf.potential_grids(delta, [0.0, 6.25], BOX)
    assert set(out) == {0.0, 6.25}
    for g in out.values():
        assert g.mean() == pytest.approx(0.0, abs=1e-9)
        assert g.std() == pytest.approx(1.0, rel=1e-9)


def test_smoothing_reduces_the_number_of_local_minima():
    rng = np.random.default_rng(3)
    delta = rng.normal(size=(32, 32, 32))
    delta -= delta.mean()
    g = lhf.potential_grids(delta, [0.0, 6.25], BOX)
    assert lhf.local_minima(g[6.25], BOX).shape[0] < lhf.local_minima(g[0.0], BOX).shape[0]


def test_local_minima_returns_cell_centres_in_mpc():
    ng = 8
    grid = np.ones((ng, ng, ng))
    grid[5, 1, 6] = -1.0
    m = lhf.local_minima(grid, BOX)
    assert m.shape == (1, 3)
    assert np.allclose(m[0], (np.array([5, 1, 6]) + 0.5) * (BOX / ng))


def test_holdout_r2_is_out_of_sample_not_in_sample():
    """Pure noise must NOT score positive; a real linear relation must."""
    rng = np.random.default_rng(4)
    y = rng.normal(size=400)
    noise = rng.normal(size=(400, 4))          # 4 features, zero signal
    assert lhf.holdout_r2(noise, y) < 0.2

    x = rng.normal(size=400)
    assert lhf.holdout_r2(x, 3.0 * x + 1.0) > 0.99


def test_holdout_r2_accepts_either_orientation_and_short_input():
    rng = np.random.default_rng(5)
    x = rng.normal(size=200)
    y = 2.0 * x
    assert lhf.holdout_r2(x, y) == pytest.approx(lhf.holdout_r2(x[None, :], y))
    assert np.isnan(lhf.holdout_r2(np.zeros(5), np.zeros(5)))


def test_holdout_r2_ignores_non_finite_rows():
    rng = np.random.default_rng(6)
    x = rng.normal(size=300)
    y = 2.0 * x + 1.0
    x2 = x.copy()
    x2[:5] = np.nan
    assert lhf.holdout_r2(x2, y) > 0.99


def test_minima_count_collapses_at_the_scale_that_broke_the_first_run():
    """Guard the bug this job actually hit.

    Localising on a 4-cell-smoothed grid left THREE minima in a 100 Mpc/h box,
    so "distance to the nearest" was a 30 Mpc/h number about the smoothing
    rather than the field. A cluster R_vir (~1.0 Mpc/h) is smaller than one LR
    cell (1.5625), so the basin has to survive at cell scale.
    """
    rng = np.random.default_rng(7)
    ng = 32
    delta = rng.normal(size=(ng, ng, ng))
    delta -= delta.mean()
    g = lhf.potential_grids(delta, [1.5625, 6.25], BOX)
    n_cell = lhf.local_minima(g[1.5625], BOX).shape[0]
    n_wide = lhf.local_minima(g[6.25], BOX).shape[0]
    assert n_cell > 5 * max(n_wide, 1)


def test_sample_grid_is_sub_cell():
    """The potential channel's value is that it interpolates.

    A minimum-finder is quantised to the grid; trilinear sampling of a linear
    ramp must recover intermediate values exactly, which is why the subhalo
    probe can beat the LR cell size.
    """
    ng = 8
    cell = BOX / ng
    ramp = np.arange(ng, dtype=np.float64).reshape(ng, 1, 1) * np.ones((1, ng, ng))
    # cell i has its CENTRE at (i+0.5)*cell, so a quarter-cell past centre 3
    # must read 3.25 -- a value no grid-quantised minimum-finder can produce.
    x = np.array([[(3.5 + 0.25) * cell, 4.5 * cell, 4.5 * cell]])
    assert lhf.sample_grid(ramp, x, BOX)[0] == pytest.approx(3.25, abs=1e-9)
    x0 = np.array([[3.5 * cell, 4.5 * cell, 4.5 * cell]])
    assert lhf.sample_grid(ramp, x0, BOX)[0] == pytest.approx(3.0, abs=1e-9)
