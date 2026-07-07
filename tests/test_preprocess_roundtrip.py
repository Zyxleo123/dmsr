import numpy as np
import pytest

from cosmo_sr.data.preprocess_srs import (
    pos_to_displacement,
    displacement_to_pos,
    particles_to_field,
)


BOX = 100.0
NG = 8


def _lattice_positions(Ng, boxsize):
    cellsize = boxsize / Ng
    lat = np.arange(Ng) * cellsize + 0.5 * cellsize
    X, Y, Z = np.meshgrid(lat, lat, lat, indexing="ij")
    return np.stack([X, Y, Z], axis=-1)


def test_zero_displacement():
    pos = _lattice_positions(NG, BOX)
    disp = pos_to_displacement(pos.copy(), BOX, NG)
    np.testing.assert_allclose(disp, 0.0, atol=1e-9)


def test_displacement_pos_roundtrip():
    rng = np.random.default_rng(0)
    disp_true = (rng.uniform(-0.4, 0.4, size=(NG, NG, NG, 3)) * BOX).astype(np.float64)
    pos = displacement_to_pos(disp_true.copy(), BOX, NG)
    assert np.all(pos >= 0) and np.all(pos < BOX + 1e-9)
    disp_rec = pos_to_displacement(pos.copy(), BOX, NG)
    np.testing.assert_allclose(disp_rec, disp_true, atol=1e-6)


def test_particles_to_field_shuffle_invariant():
    rng = np.random.default_rng(2)
    npart = NG ** 3
    ids = np.arange(npart)
    positions = rng.uniform(0, BOX, size=(npart, 3))
    velocities = rng.standard_normal((npart, 3)) * 10.0

    field_ref = particles_to_field(positions, velocities, ids, BOX, redshift=0.0)
    perm = rng.permutation(npart)
    field_shuf = particles_to_field(
        positions[perm], velocities[perm], ids[perm], BOX, redshift=0.0
    )
    np.testing.assert_allclose(field_ref, field_shuf, atol=1e-6)


def test_field_shape_and_dtype():
    rng = np.random.default_rng(3)
    npart = NG ** 3
    ids = np.arange(npart)
    positions = rng.uniform(0, BOX, size=(npart, 3))
    velocities = rng.standard_normal((npart, 3))
    field = particles_to_field(positions, velocities, ids, BOX, redshift=0.5)
    assert field.shape == (6, NG, NG, NG)
    assert field.dtype == np.float32


def test_periodic_wrap_minimal_displacement():
    # grid point (0,0,0) lattice x = 0.5*cellsize = 6.25 for Ng=8, box=100
    cellsize = BOX / NG
    pos = _lattice_positions(NG, BOX).copy()
    # place particle slightly beyond the box boundary along x
    pos[0, 0, 0, 0] = BOX + 2.0  # true wrapped position is 2.0
    disp = pos_to_displacement(pos, BOX, NG)
    expected = 2.0 - (0.5 * cellsize)  # minimal periodic displacement
    assert disp[0, 0, 0, 0] == pytest.approx(expected, abs=1e-6)
    # other grid points unaffected
    np.testing.assert_allclose(disp[1, 1, 1], 0.0, atol=1e-9)
