"""Chunk geometry, the Eulerian purity grid, and the documented core mask."""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.reward.geometry import (ChunkGrid, PurityGrid, assign_halos_to_chunks,
                                      chunk_purity_grid, lagrangian_patch_diameter_mpc)


def test_chunk_size_is_justified_by_the_halo_masses_we_score():
    # 25 Mpc/h chunks: a 1e14 host's Lagrangian patch fits, a 1e15 does not, and
    # the mask is expected to drop the latter rather than mix chunks.
    assert lagrangian_patch_diameter_mpc(1e14) < 25.0
    assert lagrangian_patch_diameter_mpc(1e15) > 25.0


def test_chunks_tile_the_box_exactly_and_ids_round_trip():
    g = ChunkGrid(ng_hr=512, chunk_hr=128)
    assert g.n_chunks == 64
    covered = np.zeros((512,) * 3, dtype=np.int8)
    for cid, sl in g.iter_slices():
        covered[sl] += 1
        assert g.index(*g.coord(cid)) == cid
    assert (covered == 1).all()


def test_chunk_hr_must_divide_the_box():
    with pytest.raises(ValueError):
        ChunkGrid(ng_hr=512, chunk_hr=100)


def test_zero_displacement_makes_every_cell_perfectly_pure():
    g = ChunkGrid(ng_hr=16, chunk_hr=8, boxsize_mpc_h=10.0)
    disp = np.zeros((3, 16, 16, 16), dtype=np.float32)
    p = chunk_purity_grid(disp, chunk_grid=g, grid=8, dis_norm_kpc_h=0.0)
    assert (p.majority_frac[p.total > 0] == 1.0).all()
    assert p.majority_id.min() >= 0
    vols = p.effective_volume_mpc3(0.8)
    assert vols.shape == (8,)
    assert np.isclose(vols.sum(), 10.0 ** 3)


def test_large_displacement_mixes_chunks_and_shrinks_the_core_volume():
    g = ChunkGrid(ng_hr=16, chunk_hr=8, boxsize_mpc_h=10.0)
    rng = np.random.default_rng(0)
    disp = rng.normal(0, 1.0, size=(3, 16, 16, 16)).astype(np.float32)
    p = chunk_purity_grid(disp, chunk_grid=g, grid=8, dis_norm_kpc_h=6000.0)
    mixed = p.effective_volume_mpc3(0.8).sum()
    assert mixed < 10.0 ** 3, "scrambled particles cannot leave every cell pure"


def _uniform_purity(grid=8, box=10.0, cid=0, frac=1.0):
    return PurityGrid(
        majority_id=np.full((grid,) * 3, cid, dtype=np.int16),
        majority_frac=np.full((grid,) * 3, frac, dtype=np.float32),
        total=np.ones((grid,) * 3, dtype=np.float32),
        grid=grid, boxsize_mpc_h=box, chunk_hr=8, ng_hr=16,
    )


def test_a_halo_inside_a_pure_region_is_assigned():
    p = _uniform_purity()
    out = assign_halos_to_chunks(np.array([[5.0, 5.0, 5.0]]), np.array([100.0]), p)
    assert out[0] == 0


def test_a_halo_straddling_an_impure_cell_is_rejected():
    p = _uniform_purity()
    p.majority_frac[3, 3, 3] = 0.4        # one bad cell inside the sphere
    # cell size = 10/8 = 1.25 Mpc/h; rvir = 1500 kpc/h spans one cell either way
    out = assign_halos_to_chunks(np.array([[4.0, 4.0, 4.0]]), np.array([1500.0]), p)
    assert out[0] == -1


def test_a_halo_spanning_two_chunks_is_rejected():
    p = _uniform_purity()
    p.majority_id[4:, :, :] = 1
    out = assign_halos_to_chunks(np.array([[5.0, 5.0, 5.0]]), np.array([1500.0]), p)
    assert out[0] == -1


def test_cluster_scale_halos_exceeding_max_half_width_are_rejected():
    p = _uniform_purity()
    out = assign_halos_to_chunks(np.array([[5.0, 5.0, 5.0]]), np.array([20000.0]), p,
                                 max_half_width=2)
    assert out[0] == -1, "a halo wider than the mask window must not be assigned by fiat"


def test_assignment_is_periodic_at_the_box_edge():
    p = _uniform_purity()
    out = assign_halos_to_chunks(np.array([[9.99, 0.01, 5.0]]), np.array([100.0]), p)
    assert out[0] == 0


def test_effective_volume_counts_only_cells_that_pass_the_same_test():
    p = _uniform_purity()
    p.majority_frac[0, 0, 0] = 0.1
    v = p.effective_volume_mpc3(0.8)
    cell = (10.0 / 8) ** 3
    assert v[0] == pytest.approx(10.0 ** 3 - cell)


def test_purity_grid_round_trips_through_npz(tmp_path):
    p = _uniform_purity()
    p.to_npz(tmp_path / "p.npz")
    back = PurityGrid.from_npz(tmp_path / "p.npz")
    assert back.grid == p.grid and back.ng_hr == p.ng_hr
    assert np.array_equal(back.majority_id, p.majority_id)


def test_field_grid_must_match_the_chunk_grid():
    g = ChunkGrid(ng_hr=32, chunk_hr=8)
    with pytest.raises(ValueError, match="ng_hr"):
        chunk_purity_grid(np.zeros((3, 16, 16, 16), dtype=np.float32),
                          chunk_grid=g, grid=8)
