"""The LR->HR mapping and the normalisations the host features rest on.

Every check here is an identity the construction must satisfy exactly, not a
tolerance on a fit: the whole point of going through particle ids is that the
Lagrangian origin of a bound particle is known rather than estimated. A failure
means a coordinate convention drifted, which is silent in a plot and fatal in a
conditioning signal.
"""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.eval.rockstar import HaloCatalog
from cosmo_sr.features.lagrangian_host import (
    LagrangianGrid, build_host_features, lagrangian_lattice_positions,
    normalization_report, periodic_circular_mean,
)
from cosmo_sr.reward.tiles import tile_of_particle_id

# Small stand-in for the real (64 -> 512, tile 64) geometry: same upsample
# factor of 8, same "tile is a whole number of LR cells" relation, 4^3 tiles.
GRID = LagrangianGrid(ng_lr=16, ng_hr=128, tile_hr=32, boxsize_mpc_h=100.0)


def _sites(coords) -> np.ndarray:
    """LR particle ids of ``(a,b,c)`` triples, C-order like field_to_particles."""
    n = GRID.ng_lr
    return np.array([(a * n + b) * n + c for a, b, c in coords], dtype=np.int64)


def _toy_box():
    """Three hosts: compact, tile-straddling, and box-edge-wrapping.

    Host 20 also carries a subhalo (id 21) so that leaf attribution and host
    attribution genuinely differ and ``remap_to_roots`` is exercised.
    """
    cat = HaloCatalog(
        ids=np.array([10, 20, 21, 30], dtype=np.int64),
        parent_ids=np.array([-1, -1, 20, -1], dtype=np.int64),
        mvir=np.array([1e13, 5e13, 2e12, 1e12], dtype=np.float64),
        rvir=np.array([300.0, 500.0, 120.0, 90.0], dtype=np.float64),
        vmax=np.zeros(4), pos=np.zeros((4, 3)), vel=np.zeros((4, 3)),
        num_p=np.array([8, 12, 4, 3], dtype=np.int64), path="toy",
    )
    owner = np.full(GRID.n_lr, -1, dtype=np.int32)

    # Host 10: 2x2x2 block wholly inside one tile.
    h10 = _sites([(a, b, c) for a in (1, 2) for b in (1, 2) for c in (1, 2)])
    owner[h10] = 10
    # Host 20: straddles the tile boundary at LR index 4 (tile_lr = 4).
    h20 = _sites([(3, 5, 5), (4, 5, 5), (3, 6, 5), (4, 6, 5)])
    owner[h20] = 20
    # ... and its subhalo, which host attribution must fold into host 20.
    h21 = _sites([(5, 5, 5), (5, 6, 5)])
    owner[h21] = 21
    # Host 30: wraps the periodic box edge on the x axis.
    h30 = _sites([(15, 9, 9), (0, 9, 9), (15, 10, 9), (0, 10, 9)])
    owner[h30] = 30
    return cat, owner, {10: h10, 20: np.concatenate([h20, h21]), 30: h30}


@pytest.fixture(scope="module")
def toy():
    cat, owner, members = _toy_box()
    feat = build_host_features(cat, owner, GRID, box="toy", source="test")
    return cat, owner, members, feat


# --------------------------------------------------------------------------
# 1. LR -> HR children
# --------------------------------------------------------------------------

def test_every_lr_site_has_exactly_upsample_cubed_children():
    f = GRID.upsample
    seen = set()
    for lr_id in (0, 1, 17, GRID.n_lr // 2, GRID.n_lr - 1):
        kids = GRID.hr_children(lr_id)
        assert kids.size == f ** 3
        assert np.unique(kids).size == f ** 3
        assert kids.min() >= 0 and kids.max() < GRID.ng_hr ** 3
        seen.update(int(k) for k in kids)
    # Distinct parents cannot share a child, so the union is the full count.
    assert len(seen) == 5 * f ** 3


def test_children_partition_the_hr_lattice():
    """Over a whole tile, the children tile the HR block with no gaps."""
    sx, sy, sz = GRID.lr_slices(0)
    lr_ids = np.arange(GRID.n_lr).reshape((GRID.ng_lr,) * 3)[sx, sy, sz].reshape(-1)
    kids = np.concatenate([GRID.hr_children(int(i)) for i in lr_ids])
    assert np.unique(kids).size == kids.size == GRID.tile_hr ** 3


# --------------------------------------------------------------------------
# 2. Host mass is a host property, not a particle property
# --------------------------------------------------------------------------

def test_log_host_mass_constant_within_a_host(toy):
    cat, _, members, feat = toy
    flat = feat.log_host_mass.reshape(-1)
    for hid, ids in members.items():
        vals = flat[ids]
        assert np.all(vals == vals[0])
        want = np.log10(float(cat.mvir[list(cat.ids).index(hid)]))
        assert vals[0] == pytest.approx(want, abs=1e-5)


def test_subhalo_particles_carry_the_host_mass(toy):
    """The satellite's own 2e12 must not leak in: leaf ownership is lifted."""
    _, _, _, feat = toy
    sub_sites = _sites([(5, 5, 5), (5, 6, 5)])
    assert feat.log_host_mass.reshape(-1)[sub_sites] == pytest.approx(
        np.log10(5e13), abs=1e-5)


def test_unowned_sites_are_exactly_zero(toy):
    _, _, _, feat = toy
    off = feat.host_member.reshape(-1) == 0
    assert np.all(feat.log_host_mass.reshape(-1)[off] == 0.0)
    assert np.all(feat.dq_over_rl.reshape(3, -1)[:, off] == 0.0)
    assert np.all(feat.host_fraction_per_tile.reshape(-1)[off] == 0.0)
    assert np.all(feat.host_index.reshape(-1)[off] == -1)


# --------------------------------------------------------------------------
# 3. Periodic offsets
# --------------------------------------------------------------------------

def test_circular_mean_handles_the_box_seam():
    box = 100.0
    pts = np.array([[99.0, 1.0, 50.0], [1.0, 99.0, 50.0]])
    c = periodic_circular_mean(pts, box)
    assert c[0] == pytest.approx(0.0, abs=1e-9) or c[0] == pytest.approx(box, abs=1e-9)
    assert c[2] == pytest.approx(50.0, abs=1e-9)


def test_dq_over_rl_matches_a_direct_periodic_computation(toy):
    """Recompute offsets from the lattice, independently of the builder."""
    _, _, members, feat = toy
    q = lagrangian_lattice_positions(GRID)
    box = GRID.boxsize_mpc_h
    dq = feat.dq_over_rl.reshape(3, -1)
    for hid, ids in members.items():
        row = feat.table.row_of(hid)
        c = feat.table.center_lag[row]
        r = feat.table.r_lag_mpc_h[row]
        d = q[ids] - c[None, :]
        d -= box * np.round(d / box)             # wrap into [-box/2, box/2)
        assert dq[:, ids].T == pytest.approx(d / r, abs=1e-5)


def test_wrapping_host_offsets_stay_small(toy):
    """The edge-wrapping host must not be spread across half the box."""
    _, _, members, feat = toy
    ids = members[30]
    off = np.linalg.norm(feat.dq_over_rl.reshape(3, -1)[:, ids], axis=0)
    # Two cells apart at most; a non-periodic centre would give |dq|/r ~ 100.
    assert off.max() < 5.0


# --------------------------------------------------------------------------
# 4. Tile fractions
# --------------------------------------------------------------------------

def test_tile_fractions_sum_to_one(toy):
    _, _, _, feat = toy
    sums = feat.table.tile_frac.astype(np.float64).sum(axis=1)
    assert sums == pytest.approx(np.ones(feat.table.n_hosts), abs=1e-6)


def test_tile_fractions_count_the_right_particles(toy):
    _, _, members, feat = toy
    for hid, ids in members.items():
        row = feat.table.row_of(hid)
        tiles = GRID.tile_of_lr_site(ids)
        want = np.bincount(tiles, minlength=GRID.n_tiles) / ids.size
        assert feat.table.tile_frac[row] == pytest.approx(want, abs=1e-6)


def test_a_straddling_host_really_spans_two_tiles(toy):
    _, _, _, feat = toy
    row = feat.table.row_of(20)
    tiles, fracs = feat.table.tiles_of(row)
    assert tiles.size >= 2
    assert fracs.sum() == pytest.approx(1.0, abs=1e-6)


def test_per_site_fraction_is_the_site_s_own_tile(toy):
    _, _, members, feat = toy
    flat = feat.host_fraction_per_tile.reshape(-1)
    for hid, ids in members.items():
        row = feat.table.row_of(hid)
        want = feat.table.tile_frac[row][GRID.tile_of_lr_site(ids)]
        assert flat[ids] == pytest.approx(want, abs=1e-6)


# --------------------------------------------------------------------------
# 5. Subhalo budget
# --------------------------------------------------------------------------

def test_budget_sums_to_n_h(toy):
    cat, owner, _, _ = toy
    want = {10: 3, 20: 7, 30: 0}
    feat = build_host_features(cat, owner, GRID, n_sub_per_host=want)
    flat_row = feat.host_index.reshape(-1)
    flat_lam = feat.subhalo_budget.reshape(-1).astype(np.float64)
    for hid, n_h in want.items():
        row = feat.table.row_of(hid)
        assert flat_lam[flat_row == row].sum() == pytest.approx(n_h, abs=1e-4)


def test_budget_is_uniform_over_a_host_s_particles(toy):
    cat, owner, members, _ = toy
    feat = build_host_features(cat, owner, GRID, n_sub_per_host={10: 3, 20: 7})
    lam = feat.subhalo_budget.reshape(-1)[members[10]]
    assert np.all(lam == lam[0])
    assert lam[0] == pytest.approx(3.0 / members[10].size, abs=1e-6)


def test_normalization_report_is_clean(toy):
    _, _, _, feat = toy
    rep = normalization_report(feat)
    assert rep["ok"], rep
    assert rep["max_abs_tile_frac_error"] < 1e-6
    assert rep["max_abs_budget_error"] < 1e-3
    assert rep["n_hosts"] == 3


# --------------------------------------------------------------------------
# 6. Cropping / HR broadcasting vs the SR2 tile coordinates
# --------------------------------------------------------------------------

def test_lr_site_and_its_hr_children_share_a_tile():
    """The two TileGrids must agree, else a feature lands in the wrong tile."""
    hr_grid = GRID.hr_tile_grid()
    rng = np.random.default_rng(0)
    for lr_id in rng.choice(GRID.n_lr, size=40, replace=False):
        t_lr = int(GRID.tile_of_lr_site(np.array([lr_id]))[0])
        t_hr = tile_of_particle_id(GRID.hr_children(int(lr_id)), hr_grid)
        assert np.all(t_hr == t_lr)


def test_tile_crop_matches_the_tile_s_lr_sites(toy):
    _, _, _, feat = toy
    stack = feat.stack_lr()
    for tile in (0, 5, GRID.n_tiles - 1):
        sx, sy, sz = GRID.lr_slices(tile)
        assert np.array_equal(feat.tile_lr(tile), stack[:, sx, sy, sz])
        # The crop's sites are exactly those the tile grid assigns to `tile`.
        ids = np.arange(GRID.n_lr).reshape((GRID.ng_lr,) * 3)[sx, sy, sz].reshape(-1)
        assert np.all(GRID.tile_of_lr_site(ids) == tile)


def test_hr_broadcast_gives_each_child_its_parent_value(toy):
    _, _, _, feat = toy
    f, s = GRID.upsample, GRID.tile_lr
    for tile in (0, 5, GRID.n_tiles - 1):
        lr = feat.tile_lr(tile)
        hr = feat.tile_hr(tile)
        assert hr.shape == (feat.n_channels,) + (GRID.tile_hr,) * 3
        # Blockwise constant, and the block value is the parent LR value.
        blocks = hr.reshape(feat.n_channels, s, f, s, f, s, f)
        assert np.array_equal(blocks[:, :, 0, :, 0, :, 0], lr)
        assert np.all(blocks == blocks[:, :, :1, :, :1, :, :1])


def test_no_dense_hr_volume_is_ever_built(toy):
    """A tile is 1/n_tiles of the box; the HR stack of the box is never made."""
    _, _, _, feat = toy
    assert feat.stack_lr().shape[1:] == (GRID.ng_lr,) * 3
    assert feat.tile_hr(0).nbytes * GRID.n_tiles > feat.stack_lr().nbytes


def test_roundtrip_npz(tmp_path, toy):
    _, _, _, feat = toy
    p = feat.to_npz(tmp_path / "feat.npz")
    back = type(feat).from_npz(p)
    assert np.array_equal(back.log_host_mass, feat.log_host_mass)
    assert np.array_equal(back.dq_over_rl, feat.dq_over_rl)
    assert np.array_equal(back.table.tile_frac, feat.table.tile_frac)
    assert back.grid == feat.grid
    assert back.table.n_sub_source == feat.table.n_sub_source


# --------------------------------------------------------------------------
# 7. The per-tile read
# --------------------------------------------------------------------------

def test_hosts_in_tile_accounts_for_every_occupied_site(toy):
    _, _, _, feat = toy
    names = feat.channel_names()
    for tile in range(GRID.n_tiles):
        rows = feat.hosts_in_tile(tile)
        crop = feat.tile_lr(tile)[names.index("host_member")]
        assert sum(r["n_sites_here"] for r in rows) == int(crop.sum())


def test_hosts_in_tile_reports_both_halves_of_the_overlap(toy):
    """n_sites_here is the tile's view; frac_of_host is the host's view."""
    _, _, members, feat = toy
    tile = int(GRID.tile_of_lr_site(members[10])[0])
    row = next(r for r in feat.hosts_in_tile(tile) if r["host_id"] == 10)
    assert row["n_particles"] == members[10].size
    assert row["frac_of_host"] == pytest.approx(
        row["n_sites_here"] / row["n_particles"], abs=1e-9)
    # ... and it agrees with the host table's own tile fraction.
    k = feat.table.row_of(10)
    assert row["frac_of_host"] == pytest.approx(
        float(feat.table.tile_frac[k][tile]), abs=1e-6)


def test_hosts_in_tile_is_mass_ordered_and_empty_where_empty(toy):
    _, _, _, feat = toy
    for tile in range(GRID.n_tiles):
        rows = feat.hosts_in_tile(tile)
        m = [r["mvir"] for r in rows]
        assert m == sorted(m, reverse=True)
        crop = feat.tile_lr(tile)[feat.channel_names().index("host_member")]
        assert (rows == []) == (crop.sum() == 0)


def test_hosts_in_tile_separates_two_hosts_sharing_one_tile():
    """The case the read exists for: one tile, several hosts, none of them whole.

    The toy fixture's hosts happen to land in different tiles, so this builds
    the overlap explicitly rather than depending on that.
    """
    cat = HaloCatalog(
        ids=np.array([1, 2], dtype=np.int64),
        parent_ids=np.array([-1, -1], dtype=np.int64),
        mvir=np.array([1e13, 4e13], dtype=np.float64),
        rvir=np.array([200.0, 400.0]), vmax=np.zeros(2), pos=np.zeros((2, 3)),
        vel=np.zeros((2, 3)), num_p=np.array([4, 6], dtype=np.int64), path="pair",
    )
    owner = np.full(GRID.n_lr, -1, dtype=np.int32)
    # Both inside tile (0,0,0) (cells 0..3), plus a tail of host 2 in the next
    # tile along x, so neither host is wholly contained in the shared tile.
    owner[_sites([(0, 0, 0), (0, 0, 1), (1, 0, 0)])] = 1
    owner[_sites([(2, 2, 2), (3, 2, 2)])] = 2
    owner[_sites([(4, 2, 2), (5, 2, 2)])] = 2
    feat = build_host_features(cat, owner, GRID)

    tile = int(GRID.tile_of_lr_site(_sites([(0, 0, 0)]))[0])
    rows = feat.hosts_in_tile(tile)
    assert [r["host_id"] for r in rows] == [2, 1]        # mass-ordered
    by_id = {r["host_id"]: r for r in rows}
    assert by_id[1]["n_sites_here"] == 3 and by_id[1]["frac_of_host"] == 1.0
    assert by_id[2]["n_sites_here"] == 2
    assert by_id[2]["n_particles"] == 4
    assert by_id[2]["frac_of_host"] == pytest.approx(0.5)
    assert by_id[2]["n_tiles_hit"] == 2
    # The tile's own occupancy is the sum, and the two hosts do not overlap.
    names = feat.channel_names()
    assert (sum(r["n_sites_here"] for r in rows)
            == int(feat.tile_lr(tile)[names.index("host_member")].sum()) == 5)
