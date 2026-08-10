"""Experiment 0: the 64^3 credit decomposition must be exact, not approximate.

Every test here is a property the leave-one-out credit silently depends on. The
synthetic `.particles` writer reproduces Rockstar's *recursive* emission
(``io/meta_io.c::print_child_particles``) rather than a flat table, because the
recursion is the part that is easy to misread: a subhalo's particles appear
twice, once under the subhalo and once under its host, and grouping by the
wrong column strips substructure out of its host's Lagrangian footprint.
"""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.eval.rockstar import HaloCatalog
from cosmo_sr.reward.catalog import CatalogBins
from cosmo_sr.reward.geometry import ChunkGrid, chunk_purity_grid
from cosmo_sr.reward.reward import RewardModel
from cosmo_sr.reward.tiles import (
    MemberWeights,
    TileGrid,
    center_tile_attribution,
    direct_full_box_stats,
    leave_one_out_credit,
    member_weights_from_particles,
    pool_tiles,
    read_tile_summaries,
    tile_of_particle_id,
    tile_summaries,
    write_tile_summaries,
)

NG = 16
TILE = 4          # 4^3 = 64 tiles, the same 8-per-axis ratio as 512/64
BOX = 100.0


@pytest.fixture
def grid() -> TileGrid:
    return TileGrid(ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX)


@pytest.fixture
def tbins() -> CatalogBins:
    return CatalogBins(
        sub_mass_edges=tuple(np.logspace(10.0, 13.0, 4).tolist()),
        host_mass_edges=tuple(np.logspace(12.0, 14.0, 3).tolist()),
        min_sub_particles=4,
        min_host_particles=4,
    )


def lagrangian_ids(coords) -> np.ndarray:
    """Flat C-order ids for a list of (ix, iy, iz), the convention of the dumps."""
    c = np.asarray(coords, dtype=np.int64).reshape(-1, 3) % NG
    return (c[:, 0] * NG + c[:, 1]) * NG + c[:, 2]


def write_particles_file(path, catalog, members):
    """Write a Rockstar-format `.particles` table for ``catalog``.

    ``members[halo_id]`` is the array of particle ids bound *directly* to that
    halo (excluding its substructure). The writer then reproduces Rockstar's
    recursion: for every halo as a root, emit the root's own particles followed
    by all of its descendants'.
    """
    ids = list(map(int, catalog.ids))
    parent = {int(i): int(p) for i, p in zip(catalog.ids, catalog.parent_ids)}
    children = {i: [c for c in ids if parent[c] == i] for i in ids}
    internal = {h: k for k, h in enumerate(ids)}

    lines = ["#Halo table:", "#id internal_id num_p", "#Particle table:",
             "#x y z vx vy vz particle_id assigned_internal_haloid "
             "internal_haloid external_haloid",
             "#Particle table begins here:"]

    def emit(root, node):
        for pid in members[node]:
            lines.append(
                f"0.0 0.0 0.0 0.0 0.0 0.0 {int(pid)} {internal[node]} "
                f"{internal[root]} {root}"
            )
        for ch in children[node]:
            emit(root, ch)

    for h in ids:
        emit(h, h)
    path.write_text("\n".join(lines) + "\n")
    return path


def two_host_catalog():
    """Two hosts (one in each host-mass bin) with 2 and 3 subhalos."""
    ids = [0, 1, 2, 3, 4, 5, 6]
    parent = [-1, -1, 0, 0, 1, 1, 1]
    mvir = [5e12, 5e13, 5e10, 5e10, 5e10, 2e11, 2e11]
    pos = np.array([
        [10.0, 10.0, 10.0],
        [60.0, 60.0, 60.0],
        [10.2, 10.0, 10.0],
        [9.8, 10.1, 10.0],
        [60.2, 60.0, 60.0],
        [59.8, 60.0, 60.1],
        [60.0, 60.3, 60.0],
    ])
    n = len(ids)
    return HaloCatalog(
        ids=np.asarray(ids, dtype=np.int64),
        parent_ids=np.asarray(parent, dtype=np.int64),
        mvir=np.asarray(mvir, dtype=np.float64),
        rvir=np.full(n, 300.0),
        vmax=np.full(n, 150.0),
        pos=pos,
        vel=np.zeros((n, 3)),
        # num_p is each halo's OWN particle list, matching straddling_members():
        # 8 for host 0, 4 for host 1, 4 for each subhalo. The hosts' streamed
        # member sets are larger because the recursion adds their substructure.
        num_p=np.asarray([8, 4, 4, 4, 4, 4, 4], dtype=np.int64),
    )


def straddling_members():
    """Member ids chosen so object 0 straddles four tiles in known proportions.

    Host 0 gets 4 particles in tile (0,0,0) and 4 in tile (1,0,0); its two
    subhalos sit wholly inside one tile each. Host 1 and its subhalos are
    entirely inside one tile, so the sum over tiles has both a split and an
    unsplit case.
    """
    host0_own = lagrangian_ids([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0),
                                (4, 0, 0), (5, 0, 0), (6, 0, 0), (7, 0, 0)])
    sub2 = lagrangian_ids([(0, 1, 0), (0, 2, 0), (1, 1, 0), (1, 2, 0)])
    sub3 = lagrangian_ids([(4, 1, 0), (5, 1, 0), (6, 1, 0), (7, 1, 0)])
    host1_own = lagrangian_ids([(8, 8, 8), (9, 8, 8), (10, 8, 8), (11, 8, 8)])
    sub4 = lagrangian_ids([(8, 9, 8), (9, 9, 8), (10, 9, 8), (11, 9, 8)])
    sub5 = lagrangian_ids([(8, 10, 8), (9, 10, 8), (10, 10, 8), (11, 10, 8)])
    sub6 = lagrangian_ids([(8, 11, 8), (9, 11, 8), (10, 11, 8), (11, 11, 8)])
    return {0: host0_own, 1: host1_own, 2: sub2, 3: sub3,
            4: sub4, 5: sub5, 6: sub6}


@pytest.fixture
def particles_file(tmp_path):
    cat = two_host_catalog()
    p = tmp_path / "halos_0.0.particles"
    write_particles_file(p, cat, straddling_members())
    return cat, p


# ---------------------------------------------------------------- id -> tile


def test_tile_of_particle_id_matches_flat_lagrangian_order(grid):
    """The id convention is the dumps', not an independent guess."""
    coords = [(0, 0, 0), (3, 3, 3), (4, 0, 0), (NG - 1, NG - 1, NG - 1), (5, 9, 13)]
    ids = lagrangian_ids(coords)
    got = tile_of_particle_id(ids, grid)
    want = [grid.index(ix // TILE, iy // TILE, iz // TILE) for ix, iy, iz in coords]
    assert got.tolist() == want


def test_tile_of_particle_id_rejects_out_of_range(grid):
    with pytest.raises(ValueError, match="outside"):
        tile_of_particle_id(np.array([NG ** 3]), grid)


def test_tiles_partition_the_box(grid):
    """Every Lagrangian site belongs to exactly one tile, and volumes sum."""
    all_ids = np.arange(NG ** 3, dtype=np.int64)
    t = tile_of_particle_id(all_ids, grid)
    counts = np.bincount(t, minlength=grid.n_tiles)
    assert counts.tolist() == [TILE ** 3] * grid.n_tiles
    assert grid.n_tiles * grid.tile_volume_mpc3 == pytest.approx(BOX ** 3)


# ------------------------------------------------------- parsing / weights


def test_weights_are_a_partition_of_unity(grid, particles_file):
    """Required test: every object's fractional contributions sum to one."""
    _, p = particles_file
    w = member_weights_from_particles(p, grid)
    sums = w.weight_sums()
    assert set(sums) == {0, 1, 2, 3, 4, 5, 6}
    assert w.max_weight_error() < 1e-12


def test_host_weights_include_substructure(grid, particles_file):
    """The recursion trap: a host's footprint must contain its subhalos'.

    Host 0 owns 8 particles directly and its two subhalos own 4 each, so the
    host's member set is 16 particles. Grouping by ``assigned_internal_haloid``
    would give 8 and a different tile split -- this test is what separates the
    two readings of the file.
    """
    _, p = particles_file
    w = member_weights_from_particles(p, grid)
    assert w.n_members[0] == 16
    assert w.n_members[1] == 16          # 4 own + 3 subhalos x 4
    assert w.n_members[2] == 4

    # Host 0's own particles split 4/4 across tiles x=0 and x=1; sub 2 sits in
    # the x=0 tile and sub 3 in the x=1 tile, so the host ends up 8/8.
    t, ww = w.rows_of(0)
    by_tile = dict(zip(t.tolist(), ww.tolist()))
    assert by_tile[grid.index(0, 0, 0)] == pytest.approx(0.5)
    assert by_tile[grid.index(1, 0, 0)] == pytest.approx(0.5)


def test_straddling_object_is_split_not_dropped(grid, particles_file):
    """The whole point of the redesign: no object is rejected for straddling."""
    _, p = particles_file
    w = member_weights_from_particles(p, grid)
    t, _ = w.rows_of(0)
    assert len(t) == 2, "host 0 straddles two tiles and must appear in both"
    assert w.weight_sums()[0] == pytest.approx(1.0)


def test_chunked_streaming_merges_split_objects(grid, particles_file):
    """A tiny read chunk splits objects across blocks; the result must not change."""
    _, p = particles_file
    ref = member_weights_from_particles(p, grid)
    small = member_weights_from_particles(p, grid, chunk_rows=3)
    assert small.halo_id.tolist() == ref.halo_id.tolist()
    assert small.tile_id.tolist() == ref.tile_id.tolist()
    np.testing.assert_allclose(small.weight, ref.weight)


def test_member_consistency_accepts_the_real_rockstar_pattern(grid, particles_file):
    """rows(o) >= num_p(o), equality only for leaves -- measured on the binary.

    Checking ``rows == num_p`` instead would fail on every host with
    substructure, which is every host this project cares about. The fixture's
    hosts have subhalos and its subhalos do not, so both branches are covered.
    """
    from cosmo_sr.reward.tiles import check_member_consistency

    cat, p = particles_file
    w = member_weights_from_particles(p, grid)
    c = check_member_consistency(cat, w)
    assert c["ok"]
    assert c["n_missing"] == 0
    assert c["n_rows_below_num_p"] == 0
    assert c["n_with_substructure"] == 2      # the two hosts
    assert c["n_leaf_exact"] == 5             # the five subhalos


def test_member_consistency_flags_a_truncated_table(grid, particles_file):
    """A short read must be caught, not averaged into the statistics."""
    from cosmo_sr.reward.tiles import check_member_consistency

    cat, p = particles_file
    w = member_weights_from_particles(p, grid)
    w.n_members[2] = 10 ** 6                  # pretend the catalog claims more
    cat.num_p[cat.ids == 2] = 10 ** 7
    assert not check_member_consistency(cat, w)["ok"]


def test_substructure_below_the_catalog_cut_still_counts_for_its_host(grid, tmp_path):
    """Rockstar's recursion absorbs clumps it refused to print.

    Measured on the binary: a 7914-particle host picked up 51 further particles
    from four 12-14 particle clumps that appear nowhere in the ASCII catalog.
    Those particles are genuinely part of the host's Lagrangian footprint, so
    the host's weights must include them.
    """
    cat = two_host_catalog()
    members = straddling_members()
    p = tmp_path / "halos_0.0.particles"
    write_particles_file(p, cat, members)

    # Append an unprinted 3-particle clump inside host 0 (external id 0, but
    # with no catalog row of its own -- exactly what Rockstar emits).
    extra = lagrangian_ids([(2, 2, 0), (2, 3, 0), (3, 2, 0)])
    lines = [f"0.0 0.0 0.0 0.0 0.0 0.0 {int(q)} 99 0 0" for q in extra]
    p.write_text(p.read_text() + "\n".join(lines) + "\n")

    w = member_weights_from_particles(p, grid)
    assert w.n_members[0] == 19, "the unprinted clump belongs to its host"
    assert w.max_weight_error() < 1e-12
    assert 99 not in w.weight_sums(), "the clump is not an object of its own"


def test_unprinted_halos_are_dropped(grid, tmp_path):
    """external_haloid = -1 marks a halo Rockstar did not print; it is not ours."""
    cat = two_host_catalog()
    p = tmp_path / "halos_0.0.particles"
    write_particles_file(p, cat, straddling_members())
    text = p.read_text().replace(" 0\n", " -1\n")     # unprint halo id 0
    p.write_text(text)
    w = member_weights_from_particles(p, grid)
    assert 0 not in w.weight_sums()
    assert w.max_weight_error() < 1e-12


# ------------------------------------------------- the deliverable identity


def test_tile_sums_reproduce_direct_full_box_stats(grid, tbins, particles_file):
    """Required test: H_b = sum_j H_jb, S_b = sum_j S_jb, O_b = S_b / H_b."""
    cat, p = particles_file
    w = member_weights_from_particles(p, grid)
    ts = tile_summaries(cat, w, tbins, grid, box="set0", source="hr")
    pooled = pool_tiles(ts.values())
    direct = direct_full_box_stats(cat, tbins)

    np.testing.assert_allclose(pooled.n_host, direct["n_host"], atol=1e-12)
    np.testing.assert_allclose(pooled.occ_numerator, direct["occ_numerator"], atol=1e-12)
    np.testing.assert_allclose(pooled.n_sub, direct["n_sub"], atol=1e-12)
    np.testing.assert_allclose(pooled.occupation(), direct["occupation"], atol=1e-12)
    assert pooled.volume_mpc3 == pytest.approx(BOX ** 3)


def test_no_object_is_double_counted(grid, tbins, particles_file):
    """Required test: total attributed host/subhalo weight equals the object count."""
    cat, p = particles_file
    w = member_weights_from_particles(p, grid)
    ts = tile_summaries(cat, w, tbins, grid, box="set0", source="hr")
    assert sum(s.n_host_objects for s in ts.values()) == pytest.approx(2.0)
    assert sum(s.n_sub_objects for s in ts.values()) == pytest.approx(5.0)


def test_tile_order_does_not_change_the_pooled_statistics(grid, tbins, particles_file):
    """Required test: tile ordering cannot change the reward."""
    cat, p = particles_file
    w = member_weights_from_particles(p, grid)
    ts = list(tile_summaries(cat, w, tbins, grid, box="set0", source="hr").values())
    rng = np.random.default_rng(0)
    shuffled = list(ts)
    rng.shuffle(shuffled)
    a, b = pool_tiles(ts), pool_tiles(shuffled)
    np.testing.assert_array_equal(a.n_host, b.n_host)
    np.testing.assert_array_equal(a.occ_numerator, b.occ_numerator)
    np.testing.assert_array_equal(a.n_sub, b.n_sub)


def test_periodic_translation_leaves_the_result_unchanged(grid, tbins, tmp_path):
    """Required test: shifting the box by a whole tile permutes tiles, nothing more.

    A translation by an exact multiple of the tile size relabels tiles by a
    permutation. The pooled statistics -- and therefore any reward -- must be
    bit-identical, and the multiset of per-tile summaries must match too.
    """
    cat = two_host_catalog()
    members = straddling_members()
    p0 = tmp_path / "a.particles"
    write_particles_file(p0, cat, members)

    shift = np.array([TILE, 2 * TILE, 3 * TILE], dtype=np.int64)
    shifted = {}
    for h, ids in members.items():
        c = np.stack([ids // (NG * NG), (ids // NG) % NG, ids % NG], axis=1)
        shifted[h] = lagrangian_ids((c + shift) % NG)
    p1 = tmp_path / "b.particles"
    write_particles_file(p1, cat, shifted)

    def stats(path):
        w = member_weights_from_particles(path, grid)
        return tile_summaries(cat, w, tbins, grid, box="set0", source="hr")

    a, b = stats(p0), stats(p1)
    pa, pb = pool_tiles(a.values()), pool_tiles(b.values())
    np.testing.assert_array_equal(pa.n_host, pb.n_host)
    np.testing.assert_array_equal(pa.occ_numerator, pb.occ_numerator)
    np.testing.assert_array_equal(pa.n_sub, pb.n_sub)

    def multiset(d):
        return sorted(tuple(np.round(s.n_host, 12)) + tuple(np.round(s.occ_numerator, 12))
                      for s in d.values())
    assert multiset(a) == multiset(b)


def test_mismatched_particles_and_catalog_is_an_error(grid, tbins, particles_file):
    """A catalog object with no member rows must raise, never be silently skipped."""
    cat, p = particles_file
    w = member_weights_from_particles(p, grid)
    keep = w.halo_id != 1
    broken = MemberWeights(w.halo_id[keep], w.tile_id[keep], w.weight[keep],
                           {k: v for k, v in w.n_members.items() if k != 1}, w.meta)
    with pytest.raises(ValueError, match="no member-particle rows"):
        tile_summaries(cat, broken, tbins, grid, box="set0", source="hr")


def test_summaries_survive_a_jsonl_roundtrip(grid, tbins, particles_file, tmp_path):
    """Fractional weights must not be floored to int on the way to disk."""
    cat, p = particles_file
    w = member_weights_from_particles(p, grid)
    ts = list(tile_summaries(cat, w, tbins, grid, box="set0", source="hr").values())
    out = write_tile_summaries(tmp_path / "t.jsonl", ts)
    back = read_tile_summaries(out)
    assert len(back) == len(ts)
    a, b = pool_tiles(ts), pool_tiles(back)
    np.testing.assert_allclose(a.n_host, b.n_host, atol=1e-12)
    np.testing.assert_allclose(a.occ_numerator, b.occ_numerator, atol=1e-12)
    # And the fractional entries really are fractional.
    assert any(0.0 < v < 1.0 for s in back for v in s.n_host)


# --------------------------------------------------------- centre fallback


def test_center_tile_fallback_keeps_partition_of_unity(grid, tbins):
    """The fallback flags boundary objects; it does not drop them."""
    cat = two_host_catalog()
    rng = np.random.default_rng(0)
    disp = rng.normal(0.0, 0.02, size=(3, NG, NG, NG)).astype(np.float32)
    purity = chunk_purity_grid(
        disp, chunk_grid=ChunkGrid(ng_hr=NG, chunk_hr=TILE, boxsize_mpc_h=BOX),
        grid=NG,
    )
    w, boundary = center_tile_attribution(cat, purity, min_purity=0.8)
    assert w.max_weight_error() < 1e-12
    assert boundary.shape == (cat.n,)
    ts = tile_summaries(cat, w, tbins, grid, box="set0", source="hr")
    pooled = pool_tiles(ts.values())
    direct = direct_full_box_stats(cat, tbins)
    # One-hot weights still reproduce the full-box totals exactly.
    np.testing.assert_allclose(pooled.n_host, direct["n_host"], atol=1e-12)
    np.testing.assert_allclose(pooled.occ_numerator, direct["occ_numerator"], atol=1e-12)


# ------------------------------------------------------------- LOO credit


def _reward_model(tbins):
    d = tbins.dim
    rng = np.random.default_rng(0)
    mu = rng.normal(0.0, 0.1, size=d)
    cov = np.eye(d) * 0.05
    return RewardModel(mu=mu, cov=cov, lam=1e-3, bins=tbins,
                       ensemble_size=8, n_draws=100, labels=tuple(tbins.labels()))


def test_loo_credit_is_removal_arithmetic_on_cached_stats(grid, tbins, particles_file):
    """A_j = R(S) - R(S - s_j), computed without re-running anything."""
    cat, p = particles_file
    w = member_weights_from_particles(p, grid)
    ts = list(tile_summaries(cat, w, tbins, grid, box="set0", source="hr").values())
    rm = _reward_model(tbins)
    a = leave_one_out_credit(ts, rm.reward)
    assert len(a) == grid.n_tiles
    full = rm.reward(pool_tiles(ts))
    for tid, val in a.items():
        drop = rm.reward(pool_tiles(ts, drop=[tid]))
        assert val == pytest.approx(full - drop)


def test_empty_tiles_get_zero_credit(grid, tbins, particles_file):
    """Most of the 64 tiles hold no objects; removing them cannot change R."""
    cat, p = particles_file
    w = member_weights_from_particles(p, grid)
    ts = list(tile_summaries(cat, w, tbins, grid, box="set0", source="hr").values())
    rm = _reward_model(tbins)
    a = leave_one_out_credit(ts, rm.reward)
    occupied = {int(t) for t in w.tile_id}
    empty = [t for t in range(grid.n_tiles) if t not in occupied]
    assert empty, "the fixture should leave most tiles empty"
    # Removing an empty tile still removes its volume, so abundance shifts a
    # little; the occupation block, which has no volume dependence, must not move.
    a_occ = leave_one_out_credit(ts, rm.reward_occupation)
    for t in empty:
        assert a_occ[t] == pytest.approx(0.0, abs=1e-12)


def test_loo_credit_is_order_independent(grid, tbins, particles_file):
    cat, p = particles_file
    w = member_weights_from_particles(p, grid)
    ts = list(tile_summaries(cat, w, tbins, grid, box="set0", source="hr").values())
    rm = _reward_model(tbins)
    a = leave_one_out_credit(ts, rm.reward)
    rng = np.random.default_rng(1)
    shuffled = list(ts)
    rng.shuffle(shuffled)
    b = leave_one_out_credit(shuffled, rm.reward)
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-12)


def test_majority_weights_preserve_the_additivity_identity():
    """One-hot attribution is still exact: every object carries weight one.

    That is the property the plan's ``sum_t N_t = N`` check rests on, and it is
    the reason collapsing to the majority tile is a variance choice rather than
    a correctness one.
    """
    from cosmo_sr.reward.tiles import MemberWeights, majority_weights

    w = MemberWeights(
        halo_id=np.array([1, 1, 1, 2, 2, 3], dtype=np.int64),
        tile_id=np.array([5, 7, 9, 4, 2, 8], dtype=np.int64),
        weight=np.array([0.2, 0.5, 0.3, 0.4, 0.6, 1.0]),
        n_members={1: 100, 2: 50, 3: 10}, meta={"source": "member_particle_ids"})
    m = majority_weights(w)

    assert list(m.tile_id) == [7, 2, 8]
    assert m.max_weight_error() == 0.0
    assert set(m.n_members) == set(w.n_members)
    assert m.meta["attribution"] == "majority"


def test_majority_weights_break_ties_deterministically():
    """Equal shares go to the lowest tile id, both times it is called."""
    from cosmo_sr.reward.tiles import MemberWeights, majority_weights

    w = MemberWeights(
        halo_id=np.array([1, 1], dtype=np.int64),
        tile_id=np.array([9, 3], dtype=np.int64),
        weight=np.array([0.5, 0.5]), n_members={1: 8}, meta={})
    assert list(majority_weights(w).tile_id) == [3]
    assert list(majority_weights(w).tile_id) == [3]


def test_majority_tile_summaries_still_sum_to_the_direct_whole_box(bins):
    """The identity every label rests on holds under BOTH attribution schemes."""
    from cosmo_sr.reward.tiles import (
        TileGrid, direct_full_box_stats, majority_weights, tile_summaries,
    )
    from tests.reward.conftest import synthetic_catalog

    grid = TileGrid(ng_hr=16, tile_hr=8, boxsize_mpc_h=100.0)
    cat = synthetic_catalog([((10.0, 10.0, 10.0), 5e13), ((60.0, 60.0, 60.0), 2e13)],
                            [2, 1], boxsize=100.0)
    rng = np.random.default_rng(0)
    halo, tile, weight = [], [], []
    for h in cat.ids:
        tiles = rng.choice(grid.n_tiles, size=3, replace=False)
        w = rng.dirichlet(np.ones(3))
        halo += [int(h)] * 3
        tile += [int(t) for t in tiles]
        weight += [float(x) for x in w]
    mw = MemberWeights(np.asarray(halo, dtype=np.int64),
                       np.asarray(tile, dtype=np.int64), np.asarray(weight),
                       {int(h): 30 for h in cat.ids}, {"source": "test"})

    direct = direct_full_box_stats(cat, bins)
    for w in (mw, majority_weights(mw)):
        s = tile_summaries(cat, w, bins, grid, box="set0", source="t")
        for key in ("n_sub", "n_host", "occ_numerator"):
            pooled = np.sum([getattr(s[t], key) for t in s], axis=0)
            assert np.allclose(pooled, direct[key], atol=1e-9), key
