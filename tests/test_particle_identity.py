"""SR2-vs-HR particle identity: the exact-correspondence claims, pinned.

The synthetic `.particles` writer reproduces Rockstar's *recursive* emission
(``io/meta_io.c::print_child_particles``), because the whole leaf-attribution
argument rests on that recursion: a satellite's particle is printed under the
satellite AND under every ancestor, and only the row where the recursion root is
the binding object identifies the object that really owns it.
"""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.eval.halo_match import match_hosts
from cosmo_sr.eval.particle_identity import (
    as_flat_catalog,
    build_owner_index,
    check_owner_consistency,
    child_map,
    descendants_of,
    displacement_stats,
    eulerian_chunk_shift,
    periodic_delta,
    profile_overlap,
    radius_fractions,
    remap_to_roots,
    set_metrics,
    stream_owner_assignment,
    tile_profile,
)
from cosmo_sr.eval.rockstar import HaloCatalog
from cosmo_sr.reward.tiles import TileGrid

NG = 16
N_PART = NG ** 3
BOX = 100.0


def lagrangian_ids(coords) -> np.ndarray:
    c = np.asarray(coords, dtype=np.int64).reshape(-1, 3) % NG
    return (c[:, 0] * NG + c[:, 1]) * NG + c[:, 2]


def write_particles_file(path, catalog, members, *, duplicate_leaf=False):
    """Rockstar-format `.particles` table with the real recursion.

    ``members[halo_id]`` are the ids bound *directly* to that halo. Every halo
    is visited as a recursion root and emits its own particles followed by all
    of its descendants', so a satellite particle appears once per ancestor.
    """
    ids = list(map(int, catalog.ids))
    parent = {int(i): int(p) for i, p in zip(catalog.ids, catalog.parent_ids)}
    children = {i: [c for c in ids if parent[c] == i] for i in ids}
    internal = {h: k for k, h in enumerate(ids)}

    lines = ["#Halo table:", "#Particle table:",
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
    if duplicate_leaf:
        # Two different objects both claiming to be the binding root of one
        # particle -- the corruption stream_owner_assignment must refuse.
        pid = int(members[ids[0]][0])
        lines.append(f"0.0 0.0 0.0 0.0 0.0 0.0 {pid} {internal[ids[1]]} "
                     f"{internal[ids[1]]} {ids[1]}")
    path.write_text("\n".join(lines) + "\n")
    return path


def nested_catalog():
    """Host 0 with subhalo 1, which itself has sub-subhalo 2; host 3 alone."""
    ids = [0, 1, 2, 3]
    parent = [-1, 0, 1, -1]
    pos = np.array([[10.0, 10.0, 10.0], [10.4, 10.0, 10.0],
                    [10.5, 10.0, 10.0], [60.0, 60.0, 60.0]])
    return HaloCatalog(
        ids=np.asarray(ids, dtype=np.int64),
        parent_ids=np.asarray(parent, dtype=np.int64),
        mvir=np.asarray([1e13, 1e11, 5e10, 5e12], dtype=np.float64),
        rvir=np.asarray([500.0, 100.0, 60.0, 400.0]),      # kpc/h
        vmax=np.full(4, 100.0),
        pos=pos, vel=np.zeros((4, 3)),
        # num_p is each halo's OWN list, which is exactly what leaf attribution
        # must reproduce.
        num_p=np.asarray([8, 4, 3, 5], dtype=np.int64),
    )


def nested_members():
    return {
        0: lagrangian_ids([(i, 0, 0) for i in range(8)]),
        1: lagrangian_ids([(i, 1, 0) for i in range(4)]),
        2: lagrangian_ids([(i, 2, 0) for i in range(3)]),
        3: lagrangian_ids([(8, 8, j) for j in range(5)]),
    }


@pytest.fixture
def owned(tmp_path):
    cat = nested_catalog()
    mem = nested_members()
    p = write_particles_file(tmp_path / "box.particles", cat, mem)
    owner = stream_owner_assignment(p, N_PART)
    return cat, mem, owner


# --------------------------------------------------------------------------
# Ownership
# --------------------------------------------------------------------------

def test_leaf_attribution_beats_the_recursion(owned):
    """A satellite particle is owned by the satellite, not by its ancestors.

    Grouping by ``external_haloid`` instead would hand every one of subhalo 1's
    particles to host 0 as well, and the id sets this module compares would
    stop being the objects' own particle lists.
    """
    cat, mem, owner = owned
    for hid, ids in mem.items():
        assert np.all(owner[ids] == hid)
    assert int(np.count_nonzero(owner >= 0)) == sum(v.size for v in mem.values())


def test_member_counts_equal_catalog_num_p(owned):
    cat, mem, owner = owned
    idx = build_owner_index(owner)
    rep = check_owner_consistency(cat, idx)
    assert rep["ok"]
    assert rep["n_exact"] == cat.n
    assert rep["max_abs_diff"] == 0
    assert idx.n_unowned == N_PART - sum(v.size for v in mem.values())


def test_two_objects_claiming_one_particle_is_refused(tmp_path):
    cat = nested_catalog()
    p = write_particles_file(tmp_path / "bad.particles", cat, nested_members(),
                             duplicate_leaf=True)
    with pytest.raises(ValueError, match="claimed by two objects"):
        stream_owner_assignment(p, N_PART)


def test_id_outside_the_lattice_is_refused(tmp_path):
    cat = nested_catalog()
    mem = nested_members()
    mem[3] = np.append(mem[3], N_PART + 5)
    p = write_particles_file(tmp_path / "oob.particles", cat, mem)
    with pytest.raises(ValueError, match="outside"):
        stream_owner_assignment(p, N_PART)


def test_members_with_substructure_is_the_full_footprint(owned):
    cat, mem, owner = owned
    idx = build_owner_index(owner)
    ch = child_map(cat)
    assert sorted(descendants_of(cat, 0, children=ch)) == [1, 2]
    full = idx.members_with_substructure(cat, 0, children=ch)
    expect = np.unique(np.concatenate([mem[0], mem[1], mem[2]]))
    assert np.array_equal(full, expect)
    # own list alone is strictly smaller: that difference IS the substructure
    assert idx.members(0).size == cat.num_p[0]


def test_root_remap_keeps_a_host_whole(owned):
    """Leaf ownership fragments a host across its own satellites; the root
    remap is what makes the host-level comparison mean what it says."""
    cat, mem, owner = owned
    leaf = {int(h) for h in np.unique(owner[np.concatenate(
        [mem[0], mem[1], mem[2]])])}
    assert leaf == {0, 1, 2}                    # three leaf owners...

    rooted = remap_to_roots(owner, cat)
    inside = np.unique(rooted[np.concatenate([mem[0], mem[1], mem[2]])])
    assert inside.tolist() == [0]               # ...but one host
    assert np.all(rooted[mem[3]] == 3)          # a host with no subs is itself
    assert np.all(rooted[owner < 0] == -1)      # unbound stays unbound


def test_unknown_halo_has_no_members(owned):
    _, _, owner = owned
    idx = build_owner_index(owner)
    assert idx.members(999).size == 0


def test_streaming_chunk_size_does_not_change_the_answer(tmp_path):
    cat = nested_catalog()
    p = write_particles_file(tmp_path / "box.particles", cat, nested_members())
    a = stream_owner_assignment(p, N_PART, chunk_rows=3)
    b = stream_owner_assignment(p, N_PART, chunk_rows=10_000)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# Granularity 1: identity
# --------------------------------------------------------------------------

def test_set_metrics_directions():
    a = np.arange(10)
    b = np.arange(5, 20)
    m = set_metrics(a, b)
    assert m["n_shared"] == 5
    assert m["completeness"] == pytest.approx(0.5)     # half of HR recovered
    assert m["purity"] == pytest.approx(5 / 15)        # third of SR2 is real
    assert m["jaccard"] == pytest.approx(5 / 20)
    assert set_metrics(a, a)["jaccard"] == 1.0
    assert set_metrics(a, np.arange(100, 110))["jaccard"] == 0.0
    assert set_metrics(a, np.zeros(0, dtype=np.int64))["purity"] == 0.0


# --------------------------------------------------------------------------
# Granularity 2: radius
# --------------------------------------------------------------------------

def test_pure_translation_is_fully_coherent():
    """The benign case: a valid object in the wrong place."""
    rng = np.random.default_rng(0)
    pos_a = rng.random((100, 3)) * BOX
    shift = np.array([0.7, -0.2, 0.1])
    pos_b = (pos_a + shift) % BOX
    st = displacement_stats(np.arange(100), pos_a, pos_b, BOX)
    assert st["residual_rms_mpc_h"] == pytest.approx(0.0, abs=1e-9)
    assert st["coherent_fraction"] == pytest.approx(1.0)
    assert st["bulk_mpc_h"] == pytest.approx(np.linalg.norm(shift))
    assert st["rms_mpc_h"] == pytest.approx(np.linalg.norm(shift))


def test_dispersal_is_incoherent():
    """The harmful case: the same particles, scattered rather than moved."""
    rng = np.random.default_rng(1)
    pos_a = np.full((2000, 3), 50.0) + rng.normal(0, 0.05, (2000, 3))
    pos_b = pos_a + rng.normal(0, 1.0, (2000, 3))
    st = displacement_stats(np.arange(2000), pos_a, pos_b, BOX)
    assert st["coherent_fraction"] < 0.05
    assert st["residual_rms_mpc_h"] > 0.9 * st["rms_mpc_h"]


def test_displacement_uses_the_short_way_round_the_box():
    pos_a = np.array([[99.9, 0.0, 0.0]])
    pos_b = np.array([[0.1, 0.0, 0.0]])
    st = displacement_stats(np.arange(1), pos_a, pos_b, BOX)
    assert st["rms_mpc_h"] == pytest.approx(0.2)
    assert periodic_delta(pos_b, pos_a, BOX)[0, 0] == pytest.approx(0.2)


def test_displacement_of_nothing_is_zero():
    st = displacement_stats(np.zeros(0, dtype=np.int64),
                            np.zeros((4, 3)), np.zeros((4, 3)), BOX)
    assert st["n"] == 0 and st["rms_mpc_h"] == 0.0


def test_radius_labels_survive_a_coincidence():
    """``2 * Rvir`` can land exactly on an absolute radius. Keying the result
    off the radius value would drop a column and shift every later one, so the
    labels are supplied by the caller and duplicates are refused."""
    pos = np.array([[0.05, 0.0, 0.0], [0.9, 0.0, 0.0]])
    f = radius_fractions(np.arange(2), pos, [0.0, 0.0, 0.0], BOX,
                         [0.5, 1.0, 0.5, 1.0],
                         ["rvir1", "rvir2", "mpc0.5", "mpc1"])
    assert list(f) == ["rvir1", "rvir2", "mpc0.5", "mpc1"]
    assert f["rvir1"] == f["mpc0.5"] == pytest.approx(0.5)
    assert f["rvir2"] == f["mpc1"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="duplicate radius labels"):
        radius_fractions(np.arange(2), pos, [0.0] * 3, BOX, [0.5, 0.5],
                         ["r", "r"])
    with pytest.raises(ValueError, match="names for"):
        radius_fractions(np.arange(2), pos, [0.0] * 3, BOX, [0.5, 1.0], ["r"])


def test_radius_fractions_are_monotone_and_periodic():
    pos = np.array([[0.05, 0.0, 0.0], [0.5, 0.0, 0.0], [99.9, 0.0, 0.0]])
    f = radius_fractions(np.arange(3), pos, [0.0, 0.0, 0.0], BOX,
                         [0.06, 0.2, 1.0])
    assert f["f_within_0.06"] == pytest.approx(1 / 3)
    assert f["f_within_0.2"] == pytest.approx(2 / 3)   # the wrapped one counts
    assert f["f_within_1"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Granularity 3: chunk
# --------------------------------------------------------------------------

def test_lagrangian_tile_profile_compares_objects_not_boxes():
    grid = TileGrid(ng_hr=NG, tile_hr=4, boxsize_mpc_h=BOX)
    a = lagrangian_ids([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
    b = lagrangian_ids([(0, 0, 0), (1, 0, 0), (4, 0, 0), (5, 0, 0)])
    far = lagrangian_ids([(8, 8, 8), (9, 8, 8)])
    ua, ub, uf = (tile_profile(x, grid) for x in (a, b, far))
    assert ua.sum() == pytest.approx(1.0)
    assert profile_overlap(ua, ua)["intersection"] == pytest.approx(1.0)
    assert profile_overlap(ua, ub)["intersection"] == pytest.approx(0.5)
    assert profile_overlap(ua, ub)["same_dominant_tile"]
    assert profile_overlap(ua, uf)["intersection"] == pytest.approx(0.0)
    assert not profile_overlap(ua, uf)["same_dominant_tile"]


def test_eulerian_chunk_shift_counts_crossings():
    box, n = 100.0, 8            # 12.5 Mpc/h chunks
    pos_a = np.array([[1.0, 1.0, 1.0], [12.0, 1.0, 1.0], [99.0, 1.0, 1.0]])
    pos_b = np.array([[2.0, 1.0, 1.0],     # same chunk
                      [13.0, 1.0, 1.0],    # crossed into the neighbour
                      [1.0, 1.0, 1.0]])    # wrapped: chunk 7 -> chunk 0, adjacent
    r = eulerian_chunk_shift(np.arange(3), pos_a, pos_b, box, n)
    assert r["frac_same_chunk"] == pytest.approx(1 / 3)
    assert r["frac_same_or_adjacent"] == pytest.approx(1.0)
    assert r["chunk_mpc_h"] == pytest.approx(12.5)


# --------------------------------------------------------------------------
# Matching adapter
# --------------------------------------------------------------------------

def test_as_flat_catalog_lets_subhalos_be_matched():
    """``match_hosts`` only ever sees top-level objects, so subhalo matching
    goes through the same greedy periodic matcher rather than a second one."""
    cat = nested_catalog()
    assert cat.subhalos().n == 2
    flat = as_flat_catalog(cat.subhalos())
    assert flat.n == 2 and np.all(flat.parent_ids == -1)
    assert flat.hosts().n == 2

    shifted = as_flat_catalog(cat.subhalos())
    shifted.pos[:] = (shifted.pos + 0.01) % BOX
    res = match_hosts(flat, shifted, boxsize_mpc_h=BOX)
    assert np.array_equal(np.sort(res.sr_ids), np.sort(flat.ids))
