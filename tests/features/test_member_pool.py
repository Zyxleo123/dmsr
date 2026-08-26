"""Multi-host selection, and the leak it is built to refuse.

The load-bearing test is :func:`test_top_host_matches_host_tiles`: the pool's
notion of "which tiles are this host's" must be identical to the one every
number in ``docs/sr2_member_gather.md`` was measured with. Two definitions in
one repository would mean the fine-tune is not working the oracle's problem.
"""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.features.member_pool import (
    HostSelection, select_hosts, split_pool, summarise_pool,
)

NG, TILE = 64, 16           # 4^3 = 64 tiles, small and exact
N_SIDE = NG // TILE


class _Cat:
    """The fields select_hosts reads off a HaloCatalog."""

    def __init__(self, ids, mvir, num_p, parent_ids):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.mvir = np.asarray(mvir, dtype=np.float64)
        self.num_p = np.asarray(num_p, dtype=np.int64)
        self.parent_ids = np.asarray(parent_ids, dtype=np.int64)


class _Owner:
    """Stands in for the CSR owner index: halo id -> flat Lagrangian ids."""

    def __init__(self, members):
        self._m = {int(k): np.asarray(v, dtype=np.int64) for k, v in members.items()}

    def members_with_substructure(self, cat, hid, children=None):
        return self._m.get(int(hid), np.zeros(0, dtype=np.int64))


def _ids_in_tile(tile_id: int, n: int) -> np.ndarray:
    """``n`` flat Lagrangian ids drawn from one tile of the NG^3 lattice."""
    ix, iy, iz = tile_id // (N_SIDE * N_SIDE), (tile_id // N_SIDE) % N_SIDE, tile_id % N_SIDE
    base = np.arange(n)
    i = ix * TILE + (base % TILE)
    j = iy * TILE + ((base // TILE) % TILE)
    k = iz * TILE + ((base // (TILE * TILE)) % TILE)
    return ((i * NG) + j) * NG + k


def _pool(members, mvir, parent=None):
    ids = sorted(members)
    cat = _Cat(ids, [mvir[i] for i in ids], [len(members[i]) for i in ids],
               parent if parent is not None else [-1] * len(ids))
    return cat, _Owner(members)


def test_selects_by_mass_and_respects_the_floor():
    members = {1: _ids_in_tile(0, 400), 2: _ids_in_tile(5, 400), 3: _ids_in_tile(9, 400)}
    mvir = {1: 1e14, 2: 1e13, 3: 1e15}
    cat, oidx = _pool(members, mvir)
    got = select_hosts(cat, oidx, "setX", n_tiles=2, min_log_mvir=13.5,
                       ng_hr=NG, tile_hr=TILE, children={})
    assert [h.halo_id for h in got] == [3, 1]      # mass order, 1e13 excluded


def test_max_hosts_caps_the_pool():
    members = {i: _ids_in_tile(i, 400) for i in range(1, 9)}
    mvir = {i: 10 ** (15.0 - 0.1 * i) for i in range(1, 9)}
    cat, oidx = _pool(members, mvir)
    got = select_hosts(cat, oidx, "setX", max_hosts=3, min_log_mvir=13.0,
                       ng_hr=NG, tile_hr=TILE, children={})
    assert len(got) == 3
    assert [h.halo_id for h in got] == [1, 2, 3]


def test_tiles_are_ranked_by_member_site_count():
    ids = np.concatenate([_ids_in_tile(7, 100), _ids_in_tile(2, 500),
                          _ids_in_tile(9, 300)])
    cat, oidx = _pool({1: ids}, {1: 1e15})
    got = select_hosts(cat, oidx, "setX", n_tiles=2, min_log_mvir=13.0,
                       ng_hr=NG, tile_hr=TILE, children={})
    assert got[0].tiles == [2, 9]
    assert got[0].tile_member_sites == [500, 300]


def test_site_coverage_is_the_fraction_inside_the_chosen_tiles():
    ids = np.concatenate([_ids_in_tile(2, 600), _ids_in_tile(9, 400)])
    cat, oidx = _pool({1: ids}, {1: 1e15})
    got = select_hosts(cat, oidx, "setX", n_tiles=1, min_log_mvir=13.0,
                       ng_hr=NG, tile_hr=TILE, children={})
    assert got[0].site_coverage == pytest.approx(0.6)


def test_hosts_with_no_members_are_skipped_not_crashed():
    cat, oidx = _pool({1: _ids_in_tile(3, 200), 2: np.zeros(0, np.int64)},
                      {1: 1e15, 2: 1e15})
    cat.num_p = np.array([200, 0])
    got = select_hosts(cat, oidx, "setX", min_log_mvir=13.0,
                       ng_hr=NG, tile_hr=TILE, children={})
    assert [h.halo_id for h in got] == [1]


def test_subhalos_are_not_selected_as_hosts():
    members = {1: _ids_in_tile(0, 400), 2: _ids_in_tile(5, 400)}
    cat, oidx = _pool(members, {1: 1e15, 2: 1e15}, parent=[-1, 1])
    got = select_hosts(cat, oidx, "setX", min_log_mvir=13.0,
                       ng_hr=NG, tile_hr=TILE, children={})
    assert [h.halo_id for h in got] == [1]


# --------------------------------------------------------------------------- #
# The split, and the leak it refuses
# --------------------------------------------------------------------------- #
def _sel(box, hid, tiles):
    return HostSelection(box=box, halo_id=hid, log_mvir=14.0, num_p=1000,
                         n_member_sites=1000, tiles=list(tiles),
                         tile_member_sites=[1] * len(tiles), site_coverage=0.4)


def test_box_split_is_clean_when_boxes_differ():
    hosts = [_sel("set3", 1, [0, 1]), _sel("set9", 2, [0, 1])]
    s = split_pool(hosts, train_boxes=["set3"], holdout_boxes=["set9"])
    assert [h.halo_id for h in s.train] == [1]
    assert [h.halo_id for h in s.holdout] == [2]
    # Identical tile ids in DIFFERENT boxes are different tiles and must not clash.
    assert s.rejected == []


def test_same_box_holdout_key_is_honoured():
    hosts = [_sel("set3", 1, [0, 1]), _sel("set3", 2, [4, 5])]
    s = split_pool(hosts, train_boxes=["set3"], holdout_boxes=[],
                   holdout_keys=["set3:h2"])
    assert [h.halo_id for h in s.train] == [1]
    assert [h.halo_id for h in s.holdout] == [2]
    assert s.rejected == []


def test_same_box_tile_overlap_is_rejected_from_both_sides():
    """The silent-leak case: a held-out host sitting in a supervised tile.

    Without the net, host 2 would report a recovery number for tile 1 that the
    run had been trained on, under a held-out label.
    """
    hosts = [_sel("set3", 1, [0, 1]), _sel("set3", 2, [1, 2])]
    s = split_pool(hosts, train_boxes=["set3"], holdout_boxes=[],
                   holdout_keys=["set3:h2"])
    # The held-out host goes, the training host stays: dropping both would
    # discard usable supervision without removing anything unsound.
    assert s.holdout == []
    assert [h.halo_id for h in s.train] == [1]
    assert [key for key, _ in s.rejected] == ["set3:h2"]
    assert "tiles [1]" in s.rejected[0][1]


def test_disjoint_tiles_in_the_same_box_survive_the_net():
    hosts = [_sel("set3", 1, [0, 1]), _sel("set3", 2, [2, 3])]
    s = split_pool(hosts, train_boxes=["set3"], holdout_boxes=[],
                   holdout_keys=["set3:h2"])
    assert [h.halo_id for h in s.train] == [1]
    assert [h.halo_id for h in s.holdout] == [2]


def test_identical_tile_ids_in_different_boxes_do_not_clash():
    """Tile ids are per-box; tile 1 of set3 and tile 1 of set9 are different."""
    hosts = [_sel("set3", 1, [0, 1]), _sel("set9", 2, [1, 2])]
    s = split_pool(hosts, train_boxes=["set3"], holdout_boxes=["set9"])
    assert [h.halo_id for h in s.train] == [1]
    assert [h.halo_id for h in s.holdout] == [2]
    assert s.rejected == []


def test_a_box_on_both_sides_is_refused_outright():
    hosts = [_sel("setA", 1, [0, 1])]
    with pytest.raises(ValueError, match="cannot be both"):
        split_pool(hosts, train_boxes=["setA"], holdout_boxes=["setA"])


def test_hosts_outside_both_lists_are_ignored():
    hosts = [_sel("set3", 1, [0]), _sel("set7", 2, [1]), _sel("set9", 3, [2])]
    s = split_pool(hosts, train_boxes=["set3"], holdout_boxes=["set9"])
    assert [h.halo_id for h in s.train] == [1]
    assert [h.halo_id for h in s.holdout] == [3]


def test_summarise_reports_coverage_and_boxes():
    hosts = [_sel("set3", 1, [0, 1]), _sel("set9", 2, [2, 3])]
    got = summarise_pool(hosts)
    assert got["n_hosts"] == 2
    assert got["n_boxes"] == 2
    assert got["n_tiles_total"] == 4
    assert got["site_coverage_median"] == pytest.approx(0.4)


def test_summarise_empty_pool_does_not_crash():
    assert summarise_pool([])["n_hosts"] == 0


# --------------------------------------------------------------------------- #
# The pin against the single-host definition
# --------------------------------------------------------------------------- #
def test_top_host_matches_host_tiles():
    """select_hosts' top host must choose the tiles host_tiles chooses.

    ``host_tiles`` ranks hosts by mvir, takes the top one, bincounts its member
    sites by tile and takes the ``n_tiles`` largest. Reproduced here on the same
    inputs; if the two ever disagree the fine-tune and the oracle are measuring
    different regions.
    """
    rng = np.random.default_rng(0)
    ids = np.concatenate([
        _ids_in_tile(t, int(n)) for t, n in
        [(1, 900), (5, 700), (9, 500), (13, 300), (21, 100)]
    ])
    rng.shuffle(ids)
    cat, oidx = _pool({7: ids}, {7: 1e15})

    got = select_hosts(cat, oidx, "setX", n_tiles=4, min_log_mvir=13.0,
                       ng_hr=NG, tile_hr=TILE, children={})[0]

    # host_tiles' arithmetic, inlined on the same inputs.
    from cosmo_sr.features.host_crops import flat_to_sites
    sites = flat_to_sites(ids, NG) // TILE
    tid = (sites[:, 0] * N_SIDE + sites[:, 1]) * N_SIDE + sites[:, 2]
    counts = np.bincount(tid, minlength=N_SIDE ** 3)
    expected = np.argsort(-counts)[:4].astype(int).tolist()

    assert got.tiles == expected
    assert got.tile_member_sites == [int(counts[t]) for t in expected]
