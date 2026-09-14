"""Index math and accumulation for the LR membership-coverage probe.

The whole job rests on one mapping (HR Lagrangian site -> LR Lagrangian site)
and one grouped count. Either can be wrong in a way that still produces a
plausible-looking coverage table, so both are pinned here.
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

lmc = pytest.importorskip("lr_membership_coverage")


class _Cat:
    """Minimal stand-in for HaloCatalog: only ids/parent_ids are used here."""

    def __init__(self, ids, parent_ids):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.parent_ids = np.asarray(parent_ids, dtype=np.int64)


def test_hr_site_to_lr_site_maps_a_whole_block_to_one_cell():
    """All 8^3 HR sites in a block must land on the same LR site."""
    f = lmc.NG_HR // lmc.NG_LR
    assert f == 8
    lx, ly, lz = 3, 5, 7
    idx = []
    for dx in range(f):
        for dy in range(f):
            for dz in range(f):
                cx, cy, cz = lx * f + dx, ly * f + dy, lz * f + dz
                idx.append(cx * lmc.NG_HR ** 2 + cy * lmc.NG_HR + cz)
    got = lmc.hr_site_to_lr_site(np.array(idx, dtype=np.int64))
    want = lx * lmc.NG_LR ** 2 + ly * lmc.NG_LR + lz
    assert (got == want).all()
    assert got.size == f ** 3


def test_hr_site_to_lr_site_is_in_range_and_covers_every_lr_cell():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, lmc.NG_HR ** 3, size=20000)
    out = lmc.hr_site_to_lr_site(idx)
    assert out.min() >= 0 and out.max() < lmc.NG_LR ** 3
    # the corners must map to the corners, not wrap
    assert lmc.hr_site_to_lr_site(np.array([0]))[0] == 0
    assert lmc.hr_site_to_lr_site(np.array([lmc.NG_HR ** 3 - 1]))[0] == lmc.NG_LR ** 3 - 1


def test_top_level_of_resolves_a_subhalo_chain_to_its_cluster():
    # 10 is the host; 11 is its subhalo; 12 is a sub-subhalo of 11
    cat = _Cat([10, 11, 12], [-1, 10, 11])
    tl = lmc.top_level_of(cat)
    assert tl == {10: 10, 11: 10, 12: 10}


def test_top_level_of_survives_a_malformed_parent_cycle():
    """A bad parent chain must not hang the job."""
    cat = _Cat([1, 2], [2, 1])
    tl = lmc.top_level_of(cat)
    assert set(tl) == {1, 2}          # returns, does not spin


@pytest.fixture
def tiny(monkeypatch):
    """Shrink the lattice to 16^3 / 2^3, keeping the real factor of 8.

    `accumulate` and `hr_site_to_lr_site` read the module constants at call
    time, so this exercises the same code on 4,096 entries instead of
    134,217,728 -- a 536 MB allocation per test otherwise.
    """
    monkeypatch.setattr(lmc, "NG_HR", 16)
    monkeypatch.setattr(lmc, "NG_LR", 2)
    return lmc.NG_HR // lmc.NG_LR


def _hr_idx(cx, cy, cz):
    return cx * lmc.NG_HR ** 2 + cy * lmc.NG_HR + cz


def test_accumulate_counts_sites_any_and_correct_host(tmp_path, tiny):
    """Three HR halos with known membership and a hand-built LR map."""
    f = tiny
    site_host = np.full(lmc.NG_LR ** 3, -1, dtype=np.int64)
    # halo 100 -> LR cell A, assigned to LR host 7 (CORRECT)
    # halo 200 -> LR cell B, assigned to LR host 9 (WRONG host)
    # halo 300 -> LR cell C, unassigned
    a = lmc.hr_site_to_lr_site(np.array([_hr_idx(0, 0, 0)]))[0]
    b = lmc.hr_site_to_lr_site(np.array([_hr_idx(f, 0, 0)]))[0]
    site_host[a] = 7
    site_host[b] = 9

    owner = np.full(lmc.NG_HR ** 3, -1, dtype=np.int32)
    for k in range(3):
        owner[_hr_idx(0, 0, k)] = 100
    for k in range(2):
        owner[_hr_idx(f, 0, k)] = 200
    for k in range(2):
        owner[_hr_idx(0, f, k)] = 300          # a third, unassigned LR cell
    p = tmp_path / "owner.npy"
    np.save(p, owner)

    hr_ids = np.array([100, 200, 300], dtype=np.int64)
    want = np.array([7, 7, 7], dtype=np.int64)
    ns, na, nk = lmc.accumulate(p, site_host, hr_ids, want, chunk=1_000_000)
    assert list(ns) == [3, 2, 2]
    assert list(na) == [3, 2, 0]     # halo 300's LR cell is unassigned
    assert list(nk) == [3, 0, 0]     # halo 200 landed on the wrong LR host


def test_accumulate_drops_owner_ids_absent_from_the_catalog(tmp_path, tiny):
    """An unknown owner id must not be aliased onto a neighbouring row."""
    site_host = np.full(lmc.NG_LR ** 3, 7, dtype=np.int64)
    owner = np.full(lmc.NG_HR ** 3, -1, dtype=np.int32)
    owner[0] = 100
    owner[1] = 555                              # not in hr_ids
    p = tmp_path / "owner.npy"
    np.save(p, owner)
    ns, na, nk = lmc.accumulate(p, site_host, np.array([100, 200], dtype=np.int64),
                                np.array([7, 7], dtype=np.int64), chunk=1_000_000)
    assert list(ns) == [1, 0]        # 555 dropped, not folded into row 200


def test_accumulate_is_chunk_invariant(tmp_path, tiny):
    rng = np.random.default_rng(1)
    owner = np.full(lmc.NG_HR ** 3, -1, dtype=np.int32)
    picks = rng.choice(lmc.NG_HR ** 3, size=200, replace=False)
    owner[picks] = rng.choice([10, 20, 30], size=200)
    p = tmp_path / "owner.npy"
    np.save(p, owner)
    site_host = rng.choice([-1, 10, 20], size=lmc.NG_LR ** 3).astype(np.int64)
    hr_ids = np.array([10, 20, 30], dtype=np.int64)
    want = np.array([10, 20, 30], dtype=np.int64)
    big = lmc.accumulate(p, site_host, hr_ids, want, chunk=lmc.NG_HR ** 3)
    small = lmc.accumulate(p, site_host, hr_ids, want, chunk=37)
    for x, y in zip(big, small):
        assert (x == y).all()


def test_rvir_bin_labels_cover_the_peak_and_end_open():
    labels = lmc.rvir_bin_labels()
    assert "1-1.5" in labels          # where HR's subhalos peak
    assert labels[-1].startswith(">")
    assert len(labels) == len(lmc.RVIR_BINS) - 1
