"""Variable-cardinality proposals from bootstrapped training-host catalogs.

The point of copying a whole donor host's satellite list, rather than sampling a
count and then sampling masses and radii independently, is that the joint
structure comes along for free. These tests pin that -- and pin the subtraction
step, which is what turns a *catalog* generator into an *edit* generator.
"""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.reward.token_bootstrap import HostTokenLibrary, subtract_existing

from conftest import synthetic_catalog

BOX = 100.0


def library(seed=0):
    hosts = [([10.0, 10.0, 10.0], 1e13), ([50.0, 50.0, 50.0], 1e13),
             ([80.0, 20.0, 40.0], 1e14)]
    cat = synthetic_catalog(hosts, [3, 5, 9], boxsize=BOX, seed=seed,
                            host_num_p=20000, sub_num_p=100, sub_mass=5e10,
                            rvir_kpc=500.0)
    return HostTokenLibrary.from_catalogs({"set0": cat}, boxsize_mpc_h=BOX,
                                          min_host_particles=1000)


def test_the_library_indexes_every_qualifying_host_with_its_own_children():
    lib = library()
    assert lib.n_hosts == 3
    assert [lib.children(h)[0].size for h in range(3)] == [3, 5, 9]
    assert lib.log_mass_ratio.size == 17


def test_ratios_and_radii_are_host_normalised():
    lib = library()
    r, d = lib.children(0)
    # 5e10 / 1e13 = 5e-3
    assert np.allclose(r, np.log10(5e-3))
    assert np.all(d >= 0) and np.all(np.isfinite(d))


def test_a_donor_of_similar_mass_is_chosen():
    lib = library()
    rng = np.random.default_rng(0)
    toks = lib.sample_tokens(host_id=42, host_mvir=1e13, rng=rng,
                             log_mass_ratio_range=(-4.0, -1.0),
                             radius_range=(0.0, 5.0), max_tokens=10)
    # Only the two 1e13 donors are within 0.15 dex, and they have 3 or 5 kids.
    assert len(toks) in (3, 5)
    assert all(t.host_id == 42 for t in toks)


def test_the_directions_are_unit_vectors_and_resampled():
    lib = library()
    a = lib.sample_tokens(host_id=1, host_mvir=1e13,
                          rng=np.random.default_rng(1),
                          log_mass_ratio_range=(-4.0, -1.0),
                          radius_range=(0.0, 5.0), max_tokens=10)
    b = lib.sample_tokens(host_id=1, host_mvir=1e13,
                          rng=np.random.default_rng(2),
                          log_mass_ratio_range=(-4.0, -1.0),
                          radius_range=(0.0, 5.0), max_tokens=10)
    for t in a:
        assert np.isclose(np.linalg.norm(np.asarray(t.direction)), 1.0)
    assert not np.allclose([t.direction for t in a][0], [t.direction for t in b][0])


def test_objects_sr2_already_has_are_subtracted():
    """The deficit, not the whole desired population: proposing what SR2 already
    produced would ask the editor to duplicate existing objects."""
    lib = library()
    rng = np.random.default_rng(3)
    full = lib.sample_tokens(host_id=1, host_mvir=1e13, rng=np.random.default_rng(3),
                             log_mass_ratio_range=(-4.0, -1.0),
                             radius_range=(0.0, 5.0), max_tokens=10)
    present = [t.log_mass_ratio for t in full[:2]]
    left = lib.sample_tokens(host_id=1, host_mvir=1e13, rng=rng,
                             existing_log_mass_ratio=present,
                             log_mass_ratio_range=(-4.0, -1.0),
                             radius_range=(0.0, 5.0), max_tokens=10)
    assert len(left) == len(full) - 2


def test_one_existing_object_cancels_exactly_one_desired_object():
    keep = subtract_existing([-2.0, -2.0, -2.0], [-2.0], tol_dex=0.25)
    assert keep.tolist() == [1, 2]


def test_an_existing_object_of_a_different_mass_cancels_nothing():
    keep = subtract_existing([-2.0, -3.0], [-1.0], tol_dex=0.25)
    assert keep.tolist() == [0, 1]


def test_out_of_range_satellites_are_dropped_before_subtraction():
    lib = library()
    toks = lib.sample_tokens(host_id=1, host_mvir=1e13,
                             rng=np.random.default_rng(4),
                             log_mass_ratio_range=(-1.5, -1.0),   # excludes 5e-3
                             radius_range=(0.0, 5.0), max_tokens=10)
    assert toks == []


def test_the_library_round_trips_through_npz(tmp_path):
    lib = library()
    p = lib.to_npz(tmp_path / "lib.npz")
    back = HostTokenLibrary.from_npz(p)
    assert back.n_hosts == lib.n_hosts
    assert np.array_equal(back.offsets, lib.offsets)
    assert np.allclose(back.log_mass_ratio, lib.log_mass_ratio)
    assert back.boxes == ("set0",)


def test_an_empty_library_returns_no_tokens_rather_than_raising():
    empty = HostTokenLibrary(np.zeros(0), np.zeros(1, dtype=np.int64),
                             np.zeros(0), np.zeros(0))
    assert empty.sample_tokens(host_id=1, host_mvir=1e13,
                               rng=np.random.default_rng(0)) == []
