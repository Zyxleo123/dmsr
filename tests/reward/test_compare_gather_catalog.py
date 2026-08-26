"""Pins the bound-subhalo measurement of compare_gather_catalog.py.

The comparison runs after a 12-hour halo-finder job, so every mistake in it is
expensive to discover. What is pinned here is the arithmetic that decides the
headline number: periodic distances (the host can sit near a box face), matching
the host by position rather than by id (three catalogs, three unrelated id
spaces), and counting subhalos in a fixed physical sphere rather than by parent
id (three different halo trees).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load():
    for p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    path = PROJECT_ROOT / "scripts" / "reward" / "compare_gather_catalog.py"
    spec = importlib.util.spec_from_file_location("compare_gather_catalog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()
from cosmo_sr.eval.rockstar import HaloCatalog  # noqa: E402


def _cat(pos, num_p, parent, mvir=None, rvir=None):
    n = len(num_p)
    return HaloCatalog(
        ids=np.arange(n, dtype=np.int64),
        parent_ids=np.asarray(parent, dtype=np.int64),
        mvir=np.asarray(mvir if mvir is not None else np.asarray(num_p) * 5.8e8,
                        dtype=float),
        rvir=np.asarray(rvir if rvir is not None else np.full(n, 500.0), dtype=float),
        vmax=np.zeros(n), pos=np.asarray(pos, dtype=float),
        vel=np.zeros((n, 3)), num_p=np.asarray(num_p, dtype=np.int64))


def test_periodic_delta_wraps_the_box():
    d = M.periodic_delta(np.array([[99.5, 0.0, 0.0]]), np.array([[0.5, 0.0, 0.0]]))
    assert d[0, 0] == pytest.approx(-1.0)


def test_subhalos_are_counted_across_a_box_face():
    """The host can sit near a face; a non-periodic distance would report zero."""
    cat = _cat([[99.8, 50.0, 50.0], [0.2, 50.0, 50.0], [50.0, 50.0, 50.0]],
               [5000, 300, 300], [-1, 0, 0])
    rows = M.subhalos_within(cat, np.array([99.8, 50.0, 50.0]), 1.0, min_p=50)
    assert rows.tolist() == [1]


def test_host_is_matched_by_position_and_mass_not_by_id():
    """Three catalogs, three unrelated id spaces -- only geometry is shared."""
    cat = _cat([[50.05, 50.0, 50.0], [50.02, 50.0, 50.0], [10.0, 10.0, 10.0]],
               [4000, 600000, 900000], [-1, -1, -1])
    r = M.match_host(cat, np.array([50.0, 50.0, 50.0]), 3.5e14, max_sep=1.0)
    assert r == 1                       # the massive one nearby, not the nearest
    assert M.match_host(cat, np.array([80.0, 80.0, 80.0]), 3.5e14,
                        max_sep=1.0) is None


def test_counting_ignores_the_parent_tree_and_the_particle_floor():
    cat = _cat([[50.0, 50.0, 50.0], [50.1, 50.0, 50.0], [50.2, 50.0, 50.0],
                [50.3, 50.0, 50.0]],
               [5000, 300, 40, 900], [-1, 0, 0, 2])
    rows = M.subhalos_within(cat, np.array([50.0, 50.0, 50.0]), 1.0, min_p=50)
    # the 40-particle object is under the floor; the sub-subhalo still counts
    assert sorted(rows.tolist()) == [1, 3]
    bins = M.count_by_bin(cat, rows)
    assert bins["200-500p"] == 1 and bins["500-2000p"] == 1 and bins["total"] == 2


def test_shell_profile_localises_a_change():
    """An edge artifact shows up as objects appearing far from the host."""
    base = _cat([[50.0, 50.0, 50.0]], [5000], [-1])
    cand = _cat([[50.0, 50.0, 50.0], [50.2, 50.0, 50.0], [58.0, 50.0, 50.0]],
                [5000, 300, 300], [-1, 0, -1])
    prof = M.shell_profile(base, cand, np.array([50.0, 50.0, 50.0]),
                           [0.0, 1.0, 4.0, 16.0], min_p=50)
    assert prof[0]["delta"] == 1        # near the host: real substructure
    assert prof[1]["delta"] == 0
    assert prof[2]["delta"] == 1        # far away: suspect the splice edge


def test_target_hit_rate_needs_position_AND_mass():
    """A wisp near the right place is not a recovered subhalo."""
    hr = _cat([[50.0, 50.0, 50.0], [50.0, 50.0, 50.0]], [5000, 400], [-1, 0],
              rvir=[1500.0, 200.0])
    hr.ids[:] = [900, 901]
    # candidate holds a 300p halo right on target 901, plus a 20p wisp
    cand = _cat([[50.01, 50.0, 50.0], [50.02, 50.0, 50.0]], [300, 20], [-1, -1])
    out = M.target_hit_rate(cand, hr, [901], radius_factor=1.0, mass_frac=0.25)
    assert out["n"] == 1 and out["hits"] == 1 and out["rows"][0]["best_num_p"] == 300

    wisp_only = _cat([[50.01, 50.0, 50.0]], [20], [-1])
    out2 = M.target_hit_rate(wisp_only, hr, [901], radius_factor=1.0, mass_frac=0.25)
    assert out2["hits"] == 0


def test_target_hit_rate_uses_a_floor_on_the_search_radius():
    """A 200-particle subhalo has r_vir well under one HR cell; without a floor
    the search sphere would be smaller than the finder's own centring error."""
    hr = _cat([[50.0, 50.0, 50.0]], [200], [0], rvir=[60.0])
    hr.ids[:] = [901]
    cand = _cat([[50.10, 50.0, 50.0]], [180], [-1])
    assert M.target_hit_rate(cand, hr, [901], min_radius=0.15)["hits"] == 1
    assert M.target_hit_rate(cand, hr, [901], min_radius=0.01)["hits"] == 0


def test_target_hit_rate_skips_ids_absent_from_the_hr_catalog():
    hr = _cat([[50.0, 50.0, 50.0]], [400], [0], rvir=[200.0])
    hr.ids[:] = [901]
    cand = _cat([[50.0, 50.0, 50.0]], [400], [-1])
    assert M.target_hit_rate(cand, hr, [901, 999])["n"] == 1
