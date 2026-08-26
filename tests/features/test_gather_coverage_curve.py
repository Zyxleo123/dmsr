"""Pins for the ceiling-vs-coverage curve.

The script's job is to say how high the ``R_vir`` ceiling could go if more
Lagrangian tiles were trained, *without* running Rockstar at every rung. So the
things that would make its answer wrong rather than absent are:

1. the ladder must walk the same tile ordering ``--n-tiles n`` trains on, and
   coverage must be monotone in it;
2. the live fraction it reads the ceiling off must be the fraction
   ``member_gather.build_member_sets`` cuts on, not a look-alike;
3. the supervision counts must apply *all* of the real cuts, because the point
   of the table is the gap between what the gate counts and what the loss is
   told about -- an over-count there is the whole error;
4. the cost columns must scale the way the loss does, or the rung that gets
   picked is picked on a fiction.

No catalog on disk, no owner array, no generator.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load():
    for p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
              PROJECT_ROOT / "scripts" / "features"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    path = PROJECT_ROOT / "scripts" / "features" / "gather_coverage_curve.py"
    spec = importlib.util.spec_from_file_location("gather_coverage_curve", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()


def _home():
    """Four subhalos with hand-placed Lagrangian occupancy.

    ``a`` sits wholly in tile 0; ``b`` is 3/4 in tile 1; ``c`` is split evenly
    across tiles 2 and 3 (so it is exactly at ``min_live_frac`` for either);
    ``d`` is wholly in tile 9, off every rung of the short ladder below.
    """
    occ_row = np.array([0, 1, 1, 2, 2, 3], dtype=np.int64)
    occ_tile = np.array([0, 1, 5, 2, 3, 9], dtype=np.int64)
    occ_count = np.array([100, 300, 100, 50, 50, 80], dtype=np.int64)
    return {
        "halo_id": np.array([1, 2, 3, 4], dtype=np.int64),
        "row": np.array([10, 11, 12, 13], dtype=np.int64),
        "tile": np.array([0, 1, 2, 9], dtype=np.int64),
        "purity": np.array([1.0, 0.75, 0.5, 1.0]),
        "n_sites": np.array([100, 400, 100, 80], dtype=np.int64),
        "num_p": np.array([100, 400, 100, 80], dtype=np.int64),
        "occ_row": occ_row, "occ_tile": occ_tile, "occ_count": occ_count,
    }


def _args(**kw):
    base = dict(min_purity=0.5, min_live_frac=0.5, min_num_p_ladder=[200, 50],
                bg_k=4096)
    base.update(kw)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# 1. the live fraction
# --------------------------------------------------------------------------- #
def test_live_fraction_is_member_particles_inside_the_trained_tiles():
    live = M.live_fractions(_home(), np.array([0, 1]))
    assert live == pytest.approx([1.0, 0.75, 0.0, 0.0])
    live = M.live_fractions(_home(), np.array([0, 1, 2, 3, 5]))
    assert live == pytest.approx([1.0, 1.0, 1.0, 0.0])


def test_live_fraction_is_monotone_in_the_tile_set():
    home = _home()
    small = M.live_fractions(home, np.array([0]))
    big = M.live_fractions(home, np.array([0, 1, 2, 3, 5, 9]))
    assert np.all(big >= small - 1e-12)
    assert big == pytest.approx(np.ones(4))


def test_live_fraction_of_no_tiles_is_zero_and_of_all_tiles_is_one():
    home = _home()
    assert M.live_fractions(home, np.array([], dtype=np.int64)) \
        == pytest.approx(np.zeros(4))


# --------------------------------------------------------------------------- #
# 2. the ceiling column
# --------------------------------------------------------------------------- #
def test_ceiling_counts_only_the_rvir_population_at_each_live_level():
    home = _home()
    in_rvir = np.array([True, True, True, False])       # `d` is outside R_vir
    live = M.live_fractions(home, np.array([0, 1]))
    r = M.rung(home, live, in_rvir, _args(), [0, 1], 0.5)
    c = r["ceiling_rvir"]
    assert c["n_rvir_total"] == 3
    assert c["live_ge_0.9"] == 1        # only `a` is wholly inside
    assert c["live_ge_0.7"] == 2        # `b` at 0.75 joins
    assert c["live_ge_0.5"] == 2
    # the threshold-free reading: 1.0 + 0.75 + 0.0, and `d` never enters
    assert c["sum_live"] == pytest.approx(1.75)


def test_a_subhalo_outside_rvir_never_raises_the_ceiling():
    home = _home()
    live = M.live_fractions(home, np.array([0, 1, 5, 9]))   # a, b, d live; c not
    all_in = M.rung(home, live, np.ones(4, bool), _args(), [0, 1, 5, 9], 0.9)
    without = M.rung(home, live, np.array([True, True, True, False]),
                     _args(), [0, 1, 5, 9], 0.9)
    assert all_in["ceiling_rvir"]["live_ge_0.9"] == 3        # a, b and d
    assert without["ceiling_rvir"]["live_ge_0.9"] == 2       # d drops out


# --------------------------------------------------------------------------- #
# 3. the supervision column -- every real cut, or the gap is understated
# --------------------------------------------------------------------------- #
def test_supervision_applies_home_tile_purity_live_and_the_particle_cut():
    home = _home()
    in_rvir = np.ones(4, bool)
    live = M.live_fractions(home, np.array([0, 1]))
    sup = M.rung(home, live, in_rvir, _args(), [0, 1], 0.5)["supervised"]
    # at >= 200p only `b` survives: `a` and `c` are too light, `d` is off-tile
    assert sup["200"]["n_sets"] == 1
    assert sup["200"]["member_particles"] == 400
    # at >= 50p `a` joins; `c`'s home tile (2) is not trained, `d`'s (9) is not
    assert sup["50"]["n_sets"] == 2
    assert sup["50"]["member_particles"] == 500


def test_a_trained_home_tile_is_not_enough_without_the_live_fraction():
    home = _home()
    # tile 2 trained: `c` is homed there and pure enough at 0.5, but only half
    # its material is inside, which is exactly the min_live_frac boundary.
    live = M.live_fractions(home, np.array([2]))
    assert M.rung(home, live, np.ones(4, bool), _args(min_live_frac=0.5),
                  [2], 0.1)["supervised"]["50"]["n_sets"] == 1
    assert M.rung(home, live, np.ones(4, bool), _args(min_live_frac=0.75),
                  [2], 0.1)["supervised"]["50"]["n_sets"] == 0


def test_lowering_min_num_p_never_lowers_the_ceiling_it_only_adds_sets():
    home = _home()
    live = M.live_fractions(home, np.array([0, 1]))
    r = M.rung(home, live, np.ones(4, bool), _args(), [0, 1], 0.5)
    assert r["supervised"]["50"]["n_sets"] >= r["supervised"]["200"]["n_sets"]
    # ... and the ceiling does not depend on min_num_p at all: it is the same
    # object either way. This is the point of separating the two knobs.
    r2 = M.rung(home, live, np.ones(4, bool),
                _args(min_num_p_ladder=[50]), [0, 1], 0.5)
    assert r2["ceiling_rvir"] == r["ceiling_rvir"]


# --------------------------------------------------------------------------- #
# 4. the cost columns
# --------------------------------------------------------------------------- #
def test_cost_columns_scale_the_way_the_run_does():
    home = _home()
    live = M.live_fractions(home, np.array([0, 1]))
    r = M.rung(home, live, np.ones(4, bool), _args(), [0, 1], 0.5)
    # free parameters are the whole tile field, 6 channels of 64^3 per tile
    assert r["delta_params"] == 2 * 6 * M.TILE ** 3
    s = r["supervised"]["50"]
    assert s["sum_n_squared"] == pytest.approx(100.0 ** 2 + 400.0 ** 2)
    assert s["bg_particles"] == 2 * 4096
