"""The census's three claims, on catalogs whose answer is known by construction.

The job reads real Rockstar ASCII and reports a verdict that decides which of
two training arms to run, so what is pinned here is the decision boundary --
that a deficit built to be small-halo-only reads SMALL-HALO and one built to be
size-independent reads FLAT -- and the environment label, which is the part with
real geometry in it and the part that would silently mislabel a whole
population if the periodic wrap or the more-massive test were wrong.
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

from cosmo_sr.eval.rockstar import HaloCatalog  # noqa: E402

census = pytest.importorskip("halo_gap_census")
BOX = census.BOX


def make_cat(pos, num_p, parent_ids, rvir_kpc) -> HaloCatalog:
    n = len(num_p)
    return HaloCatalog(
        ids=np.arange(n, dtype=np.int64),
        parent_ids=np.asarray(parent_ids, dtype=np.int64),
        mvir=np.asarray(num_p, dtype=np.float64) * 1e9,
        rvir=np.asarray(rvir_kpc, dtype=np.float64),
        vmax=np.full(n, 100.0),
        pos=np.asarray(pos, dtype=np.float64).reshape(n, 3),
        vel=np.zeros((n, 3)),
        num_p=np.asarray(num_p, dtype=np.int64),
    )


def test_environment_labels_inside_near_and_field():
    """One 1 Mpc/h host and three satellites at 0.5, 2 and 10 of its R_vir."""
    rv = 1000.0  # kpc/h -> 1 Mpc/h
    cat = make_cat(
        pos=[[50, 50, 50], [50.5, 50, 50], [52.0, 50, 50], [60.0, 50, 50]],
        num_p=[10_000, 100, 100, 100],
        parent_ids=[-1, -1, -1, -1],
        rvir_kpc=[rv, 100.0, 100.0, 100.0])
    env = census.environment(cat, host_floor=200)
    labels = [census.ENV_LABELS[e] for e in env]
    # The host itself has no MORE massive neighbour, so it is never `inside`.
    assert labels == ["field", "inside", "near", "field"]


def test_environment_is_periodic_across_the_box_face():
    """A satellite 0.5 Mpc/h from a host on the other side of the wrap."""
    cat = make_cat(
        pos=[[0.25, 50, 50], [99.75, 50, 50]],
        num_p=[10_000, 100], parent_ids=[-1, -1], rvir_kpc=[1000.0, 100.0])
    env = census.environment(cat, host_floor=200)
    # Minimum-image separation is 0.5 Mpc/h = 0.5 R_vir, not 99.5.
    assert census.ENV_LABELS[env[1]] == "inside"


def test_a_halo_is_never_inside_a_smaller_neighbour():
    """The more-massive test, which orders the pair rather than just pairing it."""
    cat = make_cat(
        pos=[[50, 50, 50], [50.2, 50, 50]],
        num_p=[300, 20_000], parent_ids=[-1, -1], rvir_kpc=[300.0, 2000.0])
    env = census.environment(cat, host_floor=200)
    assert census.ENV_LABELS[env[0]] == "inside"   # the small one, in the big
    assert census.ENV_LABELS[env[1]] == "field"    # the big one, in nothing


def _pair(hr_counts, base_counts):
    """Two catalogs whose top-level mass function is exactly as specified.

    Every halo is placed far from every other so the environment label is
    `field` throughout and cannot interact with the mass verdict.
    """
    def build(counts):
        pos, num_p = [], []
        rng = np.random.default_rng(0)
        for i, (lo, n) in enumerate(zip(census.MASS_EDGES[:-1], counts)):
            for _ in range(n):
                pos.append(rng.uniform(0, BOX, 3))
                num_p.append(lo + 1)
        n = len(num_p)
        return make_cat(pos, num_p, [-1] * n, [10.0] * n)
    return build(hr_counts), build(base_counts)


def _verdict(hr_counts, base_counts):
    hr, base = _pair(hr_counts, base_counts)
    ch = census.classify(hr, 200)
    cb = census.classify(base, 200)
    return census.verdict(census.table(ch, top=True, region_only=False),
                          census.table(cb, top=True, region_only=False),
                          census.table(ch, top=False, region_only=False),
                          census.table(cb, top=False, region_only=False))


def test_a_small_halo_deficit_reads_SMALL_HALO():
    # base recovers 30% of the smallest bin and 95% of the largest.
    hr = [1000, 1000, 1000, 1000, 0, 0, 0]
    base = [300, 500, 800, 950, 0, 0, 0]
    v = _verdict(hr, base)
    assert v["text"].startswith("SMALL-HALO"), v["text"]
    assert v["ratio_smallest_bin"] == pytest.approx(0.30, abs=1e-6)
    assert v["top_level_gap_total"] == 1450
    # 700 + 500 of the 1450 missing halos are below 200p.
    assert v["frac_of_gap_below_200p"] == pytest.approx(1200 / 1450, abs=1e-3)


def test_a_size_independent_deficit_reads_FLAT():
    hr = [1000, 1000, 1000, 1000, 0, 0, 0]
    base = [700, 700, 700, 700, 0, 0, 0]
    v = _verdict(hr, base)
    assert v["text"].startswith("FLAT"), v["text"]
    assert v["ratio_span"] < 0.15


def test_losing_the_big_halos_reads_INVERTED():
    hr = [1000, 1000, 1000, 1000, 0, 0, 0]
    base = [950, 800, 500, 300, 0, 0, 0]
    v = _verdict(hr, base)
    assert v["text"].startswith("INVERTED"), v["text"]


def test_bins_below_the_first_mass_edge_are_never_counted():
    """Rockstar's own floor moves with the field; a bin straddling it would
    report the finder's threshold as a physical deficit."""
    cat = make_cat(pos=[[10, 10, 10], [20, 20, 20]], num_p=[10, 60],
                   parent_ids=[-1, -1], rvir_kpc=[10.0, 10.0])
    t = census.table(census.classify(cat, 200), top=True, region_only=False)
    assert t["total"] == 1
    assert sum(t["by_mass"]) == 1
