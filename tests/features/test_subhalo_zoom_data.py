"""Selection and measurement logic of the per-subhalo zoom builder.

Two things here can quietly ruin the figure without failing.

*Which subhalos get drawn.* The arms' gains sit in the 50-200 particle bin, so a
panel of the nine heaviest objects would show exactly the population nobody
doubted and would answer the wrong question. ``spread`` has to actually reach
both ends of the range.

*What the panels contain.* Every arm must be binned in the same window with the
same slab, or the three columns of the figure are not comparable and the whole
tuned-vs-frozen reading is an artefact of two different crops.
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

from subhalo_zoom_data import measure, pick_subhalos  # noqa: E402

BOX = 100.0
HOST = np.array([50.0, 50.0, 50.0])


def _catalog(offsets: np.ndarray, num_p: np.ndarray) -> HaloCatalog:
    """A catalog of subhalos at ``HOST + offsets``, plus the host itself."""
    n = offsets.shape[0]
    pos = np.concatenate([HOST[None, :], HOST[None, :] + offsets], axis=0)
    npart = np.concatenate([[100000], np.asarray(num_p)]).astype(np.int64)
    return HaloCatalog(
        ids=np.arange(n + 1, dtype=np.int64),
        parent_ids=np.concatenate([[-1], np.zeros(n, dtype=np.int64)]),
        mvir=npart.astype(float) * 5.8e8,
        rvir=np.concatenate([[1500.0], 30.0 * (npart[1:] / 50.0) ** (1 / 3)]),
        vmax=np.zeros(n + 1), pos=pos, vel=np.zeros((n + 1, 3)),
        num_p=npart)


def test_spread_reaches_both_ends_where_top_does_not():
    rng = np.random.default_rng(0)
    num_p = np.unique(np.round(np.logspace(np.log10(50), np.log10(5000), 40)))
    off = rng.normal(0.0, 0.4, size=(num_p.size, 3))
    cat = _catalog(off, num_p)
    sub = cat.subhalos()

    spread = pick_subhalos(cat, HOST, 1.5, 50, 6, "spread", BOX)
    top = pick_subhalos(cat, HOST, 1.5, 50, 6, "top", BOX)

    assert spread.size == 6 and top.size == 6
    # "top" is by construction the heavy end only.
    assert sub.num_p[top].min() > 1000
    # "spread" has to reach down to the bin the arms actually populate.
    assert sub.num_p[spread].min() < 100
    assert sub.num_p[spread].max() > 1000


def test_selection_respects_the_aperture_and_the_particle_floor():
    off = np.array([[0.2, 0.0, 0.0],     # inside
                    [0.3, 0.0, 0.0],     # inside, but too few particles
                    [8.0, 0.0, 0.0]])    # outside the aperture
    cat = _catalog(off, np.array([300, 20, 900]))
    got = pick_subhalos(cat, HOST, 1.5, 50, 9, "top", BOX)
    assert got.size == 1
    assert int(cat.subhalos().num_p[got][0]) == 300


def test_selection_is_periodic():
    """A host at the box face must still find the subhalos wrapped around it."""
    off = np.array([[0.2, 0.0, 0.0]])
    cat = _catalog(off, np.array([300]))
    # Move everything so the host sits just inside 0 and the subhalo wraps.
    cat.pos[:] = (cat.pos - HOST[None, :] + np.array([0.1, 50.0, 50.0])) % BOX
    got = pick_subhalos(cat, np.array([0.1, 50.0, 50.0]), 1.5, 50, 9, "top", BOX)
    assert got.size == 1


def test_every_arm_is_measured_in_the_same_window():
    rng = np.random.default_rng(3)
    centre = np.array([50.0, 50.0, 50.0])
    r, depths, bins = 0.05, [1.0, 5.0, -1.0], 16
    # Dense enough that the aperture holds ~150 background particles: a
    # ratio measured on 0.1 of a particle is not a test of anything.
    uniform = rng.uniform(49.5, 50.5, size=(200_000, 3))
    clump = centre[None, :] + rng.normal(0.0, 0.02, size=(4_000, 3))
    pos = {"tuned": np.concatenate([uniform, clump]),
           "frozen": uniform,
           "hr": np.concatenate([uniform, clump])}

    host_centre = centre + np.array([0.0, 0.0, 1.2])
    panels, stats = measure(pos, centre, r, depths, axis=1,
                            half_xy=4.0 * r, bins=bins, host_centre=host_centre)

    for name in ("tuned", "frozen", "hr"):
        assert panels[name].shape == (len(depths), bins, bins)
        # The host-relative pair is present only because host_centre was given;
        # see test_measure_omits_host_numbers_without_a_host.
        assert set(stats[name]) == {"contrast_3d", "contrast_3d_core",
                                    "host_shell", "host_shell_core", "ladder"}
        assert len(stats[name]["ladder"]) == len(depths)
        # A wider slab can only add particles to the same sky window.
        totals = panels[name].sum(axis=(1, 2))
        assert totals[0] <= totals[1] <= totals[2]

    # The measurement has to separate the arm that has a clump from the one that
    # does not -- in the thin slab, which is the panel the figure leads with.
    assert stats["tuned"]["ladder"][0]["ratio"] > 5.0 * stats["frozen"]["ladder"][0]["ratio"]
    assert stats["frozen"]["ladder"][0]["ratio"] == pytest.approx(1.0, abs=0.5)


def test_full_column_erodes_the_separation_the_thin_slab_shows():
    """Why the figure needs the slab: the same two arms through a long column."""
    rng = np.random.default_rng(5)
    centre = np.array([50.0, 50.0, 50.0])
    r = 0.05
    # A long column along the projection axis, as the host tiles give.
    uniform = np.stack([rng.uniform(49.5, 50.5, 400_000),
                        rng.uniform(38.0, 62.0, 400_000),
                        rng.uniform(49.5, 50.5, 400_000)], axis=1)
    clump = centre[None, :] + rng.normal(0.0, 0.02, size=(4_000, 3))
    pos = {"tuned": np.concatenate([uniform, clump]), "frozen": uniform,
           "hr": uniform}
    _p, stats = measure(pos, centre, r, [1.0, -1.0], axis=1, half_xy=4 * r, bins=16)

    thin = stats["tuned"]["ladder"][0]["ratio"]
    full = stats["tuned"]["ladder"][1]["ratio"]
    assert thin > full
    assert thin / full > 5.0


def test_measure_omits_host_numbers_without_a_host():
    """No host centre, no host-relative numbers -- absent, never defaulted.

    A zero or a box centre silently substituted for the host would produce a
    background shell at the wrong radius, which is a wrong number that still
    looks like a number.
    """
    rng = np.random.default_rng(7)
    centre = np.array([50.0, 50.0, 50.0])
    pos = {"tuned": rng.uniform(49.5, 50.5, size=(50_000, 3))}
    _p, stats = measure(pos, centre, 0.05, [1.0, -1.0], axis=1,
                        half_xy=0.2, bins=8)
    assert "host_shell" not in stats["tuned"]
    assert "contrast_3d_core" in stats["tuned"]


def test_core_aperture_sees_what_the_full_radius_averages_away():
    """A concentrated core inside a diffuse envelope: the aperture decides.

    This is the tidal-truncation problem in miniature. Measured to the full
    radius, the dense core is diluted by the envelope down towards the
    background; measured to 0.3 of it, the core is plain. HR's real subhalos
    live in this regime, which is why the core aperture exists.
    """
    rng = np.random.default_rng(11)
    centre = np.array([50.0, 50.0, 50.0])
    r = 0.10
    bg = rng.uniform(49.6, 50.4, size=(300_000, 3))
    core = centre[None, :] + rng.normal(0.0, 0.012, size=(1_200, 3))
    pos = {"tuned": np.concatenate([bg, core])}
    _p, stats = measure(pos, centre, r, [1.0], axis=1, half_xy=0.4, bins=8,
                        core_frac=0.3)
    full = stats["tuned"]["contrast_3d"]["ratio"]
    small = stats["tuned"]["contrast_3d_core"]["ratio"]
    assert small > 3.0 * full
