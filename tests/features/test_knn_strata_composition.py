"""The gating diagnostic for the catalog-free strata arm.

There is no box here, so what is testable is the arithmetic that the verdict is
read off: the strata must be the same equal-count dense-first groups the loss
uses, the interior cut must be the loss's cut, the enrichments must be
enrichments (ratios to a base rate, not raw fractions of a rare class), and the
rank correlation must not silently report a number on a constant column.

The one that matters most is ``test_composition_recovers_a_planted_enrichment``:
if a density stratum really does pick out the dense population, this script has
to SAY so, and if it does not, it has to say that instead -- the stop rule is
only worth having if both outcomes are reachable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "features"):
    sys.path.insert(0, str(p))

import knn_strata_composition as ksc  # noqa: E402
from cosmo_sr.features.knn_cdf import KnnConfig, strata_partition  # noqa: E402

BOX = 100.0


def test_strata_groups_match_the_loss_partition():
    """The diagnostic and the loss must split the SAME way, or the gate is on a
    different statistic than the arm it gates."""
    rng = np.random.default_rng(3)
    z = rng.normal(size=(240, 4))
    cfg = KnnConfig(ks=(8, 32, 64, 128), n_queries=240, strata=8,
                    buffer_cells=0.0)
    mine = ksc.strata_groups(z[:, 0], 8)
    theirs = strata_partition(torch.tensor(z), cfg)
    for a, b in zip(mine, theirs):
        assert sorted(a.tolist()) == sorted(b.numpy().tolist())


def test_spearman_handles_ties_and_refuses_constants():
    assert ksc.spearman(np.arange(10.0), np.arange(10.0)) == pytest.approx(1.0)
    assert ksc.spearman(np.arange(10.0), -np.arange(10.0)) == pytest.approx(-1.0)
    # A constant column has no ranking and must read nan, not 0 or 1.
    assert np.isnan(ksc.spearman(np.ones(10), np.arange(10.0)))
    assert np.isnan(ksc.spearman(np.arange(2.0), np.arange(2.0)))
    # Ties are averaged, so a two-level column still gives a finite number.
    r = ksc.spearman(np.array([0, 0, 1, 1, 2.0]), np.array([1, 2, 3, 4, 5.0]))
    assert np.isfinite(r) and r > 0.8


def test_tile_interior_is_eulerian_and_centred_on_the_particles():
    """The cut follows ``interior_queries``: Eulerian, on the tile's own
    centroid, so a tile that has moved a whole width is still measured."""
    ng, tile = 64, 16
    cell = BOX / ng
    rows = np.arange(4096)
    pos = np.zeros((4096, 3))
    # A compact tile sitting far from its Lagrangian centre.
    rng = np.random.default_rng(7)
    pos[:] = rng.uniform(0, tile * cell, size=(4096, 3)) + np.array([70.0, 5, 5])
    inside = ksc.tile_interior(pos, rows, 0, buffer_cells=2.0, ng=ng,
                               tile=tile, box=BOX)
    assert 0 < inside.size < rows.size          # a real cut, not all or nothing
    # Everything kept is within the buffered half-side of the CIRCULAR mean --
    # which is the centre the code uses and is not the arithmetic mean, because
    # the arithmetic mean of a wrapped coordinate is not on the box.
    half = 0.5 * tile * cell - 2.0 * cell
    ang = np.mod(pos[rows], BOX) * (2 * np.pi / BOX)
    c = np.mod(np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0))
               * BOX / (2 * np.pi), BOX)
    d = np.abs(np.mod(pos[inside], BOX) - c)
    d = np.abs(d - BOX * np.round(d / BOX))
    assert d.max() <= half + 1e-9
    # A zero buffer keeps strictly more.
    assert ksc.tile_interior(pos, rows, 0, buffer_cells=0.0, ng=ng, tile=tile,
                             box=BOX).size > inside.size


def _planted(n=2000, n_dense=200, seed=5):
    """``(z_field, z_hr, cls)`` where the dense particles ARE the sub class."""
    rng = np.random.default_rng(seed)
    cls = np.full(n, ksc.CLASSES.index("host"), dtype=np.int8)
    cls[:n_dense] = ksc.SUB
    z = rng.normal(0.0, 0.3, size=(n, 4))
    z[:n_dense] -= 2.0                      # dense: small log r at every k
    return z, z.copy(), cls


def test_composition_recovers_a_planted_enrichment():
    z_f, z_h, cls = _planted()
    m = ksc.measure_tile(z_f, z_h, cls, 8, (0,), n_q=8, dead_zone=0.02,
                         strat_col=0)
    # The densest eighth is 250 rows and 200 of them are the planted class.
    assert m["composition"][0]["sub_50_200"] > 0.7
    assert m["purity_enrichment"] > 5.0
    assert m["recall_charged"] > 0.95
    assert m["recall_enrichment"] > 7.0     # 1.0 / 0.125 is the ceiling at 8
    # Field against itself: every loss form is zero.
    assert m["loss_dense"] == pytest.approx(0.0)
    assert m["loss_class_sub"] == pytest.approx(0.0)


def test_composition_reports_no_enrichment_when_density_is_unrelated():
    """The outcome that FAILS the stop rule has to be reachable, or the gate is
    decoration. Here the class is assigned at random, independent of density."""
    rng = np.random.default_rng(9)
    z = rng.normal(size=(2000, 4))
    cls = np.full(2000, ksc.CLASSES.index("host"), dtype=np.int8)
    cls[rng.choice(2000, 200, replace=False)] = ksc.SUB
    m = ksc.measure_tile(z, z.copy(), cls, 8, (0,), n_q=8, dead_zone=0.02,
                         strat_col=0)
    assert 0.4 < m["purity_enrichment"] < 2.0
    assert 0.4 < m["recall_enrichment"] < 2.0


def test_charging_more_strata_raises_recall_and_lowers_enrichment():
    z_f, z_h, cls = _planted()
    one = ksc.measure_tile(z_f, z_h, cls, 8, (0,), 8, 0.02, 0)
    two = ksc.measure_tile(z_f, z_h, cls, 8, (0, 1), 8, 0.02, 0)
    assert two["frac_charged"] > one["frac_charged"]
    assert two["recall_charged"] >= one["recall_charged"]
    # Widening the cut cannot make it MORE selective.
    assert two["purity_enrichment"] <= one["purity_enrichment"]


def test_unpaired_loss_refuses_a_sample_too_small_for_its_bins():
    z = np.random.default_rng(1).normal(size=(4, 3))
    assert np.isnan(ksc.unpaired_loss(z, z, n_q=16, dead_zone=0.02))
    assert np.isfinite(ksc.unpaired_loss(z, z, n_q=2, dead_zone=0.02))


def test_unpaired_loss_is_the_torch_loss():
    """Same bins, same hinge, same number as ``knn_cdf_loss`` -- the diagnostic
    must not be scoring a slightly different loss than the trainer charges."""
    from cosmo_sr.features.knn_cdf import knn_cdf_loss
    rng = np.random.default_rng(2)
    zc, zh = rng.normal(size=(400, 3)), rng.normal(size=(400, 3))
    cfg = KnnConfig(ks=(2, 4, 8), n_queries=400, n_quantiles=16, dead_zone=0.02,
                    buffer_cells=0.0)
    mine = ksc.unpaired_loss(zc, zh, 16, 0.02)
    theirs = float(knn_cdf_loss(torch.tensor(zc), torch.tensor(zh), cfg)[0])
    assert mine == pytest.approx(theirs, rel=1e-10)
