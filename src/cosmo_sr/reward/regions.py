"""Periodic region aggregation over the (8^3) tile grid.

Why this module exists
----------------------
The per-tile credit in :mod:`cosmo_sr.reward.tiles` is *exact* -- the tile
statistics sum to the whole-box catalog to numerical precision -- but it is not
*stable*: measured on set0, ~92% of a tile's ``dS`` label churns between two
frozen SR2 seeds of the same box and cancels on pooling
(:func:`cosmo_sr.reward.tiles.majority_weights`). A cluster's bound particles
originate in many Lagrangian tiles, so any reshuffle of which particle Rockstar
binds moves weight between tiles without moving the halo.

The obvious response -- give up on locality and predict a whole-box scalar --
throws away the thing that makes the prediction checkable and actionable. This
module takes the middle path the plan asks for: pool the exact tile statistics
into **regions** of ``width`` tiles on a side, from ``width = 1`` (the tiles
themselves) up to ``width = 8`` (the whole box), and let the diagnostic measure
at which scale the label becomes rankable. A region is still a local unit the
actor can move, but it is large enough that a halo's Lagrangian footprint mostly
falls inside one region rather than being split across the boundary.

The grid is Lagrangian and periodic
------------------------------------
Tiles cover the box as a disjoint periodic cover (see
:class:`cosmo_sr.reward.tiles.TileGrid`). A region of width ``w`` is a
``w x w x w`` block of tiles, and because the box wraps, a region may straddle
the periodic boundary. For a fixed ``w`` the box admits several *non-overlapping
partitions* into regions, one per origin **offset**: shifting the origin by one
tile along an axis produces a genuinely different grouping of tiles (until the
shift reaches ``w``, at which point the partition repeats). Enumerating the
offsets is what lets the diagnostic ask whether a candidate's ordering is stable
to *where* the region boundaries were drawn, not just to the seed.

Two exact identities anchor everything
--------------------------------------
* every partition contains every tile exactly once (:meth:`RegionPartition.check`);
* summing all regions of *any* partition reproduces the whole-box statistics,
  because summation is associative and the tiles already sum to the box.

``width = 1`` reproduces the tile summaries and ``width = 8`` yields a single
whole-box region; both are limits of the same code, not special cases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple

import numpy as np

from .tiles import TileGrid

__all__ = [
    "REGION_WIDTHS",
    "RegionGrid",
    "RegionPartition",
    "aggregate_tile_counts",
    "aggregate_tile_volume",
    "region_ids_from_tile_ids",
    "stack_tile_bags",
    "tile_bags",
]

#: The region widths the plan sweeps over, on the 8-per-axis tile grid.
REGION_WIDTHS: Tuple[int, ...] = (1, 2, 4, 8)


def region_ids_from_tile_ids(
    tile_ids: np.ndarray,
    *,
    width: int,
    n_per_axis: int = 8,
    offset: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Region id of each flat tile id, for the ``width``-tile partition at ``offset``.

    The single implementation of "which region is this tile in".
    :meth:`RegionGrid.partition` calls it, and so does the proxy's supervision
    grouping -- which needs the mapping for an arbitrary array of tile ids from a
    table, without constructing a :class:`TileGrid`. Two implementations of this
    arithmetic would eventually disagree about the periodic wrap, and the
    disagreement would surface as a proxy trained on one grouping and scored on
    another, which is silent and fatal.

    At ``width = 1`` this is the **identity**: ``region_id == tile_id``. That is
    not a special case in the code, it is what the arithmetic gives, and it is
    the property the deployed configuration relies on -- ``region_width: 1``
    means the supervision unit is the tile itself, so nothing about the existing
    per-tile labels has to be re-derived.
    """
    ids = np.asarray(tile_ids, dtype=np.int64)
    n, w = int(n_per_axis), int(width)
    if w <= 0 or n % w:
        raise ValueError(f"region width {w} does not divide tiles-per-axis {n}")
    if ids.size and (ids.min() < 0 or ids.max() >= n ** 3):
        raise ValueError(
            f"tile ids must lie in [0, {n ** 3}); got "
            f"[{int(ids.min())}, {int(ids.max())}]")
    rpa = n // w
    ox, oy, oz = (int(o) % n for o in offset)
    # Tile (ix, iy, iz) from its flat id, matching TileGrid.coord/index.
    ix = ids // (n * n)
    iy = (ids // n) % n
    iz = ids % n
    rx = ((ix - ox) % n) // w
    ry = ((iy - oy) % n) // w
    rz = ((iz - oz) % n) // w
    return ((rx * rpa + ry) * rpa + rz).astype(np.int64)


@dataclass(frozen=True)
class RegionGrid:
    """A tile grid coarsened into ``width``-tile cubic regions.

    ``width`` must divide the tiles-per-axis (8 for the deployed geometry), so a
    region is a whole number of tiles and the periodic cover stays exact.
    """

    tile_grid: TileGrid
    width: int

    def __post_init__(self) -> None:
        n = self.tile_grid.n_per_axis
        w = int(self.width)
        if w <= 0 or n % w:
            raise ValueError(
                f"region width {w} does not divide tiles-per-axis {n}")

    @property
    def n_per_axis(self) -> int:
        """Tiles per axis of the underlying tile grid."""
        return self.tile_grid.n_per_axis

    @property
    def regions_per_axis(self) -> int:
        return self.n_per_axis // int(self.width)

    @property
    def n_regions(self) -> int:
        """Regions in one partition."""
        return self.regions_per_axis ** 3

    @property
    def tiles_per_region(self) -> int:
        return int(self.width) ** 3

    @property
    def region_volume_mpc3(self) -> float:
        """Nominal region volume; also its effective volume (nothing is masked)."""
        return self.tile_grid.tile_volume_mpc3 * self.tiles_per_region

    def valid_offsets(self) -> List[Tuple[int, int, int]]:
        """Every origin offset that yields a *distinct* partition.

        Shifting an axis origin by ``width`` reproduces the same grouping of
        tiles (only the region *labels* rotate), so the distinct offsets per axis
        are ``0..width-1``. When a region spans the whole axis
        (``regions_per_axis == 1``, i.e. ``width == n_per_axis``) every offset
        yields the same single region, so only offset 0 is kept -- which is why
        ``width = 8`` reports one whole-box partition rather than 512 copies.
        """
        per_axis = 1 if self.regions_per_axis == 1 else int(self.width)
        return [(ox, oy, oz)
                for ox in range(per_axis)
                for oy in range(per_axis)
                for oz in range(per_axis)]

    def partition(self, offset: Tuple[int, int, int] = (0, 0, 0)) -> "RegionPartition":
        """The non-overlapping periodic partition at ``offset``."""
        n = self.n_per_axis
        ox, oy, oz = (int(o) % n for o in offset)
        ids = np.arange(self.tile_grid.n_tiles, dtype=np.int64)
        return RegionPartition(
            grid=self, offset=(ox, oy, oz),
            region_of_tile=region_ids_from_tile_ids(
                ids, width=int(self.width), n_per_axis=n, offset=(ox, oy, oz)))

    def partitions(self) -> Iterator["RegionPartition"]:
        for off in self.valid_offsets():
            yield self.partition(off)


@dataclass
class RegionPartition:
    """One non-overlapping periodic assignment of tiles to regions.

    ``region_of_tile[t]`` is the region id (``0..n_regions-1``) of tile ``t``.
    That single array is the whole partition: every aggregation and the
    partition-of-unity check are derived from it, so there is no second
    representation to keep in sync.
    """

    grid: RegionGrid
    offset: Tuple[int, int, int]
    region_of_tile: np.ndarray

    def __post_init__(self) -> None:
        self.region_of_tile = np.asarray(self.region_of_tile, dtype=np.int64)
        self.check()

    @property
    def n_regions(self) -> int:
        return self.grid.n_regions

    def tiles_of(self, region_id: int) -> np.ndarray:
        """Sorted tile ids belonging to ``region_id``."""
        return np.nonzero(self.region_of_tile == int(region_id))[0]

    def all_tiles_of(self) -> Dict[int, np.ndarray]:
        """``region_id -> sorted tile ids`` for every region in the partition."""
        return {r: self.tiles_of(r) for r in range(self.n_regions)}

    def check(self) -> None:
        """Every tile appears in exactly one region, and every region is full.

        Raises rather than returning a flag: a partition that is silently not a
        partition breaks the summation identity every downstream number relies
        on, and the failure would otherwise surface far away as a small
        additivity residual.
        """
        rot = self.region_of_tile
        n_tiles = self.grid.tile_grid.n_tiles
        if rot.shape != (n_tiles,):
            raise ValueError(
                f"region_of_tile has shape {rot.shape}, expected ({n_tiles},)")
        if rot.min() < 0 or rot.max() >= self.n_regions:
            raise ValueError("a tile was assigned to a region outside range")
        counts = np.bincount(rot, minlength=self.n_regions)
        if not np.all(counts == self.grid.tiles_per_region):
            bad = np.nonzero(counts != self.grid.tiles_per_region)[0]
            raise ValueError(
                f"partition offset {self.offset} is not a cover: regions "
                f"{bad[:5].tolist()} hold {counts[bad][:5].tolist()} tiles, "
                f"expected {self.grid.tiles_per_region}")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate_tile_counts(
    partition: RegionPartition, counts: Mapping[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Sum per-tile ``(n_tiles, D)`` count arrays into ``(n_regions, D)`` exactly.

    Exact because it is a plain grouped sum: region ``r``'s statistic is the sum
    of its member tiles', and the tiles already sum to the whole box, so summing
    all regions of the partition reproduces the whole-box statistic to numerical
    precision. Any count key is accepted (``n_sub``, ``n_host``,
    ``occ_numerator``); each is aggregated independently.
    """
    rot = partition.region_of_tile
    n_r = partition.n_regions
    out: Dict[str, np.ndarray] = {}
    for k, arr in counts.items():
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim != 2 or a.shape[0] != rot.shape[0]:
            raise ValueError(
                f"count '{k}' has shape {a.shape}, expected "
                f"({rot.shape[0]}, D)")
        acc = np.zeros((n_r, a.shape[1]), dtype=np.float64)
        np.add.at(acc, rot, a)
        out[k] = acc
    return out


def aggregate_tile_volume(
    partition: RegionPartition, volume: np.ndarray
) -> np.ndarray:
    """Sum a ``(n_tiles,)`` volume vector into ``(n_regions,)`` by exact membership."""
    rot = partition.region_of_tile
    v = np.asarray(volume, dtype=np.float64).reshape(-1)
    if v.shape[0] != rot.shape[0]:
        raise ValueError(
            f"volume has {v.shape[0]} entries, expected {rot.shape[0]}")
    acc = np.zeros(partition.n_regions, dtype=np.float64)
    np.add.at(acc, rot, v)
    return acc


def tile_bags(
    partition: RegionPartition, features: np.ndarray
) -> List[np.ndarray]:
    """Per-region ``(tiles_per_region, F)`` feature bags, region features kept whole.

    Unlike the count aggregation this does **not** sum: a region's input to a
    DeepSets-style proxy is the *set* of its tiles' feature vectors, and summing
    them here would destroy exactly the within-region structure the region proxy
    exists to read. Tiles within a bag are ordered by ascending tile id, so the
    bag is deterministic; a permutation-invariant encoder does not depend on the
    order, but a reproducible one makes the bag hashable and comparable.
    """
    x = np.asarray(features)
    rot = partition.region_of_tile
    if x.shape[0] != rot.shape[0]:
        raise ValueError(
            f"features has {x.shape[0]} rows, expected {rot.shape[0]}")
    return [x[partition.tiles_of(r)] for r in range(partition.n_regions)]


def stack_tile_bags(
    partition: RegionPartition, features: np.ndarray
) -> np.ndarray:
    """The tile bags stacked into ``(n_regions, tiles_per_region, F)``.

    Every region holds exactly ``tiles_per_region`` tiles, so the bags are the
    same size and stack without padding -- the shape a shared tile encoder
    consumes before the within-region sum.
    """
    bags = tile_bags(partition, features)
    return np.stack(bags, axis=0)
