"""LR-Rockstar host information, put back on the LR Lagrangian lattice.

What this is for
----------------
A halo found in the **LR** box is a statement about where structure wants to
form, and it is available before any super-resolution happens. To hand it to a
generator that produces HR tiles, it has to be expressed on the lattice the
generator indexes -- not as a catalog row. This module does that conversion and
nothing else: it does not touch SR2, and it fits no model.

Why it is exact rather than approximate
---------------------------------------
Our GADGET2 dumps set each particle ID to the flat C-order index of its
Lagrangian lattice site (:func:`cosmo_sr.eval.particles.field_to_particles`
builds them as ``arange(Ng**3)``), and Rockstar preserves 4-byte IDs verbatim.
With ``FULL_PARTICLE_CHUNKS = 1`` it prints the member IDs of every halo it
outputs. So "which LR lattice site does this bound particle come from" is a
lookup, not a nearest-neighbour match --
:func:`cosmo_sr.eval.particle_identity.stream_owner_assignment` already inverts
the member table into ``owner[particle_id]``, and everything here is built on
top of that array.

The five channels
-----------------
Per LR site ``i`` (``ng_lr**3`` of them; ``0`` everywhere no host owns the site,
see *Missing data* below):

``host_member``
    1.0 if the site's particle is bound to some host, else 0.0. The mask that
    makes the other channels interpretable -- a 0 in ``log_host_mass`` means
    "no host", not "a 1 Msun/h host".
``log_host_mass``
    ``log10(Mvir / (Msun/h))`` of the owning **host** (leaf attribution lifted
    to the top-level object with
    :func:`cosmo_sr.eval.particle_identity.remap_to_roots`, so a satellite's
    particles carry the host's mass, which is the quantity that sets how much
    substructure the region should hold).
``dq_over_rl``
    3 channels: the periodic Lagrangian offset of the site from its host's
    Lagrangian centre, divided by that host's Lagrangian radius. Dimensionless,
    so a 1e12 and a 1e14 host look alike in it, and the radial position inside
    the host is the thing being encoded.
``host_fraction_per_tile``
    Per (host, tile): the fraction of the host's member particles whose
    Lagrangian site falls in that tile. ``sum_t f[h, t] == 1`` exactly. Stored
    as the ``(H, n_tiles)`` table on :class:`HostTable`; the per-site channel
    of the same name is ``f[host(i), tile(i)]`` -- "how much of my host is in my
    own tile", i.e. how much of this host the generator can see in one forward
    pass.
``subhalo_budget`` (optional)
    Given a per-host count ``N_h``, every member particle of ``h`` carries
    ``lambda_i = N_h / N_particles,h``, so ``sum_{i in h} lambda_i == N_h``.
    A tile's share of the budget is then just the sum of ``lambda`` over the
    tile, which is what makes "how many subhalos does this tile owe" a local
    quantity.

Coordinate convention and shapes
--------------------------------
Site ``(a, b, c)`` of the ``ng_lr**3`` lattice has particle id
``(a * ng_lr + b) * ng_lr + c`` and Lagrangian position
``((a, b, c) + 0.5) * boxsize / ng_lr`` in Mpc/h -- the cell-centre convention
of ``field_to_particles``, so these features and the particle positions the
catalog was built from cannot drift apart.

Features are stored **only** at ``ng_lr**3``. A dense multi-channel
``ng_hr**3`` array is never materialised: the HR view of a tile is a
nearest-neighbour broadcast of the tile's LR crop, produced on demand by
:meth:`LagrangianHostFeatures.tile_hr`. One LR site maps to exactly
``(ng_hr // ng_lr)**3`` HR children, and those children all lie in the same
tile, which is what makes the broadcast well defined
(:meth:`LagrangianGrid.hr_children`).

Missing data
------------
Rockstar leaves a particle unowned when it is genuinely unbound or when it is
bound only to a clump below ``MIN_HALO_OUTPUT_SIZE``. Both are common -- most of
the volume is not in a halo -- and both are represented the same way: the site
gets ``host_index = -1``, ``host_member = 0`` and exact zeros in every other
channel. Zero is a *value* in ``dq_over_rl`` (the host centre), so a consumer
must read ``host_member`` alongside it rather than treating 0 as absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..eval.particle_identity import (
    UNBOUND, build_owner_index, child_map, descendants_of, periodic_delta,
    remap_to_roots,
)
from ..eval.rockstar import HaloCatalog
from ..reward.tiles import TileGrid, tile_of_particle_id

__all__ = [
    "CHANNELS",
    "HostTable",
    "LagrangianGrid",
    "LagrangianHostFeatures",
    "build_host_features",
    "lagrangian_lattice_positions",
    "normalization_report",
    "periodic_circular_mean",
]

# Channel order used by `stack_lr` / `tile_hr`. `dq_over_rl` expands to three.
CHANNELS: Tuple[str, ...] = (
    "host_member",
    "log_host_mass",
    "dq_over_rl",
    "host_fraction_per_tile",
    "subhalo_budget",
)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LagrangianGrid:
    """The LR lattice, the HR lattice, and the SR2 tile partition of both.

    The tile partition is *not* a new choice: ``tile_hr = 64`` is one SR2 output
    tile (``nsplit = 8`` over ``ng_hr = 512``), the same unit
    :class:`cosmo_sr.reward.tiles.TileGrid` assigns credit to. Because the
    upsample factor divides the tile size, the tile boundaries fall on LR cell
    boundaries too, so a tile is an integer block of ``tile_lr**3`` LR sites and
    every HR child of an LR site is in its parent's tile.
    """

    ng_lr: int = 64
    ng_hr: int = 512
    tile_hr: int = 64
    boxsize_mpc_h: float = 100.0

    def __post_init__(self) -> None:
        if self.ng_lr <= 0 or self.ng_hr % self.ng_lr:
            raise ValueError(
                f"ng_lr={self.ng_lr} does not divide ng_hr={self.ng_hr}")
        if self.tile_hr <= 0 or self.ng_hr % self.tile_hr:
            raise ValueError(
                f"tile_hr={self.tile_hr} does not divide ng_hr={self.ng_hr}")
        if self.tile_hr % self.upsample:
            raise ValueError(
                f"tile_hr={self.tile_hr} is not a whole number of LR cells "
                f"(upsample={self.upsample}); tiles would cut LR sites in half")

    # -- sizes ------------------------------------------------------------
    @property
    def upsample(self) -> int:
        return self.ng_hr // self.ng_lr

    @property
    def tile_lr(self) -> int:
        """Tile edge in LR sites."""
        return self.tile_hr // self.upsample

    @property
    def n_per_axis(self) -> int:
        return self.ng_hr // self.tile_hr

    @property
    def n_tiles(self) -> int:
        return self.n_per_axis ** 3

    @property
    def n_lr(self) -> int:
        return self.ng_lr ** 3

    @property
    def cell_mpc_h(self) -> float:
        """LR cell edge."""
        return float(self.boxsize_mpc_h) / self.ng_lr

    # -- the two TileGrids ------------------------------------------------
    def hr_tile_grid(self) -> TileGrid:
        """The generator's tile grid, indexed by HR particle id."""
        return TileGrid(ng_hr=self.ng_hr, tile_hr=self.tile_hr,
                        boxsize_mpc_h=self.boxsize_mpc_h)

    def lr_tile_grid(self) -> TileGrid:
        """The same tiles, indexed by LR particle id.

        Same ``n_per_axis``, so :func:`tile_of_particle_id` returns the *same*
        tile ids as :meth:`hr_tile_grid` does for the corresponding HR children
        -- reusing the existing function rather than writing a second one that
        could disagree with it.
        """
        return TileGrid(ng_hr=self.ng_lr, tile_hr=self.tile_lr,
                        boxsize_mpc_h=self.boxsize_mpc_h)

    def tile_of_lr_site(self, lr_ids) -> np.ndarray:
        return tile_of_particle_id(lr_ids, self.lr_tile_grid())

    def lr_slices(self, tile_id: int) -> Tuple[slice, slice, slice]:
        """The LR crop of one tile, as slices into an ``(ng_lr,)*3`` array."""
        s = self.tile_lr
        ix, iy, iz = self.lr_tile_grid().coord(int(tile_id))
        return (slice(ix * s, (ix + 1) * s),
                slice(iy * s, (iy + 1) * s),
                slice(iz * s, (iz + 1) * s))

    def hr_children(self, lr_id: int) -> np.ndarray:
        """The ``upsample**3`` HR particle ids covered by one LR site."""
        n = self.ng_lr
        i = int(lr_id)
        if not 0 <= i < self.n_lr:
            raise IndexError(f"LR id {lr_id} outside 0..{self.n_lr - 1}")
        a, b, c = i // (n * n), (i // n) % n, i % n
        f, ngh = self.upsample, self.ng_hr
        u = np.arange(f, dtype=np.int64)
        ix = (f * a + u).reshape(f, 1, 1)
        iy = (f * b + u).reshape(1, f, 1)
        iz = (f * c + u).reshape(1, 1, f)
        return ((ix * ngh + iy) * ngh + iz).reshape(-1)


def lagrangian_lattice_positions(grid: LagrangianGrid) -> np.ndarray:
    """``(ng_lr**3, 3)`` cell-centre Lagrangian positions in Mpc/h.

    The unperturbed lattice, indexed by particle id. Mirrors the ``q`` that
    :func:`cosmo_sr.eval.particles.field_to_particles` adds the displacement to,
    so the two conventions are the same by construction.
    """
    n = grid.ng_lr
    q = (np.arange(n, dtype=np.float64) + 0.5) * grid.cell_mpc_h
    out = np.empty((n, n, n, 3), dtype=np.float64)
    out[..., 0] = q.reshape(n, 1, 1)
    out[..., 1] = q.reshape(1, n, 1)
    out[..., 2] = q.reshape(1, 1, n)
    return out.reshape(-1, 3)


def periodic_circular_mean(pos: np.ndarray, box: float) -> np.ndarray:
    """Centre of a set of periodic coordinates, per axis.

    A plain mean is wrong for a host straddling the box edge -- sites at 0.1 and
    99.9 Mpc/h average to the middle of the box instead of the boundary they
    actually share. The circular mean of the phase angles has no such seam, and
    it agrees with the plain mean whenever the set does not wrap.
    """
    p = np.asarray(pos, dtype=np.float64)
    if p.size == 0:
        return np.zeros(3, dtype=np.float64)
    theta = 2.0 * np.pi * p / float(box)
    ang = np.arctan2(np.sin(theta).mean(axis=0), np.cos(theta).mean(axis=0))
    return np.mod(ang * float(box) / (2.0 * np.pi), float(box))


# --------------------------------------------------------------------------
# Per-host metadata
# --------------------------------------------------------------------------

@dataclass
class HostTable:
    """One row per host that owns at least one LR site.

    Rockstar ids live here as **metadata**, never as a feature: an id is a
    nominal label whose numeric value carries no physics, and feeding it to a
    model would invite it to memorise this particular catalog.
    """

    host_id: np.ndarray        # (H,) int64, Rockstar catalog id -- metadata only
    mvir: np.ndarray           # (H,) Msun/h
    rvir_kpc_h: np.ndarray     # (H,) Rockstar Rvir (Eulerian), for reference
    num_p_catalog: np.ndarray  # (H,) the catalog's own num_p for the host row
    n_particles: np.ndarray    # (H,) LR sites attributed to the host (+subs)
    center_lag: np.ndarray     # (H,3) Mpc/h, periodic circular mean
    r_lag_mpc_h: np.ndarray    # (H,) volume-equivalent Lagrangian radius
    rms_lag_mpc_h: np.ndarray  # (H,) rms |dq|, a shape diagnostic
    tile_frac: np.ndarray      # (H,n_tiles) float32, rows sum to 1
    n_sub: np.ndarray          # (H,) the per-host budget N_h
    n_sub_source: str = ""     # how n_sub was obtained (provenance)

    @property
    def n_hosts(self) -> int:
        return int(self.host_id.shape[0])

    def row_of(self, host_id: int) -> int:
        """Row index of a catalog id, or ``-1``. ``host_id`` is sorted."""
        k = int(np.searchsorted(self.host_id, int(host_id)))
        if k >= self.host_id.size or int(self.host_id[k]) != int(host_id):
            return -1
        return k

    def tiles_of(self, row: int) -> Tuple[np.ndarray, np.ndarray]:
        """``(tile_ids, fractions)`` of the tiles this host actually touches."""
        f = np.asarray(self.tile_frac[int(row)], dtype=np.float64)
        t = np.flatnonzero(f > 0).astype(np.int64)
        order = np.argsort(-f[t], kind="stable")
        return t[order], f[t][order]

    def n_tiles_hit(self) -> np.ndarray:
        return np.count_nonzero(self.tile_frac > 0, axis=1).astype(np.int64)


def _subhalo_counts(cat: HaloCatalog, host_ids: Iterable[int]) -> np.ndarray:
    """Number of catalog descendants of each host (sub-subhalos included)."""
    kids = child_map(cat)
    return np.array(
        [len(descendants_of(cat, int(h), children=kids)) for h in host_ids],
        dtype=np.int64,
    )


# --------------------------------------------------------------------------
# The features
# --------------------------------------------------------------------------

@dataclass
class LagrangianHostFeatures:
    """LR-resolution host conditioning channels plus their host metadata.

    Every volume is shaped ``(ng_lr, ng_lr, ng_lr)`` (``dq_over_rl`` is
    ``(3, ...)``) and is indexed so that ``vol.reshape(-1)[pid]`` is the value
    at LR particle ``pid``.
    """

    grid: LagrangianGrid
    table: HostTable
    host_index: np.ndarray            # (ng,ng,ng) int32 row into table, -1 none
    host_member: np.ndarray           # (ng,ng,ng) float32 in {0,1}
    log_host_mass: np.ndarray         # (ng,ng,ng) float32
    dq_over_rl: np.ndarray            # (3,ng,ng,ng) float32
    host_fraction_per_tile: np.ndarray  # (ng,ng,ng) float32
    subhalo_budget: Optional[np.ndarray] = None  # (ng,ng,ng) float32 or None
    box: str = ""
    source: str = ""
    catalog_path: str = ""

    # -- access -----------------------------------------------------------
    def channel_names(self) -> List[str]:
        names = ["host_member", "log_host_mass",
                 "dq_over_rl_x", "dq_over_rl_y", "dq_over_rl_z",
                 "host_fraction_per_tile"]
        if self.subhalo_budget is not None:
            names.append("subhalo_budget")
        return names

    @property
    def n_channels(self) -> int:
        return len(self.channel_names())

    def stack_lr(self) -> np.ndarray:
        """``(C, ng_lr, ng_lr, ng_lr)`` -- the whole box at LR resolution.

        Cheap: ``C * 64**3`` floats is a few MB. The HR stack of the same thing
        would be 512x larger, which is why :meth:`tile_hr` exists.
        """
        vols = [self.host_member[None], self.log_host_mass[None],
                self.dq_over_rl, self.host_fraction_per_tile[None]]
        if self.subhalo_budget is not None:
            vols.append(self.subhalo_budget[None])
        return np.concatenate(vols, axis=0).astype(np.float32, copy=False)

    def tile_lr(self, tile_id: int) -> np.ndarray:
        """``(C, tile_lr, tile_lr, tile_lr)`` -- one tile, still at LR."""
        sx, sy, sz = self.grid.lr_slices(tile_id)
        return np.ascontiguousarray(self.stack_lr()[:, sx, sy, sz])

    def tile_hr(self, tile_id: int) -> np.ndarray:
        """``(C, tile_hr, tile_hr, tile_hr)`` -- one tile, broadcast to HR.

        Nearest-neighbour repeat by ``upsample`` on each axis: every HR child
        gets its parent LR site's value, which is the only broadcast consistent
        with the features being defined per LR site. Built on demand, so the
        dense HR volume for the whole box is never allocated.
        """
        crop = self.tile_lr(tile_id)
        f = self.grid.upsample
        return np.ascontiguousarray(
            crop.repeat(f, axis=1).repeat(f, axis=2).repeat(f, axis=3))

    def hosts_in_tile(self, tile_id: int) -> List[Dict]:
        """Which hosts own sites in this tile, and how big each one is.

        The per-tile read: a tile is ``tile_lr**3`` LR sites, and they are
        typically shared between several hosts, most of which continue outside
        the tile. Each row therefore reports both halves of that -- how much of
        the *tile* the host occupies (``n_sites_here``) and how much of the
        *host* is in the tile (``frac_of_host``) -- because a host filling the
        tile and a host clipping its corner look the same from the tile alone.

        Sorted by host mass, most massive first. Sites with no host are simply
        absent; ``n_sites_here`` over all rows is the tile's occupancy.
        """
        rows = self.host_index[self.grid.lr_slices(int(tile_id))].reshape(-1)
        present = rows[rows >= 0]
        if present.size == 0:
            return []
        uniq, counts = np.unique(present, return_counts=True)
        t = self.table
        order = np.argsort(-t.mvir[uniq])
        cell = self.grid.cell_mpc_h
        out = []
        for k in order:
            r, c = int(uniq[k]), int(counts[k])
            n_tot = int(t.n_particles[r])
            out.append({
                "row": r,
                "host_id": int(t.host_id[r]),          # metadata, not a feature
                "mvir": float(t.mvir[r]),
                "log_mvir": float(np.log10(t.mvir[r])),
                "n_sites_here": c,
                "n_particles": n_tot,
                "frac_of_host": float(c / n_tot) if n_tot else 0.0,
                "r_lag_mpc_h": float(t.r_lag_mpc_h[r]),
                "r_lag_cells": float(t.r_lag_mpc_h[r] / cell),
                "n_tiles_hit": int(np.count_nonzero(t.tile_frac[r] > 0)),
                "n_sub": int(t.n_sub[r]),
                "lam": float(t.n_sub[r] / n_tot) if n_tot else 0.0,
            })
        return out

    def host_sites(self, row: int) -> np.ndarray:
        """LR particle ids owned by table row ``row``."""
        return np.flatnonzero(self.host_index.reshape(-1) == int(row)).astype(np.int64)

    # -- io ---------------------------------------------------------------
    def to_npz(self, path: str | Path) -> str:
        g, t = self.grid, self.table
        payload = dict(
            ng_lr=np.int64(g.ng_lr), ng_hr=np.int64(g.ng_hr),
            tile_hr=np.int64(g.tile_hr),
            boxsize_mpc_h=np.float64(g.boxsize_mpc_h),
            host_index=self.host_index, host_member=self.host_member,
            log_host_mass=self.log_host_mass, dq_over_rl=self.dq_over_rl,
            host_fraction_per_tile=self.host_fraction_per_tile,
            t_host_id=t.host_id, t_mvir=t.mvir, t_rvir_kpc_h=t.rvir_kpc_h,
            t_num_p_catalog=t.num_p_catalog, t_n_particles=t.n_particles,
            t_center_lag=t.center_lag, t_r_lag=t.r_lag_mpc_h,
            t_rms_lag=t.rms_lag_mpc_h, t_tile_frac=t.tile_frac,
            t_n_sub=t.n_sub,
            meta=np.array([self.box, self.source, self.catalog_path,
                           t.n_sub_source]),
        )
        if self.subhalo_budget is not None:
            payload["subhalo_budget"] = self.subhalo_budget
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **payload)
        return str(path)

    @staticmethod
    def from_npz(path: str | Path) -> "LagrangianHostFeatures":
        z = np.load(str(path), allow_pickle=False)
        meta = [str(s) for s in z["meta"]]
        grid = LagrangianGrid(ng_lr=int(z["ng_lr"]), ng_hr=int(z["ng_hr"]),
                              tile_hr=int(z["tile_hr"]),
                              boxsize_mpc_h=float(z["boxsize_mpc_h"]))
        table = HostTable(
            host_id=z["t_host_id"], mvir=z["t_mvir"],
            rvir_kpc_h=z["t_rvir_kpc_h"], num_p_catalog=z["t_num_p_catalog"],
            n_particles=z["t_n_particles"], center_lag=z["t_center_lag"],
            r_lag_mpc_h=z["t_r_lag"], rms_lag_mpc_h=z["t_rms_lag"],
            tile_frac=z["t_tile_frac"], n_sub=z["t_n_sub"],
            n_sub_source=meta[3] if len(meta) > 3 else "",
        )
        return LagrangianHostFeatures(
            grid=grid, table=table, host_index=z["host_index"],
            host_member=z["host_member"], log_host_mass=z["log_host_mass"],
            dq_over_rl=z["dq_over_rl"],
            host_fraction_per_tile=z["host_fraction_per_tile"],
            subhalo_budget=(z["subhalo_budget"] if "subhalo_budget" in z.files
                            else None),
            box=meta[0], source=meta[1], catalog_path=meta[2],
        )


def build_host_features(
    cat: HaloCatalog,
    owner: np.ndarray,
    grid: LagrangianGrid,
    *,
    n_sub_per_host: Optional[Dict[int, int] | Sequence[int] | np.ndarray] = None,
    with_subhalo_budget: bool = True,
    box: str = "",
    source: str = "",
) -> LagrangianHostFeatures:
    """Turn an LR catalog + per-particle ownership into LR-lattice features.

    ``owner`` is ``owner[lr_particle_id] -> catalog id`` as written by
    :func:`cosmo_sr.eval.particle_identity.stream_owner_assignment`, i.e. *leaf*
    attribution. It is lifted to hosts here with ``remap_to_roots``, because the
    conditioning question is about the host: a satellite's particles are part of
    the region the host governs, and attributing them to the satellite would
    make a rich host look like it lost the mass its own substructure holds.

    ``n_sub_per_host`` sets the budget ``N_h``. Pass a ``{host_id: N}`` mapping
    or an array aligned with the built table's rows. The default counts the
    host's catalog descendants, which makes the budget self-consistent with the
    catalog it came from -- a placeholder, not a target: the interesting ``N_h``
    is an HR or predicted count, and it is a caller's input for that reason.
    """
    owner = np.asarray(owner)
    if owner.size != grid.n_lr:
        raise ValueError(
            f"owner has {owner.size} entries but the LR lattice has "
            f"{grid.n_lr} sites (ng_lr={grid.ng_lr}); the ownership array and "
            "the grid describe different boxes")

    host_owner = remap_to_roots(owner, cat)
    index = build_owner_index(host_owner)
    host_ids = index.halo_id.astype(np.int64)          # sorted
    n_hosts = int(host_ids.size)

    # Catalog rows for those hosts, in table order.
    cat_row = {int(h): i for i, h in enumerate(cat.ids)}
    rows = np.array([cat_row.get(int(h), -1) for h in host_ids], dtype=np.int64)
    if n_hosts and int(rows.min()) < 0:
        missing = [int(h) for h, r in zip(host_ids, rows) if r < 0]
        raise ValueError(
            f"{len(missing)} owning host id(s) absent from the catalog "
            f"(e.g. {missing[:5]}); the ownership array and the catalog are "
            "from different Rockstar runs")

    q = lagrangian_lattice_positions(grid)
    box_l = float(grid.boxsize_mpc_h)
    v_cell = grid.cell_mpc_h ** 3

    # Per-site outputs, flat and indexed by LR particle id.
    n = grid.n_lr
    site_row = np.full(n, -1, dtype=np.int32)
    member = np.zeros(n, dtype=np.float32)
    logm = np.zeros(n, dtype=np.float32)
    dq = np.zeros((3, n), dtype=np.float32)
    frac_site = np.zeros(n, dtype=np.float32)
    budget = np.zeros(n, dtype=np.float32) if with_subhalo_budget else None

    center = np.zeros((n_hosts, 3), dtype=np.float64)
    r_lag = np.zeros(n_hosts, dtype=np.float64)
    rms_lag = np.zeros(n_hosts, dtype=np.float64)
    n_part = np.zeros(n_hosts, dtype=np.int64)
    tile_frac = np.zeros((n_hosts, grid.n_tiles), dtype=np.float32)

    mvir = np.asarray(cat.mvir, dtype=np.float64)[rows] if n_hosts else np.zeros(0)
    site_tile = grid.tile_of_lr_site(np.arange(n, dtype=np.int64))

    for k in range(n_hosts):
        ids = index.members(int(host_ids[k]))
        if ids.size == 0:                      # build_owner_index cannot emit
            continue                           # an empty group; defensive only
        n_part[k] = ids.size
        c = periodic_circular_mean(q[ids], box_l)
        center[k] = c
        d = periodic_delta(q[ids], c[None, :], box_l)
        rms_lag[k] = float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))
        # Volume-equivalent Lagrangian radius: the sphere that holds this many
        # sites at the mean lattice density. Defined for every host (including
        # a 1-site one), monotone in particle count, and independent of the
        # host's shape -- unlike rms |dq|, which is reported separately above.
        r = float((3.0 * ids.size * v_cell / (4.0 * np.pi)) ** (1.0 / 3.0))
        r_lag[k] = r

        t = site_tile[ids]
        counts = np.bincount(t, minlength=grid.n_tiles).astype(np.float64)
        f = counts / float(ids.size)
        tile_frac[k] = f.astype(np.float32)

        site_row[ids] = k
        member[ids] = 1.0
        logm[ids] = np.float32(np.log10(mvir[k])) if mvir[k] > 0 else np.float32(0.0)
        dq[:, ids] = (d / r).T.astype(np.float32)
        frac_site[ids] = f[t].astype(np.float32)

    # Budget N_h.
    if n_sub_per_host is None:
        n_sub = _subhalo_counts(cat, host_ids)
        n_sub_source = "catalog descendants of each host"
    elif isinstance(n_sub_per_host, dict):
        n_sub = np.array([int(n_sub_per_host.get(int(h), 0)) for h in host_ids],
                         dtype=np.int64)
        n_sub_source = "caller mapping {host_id: N_h}"
    else:
        n_sub = np.asarray(n_sub_per_host, dtype=np.int64).reshape(-1)
        if n_sub.size != n_hosts:
            raise ValueError(
                f"n_sub_per_host has {n_sub.size} entries for {n_hosts} hosts; "
                "pass a {host_id: N_h} mapping if the order is not the table's")
        n_sub_source = "caller array aligned with the host table"

    if budget is not None:
        for k in range(n_hosts):
            if n_part[k] == 0 or n_sub[k] == 0:
                continue
            ids = index.members(int(host_ids[k]))
            budget[ids] = np.float32(float(n_sub[k]) / float(n_part[k]))

    ng = grid.ng_lr
    shape = (ng, ng, ng)
    table = HostTable(
        host_id=host_ids,
        mvir=mvir.astype(np.float64),
        rvir_kpc_h=np.asarray(cat.rvir, dtype=np.float64)[rows] if n_hosts else np.zeros(0),
        num_p_catalog=np.asarray(cat.num_p, dtype=np.int64)[rows] if n_hosts else np.zeros(0, np.int64),
        n_particles=n_part, center_lag=center, r_lag_mpc_h=r_lag,
        rms_lag_mpc_h=rms_lag, tile_frac=tile_frac, n_sub=n_sub,
        n_sub_source=n_sub_source,
    )
    return LagrangianHostFeatures(
        grid=grid, table=table,
        host_index=site_row.reshape(shape),
        host_member=member.reshape(shape),
        log_host_mass=logm.reshape(shape),
        dq_over_rl=dq.reshape((3,) + shape),
        host_fraction_per_tile=frac_site.reshape(shape),
        subhalo_budget=None if budget is None else budget.reshape(shape),
        box=box, source=source, catalog_path=str(cat.path),
    )


def normalization_report(feat: LagrangianHostFeatures) -> Dict:
    """The identities this construction has to satisfy, measured not asserted.

    ``sum_t f[h,t] == 1`` and ``sum_{i in h} lambda_i == N_h`` are what make the
    tile fractions and the budget usable as "this tile owes this share"; if
    either drifts, a downstream consumer would be dividing a host between tiles
    that do not add up. Reported as numbers so a build log carries the evidence.
    """
    t = feat.table
    out: Dict[str, object] = {
        "n_hosts": t.n_hosts,
        "n_lr_sites": int(feat.grid.n_lr),
        "n_sites_with_host": int(np.count_nonzero(feat.host_member > 0)),
        "frac_sites_with_host": float(np.mean(feat.host_member > 0)),
        "n_tiles": int(feat.grid.n_tiles),
        "channels": feat.channel_names(),
    }
    if t.n_hosts:
        row_sums = t.tile_frac.astype(np.float64).sum(axis=1)
        out["max_abs_tile_frac_error"] = float(np.max(np.abs(row_sums - 1.0)))
        out["mass_range_log10"] = [float(np.log10(t.mvir.min())),
                                   float(np.log10(t.mvir.max()))]
        out["median_tiles_per_host"] = float(np.median(t.n_tiles_hit()))
        out["max_tiles_per_host"] = int(np.max(t.n_tiles_hit()))
        out["n_hosts_spanning_tiles"] = int(np.count_nonzero(t.n_tiles_hit() > 1))
        out["n_sub_source"] = t.n_sub_source
    else:
        out["max_abs_tile_frac_error"] = 0.0

    if feat.subhalo_budget is not None and t.n_hosts:
        flat_row = feat.host_index.reshape(-1)
        flat_lam = feat.subhalo_budget.reshape(-1).astype(np.float64)
        got = np.bincount(flat_row[flat_row >= 0], weights=flat_lam[flat_row >= 0],
                          minlength=t.n_hosts)
        want = t.n_sub.astype(np.float64)
        out["max_abs_budget_error"] = float(np.max(np.abs(got - want)))
        out["total_budget"] = float(got.sum())
    else:
        out["max_abs_budget_error"] = 0.0

    out["ok"] = bool(float(out["max_abs_tile_frac_error"]) < 1e-6
                     and float(out["max_abs_budget_error"]) < 1e-3)
    return out
