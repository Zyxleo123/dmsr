"""Lagrangian chunks, the Eulerian purity grid, and the catalog core mask.

Why this module exists
----------------------
The reward is a *catalog* statistic, so it has to be attributed to the piece of
the residual field that produced it. Two facts make that non-trivial:

1. Halo finding is only trustworthy on a **complete periodic box** -- the frozen
   pipeline never runs Rockstar on an isolated crop, and it should not start
   now. Displacements are large (median ``|Psi|`` ~ 36 HR cells against a 512
   cell box), so a Lagrangian crop does not fill any Eulerian cube it is
   compared against; ``cic_density_valid_center`` documents the same failure for
   density (``r = 0.08`` for the naive wrapped crop).
2. Credit assignment needs a *local* unit, because the reward has to say which
   chunk of residual helped.

The resolution: **generate and halo-find on the full 512^3 periodic box, and
attribute catalog objects back to Lagrangian chunks afterwards.** Nothing about
the catalog geometry changes relative to the frozen evaluation pipeline; the
only new object is the attribution map, and objects that cannot be attributed
cleanly are dropped from every summary rather than being assigned by fiat.

Chunk grid
----------
The box is cut into ``(ng_hr / chunk_hr)^3`` disjoint **Lagrangian** cubes. The
default ``chunk_hr = 128`` (25 Mpc/h) is set by the largest halo whose Lagrangian
patch must fit inside one chunk: a ``1e14 Msun/h`` host collapses from a sphere
of diameter ``2 (3M / 4 pi rho_bar)^(1/3) = 13.5 Mpc/h``, comfortably inside 25
Mpc/h, while ``1e15`` (29 Mpc/h) does not and is expected to be masked out.
That gives 64 chunks per box, so an ensemble of ``B = 16`` chunks is a quarter
of a box.

The purity grid and the core mask
---------------------------------
:func:`chunk_purity_grid` deposits every particle (nearest grid point, at
``q + Psi``) onto a coarse Eulerian mesh and records, per Eulerian cell, which
Lagrangian chunk contributed the largest share of its mass and how large that
share is. :func:`assign_halos_to_chunks` then keeps a halo only if **every**
Eulerian cell its ``Rvir`` sphere touches is dominated by the *same* chunk with
share at least ``min_purity``.

The precise core mask, as used everywhere downstream:

    A halo ``h`` belongs to chunk ``c`` iff, for every Eulerian cell ``e`` with
    ``|centre(e) - pos(h)|_inf <= radius_mult * Rvir(h)`` (periodic),
        ``majority_id[e] == c``  and  ``majority_frac[e] >= min_purity``.
    Otherwise ``h`` is boundary-contaminated and enters no summary.

    A **subhalo** additionally requires its host to be assigned to the same
    chunk, so an occupation numerator is never counted against a denominator
    that lives in a different chunk.

The effective volume of chunk ``c`` is the volume of the Eulerian cells that
pass the same test, **not** the nominal chunk volume -- otherwise the masked
region would silently bias every number density low.

This is an approximation in exactly one place: Rockstar's membership is the set
of *bound* particles, and we test the ``Rvir`` sphere instead. The sphere is a
superset of the bound set, so the test is conservative (it rejects more halos
than strictly necessary), which is the safe direction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ChunkGrid",
    "PurityGrid",
    "assign_halos_to_chunks",
    "chunk_purity_grid",
    "lagrangian_patch_diameter_mpc",
]


def lagrangian_patch_diameter_mpc(
    mass_msun_h: float,
    omega_m: float = 0.2814,
    rho_crit: float = 2.77536627e11,
) -> float:
    """Diameter (Mpc/h) of the uniform sphere that collapses into ``mass_msun_h``.

    Used to justify ``chunk_hr``: a chunk must be wider than this for the halo
    masses we intend to score, or the halo's own particles straddle chunks and
    the core mask throws it away.
    """
    rho_bar = float(omega_m) * float(rho_crit)
    return 2.0 * (3.0 * float(mass_msun_h) / (4.0 * np.pi * rho_bar)) ** (1.0 / 3.0)


@dataclass(frozen=True)
class ChunkGrid:
    """Disjoint Lagrangian cubes tiling one periodic box.

    ``chunk_hr`` is in HR cells; ``ng_hr`` must be an integer multiple of it.
    Chunk ids are the C-order flat index of ``(ix, iy, iz)``.
    """

    ng_hr: int = 512
    chunk_hr: int = 128
    boxsize_mpc_h: float = 100.0

    def __post_init__(self) -> None:
        if self.ng_hr % self.chunk_hr:
            raise ValueError(
                f"chunk_hr={self.chunk_hr} does not divide ng_hr={self.ng_hr}"
            )

    @property
    def n_per_axis(self) -> int:
        return self.ng_hr // self.chunk_hr

    @property
    def n_chunks(self) -> int:
        return self.n_per_axis ** 3

    @property
    def chunk_mpc_h(self) -> float:
        return self.boxsize_mpc_h * self.chunk_hr / self.ng_hr

    @property
    def chunk_volume_mpc3(self) -> float:
        return self.chunk_mpc_h ** 3

    def coord(self, chunk_id: int) -> Tuple[int, int, int]:
        n = self.n_per_axis
        c = int(chunk_id)
        if not 0 <= c < self.n_chunks:
            raise IndexError(f"chunk {chunk_id} outside 0..{self.n_chunks - 1}")
        return (c // (n * n), (c // n) % n, c % n)

    def index(self, ix: int, iy: int, iz: int) -> int:
        n = self.n_per_axis
        return int((ix % n) * n * n + (iy % n) * n + (iz % n))

    def slices(self, chunk_id: int) -> Tuple[slice, slice, slice]:
        """Lagrangian index slices of one chunk (never wraps: chunks tile exactly)."""
        s = self.chunk_hr
        ix, iy, iz = self.coord(chunk_id)
        return (
            slice(ix * s, (ix + 1) * s),
            slice(iy * s, (iy + 1) * s),
            slice(iz * s, (iz + 1) * s),
        )

    def origin(self, chunk_id: int) -> Tuple[int, int, int]:
        ix, iy, iz = self.coord(chunk_id)
        s = self.chunk_hr
        return (ix * s, iy * s, iz * s)

    def all_ids(self) -> List[int]:
        return list(range(self.n_chunks))

    def iter_slices(self) -> Iterator[Tuple[int, Tuple[slice, slice, slice]]]:
        for c in range(self.n_chunks):
            yield c, self.slices(c)


@dataclass
class PurityGrid:
    """Per-Eulerian-cell dominant Lagrangian chunk and its mass share.

    ``majority_id`` is ``-1`` where the cell received no particles at all (only
    possible in deep voids on a fine mesh).
    """

    majority_id: np.ndarray      # (E, E, E) int16
    majority_frac: np.ndarray    # (E, E, E) float32
    total: np.ndarray            # (E, E, E) float32, particle counts
    grid: int
    boxsize_mpc_h: float
    chunk_hr: int
    ng_hr: int

    @property
    def cell_mpc_h(self) -> float:
        return self.boxsize_mpc_h / self.grid

    def core_cells(self, min_purity: float) -> np.ndarray:
        """Boolean mask of cells that pass the purity test."""
        return (self.majority_id >= 0) & (self.majority_frac >= float(min_purity))

    def effective_volume_mpc3(self, min_purity: float) -> np.ndarray:
        """``(n_chunks,)`` Eulerian volume that each chunk owns cleanly."""
        n_chunks = (self.ng_hr // self.chunk_hr) ** 3
        ok = self.core_cells(min_purity)
        ids = self.majority_id[ok].astype(np.int64, copy=False)
        counts = np.bincount(ids, minlength=n_chunks).astype(np.float64)
        return counts * (self.cell_mpc_h ** 3)

    def to_npz(self, path) -> str:
        np.savez_compressed(
            str(path),
            majority_id=self.majority_id,
            majority_frac=self.majority_frac,
            total=self.total,
            meta=np.array(
                [self.grid, self.boxsize_mpc_h, self.chunk_hr, self.ng_hr],
                dtype=np.float64,
            ),
        )
        return str(path)

    @staticmethod
    def from_npz(path) -> "PurityGrid":
        z = np.load(str(path))
        grid, box, chunk_hr, ng_hr = z["meta"]
        return PurityGrid(
            majority_id=z["majority_id"],
            majority_frac=z["majority_frac"],
            total=z["total"],
            grid=int(grid),
            boxsize_mpc_h=float(box),
            chunk_hr=int(chunk_hr),
            ng_hr=int(ng_hr),
        )


def chunk_purity_grid(
    disp_norm: np.ndarray,
    *,
    chunk_grid: ChunkGrid,
    grid: int = 128,
    dis_norm_kpc_h: float = 6000.0,
    redshift: float = 0.0,
) -> PurityGrid:
    """Deposit each Lagrangian chunk's particles and record the dominant chunk.

    ``disp_norm`` is ``(3, ng, ng, ng)`` *normalized* displacement (catnorm), the
    same convention as :func:`cosmo_sr.eval.density.cic_density`. Deposition is
    nearest-grid-point, not CIC: the question is which Lagrangian region the mass
    at an Eulerian point came from, and CIC would smear that across cells.

    Memory is bounded by one chunk at a time plus three ``grid^3`` arrays.
    """
    from ..data.preprocess_srs import growth_D

    x = np.asarray(disp_norm)
    if x.ndim != 4 or x.shape[0] < 3:
        raise ValueError(f"expected (3, ng, ng, ng) displacement, got {x.shape}")
    ng = int(x.shape[1])
    if ng != chunk_grid.ng_hr:
        raise ValueError(f"field ng={ng} != chunk_grid.ng_hr={chunk_grid.ng_hr}")
    e = int(grid)
    ncell = e ** 3

    # normalized -> HR cells -> Eulerian mesh cells
    dis_norm = float(dis_norm_kpc_h) * float(growth_D(redshift))
    cell_kpc = chunk_grid.boxsize_mpc_h * 1000.0 / ng
    to_mesh = (dis_norm / cell_kpc) * (e / ng)      # normalized disp -> mesh cells
    q_to_mesh = e / ng                              # Lagrangian HR index -> mesh cells

    total = np.zeros(ncell, dtype=np.float32)
    best = np.zeros(ncell, dtype=np.float32)
    best_id = np.full(ncell, -1, dtype=np.int16)

    lat = np.arange(ng, dtype=np.float32) + 0.5

    for cid, (sx, sy, sz) in chunk_grid.iter_slices():
        d = x[0:3, sx, sy, sz].astype(np.float32, copy=False)
        s = d.shape[1]
        qx = (lat[sx]).reshape(s, 1, 1)
        qy = (lat[sy]).reshape(1, s, 1)
        qz = (lat[sz]).reshape(1, 1, s)
        gx = np.floor(d[0] * to_mesh + qx * q_to_mesh).astype(np.int64) % e
        gy = np.floor(d[1] * to_mesh + qy * q_to_mesh).astype(np.int64) % e
        gz = np.floor(d[2] * to_mesh + qz * q_to_mesh).astype(np.int64) % e
        flat = ((gx * e + gy) * e + gz).ravel()
        cnt = np.bincount(flat, minlength=ncell).astype(np.float32)
        total += cnt
        better = cnt > best
        best = np.where(better, cnt, best)
        best_id = np.where(better, np.int16(cid), best_id)
        del d, gx, gy, gz, flat, cnt

    frac = np.divide(best, total, out=np.zeros_like(best), where=total > 0)
    best_id = np.where(total > 0, best_id, np.int16(-1))
    shape = (e, e, e)
    return PurityGrid(
        majority_id=best_id.reshape(shape),
        majority_frac=frac.reshape(shape),
        total=total.reshape(shape),
        grid=e,
        boxsize_mpc_h=float(chunk_grid.boxsize_mpc_h),
        chunk_hr=int(chunk_grid.chunk_hr),
        ng_hr=ng,
    )


def assign_halos_to_chunks(
    pos_mpc_h: np.ndarray,
    rvir_kpc_h: np.ndarray,
    purity: PurityGrid,
    *,
    min_purity: float = 0.8,
    radius_mult: float = 1.0,
    max_half_width: int = 4,
) -> np.ndarray:
    """``(N,)`` chunk id per halo, ``-1`` when boundary-contaminated.

    A halo passes only if every Eulerian cell within ``radius_mult * Rvir``
    (Chebyshev, periodic) is dominated by the same chunk at share
    ``>= min_purity``. Halos needing a neighbourhood wider than
    ``max_half_width`` cells are rejected outright: those are the cluster-scale
    objects whose Lagrangian patch exceeds one chunk, and pretending otherwise
    would silently mix chunks.
    """
    pos = np.asarray(pos_mpc_h, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos_mpc_h must be (N, 3), got {pos.shape}")
    n = pos.shape[0]
    out = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return out

    e = purity.grid
    cell = purity.cell_mpc_h
    mid = purity.majority_id
    mfr = purity.majority_frac

    base = np.floor((pos % purity.boxsize_mpc_h) / cell).astype(np.int64) % e
    rvir_mpc = np.asarray(rvir_kpc_h, dtype=np.float64) * 1e-3
    hw = np.ceil(np.maximum(rvir_mpc, 0.0) * float(radius_mult) / cell).astype(np.int64)

    too_wide = hw > int(max_half_width)
    for w in range(0, int(max_half_width) + 1):
        sel = np.nonzero((hw == w) & ~too_wide)[0]
        if sel.size == 0:
            continue
        bx, by, bz = base[sel, 0], base[sel, 1], base[sel, 2]
        cid = mid[bx, by, bz].astype(np.int32)
        ok = (cid >= 0) & (mfr[bx, by, bz] >= float(min_purity))
        for ox in range(-w, w + 1):
            for oy in range(-w, w + 1):
                for oz in range(-w, w + 1):
                    if ox == 0 and oy == 0 and oz == 0:
                        continue
                    ix = (bx + ox) % e
                    iy = (by + oy) % e
                    iz = (bz + oz) % e
                    ok &= (mid[ix, iy, iz].astype(np.int32) == cid)
                    ok &= (mfr[ix, iy, iz] >= float(min_purity))
        out[sel] = np.where(ok, cid, -1)
    return out
