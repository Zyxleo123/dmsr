"""Fine-grained (64^3) credit: decompose a full-box catalog over generator tiles.

Why this module exists
----------------------
:mod:`cosmo_sr.reward.geometry` attributes a halo to a Lagrangian chunk by a
*purity test on an Eulerian mesh*, and drops the halo when the test fails. That
mask was measured to delete hosts in proportion to their mass (see
``docs/reward_residual_diffusion.md`` §2): retention 0.37, 0.30, 0.22, 0.12,
0.00 from ``1e12`` to ``1e14 Msun/h``. The fix adopted there was to widen the
chunk to 256 HR cells, which bought host counts at the cost of credit
resolution -- 8 credit units per box instead of 64.

This module removes the trade-off instead of re-balancing it, because the
information the purity test was *approximating* is available exactly:

* our GADGET2 dumps set the particle ID to the **flat Lagrangian grid index**
  (:mod:`cosmo_sr.eval.particles`, ``ids = arange(Ng**3)``), and Rockstar
  preserves 4-byte IDs verbatim (``io/io_gadget.c::gadget2_rescale_particles``);
* with ``FULL_PARTICLE_CHUNKS = 1`` Rockstar writes the member particle IDs of
  every printed halo.

So the Lagrangian origin of every bound particle is *known*, not inferred, and
an object can be split **fractionally** over the 64^3 generator tiles that
produced its particles:

    w[o, j] = |{ p in members(o) : tile(p) == j }| / |members(o)|,
    sum_j w[o, j] == 1   for every object, exactly.

Nothing is rejected, so nothing is mass-selected, and the effective volume of a
tile is its nominal volume -- there is no masked region to bias number densities
low.

Sufficient statistics
---------------------
Per tile ``j`` and host-mass bin ``b`` (all float, because weights are
fractional):

    H[j, b] = sum of w[h, j] over hosts h in bin b            (denominator)
    S[j, b] = sum of w[s, j] over subhalos s whose HOST is in bin b (numerator)
    N[j, m] = sum of w[s, j] over subhalos s in subhalo-mass bin m (abundance)

and the full-box statistics are recovered by summation:

    H[b] = sum_j H[j, b],   S[b] = sum_j S[j, b],   O[b] = S[b] / H[b].

Reading Rockstar's ``.particles`` table
---------------------------------------
The columns are::

    x y z vx vy vz particle_id assigned_internal_haloid internal_haloid external_haloid

``io/meta_io.c::print_child_particles`` walks each halo *and recurses into its
substructure*, so the file is emitted once per top-level recursion root:

* ``internal_haloid`` (column 8, 0-based) is the **root of the recursion** -- the
  object whose particle list is being printed;
* ``assigned_internal_haloid`` (column 7) is the object the particle is really
  bound to, which differs from the root exactly for particles inside
  substructure;
* ``external_haloid`` (column 9) is the printed catalog ``id`` of the root, or
  ``-1`` for a halo that ``_should_print`` rejected.

The loop ``for i in halos: print_child_particles(i, i, halos[i].id)`` visits
**every** halo as a root, so a particle inside a subhalo appears twice: once
under the subhalo and once under its host. That is not an error to be filtered
out -- it is precisely what we want, because hosts and subhalos are *different
objects* contributing to *different* sufficient statistics, and each of them
must receive total weight one. Grouping by ``external_haloid`` therefore gives
every catalog object its complete bound-particle set, exactly once.

The failure mode this docstring exists to prevent is grouping by
``assigned_internal_haloid`` instead: that silently removes substructure
particles from their host's set, so a rich host's weights come from its smooth
component alone. ``tests/reward/test_tiles.py`` pins the distinction.

What the grouped count is *not*
-------------------------------
Measured against the real binary on a clustered test box (138 objects, 40 with
substructure), the number of rows grouped under an object is **not** its
catalog ``num_p``, and an equality check against ``num_p`` would fail on every
rich host -- exactly the hosts this project is about. Two separate reasons:

* ``num_p`` is a halo's *own* particle list; the recursion adds its
  descendants'. So ``rows(o) = num_p(o) + sum over descendants``.
* The recursion walks the substructure tree, which contains halos Rockstar
  **rejected from the catalog** (below ``MIN_HALO_OUTPUT_SIZE``, or unbound).
  In the test box a 7914-particle host absorbed 51 further particles from four
  12-14 particle clumps that appear nowhere in the ASCII file.

Both are correct for our purpose -- the host's Lagrangian footprint should
include all of its bound material, resolved or not -- so the checkable
invariants are the ones :func:`check_member_consistency` enforces:

* every catalog object has at least one member row;
* no particle id repeats *within* an object (measured: zero repeats);
* ``rows(o) >= num_p(o)``, with equality exactly for leaves (measured: 98 of
  138, i.e. all and only the objects with no substructure).

Fallback without member IDs
---------------------------
:func:`center_tile_attribution` assigns each object entirely to the tile owning
the Eulerian cell at its centre, using the existing purity grid, and flags an
object as ``boundary`` when that cell's majority share is below ``min_purity``.
Weights still sum to one per object (they are one-hot), so every identity in
this module continues to hold; only the *sharpness* of the credit degrades.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..eval.rockstar import HaloCatalog
from .catalog import CatalogBins, EnsembleSummary

__all__ = [
    "MemberWeights",
    "TileGrid",
    "TileSummary",
    "center_tile_attribution",
    "check_member_consistency",
    "direct_full_box_stats",
    "leave_one_out_credit",
    "member_weights_from_particles",
    "pool_tiles",
    "read_tile_summaries",
    "splice_tiles",
    "stream_member_ids",
    "stream_particle_tile_counts",
    "swap_prediction",
    "tile_of_particle_id",
    "tile_summaries",
    "write_tile_summaries",
]

# Columns of the Rockstar `.particles` particle table (0-based).
COL_PARTICLE_ID = 6
COL_ASSIGNED_INTERNAL = 7
COL_INTERNAL = 8
COL_EXTERNAL = 9


@dataclass(frozen=True)
class TileGrid:
    """The generator's output tiles, as a disjoint Lagrangian cover of the box.

    ``tile_hr = 64`` is not a free parameter: it is the size of one SR2 output
    tile (``nsplit = 8`` over ``ng_hr = 512``), so a tile is exactly the unit
    the generator produces and therefore the finest unit credit can be assigned
    to without splitting a single forward pass.
    """

    ng_hr: int = 512
    tile_hr: int = 64
    boxsize_mpc_h: float = 100.0

    def __post_init__(self) -> None:
        if self.tile_hr <= 0 or self.ng_hr % self.tile_hr:
            raise ValueError(
                f"tile_hr={self.tile_hr} does not divide ng_hr={self.ng_hr}"
            )

    @property
    def n_per_axis(self) -> int:
        return self.ng_hr // self.tile_hr

    @property
    def n_tiles(self) -> int:
        return self.n_per_axis ** 3

    @property
    def tile_volume_mpc3(self) -> float:
        """Nominal volume, which is also the *effective* volume.

        Fractional attribution rejects nothing, so unlike the purity-masked
        chunk geometry there is no masked-out region and no volume correction.
        The tile volumes sum to the box volume exactly.
        """
        return (self.boxsize_mpc_h * self.tile_hr / self.ng_hr) ** 3

    def coord(self, tile_id: int) -> Tuple[int, int, int]:
        n = self.n_per_axis
        t = int(tile_id)
        if not 0 <= t < self.n_tiles:
            raise IndexError(f"tile {tile_id} outside 0..{self.n_tiles - 1}")
        return (t // (n * n), (t // n) % n, t % n)

    def index(self, ix: int, iy: int, iz: int) -> int:
        n = self.n_per_axis
        return int((ix % n) * n * n + (iy % n) * n + (iz % n))

    def slices(self, tile_id: int) -> Tuple[slice, slice, slice]:
        s = self.tile_hr
        ix, iy, iz = self.coord(tile_id)
        return (slice(ix * s, (ix + 1) * s),
                slice(iy * s, (iy + 1) * s),
                slice(iz * s, (iz + 1) * s))

    def all_ids(self) -> List[int]:
        return list(range(self.n_tiles))


def tile_of_particle_id(
    particle_ids: np.ndarray, grid: TileGrid
) -> np.ndarray:
    """Lagrangian tile id of each particle, from its GADGET particle ID.

    The ID is the C-order flat index of the Lagrangian lattice,
    ``id = (ix * Ng + iy) * Ng + iz`` (:func:`cosmo_sr.eval.particles.field_to_particles`
    builds it as ``arange(Ng**3)`` against an array of shape ``(Ng, Ng, Ng)``),
    so this is pure integer arithmetic -- no positions, no matching, no
    tolerance.
    """
    ids = np.asarray(particle_ids, dtype=np.int64)
    ng = int(grid.ng_hr)
    n_part = ng ** 3
    if ids.size and (ids.min() < 0 or ids.max() >= n_part):
        raise ValueError(
            f"particle id outside 0..{n_part - 1} "
            f"(saw {int(ids.min())}..{int(ids.max())}); "
            "the dump was not written by cosmo_sr.eval.particles"
        )
    s = int(grid.tile_hr)
    n = grid.n_per_axis
    iz = ids % ng
    iy = (ids // ng) % ng
    ix = ids // (ng * ng)
    return (ix // s) * n * n + (iy // s) * n + (iz // s)


# --------------------------------------------------------------------------
# Streaming the `.particles` table
# --------------------------------------------------------------------------


@dataclass
class MemberWeights:
    """Sparse per-object tile weights, in COO form.

    ``halo_id[k]``, ``tile_id[k]``, ``weight[k]`` -- one row per (object, tile)
    pair that actually holds particles. ``sum(weight)`` over the rows of one
    object is 1 by construction.
    """

    halo_id: np.ndarray      # (K,) int64, the Rockstar catalog id
    tile_id: np.ndarray      # (K,) int64
    weight: np.ndarray       # (K,) float64
    n_members: Dict[int, int] = dc_field(default_factory=dict)
    meta: Dict = dc_field(default_factory=dict)

    def rows_of(self, halo_id: int) -> Tuple[np.ndarray, np.ndarray]:
        m = self.halo_id == int(halo_id)
        return self.tile_id[m], self.weight[m]

    def weight_sums(self) -> Dict[int, float]:
        """Total weight per object; every entry must be 1."""
        out: Dict[int, float] = {}
        for h, w in zip(self.halo_id, self.weight):
            out[int(h)] = out.get(int(h), 0.0) + float(w)
        return out

    def max_weight_error(self) -> float:
        """``max |sum_j w[o,j] - 1|`` over objects. The partition-of-unity test."""
        s = self.weight_sums()
        if not s:
            return 0.0
        return float(max(abs(v - 1.0) for v in s.values()))

    def to_npz(self, path) -> str:
        ids = np.asarray(sorted(self.n_members), dtype=np.int64)
        cnt = np.asarray([self.n_members[int(i)] for i in ids], dtype=np.int64)
        np.savez_compressed(
            str(path),
            halo_id=self.halo_id, tile_id=self.tile_id, weight=self.weight,
            member_halo_id=ids, member_count=cnt,
            meta=np.array(json.dumps(self.meta)),
        )
        return str(path)

    @staticmethod
    def from_npz(path) -> "MemberWeights":
        z = np.load(str(path), allow_pickle=False)
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        n_members = {
            int(i): int(c)
            for i, c in zip(z["member_halo_id"], z["member_count"])
        }
        return MemberWeights(
            halo_id=z["halo_id"].astype(np.int64),
            tile_id=z["tile_id"].astype(np.int64),
            weight=z["weight"].astype(np.float64),
            n_members=n_members,
            meta=meta,
        )


#: Fraction of rows a ``.particles`` table may lose to unparseable content
#: before it is treated as a broken file rather than a slightly ragged one.
#: Not zero, because a torn final line costs one row out of ~90 million and
#: failing the whole candidate for it would be its own kind of wrong; small,
#: because anything above a handful of rows means the table and the catalog no
#: longer describe the same run and every tile statistic built from it is
#: unattributable.
MAX_UNPARSEABLE_ROW_FRACTION = 1e-6


def _stream_particle_columns(path: Path, chunk_rows: int):
    """Yield ``(particle_id, external_haloid)`` int64 blocks, skipping bad rows.

    Reads the two columns as **float64** and converts, rather than asking the C
    parser for int64 directly. Measured on this cluster: two of twenty-eight
    otherwise-identical Rockstar runs wrote a ``.particles`` table that made
    pandas raise ``cannot safely convert passed user dtype of int64 for float64
    dtyped data in column 6`` part-way through a 10 GB file, killing the job.
    A strict int64 request turns any single malformed row into a crash after
    fifteen minutes of halo finding.

    Rows whose ids are not integral, or outside the lattice, are dropped and
    **counted**. The count is returned so the caller can refuse the candidate:
    silently dropping member particles would shrink an object's weight without
    shrinking its ``num_p``, which is precisely the corruption
    :func:`check_member_consistency` exists to catch, and it should be caught
    here with a readable number rather than there as a mystery.
    """
    import pandas as pd

    reader = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[COL_PARTICLE_ID, COL_EXTERNAL],
        names=["particle_id", "external_haloid"],
        dtype=np.float64,
        engine="c",
        chunksize=int(chunk_rows),
        on_bad_lines="skip",
    )
    n_total = n_bad = 0
    examples: List[str] = []
    for block in reader:
        pid_f = block["particle_id"].to_numpy(dtype=np.float64, copy=False)
        eid_f = block["external_haloid"].to_numpy(dtype=np.float64, copy=False)
        n_total += pid_f.size
        good = (np.isfinite(pid_f) & np.isfinite(eid_f)
                & (pid_f == np.rint(pid_f)) & (eid_f == np.rint(eid_f)))
        if not good.all():
            bad = np.nonzero(~good)[0]
            n_bad += bad.size
            for i in bad[:3]:
                if len(examples) < 5:
                    examples.append(f"particle_id={pid_f[i]!r} "
                                    f"external_haloid={eid_f[i]!r}")
        yield (pid_f[good].astype(np.int64), eid_f[good].astype(np.int64))
        del block, pid_f, eid_f

    if n_bad:
        frac = n_bad / max(n_total, 1)
        msg = (f"{n_bad} of {n_total} rows ({frac:.2e}) in {path.name} could not "
               f"be read as integers; examples: {examples}")
        if frac > MAX_UNPARSEABLE_ROW_FRACTION:
            raise ValueError(
                msg + ". That is too many to be a torn line: the member table "
                "and the catalog do not describe the same run, so no tile "
                "decomposition built from it is trustworthy.")
        print(f"    WARNING: {msg} (below the "
              f"{MAX_UNPARSEABLE_ROW_FRACTION:.0e} tolerance; dropped)", flush=True)


def stream_particle_tile_counts(
    particles_path: str | Path,
    grid: TileGrid,
    *,
    chunk_rows: int = 8_000_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(halo_id, tile_id, count)`` from a Rockstar ``.particles`` file.

    Streams the table in row chunks so a ~7 GB / 8e7-line ASCII file never has
    to be resident. Grouping is by ``external_haloid`` (the recursion root's
    catalog id), which gives every catalog object its *complete* bound-particle
    set exactly once -- see the module docstring for why grouping by
    ``assigned_internal_haloid`` instead would silently strip substructure out
    of its host.

    Rows with ``external_haloid < 0`` belong to halos Rockstar chose not to
    print (``_should_print``: below ``MIN_HALO_OUTPUT_SIZE``, or unbound), so
    they are not in the catalog either and are dropped here.
    """
    path = Path(particles_path)
    if not path.is_file():
        raise FileNotFoundError(f"no member-particle table at {path}")

    n_tiles = grid.n_tiles
    keys: List[np.ndarray] = []
    counts: List[np.ndarray] = []
    for pid_all, eid_all in _stream_particle_columns(path, int(chunk_rows)):
        keep = eid_all >= 0
        if not keep.any():
            continue
        eid = eid_all[keep]
        pid = pid_all[keep]
        tile = tile_of_particle_id(pid, grid)
        k, c = np.unique(eid * n_tiles + tile, return_counts=True)
        keys.append(k)
        counts.append(c)
        del eid, pid, tile

    if not keys:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    all_keys = np.concatenate(keys)
    all_counts = np.concatenate(counts)
    # Chunk boundaries split an object's rows, so merge duplicate keys.
    uk, inv = np.unique(all_keys, return_inverse=True)
    merged = np.zeros(uk.shape[0], dtype=np.int64)
    np.add.at(merged, inv, all_counts)
    return uk // n_tiles, uk % n_tiles, merged


def stream_member_ids(
    particles_path: str | Path,
    halo_ids: Sequence[int],
    *,
    chunk_rows: int = 8_000_000,
) -> Dict[int, np.ndarray]:
    """Member particle ids of a handful of named objects, in one streaming pass.

    Experiment 1 builds its Lagrangian masks from the member ids of the target
    subhalo and of its host, and those are the only objects it needs. Extracting
    them in the *same* pass that builds the tile weights is what lets the ~7 GB
    ASCII table be deleted inside the job that produced it, instead of being
    kept around for a second reader.

    Grouping follows :func:`stream_particle_tile_counts`: rows are selected by
    ``external_haloid``, so a host's returned set contains its substructure's
    particles too -- which is correct, because the host's Lagrangian footprint
    is what the equal-count random control is drawn from.
    """
    wanted = np.asarray(sorted({int(h) for h in halo_ids}), dtype=np.int64)
    out: Dict[int, List[np.ndarray]] = {int(h): [] for h in wanted}
    if wanted.size == 0:
        return {}
    for pid_all, eid in _stream_particle_columns(Path(particles_path),
                                                 int(chunk_rows)):
        sel = np.isin(eid, wanted)
        if not sel.any():
            continue
        pid = pid_all[sel]
        for h in np.unique(eid[sel]):
            out[int(h)].append(pid[eid[sel] == h])
        del pid, sel
    return {h: (np.concatenate(v) if v else np.zeros(0, dtype=np.int64))
            for h, v in out.items()}


def member_weights_from_particles(
    particles_path: str | Path,
    grid: TileGrid,
    *,
    chunk_rows: int = 8_000_000,
) -> MemberWeights:
    """Fractional tile weights for every printed object, from member IDs."""
    halo, tile, count = stream_particle_tile_counts(
        particles_path, grid, chunk_rows=chunk_rows
    )
    return _normalize_counts(halo, tile, count, source="member_particle_ids")


def _normalize_counts(
    halo: np.ndarray, tile: np.ndarray, count: np.ndarray, *, source: str
) -> MemberWeights:
    halo = np.asarray(halo, dtype=np.int64)
    tile = np.asarray(tile, dtype=np.int64)
    count = np.asarray(count, dtype=np.int64)
    if halo.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return MemberWeights(z, z, np.zeros(0), {}, {"source": source})
    order = np.lexsort((tile, halo))
    halo, tile, count = halo[order], tile[order], count[order]
    uniq, start = np.unique(halo, return_index=True)
    totals = np.add.reduceat(count, start).astype(np.float64)
    if np.any(totals <= 0):
        raise ValueError("an object was streamed with zero member particles")
    per_row_total = np.repeat(totals, np.diff(np.append(start, halo.size)))
    return MemberWeights(
        halo_id=halo,
        tile_id=tile,
        weight=count.astype(np.float64) / per_row_total,
        n_members={int(h): int(t) for h, t in zip(uniq, totals)},
        meta={"source": source, "n_objects": int(uniq.size)},
    )


def majority_weights(weights: MemberWeights) -> MemberWeights:
    """Collapse fractional tile weights onto each object's single busiest tile.

    Why this exists
    ---------------
    Fractional attribution is *exact* -- an object's weight sums to one over the
    tiles its member particles came from, so the per-tile statistics sum to the
    whole-box catalog to numerical precision. Exactness is not the same as
    stability, and measured on set0 the two come apart badly: between two frozen
    SR2 seeds of the same box, ``sum_t |dS_t|`` is 4033 while ``|sum_t dS_t|`` is
    317. **92% of the per-tile label churn cancels on pooling.** The catalog is
    stable; its decomposition is not.

    The mechanism is that a cluster's bound particles originate in many
    Lagrangian tiles, so its weight is spread thin, and any marginal change in
    which particles Rockstar binds redistributes that weight. The halo has not
    moved and its mass has barely changed -- only the split has.

    Collapsing to the argmax tile keeps the property every downstream identity
    needs (each object still carries total weight exactly one, so the tile sums
    still reproduce :func:`direct_full_box_stats` exactly) while making the
    label move only when an object's *majority* tile changes, which requires a
    real change rather than a reshuffle.

    What it costs: a large halo genuinely does span many tiles, and this assigns
    all of it to one. The per-tile label becomes a coarser statement -- "which
    tile is most responsible" rather than "how responsible is each tile". That
    is a modelling choice, which is why both schemes are computed from the same
    Rockstar run and compared rather than one being assumed.

    Ties go to the lowest tile id, so the result is deterministic.
    """
    if weights.halo_id.size == 0:
        return MemberWeights(
            weights.halo_id.copy(), weights.tile_id.copy(), weights.weight.copy(),
            dict(weights.n_members), {**weights.meta, "attribution": "majority"})

    order = np.lexsort((weights.tile_id, weights.halo_id))
    halo = weights.halo_id[order]
    tile = weights.tile_id[order]
    w = weights.weight[order]

    uniq, start = np.unique(halo, return_index=True)
    ends = np.append(start[1:], halo.size)
    # argmax within each object's contiguous run. np.maximum.reduceat gives the
    # max; the first index attaining it is the winner, which makes ties
    # deterministic without a Python loop over 300k objects.
    peak = np.maximum.reduceat(w, start)
    is_peak = w >= np.repeat(peak, ends - start) - 1e-15
    # First index attaining the group max: mask non-peaks to +inf and take the
    # per-group minimum index. Vectorised because a Python loop over ~300k
    # objects runs on every candidate of every box.
    idx = np.where(is_peak, np.arange(halo.size), halo.size)
    pick = np.minimum.reduceat(idx, start)

    return MemberWeights(
        halo_id=uniq.astype(np.int64),
        tile_id=tile[pick].astype(np.int64),
        weight=np.ones(uniq.size, dtype=np.float64),
        n_members={int(h): int(weights.n_members.get(int(h), 0)) for h in uniq},
        meta={**weights.meta, "attribution": "majority",
              "mean_majority_share": float(np.mean(peak))},
    )


def check_member_consistency(
    cat: HaloCatalog, weights: MemberWeights
) -> Dict[str, object]:
    """Do the streamed member sets and the ASCII catalog describe one run?

    Returns diagnostics rather than raising, so a job can log the numbers and
    decide. ``ok`` is False only for conditions that make the decomposition
    wrong, which deliberately does **not** include ``rows != num_p``: see the
    module docstring for why that inequality is expected and what it means.

    * ``missing`` -- catalog objects with no member rows at all. Fatal: their
      weight is nowhere, so the tile sums cannot reproduce the full box.
    * ``rows_below_num_p`` -- objects with fewer rows than their own particle
      list. Fatal: the table is truncated or misparsed.
    * ``n_leaf_exact`` / ``n_with_substructure`` -- the expected split. If
      *every* object were "exact" the recursion would not be happening, which
      would mean substructure had been silently stripped from its host.
    """
    num_p = {int(i): int(n) for i, n in zip(cat.ids, cat.num_p)}
    has_child = set(int(p) for p in cat.parent_ids if p >= 0)
    rows = {int(k): int(v) for k, v in weights.n_members.items()}

    missing = sorted(i for i in num_p if i not in rows)
    below = sorted((i, rows[i], num_p[i]) for i in num_p
                   if i in rows and rows[i] < num_p[i])
    exact = [i for i in num_p if i in rows and rows[i] == num_p[i]]
    over = [i for i in num_p if i in rows and rows[i] > num_p[i]]

    return {
        "n_catalog_objects": len(num_p),
        "n_with_member_rows": len(rows),
        "missing": missing[:20],
        "n_missing": len(missing),
        "rows_below_num_p": below[:20],
        "n_rows_below_num_p": len(below),
        "n_leaf_exact": len(exact),
        "n_with_substructure": len(over),
        "n_flagged_as_parent_in_catalog": len(has_child),
        "max_weight_error": weights.max_weight_error(),
        "ok": bool(not missing and not below
                   and weights.max_weight_error() < 1e-9),
    }


def center_tile_attribution(
    cat: HaloCatalog,
    purity,
    *,
    min_purity: float = 0.8,
) -> Tuple[MemberWeights, np.ndarray]:
    """Fallback: one-hot weights from the tile owning each object's centre cell.

    Returns ``(weights, boundary_flag)``. ``boundary_flag[i]`` is True when the
    centre cell's majority share is below ``min_purity``, i.e. the object sits
    where tiles interleave and the one-hot assignment is a guess. Such objects
    are **still attributed and still carry total weight one** -- they are
    flagged, not dropped, because dropping them is what made the chunk mask
    mass-selective in the first place.

    ``purity`` is a :class:`cosmo_sr.reward.geometry.PurityGrid` built at the
    tile geometry (``chunk_hr = tile_hr``).
    """
    pos = np.asarray(cat.pos, dtype=np.float64)
    n = pos.shape[0]
    if n == 0:
        z = np.zeros(0, dtype=np.int64)
        return MemberWeights(z, z, np.zeros(0), {}, {"source": "center_tile"}), \
            np.zeros(0, dtype=bool)
    e = purity.grid
    cell = purity.cell_mpc_h
    base = np.floor((pos % purity.boxsize_mpc_h) / cell).astype(np.int64) % e
    tile = purity.majority_id[base[:, 0], base[:, 1], base[:, 2]].astype(np.int64)
    frac = purity.majority_frac[base[:, 0], base[:, 1], base[:, 2]]
    boundary = (tile < 0) | (frac < float(min_purity))
    # A cell that received no particles at all has majority_id -1; fall back to
    # the tile the object's own centre sits in geometrically, so the weight is
    # still assigned rather than lost.
    n_per_axis = purity.ng_hr // purity.chunk_hr
    geo = np.floor(pos % purity.boxsize_mpc_h
                   / (purity.boxsize_mpc_h / n_per_axis)).astype(np.int64) % n_per_axis
    geo_id = (geo[:, 0] * n_per_axis + geo[:, 1]) * n_per_axis + geo[:, 2]
    tile = np.where(tile < 0, geo_id, tile)
    ids = np.asarray(cat.ids, dtype=np.int64)
    w = MemberWeights(
        halo_id=ids.copy(), tile_id=tile,
        weight=np.ones(n, dtype=np.float64),
        n_members={int(i): 1 for i in ids},
        meta={"source": "center_tile", "n_boundary": int(boundary.sum())},
    )
    order = np.lexsort((w.tile_id, w.halo_id))
    w.halo_id, w.tile_id, w.weight = w.halo_id[order], w.tile_id[order], w.weight[order]
    return w, boundary


# --------------------------------------------------------------------------
# Sufficient statistics
# --------------------------------------------------------------------------


@dataclass
class TileSummary:
    """Fractional catalog statistics attributable to one 64^3 generator tile.

    Counts are **float**, not int: an object straddling tiles contributes a
    fraction to each. ``ChunkSummary`` cannot be reused for this -- its
    ``to_dict`` casts counts to int, which would silently floor every fractional
    weight to zero.
    """

    box: str
    tile_id: int
    source: str
    n_sub: np.ndarray            # (J,) N[j, m], abundance by subhalo mass
    n_host: np.ndarray           # (I,) H[j, b], host denominator by host mass
    occ_numerator: np.ndarray    # (I,) S[j, b], subhalos by their HOST's mass
    volume_mpc3: float
    n_host_objects: float = 0.0
    n_sub_objects: float = 0.0
    meta: Dict = dc_field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.box}/t{self.tile_id}"

    def to_dict(self) -> Dict:
        return {
            "box": self.box,
            "tile_id": int(self.tile_id),
            "source": self.source,
            "n_sub": [float(x) for x in np.asarray(self.n_sub)],
            "n_host": [float(x) for x in np.asarray(self.n_host)],
            "occ_numerator": [float(x) for x in np.asarray(self.occ_numerator)],
            "volume_mpc3": float(self.volume_mpc3),
            "n_host_objects": float(self.n_host_objects),
            "n_sub_objects": float(self.n_sub_objects),
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(d: Dict) -> "TileSummary":
        return TileSummary(
            box=str(d["box"]),
            tile_id=int(d["tile_id"]),
            source=str(d.get("source", "")),
            n_sub=np.asarray(d["n_sub"], dtype=np.float64),
            n_host=np.asarray(d["n_host"], dtype=np.float64),
            occ_numerator=np.asarray(d["occ_numerator"], dtype=np.float64),
            volume_mpc3=float(d["volume_mpc3"]),
            n_host_objects=float(d.get("n_host_objects", 0.0)),
            n_sub_objects=float(d.get("n_sub_objects", 0.0)),
            meta=dict(d.get("meta", {})),
        )


def _qualifying(cat: HaloCatalog, bins: CatalogBins):
    """Rows that pass the resolution cuts, and the subhalo -> host-row map.

    Mirrors :func:`cosmo_sr.reward.catalog.summarize_catalog` so the two
    geometries measure the same objects -- with one deliberate difference: there
    is no chunk-membership condition, because a subhalo and its host are now
    *shared* between tiles by weight rather than assigned to one.
    """
    ids = np.asarray(cat.ids, dtype=np.int64)
    parent = np.asarray(cat.parent_ids, dtype=np.int64)
    num_p = np.asarray(cat.num_p, dtype=np.int64)

    is_host = parent < 0
    host_ok = is_host & (num_p >= bins.min_host_particles)
    sub_ok = (~is_host) & (num_p >= bins.min_sub_particles)

    host_rows = np.nonzero(host_ok)[0]
    id_to_row = {int(ids[r]): int(r) for r in host_rows}

    keep_sub: List[int] = []
    sub_host_row: List[int] = []
    for r in np.nonzero(sub_ok)[0]:
        hr = id_to_row.get(int(parent[r]))
        if hr is None:          # host failed the resolution cut
            continue
        keep_sub.append(int(r))
        sub_host_row.append(hr)
    return (host_rows,
            np.asarray(keep_sub, dtype=np.int64),
            np.asarray(sub_host_row, dtype=np.int64))


def tile_summaries(
    cat: HaloCatalog,
    weights: MemberWeights,
    bins: CatalogBins,
    grid: TileGrid,
    *,
    box: str,
    source: str,
) -> Dict[int, TileSummary]:
    """Per-tile sufficient statistics ``H[j, b]``, ``S[j, b]``, ``N[j, m]``.

    Every qualifying object distributes its total weight of one over the tiles
    that produced its particles, so summing the returned tiles reproduces the
    direct full-box statistics exactly (:func:`direct_full_box_stats`).

    An object with no weight rows (possible only for the centre-tile fallback on
    an empty catalog) is an error, not a silent drop: it would break the
    partition of unity that every downstream identity relies on.
    """
    host_rows, sub_rows, sub_host_rows = _qualifying(cat, bins)
    mvir = np.asarray(cat.mvir, dtype=np.float64)
    ids = np.asarray(cat.ids, dtype=np.int64)

    sub_edges = np.asarray(bins.sub_mass_edges, dtype=np.float64)
    host_edges = np.asarray(bins.host_mass_edges, dtype=np.float64)
    n_j, n_i, n_t = bins.n_sub_bins, bins.n_host_bins, grid.n_tiles

    # id -> slice into the COO weight arrays
    order_ids, starts = (np.unique(weights.halo_id, return_index=True)
                         if weights.halo_id.size else
                         (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)))
    ends = np.append(starts[1:], weights.halo_id.size)
    span = {int(h): (int(a), int(b)) for h, a, b in zip(order_ids, starts, ends)}

    H = np.zeros((n_t, n_i), dtype=np.float64)
    S = np.zeros((n_t, n_i), dtype=np.float64)
    N = np.zeros((n_t, n_j), dtype=np.float64)
    n_host_obj = np.zeros(n_t, dtype=np.float64)
    n_sub_obj = np.zeros(n_t, dtype=np.float64)
    missing: List[int] = []

    def _bin(m: float, edges: np.ndarray) -> int:
        b = int(np.digitize(m, edges) - 1)
        return b if 0 <= b < len(edges) - 1 else -1

    for r in host_rows:
        b = _bin(mvir[r], host_edges)
        sp = span.get(int(ids[r]))
        if sp is None:
            missing.append(int(ids[r]))
            continue
        a, z = sp
        if b < 0:
            continue
        np.add.at(H[:, b], weights.tile_id[a:z], weights.weight[a:z])
        np.add.at(n_host_obj, weights.tile_id[a:z], weights.weight[a:z])

    for r, hr in zip(sub_rows, sub_host_rows):
        sp = span.get(int(ids[r]))
        if sp is None:
            missing.append(int(ids[r]))
            continue
        a, z = sp
        tid, w = weights.tile_id[a:z], weights.weight[a:z]
        m = _bin(mvir[r], sub_edges)
        if m >= 0:
            np.add.at(N[:, m], tid, w)
        b = _bin(mvir[hr], host_edges)
        if b >= 0:
            np.add.at(S[:, b], tid, w)
        np.add.at(n_sub_obj, tid, w)

    if missing:
        raise ValueError(
            f"{len(missing)} qualifying catalog objects have no member-particle "
            f"rows (first ids: {missing[:5]}). The `.particles` table and the "
            "ASCII catalog do not describe the same run."
        )

    vol = grid.tile_volume_mpc3
    return {
        t: TileSummary(
            box=str(box), tile_id=int(t), source=str(source),
            n_sub=N[t].copy(), n_host=H[t].copy(), occ_numerator=S[t].copy(),
            volume_mpc3=vol,
            n_host_objects=float(n_host_obj[t]),
            n_sub_objects=float(n_sub_obj[t]),
            meta={"attribution": weights.meta.get("source", "")},
        )
        for t in range(n_t)
    }


def direct_full_box_stats(
    cat: HaloCatalog, bins: CatalogBins
) -> Dict[str, np.ndarray]:
    """Whole-box ``H[b]``, ``S[b]``, ``N[m]``, ``O[b]`` with no attribution at all.

    This is the reference the tile decomposition is checked against. It touches
    neither tiles nor member IDs, so agreement is a real test rather than a
    restatement.
    """
    host_rows, sub_rows, sub_host_rows = _qualifying(cat, bins)
    mvir = np.asarray(cat.mvir, dtype=np.float64)
    H, _ = np.histogram(mvir[host_rows], bins=np.asarray(bins.host_mass_edges))
    N, _ = np.histogram(mvir[sub_rows], bins=np.asarray(bins.sub_mass_edges))
    S, _ = np.histogram(mvir[sub_host_rows], bins=np.asarray(bins.host_mass_edges))
    H = H.astype(np.float64)
    S = S.astype(np.float64)
    return {
        "n_host": H,
        "occ_numerator": S,
        "n_sub": N.astype(np.float64),
        "occupation": np.divide(S, H, out=np.full_like(S, np.nan), where=H > 0),
    }


def pool_tiles(
    summaries: Iterable[TileSummary], *, drop: Optional[Sequence[int]] = None
) -> EnsembleSummary:
    """Sum tiles into one :class:`EnsembleSummary`, optionally omitting some.

    ``drop`` is what makes the leave-one-out credit cheap: removing a tile is a
    subtraction on cached vectors, never another halo-finder run.
    """
    skip = {int(d) for d in (drop or ())}
    items = [s for s in summaries if int(s.tile_id) not in skip]
    if not items:
        raise ValueError("cannot pool an empty set of tiles")
    n_sub = np.zeros_like(np.asarray(items[0].n_sub, dtype=np.float64))
    n_host = np.zeros_like(np.asarray(items[0].n_host, dtype=np.float64))
    num = np.zeros_like(n_host)
    vol = 0.0
    for s in items:
        n_sub += np.asarray(s.n_sub, dtype=np.float64)
        n_host += np.asarray(s.n_host, dtype=np.float64)
        num += np.asarray(s.occ_numerator, dtype=np.float64)
        vol += float(s.volume_mpc3)
    return EnsembleSummary(
        n_sub, n_host, num, vol,
        tuple(sorted(s.key for s in items)),
    )


def leave_one_out_credit(
    summaries: Sequence[TileSummary],
    reward_fn,
) -> Dict[int, float]:
    """``A_j = R(S) - R(S - s_j)`` for every tile, from cached statistics only.

    ``reward_fn`` maps an :class:`EnsembleSummary` to a scalar; in production it
    is ``RewardModel.reward`` or ``RewardModel.reward_occupation``.

    This is the *removal* form the experiment brief specifies, not the
    swap-to-baseline form used by :mod:`cosmo_sr.reward.replay`. They answer
    different questions -- "what does this tile contribute?" versus "what would
    this tile have contributed had the frozen generator produced it?" -- and
    they are not interchangeable: removal also shrinks the volume and the host
    denominators, so ``A_j`` carries a size effect that the swap form cancels.
    Positive ``A_j`` means the ensemble is better *with* tile ``j`` present.

    Cost is ``n_tiles + 1`` reward evaluations on 11-vectors. The halo finder is
    never re-run, which is the only reason per-tile credit is affordable at all.
    """
    items = list(summaries)
    full = reward_fn(pool_tiles(items))
    out: Dict[int, float] = {}
    for s in items:
        if len(items) <= 1:
            out[int(s.tile_id)] = float("nan")
            continue
        out[int(s.tile_id)] = float(full - reward_fn(pool_tiles(items, drop=[s.tile_id])))
    return out


def splice_tiles(
    base: np.ndarray,
    donor: np.ndarray,
    tile_ids: Sequence[int],
    grid: TileGrid,
    *,
    channels: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Copy ``donor``'s content into ``base`` on the given Lagrangian tiles.

    Used to *verify* per-tile credit: the predicted reward change from swapping
    a tile's cached summary is compared against re-running the halo finder on a
    field where that tile really was swapped.

    The splice is hard-edged, with no blending across the tile boundary. That is
    deliberate and in-distribution: SR2 itself generates each 64^3 tile from a
    separately padded LR crop and trims to the centre, so its own output is a
    hard tile mosaic. Blending here would measure a smoother field than the
    generator can actually produce.
    """
    out = np.array(base, dtype=np.float32, copy=True)
    src = np.asarray(donor)
    if out.shape != src.shape:
        raise ValueError(f"donor {src.shape} != base {out.shape}")
    ch = list(channels) if channels is not None else list(range(out.shape[0]))
    for t in tile_ids:
        sx, sy, sz = grid.slices(int(t))
        out[ch, sx, sy, sz] = src[ch, sx, sy, sz]
    return out


def swap_prediction(
    base_tiles: Sequence[TileSummary],
    donor_tiles: Dict[int, TileSummary],
    tile_id: int,
    reward_fn,
) -> float:
    """Predicted ``dR`` from replacing one tile's summary with the donor's.

    Unlike :func:`leave_one_out_credit`, this form *is* checkable against a real
    halo-finder run, because "replace tile j with HR" is a field operation
    whereas "delete tile j" is not. The gap between this prediction and the
    measured change is exactly the cross-tile interaction that per-tile credit
    assumes away.
    """
    items = [s for s in base_tiles if int(s.tile_id) != int(tile_id)]
    donor = donor_tiles.get(int(tile_id))
    if donor is None:
        raise KeyError(f"no donor summary for tile {tile_id}")
    before = reward_fn(pool_tiles(list(base_tiles)))
    after = reward_fn(pool_tiles(items + [donor]))
    return float(after - before)


def write_tile_summaries(path: str | Path, summaries: Iterable[TileSummary]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        for s in summaries:
            fh.write(json.dumps(s.to_dict(), sort_keys=True) + "\n")
    return p


def read_tile_summaries(path: str | Path) -> List[TileSummary]:
    out: List[TileSummary] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(TileSummary.from_dict(json.loads(line)))
    return out
