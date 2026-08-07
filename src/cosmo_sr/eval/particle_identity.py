"""Are an SR2 halo and its HR counterpart made of the *same particles*?

Why this is answerable exactly
------------------------------
SR2 and HR are two displacement fields on the **same Lagrangian lattice**, and
:func:`cosmo_sr.eval.particles.field_to_particles` labels particle
``(ix, iy, iz)`` with the flat index ``(ix * Ng + iy) * Ng + iz`` in both. So
"particle 917 in SR2" and "particle 917 in HR" are the same mass element by
construction -- no matching, no tolerance, no nearest-neighbour heuristic. The
only thing that differs between the two boxes is *where* that element sits.

That makes three genuinely different questions separable, which is the point of
this module:

1. **Identity** -- does the SR2 halo hold the same *set* of Lagrangian ids as
   the HR halo it was matched to? (:func:`set_metrics`)
2. **Radius** -- for the ids the two objects do *not* share, how far away are
   they? A particle that HR binds 0.3 Rvir outside the SR2 object is a very
   different failure from one that HR binds 20 Mpc/h away.
   (:func:`radius_fractions`, :func:`displacement_stats`)
3. **Chunk** -- does the disagreement stay inside one generator tile, or does it
   cross tiles? Note this has *two* distinct meanings and both are computed:
   the **Lagrangian** tile of a particle is a function of its id alone, so it is
   identical in SR2 and HR and can only differ *between objects*
   (:func:`tile_profile`); the **Eulerian** chunk a particle occupies is a
   function of position and really can change (:func:`eulerian_chunk_shift`).

The question behind the questions
---------------------------------
The residual model is trained to map SR2 -> HR pointwise in Lagrangian space, so
it acts on particle ``i`` regardless of what SR2 made of it. If a matched pair
shares its ids and differs only by a coherent bulk shift, the residual is being
asked to *translate* a valid object -- benign, and learnable. If the id sets
disagree, the residual has to take a perfectly valid SR2 subhalo apart and
rebuild an HR one out of different mass elements, which no pointwise correction
can do without destroying the object it starts from.
:func:`displacement_stats` reports exactly that split (``bulk`` vs ``residual``
scatter), so the two cases are distinguished by measurement rather than by
assertion.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .rockstar import HaloCatalog

__all__ = [
    "OwnerIndex",
    "as_flat_catalog",
    "build_owner_index",
    "check_owner_consistency",
    "descendants_of",
    "displacement_stats",
    "eulerian_chunk_shift",
    "lagrangian_positions",
    "periodic_delta",
    "profile_overlap",
    "radius_fractions",
    "remap_to_roots",
    "root_lookup",
    "set_metrics",
    "stream_owner_assignment",
    "tile_profile",
]

# Columns of the Rockstar `.particles` table (0-based); see
# cosmo_sr.reward.tiles for the full description of the recursive emission.
COL_PARTICLE_ID = 6
COL_ASSIGNED_INTERNAL = 7
COL_INTERNAL = 8
COL_EXTERNAL = 9

UNBOUND = -1


# --------------------------------------------------------------------------
# Per-particle ownership
# --------------------------------------------------------------------------

def stream_owner_assignment(
    particles_path: str | Path,
    n_part: int,
    *,
    chunk_rows: int = 8_000_000,
) -> np.ndarray:
    """``owner[particle_id]`` -- the catalog id the particle is *bound to*.

    Rockstar's recursion prints every particle once per ancestor, so the table
    cannot be read as one row per particle. The row where the recursion root IS
    the binding object -- ``assigned_internal_haloid == internal_haloid`` --
    is unique per particle and names the deepest object that owns it, which is
    the subhalo for a satellite particle and the host for a smooth one. That is
    the leaf-most attribution, and it is what makes ``len(members(o))`` equal
    the catalog's ``num_p(o)`` (a halo's *own* particle list) rather than the
    inflated recursive count.

    Particles bound to a clump Rockstar refused to print (below
    ``MIN_HALO_OUTPUT_SIZE``) have no such row and stay ``-1``, as do genuinely
    unbound particles; :func:`check_owner_consistency` quantifies the former.

    Returns an ``int32`` array of length ``n_part``. ``-1`` means unowned.
    """
    import pandas as pd

    path = Path(particles_path)
    if not path.is_file():
        raise FileNotFoundError(f"no member-particle table at {path}")

    owner = np.full(int(n_part), UNBOUND, dtype=np.int32)
    n_rows_kept = 0
    reader = pd.read_csv(
        path, sep=r"\s+", comment="#", header=None,
        usecols=[COL_PARTICLE_ID, COL_ASSIGNED_INTERNAL, COL_INTERNAL,
                 COL_EXTERNAL],
        names=["particle_id", "assigned", "internal", "external"],
        dtype=np.int64, engine="c", chunksize=int(chunk_rows),
    )
    for block in reader:
        assigned = block["assigned"].to_numpy(dtype=np.int64, copy=False)
        internal = block["internal"].to_numpy(dtype=np.int64, copy=False)
        external = block["external"].to_numpy(dtype=np.int64, copy=False)
        keep = (assigned == internal) & (external >= 0)
        if not keep.any():
            continue
        pid = block["particle_id"].to_numpy(dtype=np.int64, copy=False)[keep]
        if pid.size and (pid.min() < 0 or pid.max() >= n_part):
            raise ValueError(
                f"particle id outside 0..{n_part - 1} (saw {int(pid.min())}.."
                f"{int(pid.max())}); the dump was not written by "
                "cosmo_sr.eval.particles"
            )
        owner[pid] = external[keep].astype(np.int32)
        n_rows_kept += int(pid.size)
        del block, assigned, internal, external, pid

    n_owned = int(np.count_nonzero(owner >= 0))
    if n_rows_kept != n_owned:
        raise ValueError(
            f"{n_rows_kept} leaf rows but only {n_owned} distinct particles "
            "own a halo: a particle was claimed by two objects, so "
            "assigned_internal_haloid == internal_haloid is not the unique "
            "leaf row it is assumed to be"
        )
    return owner


@dataclass
class OwnerIndex:
    """Members of every object, in CSR form over the owner array.

    ``particle_id[offset[k]:offset[k+1]]`` are the ids owned by ``halo_id[k]``.
    """

    halo_id: np.ndarray      # (H,) int64, sorted
    offset: np.ndarray       # (H+1,) int64
    particle_id: np.ndarray  # (M,) int64, grouped by halo
    n_part: int = 0
    n_unowned: int = 0

    def members(self, hid: int) -> np.ndarray:
        """Particles bound to ``hid`` itself, excluding its substructure."""
        k = int(np.searchsorted(self.halo_id, int(hid)))
        if k >= self.halo_id.size or int(self.halo_id[k]) != int(hid):
            return np.zeros(0, dtype=np.int64)
        return self.particle_id[self.offset[k]:self.offset[k + 1]]

    def members_with_substructure(
        self, cat: HaloCatalog, hid: int, *, children: Optional[Dict] = None
    ) -> np.ndarray:
        """``hid``'s own particles plus every descendant's, deduplicated.

        This is the object's full bound footprint, and the right set to use for
        a *host*: a satellite's mass belongs to the host too, and leaving it out
        would make a rich host look like it lost particles it never lost.
        """
        ids = descendants_of(cat, hid, children=children)
        parts = [self.members(h) for h in [int(hid)] + list(ids)]
        parts = [p for p in parts if p.size]
        if not parts:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(parts))

    def count(self, hid: int) -> int:
        return int(self.members(hid).size)

    def to_npz(self, path) -> str:
        np.savez_compressed(
            str(path), halo_id=self.halo_id, offset=self.offset,
            particle_id=self.particle_id,
            n_part=np.int64(self.n_part), n_unowned=np.int64(self.n_unowned),
        )
        return str(path)

    @staticmethod
    def from_npz(path) -> "OwnerIndex":
        z = np.load(str(path), allow_pickle=False)
        return OwnerIndex(
            halo_id=z["halo_id"].astype(np.int64),
            offset=z["offset"].astype(np.int64),
            particle_id=z["particle_id"].astype(np.int64),
            n_part=int(z["n_part"]), n_unowned=int(z["n_unowned"]),
        )


def build_owner_index(owner: np.ndarray) -> OwnerIndex:
    """Invert ``owner[pid] -> halo`` into ``halo -> [pid, ...]``.

    Only bound particles are sorted, which keeps the index roughly a third of
    the size of a full 512^3 argsort.
    """
    owner = np.asarray(owner)
    bound = np.flatnonzero(owner >= 0).astype(np.int64)
    if bound.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return OwnerIndex(z, np.zeros(1, dtype=np.int64), z,
                          n_part=int(owner.size), n_unowned=int(owner.size))
    keys = owner[bound].astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys, pid = keys[order], bound[order]
    uniq, start = np.unique(keys, return_index=True)
    offset = np.append(start, keys.size).astype(np.int64)
    return OwnerIndex(
        halo_id=uniq.astype(np.int64), offset=offset, particle_id=pid,
        n_part=int(owner.size), n_unowned=int(owner.size - bound.size),
    )


def descendants_of(
    cat: HaloCatalog, hid: int, *, children: Optional[Dict] = None
) -> List[int]:
    """Transitive substructure of ``hid`` (sub-subhalos included)."""
    if children is None:
        children = {}
        for i, p in zip(cat.ids, cat.parent_ids):
            if int(p) >= 0:
                children.setdefault(int(p), []).append(int(i))
    out: List[int] = []
    stack = list(children.get(int(hid), []))
    while stack:
        c = stack.pop()
        out.append(c)
        stack.extend(children.get(c, []))
    return out


def child_map(cat: HaloCatalog) -> Dict[int, List[int]]:
    """``parent id -> [direct children]``, built once and reused."""
    children: Dict[int, List[int]] = {}
    for i, p in zip(cat.ids, cat.parent_ids):
        if int(p) >= 0:
            children.setdefault(int(p), []).append(int(i))
    return children


def root_lookup(cat: HaloCatalog) -> np.ndarray:
    """``table[id]`` -> the top-level ancestor of ``id`` (itself, for a host).

    Indexed by catalog id, with ``-1`` for ids the catalog does not contain.
    """
    if cat.n == 0:
        return np.zeros(1, dtype=np.int32) - 1
    parent = {int(i): int(p) for i, p in zip(cat.ids, cat.parent_ids)}
    table = np.full(int(np.max(cat.ids)) + 1, -1, dtype=np.int32)
    for i in cat.ids:
        node = int(i)
        seen = set()
        while parent.get(node, -1) >= 0 and node not in seen:
            seen.add(node)
            node = parent[node]
        table[int(i)] = node
    return table


def remap_to_roots(owner: np.ndarray, cat: HaloCatalog) -> np.ndarray:
    """Per-particle ownership lifted from the leaf object to its host.

    Needed whenever *hosts* are compared: leaf attribution puts a satellite's
    particles under the satellite, so a host whose material the other box binds
    perfectly -- but into that host's own subhalos -- would otherwise be scored
    as fragmented. At host granularity those particles are in the right object.
    """
    table = root_lookup(cat)
    out = np.full_like(owner, UNBOUND)
    m = owner >= 0
    idx = owner[m].astype(np.int64)
    valid = idx < table.size
    tmp = np.full(idx.shape, UNBOUND, dtype=owner.dtype)
    tmp[valid] = table[idx[valid]].astype(owner.dtype)
    out[m] = tmp
    return out


def check_owner_consistency(cat: HaloCatalog, index: OwnerIndex) -> Dict:
    """``len(members(o))`` must equal the catalog's ``num_p(o)``.

    Leaf attribution is the whole reason this module can compare *sets*; if the
    counts disagree the sets are not the objects' particle lists and every
    Jaccard below is measuring something else. Rockstar's own bookkeeping is the
    reference, so this is a real check and not a tautology.
    """
    n_rows = np.array([index.count(int(h)) for h in cat.ids], dtype=np.int64)
    num_p = np.asarray(cat.num_p, dtype=np.int64)
    diff = n_rows - num_p
    missing = int(np.count_nonzero(n_rows == 0))
    return {
        "n_catalog_objects": int(cat.n),
        "n_with_members": int(cat.n - missing),
        "n_missing": missing,
        "n_exact": int(np.count_nonzero(diff == 0)),
        "max_abs_diff": int(np.max(np.abs(diff))) if cat.n else 0,
        "frac_exact": float(np.mean(diff == 0)) if cat.n else 1.0,
        "n_unowned_particles": int(index.n_unowned),
        "frac_particles_unowned": (float(index.n_unowned) / index.n_part
                                   if index.n_part else 0.0),
        "ok": bool(cat.n and missing == 0 and np.all(diff == 0)),
    }


# --------------------------------------------------------------------------
# Positions
# --------------------------------------------------------------------------

def lagrangian_positions(
    field, *, boxsize_mpc_h: float = 100.0, redshift: float = 0.0
) -> np.ndarray:
    """``(Ng^3, 3)`` positions in Mpc/h, indexed by particle id.

    Thin wrapper over :func:`cosmo_sr.eval.particles.field_to_particles` so the
    catnorm/growth conventions cannot drift away from the ones the GADGET dumps
    (and therefore the catalogs) were built with.
    """
    from .particles import field_to_particles

    pb = field_to_particles(
        np.asarray(field, dtype=np.float32),
        boxsize_kpc_h=float(boxsize_mpc_h) * 1000.0,
        redshift=float(redshift),
    )
    return pb.pos_mpc_h


def periodic_delta(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    """``a - b`` wrapped into ``[-box/2, box/2)``."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return d - box * np.round(d / box)


# --------------------------------------------------------------------------
# Granularity 1: identity
# --------------------------------------------------------------------------

def set_metrics(a: np.ndarray, b: np.ndarray) -> Dict:
    """Overlap of two particle-id sets.

    ``a`` is the reference (HR) set and ``b`` the candidate (SR2) set, so
    ``completeness`` is the fraction of the HR object SR2 recovered and
    ``purity`` the fraction of the SR2 object that is genuinely HR material.
    """
    a = np.unique(np.asarray(a, dtype=np.int64))
    b = np.unique(np.asarray(b, dtype=np.int64))
    shared = int(np.intersect1d(a, b, assume_unique=True).size)
    union = a.size + b.size - shared
    return {
        "n_a": int(a.size),
        "n_b": int(b.size),
        "n_shared": shared,
        "jaccard": float(shared / union) if union else 0.0,
        "completeness": float(shared / a.size) if a.size else 0.0,
        "purity": float(shared / b.size) if b.size else 0.0,
    }


# --------------------------------------------------------------------------
# Granularity 2: radius
# --------------------------------------------------------------------------

def displacement_stats(
    ids: np.ndarray,
    pos_from: np.ndarray,
    pos_to: np.ndarray,
    box: float,
) -> Dict:
    """How far, and how *coherently*, must these particles move?

    ``d_i = pos_to[i] - pos_from[i]`` is exactly the correction a pointwise
    residual has to apply to particle ``i``. Splitting it into the group mean
    and the scatter about that mean is the measurement the residual question
    turns on:

    * ``bulk_mpc_h`` large, ``residual_rms_mpc_h`` small -> SR2 built a valid
      object in the wrong place; the residual only has to translate it.
    * ``residual_rms_mpc_h`` comparable to the object's size -> the residual has
      to disperse the SR2 object and reassemble it, i.e. destroy structure that
      was internally consistent.

    ``coherent_fraction`` is the share of the mean-square displacement carried
    by the bulk term, in ``[0, 1]``.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return {"n": 0, "bulk_mpc_h": 0.0, "rms_mpc_h": 0.0,
                "residual_rms_mpc_h": 0.0, "median_mpc_h": 0.0,
                "p90_mpc_h": 0.0, "coherent_fraction": 0.0,
                "bulk_vec_mpc_h": [0.0, 0.0, 0.0]}
    d = periodic_delta(pos_to[ids], pos_from[ids], box)
    bulk = d.mean(axis=0)
    resid = d - bulk
    ms = float(np.mean(np.sum(d ** 2, axis=1)))
    ms_resid = float(np.mean(np.sum(resid ** 2, axis=1)))
    norms = np.linalg.norm(d, axis=1)
    return {
        "n": int(ids.size),
        "bulk_vec_mpc_h": [float(v) for v in bulk],
        "bulk_mpc_h": float(np.linalg.norm(bulk)),
        "rms_mpc_h": float(np.sqrt(ms)),
        "residual_rms_mpc_h": float(np.sqrt(ms_resid)),
        "median_mpc_h": float(np.median(norms)),
        "p90_mpc_h": float(np.quantile(norms, 0.9)),
        "coherent_fraction": float(1.0 - ms_resid / ms) if ms > 0 else 1.0,
    }


def radius_fractions(
    ids: np.ndarray,
    pos: np.ndarray,
    center: Sequence[float],
    box: float,
    radii_mpc_h: Sequence[float],
    names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Fraction of ``ids`` lying within each radius of ``center`` in ``pos``.

    The coarsest useful granularity: an id that is not a formal member may still
    sit inside the object, and Rockstar's boundedness cut is sharp where the
    physics is not.

    ``names`` labels the radii. Pass it whenever the radii are computed rather
    than fixed -- a caller mixing ``k * Rvir`` with absolute radii can generate
    the same number twice, and keying off the value alone would silently drop a
    column and shift every later one.
    """
    ids = np.asarray(ids, dtype=np.int64)
    radii = [float(r) for r in radii_mpc_h]
    keys = ([str(n) for n in names] if names is not None
            else [f"f_within_{r:g}" for r in radii])
    if len(keys) != len(radii):
        raise ValueError(f"{len(names or [])} names for {len(radii)} radii")
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate radius labels: {keys}")
    if ids.size == 0:
        return {k: 0.0 for k in keys}
    d = periodic_delta(pos[ids], np.asarray(center, dtype=np.float64)[None, :],
                       box)
    r = np.linalg.norm(d, axis=1)
    return {k: float(np.mean(r <= rad)) for k, rad in zip(keys, radii)}


# --------------------------------------------------------------------------
# Granularity 3: chunk
# --------------------------------------------------------------------------

def tile_profile(ids: np.ndarray, grid) -> np.ndarray:
    """Normalised histogram of the members' **Lagrangian** generator tiles.

    A particle's Lagrangian tile is a function of its id alone, so it is the
    same number in SR2 and in HR. Comparing two *objects'* profiles therefore
    asks a real question -- did SR2 assemble this halo out of the same patch of
    the initial conditions HR did? -- while asking it of one particle in two
    boxes would be vacuous.
    """
    from ..reward.tiles import tile_of_particle_id

    ids = np.asarray(ids, dtype=np.int64)
    prof = np.zeros(grid.n_tiles, dtype=np.float64)
    if ids.size == 0:
        return prof
    t = tile_of_particle_id(ids, grid)
    np.add.at(prof, t, 1.0)
    return prof / prof.sum()


def profile_overlap(u: np.ndarray, v: np.ndarray) -> Dict:
    """Agreement of two normalised tile profiles."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.sum() <= 0 or v.sum() <= 0:
        return {"intersection": 0.0, "cosine": 0.0, "n_tiles_u": 0,
                "n_tiles_v": 0, "same_dominant_tile": False}
    inter = float(np.minimum(u, v).sum())          # 1 - total variation
    cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    return {
        "intersection": inter,
        "cosine": cos,
        "n_tiles_u": int(np.count_nonzero(u)),
        "n_tiles_v": int(np.count_nonzero(v)),
        "same_dominant_tile": bool(np.argmax(u) == np.argmax(v)),
    }


def eulerian_chunk_shift(
    ids: np.ndarray,
    pos_from: np.ndarray,
    pos_to: np.ndarray,
    box: float,
    n_chunk_per_axis: int,
) -> Dict:
    """Do the particles stay in the same **Eulerian** chunk when corrected?

    This is the version of "same chunk" that can actually fail: the generator
    writes a tile of output, and a residual that must push a particle out of the
    spatial region its tile covers is asking for information the tile never saw.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return {"n": 0, "frac_same_chunk": 0.0, "frac_same_or_adjacent": 0.0}
    cell = float(box) / int(n_chunk_per_axis)
    a = np.floor(np.asarray(pos_from[ids], dtype=np.float64) / cell).astype(np.int64)
    b = np.floor(np.asarray(pos_to[ids], dtype=np.float64) / cell).astype(np.int64)
    n = int(n_chunk_per_axis)
    a %= n
    b %= n
    delta = np.abs(a - b)
    delta = np.minimum(delta, n - delta)           # periodic chunk distance
    same = np.all(delta == 0, axis=1)
    adj = np.all(delta <= 1, axis=1)
    return {
        "n": int(ids.size),
        "n_chunk_per_axis": n,
        "chunk_mpc_h": cell,
        "frac_same_chunk": float(np.mean(same)),
        "frac_same_or_adjacent": float(np.mean(adj)),
    }


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def as_flat_catalog(cat: HaloCatalog) -> HaloCatalog:
    """A copy whose objects are all top-level.

    :func:`cosmo_sr.eval.halo_match.match_hosts` matches ``cat.hosts()``. To
    match *subhalos* with the same greedy periodic-kNN logic -- rather than
    reimplementing it and risking a second, subtly different matcher -- hand it
    a catalog in which the subhalos of interest are the top-level objects.
    """
    return HaloCatalog(
        ids=cat.ids.copy(),
        parent_ids=np.full(cat.n, -1, dtype=np.int64),
        mvir=cat.mvir.copy(), rvir=cat.rvir.copy(), vmax=cat.vmax.copy(),
        pos=cat.pos.copy(), vel=cat.vel.copy(), num_p=cat.num_p.copy(),
        path=cat.path,
    )
