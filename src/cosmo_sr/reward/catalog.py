"""Chunk-level halo-catalog summaries: subhalo abundance and host occupation.

The reward's summary vector is deliberately narrow -- only the two statistics the
target failure is about:

1. ``N_sub(M_j)``  subhalo counts in subhalo-mass bins, as a number density;
2. ``<N_sub | M_host,i>``  mean subhalo occupation in host-mass bins.

Radial profiles, one-halo clustering, velocities, bispectra and HR-SR object
matching are deliberately **not** optimized, so they stay available as held-out
evidence that the model learned substructure rather than the reward.

Everything is stored as **raw counts plus the volume / host count they were
measured against**, never as a pre-divided rate. Pooling several chunks is then
a sum of numerators and a sum of denominators, which makes it exactly
associative: summarising then merging gives the same answer as merging then
summarising, and chunk order cannot matter.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..eval.rockstar import HaloCatalog

__all__ = [
    "CatalogBins",
    "ChunkSummary",
    "EnsembleSummary",
    "load_bins",
    "pool",
    "summarize_catalog",
    "summarize_full_box",
    "summary_vector",
]

# ``chunk_id`` of a whole-periodic-box summary. Negative so it can never collide
# with a real chunk id, and so a full-box row in a summaries file is skipped by
# anything iterating over chunks.
FULL_BOX_CHUNK_ID = -1


@dataclass(frozen=True)
class CatalogBins:
    """Mass bins and resolution cuts; the single source of truth is the YAML."""

    sub_mass_edges: Tuple[float, ...]
    host_mass_edges: Tuple[float, ...]
    min_sub_particles: int = 20
    min_host_particles: int = 20
    min_purity: float = 0.8
    radius_mult: float = 1.0
    abundance_transform: str = "log10_floor"
    occupation_transform: str = "log10_floor"
    abundance_floor_halos: float = 0.1     # counts floor, converted to a density
    occupation_floor: float = 0.05

    @property
    def n_sub_bins(self) -> int:
        return len(self.sub_mass_edges) - 1

    @property
    def n_host_bins(self) -> int:
        return len(self.host_mass_edges) - 1

    @property
    def dim(self) -> int:
        return self.n_sub_bins + self.n_host_bins

    def labels(self) -> List[str]:
        sm = [f"nsub_M{self.sub_mass_edges[i]:.2e}" for i in range(self.n_sub_bins)]
        ho = [f"occ_Mhost{self.host_mass_edges[i]:.2e}" for i in range(self.n_host_bins)]
        return sm + ho

    def to_dict(self) -> Dict:
        return {
            "sub_mass_edges": list(self.sub_mass_edges),
            "host_mass_edges": list(self.host_mass_edges),
            "min_sub_particles": self.min_sub_particles,
            "min_host_particles": self.min_host_particles,
            "min_purity": self.min_purity,
            "radius_mult": self.radius_mult,
            "abundance_transform": self.abundance_transform,
            "occupation_transform": self.occupation_transform,
            "abundance_floor_halos": self.abundance_floor_halos,
            "occupation_floor": self.occupation_floor,
        }


def load_bins(cfg: Dict) -> CatalogBins:
    """Build :class:`CatalogBins` from the ``reward.catalog`` block of a YAML."""
    c = dict(cfg or {})
    sub = c.get("sub_mass_bins", {})
    host = c.get("host_mass_bins", {})

    def edges(spec, default_lo, default_hi, default_n):
        if isinstance(spec, (list, tuple)):
            return tuple(float(x) for x in spec)
        s = dict(spec or {})
        return tuple(
            np.logspace(
                float(s.get("log10_min", default_lo)),
                float(s.get("log10_max", default_hi)),
                int(s.get("n_bins", default_n)) + 1,
            ).tolist()
        )

    return CatalogBins(
        sub_mass_edges=edges(sub, 10.0, 13.0, 6),
        host_mass_edges=edges(host, 12.0, 14.5, 5),
        min_sub_particles=int(c.get("min_sub_particles", 20)),
        min_host_particles=int(c.get("min_host_particles", 20)),
        min_purity=float(c.get("min_purity", 0.8)),
        radius_mult=float(c.get("radius_mult", 1.0)),
        abundance_transform=str(c.get("abundance_transform", "log10_floor")),
        occupation_transform=str(c.get("occupation_transform", "log10_floor")),
        abundance_floor_halos=float(c.get("abundance_floor_halos", 0.1)),
        occupation_floor=float(c.get("occupation_floor", 0.05)),
    )


@dataclass
class ChunkSummary:
    """Raw catalog counts attributable to one Lagrangian chunk."""

    box: str
    chunk_id: int
    source: str                       # "hr" | "base" | "candidate"
    n_sub: np.ndarray                 # (J,) subhalo counts by subhalo mass
    n_host: np.ndarray                # (I,) host counts by host mass  (denominator)
    occ_numerator: np.ndarray         # (I,) subhalo counts by HOST mass (numerator)
    volume_mpc3: float
    n_sub_total: int = 0
    n_host_total: int = 0
    n_excluded_boundary: int = 0
    n_excluded_resolution: int = 0
    meta: Dict = dc_field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.box}/c{self.chunk_id}"

    def to_dict(self) -> Dict:
        return {
            "box": self.box,
            "chunk_id": int(self.chunk_id),
            "source": self.source,
            "n_sub": np.asarray(self.n_sub).astype(int).tolist(),
            "n_host": np.asarray(self.n_host).astype(int).tolist(),
            "occ_numerator": np.asarray(self.occ_numerator).astype(int).tolist(),
            "volume_mpc3": float(self.volume_mpc3),
            "n_sub_total": int(self.n_sub_total),
            "n_host_total": int(self.n_host_total),
            "n_excluded_boundary": int(self.n_excluded_boundary),
            "n_excluded_resolution": int(self.n_excluded_resolution),
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(d: Dict) -> "ChunkSummary":
        return ChunkSummary(
            box=str(d["box"]),
            chunk_id=int(d["chunk_id"]),
            source=str(d.get("source", "")),
            n_sub=np.asarray(d["n_sub"], dtype=np.float64),
            n_host=np.asarray(d["n_host"], dtype=np.float64),
            occ_numerator=np.asarray(d["occ_numerator"], dtype=np.float64),
            volume_mpc3=float(d["volume_mpc3"]),
            n_sub_total=int(d.get("n_sub_total", 0)),
            n_host_total=int(d.get("n_host_total", 0)),
            n_excluded_boundary=int(d.get("n_excluded_boundary", 0)),
            n_excluded_resolution=int(d.get("n_excluded_resolution", 0)),
            meta=dict(d.get("meta", {})),
        )


@dataclass
class EnsembleSummary:
    """Pooled counts over the chunks of one ensemble."""

    n_sub: np.ndarray
    n_host: np.ndarray
    occ_numerator: np.ndarray
    volume_mpc3: float
    members: Tuple[str, ...] = ()

    def number_density(self) -> np.ndarray:
        return np.asarray(self.n_sub, dtype=np.float64) / max(self.volume_mpc3, 1e-30)

    def occupation(self) -> np.ndarray:
        num = np.asarray(self.occ_numerator, dtype=np.float64)
        den = np.asarray(self.n_host, dtype=np.float64)
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    def empty_host_bins(self) -> np.ndarray:
        return np.asarray(self.n_host, dtype=np.float64) <= 0

    def to_dict(self) -> Dict:
        return {
            "n_sub": np.asarray(self.n_sub).astype(float).tolist(),
            "n_host": np.asarray(self.n_host).astype(float).tolist(),
            "occ_numerator": np.asarray(self.occ_numerator).astype(float).tolist(),
            "volume_mpc3": float(self.volume_mpc3),
            "members": list(self.members),
        }

    @staticmethod
    def from_dict(d: Dict) -> "EnsembleSummary":
        return EnsembleSummary(
            n_sub=np.asarray(d["n_sub"], dtype=np.float64),
            n_host=np.asarray(d["n_host"], dtype=np.float64),
            occ_numerator=np.asarray(d["occ_numerator"], dtype=np.float64),
            volume_mpc3=float(d["volume_mpc3"]),
            members=tuple(d.get("members", ())),
        )


def pool(summaries: Iterable[ChunkSummary | EnsembleSummary]) -> EnsembleSummary:
    """Sum numerators, denominators and volumes. Order-independent by construction."""
    items = list(summaries)
    if not items:
        raise ValueError("cannot pool an empty ensemble")
    j = len(np.asarray(items[0].n_sub))
    i = len(np.asarray(items[0].n_host))
    n_sub = np.zeros(j, dtype=np.float64)
    n_host = np.zeros(i, dtype=np.float64)
    num = np.zeros(i, dtype=np.float64)
    vol = 0.0
    members: List[str] = []
    for s in items:
        if len(np.asarray(s.n_sub)) != j or len(np.asarray(s.n_host)) != i:
            raise ValueError("cannot pool summaries built with different bins")
        n_sub += np.asarray(s.n_sub, dtype=np.float64)
        n_host += np.asarray(s.n_host, dtype=np.float64)
        num += np.asarray(s.occ_numerator, dtype=np.float64)
        vol += float(s.volume_mpc3)
        members.extend(getattr(s, "members", ()) or [getattr(s, "key", "")])
    return EnsembleSummary(n_sub, n_host, num, vol, tuple(sorted(m for m in members if m)))


def _transform(x: np.ndarray, kind: str, floor: float) -> np.ndarray:
    if kind == "identity":
        return np.asarray(x, dtype=np.float64)
    if kind == "log10_floor":
        return np.log10(np.asarray(x, dtype=np.float64) + float(floor))
    if kind == "sqrt":
        return np.sqrt(np.maximum(np.asarray(x, dtype=np.float64), 0.0))
    raise ValueError(f"unknown transform {kind!r}")


def summary_vector(
    ens: EnsembleSummary,
    bins: CatalogBins,
    *,
    empty_fill: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(s, valid)``: the summary vector and a mask of informative entries.

    Host-mass bins with no hosts carry no information. Rather than emitting NaN
    (which would poison the Mahalanobis form) their entry is filled with
    ``empty_fill`` -- in practice the HR mean, so the bin contributes exactly
    zero to the distance instead of a fabricated penalty.
    """
    dens = ens.number_density()
    abundance = _transform(
        dens,
        bins.abundance_transform,
        bins.abundance_floor_halos / max(ens.volume_mpc3, 1e-30),
    )
    occ = ens.occupation()
    empty = ens.empty_host_bins()
    occ_filled = np.where(empty, 0.0, np.nan_to_num(occ, nan=0.0))
    occupation = _transform(occ_filled, bins.occupation_transform, bins.occupation_floor)
    s = np.concatenate([abundance, occupation])
    valid = np.concatenate([np.ones(bins.n_sub_bins, dtype=bool), ~empty])
    if empty_fill is not None:
        fill = np.asarray(empty_fill, dtype=np.float64)
        if fill.shape != s.shape:
            raise ValueError(f"empty_fill {fill.shape} != summary {s.shape}")
        s = np.where(valid, s, fill)
    return s, valid


def summarize_catalog(
    cat: HaloCatalog,
    chunk_of_halo: np.ndarray,
    bins: CatalogBins,
    chunk_volumes: Sequence[float],
    *,
    box: str,
    source: str,
    chunk_ids: Optional[Sequence[int]] = None,
) -> Dict[int, ChunkSummary]:
    """Per-chunk counts from one full-box catalog and its chunk assignment.

    ``chunk_of_halo`` comes from
    :func:`cosmo_sr.reward.geometry.assign_halos_to_chunks` (``-1`` = boundary
    contaminated). A subhalo is counted only when its **host** is also a
    qualifying, same-chunk halo, so an occupation numerator can never be paired
    with a denominator from a different chunk or from an unresolved host.
    """
    chunk_of_halo = np.asarray(chunk_of_halo, dtype=np.int64)
    if chunk_of_halo.shape[0] != cat.n:
        raise ValueError(
            f"assignment length {chunk_of_halo.shape[0]} != catalog size {cat.n}"
        )
    ids = np.asarray(cat.ids, dtype=np.int64)
    parent = np.asarray(cat.parent_ids, dtype=np.int64)
    mvir = np.asarray(cat.mvir, dtype=np.float64)
    num_p = np.asarray(cat.num_p, dtype=np.int64)

    is_host = parent < 0
    host_ok = is_host & (num_p >= bins.min_host_particles) & (chunk_of_halo >= 0)
    sub_res_ok = (~is_host) & (num_p >= bins.min_sub_particles)
    sub_ok = sub_res_ok & (chunk_of_halo >= 0)

    # host id -> row, restricted to qualifying hosts
    host_rows = np.nonzero(host_ok)[0]
    id_to_row = {int(ids[r]): int(r) for r in host_rows}

    sub_rows = np.nonzero(sub_ok)[0]
    keep_sub: List[int] = []
    sub_host_row: List[int] = []
    for r in sub_rows:
        hr_row = id_to_row.get(int(parent[r]))
        if hr_row is None:
            continue
        if chunk_of_halo[hr_row] != chunk_of_halo[r]:
            continue
        keep_sub.append(int(r))
        sub_host_row.append(hr_row)
    keep_sub_arr = np.asarray(keep_sub, dtype=np.int64)
    sub_host_arr = np.asarray(sub_host_row, dtype=np.int64)

    sub_edges = np.asarray(bins.sub_mass_edges, dtype=np.float64)
    host_edges = np.asarray(bins.host_mass_edges, dtype=np.float64)
    j, i = bins.n_sub_bins, bins.n_host_bins

    wanted = list(chunk_ids) if chunk_ids is not None else sorted(
        set(int(c) for c in np.unique(chunk_of_halo) if c >= 0)
    )
    out: Dict[int, ChunkSummary] = {}
    for c in wanted:
        hsel = host_rows[chunk_of_halo[host_rows] == c]
        n_host, _ = np.histogram(mvir[hsel], bins=host_edges)
        if keep_sub_arr.size:
            ssel = keep_sub_arr[chunk_of_halo[keep_sub_arr] == c]
            hosts_of = sub_host_arr[chunk_of_halo[keep_sub_arr] == c]
        else:
            ssel = np.zeros(0, dtype=np.int64)
            hosts_of = np.zeros(0, dtype=np.int64)
        n_sub, _ = np.histogram(mvir[ssel], bins=sub_edges)
        occ_num, _ = np.histogram(mvir[hosts_of], bins=host_edges)
        vol = float(chunk_volumes[c]) if c < len(chunk_volumes) else 0.0
        out[c] = ChunkSummary(
            box=str(box), chunk_id=int(c), source=str(source),
            n_sub=n_sub.astype(np.float64), n_host=n_host.astype(np.float64),
            occ_numerator=occ_num.astype(np.float64), volume_mpc3=vol,
            n_sub_total=int(ssel.size), n_host_total=int(hsel.size),
            n_excluded_boundary=int(
                np.count_nonzero((chunk_of_halo < 0) & (host_ok | sub_res_ok))
            ),
            n_excluded_resolution=int(np.count_nonzero(~(host_ok | sub_res_ok | is_host))),
        )
    return out


def summarize_full_box(
    cat: HaloCatalog,
    bins: CatalogBins,
    volume_mpc3: float,
    *,
    box: str,
    source: str,
) -> ChunkSummary:
    """The same counts, taken on the WHOLE periodic box with no chunk mask.

    Rockstar already runs on the full periodic box; it is the chunk *attribution*
    that then throws objects away. A halo whose Rvir sphere straddles a chunk
    boundary is not contaminated in any physical sense -- the box is periodic and
    the halo is entirely real -- but the purity mask drops it, and the drop rate
    was measured to rise with host mass, i.e. it removes preferentially the hosts
    the occupation reward is about. Summing chunks cannot put them back.

    So the reward statistic (Gate B) is taken here, on everything the halo finder
    found, with the nominal box volume. Chunk summaries stay exactly as they are
    and keep their one job: *marginal credit*, attributing an improvement to a
    Lagrangian region so the replay buffer can weight it.

    Returned as a :class:`ChunkSummary` with ``chunk_id = FULL_BOX_CHUNK_ID`` so
    it shares the pooling and serialisation machinery; ``meta['scope']`` says
    what it is.
    """
    ids = np.asarray(cat.ids, dtype=np.int64)
    parent = np.asarray(cat.parent_ids, dtype=np.int64)
    mvir = np.asarray(cat.mvir, dtype=np.float64)
    num_p = np.asarray(cat.num_p, dtype=np.int64)

    is_host = parent < 0
    host_ok = is_host & (num_p >= bins.min_host_particles)
    sub_ok = (~is_host) & (num_p >= bins.min_sub_particles)

    host_rows = np.nonzero(host_ok)[0]
    id_to_row = {int(ids[r]): int(r) for r in host_rows}
    keep_sub, sub_host_row = [], []
    for r in np.nonzero(sub_ok)[0]:
        hr_row = id_to_row.get(int(parent[r]))
        if hr_row is None:      # host below the resolution cut: not a pair we can use
            continue
        keep_sub.append(int(r))
        sub_host_row.append(hr_row)

    sub_edges = np.asarray(bins.sub_mass_edges, dtype=np.float64)
    host_edges = np.asarray(bins.host_mass_edges, dtype=np.float64)
    ssel = np.asarray(keep_sub, dtype=np.int64)
    hosts_of = np.asarray(sub_host_row, dtype=np.int64)
    n_host, _ = np.histogram(mvir[host_rows], bins=host_edges)
    n_sub, _ = np.histogram(mvir[ssel] if ssel.size else np.zeros(0), bins=sub_edges)
    occ_num, _ = np.histogram(
        mvir[hosts_of] if hosts_of.size else np.zeros(0), bins=host_edges)
    return ChunkSummary(
        box=str(box), chunk_id=FULL_BOX_CHUNK_ID, source=str(source),
        n_sub=n_sub.astype(np.float64), n_host=n_host.astype(np.float64),
        occ_numerator=occ_num.astype(np.float64), volume_mpc3=float(volume_mpc3),
        n_sub_total=int(ssel.size), n_host_total=int(host_rows.size),
        n_excluded_boundary=0,
        n_excluded_resolution=int(np.count_nonzero(~(host_ok | sub_ok))),
        meta={"scope": "full_box"},
    )


def write_summaries(path: str | Path, summaries: Iterable[ChunkSummary]) -> Path:
    """One JSON object per line: compact, appendable, order-insensitive."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        for s in summaries:
            fh.write(json.dumps(s.to_dict(), sort_keys=True) + "\n")
    return p


def read_summaries(path: str | Path) -> List[ChunkSummary]:
    out: List[ChunkSummary] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(ChunkSummary.from_dict(json.loads(line)))
    return out
