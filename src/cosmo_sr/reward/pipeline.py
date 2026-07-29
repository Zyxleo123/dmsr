"""Field -> catalog -> chunk summaries, shared by the oracle and evaluation stages.

Kept in one place so the oracle, the replay builder and the final evaluation
cannot drift into measuring three slightly different things.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..eval.rockstar import HaloCatalog, load_rockstar_ascii, run_rockstar_on_field
from .catalog import CatalogBins, ChunkSummary, summarize_catalog
from .geometry import ChunkGrid, PurityGrid, assign_halos_to_chunks, chunk_purity_grid

__all__ = ["CatalogResult", "existing_catalog", "field_to_chunk_summaries"]


class CatalogResult:
    def __init__(self, summaries: Dict[int, ChunkSummary], catalog: HaloCatalog,
                 purity: PurityGrid, volumes: np.ndarray, timings: Dict[str, float],
                 assignment: np.ndarray):
        self.summaries = summaries
        self.catalog = catalog
        self.purity = purity
        self.volumes = volumes
        self.timings = timings
        self.assignment = assignment

    @property
    def assigned_fraction(self) -> float:
        n = int(self.catalog.n)
        return float(np.count_nonzero(self.assignment >= 0)) / max(n, 1)

    @property
    def core_volume_fraction(self) -> float:
        box = self.purity.boxsize_mpc_h
        return float(self.volumes.sum() / (box ** 3))


def existing_catalog(work: Path, tag: str) -> Optional[Path]:
    d = Path(work) / f"{tag}_rockstar"
    for pat in ("halos*.ascii", "halos*.list", "out_*.list", "out_*.ascii"):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    return None


def field_to_chunk_summaries(
    field: np.ndarray,
    *,
    box: str,
    source: str,
    tag: str,
    grid: ChunkGrid,
    bins: CatalogBins,
    work_dir: str | Path,
    rockstar_binary: str | Path,
    rockstar_cfg: str | Path,
    boxsize_kpc_h: float = 100000.0,
    redshift: float = 0.0,
    dis_norm_kpc_h: float = 6000.0,
    purity_grid: int = 128,
    max_half_width: int = 4,
    reuse_catalog: bool = True,
    keep_gadget: bool = False,
) -> CatalogResult:
    """Run the halo finder on a full periodic box and attribute halos to chunks."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    prev = existing_catalog(work, tag) if reuse_catalog else None
    if prev is not None:
        cat = load_rockstar_ascii(prev)
    else:
        cat = run_rockstar_on_field(
            np.asarray(field), work, tag=tag, boxsize_kpc_h=boxsize_kpc_h,
            redshift=redshift, binary=rockstar_binary, cfg=rockstar_cfg,
        )
    t_halo = time.time() - t0

    t0 = time.time()
    purity = chunk_purity_grid(
        np.asarray(field[0:3]), chunk_grid=grid, grid=int(purity_grid),
        dis_norm_kpc_h=dis_norm_kpc_h, redshift=redshift,
    )
    assign = assign_halos_to_chunks(
        cat.pos, cat.rvir, purity, min_purity=float(bins.min_purity),
        radius_mult=float(bins.radius_mult), max_half_width=int(max_half_width),
    )
    volumes = purity.effective_volume_mpc3(bins.min_purity)
    t_assign = time.time() - t0

    summaries = summarize_catalog(
        cat, assign, bins, volumes, box=box, source=source, chunk_ids=grid.all_ids()
    )
    if not keep_gadget:
        for g in work.glob("*.gadget2"):
            g.unlink(missing_ok=True)
    return CatalogResult(
        summaries, cat, purity, volumes,
        {"halo_finder_s": t_halo, "attribution_s": t_assign}, assign,
    )
