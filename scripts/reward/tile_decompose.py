#!/usr/bin/env python
"""Experiment 0 deliverable: direct full-box statistics vs tile-summed, with error.

Reads the per-tile summaries cached by ``rockstar_particles.py`` and the frozen
ASCII catalogs, and writes the one table the experiment is defined by:

    H_b = sum_j H_jb ,   S_b = sum_j S_jb ,   O_b = S_b / H_b

for every host-mass bin, alongside the numerical discrepancy. Also reports the
number that motivated the whole redesign: **attribution retention per host
bin**, which the Eulerian purity mask drove to 0.37 / 0.30 / 0.22 / 0.12 / 0.00
from 1e12 to 1e14 Msun/h. Fractional member-id attribution rejects nothing, so
retention is 1.000 everywhere by construction -- the script measures it rather
than asserting it, because a retention below 1 would mean the `.particles` table
and the catalog disagree.

    python scripts/reward/tile_decompose.py --boxes set8,set9 --sources hr,base
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from _common import (  # noqa: E402
    add_common_args, banner, bins_of, load_reward_config, paths, write_json,
)

from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.reward.geometry import ChunkGrid, assign_halos_to_chunks  # noqa: E402
from cosmo_sr.reward.pipeline import existing_catalog  # noqa: E402
from cosmo_sr.reward.tiles import (  # noqa: E402
    direct_full_box_stats, pool_tiles, read_tile_summaries,
)


def host_labels(bins) -> List[str]:
    return [f"{e:.2e}" for e in bins.host_mass_edges[:-1]]


def frozen_catalog(box: str, source: str):
    p = existing_catalog(paths.subdir("halos", f"{box}__{source}__{source}"), source)
    if p is None:
        p = existing_catalog(
            paths.subdir("halos_particles", f"{box}__{source}__{source}"), source
        )
    return load_rockstar_ascii(p) if p is not None else None


def masked_retention(cat, bins, cfg) -> np.ndarray:
    """Hosts surviving the *old* Eulerian purity mask, per host bin.

    Kept as the comparison column: it is the quantity that forced chunk_hr from
    128 to 256, and the point of Experiment 0 is that it no longer applies.
    Returns NaN when the purity grid for this box was never cached, so a missing
    comparison never masquerades as a measured zero.
    """
    g = cfg.get("geometry", {})
    cg = ChunkGrid(ng_hr=int(cfg["data"]["ng_hr"]),
                   chunk_hr=int(g.get("chunk_hr", 256)),
                   boxsize_mpc_h=float(cfg["data"]["boxsize_mpc_h"]))
    from cosmo_sr.reward.geometry import PurityGrid

    hits = sorted(paths.CATALOG_CACHE().glob("*_purity.npz"))
    match = [h for h in hits if h.name.startswith(f"{cat_box(cat)}__")] if hits else []
    if not match:
        return np.full(bins.n_host_bins, np.nan)
    purity = PurityGrid.from_npz(match[0])
    assign = assign_halos_to_chunks(
        cat.pos, cat.rvir, purity, min_purity=float(g.get("min_purity", 0.8)),
        radius_mult=float(g.get("radius_mult", 1.0)),
        max_half_width=int(g.get("max_half_width", 4)),
    )
    hosts = cat.parent_ids < 0
    edges = np.asarray(bins.host_mass_edges)
    tot, _ = np.histogram(cat.mvir[hosts], bins=edges)
    kept, _ = np.histogram(cat.mvir[hosts & (assign >= 0)], bins=edges)
    _ = cg
    return np.divide(kept, tot, out=np.full(len(tot), np.nan, dtype=float),
                     where=tot > 0)


def cat_box(cat) -> str:
    """Box name recovered from the catalog path (…/halos/<box>__<src>__<tag>/…)."""
    for part in Path(cat.path).parts:
        if "__" in part:
            return part.split("__")[0]
    return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--boxes", default="set8,set9")
    ap.add_argument("--sources", default="hr,base")
    ap.add_argument("--out", default="", help="JSON report path")
    args = ap.parse_args(argv)

    cfg = load_reward_config(args)
    bins = bins_of(cfg)
    boxes = [b.strip() for b in args.boxes.split(",") if b.strip()]
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    labels = host_labels(bins)

    rows: List[Dict] = []
    for box in boxes:
        for src in sources:
            jl = paths.subdir("tile_cache") / f"{box}__{src}__{src}.jsonl"
            if not jl.is_file():
                print(f"!! missing tile summaries for {box}/{src}: {jl}", flush=True)
                continue
            ts = read_tile_summaries(jl)
            pooled = pool_tiles(ts)
            cat = frozen_catalog(box, src)
            if cat is None:
                print(f"!! no catalog for {box}/{src}", flush=True)
                continue
            direct = direct_full_box_stats(cat, bins)

            occ_tile = pooled.occupation()
            occ_direct = direct["occupation"]
            rows.append({
                "box": box, "source": src, "n_tiles": len(ts),
                "H_direct": direct["n_host"].tolist(),
                "H_tiles": pooled.n_host.tolist(),
                "S_direct": direct["occ_numerator"].tolist(),
                "S_tiles": pooled.occ_numerator.tolist(),
                "N_direct": direct["n_sub"].tolist(),
                "N_tiles": pooled.n_sub.tolist(),
                "O_direct": occ_direct.tolist(),
                "O_tiles": occ_tile.tolist(),
                "max_abs_err_H": float(np.max(np.abs(pooled.n_host - direct["n_host"]))),
                "max_abs_err_S": float(np.max(np.abs(pooled.occ_numerator
                                                     - direct["occ_numerator"]))),
                "max_abs_err_N": float(np.max(np.abs(pooled.n_sub - direct["n_sub"]))),
                "max_abs_err_O": float(np.nanmax(np.abs(occ_tile - occ_direct))),
                "volume_mpc3": float(pooled.volume_mpc3),
                "retention_member_id": (
                    np.divide(pooled.n_host, direct["n_host"],
                              out=np.full(bins.n_host_bins, np.nan),
                              where=direct["n_host"] > 0).tolist()
                ),
                "retention_purity_mask": masked_retention(cat, bins, cfg).tolist(),
            })

    if not rows:
        raise SystemExit("no tile summaries found; run rockstar_particles.py first")

    # ---------------------------------------------------------------- table
    banner("Experiment 0: direct full-box vs tile-summed statistics")
    for r in rows:
        print(f"\n{r['box']} / {r['source']}   ({r['n_tiles']} tiles, "
              f"V = {r['volume_mpc3']:.4g} (Mpc/h)^3)")
        head = (f"{'host bin':>10s} {'H direct':>10s} {'H tiles':>14s} "
                f"{'S direct':>10s} {'S tiles':>14s} "
                f"{'O direct':>9s} {'O tiles':>12s} {'|dO|':>10s}")
        print(head)
        print("-" * len(head))
        for b, lab in enumerate(labels):
            do = abs(r["O_tiles"][b] - r["O_direct"][b]) \
                if np.isfinite(r["O_tiles"][b]) and np.isfinite(r["O_direct"][b]) else np.nan
            print(f"{lab:>10s} {r['H_direct'][b]:10.0f} {r['H_tiles'][b]:14.6f} "
                  f"{r['S_direct'][b]:10.0f} {r['S_tiles'][b]:14.6f} "
                  f"{r['O_direct'][b]:9.3f} {r['O_tiles'][b]:12.6f} {do:10.2e}")
        print(f"  max |err|:  H {r['max_abs_err_H']:.3e}   "
              f"S {r['max_abs_err_S']:.3e}   N {r['max_abs_err_N']:.3e}   "
              f"O {r['max_abs_err_O']:.3e}")

    banner("Attribution retention per host bin: member ids vs the purity mask")
    head = f"{'box/src':>12s} {'method':>14s} " + " ".join(f"{l:>9s}" for l in labels)
    print(head)
    print("-" * len(head))
    for r in rows:
        key = f"{r['box']}/{r['source']}"
        print(f"{key:>12s} {'member ids':>14s} "
              + " ".join(f"{v:9.3f}" for v in r["retention_member_id"]))
        print(f"{'':>12s} {'purity mask':>14s} "
              + " ".join(f"{v:9.3f}" for v in r["retention_purity_mask"]))

    worst = max(max(r["max_abs_err_H"], r["max_abs_err_S"], r["max_abs_err_N"])
                for r in rows)
    verdict = "exact" if worst < 1e-6 else "MISMATCH"
    print(f"\nworst absolute reconstruction error over all boxes/sources: "
          f"{worst:.3e}  ->  {verdict}")

    out = Path(args.out) if args.out else \
        paths.AUDITS("tile_decomposition", create=True) / "exp0_table.json"
    write_json(out, {"rows": rows, "host_bin_labels": labels,
                     "worst_abs_error": worst, "verdict": verdict})
    print(f"\nwritten -> {out}")
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
