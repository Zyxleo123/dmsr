#!/usr/bin/env python
"""CPU stage: full-box halo finding -> chunk-level catalog summaries.

One field in, one compact JSONL of per-chunk counts out. Rockstar always runs on
the **complete periodic box** exactly as the frozen evaluation pipeline does; the
only new step is attributing each halo back to the Lagrangian chunk that produced
it (see ``cosmo_sr.reward.geometry`` for the core mask).

Summaries are cached so later stages never re-parse a 160 MB ASCII catalog, and
never re-run the finder for a counterfactual.

    python scripts/reward/catalog_summaries.py --box set12 --source hr
    python scripts/reward/catalog_summaries.py --box set12 --source base
    python scripts/reward/catalog_summaries.py --field /path/cand.npy \
        --box set12 --source candidate --tag ens003
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from _common import (add_common_args, banner, bins_of, chunk_grid, hr_path,
                     load_freeze, load_reward_config, write_json, PROJECT_ROOT)

from cosmo_sr.eval.rockstar import (default_rockstar_binary, load_rockstar_ascii,
                                    run_rockstar_on_field)
from cosmo_sr.reward import paths
from cosmo_sr.reward.base import find_base_field
from cosmo_sr.reward.catalog import (summarize_catalog, summarize_full_box,
                                     write_summaries)
from cosmo_sr.reward.geometry import assign_halos_to_chunks, chunk_purity_grid


def _find_catalog(d: Path):
    for pat in ("halos*.ascii", "halos*.list", "out_*.list", "out_*.ascii"):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    return None


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--box", required=True)
    ap.add_argument("--source", required=True, choices=["hr", "base", "candidate"])
    ap.add_argument("--field", default=None,
                    help="field .npy; defaults to the HR or cached base field")
    ap.add_argument("--tag", default=None, help="candidate identifier")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--work", default=None, help="scratch dir for GADGET2 + Rockstar")
    ap.add_argument("--out", default=None, help="summary cache dir")
    ap.add_argument("--catalog", default=None, help="reuse an existing ASCII catalog")
    ap.add_argument("--keep-gadget", action="store_true",
                    help="keep the 3.7 GB GADGET2 dump (off by default)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    freeze = load_freeze(cfg)
    grid = chunk_grid(cfg)
    bins = bins_of(cfg)
    geo = cfg.get("geometry", {})
    dcfg = cfg["data"]

    tag = args.tag or args.source
    cache = Path(args.out) if args.out else paths.CATALOG_CACHE(create=True)
    cache.mkdir(parents=True, exist_ok=True)
    stem = f"{args.box}__{args.source}__{tag}"
    summary_path = cache / f"{stem}.jsonl"
    meta_path = cache / f"{stem}.json"
    # Both files must exist: a cache written before the full-box summary existed
    # would otherwise silently keep the reward fit on chunk-pooled vectors.
    if summary_path.is_file() and (cache / f"{stem}__fullbox.jsonl").is_file() \
            and not args.overwrite:
        print(f"cached -> {summary_path}", flush=True)
        return

    if args.field:
        field_path = Path(args.field)
    elif args.source == "hr":
        field_path = hr_path(cfg, args.box)
    elif args.source == "base":
        field_path = find_base_field(args.box, args.base_seed)
        if field_path is None:
            raise SystemExit(
                f"no cached SR2 base for {args.box}; run slurm/cache_sr2_base.sbatch"
            )
    else:
        raise SystemExit("--field is required for --source candidate")

    work = Path(args.work) if args.work else paths.subdir("halos", stem, create=True)
    work.mkdir(parents=True, exist_ok=True)
    banner(f"{stem}: field={field_path.name}")

    field = np.load(field_path, mmap_mode="r")

    # ---- halo finding (full periodic box, frozen config) --------------------
    t0 = time.time()
    if args.catalog:
        cat = load_rockstar_ascii(args.catalog)
        cat_path = Path(args.catalog)
    else:
        existing = _find_catalog(work / f"{tag}_rockstar")
        if existing is not None and not args.overwrite:
            cat = load_rockstar_ascii(existing)
            cat_path = existing
        else:
            binary = PROJECT_ROOT / freeze["rockstar"]["binary"]
            cfg_path = PROJECT_ROOT / freeze["rockstar"]["config"]
            if not Path(binary).is_file():
                raise SystemExit(f"Rockstar binary missing: {binary}")
            cat = run_rockstar_on_field(
                np.asarray(field), work, tag=tag,
                boxsize_kpc_h=float(freeze["cosmology_sim"]["boxsize_kpc_h"]),
                redshift=float(freeze.get("redshift", 0.0)),
                binary=binary, cfg=cfg_path, overwrite=args.overwrite,
            )
            cat_path = Path(cat.path)
    t_halo = time.time() - t0
    print(f"  halos: {cat.n} ({t_halo:.0f}s)", flush=True)

    # ---- chunk attribution ---------------------------------------------------
    t0 = time.time()
    purity = chunk_purity_grid(
        np.asarray(field[0:3]), chunk_grid=grid,
        grid=int(geo.get("purity_grid", 128)),
        dis_norm_kpc_h=float(dcfg.get("dis_norm_kpc_h", 6000.0)),
        redshift=float(dcfg.get("redshift", 0.0)),
    )
    assign = assign_halos_to_chunks(
        cat.pos, cat.rvir, purity,
        min_purity=float(bins.min_purity),
        radius_mult=float(bins.radius_mult),
        max_half_width=int(geo.get("max_half_width", 4)),
    )
    volumes = purity.effective_volume_mpc3(bins.min_purity)
    t_assign = time.time() - t0

    summaries = summarize_catalog(
        cat, assign, bins, volumes, box=args.box, source=args.source,
        chunk_ids=grid.all_ids(),
    )
    write_summaries(summary_path, [summaries[c] for c in sorted(summaries)])
    # The whole-box statistic, unmasked, in its own file: this is what the reward
    # model is fitted on and what Gate B scores. The per-chunk file above stays
    # the credit-assignment statistic.
    full = summarize_full_box(cat, bins, float(grid.boxsize_mpc_h ** 3),
                              box=args.box, source=args.source)
    full_path = cache / f"{stem}__fullbox.jsonl"
    write_summaries(full_path, [full])

    n_assigned = int(np.count_nonzero(assign >= 0))
    core_frac = float(volumes.sum() / (grid.boxsize_mpc_h ** 3))
    write_json(meta_path, {
        "box": args.box, "source": args.source, "tag": tag,
        "field": str(field_path), "catalog": str(cat_path),
        "full_box_summary": str(full_path),
        "n_hosts_full_box": int(full.n_host_total),
        "n_subs_full_box": int(full.n_sub_total),
        "n_halos": int(cat.n),
        "n_hosts": int(cat.hosts().n), "n_subs": int(cat.subhalos().n),
        "n_assigned": n_assigned,
        "assigned_fraction": n_assigned / max(cat.n, 1),
        "core_volume_fraction": core_frac,
        "chunk_volumes_mpc3": volumes.tolist(),
        "chunk_hr": grid.chunk_hr, "n_chunks": grid.n_chunks,
        "min_purity": bins.min_purity, "radius_mult": bins.radius_mult,
        "bins": bins.to_dict(),
        "seconds": {"halo_finder": t_halo, "attribution": t_assign},
    })
    purity.to_npz(cache / f"{stem}_purity.npz")

    if not args.keep_gadget:
        for g in work.glob("*.gadget2"):
            g.unlink(missing_ok=True)

    print(
        f"  assigned {n_assigned}/{cat.n} halos ({100 * n_assigned / max(cat.n, 1):.1f}%), "
        f"core volume {100 * core_frac:.1f}% of the box ({t_assign:.0f}s)",
        flush=True,
    )
    print(f"  -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
