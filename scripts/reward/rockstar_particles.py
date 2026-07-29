#!/usr/bin/env python
"""Run Rockstar with member-particle-ID output and reduce it to per-tile weights.

Experiment 0 needs to know which 64^3 generator tile produced each bound
particle. That information exists exactly -- our GADGET2 ids are the flat
Lagrangian index -- but only if Rockstar is asked to print member ids
(``FULL_PARTICLE_CHUNKS = 1``, ``configs/sr2_baseline/rockstar_particles.cfg``).

The ``.particles`` table is ~6 GB of ASCII for a 512^3 box (78 B/row measured
on the real binary via scripts/reward/verify_member_ids.py). This script streams
it into a compact ``*_tilew.npz`` (a few MB) and **deletes it in the same job**,
so the bulky intermediate never survives the allocation and never approaches
``$HOME``. Rerunning is cheap only if the npz already exists, which is why
``--reuse`` is the default.

Because the frozen halo catalog is the thing every earlier number in this
project rests on, ``--verify-frozen`` re-derives the catalog under the new
config and compares it object-by-object with the existing frozen one. The two
configs differ in one output-only flag, so they must agree; if they do not, that
is a finding, not a nuisance, and the script exits non-zero.

    python scripts/reward/rockstar_particles.py --box set8 --source hr
    python scripts/reward/rockstar_particles.py --box set8 --source base --verify-frozen
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from _common import (  # noqa: E402
    PROJECT_ROOT, add_common_args, banner, bins_of, hr_path, load_reward_config,
    paths, write_json,
)

from cosmo_sr.eval.rockstar import (  # noqa: E402
    load_rockstar_ascii, run_rockstar_on_particles,
)
from cosmo_sr.eval.particles import field_to_particles  # noqa: E402
from cosmo_sr.reward.pipeline import existing_catalog  # noqa: E402
from cosmo_sr.reward.tiles import (  # noqa: E402
    TileGrid, check_member_consistency, direct_full_box_stats,
    member_weights_from_particles, pool_tiles, splice_tiles, stream_member_ids,
    tile_summaries, write_tile_summaries,
)

PARTICLE_CFG = PROJECT_ROOT / "configs" / "sr2_baseline" / "rockstar_particles.cfg"


def tile_grid(cfg) -> TileGrid:
    d = cfg.get("data", {})
    t = cfg.get("tiles", {})
    return TileGrid(
        ng_hr=int(d.get("ng_hr", 512)),
        tile_hr=int(t.get("tile_hr", 64)),
        boxsize_mpc_h=float(d.get("boxsize_mpc_h", 100.0)),
    )


def load_field(cfg, box: str, source: str, base_seed: int) -> np.ndarray:
    """HR from the paired dataset, ``base`` from the frozen SR2 cache."""
    if source == "hr":
        return np.load(hr_path(cfg, box), mmap_mode="r")
    cache = paths.SR2_BASE_CACHE()
    hits = sorted(Path(cache).glob(f"{box}_seed{int(base_seed)}_*.npy"))
    if not hits:
        raise SystemExit(
            f"no frozen SR2 cache for {box} seed {base_seed} under {cache}; "
            "run scripts/slurm/cache_sr2_base.sbatch first"
        )
    return np.load(hits[0], mmap_mode="r")


def catalogs_agree(a, b) -> dict:
    """Object-by-object comparison of two catalogs of the same box."""
    out = {"n_a": int(a.n), "n_b": int(b.n), "same_n": bool(a.n == b.n)}
    if a.n != b.n:
        out["agree"] = False
        return out
    order_a = np.argsort(a.ids)
    order_b = np.argsort(b.ids)
    same_ids = bool(np.array_equal(a.ids[order_a], b.ids[order_b]))
    dm = float(np.max(np.abs(a.mvir[order_a] - b.mvir[order_b]))) if a.n else 0.0
    dx = float(np.max(np.abs(a.pos[order_a] - b.pos[order_b]))) if a.n else 0.0
    dp = int(np.max(np.abs(a.num_p[order_a] - b.num_p[order_b]))) if a.n else 0
    out.update(same_ids=same_ids, max_dmvir=dm, max_dpos_mpc_h=dx, max_dnum_p=dp,
               agree=bool(same_ids and dm == 0.0 and dx == 0.0 and dp == 0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--box", required=True)
    ap.add_argument("--source", default="hr", choices=("hr", "base"))
    ap.add_argument("--tag", default="", help="override the output tag")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--field-npy", default="",
                    help="use this field instead of the cached HR/base one "
                         "(Experiment 1 intervention boxes)")
    ap.add_argument("--splice-hr-tiles", default="",
                    help="comma-separated Lagrangian tile ids to overwrite with "
                         "HR content before halo finding (Experiment 0 credit "
                         "verification). Requires --source base.")
    ap.add_argument("--out-dir", default="",
                    help="override the work directory (default: $REWARD_ROOT/halos/...)")
    ap.add_argument("--reuse", action="store_true", default=True)
    ap.add_argument("--overwrite", dest="reuse", action="store_false")
    ap.add_argument("--verify-frozen", action="store_true",
                    help="compare the catalog against the existing frozen one "
                         "and fail if the member-id config changed it")
    ap.add_argument("--extract-members", default="",
                    help="JSON with {'halo_ids': [...]}; their member particle "
                         "ids are written alongside the tile weights in the same "
                         "streaming pass (Experiment 1 mask construction)")
    ap.add_argument("--keep-particles", action="store_true",
                    help="do NOT delete the ~7 GB ASCII table (debugging only)")
    ap.add_argument("--chunk-rows", type=int, default=8_000_000)
    args = ap.parse_args(argv)

    cfg = load_reward_config(args)
    grid = tile_grid(cfg)
    bins = bins_of(cfg)
    box, source = str(args.box), str(args.source)
    tag = args.tag or source

    work = Path(args.out_dir) if args.out_dir else \
        paths.subdir("halos_particles", f"{box}__{source}__{tag}", create=True)
    work.mkdir(parents=True, exist_ok=True)
    npz = work / f"{box}_{tag}_tilew.npz"
    jsonl = paths.subdir("tile_cache", create=True) / f"{box}__{source}__{tag}.jsonl"

    if args.reuse and npz.is_file() and jsonl.is_file():
        banner(f"{box}/{tag}: tile weights already cached -> {npz}")
        return 0

    t0 = time.time()
    banner(f"{box}/{tag}: Rockstar with member-particle ids")
    field = (np.load(args.field_npy, mmap_mode="r") if args.field_npy
             else load_field(cfg, box, source, args.base_seed))
    if args.splice_hr_tiles:
        if source != "base":
            raise SystemExit("--splice-hr-tiles replaces base content with HR, "
                             "so it requires --source base")
        ids = [int(t) for t in args.splice_hr_tiles.split(",") if t.strip()]
        banner(f"splicing HR into Lagrangian tiles {ids}")
        field = splice_tiles(np.asarray(field), np.load(hr_path(cfg, box), mmap_mode="r"),
                             ids, grid)
        report_splice = ids
    else:
        report_splice = []
    d = cfg.get("data", {})
    particles = field_to_particles(
        np.asarray(field, dtype=np.float32),
        boxsize_kpc_h=float(d.get("boxsize_mpc_h", 100.0)) * 1000.0,
        redshift=float(d.get("redshift", 0.0)),
    )
    del field
    cat = run_rockstar_on_particles(
        particles, work, tag=tag, cfg=PARTICLE_CFG, overwrite=not args.reuse,
    )
    del particles
    t_halo = time.time() - t0
    print(f"    halo finding: {t_halo / 60:.1f} min, {cat.n} objects", flush=True)

    report = {"box": box, "source": source, "tag": tag,
              "halo_finder_min": t_halo / 60.0, "n_objects": int(cat.n),
              "tile_hr": grid.tile_hr, "n_tiles": grid.n_tiles,
              "spliced_hr_tiles": report_splice}

    if args.verify_frozen:
        frozen = existing_catalog(
            paths.subdir("halos", f"{box}__{source}__{source}"), source
        )
        if frozen is None:
            report["verify_frozen"] = "no frozen catalog to compare against"
            print("    !! no frozen catalog found; verification skipped", flush=True)
        else:
            cmp = catalogs_agree(load_rockstar_ascii(frozen), cat)
            report["verify_frozen"] = cmp
            print(f"    frozen-catalog comparison: {cmp}", flush=True)
            if not cmp.get("agree"):
                write_json(work / "particles_report.json", report)
                print("!!! the member-id config CHANGED the catalog. "
                      "FULL_PARTICLE_CHUNKS is supposed to be output-only; "
                      "every downstream number is now in question.", file=sys.stderr)
                return 2

    pfiles = sorted(Path(work / f"{tag}_rockstar").glob("*.particles"))
    if not pfiles:
        raise SystemExit(
            f"Rockstar wrote no .particles table in {work / (tag + '_rockstar')}; "
            f"check that {PARTICLE_CFG.name} has FULL_PARTICLE_CHUNKS = 1"
        )
    size_gb = sum(p.stat().st_size for p in pfiles) / 2 ** 30
    report["particles_gb"] = size_gb
    print(f"    member table: {size_gb:.1f} GB across {len(pfiles)} file(s)", flush=True)

    t0 = time.time()
    weights = member_weights_from_particles(pfiles[0], grid, chunk_rows=args.chunk_rows)
    report["stream_min"] = (time.time() - t0) / 60.0
    report["max_weight_error"] = weights.max_weight_error()
    report["n_objects_with_members"] = len(weights.n_members)
    print(f"    streamed in {report['stream_min']:.1f} min, "
          f"partition-of-unity error {report['max_weight_error']:.2e}", flush=True)
    weights.to_npz(npz)

    consistency = check_member_consistency(cat, weights)
    report["member_consistency"] = consistency
    print(f"    consistency: {consistency['n_with_member_rows']}/"
          f"{consistency['n_catalog_objects']} objects have members, "
          f"{consistency['n_leaf_exact']} leaves exact, "
          f"{consistency['n_with_substructure']} with substructure, "
          f"ok={consistency['ok']}", flush=True)
    if not consistency["ok"]:
        write_json(work / "particles_report.json", report)
        print("!!! member table and ASCII catalog disagree; refusing to write "
              "tile summaries from it.", file=sys.stderr)
        return 3

    if args.extract_members:
        req = json.loads(Path(args.extract_members).read_text())
        ids = [int(h) for h in req.get("halo_ids", [])]
        banner(f"extracting member ids for {len(ids)} named objects")
        members = stream_member_ids(pfiles[0], ids, chunk_rows=args.chunk_rows)
        keys = sorted(members)
        flat = np.concatenate([members[k] for k in keys]) if keys else np.zeros(0, np.int64)
        offs = np.cumsum([0] + [members[k].size for k in keys])
        mpath = work / f"{box}_{tag}_members.npz"
        np.savez_compressed(mpath, halo_id=np.asarray(keys, dtype=np.int64),
                            offset=offs.astype(np.int64), particle_id=flat)
        report["members_npz"] = str(mpath)
        report["n_members_extracted"] = int(flat.size)
        missing_ids = [k for k in keys if members[k].size == 0]
        if missing_ids:
            report["objects_with_no_members"] = missing_ids
            print(f"    !! {len(missing_ids)} requested objects had no member "
                  f"rows: {missing_ids[:5]}", flush=True)
        print(f"    member ids -> {mpath} ({flat.size} particles)", flush=True)

    ts = tile_summaries(cat, weights, bins, grid, box=box, source=source)
    write_tile_summaries(jsonl, ts.values())

    pooled = pool_tiles(ts.values())
    direct = direct_full_box_stats(cat, bins)
    err = {
        "n_host": float(np.max(np.abs(pooled.n_host - direct["n_host"]))),
        "occ_numerator": float(np.max(np.abs(pooled.occ_numerator - direct["occ_numerator"]))),
        "n_sub": float(np.max(np.abs(pooled.n_sub - direct["n_sub"]))),
    }
    report["tile_sum_vs_direct_max_abs_error"] = err
    print(f"    tile-sum vs direct full box: {err}", flush=True)

    if not args.keep_particles:
        for p in pfiles:
            p.unlink(missing_ok=True)
        report["particles_deleted"] = True
    for g in work.glob("*.gadget2"):
        g.unlink(missing_ok=True)

    write_json(work / "particles_report.json", report)
    banner(f"{box}/{tag}: tile weights -> {npz}; summaries -> {jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
