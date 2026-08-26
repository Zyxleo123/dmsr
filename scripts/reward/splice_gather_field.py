#!/usr/bin/env python
"""Splice a gather fine-tune's tuned tiles into the frozen SR2 box. CPU, minutes.

Rockstar runs on a **complete periodic box**, and the gather fine-tune supervised
four tiles of one cluster. This builds the box that isolates that change: the
cached frozen SR2 field everywhere, with the fine-tuned generator's output in the
four trained tiles. Every halo difference against ``set8__base__base`` is then
attributable to those tiles -- no second full-box generation, no GPU.

What this measures, and what it cannot
--------------------------------------
It answers the only question a field statistic cannot: **are the new clumps
bound?** Rockstar links in 6-D and will simply not report an over-hot or diffuse
overdensity as a halo.

It does **not** measure the collateral damage. The fine-tuned weights change what
the generator does at *every* site, and the last run left local peak structure
outside the supervised windows at 0.52-0.57 of frozen. This box keeps the frozen
field there, so that damage is invisible here by construction. Only a whole-box
regeneration shows it, and the host mass function gate needs that run, not this
one.

The splice edge
---------------
``splice_tiles`` is hard-edged, which is in-distribution for a mosaic of tiles
from *one* generator (``reward/tiles.py``). Here two different generators meet at
the tile faces, and the tuned field differs from frozen by rms 0.13 in normalised
displacement, so a face is a real discontinuity. ``compare_gather_catalog.py``
therefore reports halo changes against distance from the host, so a boundary
artifact is visible as a ring rather than being read as substructure.

    python scripts/reward/splice_gather_field.py --run-dir <.../set8_h271800_fine_anchored>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import banner, write_json  # noqa: E402

from cosmo_sr.reward import paths  # noqa: E402
from cosmo_sr.reward.base import find_base_field  # noqa: E402
from cosmo_sr.reward.tiles import TileGrid, splice_tiles  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True,
                    help="a host_gather run directory (holds tiles.npz)")
    ap.add_argument("--box", default="set8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="",
                    help="catalog tag; default gather_<run dir name>")
    ap.add_argument("--which", default="out", choices=("out", "frozen", "hr"),
                    help="which tiles to splice in. 'frozen' rebuilds the "
                         "control and 'hr' the ceiling, on the SAME edges")
    ap.add_argument("--out", default="")
    ap.add_argument("--force", action="store_true",
                    help="rebuild the field even if one is already on disk. "
                         "Without it an existing field of the right shape is "
                         "reused and only the sidecar is rewritten.")
    ap.add_argument("--mismatch-max", type=float, default=0.01,
                    help="reject the cached base if its own tiles differ from "
                         "the run's frozen reference by more than this rms")
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    z = np.load(run / "tiles.npz")
    tiles = [int(t) for t in z["tiles"]]
    donor_tiles = z[args.which]
    banner(f"splice {args.which} tiles {tiles} of {run.name} into frozen {args.box}")

    base_path = find_base_field(args.box, seed=int(args.seed))
    if base_path is None:
        raise SystemExit(
            f"no cached frozen SR2 box for {args.box} seed {args.seed} under "
            f"{paths.SR2_BASE_CACHE()}; produce it with "
            "scripts/reward/cache_sr2_base.py")
    base = np.load(str(base_path), mmap_mode="r")
    ng = int(base.shape[-1])
    grid = TileGrid(ng_hr=ng, tile_hr=int(donor_tiles.shape[-1]))
    n = ng // grid.tile_hr

    # The run generated its own frozen tiles; the cache generated the box. They
    # must be the same field, or the splice mixes two realisations. Float
    # nondeterminism between GPUs shows up here at ~1e-4 rms and is fine; a
    # different seed or a different assembly would not be.
    checks = []
    for i, t in enumerate(tiles):
        sx, sy, sz = grid.slices(t)
        cached = np.asarray(base[:, sx, sy, sz], dtype=np.float64)
        d = cached - z["frozen"][i].astype(np.float64)
        checks.append({"tile": t, "rms": float(d.std()), "max": float(np.abs(d).max()),
                       "field_rms": float(cached.std())})
        print(f"  tile {t:4d}: cached vs run-frozen rms {d.std():.2e} "
              f"max {np.abs(d).max():.2e}")
    worst = max(c["rms"] for c in checks)
    if worst > float(args.mismatch_max):
        raise SystemExit(
            f"cached base field disagrees with the run's frozen tiles at rms "
            f"{worst:.3e} > {args.mismatch_max}. The splice would mix two "
            f"realisations; regenerate the cache or the run with one seed.")

    tag = args.tag or f"gather_{run.name}"
    out = Path(args.out) if args.out else (
        paths.subdir("flow_rockstar", "fields", create=True)
        / f"{args.box}__{tag}__seed{args.seed}.npy")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resume. The 3.2 GiB field is the expensive artefact and it is written
    # atomically enough to trust on shape: on 2026-08-25 a node-level kill took
    # three splices out in the EPILOGUE, after every field was complete on disk,
    # and without this the only way back was to redo all of it. A field of the
    # wrong shape is a different tiling and is rebuilt, not reused.
    reused = False
    if out.is_file() and not args.force:
        existing = np.load(str(out), mmap_mode="r")
        if existing.shape == base.shape and existing.dtype == base.dtype:
            print(f"  reusing the field already at {out} "
                  f"({existing.shape}) -- pass --force to rebuild")
            field, reused = existing, True
        else:
            print(f"  {out} has shape {existing.shape}, expected {base.shape}"
                  f" -- rebuilding")
        del existing

    if not reused:
        # Assemble. A full copy of a (6, 512^3) float32 box is 3.2 GiB, and
        # splice_tiles makes one; the job asks for headroom accordingly.
        donor = np.array(base, dtype=np.float32)
        for i, t in enumerate(tiles):
            sx, sy, sz = grid.slices(t)
            donor[:, sx, sy, sz] = donor_tiles[i]
        field = splice_tiles(base, donor, tiles, grid)
        del donor
        np.save(str(out), field)

    # Chunked along the first axis. `abs(field - base).max()` over two full
    # boxes allocates two MORE 3.2 GiB temporaries on top of the field and the
    # base's resident pages; one channel at a time is 0.5 GiB and gives the
    # identical number. It also means this line works when `field` is the
    # memmap of a reused file rather than an in-memory array.
    changed = max(float(np.abs(np.asarray(field[c], dtype=np.float32)
                               - np.asarray(base[c], dtype=np.float32)).max())
                  for c in range(base.shape[0]))
    meta = {
        "ok": True, "run_dir": str(run), "box": args.box, "seed": int(args.seed),
        "which": args.which, "tag": tag, "tiles": tiles,
        "base_field": str(base_path), "field": str(out),
        "cache_consistency": checks, "worst_rms": worst,
        "max_abs_change": changed,
        # Whether this run assembled the field or found it already on disk.
        # A gate read months later should not have to guess.
        "field_reused": bool(reused),
        "n_tiles": len(tiles),
        "spliced_volume_fraction": len(tiles) / float(n ** 3),
    }
    write_json(out.with_suffix(".json"), meta)
    print(f"  {'reused' if reused else 'spliced'} {len(tiles)} of {n ** 3} tiles "
          f"({100.0 * len(tiles) / n ** 3:.2f}% of the box), "
          f"max |change| {changed:.3f}")
    print(f"  wrote {out}")
    print(f"  next: sbatch scripts/slurm/flow_rockstar_catalog_cpu.sbatch "
          f"BOX={args.box} TAG={tag} FIELD_OUT={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
