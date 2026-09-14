#!/usr/bin/env python
"""Rebuild the three host-density panels zoomed to R_vir, at full tile resolution.

The stage-1 panels are 320 bins over the 33 Mpc/h tile union -- 0.10 Mpc/h a
pixel, so the host's R_vir spans ~30 of them. Cropping those arrays to the
circle would enlarge a 30x30 thumbnail. The material is not lost, though: an
export's ``tiles.npz`` holds every spliced tile for all three arms at the full
64^3 (``out`` = tuned, ``frozen`` = control, ``hr`` = truth), so the panels can
be rebinned in a small window as finely as the particles allow. That also makes
the tuned panel reachable at all -- its 3.2 GB spliced field has been purged,
but its tiles have not.

Written in the SAME key scheme as stage 1, so the existing plot job renders it
with no new plotting code: point ``gather_holdout_figures_plot.py`` at this stem
with a matching ``--zoom-rvir``.

Two things differ from stage 1, both deliberate and both reported in the log:

* a **depth cut**. Stage 1 projects the full 25 Mpc/h Lagrangian column of the
  tiles, which at 5 R_vir the host dominates anyway. Zoomed into the circle that
  column is mostly foreground and background, and it washes out exactly the
  substructure the figure is about, so the default slab is the window's own
  half-width (``--depth-rvir 0`` restores the full column).
* **finer bins**, from ``--bins``.

Before it writes anything it re-derives the stage-1 frozen and tuned panels from
these tiles, in stage-1's window with no depth cut, and requires an exact match.
That is what proves the tiles reconstruct to the same particles as the fields
did, and that ``out`` is the arm stage 1 called ``self``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for _p in (PROJECT_ROOT / "src", SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cosmo_sr.reward.tiles import TileGrid  # noqa: E402

from gather_holdout_add_density import (  # noqa: E402
    BOX, bin_window, canonical_key, existing_bins, host_tiles, stem_path,
    stored_window, tile_block_positions, with_ext,
)

# panel name in the npz  <-  array name in tiles.npz
ARMS = {"self": "out", "frozen": "frozen", "hr": "hr"}


def arm_positions(tz, arm_key: str, rows: List[int], tiles: List[int],
                  grid: TileGrid) -> np.ndarray:
    """Positions of the host's tiles for one arm, from the export's tiles.npz."""
    blocks = tz[arm_key]
    return np.concatenate([tile_block_positions(blocks[r], t, grid)
                           for r, t in zip(rows, tiles)], axis=0)


def depth_mask(pos: np.ndarray, y_centre: float, half: float) -> np.ndarray:
    """Periodic |dy| <= half, or everything when half <= 0."""
    if half <= 0:
        return np.ones(pos.shape[0], dtype=bool)
    dy = np.abs(pos[:, 1].astype(np.float64) - y_centre)
    return np.minimum(dy, BOX - dy) <= half


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="the holdout export dir (tiles.npz + export.json)")
    ap.add_argument("--source-npz", required=True,
                    help="the stage-1 data file: the host centre, R_vir, the "
                         "subhalo overlays, and the panels used to verify")
    ap.add_argument("--host-id", type=int, default=168880)
    ap.add_argument("--zoom-rvir", type=float, default=1.15,
                    help="window half-width in R_vir; 1.15 puts the circle just "
                         "inside the panel")
    ap.add_argument("--depth-rvir", type=float, default=0.0,
                    help="slab half-thickness in R_vir along the projection "
                         "axis; 0 means take --zoom-rvir, <0 means no cut")
    ap.add_argument("--bins", type=int, default=256)
    ap.add_argument("--ng-hr", type=int, default=512)
    ap.add_argument("--tile-hr", type=int, default=64)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the stage-1 re-derivation (not recommended)")
    ap.add_argument("--out", required=True, help="output stem (.npz + .json)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    tiles_npz = run_dir / "tiles.npz"
    src_stem = stem_path(args.source_npz)
    src = with_ext(src_stem, ".npz")
    for f in (tiles_npz, src, run_dir / "export.json"):
        if not f.is_file():
            raise SystemExit(f"missing input: {f}")

    with np.load(src) as f:
        source = {canonical_key(k): f[k] for k in f.files}
    centre = np.asarray(source["host_centre"], dtype=float)
    rvir = float(source["host_rvir_mpc"][0])
    grid = TileGrid(ng_hr=args.ng_hr, tile_hr=args.tile_hr, boxsize_mpc_h=BOX)

    tiles = host_tiles(str(run_dir), args.host_id)
    tz = np.load(tiles_npz)
    all_tiles = [int(t) for t in tz["tiles"]]
    missing = [t for t in tiles if t not in all_tiles]
    if missing:
        raise SystemExit(f"tiles {missing} of host {args.host_id} are not in "
                         f"{tiles_npz}")
    rows = [all_tiles.index(t) for t in tiles]
    print(f"host {args.host_id}: tiles {tiles} = rows {rows} of "
          f"{len(all_tiles)} exported, R_vir {rvir:.3f} Mpc/h at {centre}",
          flush=True)

    pos = {}
    for name, key in ARMS.items():
        pos[name] = arm_positions(tz, key, rows, tiles, grid)
        print(f"  {name:>6} ({key}): {pos[name].shape[0]} particles", flush=True)

    # --- do these tiles reproduce the panels stage 1 built from the fields? ---
    if not args.no_verify:
        sxr, szr = stored_window(source)
        sbins = existing_bins(source, "_host_density", 320)
        ok = True
        for name in ("frozen", "self"):
            key = f"{name}_host_density"
            if key not in source:
                continue
            got, lost = bin_window(pos[name], sxr, szr, sbins)
            same = np.array_equal(got, source[key])
            ok &= same
            print(f"  verify {name}: {'OK, exact match' if same else 'MISMATCH'}"
                  f" (max |diff| {np.abs(got - source[key]).max():.6g}, "
                  f"{lost} outside stage-1's window)", flush=True)
        if not ok:
            print(">>> GATE FAILED: the tiles do not reproduce the stage-1 "
                  "panels, so a zoom built from them would not be the same "
                  "picture at higher resolution. Nothing written.", flush=True)
            print(">>> exiting 0 so dependents report the same rather than "
                  "stranding.", flush=True)
            return 0

    # --- the zoom ------------------------------------------------------------
    half = args.zoom_rvir * rvir
    depth = (args.zoom_rvir if args.depth_rvir == 0 else args.depth_rvir) * rvir
    xr = np.array([centre[0] - half, centre[0] + half])
    zr = np.array([centre[2] - half, centre[2] + half])
    pix_kpc = 2000.0 * half / args.bins
    print(f"\nzoom: +/-{half:.3f} Mpc/h ({args.zoom_rvir:g} R_vir), "
          f"{args.bins} bins = {pix_kpc:.1f} kpc/h per pixel; depth "
          f"{'no cut' if depth <= 0 else f'+/-{depth:.3f} Mpc/h'}", flush=True)

    out_data = {k: v for k, v in source.items()
                if not k.endswith(("_host_density", "_box_density"))
                and k not in ("host_density_xr", "host_density_zr")}
    for name in ARMS:
        p = pos[name][depth_mask(pos[name], centre[1], depth)]
        h, lost = bin_window(p, xr, zr, args.bins)
        out_data[f"{name}_host_density"] = h
        occ = h[h > 0]
        print(f"  {name:>6}: {p.shape[0]} in the slab, {int(h.sum())} in the "
              f"window, {h.shape} bins, {100.0 * occ.size / h.size:.1f}% of "
              f"pixels occupied, median {np.median(occ) if occ.size else 0:.0f}"
              f" max {h.max():.0f}", flush=True)
        del lost
    out_data["host_density_xr"] = xr
    out_data["host_density_zr"] = zr
    out_data["zoom_rvir"] = np.array([args.zoom_rvir])
    out_data["zoom_depth_mpc"] = np.array([depth])
    out_data["zoom_tiles"] = np.asarray(tiles, dtype=np.int64)

    stem = stem_path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    tmp = with_ext(stem, ".tmp.npz")
    np.savez_compressed(tmp, **out_data)
    tmp.replace(with_ext(stem, ".npz"))
    # the plot job reads <stem>.json for the titles; it is unchanged by the zoom
    shutil.copyfile(with_ext(src_stem, ".json"), with_ext(stem, ".json"))
    print(f"\nwrote {with_ext(stem, '.npz')} ({len(out_data)} arrays) "
          f"and {with_ext(stem, '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
