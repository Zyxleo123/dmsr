#!/usr/bin/env python
"""Add a missing density panel to an existing held-out-gather figure ``.npz``.

Why this exists: ``fig1_host_density`` draws each arm from its density array in
the stage-1 npz and falls back to a neutral ground when the key is absent --
which is why the HR panel rendered as a blank sheet of markers: the data job was
run without ``--hr-field``. Rebuilding the whole npz is not an option, because
the tuned arm's 3.2 GB spliced field has since been purged.

The panel must be built the way its neighbours were or it is not comparable.
Stage 1 does NOT project a spatial window: it selects **the particles of the
host's Lagrangian tiles** -- the exact material the operator edited -- and bins
where they landed, over the tile union padded a few Mpc/h. So this reads one
field, selects the same tile ids, bins into the same window at the same
resolution, and merges the result back. Everything else is carried through
untouched, except that legacy key names are normalised to the current scheme.

``--verify-name/--verify-field`` re-derives a panel that is ALREADY in the npz
(frozen, whose field still exists) and compares. That is the only way to know
the new panel was built like the old ones rather than merely looking plausible:
a wrongly binned or transposed panel still renders a convincing picture. On a
mismatch nothing is written.

CPU. Only the requested tiles are read, so this is ~1M particles, not 134M.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for _p in (PROJECT_ROOT / "src", SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cosmo_sr.data.preprocess_srs import disnorm  # noqa: E402
from cosmo_sr.reward.tiles import TileGrid  # noqa: E402

from gather_holdout_figures_data import BOX, project_field  # noqa: E402

LEGACY_SCALARS = {"density_xr": "host_density_xr", "density_zr": "host_density_zr"}


def stem_path(path) -> Path:
    """Strip a trailing .npz/.json without mangling a stem that contains a dot.

    ``Path.with_suffix`` reads ``zoom_rvir1.15_h168880`` as having the suffix
    ``.15_h168880`` and rewrites it to ``zoom_rvir1.npz``, which silently
    scatters a job's outputs under a name nothing downstream looks for.
    """
    path = Path(path)
    if path.name.endswith((".npz", ".json")):
        return path.with_name(path.name.rsplit(".", 1)[0])
    return path


def with_ext(stem: Path, ext: str) -> Path:
    """``stem`` + ``ext``, appended rather than substituted. See stem_path."""
    return stem.with_name(stem.name + ext)


def canonical_key(key: str) -> str:
    """Legacy stage-1 names -> the names the plot job reads."""
    if key in LEGACY_SCALARS:
        return LEGACY_SCALARS[key]
    if key.endswith("_density") and not key.endswith(("_host_density", "_box_density")):
        return f"{key[: -len('_density')]}_host_density"
    return key


def stored_window(z: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """The x/z ranges of the host panel, under either key scheme."""
    for xr, zr in (("host_density_xr", "host_density_zr"), ("density_xr", "density_zr")):
        if xr in z and zr in z:
            return np.asarray(z[xr], dtype=float), np.asarray(z[zr], dtype=float)
    raise SystemExit("npz has no window keys (host_density_xr/density_xr); "
                     "it was not written by gather_holdout_figures_data.py")


def existing_bins(z: Dict[str, np.ndarray], suffix: str, default: int) -> int:
    """Bin count of the panels already in the npz, so the new one matches."""
    for k, v in z.items():
        if k.endswith(suffix) and v.ndim == 2 and v.shape[0] == v.shape[1]:
            return int(v.shape[0])
    return default


def tile_block_positions(block: np.ndarray, tile_id: int, grid: TileGrid,
                         redshift: float = 0.0) -> np.ndarray:
    """Eulerian positions (Mpc/h) of ONE tile, from its (6, n, n, n) block.

    Split out of :func:`tile_positions` because the same block also arrives from
    an export's ``tiles.npz`` rather than from a whole field, and both must
    reconstruct it identically or the panels stop being comparable.
    """
    ng, box_kpc = grid.ng_hr, grid.boxsize_mpc_h * 1000.0
    cell = box_kpc / ng
    q = (np.arange(ng, dtype=np.float64) + 0.5) * cell
    sx, sy, sz = grid.slices(int(tile_id))
    disp = disnorm(np.asarray(block[0:3], dtype=np.float64), z=redshift, undo=True)
    pos = np.empty_like(disp)
    pos[0] = (q[sx].reshape(-1, 1, 1) + disp[0]) % box_kpc
    pos[1] = (q[sy].reshape(1, -1, 1) + disp[1]) % box_kpc
    pos[2] = (q[sz].reshape(1, 1, -1) + disp[2]) % box_kpc
    # float32, exactly as field_to_particles returns them: the panels in the
    # npz were binned from float32 positions, and at a 0.1 Mpc/h bin edge
    # the extra float64 precision moves a handful of particles one bin over.
    return np.ascontiguousarray(pos.reshape(3, -1).T * 1e-3, dtype=np.float32)


def tile_positions(field_path: str, tiles: Sequence[int], grid: TileGrid,
                   redshift: float = 0.0) -> np.ndarray:
    """Eulerian positions (Mpc/h) of the particles of the given Lagrangian tiles.

    The same reconstruction as ``field_to_particles`` -- ``x = (q + Psi) mod L``
    with the catnorm undone -- restricted to each tile's block of the lattice, so
    a handful of tiles costs a few MB instead of materialising all 512^3
    particles as float64. The returned positions are float32 for the same reason
    ``field_to_particles`` casts: the stored panels were binned at that precision.
    """
    field = np.load(field_path, mmap_mode="r")
    if field.ndim != 4 or field.shape[0] < 6:
        raise SystemExit(f"expected a (6, Ng, Ng, Ng) field, got {field.shape}")
    ng = int(field.shape[1])
    if ng != grid.ng_hr:
        raise SystemExit(f"field is {ng}^3 but the tile grid is {grid.ng_hr}^3")
    out: List[np.ndarray] = []
    for t in tiles:
        sx, sy, sz = grid.slices(int(t))
        out.append(tile_block_positions(field[0:3, sx, sy, sz], t, grid, redshift))
    del field
    return np.concatenate(out, axis=0)


def bin_window(pos: np.ndarray, xr: np.ndarray, zr: np.ndarray, bins: int
               ) -> Tuple[np.ndarray, int]:
    """Project to the x-z plane over the stored window; also count what fell out.

    Particles outside the window are dropped by ``histogram2d``, exactly as in
    stage 1 -- material that drifted past the pad is genuinely not in the panel.
    The count is returned so the log says so instead of hiding it.
    """
    h, _, _ = np.histogram2d(pos[:, 0], pos[:, 2], bins=bins,
                             range=[list(xr), list(zr)])
    return h.T, int(pos.shape[0] - h.sum())


def host_tiles(run_dir: Optional[str], host_id: int) -> List[int]:
    if not run_dir:
        return []
    export = json.loads((Path(run_dir) / "export.json").read_text())
    per_host = {h["halo_id"]: h["tiles"] for h in export["per_host"]}
    if host_id not in per_host:
        raise SystemExit(f"host {host_id} not in {run_dir}/export.json")
    return [int(t) for t in per_host[host_id]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True,
                    help="the stage-1 data file (stem or .npz), edited in place "
                         "unless --out is given")
    ap.add_argument("--field", required=True, help="the field .npy to project")
    ap.add_argument("--name", default="hr",
                    help="panel name; writes <name>_host_density")
    ap.add_argument("--tiles", default="",
                    help="comma list of Lagrangian tile ids to select "
                         "(default: the host's tiles from --run-dir)")
    ap.add_argument("--run-dir", default="",
                    help="the holdout export dir holding export.json, to look "
                         "up --host-id's tiles")
    ap.add_argument("--host-id", type=int, default=0)
    ap.add_argument("--ng-hr", type=int, default=512,
                    help="the field's grid size; with --tile-hr this is the "
                         "tile cover the ids in --tiles are numbered in")
    ap.add_argument("--tile-hr", type=int, default=64)
    ap.add_argument("--spatial", action="store_true",
                    help="bin ALL particles in a host-centred window instead of "
                         "selecting tiles -- for npz files whose panels were "
                         "built that way")
    ap.add_argument("--box-slab", type=float, default=0.0,
                    help="--spatial only: also project a full-plane cosmic-web "
                         "slab of this half-thickness in y (0 = window only)")
    ap.add_argument("--host-bins", type=int, default=0,
                    help="override the bin count (default: match the panels "
                         "already in the npz)")
    ap.add_argument("--box-bins", type=int, default=0, help="as --host-bins, box slab")
    ap.add_argument("--verify-name", default="",
                    help="a panel already in the npz to re-derive and compare, "
                         "as a check that the new one is built the same way")
    ap.add_argument("--verify-field", default="",
                    help="the field behind --verify-name")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace <name>_host_density if it is already present")
    ap.add_argument("--out", default="", help="write here instead of in place")
    args = ap.parse_args(argv)

    src = with_ext(stem_path(args.npz), ".npz")
    if not src.is_file():
        raise SystemExit(f"missing stage-1 npz: {src} "
                         f"(run scripts/slurm/gather_figures_data_cpu.sbatch)")
    if not Path(args.field).is_file():
        raise SystemExit(f"missing field: {args.field}")

    with np.load(src) as f:
        data = {k: f[k] for k in f.files}
    key = f"{args.name}_host_density"
    if canonical_key(key) in {canonical_key(k) for k in data} and not args.overwrite:
        print(f"{key} already present in {src.name}; nothing to do "
              f"(pass --overwrite to recompute)")
        return 0

    xr, zr = stored_window(data)
    bins = args.host_bins or existing_bins(data, "_host_density", 0) \
        or existing_bins(data, "_density", 320)
    grid = TileGrid(ng_hr=args.ng_hr, tile_hr=args.tile_hr, boxsize_mpc_h=BOX)

    tiles: List[int] = []
    if not args.spatial:
        tiles = ([int(t) for t in args.tiles.split(",") if t.strip()]
                 or host_tiles(args.run_dir, args.host_id))
        if not tiles:
            raise SystemExit("no tiles: pass --tiles, or --run-dir with "
                             "--host-id, or --spatial for a window-binned npz")

    def build(name: str, path: str) -> np.ndarray:
        print(f"projecting {name} from {path}", flush=True)
        pos = tile_positions(path, tiles, grid)
        h, lost = bin_window(pos, xr, zr, bins)
        print(f"  {len(tiles)} tiles, {pos.shape[0]} particles, {lost} outside "
              f"the window, {h.shape} bins over x{list(xr)} z{list(zr)}", flush=True)
        return h

    # --- is the new panel built the way the old ones were? ------------------
    if args.verify_name and args.verify_field:
        vkey = canonical_key(f"{args.verify_name}_density")
        if vkey not in {canonical_key(k) for k in data}:
            raise SystemExit(f"--verify-name {args.verify_name}: no such panel")
        stored = next(v for k, v in data.items() if canonical_key(k) == vkey)
        if args.spatial:
            raise SystemExit("--verify-* is only implemented for tile selection")
        got = build(f"{args.verify_name} (verify)", args.verify_field)
        if np.array_equal(got, stored):
            print(f"  verify OK: re-derived {args.verify_name} matches the "
                  f"stored panel exactly", flush=True)
        else:
            print(f">>> GATE FAILED: re-derived {args.verify_name} does not "
                  f"match the stored panel", flush=True)
            print(f">>>   max |diff| {np.abs(got - stored).max():.6g}, "
                  f"sums {got.sum():.0f} vs {stored.sum():.0f}, "
                  f"transpose matches: {np.array_equal(got.T, stored)}", flush=True)
            print(">>> the new panel would not be comparable with the ones "
                  "already there, so nothing was written.", flush=True)
            print(">>> exiting 0 so dependents report the same rather than "
                  "stranding.", flush=True)
            return 0

    # --- the panel that was missing -----------------------------------------
    if args.spatial:
        centre = np.asarray(data["host_centre"], dtype=float)
        mid = np.array([0.5 * (xr[0] + xr[1]), 0.5 * (zr[0] + zr[1])])
        if not np.allclose(mid, centre[[0, 2]], atol=1e-3):
            raise SystemExit(f"--spatial: stored window centre {mid} != host "
                             f"centre {centre[[0, 2]]}; the panels in this npz "
                             f"are not host-centred -- drop --spatial")
        proj = project_field(args.field, centre,
                             host_win=float(xr[1] - xr[0]) / 2.0, host_bins=bins,
                             box_slab=(args.box_slab if args.box_slab > 0 else 1.0),
                             box_bins=args.box_bins or existing_bins(
                                 data, "_box_density", 900))
        panel = proj["host"]
    else:
        panel = build(args.name, args.field)
        proj = None

    out_data = {canonical_key(k): v for k, v in data.items()}
    out_data[key] = panel
    out_data.setdefault("host_density_xr", np.asarray(xr))
    out_data.setdefault("host_density_zr", np.asarray(zr))
    if tiles:
        out_data[f"{args.name}_density_tiles"] = np.asarray(tiles, dtype=np.int64)
    if proj is not None and args.box_slab > 0:
        out_data[f"{args.name}_box_density"] = proj["box"]
        out_data.setdefault("box_slab_mpc", np.array([args.box_slab]))
    print(f"  {key}: {panel.shape}, {float(panel.sum()):.6g} particles, "
          f"max {float(panel.max()):.0f} per cell", flush=True)

    dst = with_ext(stem_path(args.out), ".npz") if args.out else src
    # savez appends ".npz" to a name that lacks it, so the temp must already end
    # in .npz or os.replace moves a file that was never written.
    tmp = with_ext(dst, ".tmp.npz")
    np.savez_compressed(tmp, **out_data)
    os.replace(tmp, dst)
    print(f"wrote {dst} ({len(out_data)} arrays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
