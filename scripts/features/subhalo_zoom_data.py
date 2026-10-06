#!/usr/bin/env python
"""One panel per subhalo, in a slab as thin as the subhalo -- not a host column.

The problem this answers
------------------------
The host figures project the full ~25 Mpc/h Lagrangian column of the host's
tiles. A 200-particle subhalo is ~0.1 Mpc/h across, so that column stacks
roughly 250 subhalo-diameters of unrelated material on top of it and the
contrast the eye is offered is ``(subhalo + column) / column`` -- a percent-level
tint *even when the subhalo is perfectly real*. "I cannot see it in the figure"
therefore says nothing about whether Rockstar's centres are genuine.

So this script stops projecting the host and starts projecting the subhalo: for
each reported centre it takes a window a few subhalo radii wide and a slab only
as deep as the subhalo is, and it does that for the same window in all three
fields -- tuned, frozen and HR -- so the panels are directly comparable.

Every panel carries numbers, because a picture cannot settle this either
-----------------------------------------------------------------------
For each subhalo it records, via :mod:`cosmo_sr.features.subhalo_zoom`:

* the **3-D** overdensity ``rho(<r) / rho(3r..5r)`` -- no projection at all;
* the **depth ladder**: the same excess after projecting slabs of ``1, 2, 5, 10``
  subhalo radii and finally the whole column.

The ladder is the experiment. A real object buried by projection starts high and
decays towards 1 as the slab opens. A phantom sits at 1 at *every* depth,
including the thinnest, where no projection is left to blame. Each rung also
carries the excess in units of its own Poisson noise, because a 5% ratio is
invisible at snr 1 and a genuine detection at snr 40, and the two questions --
"is it there" and "would I see it" -- have different answers here.

Where the particles come from
-----------------------------
By default, the export's ``tiles.npz``: the host's Lagrangian tiles at full 64^3
resolution, for all three arms, which is what the existing host figures bin and
is the only surviving source for arms whose spliced field has been purged. That
set omits material that drifted in from *other* tiles, which biases the local
background low and the contrast high -- identically in all three panels, but
still an upper bound. Pass ``--field-out/--field-frozen/--field-hr`` to bin the
complete field instead; the choice is recorded as ``particle_source`` in the
output and should be quoted with any number taken from it.

    python scripts/features/subhalo_zoom_data.py \\
        --run-dir $MG/all_blocks_self/holdout_set9 \\
        --box set9 --tag mgho_all_blocks_self_set9 --host-id 168880 \\
        --out $MG/all_blocks_self/holdout_set9/figures/subzoom_self
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for _p in (PROJECT_ROOT / "src", SCRIPT_DIR, PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cosmo_sr.eval.rockstar import HaloCatalog  # noqa: E402
from cosmo_sr.features.subhalo_zoom import (  # noqa: E402
    Window, contrast_3d, contrast_host_shell, depth_ladder, periodic_delta,
    project_window,
)
from cosmo_sr.reward.tiles import TileGrid  # noqa: E402

from compare_gather_catalog import find_catalogs  # noqa: E402
from gather_holdout_add_density import (  # noqa: E402
    BOX, host_tiles, tile_block_positions,
)

#: panel name  <-  array name in the export's tiles.npz
ARMS = {"tuned": "out", "frozen": "frozen", "hr": "hr"}

#: Slab half-thicknesses for the ladder, in units of the subhalo's own R_vir.
#: The last entry is the full column -- what the host figures show.
DEFAULT_DEPTHS = (1.0, 2.0, 5.0, 10.0, -1.0)

#: Core aperture, as a fraction of the subhalo's R_vir. A subhalo's mean density
#: inside its OWN R_vir is, by the definition of tidal truncation, comparable to
#: the host's local density there -- so the full-R_vir number is order-unity for
#: a real object and cannot discriminate. The core is where a real subhalo is
#: actually dense. Measured on HR's own subhalos, the full aperture returned
#: 0.4-1.2x for 126-3950 particle objects; that is the aperture, not the objects.
CORE_FRAC = 0.3

#: The HR grid is 512^3 over 100 Mpc/h, so one cell is 0.195 Mpc/h and the mean
#: interparticle spacing is the same. A window of +-4 R_vir around a 50-particle
#: subhalo (R_vir = 0.022) is 0.18 Mpc/h across -- ONE cell -- and binning that
#: at 96 bins draws individual particles as confetti. The window is floored so a
#: small object is shown in its surroundings, and the bin count is capped so a
#: bin is never much finer than the field can actually resolve.
CELL_MPC_H = 100.0 / 512.0
MIN_HALF_XY_MPC_H = 0.4
MIN_BIN_MPC_H = CELL_MPC_H / 3.0


def arm_positions(tz, arm_key: str, rows: Sequence[int], tiles: Sequence[int],
                  grid: TileGrid) -> np.ndarray:
    """Positions of the host's tiles for one arm, from the export's tiles.npz."""
    blocks = tz[arm_key]
    return np.concatenate([tile_block_positions(blocks[r], t, grid)
                           for r, t in zip(rows, tiles)], axis=0)


def field_positions(field_path: str, centre: np.ndarray, half: float,
                    box: float) -> np.ndarray:
    """Every particle of a whole field within a cube of half-width ``half``.

    The field is converted in one pass and cropped immediately: keeping 134M
    float64 positions alive is 3.2 GB per arm, and three arms will not fit
    beside each other.
    """
    from cosmo_sr.eval.particles import field_to_particles

    field = np.load(field_path, mmap_mode="r")
    pb = field_to_particles(np.asarray(field), boxsize_kpc_h=box * 1000.0,
                            redshift=0.0)
    del field
    d = periodic_delta(pb.pos_mpc_h, np.asarray(centre, float)[None, :], box)
    keep = (np.abs(d) <= float(half)).all(axis=1)
    out = np.ascontiguousarray(pb.pos_mpc_h[keep], dtype=np.float32)
    del pb, d, keep
    return out


def pick_subhalos(cat: HaloCatalog, centre: np.ndarray, radius: float,
                  min_p: int, n: int, how: str, box: float) -> np.ndarray:
    """Which reported subhalos to draw, as indices into the subhalo table.

    ``spread`` takes them log-uniformly across the particle-count range rather
    than the ``n`` heaviest. That matters: the arms' gains are concentrated in
    the 50-200 particle bin, so a figure of the nine biggest objects would show
    exactly the population that was never in question.
    """
    sub = cat.subhalos()
    d = periodic_delta(sub.pos, np.asarray(centre, float)[None, :], box)
    r = np.sqrt((d ** 2).sum(axis=1))
    cand = np.flatnonzero((r <= float(radius)) & (sub.num_p >= int(min_p)))
    if cand.size == 0:
        return cand
    order = cand[np.argsort(-sub.num_p[cand])]
    if how == "top" or order.size <= n:
        return order[:n]
    lo, hi = np.log10(sub.num_p[order].min()), np.log10(sub.num_p[order].max())
    targets = np.logspace(hi, lo, int(n))          # heaviest first
    chosen, used = [], set()
    for t in targets:
        j = int(np.argmin(np.abs(sub.num_p[order] - t)))
        while j in used and j + 1 < order.size:
            j += 1
        if j not in used:
            used.add(j)
            chosen.append(int(order[j]))
    return np.asarray(chosen, dtype=np.int64)


def measure(pos: Dict[str, np.ndarray], centre: np.ndarray, r: float,
            depths: Sequence[float], axis: int, half_xy: float, bins: int,
            *, host_centre: Optional[np.ndarray] = None,
            core_frac: float = CORE_FRAC,
            ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
    """Panels and contrast numbers for one subhalo, in every arm.

    Four numbers per arm, not one, because the first run of this diagnostic
    showed a single aperture and a single background cannot carry the question:

    ``contrast_3d``        full ``R_vir``, background from the local ``3r..5r``
                           annulus. The naive measurement, kept because it is
                           what a reader expects -- and because it is the one
                           HR's own subhalos fail (see ``CORE_FRAC``).
    ``contrast_3d_core``   the same, inside ``core_frac * R_vir``, where a real
                           subhalo is genuinely dense rather than tidally
                           truncated to its surroundings.
    ``host_shell``         full ``R_vir``, but background from the HOST's own
                           profile at this subhalo's host-centric radius, which
                           removes the host gradient the local annulus picks up.
    ``host_shell_core``    the core aperture against the host profile. **This is
                           the one to read**: clean aperture, clean background.

    ``host_centre`` may be None only for a caller that does not have a host --
    the two host-relative entries are then absent rather than silently wrong.
    """
    panels: Dict[str, np.ndarray] = {}
    stats: Dict[str, Dict] = {}
    r_core = float(core_frac) * float(r)
    for name, p in pos.items():
        stack = []
        for d in depths:
            half_depth = float(d) * r if float(d) > 0 else -1.0
            stack.append(project_window(
                p, Window(centre=tuple(centre), half_xy=half_xy,
                          half_depth=half_depth, axis=axis), BOX, bins))
        panels[name] = np.stack(stack, axis=0).astype(np.float32)
        entry = {
            "contrast_3d": contrast_3d(p, centre, r, BOX).to_dict(),
            "contrast_3d_core": contrast_3d(p, centre, r_core, BOX).to_dict(),
            "ladder": [c.to_dict() for c in
                       depth_ladder(p, centre, r, BOX, depths, axis=axis)],
        }
        if host_centre is not None:
            entry["host_shell"] = contrast_host_shell(
                p, centre, r, host_centre, BOX).to_dict()
            entry["host_shell_core"] = contrast_host_shell(
                p, centre, r_core, host_centre, BOX).to_dict()
        stats[name] = entry
    return panels, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="holdout export dir (tiles.npz + export.json)")
    ap.add_argument("--box", default="set9")
    ap.add_argument("--tag", required=True,
                    help="Rockstar candidate tag, e.g. mgho_all_blocks_self_set9")
    ap.add_argument("--host-id", type=int, default=168880)
    ap.add_argument("--select", default="cand", choices=("cand", "hr", "base"),
                    help="whose reported subhalos to put in the panels; 'cand' "
                         "is the tuned arm's own catalog -- the objects under "
                         "suspicion. 'hr' draws the truth objects instead, "
                         "which is the control run.")
    ap.add_argument("--pick", default="spread", choices=("spread", "top"))
    ap.add_argument("--n-subhalos", type=int, default=9)
    ap.add_argument("--min-p", type=int, default=50)
    ap.add_argument("--host-radius-mult", type=float, default=1.0,
                    help="select subhalos inside this many host R_vir")
    ap.add_argument("--half-xy-rvir", type=float, default=4.0,
                    help="panel half-width in units of the SUBHALO's R_vir")
    ap.add_argument("--depths-rvir", default=",".join(str(d) for d in DEFAULT_DEPTHS),
                    help="slab half-thicknesses in subhalo R_vir; <=0 = full column")
    ap.add_argument("--axis", type=int, default=1, choices=(0, 1, 2),
                    help="projection axis; 1 (y) matches the host figures")
    ap.add_argument("--bins", type=int, default=96,
                    help="upper bound; the actual count is capped so a bin is "
                         "never finer than --min-bin")
    ap.add_argument("--min-half-xy", type=float, default=MIN_HALF_XY_MPC_H,
                    help="floor on the panel half-width, Mpc/h. Without it a "
                         "50-particle subhalo's window is one grid cell across")
    ap.add_argument("--min-bin", type=float, default=MIN_BIN_MPC_H,
                    help="floor on the bin size, Mpc/h (default: a third of a "
                         "512^3 cell)")
    ap.add_argument("--core-frac", type=float, default=CORE_FRAC,
                    help="core aperture as a fraction of the subhalo R_vir; the "
                         "full R_vir is tidally truncated to the host's local "
                         "density and cannot discriminate")
    ap.add_argument("--ng-hr", type=int, default=512)
    ap.add_argument("--tile-hr", type=int, default=64)
    ap.add_argument("--field-out", default="")
    ap.add_argument("--field-frozen", default="")
    ap.add_argument("--field-hr", default="")
    ap.add_argument("--out", required=True, help="output stem (.npz + .json)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    depths = [float(x) for x in args.depths_rvir.split(",") if x.strip()]

    cats = find_catalogs(args.box, args.tag)
    if "hr" not in cats:
        raise SystemExit(f"no HR Rockstar catalog for {args.box}")
    if args.select not in cats:
        raise SystemExit(f"no '{args.select}' Rockstar catalog for {args.box} "
                         f"tag {args.tag}")
    hr = cats["hr"]
    h = np.flatnonzero(hr.ids == int(args.host_id))
    if h.size == 0:
        raise SystemExit(f"host {args.host_id} is not in the HR catalog")
    h = int(h[0])
    host_centre = hr.pos[h].astype(np.float64)
    host_rvir = float(hr.rvir[h]) * 1e-3
    print(f"=== host {args.host_id}: log M {np.log10(hr.mvir[h]):.3f}, "
          f"R_vir {host_rvir:.3f} Mpc/h at {host_centre}", flush=True)

    chosen = pick_subhalos(cats[args.select], host_centre,
                           args.host_radius_mult * host_rvir, args.min_p,
                           args.n_subhalos, args.pick, BOX)
    if chosen.size == 0:
        raise SystemExit(
            f"no '{args.select}' subhalos with >= {args.min_p} particles inside "
            f"{args.host_radius_mult} R_vir of host {args.host_id}: there is "
            "nothing to draw, which is itself the answer")
    sel = cats[args.select].subhalos()
    print(f"=== {chosen.size} subhalos selected ({args.pick}), "
          f"{sel.num_p[chosen].min()}-{sel.num_p[chosen].max()} particles",
          flush=True)

    # --- particles -------------------------------------------------------- #
    fields = {"tuned": args.field_out, "frozen": args.field_frozen,
              "hr": args.field_hr}
    pos: Dict[str, np.ndarray] = {}
    if all(fields.values()):
        # The widest window any subhalo will ask for, plus the background shell.
        widest = float(np.max(sel.rvir[chosen]) * 1e-3) * max(
            args.half_xy_rvir, 5.0, max(depths) if max(depths) > 0 else 0.0)
        pad = float(widest) + 1.0
        for name, fp in fields.items():
            pos[name] = field_positions(fp, host_centre, pad, BOX)
            print(f"  {name:>6}: {pos[name].shape[0]} particles from the full "
                  f"field within {pad:.2f} Mpc/h", flush=True)
        particle_source = "field"
    else:
        grid = TileGrid(ng_hr=args.ng_hr, tile_hr=args.tile_hr, boxsize_mpc_h=BOX)
        tiles = host_tiles(str(run_dir), args.host_id)
        tz = np.load(run_dir / "tiles.npz")
        all_tiles = [int(t) for t in tz["tiles"]]
        missing = [t for t in tiles if t not in all_tiles]
        if missing:
            raise SystemExit(f"tiles {missing} of host {args.host_id} are not in "
                             f"{run_dir / 'tiles.npz'}")
        rows = [all_tiles.index(t) for t in tiles]
        for name, key in ARMS.items():
            pos[name] = arm_positions(tz, key, rows, tiles, grid)
            print(f"  {name:>6} ({key}): {pos[name].shape[0]} particles "
                  f"from tiles {tiles}", flush=True)
        particle_source = "tiles"

    # --- per-subhalo panels and numbers ------------------------------------ #
    out_arrays: Dict[str, np.ndarray] = {}
    rows_json: List[Dict] = []
    for k, i in enumerate(chosen.tolist()):
        centre = sel.pos[i].astype(np.float64)
        r = float(sel.rvir[i]) * 1e-3
        half_xy = max(float(args.half_xy_rvir) * r, float(args.min_half_xy))
        # Never ask for bins the field cannot fill.
        bins = int(max(16, min(int(args.bins),
                               2.0 * half_xy / float(args.min_bin))))
        panels, stats = measure(pos, centre, r, depths, int(args.axis),
                                half_xy, bins,
                                host_centre=host_centre,
                                core_frac=float(args.core_frac))
        for name, arr in panels.items():
            out_arrays[f"sub{k:02d}_{name}"] = arr
        d_host = float(np.sqrt((periodic_delta(centre[None, :],
                                               host_centre[None, :], BOX) ** 2).sum()))
        row = {
            "panel": k,
            "halo_id": int(sel.ids[i]),
            "num_p": int(sel.num_p[i]),
            "log_mvir": float(np.log10(max(sel.mvir[i], 1.0))),
            "rvir_mpc_h": r,
            "pos_mpc_h": centre.tolist(),
            "dist_to_host_mpc_h": d_host,
            "dist_over_host_rvir": d_host / host_rvir if host_rvir else None,
            "half_xy_mpc_h": half_xy,
            "bins": bins,
            "window_floored": bool(half_xy > float(args.half_xy_rvir) * r + 1e-12),
            "arms": stats,
        }
        rows_json.append(row)
        # Print EVERY field, never just one. The first version printed
        # stats["tuned"] regardless of --select, so the `hr` control row
        # actually showed HR's centres measured in the TUNED field -- a
        # different question, and one whose low numbers were misread as the
        # control failing. The field that owns the centres is starred.
        owner = {"cand": "tuned", "hr": "hr", "base": "frozen"}[args.select]
        print(f"  [{k}] id {row['halo_id']} {row['num_p']}p r={r:.3f} "
              f"R/Rvir_host={row['dist_over_host_rvir']:.2f}  "
              f"(centres from: {args.select} -> {owner})", flush=True)
        for arm in ARMS:
            a = stats[arm]
            hs = a.get("host_shell_core", {})
            star = "*" if arm == owner else " "
            if hs and not hs.get("usable", True):
                # Print the count anyway. When the SAME aperture is populated
                # in one field and nearly empty in another -- HR 40 particles,
                # tuned 1 -- the emptiness is the result, and a bare "UNUSABLE"
                # would hide exactly the comparison the figure is about. The
                # flag means "this ratio is not reliable", never "no signal".
                print(f"     {star}{arm:>6}: host-shell core n_in="
                      f"{int(hs.get('n_in', 0)):<5d} ratio unreliable "
                      f"({hs.get('why', '')})", flush=True)
                continue
            print(f"     {star}{arm:>6}: host-shell core "
                  f"{hs.get('ratio', float('nan')):8.2f}x "
                  f"(snr {hs.get('snr', float('nan')):7.1f})   "
                  f"local-annulus core {a['contrast_3d_core']['ratio']:6.2f}x   "
                  f"full-Rvir {a['contrast_3d']['ratio']:7.2f}x   "
                  f"column {a['ladder'][-1]['ratio']:6.2f}x", flush=True)

    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(stem) + ".npz", **out_arrays)
    meta = {
        "box": args.box, "tag": args.tag, "run_dir": str(run_dir),
        "host_id": int(args.host_id),
        "host_centre_mpc_h": host_centre.tolist(),
        "host_rvir_mpc_h": host_rvir,
        "select": args.select, "pick": args.pick,
        "min_particles": int(args.min_p),
        "axis": int(args.axis), "bins": int(args.bins),
        "depths_rvir": depths,
        "half_xy_rvir": float(args.half_xy_rvir),
        "min_half_xy_mpc_h": float(args.min_half_xy),
        "min_bin_mpc_h": float(args.min_bin),
        "core_frac": float(args.core_frac),
        "particle_source": particle_source,
        "particle_source_note": (
            "tiles: only the host's Lagrangian tiles, the material the operator "
            "edited. Material that drifted in from other tiles is absent, which "
            "biases the local background low and the contrast high -- equally in "
            "all three panels. 'field' bins the complete box and has no such bias."
            if particle_source == "tiles" else
            "field: the complete box, no missing foreground or background."),
        "arms": list(ARMS.keys()),
        "subhalos": rows_json,
    }
    Path(str(stem) + ".json").write_text(json.dumps(meta, indent=2))
    print(f"=== wrote {stem}.npz and {stem}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
