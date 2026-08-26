#!/usr/bin/env python
"""Extract the data behind the held-out gather figures. CPU, reads fields once.

Two questions the box-wide compare cannot answer, and one picture worth drawing:

1. **Is there LOCAL excess?** The splice edits 32 tiles = 6.25% of the box, so
   comparing tuned to HR over the whole box is 93.75% untouched frozen SR2 and
   says nothing. This restricts every count to the union of the 32 spliced tile
   boxes, so the tuned subhalo gain has a like-for-like HR reference instead of a
   box-wide one.
2. **Where is the host damage?** hosts>=200p fell 343 against base. This locates
   those hosts in space and flags which sit in the edited region.
3. **What does the gain look like?** For the example host it projects the edited
   material's density (frozen vs tuned) and overlays the Rockstar subhalos of
   frozen / tuned / HR, so the 20 -> 366 -> 369 is a picture, not a table.

Everything is written to one ``.npz`` + ``.json`` so the plotting job re-renders
without touching a 3.2 GB field again -- the redrawable-figure rule.

A tile is LAGRANGIAN (defined on the initial lattice) but Rockstar finds a halo
at its DISPLACED position, so "in the spliced region" is a spatial test against
the union of the tile boxes, optionally dilated by a few Mpc/h to catch material
that drifted just past a Lagrangian face. The density panels instead select
particles by their Lagrangian tile id -- the exact material the operator edited
-- and show where it landed, which is the honest picture of what changed.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cosmo_sr.eval.particles import field_to_particles  # noqa: E402
from cosmo_sr.eval.rockstar import HaloCatalog, load_rockstar_ascii  # noqa: E402
from cosmo_sr.reward.tiles import TileGrid  # noqa: E402

BOX = 100.0  # Mpc/h
MASS_BINS = [(50, 100), (100, 200), (200, 500), (500, 2000), (2000, 10**12)]
BIN_LABELS = ["50-100p", "100-200p", "200-500p", "500-2000p", "2000+p"]


def catalog_path(reward_root: Path, kind: str, box: str, tag: str = "") -> str:
    if kind == "base":
        pat = f"{reward_root}/halos/{box}__base__base/*_rockstar/halos_*.ascii"
    elif kind == "hr":
        pat = f"{reward_root}/halos/{box}__hr__hr/*_rockstar/halos_*.ascii"
    else:  # candidate
        pat = (f"{reward_root}/flow_rockstar/halos/{box}__candidate__{tag}"
               f"/*/halos_*.ascii")
    hits = glob.glob(pat)
    if not hits:
        raise SystemExit(f"no catalog for kind={kind} tag={tag}: {pat}")
    return hits[0]


def tile_boxes(tiles: List[int], grid: TileGrid) -> np.ndarray:
    """``(n_tiles, 3, 2)`` low/high Mpc/h edges of each Lagrangian tile box."""
    side = BOX / grid.n_per_axis
    out = np.empty((len(tiles), 3, 2))
    for k, t in enumerate(tiles):
        c = grid.coord(int(t))
        for a in range(3):
            out[k, a] = [c[a] * side, (c[a] + 1) * side]
    return out


def in_tile_region(pos: np.ndarray, boxes: np.ndarray, dilate: float = 0.0
                   ) -> np.ndarray:
    """Boolean mask: is each ``(N,3)`` position inside any dilated tile box?

    Periodic in every axis -- a box dilated past 0 or ``BOX`` wraps.
    """
    n = pos.shape[0]
    inside = np.zeros(n, dtype=bool)
    for k in range(boxes.shape[0]):
        m = np.ones(n, dtype=bool)
        for a in range(3):
            lo, hi = boxes[k, a, 0] - dilate, boxes[k, a, 1] + dilate
            d = (pos[:, a] - lo) % BOX
            m &= d <= (hi - lo)
        inside |= m
    return inside


def counts(cat: HaloCatalog, mask: Optional[np.ndarray] = None) -> Dict[str, int]:
    """hosts>=200p and subs>=50p, matching compare_gather_catalog's definitions.

    hosts are top-level (``parent_ids < 0``); subs carry a parent. The particle
    cuts are Rockstar's own ``num_p``.
    """
    host = cat.parent_ids < 0
    sub = ~host
    sel = np.ones(cat.n, dtype=bool) if mask is None else mask
    return {
        "hosts_ge_200p": int(np.sum(sel & host & (cat.num_p >= 200))),
        "subs_ge_50p": int(np.sum(sel & sub & (cat.num_p >= 50))),
        "n_halos": int(np.sum(sel)),
    }


def within_radius_mass_function(cat: HaloCatalog, centre: np.ndarray,
                                rvir_mpc: float) -> List[int]:
    """Count halos within ``rvir`` of ``centre`` (periodic), binned by num_p.

    Excludes the single most massive halo in the sphere -- that is the host, not
    a subhalo of it. Mirrors ``compare_gather_catalog.subhalos_within_rvir``.
    """
    d = np.abs(cat.pos - centre)
    d = np.minimum(d, BOX - d)
    r = np.linalg.norm(d, axis=1)
    inside = r <= rvir_mpc
    if not np.any(inside):
        return [0] * len(MASS_BINS)
    idx = np.where(inside)[0]
    host_local = idx[np.argmax(cat.num_p[idx])]
    out = []
    for lo, hi in MASS_BINS:
        m = inside & (cat.num_p >= lo) & (cat.num_p < hi)
        m[host_local] = False
        out.append(int(np.sum(m)))
    return out


def project_field(field_path: str, centre: np.ndarray, *,
                  host_win: float, host_bins: int,
                  box_slab: float, box_bins: int
                  ) -> Dict[str, object]:
    """Two density projections of a field's particles, summed along y.

    - a **hi-res host window**: ``host_win`` Mpc/h half-width in x/z about the
      host, y within ``host_win`` too, so the subhalos are resolved;
    - a **box slab**: the whole x-z plane, y within ``box_slab`` of the host, for
      the cosmic-web comparison.

    Both from one field read. All fields share the host centre and window, so the
    panels are pixel-comparable.
    """
    field = np.load(field_path, mmap_mode="r")
    pb = field_to_particles(np.asarray(field))
    del field
    pos = pb.pos_mpc_h
    dy = np.abs(pos[:, 1] - centre[1])
    dy = np.minimum(dy, BOX - dy)

    # host window (periodic wrap handled by centring coordinates on the host)
    hxr = (centre[0] - host_win, centre[0] + host_win)
    hzr = (centre[2] - host_win, centre[2] + host_win)
    in_h = dy <= host_win
    hh, _, _ = np.histogram2d(pos[in_h, 0], pos[in_h, 2], bins=host_bins,
                              range=[hxr, hzr])

    # box slab, full plane
    in_b = dy <= box_slab
    bh, _, _ = np.histogram2d(pos[in_b, 0], pos[in_b, 2], bins=box_bins,
                              range=[[0, BOX], [0, BOX]])
    return {"host": hh.T, "host_xr": list(hxr), "host_zr": list(hzr),
            "box": bh.T}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reward-root",
                    default="/zfsauton/scratch/yixiz/DMSR/dmsr_reward")
    ap.add_argument("--box", default="set9")
    ap.add_argument("--run-dir", required=True,
                    help="the self arm's holdout export dir (holds tiles.npz + "
                         "export.json: the tile list and per-host tiles)")
    ap.add_argument("--host-id", type=int, default=168880,
                    help="the example host, scored in the HR catalog")
    ap.add_argument("--self-field", required=True,
                    help="the tuned (self) spliced field .npy")
    ap.add_argument("--frozen-field", required=True,
                    help="the frozen-control spliced field .npy")
    ap.add_argument("--base-field", default="",
                    help="the cached base SR2 field .npy, for the box web "
                         "(optional; frozen is used if absent)")
    ap.add_argument("--hr-field", default="",
                    help="the true HR field .npy, for the HR density panels "
                         "(optional; HR falls back to markers only if absent)")
    ap.add_argument("--host-win", type=float, default=4.5,
                    help="half-width Mpc/h of the hi-res host window")
    ap.add_argument("--box-slab", type=float, default=6.0,
                    help="half-thickness Mpc/h of the cosmic-web y-slab")
    ap.add_argument("--arms", default="self:mgho_all_blocks_self_set9,"
                    "frozen:mgho_frozen_set9,"
                    "nocentre:mgho_all_blocks_nocentre_set9,"
                    "full:mgho_all_blocks_full_set9,"
                    "radial:mgho_all_blocks_radial_set9",
                    help="comma list of name:rockstar_tag")
    ap.add_argument("--dilate", type=float, default=2.0,
                    help="Mpc/h to grow each tile box when masking halos, to "
                         "catch material displaced just past a Lagrangian face")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    R = Path(args.reward_root)
    grid = TileGrid(ng_hr=512, tile_hr=64, boxsize_mpc_h=BOX)
    export = json.loads((Path(args.run_dir) / "export.json").read_text())
    tiles = [int(t) for t in np.load(Path(args.run_dir) / "tiles.npz")["tiles"]]
    per_host = {h["halo_id"]: h["tiles"] for h in export["per_host"]}
    host_tiles = per_host[args.host_id]
    print(f"box {args.box}: {len(tiles)} spliced tiles, host {args.host_id} "
          f"on tiles {host_tiles}", flush=True)

    # --- catalogs -----------------------------------------------------------
    cats: Dict[str, HaloCatalog] = {
        "hr": load_rockstar_ascii(catalog_path(R, "hr", args.box)),
        "base": load_rockstar_ascii(catalog_path(R, "base", args.box)),
    }
    arm_tags = dict(a.split(":") for a in args.arms.split(",") if ":" in a)
    for name, tag in arm_tags.items():
        try:
            cats[name] = load_rockstar_ascii(catalog_path(R, "cand", args.box, tag))
        except SystemExit as e:
            print(f"  {name}: {e}", flush=True)
    for k, c in cats.items():
        print(f"  {k}: {c.n} halos", flush=True)

    boxes = tile_boxes(tiles, grid)

    # --- counts: total and restricted to the spliced tiles ------------------
    region = {}
    for k, c in cats.items():
        mask = in_tile_region(c.pos, boxes, dilate=args.dilate)
        region[k] = {"total": counts(c), "in_tiles": counts(c, mask)}
    print("\ncounts in the spliced region (dilated "
          f"{args.dilate} Mpc/h):", flush=True)
    for k in cats:
        it = region[k]["in_tiles"]
        print(f"  {k:>9}  hosts>=200p {it['hosts_ge_200p']:>5}  "
              f"subs>=50p {it['subs_ge_50p']:>5}  halos {it['n_halos']:>6}",
              flush=True)

    # --- example host: subhalo mass function within R_vir -------------------
    hr = cats["hr"]
    hi = int(np.where(hr.ids == args.host_id)[0][0])
    centre = hr.pos[hi].copy()
    rvir_mpc = float(hr.rvir[hi]) / 1000.0
    mf = {k: within_radius_mass_function(c, centre, rvir_mpc)
          for k, c in cats.items()}
    print(f"\nhost {args.host_id} at {centre}, rvir {rvir_mpc:.3f} Mpc/h", flush=True)
    print("  subhalo mass function within R_vir:", flush=True)
    for k in cats:
        print(f"    {k:>9}  {mf[k]}  total {sum(mf[k])}", flush=True)

    # --- box damage: hosts>=200p near the host, base vs self ----------------
    def hosts200(c):
        m = (c.parent_ids < 0) & (c.num_p >= 200)
        return c.pos[m], c.num_p[m]
    b_pos, b_np = hosts200(cats["base"])
    s_pos, s_np = hosts200(cats["self"]) if "self" in cats else (b_pos, b_np)
    # A base host is "lost" if no self host lies within 0.5 Mpc/h.
    def matched(a_pos, ref_pos, tol=0.5):
        out = np.zeros(a_pos.shape[0], dtype=bool)
        for i in range(a_pos.shape[0]):
            d = np.abs(ref_pos - a_pos[i]); d = np.minimum(d, BOX - d)
            out[i] = np.any(np.linalg.norm(d, axis=1) <= tol)
        return out
    b_in = in_tile_region(b_pos, boxes, dilate=args.dilate)
    b_lost = b_in & ~matched(b_pos, s_pos)
    s_in = in_tile_region(s_pos, boxes, dilate=args.dilate)
    s_gain = s_in & ~matched(s_pos, b_pos)
    print(f"\nhosts>=200p in the spliced region: base {int(b_in.sum())}, "
          f"self {int(s_in.sum())}; base-only (lost) {int(b_lost.sum())}, "
          f"self-only (gained) {int(s_gain.sum())}", flush=True)

    # --- density projections: hi-res host window + box cosmic-web slab ------
    proj_kw = dict(centre=centre, host_win=args.host_win, host_bins=360,
                   box_slab=args.box_slab, box_bins=900)
    dens = {}
    plan = [("frozen", args.frozen_field), ("self", args.self_field)]
    if args.base_field:
        plan.append(("base", args.base_field))
    if args.hr_field:
        plan.append(("hr", args.hr_field))
    for name, path in plan:
        print(f"\nprojecting field ({name})...", flush=True)
        dens[name] = project_field(path, **proj_kw)
    host_xr = dens["frozen"]["host_xr"]
    host_zr = dens["frozen"]["host_zr"]
    # the web "before" is base if given, else frozen (~base on the same edges)
    web_before = "base" if "base" in dens else "frozen"

    # subhalos of the example host for the overlay (within R_vir, >=50p, not host)
    def host_subs(c):
        d = np.abs(c.pos - centre); d = np.minimum(d, BOX - d)
        r = np.linalg.norm(d, axis=1)
        inside = (r <= rvir_mpc) & (c.num_p >= 50)
        if np.any(inside):
            inside[np.where(inside)[0][np.argmax(c.num_p[inside])]] = False
        return c.pos[inside], c.num_p[inside]
    overlay = {k: host_subs(c) for k, c in cats.items()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    host_dens = {f"{k}_host_density": dens[k]["host"] for k in dens}
    box_dens = {f"{k}_box_density": dens[k]["box"] for k in dens}
    np.savez_compressed(
        out.with_suffix(".npz"),
        host_density_xr=np.array(host_xr), host_density_zr=np.array(host_zr),
        web_before=np.array(web_before),
        host_centre=centre, host_rvir_mpc=np.array([rvir_mpc]),
        box_slab_mpc=np.array([args.box_slab]),
        base_hosts_pos=b_pos, base_hosts_lost=b_lost, base_hosts_in=b_in,
        self_hosts_pos=s_pos, self_hosts_gain=s_gain, self_hosts_in=s_in,
        **host_dens, **box_dens,
        **{f"overlay_{k}_pos": overlay[k][0] for k in cats},
        **{f"overlay_{k}_np": overlay[k][1] for k in cats},
        tile_boxes=boxes,
    )
    summary = {
        "box": args.box, "host_id": args.host_id, "n_spliced_tiles": len(tiles),
        "spliced_volume_fraction": len(tiles) / float(grid.n_tiles),
        "host_centre": centre.tolist(), "host_rvir_mpc": rvir_mpc,
        "host_tiles": host_tiles, "dilate_mpc": args.dilate,
        "region_counts": region,
        "mass_function": {k: mf[k] for k in cats},
        "mass_bin_labels": BIN_LABELS,
        "host_damage": {
            "base_in_region": int(b_in.sum()),
            "self_in_region": int(s_in.sum()),
            "base_only_lost": int(b_lost.sum()),
            "self_only_gained": int(s_gain.sum()),
        },
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out.with_suffix('.npz')} and {out.with_suffix('.json')}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
