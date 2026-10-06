#!/usr/bin/env python
"""What is different about the Rockstar subhalos SUBFIND refuses?

The gap this explains
---------------------
Inside the trained host, resolved by particle count, the confirmation rate is:

==========  ============  ==========
bin         HR confirmed  arms
==========  ============  ==========
50-100p     7%            0-1%
100-200p    40%           1-4%
200-500p    86%           1-5%
500-2000p   100%          12-29%
2000+p      100%          86-100%
==========  ============  ==========

Below 200 particles HR fails too, so that end is SUBFIND's resolution limit and
carries no information about the arms. **Between 200 and 2000 particles HR
confirms 86-100% and the arms confirm 1-29%** -- objects SUBFIND demonstrably
finds when they are real, and refuses here. This script measures why, on that
band, from the particles themselves.

SUBFIND requires two things Rockstar does not, and each leaves a signature
(:mod:`cosmo_sr.features.subhalo_forensics`):

* **a density peak delimited by a saddle** -> ``r_half_over_r`` (0.3-0.4 for a
  peaked object, 0.794 for a uniform patch) and ``host_radius_over_rvir``;
* **survival of unbinding** -> ``virial_ratio`` = ``sigma_v / v_circ``, near 1
  for a bound clump and well above it for one too hot to hold together.

Every property is computed from the particles inside the aperture, never from
Rockstar's own columns: the question is whether Rockstar's object exists as a
bound compact peak, so taking its mass or radius on trust would assume the
answer. ``n_in`` against Rockstar's ``num_p`` is itself diagnostic -- a large
excess means the aperture is mostly host, not object.

The comparison that matters is **confirmed vs unconfirmed at matched particle
count**, within a field and then across fields. Reporting the arms alone would
not separate "these objects are bad" from "SUBFIND is strict".

    python scripts/features/subfind_miss_forensics.py \\
        --box set9 --tag mgho_all_blocks_self_set9 \\
        --subfind $SUBFIND_ROOT/set9__mgho_all_blocks_self_set9/subfind_catalog.npz \\
        --field $FIELDS/set9__mgho_all_blocks_self_set9__seed0.npy \\
        --host-id 168880 --out $SUBFIND_ROOT/forensics/set9__self
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for _p in (PROJECT_ROOT / "src", SCRIPT_DIR, PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cosmo_sr.eval.particles import field_to_particles  # noqa: E402
from cosmo_sr.features.subhalo_forensics import (  # noqa: E402
    UNIFORM_R_HALF_OVER_R, clump_properties, summarize_split,
)

from compare_gather_catalog import BOXSIZE, find_catalogs  # noqa: E402
from subfind_compare import SubfindCat, match, periodic_delta  # noqa: E402


def region_particles(field_path: str, centre: np.ndarray, half: float,
                     box: float):
    """Positions, velocities and particle mass within a cube about the host.

    Converted in one pass and cropped immediately: 134M float64 positions plus
    velocities is ~6.4 GB live, and nothing downstream needs the other 94% of
    the box.
    """
    field = np.load(field_path, mmap_mode="r")
    pb = field_to_particles(np.asarray(field), boxsize_kpc_h=box * 1000.0,
                            redshift=0.0)
    del field
    d = periodic_delta(pb.pos_mpc_h, np.asarray(centre, float)[None, :], box)
    keep = (np.abs(d) <= float(half)).all(axis=1)
    pos = np.ascontiguousarray(pb.pos_mpc_h[keep], dtype=np.float64)
    vel = np.ascontiguousarray(pb.vel_kms[keep], dtype=np.float64)
    m_p = float(pb.particle_mass_msun_h)
    del pb, d, keep
    return pos, vel, m_p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box", default="set9")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subfind", required=True,
                    help="subfind_catalog.npz of the field being examined")
    ap.add_argument("--field", required=True,
                    help="the (6, Ng, Ng, Ng) .npy the catalogues came from")
    ap.add_argument("--source", default="cand", choices=("cand", "hr", "base"),
                    help="which Rockstar catalog goes with --field")
    ap.add_argument("--host-id", type=int, default=168880)
    ap.add_argument("--radius-mult", type=float, default=1.0)
    ap.add_argument("--min-p", type=int, default=200,
                    help="lower edge of the band to examine. 200 by default, "
                         "because below it HR fails too and the comparison "
                         "carries no signal about the arms")
    ap.add_argument("--max-p", type=int, default=2000,
                    help="upper edge; above it every field confirms ~100% and "
                         "the objects are the shared central structure")
    ap.add_argument("--match-tol-mpc", type=float, default=0.05)
    ap.add_argument("--match-rvir-mult", type=float, default=1.0)
    ap.add_argument("--out", required=True, help="output stem (.json)")
    args = ap.parse_args(argv)

    cats = find_catalogs(args.box, args.tag)
    for need in ("hr", args.source):
        if need not in cats:
            raise SystemExit(f"no Rockstar '{need}' catalog for {args.box}")
    hr = cats["hr"]
    h = np.flatnonzero(hr.ids == int(args.host_id))
    if h.size == 0:
        raise SystemExit(f"host {args.host_id} is not in the HR catalog")
    h = int(h[0])
    host_centre = hr.pos[h].astype(np.float64)
    host_rvir = float(hr.rvir[h]) * 1e-3
    radius = float(args.radius_mult) * host_rvir

    rs = cats[args.source]
    sub = rs.subhalos()
    d = np.sqrt((periodic_delta(sub.pos, host_centre[None, :], BOXSIZE) ** 2).sum(1))
    band = ((d <= radius) & (sub.num_p >= int(args.min_p))
            & (sub.num_p < int(args.max_p)))
    keep = np.flatnonzero(band)
    print(f"=== {args.tag} ({args.source}): {keep.size} Rockstar subhalos in "
          f"{args.min_p}-{args.max_p}p inside {radius:.3f} Mpc/h", flush=True)
    if keep.size == 0:
        raise SystemExit("no subhalos in the band; nothing to explain")

    sf = SubfindCat(args.subfind)
    sf_keep = sf.within(host_centre, radius, int(args.min_p), BOXSIZE)
    idx, sep = match(sub.pos[keep], sub.num_p[keep], sub.rvir[keep] * 1e-3,
                     sf.pos[sf_keep], sf.len[sf_keep], BOXSIZE,
                     tol_mpc=args.match_tol_mpc, rvir_mult=args.match_rvir_mult)
    confirmed = idx >= 0
    print(f"=== {int(confirmed.sum())} of {keep.size} confirmed by SUBFIND "
          f"({confirmed.mean():.1%})", flush=True)

    pad = radius + 2.0
    pos, vel, m_p = region_particles(args.field, host_centre, pad, BOXSIZE)
    print(f"=== {pos.shape[0]} particles within {pad:.2f} Mpc/h, "
          f"m_p = {m_p:.3e} Msun/h", flush=True)

    rows: List[Dict] = []
    for n, i in enumerate(keep.tolist()):
        r = float(sub.rvir[i]) * 1e-3
        p = clump_properties(pos, vel, sub.pos[i].astype(np.float64), r, m_p,
                             BOXSIZE, host_centre=host_centre)
        p.update(halo_id=int(sub.ids[i]), num_p=int(sub.num_p[i]),
                 rvir_mpc_h=r, confirmed=bool(confirmed[n]),
                 separation_mpc_h=(None if not confirmed[n] else float(sep[n])))
        if "host_radius_mpc_h" in p:
            p["host_radius_over_rvir"] = p["host_radius_mpc_h"] / host_rvir
        # n_in / num_p: how much of the aperture is the object at all.
        p["n_in_over_num_p"] = (p["n_in"] / p["num_p"]) if p["num_p"] else None
        rows.append(p)

    summary = summarize_split(rows)
    result = {
        "box": args.box, "tag": args.tag, "source": args.source,
        "field": str(args.field), "host_id": int(args.host_id),
        "host_rvir_mpc_h": host_rvir, "aperture_mpc_h": radius,
        "band_p": [int(args.min_p), int(args.max_p)],
        "n_rockstar": int(keep.size),
        "n_confirmed": int(confirmed.sum()),
        "confirmed_frac": float(confirmed.mean()),
        "uniform_r_half_over_r": UNIFORM_R_HALF_OVER_R,
        "summary": summary,
        "subhalos": rows,
    }
    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = stem.with_name(stem.name + ".json") if not str(stem).endswith(".json") \
        else stem
    out.write_text(json.dumps(result, indent=2))

    print(f"\n=== medians, {args.min_p}-{args.max_p}p, confirmed vs not "
          f"(uniform sphere would give r_half/r = {UNIFORM_R_HALF_OVER_R:.3f})",
          flush=True)
    print(f"{'property':<24} {'confirmed':>12} {'unconfirmed':>12}", flush=True)
    for f in ("num_p", "n_in", "n_in_over_num_p", "virial_ratio",
              "r_half_over_r", "sigma_v", "v_circ", "host_radius_over_rvir"):
        a = summary.get(f"{f}__confirmed")
        b = summary.get(f"{f}__unconfirmed")
        fmt = lambda v: "     -" if v is None else f"{v:12.3f}"
        print(f"{f:<24} {fmt(a)} {fmt(b)}", flush=True)
    print(f"\n=== wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
