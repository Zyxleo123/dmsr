#!/usr/bin/env python
"""What actually happened to the hosts the `self` arm "destroyed"?

`gather-holdout-rockstar-gate` reports an aggregate: in the edited region HR
resolves 3775 top-level hosts (>=200p), base 3028, the tuned `self` arm 2708 --
a -320 the write-up calls "host damage" but never opens up. The count is a NET
change over a masked region (`hosts_ge_200p = sum(parent_ids < 0 & num_p >= 200
& in_region)`), so it silently lumps three very different fates together:

  * SHRUNK   -- the host is still top-level but fell below 200p (particle loss).
               If this dominates, the damage is a resolution-floor effect and
               should concentrate in the SMALL hosts (near the floor), exactly
               the mass dependence the user predicted.
  * DEMOTED  -- `parent_ids` flipped from <0 to >=0: the object is still there,
               still bound, but Rockstar now sees it inside a bigger halo's
               radius and files it as substructure. This is a BOOKKEEPING move,
               not destruction -- it can even be correct if HR agrees.
  * VANISHED -- no descendant within the search radius: dissolved or absorbed
               with its centre moved away.

And whether any of it is REAL damage (vs base over-resolving relative to truth)
needs HR: a base host that HR ALSO resolves as a top-level >=200p host, but the
arm does not, is genuine divergence from truth; a base host HR does not resolve
either is not the arm's fault.

This job reads three ASCII catalogs (base, hr, tuned self) that already sit in
`dmsr_reward`, matches every base host to a descendant in each by periodic
nearest-most-massive, classifies the fate, cross-checks against HR, and writes:

  * <out>.json   -- counts per fate x mass-bin x (genuine-damage | bookkeeping),
                    plus the reconciliation of the net region host change.
  * <out>.jsonl  -- one row per base host (all fields) so figures redraw cheaply.

CPU-only and submitted: parsing catalogs is real work the login node kills, but
there is no field read and no halo finder here -- it is minutes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402
from gather_holdout_figures_data import (  # noqa: E402
    BOX, MASS_BINS, catalog_path, in_tile_region, load_rockstar_ascii,
    tile_boxes,
)
from cosmo_sr.eval.rockstar import HaloCatalog  # noqa: E402
from cosmo_sr.reward.tiles import TileGrid  # noqa: E402

HOST_FLOOR = 200  # a host is a top-level halo with at least this many particles


def _periodic_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimum-image displacement ``a - b`` (Mpc/h) in the periodic box."""
    d = a - b
    return d - BOX * np.round(d / BOX)


def match_descendant(pos: np.ndarray, dst: HaloCatalog,
                     search_mpc: float) -> int:
    """Row in ``dst`` of the most-massive halo within ``search_mpc`` of ``pos``.

    Nearest-most-massive, not nearest: a host that fragments leaves several
    small clumps and one dominant remnant, and the remnant (its mass mostly
    intact across a 6.25%-spliced field) is the physical continuation. Returns
    -1 when nothing lies within the radius -- the VANISHED case.
    """
    d = _periodic_delta(dst.pos, pos[None, :])
    r2 = np.einsum("ij,ij->i", d, d)
    near = np.where(r2 <= search_mpc * search_mpc)[0]
    if near.size == 0:
        return -1
    return int(near[np.argmax(dst.num_p[near])])


def mass_bin(num_p: int) -> str:
    for lo, hi in MASS_BINS:
        if lo <= num_p < hi:
            return f"{lo}-{hi if hi < 10**11 else 'inf'}"
    return "other"


def classify(num_p0: int, drow: int, dst: HaloCatalog) -> str:
    """Fate of a base host given its matched descendant row (``-1`` = none)."""
    if drow < 0:
        return "VANISHED"
    if dst.parent_ids[drow] >= 0:
        return "DEMOTED"
    if dst.num_p[drow] < HOST_FLOOR:
        return "SHRUNK"
    return "SURVIVED"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reward-root",
                    default="/zfsauton/scratch/yixiz/DMSR/dmsr_reward")
    ap.add_argument("--box", default="set9")
    ap.add_argument("--run-dir", required=True,
                    help="the arm's holdout export dir (export.json -> tiles)")
    ap.add_argument("--arm-tag", required=True,
                    help="Rockstar tag of the arm under test, e.g. "
                         "mgho_all_blocks_self_set9")
    ap.add_argument("--arm-name", default="self")
    ap.add_argument("--dilate", type=float, default=2.0,
                    help="tile dilation (Mpc/h); matches the gate's region")
    ap.add_argument("--search-rvir", type=float, default=2.0,
                    help="descendant search radius in units of the base host's "
                         "R_vir, floored at --search-floor")
    ap.add_argument("--search-floor", type=float, default=0.3,
                    help="minimum descendant search radius (Mpc/h)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    R = Path(a.reward_root)
    export = json.loads((Path(a.run_dir) / "export.json").read_text())
    tiles = ([int(t) for t in export["tiles"]] if "tiles" in export
             else sorted({int(t) for h in export["per_host"] for t in h["tiles"]}))
    grid = TileGrid()
    boxes = tile_boxes(tiles, grid)

    base = load_rockstar_ascii(catalog_path(R, "base", a.box))
    hr = load_rockstar_ascii(catalog_path(R, "hr", a.box))
    arm = load_rockstar_ascii(catalog_path(R, "cand", a.box, a.arm_tag))
    print(f"box {a.box}: {len(tiles)} tiles, dilate {a.dilate} Mpc/h; "
          f"base {base.n}, hr {hr.n}, {a.arm_name} {arm.n} halos", flush=True)

    # The base hosts inside the edited region -- the objects whose fate we trace.
    b_host = (base.parent_ids < 0) & (base.num_p >= HOST_FLOOR)
    b_region = in_tile_region(base.pos, boxes, dilate=a.dilate)
    rows = np.where(b_host & b_region)[0]
    print(f"tracing {rows.size} base hosts (>= {HOST_FLOOR}p) in the region",
          flush=True)

    per_host: List[Dict] = []
    for r in rows:
        pos0 = base.pos[r]
        num_p0 = int(base.num_p[r])
        rvir0_mpc = float(base.rvir[r]) / 1000.0  # kpc/h -> Mpc/h
        search = max(a.search_rvir * rvir0_mpc, a.search_floor)

        arow = match_descendant(pos0, arm, search)
        hrow = match_descendant(pos0, hr, search)
        fate = classify(num_p0, arow, arm)

        # HR's opinion of this base host: is it a top-level >=200p host there
        # too? Only then is the arm's failure to keep it genuine divergence.
        hr_is_host = (hrow >= 0 and hr.parent_ids[hrow] < 0
                      and hr.num_p[hrow] >= HOST_FLOOR)
        lost = fate != "SURVIVED"
        row = {
            "base_id": int(base.ids[r]),
            "num_p_base": num_p0,
            "mass_bin": mass_bin(num_p0),
            "log_mvir_base": float(np.log10(max(base.mvir[r], 1.0))),
            "vmax_base": float(base.vmax[r]),
            "pos": [float(x) for x in pos0],
            "fate": fate,
            "lost": bool(lost),
            # the descendant in the arm catalog (what the host became)
            "num_p_arm": int(arm.num_p[arow]) if arow >= 0 else 0,
            "vmax_arm": float(arm.vmax[arow]) if arow >= 0 else 0.0,
            "arm_is_sub": bool(arow >= 0 and arm.parent_ids[arow] >= 0),
            "arm_parent_id": int(arm.parent_ids[arow]) if arow >= 0 else -1,
            # HR cross-check
            "hr_is_host": bool(hr_is_host),
            "num_p_hr": int(hr.num_p[hrow]) if hrow >= 0 else 0,
            # the interpretation: a lost host HR keeps is REAL damage; a lost
            # host HR also drops is base over-resolving, not the arm's fault.
            "genuine_damage": bool(lost and hr_is_host),
        }
        # Physical signatures for the SHRUNK/DEMOTED split the user asked about:
        # particle loss (below floor) vs a shallower potential (vmax collapse
        # beyond what mass loss implies). Full 2T/|W| needs particles; vmax is
        # the catalog-level proxy for the well depth.
        if arow >= 0:
            row["dnum_p_frac"] = (row["num_p_arm"] - num_p0) / max(num_p0, 1)
            row["dvmax_frac"] = (row["vmax_arm"] - row["vmax_base"]) / max(
                row["vmax_base"], 1e-6)
        per_host.append(row)

    # ---- aggregate: fate x mass-bin x damage, and the net reconciliation -----
    fates = ("SURVIVED", "SHRUNK", "DEMOTED", "VANISHED")
    by_fate = {f: sum(h["fate"] == f for h in per_host) for f in fates}
    by_fate_damage = {
        f: sum(h["fate"] == f and h["genuine_damage"] for h in per_host)
        for f in fates}
    bins = [f"{lo}-{hi if hi < 10**11 else 'inf'}" for lo, hi in MASS_BINS]
    by_bin = {b: {f: sum(h["mass_bin"] == b and h["fate"] == f
                         for h in per_host) for f in fates} for b in bins}

    # The other direction: arm hosts in the region with NO base progenitor are
    # newly resolved hosts. net = gained - lost closes the 3028 -> 2708 gap.
    a_host = (arm.parent_ids < 0) & (arm.num_p >= HOST_FLOOR)
    a_region = in_tile_region(arm.pos, boxes, dilate=a.dilate)
    arm_host_rows = np.where(a_host & a_region)[0]
    # count arm hosts that are the SURVIVED descendant of some base host, so the
    # rest are genuinely NEW hosts (gained), not double-counted survivors.
    survived_arm = set()
    for r in rows:
        pos0 = base.pos[r]
        rvir0_mpc = float(base.rvir[r]) / 1000.0
        search = max(a.search_rvir * rvir0_mpc, a.search_floor)
        arow = match_descendant(pos0, arm, search)
        if arow >= 0 and arm.parent_ids[arow] < 0 and arm.num_p[arow] >= HOST_FLOOR:
            survived_arm.add(int(arow))
    gained = int(sum(1 for ar in arm_host_rows if int(ar) not in survived_arm))

    base_hosts_in = int(rows.size)
    arm_hosts_in = int(arm_host_rows.size)
    lost_total = int(sum(h["lost"] for h in per_host))
    summary = {
        "ok": True,
        "box": a.box,
        "arm": a.arm_name,
        "n_tiles": len(tiles),
        "dilate": a.dilate,
        "region_host_counts": {
            "base": base_hosts_in, a.arm_name: arm_hosts_in,
            "hr": int(np.sum((hr.parent_ids < 0) & (hr.num_p >= HOST_FLOOR)
                             & in_tile_region(hr.pos, boxes, dilate=a.dilate))),
        },
        "reconciliation": {
            "base_hosts": base_hosts_in,
            "lost": lost_total,
            "gained_new": gained,
            "net": gained - lost_total,
            "arm_hosts_predicted": base_hosts_in - lost_total + gained,
            "arm_hosts_actual": arm_hosts_in,
        },
        "by_fate": by_fate,
        "by_fate_genuine_damage": by_fate_damage,
        "by_mass_bin": by_bin,
        "genuine_damage_total": int(sum(h["genuine_damage"] for h in per_host)),
        "bookkeeping_only": int(sum(h["lost"] and not h["genuine_damage"]
                                    for h in per_host)),
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out.with_suffix(".json"), summary)
    with open(out.with_suffix(".jsonl"), "w") as fh:
        for h in per_host:
            fh.write(json.dumps(h) + "\n")

    # ---- human-readable verdict --------------------------------------------
    print(f"\n=== fate of {base_hosts_in} base hosts in the region")
    for f in fates:
        print(f"  {f:<9} {by_fate[f]:>5}   (genuine damage: "
              f"{by_fate_damage[f]})")
    print(f"\n=== net reconciliation (region hosts >= {HOST_FLOOR}p)")
    rc = summary["reconciliation"]
    print(f"  base {rc['base_hosts']}  - lost {rc['lost']}  + gained "
          f"{rc['gained_new']}  = predicted {rc['arm_hosts_predicted']}  "
          f"(actual {rc['arm_hosts_actual']})")
    print(f"\n=== the answer to 'what got violated'")
    dominant = max(("SHRUNK", "DEMOTED", "VANISHED"), key=lambda f: by_fate[f])
    print(f"  dominant loss fate: {dominant}")
    print(f"  genuine damage (HR keeps the host, arm does not): "
          f"{summary['genuine_damage_total']}")
    print(f"  bookkeeping only (HR drops it too / re-parented): "
          f"{summary['bookkeeping_only']}")
    print(f"\n=== loss by mass bin (SHRUNK = floor effect, small hosts)")
    for b in bins:
        d = by_bin[b]
        n_lost = d["SHRUNK"] + d["DEMOTED"] + d["VANISHED"]
        print(f"  {b:>10}p : lost {n_lost:>4}  "
              f"(shrunk {d['SHRUNK']}, demoted {d['DEMOTED']}, "
              f"vanished {d['VANISHED']})")
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
