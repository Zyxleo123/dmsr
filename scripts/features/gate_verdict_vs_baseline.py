#!/usr/bin/env python
"""A/B a newly-gated member-gather arm against a BASELINE arm, on real Rockstar.

The existing `compare_gather_catalog.py` scores an arm against `base` and `HR`.
It cannot answer the two questions a follow-up arm actually raises, because both
are relative to the arm it is trying to improve on:

  Q1  Did subhalo abundance DETERIORATE from the baseline?
      `all_blocks_self` reproduced HR's subhalo mass function within R_vir
      (366 vs 369) and recovered ~59% of the local subhalo deficit (9187 vs
      base 4541, HR 12367). A change that fixes the velocity field is only
      worth having if it does not give that back.

  Q2  Was the HOST DAMAGE recovered?
      That is the defect section 11 could not explain: HR wants MORE resolved
      hosts in the edited region than base has (3775 vs 3028) and tuning made
      FEWER (2708). The prime suspect was velocity cooling, so an arm with a
      velocity term is the direct test.

Both are counted in the EDITED REGION, never box-wide. `compare_gather_catalog`
reports `hosts_ge_200p_change` over the whole box, where 93.75% is untouched
frozen SR2 -- an earlier read of that scope produced a "7000 spurious halos"
claim that had to be retracted. The region, the host cut and the subhalo cut are
imported from `gather_holdout_figures_data` rather than restated, so there is one
definition of "the edited region" in the tree. That module's field loading is
NOT used: the density panels need two 3.2 GB .npy reads and this asks only for
catalog counts.
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
    MASS_BINS, catalog_path, counts, in_tile_region, load_rockstar_ascii,
    tile_boxes, within_radius_mass_function,
)
from cosmo_sr.reward.tiles import TileGrid  # noqa: E402


def _load(reward_root: Path, box: str, kind: str, tag: str = ""):
    try:
        return load_rockstar_ascii(catalog_path(reward_root, kind, box, tag))
    except SystemExit as e:
        print(f"    (missing: {e})", flush=True)
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reward-root",
                    default="/zfsauton/scratch/yixiz/DMSR/dmsr_reward")
    ap.add_argument("--box", default="set9")
    ap.add_argument("--run-dir", required=True,
                    help="the NEW arm's holdout export dir (export.json gives "
                         "the spliced tile list and the per-host tiles)")
    ap.add_argument("--arm-tag", required=True,
                    help="Rockstar tag of the new arm, e.g. "
                         "mgho_all_blocks_selfvel_set9")
    ap.add_argument("--baseline-tag", required=True,
                    help="Rockstar tag of the arm being improved on. Empty "
                         "means there is nothing to A/B against -- the region "
                         "counts are still the point, so the run reports them "
                         "and returns a REGION ONLY verdict instead of Q1/Q2.")
    ap.add_argument("--baseline-name", default="self")
    ap.add_argument("--host-id", type=int, default=168880)
    ap.add_argument("--dilate", type=float, default=2.0)
    # A tolerance, not a coin flip: the gate's own resplice noise is +-9 targets
    # on a base of ~220, i.e. ~4%. Anything inside that is "unchanged".
    ap.add_argument("--abundance-tol", type=float, default=0.04,
                    help="fractional drop in edited-region subs>=50p that still "
                         "counts as 'not deteriorated' (default 4%%, the "
                         "measured resplice noise)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    R = Path(a.reward_root)
    export = json.loads((Path(a.run_dir) / "export.json").read_text())
    tiles = [int(t) for t in export["tiles"]] if "tiles" in export else None
    if tiles is None:
        tiles = sorted({int(t) for h in export["per_host"] for t in h["tiles"]})
    grid = TileGrid()
    boxes = tile_boxes(tiles, grid)
    print(f"box {a.box}: {len(tiles)} spliced tiles, dilate {a.dilate} Mpc/h")

    cats = {"hr": _load(R, a.box, "hr"), "base": _load(R, a.box, "base"),
            a.baseline_name: _load(R, a.box, "cand", a.baseline_tag),
            "new": _load(R, a.box, "cand", a.arm_tag)}
    missing = [k for k, v in cats.items() if v is None]
    if "new" in missing or "hr" in missing or "base" in missing:
        # Gate failure, not job failure: exiting non-zero would strand every
        # dependent job in DependencyNeverSatisfied with no explanation.
        print(f"\nGATE INCOMPLETE: no catalog for {missing}. "
              f"Nothing is readable; not a verdict.")
        write_json(Path(a.out).with_suffix(".json"),
                   {"ok": False, "missing": missing})
        return 0

    region: Dict[str, Dict] = {}
    for k, c in cats.items():
        if c is None:
            continue
        m = in_tile_region(c.pos, boxes, dilate=a.dilate)
        region[k] = {"total": counts(c), "in_tiles": counts(c, m)}

    print(f"\n=== EDITED REGION ({len(tiles)} tiles, dilated {a.dilate} Mpc/h)")
    print(f"{'':<10} {'hosts>=200p':>12} {'subs>=50p':>11} {'halos':>8}")
    for k in ("base", a.baseline_name, "new", "hr"):
        if k not in region:
            continue
        it = region[k]["in_tiles"]
        print(f"{k:<10} {it['hosts_ge_200p']:>12} {it['subs_ge_50p']:>11} "
              f"{it['n_halos']:>8}")

    hr = cats["hr"]
    where = np.where(hr.ids == a.host_id)[0]
    mf = {}
    if where.size:
        hi = int(where[0])
        centre, rvir = hr.pos[hi].copy(), float(hr.rvir[hi]) / 1000.0
        mf = {k: within_radius_mass_function(c, centre, rvir)
              for k, c in cats.items() if c is not None}
        print(f"\n=== subhalo mass function within R_vir of host {a.host_id} "
              f"({rvir:.2f} Mpc/h)")
        labels = [f"{lo}-{'inf' if hi == np.inf else hi}p" for lo, hi in MASS_BINS]
        print(f"{'':<10} " + " ".join(f"{b:>10}" for b in labels)
              + f" {'total':>8}")
        for k in ("base", a.baseline_name, "new", "hr"):
            if k in mf:
                print(f"{k:<10} " + " ".join(f"{v:>10}" for v in mf[k])
                      + f" {sum(mf[k]):>8}")
        mf = {k: dict(zip(labels, v), total=int(sum(v))) for k, v in mf.items()}

    # --- Q1: abundance vs the baseline --------------------------------------
    b_sub = region[a.baseline_name]["in_tiles"]["subs_ge_50p"] if a.baseline_name in region else None
    n_sub = region["new"]["in_tiles"]["subs_ge_50p"]
    base_sub = region["base"]["in_tiles"]["subs_ge_50p"]
    hr_sub = region["hr"]["in_tiles"]["subs_ge_50p"]
    q1 = None
    if b_sub:
        keep = n_sub / b_sub
        q1 = bool(keep >= 1.0 - a.abundance_tol)
        rec_b = (b_sub - base_sub) / max(hr_sub - base_sub, 1)
        rec_n = (n_sub - base_sub) / max(hr_sub - base_sub, 1)
        print(f"\n=== Q1  subhalo abundance vs {a.baseline_name}")
        print(f"  edited-region subs>=50p: base {base_sub}, "
              f"{a.baseline_name} {b_sub}, new {n_sub}, HR {hr_sub}")
        print(f"  new/{a.baseline_name} = {keep:.3f}  "
              f"(tol {1 - a.abundance_tol:.2f})")
        print(f"  deficit recovered: {a.baseline_name} {100 * rec_b:.1f}%  ->  "
              f"new {100 * rec_n:.1f}%")
        print(f"  {'PASS' if q1 else 'FAIL'}: abundance "
              f"{'held' if q1 else 'DETERIORATED'}")

    # --- Q2: host damage ----------------------------------------------------
    b_h = region[a.baseline_name]["in_tiles"]["hosts_ge_200p"] if a.baseline_name in region else None
    n_h = region["new"]["in_tiles"]["hosts_ge_200p"]
    base_h = region["base"]["in_tiles"]["hosts_ge_200p"]
    hr_h = region["hr"]["in_tiles"]["hosts_ge_200p"]
    q2 = None
    if b_h:
        dmg_b, dmg_n = b_h - base_h, n_h - base_h
        q2 = bool(n_h > b_h)
        print(f"\n=== Q2  host damage vs {a.baseline_name}")
        print(f"  edited-region hosts>=200p: base {base_h}, "
              f"{a.baseline_name} {b_h}, new {n_h}, HR {hr_h}")
        print(f"  damage vs base: {a.baseline_name} {dmg_b:+d}  ->  "
              f"new {dmg_n:+d}")
        if dmg_b < 0:
            print(f"  recovered {100 * (n_h - b_h) / abs(dmg_b):.1f}% of the "
                  f"baseline's host loss")
        print(f"  {'PASS' if q2 else 'FAIL'}: host count "
              f"{'recovered' if q2 else 'NOT recovered'}")
        print(f"  NOTE: HR wants {hr_h - base_h:+d} vs base, so even a full "
              f"recovery to {base_h} is only half the target.")

    if q1 is None and q2 is None:
        # No baseline catalog: the A/B never ran. Saying "abundance LOST" here
        # would invent a comparison that was not made.
        verdict = ("REGION ONLY: no baseline catalog for "
                   f"'{a.baseline_tag or a.baseline_name}'; region counts and "
                   "the mass function are reported, no A/B")
    else:
        verdict = ("PASS: abundance held and hosts recovered" if q1 and q2 else
                   "PARTIAL: " + ("abundance held" if q1 else "abundance LOST")
                   + ", " + ("hosts recovered" if q2 else "hosts NOT recovered"))
    print(f"\n=== VERDICT  {verdict}")
    print("  FEASIBILITY vs the baseline only -- this is real Rockstar on a "
          "held-out box,\n  but it is a TILE SPLICE: 93.75% of the box is held "
          "frozen by construction.")

    write_json(Path(a.out).with_suffix(".json"), {
        "ok": True, "box": a.box, "host_id": a.host_id,
        "arm_tag": a.arm_tag, "baseline_tag": a.baseline_tag,
        "n_tiles": len(tiles), "dilate": a.dilate,
        "region_counts": region, "mass_function": mf,
        "q1_abundance_held": q1, "q2_hosts_recovered": q2,
        "verdict": verdict})
    print(f"\n=== wrote {Path(a.out).with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
