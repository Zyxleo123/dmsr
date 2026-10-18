#!/usr/bin/env python
"""Is there enough overdensity in a field to clump into HR's missing satellites?

The low-contrast census (``host_phase_profile.py --peaks --peak-min-ratio 3``)
records every density bump above 3x its shell mean around the 60 massive region
hosts. Restricted to one r/R_vir range (default 0.25-1, where SUBFIND counts
the largest real-object deficit: fine-tuned 23% of HR), this asks three things:

1. SUPPLY -- bumps per contrast band, per field, against HR's;
2. COVERAGE -- what fraction of HR's own objects in that range (Rockstar >=20p
   and SUBFIND satellites >=20p) sit on a bump at all: clumping existing bumps
   can only ever reach objects that grow from one;
3. CEILING -- if every bump of a field turned into an object at HR's rate for its
   contrast, how many objects would it get, against HR's own bump-hosted count.

Catalogs + saved records only; seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402
from host_phase_profile import select_region_hosts  # noqa: E402
from peak_halo_match import match  # noqa: E402
from placement_forensics import min_image, read_rockstar  # noqa: E402

BANDS = (3.0, 10.0, 30.0, 100.0, np.inf)


def nearest_host_x(pos, centres, rvir):
    x = np.full(pos.shape[0], np.inf)
    for c, r in zip(centres, rvir):
        d = np.linalg.norm(min_image(pos - c), axis=1) / r
        np.minimum(x, d, out=x)
    return x


def band_of(ratio):
    return np.digitize(ratio, BANDS[1:-1], right=True)   # 0..3 for ratio > 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields", required=True,
                    help="name=peaks.npz:rockstar.ascii,... ; the FIRST is HR")
    ap.add_argument("--hr-subfind", default="", help="HR subfind_catalog.npz")
    ap.add_argument("--region-export", required=True)
    ap.add_argument("--x-min", type=float, default=0.25)
    ap.add_argument("--x-max", type=float, default=1.0)
    ap.add_argument("--min-match", type=float, default=0.03)
    ap.add_argument("--match-rvir", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    items = [s.split("=", 1) for s in a.fields.split(",")]
    res = {"x_range": [a.x_min, a.x_max], "bands": [f"{lo:g}-{hi:g}" for lo, hi in
                                                    zip(BANDS[:-1], BANDS[1:])],
           "fields": {}}
    conv_hr = None
    hr_bump = None
    for k, (name, rest) in enumerate(items):
        pk_path, cat_path = rest.split(":")[:2]
        pk = dict(np.load(pk_path))
        cat = read_rockstar(cat_path)
        keep = cat["num_p"] >= 20
        hit = match(pk["pos"], cat["pos"][keep], cat["rvir"][keep] / 1000.0 * a.match_rvir,
                    a.min_match) >= 0
        sel = (pk["x"] >= a.x_min) & (pk["x"] < a.x_max) & (pk["ratio"] > BANDS[0])
        b = band_of(pk["ratio"][sel])
        n = np.bincount(b, minlength=4)[:4]
        nc = np.bincount(b, weights=hit[sel].astype(float), minlength=4)[:4]
        conv = np.where(n > 0, nc / np.maximum(n, 1), np.nan)
        if k == 0:
            conv_hr = conv
        ceiling = float(np.nansum(n * conv_hr))
        rec = {"n_bumps": n.tolist(), "n_bumps_total": int(n.sum()),
               "conversion": conv.tolist(), "objects_on_bumps": int(nc.sum()),
               "ceiling_at_hr_conversion": ceiling}
        if k == 0:
            hr_bump = float(nc.sum())
            # --- coverage: HR's objects in the range that sit on a bump ------
            hr = cat
            sel_h = select_region_hosts(hr, a.region_export, 2.0, 10000, 60)
            centres = hr["pos"][sel_h]
            rv = hr["rvir"][sel_h] / 1000.0
            is_host = np.zeros(hr["id"].size, dtype=bool)
            is_host[sel_h] = True
            obj = np.where((hr["num_p"] >= 20) & ~is_host)[0]
            xo = nearest_host_x(hr["pos"][obj], centres, rv)
            obj = obj[(xo >= a.x_min) & (xo < a.x_max)]
            from scipy.spatial import cKDTree
            tree = cKDTree(np.mod(pk["pos"][pk["ratio"] > BANDS[0]], 100.0), boxsize=100.0)
            rad = np.maximum(hr["rvir"][obj] / 1000.0 * a.match_rvir, a.min_match)
            dist, _ = tree.query(np.mod(hr["pos"][obj], 100.0))
            on = dist < rad
            rec["hr_rockstar_objects_in_range"] = int(obj.size)
            rec["hr_rockstar_on_a_bump"] = float(np.mean(on)) if obj.size else np.nan
            for lo in (50, 200):
                m = hr["num_p"][obj] >= lo
                rec[f"hr_rockstar_on_a_bump_ge{lo}"] = float(np.mean(on[m])) if m.any() else np.nan
            if a.hr_subfind and Path(a.hr_subfind).is_file():
                z = np.load(a.hr_subfind)
                s = np.where(z["is_satellite"] & (z["SubhaloLen"] >= 20))[0]
                xs = nearest_host_x(z["SubhaloPos"][s].astype(np.float64), centres, rv)
                s = s[(xs >= a.x_min) & (xs < a.x_max)]
                rs = np.maximum(2.0 * z["SubhaloHalfmassRad"][s].astype(np.float64), a.min_match)
                ds, _ = tree.query(np.mod(z["SubhaloPos"][s].astype(np.float64), 100.0))
                rec["hr_subfind_sats_in_range"] = int(s.size)
                rec["hr_subfind_on_a_bump"] = float(np.mean(ds < rs)) if s.size else np.nan
        res["fields"][name] = rec
        print(f"{name:14s} bumps {n.tolist()} (total {int(n.sum())})  conv "
              f"{np.round(conv, 2).tolist()}  objects-on-bumps {int(nc.sum())}  "
              f"ceiling@HR-conv {ceiling:.0f}" + (f" ({ceiling / hr_bump:.2f}x HR)"
                                                 if hr_bump else ""), flush=True)
        if k == 0:
            print(f"    HR coverage: {rec['hr_rockstar_objects_in_range']} Rockstar objects >=20p "
                  f"in range, {rec['hr_rockstar_on_a_bump']:.2f} on a >3x bump "
                  f"(>=50p {rec['hr_rockstar_on_a_bump_ge50']:.2f}, >=200p "
                  f"{rec['hr_rockstar_on_a_bump_ge200']:.2f})"
                  + (f"; SUBFIND sats {rec['hr_subfind_sats_in_range']}, "
                     f"{rec['hr_subfind_on_a_bump']:.2f} on a bump"
                     if "hr_subfind_sats_in_range" in rec else ""), flush=True)
    write_json(Path(a.out), res)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
