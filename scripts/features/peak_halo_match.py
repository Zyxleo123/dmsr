#!/usr/bin/env python
"""Which density peaks does the halo finder turn into objects, and why not the others?

The peak census (``host_phase_profile.py --peaks``) found SR's host outskirts
hold HR's number of compact density peaks, with HR's shapes, yet Rockstar and
SUBFIND find 3-5x fewer objects there. This matches each field's own peaks
against its own catalogs and reports, per r/R_vir bin, the CONVERSION RATE --
the fraction of peaks that carry a halo-finder object -- split by the two
properties a halo finder needs and the census cannot see by eye:

* Lagrangian spread of the peak's 64 nearest particles (lattice cells): one
  collapsed patch (a halo) vs material assembled from distant patches;
* kinematic distinctness, |v_core - v_bg| / sigma_bg, and the core's own
  dispersion over the background's.

A peak "has an object" when a catalog centre lies within
max(--min-match, --match-rvir * r_vir) of it (Rockstar), or within
max(--min-match, 2 x half-mass radius) (SUBFIND). Catalogs only + the saved
peak records: seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402
from host_phase_profile import EDGES_X  # noqa: E402
from placement_forensics import read_rockstar  # noqa: E402

BOX = 100.0


def match(peak_pos: np.ndarray, obj_pos: np.ndarray, obj_rad: np.ndarray,
          min_match: float) -> np.ndarray:
    """Index of the object matched to each peak, -1 if none within its radius.

    Among the objects whose matching radius contains the peak, the nearest.
    """
    from scipy.spatial import cKDTree
    out = np.full(peak_pos.shape[0], -1, dtype=np.int64)
    if obj_pos.shape[0] == 0 or peak_pos.shape[0] == 0:
        return out
    rad = np.maximum(obj_rad, min_match)
    tree = cKDTree(np.mod(obj_pos, BOX), boxsize=BOX)
    cand = tree.query_ball_point(np.mod(peak_pos, BOX), float(rad.max()))
    for i, js in enumerate(cand):
        if not js:
            continue
        js = np.asarray(js)
        dd = obj_pos[js] - peak_pos[i]
        dd -= BOX * np.round(dd / BOX)
        r = np.sqrt(np.einsum("ij,ij->i", dd, dd))
        ok = r < rad[js]
        if ok.any():
            out[i] = int(js[ok][np.argmin(r[ok])])
    return out


def summarise(pk: Dict[str, np.ndarray], hit: np.ndarray, label: str,
              lag_split: float, dv_split: float) -> Dict:
    nb = EDGES_X.size - 1
    b = pk["bin"].astype(int)
    rows = []
    for i in range(nb):
        m = b == i
        n = int(m.sum())
        row = {"bin": f"{EDGES_X[i]:g}-{EDGES_X[i + 1]:g}", "n_peaks": n,
               f"conv_{label}": float(np.mean(hit[m])) if n else float("nan")}
        lag, dv = pk["lag"][m], pk["dvbg"][m]
        h = hit[m]
        for nm, sel in (("compact", lag < lag_split), ("spread", lag >= lag_split),
                        ("distinct", dv >= dv_split), ("comoving", dv < dv_split)):
            row[f"n_{nm}"] = int(np.sum(sel))
            row[f"conv_{label}_{nm}"] = float(np.mean(h[sel])) if sel.any() else float("nan")
        with np.errstate(all="ignore"):
            row["lag_median"] = float(np.nanmedian(lag)) if n else float("nan")
            row["dvbg_median"] = float(np.nanmedian(dv)) if n else float("nan")
            row["coldbg_median"] = float(np.nanmedian(pk["coldbg"][m])) if n else float("nan")
            row["frac_compact"] = float(np.mean(lag < lag_split)) if n else float("nan")
            row["frac_distinct"] = float(np.nanmean(dv >= dv_split)) if n else float("nan")
        rows.append(row)
    return {"by_bin": rows, "conv_all": float(np.mean(hit)) if hit.size else float("nan"),
            "n_peaks": int(hit.size)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields", required=True,
                    help="comma list name=peaks.npz:rockstar.ascii[:subfind.npz]")
    ap.add_argument("--min-match", type=float, default=0.03, help="Mpc/h")
    ap.add_argument("--match-rvir", type=float, default=0.5)
    ap.add_argument("--min-num-p", type=int, default=20)
    ap.add_argument("--lag-split", type=float, default=4.0,
                    help="cells: below = one compact Lagrangian patch")
    ap.add_argument("--dv-split", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    res = {}
    for item in a.fields.split(","):
        name, rest = item.split("=", 1)
        parts = rest.split(":")
        pk = dict(np.load(parts[0]))
        cat = read_rockstar(parts[1])
        keep = cat["num_p"] >= a.min_num_p
        rs = match(pk["pos"], cat["pos"][keep], cat["rvir"][keep] / 1000.0 * a.match_rvir,
                   a.min_match)
        hit_rs = rs >= 0
        sub = (cat["parent_id"][keep] >= 0)
        np50 = cat["num_p"][keep] >= 50
        hit_rs_sub50 = np.zeros_like(hit_rs)
        hit_rs_sub50[hit_rs] = sub[rs[hit_rs]] & np50[rs[hit_rs]]
        out = {"rockstar": summarise(pk, hit_rs, "rs", a.lag_split, a.dv_split),
               "rockstar_sub50": summarise(pk, hit_rs_sub50, "rs50", a.lag_split, a.dv_split)}
        if len(parts) > 2 and Path(parts[2]).is_file():
            z = np.load(parts[2])
            ks = z["SubhaloLen"] >= a.min_num_p
            sfm = match(pk["pos"], z["SubhaloPos"][ks].astype(np.float64),
                        2.0 * z["SubhaloHalfmassRad"][ks].astype(np.float64), a.min_match)
            out["subfind"] = summarise(pk, sfm >= 0, "sf", a.lag_split, a.dv_split)
        res[name] = out
        rows = out["rockstar"]["by_bin"]
        print(f"=== {name}: {out['rockstar']['n_peaks']} peaks, Rockstar conversion "
              f"{out['rockstar']['conv_all']:.2f}", flush=True)
        print(f"  {'bin':>10s} {'peaks':>6s} {'conv':>5s} {'cmpct':>6s} {'c_cmp':>6s} "
              f"{'c_spr':>6s} {'dist':>5s} {'c_dst':>6s} {'c_cmv':>6s} {'lag':>6s} {'dvbg':>5s}"
              + ("  sf_conv" if "subfind" in out else ""))
        for i, r in enumerate(rows):
            sfc = out["subfind"]["by_bin"][i]["conv_sf"] if "subfind" in out else None
            print(f"  {r['bin']:>10s} {r['n_peaks']:6d} {r['conv_rs']:5.2f} "
                  f"{r['frac_compact']:6.2f} {r['conv_rs_compact']:6.2f} {r['conv_rs_spread']:6.2f} "
                  f"{r['frac_distinct']:5.2f} {r['conv_rs_distinct']:6.2f} {r['conv_rs_comoving']:6.2f} "
                  f"{r['lag_median']:6.1f} {r['dvbg_median']:5.2f}"
                  + (f"  {sfc:7.2f}" if sfc is not None else ""), flush=True)
    write_json(Path(a.out), {"min_match": a.min_match, "match_rvir": a.match_rvir,
                             "lag_split": a.lag_split, "dv_split": a.dv_split,
                             "results": res})
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
