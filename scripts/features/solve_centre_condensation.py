#!/usr/bin/env python
"""Did the generator build an object where the soft solve put its centre?

The membership dump (``finetune_member_gather.py --dump-membership-npz``) on an
arm's own tuned field records the converged candidate centres ``c_j``. Its
radial figure showed them at HR-like radii (median 1.42 R_vir vs HR 1.19) while
Rockstar's subhalos sit at 0.35. Matching HR's exact address is NOT required
here -- only the right host-centric radius -- so this asks, per set and per
radial bin of HR's own placement:

1. **radius**: ``r(c_j) / r(HR)``, both against the set's OWN HR parent
   (never ``centre_host``, see ``host_profile_probe.parent_geometry``);
2. **condensation**: is there a halo in the ARM's Rockstar catalog within one
   search radius of ``c_j`` (wherever ``c_j`` is), and how compact is the
   solved material around ``c_j`` (weighted r_50 / r_rms, in units of the HR
   subhalo's own R_vir).

CPU, reads only saved artefacts. Writes ``<out>/<npz stem>_condensation.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402

BOX = 100.0
MIN_RADIUS = 0.15                      # compare_gather_catalog's min_radius
BINS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf]


def mind(d: np.ndarray) -> np.ndarray:
    return d - BOX * np.round(d / BOX)


def wmedian(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def nearest(cat_pos: np.ndarray, p: np.ndarray):
    d = np.linalg.norm(mind(cat_pos - p[None, :]), axis=1)
    k = int(np.argmin(d))
    return float(d[k]), k


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--hr-ascii", required=True)
    ap.add_argument("--arm-ascii", required=True)
    ap.add_argument("--min-num-p", type=int, default=50,
                    help="arm halos below this are not counted as objects")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)

    z = np.load(a.npz, allow_pickle=True)
    hr = load_rockstar_ascii(a.hr_ascii)
    arm = load_rockstar_ascii(a.arm_ascii)
    keep = np.asarray(arm.num_p) >= a.min_num_p
    apos = np.asarray(arm.pos, dtype=np.float64)[keep]
    anp = np.asarray(arm.num_p)[keep]
    asub = (np.asarray(arm.parent_ids) >= 0)[keep]

    id2row = {int(h): i for i, h in enumerate(hr.ids)}
    hpos = np.asarray(hr.pos, dtype=np.float64)
    hrv = np.asarray(hr.rvir, dtype=np.float64) / 1000.0

    x = z["x"].astype(np.float64)
    setid = z["setid"]
    w_fin = z["sweep_w"][-1].astype(np.float64)
    c_all = z["sweep_c"].astype(np.float64)
    ref = z["centre_ref"].astype(np.float64)

    rows = []
    for j, h in enumerate(z["halo_id"]):
        r = id2row.get(int(h))
        if r is None:
            continue
        pid = int(hr.parent_ids[r])
        pr = id2row.get(pid) if pid >= 0 else None
        if pr is None or hrv[pr] <= 0:
            continue
        P, Rv = hpos[pr], hrv[pr]
        rad = lambda c: float(np.linalg.norm(mind(c - P))) / Rv  # noqa: E731
        search = max(hrv[r], MIN_RADIUS)
        m = setid == j
        w = w_fin[m]
        cj = c_all[-1, j]
        dist = np.linalg.norm(mind(x[m] - cj[None, :]), axis=1)
        ok = w > 0
        d_cj, k_cj = nearest(apos, cj)
        d_hr, k_hr = nearest(apos, ref[j])
        rows.append({
            "halo_id": int(h), "parent_id": pid, "num_p_hr": int(hr.num_p[r]),
            "r_hr": rad(ref[j]), "r_init": rad(c_all[0, j]), "r_fin": rad(cj),
            "search_mpc_h": search,
            "w_mass": float(w.sum()),
            "r50_over_rvir_sub": (wmedian(dist[ok], w[ok]) / hrv[r]
                                  if ok.any() else float("nan")),
            "rrms_over_rvir_sub": (float(np.sqrt((w * dist ** 2).sum()
                                                 / max(w.sum(), 1e-12)))
                                   / hrv[r]),
            "d_obj_cj_search": d_cj / search,
            "obj_cj_num_p": int(anp[k_cj]), "obj_cj_is_sub": bool(asub[k_cj]),
            "d_obj_hr_search": d_hr / search,
        })

    def summ(sel):
        t = [rows[i] for i in sel]
        if not t:
            return {"n": 0}
        g = lambda k: np.array([q[k] for q in t], dtype=np.float64)  # noqa: E731
        hit_cj, hit_hr = g("d_obj_cj_search") < 1.0, g("d_obj_hr_search") < 1.0
        return {
            "n": len(t),
            "median_r_hr": float(np.median(g("r_hr"))),
            "median_r_init": float(np.median(g("r_init"))),
            "median_r_fin": float(np.median(g("r_fin"))),
            "median_fin_over_hr": float(np.median(g("r_fin") / g("r_hr"))),
            "frac_fin_within_x1.5_of_hr": float(np.mean(
                np.abs(np.log(g("r_fin") / g("r_hr"))) < np.log(1.5))),
            "frac_object_at_cj": float(hit_cj.mean()),
            "frac_object_at_hr": float(hit_hr.mean()),
            "median_d_obj_cj_search": float(np.median(g("d_obj_cj_search"))),
            "median_r50_over_rvir_sub": float(np.nanmedian(g("r50_over_rvir_sub"))),
            "median_rrms_over_rvir_sub": float(np.median(g("rrms_over_rvir_sub"))),
        }

    rhr = np.array([q["r_hr"] for q in rows])
    by_bin = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        s = np.flatnonzero((rhr >= lo) & (rhr < hi))
        by_bin.append({"bin": f"{lo:g}-{hi:g}", **summ(s)})
    # Per band by where each centre ACTUALLY sits, beside HR's own count: the
    # scoreboard. A within-bin median ratio of ~1 hides the scatter that
    # smears these counts.
    rfin = np.array([q["r_fin"] for q in rows])
    hit = np.array([q["d_obj_cj_search"] < 1.0 for q in rows])
    bands = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        a_ = (rhr >= lo) & (rhr < hi)
        b_ = (rfin >= lo) & (rfin < hi)
        bands.append({"bin": f"{lo:g}-{hi:g}", "n_hr": int(a_.sum()),
                      "n_cj": int(b_.sum()), "n_cj_obj": int((b_ & hit).sum())})
    rep = {"npz": a.npz, "band_counts": bands, "arm_ascii": a.arm_ascii, "hr_ascii": a.hr_ascii,
           "min_num_p": a.min_num_p, "n_total": int(z["halo_id"].size),
           "n_matched": len(rows), "all": summ(range(len(rows))),
           "by_hr_radius": by_bin, "rows": rows}

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / (Path(a.npz).stem + "_condensation.json")
    f.write_text(json.dumps(rep, indent=1))

    hdr = ("bin(r_HR/Rvir)  n  r_HR  r_init  r_fin  fin/HR  |log|<1.5x  "
           "obj@c_j  obj@HR  r50/Rv_sub  rrms/Rv_sub")
    print(hdr)
    for b in by_bin + [{"bin": "ALL", **rep["all"]}]:
        if not b.get("n"):
            print(f"{b['bin']:>12}  0")
            continue
        print(f"{b['bin']:>12} {b['n']:3d}  {b['median_r_hr']:.2f}  "
              f"{b['median_r_init']:.2f}  {b['median_r_fin']:.2f}  "
              f"{b['median_fin_over_hr']:.2f}   {b['frac_fin_within_x1.5_of_hr']:.2f}"
              f"      {b['frac_object_at_cj']:.2f}    {b['frac_object_at_hr']:.2f}"
              f"     {b['median_r50_over_rvir_sub']:.2f}       "
              f"{b['median_rrms_over_rvir_sub']:.2f}")
    print("\nband(R_vir)   HR  c_j  c_j&obj")
    for b in bands:
        print(f"{b['bin']:>10}  {b['n_hr']:4d} {b['n_cj']:4d} {b['n_cj_obj']:6d}")
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
