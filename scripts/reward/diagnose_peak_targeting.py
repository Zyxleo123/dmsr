#!/usr/bin/env python
"""Can a frozen-SR2 density peak mark a *missing* subhalo the catalog lacks?

The deployable-partition test found the blocker: the nearest frozen-SR2 *subhalo*
to a missing target is a different, already-formed structure, so a bag centred
there shows no SR2->HR rise. But a missing subhalo's members are pre-gathered in
frozen SR2 (Q1: median ~0.74 host-Rvir from the centre) -- a loose overdensity
Rockstar has not bound yet. If a peak finder can flag that proto-overdensity, a
peak-seeded candidate could target the missing structure that no subhalo marks.

The catch is scale. A plain density peak at host smoothing finds the host centre,
not the subhalo bumps riding on the host envelope. So peaks are found with a
**difference of Gaussians** -- density smoothed at the subhalo scale minus the
same density smoothed at the host scale -- whose local maxima are precisely the
subhalo-scale excesses over the smooth host profile.

Two questions, both cheap post-processing of artifacts already on disk:

1. **Proximity** -- for each missing subhalo, how far is the nearest DoG peak,
   against the object's own scale ``r_ref`` and against the nearest existing
   *subhalo*? A peak within ~``r_ref`` means the missing structure is markable.
2. **Scoring** -- re-run the deployable score with **peak-only** centres (no
   subhalo suppression) and see whether ``HR>SR2`` recovers from the 0.17 that
   subhalo centres gave.

    python scripts/reward/diagnose_peak_targeting.py --boxes set8,set9
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from _common import append_jsonl, banner, paths, write_json  # noqa: E402

from cosmo_sr.reward.candidate_score import CandidateScoreConfig  # noqa: E402
from cosmo_sr.reward.oracle_hr import _periodic_gaussian  # noqa: E402

from diagnose_candidate_partition import (  # noqa: E402
    SpatialHash, _FieldCfg, _periodic_dist, load_catalog, load_owner_index,
    load_targets,
)
from diagnose_candidate_partition import load_members as _load_members  # noqa: E402
from diagnose_deployable_partition import score_condition  # noqa: E402
from diagnose_progress_signal import particles_of  # noqa: E402
from rockstar_particles import load_field  # noqa: E402


def dog_residual(pos: np.ndarray, box_l: float, *, ng: int, fine_cells: float,
                 coarse_cells: float):
    """``(resid, local_max, cell)`` -- the difference-of-Gaussians field.

    ``fine_cells < coarse_cells``; ``resid = G_fine(1+delta) - G_coarse`` isolates
    structure between the two scales, so its local maxima are the subhalo-scale
    bumps rather than the host centre a single-scale peak returns. The two
    Gaussian smooths and the ``maximum_filter`` are the cost and are independent
    of any threshold, so this runs once and :func:`peaks_from_residual` thresholds
    it as many times as the sweep needs.
    """
    from scipy.ndimage import maximum_filter

    cell = box_l / ng
    ijk = np.floor(pos / cell).astype(np.int64) % ng
    flat = (ijk[:, 0] * ng + ijk[:, 1]) * ng + ijk[:, 2]
    dens = np.bincount(flat, minlength=ng ** 3).astype(np.float64)
    dens = (dens / dens.mean()).reshape(ng, ng, ng).astype(np.float32)
    resid = _periodic_gaussian(dens, float(fine_cells)) \
        - _periodic_gaussian(dens, float(coarse_cells))
    return resid, maximum_filter(resid, size=3, mode="wrap"), cell


def peaks_from_residual(resid, local_max, cell: float, *, threshold: float,
                        max_peaks: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(centres_mpc_h, strengths)`` local maxima of ``resid`` above ``threshold``."""
    peak = (resid >= local_max) & (resid >= float(threshold))
    pi = np.argwhere(peak)
    if pi.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros(0)
    strength = resid[peak]
    if pi.shape[0] > int(max_peaks):
        keep = np.argsort(strength)[::-1][:int(max_peaks)]
        pi, strength = pi[keep], strength[keep]
    return (pi.astype(np.float64) + 0.5) * cell, strength


def _objects(box, targets, members, hr_cat, pos_hr):
    """Missing targets with their true centre, host Rvir and HR half-mass r_ref."""
    objs = []
    for t in targets:
        sid = int(t["hr_sub_id"])
        mem = members.get(sid)
        if mem is None or mem.size == 0:
            continue
        qh = torch.from_numpy(pos_hr[mem])
        r_ref = float(torch.linalg.norm(qh - qh.mean(0), dim=1).median().clamp_min(1e-3))
        objs.append({"kind": "missing_target", "hr_sub_id": sid,
                     "member_ids": np.asarray(mem, dtype=np.int64),
                     "true_center": np.asarray(t["sub_pos_mpc_h"], dtype=np.float64),
                     "host_rvir_mpc_h": float(t["host_rvir_kpc_h"]) * 1e-3,
                     "r_ref_mpc_h": r_ref})
    return objs


def run_box(box: str, args, out_dir: Path) -> Dict:
    banner(f"{box}: loading fields")
    box_l = float(args.boxsize_mpc_h)
    sr2 = load_field(_FieldCfg(args), box, "base", args.base_seed)
    pos_sr2, vel_sr2, m_p, n_part = particles_of(sr2, box_l)
    del sr2
    hr = load_field(_FieldCfg(args), box, "hr", args.base_seed)
    pos_hr, vel_hr, _, _ = particles_of(hr, box_l)
    del hr

    base_cat = load_catalog(box, "base")
    hr_cat = load_catalog(box, "hr")
    targets = load_targets(box)
    members = _load_members(box)
    objs = _objects(box, targets, members, hr_cat, pos_hr)
    sub_centres = base_cat.subhalos().pos.astype(np.float64)

    ng = int(args.ng_peak)
    thresholds = [float(x) for x in str(args.dog_thresholds).split(",")]

    # ---- Part 1: proximity sweep over DoG threshold -------------------------
    banner(f"{box}: Part 1 -- DoG peak proximity (ng={ng}, fine={args.dog_fine}, "
           f"coarse={args.dog_coarse})")
    prox_p = out_dir / box / "proximity.jsonl"
    if prox_p.exists():
        prox_p.unlink()
    resid, local_max, cell = dog_residual(
        pos_sr2, box_l, ng=ng, fine_cells=args.dog_fine,
        coarse_cells=args.dog_coarse)
    peak_sets: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for thr in thresholds:
        peaks, pstr = peaks_from_residual(resid, local_max, cell,
                                          threshold=thr, max_peaks=args.max_peaks)
        peak_sets[thr] = (peaks, pstr)
        for o in objs:
            tc, rref = o["true_center"], o["r_ref_mpc_h"]
            d_peak = (float(_periodic_dist(peaks, tc, box_l).min())
                      if peaks.shape[0] else float("inf"))
            d_sub = float(_periodic_dist(sub_centres, tc, box_l).min())
            append_jsonl(prox_p, {
                "box": box, "hr_sub_id": o["hr_sub_id"], "threshold": thr,
                "n_peaks": int(peaks.shape[0]), "r_ref_mpc_h": rref,
                "nearest_peak_mpc_h": d_peak, "nearest_sub_mpc_h": d_sub,
                "peak_within_rref": bool(d_peak <= rref),
                "peak_within_half_rref": bool(d_peak <= 0.5 * rref),
                "peak_closer_than_sub": bool(d_peak < d_sub),
            })

    # ---- Part 2: peak-only scoring at the chosen threshold ------------------
    thr0 = float(args.score_threshold)
    if thr0 in peak_sets:
        peaks0 = peak_sets[thr0][0]
    else:
        peaks0 = peaks_from_residual(resid, local_max, cell, threshold=thr0,
                                     max_peaks=args.max_peaks)[0]
    bg = n_part / (box_l ** 3)
    scales = tuple(float(x) for x in str(args.scale_mults).split(","))
    cfg = CandidateScoreConfig(scale_mults=scales, particle_mass_msun_h=m_p,
                               bg_number_density_mpc3=bg)
    cfg_narrow = CandidateScoreConfig(scale_mults=(min(scales),),
                                      particle_mass_msun_h=m_p,
                                      bg_number_density_mpc3=bg)
    r_mult = float(args.r_mult)
    r_max = r_mult * max((o["host_rvir_mpc_h"] for o in objs), default=1.0)
    shash = SpatialHash(pos_sr2, box_l, cell=max(r_max, 0.5))

    banner(f"{box}: Part 2 -- peak-only scoring at threshold {thr0} "
           f"({peaks0.shape[0]} peaks)")
    cond_p = out_dir / box / "conditions.jsonl"
    if cond_p.exists():
        cond_p.unlink()
    rows = []
    for o in objs:
        o["box"] = box
        mem = o["member_ids"]
        tc, rref = o["true_center"], o["r_ref_mpc_h"]
        rc = max(r_mult * o["host_rvir_mpc_h"], 0.2)
        # A oracle baseline
        o["center_offset_mpc_h"] = 0.0
        rows.append(score_condition(mem, tc, rref, pos_sr2, vel_sr2, pos_hr,
                                    vel_hr, box_l, cfg, cfg_narrow, args.n_t,
                                    args.seed, cond="A_oracle", o=o))
        # C_peak: nearest DoG peak centre + radius bag around it
        if peaks0.shape[0]:
            d = _periodic_dist(peaks0, tc, box_l)
            j = int(np.argmin(d))
            pc = peaks0[j]
            o["center_offset_mpc_h"] = float(d[j])
            bag = shash.query(pc, rc)
            rows.append(score_condition(bag, pc, rref, pos_sr2, vel_sr2, pos_hr,
                                        vel_hr, box_l, cfg, cfg_narrow, args.n_t,
                                        args.seed, cond="C_peak", o=o))
    for r in rows:
        append_jsonl(cond_p, r)

    summary = {"box": box, "n_objects": len(objs),
               "dog": {"ng": ng, "fine": args.dog_fine, "coarse": args.dog_coarse},
               "proximity": _proximity_summary(prox_p),
               "scoring": _scoring_summary(rows)}
    write_json(out_dir / box / "summary.json", summary)
    _print_summary(box, summary)
    return summary


def _proximity_summary(path: Path) -> Dict:
    import json
    rows = [json.loads(l) for l in open(path)]
    out: Dict[str, Dict] = {}
    for thr in sorted({r["threshold"] for r in rows}):
        sel = [r for r in rows if r["threshold"] == thr]
        out[f"thr={thr:g}"] = {
            "n_peaks": int(sel[0]["n_peaks"]),
            "frac_within_rref": float(np.mean([r["peak_within_rref"] for r in sel])),
            "frac_within_half_rref": float(np.mean([r["peak_within_half_rref"] for r in sel])),
            "frac_peak_closer_than_sub": float(np.mean([r["peak_closer_than_sub"] for r in sel])),
            "median_nearest_peak_mpc_h": float(np.median([r["nearest_peak_mpc_h"] for r in sel])),
            "median_nearest_sub_mpc_h": float(np.median([r["nearest_sub_mpc_h"] for r in sel])),
        }
    return out


def _scoring_summary(rows: List[Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    for cond in sorted({r["condition"] for r in rows}):
        sel = [r for r in rows if r["condition"] == cond]
        out[cond] = {
            "n": len(sel),
            "frac_hr_gt_sr2_bind": float(np.mean([r["hr_gt_sr2_bind"] for r in sel])),
            "frac_bound_t1": float(np.mean([r["bound_t1"] for r in sel])),
            "median_center_offset_mpc_h": float(np.median([r["center_offset_mpc_h"] for r in sel])),
            "scram_ratio_median": float(np.median([r["scram_ratio"] for r in sel])),
        }
    return out


def _print_summary(box: str, s: Dict) -> None:
    print(f"\n### {box} peak-targeting summary", flush=True)
    print(" Part 1 -- DoG peak proximity to missing subhalos:", flush=True)
    for k, v in s["proximity"].items():
        print(f"   {k:10s} npeaks={v['n_peaks']:6d}  within Rref={v['frac_within_rref']:.2f}"
              f"  within 0.5Rref={v['frac_within_half_rref']:.2f}"
              f"  closer-than-sub={v['frac_peak_closer_than_sub']:.2f}"
              f"  med d_peak={v['median_nearest_peak_mpc_h']:.2f} vs d_sub="
              f"{v['median_nearest_sub_mpc_h']:.2f}", flush=True)
    print(" Part 2 -- peak-only scoring:", flush=True)
    for k, v in s["scoring"].items():
        print(f"   {k:10s} n={v['n']:3d}  HR>SR2 bind={v['frac_hr_gt_sr2_bind']:.2f}"
              f"  bound_t1={v['frac_bound_t1']:.2f}  off={v['median_center_offset_mpc_h']:.2f}"
              f"  scram={v['scram_ratio_median']:.2f}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes", default="set8,set9")
    ap.add_argument("--data-root", default="/zfsauton/scratch/yixiz/DMSR/paired_catnorm")
    ap.add_argument("--boxsize-mpc-h", type=float, default=100.0)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ng-peak", type=int, default=512,
                    help="density grid; 512 -> 0.195 Mpc/h cells, subhalo-resolving")
    ap.add_argument("--dog-fine", type=float, default=1.5,
                    help="fine Gaussian sigma in cells (subhalo scale)")
    ap.add_argument("--dog-coarse", type=float, default=8.0,
                    help="coarse Gaussian sigma in cells (host envelope)")
    ap.add_argument("--dog-thresholds", default="0.5,1,2,4",
                    help="residual-height thresholds to sweep in Part 1")
    ap.add_argument("--score-threshold", type=float, default=1.0,
                    help="DoG threshold used for the Part 2 peak-only scoring")
    ap.add_argument("--max-peaks", type=int, default=200000)
    ap.add_argument("--scale-mults", default="1,2,4")
    ap.add_argument("--n-t", type=int, default=11)
    ap.add_argument("--r-mult", type=float, default=2.0)
    ap.add_argument("--out-name", default="peak_targeting")
    args = ap.parse_args(argv)

    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "8")))
    out_dir = paths.subdir("audits", args.out_name, create=True)
    boxes = [b.strip() for b in args.boxes.split(",") if b.strip()]
    summaries = [run_box(b, args, out_dir) for b in boxes]
    write_json(out_dir / "summary.json", {"boxes": boxes, "per_box": summaries})
    banner(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
