#!/usr/bin/env python
"""Does the score still measure formation once the ORACLE is removed?

The progress-signal test (:mod:`scripts/reward/diagnose_progress_signal.py`)
validated the multiscale score using the *oracle* partition -- exact HR member
ids, true HR centres -- to isolate the score question from the partition
question. Both are now validated in isolation; this test puts them together and
drops every oracle, because deployment has neither the true centre nor the exact
member set.

Two degradations separate cleanly, so three conditions are scored per object:

* **A oracle** -- true HR centre, exact HR member ids. The baseline the earlier
  test established; reproduced here with a *fixed* (detached) centre so A/B/C are
  compared on equal footing.
* **B contaminated bag** -- true HR centre, but the bag is every particle within
  ``R_c`` of it **in frozen SR2** (fixed by id). This is the deployable bag:
  ~88% complete but only ~10% pure at 2 Rvir (Q1), so it adds background the
  soft radial weight has to suppress. Isolates the contamination effect.
* **C deployable** -- a frozen-SR2 candidate centre (its own subhalos + density
  peaks + NMS, the Q2 machinery), the nearest one to the object, with the same
  radius bag collected around *it*. Adds the centre offset on top of the
  contamination. This is what a training step actually sees.

The bag is fixed at collection time (the note's detached partition); the soft
radial weight in the score is recomputed from the live positions, so background
that drifts away is down-weighted without changing the id set. Each condition is
then scored along the frozen-SR2 -> HR interpolation exactly as before, and the
question is whether HR is still scored above SR2, still becomes bound, and still
rejects the scrambled-velocity control -- now under deployment conditions.

    python scripts/reward/diagnose_deployable_partition.py --boxes set8,set9
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from _common import append_jsonl, banner, paths, write_json  # noqa: E402

from cosmo_sr.eval.particle_identity import periodic_delta  # noqa: E402
from cosmo_sr.reward.candidate_score import (  # noqa: E402
    CandidateScoreConfig, candidate_features, feature_names, formation_score,
)

from diagnose_candidate_partition import (  # noqa: E402
    SpatialHash, _FieldCfg, _periodic_dist, density_peaks, load_catalog,
    load_owner_index, load_targets, nms,
)
from diagnose_candidate_partition import load_members as _load_members  # noqa: E402
from diagnose_progress_signal import particles_of  # noqa: E402
from rockstar_particles import load_field  # noqa: E402


def _objects(box: str, targets, members, owner, hr_cat, *, n_controls, seed):
    """Missing targets (with host Rvir) plus present-subhalo controls."""
    objs = []
    for t in targets:
        sid = int(t["hr_sub_id"])
        mem = members.get(sid)
        if mem is None or mem.size == 0:
            continue
        objs.append({"kind": "missing_target", "hr_sub_id": sid,
                     "member_ids": np.asarray(mem, dtype=np.int64),
                     "true_center": np.asarray(t["sub_pos_mpc_h"], dtype=np.float64),
                     "host_rvir_mpc_h": float(t["host_rvir_kpc_h"]) * 1e-3})
    if owner is None or n_controls <= 0:
        return objs
    subs = hr_cat.subhalos()
    host_row = {int(i): k for k, i in enumerate(hr_cat.ids)}
    rng = np.random.default_rng(int(seed))
    idx = np.arange(subs.n)
    rng.shuffle(idx)
    taken = 0
    for k in idx:
        if taken >= int(n_controls):
            break
        sid = int(subs.ids[k])
        mem = owner.members(sid)
        if mem.size < 200:
            continue
        prow = host_row.get(int(subs.parent_ids[k]))
        rvir = (float(hr_cat.rvir[prow]) if prow is not None
                else float(subs.rvir[k])) * 1e-3
        objs.append({"kind": "present_control", "hr_sub_id": sid,
                     "member_ids": np.asarray(mem, dtype=np.int64),
                     "true_center": subs.pos[k].astype(np.float64),
                     "host_rvir_mpc_h": rvir})
        taken += 1
    return objs


def score_condition(ids: np.ndarray, center_np: np.ndarray, r_ref: float,
                    pos_sr2, vel_sr2, pos_hr, vel_hr, box_l: float,
                    cfg: CandidateScoreConfig, cfg_narrow: CandidateScoreConfig,
                    n_t: int, seed: int, *, cond: str, o: Dict) -> Dict:
    """Score one fixed bag around one fixed centre along the SR2->HR path."""
    ids = np.asarray(ids, dtype=np.int64)
    qs = torch.from_numpy(pos_sr2[ids])
    vs = torch.from_numpy(vel_sr2[ids])
    dq = torch.from_numpy(periodic_delta(pos_hr[ids], pos_sr2[ids], box_l))
    dv = torch.from_numpy(vel_hr[ids] - vel_sr2[ids])
    center = torch.from_numpy(np.asarray(center_np, dtype=np.float64))
    thr = float(cfg.virial_thresh)
    mid = cfg.scale_mults[len(cfg.scale_mults) // 2]

    def one(t: float):
        q = qs + t * dq
        v = vs + t * dv
        f = candidate_features(q, v, center, r_ref, cfg, box=box_l)
        s_b = float(formation_score(q, v, center, r_ref, cfg, box=box_l, use_binding=True))
        s_d = float(formation_score(q, v, center, r_ref, cfg, box=box_l, use_binding=False))
        s_n = float(formation_score(q, v, center, r_ref, cfg_narrow, box=box_l, use_binding=False))
        return s_b, s_d, s_n, float(f[f"virial_ratio_s{mid:g}"])

    ts = np.linspace(0.0, 1.0, int(n_t))
    sb, sd, sn, vr = zip(*[one(float(t)) for t in ts])
    sb, sd, sn, vr = map(np.asarray, (sb, sd, sn, vr))

    def early_slope(y):
        rise = y[-1] - y[0]
        if abs(rise) < 1e-12:
            return 0.0
        half = max(2, len(y) // 2)
        return float((y[half - 1] - y[0]) / rise)

    # Scrambled-velocity control at t=1: HR positions of the bag, hot velocities.
    qh = qs + dq
    g = torch.Generator().manual_seed(int(seed) + int(o["hr_sub_id"]))
    v_scram = torch.randn(ids.size, 3, generator=g) * 800.0
    s_real = float(formation_score(qh, vs + dv, center, r_ref, cfg, box=box_l, use_binding=True))
    s_scram = float(formation_score(qh, v_scram, center, r_ref, cfg, box=box_l, use_binding=True))
    return {
        "box": o.get("box"), "kind": o["kind"], "hr_sub_id": o["hr_sub_id"],
        "condition": cond, "n_bag": int(ids.size), "r_ref_mpc_h": float(r_ref),
        "center_offset_mpc_h": float(o.get("center_offset_mpc_h", 0.0)),
        "score_bind_t0": float(sb[0]), "score_bind_t1": float(sb[-1]),
        "score_dens_t0": float(sd[0]), "score_dens_t1": float(sd[-1]),
        "hr_gt_sr2_bind": bool(sb[-1] > sb[0]), "hr_gt_sr2_dens": bool(sd[-1] > sd[0]),
        "early_slope_bind": early_slope(sb), "early_slope_narrow": early_slope(sn),
        "virial_t0": float(vr[0]), "virial_t1": float(vr[-1]),
        "bound_t0": bool(vr[0] < thr), "bound_t1": bool(vr[-1] < thr),
        "scram_ratio": float(s_scram / s_real) if s_real else 0.0,
        "score_real_t1": s_real, "score_scram_t1": s_scram,
    }


def run_box(box: str, args, out_dir: Path) -> Dict:
    banner(f"{box}: loading fields")
    box_l = float(args.boxsize_mpc_h)
    sr2 = load_field(_FieldCfg(args), box, "base", args.base_seed)
    pos_sr2, vel_sr2, m_p, n_part = particles_of(sr2, box_l)
    del sr2
    hr = load_field(_FieldCfg(args), box, "hr", args.base_seed)
    pos_hr, vel_hr, _, _ = particles_of(hr, box_l)
    del hr

    hr_cat = load_catalog(box, "hr")
    base_cat = load_catalog(box, "base")
    targets = load_targets(box)
    members = _load_members(box)
    owner = load_owner_index(box)

    # Deployable candidate centres from frozen SR2 only (Q2 machinery).
    banner(f"{box}: building frozen-SR2 candidate centres")
    peaks, pstr = density_peaks(pos_sr2, box_l, ng_peak=args.ng_peak,
                               smooth_cells=2.0, min_overdensity=3.0,
                               max_peaks=20000)
    sub_centres = base_cat.subhalos().pos.astype(np.float64)
    all_c = np.vstack([sub_centres, peaks]) if peaks.shape[0] else sub_centres
    all_s = np.concatenate([np.full(sub_centres.shape[0], np.inf),
                            pstr if peaks.shape[0] else np.zeros(0)])
    candidates = nms(all_c, all_s, box_l, args.d_nms_mpc_h)

    bg = n_part / (box_l ** 3)
    scales = tuple(float(x) for x in str(args.scale_mults).split(","))
    cfg = CandidateScoreConfig(scale_mults=scales, particle_mass_msun_h=m_p,
                               bg_number_density_mpc3=bg)
    cfg_narrow = CandidateScoreConfig(scale_mults=(min(scales),),
                                      particle_mass_msun_h=m_p,
                                      bg_number_density_mpc3=bg)

    objs = _objects(box, targets, members, owner, hr_cat,
                    n_controls=args.n_controls, seed=args.seed)
    r_mult = float(args.r_mult)
    r_max = r_mult * max((o["host_rvir_mpc_h"] for o in objs), default=1.0)
    shash = SpatialHash(pos_sr2, box_l, cell=max(r_max, 0.5))

    banner(f"{box}: {len(objs)} objects x 3 conditions (R_c = {r_mult} x host Rvir)")
    rows_p = out_dir / box / "conditions.jsonl"
    if rows_p.exists():
        rows_p.unlink()
    summaries = []
    for o in objs:
        o["box"] = box
        mem = o["member_ids"]
        true_c = o["true_center"]
        # r_ref: the object's own HR half-mass radius (fixed across conditions so
        # the partition, not the scale, is what varies).
        qh = torch.from_numpy(pos_hr[mem])
        r_ref = float(torch.linalg.norm(qh - qh.mean(0), dim=1).median().clamp_min(1e-3))
        rc = max(r_mult * o["host_rvir_mpc_h"], 0.2)

        # Nearest frozen-SR2 candidate to the true position -> deployable centre.
        d = _periodic_dist(candidates, true_c, box_l)
        j = int(np.argmin(d))
        deploy_c = candidates[j]
        # A: oracle centre + member ids
        o["center_offset_mpc_h"] = 0.0
        rowA = score_condition(mem, true_c, r_ref, pos_sr2, vel_sr2, pos_hr, vel_hr,
                               box_l, cfg, cfg_narrow, args.n_t, args.seed,
                               cond="A_oracle", o=o)
        # B: true centre + radius bag
        bagB = shash.query(true_c, rc)
        rowB = score_condition(bagB, true_c, r_ref, pos_sr2, vel_sr2, pos_hr, vel_hr,
                               box_l, cfg, cfg_narrow, args.n_t, args.seed,
                               cond="B_contam_bag", o=o)
        # C: deployable centre + radius bag around it
        o["center_offset_mpc_h"] = float(d[j])
        bagC = shash.query(deploy_c, rc)
        rowC = score_condition(bagC, deploy_c, r_ref, pos_sr2, vel_sr2, pos_hr, vel_hr,
                               box_l, cfg, cfg_narrow, args.n_t, args.seed,
                               cond="C_deployable", o=o)
        for r in (rowA, rowB, rowC):
            append_jsonl(rows_p, r)
            summaries.append(r)

    summary = {"box": box, "n_objects": len(objs), "r_mult": r_mult,
               "n_candidates": int(candidates.shape[0]),
               "aggregate": _aggregate(summaries)}
    write_json(out_dir / box / "summary.json", summary)
    _print_summary(box, summary)
    return summary


def _aggregate(rows: List[Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    for kind in sorted({r["kind"] for r in rows}):
        for cond in sorted({r["condition"] for r in rows if r["kind"] == kind}):
            sel = [r for r in rows if r["kind"] == kind and r["condition"] == cond]
            out[f"{kind}|{cond}"] = {
                "n": len(sel),
                "frac_hr_gt_sr2_bind": float(np.mean([r["hr_gt_sr2_bind"] for r in sel])),
                "frac_hr_gt_sr2_dens": float(np.mean([r["hr_gt_sr2_dens"] for r in sel])),
                "median_center_offset_mpc_h": float(np.median([r["center_offset_mpc_h"] for r in sel])),
                "median_n_bag": float(np.median([r["n_bag"] for r in sel])),
                "virial_t1_median": float(np.median([r["virial_t1"] for r in sel])),
                "frac_bound_t1": float(np.mean([r["bound_t1"] for r in sel])),
                "scram_ratio_median": float(np.median([r["scram_ratio"] for r in sel])),
            }
    return out


def _print_summary(box: str, s: Dict) -> None:
    print(f"\n### {box} deployable-partition summary (R_c={s['r_mult']}xRvir, "
          f"{s['n_candidates']} candidates)", flush=True)
    for k, v in s["aggregate"].items():
        print(f" {k:28s} n={v['n']:3d}  HR>SR2 bind|dens={v['frac_hr_gt_sr2_bind']:.2f}"
              f"|{v['frac_hr_gt_sr2_dens']:.2f}  bound_t1={v['frac_bound_t1']:.2f}"
              f"  scram={v['scram_ratio_median']:.2f}  off={v['median_center_offset_mpc_h']:.2f}"
              f"  nbag={v['median_n_bag']:.0f}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes", default="set8,set9")
    ap.add_argument("--data-root", default="/zfsauton/scratch/yixiz/DMSR/paired_catnorm")
    ap.add_argument("--boxsize-mpc-h", type=float, default=100.0)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale-mults", default="1,2,4")
    ap.add_argument("--n-t", type=int, default=11)
    ap.add_argument("--r-mult", type=float, default=2.0,
                    help="bag radius as a multiple of host Rvir (Q1: ~88%% complete at 2)")
    ap.add_argument("--ng-peak", type=int, default=256)
    ap.add_argument("--d-nms-mpc-h", type=float, default=0.5)
    ap.add_argument("--n-controls", type=int, default=24)
    ap.add_argument("--out-name", default="deployable_partition")
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
