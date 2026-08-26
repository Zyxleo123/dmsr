#!/usr/bin/env python
"""Does the multiscale candidate score measure genuine formation progress?

The note's cheapest decisive experiment. Take the true HR member particles of a
subhalo, read the *same ids* from frozen SR2, and interpolate their phase-space
coordinates along the straight path from the frozen-SR2 configuration to the HR
one,

    q(t) = q_SR2 + t (q_HR - q_SR2),   v(t) = v_SR2 + t (v_HR - v_SR2),   t in [0,1]

with q differences taken as minimal periodic images. As the object collapses,
does :func:`cosmo_sr.reward.candidate_score.formation_score` rise -- and does it
rise *with usable gradient the whole way*, not just at the end? A reward that is
flat until the object is already formed cannot guide anything there.

Three things are measured, per subhalo, with no Rockstar and no training:

* the score path ``S(t)`` and its slope ``dS/dt`` -- monotone and non-vanishing
  is the claim. Reported for the full binding-gated multiscale score, its
  density-only ablation, and a single-narrow-scale score (the last stands in for
  the pairwise sum's "no early gradient" pathology);
* the binding margin and velocity dispersion along the path -- the object should
  cross into the bound regime as it forms;
* a **scrambled-velocity control** at ``t = 1``: the HR *positions* with hot,
  random velocities. Same density as the real formed object, so a density-only
  score is unchanged, but the binding-gated score must discount it. That is the
  density/velocity distinction the pairwise sum is blind to, tested directly.

    python scripts/reward/diagnose_progress_signal.py --boxes set8,set9
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from _common import append_jsonl, banner, paths, write_json  # noqa: E402

from cosmo_sr.eval.particles import field_to_particles  # noqa: E402
from cosmo_sr.eval.particle_identity import periodic_delta  # noqa: E402
from cosmo_sr.reward.candidate_score import (  # noqa: E402
    CandidateScoreConfig, candidate_features, feature_names, formation_score,
)

from diagnose_candidate_partition import (  # noqa: E402
    _FieldCfg, load_owner_index, load_targets,
)
from diagnose_candidate_partition import load_members as _load_members  # noqa: E402
from rockstar_particles import load_field  # noqa: E402


def particles_of(field, box_l: float):
    """``(pos_mpc_h, vel_kms, particle_mass, n)`` from one field."""
    pb = field_to_particles(np.asarray(field), boxsize_kpc_h=box_l * 1000.0)
    return (pb.pos_mpc_h.astype(np.float64), pb.vel_kms.astype(np.float64),
            float(pb.particle_mass_msun_h), int(pb.pos_mpc_h.shape[0]))


def _object_pool(targets, members, owner, *, n_controls, seed) -> List[Dict]:
    objs = []
    for t in targets:
        sid = int(t["hr_sub_id"])
        mem = members.get(sid)
        if mem is None or mem.size == 0:
            continue
        objs.append({"kind": "missing_target", "hr_sub_id": sid,
                     "member_ids": np.asarray(mem, dtype=np.int64),
                     "sub_mvir": float(t["sub_mvir"])})
    if owner is None or n_controls <= 0:
        return objs
    rng = np.random.default_rng(int(seed))
    hids = owner.halo_id.copy()
    rng.shuffle(hids)
    taken = 0
    for h in hids:
        if taken >= int(n_controls):
            break
        mem = owner.members(int(h))
        if mem.size < 200:
            continue
        objs.append({"kind": "present_control", "hr_sub_id": int(h),
                     "member_ids": np.asarray(mem, dtype=np.int64),
                     "sub_mvir": 0.0})
        taken += 1
    return objs


def _score_variants(q, v, center, r_ref, cfg, cfg_narrow, box_l):
    """The three scores at one path point (all torch scalars -> floats)."""
    s_bind = formation_score(q, v, center, r_ref, cfg, box=box_l, use_binding=True)
    s_dens = formation_score(q, v, center, r_ref, cfg, box=box_l, use_binding=False)
    s_narrow = formation_score(q, v, center, r_ref, cfg_narrow, box=box_l,
                               use_binding=False)
    return float(s_bind), float(s_dens), float(s_narrow)


def run_object(o: Dict, pos_sr2, vel_sr2, pos_hr, vel_hr, box_l: float,
               cfg: CandidateScoreConfig, cfg_narrow: CandidateScoreConfig,
               n_t: int, seed: int) -> Dict:
    mem = o["member_ids"]
    qs = torch.from_numpy(pos_sr2[mem])
    vs = torch.from_numpy(vel_sr2[mem])
    qh = torch.from_numpy(pos_hr[mem])
    vh = torch.from_numpy(vel_hr[mem])
    # Minimal-image displacement SR2 -> HR, so the interpolation is the short way
    # round the box rather than across it.
    dq = torch.from_numpy(periodic_delta(pos_hr[mem], pos_sr2[mem], box_l))

    # Reference radius: the object's own HR half-mass radius. The scale at which
    # "concentrated" is defined, per object, so a cluster and a group are judged
    # on their own size rather than a shared absolute radius.
    ch = qh.mean(0)
    r_ref = float(torch.linalg.norm(qh - ch, dim=1).median().clamp_min(1e-3))

    path: List[Dict] = []
    for t in np.linspace(0.0, 1.0, int(n_t)):
        q = qs + float(t) * dq
        v = vs + float(t) * (vh - vs)
        center = q.mean(0)
        f = candidate_features(q, v, center, r_ref, cfg, box=box_l)
        s_bind, s_dens, s_narrow = _score_variants(
            q, v, center, r_ref, cfg, cfg_narrow, box_l)
        row = {"kind": o["kind"], "hr_sub_id": o["hr_sub_id"], "t": float(t),
               "r_ref_mpc_h": r_ref, "n_members": int(mem.size),
               "score_bind": s_bind, "score_dens": s_dens, "score_narrow": s_narrow}
        mid = cfg.scale_mults[len(cfg.scale_mults) // 2]
        row["sigma_v_kms"] = float(f[f"sigma_v_kms_s{mid:g}"])
        row["virial_ratio"] = float(f[f"virial_ratio_s{mid:g}"])
        row["binding_margin"] = float(f[f"binding_margin_s{mid:g}"])
        row["concentration"] = float(f["concentration"])
        path.append(row)

    # Early slope of each score, in units of the total rise, so a flat-then-jump
    # signal and a steady climb are told apart by one number.
    def early_slope(key: str) -> float:
        y = np.array([r[key] for r in path])
        rise = float(y[-1] - y[0])
        if abs(rise) < 1e-12:
            return 0.0
        # mean slope over the first half of the path, normalised by total rise.
        half = max(2, len(y) // 2)
        return float((y[half - 1] - y[0]) / rise)

    # Scrambled-velocity control at t = 1: HR positions, hot random velocities.
    g = torch.Generator().manual_seed(int(seed) + int(o["hr_sub_id"]))
    v_scram = torch.randn(mem.size, 3, generator=g) * 800.0
    center_h = qh.mean(0)
    s_real = float(formation_score(qh, vh, center_h, r_ref, cfg, box=box_l,
                                   use_binding=True))
    s_scram = float(formation_score(qh, v_scram, center_h, r_ref, cfg, box=box_l,
                                    use_binding=True))
    thr = float(cfg.virial_thresh)
    summary = {
        "kind": o["kind"], "hr_sub_id": o["hr_sub_id"], "n_members": int(mem.size),
        "sub_mvir": o["sub_mvir"], "r_ref_mpc_h": r_ref,
        "score_bind_t0": path[0]["score_bind"], "score_bind_t1": path[-1]["score_bind"],
        "score_dens_t0": path[0]["score_dens"], "score_dens_t1": path[-1]["score_dens"],
        # The straight-line SR2->HR morph overshoots (peaks mid-path), so strict
        # monotonicity is the wrong test; the right one is the endpoint -- is the
        # formed (HR) object scored above the frozen (SR2) one?
        "hr_gt_sr2_bind": bool(path[-1]["score_bind"] > path[0]["score_bind"]),
        "hr_gt_sr2_dens": bool(path[-1]["score_dens"] > path[0]["score_dens"]),
        "monotone_bind": bool(all(
            path[i]["score_bind"] <= path[i + 1]["score_bind"] + 1e-9
            for i in range(len(path) - 1))),
        "early_slope_bind": early_slope("score_bind"),
        "early_slope_narrow": early_slope("score_narrow"),
        "virial_t0": path[0]["virial_ratio"], "virial_t1": path[-1]["virial_ratio"],
        "bound_t0": bool(path[0]["virial_ratio"] < thr),
        "bound_t1": bool(path[-1]["virial_ratio"] < thr),
        "scram_ratio": float(s_scram / s_real) if s_real else 0.0,
        "score_real_t1": s_real, "score_scram_t1": s_scram,
    }
    return {"path": path, "summary": summary}


def run_box(box: str, args, out_dir: Path) -> Dict:
    banner(f"{box}: loading fields")
    box_l = float(args.boxsize_mpc_h)
    sr2 = load_field(_FieldCfg(args), box, "base", args.base_seed)
    pos_sr2, vel_sr2, m_p, n_part = particles_of(sr2, box_l)
    del sr2
    hr = load_field(_FieldCfg(args), box, "hr", args.base_seed)
    pos_hr, vel_hr, _, _ = particles_of(hr, box_l)
    del hr

    targets = load_targets(box)
    members = _load_members(box)
    owner = load_owner_index(box)

    bg = n_part / (box_l ** 3)
    scales = tuple(float(x) for x in str(args.scale_mults).split(","))
    cfg = CandidateScoreConfig(scale_mults=scales, particle_mass_msun_h=m_p,
                               bg_number_density_mpc3=bg)
    cfg_narrow = CandidateScoreConfig(scale_mults=(min(scales),),
                                      particle_mass_msun_h=m_p,
                                      bg_number_density_mpc3=bg)

    objs = _object_pool(targets, members, owner, n_controls=args.n_controls,
                        seed=args.seed)
    banner(f"{box}: {len(objs)} objects x {args.n_t} path points")
    path_p = out_dir / box / "path.jsonl"
    sum_p = out_dir / box / "object_summary.jsonl"
    for p in (path_p, sum_p):
        if p.exists():
            p.unlink()
    summaries = []
    for o in objs:
        res = run_object(o, pos_sr2, vel_sr2, pos_hr, vel_hr, box_l, cfg,
                         cfg_narrow, args.n_t, args.seed)
        for r in res["path"]:
            append_jsonl(path_p, r)
        append_jsonl(sum_p, res["summary"])
        summaries.append(res["summary"])

    summary = {"box": box, "n_objects": len(objs),
               "scale_mults": list(scales), "feature_names": list(feature_names(cfg)),
               "aggregate": _aggregate(summaries)}
    write_json(out_dir / box / "summary.json", summary)
    _print_summary(box, summary)
    return summary


def _aggregate(sums: List[Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    for kind in sorted({s["kind"] for s in sums}):
        sel = [s for s in sums if s["kind"] == kind]
        if not sel:
            continue
        out[kind] = {
            "n": len(sel),
            # Endpoint test (the correct one -- the linear path overshoots):
            "frac_hr_gt_sr2_bind": float(np.mean([s["hr_gt_sr2_bind"] for s in sel])),
            "frac_hr_gt_sr2_dens": float(np.mean([s["hr_gt_sr2_dens"] for s in sel])),
            "median_delta_dens": float(np.median(
                [s["score_dens_t1"] - s["score_dens_t0"] for s in sel])),
            "early_slope_bind_median": float(np.median([s["early_slope_bind"] for s in sel])),
            "early_slope_narrow_median": float(np.median([s["early_slope_narrow"] for s in sel])),
            "frac_monotone_bind": float(np.mean([s["monotone_bind"] for s in sel])),
            "virial_t1_median": float(np.median([s["virial_t1"] for s in sel])),
            "frac_bound_t0": float(np.mean([s["bound_t0"] for s in sel])),
            "frac_bound_t1": float(np.mean([s["bound_t1"] for s in sel])),
            "scram_ratio_median": float(np.median([s["scram_ratio"] for s in sel])),
        }
    return out


def _print_summary(box: str, s: Dict) -> None:
    print(f"\n### {box} progress-signal summary", flush=True)
    for kind, v in s["aggregate"].items():
        print(f" {kind}: n={v['n']}", flush=True)
        print(f"   HR>SR2 (endpoint) bind|dens = {v['frac_hr_gt_sr2_bind']:.3f}"
              f" | {v['frac_hr_gt_sr2_dens']:.3f}   (median dens delta "
              f"{v['median_delta_dens']:+.3f})", flush=True)
        print(f"   early-slope bind|narrow     = {v['early_slope_bind_median']:.3f}"
              f" | {v['early_slope_narrow_median']:.3f}", flush=True)
        print(f"   virial ratio @t1 (median)   = {v['virial_t1_median']:.2f}"
              f"   bound frac t0->t1 = {v['frac_bound_t0']:.3f} -> "
              f"{v['frac_bound_t1']:.3f}", flush=True)
        print(f"   scram/real score @t1        = {v['scram_ratio_median']:.3f}"
              f"   (<1 means hot clump discounted)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes", default="set8,set9")
    ap.add_argument("--data-root", default="/zfsauton/scratch/yixiz/DMSR/paired_catnorm")
    ap.add_argument("--boxsize-mpc-h", type=float, default=100.0)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale-mults", default="1,2,4")
    ap.add_argument("--n-t", type=int, default=11)
    ap.add_argument("--n-controls", type=int, default=24)
    ap.add_argument("--out-name", default="progress_signal")
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
