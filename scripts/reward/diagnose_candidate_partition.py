#!/usr/bin/env python
"""Does the candidate/partitioning step of the shaping reward actually work?

The shaping idea (HackMD note) never optimises a global pairwise sum. It places
*candidate centres* inside frozen-SR2 hosts, collects the nearby high-resolution
tracer particles once, freezes that particle-id set, and lets gradients flow
through those fixed particles' live positions. Two things have to be true before
any of the scoring or survival-head machinery is worth building, and neither
needs a reward, a proxy, or a new Rockstar run -- both are pure post-processing
of artifacts already on disk:

Q1 -- RETRIEVAL (the make-or-break test)
    Put a candidate centre at an HR subhalo's Eulerian position (the
    proxy-training oracle centre) and collect the particles within ``R_c`` **in
    the frozen SR2 configuration**. Gradients only ever reach the particles in
    that bag, so if the subhalo's *true HR member particles* are not in it, no
    optimisation over a fixed bag can build the object. We measure, against the
    real HR member-id set:

    * ``completeness`` -- fraction of the true members the bag caught, in frozen
      SR2. This is the number that decides go/no-go: low completeness means the
      members are dispersed far from the centre in frozen SR2 and the fixed-id
      partition cannot reach them.
    * ``purity`` -- fraction of the bag that is genuine member material (how much
      background the bag drags in; the note's explicit background assignment
      tolerates some, so this is secondary).
    * an HR ceiling -- the same completeness measured in the *HR* configuration,
      i.e. the best any partition centred here could ever do.

    ``R_c`` is swept as multiples of the host's virial radius so the
    completeness-vs-``R_c`` curve tells us how large a bag the deployed scheme
    must gather, rather than guessing one radius.

Q2 -- COVERAGE (candidate discovery, with no HR answers)
    Build candidate centres from **frozen SR2 alone** -- its own subhalos plus
    density peaks -- suppress duplicates by non-maximum suppression, and ask
    whether every real HR subhalo, and especially the *missing* ones the oracle
    targets, has a candidate centre near it. Missing subhalos are the crux:
    frozen SR2 has no subhalo there by construction, so their coverage rests
    entirely on the weak density peaks.

Both questions write one JSONL row per object so the figures
(:mod:`scripts/reward/plot_candidate_partition.py`) redraw without recompute.

    python scripts/reward/diagnose_candidate_partition.py --boxes set8,set9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from _common import append_jsonl, banner, paths, write_json  # noqa: E402

from cosmo_sr.eval.particle_identity import (  # noqa: E402
    build_owner_index, lagrangian_positions, periodic_delta,
)
from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.reward.oracle_hr import _periodic_gaussian  # noqa: E402
from cosmo_sr.reward.pipeline import existing_catalog  # noqa: E402

from rockstar_particles import load_field  # noqa: E402


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_catalog(box: str, source: str):
    """The Rockstar ASCII catalog for ``box``/``source`` (hr | base)."""
    for root in ("halos", "halos_particles"):
        p = existing_catalog(paths.subdir(root, f"{box}__{source}__{source}"), source)
        if p is not None:
            return load_rockstar_ascii(p)
    raise SystemExit(f"no {source} catalog for {box}")


def load_members(box: str) -> Dict[int, np.ndarray]:
    """``halo id -> HR member particle ids`` from the extracted npz (targets+hosts)."""
    hits = sorted(paths.subdir("halos_particles", f"{box}__hr__hr")
                  .glob(f"{box}_hr_members.npz"))
    if not hits:
        raise SystemExit(f"no member ids for {box} (set8_hr_members.npz etc.)")
    z = np.load(hits[0])
    ids, off, pid = z["halo_id"], z["offset"], z["particle_id"]
    return {int(h): pid[off[k]:off[k + 1]] for k, h in enumerate(ids)}


def load_owner_index(box: str):
    """Full ``halo -> member ids`` index from the per-particle owner array, or None.

    Present only where ``<box>_hr_owner.npy`` was written (set8); it lets Q1 add
    *present* HR subhalos as controls, not only the missing targets.
    """
    hits = sorted(paths.subdir("halos_particles", f"{box}__hr__hr")
                  .glob(f"{box}_hr_owner.npy"))
    if not hits:
        return None
    owner = np.load(hits[0], mmap_mode="r")
    return build_owner_index(np.asarray(owner))


def load_targets(box: str) -> List[Dict]:
    p = paths.subdir("oracle_hr") / f"targets_{box}.json"
    if not p.is_file():
        raise SystemExit(f"no targets_{box}.json under {p.parent}")
    return list(json.loads(p.read_text())["targets"])


# --------------------------------------------------------------------------- #
# Eulerian spatial hash (the note's neighbour list, detached indices)
# --------------------------------------------------------------------------- #
class SpatialHash:
    """Uniform periodic grid over Eulerian positions, for radius queries.

    Built once over all ``Ng^3`` frozen-SR2 particles; a query gathers the cells
    of the ``[mu - R, mu + R]`` cube (wrapped) and returns the exact-distance
    members. Cell size is set to the largest query radius so a query touches at
    most ``3^3`` cells.
    """

    def __init__(self, pos: np.ndarray, box: float, cell: float):
        self.box = float(box)
        self.n = max(1, int(np.floor(self.box / float(cell))))
        self.cell = self.box / self.n
        ijk = np.floor(pos / self.cell).astype(np.int64) % self.n
        flat = (ijk[:, 0] * self.n + ijk[:, 1]) * self.n + ijk[:, 2]
        self.order = np.argsort(flat, kind="stable")
        self.flat_sorted = flat[self.order]
        self.pos = pos

    def query(self, center: np.ndarray, radius: float) -> np.ndarray:
        """Particle ids within ``radius`` of ``center`` (periodic)."""
        n, cell = self.n, self.cell
        c = np.floor(np.asarray(center) / cell).astype(np.int64)
        reach = int(np.ceil(radius / cell))
        ids: List[np.ndarray] = []
        rng = range(-reach, reach + 1)
        for dx in rng:
            for dy in rng:
                for dz in rng:
                    cx = (c[0] + dx) % n
                    cy = (c[1] + dy) % n
                    cz = (c[2] + dz) % n
                    key = (cx * n + cy) * n + cz
                    lo = int(np.searchsorted(self.flat_sorted, key, "left"))
                    hi = int(np.searchsorted(self.flat_sorted, key, "right"))
                    if hi > lo:
                        ids.append(self.order[lo:hi])
        if not ids:
            return np.zeros(0, dtype=np.int64)
        cand = np.concatenate(ids)
        d = _periodic_dist(self.pos[cand], np.asarray(center), self.box)
        return cand[d <= radius]


def _periodic_dist(pts: np.ndarray, center: np.ndarray, box: float) -> np.ndarray:
    d = periodic_delta(pts, np.asarray(center)[None, :], box)
    return np.sqrt(np.sum(d * d, axis=1))


# --------------------------------------------------------------------------- #
# Q1: retrieval
# --------------------------------------------------------------------------- #
def _object_pool(box: str, targets: List[Dict], members: Dict[int, np.ndarray],
                 owner, hr, *, n_controls: int, seed: int) -> List[Dict]:
    """The objects Q1 tests: every missing target, plus present-subhalo controls.

    A control is a present HR subhalo (one Rockstar already finds), sampled
    across the mass range spanned by the targets. It answers the sibling
    question -- does the frozen field already keep a *detected* subhalo's members
    near its HR centre? -- which calibrates what "good" retrieval looks like.
    """
    objs: List[Dict] = []
    for t in targets:
        sid = int(t["hr_sub_id"])
        mem = members.get(sid)
        if mem is None or mem.size == 0:
            continue
        objs.append({
            "kind": "missing_target", "hr_sub_id": sid,
            "center": [float(v) for v in t["sub_pos_mpc_h"]],
            "host_rvir_mpc_h": float(t["host_rvir_kpc_h"]) * 1e-3,
            "host_bin": int(t["host_bin"]), "host_mvir": float(t["host_mvir"]),
            "sub_mvir": float(t["sub_mvir"]), "member_ids": mem,
        })
    if owner is None or n_controls <= 0:
        return objs

    subs = hr.subhalos()
    missing = {int(t["hr_sub_id"]) for t in targets}
    lo = min((o["sub_mvir"] for o in objs), default=1e12)
    hi = max((o["sub_mvir"] for o in objs), default=1e14)
    ok = (subs.mvir >= lo) & (subs.mvir <= hi)
    idx = np.nonzero(ok)[0]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    # Host rvir for a subhalo control: its own parent's rvir if we can find it,
    # else the subhalo's rvir. Only sets the R_c ladder's unit, not the members.
    host_row = {int(i): k for k, i in enumerate(hr.ids)}
    taken = 0
    for k in idx:
        if taken >= int(n_controls):
            break
        sid = int(subs.ids[k])
        if sid in missing:
            continue
        mem = owner.members(sid)
        if mem.size < 50:
            continue
        prow = host_row.get(int(subs.parent_ids[k]))
        rvir_kpc = float(hr.rvir[prow]) if prow is not None else float(subs.rvir[k])
        objs.append({
            "kind": "present_control", "hr_sub_id": sid,
            "center": [float(v) for v in subs.pos[k]],
            "host_rvir_mpc_h": rvir_kpc * 1e-3, "host_bin": -1,
            "host_mvir": 0.0, "sub_mvir": float(subs.mvir[k]),
            "member_ids": mem,
        })
        taken += 1
    return objs


def retrieval_rows(box: str, q_sr2: np.ndarray, q_hr: np.ndarray,
                   objs: List[Dict], box_l: float,
                   *, r_mults: Tuple[float, ...], r_floor_mpc_h: float,
                   spatial_hash: SpatialHash) -> List[Dict]:
    rows: List[Dict] = []
    for o in objs:
        mu = np.asarray(o["center"], dtype=np.float64)
        mem = np.asarray(o["member_ids"], dtype=np.int64)
        rvir = max(float(o["host_rvir_mpc_h"]), 1e-6)
        d_sr2 = _periodic_dist(q_sr2[mem], mu, box_l)
        d_hr = _periodic_dist(q_hr[mem], mu, box_l)
        base = {k: o[k] for k in ("kind", "hr_sub_id", "host_bin", "host_mvir",
                                  "sub_mvir")}
        base.update(box=box, n_members=int(mem.size),
                    host_rvir_mpc_h=rvir,
                    member_median_rvir=float(np.median(d_sr2) / rvir),
                    member_p90_rvir=float(np.quantile(d_sr2, 0.9) / rvir))
        for mult in r_mults:
            rc = max(float(mult) * rvir, float(r_floor_mpc_h))
            collected = spatial_hash.query(mu, rc)
            n_col = int(collected.size)
            shared = int(np.intersect1d(collected, mem, assume_unique=False).size)
            row = dict(base)
            row.update(
                r_mult=float(mult), r_c_mpc_h=float(rc),
                completeness_sr2=float(np.mean(d_sr2 <= rc)),
                completeness_hr=float(np.mean(d_hr <= rc)),
                n_collected=n_col,
                purity=float(shared / n_col) if n_col else 0.0,
            )
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Q2: coverage
# --------------------------------------------------------------------------- #
def density_peaks(q_sr2: np.ndarray, box_l: float, *, ng_peak: int,
                  smooth_cells: float, min_overdensity: float,
                  max_peaks: int) -> Tuple[np.ndarray, np.ndarray]:
    """Local maxima of the smoothed frozen-SR2 number density, as candidate centres.

    Returns ``(centres_mpc_h, strengths)`` -- the second used only to order NMS.
    """
    from scipy.ndimage import maximum_filter

    cell = box_l / ng_peak
    ijk = np.floor(q_sr2 / cell).astype(np.int64) % ng_peak
    flat = (ijk[:, 0] * ng_peak + ijk[:, 1]) * ng_peak + ijk[:, 2]
    counts = np.bincount(flat, minlength=ng_peak ** 3).astype(np.float64)
    counts = counts.reshape(ng_peak, ng_peak, ng_peak)
    dens = counts / counts.mean()                      # 1 + delta
    sm = _periodic_gaussian(dens.astype(np.float32), float(smooth_cells))
    mx = maximum_filter(sm, size=3, mode="wrap")
    peak = (sm >= mx) & (sm >= float(min_overdensity))
    pi = np.argwhere(peak)
    if pi.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros(0)
    strength = sm[peak]
    if pi.shape[0] > int(max_peaks):
        keep = np.argsort(strength)[::-1][:int(max_peaks)]
        pi, strength = pi[keep], strength[keep]
    centres = (pi.astype(np.float64) + 0.5) * cell
    return centres, strength


def nms(centres: np.ndarray, strengths: np.ndarray, box_l: float,
        d_nms_mpc_h: float) -> np.ndarray:
    """Greedy periodic non-maximum suppression: strongest first, drop near-duplicates."""
    if centres.shape[0] == 0:
        return centres
    order = np.argsort(strengths)[::-1]
    kept: List[int] = []
    kept_pos = np.zeros((0, 3))
    for k in order:
        c = centres[k]
        if kept_pos.shape[0]:
            d = _periodic_dist(kept_pos, c, box_l)
            if float(d.min()) < float(d_nms_mpc_h):
                continue
        kept.append(int(k))
        kept_pos = np.vstack([kept_pos, c[None, :]])
    return centres[np.asarray(kept, dtype=np.int64)]


def coverage_rows(box: str, hr, base_cat, q_sr2: np.ndarray, targets: List[Dict],
                  box_l: float, *, ng_peak: int, smooth_cells: float,
                  min_overdensity: float, max_peaks: int, d_nms_mpc_h: float,
                  ) -> Tuple[List[Dict], Dict]:
    """Nearest frozen-SR2 candidate for every HR subhalo, sub-only and sub+peak."""
    sub_centres = base_cat.subhalos().pos.astype(np.float64)
    peaks, pstr = density_peaks(q_sr2, box_l, ng_peak=ng_peak,
                                smooth_cells=smooth_cells,
                                min_overdensity=min_overdensity,
                                max_peaks=max_peaks)
    all_centres = np.vstack([sub_centres, peaks]) if peaks.shape[0] else sub_centres
    all_str = np.concatenate([
        np.full(sub_centres.shape[0], np.inf),      # subhalos win NMS ties
        pstr if peaks.shape[0] else np.zeros(0)])
    cand_sub = nms(sub_centres, np.full(sub_centres.shape[0], np.inf),
                   box_l, d_nms_mpc_h)
    cand_all = nms(all_centres, all_str, box_l, d_nms_mpc_h)

    def nearest(pos: np.ndarray, centres: np.ndarray) -> float:
        if centres.shape[0] == 0:
            return float("inf")
        return float(_periodic_dist(centres, pos, box_l).min())

    subs = hr.subhalos()
    missing = {int(t["hr_sub_id"]) for t in targets}
    rows: List[Dict] = []
    for k in range(subs.n):
        sid = int(subs.ids[k])
        pos = subs.pos[k].astype(np.float64)
        rows.append({
            "box": box, "hr_sub_id": sid, "sub_mvir": float(subs.mvir[k]),
            "sub_num_p": int(subs.num_p[k]),
            "is_missing": bool(sid in missing),
            "nearest_sub_mpc_h": nearest(pos, cand_sub),
            "nearest_all_mpc_h": nearest(pos, cand_all),
        })
    meta = {
        "n_hr_subhalos": int(subs.n),
        "n_missing_targets": int(len(missing)),
        "n_base_subhalos": int(sub_centres.shape[0]),
        "n_density_peaks": int(peaks.shape[0]),
        "n_candidates_sub": int(cand_sub.shape[0]),
        "n_candidates_all": int(cand_all.shape[0]),
        "d_nms_mpc_h": float(d_nms_mpc_h),
    }
    return rows, meta


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_box(box: str, args, out_dir: Path) -> Dict:
    banner(f"{box}: loading fields and catalogs")
    sr2 = load_field(_FieldCfg(args), box, "base", args.base_seed)
    hr = load_field(_FieldCfg(args), box, "hr", args.base_seed)
    box_l = float(args.boxsize_mpc_h)
    banner(f"{box}: computing Eulerian positions (2 x Ng^3)")
    q_sr2 = lagrangian_positions(np.asarray(sr2), boxsize_mpc_h=box_l).astype(np.float64)
    q_hr = lagrangian_positions(np.asarray(hr), boxsize_mpc_h=box_l).astype(np.float64)
    del sr2, hr

    hr_cat = load_catalog(box, "hr")
    base_cat = load_catalog(box, "base")
    targets = load_targets(box)
    members = load_members(box)
    owner = load_owner_index(box)

    r_mults = tuple(float(x) for x in str(args.r_mults).split(","))
    r_max = max(r_mults) * max((float(t["host_rvir_kpc_h"]) * 1e-3
                                for t in targets), default=1.0)
    r_max = max(r_max, args.r_floor_mpc_h)
    shash = SpatialHash(q_sr2, box_l, cell=max(r_max, 0.5))

    banner(f"{box}: Q1 retrieval")
    objs = _object_pool(box, targets, members, owner, hr_cat,
                        n_controls=args.n_controls, seed=args.seed)
    ret = retrieval_rows(box, q_sr2, q_hr, objs, box_l, r_mults=r_mults,
                         r_floor_mpc_h=args.r_floor_mpc_h, spatial_hash=shash)
    ret_path = out_dir / box / "retrieval.jsonl"
    if ret_path.exists():
        ret_path.unlink()
    for r in ret:
        append_jsonl(ret_path, r)

    banner(f"{box}: Q2 coverage")
    cov, meta = coverage_rows(
        box, hr_cat, base_cat, q_sr2, targets, box_l,
        ng_peak=args.ng_peak, smooth_cells=args.smooth_cells,
        min_overdensity=args.min_overdensity, max_peaks=args.max_peaks,
        d_nms_mpc_h=args.d_nms_mpc_h)
    cov_path = out_dir / box / "coverage.jsonl"
    if cov_path.exists():
        cov_path.unlink()
    for r in cov:
        append_jsonl(cov_path, r)

    summary = {"box": box, "coverage_meta": meta,
               "retrieval": _retrieval_summary(ret),
               "coverage": _coverage_summary(cov)}
    write_json(out_dir / box / "summary.json", summary)
    _print_summary(box, summary)
    return summary


def _retrieval_summary(rows: List[Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    for kind in sorted({r["kind"] for r in rows}):
        for mult in sorted({r["r_mult"] for r in rows if r["kind"] == kind}):
            sel = [r for r in rows if r["kind"] == kind and r["r_mult"] == mult]
            out[f"{kind}@{mult:g}Rvir"] = {
                "n": len(sel),
                "completeness_sr2_median":
                    float(np.median([r["completeness_sr2"] for r in sel])),
                "completeness_hr_median":
                    float(np.median([r["completeness_hr"] for r in sel])),
                "purity_median": float(np.median([r["purity"] for r in sel])),
            }
    return out


def _coverage_summary(rows: List[Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    for label, sel in (("all", rows),
                       ("missing", [r for r in rows if r["is_missing"]]),
                       ("present", [r for r in rows if not r["is_missing"]])):
        if not sel:
            continue
        block = {"n": len(sel)}
        for thr in (0.5, 1.0, 2.0):
            block[f"cov_sub_within_{thr}"] = float(
                np.mean([r["nearest_sub_mpc_h"] <= thr for r in sel]))
            block[f"cov_all_within_{thr}"] = float(
                np.mean([r["nearest_all_mpc_h"] <= thr for r in sel]))
        out[label] = block
    return out


def _print_summary(box: str, s: Dict) -> None:
    print(f"\n### {box} candidate/partition summary", flush=True)
    print(" Q1 retrieval (median completeness in frozen SR2 | HR ceiling):", flush=True)
    for k, v in s["retrieval"].items():
        print(f"   {k:28s} n={v['n']:3d}  sr2={v['completeness_sr2_median']:.3f}"
              f"  hr={v['completeness_hr_median']:.3f}"
              f"  purity={v['purity_median']:.3f}", flush=True)
    print(" Q2 coverage (fraction with a candidate within R Mpc/h):", flush=True)
    for k, v in s["coverage"].items():
        print(f"   {k:8s} n={v['n']:4d}  sub<=1={v.get('cov_sub_within_1.0', 0):.3f}"
              f"  all<=1={v.get('cov_all_within_1.0', 0):.3f}"
              f"  all<=0.5={v.get('cov_all_within_0.5', 0):.3f}", flush=True)
    m = s["coverage_meta"]
    print(f"   candidates: {m['n_candidates_sub']} sub-only, "
          f"{m['n_candidates_all']} sub+peak "
          f"(from {m['n_base_subhalos']} base subs, {m['n_density_peaks']} peaks)",
          flush=True)


class _FieldCfg:
    """Minimal config shim so ``rockstar_particles.load_field`` finds HR paths."""

    def __init__(self, args):
        self._d = {"data": {"root": args.data_root, "boxsize_mpc_h": args.boxsize_mpc_h}}

    def __getitem__(self, k):
        return self._d[k]

    def get(self, k, default=None):
        return self._d.get(k, default)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes", default="set8,set9")
    ap.add_argument("--data-root", default="/zfsauton/scratch/yixiz/DMSR/paired_catnorm")
    ap.add_argument("--boxsize-mpc-h", type=float, default=100.0)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    # Q1
    ap.add_argument("--r-mults", default="0.5,1,2,4",
                    help="R_c ladder as multiples of the host virial radius")
    ap.add_argument("--r-floor-mpc-h", type=float, default=0.2)
    ap.add_argument("--n-controls", type=int, default=32,
                    help="present HR subhalos sampled as retrieval controls "
                         "(needs <box>_hr_owner.npy; 0 to disable)")
    # Q2
    ap.add_argument("--ng-peak", type=int, default=256,
                    help="grid for the frozen-SR2 density-peak finder")
    ap.add_argument("--smooth-cells", type=float, default=2.0)
    ap.add_argument("--min-overdensity", type=float, default=3.0)
    ap.add_argument("--max-peaks", type=int, default=20000)
    ap.add_argument("--d-nms-mpc-h", type=float, default=0.5)
    ap.add_argument("--out-name", default="candidate_partition")
    args = ap.parse_args(argv)

    out_dir = paths.subdir("audits", args.out_name, create=True)
    boxes = [b.strip() for b in args.boxes.split(",") if b.strip()]
    summaries = [run_box(b, args, out_dir) for b in boxes]
    write_json(out_dir / "summary.json", {"boxes": boxes, "per_box": summaries})
    banner(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
