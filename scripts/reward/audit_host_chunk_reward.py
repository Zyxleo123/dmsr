#!/usr/bin/env python
"""Host-level chunk-attributed catalog reward vs naive bare-chunk Rockstar.

Implements the credit scheme:

* Halo finding stays on the **full periodic box** (existing Rockstar catalogs).
* Each HR host is assigned to exactly one 64^3 tile by its Eulerian centre.
* Generated / SR2 hosts are matched to HR hosts (periodic pos+mass).
* Reward credit is local to the tile; environment is whatever the full-box
  finder already saw -- no bare periodic crop, no face-wrapping artefacts.

Per-host error terms are reported **separately** (not only the sum)::

    miss, false_positive, dlogM (Huber), dN_sub (Huber), W1(mass ratio)

Radial / velocity W1 terms are optional (``--full-phase``); default off, as in
the design note: start with host preservation, count, and mass distribution.

Also pools tiles into 128^3 superchunks and compares against the naive
``audit_chunk_rockstar`` report when present.

    python scripts/reward/audit_host_chunk_reward.py --box set8
    python scripts/reward/audit_host_chunk_reward.py --box set8 --tiles 0,1,8,9
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wasserstein_distance

from _common import (PROJECT_ROOT, add_common_args, banner, bins_of,
                     load_reward_config, write_json)

from cosmo_sr.eval.halo_match import match_hosts
from cosmo_sr.eval.rockstar import HaloCatalog, load_rockstar_ascii
from cosmo_sr.reward import paths
from cosmo_sr.reward.tiles import TileGrid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def huber(x: float, delta: float = 1.0) -> float:
    ax = abs(float(x))
    d = float(delta)
    return 0.5 * ax * ax if ax <= d else d * (ax - 0.5 * d)


def _periodic_delta(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return d - box * np.round(d / box)


def vvir_kms(mvir: float, rvir_kpc_h: float) -> float:
    """Circular velocity (km/s) from Mvir / Rvir; Rvir is Rockstar kpc/h."""
    r = max(float(rvir_kpc_h) * 1e-3, 1e-6)  # Mpc/h
    # G = 4.30091e-9 Mpc/h (Msun/h)^{-1} (km/s)^2
    return float(np.sqrt(4.30091e-9 * float(mvir) / r))


def load_box_catalog(box: str, source: str) -> HaloCatalog:
    root = paths.subdir("halos") / f"{box}__{source}__{source}"
    tag = "hr" if source == "hr" else "base"
    cand = [
        root / f"{tag}_rockstar" / "halos_0.0.ascii",
        root / f"{tag}_rockstar" / "halos_0.0.list",
    ]
    for p in cand:
        if p.is_file():
            return load_rockstar_ascii(p)
    hits = sorted((root / f"{tag}_rockstar").glob("halos*.ascii")) if \
        (root / f"{tag}_rockstar").is_dir() else []
    if not hits:
        raise SystemExit(f"no Rockstar ASCII under {root}")
    return load_rockstar_ascii(hits[0])


def tile_of_centres(pos: np.ndarray, grid: TileGrid) -> np.ndarray:
    """Geometric centre → tile id (C-order). No purity rejection."""
    box = float(grid.boxsize_mpc_h)
    n = grid.n_per_axis
    cell = box / n
    g = np.floor((np.asarray(pos, dtype=np.float64) % box) / cell).astype(np.int64) % n
    return (g[:, 0] * n + g[:, 1]) * n + g[:, 2]


def build_children(cat: HaloCatalog) -> Dict[int, np.ndarray]:
    """parent_id -> row indices of subhalos in ``cat`` (full catalog rows)."""
    parent = np.asarray(cat.parent_ids, dtype=np.int64)
    out: Dict[int, List[int]] = defaultdict(list)
    for r, p in enumerate(parent):
        if p >= 0:
            out[int(p)].append(int(r))
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def host_row_index(hosts: HaloCatalog) -> Dict[int, int]:
    return {int(i): r for r, i in enumerate(np.asarray(hosts.ids, dtype=np.int64))}


def qualifying_subs(cat: HaloCatalog, host_id: int, children: Dict[int, np.ndarray],
                    min_sub_particles: int, min_sub_mass: float) -> np.ndarray:
    rows = children.get(int(host_id))
    if rows is None or rows.size == 0:
        return np.zeros(0, dtype=np.int64)
    num_p = np.asarray(cat.num_p, dtype=np.int64)
    mvir = np.asarray(cat.mvir, dtype=np.float64)
    ok = (num_p[rows] >= int(min_sub_particles)) & (mvir[rows] >= float(min_sub_mass))
    return rows[ok]


def sub_features(cat: HaloCatalog, host_row_in_hosts: int, hosts: HaloCatalog,
                 sub_rows: np.ndarray, box: float
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mass ratios, r/Rvir, v/Vvir for the given sub rows of ``cat``."""
    if sub_rows.size == 0:
        z = np.zeros(0, dtype=np.float64)
        return z, z, z
    mh = float(hosts.mvir[host_row_in_hosts])
    rh = max(float(hosts.rvir[host_row_in_hosts]) * 1e-3, 1e-6)
    vh = max(vvir_kms(mh, float(hosts.rvir[host_row_in_hosts])), 1e-6)
    hp = hosts.pos[host_row_in_hosts]
    hv = hosts.vel[host_row_in_hosts]
    m = np.asarray(cat.mvir[sub_rows], dtype=np.float64)
    ratios = m / max(mh, 1e-30)
    dr = _periodic_delta(cat.pos[sub_rows], hp, box)
    r = np.linalg.norm(dr, axis=-1) / rh
    dv = np.asarray(cat.vel[sub_rows], dtype=np.float64) - hv
    v = np.linalg.norm(dv, axis=-1) / vh
    return ratios, r, v


def w1_safe(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size == 0 or b.size == 0:
        return None
    return float(wasserstein_distance(a, b))


def mass_bin_edges(bins) -> np.ndarray:
    return np.asarray(bins.host_mass_edges, dtype=np.float64)


def estimate_scales(hr_hosts: HaloCatalog, hr_cat: HaloCatalog,
                    hr_children: Dict[int, np.ndarray],
                    min_sub_p: int, min_sub_m: float,
                    host_edges: np.ndarray
                    ) -> Tuple[float, np.ndarray]:
    """s_M from a floor; s_N(M) = max(1, std of N_sub in each host-mass bin)."""
    s_M = 0.15  # dex; typical Rockstar host-mass scatter / match tolerance
    n_bins = len(host_edges) - 1
    counts: List[List[int]] = [[] for _ in range(n_bins)]
    for r, hid in enumerate(np.asarray(hr_hosts.ids, dtype=np.int64)):
        b = int(np.digitize(float(hr_hosts.mvir[r]), host_edges) - 1)
        if not 0 <= b < n_bins:
            continue
        n = int(qualifying_subs(hr_cat, int(hid), hr_children,
                                min_sub_p, min_sub_m).size)
        counts[b].append(n)
    s_N = np.ones(n_bins, dtype=np.float64)
    for b, xs in enumerate(counts):
        if len(xs) >= 2:
            s_N[b] = max(1.0, float(np.std(xs)))
        elif len(xs) == 1:
            s_N[b] = max(1.0, math.sqrt(max(xs[0], 1)))
    return s_M, s_N


# ---------------------------------------------------------------------------
# Per-host / per-tile errors
# ---------------------------------------------------------------------------

def score_tile(
    tile_id: int,
    hr_hosts: HaloCatalog,
    gen_hosts: HaloCatalog,
    hr_cat: HaloCatalog,
    gen_cat: HaloCatalog,
    hr_children: Dict[int, np.ndarray],
    gen_children: Dict[int, np.ndarray],
    hr_tile: np.ndarray,
    gen_tile: np.ndarray,
    match_sr_of_hr: Dict[int, int],
    matched_gen_ids: set,
    *,
    box: float,
    min_sub_p: int,
    min_sub_m: float,
    s_M: float,
    s_N: np.ndarray,
    host_edges: np.ndarray,
    w_M: float,
    w_N: float,
    w_m: float,
    w_r: float,
    w_v: float,
    lam_miss: float,
    lam_fp: float,
    full_phase: bool,
    mhost_min: float,
) -> Dict:
    hr_idx = np.nonzero(
        (hr_tile == tile_id) & (hr_hosts.mvir >= mhost_min)
    )[0]
    gen_idx = np.nonzero(
        (gen_tile == tile_id) & (gen_hosts.mvir >= mhost_min)
    )[0]

    terms = {
        "miss": 0.0,
        "false_positive": 0.0,
        "dlogM": 0.0,
        "dN_sub": 0.0,
        "W1_mass_ratio": 0.0,
        "W1_radial": 0.0,
        "W1_velocity": 0.0,
    }
    n_miss = n_match = n_fp = 0
    n_w1_m = n_w1_r = n_w1_v = 0
    host_rows: List[Dict] = []

    hr_id_to_row = host_row_index(hr_hosts)
    gen_id_to_row = host_row_index(gen_hosts)

    for r in hr_idx:
        hid = int(hr_hosts.ids[r])
        gid = match_sr_of_hr.get(hid, -1)
        if gid < 0 or gid not in gen_id_to_row:
            terms["miss"] += float(lam_miss)
            n_miss += 1
            host_rows.append({"hr_id": hid, "gen_id": None, "status": "miss",
                              "e_h": float(lam_miss)})
            continue
        g = gen_id_to_row[gid]
        n_match += 1
        dlogM = (math.log10(max(float(gen_hosts.mvir[g]), 1e-30))
                 - math.log10(max(float(hr_hosts.mvir[r]), 1e-30)))
        e_M = huber(dlogM / s_M)

        hr_subs = qualifying_subs(hr_cat, hid, hr_children, min_sub_p, min_sub_m)
        gen_subs = qualifying_subs(gen_cat, gid, gen_children, min_sub_p, min_sub_m)
        n_hr, n_gen = int(hr_subs.size), int(gen_subs.size)
        b = int(np.digitize(float(hr_hosts.mvir[r]), host_edges) - 1)
        sNb = float(s_N[b]) if 0 <= b < len(s_N) else 1.0
        dN = (math.log1p(n_gen) - math.log1p(n_hr)) / max(sNb, 1e-6)
        # Normalise by a mild scale so Huber sees O(1); s_N is in count units
        # while the residual is already log — use s_log ≈ s_N / (1+mean).
        s_log = max(sNb / max(1.0 + n_hr, 1.0), 0.25)
        e_N = huber((math.log1p(n_gen) - math.log1p(n_hr)) / s_log)

        e_m = e_r = e_v = 0.0
        used_m = used_r = used_v = False
        if n_hr > 0 and n_gen > 0:
            hr_rat, hr_rad, hr_vel = sub_features(
                hr_cat, r, hr_hosts, hr_subs, box)
            gen_rat, gen_rad, gen_vel = sub_features(
                gen_cat, g, gen_hosts, gen_subs, box)
            wm = w1_safe(hr_rat, gen_rat)
            if wm is not None:
                e_m = float(wm)
                used_m = True
                n_w1_m += 1
            if full_phase:
                wr = w1_safe(hr_rad, gen_rad)
                wv = w1_safe(hr_vel, gen_vel)
                if wr is not None:
                    e_r = float(wr); used_r = True; n_w1_r += 1
                if wv is not None:
                    e_v = float(wv); used_v = True; n_w1_v += 1

        e_h = (w_M * e_M + w_N * e_N
               + (w_m * e_m if used_m else 0.0)
               + (w_r * e_r if used_r else 0.0)
               + (w_v * e_v if used_v else 0.0))
        terms["dlogM"] += w_M * e_M
        terms["dN_sub"] += w_N * e_N
        if used_m:
            terms["W1_mass_ratio"] += w_m * e_m
        if used_r:
            terms["W1_radial"] += w_r * e_r
        if used_v:
            terms["W1_velocity"] += w_v * e_v
        host_rows.append({
            "hr_id": hid, "gen_id": gid, "status": "matched",
            "mvir_hr": float(hr_hosts.mvir[r]), "mvir_gen": float(gen_hosts.mvir[g]),
            "dlogM": dlogM, "n_sub_hr": n_hr, "n_sub_gen": n_gen,
            "e_M": e_M, "e_N": e_N, "e_m": e_m if used_m else None,
            "e_r": e_r if used_r else None, "e_v": e_v if used_v else None,
            "e_h": e_h,
        })

    for g in gen_idx:
        gid = int(gen_hosts.ids[g])
        if gid not in matched_gen_ids:
            terms["false_positive"] += float(lam_fp)
            n_fp += 1

    n_hr = int(hr_idx.size)
    denom = max(1, n_hr)
    E = (terms["miss"] + terms["false_positive"]
         + terms["dlogM"] + terms["dN_sub"]
         + terms["W1_mass_ratio"] + terms["W1_radial"] + terms["W1_velocity"]
         ) / denom
    return {
        "tile_id": int(tile_id),
        "n_hr_hosts": n_hr,
        "n_gen_hosts": int(gen_idx.size),
        "n_matched": n_match,
        "n_miss": n_miss,
        "n_false_positive": n_fp,
        "n_w1_mass_ratio": n_w1_m,
        "E_cat": float(E),
        "terms_sum": {k: float(v) for k, v in terms.items()},
        "terms_mean": {k: float(v) / denom for k, v in terms.items()},
        "hosts": host_rows,
    }


def pool_tiles(tile_scores: Sequence[Dict], tile_ids: Sequence[int]) -> Dict:
    sel = [t for t in tile_scores if int(t["tile_id"]) in set(int(i) for i in tile_ids)]
    if not sel:
        return {"E_cat": float("nan"), "n_hr_hosts": 0, "terms_mean": {}}
    # Re-average with host weighting (sum of numerators / sum of denoms).
    n = sum(int(t["n_hr_hosts"]) for t in sel)
    denom = max(1, n)
    terms = defaultdict(float)
    for t in sel:
        for k, v in t["terms_sum"].items():
            terms[k] += float(v)
    E = sum(terms.values()) / denom
    return {
        "tile_ids": list(map(int, tile_ids)),
        "n_hr_hosts": n,
        "n_matched": sum(int(t["n_matched"]) for t in sel),
        "n_miss": sum(int(t["n_miss"]) for t in sel),
        "n_false_positive": sum(int(t["n_false_positive"]) for t in sel),
        "E_cat": float(E),
        "terms_mean": {k: float(v) / denom for k, v in terms.items()},
        "terms_sum": {k: float(v) for k, v in terms.items()},
    }


def tiles_in_chunk128(chunk_id: int, tile_grid: TileGrid) -> List[int]:
    """Map a chunk_hr=128 id (4^3) onto the eight 64^3 tiles it contains."""
    # chunk128 uses n=4; tile uses n=8.
    n_c = 4
    ix = chunk_id // (n_c * n_c)
    iy = (chunk_id // n_c) % n_c
    iz = chunk_id % n_c
    out = []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                tx, ty, tz = 2 * ix + dx, 2 * iy + dy, 2 * iz + dz
                out.append(int(tile_grid.index(tx, ty, tz)))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--box", default="set8")
    ap.add_argument("--tiles", default=None,
                    help="comma-separated tile ids (default: all non-empty HR tiles)")
    ap.add_argument("--compare-naive", default=None,
                    help="path to audit_chunk_rockstar JSON (default: auto)")
    ap.add_argument("--mhost-min", type=float, default=1e12)
    ap.add_argument("--min-sub-particles", type=int, default=None)
    ap.add_argument("--min-sub-mass", type=float, default=1.2e10)
    ap.add_argument("--lam-miss", type=float, default=1.0)
    ap.add_argument("--lam-fp", type=float, default=0.5)
    ap.add_argument("--w-M", type=float, default=1.0)
    ap.add_argument("--w-N", type=float, default=1.0)
    ap.add_argument("--w-m", type=float, default=1.0)
    ap.add_argument("--w-r", type=float, default=0.5)
    ap.add_argument("--w-v", type=float, default=0.5)
    ap.add_argument("--full-phase", action="store_true",
                    help="include radial and velocity W1 terms")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-host-rows", type=int, default=20,
                    help="per-tile host detail rows kept in the JSON")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    bins = bins_of(cfg)
    box_mpc = float(cfg["data"]["boxsize_mpc_h"])
    tile_hr = int(cfg.get("tiles", {}).get("tile_hr", 64))
    grid = TileGrid(ng_hr=int(cfg["data"]["ng_hr"]), tile_hr=tile_hr,
                    boxsize_mpc_h=box_mpc)
    min_sub_p = int(args.min_sub_particles
                    if args.min_sub_particles is not None
                    else bins.min_sub_particles)
    host_edges = mass_bin_edges(bins)

    banner(f"host-level chunk-attributed reward: box={args.box} "
           f"tile_hr={tile_hr} ({grid.n_tiles} tiles) "
           f"phase={'full' if args.full_phase else 'M+N+massW1'}")

    t0 = time.time()
    hr_cat = load_box_catalog(args.box, "hr")
    base_cat = load_box_catalog(args.box, "base")
    print(f"  catalogs: HR={hr_cat.n} rows, base={base_cat.n} rows "
          f"({time.time() - t0:.1f}s)", flush=True)

    hr_hosts, base_hosts = hr_cat.hosts(), base_cat.hosts()
    hr_children = build_children(hr_cat)
    base_children = build_children(base_cat)
    hr_tile = tile_of_centres(hr_hosts.pos, grid)
    base_tile = tile_of_centres(base_hosts.pos, grid)

    t1 = time.time()
    match = match_hosts(hr_cat, base_cat, boxsize_mpc_h=box_mpc,
                        mhost_min=float(args.mhost_min))
    print(f"  matched {int(np.count_nonzero(match.sr_ids >= 0))}/"
          f"{match.hr_ids.size} HR hosts ({time.time() - t1:.1f}s)", flush=True)
    match_sr_of_hr = {int(h): int(s) for h, s in zip(match.hr_ids, match.sr_ids)
                      if int(s) >= 0}
    matched_gen_ids = set(match_sr_of_hr.values())

    s_M, s_N = estimate_scales(hr_hosts, hr_cat, hr_children,
                               min_sub_p, float(args.min_sub_mass), host_edges)
    print(f"  scales: s_M={s_M:.3g} dex; s_N(bins)={np.round(s_N, 2).tolist()}",
          flush=True)

    if args.tiles:
        tile_ids = [int(x) for x in args.tiles.split(",") if x.strip() != ""]
    else:
        # All tiles that own at least one HR host above the mass floor.
        mask = hr_hosts.mvir >= float(args.mhost_min)
        tile_ids = sorted(set(int(t) for t in hr_tile[mask]))

    scores: List[Dict] = []
    for tid in tile_ids:
        sc = score_tile(
            tid, hr_hosts, base_hosts, hr_cat, base_cat,
            hr_children, base_children, hr_tile, base_tile,
            match_sr_of_hr, matched_gen_ids,
            box=box_mpc, min_sub_p=min_sub_p,
            min_sub_m=float(args.min_sub_mass),
            s_M=s_M, s_N=s_N, host_edges=host_edges,
            w_M=float(args.w_M), w_N=float(args.w_N), w_m=float(args.w_m),
            w_r=float(args.w_r), w_v=float(args.w_v),
            lam_miss=float(args.lam_miss), lam_fp=float(args.lam_fp),
            full_phase=bool(args.full_phase),
            mhost_min=float(args.mhost_min),
        )
        # Trim host detail for the JSON.
        sc["hosts"] = sc["hosts"][: int(args.max_host_rows)]
        scores.append(sc)

    # Global mean over scored tiles (host-weighted).
    global_pool = pool_tiles(scores, tile_ids)

    # 128^3 superchunks overlapping the naive audit (chunks 0..3).
    naive_path = Path(args.compare_naive) if args.compare_naive else \
        paths.AUDITS("chunk_rockstar") / "set8_chunks0-1-2-3.json"
    naive = None
    if naive_path.is_file():
        naive = json.loads(naive_path.read_text())

    comparisons: List[Dict] = []
    for cid in range(4):
        tids = tiles_in_chunk128(cid, grid)
        pooled = pool_tiles(scores, tids)
        row = {
            "chunk128_id": cid,
            "tile_ids": tids,
            "host_attributed": {
                "E_cat_base_vs_hr": pooled["E_cat"],
                "terms_mean": pooled["terms_mean"],
                "n_hr_hosts": pooled["n_hr_hosts"],
                "n_matched": pooled["n_matched"],
                "n_miss": pooled["n_miss"],
                "n_false_positive": pooled["n_false_positive"],
            },
        }
        if naive is not None:
            # Find HR/base naive rows for this chunk.
            by = {(r["chunk_id"], r["source"]): r for r in naive.get("rows", [])}
            hr_n = by.get((cid, "hr"), {}).get("chunk_rockstar")
            base_n = by.get((cid, "base"), {}).get("chunk_rockstar")
            if hr_n and base_n:
                row["naive_chunk_rockstar"] = {
                    "hr_R_occ": hr_n["R_occ"],
                    "base_R_occ": base_n["R_occ"],
                    "dR_occ_hr_minus_base": hr_n["R_occ"] - base_n["R_occ"],
                    "hr_n_sub": hr_n["n_sub_total"],
                    "base_n_sub": base_n["n_sub_total"],
                    "d_n_sub": hr_n["n_sub_total"] - base_n["n_sub_total"],
                    "hr_n_host": hr_n["n_host_total"],
                    "base_n_host": base_n["n_host_total"],
                }
        comparisons.append(row)

    # Term dominance across non-empty tiles.
    nonempty = [t for t in scores if t["n_hr_hosts"] > 0]
    term_means = defaultdict(list)
    for t in nonempty:
        for k, v in t["terms_mean"].items():
            term_means[k].append(float(v))
    term_summary = {
        k: {"mean": float(np.mean(vs)), "median": float(np.median(vs)),
            "p90": float(np.percentile(vs, 90))}
        for k, vs in term_means.items() if vs
    }

    out_dir = paths.AUDITS("host_chunk_reward", create=True)
    out_path = Path(args.out) if args.out else out_dir / f"{args.box}_host_chunk_reward.json"
    report = {
        "box": args.box,
        "tile_hr": tile_hr,
        "n_tiles": grid.n_tiles,
        "method": (
            "full-box Rockstar; centre→64^3 attribution; host match; "
            "per-host catalog error with separated terms"
        ),
        "contrast_with_naive": (
            "Naive audit_chunk_rockstar crops a Lagrangian cube, sets PERIODIC=1 "
            "on that cube alone, and scores a binned Mahalanobis reward. This "
            "audit keeps full-box (periodic) finding, scores only core-tile hosts, "
            "and reports miss/fp/dlogM/dN/W1 terms separately."
        ),
        "scales": {"s_M_dex": s_M, "s_N_per_host_bin": s_N.tolist()},
        "weights": {
            "lam_miss": args.lam_miss, "lam_fp": args.lam_fp,
            "w_M": args.w_M, "w_N": args.w_N, "w_m": args.w_m,
            "w_r": args.w_r, "w_v": args.w_v, "full_phase": bool(args.full_phase),
        },
        "match": {
            "n_hr_hosts": int(hr_hosts.n),
            "n_base_hosts": int(base_hosts.n),
            "n_matched": int(np.count_nonzero(match.sr_ids >= 0)),
            "method": match.method,
        },
        "global": global_pool,
        "term_summary_over_tiles": term_summary,
        "tiles": [{k: v for k, v in t.items() if k != "hosts"} for t in scores],
        "tile_host_samples": {str(t["tile_id"]): t["hosts"] for t in scores
                              if t["hosts"]},
        "vs_naive_chunk128": comparisons,
        "naive_report": str(naive_path) if naive is not None else None,
    }
    write_json(out_path, report)

    print(f"\n  global E_cat(base|HR) = {global_pool['E_cat']:.4g} "
          f"over {global_pool['n_hr_hosts']} HR hosts", flush=True)
    print("  term means (host-weighted global):", flush=True)
    for k, v in sorted(global_pool["terms_mean"].items(), key=lambda kv: -kv[1]):
        if v == 0 and k in ("W1_radial", "W1_velocity") and not args.full_phase:
            continue
        print(f"    {k:16s}  {v:.4g}", flush=True)
    print("\n  vs naive chunk-Rockstar (128^3 chunks 0..3):", flush=True)
    for c in comparisons:
        ha = c["host_attributed"]
        tm = ha.get("terms_mean") or {}
        line = (f"    chunk {c['chunk128_id']}: E_cat={ha['E_cat_base_vs_hr']:.3g} "
                f"(miss={tm.get('miss', 0):.3g} "
                f"dN={tm.get('dN_sub', 0):.3g} "
                f"W1m={tm.get('W1_mass_ratio', 0):.3g} "
                f"dM={tm.get('dlogM', 0):.3g} "
                f"fp={tm.get('false_positive', 0):.3g})")
        nv = c.get("naive_chunk_rockstar")
        if nv:
            line += (f"  | naive dR_occ={nv['dR_occ_hr_minus_base']:+.3g} "
                     f"d_nsub={nv['d_n_sub']:+d}")
        print(line, flush=True)
    print(f"\n  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
