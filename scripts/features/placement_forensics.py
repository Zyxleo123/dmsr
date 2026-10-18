#!/usr/bin/env python
"""Placement forensics: what IS the inverted subhalo radial distribution?

Every gate since 2026-09-14 reports where the region's subhalos sit in their
hosts, and every tuned arm is inverted: HR's median is 1.13 R_vir with 2.5%
inside 0.25 R_vir, `coarse_soft` 0.35 with 40% there (6781 vs HR's 593), and
the 1-1.5 R_vir band holds 1129 against HR's 6545. Every lever tried on that
number -- shell terms, radial prior, relocation, host channel, isolation --
acted on supervised member sets, and none moved it out of sample.

The number is a count of Rockstar objects, and two measurements on ONE host
(168880, 200-2000p) say the count may not be what it looks like: the arms'
subhalos there are fragments of 5-17 blobs (SUBFIND finds the blob count), and
they are ~3x too cold. This script asks, region-wide and for every catalog on
disk, which of three things the inversion is made of:

  1. CORE FRAGMENTS -- threshold-scale pieces of a few cold blobs, co-moving
     with the host, that Rockstar's 6D linking splits and SUBFIND does not.
  2. REAL MISPLACED OBJECTS -- self-bound satellites deeper than HR's.
  3. AN OUTSKIRT MASS SHIFT -- HR's outer objects present but under-massive,
     so they fall under the 50p floor.

Per catalog, all in the gate's EDITED REGION (subhalos whose parent centre is
inside the dilated spliced tiles, the gate's own definition), it writes:

* ``radial_by_mass``  -- counts per r/R_vir bin x subhalo num_p bin, with the
  gate's convention (immediate parent) and host-centric (top-level host).
* ``structures``      -- subhalos linked when one centre lies inside the
  other's r_vir, counted per radial bin (the fragment test of
  docs/sr2_subfind_crosscheck.md, region-wide).
* ``kinematics``      -- per radial bin x mass bin medians of T/|U|,
  vrms/vmax, mbound/mvir, Xoff/r_vir and |v_sub - v_host| / vrms_host.
* ``outskirt_cumulative`` -- N(>= num_p) at 0.75-2 and 1-1.5 R_vir, so a
  mass shift reads as a horizontal offset of the same curve.
* ``by_host_mass``    -- median physical radius per host-mass bin, with and
  without the core (the "flat in host mass" finding, core removed).
* ``host_match``      -- this catalog's host at each massive HR host: offset,
  r_vir ratio, num_p ratio (an R_vir that swelled shrinks every r/R_vir).

SUBFIND npz catalogs (``subfind_catalog.py``) get the same radial and
kinematic tables for their satellites, with R_vir from the FOF group mass on
the Bryan-Norman overdensity Rockstar's "vir" also uses.

Finally a CROSS-ARM table joins each member-gather arm's catalog numbers to
the held-out field diagnostics its own trainer logged (`summary.json`), and
reports Spearman correlations: if the core excess is velocity cooling, it
tracks ``velhighk_ratio`` across ~50 arms.

Catalog geometry only: no field is read, so a whole run is minutes of CPU.
The gate's own radial counts are reproduced as a check (``--check-verdict``)
before anything else is trusted.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402
from gather_holdout_figures_data import (  # noqa: E402
    catalog_path, in_tile_region, tile_boxes,
)
from subhalo_radial import RVIR_BINS  # noqa: E402  (one definition of the bins)
from cosmo_sr.reward.tiles import TileGrid  # noqa: E402

BOX = 100.0
#: Subhalo num_p bins. Rockstar's MIN_HALO_OUTPUT_SIZE is 20, so the first bin
#: is everything the gate's radial distribution counts below the 50p floor.
SUB_MASS_EDGES = (20, 50, 100, 200, 1000, 10 ** 12)
HOST_MASS_EDGES = (200, 1000, 5000, 20000, 100000, 10 ** 12)
CUM_THRESHOLDS = (20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 1000)
#: Rockstar ASCII columns this script needs (lower-cased header names).
ROCKSTAR_COLUMNS = ("id", "num_p", "mvir", "mbound_vir", "rvir", "vmax",
                    "vrms", "x", "y", "z", "vx", "vy", "vz", "xoff", "voff",
                    "t/|u|", "idx", "i_so")
#: Held-out field diagnostics logged by finetune_member_gather.py.
DIAG_FIELDS = ("velhighk_ratio", "velhighk_ratio_min", "vel_rms_ratio",
               "highk_ratio", "highk_ratio_max", "virial", "bound_hard",
               "iso_ratio", "r_rms_over_hr", "sigma_v_over_hr", "low_k",
               "centre_offset_radii", "soft_hr_fraction")
RHO_CRIT = 2.77536627e11          # (Msun/h) / (Mpc/h)^3
OMEGA_M = 0.2814                  # the simulations' cosmology (catalog header)


def r_labels() -> List[str]:
    lab = [f"{lo:g}-{hi:g}" for lo, hi in zip(RVIR_BINS[:-1], RVIR_BINS[1:])]
    return lab + [f">{RVIR_BINS[-1]:g}"]


def m_labels(edges: Sequence[int]) -> List[str]:
    return [f"{lo}-{'inf' if hi >= 10 ** 12 else hi}p"
            for lo, hi in zip(edges[:-1], edges[1:])]


def r_bin(x: np.ndarray) -> np.ndarray:
    """Index into the 9 radial bins (8 intervals + beyond the last edge).

    Same rule as gate_verdict_vs_baseline.radial_distribution.
    """
    return np.digitize(x, RVIR_BINS[1:], right=False)


def m_bin(n: np.ndarray, edges: Sequence[int]) -> np.ndarray:
    """Index of each num_p in ``edges``; -1 below the first edge."""
    return np.searchsorted(np.asarray(edges), n, side="right") - 1


def bryan_norman_delta_c(omega_m: float) -> float:
    """Virial overdensity relative to CRITICAL at z=0 (Bryan & Norman 1998)."""
    x = omega_m - 1.0
    return 18.0 * np.pi ** 2 + 82.0 * x - 39.0 * x ** 2


def rvir_from_mass(m_msun_h: np.ndarray, omega_m: float = OMEGA_M) -> np.ndarray:
    """R_vir in Mpc/h of a mass in Msun/h, on the Bryan-Norman overdensity."""
    dc = bryan_norman_delta_c(omega_m)
    return np.cbrt(3.0 * np.asarray(m_msun_h, dtype=np.float64)
                   / (4.0 * np.pi * dc * RHO_CRIT))


def min_image(d: np.ndarray, box: float = BOX) -> np.ndarray:
    return d - box * np.round(d / box)


# --------------------------------------------------------------------------
# Rockstar
# --------------------------------------------------------------------------

def read_rockstar(path: str) -> Dict[str, np.ndarray]:
    """The columns in ROCKSTAR_COLUMNS plus ``parent_id``, by header name.

    The parent mapping is load_rockstar_ascii's: ``i_so`` is the internal
    index of the parent, remapped to the printed ``id`` through ``idx``. The
    column count is checked against the header -- a silent shift would
    misread every column after it.
    """
    import pandas as pd
    header = None
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            toks = re.split(r"\s+", line.lstrip("#").strip().lower())
            if toks and toks[0] == "id" and header is None:
                header = [t.strip("()") for t in toks]
    if header is None:
        raise SystemExit(f"{path}: no '#id ...' header line")
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="c")
    if df.shape[1] != len(header):
        raise SystemExit(f"{path}: {df.shape[1]} data columns vs "
                         f"{len(header)} header names -- refusing to guess")
    col = {name: i for i, name in enumerate(header)}
    missing = [c for c in ROCKSTAR_COLUMNS if c not in col]
    if missing:
        raise SystemExit(f"{path}: header lacks {missing}")
    out = {c: df.iloc[:, col[c]].to_numpy() for c in ROCKSTAR_COLUMNS}
    for c in ("id", "num_p", "idx", "i_so"):
        out[c] = out[c].astype(np.int64)
    idx, i_so, ids = out["idx"], out["i_so"], out["id"]
    order = np.argsort(idx)
    j = np.clip(np.searchsorted(idx, i_so, sorter=order), 0, max(idx.size - 1, 0))
    row = order[j] if idx.size else j
    hit = (i_so >= 0) & (idx[row] == i_so) if idx.size else np.zeros(0, bool)
    out["parent_id"] = np.where(hit, ids[row], -1).astype(np.int64)
    out["pos"] = np.stack([out["x"], out["y"], out["z"]], axis=1).astype(np.float64)
    out["vel"] = np.stack([out["vx"], out["vy"], out["vz"]], axis=1).astype(np.float64)
    return out


def parent_rows(ids: np.ndarray, parent_id: np.ndarray) -> np.ndarray:
    """Row of each object's parent, -1 for a host or a missing parent."""
    order = np.argsort(ids)
    j = np.clip(np.searchsorted(ids, parent_id, sorter=order), 0, ids.size - 1)
    row = order[j]
    ok = (parent_id >= 0) & (ids[row] == parent_id)
    return np.where(ok, row, -1)


def top_rows(prow: np.ndarray, max_depth: int = 32) -> np.ndarray:
    """Row of each object's TOP-LEVEL host (itself for a host).

    Walks the parent pointers; an object whose chain does not end within
    ``max_depth`` (a cycle would be a catalog bug) is returned as -1.
    """
    top = np.arange(prow.size)
    for _ in range(max_depth):
        nxt = prow[top]
        moving = nxt >= 0
        if not moving.any():
            return top
        top = np.where(moving, nxt, top)
    return np.where(prow[top] >= 0, -1, top)


def link_structures(pos: np.ndarray, rad: np.ndarray, box: float = BOX
                    ) -> np.ndarray:
    """Connected-component label per object: i~j when |x_i - x_j| < max(r_i, r_j).

    That is "one centre lies inside the other's sphere", the linking rule of
    the fragment test. Periodic. ``rad`` in the same units as ``pos``.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    n = pos.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    tree = cKDTree(np.mod(pos, box), boxsize=box)
    pairs = tree.query_pairs(r=float(np.max(rad)), output_type="ndarray")
    if pairs.size:
        d = np.linalg.norm(min_image(pos[pairs[:, 0]] - pos[pairs[:, 1]], box),
                           axis=1)
        keep = d < np.maximum(rad[pairs[:, 0]], rad[pairs[:, 1]])
        pairs = pairs[keep]
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                   shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    _, lab = connected_components(g, directed=False)
    return lab.astype(np.int64)


def representatives(labels: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Index of the heaviest member of each component (ties: lowest index)."""
    if labels.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((-weight, labels))       # by label, heaviest first
    first = np.ones(order.size, dtype=bool)
    first[1:] = labels[order][1:] != labels[order][:-1]
    return order[first]


def cumulative_counts(num_p: np.ndarray, thresholds=CUM_THRESHOLDS) -> List[int]:
    return [int(np.sum(num_p >= t)) for t in thresholds]


def median_or_nan(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def binned_medians(rb: np.ndarray, mb: np.ndarray, values: Dict[str, np.ndarray],
                   n_r: int, n_m: int) -> Dict[str, List[List[float]]]:
    """Median of each value per (radial bin, mass bin) cell, plus the count."""
    out: Dict[str, List[List[float]]] = {"n": []}
    for k in values:
        out[k] = []
    for i in range(n_r):
        row_n, rows = [], {k: [] for k in values}
        for j in range(n_m):
            m = (rb == i) & (mb == j)
            row_n.append(int(m.sum()))
            for k, v in values.items():
                rows[k].append(median_or_nan(v[m]))
        out["n"].append(row_n)
        for k in values:
            out[k].append(rows[k])
    return out


def radial_table(x: np.ndarray, num_p: np.ndarray, edges=SUB_MASS_EDGES
                 ) -> Dict[str, object]:
    rb, mb = r_bin(x), m_bin(num_p, edges)
    n_r, n_m = len(RVIR_BINS), len(edges) - 1
    counts = np.zeros((n_r, n_m), dtype=np.int64)
    ok = (mb >= 0) & (mb < n_m)
    np.add.at(counts, (rb[ok], mb[ok]), 1)
    return {"counts": counts.tolist(),
            "total_by_r": counts.sum(axis=1).tolist(),
            "median_x": median_or_nan(x),
            "median_x_ge50": median_or_nan(x[num_p >= 50]),
            "n": int(ok.sum())}


def analyse_rockstar(cat: Dict[str, np.ndarray], boxes: np.ndarray,
                     dilate: float, hr_hosts: Optional[np.ndarray] = None
                     ) -> Dict[str, object]:
    ids, num_p = cat["id"], cat["num_p"]
    pos, vel = cat["pos"], cat["vel"]
    rvir = cat["rvir"] / 1000.0                        # kpc/h -> Mpc/h
    prow = parent_rows(ids, cat["parent_id"])
    is_sub = cat["parent_id"] >= 0
    n_orphan = int(np.sum(is_sub & (prow < 0)))
    sub = np.where(is_sub & (prow >= 0))[0]
    par = prow[sub]
    inreg = in_tile_region(pos[par], boxes, dilate=dilate)
    sub, par = sub[inreg], par[inreg]
    good = rvir[par] > 0
    sub, par = sub[good], par[good]
    d = min_image(pos[sub] - pos[par])
    r = np.linalg.norm(d, axis=1)
    x = r / rvir[par]                                   # the gate's convention
    trow = top_rows(prow)
    top = trow[sub]
    okt = top >= 0
    xt = np.full(sub.size, np.nan)
    rt = np.full(sub.size, np.nan)
    dt = min_image(pos[sub[okt]] - pos[top[okt]])
    rt[okt] = np.linalg.norm(dt, axis=1)
    xt[okt] = rt[okt] / rvir[top[okt]]
    npp = num_p[sub]

    res: Dict[str, object] = {
        "n_objects": int(ids.size), "n_orphan": n_orphan,
        "n_region_subs": int(sub.size),
        "n_region_subs_ge50": int(np.sum(npp >= 50)),
        "radial_by_mass": radial_table(x, npp),
        "radial_by_mass_top": radial_table(xt[okt], npp[okt]),
    }

    # --- fragment linking among the region's subhalos ----------------------
    struct = {}
    for floor in (20, 50):
        m = npp >= floor
        lab = link_structures(pos[sub[m]], rvir[sub[m]])
        rep = representatives(lab, npp[m].astype(np.float64))
        rb_rep = r_bin(x[m][rep])
        rb_all = r_bin(x[m])
        n_r = len(RVIR_BINS)
        n_struct = np.bincount(rb_rep, minlength=n_r)[:n_r]
        n_frag = np.bincount(rb_all, minlength=n_r)[:n_r]
        sizes = np.bincount(lab)
        struct[f"ge{floor}"] = {
            "n_subs": int(m.sum()), "n_structures": int(rep.size),
            "structures_by_r": n_struct.tolist(),
            "subs_by_r": n_frag.tolist(),
            "frag_per_structure_by_r": [float(a / b) if b else float("nan")
                                        for a, b in zip(n_frag, n_struct)],
            "largest_structure": int(sizes.max()) if sizes.size else 0,
            "frac_in_multi": float(np.mean(sizes[lab] > 1)) if lab.size else 0.0,
            "median_x_structures": median_or_nan(x[m][rep]),
        }
    res["structures"] = struct

    # --- per-object kinematics by radius and mass --------------------------
    vrms_host = cat["vrms"][par]
    dv = np.linalg.norm(vel[sub] - vel[par], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = {
            "t_over_u": cat["t/|u|"][sub].astype(np.float64),
            "vrms_over_vmax": cat["vrms"][sub] / cat["vmax"][sub],
            "bound_frac": cat["mbound_vir"][sub] / cat["mvir"][sub],
            "xoff_over_rvir": cat["xoff"][sub] / cat["rvir"][sub],
            "dv_over_host_vrms": dv / vrms_host,
        }
    rb, mb = r_bin(x), m_bin(npp, SUB_MASS_EDGES)
    res["kinematics"] = binned_medians(rb, mb, vals, len(RVIR_BINS),
                                       len(SUB_MASS_EDGES) - 1)
    # Co-moving fraction: an orbiting satellite near a host's centre moves at
    # ~ the host's dispersion; a fragment of the host's own core does not.
    comove = vals["dv_over_host_vrms"] < 0.3
    res["comoving_frac_by_r"] = [
        float(np.mean(comove[rb == i])) if np.any(rb == i) else float("nan")
        for i in range(len(RVIR_BINS))]

    # --- outskirts: is the outer population present but under-massive? -----
    res["outskirt_cumulative"] = {
        "thresholds": list(CUM_THRESHOLDS),
        "0.75-2": cumulative_counts(npp[(x >= 0.75) & (x < 2.0)]),
        "1-1.5": cumulative_counts(npp[(x >= 1.0) & (x < 1.5)]),
        "0-0.25": cumulative_counts(npp[x < 0.25]),
    }

    # --- physical radius by TOP-LEVEL host mass, core in and out -----------
    hm = m_bin(num_p[top[okt]], HOST_MASS_EDGES)
    rt_ok, xt_ok = rt[okt], xt[okt]
    bhm = []
    for j in range(len(HOST_MASS_EDGES) - 1):
        m = hm == j
        mo = m & (xt_ok >= 0.25)
        bhm.append({"n": int(m.sum()), "median_r": median_or_nan(rt_ok[m]),
                    "n_outside_core": int(mo.sum()),
                    "median_r_outside_core": median_or_nan(rt_ok[mo]),
                    "core_frac": float(np.mean(xt_ok[m] < 0.25)) if m.any()
                    else float("nan")})
    res["by_host_mass"] = bhm

    # --- this catalog's host at each massive HR host -----------------------
    if hr_hosts is not None and hr_hosts.size:
        from scipy.spatial import cKDTree
        hosts = np.where((cat["parent_id"] < 0) & (num_p >= 200))[0]
        tree = cKDTree(np.mod(pos[hosts], BOX), boxsize=BOX)
        dist, k = tree.query(np.mod(hr_hosts[:, :3], BOX))
        hrow = hosts[k]
        res["host_match"] = {
            "n": int(hr_hosts.shape[0]),
            "median_offset_over_rvir_hr": float(np.median(dist / hr_hosts[:, 3])),
            "median_rvir_ratio": float(np.median(rvir[hrow] / hr_hosts[:, 3])),
            "median_num_p_ratio": float(np.median(num_p[hrow] / hr_hosts[:, 4])),
            "frac_matched_within_0.3rvir": float(np.mean(dist < 0.3 * hr_hosts[:, 3])),
        }
    return res


# --------------------------------------------------------------------------
# SUBFIND
# --------------------------------------------------------------------------

def analyse_subfind(npz_path: str, boxes: np.ndarray, dilate: float
                    ) -> Dict[str, object]:
    z = np.load(npz_path)
    sat = np.where(z["is_satellite"])[0]
    g = z["SubhaloGroupNr"][sat].astype(np.int64)
    gpos = z["GroupPos"][g].astype(np.float64)
    inreg = in_tile_region(gpos, boxes, dilate=dilate)
    sat, g, gpos = sat[inreg], g[inreg], gpos[inreg]
    rv = rvir_from_mass(z["GroupMass"][g])
    r = np.linalg.norm(min_image(z["SubhaloPos"][sat].astype(np.float64) - gpos),
                       axis=1)
    x = r / rv
    npp = z["SubhaloLen"][sat].astype(np.int64)
    cen = z["GroupFirstSub"][g].astype(np.int64)
    dv = np.linalg.norm(z["SubhaloVel"][sat] - z["SubhaloVel"][cen], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = {"veldisp_over_vmax": z["SubhaloVelDisp"][sat] / z["SubhaloVmax"][sat],
                "dv_over_central_veldisp": dv / z["SubhaloVelDisp"][cen],
                "halfmass_over_rvir_host": z["SubhaloHalfmassRad"][sat] / rv}
    rb, mb = r_bin(x), m_bin(npp, SUB_MASS_EDGES)
    glen = z["GroupLen"][g].astype(np.int64)
    hm = m_bin(glen, HOST_MASS_EDGES)
    bhm = []
    for j in range(len(HOST_MASS_EDGES) - 1):
        m = hm == j
        mo = m & (x >= 0.25)
        bhm.append({"n": int(m.sum()), "median_r": median_or_nan(r[m]),
                    "n_outside_core": int(mo.sum()),
                    "median_r_outside_core": median_or_nan(r[mo]),
                    "core_frac": float(np.mean(x[m] < 0.25)) if m.any()
                    else float("nan")})
    return {
        "n_region_sats": int(sat.size),
        "n_region_sats_ge50": int(np.sum(npp >= 50)),
        "n_groups_total": int(z["GroupLen"].size),
        "radial_by_mass": radial_table(x, npp),
        "kinematics": binned_medians(rb, mb, vals, len(RVIR_BINS),
                                     len(SUB_MASS_EDGES) - 1),
        "outskirt_cumulative": {
            "thresholds": list(CUM_THRESHOLDS),
            "0.75-2": cumulative_counts(npp[(x >= 0.75) & (x < 2.0)]),
            "1-1.5": cumulative_counts(npp[(x >= 1.0) & (x < 1.5)]),
            "0-0.25": cumulative_counts(npp[x < 0.25]),
        },
        "by_host_mass": bhm,
    }


# --------------------------------------------------------------------------
# cross-arm table
# --------------------------------------------------------------------------

def history_row(summary: Dict, step: Optional[int]) -> Optional[Dict]:
    """The held-out eval row at ``step`` (the last at or before it), else last."""
    hist = summary.get("history") or []
    if not hist:
        return None
    if step is None:
        return hist[-1]
    rows = [h for h in hist if int(h.get("step", -1)) <= step]
    return rows[-1] if rows else hist[0]


def spearman(a: Sequence[float], b: Sequence[float]) -> Tuple[float, int, float]:
    """Spearman rho, n and a two-sided permutation p (2000 shuffles, seed 0)."""
    from scipy.stats import rankdata
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = int(a.size)
    if n < 5:
        return float("nan"), n, float("nan")
    ra, rb = rankdata(a), rankdata(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(0)
    null = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1]
                     for _ in range(2000)])
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (null.size + 1))
    return rho, n, p


def discover_arms(reward_root: Path, box: str) -> List[Tuple[str, str, Optional[int]]]:
    """(label, rockstar tag, history step) for every gated member-gather export.

    A ``holdout_<box>_step<N>`` export was cut from ``tuned_step<N>.pt`` and is
    matched to that step's eval row, not the final one.
    """
    out = []
    for exp in sorted(glob.glob(str(reward_root / "member_gather" / "*"
                                    / f"holdout_{box}*" / "export.json"))):
        p = Path(exp)
        arm, sub = p.parent.parent.name, p.parent.name
        m = re.fullmatch(rf"holdout_{box}(?:_step(\d+))?", sub)
        if not m:
            continue
        step = int(m.group(1)) if m.group(1) else None
        tag = f"mgho_{arm}{'_step' + m.group(1) if step else ''}_{box}"
        label = arm + (f"@{step}" if step else "")
        out.append((label, tag, step))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reward-root", default="/zfsauton/scratch/yixiz/DMSR/dmsr_reward")
    ap.add_argument("--box", default="set9")
    ap.add_argument("--region-export", required=True,
                    help="export.json whose tile list defines the edited region "
                         "for EVERY catalog (one common region)")
    ap.add_argument("--dilate", type=float, default=2.0)
    ap.add_argument("--arms", default="auto",
                    help="'auto' = every gated member-gather export on --box; "
                         "or a comma list of label=rockstar_tag")
    ap.add_argument("--extra", default="",
                    help="comma list of name=path/to/halos_*.ascii (swap / m2m "
                         "catalogs); a glob is allowed")
    ap.add_argument("--subfind", default="auto",
                    help="'auto' = every subfind/<box>__*/subfind_catalog.npz; "
                         "or a comma list of name=path.npz; '' for none")
    ap.add_argument("--check-verdict", default="",
                    help="a gate verdict json; its hr/base radial counts must be "
                         "reproduced exactly or the run stops")
    ap.add_argument("--hr-host-min-num-p", type=int, default=20000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    R = Path(a.reward_root)
    export = json.loads(Path(a.region_export).read_text())
    tiles = [int(t) for t in export["tiles"]] if "tiles" in export else sorted(
        {int(t) for h in export["per_host"] for t in h["tiles"]})
    boxes = tile_boxes(tiles, TileGrid())
    print(f"region: {len(tiles)} tiles from {a.region_export}, dilate {a.dilate}")

    # (name, path, history step, member-gather arm, kind)
    catalogs: List[Tuple[str, str, Optional[int], Optional[str], str]] = [
        ("hr", catalog_path(R, "hr", a.box), None, None, "reference"),
        ("base", catalog_path(R, "base", a.box), None, None, "reference"),
    ]
    if a.arms == "auto":
        for label, tag, step in discover_arms(R, a.box):
            try:
                catalogs.append((label, catalog_path(R, "cand", a.box, tag),
                                 step, label.split("@")[0], "arm"))
            except SystemExit:
                print(f"    {label}: no Rockstar catalog for tag {tag} -- skipped")
    elif a.arms.strip():
        for item in [s for s in a.arms.split(",") if s.strip()]:
            label, tag = item.split("=", 1)
            catalogs.append((label, catalog_path(R, "cand", a.box, tag), None,
                             label.split("@")[0], "arm"))
    for item in [s for s in a.extra.split(",") if s.strip()]:
        name, pat = item.split("=", 1)
        hits = sorted(glob.glob(pat))
        if not hits:
            print(f"    extra {name}: nothing matches {pat} -- skipped")
            continue
        catalogs.append((name, hits[0], None, None, "extra"))

    hr = read_rockstar(catalogs[0][1])
    hrv = hr["rvir"] / 1000.0
    big = np.where((hr["parent_id"] < 0) & (hr["num_p"] >= a.hr_host_min_num_p)
                   & in_tile_region(hr["pos"], boxes, dilate=a.dilate))[0]
    hr_hosts = np.column_stack([hr["pos"][big], hrv[big], hr["num_p"][big]])
    print(f"{big.size} HR hosts >= {a.hr_host_min_num_p}p in the region")

    results: Dict[str, Dict] = {}
    for name, path, step, arm, kind in catalogs:
        print(f"--- {name}: {path}", flush=True)
        cat = hr if name == "hr" else read_rockstar(path)
        res = analyse_rockstar(cat, boxes, a.dilate, hr_hosts)
        res["path"], res["step"], res["arm"], res["kind"] = path, step, arm, kind
        results[name] = res
        rb = res["radial_by_mass"]["total_by_r"]
        print(f"    region subs {res['n_region_subs']} (>=50p "
              f"{res['n_region_subs_ge50']}), core {rb[0]}, 1-1.5 {rb[4]}, "
              f"median x {res['radial_by_mass']['median_x']:.3f}, structures>=20 "
              f"{res['structures']['ge20']['n_structures']}", flush=True)

    if a.check_verdict:
        v = json.loads(Path(a.check_verdict).read_text())["radial_distribution"]
        for k in ("hr", "base"):
            mine = results[k]["radial_by_mass"]["total_by_r"]
            if list(mine) != list(v[k]["counts"]):
                raise SystemExit(f"CHECK FAILED: {k} radial counts {mine} != the "
                                 f"gate's {v[k]['counts']}; nothing here is "
                                 f"comparable with the gates")
        print("check: hr/base radial counts reproduce the gate's exactly")

    sf: Dict[str, Dict] = {}
    if a.subfind.strip() == "auto":
        items = [(Path(p).parent.name.split("__", 1)[1], p) for p in sorted(
            glob.glob(str(R / "subfind" / f"{a.box}__*" / "subfind_catalog.npz")))]
    else:
        items = [tuple(s.split("=", 1)) for s in a.subfind.split(",") if s.strip()]
    for name, p in items:
        if not Path(p).is_file():
            print(f"--- SUBFIND {name}: no catalog at {p} -- skipped", flush=True)
            continue
        print(f"--- SUBFIND {name}: {p}", flush=True)
        sf[name] = analyse_subfind(p, boxes, a.dilate)
        t = sf[name]["radial_by_mass"]["total_by_r"]
        print(f"    region sats {sf[name]['n_region_sats']} (>=50p "
              f"{sf[name]['n_region_sats_ge50']}), core {t[0]}, 1-1.5 {t[4]}, "
              f"median x {sf[name]['radial_by_mass']['median_x']:.3f}")

    # --- cross-arm: catalog numbers vs the trainer's held-out diagnostics ----
    rows = []
    for name, res in results.items():
        row = {"name": name, "kind": res["kind"],
               "region_subs": res["n_region_subs"],
               "region_subs_ge50": res["n_region_subs_ge50"],
               "core": res["radial_by_mass"]["total_by_r"][0],
               "core_ge50": int(np.sum(np.asarray(
                   res["radial_by_mass"]["counts"])[0, 1:])),
               "band_1_1.5": res["radial_by_mass"]["total_by_r"][4],
               "band_1_1.5_ge50": int(np.sum(np.asarray(
                   res["radial_by_mass"]["counts"])[4, 1:])),
               "median_x": res["radial_by_mass"]["median_x"],
               "core_structures": res["structures"]["ge20"]["structures_by_r"][0],
               "core_frag_per_structure":
                   res["structures"]["ge20"]["frag_per_structure_by_r"][0],
               "core_comoving_frac": res["comoving_frac_by_r"][0]}
        row["core_frac"] = row["core"] / max(row["region_subs"], 1)
        if name == "base":
            row.update({"velhighk_ratio": 1.0, "vel_rms_ratio": 1.0})
        arm = res.get("arm")
        if arm:
            sp = R / "member_gather" / arm / "summary.json"
            if sp.is_file():
                s = json.loads(sp.read_text())
                h = history_row(s, res.get("step"))
                if h and "holdout" in h:
                    for k in DIAG_FIELDS:
                        val = h["holdout"].get(k)
                        row[k] = float(val) if isinstance(val, (int, float)) else None
                    row["diag_step"] = int(h.get("step", -1))
                cfg = s.get("config", {})
                row["config"] = {k: cfg.get(k) for k in sorted(cfg)
                                 if k.startswith("w_") or k in (
                                     "centre_mode", "soft_membership", "rung",
                                     "steps", "min_num_p", "population")}
        rows.append(row)

    targets = ("core_frac", "core", "core_ge50", "core_structures",
               "core_frag_per_structure", "core_comoving_frac", "median_x",
               "band_1_1.5_ge50")
    preds = ("velhighk_ratio", "vel_rms_ratio", "highk_ratio_max", "virial",
             "bound_hard", "iso_ratio", "r_rms_over_hr", "sigma_v_over_hr",
             "region_subs_ge50")
    # Arms plus frozen SR2 (velocity ratios 1 by construction); HR and the
    # swap / m2m extras are not draws from the same family of objectives.
    cand = [r for r in rows if r["kind"] == "arm" or r["name"] == "base"]
    corr = {}
    for t in targets:
        corr[t] = {}
        for p in preds:
            rho, n, pv = spearman([r.get(t) if r.get(t) is not None else np.nan
                                   for r in cand],
                                  [r.get(p) if r.get(p) is not None else np.nan
                                   for r in cand])
            corr[t][p] = {"rho": rho, "n": n, "p": pv}

    print("\n=== cross-arm Spearman (candidates incl. base; rho [n, perm p])")
    print(f"{'':<24}" + "".join(f"{p[:14]:>16}" for p in preds))
    for t in targets:
        print(f"{t:<24}" + "".join(
            f"{corr[t][p]['rho']:>8.2f} [{corr[t][p]['n']:>2d},{corr[t][p]['p']:.2f}]"
            if np.isfinite(corr[t][p]['rho']) else f"{'-':>16}" for p in preds))

    write_json(Path(a.out), {
        "box": a.box, "dilate": a.dilate, "n_tiles": len(tiles),
        "region_export": a.region_export,
        "r_bins": r_labels(), "sub_mass_bins": m_labels(SUB_MASS_EDGES),
        "host_mass_bins": m_labels(HOST_MASS_EDGES),
        "n_hr_hosts_matched": int(big.size),
        "rockstar": results, "subfind": sf,
        "cross_arm": rows, "correlations": corr})
    print(f"\n=== wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
