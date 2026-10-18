#!/usr/bin/env python
"""Stacked host-centric phase-space profiles of one field: is the core cold or dense?

Halo-finder counts say the tuned arms pile "subhalos" into host cores (40% of
them inside 0.25 R_vir against HR's 2.5%), but a count cannot say what the
field looks like there. This reads the PARTICLES around every massive HR host
in the edited region and measures, per bin of r / R_vir (HR's R_vir, so every
field is binned on the same physical shells):

* ``rho``          particle number density / box mean (the host profile),
* ``vr``           mean radial velocity / V_vir (infall),
* ``sigma3d``      3D velocity dispersion about the bin mean / V_vir,
* ``sigma_loc``    median velocity dispersion of each particle's k=32
                   neighbours / V_vir -- the small-scale temperature a 6D
                   finder sees,
* ``coldness``     median sigma_loc / sigma3d (~1 in a well-mixed hot halo,
                   << 1 for cold streams and clumps),
* ``f_clump10``    fraction of the bin's particles whose k=32 density exceeds
                   10x the bin mean -- a finder-free substructure mass
                   fraction (also 30x),
* ``f_coldclump``  dense (>10x) AND cold (sigma_loc < 0.5 sigma3d): what
                   Rockstar's 6D linking picks up as a halo.

The host centre is re-found in each field by shrinking spheres from HR's
centre, so a displaced core is measured about itself, and the offset is
recorded. Host-to-host scatter of HR's own profiles is the learnability
check: a stacked target is only a rule if HR's hosts agree on it.

Writes ``<out>.npz`` (per-host profiles, redrawable) and ``<out>.json``
(stacked medians, 16/84 percentiles, per-host scalars). CPU only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402
from gather_holdout_figures_data import in_tile_region, tile_boxes  # noqa: E402
from placement_forensics import min_image, read_rockstar  # noqa: E402
from cosmo_sr.eval.particles import field_to_particles  # noqa: E402
from cosmo_sr.reward.tiles import TileGrid  # noqa: E402

BOX = 100.0
EDGES_X = np.array([0.0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25,
                    1.5, 2.0, 2.5, 3.0])
K_NN = 32
G_MPC = 4.30091e-9               # Mpc (km/s)^2 / Msun ; h cancels in V_vir
QUANTITIES = ("n", "rho", "vr", "sigma3d", "sigma_loc", "coldness",
              "f_clump10", "f_clump30", "f_coldclump")


class CellIndex:
    """Particles sorted into a periodic grid of cubic cells, for sphere gathers."""

    def __init__(self, pos: np.ndarray, cell: float = 2.0, box: float = BOX):
        self.n = int(np.floor(box / cell))
        self.cell = box / self.n
        self.box = box
        ijk = np.floor(np.mod(pos, box) / self.cell).astype(np.int64)
        np.clip(ijk, 0, self.n - 1, out=ijk)
        key = (ijk[:, 0] * self.n + ijk[:, 1]) * self.n + ijk[:, 2]
        self.order = np.argsort(key, kind="stable")
        self.starts = np.searchsorted(key[self.order],
                                      np.arange(self.n ** 3 + 1))

    def gather(self, centre: np.ndarray, radius: float) -> np.ndarray:
        """Indices of every particle in the cells overlapping the sphere's cube."""
        lo = np.floor((centre - radius) / self.cell).astype(int)
        hi = np.floor((centre + radius) / self.cell).astype(int)
        rng = [np.arange(lo[a], hi[a] + 1) % self.n for a in range(3)]
        rng = [np.unique(r) for r in rng]
        keys = ((rng[0][:, None, None] * self.n + rng[1][None, :, None])
                * self.n + rng[2][None, None, :]).ravel()
        parts = [self.order[self.starts[k]:self.starts[k + 1]] for k in keys]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def shrinking_centre(d: np.ndarray, r0: float, r_min: float,
                     factor: float = 0.8, min_n: int = 200) -> np.ndarray:
    """Shrinking-sphere centre of ``d`` (positions relative to a first guess)."""
    c = np.zeros(3)
    r = r0
    while r > r_min:
        m = np.einsum("ij,ij->i", d - c, d - c) < r * r
        if m.sum() < min_n:
            break
        c = d[m].mean(axis=0)
        r *= factor
    return c


def host_profile(d: np.ndarray, v: np.ndarray, rvir: float, vvir: float,
                 mean_n: float, rng: np.random.Generator,
                 max_query_per_bin: int = 20000) -> Dict[str, np.ndarray]:
    """Profiles of one host. ``d`` = positions relative to the field's centre."""
    from scipy.spatial import cKDTree
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    x = r / rvir
    inner = x < 0.1
    if inner.sum() < 20:
        inner = x < 0.2
    v = v - v[inner].mean(axis=0) if inner.any() else v
    nb = EDGES_X.size - 1
    b = np.digitize(x, EDGES_X) - 1                   # -1 never (x >= 0)
    in_range = b < nb
    out = {q: np.full(nb, np.nan) for q in QUANTITIES}
    vol = 4.0 / 3.0 * np.pi * ((EDGES_X[1:] * rvir) ** 3
                               - (EDGES_X[:-1] * rvir) ** 3)
    tree = cKDTree(d)
    for i in range(nb):
        m = np.where(in_range & (b == i))[0]
        out["n"][i] = m.size
        out["rho"][i] = m.size / vol[i] / mean_n
        if m.size < K_NN + 2:
            continue
        rhat = d[m] / np.maximum(r[m, None], 1e-12)
        out["vr"][i] = float(np.mean(np.einsum("ij,ij->i", v[m], rhat))) / vvir
        vm = v[m] - v[m].mean(axis=0)
        s3 = float(np.sqrt(np.mean(np.einsum("ij,ij->i", vm, vm))))
        out["sigma3d"][i] = s3 / vvir
        q = m if m.size <= max_query_per_bin else rng.choice(
            m, max_query_per_bin, replace=False)
        dist, nbr = tree.query(d[q], k=K_NN + 1)
        nbr = nbr[:, 1:]                               # drop the particle itself
        rk = dist[:, -1]
        n_loc = K_NN / (4.0 / 3.0 * np.pi * np.maximum(rk, 1e-9) ** 3)
        vn = v[nbr]                                    # (Q, K, 3)
        vn = vn - vn.mean(axis=1, keepdims=True)
        s_loc = np.sqrt(np.mean(np.einsum("qkj,qkj->qk", vn, vn), axis=1))
        n_bin = m.size / vol[i]
        out["sigma_loc"][i] = float(np.median(s_loc)) / vvir
        out["coldness"][i] = float(np.median(s_loc)) / s3 if s3 > 0 else np.nan
        out["f_clump10"][i] = float(np.mean(n_loc > 10.0 * n_bin))
        out["f_clump30"][i] = float(np.mean(n_loc > 30.0 * n_bin))
        out["f_coldclump"][i] = float(np.mean((n_loc > 10.0 * n_bin)
                                              & (s_loc < 0.5 * s3)))
    return out


PEAK_QUANTITIES = ("n_peaks10", "n_peaks30", "peak_c_over_a", "peak_b_over_a",
                   "peak_tu", "peak_lag", "peak_dvbg", "peak_coldbg")
BG_MIN_X = 0.5                   # distinctness is measured from 0.5 R_vir out
K_BIND = 512                     # neighbourhood an isolated-clump unbinding sees
MASS_KS = (32, 64, 128, 256, 512)


def bound_count(dx: np.ndarray, vx: np.ndarray, m_p: float, iters: int = 10) -> int:
    """Particles left after iterative unbinding of one neighbourhood in isolation.

    Each pass keeps the particles whose kinetic energy about the bulk velocity
    of the currently-bound set is below their (softened, pairwise) binding to
    that set -- the halo finders' unbinding step without the 6D linking or the
    host's tidal field.
    """
    n = dx.shape[0]
    bound = np.ones(n, dtype=bool)
    dd = dx[:, None, :] - dx[None, :, :]
    inv = 1.0 / np.sqrt(np.einsum("ijk,ijk->ij", dd, dd) + SOFT_MPC ** 2)
    np.fill_diagonal(inv, 0.0)
    for _ in range(iters):
        if bound.sum() < 2:
            return int(bound.sum())
        vb = vx[bound].mean(axis=0)
        kin = 0.5 * np.einsum("ij,ij->i", vx - vb, vx - vb)
        pot = G_MPC * m_p * (inv[:, bound].sum(axis=1))
        new = kin < pot
        if np.array_equal(new, bound):
            break
        bound = new
    return int(bound.sum())
NG = 512                         # the Lagrangian lattice (ids are flat indices)
K_OBJ = 64                       # neighbours that stand for "the object" at a peak
SOFT_MPC = 0.005                 # Plummer softening (Rockstar's force resolution)


def lagrangian_q(ids: np.ndarray, ng: int = NG, box: float = BOX) -> np.ndarray:
    """Lagrangian lattice position (Mpc/h) of flat particle ids, cell centres."""
    i, j, k = np.unravel_index(np.asarray(ids, dtype=np.int64), (ng, ng, ng))
    cell = box / ng
    return np.stack([(i + 0.5) * cell, (j + 0.5) * cell, (k + 0.5) * cell], axis=-1)


def peak_profile(d: np.ndarray, v: np.ndarray, rvir: float, m_p: float,
                 tree=None, chunk: int = 500000, ids=None,
                 records: bool = False, bind: bool = False,
                 min_ratio: float = 10.0):
    """Finder-free census of compact objects per radial bin.

    A PEAK is a particle whose k=32 density is the maximum of its own
    neighbourhood and exceeds 10x (or 30x) the mean density of its shell -- one
    per density maximum, so a count of clumps, not of clumped mass. At each
    10x peak the K_OBJ nearest particles stand for the object:

    * ``peak_c_over_a`` / ``peak_b_over_a`` -- axis ratios of their inertia
      tensor about the peak (a caustic sheet has small c/a, a filament small
      b/a, a halo ~0.5-1 for both),
    * ``peak_tu`` -- their T/|U|: kinetic energy about their mean velocity over
      the softened pairwise potential (~0.5 virialised, >> 1 a flow),
    * ``peak_lag`` -- (needs ``ids``) rms spread of their LAGRANGIAN positions,
      in lattice cells: a halo is one collapsed patch (small), a stream crossing
      is assembled from distant patches (large),
    * ``peak_dvbg`` -- |mean v of the 32 nearest - mean v of an annulus
      OUTSIDE the object, [max(4 r64, 0.1), that + 0.15] Mpc/h| / the annulus
      dispersion: an orbiting subhalo moves apart from its surroundings, a knot
      in a flow does not. ``peak_coldbg`` is the 32 nearest's own dispersion
      over the annulus's. Both only from ``BG_MIN_X`` R_vir out (a core annulus
      would hold ~1e6 particles and is not the population in question).

    Counts are per host; the rest are medians over the peaks in the bin. With
    ``records`` the per-peak values are returned too (key ``_records``).
    """
    from scipy.spatial import cKDTree
    if tree is None:
        tree = cKDTree(d)
    n = d.shape[0]
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    x = r / rvir
    nb = EDGES_X.size - 1
    b = np.digitize(x, EDGES_X) - 1
    vol = 4.0 / 3.0 * np.pi * ((EDGES_X[1:] * rvir) ** 3 - (EDGES_X[:-1] * rvir) ** 3)
    n_shell = np.array([np.sum(b == i) / vol[i] for i in range(nb)])
    n_loc = np.empty(n)
    nbr_all = np.empty((n, K_NN), dtype=np.int64)
    for s in range(0, n, chunk):
        dist, nbr = tree.query(d[s:s + chunk], k=K_NN + 1, workers=-1)
        n_loc[s:s + chunk] = K_NN / (4.0 / 3.0 * np.pi * np.maximum(dist[:, -1], 1e-9) ** 3)
        nbr_all[s:s + chunk] = nbr[:, 1:]
    is_max = n_loc >= n_loc[nbr_all].max(axis=1)
    out = {q: np.full(nb, np.nan) for q in PEAK_QUANTITIES}
    cell = BOX / NG
    rec = {k: [] for k in ("row", "bin", "ratio", "c_over_a", "tu", "lag", "dvbg",
                           "coldbg", "nbound", *[f"enc{k}" for k in MASS_KS])}
    for i in range(nb):
        inb = (b == i) & is_max
        # ``min_ratio`` is the contrast a peak needs to be MEASURED and recorded
        # (10 = every census before 2026-10-15). n_peaks10/30 are counted at their
        # own thresholds whatever it is, so the count columns stay comparable.
        out["n_peaks10"][i] = int(np.sum(inb & (n_loc > 10.0 * n_shell[i])))
        p10 = np.where(inb & (n_loc > float(min_ratio) * n_shell[i]))[0]
        out["n_peaks30"][i] = int(np.sum(inb & (n_loc > 30.0 * n_shell[i])))
        if p10.size == 0:
            continue
        dist64, obj = tree.query(d[p10], k=K_OBJ, workers=-1)  # includes the peak
        ca, ba, tu, lag, dvb, cbg = [], [], [], [], [], []
        nbd, enc = [], {k_: [] for k_ in MASS_KS}
        iu = np.triu_indices(K_OBJ, 1)
        for n_row, row in enumerate(obj):
            xo = d[row] - d[row[0]]
            ev = np.linalg.eigvalsh(xo.T @ xo / K_OBJ)             # ascending
            if ev[2] > 0:
                ca.append(np.sqrt(max(ev[0], 0.0) / ev[2]))
                ba.append(np.sqrt(max(ev[1], 0.0) / ev[2]))
            vo = v[row] - v[row].mean(axis=0)
            kin = 0.5 * float(np.einsum("ij,ij->", vo, vo))
            dd = d[row][:, None, :] - d[row][None, :, :]
            rij = np.sqrt(np.einsum("ijk,ijk->ij", dd, dd)[iu] + SOFT_MPC ** 2)
            pot = G_MPC * m_p * float(np.sum(1.0 / rij))
            tu.append(kin / pot if pot > 0 else np.nan)
            if ids is not None:
                q = lagrangian_q(ids[row])
                dq = q - q[0]
                dq -= BOX * np.round(dq / BOX)
                dq -= dq.mean(axis=0)
                lag.append(float(np.sqrt(np.mean(np.einsum("ij,ij->i", dq, dq)))) / cell)
            if x[row[0]] < BG_MIN_X:
                dvb.append(np.nan)
                cbg.append(np.nan)
                if bind:
                    nbd.append(np.nan)
                    for k_ in MASS_KS:
                        enc[k_].append(np.nan)
                continue
            if bind:
                db, nb_ = tree.query(d[row[0]], k=K_BIND)
                nbd.append(bound_count(d[nb_] - d[row[0]], v[nb_], m_p))
                for k_ in MASS_KS:
                    rk_ = max(float(db[k_ - 1]), 1e-9)
                    enc[k_].append(k_ / (4.0 / 3.0 * np.pi * rk_ ** 3) / n_shell[i])
            r_in = max(4.0 * float(dist64[n_row, -1]), 0.1)
            r_out = r_in + 0.15
            shell = np.asarray(tree.query_ball_point(d[row[0]], r_out), dtype=np.int64)
            dd = d[shell] - d[row[0]]
            bg = shell[np.einsum("ij,ij->i", dd, dd) > r_in * r_in]
            if bg.size >= 16:
                vb = v[bg] - v[bg].mean(axis=0)
                sb = float(np.sqrt(np.mean(np.einsum("ij,ij->i", vb, vb))))
                vk = v[row[:K_NN]]
                vc = vk.mean(axis=0)
                sk = float(np.sqrt(np.mean(np.einsum("ij,ij->i", vk - vc, vk - vc))))
                dvb.append(float(np.linalg.norm(vc - v[bg].mean(axis=0)) / sb)
                           if sb > 0 else np.nan)
                cbg.append(sk / sb if sb > 0 else np.nan)
            else:
                dvb.append(np.nan)
                cbg.append(np.nan)
        out["peak_c_over_a"][i] = float(np.median(ca)) if ca else np.nan
        out["peak_b_over_a"][i] = float(np.median(ba)) if ba else np.nan
        out["peak_tu"][i] = float(np.nanmedian(tu)) if tu else np.nan
        out["peak_lag"][i] = float(np.median(lag)) if lag else np.nan
        with np.errstate(all="ignore"):
            out["peak_dvbg"][i] = float(np.nanmedian(dvb)) if dvb else np.nan
            out["peak_coldbg"][i] = float(np.nanmedian(cbg)) if cbg else np.nan
        if records:
            rec["row"].extend(p10.tolist())
            rec["bin"].extend([i] * p10.size)
            rec["ratio"].extend((n_loc[p10] / n_shell[i]).tolist())
            rec["c_over_a"].extend(ca if len(ca) == p10.size else [np.nan] * p10.size)
            rec["tu"].extend(tu)
            rec["lag"].extend(lag if lag else [np.nan] * p10.size)
            rec["dvbg"].extend(dvb)
            rec["coldbg"].extend(cbg)
            rec["nbound"].extend(nbd if bind else [np.nan] * p10.size)
            for k_ in MASS_KS:
                rec[f"enc{k_}"].extend(enc[k_] if bind else [np.nan] * p10.size)
    if records:
        out["_records"] = {k: np.asarray(vv) for k, vv in rec.items()}
    return out


def select_region_hosts(hr: Dict[str, np.ndarray], region_export: str, dilate: float,
                        min_num_p: int, max_hosts: int) -> np.ndarray:
    """Rows of the HR hosts every census script measures: top-level, >= min_num_p,
    centre inside the dilated spliced tiles ('' = whole box), largest first.

    One definition, because peak records index hosts by position in this list.
    """
    if region_export.strip():
        ex = json.loads(Path(region_export).read_text())
        tiles = [int(t) for t in ex["tiles"]] if "tiles" in ex else sorted(
            {int(t) for h in ex["per_host"] for t in h["tiles"]})
        inreg = in_tile_region(hr["pos"], tile_boxes(tiles, TileGrid()), dilate=dilate)
    else:
        inreg = np.ones(hr["id"].size, dtype=bool)
    sel = np.where((hr["parent_id"] < 0) & (hr["num_p"] >= min_num_p) & inreg)[0]
    return sel[np.argsort(-hr["num_p"][sel])][:max_hosts]


def stack(per_host: Dict[str, np.ndarray]) -> Dict[str, Dict[str, List[float]]]:
    res = {}
    for q, a in per_host.items():
        a = np.asarray(a, dtype=np.float64)
        with np.errstate(all="ignore"):
            p16, p50, p84 = np.nanpercentile(a, [16, 50, 84], axis=0)
        res[q] = {"p16": p16.tolist(), "median": p50.tolist(),
                  "p84": p84.tolist(),
                  "n_hosts": np.sum(np.isfinite(a), axis=0).tolist()}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True, help="(6,512,512,512) catnorm .npy")
    ap.add_argument("--name", required=True)
    ap.add_argument("--hr-catalog", required=True,
                    help="HR Rockstar halos_*.ascii: defines the hosts and R_vir")
    ap.add_argument("--region-export", default="",
                    help="export.json whose tiles select the hosts; '' = whole box")
    ap.add_argument("--dilate", type=float, default=2.0)
    ap.add_argument("--min-host-num-p", type=int, default=10000)
    ap.add_argument("--max-hosts", type=int, default=60)
    ap.add_argument("--max-query-per-bin", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--peak-min-ratio", type=float, default=10.0,
                    help="with --peaks: contrast (k=32 density / shell mean) a peak "
                         "needs to be measured and recorded; 10 = the original census")
    ap.add_argument("--bind", action="store_true",
                    help="with --peaks: per-peak isolated unbinding of the 512 nearest "
                         "particles and the enclosed-density profile (outskirt peaks)")
    ap.add_argument("--peaks", action="store_true",
                    help="also count compact density peaks per bin, with their "
                         "shapes and object-scale T/|U| (peak_profile)")
    ap.add_argument("--out", required=True, help="output stem (.npz/.json added)")
    a = ap.parse_args(argv)

    hr = read_rockstar(a.hr_catalog)
    # '' = whole box: the training boxes have no gate region, and the
    # universality check wants hosts that were never region-selected
    sel = select_region_hosts(hr, a.region_export, a.dilate, a.min_host_num_p,
                              a.max_hosts)
    rvir = hr["rvir"][sel] / 1000.0
    vvir = np.sqrt(G_MPC * hr["mvir"][sel] / rvir)
    print(f"{sel.size} HR hosts >= {a.min_host_num_p}p in the region "
          f"(num_p {hr['num_p'][sel].min()}-{hr['num_p'][sel].max()})", flush=True)

    t0 = time.time()
    field = np.load(a.field, mmap_mode="r")
    pb = field_to_particles(field, boxsize_kpc_h=BOX * 1000.0)
    pos, vel = pb.pos_mpc_h, pb.vel_kms
    del field
    mean_n = pos.shape[0] / BOX ** 3
    ci = CellIndex(pos)
    print(f"particles + cell index in {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(a.seed)
    qnames = QUANTITIES + (PEAK_QUANTITIES if a.peaks else ())
    per_host = {q: [] for q in qnames}
    peak_rec = {k: [] for k in ("host", "pos", "x", "bin", "ratio", "c_over_a",
                                "tu", "lag", "dvbg", "coldbg", "nbound",
                                *[f"enc{k}" for k in MASS_KS])}
    scal = {k: [] for k in ("host_id", "num_p_hr", "rvir_hr", "vvir_hr",
                            "centre_offset_over_rvir", "n_within_rvir_over_hr",
                            "conc_n025_over_n1")}
    for j, h in enumerate(sel):
        c_hr = hr["pos"][h]
        rv = float(rvir[j])
        reach = 3.0 * rv + max(0.3 * rv, 0.3)
        idx = ci.gather(c_hr, reach)
        d = min_image(pos[idx].astype(np.float64) - c_hr)
        keep = np.einsum("ij,ij->i", d, d) < reach * reach
        idx, d = idx[keep], d[keep]
        c = shrinking_centre(d, 0.5 * rv, 0.05 * rv)
        d = d - c
        prof = host_profile(d, vel[idx].astype(np.float64), rv, float(vvir[j]),
                            mean_n, rng, a.max_query_per_bin)
        if a.peaks:
            pk = peak_profile(d, vel[idx].astype(np.float64), rv,
                              pb.particle_mass_msun_h, ids=pb.ids[idx], records=True,
                              bind=a.bind, min_ratio=a.peak_min_ratio)
            r_ = pk.pop("_records")
            prof.update(pk)
            rows = r_["row"].astype(np.int64)
            peak_rec["host"].append(np.full(rows.size, j))
            peak_rec["pos"].append(np.mod(d[rows] + c + c_hr, BOX))
            peak_rec["x"].append(np.sqrt(np.einsum("ij,ij->i", d[rows], d[rows])) / rv)
            for k_ in ("bin", "ratio", "c_over_a", "tu", "lag", "dvbg", "coldbg",
                       "nbound", *[f"enc{k}" for k in MASS_KS]):
                peak_rec[k_].append(r_[k_])
        for q in qnames:
            per_host[q].append(prof[q])
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        n1 = int(np.sum(r < rv))
        scal["host_id"].append(int(hr["id"][h]))
        scal["num_p_hr"].append(int(hr["num_p"][h]))
        scal["rvir_hr"].append(rv)
        scal["vvir_hr"].append(float(vvir[j]))
        scal["centre_offset_over_rvir"].append(float(np.linalg.norm(c) / rv))
        scal["n_within_rvir_over_hr"].append(n1 / float(hr["num_p"][h]))
        scal["conc_n025_over_n1"].append(float(np.sum(r < 0.25 * rv)) / max(n1, 1))
        if j % 10 == 0:
            print(f"  host {j + 1}/{sel.size} id {hr['id'][h]} num_p "
                  f"{hr['num_p'][h]}: {idx.size} particles, offset "
                  f"{scal['centre_offset_over_rvir'][-1]:.3f} R_vir, "
                  f"{time.time() - t0:.0f}s", flush=True)

    arrays = {q: np.asarray(v, dtype=np.float64) for q, v in per_host.items()}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".npz"), edges_x=EDGES_X, **arrays,
             **{k: np.asarray(v) for k, v in scal.items()})
    if a.peaks:
        # every 10x peak, absolute position included, so peaks can be matched
        # against a halo catalog without recomputing anything
        np.savez(out.parent / (out.name + "_peaks.npz"),
                 **{k: np.concatenate(v) if v else np.zeros(0)
                    for k, v in peak_rec.items()})
    big = np.asarray(scal["num_p_hr"]) >= 100000
    summary = {
        "name": a.name, "field": a.field, "n_hosts": int(sel.size),
        "k_nn": K_NN, "edges_x": EDGES_X.tolist(),
        "stack_all": stack(arrays),
        "stack_ge100k": stack({q: v[big] for q, v in arrays.items()}),
        "stack_lt100k": stack({q: v[~big] for q, v in arrays.items()}),
        "n_ge100k": int(big.sum()),
        "scalars_median": {k: float(np.median(v)) for k, v in scal.items()
                           if k != "host_id"},
        "per_host_scalars": scal,
    }
    write_json(out.with_suffix(".json"), summary)
    print(f"wrote {out.with_suffix('.npz')} and .json in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
