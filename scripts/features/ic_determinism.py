#!/usr/bin/env python
"""Do the IC ripples finer than the LR grid fix where subhalos sit today?

The IC-conditioning proposal rests on one claim: the initial ripples that the LR
grid cannot hold (wavelengths 0.39-3.1 Mpc/h) decide which outskirt objects
exist and where they sit at z=0, 13.8 Gyr later. The multi-resolution suite
tests that without training anything. If the 256^3 run of a box shares the
512^3 run's IC down to its own grid (0.78 Mpc/h), an outskirt object of the
512^3 run should reappear in the 256^3 run at the same host-relative place,
made of the same IC patch. Frozen SR2 sees only LR, so its objects should match
no better than an angular-shuffle null. The 256^3 run also differs numerically
(resolution, softening, time steps), so its match rate is a conservative
ceiling for any model that is given the IC.

Subcommands, all CPU:

  ripples   r(k) of the Lagrangian displacement and velocity fields on the
            256^3 grid: HR (2^3 block mean, and corner subsample in case the
            lattices are offset by half an HR cell) against dmo-256 and against
            frozen SR2. Gate: does dmo-256 share HR's fine ripples at all?
  rockstar  Rockstar on the dmo-256 box through the same pipeline and config as
            HR's catalog (the particle mass comes from the snapshot header).
  match     around every massive HR host, each HR object >= --obj-min-np at
            0.1-3 R_vir is looked for in a candidate catalog at the same
            host-relative position (mass within --mass-tol), against an
            angular-shuffle null. Matched pairs are checked for the same IC
            patch (periodic mean IC position of each object's core particles).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402

BOX = 100.0                                         # Mpc/h
NG_HR = 512
CELL_HR = BOX / NG_HR
R_BINS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)  # r / R_vir of the HR host
M_BINS = (400.0, 1000.0, 3000.0, np.inf)            # HR-particle equivalents
POOLED = {"inner_0.1_0.5": (0.1, 0.5), "mid_0.5_1": (0.5, 1.0),
          "outskirts_1_3": (1.0, 3.0)}
LR_NYQUIST = np.pi * 64 / BOX                       # 2.01 h/Mpc


def min_image(d: np.ndarray, box: float = BOX) -> np.ndarray:
    return d - box * np.round(d / box)


def wrap(x: np.ndarray, box: float = BOX) -> np.ndarray:
    """Into [0, box): np.mod can return box itself for tiny negatives."""
    y = np.mod(np.asarray(x, dtype=np.float64), box)
    y[y >= box] = 0.0
    return y


# ---------------------------------------------------------------- spectra

def block_mean(a: np.ndarray, f: int = 2) -> np.ndarray:
    """Mean over f^3 blocks of a cubic (Ng, Ng, Ng) array, as float64."""
    n = a.shape[0]
    if a.ndim != 3 or a.shape != (n, n, n) or n % f:
        raise ValueError(f"need a cubic array divisible by {f}, got {a.shape}")
    m = n // f
    return a.reshape(m, f, m, f, m, f).mean(axis=(1, 3, 5), dtype=np.float64)


def corner(a: np.ndarray, f: int = 2) -> np.ndarray:
    """Every f-th lattice point: right if the coarse lattice sits on the fine one's corners."""
    return np.ascontiguousarray(a[::f, ::f, ::f], dtype=np.float64)


def k_edges(ng: int, nbins: int = 32, box: float = BOX) -> np.ndarray:
    """Log-spaced |k| edges [h/Mpc], fundamental to the Nyquist of an ng grid."""
    return np.geomspace(2 * np.pi / box, np.pi * ng / box, nbins + 1)


def _shell_index(ng: int, edges: np.ndarray, box: float = BOX):
    kf = 2 * np.pi / box
    k1 = np.fft.fftfreq(ng, d=1.0 / ng) * kf
    kz = np.fft.rfftfreq(ng, d=1.0 / ng) * kf
    kk = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2
                 + kz[None, None, :] ** 2)
    idx = np.digitize(kk, edges).ravel() - 1
    # rfftn keeps one of each +-kz pair, so interior kz planes stand for two modes
    w = np.full(kz.size, 2.0)
    w[0] = 1.0
    if ng % 2 == 0:
        w[-1] = 1.0
    w = np.broadcast_to(w, kk.shape).ravel()
    ok = (idx >= 0) & (idx < edges.size - 1)
    return idx[ok], w[ok], ok


def cross_r(fa: Sequence[np.ndarray], fb: Sequence[np.ndarray],
            edges: np.ndarray, box: float = BOX) -> Tuple[np.ndarray, np.ndarray]:
    """r(k) of two multi-channel fields, channels pooled per shell.

    r = sum Re(A B*) / sqrt(sum |A|^2 sum |B|^2): 1 for identical fields, ~0
    for independent ones, whatever the amplitudes. Returns (r, weighted modes).
    """
    ng = fa[0].shape[0]
    idx, w, ok = _shell_index(ng, edges, box)
    nb = edges.size - 1
    sab, saa, sbb = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    for a, b in zip(fa, fb):
        A = np.fft.rfftn(np.asarray(a, np.float64) - np.mean(a)).ravel()[ok]
        B = np.fft.rfftn(np.asarray(b, np.float64) - np.mean(b)).ravel()[ok]
        sab += np.bincount(idx, weights=w * (A * B.conj()).real, minlength=nb)
        saa += np.bincount(idx, weights=w * np.abs(A) ** 2, minlength=nb)
        sbb += np.bincount(idx, weights=w * np.abs(B) ** 2, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = sab / np.sqrt(saa * sbb)
    return r, np.bincount(idx, weights=w, minlength=nb)


def band_mean(r: np.ndarray, nmodes: np.ndarray, edges: np.ndarray,
              lo: float, hi: float) -> float:
    """Mode-weighted mean of r over the shells whose centre lies in [lo, hi]."""
    c = np.sqrt(edges[:-1] * edges[1:])
    sel = (c >= lo) & (c <= hi) & np.isfinite(r) & (nmodes > 0)
    if not sel.any():
        return float("nan")
    return float(np.sum(r[sel] * nmodes[sel]) / np.sum(nmodes[sel]))


def _undo(c: int, a: np.ndarray, z: float = 0.0) -> np.ndarray:
    """Catnorm channel -> physical (kpc/h displacement, km/s velocity)."""
    from cosmo_sr.data.preprocess_srs import disnorm, velnorm
    return (disnorm if c < 3 else velnorm)(a, z=z, undo=True)


def cmd_ripples(a) -> int:
    t0 = time.time()
    xh = np.load(a.hr_field, mmap_mode="r")
    xs = np.load(a.sr2_field, mmap_mode="r")
    d = Path(a.dmo_dir)
    dd = np.load(d / "disp.npy", mmap_mode="r")
    dv = np.load(d / "vel.npy", mmap_mode="r")
    ng = int(dd.shape[1])
    if xh.shape[1] % ng or xs.shape[1] != xh.shape[1]:
        raise SystemExit(f"grids do not nest: HR {xh.shape}, SR2 {xs.shape}, dmo {dd.shape}")
    f = xh.shape[1] // ng
    hr_m, hr_c, sr2_m = [], [], []
    for c in range(6):
        ch = np.asarray(xh[c], dtype=np.float32)
        hr_m.append(_undo(c, block_mean(ch, f)))
        hr_c.append(_undo(c, corner(ch, f)))
        del ch
        sr2_m.append(_undo(c, block_mean(np.asarray(xs[c], dtype=np.float32), f)))
        print(f"  channel {c} reduced ({time.time() - t0:.0f}s)", flush=True)
    dmo = ([np.asarray(dd[c], np.float64) for c in range(3)]
           + [np.asarray(dv[c], np.float64) for c in range(3)])
    edges = k_edges(ng, a.nbins)
    kc = np.sqrt(edges[:-1] * edges[1:])
    lo, hi = a.band
    out: Dict = {"hr_field": a.hr_field, "sr2_field": a.sr2_field,
                 "dmo_dir": a.dmo_dir, "ng": ng, "k_edges": edges, "k_centre": kc,
                 "lr_nyquist": LR_NYQUIST, "dmo_nyquist": np.pi * ng / BOX,
                 "band": [lo, hi], "r": {}, "band_r": {}, "lowk_r": {}, "rms": {}}
    for kind, sl in (("disp", slice(0, 3)), ("vel", slice(3, 6))):
        r_mean, nm = cross_r(hr_m[sl], dmo[sl], edges)
        r_corner, _ = cross_r(hr_c[sl], dmo[sl], edges)
        r_sr2, _ = cross_r(hr_m[sl], sr2_m[sl], edges)
        out["nmodes"] = nm
        out["r"][kind] = {"dmo_mean": r_mean, "dmo_corner": r_corner, "sr2": r_sr2}
        out["band_r"][kind] = {k: band_mean(v, nm, edges, lo, hi)
                               for k, v in out["r"][kind].items()}
        out["lowk_r"][kind] = {k: band_mean(v, nm, edges, 0.1, 1.0)
                               for k, v in out["r"][kind].items()}
        out["rms"][kind] = {
            n: float(np.sqrt(np.mean([np.var(x) for x in fs[sl]])))
            for n, fs in (("hr_mean", hr_m), ("dmo", dmo), ("sr2_mean", sr2_m))}

    b = out["band_r"]["disp"]
    variant = "mean" if b["dmo_mean"] >= b["dmo_corner"] else "corner"
    r_dmo, r_sr2 = b[f"dmo_{variant}"], b["sr2"]
    low = out["lowk_r"]["disp"][f"dmo_{variant}"]
    if not (np.isfinite(low) and low >= 0.8):
        shared, why = False, (f"low-k r = {low:.2f} < 0.8: the {ng}^3 run is not the "
                              f"same large-scale box as HR")
    elif r_dmo >= a.gate_min and r_dmo - r_sr2 >= a.gate_margin:
        shared, why = True, (f"r = {r_dmo:.2f} at {lo}-{hi} h/Mpc vs frozen SR2 "
                             f"{r_sr2:.2f}: the {ng}^3 run shares HR's fine ripples")
    else:
        shared, why = False, (f"r = {r_dmo:.2f} at {lo}-{hi} h/Mpc vs frozen SR2 "
                              f"{r_sr2:.2f} (need >= {a.gate_min} and a margin of "
                              f"{a.gate_margin}): different fine ripples")
    out["gate"] = {"shared": shared, "reason": why, "variant": variant,
                   "r_dmo": r_dmo, "r_sr2": r_sr2, "lowk_r_dmo": low,
                   "gate_min": a.gate_min, "gate_margin": a.gate_margin}

    print(f"\n r(k) of the Lagrangian displacement, HR vs ... "
          f"(LR Nyquist {LR_NYQUIST:.2f}, dmo Nyquist {np.pi * ng / BOX:.2f} h/Mpc)")
    print(f" {'k':>7} {'modes':>9} {'dmo mean':>9} {'dmo corner':>10} {'SR2':>7}"
          f" | vel: {'dmo':>6} {'SR2':>6}")
    rd, rv = out["r"]["disp"], out["r"]["vel"]
    for i in range(kc.size):
        print(f" {kc[i]:7.2f} {nm[i]:9.0f} {rd['dmo_mean'][i]:9.3f}"
              f" {rd['dmo_corner'][i]:10.3f} {rd['sr2'][i]:7.3f}"
              f" |      {rv[f'dmo_{variant}'][i]:6.3f} {rv['sr2'][i]:6.3f}")
    print(f"\n rms (kpc/h, km/s): {out['rms']}")
    print(f" GATE shared={shared}: {why}")
    write_json(a.out, out)
    print(f" wrote {a.out}  ({time.time() - t0:.0f}s)")
    return 0


# --------------------------------------------------------------- rockstar

def cmd_rockstar(a) -> int:
    from cosmo_sr.data.preprocess_srs import disp_vel_to_field
    from cosmo_sr.eval.particles import particle_mass_msun_h
    from cosmo_sr.eval.rockstar import run_rockstar_on_field

    t0 = time.time()
    d = Path(a.dmo_dir)
    field = disp_vel_to_field(d / "disp.npy", d / "vel.npy", out_path=None,
                              redshift=a.redshift)
    ng = int(field.shape[1])
    cat = run_rockstar_on_field(field, a.out_dir, tag=a.tag, redshift=a.redshift)
    host = cat.parent_ids < 0
    summary = {"catalog": cat.path, "dmo_dir": a.dmo_dir, "ng": ng,
               "particle_mass_msun_h": particle_mass_msun_h(0.2814, BOX, ng ** 3),
               "n_halos": int(cat.num_p.size), "n_hosts": int(host.sum()),
               "n_subs": int((~host).sum()), "seconds": time.time() - t0}
    write_json(Path(a.out_dir) / "rockstar_summary.json", summary)
    print(summary)
    return 0


# ------------------------------------------------------------- catalogs

def load_catalog(path: str, np_ratio: float) -> Dict[str, np.ndarray]:
    """Rockstar catalog with num_p in HR-particle equivalents and rvir in Mpc/h."""
    from placement_forensics import read_rockstar
    c = read_rockstar(path)
    return {"id": c["id"], "parent": c["parent_id"], "pos": wrap(c["pos"]),
            "rvir": c["rvir"] / 1000.0, "np": c["num_p"] * float(np_ratio),
            "np_ratio": float(np_ratio), "path": str(path)}


def massive_hosts(cat: Dict, min_np: float) -> np.ndarray:
    return np.flatnonzero((cat["parent"] < 0) & (cat["np"] >= min_np))


def assign_to_hosts(cat: Dict, hosts: np.ndarray, rows: np.ndarray,
                    r_min: float, r_max: float):
    """Each object's nearest massive host in units of that host's R_vir.

    Returns (rows kept, index into ``hosts``, r/R_vir) for r_min <= r/R_vir <= r_max.
    """
    best = np.full(rows.size, np.inf)
    who = np.full(rows.size, -1)
    for j, h in enumerate(hosts):
        x = np.linalg.norm(min_image(cat["pos"][rows] - cat["pos"][h]), axis=1)
        x /= cat["rvir"][h]
        better = x < best
        best[better], who[better] = x[better], j
    keep = (best >= r_min) & (best <= r_max)
    return rows[keep], who[keep], best[keep]


def frame_offset(hr: Dict, hosts: np.ndarray, cand: Dict) -> np.ndarray:
    """Median offset candidate - HR of the massive hosts (a global frame shift)."""
    from scipy.spatial import cKDTree
    pool = np.flatnonzero((cand["parent"] < 0)
                          & (cand["np"] >= hr["np"][hosts].min() / 3.0))
    if pool.size == 0:
        return np.zeros(3)
    dist, k = cKDTree(cand["pos"][pool], boxsize=BOX).query(hr["pos"][hosts])
    off = min_image(cand["pos"][pool[k]] - hr["pos"][hosts])
    near = dist < hr["rvir"][hosts]
    if near.sum() < 5:
        near = np.ones_like(near)
    return np.median(off[near], axis=0)


def match_hosts(hr: Dict, hosts: np.ndarray, cand: Dict, offset: np.ndarray,
                mass_tol: float = 2.0, max_rvir: float = 0.5) -> np.ndarray:
    """Candidate row of each massive HR host (-1 if none within max_rvir R_vir)."""
    from scipy.spatial import cKDTree
    pool = np.flatnonzero(cand["parent"] < 0)
    tree = cKDTree(cand["pos"][pool], boxsize=BOX)
    out = np.full(hosts.size, -1, dtype=np.int64)
    pred = wrap(hr["pos"][hosts] + offset)
    hits = tree.query_ball_point(pred, max_rvir * hr["rvir"][hosts])
    for i, h in enumerate(hosts):
        rows = pool[np.asarray(hits[i], dtype=np.int64)]
        ratio = cand["np"][rows] / hr["np"][h]
        rows = rows[(ratio >= 1 / mass_tol) & (ratio <= mass_tol)]
        if rows.size:
            d = np.linalg.norm(min_image(cand["pos"][rows] - pred[i]), axis=1)
            out[i] = rows[np.argmin(d)]
    return out


def match_points(tree, cat: Dict, points: np.ndarray, radius: np.ndarray,
                 np_target: np.ndarray, mass_tol: float) -> np.ndarray:
    """Nearest catalog row within ``radius`` whose mass is within a factor
    ``mass_tol`` of ``np_target`` (-1 if none). ``tree`` indexes cat['pos']."""
    hits = tree.query_ball_point(wrap(points), radius)
    out = np.full(points.shape[0], -1, dtype=np.int64)
    for i, h in enumerate(hits):
        if not h:
            continue
        rows = np.asarray(h, dtype=np.int64)
        ratio = cat["np"][rows] / np_target[i]
        rows = rows[(ratio >= 1 / mass_tol) & (ratio <= mass_tol)]
        if rows.size:
            d = np.linalg.norm(min_image(cat["pos"][rows] - points[i]), axis=1)
            out[i] = rows[np.argmin(d)]
    return out


def random_directions(rng: np.random.Generator, n_null: int, n: int) -> np.ndarray:
    u = rng.normal(size=(n_null, n, 3))
    return u / np.linalg.norm(u, axis=-1, keepdims=True)


def match_catalogs(hr: Dict, hosts: np.ndarray, rows: np.ndarray,
                   hidx: np.ndarray, cand: Dict, *, mass_tol: float,
                   match_rvir: float, match_min_mpc: float, n_null: int,
                   rng: np.random.Generator, host_mass_tol: float = 2.0,
                   host_match_rvir: float = 0.5) -> Dict[str, np.ndarray]:
    """Look for each HR object in ``cand`` at the same host-relative position.

    The prediction is the matched candidate host plus the HR object's offset
    from its own host (the HR position plus the global frame offset when the
    host has no match). The null keeps each object's distance from the host
    and draws a random direction, ``n_null`` times.
    """
    from scipy.spatial import cKDTree
    off = frame_offset(hr, hosts, cand)
    hmatch = match_hosts(hr, hosts, cand, off, host_mass_tol, host_match_rvir)
    dvec = min_image(hr["pos"][rows] - hr["pos"][hosts][hidx])
    hm = hmatch[hidx]
    centre = np.where((hm >= 0)[:, None], cand["pos"][np.maximum(hm, 0)],
                      wrap(hr["pos"][hosts][hidx] + off))
    radius = np.maximum(match_min_mpc, match_rvir * hr["rvir"][rows])
    np_obj = hr["np"][rows]
    tree = cKDTree(cand["pos"], boxsize=BOX)
    match = match_points(tree, cand, wrap(centre + dvec), radius, np_obj, mass_tol)
    dist = np.linalg.norm(dvec, axis=1)
    u = random_directions(rng, n_null, rows.size)
    null = np.zeros(rows.size)
    for t in range(n_null):
        null += match_points(tree, cand, wrap(centre + dist[:, None] * u[t]),
                             radius, np_obj, mass_tol) >= 0
    return {"offset": off, "host_match": hmatch, "match": match,
            "null_frac": null / max(n_null, 1), "host_matched": hm >= 0,
            "radius": radius}


def reverse_match(hr: Dict, hosts: np.ndarray, cand: Dict, hmatch: np.ndarray,
                  *, obj_min_np: float, host_min_np: float, r_min: float,
                  r_max: float, mass_tol: float, match_rvir: float,
                  match_min_mpc: float, n_null: int, rng: np.random.Generator):
    """The other direction: candidate objects around matched hosts, looked for
    in HR at the same host-relative place. Binned by the HR host's R_vir."""
    from scipy.spatial import cKDTree
    ok = hmatch >= 0
    ch, hh = hmatch[ok], hosts[ok]
    if ch.size == 0:
        return np.zeros(0), np.zeros(0, bool), np.zeros(0)
    crow = np.flatnonzero((cand["np"] >= obj_min_np) & (cand["np"] < host_min_np))
    best = np.full(crow.size, np.inf)
    who = np.full(crow.size, -1)
    for j in range(ch.size):
        x = np.linalg.norm(min_image(cand["pos"][crow] - cand["pos"][ch[j]]), axis=1)
        x /= hr["rvir"][hh[j]]
        better = x < best
        best[better], who[better] = x[better], j
    keep = (best >= r_min) & (best <= r_max)
    crow, who, x = crow[keep], who[keep], best[keep]
    dvec = min_image(cand["pos"][crow] - cand["pos"][ch[who]])
    centre = hr["pos"][hh[who]]
    radius = np.maximum(match_min_mpc, match_rvir * cand["rvir"][crow])
    tree = cKDTree(hr["pos"], boxsize=BOX)
    m = match_points(tree, hr, wrap(centre + dvec), radius, cand["np"][crow],
                     mass_tol) >= 0
    dist = np.linalg.norm(dvec, axis=1)
    u = random_directions(rng, n_null, crow.size)
    null = np.zeros(crow.size)
    for t in range(n_null):
        null += match_points(tree, hr, wrap(centre + dist[:, None] * u[t]), radius,
                             cand["np"][crow], mass_tol) >= 0
    return x, m, null / max(n_null, 1)


# ------------------------------------------------------- IC patches

def field_ng(field: str, kind: str) -> int:
    if kind == "raw":
        return int(np.load(Path(field) / "disp.npy", mmap_mode="r").shape[1])
    if kind == "catnorm":
        return int(np.load(field, mmap_mode="r").shape[1])
    raise SystemExit(f"unknown field kind {kind!r} (raw or catnorm)")


def particle_positions(field: str, kind: str) -> Tuple[np.ndarray, int]:
    """z=0 positions (N, 3) float64 in Mpc/h, row = flat Lagrangian index.

    ``catnorm``: a (6, Ng^3) normalised field (displacement x 6000 kpc/h at
    z=0); ``raw``: a directory whose disp.npy is (3, Ng^3) in kpc/h. The lattice
    is cell-centred, as in cosmo_sr.eval.particles.field_to_particles.
    """
    from cosmo_sr.data.preprocess_srs import growth_D
    if kind == "catnorm":
        x = np.load(field, mmap_mode="r")
        scale = 6000.0 * growth_D(0.0) / 1000.0
    elif kind == "raw":
        x = np.load(Path(field) / "disp.npy", mmap_mode="r")
        scale = 1.0 / 1000.0
    else:
        raise SystemExit(f"unknown field kind {kind!r} (raw or catnorm)")
    ng = int(x.shape[1])
    q = (np.arange(ng, dtype=np.float64) + 0.5) * (BOX / ng)
    pos = np.empty((ng ** 3, 3), dtype=np.float64)
    for c in range(3):
        shape = [1, 1, 1]
        shape[c] = ng
        pos[:, c] = (np.asarray(x[c], dtype=np.float64) * scale
                     + q.reshape(shape)).ravel()
    np.mod(pos, BOX, out=pos)
    pos[pos >= BOX] = 0.0
    return pos, ng


def particle_tree(pos: np.ndarray):
    from scipy.spatial import cKDTree
    return cKDTree(pos, leafsize=32, balanced_tree=False, compact_nodes=False,
                   copy_data=False, boxsize=BOX)


def lagrangian_centroids(tree, ng: int, centres: np.ndarray,
                         ks: np.ndarray) -> np.ndarray:
    """Periodic mean IC position (Mpc/h) of each object's ks[i] nearest particles."""
    out = np.full((len(centres), 3), np.nan)
    if len(centres) == 0:
        return out
    kmax = int(max(1, np.max(ks)))
    _, idx = tree.query(wrap(centres), k=kmax, workers=-1)
    idx = np.asarray(idx).reshape(len(centres), kmax)
    for i, k in enumerate(ks):
        p = idx[i, :max(int(k), 1)]
        q = (np.stack(np.unravel_index(p, (ng, ng, ng)), axis=1) + 0.5) * (BOX / ng)
        th = 2 * np.pi * q / BOX
        out[i] = wrap(np.arctan2(np.sin(th).mean(0), np.cos(th).mean(0))
                      * BOX / (2 * np.pi))
    return out


def lagrangian_radius(np_hr: np.ndarray) -> np.ndarray:
    """Radius (Mpc/h) of the IC sphere holding np_hr HR particles."""
    return (3.0 * np.asarray(np_hr, dtype=np.float64) / (4 * np.pi)) ** (1 / 3) * CELL_HR


def field_consistency(tree, cat: Dict, rows: np.ndarray, n_max: int = 2000,
                      seed: int = 0) -> float:
    """Median (field particles within r_vir) / (catalog num_p). About 1-5 when the
    catalog was found on this field; far below 1 when catalog and field differ."""
    if rows.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    pick = rows if rows.size <= n_max else rng.choice(rows, n_max, replace=False)
    n_in = tree.query_ball_point(wrap(cat["pos"][pick]), cat["rvir"][pick],
                                 return_length=True, workers=-1)
    return float(np.median(n_in / (cat["np"][pick] / cat["np_ratio"])))


# ------------------------------------------------------------- tracking

def periodic_mean(x: np.ndarray, box: float = BOX) -> np.ndarray:
    """Periodic mean of points (N, 3)."""
    th = 2 * np.pi * np.asarray(x, dtype=np.float64) / box
    return wrap(np.arctan2(np.sin(th).mean(0), np.cos(th).mean(0)) * box / (2 * np.pi))


def core_particles(tree, centres: np.ndarray, ks: np.ndarray) -> List[np.ndarray]:
    """Flat indices of each object's ks[i] particles nearest its centre."""
    if len(centres) == 0:
        return []
    kmax = int(max(1, np.max(ks)))
    _, idx = tree.query(wrap(centres), k=kmax, workers=-1)
    idx = np.asarray(idx).reshape(len(centres), kmax)
    return [idx[i, :max(int(k), 1)] for i, k in enumerate(ks)]


def map_to_grid(p: np.ndarray, ng_from: int, ng_to: int) -> np.ndarray:
    """Flat Lagrangian index on an ng_from lattice -> the ng_to cell holding that IC point.

    Both lattices are cell-centred (the ripples stage confirmed it), so fine cell
    i lies in coarse cell i // (ng_from // ng_to).
    """
    if ng_from == ng_to:
        return np.asarray(p)
    if ng_from % ng_to:
        raise ValueError(f"lattices do not nest: {ng_from} -> {ng_to}")
    f = ng_from // ng_to
    i, j, k = np.unravel_index(np.asarray(p), (ng_from,) * 3)
    return np.ravel_multi_index((i // f, j // f, k // f), (ng_to,) * 3)


def matter_stats(pos: np.ndarray, members: Sequence[np.ndarray],
                 r_vir: np.ndarray):
    """Per member list: periodic centroid, rms distance from it, and the fraction
    of members within r_vir of it (how compact the same matter is)."""
    n = len(members)
    c, s, f = np.empty((n, 3)), np.empty(n), np.empty(n)
    for i, m in enumerate(members):
        x = pos[m]
        ci = periodic_mean(x)
        d = np.linalg.norm(min_image(x - ci), axis=1)
        c[i], s[i], f[i] = ci, np.sqrt(np.mean(d ** 2)), np.mean(d <= r_vir[i])
    return c, s, f


def radial_geometry(x: np.ndarray, centre: np.ndarray, dvec_hr: np.ndarray,
                    rvir_host: np.ndarray):
    """(r / R_vir of x about ``centre``, cosine between its host-relative direction
    and the HR object's ``dvec_hr``)."""
    d = min_image(x - centre)
    r = np.linalg.norm(d, axis=1)
    rh = np.linalg.norm(dvec_hr, axis=1)
    cos = np.sum(d * dvec_hr, axis=1) / np.maximum(r * rh, 1e-12)
    return r / rvir_host, cos


def same_patch_objects(cand: Dict, rows_c: np.ndarray, q_c: np.ndarray,
                       q_target: np.ndarray, r_l: np.ndarray,
                       np_target: np.ndarray, lag_tol: float,
                       mass_tol: float) -> np.ndarray:
    """For each target IC centroid, the candidate row whose IC centroid is nearest
    within lag_tol * R_L, mass within a factor mass_tol (-1 if none) -- the same
    object found wherever it sits today."""
    from scipy.spatial import cKDTree
    out = np.full(len(q_target), -1, dtype=np.int64)
    if len(rows_c) == 0 or len(q_target) == 0:
        return out
    tree = cKDTree(wrap(q_c), boxsize=BOX)
    hits = tree.query_ball_point(wrap(q_target), lag_tol * np.asarray(r_l))
    for i, h in enumerate(hits):
        if not h:
            continue
        h = np.asarray(h, dtype=np.int64)
        ratio = cand["np"][rows_c[h]] / np_target[i]
        h = h[(ratio >= 1 / mass_tol) & (ratio <= mass_tol)]
        if h.size:
            d = np.linalg.norm(min_image(q_c[h] - q_target[i]), axis=1)
            out[i] = rows_c[h[np.argmin(d)]]
    return out


def shuffled_within_bins(x: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float,
                         rng: np.random.Generator, n_perm: int = 50) -> np.ndarray:
    """Null for 'same radius': per object, the chance |log(b/a)| <= log(tol) when b is
    re-paired at random among objects of the same r/R_vir bin."""
    out = np.full(x.size, np.nan)
    edges = np.asarray(R_BINS)
    which = np.clip(np.digitize(x, edges) - 1, 0, edges.size - 2)
    for k in range(edges.size - 1):
        sel = np.flatnonzero(which == k)
        if sel.size < 2:
            continue
        hits = np.zeros(sel.size)
        for _ in range(n_perm):
            hits += np.abs(np.log(b[rng.permutation(sel)] / a[sel])) <= np.log(tol)
        out[sel] = hits / n_perm
    return out


# ------------------------------------------------------------ summaries

def _rows_in(x: np.ndarray, lo: float, hi: float, last: bool) -> np.ndarray:
    return (x >= lo) & ((x <= hi) if last else (x < hi))


def radial_table(x: np.ndarray, flags: Dict[str, np.ndarray],
                 np_obj: Optional[np.ndarray] = None) -> List[Dict]:
    """Per r/R_vir bin: n and the mean of each per-object flag (and per mass bin)."""
    out = []
    edges = list(zip(R_BINS[:-1], R_BINS[1:]))
    for i, (lo, hi) in enumerate(edges):
        sel = _rows_in(x, lo, hi, i == len(edges) - 1)
        row = {"r_lo": lo, "r_hi": hi, "n": int(sel.sum())}
        for k, v in flags.items():
            row[k] = float(np.mean(v[sel])) if sel.any() else None
        if np_obj is not None:
            row["by_mass"] = []
            for mlo, mhi in zip(M_BINS[:-1], M_BINS[1:]):
                s2 = sel & (np_obj >= mlo) & (np_obj < mhi)
                cell = {"np_lo": mlo, "np_hi": mhi, "n": int(s2.sum())}
                for k, v in flags.items():
                    cell[k] = float(np.mean(v[s2])) if s2.any() else None
                row["by_mass"].append(cell)
        out.append(row)
    return out


def pooled(x: np.ndarray, flags: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    out = {}
    for name, (lo, hi) in POOLED.items():
        sel = _rows_in(x, lo, hi, hi >= R_BINS[-1])
        out[name] = {"n": int(sel.sum()),
                     **{k: (float(np.mean(v[sel])) if sel.any() else None)
                        for k, v in flags.items()}}
    return out


def verdict(gate: Optional[Dict], res: Dict, cand_name: str,
            control: Optional[str]) -> Dict:
    """Pre-registered reading of the outskirt (1-3 R_vir) numbers."""
    rule = ("IC CARRIES PLACEMENT if the fine ripples are shared and the 256^3 run "
            "recovers >= 0.30 above its null at 1-3 R_vir, >= 0.20 more excess than "
            "the LR-only control (frozen SR2), with at least half of its matches from "
            "the same IC patch; NO if its excess is < 0.10; otherwise INCONCLUSIVE")
    o = res[cand_name]["pooled"]["outskirts_1_3"]
    s = res[control]["pooled"]["outskirts_1_3"] if control in res else None
    if o["n"] == 0 or o["match"] is None:
        return {"call": "NO DATA", "rule": rule}
    exc = o["match"] - o["null"]
    exc_sr2 = (s["match"] - s["null"]) if s and s["match"] is not None else float("nan")
    patch_share = (o["same_patch"] / o["match"]) if o["match"] else 0.0
    shared = bool(gate and gate.get("shared"))
    if not shared:
        call = ("NOT A TEST OF THE IC: dmo-256 does not share HR's fine ripples, so "
                "this measures what the large ripples alone fix")
    elif exc >= 0.30 and exc - exc_sr2 >= 0.20 and patch_share >= 0.5:
        call = "IC CARRIES PLACEMENT"
    elif exc < 0.10:
        call = "IC DOES NOT CARRY PLACEMENT"
    else:
        call = "INCONCLUSIVE"
    return {"call": call, "rule": rule, "excess_outskirts": exc,
            "excess_outskirts_sr2": exc_sr2, "same_patch_share": patch_share,
            "gate_shared": shared}


# ----------------------------------------------------------------- match

def _read_gate(path: Optional[str]) -> Optional[Dict]:
    import json
    if not path or not Path(path).is_file():
        print(f" (no ripples gate at {path}: the reading will not know whether "
              f"the fine ripples are shared)")
        return None
    return json.loads(Path(path).read_text()).get("gate")


def cmd_match(a) -> int:
    t0 = time.time()
    gate = _read_gate(a.gate_json)
    if gate is not None:
        print(f" ripples gate: shared={gate['shared']} -- {gate['reason']}")
    rng = np.random.default_rng(a.seed)
    hr = load_catalog(a.hr_catalog, 1.0)
    hosts = massive_hosts(hr, a.host_min_np)
    is_host = np.zeros(hr["np"].size, bool)
    is_host[hosts] = True
    rows = np.flatnonzero((hr["np"] >= a.obj_min_np) & ~is_host)
    rows, hidx, x = assign_to_hosts(hr, hosts, rows, a.r_min, a.r_max)
    np_obj = hr["np"][rows]
    print(f" HR: {hosts.size} hosts >= {a.host_min_np:.0f} p; {rows.size} objects "
          f">= {a.obj_min_np:.0f} p at {a.r_min}-{a.r_max} R_vir "
          f"({time.time() - t0:.0f}s)", flush=True)

    pos, ng_hr = particle_positions(a.hr_field, "catnorm")
    tree = particle_tree(pos)
    k_hr = np.clip(np_obj.astype(np.int64) // 2, 1, a.lag_k)
    q_hr = lagrangian_centroids(tree, ng_hr, hr["pos"][rows], k_hr)
    cons = {"hr": field_consistency(tree, hr, rows, seed=a.seed)}
    del tree, pos
    r_l = lagrangian_radius(np_obj)
    print(f" HR IC patches done; field/catalog n_in/num_p median "
          f"{cons['hr']:.2f} ({time.time() - t0:.0f}s)", flush=True)

    res: Dict = {}
    per_obj: Dict[str, np.ndarray] = {
        "hr_id": hr["id"][rows], "hr_host_id": hr["id"][hosts][hidx],
        "r_over_rvir": x, "np": np_obj, "q_hr": q_hr}
    for name, cat_path, field, kind in a.cand:
        ng = field_ng(field, kind)
        cand = load_catalog(cat_path, (NG_HR / ng) ** 3)
        m = match_catalogs(hr, hosts, rows, hidx, cand, mass_tol=a.mass_tol,
                           match_rvir=a.match_rvir, match_min_mpc=a.match_min_mpc,
                           n_null=a.n_null, rng=rng)
        got = m["match"] >= 0
        pos, ngc = particle_positions(field, kind)
        tree = particle_tree(pos)
        k_c = np.clip((cand["np"][m["match"][got]] / cand["np_ratio"]).astype(np.int64)
                      // 2, 1, a.lag_k)
        q_c = lagrangian_centroids(tree, ngc, cand["pos"][m["match"][got]], k_c)
        cand_rows = np.flatnonzero(cand["np"] >= a.obj_min_np)
        cons[name] = field_consistency(tree, cand, cand_rows, seed=a.seed)
        del tree, pos
        dq = np.full(rows.size, np.nan)
        dq[got] = np.linalg.norm(min_image(q_c - q_hr[got]), axis=1) / r_l[got]
        same_patch = got & (dq <= a.lag_tol)
        rx, rm, rnull = reverse_match(
            hr, hosts, cand, m["host_match"], obj_min_np=a.obj_min_np,
            host_min_np=a.host_min_np, r_min=a.r_min, r_max=a.r_max,
            mass_tol=a.mass_tol, match_rvir=a.match_rvir,
            match_min_mpc=a.match_min_mpc, n_null=a.n_null, rng=rng)
        flags = {"match": got.astype(float), "null": m["null_frac"],
                 "same_patch": same_patch.astype(float)}
        res[name] = {
            "catalog": cat_path, "field": field, "kind": kind, "ng": ng,
            "frame_offset_mpc": m["offset"],
            "hosts_matched": int((m["host_match"] >= 0).sum()),
            "objects_with_matched_host": float(m["host_matched"].mean()) if rows.size else None,
            "field_consistency": cons[name],
            "radial": radial_table(x, flags, np_obj), "pooled": pooled(x, flags),
            "median_dq_over_rl_matched": float(np.nanmedian(dq)) if got.any() else None,
            "reverse": {"radial": radial_table(rx, {"match": rm.astype(float),
                                                    "null": rnull}),
                        "pooled": pooled(rx, {"match": rm.astype(float),
                                              "null": rnull})}}
        per_obj[f"{name}_match_id"] = np.where(got, cand["id"][np.maximum(m["match"], 0)], -1)
        per_obj[f"{name}_null_frac"] = m["null_frac"]
        per_obj[f"{name}_dq_over_rl"] = dq
        print(f"\n === {name}: frame offset {np.round(m['offset'], 3)} Mpc/h, "
              f"{res[name]['hosts_matched']}/{hosts.size} hosts matched, field/catalog "
              f"n_in/num_p median {cons[name]:.2f} ({time.time() - t0:.0f}s)")
        print(f" {'r/Rvir':>10} {'n':>6} {'match':>6} {'null':>6} {'samepatch':>9}"
              f" | reverse {'n':>6} {'match':>6} {'null':>6}")
        for row, rrow in zip(res[name]["radial"], res[name]["reverse"]["radial"]):
            f = lambda v: "   nan" if v is None else f"{v:6.3f}"
            print(f" {row['r_lo']:4.2f}-{row['r_hi']:<4.2f} {row['n']:6d} "
                  f"{f(row['match'])} {f(row['null'])} {f(row['same_patch']):>9} |"
                  f"         {rrow['n']:6d} {f(rrow['match'])} {f(rrow['null'])}",
                  flush=True)
        if cons[name] < 0.3:
            print(f" WARNING {name}: catalog objects hold {cons[name]:.2f} of their "
                  f"num_p in this field -- catalog and field may not belong together")

    v = verdict(gate, res, a.cand[0][0], a.cand[1][0] if len(a.cand) > 1 else None)
    out = {"hr_catalog": a.hr_catalog, "hr_field": a.hr_field, "gate": gate,
           "params": {k: getattr(a, k) for k in (
               "host_min_np", "obj_min_np", "r_min", "r_max", "mass_tol",
               "match_rvir", "match_min_mpc", "n_null", "lag_k", "lag_tol", "seed")},
           "r_bins": R_BINS, "m_bins": [m for m in M_BINS if np.isfinite(m)],
           "n_hosts": int(hosts.size), "n_objects": int(rows.size),
           "field_consistency_hr": cons["hr"], "candidates": res,
           "primary": a.cand[0][0], "verdict": v, "seconds": time.time() - t0}
    write_json(a.out, out)
    np.savez_compressed(Path(a.out).with_suffix(".objects.npz"), **per_obj)
    print(f"\n VERDICT: {v['call']}")
    print(f" rule: {v['rule']}")
    print(f" wrote {a.out} (+ .objects.npz)  ({time.time() - t0:.0f}s)")
    return 0


# ----------------------------------------------------------------- track

def lattice_centroid(m: np.ndarray, ng: int) -> np.ndarray:
    """Periodic mean IC position (Mpc/h) of flat Lagrangian indices m."""
    return periodic_mean((np.stack(np.unravel_index(m, (ng,) * 3), axis=1) + 0.5)
                         * (BOX / ng))


def per_bin(x: np.ndarray, fn) -> List[Dict]:
    out = []
    edges = list(zip(R_BINS[:-1], R_BINS[1:]))
    for i, (lo, hi) in enumerate(edges):
        sel = _rows_in(x, lo, hi, i == len(edges) - 1)
        row = {"r_lo": lo, "r_hi": hi, "n": int(sel.sum())}
        if sel.any():
            row.update(fn(sel))
        out.append(row)
    return out


def per_pool(x: np.ndarray, fn) -> Dict[str, Dict]:
    out = {}
    for name, (lo, hi) in POOLED.items():
        sel = _rows_in(x, lo, hi, hi >= R_BINS[-1])
        out[name] = {"n": int(sel.sum()), **(fn(sel) if sel.any() else {})}
    return out


def _med(v: np.ndarray) -> Optional[float]:
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else None


def _frac(v: np.ndarray) -> Optional[float]:
    return float(np.mean(v)) if v.size else None


def cmd_track(a) -> int:
    """Two questions per HR object, in each candidate field:
    1. the same matter: where do this object's own core particles end up, and how
       compact are they? (does the candidate know WHERE the matter goes?)
    2. the same object: the candidate object built from the same IC patch,
       wherever it sits -- same radius? same direction? (orbit size vs phase)
    """
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    hr = load_catalog(a.hr_catalog, 1.0)
    hosts = massive_hosts(hr, a.host_min_np)
    is_host = np.zeros(hr["np"].size, bool)
    is_host[hosts] = True
    rows = np.flatnonzero((hr["np"] >= a.obj_min_np) & ~is_host)
    rows, hidx, x = assign_to_hosts(hr, hosts, rows, a.r_min, a.r_max)
    np_obj, rvir_obj = hr["np"][rows], hr["rvir"][rows]
    rvir_host = hr["rvir"][hosts][hidx]
    dvec = min_image(hr["pos"][rows] - hr["pos"][hosts][hidx])
    r_l = lagrangian_radius(np_obj)
    radius = np.maximum(a.match_min_mpc, a.match_rvir * rvir_obj)
    print(f" HR: {hosts.size} hosts, {rows.size} objects ({time.time() - t0:.0f}s)",
          flush=True)

    pos, ng_hr = particle_positions(a.hr_field, "catnorm")
    tree = particle_tree(pos)
    members = core_particles(tree, hr["pos"][rows],
                             np.clip(np_obj.astype(np.int64) // 2, 1, a.lag_k))
    _, s_hr, f_hr = matter_stats(pos, members, rvir_obj)
    del tree, pos
    q_hr = np.array([lattice_centroid(m, ng_hr) for m in members]).reshape(-1, 3)
    print(f" HR core particles + IC patches ({time.time() - t0:.0f}s)", flush=True)

    tol, ltol = a.radius_tol, np.log(a.radius_tol)
    res: Dict = {}
    per_obj: Dict[str, np.ndarray] = {"hr_id": hr["id"][rows], "r_over_rvir": x,
                                      "np": np_obj, "hr_compact": f_hr}
    for name, cat_path, field, kind in a.cand:
        ng = field_ng(field, kind)
        cand = load_catalog(cat_path, (NG_HR / ng) ** 3)
        off = frame_offset(hr, hosts, cand)
        hmatch = match_hosts(hr, hosts, cand, off)
        hm = hmatch[hidx]
        centre = np.where((hm >= 0)[:, None], cand["pos"][np.maximum(hm, 0)],
                          wrap(hr["pos"][hosts][hidx] + off))
        pred = wrap(centre + dvec)

        pos, ngc = particle_positions(field, kind)
        mapped = [map_to_grid(m, ng_hr, ngc) for m in members]
        c_m, s_m, f_m = matter_stats(pos, mapped, rvir_obj)
        off_m = np.linalg.norm(min_image(c_m - pred), axis=1)
        r_m, cos_m = radial_geometry(c_m, centre, dvec, rvir_host)

        tree = particle_tree(pos)
        hc = hmatch[hmatch >= 0]
        pool = np.flatnonzero((cand["np"] >= a.obj_min_np / a.mass_tol)
                              & (cand["np"] < a.host_min_np))
        rows_c, _, _ = assign_to_hosts(cand, hc, pool, 0.0, a.pool_rmax)
        kc = np.clip((cand["np"][rows_c] / cand["np_ratio"]).astype(np.int64) // 2,
                     1, a.lag_k)
        q_c = lagrangian_centroids(tree, ngc, cand["pos"][rows_c], kc)
        del tree, pos
        sp = same_patch_objects(cand, rows_c, q_c, q_hr, r_l, np_obj, a.lag_tol,
                                a.mass_tol)
        got = sp >= 0
        r_p, cos_p = np.full(rows.size, np.nan), np.full(rows.size, np.nan)
        place_p = np.full(rows.size, np.nan)
        if got.any():
            r_p[got], cos_p[got] = radial_geometry(cand["pos"][sp[got]], centre[got],
                                                   dvec[got], rvir_host[got])
            place_p[got] = np.linalg.norm(min_image(cand["pos"][sp[got]] - pred[got]),
                                          axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr_m = np.abs(np.log(r_m / x))
            lr_p = np.abs(np.log(r_p / x))
        null_r = np.full(rows.size, np.nan)
        if got.sum() >= 2:
            null_r[got] = shuffled_within_bins(x[got], x[got], r_p[got], tol, rng)

        def stats(sel, got=got):
            g = got[sel]
            return {
                "matter_offset_mpc": _med(off_m[sel]),
                "matter_offset_over_rvir_obj": _med(off_m[sel] / rvir_obj[sel]),
                "matter_r_ratio": _med(r_m[sel] / x[sel]),
                "matter_same_r": _frac(lr_m[sel] <= ltol),
                "matter_cos": _med(cos_m[sel]),
                "matter_compact": _med(f_m[sel]),
                "matter_spread_over_hr": _med(s_m[sel] / s_hr[sel]),
                "hr_compact": _med(f_hr[sel]),
                "rvir_obj_mpc": _med(rvir_obj[sel]),
                "patch_found": _frac(g.astype(float)),
                "patch_same_place": _frac(place_p[sel][g] <= radius[sel][g]),
                "patch_same_r": _frac(lr_p[sel][g] <= ltol),
                "patch_same_r_null": _frac(null_r[sel][g]),
                "patch_same_dir": _frac(cos_p[sel][g] >= a.dir_cos),
                "patch_same_dir_null": (1.0 - a.dir_cos) / 2.0,
                "patch_abs_log_r": _med(lr_p[sel][g]),
            }

        res[name] = {"catalog": cat_path, "field": field, "kind": kind, "ng": ng,
                     "frame_offset_mpc": off, "hosts_matched": int(hc.size),
                     "pool_objects": int(rows_c.size),
                     "radial": per_bin(x, stats), "pooled": per_pool(x, stats)}
        for k, v in (("matter_offset_mpc", off_m), ("matter_r_over_rvir", r_m),
                     ("matter_cos", cos_m), ("matter_compact", f_m),
                     ("patch_id", np.where(got, cand["id"][np.maximum(sp, 0)], -1)),
                     ("patch_r_over_rvir", r_p), ("patch_cos", cos_p),
                     ("patch_offset_mpc", place_p)):
            per_obj[f"{name}_{k}"] = v

        print(f"\n === {name}: {hc.size}/{hosts.size} hosts matched, {rows_c.size} "
              f"pool objects ({time.time() - t0:.0f}s)")
        print("   (A) the same matter: where HR's object particles end up")
        print(f" {'r/Rvir':>10} {'n':>5} {'offset':>7} {'off/rv':>6} {'r/r_HR':>6}"
              f" {'same r':>6} {'cos':>5} {'compact':>7} {'HR cmp':>6}")
        f = lambda v, w=6, p=2: f"{'nan':>{w}}" if v is None else f"{v:{w}.{p}f}"
        for row in res[name]["radial"]:
            g = row.get
            print(f" {row['r_lo']:4.2f}-{row['r_hi']:<4.2f} {row['n']:5d}"
                  f" {f(g('matter_offset_mpc'), 7)} {f(g('matter_offset_over_rvir_obj'), 6, 1)}"
                  f" {f(g('matter_r_ratio'))} {f(g('matter_same_r'))}"
                  f" {f(g('matter_cos'), 5)} {f(g('matter_compact'), 7)} {f(g('hr_compact'))}")
        print(f"   (B) the same object (same IC patch), found anywhere; radius within "
              f"x{tol:g}, direction cos >= {a.dir_cos:g}")
        print(f" {'r/Rvir':>10} {'n':>5} {'found':>6} {'place':>6} {'same r':>6}"
              f" {'null':>5} {'samedir':>7} {'null':>5} {'|dlnr|':>6}")
        for row in res[name]["radial"]:
            g = row.get
            print(f" {row['r_lo']:4.2f}-{row['r_hi']:<4.2f} {row['n']:5d}"
                  f" {f(g('patch_found'))} {f(g('patch_same_place'))}"
                  f" {f(g('patch_same_r'))} {f(g('patch_same_r_null'), 5)}"
                  f" {f(g('patch_same_dir'), 7)} {f(g('patch_same_dir_null'), 5)}"
                  f" {f(g('patch_abs_log_r'))}", flush=True)

    out = {"hr_catalog": a.hr_catalog, "hr_field": a.hr_field,
           "params": {k: getattr(a, k) for k in (
               "host_min_np", "obj_min_np", "r_min", "r_max", "mass_tol",
               "match_rvir", "match_min_mpc", "lag_k", "lag_tol", "radius_tol",
               "dir_cos", "pool_rmax", "seed")},
           "r_bins": R_BINS, "n_hosts": int(hosts.size), "n_objects": int(rows.size),
           "candidates": res, "primary": a.cand[0][0], "seconds": time.time() - t0}
    write_json(a.out, out)
    np.savez_compressed(Path(a.out).with_suffix(".objects.npz"), **per_obj)
    print(f"\n wrote {a.out} (+ .objects.npz)  ({time.time() - t0:.0f}s)")
    return 0


# --------------------------------------------------------------- scaling

#: object size in particles of the COARSE INPUT (the run whose ripples are given)
SCALE_BINS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 3000)
SCALE_RANGES = {"outskirts_1_3": (1.0, 3.0), "mid_0.25_1": (0.25, 1.0)}


def scaling_table(n_coarse: np.ndarray, x: np.ndarray, flags: Dict[str, np.ndarray],
                  lo: float, hi: float, edges=SCALE_BINS) -> List[Dict]:
    """Per bin of object size in coarse-input particles, at lo <= r/R_vir <= hi."""
    in_r = (x >= lo) & (x <= hi)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = in_r & (n_coarse >= a) & (n_coarse < b)
        row = {"n_lo": a, "n_hi": b, "n": int(sel.sum())}
        for k, v in flags.items():
            row[k] = float(np.mean(v[sel])) if sel.any() else None
        rows.append(row)
    return rows


def cmd_scaling(a) -> int:
    """Is recovery a function of how many of the INPUT's own particles an object
    has? If the 256^3 run, the 128^3 run and SR2 (64^3 input) fall on one curve,
    an object is recoverable once the input resolves the ripple that forms it --
    which full 512^3 ICs do even for HR's smallest subhalos."""
    out: Dict = {"bins": SCALE_BINS, "ranges": SCALE_RANGES, "lag_tol": a.lag_tol,
                 "runs": {}}
    for label, npz, name, ng in a.run:
        d = np.load(npz)
        n_coarse = d["np"] * (int(ng) / NG_HR) ** 3
        with np.errstate(invalid="ignore"):
            flags = {"match": (d[f"{name}_match_id"] >= 0).astype(float),
                     "null": d[f"{name}_null_frac"],
                     "same_patch": (d[f"{name}_dq_over_rl"] <= a.lag_tol).astype(float)}
        out["runs"][label] = {
            "npz": npz, "cand": name, "coarse_ng": int(ng),
            "finest_ripple_mpc": 2 * BOX / int(ng),
            **{k: scaling_table(n_coarse, d["r_over_rvir"], flags, lo, hi)
               for k, (lo, hi) in SCALE_RANGES.items()}}
    for k in SCALE_RANGES:
        print(f"\n {k}: match (null) per object size in the input's own particles")
        print(f" {'n_coarse':>11} " + " ".join(f"{lab:>18}" for lab in out["runs"]))
        for i, (lo, hi) in enumerate(zip(SCALE_BINS[:-1], SCALE_BINS[1:])):
            cells = []
            for lab, r in out["runs"].items():
                row = r[k][i]
                cells.append(f"{'':>18}" if row["match"] is None else
                             f"{row['match']:5.2f} ({row['null']:.2f}) n={row['n']:<4d}")
            print(f" {lo:>5}-{hi:<5} " + " ".join(cells))
    write_json(a.out, out)
    print(f"\n wrote {a.out}")
    return 0


# ------------------------------------------------------------------ main

def _selection_args(m: argparse.ArgumentParser) -> None:
    m.add_argument("--hr-catalog", required=True)
    m.add_argument("--hr-field", required=True)
    m.add_argument("--cand", nargs=4, action="append", required=True,
                   metavar=("NAME", "CATALOG", "FIELD", "KIND"),
                   help="KIND: raw (dir with disp.npy, kpc/h) or catnorm (6,Ng^3); "
                        "the first --cand is the primary one")
    m.add_argument("--host-min-np", type=float, default=20000)
    m.add_argument("--obj-min-np", type=float, default=400)
    m.add_argument("--r-min", type=float, default=R_BINS[0])
    m.add_argument("--r-max", type=float, default=R_BINS[-1])
    m.add_argument("--mass-tol", type=float, default=3.0)
    m.add_argument("--match-rvir", type=float, default=1.0)
    m.add_argument("--match-min-mpc", type=float, default=0.1)
    m.add_argument("--lag-k", type=int, default=512)
    m.add_argument("--lag-tol", type=float, default=1.0,
                   help="same IC patch if the centroids differ by <= this x R_L")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", required=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("ripples", help="r(k): do the 256^3 and 512^3 runs share fine ripples?")
    r.add_argument("--hr-field", required=True, help="(6,512^3) catnorm HR field")
    r.add_argument("--sr2-field", required=True, help="(6,512^3) catnorm frozen SR2 field")
    r.add_argument("--dmo-dir", required=True, help="dir with raw disp.npy/vel.npy (3,Ng^3)")
    r.add_argument("--nbins", type=int, default=32)
    r.add_argument("--band", type=float, nargs=2, default=(2.5, 6.0),
                   help="gate band in h/Mpc (between the LR and dmo Nyquists)")
    r.add_argument("--gate-min", type=float, default=0.3)
    r.add_argument("--gate-margin", type=float, default=0.2)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_ripples)

    s = sub.add_parser("rockstar", help="Rockstar on a raw dmo box")
    s.add_argument("--dmo-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--tag", default="dmo256")
    s.add_argument("--redshift", type=float, default=0.0)
    s.set_defaults(func=cmd_rockstar)

    m = sub.add_parser("match", help="host-relative + IC-patch match against HR")
    _selection_args(m)
    m.add_argument("--gate-json", default=None, help="ripples output (for the reading)")
    m.add_argument("--n-null", type=int, default=10)
    m.set_defaults(func=cmd_match)

    t = sub.add_parser("track", help="same matter + same-IC-patch object, radius vs direction")
    _selection_args(t)
    t.add_argument("--radius-tol", type=float, default=1.25,
                   help="'same radius' if r_cand / r_HR is within this factor")
    t.add_argument("--dir-cos", type=float, default=0.9,
                   help="'same direction' if the host-relative cosine is >= this")
    t.add_argument("--pool-rmax", type=float, default=3.5,
                   help="candidate objects searched for the same IC patch: within this "
                        "many R_vir of a matched host")
    t.set_defaults(func=cmd_track)

    c = sub.add_parser("scaling", help="recovery vs object size in the input's own particles")
    c.add_argument("--run", nargs=4, action="append", required=True,
                   metavar=("LABEL", "MATCH_NPZ", "CAND", "COARSE_NG"),
                   help="a match .objects.npz, the candidate name inside it, and the "
                        "grid of the input whose ripples it had (SR2: 64)")
    c.add_argument("--lag-tol", type=float, default=1.0)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_scaling)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
