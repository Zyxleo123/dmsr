#!/usr/bin/env python
"""Can an LR-derived host channel carry the (centre, scale) the SR operator lacks?

MEASURED context that makes this the gating question. On held-out set9,
region-restricted, HR's median subhalo host-centric radius scales with host
mass over a factor of 6.5 (0.181 Mpc/h at 200-1k p to 1.186 at >=100k p, i.e.
``r ~ R_vir ~ M^(1/3)``), while every candidate catalog is FLAT at ~0.1 Mpc/h.
A local translation-equivariant operator fed only ``(lr, noise)`` cannot
separate "1.2 Mpc from a cluster" from "0.3 Mpc from a group" -- the two look
alike locally -- so it emits the conditional mean, which IS a flat profile.
Fixing that means giving the model a non-local channel carrying the host's
centre and scale.

Two candidate channels, and this job prices both against each other:

  * ``catalog`` -- pre-run Rockstar on the LR box and feed, per position, the
    nearest resolved host's centre / mass / R_vir. Explicit and exactly the
    quantity wanted, but LR is **64^3** (262,144 particles, m_p 2.98e11
    Msun/h), so a 20-particle floor is ~6e12 Msun/h: everything below roughly
    10^4 HR particles has NO LR host at all, and the channel falls off a cliff
    rather than degrading.
  * ``potential`` -- the LR gravitational potential from one FFT
    (``phi_k = -delta_k / k^2``) at a few smoothing scales. No halo finder, no
    completeness cliff, smooth and differentiable; a 10^13 halo still dents it
    even when Rockstar cannot resolve it as an object. But it carries no
    explicit R_vir, so the network must infer the scale.

Neither is leakage: both are computed from LR alone and are available at
inference for every LR box.

The verdict this job returns, per HR host-mass bin:

  * completeness   -- fraction of HR hosts with an LR host inside R_vir
  * centre error   -- |x_LR - x_HR|, in Mpc/h and in HR R_vir
  * scale recovery -- out-of-sample R^2 for log10 Mvir_HR from each channel
  * **the decision number** -- centre error / HR's own median subhalo radius
    for hosts in that bin. Below ~0.3 the channel is a usable coordinate;
    at or above 1 it is blur and cannot place a subhalo at all.

CPU-only and submitted. The LR Rockstar run is small (262k particles) but the
HR ASCII parse is real work the login node kills.

Writes, under ``--out-dir``:
  * ``<box>_hosts.jsonl`` -- one row per HR host (every field), for re-plots
  * ``<box>.npz``         -- the same as arrays
  * ``summary.json``      -- per-box, per-mass-bin verdict table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import add_common_args, load_reward_config, lr_path, write_json  # noqa: E402
from gather_holdout_figures_data import BOX, catalog_path  # noqa: E402
from cosmo_sr.eval.particles import field_to_particles  # noqa: E402
from cosmo_sr.eval.rockstar import (  # noqa: E402
    HaloCatalog, load_rockstar_ascii, run_rockstar_on_particles,
)

LR_CFG = PROJECT_ROOT / "configs" / "sr2_baseline" / "rockstar_lr_particles.cfg"

#: HR host num_p bins -- the same ladder the radial measurement used, so the
#: "centre error vs median subhalo radius" comparison is bin-for-bin.
HOST_BINS = [(200, 1000), (1000, 5000), (5000, 20000),
             (20000, 100000), (100000, 10 ** 12)]
HOST_LABELS = ["200-1k", "1k-5k", "5k-20k", "20k-100k", "100k+"]


def periodic_delta(a: np.ndarray, b: np.ndarray, box: float = BOX) -> np.ndarray:
    """Minimum-image ``a - b``; broadcasts over the leading axes."""
    d = a - b
    return d - box * np.round(d / box)


# --------------------------------------------------------------------------- #
# The LR catalog
# --------------------------------------------------------------------------- #
def lr_catalog(cfg, box: str, work_root: Path, *, rerun: bool):
    """``(catalog, particles)`` for the LR box, running Rockstar if needed.

    Reuses an existing ``<box>_lr_rockstar`` -- ``build_lagrangian_host.py``
    already produced one for set8 under the SAME config, and re-finding it
    would only risk a different answer for no gain.
    """
    work = work_root / box
    work.mkdir(parents=True, exist_ok=True)
    field = np.load(lr_path(cfg, box), mmap_mode="r")
    particles = field_to_particles(np.asarray(field), boxsize_kpc_h=BOX * 1000.0)
    existing = sorted((work / f"{box}_lr_rockstar").glob("halos_*.ascii"))
    if existing and not rerun:
        print(f"  reusing LR catalog {existing[0]}")
        return load_rockstar_ascii(existing[0]), particles
    print(f"  running Rockstar on {particles.pos_mpc_h.shape[0]} LR particles "
          f"(m_p {particles.particle_mass_msun_h:.3e} Msun/h)")
    cat = run_rockstar_on_particles(
        particles, work, cfg=LR_CFG, tag=f"{box}_lr", overwrite=bool(rerun))
    return cat, particles


# --------------------------------------------------------------------------- #
# The potential channel
# --------------------------------------------------------------------------- #
def cic_overdensity(pos_mpc: np.ndarray, ng: int, box: float = BOX) -> np.ndarray:
    """Cloud-in-cell overdensity ``delta`` on an ``ng^3`` periodic grid."""
    # Cell CENTRES, matching `sample_grid` and `local_minima`. Depositing on
    # corners and reading back on centres would put a half-cell (0.78 Mpc/h at
    # ng=64) systematic into every potential distance this job reports.
    g = (pos_mpc / box) * ng - 0.5
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    rho = np.zeros((ng, ng, ng), dtype=np.float64)
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1.0 - f[:, 0]
        ix = (i0[:, 0] + dx) % ng
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1.0 - f[:, 1]
            iy = (i0[:, 1] + dy) % ng
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1.0 - f[:, 2]
                iz = (i0[:, 2] + dz) % ng
                np.add.at(rho, (ix, iy, iz), wx * wy * wz)
    mean = rho.mean()
    if mean <= 0:
        raise ValueError("empty CIC grid")
    return rho / mean - 1.0


def potential_grids(delta: np.ndarray, scales_mpc: List[float],
                    box: float = BOX) -> Dict[float, np.ndarray]:
    """``phi_k = -delta_k / k^2``, Gaussian-smoothed at each scale.

    Units are arbitrary (the 4 pi G rho_bar prefactor is a constant), which is
    all this job needs: it scores R^2 and distances, never an absolute depth.
    Each grid is standardised so the regression design is well conditioned.
    """
    ng = delta.shape[0]
    kf = 2.0 * np.pi / box
    k1 = np.fft.fftfreq(ng, d=1.0 / ng) * kf
    kx = k1.reshape(ng, 1, 1)
    ky = k1.reshape(1, ng, 1)
    kz = np.fft.rfftfreq(ng, d=1.0 / ng).reshape(1, 1, -1) * kf
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    dk = np.fft.rfftn(delta)
    inv = np.zeros_like(k2)
    nz = k2 > 0
    inv[nz] = -1.0 / k2[nz]
    phik = dk * inv
    out: Dict[float, np.ndarray] = {}
    for R in scales_mpc:
        w = np.exp(-0.5 * k2 * float(R) ** 2) if R > 0 else 1.0
        phi = np.fft.irfftn(phik * w, s=delta.shape, axes=(0, 1, 2))
        sd = phi.std()
        out[float(R)] = (phi - phi.mean()) / (sd if sd > 0 else 1.0)
    return out


def sample_grid(grid: np.ndarray, pos_mpc: np.ndarray, box: float = BOX) -> np.ndarray:
    """Trilinear sample of a periodic grid at ``(N,3)`` positions."""
    ng = grid.shape[0]
    g = (pos_mpc / box) * ng - 0.5
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    out = np.zeros(pos_mpc.shape[0], dtype=np.float64)
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1.0 - f[:, 0]
        ix = (i0[:, 0] + dx) % ng
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1.0 - f[:, 1]
            iy = (i0[:, 1] + dy) % ng
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1.0 - f[:, 2]
                iz = (i0[:, 2] + dz) % ng
                out += wx * wy * wz * grid[ix, iy, iz]
    return out


def local_minima(grid: np.ndarray, box: float = BOX) -> np.ndarray:
    """``(M,3)`` Mpc/h centres of cells strictly below all 26 neighbours."""
    ng = grid.shape[0]
    keep = np.ones_like(grid, dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                keep &= grid < np.roll(grid, (dx, dy, dz), axis=(0, 1, 2))
    idx = np.argwhere(keep)
    return (idx + 0.5) * (box / ng)


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def nearest(query: np.ndarray, ref: np.ndarray, box: float = BOX,
            chunk: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """``(index, distance)`` of the nearest ``ref`` point to each ``query``.

    Brute force in chunks. The reference set is the LR host list (hundreds to
    a few thousand rows), so this is tens of millions of float ops -- cheaper
    and far more predictable than a periodic tree, and it has no dependency.
    """
    n = query.shape[0]
    if ref.shape[0] == 0:
        return np.full(n, -1, dtype=np.int64), np.full(n, np.inf)
    out_i = np.empty(n, dtype=np.int64)
    out_d = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        q = query[s:s + chunk]
        d = periodic_delta(q[:, None, :], ref[None, :, :], box)
        dist = np.sqrt((d ** 2).sum(axis=2))
        j = np.argmin(dist, axis=1)
        out_i[s:s + chunk] = j
        out_d[s:s + chunk] = dist[np.arange(q.shape[0]), j]
    return out_i, out_d


def holdout_r2(x: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Out-of-sample R^2 of a linear fit ``y ~ [1, x]``, 50/50 split by host.

    In-sample R^2 on a 4-feature fit over a few hundred hosts would flatter the
    channel; the question is whether the relation transfers, so it is scored on
    hosts the fit never saw.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.shape[0] != y.shape[0]:
        x = x.T
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[ok], y[ok]
    if y.size < 20:
        return float("nan")
    rng = np.random.default_rng(seed)
    m = rng.random(y.size) < 0.5
    if m.sum() < 10 or (~m).sum() < 10:
        return float("nan")
    A = np.c_[np.ones(y.size), x]
    beta, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
    pred = A[~m] @ beta
    yy = y[~m]
    denom = ((yy - yy.mean()) ** 2).sum()
    if denom <= 0:
        return float("nan")
    return float(1.0 - ((yy - pred) ** 2).sum() / denom)


# --------------------------------------------------------------------------- #
# One box
# --------------------------------------------------------------------------- #
def run_box(cfg, box: str, args, out_dir: Path) -> Dict:
    print(f"\n=== {box}")
    hr = load_rockstar_ascii(catalog_path(Path(args.reward_root), "hr", box))
    print(f"  HR catalog {hr.n} rows")
    lr, lrpart = lr_catalog(cfg, box, Path(args.lr_work_root),
                            rerun=args.rerun_rockstar)
    m_lr = float(lrpart.particle_mass_msun_h)

    hr_host = np.flatnonzero((hr.parent_ids < 0) & (hr.num_p >= args.hr_min_num_p))
    lr_host = np.flatnonzero(lr.parent_ids < 0)
    print(f"  HR hosts >={args.hr_min_num_p}p: {hr_host.size}    "
          f"LR top-level halos: {lr_host.size}")
    if lr_host.size == 0:
        raise SystemExit(f"{box}: LR catalog has no top-level halos")

    hpos = np.asarray(hr.pos[hr_host], dtype=np.float64)
    hrvir = np.asarray(hr.rvir[hr_host], dtype=np.float64) / 1000.0   # kpc/h -> Mpc/h
    hnump = np.asarray(hr.num_p[hr_host], dtype=np.int64)
    hmvir = np.asarray(hr.mvir[hr_host], dtype=np.float64)
    lpos = np.asarray(lr.pos[lr_host], dtype=np.float64)

    j, dr = nearest(hpos, lpos, BOX)
    lr_np = np.asarray(lr.num_p[lr_host], dtype=np.int64)[j]
    lr_mv = np.asarray(lr.mvir[lr_host], dtype=np.float64)[j]
    # How many HR hosts land on the SAME LR host. This is a fidelity limit in
    # its own right -- a shared match means the channel cannot tell those HR
    # hosts apart -- and it is the bug `host-row-overlay-collision` recorded as
    # a viewer artefact. Counted, not silently resolved.
    share = np.bincount(j, minlength=lr_host.size)[j]

    # --- the potential channel -------------------------------------------- #
    delta = cic_overdensity(np.asarray(lrpart.pos_mpc_h, dtype=np.float64),
                            args.pot_ng, BOX)
    scales = [float(s) for s in str(args.pot_scales).split(",") if s.strip()]
    phis = potential_grids(delta, scales, BOX)
    phi_at = np.column_stack([sample_grid(phis[s], hpos, BOX) for s in scales])
    # Localisation. MEASURED 2026-09-10, and it is why this is a parameter and
    # not `max(scales)`: at R=6.25 Mpc/h (4 LR cells) the whole 100 Mpc/h box
    # holds THREE local minima, so "distance to the nearest one" was a 30 Mpc/h
    # number about nothing. A cluster's R_vir is ~1.0 Mpc/h -- SMALLER than the
    # 1.5625 Mpc/h LR cell -- so the basin must be resolved at cell scale or the
    # measurement is about the smoothing, not the field.
    min_scale = float(args.pot_min_scale)
    if min_scale not in phis:
        raise SystemExit(f"--pot-min-scale {min_scale} is not among --pot-scales "
                         f"{sorted(phis)}")
    n_minima = {float(s): int(local_minima(phis[s], BOX).shape[0]) for s in scales}
    minima = local_minima(phis[min_scale], BOX)
    _, pot_dr = nearest(hpos, minima, BOX)
    print(f"  potential: {len(scales)} scales, local minima per scale "
          + ", ".join(f"R={s}:{n}" for s, n in sorted(n_minima.items()))
          + f"  (localising on R={min_scale})")

    # --- HR's own median subhalo radius per host, the yardstick ------------ #
    sub = np.flatnonzero(hr.parent_ids >= 0)
    id_to_row = {int(v): i for i, v in enumerate(hr.ids)}
    host_row_of_sub = np.array([id_to_row.get(int(p), -1)
                                for p in hr.parent_ids[sub]], dtype=np.int64)
    ok = host_row_of_sub >= 0
    sub, host_row_of_sub = sub[ok], host_row_of_sub[ok]
    sub_r = np.sqrt((periodic_delta(np.asarray(hr.pos[sub], dtype=np.float64),
                                    np.asarray(hr.pos[host_row_of_sub],
                                               dtype=np.float64), BOX) ** 2
                     ).sum(axis=1))
    med_sub_r = {}
    for hrow, rr in zip(host_row_of_sub, sub_r):
        med_sub_r.setdefault(int(hrow), []).append(float(rr))

    # What the operator would actually read: phi at ITS OWN position. Regress
    # each HR subhalo's true host-centric radius on the multi-scale phi sampled
    # where that subhalo sits. This is the conditioning question stated exactly
    # -- "how far into a host am I, in units of its R_vir?" -- and unlike a
    # minimum-finder it is sub-cell, because trilinear sampling of a smooth
    # field is.
    sub_pos = np.asarray(hr.pos[sub], dtype=np.float64)
    sub_phi = np.column_stack([sample_grid(phis[s], sub_pos, BOX) for s in scales])
    sub_host_rvir = np.asarray(hr.rvir[host_row_of_sub], dtype=np.float64) / 1000.0
    sub_host_np = np.asarray(hr.num_p[host_row_of_sub], dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sub_log_r = np.log10(np.where(sub_r > 0, sub_r, np.nan))
        sub_log_rrv = np.log10(np.where((sub_r > 0) & (sub_host_rvir > 0),
                                        sub_r / np.maximum(sub_host_rvir, 1e-9),
                                        np.nan))
    print(f"  subhalo-position probe over {sub.size} HR subhalos")

    rows: List[Dict] = []
    for k, row in enumerate(hr_host):
        rs = med_sub_r.get(int(row), [])
        rows.append({
            "hr_row": int(row), "hr_id": int(hr.ids[row]),
            "num_p": int(hnump[k]), "log_mvir": float(np.log10(max(hmvir[k], 1.0))),
            "rvir_mpc": float(hrvir[k]),
            "lr_dr_mpc": float(dr[k]),
            "lr_dr_over_rvir": float(dr[k] / hrvir[k]) if hrvir[k] > 0 else float("nan"),
            "lr_num_p": int(lr_np[k]),
            "lr_log_mvir": float(np.log10(max(lr_mv[k], 1.0))),
            "lr_share": int(share[k]),
            "pot_dr_mpc": float(pot_dr[k]),
            "pot_dr_over_rvir": float(pot_dr[k] / hrvir[k]) if hrvir[k] > 0 else float("nan"),
            "phi": [float(v) for v in phi_at[k]],
            "n_sub": len(rs),
            "median_sub_r_mpc": float(np.median(rs)) if rs else float("nan"),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{box}_hosts.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    np.savez_compressed(
        out_dir / f"{box}.npz",
        num_p=hnump, log_mvir=np.array([r["log_mvir"] for r in rows]),
        rvir_mpc=hrvir, lr_dr_mpc=dr,
        lr_dr_over_rvir=np.array([r["lr_dr_over_rvir"] for r in rows]),
        lr_num_p=lr_np, lr_log_mvir=np.array([r["lr_log_mvir"] for r in rows]),
        lr_share=share, pot_dr_mpc=pot_dr,
        pot_dr_over_rvir=np.array([r["pot_dr_over_rvir"] for r in rows]),
        phi=phi_at, n_sub=np.array([r["n_sub"] for r in rows]),
        median_sub_r_mpc=np.array([r["median_sub_r_mpc"] for r in rows]),
        pot_scales=np.array(scales),
    )

    # --- the verdict table ------------------------------------------------- #
    #: HR's OWN median subhalo radius per host -- the yardstick the centre
    #: error is judged against. NaN for a host with no subhalos.
    med_sub_arr = np.array([r["median_sub_r_mpc"] for r in rows], dtype=np.float64)
    bins = []
    for (lo, hi), lab in zip(HOST_BINS, HOST_LABELS):
        m = (hnump >= lo) & (hnump < hi)
        if not m.any():
            bins.append({"bin": lab, "n_hosts": 0})
            continue
        inside = dr[m] <= args.match_rvir * hrvir[m]
        sel = med_sub_arr[m]
        med_r = float(np.nanmedian(sel)) if np.isfinite(sel).any() else float("nan")
        cerr = float(np.median(dr[m]))
        perr = float(np.median(pot_dr[m]))
        row = {
            "bin": lab, "n_hosts": int(m.sum()),
            "completeness_within_rvir": float(inside.mean()),
            "lr_centre_err_mpc_median": cerr,
            "lr_centre_err_over_rvir_median": float(np.median(dr[m] / np.maximum(hrvir[m], 1e-9))),
            "pot_centre_err_mpc_median": perr,
            "hr_median_sub_r_mpc": med_r,
            "shared_lr_host_frac": float((share[m] > 1).mean()),
            "lr_num_p_median": float(np.median(lr_np[m])),
            "r2_logM_from_lr_catalog": holdout_r2(
                np.log10(np.maximum(lr_mv[m], 1.0)), np.log10(np.maximum(hmvir[m], 1.0))),
            "r2_logM_from_potential": holdout_r2(phi_at[m], np.log10(np.maximum(hmvir[m], 1.0))),
        }
        sm = (sub_host_np >= lo) & (sub_host_np < hi)
        row["n_sub_in_bin"] = int(sm.sum())
        row["r2_log_r_from_potential"] = holdout_r2(sub_phi[sm], sub_log_r[sm])
        row["r2_log_r_over_rvir_from_potential"] = holdout_r2(sub_phi[sm], sub_log_rrv[sm])
        for key, err in (("lr", cerr), ("pot", perr)):
            if not np.isfinite(med_r) or med_r <= 0:
                row[f"{key}_verdict"] = "NO YARDSTICK"
            else:
                ratio = err / med_r
                row[f"{key}_err_over_sub_r"] = float(ratio)
                row[f"{key}_verdict"] = ("USABLE" if ratio < 0.3 else
                                         "MARGINAL" if ratio < 1.0 else "BLUR")
        bins.append(row)

    # Whole-population scale recovery, the headline for the channel design.
    allm = np.ones(hnump.size, dtype=bool)
    summary = {
        "box": box, "n_hr_hosts": int(hr_host.size), "n_lr_hosts": int(lr_host.size),
        "lr_particle_mass_msun_h": float(m_lr),
        "hr_min_num_p": int(args.hr_min_num_p),
        "match_rvir": float(args.match_rvir),
        "pot_scales_mpc": scales, "pot_ng": int(args.pot_ng),
        "pot_min_scale": min_scale, "n_local_minima_by_scale": n_minima,
        "n_hr_subhalos": int(sub.size),
        "r2_log_r_from_potential_all": holdout_r2(sub_phi, sub_log_r),
        "r2_log_r_over_rvir_from_potential_all": holdout_r2(sub_phi, sub_log_rrv),
        "r2_logM_from_lr_catalog_all": holdout_r2(
            np.log10(np.maximum(lr_mv[allm], 1.0)), np.log10(np.maximum(hmvir[allm], 1.0))),
        "r2_logM_from_potential_all": holdout_r2(
            phi_at[allm], np.log10(np.maximum(hmvir[allm], 1.0))),
        "bins": bins,
    }
    print(f"\n  {'bin':10s} {'nhost':>6s} {'compl':>6s} {'dr_LR':>7s} {'dr_pot':>7s} "
          f"{'sub_r':>7s} {'LR':>9s} {'POT':>9s}")
    for b in bins:
        if not b.get("n_hosts"):
            continue
        print(f"  {b['bin']:10s} {b['n_hosts']:6d} "
              f"{b['completeness_within_rvir']:6.2f} "
              f"{b['lr_centre_err_mpc_median']:7.3f} "
              f"{b['pot_centre_err_mpc_median']:7.3f} "
              f"{b['hr_median_sub_r_mpc']:7.3f} "
              f"{b.get('lr_verdict',''):>9s} {b.get('pot_verdict',''):>9s}")
    print(f"\n  {'bin':10s} {'n_sub':>7s} {'R2 log r':>9s} {'R2 log r/Rvir':>14s}"
          "   <- phi at the SUBHALO's own position")
    for b in bins:
        if not b.get("n_sub_in_bin"):
            continue
        print(f"  {b['bin']:10s} {b['n_sub_in_bin']:7d} "
              f"{b['r2_log_r_from_potential']:9.3f} "
              f"{b['r2_log_r_over_rvir_from_potential']:14.3f}")
    print(f"\n  scale recovery (holdout R^2 for log10 Mvir_HR): "
          f"LR catalog {summary['r2_logM_from_lr_catalog_all']:.3f}   "
          f"potential {summary['r2_logM_from_potential_all']:.3f}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--boxes", default="set9",
                    help="comma-separated boxes (default set9)")
    ap.add_argument("--reward-root", required=True,
                    help="dmsr_reward root holding halos/<box>__hr__hr")
    ap.add_argument("--lr-work-root", required=True,
                    help="where LR gadget2 dumps and Rockstar output live")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hr-min-num-p", type=int, default=200,
                    help="HR top-level floor; 200 matches the radial ladder")
    ap.add_argument("--match-rvir", type=float, default=1.0,
                    help="an LR host within this many HR R_vir counts as found")
    ap.add_argument("--pot-ng", type=int, default=64,
                    help="grid for the LR potential (LR is 64^3; no gain above)")
    ap.add_argument("--pot-scales", default="0,1.5625,3.125,6.25",
                    help="Gaussian smoothing scales in Mpc/h (0 = unsmoothed)")
    ap.add_argument("--pot-min-scale", type=float, default=1.5625,
                    help="scale whose local minima localise a centre; must be "
                         "one of --pot-scales. 6.25 leaves 3 minima in the box")
    ap.add_argument("--rerun-rockstar", action="store_true",
                    help="re-find the LR catalog even if one is on disk")
    args = ap.parse_args(argv)

    cfg = load_reward_config(args)
    out_dir = Path(args.out_dir)
    boxes = [b.strip() for b in args.boxes.split(",") if b.strip()]
    per_box = [run_box(cfg, b, args, out_dir) for b in boxes]
    write_json(out_dir / "summary.json",
               {"ok": True, "boxes": boxes, "per_box": per_box})
    print(f"\n=== wrote {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
