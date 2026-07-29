"""Halo / subhalo population metrics for the SR2 subhalo study.

All estimators take :class:`~cosmo_sr.eval.rockstar.HaloCatalog` objects produced
by the *same* frozen Rockstar config for HR and SR. Stratification helpers take
redshift / host-mass / subhalo-mass / radius bins as explicit arrays so Stage-1
tables stay comparable across seeds and boxes.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .rockstar import HaloCatalog

__all__ = [
    "mass_function",
    "vmax_function",
    "nsub_vs_mhost",
    "subhalo_radial_profile",
    "one_halo_correlation",
    "host_velocity_dispersion",
]


def mass_function(
    masses: np.ndarray,
    boxsize_mpc_h: float,
    bins: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``dn/dlnM`` in ``(h/Mpc)^3``."""
    m = np.asarray(masses, dtype=np.float64)
    m = m[np.isfinite(m) & (m > 0)]
    if bins is None:
        if m.size == 0:
            bins = np.logspace(8, 15, 15)
        else:
            bins = np.logspace(np.log10(m.min()) - 0.1, np.log10(m.max()) + 0.1, 15)
    hist, edges = np.histogram(m, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    dlnm = np.log(edges[1:]) - np.log(edges[:-1])
    vol = float(boxsize_mpc_h) ** 3
    return centers, hist / (vol * np.maximum(dlnm, 1e-30))


def vmax_function(
    vmax: np.ndarray,
    boxsize_mpc_h: float,
    bins: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vmax, dtype=np.float64)
    v = v[np.isfinite(v) & (v > 0)]
    if bins is None:
        bins = np.logspace(1, 3, 20) if v.size else np.logspace(1, 3, 20)
    hist, edges = np.histogram(v, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    dln = np.log(edges[1:]) - np.log(edges[:-1])
    vol = float(boxsize_mpc_h) ** 3
    return centers, hist / (vol * np.maximum(dln, 1e-30))


def nsub_vs_mhost(
    cat: HaloCatalog,
    host_bins: Optional[np.ndarray] = None,
    msub_min: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean subhalo count per host-mass bin (and host counts)."""
    hosts = cat.hosts()
    subs = cat.subhalos()
    if host_bins is None:
        host_bins = np.logspace(11, 15, 9)
    # Map parent id -> host mass
    id_to_m = {int(i): float(m) for i, m in zip(hosts.ids, hosts.mvir)}
    counts = np.zeros(len(host_bins) - 1, dtype=np.float64)
    nhost = np.zeros(len(host_bins) - 1, dtype=np.float64)
    host_which = np.digitize(hosts.mvir, host_bins) - 1
    for w in host_which:
        if 0 <= w < len(nhost):
            nhost[w] += 1
    for pid, msub in zip(subs.parent_ids, subs.mvir):
        if msub < msub_min:
            continue
        mhost = id_to_m.get(int(pid))
        if mhost is None:
            continue
        w = int(np.digitize([mhost], host_bins)[0] - 1)
        if 0 <= w < len(counts):
            counts[w] += 1
    mean = np.divide(counts, nhost, out=np.zeros_like(counts), where=nhost > 0)
    centers = np.sqrt(host_bins[:-1] * host_bins[1:])
    return centers, mean, nhost


def subhalo_radial_profile(
    cat: HaloCatalog,
    r_bins: Optional[np.ndarray] = None,
    boxsize_mpc_h: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """``dN/d(r/Rvir)`` stacked over all host–subhalo pairs."""
    if r_bins is None:
        r_bins = np.linspace(0.0, 1.0, 11)
    hosts = cat.hosts()
    subs = cat.subhalos()
    id_to_pos = {int(i): p for i, p in zip(hosts.ids, hosts.pos)}
    id_to_r = {int(i): float(r) for i, r in zip(hosts.ids, hosts.rvir)}
    # Rockstar rvir is kpc/h; positions are Mpc/h.
    rr = []
    for pid, spos in zip(subs.parent_ids, subs.pos):
        hpos = id_to_pos.get(int(pid))
        rvir_kpc = id_to_r.get(int(pid))
        if hpos is None or rvir_kpc is None or rvir_kpc <= 0:
            continue
        d = spos - hpos
        d -= boxsize_mpc_h * np.round(d / boxsize_mpc_h)
        dist = float(np.linalg.norm(d))
        rr.append(dist / (rvir_kpc * 1e-3))
    rr = np.asarray(rr, dtype=np.float64)
    hist, edges = np.histogram(rr, bins=r_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dens = hist / np.maximum(edges[1:] - edges[:-1], 1e-30)
    if dens.sum() > 0:
        dens = dens / dens.sum()
    return centers, dens


def one_halo_correlation(
    cat: HaloCatalog,
    boxsize_mpc_h: float,
    rmax_mpc_h: float = 0.5,
    n_bins: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pair counts of (host, subhalo) separations below ``rmax`` as a proxy ξ_1h.

    This is *not* a full Landy–Szalay estimator; it is the stacked host–subhalo
    pair histogram normalised by the RR expectation for a uniform sphere, used
    as a relative SR-vs-HR diagnostic.
    """
    hosts = cat.hosts()
    subs = cat.subhalos()
    id_to_pos = {int(i): p for i, p in zip(hosts.ids, hosts.pos)}
    edges = np.logspace(np.log10(0.02), np.log10(rmax_mpc_h), n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.float64)
    n_pairs = 0
    for pid, spos in zip(subs.parent_ids, subs.pos):
        hpos = id_to_pos.get(int(pid))
        if hpos is None:
            continue
        d = spos - hpos
        d -= boxsize_mpc_h * np.round(d / boxsize_mpc_h)
        dist = float(np.linalg.norm(d))
        if dist <= 0 or dist > rmax_mpc_h:
            continue
        w = int(np.digitize([dist], edges)[0] - 1)
        if 0 <= w < n_bins:
            counts[w] += 1
            n_pairs += 1
    centers = np.sqrt(edges[:-1] * edges[1:])
    # RR ~ r^2 dr for isotropic pairs
    rr = centers ** 2 * (edges[1:] - edges[:-1])
    xi = np.divide(counts, rr, out=np.zeros_like(counts), where=rr > 0)
    if xi.sum() > 0:
        xi = xi / xi.sum() * (n_pairs if n_pairs else 1.0)
    return centers, xi


def host_velocity_dispersion(cat: HaloCatalog) -> Dict[str, float]:
    """Host Vrms proxy: Rockstar ``vrms`` is not stored; use host vel std across catalog."""
    hosts = cat.hosts()
    if hosts.n == 0:
        return {"host_speed_mean": float("nan"), "host_speed_rms": float("nan"),
                "sub_speed_mean": float("nan"), "sub_speed_rms": float("nan")}
    hs = np.linalg.norm(hosts.vel, axis=1)
    ss = np.linalg.norm(cat.subhalos().vel, axis=1) if cat.subhalos().n else np.array([])
    return {
        "host_speed_mean": float(hs.mean()) if hs.size else float("nan"),
        "host_speed_rms": float(np.sqrt(np.mean(hs ** 2))) if hs.size else float("nan"),
        "sub_speed_mean": float(ss.mean()) if ss.size else float("nan"),
        "sub_speed_rms": float(np.sqrt(np.mean(ss ** 2))) if ss.size else float("nan"),
    }
