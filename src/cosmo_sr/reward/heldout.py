"""Catalog statistics that the reward deliberately does **not** optimize.

The reward sees only subhalo abundance and mean occupation. Everything here is
downstream evidence: if the distilled model improves these too, it learned
substructure; if it improves the reward while these get worse, it learned the
reward. Keeping the two sets in separate modules makes it hard to accidentally
promote a held-out statistic into the objective.

``two_halo_correlation`` and ``subhalo_relative_velocity`` are new; the rest wrap
:mod:`cosmo_sr.eval.halo_metrics` so the frozen definitions stay authoritative.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..eval.halo_metrics import (mass_function, nsub_vs_mhost, one_halo_correlation,
                                 subhalo_radial_profile)
from ..eval.rockstar import HaloCatalog

__all__ = [
    "bootstrap_ci",
    "held_out_metrics",
    "isolated_vs_sub_counts",
    "subhalo_relative_velocity",
    "two_halo_correlation",
]


def two_halo_correlation(
    cat: HaloCatalog,
    boxsize_mpc_h: float,
    *,
    r_min: float = 1.0,
    r_max: float = 20.0,
    n_bins: int = 10,
    mass_min: float = 0.0,
    max_hosts: int = 20000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Host-host ``xi(r)`` by the natural estimator ``DD/RR - 1``.

    RR is analytic (a periodic box has a uniform random expectation), which is
    exact here and avoids generating a random catalog. Hosts are subsampled to
    ``max_hosts`` because the pair count is O(N^2); the estimator is unbiased
    under subsampling and only the noise grows.
    """
    hosts = cat.hosts()
    sel = np.asarray(hosts.mvir >= float(mass_min)).nonzero()[0]
    pos = np.asarray(hosts.pos, dtype=np.float64)[sel]
    n = pos.shape[0]
    edges = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    if n < 2:
        return centers, np.full(n_bins, np.nan)
    if n > max_hosts:
        rng = np.random.default_rng(int(seed))
        pos = pos[rng.choice(n, size=int(max_hosts), replace=False)]
        n = pos.shape[0]

    box = float(boxsize_mpc_h)
    counts = np.zeros(n_bins, dtype=np.float64)
    block = 512
    for i0 in range(0, n, block):
        a = pos[i0:i0 + block]
        d = a[:, None, :] - pos[None, :, :]
        d -= box * np.round(d / box)
        r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
        # Each unordered pair appears twice over the full double loop; the
        # self-distance 0 falls below r_min and is dropped by the histogram.
        counts += np.histogram(r.ravel(), bins=edges)[0]
    dd = counts / 2.0

    vol_shell = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rr = 0.5 * n * (n - 1) * vol_shell / (box ** 3)
    xi = np.divide(dd, rr, out=np.full(n_bins, np.nan), where=rr > 0) - 1.0
    return centers, xi


def subhalo_relative_velocity(
    cat: HaloCatalog, *, bins: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """PDF of ``|v_sub - v_host|`` over host-subhalo pairs, plus its moments."""
    hosts = cat.hosts()
    subs = cat.subhalos()
    if bins is None:
        bins = np.linspace(0.0, 1500.0, 31)
    id_to_v = {int(i): v for i, v in zip(hosts.ids, hosts.vel)}
    dv = []
    for pid, sv in zip(subs.parent_ids, subs.vel):
        hv = id_to_v.get(int(pid))
        if hv is None:
            continue
        dv.append(float(np.linalg.norm(np.asarray(sv) - np.asarray(hv))))
    dv = np.asarray(dv, dtype=np.float64)
    hist, edges = np.histogram(dv, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    moments = {
        "n_pairs": float(dv.size),
        "mean": float(dv.mean()) if dv.size else float("nan"),
        "median": float(np.median(dv)) if dv.size else float("nan"),
        "rms": float(np.sqrt(np.mean(dv ** 2))) if dv.size else float("nan"),
    }
    return centers, hist, moments


def isolated_vs_sub_counts(
    cat: HaloCatalog, *, min_particles: int = 20
) -> Dict[str, float]:
    """How the population splits between field halos and substructure.

    The SR2 failure shows up here as too few subhalos *and* too many isolated
    halos of the same mass: unresolved substructure does not vanish, it is
    reported as a separate field halo.
    """
    ok = np.asarray(cat.num_p) >= int(min_particles)
    is_host = np.asarray(cat.parent_ids) < 0
    n_host = int(np.count_nonzero(ok & is_host))
    n_sub = int(np.count_nonzero(ok & ~is_host))
    total = n_host + n_sub
    return {
        "n_isolated": float(n_host),
        "n_subhalo": float(n_sub),
        "subhalo_fraction": float(n_sub / total) if total else float("nan"),
        "sub_per_host": float(n_sub / n_host) if n_host else float("nan"),
    }


def held_out_metrics(
    cat: HaloCatalog,
    *,
    boxsize_mpc_h: float = 100.0,
    host_bins: Optional[np.ndarray] = None,
    sub_bins: Optional[np.ndarray] = None,
    min_particles: int = 20,
    two_halo_mass_min: float = 1e12,
    seed: int = 0,
) -> Dict[str, object]:
    """Every unrewarded catalog statistic for one full-box catalog."""
    hosts = cat.hosts()
    subs = cat.subhalos()
    out: Dict[str, object] = {}

    mc, hmf = mass_function(hosts.mvir, boxsize_mpc_h, bins=host_bins)
    out["host_mass_function"] = {"centers": mc.tolist(), "dn_dlnm": hmf.tolist()}
    sc, shmf = mass_function(subs.mvir, boxsize_mpc_h, bins=sub_bins)
    out["subhalo_mass_function"] = {"centers": sc.tolist(), "dn_dlnm": shmf.tolist()}

    hc, occ, nh = nsub_vs_mhost(cat, host_bins=host_bins)
    out["occupation"] = {"centers": hc.tolist(), "mean_nsub": occ.tolist(),
                         "n_host": nh.tolist()}

    rc, prof = subhalo_radial_profile(cat, boxsize_mpc_h=boxsize_mpc_h)
    out["radial_profile"] = {"r_over_rvir": rc.tolist(), "pdf": prof.tolist()}

    oc, xi1 = one_halo_correlation(cat, boxsize_mpc_h)
    out["one_halo"] = {"r": oc.tolist(), "xi": xi1.tolist()}

    tc, xi2 = two_halo_correlation(cat, boxsize_mpc_h,
                                   mass_min=float(two_halo_mass_min), seed=seed)
    out["two_halo"] = {"r": tc.tolist(), "xi": xi2.tolist()}

    vc, vpdf, vmom = subhalo_relative_velocity(cat)
    out["relative_velocity"] = {"v": vc.tolist(), "pdf": vpdf.tolist(), **vmom}

    out["counts"] = isolated_vs_sub_counts(cat, min_particles=min_particles)
    return out


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile CI over **boxes**.

    The resampling unit must be the box. Chunks inside a box share the same
    large-scale modes and the same LR field, so bootstrapping them as if they
    were independent boxes would shrink every interval by roughly the square
    root of the number of chunks and manufacture significance.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    if v.size == 1:
        return {"mean": float(v[0]), "lo": float("nan"), "hi": float("nan"), "n": 1}
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(v, size=(int(n_boot), v.size), replace=True).mean(axis=1)
    return {
        "mean": float(v.mean()),
        "lo": float(np.quantile(draws, alpha / 2)),
        "hi": float(np.quantile(draws, 1 - alpha / 2)),
        "n": int(v.size),
    }
