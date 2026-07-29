"""HR↔SR host / subhalo matching and miss classification.

Host matching uses **periodic** distances, an absolute linking-length floor
(so small-Rvir hosts are not effectively unmatchable), mass-descending greedy
assignment, and a documented permissive fallback.

Subhalo classes (within matched hosts):

1. recovered
2. spatially_shifted
3. recovered_biased       — mass / Vmax bias
4. velocity_incoherent    — position OK, velocity mismatch
5. merged_into_host       — no SR sub; HR sub is central (Rockstar absorption)
6. missing                — no SR counterpart (peak absent / too diffuse / far)
7. diffuse_peak           — optional: density peak exists in SR but no Rockstar sub
8. absent_peak            — optional: no SR density peak near HR sub
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .rockstar import HaloCatalog

__all__ = [
    "MatchResult",
    "SubhaloClass",
    "classify_subhalos",
    "match_hosts",
    "phase_space_match",
    "self_match_after_translation",
]


SubhaloClass = str


@dataclass
class MatchResult:
    hr_ids: np.ndarray
    sr_ids: np.ndarray          # -1 if unmatched
    score: np.ndarray
    method: str


def _periodic_delta(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    d = a - b
    return d - box * np.round(d / box)


def _periodic_dist(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    return np.linalg.norm(_periodic_delta(a, b, box), axis=-1)


def match_hosts(
    hr: HaloCatalog,
    sr: HaloCatalog,
    boxsize_mpc_h: float = 100.0,
    *,
    link_mpc_h: float = 1.0,
    max_dx_rvir: float = 3.0,
    mass_ratio_max: float = 10.0,
    mhost_min: float = 0.0,
    knn: int = 64,
) -> MatchResult:
    """Greedy host matching with periodic distance and permissive candidates.

    Search radius for host ``i`` is ``max(link_mpc_h, max_dx_rvir * Rvir_i)``.
    HR hosts are matched in **descending mass** order so massive systems claim
    counterparts before small satellites can steal them.

    Uses a periodic ``cKDTree`` bulk k-NN query (``knn`` neighbours) so cost is
    ~O(N log N), not O(N²).
    """
    from scipy.spatial import cKDTree

    h = hr.hosts()
    s = sr.hosts()
    used = np.zeros(s.n, dtype=bool)
    sr_ids = np.full(h.n, -1, dtype=np.int64)
    scores = np.full(h.n, np.inf, dtype=np.float64)
    if h.n == 0 or s.n == 0:
        return MatchResult(hr_ids=h.ids.copy(), sr_ids=sr_ids, score=scores,
                           method="host_nn_periodic")

    box = float(boxsize_mpc_h)
    tree = cKDTree(s.pos, boxsize=box)
    k = int(min(max(knn, 8), s.n))
    # Bulk k-NN; scipy returns inf when fewer than k within half-box.
    dists_all, idxs_all = tree.query(h.pos, k=k)
    if k == 1:
        dists_all = dists_all[:, None]
        idxs_all = idxs_all[:, None]

    order = np.argsort(-h.mvir)  # massive first
    for i in order:
        if h.mvir[i] < mhost_min:
            continue
        rvir_mpc = max(float(h.rvir[i]) * 1e-3, 1e-6)
        r_search = max(float(link_mpc_h), float(max_dx_rvir) * rvir_mpc)
        r_fb = 2.0 * float(link_mpc_h)

        idxs = np.asarray(idxs_all[i], dtype=np.int64)
        dists = np.asarray(dists_all[i], dtype=np.float64)
        finite = np.isfinite(dists) & (idxs >= 0) & (idxs < s.n)
        idxs, dists = idxs[finite], dists[finite]
        if idxs.size == 0:
            continue

        # Prefer mass-consistent candidates inside r_search; else permissive.
        ratio = np.maximum(
            s.mvir[idxs] / max(float(h.mvir[i]), 1e-30),
            float(h.mvir[i]) / np.maximum(s.mvir[idxs], 1e-30),
        )
        free = ~used[idxs]
        primary = free & (dists <= r_search) & (ratio <= mass_ratio_max)
        if not np.any(primary):
            primary = free & (dists <= r_fb)
        if not np.any(primary):
            primary = free & (dists <= r_search)  # ignore mass
        if not np.any(primary):
            continue
        idxs, dists, ratio = idxs[primary], dists[primary], ratio[primary]
        sc = dists / rvir_mpc + 0.1 * np.abs(np.log(np.maximum(ratio, 1e-30)))
        jloc = int(np.argmin(sc))
        j = int(idxs[jloc])
        used[j] = True
        sr_ids[i] = s.ids[j]
        scores[i] = float(sc[jloc])
    return MatchResult(
        hr_ids=h.ids.copy(), sr_ids=sr_ids, score=scores,
        method="host_nn_periodic",
    )


def self_match_after_translation(
    cat: HaloCatalog,
    boxsize_mpc_h: float = 100.0,
    shift_mpc_h: Tuple[float, float, float] = (0.05, -0.03, 0.04),
    **match_kw,
) -> Dict[str, float]:
    """Validate host matching: catalog vs itself after a known periodic shift.

    Use a **small** shift (≪ mean host spacing) so the true counterpart is the
    unique nearest neighbour. Expects near-100% host recovery.
    """
    hosts = cat.hosts()
    shift = np.asarray(shift_mpc_h, dtype=np.float64).reshape(1, 3)
    shifted = HaloCatalog(
        ids=hosts.ids.copy(),
        parent_ids=hosts.parent_ids.copy(),
        mvir=hosts.mvir.copy(),
        rvir=hosts.rvir.copy(),
        vmax=hosts.vmax.copy(),
        pos=(hosts.pos + shift) % boxsize_mpc_h,
        vel=hosts.vel.copy(),
        num_p=hosts.num_p.copy(),
        path=hosts.path,
    )
    # Treat shifted catalog as "SR": wrap as full catalog of hosts only.
    fake_sr = shifted
    # Build a minimal "HR" catalog that only has hosts (no subs needed).
    fake_hr = hosts
    # match_hosts calls .hosts() which filters parent_ids < 0 — already hosts.
    m = match_hosts(fake_hr, fake_sr, boxsize_mpc_h, **match_kw)
    matched = m.sr_ids >= 0
    # Correct if matched SR id equals the same host id (shift preserves ids).
    correct = matched & (m.sr_ids == m.hr_ids)
    n = max(int(hosts.n), 1)
    return {
        "n_hosts": float(hosts.n),
        "match_rate": float(matched.mean()) if hosts.n else 0.0,
        "correct_id_rate": float(correct.mean()) if hosts.n else 0.0,
        "n_matched": float(matched.sum()),
        "n_correct": float(correct.sum()),
        "shift_mpc_h": list(map(float, shift_mpc_h)),
    }


def phase_space_match(
    hr_pos: np.ndarray,
    hr_vel: np.ndarray,
    hr_mass: np.ndarray,
    sr_pos: np.ndarray,
    sr_vel: np.ndarray,
    sr_mass: np.ndarray,
    boxsize_mpc_h: float,
    *,
    pos_scale_mpc: float = 0.1,
    vel_scale_kms: float = 100.0,
    mass_weight: float = 0.2,
    max_pos_mpc: Optional[float] = None,
) -> MatchResult:
    """One-to-one greedy match of subhalo lists inside one host."""
    n_h, n_s = len(hr_pos), len(sr_pos)
    hr_ids = np.arange(n_h, dtype=np.int64)
    sr_ids = np.full(n_h, -1, dtype=np.int64)
    scores = np.full(n_h, np.inf)
    used = np.zeros(n_s, dtype=bool)
    max_pos = float(max_pos_mpc) if max_pos_mpc is not None else 5.0 * pos_scale_mpc
    for i in range(n_h):
        if n_s == 0:
            break
        dpos = _periodic_dist(sr_pos, hr_pos[i], boxsize_mpc_h)
        dvel = np.linalg.norm(sr_vel - hr_vel[i], axis=1)
        dm = np.abs(np.log(np.maximum(sr_mass, 1e-30) / max(hr_mass[i], 1e-30)))
        sc = dpos / pos_scale_mpc + dvel / vel_scale_kms + mass_weight * dm
        sc = np.where(used, np.inf, sc)
        j = int(np.argmin(sc))
        if not np.isfinite(sc[j]) or dpos[j] > max_pos:
            continue
        used[j] = True
        sr_ids[i] = j
        scores[i] = sc[j]
    return MatchResult(hr_ids=hr_ids, sr_ids=sr_ids, score=scores, method="phase_space")


def classify_subhalos(
    hr: HaloCatalog,
    sr: HaloCatalog,
    host_match: MatchResult,
    boxsize_mpc_h: float = 100.0,
    *,
    shift_rvir: float = 0.3,
    mass_bias: float = 0.5,
    vmax_bias: float = 0.5,
    vel_bias: float = 0.75,
    only_matched_hosts: bool = True,
) -> List[Dict]:
    """Classify HR subhalos; by default only inside successfully matched hosts."""
    hr_hosts = hr.hosts()
    hr_subs = hr.subhalos()
    sr_subs = sr.subhalos()
    hr_by_parent: Dict[int, List[int]] = {}
    for k, pid in enumerate(hr_subs.parent_ids):
        hr_by_parent.setdefault(int(pid), []).append(k)
    sr_by_parent: Dict[int, List[int]] = {}
    for k, pid in enumerate(sr_subs.parent_ids):
        sr_by_parent.setdefault(int(pid), []).append(k)

    out: List[Dict] = []
    for hi, (hr_hid, sr_host_id) in enumerate(zip(host_match.hr_ids, host_match.sr_ids)):
        sr_host_id = int(sr_host_id)
        if only_matched_hosts and sr_host_id < 0:
            continue
        hr_kids = hr_by_parent.get(int(hr_hid), [])
        if not hr_kids:
            continue
        hpos = hr_hosts.pos[hi]
        rvir_mpc = max(hr_hosts.rvir[hi] * 1e-3, 1e-6)
        kids = sr_by_parent.get(sr_host_id, []) if sr_host_id >= 0 else []
        hidx = np.asarray(hr_kids, dtype=np.int64)

        if kids and sr_host_id >= 0:
            kidx = np.asarray(kids, dtype=np.int64)
            match = phase_space_match(
                hr_subs.pos[hidx], hr_subs.vel[hidx], hr_subs.mvir[hidx],
                sr_subs.pos[kidx], sr_subs.vel[kidx], sr_subs.mvir[kidx],
                boxsize_mpc_h,
                pos_scale_mpc=max(0.05, 0.25 * rvir_mpc),
                max_pos_mpc=max(0.2, 1.0 * rvir_mpc),
            )
        else:
            match = MatchResult(
                hr_ids=np.arange(len(hidx), dtype=np.int64),
                sr_ids=np.full(len(hidx), -1, dtype=np.int64),
                score=np.full(len(hidx), np.inf),
                method="phase_space",
            )

        for local_i, i in enumerate(hidx):
            r_rvir = float(np.linalg.norm(
                _periodic_delta(hr_subs.pos[i], hpos, boxsize_mpc_h)
            ) / rvir_mpc)
            rec = {
                "hr_id": int(hr_subs.ids[i]),
                "host_hr_id": int(hr_hid),
                "host_sr_id": sr_host_id,
                "class": "missing",
                "sr_id": -1,
                "dx_rvir": float("nan"),
                "mass_ratio": float("nan"),
                "vmax_ratio": float("nan"),
                "vel_ratio": float("nan"),
                "r_rvir": r_rvir,
            }
            if sr_host_id < 0 or match.sr_ids[local_i] < 0:
                # Central and unmatched → likely absorbed into host by finder.
                rec["class"] = "merged_into_host" if r_rvir < 0.25 else "missing"
                out.append(rec)
                continue

            j = int(kidx[int(match.sr_ids[local_i])])
            rec["sr_id"] = int(sr_subs.ids[j])
            dx = float(_periodic_dist(
                sr_subs.pos[j:j + 1], hr_subs.pos[i], boxsize_mpc_h
            )[0] / rvir_mpc)
            mr = float(sr_subs.mvir[j] / max(hr_subs.mvir[i], 1e-30))
            vr = float(sr_subs.vmax[j] / max(hr_subs.vmax[i], 1e-30))
            dvel = float(np.linalg.norm(sr_subs.vel[j] - hr_subs.vel[i]))
            v_scale = max(float(hr_subs.vmax[i]), 50.0)
            vel_ratio = dvel / v_scale
            rec.update(dx_rvir=dx, mass_ratio=mr, vmax_ratio=vr, vel_ratio=vel_ratio)

            mass_ok = abs(np.log(max(mr, 1e-30))) < mass_bias
            vmax_ok = abs(np.log(max(vr, 1e-30))) < vmax_bias
            pos_ok = dx <= shift_rvir
            vel_ok = vel_ratio < vel_bias

            if pos_ok and mass_ok and vmax_ok and vel_ok:
                rec["class"] = "recovered"
            elif pos_ok and not vel_ok and mass_ok:
                rec["class"] = "velocity_incoherent"
            elif (not pos_ok) and mass_ok:
                rec["class"] = "spatially_shifted"
            elif pos_ok and (not mass_ok or not vmax_ok):
                rec["class"] = "recovered_biased"
            else:
                rec["class"] = "spatially_shifted"
            out.append(rec)
    return out
