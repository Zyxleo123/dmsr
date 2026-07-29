"""Density-peak probes for missing SR subhalos.

For HR subhalos classified as ``missing`` / ``merged_into_host``, inspect the
SR (and optionally HR) CIC density field near the HR sub position:

* ``absent_peak``  — no local SR density maximum above background
* ``diffuse_peak`` — a peak exists but is too weak / broad for Rockstar
* (catalog classes handle displaced / velocity / Rockstar absorption)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["local_density_stats", "refine_missing_with_density"]


def _wrap_index(i: int, n: int) -> int:
    return int(i) % int(n)


def local_density_stats(
    dens: np.ndarray,
    pos_mpc_h: np.ndarray,
    boxsize_mpc_h: float,
    *,
    half_window: int = 3,
    peak_contrast: float = 1.5,
) -> Dict[str, float]:
    """Return local δ and whether a discrete peak sits in a small window."""
    dens = np.asarray(dens)
    ng = dens.shape[0]
    cell = float(boxsize_mpc_h) / ng
    ix = [int(np.floor(float(p) / cell)) % ng for p in pos_mpc_h]
    # Neighbourhood mean / max
    vals = []
    peak = True
    cval = float(dens[ix[0], ix[1], ix[2]])
    for dx in range(-half_window, half_window + 1):
        for dy in range(-half_window, half_window + 1):
            for dz in range(-half_window, half_window + 1):
                v = float(dens[
                    _wrap_index(ix[0] + dx, ng),
                    _wrap_index(ix[1] + dy, ng),
                    _wrap_index(ix[2] + dz, ng),
                ])
                vals.append(v)
                if (dx, dy, dz) != (0, 0, 0) and v > cval:
                    peak = False
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(arr.mean())
    mx = float(arr.max())
    # δ relative to global mean if dens is 1+δ; else relative to window.
    gmean = float(np.mean(dens))
    contrast = (cval + 1e-6) / (gmean + 1e-6)
    return {
        "delta_center": cval,
        "delta_window_mean": mean,
        "delta_window_max": mx,
        "contrast_global": contrast,
        "is_local_max": float(peak),
        "has_peak": float(peak and contrast >= peak_contrast),
    }


def refine_missing_with_density(
    records: Sequence[Dict],
    dens_sr: np.ndarray,
    dens_hr: Optional[np.ndarray],
    boxsize_mpc_h: float,
    hr_pos_by_id: Dict[int, np.ndarray],
    *,
    half_window: int = 3,
    peak_contrast: float = 1.5,
    diffuse_contrast: float = 1.2,
) -> List[Dict]:
    """Upgrade ``missing`` classes using SR density morphology."""
    out: List[Dict] = []
    for rec in records:
        r = dict(rec)
        if r.get("class") not in ("missing",):
            out.append(r)
            continue
        pos = hr_pos_by_id.get(int(r["hr_id"]))
        if pos is None:
            out.append(r)
            continue
        sr = local_density_stats(
            dens_sr, pos, boxsize_mpc_h,
            half_window=half_window, peak_contrast=peak_contrast,
        )
        r["sr_density"] = sr
        if dens_hr is not None:
            r["hr_density"] = local_density_stats(
                dens_hr, pos, boxsize_mpc_h,
                half_window=half_window, peak_contrast=peak_contrast,
            )
        if sr["has_peak"] >= 0.5:
            # Peak present in SR density but Rockstar found no sub → diffuse /
            # unbound / absorbed into host finder.
            r["class"] = "diffuse_peak"
        elif sr["contrast_global"] >= diffuse_contrast:
            r["class"] = "diffuse_peak"
        else:
            r["class"] = "absent_peak"
        out.append(r)
    return out
