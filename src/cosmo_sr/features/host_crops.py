"""Host-frame Lagrangian crops -- the geometry of Option A.

``docs/sr2_substructure_module.md`` section 3 proposes cropping a cube around a
host's **Lagrangian** centroid, side ``2 R_L(host)``, and resampling it to a
fixed grid. Everything here is the index arithmetic of that crop, kept apart
from the I/O so it can be pinned by ``tests/features/test_host_crops.py``.

Two conventions carried in from :mod:`cosmo_sr.features.lagrangian_host`:

* a particle id **is** the flat C-order index of its Lagrangian lattice site,
  because ``field_to_particles`` writes ids as ``arange(ng**3)``;
* the lattice is periodic, so every centre is a circular mean and every offset
  is taken modulo ``ng``. A host straddling the box face is not a special case.

The crop is defined on the **native** lattice and never interpolated here. The
resample to Option A's fixed ``96^3`` is reported as a ratio
(:func:`resample_report`) rather than performed, so that what is measured is the
content of the crop and not an interpolation of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "CropFrame",
    "auc",
    "block_reduce",
    "crop_frame",
    "flat_to_sites",
    "lagrangian_radius_sites",
    "periodic_site_centre",
    "resample_report",
    "roc_curve",
    "sites_to_flat",
    "to_crop_coords",
]


# ---------------------------------------------------------------------------
# Lattice <-> id
# ---------------------------------------------------------------------------

def flat_to_sites(pid: np.ndarray, ng: int) -> np.ndarray:
    """``(N,)`` flat C-order ids -> ``(N, 3)`` lattice sites."""
    p = np.asarray(pid, dtype=np.int64).reshape(-1)
    out = np.empty((p.size, 3), dtype=np.int64)
    out[:, 0] = p // (ng * ng)
    out[:, 1] = (p // ng) % ng
    out[:, 2] = p % ng
    return out


def sites_to_flat(sites: np.ndarray, ng: int) -> np.ndarray:
    """Inverse of :func:`flat_to_sites`; sites are taken modulo ``ng``."""
    s = np.asarray(sites, dtype=np.int64).reshape(-1, 3) % ng
    return (s[:, 0] * ng + s[:, 1]) * ng + s[:, 2]


def periodic_site_centre(sites: np.ndarray, ng: int) -> np.ndarray:
    """Circular mean of lattice sites, per axis, in ``[0, ng)``.

    The arithmetic mean is wrong at the box face: sites at 1 and ``ng - 1``
    average to the middle of the box. Mirrors
    :func:`cosmo_sr.features.lagrangian_host.periodic_circular_mean`, but in
    site units rather than Mpc/h.
    """
    s = np.asarray(sites, dtype=np.float64).reshape(-1, 3)
    if s.size == 0:
        return np.zeros(3)
    ang = 2.0 * np.pi * s / float(ng)
    mean = np.arctan2(np.sin(ang).mean(axis=0), np.cos(ang).mean(axis=0))
    return (mean % (2.0 * np.pi)) * float(ng) / (2.0 * np.pi)


def lagrangian_radius_sites(n_particles: float) -> float:
    """``R_L`` in lattice sites: the volume-equivalent sphere of ``n`` sites.

    One site carries exactly one particle, so the mean-density volume of ``n``
    particles is ``n`` cells and ``R_L = (3 n / 4 pi)^(1/3)``. Defined for a
    one-particle object and monotone in count, unlike anything shape-based.
    """
    n = max(float(n_particles), 0.0)
    return float((3.0 * n / (4.0 * np.pi)) ** (1.0 / 3.0))


# ---------------------------------------------------------------------------
# The crop
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CropFrame:
    """A periodic cube of the Lagrangian lattice, in native sites."""

    centre: np.ndarray     # (3,) float, the host's circular-mean site
    start: np.ndarray      # (3,) int, lower corner before the modulo
    side: int              # edge in native sites
    ng: int

    @property
    def n_sites(self) -> int:
        return int(self.side) ** 3

    def axes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-axis wrapped index arrays; ``np.ix_(*axes())`` gathers the cube."""
        r = np.arange(int(self.side), dtype=np.int64)
        return tuple((int(self.start[a]) + r) % int(self.ng) for a in range(3))

    def flat_ids(self) -> np.ndarray:
        """``(side^3,)`` particle ids of the crop, in C order over the cube."""
        ax, ay, az = self.axes()
        ng = int(self.ng)
        return ((ax[:, None, None] * ng + ay[None, :, None]) * ng
                + az[None, None, :]).reshape(-1)


def crop_frame(sites: np.ndarray, ng: int, *, scale: float = 1.0,
               min_side: int = 16, max_side: int = 256,
               n_particles: float | None = None) -> CropFrame:
    """Option A's cube: centred on the host's Lagrangian centroid, side ``2 R_L``.

    ``scale`` multiplies the half-side, so ``scale=1`` is the note's
    ``side ~ 2 R_L``. ``side`` is forced even, which keeps the centre on a cell
    corner and makes the block reduction of :func:`block_reduce` exact whenever
    the output side divides it.
    """
    s = np.asarray(sites, dtype=np.int64).reshape(-1, 3)
    centre = periodic_site_centre(s, ng)
    n = float(s.shape[0]) if n_particles is None else float(n_particles)
    side = int(2.0 * np.ceil(scale * lagrangian_radius_sites(n)))
    side = int(np.clip(side, min_side, min(max_side, ng)))
    side += side % 2
    start = np.floor(centre).astype(np.int64) - side // 2
    return CropFrame(centre=centre, start=start, side=side, ng=int(ng))


def to_crop_coords(sites: np.ndarray, frame: CropFrame) -> np.ndarray:
    """Lattice sites (or float positions) -> crop coordinates in ``[0, side)``.

    Values outside the crop come back outside ``[0, side)`` -- specifically in
    ``[side, ng)`` -- so ``(u >= 0) & (u < side)`` is the membership test and
    nothing needs a separate mask.
    """
    s = np.asarray(sites, dtype=np.float64).reshape(-1, 3)
    return (s - frame.start.astype(np.float64)[None, :]) % float(frame.ng)


def resample_report(frame: CropFrame, target: int = 96) -> dict:
    """What Option A's fixed-grid resample would do to this crop.

    ``ratio > 1`` is interpolation inventing sites (the "resampling waste at the
    bottom" of section 3); ``ratio < 1`` is a cluster being decimated.
    """
    return {
        "native_side": int(frame.side),
        "target_side": int(target),
        "ratio": float(target) / float(frame.side),
        "native_sites": frame.n_sites,
        "target_sites": int(target) ** 3,
    }


# ---------------------------------------------------------------------------
# Reduction and ranking
# ---------------------------------------------------------------------------

def block_reduce(vol: np.ndarray, max_side: int, how: str = "max"):
    """Cube -> at most ``max_side^3`` by whole blocks. Returns ``(out, extent)``.

    ``how='max'`` is the default because the page is looking for *clumps*: a
    block mean of a log density dilutes a 4-site subhalo into its smooth
    surroundings and the thing being visualised disappears. ``how='mean'`` is
    right for a fraction-valued channel like a membership mask.

    The output side is **derived from the factor**, not fixed: choosing
    ``f = ceil(n / max_side)`` and then ``out = ceil(n / f)`` leaves at most
    ``f - 1`` sites of padding, where fixing ``out = max_side`` and padding up
    to ``f * max_side`` can leave a quarter of the cube fabricated (n=148,
    max_side=48 pads 44 sites). Padding is ``nan`` and ignored by the
    reduction, so a trailing block reports its real sites rather than a
    duplicate of the last row -- edge replication is what smears a bright
    boundary row into a band and is never a measurement.

    ``extent`` is the number of native sites the returned cube spans
    (``out * f``). Every consumer that maps a native coordinate onto this cube
    must scale by ``extent``, not by ``n``; that mismatch is exactly what puts
    an overlay in the wrong place.
    """
    v = np.asarray(vol, dtype=np.float32)
    n = v.shape[0]
    if v.ndim != 3 or v.shape != (n, n, n):
        raise ValueError(f"expected a cube, got {v.shape}")
    if max_side >= n:
        return v.astype(np.float32, copy=True), n
    f = int(np.ceil(n / max_side))
    out_side = int(np.ceil(n / f))
    extent = out_side * f
    pad = extent - n
    if pad:
        v = np.pad(v, ((0, pad), (0, pad), (0, pad)),
                   mode="constant", constant_values=np.nan)
    v = v.reshape(out_side, f, out_side, f, out_side, f)
    red = np.nanmax if how == "max" else np.nanmean
    with np.errstate(invalid="ignore"):
        out = red(v, axis=(1, 3, 5))
    return np.asarray(out, dtype=np.float32), extent


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score of a positive > score of a negative).

    Rank-based, so it costs one sort and is invariant to any monotone rescaling
    of ``scores`` -- which is the point. The question this answers is whether
    the *ordering* a scalar induces on Lagrangian sites agrees with where HR put
    its subhalos, not whether its units are calibrated.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).reshape(-1) > 0
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    # Average ranks within ties, or a constant score would not score 0.5.
    su = np.sort(s)
    start = 0
    for end in np.flatnonzero(np.diff(su) != 0).tolist() + [s.size - 1]:
        if end > start:
            ranks[order[start:end + 1]] = 0.5 * (start + end) + 1.0
        start = end + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve(scores: np.ndarray, labels: np.ndarray, n_points: int = 64) -> dict:
    """``n_points`` samples of the ROC, for drawing rather than for scoring."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).reshape(-1) > 0
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return {"fpr": [], "tpr": []}
    order = np.argsort(-s, kind="stable")
    tp = np.cumsum(y[order]).astype(np.float64)
    fp = np.cumsum(~y[order]).astype(np.float64)
    idx = np.unique(np.linspace(0, s.size - 1, n_points).astype(np.int64))
    return {
        "fpr": (fp[idx] / n_neg).round(5).tolist(),
        "tpr": (tp[idx] / n_pos).round(5).tolist(),
    }
