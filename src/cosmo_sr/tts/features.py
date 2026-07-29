"""Stage 2: candidate features computable **without the test HR box**.

Everything in this module is available at inference time on a box we have never
seen the truth for. They fall into four groups:

``operator``
    ``|A(x_SR) - y|`` per channel group. Cheap and always available. Note ``A``
    (block average) is a *misspecified* model of this dataset's LR simulations,
    so this is a consistency signal, not a quality guarantee.
``artefact``
    Tile-seam discontinuity of the stitched field, and -- when the candidate was
    produced with coordinate-indexed global noise -- the disagreement between
    overlapping tiles.
``plausibility``
    How far the candidate's density power spectrum and ``log(1 + delta)`` PDF sit
    from the *training* HR reference, plus joint density-velocity statistics
    (velocity dispersion inside overdense regions, divergence-density
    correlation) that no single-field summary sees.
``equivariance``
    ``R^-1 G(R x, R z)`` vs ``G(x, z)`` for cubic rotations and flips, with the
    **same noise realisation transformed along with the input**. Comparing
    against an independently drawn noise would only measure the model's
    stochasticity, which says nothing about this candidate.

    Translation is deliberately *not* in the feature set: a fully convolutional
    generator fed input and noise shifted together is exactly translation
    equivariant, so the residual is float noise and carries no signal. It is
    still checked -- as a correctness test of the global-noise indexing, in
    :mod:`cosmo_sr.tts.tiling` and its tests.
``noise``
    Per-site mean and std of the realisation itself. Uninformative for i.i.d.
    draws (they are N(0, 1) by construction) but the tripwire that catches
    Stage 4 refinement wandering out of distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data.crops import periodic_crop
from .metrics import (
    DensityGeometry,
    boundary_discontinuity,
    cic_density_slabs,
    density_profile,
    lr_reconstruction,
)
from .sampling import tile_noise, tile_starts
from .srs_noise import NOISE_SITES

__all__ = [
    "FEATURE_KEYS",
    "HRReference",
    "candidate_features",
    "equivariance_features",
    "joint_density_velocity",
    "rotate_field",
]


# --------------------------------------------------------------------------- #
# Training-set reference statistics
# --------------------------------------------------------------------------- #
@dataclass
class HRReference:
    """Mean density power spectrum and log-density PDF of the *training* HR boxes.

    Fitted on training boxes only. A candidate's distance from these curves is a
    plausibility feature: it asks "does this look like a real universe at this
    resolution", which needs no paired truth for the box being scored.
    """

    pk: np.ndarray
    pdf: np.ndarray
    n_boxes: int = 0

    @classmethod
    def fit(cls, densities: Sequence[torch.Tensor], n_bins: int = 24) -> "HRReference":
        pks, pdfs = [], []
        for rho in densities:
            prof = density_profile(rho, n_bins=n_bins)
            pks.append(prof["density_pk"])
            pdfs.append(prof["log_density_pdf"])
        if not pks:
            raise ValueError("HRReference.fit needs at least one density field")
        return cls(pk=np.mean(pks, axis=0), pdf=np.mean(pdfs, axis=0), n_boxes=len(pks))

    def save(self, path) -> None:
        np.savez(path, pk=self.pk, pdf=self.pdf, n_boxes=np.array(self.n_boxes))

    @classmethod
    def load(cls, path) -> "HRReference":
        d = np.load(Path(path))
        return cls(pk=d["pk"], pdf=d["pdf"], n_boxes=int(d["n_boxes"]))

    def distance(self, rho: torch.Tensor, n_bins: int = 24) -> Dict[str, float]:
        prof = density_profile(rho, n_bins=n_bins)
        pk = np.asarray(prof["density_pk"], dtype=np.float64)
        ref = np.asarray(self.pk, dtype=np.float64)
        n = min(len(pk), len(ref))
        logratio = np.log(np.maximum(pk[:n], 1e-30) / np.maximum(ref[:n], 1e-30))
        return {
            "plaus_pk_logdist": float(np.abs(logratio).mean()),
            "plaus_pk_logdist_high": float(np.abs(logratio[n // 2:]).mean()),
            "plaus_pdf_l1": float(np.abs(prof["log_density_pdf"] - self.pdf).sum()),
        }


# --------------------------------------------------------------------------- #
# Joint density-velocity statistics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def joint_density_velocity(
    sr: torch.Tensor, rho: torch.Tensor, quantile: float = 0.9
) -> Dict[str, float]:
    """Statistics that couple the density and velocity fields.

    A candidate can have a perfect density PDF and a perfect velocity power
    spectrum while pairing them wrongly -- fast material in voids, quiet material
    in haloes. These summaries are sensitive to that, and they are exactly the
    kind of structure the SR2 paper checks by eye in halo profiles.

    * ``jdv_speed_in_dense``: rms speed in the densest cells, over the global rms.
    * ``jdv_div_rho_corr``: Pearson correlation of ``div v`` with ``delta``
      (negative in gravitational collapse -- infall onto overdensities).
    * ``jdv_speed_rho_corr``: correlation of speed with ``log(1 + delta)``.
    """
    from ..dmsr.evaluate import divergence_field

    v = sr[:, 3:6].float()
    speed = v.pow(2).sum(dim=1, keepdim=True).sqrt()
    d = rho.float()
    logd = torch.log10((d + 1.0).clamp_min(1e-6))
    div = divergence_field(v)

    thresh = torch.quantile(d.reshape(-1)[:: max(1, d.numel() // 2_000_000)], quantile)
    dense = d >= thresh
    rms_all = max(float(speed.pow(2).mean().sqrt()), 1e-30)
    out = {
        "jdv_speed_in_dense": float(speed[dense].pow(2).mean().sqrt()) / rms_all,
        "jdv_speed_in_void": float(speed[~dense].pow(2).mean().sqrt()) / rms_all,
    }

    def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.reshape(-1).float(); b = b.reshape(-1).float()
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).mean() / (a.std() * b.std()).clamp_min(1e-30))

    out["jdv_div_rho_corr"] = _corr(div, d)
    out["jdv_speed_rho_corr"] = _corr(speed, logd)
    return out


# --------------------------------------------------------------------------- #
# Cubic-symmetry equivariance
# --------------------------------------------------------------------------- #
#: The three axis-pair rotations by 90 degrees, as ``(axis_a, axis_b)`` on the
#: spatial dims of a ``(B, C, X, Y, Z)`` field.
_ROT_AXES: Tuple[Tuple[int, int], ...] = ((-3, -2), (-3, -1), (-2, -1))


def rotate_field(
    x: torch.Tensor, axes: Tuple[int, int], k: int = 1, vector_slices=((0, 3), (3, 6))
) -> torch.Tensor:
    """Rotate a field by ``k * 90`` degrees about the axis normal to ``axes``.

    Rotating the *spatial* lattice alone is wrong for this data: channels 0-2 and
    3-5 are the components of physical 3-vectors (displacement and velocity) and
    must be rotated with the frame. A 90-degree rotation in the (i, j) plane maps
    ``(v_i, v_j) -> (-v_j, v_i)``; applying it ``k`` times gives the general case.
    """
    y = torch.rot90(x, k=k, dims=list(axes))
    ai, aj = (a + x.dim() for a in axes)          # spatial axis -> component index
    ci, cj = ai - 2, aj - 2                       # (B, C, X, Y, Z): dim 2 == component 0
    y = y.clone()
    for lo, hi in vector_slices:
        if hi > y.shape[1]:
            continue
        vi, vj = lo + ci, lo + cj
        if vj >= hi:
            continue
        for _ in range(k % 4):
            comp_i, comp_j = y[:, vi].clone(), y[:, vj].clone()
            y[:, vi], y[:, vj] = -comp_j, comp_i
    return y


def _rotate_noise(z: Dict[str, torch.Tensor], axes, k: int) -> Dict[str, torch.Tensor]:
    """Noise is a scalar field: rotate the lattice, no component mixing."""
    return {s: torch.rot90(t, k=k, dims=list(axes)) for s, t in z.items()}


@torch.no_grad()
def equivariance_features(
    generator,
    lr_field,
    seed: int,
    nsplit: int,
    pad: int,
    scale_factor: int = 8,
    device=None,
    n_probes: int = 4,
    noise_mode: str = "per_tile",
    global_field=None,
    probe_seed: int = 0,
) -> Dict[str, float]:
    """Rotation/flip self-consistency of the candidate, on a few probe tiles.

    For each probe tile we compute ``y = G(x, z)`` and, for each symmetry ``T``,
    ``y_T = T^-1 G(T x, T z)`` -- transforming the *same* noise realisation with
    the input, as required. The relative rms of ``y - y_T`` is the feature.

    A 3-D CNN is not rotation equivariant, so this is not expected to be zero;
    the useful part is that it varies between candidates, and it does so without
    ever touching the truth. Only ``n_probes`` tiles are used, since the whole
    point of a test-time feature is that it must be cheap relative to sampling.
    """
    lr_np = np.ascontiguousarray(np.asarray(lr_field), dtype=np.float32)
    ng = lr_np.shape[1]
    chunk = ng // nsplit
    lr_size = chunk + 2 * pad
    device = device or next(generator.parameters()).device

    starts = tile_starts(ng, nsplit)
    rng = np.random.default_rng(probe_seed)
    probes = [starts[i] for i in rng.choice(len(starts), size=min(n_probes, len(starts)),
                                            replace=False)]

    rot_rel: List[float] = []
    flip_rel: List[float] = []
    for start in probes:
        crop = periodic_crop(lr_np, start, chunk, pad=pad)
        x = torch.from_numpy(np.ascontiguousarray(crop)).float().unsqueeze(0).to(device)
        z = tile_noise(seed, start, lr_size, scale_factor, device, pad=pad,
                       mode=noise_mode, global_field=global_field)
        y = generator(x, noise=z)
        denom = max(float(y.pow(2).mean().sqrt()), 1e-30)

        for axes in _ROT_AXES:
            y_t = rotate_field(
                generator(rotate_field(x, axes, 1), noise=_rotate_noise(z, axes, 1)),
                axes, -1,
            )
            rot_rel.append(float((y - y_t).pow(2).mean().sqrt()) / denom)

        for dim in (-3, -2, -1):
            xf = torch.flip(x, dims=[dim])
            comp = dim + x.dim() - 2          # flipped spatial axis -> vector component
            xf = xf.clone()
            for lo in (0, 3):
                if lo + comp < xf.shape[1]:
                    xf[:, lo + comp] = -xf[:, lo + comp]
            zf = {s: torch.flip(t, dims=[dim]) for s, t in z.items()}
            y_f = generator(xf, noise=zf)
            y_f = torch.flip(y_f, dims=[dim]).clone()
            for lo in (0, 3):
                if lo + comp < y_f.shape[1]:
                    y_f[:, lo + comp] = -y_f[:, lo + comp]
            flip_rel.append(float((y - y_f).pow(2).mean().sqrt()) / denom)

    return {
        "equiv_rotation_rel": float(np.mean(rot_rel)),
        "equiv_rotation_max": float(np.max(rot_rel)),
        "equiv_flip_rel": float(np.mean(flip_rel)),
        "equiv_n_probes": float(len(probes)),
    }


# --------------------------------------------------------------------------- #
# Noise diagnostics
# --------------------------------------------------------------------------- #
def noise_diagnostics(noise: Optional[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """Per-site mean/std and distance from N(0, 1).

    Flat for freshly drawn noise; the point is Stage 4, where an optimiser can
    push a site far off distribution to game the verifier. ``noise_max_absmean``
    and ``noise_max_sigma_dev`` are the rejection tripwires.
    """
    if not noise:
        return {}
    out: Dict[str, float] = {}
    means, sigmas = [], []
    for site in NOISE_SITES:
        if site not in noise:
            continue
        z = noise[site].detach().float()
        m, s = float(z.mean()), float(z.std())
        out[f"noise_{site}_mean"] = m
        out[f"noise_{site}_std"] = s
        means.append(abs(m)); sigmas.append(abs(s - 1.0))
    if means:
        out["noise_max_absmean"] = float(max(means))
        out["noise_max_sigma_dev"] = float(max(sigmas))
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
#: Feature columns consumed by the verifier, in a fixed order.
FEATURE_KEYS: Tuple[str, ...] = (
    "lr_recon_rel_disp", "lr_recon_rel_vel",
    "boundary_ratio",
    "density_sigma", "density_skew", "vel_rms", "vel_div_rms",
    "plaus_pk_logdist", "plaus_pk_logdist_high", "plaus_pdf_l1",
    "jdv_speed_in_dense", "jdv_speed_in_void", "jdv_div_rho_corr", "jdv_speed_rho_corr",
    "equiv_rotation_rel", "equiv_flip_rel",
)


@torch.no_grad()
def candidate_features(
    sr: torch.Tensor,
    lr: torch.Tensor,
    factor: int = 8,
    geometry: Optional[DensityGeometry] = None,
    reference: Optional[HRReference] = None,
    rho: Optional[torch.Tensor] = None,
    tile_size: int = 0,
    n_bins: int = 24,
    extra: Optional[Dict[str, float]] = None,
    slab: int = 32,
) -> Dict[str, float]:
    """All HR-free features for one ``(1, 6, N, N, N)`` candidate.

    ``extra`` merges in features that need the generator rather than the field
    (:func:`equivariance_features`, :func:`noise_diagnostics`).
    """
    n = sr.shape[-1]
    geo = (geometry or DensityGeometry(ng=n)).for_grid(n)
    if rho is None:
        rho = cic_density_slabs(sr[:, 0:3], geo.cellsize, geo.dis_norm, slab=slab)

    feats: Dict[str, float] = {}
    feats.update(lr_reconstruction(sr, lr, factor))
    if tile_size:
        feats["boundary_ratio"] = boundary_discontinuity(sr, tile_size)
    feats["density_sigma"] = float(rho.std())
    feats["density_skew"] = float(
        ((rho - rho.mean()) ** 3).mean() / rho.std().pow(3).clamp_min(1e-30)
    )
    feats["vel_rms"] = float(sr[:, 3:6].pow(2).mean().sqrt())
    feats["disp_rms"] = float(sr[:, 0:3].pow(2).mean().sqrt())
    from ..dmsr.evaluate import divergence_field
    feats["vel_div_rms"] = float(divergence_field(sr[:, 3:6]).pow(2).mean().sqrt())
    if reference is not None:
        feats.update(reference.distance(rho, n_bins=n_bins))
    feats.update(joint_density_velocity(sr, rho))
    if extra:
        feats.update({k: float(v) for k, v in extra.items()})
    return feats


def feature_matrix(rows: Sequence[Dict[str, float]],
                   keys: Sequence[str] = FEATURE_KEYS) -> np.ndarray:
    """``(n_rows, n_features)`` matrix; missing/NaN entries become 0."""
    x = np.asarray([[float(r.get(k, np.nan)) for k in keys] for r in rows], dtype=np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# Differentiable subset (Stage 4)
# --------------------------------------------------------------------------- #
#: Features that survive backpropagation to the injected noise. Excluded:
#: everything histogram-based (``plaus_pdf_l1``, the PDF errors) -- ``histc`` has
#: zero gradient -- and ``boundary_ratio``, which is a property of the tiling
#: rather than of any one tile.
DIFFERENTIABLE_FEATURE_KEYS: Tuple[str, ...] = (
    "lr_recon_rel_disp", "lr_recon_rel_vel",
    "density_sigma", "density_skew", "vel_rms", "vel_div_rms",
    "plaus_pk_logdist", "plaus_pk_logdist_high",
    "jdv_speed_in_dense", "jdv_speed_in_void", "jdv_div_rho_corr", "jdv_speed_rho_corr",
)


def differentiable_features(
    sr: torch.Tensor,
    lr: torch.Tensor,
    factor: int = 8,
    geometry: Optional[DensityGeometry] = None,
    reference: Optional[HRReference] = None,
    rho: Optional[torch.Tensor] = None,
    slab: int = 32,
) -> Dict[str, torch.Tensor]:
    """The :data:`DIFFERENTIABLE_FEATURE_KEYS` as **torch scalars** with gradients.

    Same definitions as :func:`candidate_features` -- the numbers agree -- but
    returned before the ``float()`` cast so Stage 4 can differentiate a verifier
    score built from them all the way back to the six noise tensors.
    """
    import torch.nn.functional as F
    from ..dmsr.evaluate import auto_cross_power, divergence_field

    n = sr.shape[-1]
    geo = (geometry or DensityGeometry(ng=n)).for_grid(n)
    if rho is None:
        rho = cic_density_slabs(sr[:, 0:3], geo.cellsize, geo.dis_norm, slab=slab)

    a = F.avg_pool3d(sr, kernel_size=int(factor))
    out: Dict[str, torch.Tensor] = {}
    for name, sl in (("disp", slice(0, 3)), ("vel", slice(3, 6))):
        num = (a[:, sl] - lr[:, sl]).pow(2).mean()
        out[f"lr_recon_rel_{name}"] = num / lr[:, sl].pow(2).mean().clamp_min(1e-30)

    out["density_sigma"] = rho.std()
    out["density_skew"] = ((rho - rho.mean()) ** 3).mean() / rho.std().pow(3).clamp_min(1e-30)
    out["vel_rms"] = sr[:, 3:6].pow(2).mean().sqrt()
    out["vel_div_rms"] = divergence_field(sr[:, 3:6]).pow(2).mean().sqrt()

    if reference is not None:
        pk, _, _, _ = auto_cross_power(rho, rho, len(reference.pk))
        ref = torch.as_tensor(reference.pk, device=pk.device, dtype=pk.dtype)
        m = min(len(pk), len(ref))
        logratio = torch.log(pk[:m].clamp_min(1e-30) / ref[:m].clamp_min(1e-30))
        out["plaus_pk_logdist"] = logratio.abs().mean()
        out["plaus_pk_logdist_high"] = logratio[m // 2:].abs().mean()

    v = sr[:, 3:6]
    speed = v.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-30).sqrt()
    thresh = torch.quantile(rho.detach().reshape(-1)[:: max(1, rho.numel() // 2_000_000)], 0.9)
    dense = (rho >= thresh).float()
    rms_all = speed.pow(2).mean().clamp_min(1e-30).sqrt()
    out["jdv_speed_in_dense"] = (
        (speed.pow(2) * dense).sum() / dense.sum().clamp_min(1.0)
    ).clamp_min(1e-30).sqrt() / rms_all
    out["jdv_speed_in_void"] = (
        (speed.pow(2) * (1 - dense)).sum() / (1 - dense).sum().clamp_min(1.0)
    ).clamp_min(1e-30).sqrt() / rms_all

    def _corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1) - x.mean()
        y = y.reshape(-1) - y.mean()
        return (x * y).mean() / (x.std() * y.std()).clamp_min(1e-30)

    out["jdv_div_rho_corr"] = _corr(divergence_field(v), rho)
    out["jdv_speed_rho_corr"] = _corr(speed, torch.log10((rho + 1.0).clamp_min(1e-6)))
    return out
