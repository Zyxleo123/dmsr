"""Differentiable structural features of a displacement crop.

The catalog reward is a function of a halo catalog, and a halo catalog is
produced by a non-differentiable binary. Something has to stand in for it inside
a training step. This module is the *first* thing tried, deliberately: a small
fixed feature vector rather than a learned 3-D critic.

Why start here and not with a patch network
-------------------------------------------
A critic trained on "HR versus SR2" learns the easiest available discriminator,
and on this data that is a global one -- SR2's density power is biased at high
k, which separates the two classes perfectly without saying anything about
whether *this* tile gained a subhalo. The resulting gradient points at "look
more like HR overall", which the density guard then has to fight. A short,
interpretable feature vector cannot hide such a shortcut: every coordinate is
nameable, and the ablation "does the proxy still work without envelope mass?"
is a one-line experiment rather than an interpretability project.

If the feature proxy fails its gate (section 7 of the plan), the fallback is a
patch network built on :class:`cosmo_sr.tts.verifier.PatchVerifier`. It is a
fallback, not a starting point.

What the features are
---------------------
Everything is computed from the Eulerian overdensity ``delta`` obtained with
:func:`cosmo_sr.eval.density.cic_density_valid_center` -- the *valid-centre* CIC,
because a crop's particles do not stay inside the crop (only ~10% do) and the
periodic-wrap version of CIC scrambles a crop's density into something correlated
``r = 0.08`` with the truth.

``compact_mass``
    Mass fraction in cells with ``delta`` above ~100, through a sigmoid rather
    than an indicator so the boundary carries gradient. This is the "is there a
    collapsed object here" coordinate.
``envelope_mass``
    Mass fraction in ``10 < delta < 100``: the material that is clustered but
    has not collapsed. A generator that smears a halo moves mass from the first
    coordinate to this one at constant total mass, which is exactly the
    distinction a scalar density statistic cannot make.
``peak_s{scale}``
    Soft excursion-set volume at three smoothing scales. A single compact clump
    survives smoothing at scale 1 and vanishes by scale 4; a diffuse blob of the
    same mass does the opposite. Three scales is the minimum that separates
    "compact", "extended" and "smooth".
``peak_contrast_s{scale}``
    How far a cell rises above its own smoothed neighbourhood, mass-weighted.
    Threshold-free, so it is the coordinate least able to be gamed by a global
    amplitude shift.
``second_moment``
    Compact-mass-weighted second spatial moment about its own centroid, in units
    of the region half-width. With one clump this *is* concentration. With
    several it measures how spread the clumps are, which is still monotone in
    smearing; the synthetic gate in :func:`validate_soft_features` is the
    single-clump case where the interpretation is unambiguous.
``logP_b{i}``
    Log density power in a few radial bands. Present so the proxy can see the
    statistic the density guard blocks on, rather than having to infer it.

:func:`paired_features` returns these for a candidate **and their difference
from the frozen SR2 field at the same ``(Y, z)``**. The difference is the part
the actor can actually move: absolute features are dominated by which tile of
which box we are looking at, which no amount of fine-tuning changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..eval.density import cic_density_valid_center
from ..losses.flow import _kbin_index

__all__ = [
    "SoftStructureConfig",
    "density_from_disp",
    "feature_names",
    "paired_feature_names",
    "paired_features",
    "soft_structure_features",
    "structural_diversity",
    "validate_soft_features",
]


@dataclass(frozen=True)
class SoftStructureConfig:
    """Thresholds and scales. Every number is a modelling choice, so it is named.

    All soft thresholds are applied to ``u = log1p(delta)``, not to ``delta``.
    Density here spans ``-1`` to several thousand, and a sigmoid of width
    proportional to the threshold cannot cover that: at ``delta_threshold = 100``
    with a 35% width, an *empty* cell still scores ``sigmoid(-2.86) = 0.054``, so
    a completely smooth field reported 5.4% "compact mass" and the real signal
    (a few percent) sat on top of a constant five times its size. In log space
    the same cell scores ``1e-4`` and the gradient is well conditioned across
    the whole range, which is the property that made this the right coordinate
    rather than a smaller ``tau``.

    ``tau_log`` is therefore an absolute width in natural-log units: 0.35 means
    the transition takes about a factor of 1.4 in ``1 + delta``. Measured
    pedestals on a uniform field at that width are ``2e-6`` for
    ``compact_mass`` and ``1e-3`` for ``envelope_mass``, against feature
    ranges of 0.06 and 0.07 across a concentration ladder -- i.e. under 2% of
    the signal, i.e. a constant a linear head absorbs into its bias.
    """

    compact_delta: float = 100.0
    envelope_lo: float = 10.0
    envelope_hi: float = 100.0
    tau_log: float = 0.35
    peak_scales: Tuple[int, ...] = (1, 2, 4)
    #: Excursion threshold per smoothing scale. Falls with scale because
    #: smoothing lowers peaks; these are the values a single compact clump of
    #: the mass Rockstar can resolve stays above.
    peak_deltas: Tuple[float, ...] = (100.0, 30.0, 10.0)
    #: Margin, in natural-log units of ``1 + delta``, by which a cell must beat
    #: its own smoothed neighbourhood to count as a local peak. 1.5 is a factor
    #: of ~4.5, which leaves a uniform field at ``sigmoid(-3) = 0.047`` rather
    #: than on a 0.27 pedestal that would compress the useful range.
    contrast_margin: float = 1.5
    n_power_bands: int = 4
    #: Scored sub-cube as a fraction of the crop. 0.5 is the value the measured
    #: table in ``cic_density_valid_center`` puts at rel-RMS 0.03 / corr 0.9995.
    region_fraction: float = 0.5
    grid_mult: int = 1
    #: Grid geometry, in the on-disk conventions used everywhere in this repo.
    boxsize_mpc_h: float = 100.0
    ng_hr: int = 512
    dis_norm_kpc_h: float = 6000.0

    @property
    def cellsize_kpc_h(self) -> float:
        return float(self.boxsize_mpc_h) * 1000.0 / float(self.ng_hr)

    def region_of(self, crop: int) -> int:
        r = int(round(float(self.region_fraction) * int(crop)))
        r = max(4, min(int(crop), r - (r % 2)))
        return r

    def __post_init__(self) -> None:
        if len(self.peak_scales) != len(self.peak_deltas):
            raise ValueError(
                f"peak_scales {self.peak_scales} and peak_deltas "
                f"{self.peak_deltas} must have the same length"
            )
        if not 0.0 < self.tau_log:
            raise ValueError("tau_log must be positive")
        if not self.envelope_lo < self.envelope_hi:
            raise ValueError("envelope_lo must be below envelope_hi")


# --------------------------------------------------------------------------- #
# Density
# --------------------------------------------------------------------------- #
def density_from_disp(
    disp: torch.Tensor, cfg: Optional[SoftStructureConfig] = None
) -> torch.Tensor:
    """``(B, 1, R, R, R)`` valid-centre overdensity from ``(B, 3, N, N, N)`` displacement.

    Displacement channels only -- passing a six-channel field here would deposit
    particles at ``q + velocity``, which is not a position.
    """
    cfg = cfg or SoftStructureConfig()
    if disp.dim() != 5:
        raise ValueError(f"expected (B, C, N, N, N), got {tuple(disp.shape)}")
    if disp.shape[1] < 3:
        raise ValueError(f"need at least 3 displacement channels, got {disp.shape[1]}")
    d = disp[:, 0:3]
    return cic_density_valid_center(
        d, cfg.cellsize_kpc_h, float(cfg.dis_norm_kpc_h),
        region=cfg.region_of(int(d.shape[-1])), grid_mult=int(cfg.grid_mult),
    )


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def feature_names(cfg: Optional[SoftStructureConfig] = None) -> List[str]:
    cfg = cfg or SoftStructureConfig()
    names = ["compact_mass", "envelope_mass"]
    names += [f"peak_s{s}" for s in cfg.peak_scales]
    names += [f"peak_contrast_s{s}" for s in cfg.peak_scales]
    names += ["second_moment"]
    names += [f"logP_b{i}" for i in range(int(cfg.n_power_bands))]
    return names


def _smooth(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Box-average over a ``(2*scale+1)^3`` window, edges replicated.

    Replicate rather than zero padding: a crop's face is not empty space, and a
    zero pad would manufacture a density gradient at every boundary that the
    contrast feature would then read as structure.
    """
    s = int(scale)
    if s <= 0:
        return x
    k = 2 * s + 1
    xp = F.pad(x, (s,) * 6, mode="replicate")
    return F.avg_pool3d(xp, kernel_size=k, stride=1)


def _radial_log_power(delta: torch.Tensor, n_bands: int) -> torch.Tensor:
    """``(B, n_bands)`` log power, per sample (not averaged over the batch)."""
    x = delta.float()
    b, _, n = x.shape[0], x.shape[1], x.shape[-1]
    f = torch.fft.rfftn(x, dim=(-3, -2, -1))
    p = (f.real ** 2 + f.imag ** 2) / float(n ** 3)
    p = p.reshape(b, -1)
    idx = _kbin_index(n, int(n_bands), x.device).reshape(-1)
    sums = torch.zeros(b, int(n_bands), device=x.device, dtype=p.dtype)
    counts = torch.zeros(int(n_bands), device=x.device, dtype=p.dtype)
    sums.index_add_(1, idx, p)
    counts.index_add_(0, idx, torch.ones_like(p[0]))
    return torch.log((sums / counts.clamp_min(1.0)).clamp_min(1e-12))


def soft_structure_features(
    delta: torch.Tensor, cfg: Optional[SoftStructureConfig] = None
) -> torch.Tensor:
    """``(B, F)`` differentiable features of a ``(B, 1, R, R, R)`` overdensity."""
    cfg = cfg or SoftStructureConfig()
    if delta.dim() != 5 or delta.shape[1] != 1:
        raise ValueError(f"expected (B, 1, R, R, R), got {tuple(delta.shape)}")
    d = delta.float()
    b, r = int(d.shape[0]), int(d.shape[-1])
    tau = float(cfg.tau_log)

    # Mass, not volume: a cell's contribution is what it carries. Clamped at 0
    # because valid-centre CIC subtracts the uniform expectation, and a cell that
    # lost a sliver of mass at the region face can land a hair below -1; a
    # negative mass weight there would flip the sign of a feature.
    mass = (1.0 + d).clamp_min(0.0)
    total = mass.sum(dim=(1, 2, 3, 4)).clamp_min(1e-12)
    u = torch.log(mass.clamp_min(1e-6))          # log(1 + delta), guarded

    compact_u = float(torch.log1p(torch.tensor(cfg.compact_delta)))
    lo_u = float(torch.log1p(torch.tensor(cfg.envelope_lo)))
    hi_u = float(torch.log1p(torch.tensor(cfg.envelope_hi)))

    w_compact = torch.sigmoid((u - compact_u) / tau)
    w_env = torch.sigmoid((u - lo_u) / tau) * torch.sigmoid((hi_u - u) / tau)

    compact = (w_compact * mass).sum(dim=(1, 2, 3, 4)) / total
    envelope = (w_env * mass).sum(dim=(1, 2, 3, 4)) / total

    peaks: List[torch.Tensor] = []
    contrasts: List[torch.Tensor] = []
    for scale, thr in zip(cfg.peak_scales, cfg.peak_deltas):
        us = torch.log(_smooth(mass, int(scale)).clamp_min(1e-6))
        thr_u = float(torch.log1p(torch.tensor(float(thr))))
        peaks.append(torch.sigmoid((us - thr_u) / tau).mean(dim=(1, 2, 3, 4)))
        excess = u - us - float(cfg.contrast_margin)
        contrasts.append(
            (torch.sigmoid(excess / tau) * mass).sum(dim=(1, 2, 3, 4)) / total
        )

    # Second spatial moment of the compact mass about its own centroid, in units
    # of the region half-width, so it is dimensionless and O(1).
    wm = (w_compact * mass).reshape(b, -1)
    wsum = wm.sum(dim=1).clamp_min(1e-8)
    lat = torch.arange(r, device=d.device, dtype=d.dtype) + 0.5
    grid = torch.stack(torch.meshgrid(lat, lat, lat, indexing="ij")).reshape(3, -1)
    centroid = (wm.unsqueeze(1) * grid.unsqueeze(0)).sum(dim=2) / wsum.unsqueeze(1)
    sq = ((grid.unsqueeze(0) - centroid.unsqueeze(2)) ** 2).sum(dim=1)
    half = 0.5 * float(r)
    second = (wm * sq).sum(dim=1) / wsum / (half ** 2)

    bands = _radial_log_power(d, int(cfg.n_power_bands))

    cols = [compact, envelope, *peaks, *contrasts, second]
    return torch.cat([torch.stack(cols, dim=1), bands], dim=1)


def paired_feature_names(cfg: Optional[SoftStructureConfig] = None) -> List[str]:
    base = feature_names(cfg)
    return list(base) + [f"d_{n}" for n in base]


def paired_features(
    candidate_disp: torch.Tensor,
    frozen_disp: torch.Tensor,
    cfg: Optional[SoftStructureConfig] = None,
) -> torch.Tensor:
    """``(B, 2F)``: the candidate's features and their difference from frozen.

    The frozen branch is detached. It is a *reference*, and a gradient through
    it would let the optimiser reduce the difference by moving the baseline --
    which it cannot do in reality, since the frozen weights never change, so any
    such gradient is a pure artefact of the graph.
    """
    cfg = cfg or SoftStructureConfig()
    cand = soft_structure_features(density_from_disp(candidate_disp, cfg), cfg)
    with torch.no_grad():
        base = soft_structure_features(density_from_disp(frozen_disp, cfg), cfg)
    return torch.cat([cand, cand - base.detach()], dim=1)


# --------------------------------------------------------------------------- #
# Diversity
# --------------------------------------------------------------------------- #
def structural_diversity(
    disp_draws: torch.Tensor, cfg: Optional[SoftStructureConfig] = None
) -> Dict[str, torch.Tensor]:
    """Spread across noise draws, in the coordinates that matter.

    ``disp_draws`` is ``(B, D, C, N, N, N)``: the same LR tile under ``D``
    different ``z``. Returns ``displacement``, ``density`` and ``soft_peak``
    spreads plus a combined ``d_struct``.

    Deliberately **not** the RMS spread of the six-channel output. Velocity
    carries most of that variance and none of the substructure question, so a
    six-channel diversity floor is satisfied by a generator whose displacement
    field has collapsed to a deterministic map -- the exact failure this guard
    exists to catch.
    """
    cfg = cfg or SoftStructureConfig()
    if disp_draws.dim() != 6:
        raise ValueError(f"expected (B, D, C, N, N, N), got {tuple(disp_draws.shape)}")
    b, dd = int(disp_draws.shape[0]), int(disp_draws.shape[1])
    if dd < 2:
        raise ValueError("structural diversity needs at least 2 noise draws")

    psi = disp_draws[:, :, 0:3]
    spread = psi.std(dim=1, unbiased=True)
    scale = psi.reshape(b, -1).pow(2).mean(dim=1).clamp_min(1e-20).sqrt()
    # clamp_min before every sqrt: a collapsed sampler makes the variance
    # exactly zero, where d(sqrt)/dx is infinite -- and this quantity is
    # inside a loss, so that infinity becomes NaN weights.
    disp_div = spread.reshape(b, -1).pow(2).mean(dim=1).clamp_min(1e-24).sqrt() / scale

    flat = disp_draws.reshape(b * dd, *disp_draws.shape[2:])
    delta = density_from_disp(flat, cfg)
    dv = delta.reshape(b, dd, -1)
    den_div = (dv.std(dim=1, unbiased=True).pow(2).mean(dim=1).clamp_min(1e-24).sqrt()
               / dv.reshape(b, -1).pow(2).mean(dim=1).clamp_min(1e-20).sqrt())

    feats = soft_structure_features(delta, cfg).reshape(b, dd, -1)
    n_peaks = 2 * len(cfg.peak_scales)
    peak = feats[:, :, 2:2 + n_peaks]
    peak_div = (peak.std(dim=1, unbiased=True).pow(2).mean(dim=1).clamp_min(1e-24).sqrt()
                / peak.reshape(b, -1).pow(2).mean(dim=1).clamp_min(1e-20).sqrt())

    return {
        "displacement": disp_div,
        "density": den_div,
        "soft_peak": peak_div,
        # Geometric-mean-free: the minimum is what a floor should bind on, since
        # a healthy density spread must not be allowed to excuse a collapsed
        # displacement field.
        "d_struct": torch.minimum(torch.minimum(disp_div, den_div), peak_div),
    }


# --------------------------------------------------------------------------- #
# Synthetic validation
# --------------------------------------------------------------------------- #
def _synthetic_delta(n: int, sigma_cells: float, mass: float,
                     device=None, dtype=torch.float32) -> torch.Tensor:
    """A single Gaussian clump of fixed total mass and adjustable width.

    Built directly in density space rather than through CIC, because the claim
    being tested is about the *features*, not about the deposit: "same mass,
    different concentration, higher compactness score".
    """
    dev = device or torch.device("cpu")
    lat = torch.arange(n, device=dev, dtype=dtype) - (n - 1) / 2.0
    g = torch.stack(torch.meshgrid(lat, lat, lat, indexing="ij"))
    r2 = (g ** 2).sum(dim=0)
    blob = torch.exp(-0.5 * r2 / max(float(sigma_cells), 1e-6) ** 2)
    blob = blob / blob.sum().clamp_min(1e-12) * float(mass)
    # Uniform background of one unit per cell, plus the clump's excess.
    rho = 1.0 + blob
    return (rho / rho.mean() - 1.0).reshape(1, 1, n, n, n)


def validate_soft_features(
    cfg: Optional[SoftStructureConfig] = None,
    *,
    n: int = 32,
    mass: float = 4000.0,
    sigmas: Sequence[float] = (0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
) -> Dict[str, object]:
    """Do the features order a *ladder* of concentrations at constant mass?

    The plan's requirement is "compact clumps score higher than smeared fields
    with the same mass". Checking two points would pass for a feature that is
    non-monotone in between -- which is a real risk here, because
    ``envelope_mass`` genuinely is a band statistic and *should* be non-monotone.
    So this sweeps a ladder of Gaussian widths at fixed total mass and asks for
    monotonicity of the two coordinates that are supposed to be monotone:

    * ``compact_mass`` strictly decreasing as the clump is smeared out;
    * ``second_moment`` strictly increasing.

    ``envelope_mass`` is reported and explicitly **not** required to be monotone;
    what is required of it is that the smearing moves mass *into* it from
    ``compact_mass`` somewhere along the ladder, which is the trade the feature
    pair exists to expose.

    Returns numbers rather than asserting, so the unit test and the proxy-gate
    script consume the same measurement.
    """
    cfg = cfg or SoftStructureConfig()
    names = feature_names(cfg)
    sig = [float(s) for s in sigmas]
    if len(sig) < 3:
        raise ValueError("need at least three widths to call anything monotone")
    fields = [_synthetic_delta(int(n), s, float(mass)) for s in sig]
    feats = torch.cat([soft_structure_features(f, cfg) for f in fields], dim=0)

    i_c = names.index("compact_mass")
    i_e = names.index("envelope_mass")
    i_m = names.index("second_moment")
    compact = feats[:, i_c].tolist()
    envelope = feats[:, i_e].tolist()
    moment = feats[:, i_m].tolist()
    total_mass = [float((1.0 + f).sum()) for f in fields]

    mono_down = all(compact[i] > compact[i + 1] for i in range(len(sig) - 1))
    mono_up = all(moment[i] < moment[i + 1] for i in range(len(sig) - 1))
    # Mass must actually be conserved along the ladder, or "same mass" is a
    # claim about the test rather than about the features.
    mass_spread = (max(total_mass) - min(total_mass)) / max(abs(total_mass[0]), 1e-12)
    traded = max(envelope) > envelope[0]

    return {
        "sigmas": sig,
        "feature_names": names,
        "compact_mass": compact,
        "envelope_mass": envelope,
        "second_moment": moment,
        "total_mass": total_mass,
        "total_mass_relative_spread": float(mass_spread),
        "compact_mass_monotone_decreasing": bool(mono_down),
        "second_moment_monotone_increasing": bool(mono_up),
        "envelope_gains_from_smearing": bool(traded),
        "compact_mass_range": float(max(compact) - min(compact)),
        "ok": bool(mono_down and mono_up and traded and mass_spread < 1e-3),
    }
