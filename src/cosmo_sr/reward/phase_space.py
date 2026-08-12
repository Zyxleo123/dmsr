"""Differentiable phase-space summaries of a displacement+velocity crop.

Arm B of the proxy comparison. :mod:`cosmo_sr.reward.soft_structure` reduces a
tile to thirteen numbers computed from the Eulerian *density* alone; this module
adds nine computed from the density **and the velocity field**, and nothing
else changes -- same tiles, same labels, same heads, same optimiser. The
comparison is therefore a clean answer to one question: does a proxy that can
see velocity rank tiles better than one that cannot?

Why velocity might matter at all
--------------------------------
Rockstar is a *phase-space* halo finder. It does not ask "is this region dense";
it links particles in six dimensions and keeps the bound ones. Two fields with
the same density can therefore produce different catalogs: a clump whose
particles share a coherent flow is one object passing through, while the same
clump with a large internal velocity dispersion is either a bound halo or an
unbound superposition depending on how that dispersion compares to its binding
energy. A density-only summary is blind to that distinction by construction, and
subhalo *survival* inside a host is exactly where the distinction bites -- which
is the statistic (``S``, and through it occupation) the reward is built on.

That is an argument for testing Arm B, not for assuming it. If the extra
coordinates buy nothing, the honest outcome is to keep Arm A and say so;
:mod:`scripts/reward/proxy_benchmark.py` is where that decision is written down.

What the features are
---------------------
Everything is mass-weighted over the same valid-centre sub-cube the density
features use, from a single CIC pass that deposits mass, momentum and squared
speed together (:func:`cosmo_sr.eval.density.cic_deposit_valid_center`). One
pass, one geometry: a momentum grid built by a second, independent deposit could
disagree with the mass grid about which particles landed in which cell, and
every feature here is a ratio of the two.

The crop's own bulk velocity is subtracted first. It is a large-scale mode --
the same in the candidate and in the frozen reference, and not something
fine-tuning can or should move -- so leaving it in would put a large common
offset under every coordinate and compress the range that carries the signal.
The intra-cell dispersion ``sigma^2`` is already bulk-free and is not touched.

``vdisp_compact`` / ``vdisp_envelope``
    RMS intra-cell velocity dispersion in the collapsed and in the clustered-
    but-not-collapsed material, in units of ``v_ref_km_s``. The direct measure
    of "is this clump hot or cold", which is what decides whether Rockstar's
    6-D linking holds it together.
``ke_contrast``
    ``log`` of the ratio of specific kinetic energy in the compact material to
    that in the envelope. Scale-free, so a global velocity-amplitude error
    cancels out of it and it reports the *shape* of the velocity field rather
    than its normalisation.
``infall_coherence``
    Mass-weighted mean *radial* velocity of the compact material about its own
    centroid, in units of its rms velocity, signed so positive is infall. Around
    1 for material streaming in, 0 for virialised motion, negative for an
    expanding clump. Coherent infall and virialised motion look identical in
    density and opposite here.
``div_v_delta_corr``
    Pearson correlation across cells between ``log(1 + delta)`` and ``-div v``.
    Infall onto overdensities makes this positive; a field whose velocities have
    been generated without regard to its own density has it near zero.
``kinetic_binding_ratio``
    ``log10`` of ``sigma^2 R / (G M)`` for the compact material -- kinetic energy
    against binding energy, with ``R`` the mass-weighted rms radius about the
    compact centroid. Around 0 for a virialised object; strongly positive for a
    clump that is dense but unbound and would therefore *not* appear in the
    catalog.
``vcoh_s{scale}``
    Fraction of the compact material's kinetic energy that survives box
    smoothing of the velocity field at each scale. A coherent flow keeps it;
    random motion loses it, faster the smaller its correlation length. Three
    scales is the minimum that distinguishes "coherent", "clumpy" and "hot".

All nine are dimensionless and O(1) by construction, so the standardiser in
:class:`cosmo_sr.reward.catalog_proxy.CatalogProxy` starts from a well-scaled
input rather than having to absorb three orders of magnitude of units.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..eval.density import cic_deposit_valid_center
# The arm list lives in the torch-free :mod:`cosmo_sr.reward.arms` (so the
# submitter can read it on a login node) and is re-exported here because every
# existing caller imports it from this module.
from .arms import ARMS
# `_smooth` is private to soft_structure but shared deliberately: the coherence
# ladder here must smooth exactly the way the density peak ladder does, or the
# two blocks would be describing different scales under the same names.
from .soft_structure import (
    SoftStructureConfig, _smooth, density_from_disp, feature_names,
    soft_structure_features,
)

__all__ = [
    "ARMS",
    "FLAT_ARMS",
    "PhaseSpaceConfig",
    "arm_feature_names",
    "arm_features",
    "arm_paired_features",
    "deposit_phase_space",
    "phase_space_feature_names",
    "phase_space_features",
    "phase_space_grid",
    "phase_space_grid_channel_names",
    "phase_space_paired_grid",
    "validate_phase_space_features",
]

#: The arms whose per-tile features are a FLAT vector this module can compute.
#: Arm ``c`` is a token grid and lives in :mod:`cosmo_sr.reward.soft_rockstar`;
#: asking this module for it must raise, not silently fall back to arm ``b``.
FLAT_ARMS: Tuple[str, ...] = ("a", "b")

#: Newton's constant in ``Mpc (km/s)^2 / Msun``. In ``h``-units the factors of
#: ``h`` in ``M`` and ``R`` cancel, so the same number applies to ``Msun/h`` and
#: ``Mpc/h`` without correction.
G_MPC_KMS2_PER_MSUN = 4.30091e-9


@dataclass(frozen=True)
class PhaseSpaceConfig:
    """The velocity-side modelling choices, each one named.

    ``vel_norm_km_s`` is the constant :func:`cosmo_sr.data.preprocess_srs.velnorm`
    divides by, i.e. what turns a stored velocity channel back into km/s. It is
    313.42 at ``z = 0``. It is a *config* value rather than a call to the helper
    so that a feature vector computed at one redshift can never be silently
    compared with one computed at another: the number is written into every
    manifest.
    """

    vel_norm_km_s: float = 313.42210244571896
    #: Velocity scale the dispersions are reported in. 300 km/s is the dispersion
    #: of a ~1e13 Msun/h host, i.e. the middle of the mass range occupation is
    #: decided on, so the compact-region features sit near 1 rather than near
    #: 0.001 where a standardiser has to undo the units.
    v_ref_km_s: float = 300.0
    #: Smoothing scales for the coherence ladder, in HR cells. Matched to
    #: ``SoftStructureConfig.peak_scales`` so the two blocks describe the same
    #: three scales and a difference between them is about velocity, not about
    #: resolution.
    coherence_scales: Tuple[int, ...] = (1, 2, 4)
    #: Mass of one HR particle, ``Msun/h``. 5.819e8 for the deployed
    #: (Om = 0.2814, 100 Mpc/h, 512^3) configuration.
    particle_mass_msun_h: float = 581881454.8686146

    def __post_init__(self) -> None:
        if self.vel_norm_km_s <= 0 or self.v_ref_km_s <= 0:
            raise ValueError("velocity normalisations must be positive")
        if not self.coherence_scales:
            raise ValueError("need at least one coherence scale")
        if self.particle_mass_msun_h <= 0:
            raise ValueError("particle mass must be positive")


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def phase_space_feature_names(ps: Optional[PhaseSpaceConfig] = None) -> List[str]:
    ps = ps or PhaseSpaceConfig()
    return [
        "vdisp_compact", "vdisp_envelope", "ke_contrast", "infall_coherence",
        "div_v_delta_corr", "kinetic_binding_ratio",
    ] + [f"vcoh_s{s}" for s in ps.coherence_scales]


def _central_divergence(v: torch.Tensor) -> torch.Tensor:
    """``(B, 1, R, R, R)`` divergence of a ``(B, 3, R, R, R)`` field, in 1/cell.

    Central differences with replicate padding, matching
    :func:`cosmo_sr.reward.soft_structure._smooth`: a crop face is not empty
    space, and zero padding would manufacture a divergence at every boundary
    that the density correlation would then read as infall.
    """
    p = F.pad(v, (1,) * 6, mode="replicate")
    dx = p[:, 0:1, 2:, 1:-1, 1:-1] - p[:, 0:1, :-2, 1:-1, 1:-1]
    dy = p[:, 1:2, 1:-1, 2:, 1:-1] - p[:, 1:2, 1:-1, :-2, 1:-1]
    dz = p[:, 2:3, 1:-1, 1:-1, 2:] - p[:, 2:3, 1:-1, 1:-1, :-2]
    return 0.5 * (dx + dy + dz)


def _weighted_pearson(a: torch.Tensor, b: torch.Tensor,
                      w: torch.Tensor) -> torch.Tensor:
    """``(B,)`` weighted Pearson correlation of two ``(B, 1, R, R, R)`` fields."""
    dims = (1, 2, 3, 4)
    tot = w.sum(dim=dims).clamp_min(1e-12)
    ma = (w * a).sum(dim=dims) / tot
    mb = (w * b).sum(dim=dims) / tot
    da = a - ma.view(-1, 1, 1, 1, 1)
    db = b - mb.view(-1, 1, 1, 1, 1)
    cov = (w * da * db).sum(dim=dims) / tot
    va = (w * da * da).sum(dim=dims) / tot
    vb = (w * db * db).sum(dim=dims) / tot
    return cov / (va.clamp_min(1e-20) * vb.clamp_min(1e-20)).sqrt()


def deposit_phase_space(
    disp: torch.Tensor,
    vel: torch.Tensor,
    cfg: Optional[SoftStructureConfig] = None,
    ps: Optional[PhaseSpaceConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(m, vbar, sigma2)`` from ONE CIC pass over the valid-centre region.

    ``m`` is ``1 + delta`` (clamped non-negative), ``vbar`` the per-cell
    mass-weighted mean velocity in km/s, ``sigma2`` the intra-cell velocity
    dispersion in (km/s)^2. Shared by arm B's crop summaries and arm C's token
    grid so the two arms cannot disagree about which particles landed in which
    cell -- a second, independent deposit could.
    """
    cfg = cfg or SoftStructureConfig()
    ps = ps or PhaseSpaceConfig()
    if disp.dim() != 5 or disp.shape[1] != 3:
        raise ValueError(f"expected disp (B, 3, N, N, N), got {tuple(disp.shape)}")
    if vel.shape != disp.shape:
        raise ValueError(f"vel {tuple(vel.shape)} must match disp {tuple(disp.shape)}")

    v = vel.float() * float(ps.vel_norm_km_s)                       # km/s
    values = torch.cat([v, (v ** 2).sum(dim=1, keepdim=True)], dim=1)   # (B,4,...)
    raw_mass, acc = cic_deposit_valid_center(
        disp, values, cfg.cellsize_kpc_h, float(cfg.dis_norm_kpc_h),
        region=cfg.region_of(int(disp.shape[-1])), grid_mult=int(cfg.grid_mult))

    m = raw_mass * float(int(cfg.grid_mult) ** 3)      # 1 + delta, as in the
    m = m.clamp_min(0.0)                               # density-only branch
    # Cells with a sliver of mass would make `momentum / mass` explode; the floor
    # is small against a mean of 1 and only ever binds where there is nothing to
    # measure. Every feature downstream is mass-weighted, so those cells
    # contribute ~nothing regardless of what their ratio evaluates to.
    msafe = raw_mass.clamp_min(1e-6)
    vbar = acc[:, 0:3] / msafe                                       # (B,3,R,R,R)
    v2bar = acc[:, 3:4] / msafe                                      # (B,1,R,R,R)
    # Intra-cell dispersion: <|v|^2> - |<v>|^2 within the cell. Non-negative in
    # exact arithmetic; clamped because it is a difference of two large numbers.
    sigma2 = (v2bar - (vbar ** 2).sum(dim=1, keepdim=True)).clamp_min(0.0)
    return m, vbar, sigma2


def phase_space_features(
    disp: torch.Tensor,
    vel: torch.Tensor,
    cfg: Optional[SoftStructureConfig] = None,
    ps: Optional[PhaseSpaceConfig] = None,
) -> torch.Tensor:
    """``(B, 9)`` phase-space features of a crop.

    ``disp`` and ``vel`` are both ``(B, 3, N, N, N)`` in the stored normalised
    units -- the first three and last three channels of a six-channel field.
    Differentiable in both.
    """
    cfg = cfg or SoftStructureConfig()
    ps = ps or PhaseSpaceConfig()
    m, vbar, sigma2 = deposit_phase_space(disp, vel, cfg, ps)
    # Scalar fields are (B, 1, R, R, R) and reduce over `dims`; vector fields are
    # (B, 3, R, R, R) and must keep their component axis, so they reduce over
    # `sdims`. Using `dims` on a vector silently sums x, y and z together.
    dims, sdims = (1, 2, 3, 4), (2, 3, 4)

    total = m.sum(dim=dims).clamp_min(1e-12)
    v0 = (m * vbar).sum(dim=sdims) / total.unsqueeze(1)              # (B,3)
    dv = vbar - v0.view(-1, 3, 1, 1, 1)
    dv2 = (dv ** 2).sum(dim=1, keepdim=True)

    tau = float(cfg.tau_log)
    u = torch.log(m.clamp_min(1e-6))
    compact_u = float(torch.log1p(torch.tensor(cfg.compact_delta)))
    lo_u = float(torch.log1p(torch.tensor(cfg.envelope_lo)))
    hi_u = float(torch.log1p(torch.tensor(cfg.envelope_hi)))
    wc = torch.sigmoid((u - compact_u) / tau) * m
    we = torch.sigmoid((u - lo_u) / tau) * torch.sigmoid((hi_u - u) / tau) * m
    sc = wc.sum(dim=dims).clamp_min(1e-12)
    se = we.sum(dim=dims).clamp_min(1e-12)

    vref = float(ps.v_ref_km_s)
    vdisp_c = ((wc * sigma2).sum(dim=dims) / sc).clamp_min(0.0).sqrt() / vref
    vdisp_e = ((we * sigma2).sum(dim=dims) / se).clamp_min(0.0).sqrt() / vref

    # Specific kinetic energy: bulk-relative streaming plus internal dispersion,
    # which together are the total kinetic energy per unit mass in the cell.
    ke = dv2 + sigma2
    ke_c = (wc * ke).sum(dim=dims) / sc
    ke_e = (we * ke).sum(dim=dims) / se
    ke_contrast = torch.log((ke_c + 1e-6) / (ke_e + 1e-6))

    divv = _central_divergence(dv)
    corr = _weighted_pearson(u, -divv, m)

    # Geometry of the compact material: its centroid, the unit vector pointing
    # away from it, and its mass-weighted rms radius. Shared by the infall and
    # the binding features so both describe the same object.
    r = int(m.shape[-1])
    b = int(m.shape[0])
    cell_mpc = float(cfg.boxsize_mpc_h) / float(cfg.ng_hr) / float(cfg.grid_mult)
    lat = torch.arange(r, device=m.device, dtype=m.dtype) + 0.5
    grid = torch.stack(torch.meshgrid(lat, lat, lat, indexing="ij")).reshape(3, -1)
    wcf = wc.reshape(b, -1)
    wsum = wcf.sum(dim=1).clamp_min(1e-12)
    centroid = (wcf.unsqueeze(1) * grid.unsqueeze(0)).sum(dim=2) / wsum.unsqueeze(1)
    offset = grid.unsqueeze(0) - centroid.unsqueeze(2)                # (B,3,R^3)
    rad2 = (offset ** 2).sum(dim=1)
    rhat = (offset / rad2.clamp_min(1e-12).sqrt().unsqueeze(1)).reshape(b, 3, r, r, r)
    r_rms = ((wcf * rad2).sum(dim=1) / wsum).clamp_min(1e-12).sqrt() * cell_mpc

    # Coherent infall against random motion: the mass-weighted mean *radial*
    # velocity in units of the rms velocity, signed so that positive is infall.
    # Not `|<v>|^2 / <|v|^2>`, which after the crop bulk flow has been removed
    # measures a residual dipole and is ~0 for both a virialised clump and a
    # spherically infalling one -- the two cases this coordinate exists to tell
    # apart.
    rms_v = ((wc * dv2).sum(dim=dims) / sc).clamp_min(1e-12).sqrt()
    infall = -((wc * (dv * rhat).sum(dim=1, keepdim=True)).sum(dim=dims) / sc) / rms_v

    # Kinetic vs binding energy of the compact material, as one lump: sigma^2
    # against G M / R. A crude virial estimator on purpose -- it is a *feature*,
    # not a measurement, and anything fancier would need the object list this
    # module exists to avoid needing.
    mass_msun = wsum / float(int(cfg.grid_mult) ** 3) * float(ps.particle_mass_msun_h)
    binding = G_MPC_KMS2_PER_MSUN * mass_msun / r_rms.clamp_min(1e-6)
    sigma2_c = (wc * (dv2 + sigma2)).sum(dim=dims) / sc
    kb = torch.log10((sigma2_c + 1e-6) / (binding + 1e-6))

    coh: List[torch.Tensor] = []
    denom = (wc * dv2).sum(dim=dims).clamp_min(1e-12)
    for s in ps.coherence_scales:
        # Smooth momentum and mass, not the velocity: a mass-weighted mean is
        # the only average of a velocity field that is not dominated by empty
        # cells whose `momentum / mass` is noise.
        num = _smooth(m * dv, int(s))
        den = _smooth(m, int(s)).clamp_min(1e-6)
        vs = num / den
        coh.append((wc * (vs ** 2).sum(dim=1, keepdim=True)).sum(dim=dims) / denom)

    return torch.stack(
        [vdisp_c, vdisp_e, ke_contrast, infall, corr, kb, *coh], dim=1)


# --------------------------------------------------------------------------- #
# The dense Eulerian phase-space grid (arm E)
# --------------------------------------------------------------------------- #
def phase_space_grid_channel_names() -> List[str]:
    """The five channels of :func:`phase_space_grid`, in order."""
    return ["log_density", "dv_x", "dv_y", "dv_z", "vdisp"]


def phase_space_grid(
    disp: torch.Tensor,
    vel: torch.Tensor,
    cfg: Optional[SoftStructureConfig] = None,
    ps: Optional[PhaseSpaceConfig] = None,
) -> torch.Tensor:
    """``(B, 5, R, R, R)`` dense phase-space cells of a crop -- arm E's input.

    NOT summarised: every valid-centre cell is kept. The five channels, all
    dimensionless and O(1) so the arm's standardiser starts from a well-scaled
    input, are

    1. ``log(1 + delta)`` -- the cell's log density from the shared CIC deposit;
    2-4. the three components of the **bulk-subtracted** mean velocity in units
       of ``v_ref``. The crop's own mass-weighted bulk flow is a large-scale mode
       -- identical in candidate and frozen and not something fine-tuning can move
       -- so it is removed exactly as arm B removes it, leaving the internal
       velocity structure the catalog actually depends on;
    5. the intra-cell velocity dispersion ``sqrt(sigma^2)`` in units of
       ``v_ref``, straight from the deposit (already bulk-free).

    Differentiable in the field, and computed from the SAME
    :func:`deposit_phase_space` pass as arms B and C so the arms cannot disagree
    about which particles landed in which cell.
    """
    cfg = cfg or SoftStructureConfig()
    ps = ps or PhaseSpaceConfig()
    if disp.dim() != 5 or disp.shape[1] != 3:
        raise ValueError(f"expected disp (B, 3, N, N, N), got {tuple(disp.shape)}")
    m, vbar, sigma2 = deposit_phase_space(disp, vel, cfg, ps)
    vref = float(ps.v_ref_km_s)

    total = m.sum(dim=(1, 2, 3, 4)).clamp_min(1e-12)
    v0 = (m * vbar).sum(dim=(2, 3, 4)) / total.unsqueeze(1)             # (B,3)
    dv = (vbar - v0.view(-1, 3, 1, 1, 1)) / vref                        # (B,3,R,R,R)

    log_dens = torch.log(m.clamp_min(1e-6))                            # (B,1,R,R,R)
    # A per-cell sqrt is not pooled first the way arm C's is, so the many empty
    # cells sit at sigma^2 = 0 exactly, where d/dx sqrt(x) is infinite and the
    # actor's field gradient would come back NaN. The epsilon (1e-8 (km/s)^2,
    # i.e. a 1e-4 km/s floor -- ~3e-7 in v_ref units) bounds the gradient without
    # moving the feature: it only ever matters where there is nothing to measure.
    vdisp = (sigma2.clamp_min(0.0) + 1e-8).sqrt() / vref               # (B,1,R,R,R)
    return torch.cat([log_dens, dv, vdisp], dim=1)


def phase_space_paired_grid(
    candidate: torch.Tensor,
    frozen: torch.Tensor,
    cfg: Optional[SoftStructureConfig] = None,
    ps: Optional[PhaseSpaceConfig] = None,
) -> torch.Tensor:
    """``(B, 2, 5, R, R, R)``: candidate grid and candidate-minus-frozen grid.

    Same slot convention as :func:`arm_paired_features` and
    :func:`cosmo_sr.reward.soft_rockstar.soft_rockstar_paired_tokens`: the first
    block is the candidate, the second the candidate minus the frozen reference.
    The frozen branch is detached -- a gradient through the reference would let
    the optimiser shrink the difference by moving a baseline whose weights never
    change. ``candidate`` and ``frozen`` are both ``(B, 6, N, N, N)`` fields.
    """
    cand = phase_space_grid(candidate[:, 0:3], candidate[:, 3:6], cfg, ps)
    with torch.no_grad():
        base = phase_space_grid(frozen[:, 0:3], frozen[:, 3:6], cfg, ps)
    return torch.stack([cand, cand - base.detach()], dim=1)


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def _check_arm(arm: str) -> str:
    a = str(arm).lower()
    if a not in FLAT_ARMS:
        raise ValueError(
            f"this module computes flat feature vectors for arms "
            f"{list(FLAT_ARMS)}, got {arm!r}"
            + (" (arm 'c' is a token grid; use cosmo_sr.reward.soft_rockstar)"
               if a == "c" else ""))
    return a


def arm_feature_names(arm: str, cfg: Optional[SoftStructureConfig] = None,
                      ps: Optional[PhaseSpaceConfig] = None) -> List[str]:
    """Names of the *base* block of one arm, before the frozen-relative pairing."""
    names = list(feature_names(cfg))
    if _check_arm(arm) == "b":
        names += phase_space_feature_names(ps)
    return names


def arm_paired_feature_names(arm: str, cfg: Optional[SoftStructureConfig] = None,
                             ps: Optional[PhaseSpaceConfig] = None) -> List[str]:
    base = arm_feature_names(arm, cfg, ps)
    return list(base) + [f"d_{n}" for n in base]


def arm_features(field: torch.Tensor, arm: str,
                 cfg: Optional[SoftStructureConfig] = None,
                 ps: Optional[PhaseSpaceConfig] = None) -> torch.Tensor:
    """``(B, F)`` base features of one arm from a ``(B, C>=3, N, N, N)`` crop.

    Arm ``a`` needs only the displacement channels; arm ``b`` needs all six and
    says so rather than silently scoring velocity as zero.
    """
    cfg = cfg or SoftStructureConfig()
    a = _check_arm(arm)
    dens = soft_structure_features(density_from_disp(field, cfg), cfg)
    if a == "a":
        return dens
    if field.shape[1] < 6:
        raise ValueError(
            f"arm 'b' needs 6 channels (displacement + velocity), got "
            f"{field.shape[1]}; a 3-channel field has no phase space to summarise")
    return torch.cat(
        [dens, phase_space_features(field[:, 0:3], field[:, 3:6], cfg, ps)], dim=1)


def arm_paired_features(candidate: torch.Tensor, frozen: torch.Tensor, arm: str,
                        cfg: Optional[SoftStructureConfig] = None,
                        ps: Optional[PhaseSpaceConfig] = None) -> torch.Tensor:
    """``(B, 2F)``: one arm's features and their difference from frozen SR2.

    Same contract as :func:`cosmo_sr.reward.soft_structure.paired_features`, and
    for arm ``a`` it returns exactly that -- the frozen branch is detached,
    because a gradient through the reference would let the optimiser shrink the
    difference by moving a baseline whose weights never change.
    """
    cand = arm_features(candidate, arm, cfg, ps)
    with torch.no_grad():
        base = arm_features(frozen, arm, cfg, ps)
    return torch.cat([cand, cand - base.detach()], dim=1)


# --------------------------------------------------------------------------- #
# Synthetic validation
# --------------------------------------------------------------------------- #
def _synthetic_phase_space(n: int, *, infall_fraction: float,
                           v_total_km_s: float = 300.0, seed: int = 0,
                           ps: Optional[PhaseSpaceConfig] = None):
    """A collapsed clump whose motion is part radial infall, part random noise.

    One displacement field for every call -- particles pulled a fixed fraction
    of the way to the crop centre -- so the *density* is identical across the
    ladder and any density feature that moves is a bug. Only the velocity split
    changes: ``infall_fraction`` of the rms speed is coherent inward motion, the
    rest is isotropic noise, at constant total speed.
    """
    ps = ps or PhaseSpaceConfig()
    g = torch.Generator().manual_seed(int(seed))
    lat = torch.arange(n, dtype=torch.float32) - (n - 1) / 2.0
    q = torch.stack(torch.meshgrid(lat, lat, lat, indexing="ij")).unsqueeze(0)
    rhat = q / q.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-6).sqrt()
    disp = (-q * 0.75 / 6000.0 * (100000.0 / 512.0))

    f = float(infall_fraction)
    noise = torch.randn(1, 3, n, n, n, generator=g)
    noise = noise / noise.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
    v = float(v_total_km_s) * (-f * rhat + (1.0 - f ** 2) ** 0.5 * noise)
    return disp, v / float(ps.vel_norm_km_s)


def validate_phase_space_features(
    cfg: Optional[SoftStructureConfig] = None,
    ps: Optional[PhaseSpaceConfig] = None,
    *,
    n: int = 32,
    infall_fractions: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    dispersions: Sequence[float] = (50.0, 150.0, 300.0, 600.0),
) -> dict:
    """Do the velocity coordinates respond to velocity and not to density?

    Three claims, checked as ladders rather than at two points, for the same
    reason :func:`cosmo_sr.reward.soft_structure.validate_soft_features` does:

    * at **fixed density and fixed speed**, moving kinetic energy from random
      motion into coherent infall must raise ``infall_coherence``
      monotonically. That is the coordinate whose whole justification is telling
      a virialised clump from an infalling one, and density cannot;
    * at fixed density and fixed infall fraction, raising the speed must raise
      ``vdisp_compact`` monotonically;
    * the *density* coordinates must be untouched by any of it. The two blocks
      are supposed to be independent, and a density feature that moved when only
      the velocities changed would make the arm comparison meaningless -- Arm B
      would differ from Arm A in its shared coordinates too.

    Returns numbers rather than asserting, so the unit test and any report
    consume the same measurement.
    """
    cfg = cfg or SoftStructureConfig()
    ps = ps or PhaseSpaceConfig()
    names = phase_space_feature_names(ps)
    n_dens = len(feature_names(cfg))
    i_inf = names.index("infall_coherence")
    i_disp = names.index("vdisp_compact")

    def row(**kw):
        d, v = _synthetic_phase_space(int(n), ps=ps, **kw)
        f = torch.cat([d, v], dim=1)
        return arm_features(f, "b", cfg, ps)[0]

    inf_rows = torch.stack([row(infall_fraction=float(x)) for x in infall_fractions])
    spd_rows = torch.stack([row(infall_fraction=0.5, v_total_km_s=float(s))
                            for s in dispersions])

    infall = inf_rows[:, n_dens + i_inf].tolist()
    vdisp = spd_rows[:, n_dens + i_disp].tolist()
    dens = torch.cat([inf_rows[:, :n_dens], spd_rows[:, :n_dens]], dim=0)
    dens_spread = float((dens.max(dim=0).values - dens.min(dim=0).values).abs().max())
    dens_scale = float(dens.abs().max().clamp_min(1e-12))

    mono_inf = all(infall[i] < infall[i + 1] for i in range(len(infall) - 1))
    mono_v = all(vdisp[i] < vdisp[i + 1] for i in range(len(vdisp) - 1))
    unchanged = dens_spread <= 1e-4 * dens_scale
    return {
        "infall_fractions": [float(x) for x in infall_fractions],
        "speeds_km_s": [float(x) for x in dispersions],
        "phase_space_feature_names": names,
        "infall_coherence": infall,
        "vdisp_compact": vdisp,
        "infall_monotone_increasing": bool(mono_inf),
        "vdisp_monotone_increasing": bool(mono_v),
        "density_feature_max_abs_spread": dens_spread,
        "density_features_unchanged": bool(unchanged),
        "ok": bool(mono_inf and mono_v and unchanged),
    }
