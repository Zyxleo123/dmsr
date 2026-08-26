"""Sparse, local, multiscale candidate score (HackMD note, step 3).

The shaping reward never scores the box globally. It scores one *candidate* at a
time -- a centre :math:`\\mu_c` and the fixed set of tracer particles collected
around it (:mod:`scripts/reward/diagnose_candidate_partition.py` validated that
this set contains the material) -- from a handful of physical summaries computed
at several spatial scales, and passes them to a small survival head. This module
is those summaries and that head, differentiable in the particles' live
positions and velocities so gradients reach the SR2 field.

Why multiscale, and why velocity
--------------------------------
A single narrow kernel rewards only the final concentration and gives almost no
gradient while the material is still spread out; a single broad kernel rewards
gathering but never distinguishes a diffuse cloud from a collapsed core. The note
asks for a *ladder* -- here :math:`\\{1, 2, 4\\}\\times R_{ref}` -- so the broad
rungs carry the early signal and the narrow rung the late one, and the score has
usable gradient along the whole collapse path.

Density is not enough on its own: Rockstar keeps *bound* objects, so a dense clump
whose particles are flying apart is not a subhalo. Every scale therefore also
carries the internal velocity dispersion and a crude virial binding margin

    b(R) = G M(R) / R - sigma_v^2(R),

positive for a bound object and negative for one that is dense but unbound. This
is the coordinate that lets the score reject the failure the pairwise sum cannot
see -- identical density, hot velocities.

Everything is a *soft* count (a sigmoid radial indicator), so the summaries are
smooth functions of the particle positions and the whole thing is one
differentiable expression. Nothing here selects particles or centres; those
discrete choices are the detached partition step and live upstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

__all__ = [
    "CandidateScoreConfig",
    "G_MPC_KMS2_PER_MSUN",
    "SurvivalHead",
    "candidate_features",
    "feature_names",
    "formation_score",
    "unwrap_about",
]

#: Newton's constant in ``Mpc (km/s)^2 / Msun`` (h-units; the h factors cancel).
G_MPC_KMS2_PER_MSUN = 4.30091e-9


@dataclass(frozen=True)
class CandidateScoreConfig:
    """The candidate-score modelling choices, each one named.

    ``scale_mults`` is the radius ladder in units of the candidate's reference
    radius ``R_ref`` (the object's own scale, e.g. its HR half-mass radius). The
    broadest rung supplies early gradient, the narrowest the final concentration.

    ``tau_frac`` is the width of the soft radial indicator as a fraction of each
    radius, so the boundary is equally soft at every scale rather than sharp at
    the small ones and mushy at the large ones.

    ``bg_number_density_mpc3`` is the mean tracer number density of the box
    (``N / L^3``); overdensities are reported against it. Passed rather than
    inferred so a candidate cropped from a sub-region still compares to the whole
    box's mean, which is what "overdense" means physically.
    """

    scale_mults: Tuple[float, ...] = (1.0, 2.0, 4.0)
    tau_frac: float = 0.15
    particle_mass_msun_h: float = 581881454.8686146
    bg_number_density_mpc3: float = 512.0 ** 3 / 100.0 ** 3
    #: Dimensionless virial ratio ``sigma_v^2 / (G M / R)`` at or below which a
    #: candidate is treated as bound. The raw margin ``G M / R - sigma_v^2`` has
    #: no calibrated zero -- a *virialised* halo sits at some object-dependent
    #: negative value, not at zero -- so the score gates on this ratio instead.
    #: A stable self-gravitating clump is O(1); an unbound one is >> 1. The first
    #: run measured HR halos at ~1.9 and a hot control at ~9, so the threshold
    #: sits between them and its width spans that gap.
    virial_thresh: float = 2.5
    virial_width: float = 1.5
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if not self.scale_mults or any(s <= 0 for s in self.scale_mults):
            raise ValueError("scale_mults must be positive and non-empty")
        if self.tau_frac <= 0:
            raise ValueError("tau_frac must be positive")
        if self.particle_mass_msun_h <= 0 or self.bg_number_density_mpc3 <= 0:
            raise ValueError("mass and background density must be positive")
        if self.virial_thresh <= 0 or self.virial_width <= 0:
            raise ValueError("virial threshold and width must be positive")


def feature_names(cfg: Optional[CandidateScoreConfig] = None) -> Tuple[str, ...]:
    cfg = cfg or CandidateScoreConfig()
    per = ["log1p_overdensity", "sigma_v_kms", "virial_ratio"]
    names = [f"{p}_s{m:g}" for m in cfg.scale_mults for p in per]
    return tuple(names + ["concentration"])


def unwrap_about(pos: torch.Tensor, center: torch.Tensor, box: float) -> torch.Tensor:
    """Move ``pos`` to the periodic image nearest ``center`` (differentiable).

    A candidate straddling the box face would otherwise have members a full box
    apart from their own centre; unwrapping first makes the Euclidean radius the
    physical one. ``center`` is broadcast over the particle axis.
    """
    d = pos - center
    d = d - float(box) * torch.round(d / float(box))
    return center + d


def _soft_weight(r: torch.Tensor, radius: float, cfg: CandidateScoreConfig
                 ) -> torch.Tensor:
    """``sigmoid((R - r) / (tau R))`` -- a smooth indicator of "inside radius R"."""
    tau = max(cfg.tau_frac * float(radius), cfg.eps)
    return torch.sigmoid((float(radius) - r) / tau)


def candidate_features(
    pos: torch.Tensor,
    vel: torch.Tensor,
    center: torch.Tensor,
    r_ref: float,
    cfg: Optional[CandidateScoreConfig] = None,
    box: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Multiscale physical summaries of one candidate's collected particles.

    ``pos`` ``(M, 3)`` in Mpc/h, ``vel`` ``(M, 3)`` in km/s, ``center`` ``(3,)``.
    Differentiable in ``pos`` and ``vel``. Returns a dict of scalars, one entry
    per feature in :func:`feature_names`, plus the raw per-scale pieces the
    survival head and the diagnostics read.
    """
    cfg = cfg or CandidateScoreConfig()
    if pos.dim() != 2 or pos.shape[1] != 3 or vel.shape != pos.shape:
        raise ValueError(f"pos/vel must be (M, 3); got {tuple(pos.shape)}, "
                         f"{tuple(vel.shape)}")
    if box is not None:
        pos = unwrap_about(pos, center.view(1, 3), box)
    r = torch.linalg.norm(pos - center.view(1, 3), dim=1)               # (M,)
    m_p = float(cfg.particle_mass_msun_h)

    out: Dict[str, torch.Tensor] = {}
    masses = []
    for mult in cfg.scale_mults:
        R = float(mult) * float(r_ref)
        w = _soft_weight(r, R, cfg)                                     # (M,)
        n = w.sum().clamp_min(cfg.eps)
        mass = n * m_p
        vol = 4.0 / 3.0 * torch.pi * (R ** 3)
        n_expected = cfg.bg_number_density_mpc3 * vol
        overdensity = n / max(n_expected, cfg.eps)
        bulk = (w.unsqueeze(1) * vel).sum(0) / n                        # (3,)
        dv2 = ((vel - bulk.view(1, 3)) ** 2).sum(1)                     # (M,)
        sigma_v2 = (w * dv2).sum() / n
        sigma_v = (sigma_v2 + cfg.eps).sqrt()
        # Circular velocity squared at R from the enclosed soft mass; the virial
        # ratio against it is the dimensionless "how bound" coordinate. Both use
        # the same aperture R, so a global velocity-amplitude error cancels.
        gm_over_r = G_MPC_KMS2_PER_MSUN * mass / R
        virial_ratio = sigma_v2 / (gm_over_r + cfg.eps)
        out[f"log1p_overdensity_s{mult:g}"] = torch.log1p(overdensity)
        out[f"sigma_v_kms_s{mult:g}"] = sigma_v
        out[f"virial_ratio_s{mult:g}"] = virial_ratio
        # The raw margin is still reported for continuity, but nothing gates on
        # it -- it has no calibrated zero (see CandidateScoreConfig.virial_thresh).
        out[f"binding_margin_s{mult:g}"] = gm_over_r - sigma_v2
        masses.append(mass)
    # Concentration: innermost enclosed mass over outermost. Rises as material
    # moves from the envelope into the core even when the total is unchanged.
    out["concentration"] = masses[0] / masses[-1].clamp_min(cfg.eps)
    return out


def formation_score(
    pos: torch.Tensor,
    vel: torch.Tensor,
    center: torch.Tensor,
    r_ref: float,
    cfg: Optional[CandidateScoreConfig] = None,
    box: Optional[float] = None,
    *,
    use_binding: bool = True,
) -> torch.Tensor:
    """One differentiable scalar: how much this candidate looks like a bound clump.

    The mean over scales of the log-overdensity, each rung gated by a soft
    virial factor ``sigmoid((thresh - sigma_v^2/(GM/R)) / width)`` when
    ``use_binding`` -- so a dense but *unbound* clump (virial ratio >> 1) is
    discounted while a bound one (ratio O(1)) passes, which is exactly the
    density/velocity distinction the pairwise sum is blind to. Setting
    ``use_binding=False`` recovers the density-only score, the ablation the
    progress diagnostic uses to show the velocity terms carry their weight.
    """
    cfg = cfg or CandidateScoreConfig()
    f = candidate_features(pos, vel, center, r_ref, cfg, box)
    terms = []
    for mult in cfg.scale_mults:
        od = f[f"log1p_overdensity_s{mult:g}"]
        if use_binding:
            vr = f[f"virial_ratio_s{mult:g}"]
            gate = torch.sigmoid((cfg.virial_thresh - vr) / cfg.virial_width)
            terms.append(od * gate)
        else:
            terms.append(od)
    return torch.stack(terms).mean()


class SurvivalHead(torch.nn.Module):
    """Small MLP mapping candidate features to ``p_c in [0, 1]``.

    Fractional existence, summed (capped) into the ``N, H, S`` count statistics
    the reward is built on -- so one enormous dense candidate cannot dominate
    through an ``n_c^2`` pair count, which is the whole reason the note routes
    the score through a capped head rather than summing kernels directly. Left
    untrained here: the progress diagnostic validates the *features* first;
    fitting the head against real HR subhalo existence is the following step.
    """

    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features)).squeeze(-1)
