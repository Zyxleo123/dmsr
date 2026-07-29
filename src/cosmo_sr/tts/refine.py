"""Stage 4: refining a candidate's injected noise, with SR2 and the verifier frozen.

Best-of-K only ever reaches the best of ``K`` random draws. Refinement asks for
more: hold the generator and the scorer fixed and move the *noise* itself
downhill. The optimisation variables are the six noise tensors of
:class:`cosmo_sr.tts.srs_noise.ControlledG`, nothing else.

Two things keep this honest:

**Coarse-to-fine.** Noise sites act on very different scales (``z0`` at the LR
resolution, ``z5`` at the HR resolution). Unlocking them all at once lets the
optimiser spend its budget on the ~10^8 fine-scale variables, which is where
verifier exploitation is easiest and where the physical effect is smallest.
Phases unlock ``coarse`` -> ``+middle`` -> ``+fine`` with a shrinking step, so
large-scale structure is settled before detail is touched.

**A prior term.** The generator was trained with ``z ~ N(0, 1)``; an unconstrained
optimiser will happily leave that distribution and produce a field that scores
well and looks wrong. Every step therefore pays

.. math::
    L_{noise} = \\lambda_\\mu \\mu(z)^2
              + \\lambda_\\sigma (\\sigma(z) - 1)^2
              + \\lambda_2 \\frac{\\lVert z - z_0 \\rVert_2^2}{N}

per site, with ``z_0`` the originally sampled noise. The trust-region term is
normalised by the element count so the same ``lambda_2`` means the same thing at
every site despite their 512x size difference.

A gradient-free cross-entropy-method refiner is provided as a control: if CEM
matches gradient refinement, the gain is from search, not from gradients; if
gradient refinement wins by a lot while its noise statistics drift, that is
verifier exploitation, not quality.

Scope note: refinement runs **per tile**. A full 512^3 box is 512 tiles, and
backpropagating through all of them at once is not memory-feasible; the score
functions used here are local statistics that are well defined on a tile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .srs_noise import NOISE_SITES, STAGE_NAMES, STAGE_SITES

__all__ = [
    "LinearFeatureObjective",
    "NoiseRegularizer",
    "RefinementResult",
    "cem_refine_tile",
    "default_schedule",
    "noise_statistics",
    "refine_tile_noise",
]


class LinearFeatureObjective:
    """A differentiable verifier score built from the summary features.

    The trained :class:`cosmo_sr.tts.verifier.FeatureRanker` reads features that
    include histograms, whose gradient is identically zero. This wraps the
    *linear* part of a ranker -- weights over the differentiable feature subset,
    applied to the standardised features -- so the same ordering can be
    optimised, not just evaluated.

    Restricting to a linear head is not a limitation in practice: the ranker is
    linear or a one-hidden-layer MLP by design (there are only a handful of
    independent boxes), and a linear surrogate is exactly what makes the
    refinement objective interpretable when it has to be audited for
    exploitation.
    """

    def __init__(self, weights: Dict[str, float], mean: Dict[str, float],
                 std: Dict[str, float], feature_fn: Callable[[torch.Tensor], Dict[str, torch.Tensor]]):
        self.weights = dict(weights)
        self.mean = dict(mean)
        self.std = dict(std)
        self.feature_fn = feature_fn

    def __call__(self, sr: torch.Tensor) -> torch.Tensor:
        feats = self.feature_fn(sr)
        total = None
        for key, w in self.weights.items():
            if key not in feats:
                continue
            z = (feats[key] - self.mean.get(key, 0.0)) / max(self.std.get(key, 1.0), 1e-12)
            term = w * z
            total = term if total is None else total + term
        if total is None:
            raise ValueError("no differentiable features overlapped the verifier weights")
        return total


@dataclass
class NoiseRegularizer:
    """Keeps optimised noise close to the distribution the generator was trained on."""

    lam_mu: float = 1.0
    lam_sigma: float = 1.0
    lam_l2: float = 0.1

    def __call__(self, z: Dict[str, torch.Tensor],
                 z0: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for site, t in z.items():
            mu = t.mean()
            sd = t.std()
            l2 = (t - z0[site]).pow(2).mean()      # ||.||^2 / N
            term = self.lam_mu * mu.pow(2) + self.lam_sigma * (sd - 1.0).pow(2) + self.lam_l2 * l2
            total = term if total is None else total + term
        if total is None:
            raise ValueError("no noise tensors to regularise")
        return total


@dataclass
class Phase:
    """One coarse-to-fine phase: which stages move, for how long, how fast."""

    stages: Tuple[str, ...]
    steps: int
    lr: float


def default_schedule(steps: int = 60, lr: float = 0.05) -> List[Phase]:
    """``coarse`` -> ``coarse+middle`` -> everything, with a decaying step size."""
    return [
        Phase(("coarse",), steps, lr),
        Phase(("coarse", "middle"), steps, lr * 0.5),
        Phase(("coarse", "middle", "fine"), max(1, steps // 2), lr * 0.2),
    ]


def _sites_for(stages: Sequence[str]) -> List[str]:
    return [s for stage in stages for s in STAGE_SITES[stage]]


def noise_statistics(z: Dict[str, torch.Tensor],
                     z0: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
    """Per-site mean/std and distance from the original draw, for the audit trail."""
    out: Dict[str, float] = {}
    for site in NOISE_SITES:
        if site not in z:
            continue
        t = z[site].detach().float()
        out[f"{site}_mean"] = float(t.mean())
        out[f"{site}_std"] = float(t.std())
        if z0 is not None and site in z0:
            d = (t - z0[site].detach().float())
            out[f"{site}_dist"] = float(d.pow(2).mean().sqrt())
    means = [abs(v) for k, v in out.items() if k.endswith("_mean")]
    sds = [abs(v - 1.0) for k, v in out.items() if k.endswith("_std")]
    dists = [v for k, v in out.items() if k.endswith("_dist")]
    if means:
        out["max_absmean"] = float(max(means))
        out["max_sigma_dev"] = float(max(sds))
    if dists:
        out["max_dist"] = float(max(dists))
    return out


@dataclass
class RefinementResult:
    """Refined noise plus everything needed to judge whether to trust it."""

    noise: Dict[str, torch.Tensor]
    score: float
    initial_score: float
    trajectory: List[float] = field(default_factory=list)
    reg_trajectory: List[float] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)
    rejected: bool = False
    reject_reason: str = ""

    @property
    def improvement(self) -> float:
        return float(self.initial_score - self.score)


def _check_out_of_distribution(
    stats: Dict[str, float], max_absmean: float, max_sigma_dev: float
) -> str:
    if stats.get("max_absmean", 0.0) > max_absmean:
        return f"noise mean drifted to {stats['max_absmean']:.3f} (limit {max_absmean})"
    if stats.get("max_sigma_dev", 0.0) > max_sigma_dev:
        return f"noise std drifted by {stats['max_sigma_dev']:.3f} (limit {max_sigma_dev})"
    return ""


def refine_tile_noise(
    generator,
    x: torch.Tensor,
    z0: Dict[str, torch.Tensor],
    objective: Callable[[torch.Tensor], torch.Tensor],
    schedule: Optional[Sequence[Phase]] = None,
    regularizer: Optional[NoiseRegularizer] = None,
    max_absmean: float = 0.15,
    max_sigma_dev: float = 0.25,
    verbose: bool = False,
) -> RefinementResult:
    """Gradient refinement of one tile's noise. ``objective`` returns a scalar to minimise.

    ``objective`` must be differentiable in the generator output -- a
    :class:`cosmo_sr.tts.verifier.PatchVerifier` score, or a differentiable
    summary statistic. Feature rankers that read histogram-based features are not
    differentiable end to end; use :func:`cem_refine_tile` for those.

    The result is marked ``rejected`` (rather than silently returned) when the
    optimised noise leaves the training distribution -- a refined sample that
    scores well only because its noise is out of distribution is not a better
    sample, it is a broken one.
    """
    schedule = list(schedule or default_schedule())
    regularizer = regularizer or NoiseRegularizer()
    z = {k: v.detach().clone() for k, v in z0.items()}
    z_orig = {k: v.detach().clone() for k, v in z0.items()}

    with torch.no_grad():
        initial = float(objective(generator(x, noise=z)))

    traj: List[float] = [initial]
    reg_traj: List[float] = []
    for phase in schedule:
        sites = _sites_for(phase.stages)
        for s in z:
            z[s] = z[s].detach()
            z[s].requires_grad_(s in sites)
        opt = torch.optim.Adam([z[s] for s in sites], lr=phase.lr)
        for _ in range(int(phase.steps)):
            opt.zero_grad(set_to_none=True)
            y = generator(x, noise=z)
            obj = objective(y)
            reg = regularizer(z, z_orig)
            (obj + reg).backward()
            opt.step()
            traj.append(float(obj.detach()))
            reg_traj.append(float(reg.detach()))
        if verbose:
            print(f"  phase {phase.stages}: score {traj[-1]:.5f}", flush=True)

    z = {k: v.detach() for k, v in z.items()}
    with torch.no_grad():
        final = float(objective(generator(x, noise=z)))
    stats = noise_statistics(z, z_orig)
    reason = _check_out_of_distribution(stats, max_absmean, max_sigma_dev)
    return RefinementResult(noise=z, score=final, initial_score=initial, trajectory=traj,
                            reg_trajectory=reg_traj, stats=stats,
                            rejected=bool(reason), reject_reason=reason)


def cem_refine_tile(
    generator,
    x: torch.Tensor,
    z0: Dict[str, torch.Tensor],
    objective: Callable[[torch.Tensor], float],
    stages: Sequence[str] = STAGE_NAMES,
    iterations: int = 8,
    population: int = 16,
    elite_frac: float = 0.25,
    sigma: float = 0.3,
    sigma_decay: float = 0.85,
    regularizer: Optional[NoiseRegularizer] = None,
    seed: int = 0,
    max_absmean: float = 0.15,
    max_sigma_dev: float = 0.25,
) -> RefinementResult:
    """Gradient-free control: cross-entropy method over perturbations of ``z0``.

    Each iteration perturbs the *mean* noise by ``sigma * eps``, keeps the elite
    fraction and recentres. Uses only forward passes, so it accepts any scoring
    function -- including the feature ranker, which is not differentiable.

    Its role is diagnostic: if CEM matches gradient refinement, the improvement
    comes from search over the noise distribution, not from exploiting the
    verifier's gradients.
    """
    regularizer = regularizer or NoiseRegularizer()
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    sites = _sites_for(stages)
    mean = {k: v.detach().clone() for k, v in z0.items()}
    z_orig = {k: v.detach().clone() for k, v in z0.items()}

    def _score(z: Dict[str, torch.Tensor]) -> float:
        with torch.no_grad():
            return float(objective(generator(x, noise=z))) + float(regularizer(z, z_orig))

    initial = _score(mean)
    best_z, best = {k: v.clone() for k, v in mean.items()}, initial
    traj = [initial]
    n_elite = max(1, int(population * elite_frac))
    for _ in range(int(iterations)):
        samples, scores = [], []
        for _p in range(int(population)):
            z = {k: v.clone() for k, v in mean.items()}
            for s in sites:
                eps = torch.randn(z[s].shape, generator=g, dtype=torch.float32)
                z[s] = z[s] + sigma * eps.to(z[s].device, z[s].dtype)
            samples.append(z)
            scores.append(_score(z))
        order = np.argsort(scores)[:n_elite]
        mean = {
            s: torch.stack([samples[i][s] for i in order]).mean(dim=0) for s in mean
        }
        m_score = _score(mean)
        traj.append(m_score)
        if m_score < best:
            best, best_z = m_score, {k: v.clone() for k, v in mean.items()}
        sigma *= sigma_decay

    stats = noise_statistics(best_z, z_orig)
    reason = _check_out_of_distribution(stats, max_absmean, max_sigma_dev)
    return RefinementResult(noise=best_z, score=best, initial_score=initial, trajectory=traj,
                            stats=stats, rejected=bool(reason), reject_reason=reason)
