"""Offline reward-weighted residual diffusion on top of the frozen SR2 generator.

The frozen SR2 generator ``G_SR2`` reproduces global density statistics but loses
low-mass subhalos (~55% deficit; ~89% of HR subs in matched hosts are ``missing``
-- see ``docs/sr2_subhalo_study.md``). SR2's *own* noise diversity does not fix
this: Gate A on nested SR2 seeds returned ``fail_unhelpful_diversity``. So the
extra degrees of freedom have to come from somewhere else.

This package adds a **stochastic residual** on top of the frozen output,

    Psi_hat = Psi_base + a * dPsi,        dPsi ~ p_phi(. | y, Psi_base)

trains ``p_phi`` first by ordinary paired supervision, then reweights it offline
toward residual samples that a *non-differentiable* halo-catalog reward likes.
No policy gradients: the reward only ever selects and weights samples that the
model already produces, so the model never learns to exploit a differentiable
surrogate of Rockstar.

Layout
------
``paths``        $ZFS artifact locations (nothing bulky lands on home).
``geometry``     Lagrangian chunk grid, Eulerian purity grid, core mask.
``base``         Frozen SR2 wrapper + residual composition.
``diffusion``    Continuous-time VP cosine schedule, eps loss, DDIM sampler.
``model``        Conditional 3D residual denoiser (reuses ``Map2MapUNet3D``).
``targets``      Paired residual targets ``dPsi* = Psi_HR - Psi_base`` + cache.
``catalog``      Chunk-level catalog summaries (SHMF + occupation) and pooling.
``reward``       Ensemble Mahalanobis catalog reward against HR.
``constraints``  Field-fidelity feasibility filter.
``replay``       Per-chunk marginal contributions and the elite replay buffer.
``tiles``        Exact 64^3 credit from Rockstar member ids (Experiment 0).
``oracle_hr``    Targeted HR-residual oracle (Experiment 1).

``geometry`` and ``tiles`` are not two resolutions of one idea. ``geometry``
attributes a halo by an Eulerian purity test and *drops* it when the test fails,
which was measured to reject hosts in proportion to their mass. ``tiles``
attributes every object *fractionally* by its member particles' Lagrangian
origin, rejecting nothing. Both are live; see ``docs/catalog_reward_oracle.md``.
"""
from __future__ import annotations

__all__ = [
    "base",
    "catalog",
    "constraints",
    "diffusion",
    "geometry",
    "model",
    "oracle_hr",
    "paths",
    "replay",
    "reward",
    "targets",
    "tiles",
]
