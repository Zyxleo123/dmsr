"""DMSR stage: exact-consistent stochastic null-space flow + HR-space critic.

This subpackage implements the single-operator (``s=8``, 64^3 -> 512^3) stage:

    x_hat = A_plus(y) + P_A(r_theta(y, z))

so that ``A(x_hat) = y`` holds *by construction*, and the only learned degrees of
freedom live in ``ker A`` -- the HR detail the LR simulation cannot see.

Deliberately **not** implemented here (excluded by design, see ``docs/dmsr_stage.md``):
pseudo-HR pairs, learned-degradation consistency, cycle consistency, virtual
shifted measurements, posterior distillation.

Module map::

    operator.py  A / A_plus / P_A and the exact-consistency combine
    cubic.py     the 24 orientation-preserving cube rotations (voxels + vectors)
    density.py   differentiable CIC density and the high-pass density channel
    encoder.py   the LR condition encoder (separately pretrainable)
    flow.py      null-space rectified flow: generator, loss, ODE sampler
    critic.py    spectral-norm 3D PatchGAN on (residual, rho_high), hinge + R1
    env.py       environment descriptors, balanced sampler, source-classifier AUC
    data.py      box-level-split paired and LR-only crop datasets
    ssl.py       masked 3D reconstruction pretraining for the condition encoder
    evaluate.py  held-out metrics: r(k), T(k), PDFs, bispectra, diversity, ...
"""
from __future__ import annotations

from .operator import NullSpaceOperator

__all__ = ["NullSpaceOperator"]
