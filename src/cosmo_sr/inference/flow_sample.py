"""Sampling for the residual null-space flow: one octave step and 64->512 cascade.

At each octave we integrate the flow ODE ``dr/dt = v_phi(r, t, y_R, R)`` from
``t=0`` (``r = z ~ N(0, I)``) to ``t=1`` with forward Euler, project the result
onto the null space, then add the consistent base::

    x_2R = B_cons_R(y_R) + P_null_R(r(1))   =>   A_R(x_2R) = y_R exactly.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from ..operators.multiscale import MultiScaleOperators
from ..operators.base_upscaler import BaseUpscaler, consistent_base


def integrate_residual(
    model: torch.nn.Module,
    ops: MultiScaleOperators,
    y_R: torch.Tensor,
    R: float,
    n_steps: int = 20,
    z: Optional[torch.Tensor] = None,
    project_each_step: bool = False,
    bp_steps: Optional[int] = None,
) -> torch.Tensor:
    """Integrate the flow ODE and return a null-space residual at ``2R``.

    Grad flows through the integration (used by LR-only training). Wrap in
    ``torch.no_grad()`` for pure inference.

    ``bp_steps`` bounds the backprop-through-time depth: only the last
    ``bp_steps`` Euler steps keep gradients (earlier steps run under
    ``torch.no_grad()`` and are detached). This makes training memory
    independent of ``n_steps`` -- without it, gradient checkpoints for every
    step of the unrolled ODE stay resident simultaneously, which is the main
    memory driver for LR-only training at large volumes. ``None`` (default)
    keeps full backprop through all steps (previous behaviour).
    """
    b, c = y_R.shape[0], y_R.shape[1]
    n2 = y_R.shape[-1] * ops.factor
    if z is None:
        z = torch.randn(b, c, n2, n2, n2, device=y_R.device, dtype=y_R.dtype)
    r = z
    dt = 1.0 / n_steps
    cutoff = 0 if bp_steps is None else max(n_steps - bp_steps, 0)
    for i in range(n_steps):
        t = torch.full((b,), i * dt, device=y_R.device, dtype=y_R.dtype)
        if i < cutoff:
            with torch.no_grad():
                v = model(r, t, y_R, R)
                r = r + dt * v
                if project_each_step:
                    r = ops.P_null(r)
        else:
            v = model(r, t, y_R, R)
            r = r + dt * v
            if project_each_step:
                r = ops.P_null(r)
    return ops.P_null(r)


def sample_step(
    model: torch.nn.Module,
    ops: MultiScaleOperators,
    base_upscaler: BaseUpscaler,
    y_R: torch.Tensor,
    R: float,
    n_steps: int = 20,
    z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One octave: ``y_R (grid R) -> x_2R (grid 2R)`` with hard LR consistency."""
    base = consistent_base(base_upscaler, ops, y_R)
    r = integrate_residual(model, ops, y_R, R, n_steps=n_steps, z=z)
    return base + r


@torch.no_grad()
def super_resolve_cascade(
    model: torch.nn.Module,
    ops: MultiScaleOperators,
    base_upscaler: BaseUpscaler,
    y_64: torch.Tensor,
    resolutions=(64, 128, 256),
    n_steps: int = 20,
    seed: Optional[int] = None,
) -> Dict[int, torch.Tensor]:
    """Cascade 64 -> 128 -> 256 -> 512, returning each generated level.

    ``resolutions`` are the *coarse* octave labels fed to the shared model.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    out: Dict[int, torch.Tensor] = {}
    cur = y_64
    for R in resolutions:
        cur = sample_step(model, ops, base_upscaler, cur, float(R), n_steps=n_steps)
        out[2 * R] = cur
    return out
