"""Sampling for the latent residual flow with classifier-free guidance (CFG).

Given a coarse field ``x_R`` at grid ``R``:

1. draw ``z ~ N(0, I)`` on the AE latent grid,
2. integrate ``dz/dt = v_cfg`` with forward Euler, where
   ``v_cfg = v_uncond + s * (v_cond - v_uncond)``,
3. decode, project onto the null space, and add the consistent base::

       base     = B_cons_R(x_R)
       r_hat    = P_null_R(ae.decode(z))
       x_2R_hat = base + r_hat            =>   A_R(x_2R_hat) = x_R exactly.

Hard LR consistency holds *by construction* regardless of the AE / flow quality.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ..operators.multiscale import MultiScaleOperators
from ..operators.base_upscaler import BaseUpscaler, consistent_base


def latent_shape_from_cond(
    ae: torch.nn.Module, ops: MultiScaleOperators, x_R: torch.Tensor
) -> Tuple[int, int, int, int, int]:
    """Latent tensor shape produced by encoding ``r_star`` at grid ``2R``."""
    b = x_R.shape[0]
    n2 = x_R.shape[-1] * ops.factor
    d = ae.downsample_factor
    if n2 % d != 0:
        raise ValueError(f"2R grid {n2} not divisible by AE downsample factor {d}")
    m = n2 // d
    return (b, ae.latent_channels, m, m, m)


def cfg_velocity(
    model: torch.nn.Module,
    z: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
    null_cond: torch.Tensor,
    R,
    cfg_scale: float,
) -> torch.Tensor:
    """``v_uncond + cfg_scale * (v_cond - v_uncond)``.

    ``cfg_scale=0`` returns the unconditional velocity; ``cfg_scale=1`` returns
    the conditional velocity.
    """
    v_uncond = model(z, t, null_cond, R)
    if cfg_scale == 0.0:
        return v_uncond
    v_cond = model(z, t, cond, R)
    return v_uncond + float(cfg_scale) * (v_cond - v_uncond)


def integrate_latent(
    model: torch.nn.Module,
    x_R: torch.Tensor,
    R: float,
    latent_shape: Tuple[int, ...],
    n_steps: int = 20,
    cfg_scale: float = 1.0,
    z: Optional[torch.Tensor] = None,
    null_cond: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Euler-integrate the latent flow ODE from ``t=0`` (noise) to ``t=1``."""
    device = x_R.device
    if z is None:
        z = torch.randn(*latent_shape, device=device, dtype=x_R.dtype)
    if null_cond is None:
        null_cond = torch.zeros_like(x_R)
    b = z.shape[0]
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((b,), i * dt, device=device, dtype=z.dtype)
        v = cfg_velocity(model, z, t, x_R, null_cond, R, cfg_scale)
        z = z + dt * v
    return z


def sample_latent_step(
    model: torch.nn.Module,
    ae: torch.nn.Module,
    ops: MultiScaleOperators,
    base_upscaler: BaseUpscaler,
    x_R: torch.Tensor,
    R: float,
    n_steps: int = 20,
    cfg_scale: float = 1.0,
    z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One octave ``x_R (grid R) -> x_2R (grid 2R)`` with hard LR consistency."""
    latent_shape = latent_shape_from_cond(ae, ops, x_R)
    z1 = integrate_latent(
        model, x_R, float(R), latent_shape, n_steps=n_steps, cfg_scale=cfg_scale, z=z
    )
    r_hat = ops.P_null(ae.decode(z1))
    base = consistent_base(base_upscaler, ops, x_R)
    x_2R = base + r_hat
    return x_2R


@torch.no_grad()
def super_resolve_latent_cascade(
    model: torch.nn.Module,
    ae: torch.nn.Module,
    ops: MultiScaleOperators,
    base_upscaler: BaseUpscaler,
    y_64: torch.Tensor,
    resolutions=(64, 128, 256),
    n_steps: int = 20,
    cfg_scale: float = 1.0,
    seed: Optional[int] = None,
) -> Dict[int, torch.Tensor]:
    """Cascade 64 -> 128 -> 256 -> 512 in latent space, returning each level."""
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    ae.eval()
    out: Dict[int, torch.Tensor] = {}
    cur = y_64
    for R in resolutions:
        cur = sample_latent_step(
            model, ae, ops, base_upscaler, cur, float(R),
            n_steps=n_steps, cfg_scale=cfg_scale,
        )
        out[2 * R] = cur
    return out
