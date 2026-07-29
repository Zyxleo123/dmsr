"""Continuous-time VP diffusion on the residual, with epsilon prediction.

The repository already has flow matching (``losses/flow.py``) and an
``x_0``-prediction ambient denoiser (``losses/ambient_denoise.py``), but the plan
calls for the ordinary denoising objective

    L_sup = E_{t, eps} || eps - eps_phi(dPsi_t, t, y, Psi_base) ||^2 ,

so this module supplies the missing epsilon-prediction pieces on top of the
existing :class:`~cosmo_sr.models.operator_denoiser.CosineSchedule`
(``alpha = cos(pi t / 2)``, ``sigma = sin(pi t / 2)``, ``t in [0, 1]``).

Residual scaling
----------------
The paired residual ``dPsi* = Psi_HR - Psi_base`` is small in catnorm units
(order 1e-2), so feeding it raw to a unit-variance diffusion would spend the
whole schedule on noise. Everything here therefore works in **whitened** units
``u = dPsi / sigma_res`` with a fixed per-channel ``sigma_res`` measured once by
the target audit and stored in the checkpoint. ``sigma_res`` is a constant, not
a learned parameter, so the mapping back to physical units is exact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

from ..models.operator_denoiser import CosineSchedule

__all__ = [
    "DiffusionConfig",
    "ddim_sample",
    "denoising_loss",
    "sample_timesteps",
    "whiten",
    "unwhiten",
]


@dataclass
class DiffusionConfig:
    """Sampling / loss knobs.

    ``x0_clip`` bounds the predicted clean residual *in whitened units*, where
    unit variance is the definition of the scale. Without it the first DDIM step
    divides by ``alpha(t_max) ~ 0``, so any model whose ``eps_hat`` is not yet
    accurate -- most of all the zero-initialised head, for which ``eps_hat = 0``
    exactly -- returns a residual larger than the field it corrects. Set to 0 to
    disable.
    """

    n_steps: int = 40
    t_min: float = 1e-3
    t_max: float = 0.999
    churn: float = 0.0            # 0 = deterministic DDIM, 1 = ancestral DDPM
    x0_clip: float = 4.0          # in units of sigma_res; 0 disables
    loss_weighting: str = "uniform"   # "uniform" | "snr" (min-SNR-gamma style)
    min_snr_gamma: float = 5.0


def whiten(x: torch.Tensor, sigma_res: torch.Tensor) -> torch.Tensor:
    """Physical residual -> unit-scale diffusion variable."""
    return x / sigma_res.to(x.device, x.dtype).view(1, -1, 1, 1, 1)


def unwhiten(u: torch.Tensor, sigma_res: torch.Tensor) -> torch.Tensor:
    """Diffusion variable -> physical residual (exact inverse of :func:`whiten`)."""
    return u * sigma_res.to(u.device, u.dtype).view(1, -1, 1, 1, 1)


def sample_timesteps(
    batch: int,
    cfg: DiffusionConfig,
    device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Uniform ``t`` in ``[t_min, t_max]``; one per batch element."""
    u = torch.rand(batch, device=device, generator=generator)
    return cfg.t_min + (cfg.t_max - cfg.t_min) * u


def _loss_weight(t: torch.Tensor, cfg: DiffusionConfig) -> torch.Tensor:
    if cfg.loss_weighting == "uniform":
        return torch.ones_like(t)
    if cfg.loss_weighting == "snr":
        sched = CosineSchedule()
        a, s = sched.alpha_sigma(t)
        snr = (a / s.clamp_min(1e-8)) ** 2
        return torch.clamp(snr, max=cfg.min_snr_gamma) / snr.clamp_min(1e-8)
    raise ValueError(f"unknown loss_weighting {cfg.loss_weighting!r}")


def denoising_loss(
    model,
    u0: torch.Tensor,
    cond: dict,
    cfg: DiffusionConfig,
    *,
    generator: Optional[torch.Generator] = None,
    per_sample_weight: Optional[torch.Tensor] = None,
    reduce: bool = True,
):
    """``|| eps - eps_phi(u_t, t, cond) ||^2`` on whitened residuals ``u0``.

    ``per_sample_weight`` is the reward weight ``w_i`` of section 8; it multiplies
    each element's mean squared error before the batch mean, so a batch mixing
    supervised and elite examples can be scored in one pass.

    Returns ``(loss, aux)`` where ``aux`` carries ``t``, ``eps``, ``eps_hat``.
    """
    if u0.dim() != 5:
        raise ValueError(f"u0 must be (B, C, D, H, W), got {tuple(u0.shape)}")
    b = u0.shape[0]
    sched = CosineSchedule()
    t = sample_timesteps(b, cfg, u0.device, generator)
    a, s = sched.broadcast(t, ndim=5)
    eps = torch.randn(u0.shape, device=u0.device, dtype=u0.dtype, generator=generator)
    u_t = a * u0 + s * eps
    eps_hat = model(u_t, t, **cond)
    per_elem = (eps_hat - eps) ** 2
    per_sample = per_elem.flatten(1).mean(dim=1) * _loss_weight(t, cfg)
    if per_sample_weight is not None:
        per_sample = per_sample * per_sample_weight.reshape(-1).to(per_sample.dtype)
    loss = per_sample.mean() if reduce else per_sample
    return loss, {"t": t, "eps": eps, "eps_hat": eps_hat, "u_t": u_t}


def _clip_record(u0: torch.Tensor, x0_clip: float, step: int, t: float) -> dict:
    """Clip statistics for one DDIM step, in standardized residual units.

    ``u0`` is already whitened, so 1.0 is one ``sigma_res`` and the numbers are
    comparable across channels, resolutions and runs.
    """
    hit = (u0.abs() >= x0_clip)
    return {
        "step": int(step),
        "t": float(t),
        "x0_clip": float(x0_clip),
        "clip_fraction": float(hit.float().mean().item()),
        "clip_fraction_per_channel": [
            float(v) for v in hit.float().mean(dim=(0, 2, 3, 4)).tolist()
        ] if u0.ndim == 5 else [],
        "abs_max_sigma": float(u0.abs().max().item()),
        "rms_sigma": float(u0.float().pow(2).mean().sqrt().item()),
    }


@torch.no_grad()
def ddim_sample(
    model,
    shape: Sequence[int],
    cond: dict,
    cfg: DiffusionConfig,
    *,
    device,
    dtype=torch.float32,
    generator: Optional[torch.Generator] = None,
    tile_fn: Optional[Callable] = None,
    clip_log: Optional[list] = None,
) -> torch.Tensor:
    """Deterministic DDIM (``churn=0``) or ancestral (``churn>0``) sampling.

    ``tile_fn`` lets a caller replace the plain ``model(u_t, t, **cond)`` call
    with a tiled / blended evaluation over a volume too large for one forward
    pass (see :func:`cosmo_sr.reward.sampling.multidiffusion_eps`); it receives
    ``(u_t, t)`` and returns ``eps_hat`` of the same shape.

    ``clip_log``, if given, is appended one dict per step recording the fraction
    of voxels that hit ``x0_clip``. This is not diagnostics for its own sake: a
    zero-initialised head returns ``eps_hat = 0``, for which ``u_0 = u_t/alpha``
    is the *initial noise rescaled* rather than zero, and only the clip keeps the
    first steps finite. A run whose clip fraction is not small is one where the
    clip -- not the model -- is setting the residual amplitude, and the samples
    are artifacts of the bound.
    """
    sched = CosineSchedule()
    u = torch.randn(tuple(shape), device=device, dtype=dtype, generator=generator)
    ts = torch.linspace(cfg.t_max, cfg.t_min, int(cfg.n_steps) + 1, device=device)
    for i in range(int(cfg.n_steps)):
        t_cur = ts[i].expand(shape[0])
        t_nxt = ts[i + 1].expand(shape[0])
        a_c, s_c = sched.broadcast(t_cur, ndim=5)
        a_n, s_n = sched.broadcast(t_nxt, ndim=5)
        eps_hat = tile_fn(u, t_cur) if tile_fn is not None else model(u, t_cur, **cond)
        u0_hat = (u - s_c * eps_hat) / a_c.clamp_min(1e-8)
        if cfg.x0_clip > 0:
            if clip_log is not None:
                clip_log.append(_clip_record(u0_hat, float(cfg.x0_clip),
                                             int(i), float(ts[i])))
            u0_hat = u0_hat.clamp(-cfg.x0_clip, cfg.x0_clip)
        if cfg.churn <= 0.0:
            u = a_n * u0_hat + s_n * eps_hat
        else:
            # Ancestral step: replace a `churn` fraction of the carried noise.
            keep = math.sqrt(max(0.0, 1.0 - cfg.churn ** 2))
            fresh = torch.randn(u.shape, device=device, dtype=dtype, generator=generator)
            u = a_n * u0_hat + s_n * (keep * eps_hat + cfg.churn * fresh)
    return u
