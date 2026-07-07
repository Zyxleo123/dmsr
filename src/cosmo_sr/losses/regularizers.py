"""Optional weak, deterministic regularizers.

Implemented:
  * HR total variation (spatial smoothness)
  * output finite-value penalty (soft bound on magnitudes)
  * channel-wise mean/std matching to a reference

No stochastic / entropy regularization is included yet (deterministic first).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def assert_finite(value: torch.Tensor, name: str = "loss") -> torch.Tensor:
    """Raise ``ValueError`` if ``value`` contains NaN or Inf."""
    if not torch.isfinite(value).all():
        raise ValueError(f"Non-finite value encountered in {name}: {value}")
    return value


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Mean absolute spatial gradient of a ``(B, C, N, N, N)`` field."""
    if x.dim() != 5:
        raise ValueError(f"total_variation expects 5D input, got {tuple(x.shape)}")
    dx = (x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).abs().mean()
    dy = (x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).abs().mean()
    dz = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs().mean()
    return (dx + dy + dz) / 3.0


def finite_value_penalty(x: torch.Tensor, limit: float = 10.0) -> torch.Tensor:
    """Quadratic penalty on the amount by which ``|x|`` exceeds ``limit``.

    Keeps outputs from drifting to huge values without affecting the in-range
    signal.
    """
    excess = F.relu(x.abs() - float(limit))
    return (excess ** 2).mean()


def channel_mean_std_match(x_hat: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Match per-channel mean and std of ``x_hat`` to ``reference``.

    Both tensors are ``(B, C, ...)``; statistics are computed over all non-channel
    dimensions.
    """
    if x_hat.shape[1] != reference.shape[1]:
        raise ValueError("channel_mean_std_match requires matching channel counts")
    dims = (0,) + tuple(range(2, x_hat.dim()))
    mean_hat = x_hat.mean(dim=dims)
    std_hat = x_hat.std(dim=dims)
    mean_ref = reference.mean(dim=dims)
    std_ref = reference.std(dim=dims)
    return F.mse_loss(mean_hat, mean_ref) + F.mse_loss(std_hat, std_ref)


def regularization_loss(
    x_hat: torch.Tensor,
    cfg: Optional[Dict[str, Any]] = None,
    reference: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Weighted sum of enabled regularizers.

    ``cfg`` keys (all optional, default 0 / disabled):
      * ``lambda_tv``: weight of total variation
      * ``lambda_finite`` + ``finite_limit``: weight/limit of finite-value penalty
      * ``lambda_meanstd``: weight of channel mean/std matching (needs ``reference``)

    Returns ``(loss_reg, components)``. ``loss_reg`` is a scalar tensor on the
    same device as ``x_hat`` (zero if nothing enabled).
    """
    cfg = cfg or {}
    components: Dict[str, torch.Tensor] = {}
    loss_reg = x_hat.new_zeros(())

    lambda_tv = float(cfg.get("lambda_tv", 0.0))
    if lambda_tv != 0.0:
        tv = total_variation(x_hat)
        components["tv"] = tv
        loss_reg = loss_reg + lambda_tv * tv

    lambda_finite = float(cfg.get("lambda_finite", 0.0))
    if lambda_finite != 0.0:
        fv = finite_value_penalty(x_hat, limit=float(cfg.get("finite_limit", 10.0)))
        components["finite"] = fv
        loss_reg = loss_reg + lambda_finite * fv

    lambda_meanstd = float(cfg.get("lambda_meanstd", 0.0))
    if lambda_meanstd != 0.0:
        if reference is None:
            raise ValueError("channel mean/std matching requires a reference tensor")
        ms = channel_mean_std_match(x_hat, reference)
        components["meanstd"] = ms
        loss_reg = loss_reg + lambda_meanstd * ms

    return loss_reg, components


def regularizers_enabled(cfg: Optional[Dict[str, Any]]) -> bool:
    if not cfg:
        return False
    return any(
        float(cfg.get(k, 0.0)) != 0.0
        for k in ("lambda_tv", "lambda_finite", "lambda_meanstd")
    )
