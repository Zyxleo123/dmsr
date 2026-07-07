"""Ambient LR-consistency loss.

Given a generator ``G`` and fixed degrader ``A``::

    x_hat_hr = G(y_lr)
    y_recon  = A(x_hat_hr)
    loss_ambient = mse(y_recon, y_lr)

This lets us learn from LR-only ("ambient") data by requiring that degrading the
generated HR reproduces the observed LR.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def ambient_loss(
    y_recon: torch.Tensor, y_lr: torch.Tensor, kind: str = "mse", huber_delta: float = 1.0
) -> torch.Tensor:
    """Reconstruction error between degraded-generated LR and observed LR."""
    if y_recon.shape != y_lr.shape:
        raise ValueError(
            f"Shape mismatch in ambient loss: {tuple(y_recon.shape)} vs "
            f"{tuple(y_lr.shape)}. Check scale_factor / crop alignment."
        )
    if kind == "mse":
        return F.mse_loss(y_recon, y_lr)
    if kind == "huber":
        return F.huber_loss(y_recon, y_lr, delta=float(huber_delta))
    raise ValueError(f"Unknown ambient loss kind {kind!r} (use 'mse' or 'huber')")


def compute_ambient(
    generator: nn.Module, degrader: nn.Module, y_lr: torch.Tensor,
    kind: str = "mse", huber_delta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the ambient forward pass.

    Returns ``(loss_ambient, x_hat_hr, y_recon)``.
    """
    x_hat_hr = generator(y_lr)
    y_recon = degrader(x_hat_hr)
    loss = ambient_loss(y_recon, y_lr, kind=kind, huber_delta=huber_delta)
    return loss, x_hat_hr, y_recon
