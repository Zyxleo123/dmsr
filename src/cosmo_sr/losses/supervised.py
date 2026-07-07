"""Supervised paired loss: ``mse(G(y_lr_paired), x_hr_paired)``."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor, kind: str = "mse", huber_delta: float = 1.0
) -> torch.Tensor:
    """Reconstruction loss supporting ``"mse"`` and ``"huber"``."""
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch in reconstruction loss: {tuple(pred.shape)} vs "
            f"{tuple(target.shape)}"
        )
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "huber":
        return F.huber_loss(pred, target, delta=float(huber_delta))
    raise ValueError(f"Unknown reconstruction loss kind {kind!r} (use 'mse' or 'huber')")


def supervised_loss(
    x_hat_hr: torch.Tensor, x_hr: torch.Tensor, kind: str = "mse", huber_delta: float = 1.0
) -> torch.Tensor:
    """Supervised loss between generated HR and target HR (MSE or Huber)."""
    return reconstruction_loss(x_hat_hr, x_hr, kind=kind, huber_delta=huber_delta)
