"""Loss functions and the combined objective for our method.

``compute_losses`` assembles ambient LR-consistency, scarce paired-HR and
optional regularizers into a single dict::

    {
      "loss":         total (weighted) loss,
      "loss_ambient": ambient LR-consistency term (if unpaired batch given),
      "loss_pair":    supervised HR term      (if paired batch given),
      "loss_reg":     regularizer term        (if regularizers enabled),
    }
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .ambient import ambient_loss, compute_ambient
from .ambient_denoise import (
    clean_denoise_loss,
    ambient_denoise_loss,
    build_ambient_target,
)
from .supervised import supervised_loss
from .regularizers import (
    assert_finite,
    total_variation,
    finite_value_penalty,
    channel_mean_std_match,
    regularization_loss,
    regularizers_enabled,
)

__all__ = [
    "ambient_loss",
    "compute_ambient",
    "clean_denoise_loss",
    "ambient_denoise_loss",
    "build_ambient_target",
    "supervised_loss",
    "assert_finite",
    "total_variation",
    "finite_value_penalty",
    "channel_mean_std_match",
    "regularization_loss",
    "regularizers_enabled",
    "compute_losses",
]


def compute_losses(
    generator: nn.Module,
    degrader: nn.Module,
    *,
    y_lr_unpaired: Optional[torch.Tensor] = None,
    y_lr_paired: Optional[torch.Tensor] = None,
    x_hr_paired: Optional[torch.Tensor] = None,
    lambda_ambient: float = 1.0,
    lambda_pair: float = 1.0,
    lambda_reg: float = 1.0,
    reg_cfg: Optional[Dict[str, Any]] = None,
    raise_on_nonfinite: bool = True,
    recon: str = "mse",
    huber_delta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute the combined objective, returning a dict of scalar tensors.

    At least one of an unpaired or paired batch must be supplied (subject to the
    corresponding non-zero weight). Regularizers are applied to the concatenation
    of whatever generated HR is available.
    """
    lambda_ambient = float(lambda_ambient)
    lambda_pair = float(lambda_pair)
    lambda_reg = float(lambda_reg)

    out: Dict[str, torch.Tensor] = {}
    total = None
    hr_for_reg = []

    use_ambient = lambda_ambient != 0.0 and y_lr_unpaired is not None
    use_pair = lambda_pair != 0.0 and y_lr_paired is not None and x_hr_paired is not None

    if lambda_ambient != 0.0 and y_lr_unpaired is None:
        raise ValueError("lambda_ambient != 0 but no unpaired LR batch was provided.")
    if lambda_pair != 0.0 and (y_lr_paired is None or x_hr_paired is None):
        raise ValueError("lambda_pair != 0 but no paired LR/HR batch was provided.")

    if use_ambient:
        la, x_hat_unpaired, _y_recon = compute_ambient(
            generator, degrader, y_lr_unpaired, kind=recon, huber_delta=huber_delta
        )
        out["loss_ambient"] = la
        total = lambda_ambient * la if total is None else total + lambda_ambient * la
        hr_for_reg.append(x_hat_unpaired)

    if use_pair:
        x_hat_paired = generator(y_lr_paired)
        lp = supervised_loss(x_hat_paired, x_hr_paired, kind=recon, huber_delta=huber_delta)
        out["loss_pair"] = lp
        total = lambda_pair * lp if total is None else total + lambda_pair * lp
        hr_for_reg.append(x_hat_paired)

    if total is None:
        raise ValueError(
            "No active loss terms: provide an unpaired and/or paired batch with "
            "non-zero weights."
        )

    if lambda_reg != 0.0 and regularizers_enabled(reg_cfg):
        x_hat_all = torch.cat(hr_for_reg, dim=0)
        reference = x_hr_paired if use_pair else None
        loss_reg, _components = regularization_loss(x_hat_all, reg_cfg, reference=reference)
        out["loss_reg"] = loss_reg
        total = total + lambda_reg * loss_reg

    out["loss"] = total

    if raise_on_nonfinite:
        for name, value in out.items():
            assert_finite(value, name)

    return out
