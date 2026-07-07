"""Voxel and field-statistic metrics for evaluation."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .spectra import _to_numpy, cross_correlation_coefficient


def voxel_mse(a, b) -> float:
    a = _to_numpy(a).astype(np.float64)
    b = _to_numpy(b).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"voxel_mse shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def relative_mse(a, b) -> float:
    """MSE(a, b) normalized by the mean square of ``b`` (the reference)."""
    a = _to_numpy(a).astype(np.float64)
    b = _to_numpy(b).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"relative_mse shape mismatch: {a.shape} vs {b.shape}")
    denom = float(np.mean(b ** 2))
    num = float(np.mean((a - b) ** 2))
    if denom == 0:
        return num
    return num / denom


def lr_reconstruction_mse(degrader, x_hat_hr, y_lr) -> float:
    """``mse(A(x_hat_hr), y_lr)`` computed with torch degrader on numpy/torch input."""
    import torch

    x = x_hat_hr
    if not torch.is_tensor(x):
        x = torch.from_numpy(_to_numpy(x)).float()
    if x.dim() == 4:
        x = x.unsqueeze(0)
    with torch.no_grad():
        y_recon = degrader(x)
    y = y_lr
    if not torch.is_tensor(y):
        y = torch.from_numpy(_to_numpy(y)).float()
    if y.dim() == 4:
        y = y.unsqueeze(0)
    return voxel_mse(y_recon, y)


def channel_mean_std(field) -> Dict[str, list]:
    """Per-channel mean and std of a ``(C, N, N, N)`` field."""
    field = _to_numpy(field).astype(np.float64)
    if field.ndim != 4:
        raise ValueError(f"channel_mean_std expects (C, N, N, N), got {field.shape}")
    means = field.mean(axis=(1, 2, 3))
    stds = field.std(axis=(1, 2, 3))
    return {"mean": means.tolist(), "std": stds.tolist()}


def compute_metrics(
    degrader,
    x_hat_hr,
    y_lr,
    x_hr: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute the full metric dict for a single ``(C, N, N, N)`` generated field.

    Always computes LR reconstruction MSE. If ``x_hr`` (ground truth) is given,
    also computes HR MSE, relative MSE and per-channel cross-correlation ``r(k)``
    summary.
    """
    x_hat_np = _to_numpy(x_hat_hr)
    metrics: Dict[str, Any] = {}
    metrics["lr_recon_mse"] = lr_reconstruction_mse(degrader, x_hat_hr, y_lr)
    metrics["gen_channel_stats"] = channel_mean_std(x_hat_np)

    if x_hr is not None:
        x_hr_np = _to_numpy(x_hr)
        metrics["hr_mse"] = voxel_mse(x_hat_np, x_hr_np)
        metrics["hr_relative_mse"] = relative_mse(x_hat_np, x_hr_np)
        metrics["hr_channel_stats"] = channel_mean_std(x_hr_np)
        # mean cross-correlation coefficient per channel (averaged over k)
        r_means = []
        for c in range(x_hat_np.shape[0]):
            _, r = cross_correlation_coefficient(x_hat_np[c], x_hr_np[c])
            r_means.append(float(np.mean(r)) if r.size else float("nan"))
        metrics["hr_cross_corr_mean"] = r_means
    return metrics
