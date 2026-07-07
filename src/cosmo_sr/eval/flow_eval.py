"""Evaluation for the residual null-space flow cascade.

Metrics (in increasing cost/order):

1. :func:`consistency_error`         -- ``A_R(x_hat) vs y_R`` at each octave.
2. :func:`highk_power_ratio`         -- high-k power recovery vs ground truth.
3. :func:`residual_power_per_octave` -- null-space residual power per octave.
4. :func:`z_diversity`               -- variability across noise draws for fixed ``y``.

:func:`evaluate_cascade` orchestrates 1-4 given a model and a ground-truth
pyramid. SR2-style summary statistics (power-spectrum recovery, cross-correlation)
are included via :func:`sr2_power_summary`; heavier SR2 statistics (bispectrum,
halo/subhalo abundance) are provided as optional hooks in :mod:`sr2_stats`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .spectra import power_spectrum, cross_correlation_coefficient
from ..operators.multiscale import MultiScaleOperators
from ..operators.base_upscaler import BaseUpscaler, consistent_base
from ..inference.flow_sample import sample_step


def _as5d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x.unsqueeze(0)
    if x.dim() != 5:
        raise ValueError(f"expected 4D or 5D tensor, got {tuple(x.shape)}")
    return x


def consistency_error(
    ops: MultiScaleOperators, x_2R: torch.Tensor, y_R: torch.Tensor
) -> Dict[str, float]:
    """LR-consistency of a generated fine field: ``A_R(x_2R)`` vs ``y_R``."""
    x_2R = _as5d(x_2R)
    y_R = _as5d(y_R)
    with torch.no_grad():
        a = ops.A(x_2R)
        mse = float(torch.mean((a - y_R) ** 2))
        denom = float(torch.mean(y_R ** 2))
    return {"consistency_mse": mse, "consistency_rel": mse / denom if denom > 0 else mse}


def mean_power_spectrum(field: torch.Tensor):
    """Channel-averaged isotropic power spectrum of a ``(C, N, N, N)`` field."""
    field = _as5d(field)[0]
    ks, pks = None, []
    for c in range(field.shape[0]):
        k, pk = power_spectrum(field[c])
        ks = k
        pks.append(pk)
    return ks, np.mean(np.stack(pks, axis=0), axis=0)


def highk_power_ratio(
    gen: torch.Tensor, true: torch.Tensor, frac: float = 0.5
) -> Dict[str, float]:
    """Mean ``P_gen/P_true`` over the top ``frac`` of k-bins (high-k recovery)."""
    kg, pg = mean_power_spectrum(gen)
    kt, pt = mean_power_spectrum(true)
    n = min(len(pg), len(pt))
    if n == 0:
        return {"highk_power_ratio": float("nan"), "allk_power_ratio": float("nan")}
    lo = int((1.0 - frac) * n)
    ratio = pg[:n] / np.clip(pt[:n], 1e-30, None)
    return {
        "highk_power_ratio": float(np.mean(ratio[lo:])),
        "allk_power_ratio": float(np.mean(ratio)),
    }


def residual_power_per_octave(
    ops: MultiScaleOperators,
    base_upscaler: BaseUpscaler,
    x_R: torch.Tensor,
    x_2R: torch.Tensor,
):
    """Isotropic power spectrum of the null-space residual ``P_null(x_2R - B_cons)``."""
    x_R = _as5d(x_R)
    x_2R = _as5d(x_2R)
    with torch.no_grad():
        r = ops.P_null(x_2R - consistent_base(base_upscaler, ops, x_R))
    k, pk = mean_power_spectrum(r[0])
    return k, pk


@torch.no_grad()
def z_diversity(
    model: torch.nn.Module,
    ops: MultiScaleOperators,
    base_upscaler: BaseUpscaler,
    y_R: torch.Tensor,
    R: float,
    n_samples: int = 4,
    n_steps: int = 20,
) -> Dict[str, float]:
    """Variability of generated fine fields across noise draws for a fixed ``y``."""
    y_R = _as5d(y_R)
    samples = [
        sample_step(model, ops, base_upscaler, y_R, R, n_steps=n_steps)
        for _ in range(n_samples)
    ]
    stack = torch.stack(samples, dim=0)  # (S, 1, C, N, N, N)
    per_voxel_std = stack.std(dim=0)
    signal_std = stack.mean(dim=0).std()
    rel = float(per_voxel_std.mean() / signal_std.clamp_min(1e-12))
    return {
        "z_voxel_std_mean": float(per_voxel_std.mean()),
        "z_rel_diversity": rel,
        "n_samples": n_samples,
    }


@torch.no_grad()
def evaluate_cascade(
    model: torch.nn.Module,
    ops: MultiScaleOperators,
    base_upscaler: BaseUpscaler,
    pyramid: Dict[int, torch.Tensor],
    resolutions=(64, 128, 256),
    n_steps: int = 20,
    diversity_samples: int = 4,
) -> Dict[str, Any]:
    """Run the full cascade from the coarsest level and score each octave.

    ``pyramid`` maps resolution label -> ground-truth field (``(C,N,N,N)`` or 5D).
    Generation starts from ``pyramid[min(resolutions)]`` and proceeds upward.
    """
    model.eval()
    out: Dict[str, Any] = {"octaves": {}}
    y = _as5d(pyramid[min(resolutions)])
    for R in resolutions:
        x_hat = sample_step(model, ops, base_upscaler, y, float(R), n_steps=n_steps)
        rec: Dict[str, Any] = {}
        rec.update(consistency_error(ops, x_hat, y))
        true_2R = pyramid.get(2 * R)
        if true_2R is not None:
            true_2R = _as5d(true_2R)
            rec.update(highk_power_ratio(x_hat, true_2R))
            k, pk_gen = residual_power_per_octave(ops, base_upscaler, y, x_hat)
            _, pk_true = residual_power_per_octave(ops, base_upscaler, y, true_2R)
            rec["residual_power_gen"] = pk_gen.tolist()
            rec["residual_power_true"] = pk_true.tolist()
            rec["residual_k"] = k.tolist()
        rec.update(
            z_diversity(model, ops, base_upscaler, y, float(R),
                        n_samples=diversity_samples, n_steps=n_steps)
        )
        out["octaves"][int(2 * R)] = rec
        # cascade: feed generated level as next coarse input
        y = x_hat
    return out


def sr2_power_summary(gen: torch.Tensor, true: torch.Tensor) -> Dict[str, Any]:
    """SR2-style summary: power-spectrum recovery + mean cross-correlation r(k)."""
    kg, pg = mean_power_spectrum(gen)
    kt, pt = mean_power_spectrum(true)
    n = min(len(pg), len(pt))
    gen4 = _as5d(gen)[0]
    true4 = _as5d(true)[0]
    r_means: List[float] = []
    for c in range(gen4.shape[0]):
        _, r = cross_correlation_coefficient(gen4[c], true4[c])
        r_means.append(float(np.mean(r)) if r.size else float("nan"))
    return {
        "k": kt[:n].tolist(),
        "power_ratio": (pg[:n] / np.clip(pt[:n], 1e-30, None)).tolist(),
        "cross_corr_mean_per_channel": r_means,
    }
