"""Clean + operator-conditioned ambient branches for the HR denoiser.

This is the *non-vacuous* ambient training path (contrast ``losses/ambient.py``,
which is ``mse(A(G(y)), y)`` and teaches nothing about ``null(A)``). Here the
ambient branch injects **operator diversity** via subcell-shift operators
``H_g``, feeding an HR-grid backprojection of a noisy measurement and scoring in
measurement space -- so LR-only data can shape the HR prior. See
``docs/gate1_operator_coverage.md``.

Branches (shared denoiser ``D_psi``, schedule with ``alpha(t), sigma(t)``):

    clean   :  x_t   = alpha x + sigma eps
               L_clean = w(t) || D_psi(x_t, t, identity) - x ||^2

    ambient :  eps ~ N(0, I) on the HR grid
               y_t   = alpha y + sigma (H_g eps)                (measurement grid)
               input = H_g^+(y_t)
               L_amb = || H_g D_psi(input, t, shifted_g) - y ||^2 / n_LR

The C1/C2/C3 constructions differ only in how the ``(y, g)`` pairs are built
(:func:`build_ambient_target`); the loss is identical.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..models.operator_denoiser import OperatorConditionedDenoiser, CosineSchedule
from ..operators.shifted_operator import ShiftedDownsampleOperator
from ..operators.symmetry import Shift


def _sample_t(batch: int, device, eps: float = 1e-3) -> torch.Tensor:
    """Uniform ``t`` in ``[eps, 1-eps]`` (avoid the exact schedule endpoints)."""
    return torch.rand(batch, device=device) * (1 - 2 * eps) + eps


def _energies(x0_hat: torch.Tensor, operator: ShiftedDownsampleOperator, g) -> Dict[str, float]:
    """Range/null energy of the prediction (watch for a trivial backprojection)."""
    rng = operator.project_range(x0_hat, g)
    null = x0_hat - rng
    tot = x0_hat.pow(2).mean().item() + 1e-12
    return {
        "range_energy": rng.pow(2).mean().item(),
        "null_energy": null.pow(2).mean().item(),
        "null_frac": null.pow(2).mean().item() / tot,
    }


def clean_denoise_loss(
    denoiser: OperatorConditionedDenoiser,
    x: torch.Tensor,
    schedule: CosineSchedule,
    t: Optional[torch.Tensor] = None,
    weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """``L_clean = w(t) || D_psi(x_t, t, identity) - x ||^2`` (uniform ``w`` by default)."""
    if t is None:
        t = _sample_t(x.shape[0], x.device)
    a, s = schedule.broadcast(t, ndim=x.dim())
    eps = torch.randn_like(x)
    x_t = a * x + s * eps
    x0_hat = denoiser(x_t, t, shift=(0, 0, 0), kind="identity")
    loss = weight * F.mse_loss(x0_hat, x)
    return loss, {"loss_clean": loss.item()}


def ambient_denoise_loss(
    denoiser: OperatorConditionedDenoiser,
    operator: ShiftedDownsampleOperator,
    y: torch.Tensor,
    g: Shift,
    schedule: CosineSchedule,
    t: Optional[torch.Tensor] = None,
    kind: str = "shifted",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Measurement-space ambient loss under operator ``H_g`` (see module docstring).

    ``y`` is the observed measurement (LR grid); ``g`` the assigned subcell shift.
    HR-grid noise is measured through ``H_g`` so ``y_t`` carries the operator's
    induced noise covariance, then backprojected to the HR grid as the input.
    """
    b, c = y.shape[0], y.shape[1]
    n_hr = y.shape[-1] * operator.factor
    if t is None:
        t = _sample_t(b, y.device)
    a, s = schedule.broadcast(t, ndim=y.dim())
    eps = torch.randn(b, c, n_hr, n_hr, n_hr, device=y.device, dtype=y.dtype)
    y_t = a * y + s * operator.forward(eps, g)          # noisy measurement (LR grid)
    input_hr = operator.pseudoinverse(y_t, g)           # HR-grid backprojection
    x0_hat = denoiser(input_hr, t, shift=g, kind=kind)
    y_hat = operator.forward(x0_hat, g)
    loss = F.mse_loss(y_hat, y)                          # / n_LR elements
    diag = {"loss_ambient": loss.item(), **_energies(x0_hat, operator, g)}
    return loss, diag


# --------------------------------------------------------------------------- #
# C1 / C2 / C3 ambient-data constructions (controlled study)
# --------------------------------------------------------------------------- #
def build_ambient_target(
    x: torch.Tensor,
    operator: ShiftedDownsampleOperator,
    mode: str,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, Shift, str]:
    """Build a ``(y, g, kind)`` ambient target from clean HR ``x``.

    * ``"fixed"``  (C1): ``y = A x``,  ``g = 0``           -- weak fixed-null baseline.
    * ``"true_shift"`` (C2): sample ``g``, ``y = H_g x``   -- genuine operator diversity.
    * ``"virtual_shift"`` (C3): ``y = A x``, ``g`` sampled *independently* -- the
      symmetry-induced virtual operator (only distributionally justified).
    """
    if mode == "fixed":
        g: Shift = (0, 0, 0)
        return operator.forward(x, g), g, "fixed"
    if mode == "true_shift":
        g = operator.sym.sample_shift(rng)
        return operator.forward(x, g), g, "shifted"
    if mode == "virtual_shift":
        g = operator.sym.sample_shift(rng)
        return operator.forward(x, (0, 0, 0)), g, "shifted"  # measurement is A x
    raise ValueError(f"unknown ambient mode {mode!r} (use fixed|true_shift|virtual_shift)")
