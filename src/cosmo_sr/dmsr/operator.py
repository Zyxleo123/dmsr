"""The analytic degradation ``A``, its right inverse ``A_plus``, and ``P_A``.

This is the single-operator framing of the DMSR stage: one ``A`` taking a
``512^3`` HR field straight to the ``64^3`` LR grid (``factor = 8``), matching
the paired data on disk (``paired_catnorm/{hr,lr}``) and, crucially, matching the
native resolution of the 350 LR-only boxes so they can condition the model
directly.

The three operators are thin wrappers over the already-verified primitives in
:mod:`cosmo_sr.operators.multiscale` (``block_average`` / ``block_upsample`` /
``null_projection``), instantiated at ``factor=8`` instead of the octave-cascade's
``factor=2``. Nothing about those primitives is factor-specific, so the identities
they already satisfy in ``tests/test_multiscale.py`` carry over:

    A(A_plus(y))  = y      (A_plus is an exact right inverse)
    A(P_A(x))     = 0      (P_A lands in ker A)
    P_A(P_A(x))   = P_A(x) (P_A is idempotent)

Both are exact up to float rounding: ``A`` averages ``s^3`` blocks and ``A_plus``
broadcasts a constant over each block, so ``A A_plus`` is the identity by
construction rather than by approximation.

Exact consistency of the generated field is therefore *structural*::

    x_hat = A_plus(y) + P_A(r)   ==>   A(x_hat) = y + 0 = y

which is why ``||A(x_hat) - y||`` is logged as a **diagnostic only** and never
used as a training loss. Training it would be the known-degenerate objective:
under this parameterization the term is identically zero, so it supplies no
gradient to the null-space residual at all (regression-tested in
``tests/dmsr/test_adversarial_grad.py``).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ..operators.multiscale import block_average, block_upsample


class NullSpaceOperator(nn.Module):
    """Bundle of ``A``, ``A_plus``, ``P_A`` for a factor-``s`` block degradation.

    Stateless (no parameters), but a ``Module`` so it moves with ``.to(device)``
    and composes inside larger models.

    Parameters
    ----------
    factor:
        Integer downsampling ratio ``N_hr / N_lr`` per axis. ``8`` for the paired
        boxes here (512 -> 64).
    """

    def __init__(self, factor: int = 8):
        super().__init__()
        if not isinstance(factor, int) or factor < 2:
            raise ValueError(f"factor must be an int >= 2, got {factor!r}")
        self.factor = int(factor)

    # -- the three operators ------------------------------------------------ #
    def A(self, x_hr: torch.Tensor) -> torch.Tensor:
        """``A``: HR -> LR by non-overlapping ``s^3`` block mean."""
        return block_average(x_hr, self.factor)

    def A_plus(self, y_lr: torch.Tensor) -> torch.Tensor:
        """``A_plus``: LR -> HR by block broadcast. Exact right inverse of ``A``."""
        return block_upsample(y_lr, self.factor)

    def P_A(self, x_hr: torch.Tensor) -> torch.Tensor:
        """``P_A(x) = x - A_plus(A(x))``: project onto ``ker A`` (remove block means)."""
        return x_hr - self.A_plus(self.A(x_hr))

    # -- the generator parameterization ------------------------------------- #
    def combine(self, y_lr: torch.Tensor, r_raw: torch.Tensor) -> torch.Tensor:
        """``x_hat = A_plus(y) + P_A(r_raw)`` -- exactly consistent with ``y``.

        ``r_raw`` is the *unprojected* generator output; projecting here (rather
        than trusting the network to emit a null-space field) is what makes the
        guarantee structural.
        """
        if r_raw.shape[-1] != y_lr.shape[-1] * self.factor:
            raise ValueError(
                f"combine: residual grid {r_raw.shape[-1]} != LR grid "
                f"{y_lr.shape[-1]} * factor {self.factor}"
            )
        return self.A_plus(y_lr) + self.P_A(r_raw)

    # -- diagnostics (never a training loss) -------------------------------- #
    @torch.no_grad()
    def consistency_error(
        self, x_hat: torch.Tensor, y_lr: torch.Tensor
    ) -> Tuple[float, float]:
        """``(absolute, relative)`` L2 error of ``A(x_hat)`` against ``y``.

        Diagnostic only -- see the module docstring. Relative error is normalised
        by ``||y||`` so it is comparable across crops and channel subsets.
        """
        resid = self.A(x_hat) - y_lr
        abs_err = float(resid.norm())
        rel_err = abs_err / float(y_lr.norm().clamp_min(1e-12))
        return abs_err, rel_err

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"factor={self.factor}"
