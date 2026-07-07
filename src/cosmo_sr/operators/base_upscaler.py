"""Base upscalers ``B_R(y_R)`` and the always-consistent correction ``B_cons``.

``B_R`` maps a coarse field at grid ``R`` to a candidate fine field at ``2R``.
Any base upscaler can be made *exactly LR-consistent* via::

    B_cons(y) = B_R(y) + U_R(y - A_R(B_R(y)))

which satisfies ``A_R(B_cons(y)) = y`` regardless of ``B_R`` (since ``A U = I``).

Implemented base upscalers:

* :class:`IdentityUpscaler` -- ``B_R(y) = U_R(y)`` (nearest broadcast). This is the
  simplest starting point; ``B_cons`` then equals ``U_R(y)`` exactly.
* :class:`BackboneUpscaler` -- wraps a learned SR backbone (e.g. a factor-2
  :class:`SimpleSRGenerator`, i.e. an "SR2" backbone) as ``B_R``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .multiscale import MultiScaleOperators


class BaseUpscaler(nn.Module):
    """Interface: map coarse ``y_R`` (grid ``R``) to fine candidate at ``2R``."""

    factor: int = 2

    def forward(self, y_R: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


class IdentityUpscaler(BaseUpscaler):
    """``B_R(y) = U_R(y)`` -- parameter-free nearest broadcast upsample."""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = int(factor)
        self.ops = MultiScaleOperators(self.factor)

    def forward(self, y_R: torch.Tensor) -> torch.Tensor:
        return self.ops.U(y_R)


class BackboneUpscaler(BaseUpscaler):
    """Wrap a learned factor-2 SR backbone as the base upscaler ``B_R``.

    The backbone must accept a ``(B, C, R, R, R)`` field and return a
    ``(B, C, 2R, 2R, 2R)`` field (e.g. ``SimpleSRGenerator(scale_factor=2)``).
    """

    def __init__(self, backbone: nn.Module, factor: int = 2):
        super().__init__()
        self.factor = int(factor)
        self.backbone = backbone

    def forward(self, y_R: torch.Tensor) -> torch.Tensor:
        return self.backbone(y_R)


def consistent_base(
    base_upscaler: BaseUpscaler, ops: MultiScaleOperators, y_R: torch.Tensor
) -> torch.Tensor:
    """Return ``B_cons(y_R) = B_R(y_R) + U_R(y_R - A_R(B_R(y_R)))``.

    Guarantees ``A_R(consistent_base(...)) = y_R`` up to float error.
    """
    b = base_upscaler(y_R)
    return b + ops.U(y_R - ops.A(b))
