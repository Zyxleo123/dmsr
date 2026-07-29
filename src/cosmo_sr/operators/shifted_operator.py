"""Symmetry-shifted degradation operators ``H_g = A o T_g``.

Built on the factor-``s`` :class:`MultiScaleOperators` (``A`` = block-average,
``U`` = block-broadcast right inverse) and :class:`SubcellShift` (``T_g``). With
``g = (0, 0, 0)`` this reduces to the fixed analytic degradation.

Maps (``x`` on the HR grid ``2R``/``sR``, ``y`` on the measurement grid ``R``):

* forward       ``H_g x        = A(T_g x)``
* pseudoinverse ``H_g^+ y      = T_g^{-1}(U y)``          (right inverse: ``H_g H_g^+ = I``)
* adjoint       ``H_g^T y      = T_g^{-1}(A^T y)``,  ``A^T = U / s^3``
* project_null  ``x - H_g^+ H_g x = T_g^{-1} P_null(T_g x)``

``T_g`` is orthogonal (a periodic permutation) so ``T_g^T = T_g^{-1}`` and the
identities ``A U = I`` / ``A P_null = 0`` carry over verbatim in the shifted frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .base import LinearMeasurementOperator
from .multiscale import MultiScaleOperators
from .symmetry import SubcellShift, Shift, as_shift


@dataclass(frozen=True)
class OperatorContext:
    """What operator to apply, and metadata for conditioning embeddings.

    ``kind``: ``"identity"`` (H = I, clean branch), ``"fixed"`` (g = 0), or
    ``"shifted"``. ``shift`` is the subcell translation ``g`` in HR voxels.
    """

    shift: Shift = (0, 0, 0)
    factor: int = 2
    kind: str = "shifted"


class ShiftedDownsampleOperator(LinearMeasurementOperator):
    """``H_g = A o T_g`` with an exact right inverse and adjoint.

    ``operator_context`` may be an :class:`OperatorContext`, a raw shift 3-tuple,
    or ``None`` (the fixed operator ``g = 0``).
    """

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = int(factor)
        self.ops = MultiScaleOperators(self.factor)
        self.sym = SubcellShift(self.factor)

    def _g(self, operator_context: Any) -> Shift:
        return as_shift(operator_context, self.factor)

    def forward(self, x_hr: torch.Tensor, operator_context: Any = None) -> torch.Tensor:
        g = self._g(operator_context)
        return self.ops.A(self.sym.apply_field(x_hr, g))

    def pseudoinverse(self, y: torch.Tensor, operator_context: Any = None) -> torch.Tensor:
        g = self._g(operator_context)
        return self.sym.invert_field(self.ops.U(y), g)

    def adjoint(self, y: torch.Tensor, operator_context: Any = None) -> torch.Tensor:
        g = self._g(operator_context)
        a_t = self.ops.U(y) / float(self.factor ** 3)  # A^T = U / s^3
        return self.sym.invert_field(a_t, g)
