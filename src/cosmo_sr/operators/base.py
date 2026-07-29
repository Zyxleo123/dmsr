"""Common linear-measurement-operator interface.

Every measurement operator ``H`` used for conditional generation is linear in the
HR field and exposes the same five maps, so training/sampling code can treat the
fixed degradation and the symmetry-shifted degradations uniformly:

* :meth:`forward`        -- ``H x``            (HR grid -> measurement grid)
* :meth:`adjoint`        -- ``H^T y``          (measurement grid -> HR grid)
* :meth:`pseudoinverse`  -- ``H^+ y``, a right inverse (``H H^+ = I``)
* :meth:`project_range`  -- ``H^+ H x``        (component of ``x`` seen by ``H``)
* :meth:`project_null`   -- ``x - H^+ H x``    (component of ``x`` in ``ker H``)

``operator_context`` carries which concrete operator to apply (e.g. a subcell
shift ``g``); subclasses define its type. ``None`` means the canonical operator.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class LinearMeasurementOperator(nn.Module):
    """Abstract linear HR -> measurement operator (see module docstring)."""

    def forward(self, x_hr: torch.Tensor, operator_context: Any = None) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def adjoint(self, y: torch.Tensor, operator_context: Any = None) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def pseudoinverse(self, y: torch.Tensor, operator_context: Any = None) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def project_range(self, x_hr: torch.Tensor, operator_context: Any = None) -> torch.Tensor:
        """``H^+ H x`` -- the part of ``x`` the measurement constrains."""
        return self.pseudoinverse(self.forward(x_hr, operator_context), operator_context)

    def project_null(self, x_hr: torch.Tensor, operator_context: Any = None) -> torch.Tensor:
        """``x - H^+ H x`` -- the part of ``x`` invisible to the measurement."""
        return x_hr - self.project_range(x_hr, operator_context)
