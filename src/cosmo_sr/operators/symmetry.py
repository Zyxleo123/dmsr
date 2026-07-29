"""Periodic subcell translations ``T_g`` and the ``SymmetryTransform`` interface.

For a factor-``s`` degradation, ``A`` block-averages non-overlapping ``s^3`` HR
blocks. A **subcell** translation by ``g = (gx, gy, gz)`` HR voxels with
``g_i in {0..s-1}`` re-tiles the HR grid into a *different* set of averaging
blocks, so ``H_g = A o T_g`` is a genuinely different measurement operator (this
is what expands operator coverage; see ``docs/gate1_operator_coverage.md``).
Coarse translations by multiples of ``s`` merely permute the LR voxels
(``A o T_{s*m} = shift_m o A``) and are *not* useful diversity -- see
:func:`SubcellShift.is_subcell`.

Only **translations** are implemented (the plan's Phase A). On the periodic grid
they act via :func:`torch.roll` and do **not** rotate vector components, so scalar
and 3-vector channels transform identically. Proper cubic rotations (later) must
*additionally* rotate the ``disp``/``vel`` 3-vector channels; that is intentionally
left unimplemented rather than applied incorrectly.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch

Shift = Tuple[int, int, int]


class SymmetryTransform:
    """Interface: a spatial symmetry acting on fields and on HR-space noise."""

    def apply_field(self, x: torch.Tensor, context) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def invert_field(self, x: torch.Tensor, context) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def apply_noise(self, z: torch.Tensor, context) -> torch.Tensor:
        """Transform HR-space noise the same way as a field (identical for
        translations; a hook for rotations, which act on noise spatially only)."""
        return self.apply_field(z, context)


def as_shift(context, factor: int | None = None) -> Shift:
    """Coerce ``context`` (a shift tuple, ``None``, or an object with ``.shift``)
    into a validated integer ``(gx, gy, gz)``."""
    if context is None:
        return (0, 0, 0)
    g = getattr(context, "shift", context)
    g = tuple(int(v) for v in g)
    if len(g) != 3:
        raise ValueError(f"shift must have 3 components, got {g!r}")
    if factor is not None and not all(0 <= v < factor for v in g):
        raise ValueError(f"subcell shift {g} out of range for factor {factor}")
    return g  # type: ignore[return-value]


class SubcellShift(SymmetryTransform):
    """Periodic HR-grid translation ``T_g`` by ``g`` HR voxels (dims ``-3,-2,-1``).

    ``T_g x = roll(x, g)`` and ``T_g^{-1} x = roll(x, -g)``. ``context`` is the
    shift ``g`` (a 3-tuple or anything with a ``.shift`` attribute).
    """

    def __init__(self, factor: int = 2):
        if not isinstance(factor, int) or factor < 2:
            raise ValueError(f"factor must be an int >= 2, got {factor!r}")
        self.factor = int(factor)

    # -- group helpers ------------------------------------------------------ #
    def is_subcell(self, g: Shift) -> bool:
        """``True`` unless every component is 0 mod ``factor`` (a coarse shift
        that commutes with ``A`` and adds no coverage)."""
        return any(v % self.factor != 0 for v in g)

    def all_shifts(self) -> list[Shift]:
        f = self.factor
        return [(x, y, z) for x in range(f) for y in range(f) for z in range(f)]

    def sample_shift(self, rng: np.random.Generator) -> Shift:
        f = self.factor
        return (int(rng.integers(f)), int(rng.integers(f)), int(rng.integers(f)))

    # -- transforms --------------------------------------------------------- #
    def apply_field(self, x: torch.Tensor, context) -> torch.Tensor:
        g = as_shift(context, self.factor)
        return torch.roll(x, shifts=g, dims=(-3, -2, -1))

    def invert_field(self, x: torch.Tensor, context) -> torch.Tensor:
        g = as_shift(context, self.factor)
        return torch.roll(x, shifts=tuple(-v for v in g), dims=(-3, -2, -1))
