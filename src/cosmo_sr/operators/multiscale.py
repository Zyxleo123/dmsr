"""Factor-2 multiscale operators for the residual null-space cascade.

At resolution octave ``R`` we relate a coarse field at grid size ``R`` to a fine
field at grid size ``2R`` via three operators (all parameter-free):

* ``A_R`` : block-average a ``2R`` field down to ``R`` (``avg_pool3d`` factor 2).
* ``U_R`` : block-broadcast (nearest) upsample an ``R`` field up to ``2R``.
* ``P_null_R(h) = h - U_R(A_R(h))`` : project a ``2R`` field onto the null space
  of ``A_R`` (removes per-block means).

Exact identities (up to float error), verified in tests:

* ``A_R(U_R(y)) = y``          (``U_R`` is a right inverse of ``A_R``)
* ``A_R(P_null_R(h)) = 0``     (``P_null_R`` outputs live in ``ker A_R``)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_cubic5d(x: torch.Tensor, name: str) -> None:
    if x.dim() != 5:
        raise ValueError(f"{name}: expected 5D (B, C, N, N, N), got {tuple(x.shape)}")
    _, _, nx, ny, nz = x.shape
    if not (nx == ny == nz):
        raise ValueError(f"{name}: only cubic volumes supported, got {(nx, ny, nz)}")


def block_average(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """``A_R``: average non-overlapping ``factor**3`` blocks (``2R -> R``)."""
    _check_cubic5d(x, "block_average")
    n = x.shape[-1]
    if n % factor != 0:
        raise ValueError(f"block_average: size {n} not divisible by factor {factor}")
    return F.avg_pool3d(x, kernel_size=factor, stride=factor)


def block_upsample(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """``U_R``: nearest (block-broadcast) upsample (``R -> 2R``)."""
    _check_cubic5d(x, "block_upsample")
    return F.interpolate(x, scale_factor=factor, mode="nearest")


def null_projection(h: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """``P_null_R``: remove the component of ``h`` seen by ``A_R``."""
    return h - block_upsample(block_average(h, factor), factor)


class MultiScaleOperators(nn.Module):
    """Bundle of the factor-2 operators ``A_R``, ``U_R``, ``P_null_R``.

    Stateless (no parameters); wrapped as a module so it moves with ``.to(device)``
    and composes cleanly inside larger models.
    """

    def __init__(self, factor: int = 2):
        super().__init__()
        if not isinstance(factor, int) or factor < 2:
            raise ValueError(f"factor must be an int >= 2, got {factor!r}")
        self.factor = int(factor)

    def A(self, x: torch.Tensor) -> torch.Tensor:
        return block_average(x, self.factor)

    def U(self, y: torch.Tensor) -> torch.Tensor:
        return block_upsample(y, self.factor)

    def P_null(self, h: torch.Tensor) -> torch.Tensor:
        return null_projection(h, self.factor)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"factor={self.factor}"
