"""Fixed (non-learned) degradation operator ``A``: HR field -> LR field.

This is the first core method component. Only ``mode="average"`` (mean pooling
over non-overlapping ``scale_factor**3`` blocks) is implemented for now. Other
modes (CIC mass assignment, Fourier low-pass, particle degradation) are
intentionally deferred.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_SUPPORTED_MODES = ("average",)


class FixedDegrader(nn.Module):
    """Deterministic downsampling operator.

    Parameters
    ----------
    scale_factor:
        Integer ratio ``N_hr / N_lr`` along each spatial axis.
    mode:
        Only ``"average"`` is supported (block mean pooling via ``avg_pool3d``).
    """

    def __init__(self, scale_factor: int, mode: str = "average"):
        super().__init__()
        if not isinstance(scale_factor, int) or scale_factor < 1:
            raise ValueError(f"scale_factor must be a positive int, got {scale_factor!r}")
        if mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"mode {mode!r} not supported; available modes: {_SUPPORTED_MODES}"
            )
        self.scale_factor = int(scale_factor)
        self.mode = mode

    def forward(self, x_hr: torch.Tensor) -> torch.Tensor:
        if x_hr.dim() != 5:
            raise ValueError(
                f"Expected 5D input (B, C, N, N, N), got shape {tuple(x_hr.shape)}"
            )
        b, c, nx, ny, nz = x_hr.shape
        if not (nx == ny == nz):
            raise ValueError(
                f"Only cubic HR volumes are supported; got spatial dims {(nx, ny, nz)}"
            )
        s = self.scale_factor
        if nx % s != 0:
            raise ValueError(
                f"HR size {nx} is not divisible by scale_factor {s}; "
                "cannot degrade with non-overlapping blocks."
            )
        if self.mode == "average":
            return F.avg_pool3d(x_hr, kernel_size=s, stride=s)
        raise ValueError(f"Unhandled mode {self.mode!r}")  # pragma: no cover

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"scale_factor={self.scale_factor}, mode={self.mode!r}"
