"""A small deterministic 3D super-resolution generator ``G(y_lr) -> x_hr``.

Design: trilinear upsample by ``scale_factor`` followed by a stack of Conv3d
residual blocks and an output convolution. Spatial size is preserved by the
convolutions (padding=1), so the output is exactly ``scale_factor`` times the
input size along each axis.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3d(nn.Module):
    """Two 3x3x3 convolutions with a residual connection."""

    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv3d(width, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(width, width, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


class SimpleSRGenerator(nn.Module):
    """Deterministic 3D SR generator.

    Parameters
    ----------
    in_channels, out_channels:
        Channel counts (6 for displacement+velocity).
    scale_factor:
        Upsampling ratio per axis.
    width:
        Number of feature channels in the residual body.
    depth:
        Number of residual blocks.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 6,
        scale_factor: int = 8,
        width: int = 64,
        depth: int = 4,
    ):
        super().__init__()
        if not isinstance(scale_factor, int) or scale_factor < 1:
            raise ValueError(f"scale_factor must be a positive int, got {scale_factor!r}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = int(scale_factor)

        self.head = nn.Conv3d(in_channels, width, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResBlock3d(width) for _ in range(depth)])
        self.tail = nn.Conv3d(width, out_channels, kernel_size=3, padding=1)

    def forward(self, y_lr: torch.Tensor) -> torch.Tensor:
        if y_lr.dim() != 5:
            raise ValueError(
                f"Expected 5D input (B, C, N, N, N), got shape {tuple(y_lr.shape)}"
            )
        if y_lr.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {y_lr.shape[1]}"
            )
        x = F.interpolate(
            y_lr, scale_factor=self.scale_factor, mode="trilinear", align_corners=False
        )
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class NullSpaceSRGenerator(nn.Module):
    """Range/null-space decomposition generator with hard data consistency.

    For the average-pool degrader ``A`` (block mean over ``scale**3`` blocks), a
    right inverse is ``A+`` = block broadcast (nearest upsample), since
    ``A(A+ y) = y``. This generator outputs::

        x_hat = A+(y)  +  (I - A+ A) H(y, z)

    where ``H`` is a :class:`SimpleSRGenerator` backbone. The second term lives in
    the null space of ``A``, so ``A(x_hat) = y`` holds **exactly by construction**
    (up to float error) and the ambient LR-consistency loss is satisfied for free.

    ``z_channels`` optionally appends noise channels to the input of ``H`` (for a
    future stochastic variant); it defaults to 0 (deterministic).
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 6,
        scale_factor: int = 8,
        width: int = 64,
        depth: int = 4,
        z_channels: int = 0,
    ):
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("NullSpaceSRGenerator requires in_channels == out_channels")
        if not isinstance(scale_factor, int) or scale_factor < 1:
            raise ValueError(f"scale_factor must be a positive int, got {scale_factor!r}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = int(scale_factor)
        self.z_channels = int(z_channels)
        self.backbone = SimpleSRGenerator(
            in_channels=in_channels + self.z_channels,
            out_channels=out_channels,
            scale_factor=scale_factor,
            width=width,
            depth=depth,
        )

    def _a_pinv(self, y_lr: torch.Tensor) -> torch.Tensor:
        # A+ : broadcast each LR voxel over its scale**3 HR block (nearest)
        return F.interpolate(y_lr, scale_factor=self.scale_factor, mode="nearest")

    def _degrade(self, x_hr: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(x_hr, kernel_size=self.scale_factor, stride=self.scale_factor)

    def forward(self, y_lr: torch.Tensor) -> torch.Tensor:
        if y_lr.dim() != 5:
            raise ValueError(
                f"Expected 5D input (B, C, N, N, N), got shape {tuple(y_lr.shape)}"
            )
        if y_lr.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {y_lr.shape[1]}"
            )
        h_in = y_lr
        if self.z_channels > 0:
            b, _, n0, n1, n2 = y_lr.shape
            z = torch.randn(b, self.z_channels, n0, n1, n2, device=y_lr.device, dtype=y_lr.dtype)
            h_in = torch.cat([y_lr, z], dim=1)

        base = self._a_pinv(y_lr)               # A+ y  (range component)
        h = self.backbone(h_in)                 # candidate HR detail
        h_null = h - self._a_pinv(self._degrade(h))  # (I - A+ A) H
        return base + h_null
