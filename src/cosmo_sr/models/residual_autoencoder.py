"""3D convolutional residual autoencoder for null-space HR residuals.

The autoencoder compresses a null-space residual field ``r_star`` (living at grid
``2R``) into a small latent ``z`` and reconstructs it. It is the first stage of
the *latent residual flow* extension: the latent conditional flow later learns
the distribution of these latents.

Shapes::

    r_star : (B, C, N, N, N)             # N = 2R crop size, C = 6
    z      : (B, C_lat, N/D, N/D, N/D)   # D = 2**n_down downsample factor
    recon  : (B, C, N, N, N)

The input spatial size ``N`` must be divisible by ``D = 2**n_down`` (mirrors the
existing ``crop_hr`` divisibility requirement in :class:`PyramidCropDataset`).

Consistency note: this module reconstructs the *raw* residual; exact LR
consistency is enforced downstream by projecting the decode through
``ops.P_null`` (so ``A_R(P_null(decode(z))) == 0``), never by the AE itself.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


def _groups(groups: int, width: int) -> int:
    return math.gcd(groups, width) or 1


class ResBlock3d(nn.Module):
    """GroupNorm + SiLU 3D conv residual block (channel-preserving)."""

    def __init__(self, width: int, groups: int = 8):
        super().__init__()
        g = _groups(groups, width)
        self.norm1 = nn.GroupNorm(g, width)
        self.conv1 = nn.Conv3d(width, width, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g, width)
        self.conv2 = nn.Conv3d(width, width, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class ResidualAutoencoder(nn.Module):
    """Symmetric 3D residual autoencoder.

    Parameters
    ----------
    channels:
        Field channels (6 for displacement+velocity).
    width:
        Base feature width at the finest level.
    ch_mults:
        Channel multipliers per level; ``len == n_down + 1``. The number of
        strided downsamples is ``n_down = len(ch_mults) - 1``.
    latent_channels:
        Number of latent channels ``C_lat``.
    n_res:
        Residual blocks per level.
    groups:
        GroupNorm group hint (reduced to gcd with width).
    """

    def __init__(
        self,
        channels: int = 6,
        width: int = 32,
        ch_mults: Sequence[int] = (1, 2, 2),
        latent_channels: int = 16,
        n_res: int = 1,
        groups: int = 8,
    ):
        super().__init__()
        ch_mults = tuple(int(m) for m in ch_mults)
        if len(ch_mults) < 1:
            raise ValueError("ch_mults must have at least one entry")
        self.channels = int(channels)
        self.latent_channels = int(latent_channels)
        self.n_down = len(ch_mults) - 1
        self.downsample_factor = 2 ** self.n_down
        widths = [int(width * m) for m in ch_mults]

        # ---- encoder ----
        self.stem = nn.Conv3d(channels, widths[0], kernel_size=3, padding=1)
        enc: list[nn.Module] = []
        for i in range(self.n_down):
            for _ in range(n_res):
                enc.append(ResBlock3d(widths[i], groups))
            enc.append(nn.Conv3d(widths[i], widths[i + 1], kernel_size=4, stride=2, padding=1))
        self.encoder = nn.ModuleList(enc)
        self.enc_mid = ResBlock3d(widths[-1], groups)
        self.to_latent = nn.Conv3d(widths[-1], latent_channels, kernel_size=3, padding=1)

        # ---- decoder ----
        self.from_latent = nn.Conv3d(latent_channels, widths[-1], kernel_size=3, padding=1)
        self.dec_mid = ResBlock3d(widths[-1], groups)
        dec: list[nn.Module] = []
        for i in reversed(range(self.n_down)):
            dec.append(
                nn.ConvTranspose3d(widths[i + 1], widths[i], kernel_size=4, stride=2, padding=1)
            )
            for _ in range(n_res):
                dec.append(ResBlock3d(widths[i], groups))
        self.decoder = nn.ModuleList(dec)
        self.out_norm = nn.GroupNorm(_groups(groups, widths[0]), widths[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv3d(widths[0], channels, kernel_size=3, padding=1)

    def _check_size(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError(f"expected 5D (B,C,N,N,N), got {tuple(x.shape)}")
        n = x.shape[-1]
        if n % self.downsample_factor != 0:
            raise ValueError(
                f"input size {n} must be divisible by downsample factor "
                f"{self.downsample_factor} (n_down={self.n_down})"
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._check_size(x)
        h = self.stem(x)
        for layer in self.encoder:
            h = layer(h)
        h = self.enc_mid(h)
        return self.to_latent(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = self.dec_mid(h)
        for layer in self.decoder:
            h = layer(h)
        return self.out_conv(self.out_act(self.out_norm(h)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
