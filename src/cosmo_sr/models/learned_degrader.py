"""Learned degrader ``D_phi`` as a correction to the known average-pool degrader.

    D_phi(x_2R, R) = A_R(x_2R) + Delta_phi(x_2R, R)

``A_R`` is the exact block-average operator; ``Delta_phi`` is a small CNN that
downsamples ``x_2R`` (grid ``2R``) by factor 2 to ``R`` and predicts a residual
correction. The final conv of ``Delta_phi`` is zero-initialised, so at init
``D_phi == A_R`` exactly (the safe baseline).

In the latent-flow pipeline ``D_phi`` is trained separately, then **frozen** and
used only as an auxiliary physical-degrader consistency regulariser -- it never
replaces the paired HR residual supervision nor the hard null-space consistency.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..operators.multiscale import MultiScaleOperators
from .residual_flow import sinusoidal_embedding


def _groups(groups: int, width: int) -> int:
    return math.gcd(groups, width) or 1


class _FiLMResBlock3d(nn.Module):
    """GroupNorm + SiLU conv residual block with optional FiLM conditioning."""

    def __init__(self, width: int, cond_dim: int = 0, groups: int = 8):
        super().__init__()
        g = _groups(groups, width)
        self.norm1 = nn.GroupNorm(g, width)
        self.conv1 = nn.Conv3d(width, width, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g, width)
        self.conv2 = nn.Conv3d(width, width, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.film = nn.Linear(cond_dim, 2 * width) if cond_dim > 0 else None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.norm2(h)
        if self.film is not None and cond is not None:
            scale, shift = self.film(cond).chunk(2, dim=1)
            scale = scale.view(scale.shape[0], -1, 1, 1, 1)
            shift = shift.view(shift.shape[0], -1, 1, 1, 1)
            h = h * (1 + scale) + shift
        h = self.conv2(self.act(h))
        return x + h


class LearnedDegrader(nn.Module):
    """``D_phi(x_2R, R) = A_R(x_2R) + Delta_phi(x_2R, R)``.

    Parameters
    ----------
    channels:
        Field channels (6).
    width, depth:
        Feature width and number of residual blocks in ``Delta_phi``.
    factor:
        Downsample factor (2).
    use_res_embed:
        If ``True`` inject a resolution embedding (from ``log2(R)``) via FiLM.
    embed_dim:
        Resolution-embedding dimensionality (used when ``use_res_embed``).
    """

    def __init__(
        self,
        channels: int = 6,
        width: int = 32,
        depth: int = 2,
        factor: int = 2,
        use_res_embed: bool = True,
        embed_dim: int = 64,
        groups: int = 8,
    ):
        super().__init__()
        self.channels = int(channels)
        self.factor = int(factor)
        self.use_res_embed = bool(use_res_embed)
        self.embed_dim = int(embed_dim)
        self.ops = MultiScaleOperators(self.factor)

        cond_dim = embed_dim if use_res_embed else 0
        if use_res_embed:
            self.r_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
            )

        # strided conv downsamples 2R -> R; kernel scales with factor so every
        # input voxel falls inside at least one output's receptive field (at
        # factor=2 this reduces to the original kernel=4, stride=2, pad=1).
        self.head = nn.Conv3d(
            channels, width, kernel_size=2 * factor, stride=factor, padding=factor // 2
        )
        self.blocks = nn.ModuleList(
            [_FiLMResBlock3d(width, cond_dim, groups) for _ in range(max(depth, 1))]
        )
        self.out_norm = nn.GroupNorm(_groups(groups, width), width)
        self.out_act = nn.SiLU()
        self.final = nn.Conv3d(width, channels, kernel_size=3, padding=1)
        # zero-init final conv so Delta_phi == 0 and D_phi == A_R at init
        nn.init.zeros_(self.final.weight)
        if self.final.bias is not None:
            nn.init.zeros_(self.final.bias)

    def _res_cond(self, R, batch: int, device) -> Optional[torch.Tensor]:
        if not self.use_res_embed:
            return None
        R = torch.as_tensor(R, device=device, dtype=torch.float32).reshape(-1)
        if R.numel() == 1:
            R = R.expand(batch)
        emb = sinusoidal_embedding(torch.log2(R.clamp_min(1.0)), self.embed_dim)
        return self.r_mlp(emb)

    def delta(self, x_2R: torch.Tensor, R) -> torch.Tensor:
        """Correction ``Delta_phi(x_2R, R)`` at grid ``R`` (zero at init)."""
        b = x_2R.shape[0]
        cond = self._res_cond(R, b, x_2R.device)
        h = self.head(x_2R)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.final(self.out_act(self.out_norm(h)))

    def forward(self, x_2R: torch.Tensor, R) -> torch.Tensor:
        return self.ops.A(x_2R) + self.delta(x_2R, R)
