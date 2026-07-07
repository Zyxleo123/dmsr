"""Scale-shared residual flow model ``v_phi(r_t, t, y_R, R)``.

A single network, shared across resolution octaves ``R in {64, 128, 256}``,
predicts the flow-matching velocity of a *null-space residual* living at grid
``2R``. Conditioning:

* ``r_t``    : current residual state at ``2R`` (``C`` channels).
* ``y_R``    : coarse field at ``R``; broadcast-upsampled (``U_R``) to ``2R`` and
  concatenated as conditioning (``C`` channels).
* ``t``      : flow time in ``[0, 1]`` -> sinusoidal embedding.
* ``R``      : resolution octave -> embedding from ``log2(R)`` (so the same
  weights specialise per octave while sharing structure).
* ``context``: optional extra conditioning channels at ``2R`` (default none).

Time/resolution conditioning is injected via FiLM (per-channel affine) in each
residual block, so the network stays fully convolutional and works at any size.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt

from ..operators.multiscale import block_upsample


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of a ``(B,)`` tensor into ``(B, dim)``."""
    if dim % 2 != 0:
        raise ValueError(f"embedding dim must be even, got {dim}")
    device = values.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    args = values.float().view(-1, 1) * freqs.view(1, -1)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class FiLMResBlock3d(nn.Module):
    """GroupNorm + SiLU conv residual block with FiLM conditioning."""

    def __init__(self, width: int, cond_dim: int, groups: int = 8):
        super().__init__()
        g = math.gcd(groups, width) or 1
        self.norm1 = nn.GroupNorm(g, width)
        self.conv1 = nn.Conv3d(width, width, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g, width)
        self.conv2 = nn.Conv3d(width, width, kernel_size=3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * width)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).chunk(2, dim=1)
        scale = scale.view(scale.shape[0], -1, 1, 1, 1)
        shift = shift.view(shift.shape[0], -1, 1, 1, 1)
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.act(h))
        return x + h


class ResidualFlowModel(nn.Module):
    """Shared-across-scale velocity network for null-space residuals.

    Parameters
    ----------
    channels:
        Field channels (6 for displacement+velocity).
    width, depth:
        Feature width and number of FiLM residual blocks.
    embed_dim:
        Dimensionality of the time/resolution conditioning vector.
    context_channels:
        Optional extra conditioning channels (concatenated at ``2R``).
    factor:
        Upsampling factor per octave (2).
    use_checkpoint:
        If ``True``, wrap each residual block in
        ``torch.utils.checkpoint`` during training. Trades ~30% extra compute
        (one recomputed forward pass per block on backward) for activation
        memory that scales with ~1 block instead of ``depth`` blocks -- this
        matters a lot here since the LR-only branch unrolls several full
        forward passes through the network for the ODE integration, each of
        which would otherwise retain its own full activation stack.
    """

    def __init__(
        self,
        channels: int = 6,
        width: int = 64,
        depth: int = 4,
        embed_dim: int = 128,
        context_channels: int = 0,
        factor: int = 2,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.channels = int(channels)
        self.context_channels = int(context_channels)
        self.factor = int(factor)
        self.embed_dim = int(embed_dim)
        self.use_checkpoint = bool(use_checkpoint)

        self.t_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.r_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )

        in_ch = channels + channels + self.context_channels  # r_t + U(y_R) + context
        self.head = nn.Conv3d(in_ch, width, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [FiLMResBlock3d(width, embed_dim) for _ in range(depth)]
        )
        self.tail = nn.Sequential(
            nn.GroupNorm(math.gcd(8, width) or 1, width),
            nn.SiLU(),
            nn.Conv3d(width, channels, kernel_size=3, padding=1),
        )

    def _cond(self, t: torch.Tensor, R: torch.Tensor, batch: int, device) -> torch.Tensor:
        t = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1)
        R = torch.as_tensor(R, device=device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(batch)
        if R.numel() == 1:
            R = R.expand(batch)
        t_emb = self.t_mlp(sinusoidal_embedding(t, self.embed_dim))
        # log2(R) normalised so {64,128,256} -> compact range; robust to unseen R
        r_emb = self.r_mlp(sinusoidal_embedding(torch.log2(R.clamp_min(1.0)), self.embed_dim))
        return t_emb + r_emb

    def forward(
        self,
        r_t: torch.Tensor,
        t: torch.Tensor,
        y_R: torch.Tensor,
        R: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if r_t.dim() != 5:
            raise ValueError(f"r_t must be 5D, got {tuple(r_t.shape)}")
        b = r_t.shape[0]
        y_up = block_upsample(y_R, self.factor)
        if y_up.shape[-1] != r_t.shape[-1]:
            raise ValueError(
                f"U(y_R) size {tuple(y_up.shape)} != r_t size {tuple(r_t.shape)}; "
                "check that r_t is at 2R and y_R is at R."
            )
        feats = [r_t, y_up]
        if self.context_channels > 0:
            if context is None:
                raise ValueError("context_channels > 0 but no context tensor was passed")
            feats.append(context)
        x = torch.cat(feats, dim=1)
        cond = self._cond(t, R, b, r_t.device)
        x = self.head(x)
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = _ckpt.checkpoint(blk, x, cond, use_reentrant=False)
            else:
                x = blk(x, cond)
        return self.tail(x)
