"""Latent conditional flow velocity model ``v_theta(z_t, t, cond, R)``.

Operates on autoencoder latents (see :class:`ResidualAutoencoder`). A single
ResNet-style 3D network, shared across octaves, predicts the flow-matching
velocity of a latent ``z`` under a linear interpolant. Conditioning:

* ``z_t``  : current latent state ``(B, C_lat, n, n, n)``.
* ``cond`` : coarse field ``x_R`` at grid ``R`` (``C`` channels); resampled to the
  latent grid and concatenated. Pass a zero tensor for the null (unconditional)
  condition used by classifier-free guidance.
* ``t``    : flow time ``[0, 1]`` -> sinusoidal embedding (FiLM).
* ``R``    : resolution octave -> embedding from ``log2(R)`` (FiLM).

DiT is intentionally *not* used for the first pass (kept convolutional / FiLM).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt

from .residual_flow import FiLMResBlock3d, sinusoidal_embedding


class LatentFlowModel(nn.Module):
    """Shared-across-scale latent velocity network.

    Parameters
    ----------
    latent_channels:
        Number of AE latent channels ``C_lat``.
    cond_channels:
        Channels of the coarse conditioning field ``x_R`` (6).
    width, depth:
        Feature width and number of FiLM residual blocks.
    embed_dim:
        Time / resolution conditioning dimensionality.
    cond_mode:
        Interpolation mode used to resample ``cond`` to the latent grid.
    use_checkpoint:
        Wrap residual blocks in gradient checkpointing during training.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        cond_channels: int = 6,
        width: int = 64,
        depth: int = 4,
        embed_dim: int = 128,
        cond_mode: str = "trilinear",
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.latent_channels = int(latent_channels)
        self.cond_channels = int(cond_channels)
        self.embed_dim = int(embed_dim)
        self.cond_mode = str(cond_mode)
        self.use_checkpoint = bool(use_checkpoint)

        self.t_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.r_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )

        in_ch = latent_channels + cond_channels
        self.head = nn.Conv3d(in_ch, width, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [FiLMResBlock3d(width, embed_dim) for _ in range(depth)]
        )
        self.tail = nn.Sequential(
            nn.GroupNorm(math.gcd(8, width) or 1, width),
            nn.SiLU(),
            nn.Conv3d(width, latent_channels, kernel_size=3, padding=1),
        )

    def _cond_emb(self, t: torch.Tensor, R: torch.Tensor, batch: int, device) -> torch.Tensor:
        t = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1)
        R = torch.as_tensor(R, device=device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(batch)
        if R.numel() == 1:
            R = R.expand(batch)
        t_emb = self.t_mlp(sinusoidal_embedding(t, self.embed_dim))
        r_emb = self.r_mlp(sinusoidal_embedding(torch.log2(R.clamp_min(1.0)), self.embed_dim))
        return t_emb + r_emb

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        R: torch.Tensor,
    ) -> torch.Tensor:
        if z_t.dim() != 5:
            raise ValueError(f"z_t must be 5D, got {tuple(z_t.shape)}")
        b = z_t.shape[0]
        target_size = z_t.shape[-3:]
        if cond.shape[-3:] != target_size:
            align = None if self.cond_mode == "nearest" else False
            cond_rs = F.interpolate(cond, size=target_size, mode=self.cond_mode, align_corners=align)
        else:
            cond_rs = cond
        x = torch.cat([z_t, cond_rs], dim=1)
        emb = self._cond_emb(t, R, b, z_t.device)
        x = self.head(x)
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = _ckpt.checkpoint(blk, x, emb, use_reentrant=False)
            else:
                x = blk(x, emb)
        return self.tail(x)
