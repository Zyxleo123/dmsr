"""Operator-conditioned HR denoiser ``D_psi(x_t, t, operator_context) -> x0_hat``.

One shared 3D U-Net (the :class:`Map2MapUNet3D` backbone) predicts the clean HR
field ``x0`` from a noisy HR-grid input, conditioned via FiLM on

* diffusion time ``t``            (sinusoidal + MLP),
* operator ``kind``               (identity / fixed / shifted),
* subcell shift ``g``             (normalised, sinusoidal per-axis + MLP),
* scale factor ``s``              (``log2 s``; constant in the factor-2 study).

The *same* network serves two input constructions (built in the training
branches, see ``losses/ambient_denoise.py``):

    clean   :  x_t   = alpha(t) x + sigma(t) eps                      (kind="identity")
    ambient :  input = H_g^+( alpha(t) y + sigma(t) H_g eps )         (kind="shifted", g)

so the ambient branch feeds an HR-grid *backprojection* of a noisy measurement
and the network never sees clean HR for LR-only data.
"""
from __future__ import annotations

import copy
import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from .flow_unet import Map2MapUNet3D
from .residual_flow import sinusoidal_embedding

KIND_TO_IDX = {"identity": 0, "fixed": 1, "shifted": 2}
ShiftLike = Union[Sequence[int], torch.Tensor, None]
KindLike = Union[str, torch.Tensor]


class CosineSchedule:
    """Variance-preserving cosine schedule: ``alpha=cos(pi t/2)``, ``sigma=sin(pi t/2)``.

    ``t in [0, 1]`` (``t=0`` clean, ``t=1`` pure noise); ``alpha^2 + sigma^2 = 1``.
    """

    def alpha_sigma(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.cos(0.5 * math.pi * t)
        s = torch.sin(0.5 * math.pi * t)
        return a, s

    def broadcast(self, t: torch.Tensor, ndim: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        """``alpha, sigma`` reshaped to ``(B, 1, 1, 1, 1)`` for a 5D field."""
        a, s = self.alpha_sigma(t.reshape(-1))
        shape = (-1,) + (1,) * (ndim - 1)
        return a.reshape(shape), s.reshape(shape)


class ModelEMA:
    """Exponential moving average of a model's parameters (kept as ``.module``)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ep, p in zip(self.module.parameters(), model.parameters()):
            ep.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for eb, b in zip(self.module.buffers(), model.buffers()):
            eb.copy_(b)


class OperatorConditionedDenoiser(nn.Module):
    """Shared U-Net predicting clean HR ``x0`` under a specified measurement operator."""

    def __init__(
        self,
        channels: int = 6,
        width: int = 64,
        num_levels: int = 3,
        blocks_per_level: int = 1,
        embed_dim: int = 128,
        factor: int = 2,
        norm: str = "group",
        num_groups: int = 8,
        activation: str = "silu",
        use_resblocks: bool = True,
        use_attention: bool = True,
        attention_heads: int = 4,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.factor = int(factor)
        self.embed_dim = int(embed_dim)

        self.t_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.kind_emb = nn.Embedding(len(KIND_TO_IDX), embed_dim)
        self.shift_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )

        self.backbone = Map2MapUNet3D(
            in_channels=channels,
            out_channels=channels,
            width=width,
            num_levels=num_levels,
            blocks_per_level=blocks_per_level,
            padding="same",
            norm=norm,
            num_groups=num_groups,
            activation=activation,
            use_resblocks=use_resblocks,
            cond_dim=embed_dim,          # enables FiLM in every conv stage
            use_attention=use_attention,
            attention_heads=attention_heads,
            use_checkpoint=use_checkpoint,
            global_bypass=False,          # predict x0 directly (no raw input skip)
        )

    # -- conditioning ------------------------------------------------------- #
    def _as_kind_idx(self, kind: KindLike, batch: int, device) -> torch.Tensor:
        if isinstance(kind, str):
            return torch.full((batch,), KIND_TO_IDX[kind], dtype=torch.long, device=device)
        idx = torch.as_tensor(kind, device=device, dtype=torch.long).reshape(-1)
        return idx.expand(batch) if idx.numel() == 1 else idx

    def _as_shift(self, shift: ShiftLike, batch: int, device) -> torch.Tensor:
        if shift is None:
            shift = (0, 0, 0)
        g = torch.as_tensor(shift, device=device, dtype=torch.float32)
        if g.dim() == 1:
            g = g.reshape(1, 3).expand(batch, 3)
        if g.shape != (batch, 3):
            raise ValueError(f"shift must broadcast to ({batch}, 3), got {tuple(g.shape)}")
        return g

    def _cond(self, t, shift, kind, batch, device) -> torch.Tensor:
        t = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1)
        t = t.expand(batch) if t.numel() == 1 else t
        cond = self.t_mlp(sinusoidal_embedding(t, self.embed_dim))
        cond = cond + self.kind_emb(self._as_kind_idx(kind, batch, device))
        g = self._as_shift(shift, batch, device) / float(self.factor)  # normalise to [0,1)
        g_emb = sum(sinusoidal_embedding(g[:, ax], self.embed_dim) for ax in range(3))
        cond = cond + self.shift_mlp(g_emb)
        return cond

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        shift: ShiftLike = None,
        kind: KindLike = "shifted",
    ) -> torch.Tensor:
        if x_t.dim() != 5:
            raise ValueError(f"x_t must be 5D (B, C, N, N, N), got {tuple(x_t.shape)}")
        cond = self._cond(t, shift, kind, x_t.shape[0], x_t.device)
        return self.backbone(x_t, cond)
