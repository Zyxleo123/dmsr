"""map2map-style "H-block" multi-scale velocity backbone for flow matching.

map2map's SRSGAN generator (``map2map/models/srsgan.py``) builds its output by
a *projection-and-accumulate* pattern: at every stage of its progressive-growing
stack, features are projected to the output channel count and summed with the
(upsampled) accumulator from the previous, coarser stage -- see the ``HBlock``
docstring there and Fig. 7(b) of the StyleGAN2 paper.

That accumulate-while-upsampling idea is architecturally orthogonal to *why*
map2map's cascade exists (a fixed-depth deterministic/adversarial single pass
that changes resolution as it goes). Flow matching instead needs a network
that runs at a *fixed* resolution -- input and output both at grid ``2R``,
called many times per octave to integrate the ODE -- with FiLM conditioning on
flow time ``t`` and octave ``R``. This module keeps the map2map projection/
accumulate pattern but drops the parts tied to its single-pass cascade
(no noise injection, no valid-conv cropping, no depth tied to an overall
``scale_factor``): it downsamples internally to build a small feature
pyramid, then reconstructs the velocity field by accumulating each level's
projected contribution while upsampling back to full resolution.

:class:`HBlockResidualFlowModel` wraps this behind the exact interface of
:class:`~cosmo_sr.models.residual_flow.ResidualFlowModel`, so it drops into
``train_flow`` / ``flow_sample`` unchanged: ``v = model(r_t, t, y_R, R)``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt

from ..operators.multiscale import block_upsample
from .residual_flow import sinusoidal_embedding


class _FiLMConvBlock3d(nn.Module):
    """GroupNorm + FiLM(t, R) + SiLU + same-padding conv."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8):
        super().__init__()
        g = math.gcd(groups, in_ch) or 1
        self.norm = nn.GroupNorm(g, in_ch)
        self.film = nn.Linear(cond_dim, 2 * in_ch)
        self.act = nn.SiLU()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).chunk(2, dim=1)
        scale = scale.view(scale.shape[0], -1, 1, 1, 1)
        shift = shift.view(shift.shape[0], -1, 1, 1, 1)
        h = self.norm(x) * (1 + scale) + shift
        return self.conv(self.act(h))


class _HBlockDown3d(nn.Module):
    """Refine at the current scale, then strided-conv downsample by 2."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8):
        super().__init__()
        self.refine = _FiLMConvBlock3d(in_ch, in_ch, cond_dim, groups)
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.refine(x, cond)
        return self.act(self.down(x))


class HBlockResidualFlowModel(nn.Module):
    """Multi-scale "H-block"-style velocity network ``v_phi(r_t, t, y_R, R)``.

    An internal feature pyramid (``num_levels`` strided-conv downsamples) is
    built from the concatenated input, then reconstructed coarse-to-fine by
    projecting each level's refined features to ``channels`` and summing with
    the upsampled accumulator from the coarser level -- map2map's H-block
    pattern, run at fixed resolution instead of growing the output.

    Parameters
    ----------
    channels:
        Field channels (6 for displacement+velocity).
    width:
        Feature width at the finest level; doubles per level (capped at 4x).
    num_levels:
        Number of internal downsample stages (0 = plain single-scale conv net).
    embed_dim:
        Dimensionality of the time/resolution conditioning vector.
    context_channels:
        Optional extra conditioning channels (concatenated at ``2R``).
    factor:
        Upsampling factor per octave (2), used to broadcast ``y_R`` to ``2R``.
    zero_init_tail:
        Zero-init every projection conv so the network outputs exactly zero
        velocity at init (matches the other backbones' safe-start behaviour).
    use_checkpoint:
        Gradient-checkpoint each down/refine stage during training.
    """

    def __init__(
        self,
        channels: int = 6,
        width: int = 64,
        num_levels: int = 3,
        embed_dim: int = 128,
        context_channels: int = 0,
        factor: int = 2,
        groups: int = 8,
        zero_init_tail: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if num_levels < 0:
            raise ValueError(f"num_levels must be >= 0, got {num_levels}")
        self.channels = int(channels)
        self.context_channels = int(context_channels)
        self.factor = int(factor)
        self.embed_dim = int(embed_dim)
        self.num_levels = int(num_levels)
        self.use_checkpoint = bool(use_checkpoint)

        self.t_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.r_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )

        in_ch = channels + channels + self.context_channels  # r_t + U(y_R) + context
        self.head = nn.Conv3d(in_ch, width, kernel_size=3, padding=1)

        # Widths per pyramid level, finest (0) -> coarsest (num_levels).
        widths = [width * (2 ** min(i, 2)) for i in range(num_levels + 1)]
        self.downs = nn.ModuleList(
            [_HBlockDown3d(widths[i], widths[i + 1], embed_dim, groups) for i in range(num_levels)]
        )
        self.refine = nn.ModuleList(
            [_FiLMConvBlock3d(w, w, embed_dim, groups) for w in widths]
        )
        self.proj = nn.ModuleList([nn.Conv3d(w, channels, kernel_size=1) for w in widths])
        if zero_init_tail:
            for p in self.proj:
                nn.init.zeros_(p.weight)
                if p.bias is not None:
                    nn.init.zeros_(p.bias)

    def _cond(self, t: torch.Tensor, R: torch.Tensor, batch: int, device) -> torch.Tensor:
        t = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1)
        R = torch.as_tensor(R, device=device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(batch)
        if R.numel() == 1:
            R = R.expand(batch)
        t_emb = self.t_mlp(sinusoidal_embedding(t, self.embed_dim))
        r_emb = self.r_mlp(sinusoidal_embedding(torch.log2(R.clamp_min(1.0)), self.embed_dim))
        return t_emb + r_emb

    def _maybe_ckpt(self, fn, *args):
        if self.use_checkpoint and self.training:
            return _ckpt.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

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

        # Encoder: finest (index 0, full res) -> coarsest (index num_levels).
        pyramid = [self.head(x)]
        for down in self.downs:
            pyramid.append(self._maybe_ckpt(down, pyramid[-1], cond))

        # Decoder: accumulate projections coarse -> fine (H-block pattern).
        y = None
        for level in range(len(pyramid) - 1, -1, -1):
            feat = self._maybe_ckpt(self.refine[level], pyramid[level], cond)
            proj = self.proj[level](feat)
            y = proj if y is None else F.interpolate(
                y, scale_factor=2, mode="trilinear", align_corners=False
            ) + proj
        return y
