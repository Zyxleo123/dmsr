"""map2map-style 3D U-Net velocity backbone with optional modern gadgets.

:class:`Map2MapUNet3D` is a clean-room reimplementation of the U-Net design
used by map2map (https://github.com/eelregit/map2map, GPL-3.0 -- no code was
copied): a 3D conv encoder-decoder with two stride-2 conv downsample stages, a
bottleneck, two transposed-conv upsample stages with skip concatenation,
BatchNorm + LeakyReLU, valid (unpadded) convolutions, and an optional global
input->output bypass. Because valid convolutions shrink the volume, skips and
the bypass are aligned by center-cropping (map2map's ``narrow_like``).

Every departure from that baseline is a config switch that defaults off, so
``map2map_compat=True`` recovers the plain behaviour exactly:

* ``padding="same"``    -- shape-preserving convs (required for flow training).
* ``norm`` / ``activation`` -- GroupNorm/InstanceNorm/none, SiLU/ReLU.
* ``use_resblocks``     -- residual double-conv stages.
* ``use_film``          -- FiLM conditioning on flow time ``t`` and octave ``R``.
* ``use_attention``     -- multi-head self-attention at the bottleneck.
* ``zero_init_tail``    -- zero-init the output conv (velocity starts at 0).
* ``use_checkpoint``    -- gradient checkpointing of the conv stages.

:class:`UNetResidualFlowModel` wraps the U-Net behind the exact interface of
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


def center_crop_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Center-crop the spatial dims of ``x`` to match ``ref`` (map2map ``narrow_like``)."""
    for d in range(2, x.dim()):
        extra = x.shape[d] - ref.shape[d]
        if extra < 0:
            raise ValueError(
                f"center_crop_like: x {tuple(x.shape)} smaller than ref {tuple(ref.shape)} "
                f"along dim {d}"
            )
        if extra > 0:
            x = x.narrow(d, extra // 2, ref.shape[d])
    return x


class ChannelGroupNorm3d(nn.Module):
    """GroupNorm whose statistics are taken over channels only, never over space.

    ``nn.GroupNorm`` reduces over ``(C/G, D, H, W)``, so every output voxel depends
    on the mean and variance of the WHOLE input window. That makes the effective
    receptive field window-global regardless of kernel reach, and it makes a crop
    of size A and a crop of size B renormalise the same physical region
    differently. Measured on ``t13_unconstrained_s0`` (set14): the LR conditioning
    features of a fixed 8^3 region computed in an 8^3 window correlate only
    r = 0.474 with the same region computed in the full 64^3 box, and **92.7% of
    that discrepancy is a per-channel affine rescale** -- i.e. it is this layer,
    not missing spatial information. Training used ``crop_lr=8`` while full-box
    inference tiles at 32^3 LR windows, so the shift is also a train/eval
    mismatch.

    This variant normalises the ``C/G`` channels of each group at each voxel
    independently, so it is strictly pointwise and carries no spatial dependence.
    Same parameter count and same ``weight``/``bias`` semantics as ``nn.GroupNorm``.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        if num_channels % num_groups:
            raise ValueError(f"channels {num_channels} not divisible by groups {num_groups}")
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0], x.shape[1]
        g = self.num_groups
        xs = x.reshape(b, g, c // g, *x.shape[2:])
        mean = xs.mean(dim=2, keepdim=True)
        var = xs.var(dim=2, keepdim=True, unbiased=False)
        xs = (xs - mean) * torch.rsqrt(var + self.eps)
        x = xs.reshape(b, c, *x.shape[2:])
        shape = (1, -1) + (1,) * (x.dim() - 2)
        return x * self.weight.view(shape) + self.bias.view(shape)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}"


def _make_norm(kind: str, channels: int, num_groups: int = 8) -> nn.Module:
    kind = kind.lower()
    if kind == "batch":
        return nn.BatchNorm3d(channels)
    if kind == "group":
        return nn.GroupNorm(math.gcd(num_groups, channels) or 1, channels)
    if kind == "channel":
        return ChannelGroupNorm3d(math.gcd(num_groups, channels) or 1, channels)
    if kind == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if kind == "none":
        return nn.Identity()
    raise ValueError(
        f"Unknown norm {kind!r} (use 'batch'|'group'|'channel'|'instance'|'none')")


def _make_act(kind: str) -> nn.Module:
    kind = kind.lower()
    if kind == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    if kind == "silu":
        return nn.SiLU(inplace=True)
    if kind == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unknown activation {kind!r} (use 'leaky_relu'|'silu'|'relu')")


class ConvBlock3d(nn.Module):
    """Double conv stage ``[Conv -> Norm -> Act] x 2`` with optional FiLM/residual.

    FiLM (when ``cond_dim`` is set) modulates the second norm output with a
    per-channel affine predicted from the conditioning vector, mirroring
    :class:`~cosmo_sr.models.residual_flow.FiLMResBlock3d`. With ``residual=True``
    the (1x1-projected, center-cropped) input is added before the last
    activation, so the block works under both same and valid padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: str = "valid",
        norm: str = "batch",
        num_groups: int = 8,
        activation: str = "leaky_relu",
        cond_dim: Optional[int] = None,
        residual: bool = False,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        pad = kernel_size // 2 if padding == "same" else 0
        pm = dict(padding_mode=padding_mode) if pad else {}
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=pad, **pm)
        self.norm1 = _make_norm(norm, out_channels, num_groups)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=pad, **pm)
        self.norm2 = _make_norm(norm, out_channels, num_groups)
        self.act = _make_act(activation)
        self.film = nn.Linear(cond_dim, 2 * out_channels) if cond_dim else None
        self.residual = bool(residual)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if residual and in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        if self.film is not None:
            if cond is None:
                raise ValueError("ConvBlock3d built with cond_dim but no cond was passed")
            scale, shift = self.film(cond).chunk(2, dim=1)
            h = h * (1 + scale.view(scale.shape[0], -1, 1, 1, 1)) + shift.view(
                shift.shape[0], -1, 1, 1, 1
            )
        if self.residual:
            idt = x if self.skip is None else self.skip(x)
            h = h + center_crop_like(idt, h)
        return self.act(h)


class Resample3d(nn.Module):
    """Stride-2 conv down / transposed-conv up, followed by norm + act."""

    def __init__(
        self,
        channels: int,
        direction: str,
        norm: str = "batch",
        num_groups: int = 8,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        if direction == "down":
            self.conv = nn.Conv3d(channels, channels, kernel_size=2, stride=2)
        elif direction == "up":
            self.conv = nn.ConvTranspose3d(channels, channels, kernel_size=2, stride=2)
        else:
            raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")
        self.norm = _make_norm(norm, channels, num_groups)
        self.act = _make_act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class AttentionBlock3d(nn.Module):
    """Residual multi-head self-attention over flattened ``D*H*W`` tokens."""

    def __init__(self, channels: int, heads: int = 4, num_groups: int = 8):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
        self.heads = heads
        self.norm = nn.GroupNorm(math.gcd(num_groups, channels) or 1, channels)
        self.qkv = nn.Conv3d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, d * h * w)
        q, k, v = qkv.transpose(-2, -1).unbind(1)  # each (b, heads, tokens, head_dim)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-2, -1).reshape(b, c, d, h, w)
        return x + self.proj(out)


class Map2MapUNet3D(nn.Module):
    """map2map-style 3D U-Net with configurable, default-off modern gadgets.

    Parameters
    ----------
    in_channels, out_channels:
        I/O channel counts. Feature width is constant (``width``) across levels,
        as in map2map (no channel doubling).
    num_levels:
        Number of downsample stages (2 in map2map).
    padding:
        ``"valid"`` (unpadded convs, output smaller than input -- map2map) or
        ``"same"`` (shape-preserving -- required for flow training).
    global_bypass:
        ``"auto"`` adds a center-cropped input->output skip only when
        ``in_channels == out_channels``; ``True``/``False`` force it.
    cond_dim:
        Conditioning vector width; enables FiLM inside every conv stage.
    use_checkpoint:
        Checkpoint each conv stage during training (memory for compute).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        num_levels: int = 2,
        blocks_per_level: int = 1,
        kernel_size: int = 3,
        padding: str = "valid",
        norm: str = "batch",
        num_groups: int = 8,
        activation: str = "leaky_relu",
        use_resblocks: bool = False,
        cond_dim: Optional[int] = None,
        use_attention: bool = False,
        attention_heads: int = 4,
        zero_init_tail: bool = False,
        use_checkpoint: bool = False,
        global_bypass: str | bool = "auto",
        padding_mode: str = "zeros",
    ):
        super().__init__()
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        if blocks_per_level < 1:
            raise ValueError(f"blocks_per_level must be >= 1, got {blocks_per_level}")
        if padding not in ("same", "valid"):
            raise ValueError(f"padding must be 'same' or 'valid', got {padding!r}")
        if global_bypass == "auto":
            global_bypass = in_channels == out_channels
        elif global_bypass and in_channels != out_channels:
            raise ValueError(
                f"global_bypass=True needs in_channels == out_channels, "
                f"got {in_channels} != {out_channels}"
            )
        self.num_levels = int(num_levels)
        self.use_checkpoint = bool(use_checkpoint)
        self.global_bypass = bool(global_bypass)

        def stage(in_ch: int) -> nn.ModuleList:
            blocks = []
            for i in range(blocks_per_level):
                blocks.append(ConvBlock3d(
                    in_ch if i == 0 else width, width,
                    kernel_size=kernel_size, padding=padding, norm=norm,
                    num_groups=num_groups, activation=activation,
                    cond_dim=cond_dim, residual=use_resblocks,
                    padding_mode=padding_mode,
                ))
            return nn.ModuleList(blocks)

        def resample(direction: str) -> Resample3d:
            return Resample3d(width, direction, norm=norm, num_groups=num_groups,
                              activation=activation)

        self.enc_stages = nn.ModuleList(
            [stage(in_channels if l == 0 else width) for l in range(num_levels)]
        )
        self.downs = nn.ModuleList([resample("down") for _ in range(num_levels)])
        self.bottleneck = stage(width)
        self.attn = (
            AttentionBlock3d(width, heads=attention_heads, num_groups=num_groups)
            if use_attention else None
        )
        self.ups = nn.ModuleList([resample("up") for _ in range(num_levels)])
        self.dec_stages = nn.ModuleList([stage(2 * width) for _ in range(num_levels)])
        pad = kernel_size // 2 if padding == "same" else 0
        pm = dict(padding_mode=padding_mode) if pad else {}
        self.conv_out = nn.Conv3d(width, out_channels, kernel_size, padding=pad, **pm)
        if zero_init_tail:
            nn.init.zeros_(self.conv_out.weight)
            nn.init.zeros_(self.conv_out.bias)

    def _run_stage(self, blocks: nn.ModuleList, x: torch.Tensor,
                   cond: Optional[torch.Tensor]) -> torch.Tensor:
        for blk in blocks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = _ckpt.checkpoint(blk, x, cond, use_reentrant=False)
            else:
                x = blk(x, cond)
        return x

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"expected 5D input (B, C, D, H, W), got {tuple(x.shape)}")
        skips = []
        h = x
        for l in range(self.num_levels):
            h = self._run_stage(self.enc_stages[l], h, cond)
            skips.append(h)
            h = self.downs[l](h)
        h = self._run_stage(self.bottleneck, h, cond)
        if self.attn is not None:
            h = self.attn(h)
        for l in reversed(range(self.num_levels)):
            h = self.ups[l](h)
            h = torch.cat([center_crop_like(skips[l], h), h], dim=1)
            h = self._run_stage(self.dec_stages[l], h, cond)
        out = self.conv_out(h)
        if self.global_bypass:
            out = out + center_crop_like(x, out)
        return out


class UNetResidualFlowModel(nn.Module):
    """U-Net velocity network with the :class:`ResidualFlowModel` interface.

    Conditioning matches ``ResidualFlowModel``: ``U(y_R)`` (and optional
    ``context``) are concatenated to ``r_t`` channel-wise; ``t`` and ``R`` enter
    via FiLM when ``use_film=True`` and are ignored otherwise (plain U-Net).

    ``map2map_compat=True`` forces the plain map2map-style configuration
    (valid padding, BatchNorm, LeakyReLU, all gadgets off). For actual flow
    training use ``padding="same"`` so the velocity matches ``r_t``'s grid --
    then input sizes must be divisible by ``2**num_levels``.
    """

    def __init__(
        self,
        channels: int = 6,
        width: int = 64,
        num_levels: int = 2,
        blocks_per_level: int = 1,
        kernel_size: int = 3,
        padding: str = "same",
        norm: str = "group",
        num_groups: int = 8,
        activation: str = "silu",
        use_resblocks: bool = False,
        use_film: bool = False,
        embed_dim: int = 128,
        use_attention: bool = False,
        attention_heads: int = 4,
        zero_init_tail: bool = False,
        use_checkpoint: bool = False,
        context_channels: int = 0,
        factor: int = 2,
        map2map_compat: bool = False,
        global_bypass: str | bool = "auto",
        padding_mode: str = "zeros",
    ):
        super().__init__()
        if map2map_compat:
            padding, norm, activation = "valid", "batch", "leaky_relu"
            use_resblocks = use_film = use_attention = False
            zero_init_tail = use_checkpoint = False
            padding_mode = "zeros"
        self.channels = int(channels)
        self.context_channels = int(context_channels)
        self.factor = int(factor)
        self.embed_dim = int(embed_dim)
        self.use_film = bool(use_film)
        self.padding = padding

        if self.use_film:
            self.t_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
            )
            self.r_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
            )

        in_ch = channels + channels + self.context_channels  # r_t + U(y_R) + context
        self.unet = Map2MapUNet3D(
            in_channels=in_ch,
            out_channels=channels,
            width=width,
            num_levels=num_levels,
            blocks_per_level=blocks_per_level,
            kernel_size=kernel_size,
            padding=padding,
            norm=norm,
            num_groups=num_groups,
            activation=activation,
            use_resblocks=use_resblocks,
            cond_dim=embed_dim if self.use_film else None,
            use_attention=use_attention,
            attention_heads=attention_heads,
            zero_init_tail=zero_init_tail,
            use_checkpoint=use_checkpoint,
            global_bypass=global_bypass,
            padding_mode=padding_mode,
        )

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
        cond = self._cond(t, R, r_t.shape[0], r_t.device) if self.use_film else None
        out = self.unet(x, cond)
        if self.padding == "same" and out.shape != r_t.shape:
            raise ValueError(
                f"output {tuple(out.shape)} != r_t {tuple(r_t.shape)}; with "
                f"padding='same' the input size must be divisible by "
                f"2**num_levels ({2 ** self.unet.num_levels})"
            )
        return out
