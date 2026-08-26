"""Arms D, E and F: spatial proxies that keep the *layout* a summary discards.

Arm C reduces a tile's token grid to a permutation-invariant DeepSets pool, so
by construction it cannot use *where* a token sits relative to its neighbours.
These three arms each keep some spatial structure and let a 3-D convolution model
adjacency before anything is pooled:

``d`` :class:`SpatialTokenProxy`
    The SAME 8^3 token grid arm C reads (``tokens_c.npy``, not a second cache),
    but reshaped to ``(B, 2F, 8, 8, 8)`` and run through local 3^3 convolutions
    with GroupNorm/SiLU and one downsampling stage before global mean+max
    pooling. It is arm C's control: identical features, adjacency-aware model.
``e`` :class:`FullGridProxy`
    The complete 32^3 Eulerian phase-space grid -- five channels (log density,
    three bulk-subtracted mean velocities over ``v_ref``, one dispersion over
    ``v_ref``), candidate and candidate-minus-frozen concatenated to ten -- run
    through a strided ``10->16->32->64`` trunk. No hand-designed summaries at all;
    the convolutions build whatever the catalog needs from the raw cells.
``f`` :class:`SR2DiscriminatorProxy`
    The original SR2 discriminator, unchanged below the head: the exact
    20-channel :func:`cosmo_sr.reward.sr2_adversarial.critic_input` through the
    :class:`cosmo_sr.reward.sr2_adversarial.SR2Critic` convolutional body, with
    the scalar critic head replaced by the common 16-output catalog head. Its
    weights are freshly initialised -- no compatible SR2 discriminator checkpoint
    exists (see that module) -- so this is a test of the *architecture*, not a
    fine-tune of a pretrained critic.

All three predict the same three catalog count blocks as arms A-C, through the
shared :class:`cosmo_sr.reward.catalog_proxy.ProxyBase` head, and all three use
the frozen-relative residual parameterisation: the head is a signed log-count
residual reconstructed against the measured frozen tile summary, and its final
layer is zero-initialised so the untrained proxy predicts "no change from
frozen".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .catalog_proxy import ProxyBase, register_proxy
from .torch_reward import TorchSummary

__all__ = [
    "FullGridProxy",
    "FullGridProxyConfig",
    "SR2DiscriminatorProxy",
    "SR2DiscriminatorProxyConfig",
    "SpatialTokenProxy",
    "SpatialTokenProxyConfig",
]


# --------------------------------------------------------------------------- #
# A shared 3-D conv trunk
# --------------------------------------------------------------------------- #
def _conv_block(cin: int, cout: int, *, stride: int) -> List[nn.Module]:
    """One ``Conv3d -> GroupNorm -> SiLU`` block with a 3^3 kernel.

    GroupNorm rather than BatchNorm: these models sit next to the generator in an
    actor step where the batch is 1-2 tiles, and a batch statistic is meaningless
    (and, in the actor's per-sample reward, wrong) at that size. The group count
    is capped at 8 and never exceeds the channel count.
    """
    groups = max(1, min(8, cout))
    return [
        nn.Conv3d(cin, cout, kernel_size=3, stride=int(stride), padding=1),
        nn.GroupNorm(groups, cout),
        nn.SiLU(),
    ]


def _global_mean_max(h: torch.Tensor) -> torch.Tensor:
    """``(B, 2C)`` from ``(B, C, D, D, D)``: mean and max pooled and concatenated.

    Mean and max because they answer different questions -- "how much on average"
    and "how extreme at its peak" -- and a tile's catalog depends on both its
    typical density and its single densest clump.
    """
    flat = h.reshape(h.shape[0], h.shape[1], -1)
    return torch.cat([flat.mean(dim=2), flat.max(dim=2).values], dim=1)


class _ChannelStandardizer:
    """Per-channel mean/std buffers and the standardize op, mixed into a model.

    Per channel and not per cell/token: a cell is a position, and a per-position
    statistic would build the grid index into the model. Fitted on a batch of
    training rows the caller supplies (the full grid stack is too large to hold),
    exactly as the flat arms fit theirs on the training rows only.
    """

    def _init_standardizer(self, n_channels: int) -> None:
        self.register_buffer("feat_mean", torch.zeros(int(n_channels), dtype=torch.float64))
        self.register_buffer("feat_std", torch.ones(int(n_channels), dtype=torch.float64))

    def fit_standardizer(self, grids: torch.Tensor | np.ndarray) -> "_ChannelStandardizer":
        x = self._to_grid(torch.as_tensor(np.asarray(grids, dtype=np.float32)).to(torch.float64))
        flat = x.transpose(0, 1).reshape(x.shape[1], -1)          # (C, N*cells)
        # ``copy_`` keeps the registered buffers (and their device); assignment
        # would replace them with CPU tensors that ``.to(cuda)`` never moves.
        mean = flat.mean(dim=1).to(device=self.feat_mean.device,
                                   dtype=self.feat_mean.dtype)
        std = flat.std(dim=1, unbiased=False).to(device=self.feat_std.device,
                                                  dtype=self.feat_std.dtype)
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(torch.where(std < 1e-12, torch.ones_like(std), std))
        return self

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        shape = (1, -1) + (1,) * (x.dim() - 2)
        mean = self.feat_mean.to(device=x.device, dtype=x.dtype).reshape(shape)
        std = self.feat_std.to(device=x.device, dtype=x.dtype).reshape(shape)
        return (x - mean) / std


# --------------------------------------------------------------------------- #
# Arm D: a CNN on arm C's token grid
# --------------------------------------------------------------------------- #
@dataclass
class SpatialTokenProxyConfig:
    """Shape of one arm-D proxy. One fixed configuration; there is no sweep."""

    n_token_features: int = 8
    n_tokens: int = 512
    n_sub_bins: int = 6
    n_host_bins: int = 5
    channels: Tuple[int, ...] = (32, 64, 64)
    #: Which conv stages downsample (stride 2). One stage here: 8^3 -> 4^3, so
    #: the last conv still sees a neighbourhood rather than a single cell.
    strides: Tuple[int, ...] = (1, 2, 1)
    dropout: float = 0.0
    output_scale: Tuple[float, ...] = ()
    residual_head: bool = True

    def __post_init__(self) -> None:
        t = round(int(self.n_tokens) ** (1.0 / 3.0))
        if t ** 3 != int(self.n_tokens):
            raise ValueError(
                f"n_tokens={self.n_tokens} is not a perfect cube; arm D needs an "
                "ordered t^3 token grid to convolve over")
        if len(self.channels) != len(self.strides):
            raise ValueError("channels and strides must have the same length")

    @property
    def tokens_per_axis(self) -> int:
        return round(int(self.n_tokens) ** (1.0 / 3.0))

    @property
    def n_outputs(self) -> int:
        return int(self.n_sub_bins) + 2 * int(self.n_host_bins)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["channels"] = [int(c) for c in self.channels]
        d["strides"] = [int(s) for s in self.strides]
        d["output_scale"] = [float(x) for x in self.output_scale]
        return d

    @staticmethod
    def from_dict(d: Mapping) -> "SpatialTokenProxyConfig":
        return SpatialTokenProxyConfig(
            n_token_features=int(d["n_token_features"]),
            n_tokens=int(d["n_tokens"]),
            n_sub_bins=int(d["n_sub_bins"]),
            n_host_bins=int(d["n_host_bins"]),
            channels=tuple(int(c) for c in d.get("channels", (32, 64, 64))),
            strides=tuple(int(s) for s in d.get("strides", (1, 2, 1))),
            dropout=float(d.get("dropout", 0.0)),
            output_scale=tuple(float(x) for x in d.get("output_scale", ())),
            residual_head=bool(d.get("residual_head", False)),
        )


@register_proxy
class SpatialTokenProxy(ProxyBase, _ChannelStandardizer):
    """Arm D: a 3-D CNN over arm C's ``(B, 2, T, F)`` token grid.

    The tokens are the candidate block and the candidate-minus-frozen block, in
    the SAME order arm C stores them (C-order over the ``t^3`` grid). They are
    reshaped to ``(B, 2F, t, t, t)`` -- ``2F`` channels over the ordered token
    grid -- so a convolution sees which tokens are adjacent, which is exactly the
    information arm C's permutation-invariant pooling throws away. Everything from
    the pooled vector on is the shared head.
    """

    CONFIG = SpatialTokenProxyConfig

    def __init__(self, cfg: Optional[SpatialTokenProxyConfig] = None, *,
                 seed: Optional[int] = None):
        super().__init__()
        self.cfg = cfg or SpatialTokenProxyConfig()
        if seed is not None:
            torch.manual_seed(int(seed))
        cin = 2 * int(self.cfg.n_token_features)
        layers: List[nn.Module] = []
        c = cin
        for cout, stride in zip(self.cfg.channels, self.cfg.strides):
            layers += _conv_block(c, int(cout), stride=int(stride))
            if self.cfg.dropout > 0:
                layers.append(nn.Dropout3d(float(self.cfg.dropout)))
            c = int(cout)
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(2 * c, self.cfg.n_outputs)
        self._zero_init_head(self.head)
        self.seed = None if seed is None else int(seed)

        self._init_output_scale()
        self._init_standardizer(cin)

    def _init_output_scale(self) -> None:
        scale = list(self.cfg.output_scale) or [1.0] * self.cfg.n_outputs
        if len(scale) != self.cfg.n_outputs:
            raise ValueError(
                f"output_scale has {len(scale)} entries, expected {self.cfg.n_outputs}")
        self.register_buffer("output_scale", torch.as_tensor(scale, dtype=torch.float64))

    @property
    def n_features(self) -> int:
        return 2 * int(self.cfg.n_token_features)

    def _to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, 2, T, F) -> (B, 2F, t, t, t)`` in the stored C-order."""
        if tokens.dim() != 4 or tokens.shape[1] != 2 \
                or tokens.shape[-1] != int(self.cfg.n_token_features):
            raise ValueError(
                f"expected (B, 2, T, {self.cfg.n_token_features}) tokens, got "
                f"{tuple(tokens.shape)}")
        b, t = tokens.shape[0], self.cfg.tokens_per_axis
        x = tokens.reshape(b, 2, t, t, t, int(self.cfg.n_token_features))
        return x.permute(0, 1, 5, 2, 3, 4).reshape(b, self.n_features, t, t, t)

    def forward(self, tokens: torch.Tensor,
                frozen: Optional[TorchSummary] = None) -> Dict[str, torch.Tensor]:
        x = self._standardize(self._to_grid(tokens.to(device=self.feat_mean.device,
                                                      dtype=self.feat_mean.dtype)))
        h = self.body(x.to(next(self.body.parameters()).dtype))
        raw = self.head(_global_mean_max(h))
        return self._counts_from_raw(raw, frozen)


# --------------------------------------------------------------------------- #
# Arm E: a CNN on the full Eulerian phase-space grid
# --------------------------------------------------------------------------- #
@dataclass
class FullGridProxyConfig:
    """Shape of one arm-E proxy. One fixed configuration; there is no sweep."""

    n_grid_channels: int = 5
    grid_size: int = 32
    n_sub_bins: int = 6
    n_host_bins: int = 5
    #: Strided trunk 10 -> 16 -> 32 -> 64 over the 32^3 grid.
    channels: Tuple[int, ...] = (16, 32, 64)
    strides: Tuple[int, ...] = (2, 2, 2)
    dropout: float = 0.0
    output_scale: Tuple[float, ...] = ()
    residual_head: bool = True

    def __post_init__(self) -> None:
        if len(self.channels) != len(self.strides):
            raise ValueError("channels and strides must have the same length")

    @property
    def n_input_channels(self) -> int:
        """Candidate and candidate-minus-frozen stacked: ``2 * n_grid_channels``."""
        return 2 * int(self.n_grid_channels)

    @property
    def n_outputs(self) -> int:
        return int(self.n_sub_bins) + 2 * int(self.n_host_bins)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["channels"] = [int(c) for c in self.channels]
        d["strides"] = [int(s) for s in self.strides]
        d["output_scale"] = [float(x) for x in self.output_scale]
        return d

    @staticmethod
    def from_dict(d: Mapping) -> "FullGridProxyConfig":
        return FullGridProxyConfig(
            n_grid_channels=int(d.get("n_grid_channels", 5)),
            grid_size=int(d.get("grid_size", 32)),
            n_sub_bins=int(d["n_sub_bins"]),
            n_host_bins=int(d["n_host_bins"]),
            channels=tuple(int(c) for c in d.get("channels", (16, 32, 64))),
            strides=tuple(int(s) for s in d.get("strides", (2, 2, 2))),
            dropout=float(d.get("dropout", 0.0)),
            output_scale=tuple(float(x) for x in d.get("output_scale", ())),
            residual_head=bool(d.get("residual_head", False)),
        )


@register_proxy
class FullGridProxy(ProxyBase, _ChannelStandardizer):
    """Arm E: a strided 3-D CNN over the full ``(B, 2, 5, 32, 32, 32)`` grid.

    The two blocks are the candidate grid and the candidate-minus-frozen grid,
    reshaped to ``(B, 10, 32, 32, 32)``. No pooling to summaries beforehand: the
    whole point of this arm is that the convolutions, not a human, decide what to
    aggregate out of the raw phase-space cells.
    """

    CONFIG = FullGridProxyConfig

    def __init__(self, cfg: Optional[FullGridProxyConfig] = None, *,
                 seed: Optional[int] = None):
        super().__init__()
        self.cfg = cfg or FullGridProxyConfig()
        if seed is not None:
            torch.manual_seed(int(seed))
        cin = self.cfg.n_input_channels
        layers: List[nn.Module] = []
        c = cin
        for cout, stride in zip(self.cfg.channels, self.cfg.strides):
            layers += _conv_block(c, int(cout), stride=int(stride))
            if self.cfg.dropout > 0:
                layers.append(nn.Dropout3d(float(self.cfg.dropout)))
            c = int(cout)
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(2 * c, self.cfg.n_outputs)
        self._zero_init_head(self.head)
        self.seed = None if seed is None else int(seed)

        scale = list(self.cfg.output_scale) or [1.0] * self.cfg.n_outputs
        if len(scale) != self.cfg.n_outputs:
            raise ValueError(
                f"output_scale has {len(scale)} entries, expected {self.cfg.n_outputs}")
        self.register_buffer("output_scale", torch.as_tensor(scale, dtype=torch.float64))
        self._init_standardizer(cin)

    @property
    def n_features(self) -> int:
        return self.cfg.n_input_channels

    def _to_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """``(B, 2, C, D, D, D) -> (B, 2C, D, D, D)``, candidate then difference."""
        if grid.dim() != 6 or grid.shape[1] != 2 \
                or grid.shape[2] != int(self.cfg.n_grid_channels):
            raise ValueError(
                f"expected (B, 2, {self.cfg.n_grid_channels}, D, D, D) grid, got "
                f"{tuple(grid.shape)}")
        b = grid.shape[0]
        return grid.reshape(b, self.n_features, *grid.shape[3:])

    def forward(self, grid: torch.Tensor,
                frozen: Optional[TorchSummary] = None) -> Dict[str, torch.Tensor]:
        x = self._standardize(self._to_grid(grid.to(device=self.feat_mean.device,
                                                    dtype=self.feat_mean.dtype)))
        h = self.body(x.to(next(self.body.parameters()).dtype))
        raw = self.head(_global_mean_max(h))
        return self._counts_from_raw(raw, frozen)


# --------------------------------------------------------------------------- #
# Arm F: the SR2 discriminator architecture with a catalog head
# --------------------------------------------------------------------------- #
@dataclass
class SR2DiscriminatorProxyConfig:
    """Shape of one arm-F proxy. Mirrors the SR2 critic below the head."""

    in_channels: int = 20
    width: int = 64
    depth: int = 4
    n_sub_bins: int = 6
    n_host_bins: int = 5
    output_scale: Tuple[float, ...] = ()
    residual_head: bool = True

    @property
    def n_outputs(self) -> int:
        return int(self.n_sub_bins) + 2 * int(self.n_host_bins)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["output_scale"] = [float(x) for x in self.output_scale]
        return d

    @staticmethod
    def from_dict(d: Mapping) -> "SR2DiscriminatorProxyConfig":
        return SR2DiscriminatorProxyConfig(
            in_channels=int(d.get("in_channels", 20)),
            width=int(d.get("width", 64)),
            depth=int(d.get("depth", 4)),
            n_sub_bins=int(d["n_sub_bins"]),
            n_host_bins=int(d["n_host_bins"]),
            output_scale=tuple(float(x) for x in d.get("output_scale", ())),
            residual_head=bool(d.get("residual_head", False)),
        )


@register_proxy
class SR2DiscriminatorProxy(ProxyBase):
    """Arm F: :class:`SR2Critic`'s body with the scalar head swapped for 16 counts.

    The input is the exact 20-channel :func:`critic_input` the SR2 discriminator
    was trained with (6 upsampled LR + 6 candidate displacement/velocity + 8
    inverse-pixel-shuffled fine density), built per tile and passed in as
    ``features``; nothing is cached. Only the head differs from
    :class:`~cosmo_sr.reward.sr2_adversarial.SR2Critic`: a ``Linear(hidden, 16)``
    in place of ``Linear(hidden, 1)``. The weights are freshly initialised
    because no compatible SR2 discriminator checkpoint exists, so this measures an
    architecture, not a pretraining. And that architecture is SR2Critic's, which
    is an SR2-STYLE reconstruction of the discriminator, not a verified copy of
    the original (see SR2Critic's docstring): a negative arm-F result is evidence
    about this network, not proof that "the original SR2 discriminator failed".

    No input standardiser on purpose: keeping the untouched critic input is what
    makes this a clean discriminator-architecture test, and the critic's own
    GroupNorm handles the internal scale.
    """

    CONFIG = SR2DiscriminatorProxyConfig

    def __init__(self, cfg: Optional[SR2DiscriminatorProxyConfig] = None, *,
                 seed: Optional[int] = None):
        super().__init__()
        # Imported here rather than at module scope so a login-node import of the
        # arm registry does not drag in the critic's density stack.
        from .sr2_adversarial import SR2Critic

        self.cfg = cfg or SR2DiscriminatorProxyConfig()
        if seed is not None:
            torch.manual_seed(int(seed))
        critic = SR2Critic(width=int(self.cfg.width), depth=int(self.cfg.depth),
                           in_chan=int(self.cfg.in_channels))
        self.in_chan = int(critic.in_chan)
        self.body = critic.body            # the exact SR2 convolutional trunk
        hidden = int(critic.head.in_features)
        self.head = nn.Linear(hidden, self.cfg.n_outputs)
        self._zero_init_head(self.head)
        self.seed = None if seed is None else int(seed)

        scale = list(self.cfg.output_scale) or [1.0] * self.cfg.n_outputs
        if len(scale) != self.cfg.n_outputs:
            raise ValueError(
                f"output_scale has {len(scale)} entries, expected {self.cfg.n_outputs}")
        self.register_buffer("output_scale", torch.as_tensor(scale, dtype=torch.float64))

    @property
    def n_features(self) -> int:
        return int(self.in_chan)

    def forward(self, critic_input_tensor: torch.Tensor,
                frozen: Optional[TorchSummary] = None) -> Dict[str, torch.Tensor]:
        x = critic_input_tensor
        if x.dim() != 5 or x.shape[1] != self.in_chan:
            raise ValueError(
                f"arm F expects the {self.in_chan}-channel SR2 critic input "
                f"(B, {self.in_chan}, N, N, N), got {tuple(x.shape)}")
        h = self.body(x.to(next(self.body.parameters()).dtype)).mean(dim=(2, 3, 4))
        raw = self.head(h)
        return self._counts_from_raw(raw, frozen)
