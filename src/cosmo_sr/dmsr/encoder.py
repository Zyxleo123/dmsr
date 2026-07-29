"""The LR condition encoder -- the one component shared between SSL and the flow.

Kept as a standalone module (rather than fused into the velocity network) for one
reason: Stage B initialises it from masked-reconstruction pretraining on *all*
training LR boxes, including the ~350 LR-only ones. That ablation is only
meaningful if exactly the same weights can be loaded into the flow, so the
encoder's interface is frozen:

    y (B, C, n, n, n)  ->  features (B, cond_channels, n, n, n)

i.e. it never changes spatial resolution. The flow block-upsamples those features
to the HR grid and feeds them to
:class:`~cosmo_sr.models.flow_unet.UNetResidualFlowModel` as ``context``; the SSL
task feeds them to a light decoder head instead.

Receptive field matters more than depth here: a ``crop_lr=8`` condition is only 8
voxels across, so the encoder uses dilated convolutions to see the whole crop
without downsampling it away.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


def _norm(channels: int, num_groups: int = 8) -> nn.Module:
    import math

    return nn.GroupNorm(math.gcd(num_groups, channels) or 1, channels)


class _ResBlock3d(nn.Module):
    """``Conv-Norm-SiLU-Conv-Norm`` + skip, with configurable dilation.

    Circular padding throughout: the boxes are periodic, and a crop of a periodic
    field is better served by wrap-around than by zeros at the faces (zeros would
    manufacture a fake density gradient at every crop boundary).
    """

    def __init__(self, channels: int, dilation: int = 1, num_groups: int = 8):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv3d(
            channels, channels, 3, padding=pad, dilation=dilation, padding_mode="circular"
        )
        self.norm1 = _norm(channels, num_groups)
        self.conv2 = nn.Conv3d(
            channels, channels, 3, padding=pad, dilation=dilation, padding_mode="circular"
        )
        self.norm2 = _norm(channels, num_groups)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class LRConditionEncoder(nn.Module):
    """Resolution-preserving encoder over an LR crop.

    Parameters
    ----------
    in_channels:
        Channels of the LR field (6 for ``disp+vel``, 3 for displacement-only).
    width:
        Hidden width.
    cond_channels:
        Output feature channels handed to the flow as ``context``.
    dilations:
        One residual block per entry, with that dilation. The default
        ``(1, 2, 4)`` reaches a 15-voxel receptive field, covering an 8^3 crop.
    """

    def __init__(
        self,
        in_channels: int = 6,
        width: int = 64,
        cond_channels: int = 32,
        dilations: Sequence[int] = (1, 2, 4),
        num_groups: int = 8,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.cond_channels = int(cond_channels)
        self.width = int(width)
        self.stem = nn.Conv3d(
            self.in_channels, self.width, 3, padding=1, padding_mode="circular"
        )
        self.blocks = nn.ModuleList(
            [_ResBlock3d(self.width, dilation=int(d), num_groups=num_groups) for d in dilations]
        )
        self.head = nn.Conv3d(self.width, self.cond_channels, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 5:
            raise ValueError(f"encoder expects (B, C, n, n, n), got {tuple(y.shape)}")
        if y.shape[1] != self.in_channels:
            raise ValueError(
                f"encoder built for {self.in_channels} channels, got {y.shape[1]}"
            )
        h = self.stem(y)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


class MaskedReconstructionHead(nn.Module):
    """Light decoder mapping encoder features back to LR field channels.

    Intentionally shallow (two convs). A heavy decoder would let the SSL task be
    solved in the decoder, leaving the encoder -- the part we actually transfer --
    under-trained.
    """

    def __init__(self, cond_channels: int = 32, width: int = 64, out_channels: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cond_channels, width, 3, padding=1, padding_mode="circular"),
            _norm(width),
            nn.SiLU(inplace=True),
            nn.Conv3d(width, out_channels, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


class LRMaskedAutoencoder(nn.Module):
    """Encoder + reconstruction head, used only during SSL pretraining.

    ``save_encoder`` writes a checkpoint that
    :func:`cosmo_sr.dmsr.flow.load_pretrained_encoder` can load into a flow whose
    encoder was built with the same constructor arguments.
    """

    def __init__(
        self,
        in_channels: int = 6,
        width: int = 64,
        cond_channels: int = 32,
        dilations: Sequence[int] = (1, 2, 4),
    ):
        super().__init__()
        self.encoder = LRConditionEncoder(
            in_channels=in_channels,
            width=width,
            cond_channels=cond_channels,
            dilations=dilations,
        )
        self.head = MaskedReconstructionHead(
            cond_channels=cond_channels, width=width, out_channels=in_channels
        )

    def forward(self, y_masked: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(y_masked))

    def encoder_config(self) -> dict:
        enc = self.encoder
        return {
            "in_channels": enc.in_channels,
            "width": enc.width,
            "cond_channels": enc.cond_channels,
        }

    def save_encoder(self, path) -> None:
        torch.save(
            {"encoder": self.encoder.state_dict(), "encoder_config": self.encoder_config()},
            path,
        )
