"""Generator factory and lightweight wrappers.

We keep our generator independent of ``map2map`` internals. The factory returns
our :class:`SimpleSRGenerator` by default. A trivial :class:`NearestUpsampler`
is provided for testing tiled inference against direct inference (it is an exact,
seam-free, deterministic upsampler).
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_baseline import SimpleSRGenerator, NullSpaceSRGenerator

_REGISTRY = {
    "SimpleSRGenerator": SimpleSRGenerator,
    "NullSpaceSRGenerator": NullSpaceSRGenerator,
}


class NearestUpsampler(nn.Module):
    """Parameter-free nearest-neighbour upsampler (exact and seam-free).

    Useful as an identity-like reference model for inference tests: tiled
    inference must equal direct full-field inference.
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 6, scale_factor: int = 8):
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("NearestUpsampler requires in_channels == out_channels")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = int(scale_factor)

    def forward(self, y_lr: torch.Tensor) -> torch.Tensor:
        if y_lr.dim() != 5:
            raise ValueError(
                f"Expected 5D input (B, C, N, N, N), got shape {tuple(y_lr.shape)}"
            )
        return F.interpolate(y_lr, scale_factor=self.scale_factor, mode="nearest")


def build_generator(name: str = "SimpleSRGenerator", **kwargs: Any) -> nn.Module:
    """Instantiate a generator by name.

    Currently supports ``"SimpleSRGenerator"`` and ``"NearestUpsampler"``.
    """
    if name == "NearestUpsampler":
        return NearestUpsampler(
            in_channels=kwargs.get("in_channels", 6),
            out_channels=kwargs.get("out_channels", 6),
            scale_factor=kwargs.get("scale_factor", 8),
        )
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown generator {name!r}; available: {sorted(_REGISTRY) + ['NearestUpsampler']}"
        )
    cls = _REGISTRY[name]
    return cls(**kwargs)


def model_config_kwargs(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract generator kwargs from a config ``model`` section (drops ``name``)."""
    kwargs = dict(model_cfg)
    kwargs.pop("name", None)
    return kwargs
