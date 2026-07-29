"""Checkpoint-compatible SR2/SRS generator with **explicit** noise control.

The upstream generator (``external/SRS-map2map/map2map/models/srsgan.py``) draws
its stochasticity with ``torch.randn_like`` inside :class:`AddNoise`, six times
per forward pass (three ``SkipBlock``s x two injections). The only handle the
rest of the repo has on that is the *global* torch RNG, which makes per-tile
reproducibility, noise replay and noise optimisation impossible.

This module re-implements ``G``/``SkipBlock``/``AddNoise`` **locally** (upstream
is read-only) with three additions:

* ``forward(x, noise=...)`` accepts an explicit noise tensor per injection site;
* ``record=True`` returns the internally sampled noises so a realisation can be
  replayed bit-exactly;
* omitting noise falls back to ``torch.randn_like`` -- the original stochastic
  distribution, drawn in the original order, so a global ``manual_seed`` gives
  byte-identical output to the upstream module (see ``tests/tts``).

State-dict keys are identical to upstream, so ``SRmodel/G_z0.pt`` loads directly.

Site naming
-----------
``G(scale_factor=8)`` has ``log2(8) = 3`` blocks; we call them ``coarse`` (block
0, acting at the LR resolution), ``middle`` (block 1) and ``fine`` (block 2,
acting at 4x/8x). Sites are indexed ``z0..z5`` in forward order::

    coarse: z0 (pre-upsample, res x1)   z1 (post-upsample+conv, res x2)
    middle: z2 (res x2)                 z3 (res x4)
    fine:   z4 (res x4)                 z5 (res x8)

The pretrained ``G_z0.pt`` has ``|std|`` of order 1e-3 at z0..z4 and 5e-2 at z5,
i.e. its stochasticity is overwhelmingly a *fine-scale* effect. That is not an
assumption of this code, just a fact worth knowing before interpreting any
test-time-scaling result.

Faithfulness note (the double add)
----------------------------------
Upstream ``AddNoise.forward`` does ``x = x + noise`` and then ``return x + noise``
-- the noise is added **twice**. That looks like a typo, but the checkpoint was
trained with it, so it is part of the learned function and is reproduced here
verbatim. :func:`AddNoise.forward` is the only place this matters.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "AddNoise",
    "ControlledG",
    "NOISE_SITES",
    "STAGE_NAMES",
    "STAGE_SITES",
    "flatten_noise",
    "load_controlled_generator",
    "noise_site_layout",
    "site_shapes",
    "unflatten_noise",
]

STAGE_NAMES: Tuple[str, str, str] = ("coarse", "middle", "fine")
#: ``z0..z5`` in forward order.
NOISE_SITES: Tuple[str, ...] = tuple(f"z{i}" for i in range(6))
#: stage name -> the two site names it owns.
STAGE_SITES: Dict[str, Tuple[str, str]] = {
    "coarse": ("z0", "z1"),
    "middle": ("z2", "z3"),
    "fine": ("z4", "z5"),
}


def _narrow_by(a: torch.Tensor, c: int) -> torch.Tensor:
    """Symmetric narrowing on every spatial axis (copy of ``map2map.models.narrow``)."""
    ind = (slice(None),) * 2 + (slice(c, -c),) * (a.dim() - 2)
    return a[ind]


class Upsample2(nn.Module):
    """``Resampler(3, 2)``: trilinear x2, then drop the inaccurate 1-cell edge.

    Parameter-free, so its presence at ``conv[1]`` costs no state-dict keys and
    the upstream index layout (``conv.0``, ``conv.2``, ``conv.4``, ``conv.5``)
    is preserved.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return _narrow_by(x, 1)


class AddNoise(nn.Module):
    """Add (or concatenate) per-voxel noise, matching upstream exactly.

    Parameters
    ----------
    cat:
        Concatenate instead of add. The pretrained ``G_z0``/``G_z2`` use
        ``cat=False``; ``cat=True`` is reproduced for completeness.
    chan:
        Number of learned per-channel noise scales (``self.std``).
    """

    def __init__(self, cat: bool, chan: int = 1):
        super().__init__()
        self.cat = bool(cat)
        if not self.cat:
            self.std = nn.Parameter(torch.zeros([chan]))

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        """Returns ``(out, z_used)``. ``z`` has the shape of ``x[:, :1]``."""
        if z is None:
            z = torch.randn_like(x[:, :1])
        elif tuple(z.shape) != tuple(x[:, :1].shape):
            raise ValueError(
                f"explicit noise shape {tuple(z.shape)} != expected {tuple(x[:, :1].shape)}"
            )
        else:
            z = z.to(dtype=x.dtype)

        if self.cat:
            noise = z
            x = torch.cat([x, noise], dim=1)
        else:
            std_shape = (-1,) + (1,) * (x.dim() - 2)
            noise = self.std.view(std_shape) * z
            x = x + noise
        # NOTE: upstream adds `noise` a second time here. Kept verbatim -- the
        # pretrained weights were fit with this behaviour.
        return x + noise, z


class SkipBlock(nn.Module):
    """StyleGAN2 "I" block; module layout mirrors upstream for state-dict parity.

    ``self.conv`` keeps the upstream ``nn.Sequential`` indices
    ``[AddNoise, upsample, Conv3d, LeakyReLU, AddNoise, Conv3d, LeakyReLU]`` so
    keys read ``conv.0.std``, ``conv.2.weight``, ``conv.4.std``, ``conv.5.weight``.
    The forward pass walks that list explicitly so noise can be injected.
    """

    def __init__(self, prev_chan: int, next_chan: int, out_chan: int, cat_noise: bool):
        super().__init__()
        self.cat_noise = bool(cat_noise)
        self.upsample = Upsample2()
        c = int(cat_noise)
        self.conv = nn.Sequential(
            AddNoise(cat_noise, chan=prev_chan),
            self.upsample,
            nn.Conv3d(prev_chan + c, next_chan, 3),
            nn.LeakyReLU(0.2, True),
            AddNoise(cat_noise, chan=next_chan),
            nn.Conv3d(next_chan + c, next_chan, 3),
            nn.LeakyReLU(0.2, True),
        )
        self.proj = nn.Sequential(
            nn.Conv3d(next_chan + c, out_chan, 1),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x, y, z_a=None, z_b=None):
        x, z_a = self.conv[0](x, z_a)          # AddNoise
        x = self.conv[1](x)                    # Resampler(3, 2)
        x = self.conv[3](self.conv[2](x))      # Conv3d + LeakyReLU
        x, z_b = self.conv[4](x, z_b)          # AddNoise
        x = self.conv[6](self.conv[5](x))      # Conv3d + LeakyReLU

        if y is None:
            y = self.proj(x)
        else:
            y = self.upsample(y)
            y = _narrow_by(y, 2)
            y = y + self.proj(x)
        return x, y, z_a, z_b


class ControlledG(nn.Module):
    """SR2 generator with explicit / recordable noise.

    Drop-in for ``map2map.models.srsgan.G``: same constructor signature, same
    state-dict keys, same output for the same noise.

    Examples
    --------
    >>> G = ControlledG(6, 6, 8)                     # doctest: +SKIP
    >>> y, z = G(lr, record=True)                    # doctest: +SKIP
    >>> y2, _ = G(lr, noise=z)                       # doctest: +SKIP
    >>> torch.allclose(y, y2)                        # doctest: +SKIP
    True
    """

    def __init__(self, in_chan: int, out_chan: int, scale_factor: int = 16,
                 chan_base: int = 512, chan_min: int = 64, chan_max: int = 512,
                 cat_noise: bool = False):
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.num_blocks = round(log2(self.scale_factor))
        if chan_min > chan_max:
            raise ValueError("chan_min must be <= chan_max")

        def chan(b: int) -> int:
            return min(max(chan_base >> b, chan_min), chan_max)

        self.block0 = nn.Sequential(nn.Conv3d(in_chan, chan(0), 1), nn.LeakyReLU(0.2, True))
        self.blocks = nn.ModuleList(
            SkipBlock(chan(b), chan(b + 1), out_chan, cat_noise) for b in range(self.num_blocks)
        )

    # -- noise plumbing ---------------------------------------------------- #
    def _resolve(self, noise) -> List[Optional[torch.Tensor]]:
        """Normalise ``noise`` (None / dict-by-stage / dict-by-site / sequence)."""
        n_sites = 2 * self.num_blocks
        if noise is None:
            return [None] * n_sites
        if isinstance(noise, dict):
            out: List[Optional[torch.Tensor]] = [None] * n_sites
            for key, value in noise.items():
                if key in STAGE_SITES:                       # {'coarse': [z0, z1], ...}
                    b = STAGE_NAMES.index(key)
                    if len(value) != 2:
                        raise ValueError(f"stage {key!r} needs exactly 2 noise tensors")
                    out[2 * b], out[2 * b + 1] = value[0], value[1]
                elif key in NOISE_SITES:                     # {'z0': ..., 'z5': ...}
                    out[NOISE_SITES.index(key)] = value
                else:
                    raise KeyError(
                        f"unknown noise key {key!r}; expected one of "
                        f"{sorted(STAGE_SITES)} or {list(NOISE_SITES[:n_sites])}"
                    )
            return out
        seq = list(noise)
        if len(seq) != n_sites:
            raise ValueError(f"expected {n_sites} noise tensors, got {len(seq)}")
        return seq

    def forward(self, x: torch.Tensor, noise=None, record: bool = False):
        """``y`` (and, if ``record``, the ``{site: z}`` dict actually used).

        ``noise`` may be ``None`` (sample internally), a length-6 sequence, a
        ``{'coarse': [z0, z1], 'middle': [...], 'fine': [...]}`` dict, or a
        ``{'z3': ...}`` dict naming individual sites. Sites left as ``None`` are
        sampled from ``torch.randn_like`` exactly as upstream does.
        """
        zs = self._resolve(noise)
        y = x
        x = self.block0(x)
        used: List[torch.Tensor] = []
        for b, block in enumerate(self.blocks):
            x, y, za, zb = block(x, y, zs[2 * b], zs[2 * b + 1])
            used += [za, zb]
        if record:
            return y, {NOISE_SITES[i]: z for i, z in enumerate(used)}
        return y


# --------------------------------------------------------------------------- #
# Site geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SiteLayout:
    """Where one injection site lives, in *global* coordinates.

    Attributes
    ----------
    site:
        ``'z0'``..``'z5'``.
    stage:
        ``'coarse'`` / ``'middle'`` / ``'fine'``.
    res:
        Resolution multiplier relative to the LR grid (1, 2, 4 or 8).
    size:
        Spatial extent (cells) of the noise tensor for this tile.
    offset:
        Index, on the global ``res * Ng_lr`` lattice, of this tile's first noise
        cell. Derived from the tile's padded LR start, so two tiles whose padded
        inputs overlap read the *same* global noise cells in the overlap.
    """

    site: str
    stage: str
    res: int
    size: int
    offset: int


def noise_site_layout(
    lr_size: int,
    scale_factor: int = 8,
    start: int = 0,
    pad: int = 0,
) -> List[SiteLayout]:
    """Per-site shapes and global offsets for one tile.

    ``lr_size`` is the size of the tensor actually fed to the generator, i.e. the
    *padded* tile (``chunk + 2 * pad``). ``start`` is the unpadded tile start on
    the LR grid; the padded input therefore begins at ``start - pad``.

    The offsets come from tracking the valid-convolution / upsample bookkeeping::

        upsample x2 (narrow 1)  : coord -> 2 * coord + 1
        conv3d k3 (valid)       : coord -> coord + 1

    which is exact, so :func:`noise_site_layout` is what makes globally coherent
    tiling (Stage 5) possible: adjacent tiles indexing the same global noise
    field agree bit-for-bit in their shared valid region.
    """
    num_blocks = round(log2(int(scale_factor)))
    size = int(lr_size)
    res = 1
    offset = int(start) - int(pad)      # LR-grid coordinate of the padded input
    out: List[SiteLayout] = []
    for b in range(num_blocks):
        stage = STAGE_NAMES[b] if b < len(STAGE_NAMES) else f"block{b}"
        # site a: acts on the block input, unchanged resolution
        out.append(SiteLayout(f"z{2 * b}", stage, res, size, offset))
        # upsample (x2, narrow 1) then conv3d k3 -> +2 on the fine lattice
        res, offset, size = 2 * res, 2 * offset + 2, 2 * size - 4
        out.append(SiteLayout(f"z{2 * b + 1}", stage, res, size, offset))
        # second conv3d k3
        offset, size = offset + 1, size - 2
    return out


def site_shapes(
    lr_size: int, scale_factor: int = 8, batch: int = 1
) -> Dict[str, Tuple[int, int, int, int, int]]:
    """``{site: (B, 1, s, s, s)}`` for an input of spatial size ``lr_size``."""
    return {
        lay.site: (int(batch), 1, lay.size, lay.size, lay.size)
        for lay in noise_site_layout(lr_size, scale_factor)
    }


def flatten_noise(noise: Dict[str, torch.Tensor], n_sites: int = 6) -> List[torch.Tensor]:
    """``{site: z}`` -> ``[z0, ..., z5]`` (forward order)."""
    return [noise[NOISE_SITES[i]] for i in range(n_sites)]


def unflatten_noise(seq: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """``[z0, ...]`` -> ``{site: z}``."""
    return {NOISE_SITES[i]: z for i, z in enumerate(seq)}


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #
def load_controlled_generator(
    model_path,
    in_chan: int = 6,
    out_chan: int = 6,
    scale_factor: int = 8,
    device=None,
    eval_mode: bool = True,
    **chan_kwargs,
) -> ControlledG:
    """Load ``SRmodel/G_z0.pt`` (or ``G_z2.pt``) into a :class:`ControlledG`.

    No dependency on ``external/SRS-map2map`` at all -- the checkpoint is a plain
    ``{'epoch', 'model'}`` dict and the key layout is reproduced here, so this
    avoids the ``sys.path`` juggling that :mod:`cosmo_sr.eval.baseline_srs` needs.
    """
    import os

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        state = torch.load(os.fspath(model_path), map_location="cpu")
    except Exception:  # pragma: no cover - older/pickled checkpoints
        state = torch.load(os.fspath(model_path), map_location="cpu", weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state

    G = ControlledG(in_chan, out_chan, scale_factor, **chan_kwargs)
    G.load_state_dict(sd, strict=True)   # raises on any key/shape mismatch
    G.to(device)
    if eval_mode:
        G.eval()
    return G
