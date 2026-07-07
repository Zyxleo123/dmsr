"""Adapter for the paper baseline: the pretrained SRS-map2map SR-GAN generator.

`SRS-map2map` is the code for "AI-assisted super-resolution cosmological
simulations" (Ni et al.). Its generator (``map2map.models.srsgan.G``) is a
StyleGAN2-style, *stochastic* SR-GAN using valid (unpadded) convolutions, so a
chunk of LR size ``L`` maps to HR size ``scale*L - 42`` (for scale 8). This
module loads the pretrained weights and performs SRS-style tiled inference
(periodic pad + narrow/trim) returning a *normalized* ``(6, Ng*scale, ...)``
field in the same space as our fields, so both can be scored with the same
metrics.

The SRS repo ships its own ``map2map`` fork (with ``srsgan``), distinct from the
installed ``eelregit/map2map``. We prepend the SRS repo to ``sys.path`` so
``import map2map`` resolves to that fork. Do not import ``map2map`` elsewhere in
the process before calling this.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..data.crops import periodic_crop


def _default_srs_dir() -> str:
    # src/cosmo_sr/eval/baseline_srs.py -> project root is parents[3]
    root = Path(__file__).resolve().parents[3]
    return str(root / "external" / "SRS-map2map")


def _import_srs_models(srs_dir: Optional[str] = None):
    srs_dir = os.fspath(srs_dir or _default_srs_dir())
    srs_dir = str(Path(srs_dir).resolve())
    if srs_dir not in sys.path:
        sys.path.insert(0, srs_dir)
    import map2map  # noqa: WPS433

    if "SRS-map2map" not in os.path.realpath(map2map.__file__):
        raise ImportError(
            "Expected the SRS-map2map fork of `map2map` (with srsgan.G), but "
            f"`import map2map` resolved to {map2map.__file__}. Ensure no other "
            "map2map was imported earlier in this process."
        )
    from map2map import models  # noqa: WPS433

    return models


def load_srs_generator(
    model_path: str | os.PathLike,
    srs_dir: Optional[str] = None,
    in_chan: int = 6,
    out_chan: int = 6,
    scale_factor: int = 8,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Load the pretrained SRS generator (e.g. ``SRmodel/G_z0.pt``)."""
    models = _import_srs_models(srs_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = models.G(in_chan, out_chan, scale_factor)
    state = torch.load(os.fspath(model_path), map_location=device)
    G.load_state_dict(state["model"])
    G.eval().to(device)
    return G


def _narrow_like(box: np.ndarray, tgt: int) -> np.ndarray:
    """Center-trim a ``(C,N,N,N)`` box to ``(C,tgt,tgt,tgt)``."""
    width = box.shape[1] - tgt
    if width < 0:
        raise ValueError(f"cannot narrow size {box.shape[1]} to larger tgt {tgt}")
    hw = width // 2
    return box[:, hw:hw + tgt, hw:hw + tgt, hw:hw + tgt]


def super_resolve_srs(
    generator: torch.nn.Module,
    lr_field,
    scale_factor: int = 8,
    nsplit: int = 4,
    pad: int = 3,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """SRS-style tiled inference returning a normalized ``(6, Ng*scale, ...)`` field.

    Mirrors ``SRS-map2map/lr2sr.py`` (periodic crop with ``pad`` + ``narrow_like``
    trim) but keeps the output in normalized units for metric comparison. The
    generator is stochastic; ``seed`` fixes the noise for reproducibility.
    """
    lr_np = np.ascontiguousarray(np.asarray(lr_field), dtype=np.float32)
    if lr_np.ndim != 4 or not (lr_np.shape[1] == lr_np.shape[2] == lr_np.shape[3]):
        raise ValueError(f"lr_field must be cubic (C,Ng,Ng,Ng); got {lr_np.shape}")
    C, Ng = lr_np.shape[0], lr_np.shape[1]
    if Ng % nsplit != 0:
        raise ValueError(f"nsplit={nsplit} does not divide Ng={Ng}")

    device = device or next(generator.parameters()).device
    torch.manual_seed(seed)

    chunk = Ng // nsplit
    tgt = chunk * scale_factor
    Ng_hr = Ng * scale_factor
    out = np.zeros((C, Ng_hr, Ng_hr, Ng_hr), dtype=np.float32)

    for ix in range(nsplit):
        for iy in range(nsplit):
            for iz in range(nsplit):
                start = (ix * chunk, iy * chunk, iz * chunk)
                crop = periodic_crop(lr_np, start, chunk, pad=pad)
                t = torch.from_numpy(np.ascontiguousarray(crop)).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    sr = generator(t).squeeze(0).cpu().numpy()
                sr = _narrow_like(sr, tgt)
                hs = (ix * tgt, iy * tgt, iz * tgt)
                out[:, hs[0]:hs[0] + tgt, hs[1]:hs[1] + tgt, hs[2]:hs[2] + tgt] = sr
    return out
