"""Reproducible multi-sample SR2 inference (test-time scaling, Stage 0).

The existing adapter (:func:`cosmo_sr.eval.baseline_srs.super_resolve_srs`) seeds
the *global* torch RNG once and then walks the tiles in a fixed order, so the
noise a given tile receives depends on how many tiles were drawn before it. That
makes a realisation an artefact of the traversal and batching, and gives no way
to ask for "the same box again, different noise".

Here a **root seed defines one complete full-box realisation**. Each tile's six
noise tensors are derived from ``(root_seed, tile_coordinate, site)`` through a
BLAKE2b hash, so:

* the same ``(lr, root_seed)`` always gives the same box;
* traversal order, batching and device never enter the result;
* different root seeds give genuinely independent realisations.

Two noise modes are available:

``"per_tile"`` (default)
    Each tile draws i.i.d. noise. Tiles are stitched by SRS's crop-and-trim, so
    the noise is discontinuous across tile seams -- the same regime the upstream
    ``lr2sr.py`` runs in.
``"global"``
    Noise is drawn once for the whole box on the six site lattices and each tile
    reads the spatially corresponding window (see
    :func:`cosmo_sr.tts.srs_noise.noise_site_layout`). Overlapping tiles then
    agree exactly in their shared valid region, which is what Stage 5 needs.

Nothing here averages candidates: the whole point is to keep the individual
realisations so a selector can choose among them.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import asdict, dataclass, field as dc_field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data.crops import periodic_crop
from .srs_noise import NOISE_SITES, ControlledG, noise_site_layout

__all__ = [
    "Candidate",
    "GlobalNoiseField",
    "InferenceConfig",
    "derive_tile_seed",
    "generate_srs_candidates",
    "iter_srs_candidates",
    "super_resolve_srs_seeded",
    "tile_starts",
]


# --------------------------------------------------------------------------- #
# Seed derivation
# --------------------------------------------------------------------------- #
def derive_tile_seed(root_seed: int, tile_coord: Sequence[int], site: int = -1) -> int:
    """A 63-bit seed determined by ``(root_seed, tile_coord, site)``.

    Hashing rather than arithmetic mixing (``root * P1 + x * P2 + ...``) because
    tile coordinates are small consecutive integers: an affine mix would make
    neighbouring tiles' seeds correlated, and torch's Philox/MT streams from
    nearby seeds are not guaranteed independent. BLAKE2b decorrelates them.

    ``site = -1`` means "the whole tile"; ``site >= 0`` names one injection site.
    """
    coord = tuple(int(c) for c in tile_coord)
    if len(coord) != 3:
        raise ValueError(f"tile_coord must be a 3-tuple, got {tile_coord!r}")
    payload = struct.pack("<qqqqq", int(root_seed), *coord, int(site))
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def tile_starts(ng: int, nsplit: int) -> List[Tuple[int, int, int]]:
    """Unpadded LR start coordinates of every tile, in canonical (x, y, z) order."""
    if ng % nsplit:
        raise ValueError(f"nsplit={nsplit} does not divide Ng={ng}")
    chunk = ng // nsplit
    return [
        (ix * chunk, iy * chunk, iz * chunk)
        for ix in range(nsplit)
        for iy in range(nsplit)
        for iz in range(nsplit)
    ]


# --------------------------------------------------------------------------- #
# Noise sources
# --------------------------------------------------------------------------- #
def _randn(shape, seed: int, device, dtype=torch.float32) -> torch.Tensor:
    """``randn`` drawn on the CPU from ``seed``, then moved to ``device``.

    Deliberately CPU-side: CUDA's generator produces a different stream for the
    same seed, and a realisation must not depend on which device produced it.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)


class GlobalNoiseField:
    """Full-box noise on all six injection lattices, sliced per tile.

    Site ``s`` lives on a ``(res_s * ng_lr)`` cubic lattice; a tile reads the
    window starting at :attr:`SiteLayout.offset`, wrapping periodically. Two
    tiles whose padded inputs overlap therefore read *identical* noise values in
    the overlap, which (with valid convolutions and a translation-equivariant
    network) makes their outputs agree exactly there.

    Fields are materialised lazily and cached: for ``ng_lr = 64`` the six
    lattices total ~172M floats (~690 MB in float32), so hold one realisation at
    a time.
    """

    def __init__(self, root_seed: int, ng_lr: int, scale_factor: int = 8,
                 device=None, dtype=torch.float32):
        self.root_seed = int(root_seed)
        self.ng_lr = int(ng_lr)
        self.scale_factor = int(scale_factor)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self._cache: Dict[str, torch.Tensor] = {}

    def field(self, site: str, res: int) -> torch.Tensor:
        """The ``(res * ng_lr)^3`` noise lattice for one site."""
        if site not in self._cache:
            n = res * self.ng_lr
            seed = derive_tile_seed(self.root_seed, (-1, -1, -1), NOISE_SITES.index(site))
            self._cache[site] = _randn((n, n, n), seed, self.device, self.dtype)
        return self._cache[site]

    def window(self, site: str, res: int, size: int,
               offsets: Sequence[int]) -> torch.Tensor:
        """``(1, 1, s, s, s)`` slice of one site's lattice, wrapping periodically.

        ``offsets`` is per-axis: the tile's ``SiteLayout.offset`` computed from
        each of its three start coordinates.
        """
        f = self.field(site, res)
        n = f.shape[0]
        if size > n:
            raise ValueError(
                f"site {site} window {size} exceeds the global lattice {n}; "
                "use a larger box or fewer/larger tiles"
            )
        idx = [
            (torch.arange(size, device=f.device) + int(o)) % n for o in offsets
        ]
        sub = f.index_select(0, idx[0]).index_select(1, idx[1]).index_select(2, idx[2])
        return sub.reshape(1, 1, *sub.shape)

    def free(self) -> None:
        self._cache.clear()


def tile_noise(
    root_seed: int,
    tile_coord: Sequence[int],
    lr_size: int,
    scale_factor: int,
    device,
    pad: int = 0,
    mode: str = "per_tile",
    global_field: Optional[GlobalNoiseField] = None,
    dtype=torch.float32,
) -> Dict[str, torch.Tensor]:
    """The six noise tensors for one tile, in the requested mode."""
    layouts = noise_site_layout(lr_size, scale_factor, start=0, pad=0)
    if mode == "global":
        if global_field is None:
            raise ValueError("mode='global' requires a GlobalNoiseField")
        # noise_site_layout tracks one axis at a time; the offsets are affine in
        # `start`, so evaluate it once per coordinate to get the 3-D window.
        per_axis = [
            noise_site_layout(lr_size, scale_factor, start=int(c), pad=pad)
            for c in tile_coord
        ]
        return {
            lay.site: global_field.window(
                lay.site, lay.res, lay.size, [per_axis[d][i].offset for d in range(3)]
            ).to(device=device, dtype=dtype)
            for i, lay in enumerate(layouts)
        }
    if mode != "per_tile":
        raise ValueError(f"unknown noise mode {mode!r}; expected 'per_tile' or 'global'")
    return {
        lay.site: _randn(
            (1, 1, lay.size, lay.size, lay.size),
            derive_tile_seed(root_seed, tile_coord, i),
            device,
            dtype,
        )
        for i, lay in enumerate(layouts)
    }


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #
@dataclass
class InferenceConfig:
    """Everything needed to regenerate a candidate byte-for-byte."""

    scale_factor: int = 8
    nsplit: int = 8
    pad: int = 3
    noise_mode: str = "per_tile"
    ng_lr: int = 0
    chunk: int = 0
    model_path: str = ""
    device: str = ""
    dtype: str = "float32"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Candidate:
    """One full-box SR realisation plus the provenance needed to reproduce it."""

    seed: int
    field: Optional[np.ndarray]
    config: Dict = dc_field(default_factory=dict)
    box: str = ""

    @property
    def disp(self) -> np.ndarray:
        return self.field[0:3]

    @property
    def vel(self) -> np.ndarray:
        return self.field[3:6]


def _narrow_like(box: torch.Tensor, tgt: int) -> torch.Tensor:
    """Centre-trim a ``(C, N, N, N)`` box to ``(C, tgt, tgt, tgt)``."""
    width = box.shape[-1] - tgt
    if width < 0:
        raise ValueError(f"cannot narrow size {box.shape[-1]} up to {tgt}")
    hw = width // 2
    return box[..., hw:hw + tgt, hw:hw + tgt, hw:hw + tgt]


@torch.no_grad()
def super_resolve_srs_seeded(
    generator,
    lr_field,
    seed: int,
    scale_factor: int = 8,
    nsplit: int = 8,
    pad: int = 3,
    device=None,
    noise_mode: str = "per_tile",
    global_field: Optional[GlobalNoiseField] = None,
    tile_order: Optional[Sequence[Tuple[int, int, int]]] = None,
    out_dtype=np.float32,
    progress: bool = False,
) -> np.ndarray:
    """One full-box realisation, ``(6, Ng*scale, ...)`` in normalized units.

    Same tiling geometry as :func:`cosmo_sr.eval.baseline_srs.super_resolve_srs`
    (periodic pad, centre-trim), but the noise is derived per tile from ``seed``
    rather than consumed from the global RNG, so ``tile_order`` cannot change the
    result -- only the order in which identical values are written.
    """
    lr_np = np.ascontiguousarray(np.asarray(lr_field), dtype=np.float32)
    if lr_np.ndim != 4 or not (lr_np.shape[1] == lr_np.shape[2] == lr_np.shape[3]):
        raise ValueError(f"lr_field must be cubic (C, Ng, Ng, Ng); got {lr_np.shape}")
    c, ng = lr_np.shape[0], lr_np.shape[1]
    if ng % nsplit:
        raise ValueError(f"nsplit={nsplit} does not divide Ng={ng}")

    if not isinstance(generator, ControlledG):
        raise TypeError(
            "seeded inference needs a cosmo_sr.tts.srs_noise.ControlledG (the upstream "
            "map2map G draws noise from the global RNG and cannot be steered per tile); "
            "load one with load_controlled_generator()"
        )
    device = device or next(generator.parameters()).device
    chunk = ng // nsplit
    lr_size = chunk + 2 * pad
    tgt = chunk * scale_factor
    ng_hr = ng * scale_factor
    out = np.zeros((c, ng_hr, ng_hr, ng_hr), dtype=out_dtype)

    starts = list(tile_order) if tile_order is not None else tile_starts(ng, nsplit)
    for n_done, start in enumerate(starts):
        crop = periodic_crop(lr_np, start, chunk, pad=pad)
        t = torch.from_numpy(np.ascontiguousarray(crop)).float().unsqueeze(0).to(device)
        noise = tile_noise(
            seed, start, lr_size, scale_factor, device,
            pad=pad, mode=noise_mode, global_field=global_field,
        )
        sr = generator(t, noise=noise).squeeze(0)
        sr = _narrow_like(sr, tgt).to("cpu").numpy().astype(out_dtype, copy=False)
        hs = tuple(s * scale_factor for s in start)
        out[:, hs[0]:hs[0] + tgt, hs[1]:hs[1] + tgt, hs[2]:hs[2] + tgt] = sr
        if progress and (n_done + 1) % max(1, len(starts) // 10) == 0:
            print(f"    tile {n_done + 1}/{len(starts)}", flush=True)
    return out


def iter_srs_candidates(
    generator,
    lr_field,
    seeds: Iterable[int],
    nsplit: int = 8,
    pad: int = 3,
    scale_factor: int = 8,
    device=None,
    noise_mode: str = "per_tile",
    box: str = "",
    model_path: str = "",
    out_dtype=np.float32,
    progress: bool = False,
) -> Iterator[Candidate]:
    """Yield one :class:`Candidate` per root seed, holding one box at a time.

    Streaming matters: a 512^3 six-channel realisation is 3.2 GB, so ``K = 32``
    materialised at once is 100 GB. Consumers that only need per-candidate
    summaries should iterate rather than call :func:`generate_srs_candidates`.
    """
    lr_np = np.ascontiguousarray(np.asarray(lr_field), dtype=np.float32)
    ng = lr_np.shape[1]
    device = device or next(generator.parameters()).device
    cfg = InferenceConfig(
        scale_factor=int(scale_factor), nsplit=int(nsplit), pad=int(pad),
        noise_mode=str(noise_mode), ng_lr=int(ng), chunk=int(ng // nsplit),
        model_path=str(model_path), device=str(device), dtype=str(np.dtype(out_dtype)),
    )
    for seed in seeds:
        gfield = (
            GlobalNoiseField(int(seed), ng, scale_factor, device=torch.device("cpu"))
            if noise_mode == "global" else None
        )
        field = super_resolve_srs_seeded(
            generator, lr_np, int(seed), scale_factor=scale_factor, nsplit=nsplit,
            pad=pad, device=device, noise_mode=noise_mode, global_field=gfield,
            out_dtype=out_dtype, progress=progress,
        )
        if gfield is not None:
            gfield.free()
        yield Candidate(seed=int(seed), field=field, config=cfg.to_dict(), box=box)


def generate_srs_candidates(
    generator,
    lr_field,
    seeds: Sequence[int],
    nsplit: int = 8,
    pad: int = 3,
    scale_factor: int = 8,
    device=None,
    noise_mode: str = "per_tile",
    box: str = "",
    model_path: str = "",
    out_dtype=np.float32,
    progress: bool = False,
) -> List[Candidate]:
    """``K`` independent full-box realisations of the same LR input.

    Candidates are **not** averaged or combined -- selection happens downstream.
    Materialises all ``K`` boxes; prefer :func:`iter_srs_candidates` when ``K``
    or the box size is large.
    """
    return list(
        iter_srs_candidates(
            generator, lr_field, seeds, nsplit=nsplit, pad=pad,
            scale_factor=scale_factor, device=device, noise_mode=noise_mode,
            box=box, model_path=model_path, out_dtype=out_dtype, progress=progress,
        )
    )
