"""Stage 5: globally coherent tiled inference.

The upstream tiling draws independent noise per tile and butt-joins the trimmed
outputs. Two consequences: the realisation depends on the tile grid, and the
seams are visible discontinuities that every seam-sensitive statistic
(bispectrum, halo finding, velocity divergence) picks up.

The fix has two halves:

1. **Coordinate-indexed noise.** One global noise lattice per injection site for
   the whole box; a tile reads the window its own coordinates point at
   (:func:`cosmo_sr.tts.srs_noise.noise_site_layout`). Because the generator is
   fully convolutional over valid convolutions, two overlapping tiles reading the
   same global noise produce *identical* values in their shared valid region --
   not merely similar. Periodicity across opposite box faces follows from
   wrapping both the LR crop and the noise indices.

2. **Overlapping tiles, selected jointly.** With overlaps available we can score
   whole tile *combinations*

   .. math::
       S_{joint} = \\sum_i S_{verifier}(x_i, y_i)
                 + \\lambda_{overlap} \\sum_{(i,j)} \\lVert x_i - x_j \\rVert^2_{overlap}

   and minimise it by coordinate descent over each tile's candidate list.
   Cropping or blending happens **after** selection: blending first would average
   away exactly the small-scale variance the candidates differ in, which looks
   like a smoother seam and is in fact a worse field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data.crops import periodic_crop
from .sampling import GlobalNoiseField, tile_noise
from .srs_noise import ControlledG

__all__ = [
    "TileGrid",
    "blend_window",
    "generate_tiles",
    "joint_score",
    "select_tiles_coordinate_descent",
    "stitch_overlapping",
    "tiled_inference",
]


@dataclass(frozen=True)
class TileGrid:
    """Overlapping tile geometry on a periodic LR box.

    ``chunk`` is the LR extent each tile *claims*; ``stride`` is how far apart
    tile origins sit. ``stride == chunk`` reproduces the non-overlapping upstream
    layout. Requires ``stride >= chunk / 2`` so at most two tiles cover any point
    per axis, which is what makes the complementary blend window exact.
    """

    ng: int
    chunk: int
    stride: int
    pad: int = 3
    scale: int = 8

    def __post_init__(self):
        if self.ng % self.stride:
            raise ValueError(f"stride {self.stride} must divide Ng {self.ng}")
        if not (self.chunk // 2 <= self.stride <= self.chunk):
            raise ValueError(
                f"need chunk/2 <= stride <= chunk, got chunk={self.chunk}, stride={self.stride}"
            )
        if self.chunk + 2 * self.pad > self.ng:
            raise ValueError("padded tile exceeds the box; use a smaller chunk")

    @property
    def n_per_axis(self) -> int:
        return self.ng // self.stride

    @property
    def overlap_lr(self) -> int:
        return self.chunk - self.stride

    @property
    def tile_hr(self) -> int:
        return self.chunk * self.scale

    @property
    def lr_size(self) -> int:
        return self.chunk + 2 * self.pad

    def starts(self) -> List[Tuple[int, int, int]]:
        r = range(0, self.ng, self.stride)
        return [(x, y, z) for x in r for y in r for z in r]


def blend_window(size: int, ramp: int, device=None) -> torch.Tensor:
    """1-D weight: cosine ramp up over ``ramp`` cells, flat, ramp down.

    Complementary by construction (``w(t) + w(1 - t) = 1``), so two tiles whose
    ramps coincide sum to exactly one and no renormalisation is needed.
    """
    w = torch.ones(size, device=device, dtype=torch.float32)
    if ramp > 0:
        t = (torch.arange(ramp, device=device, dtype=torch.float32) + 0.5) / ramp
        up = 0.5 * (1 - torch.cos(np.pi * t))
        w[:ramp] = up
        w[size - ramp:] = torch.flip(up, dims=[0])
    return w


@torch.no_grad()
def generate_tiles(
    generator: ControlledG,
    lr_field,
    grid: TileGrid,
    seed: int,
    device=None,
    noise_mode: str = "global",
    global_field: Optional[GlobalNoiseField] = None,
    starts: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> Dict[Tuple[int, int, int], torch.Tensor]:
    """``{start: (6, tile_hr, tile_hr, tile_hr)}`` for one realisation.

    In ``"global"`` mode all tiles read one coordinate-indexed noise field, so the
    result is a genuine sub-sampling of a single full-box realisation.
    """
    lr_np = np.ascontiguousarray(np.asarray(lr_field), dtype=np.float32)
    device = device or next(generator.parameters()).device
    if noise_mode == "global" and global_field is None:
        global_field = GlobalNoiseField(int(seed), grid.ng, grid.scale,
                                        device=torch.device("cpu"))
    out: Dict[Tuple[int, int, int], torch.Tensor] = {}
    for start in (starts if starts is not None else grid.starts()):
        crop = periodic_crop(lr_np, start, grid.chunk, pad=grid.pad)
        x = torch.from_numpy(np.ascontiguousarray(crop)).float().unsqueeze(0).to(device)
        z = tile_noise(int(seed), start, grid.lr_size, grid.scale, device, pad=grid.pad,
                       mode=noise_mode, global_field=global_field)
        y = generator(x, noise=z).squeeze(0)
        w = (y.shape[-1] - grid.tile_hr) // 2
        out[tuple(start)] = y[..., w:w + grid.tile_hr, w:w + grid.tile_hr, w:w + grid.tile_hr]
    return out


def _overlap_slices(a: int, b: int, grid: TileGrid) -> Optional[Tuple[slice, slice]]:
    """Local HR slices of the shared region of two tiles offset along one axis."""
    n_hr = grid.ng * grid.scale
    d = ((b - a) % grid.ng) * grid.scale
    if d == 0:
        return (slice(None), slice(None))
    width = grid.tile_hr - d
    if width <= 0:
        d2 = n_hr - d
        width = grid.tile_hr - d2
        if width <= 0:
            return None
        return (slice(0, width), slice(grid.tile_hr - width, grid.tile_hr))
    return (slice(d, grid.tile_hr), slice(0, width))


def overlap_pairs(grid: TileGrid) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Neighbouring tile pairs (one per axis direction), with periodic wrap."""
    starts = grid.starts()
    have = set(starts)
    pairs = []
    for s in starts:
        for axis in range(3):
            t = list(s)
            t[axis] = (s[axis] + grid.stride) % grid.ng
            t = tuple(t)
            if t in have and grid.overlap_lr > 0:
                pairs.append((s, t))
    return pairs


def overlap_disagreement(a: torch.Tensor, b: torch.Tensor, axis: int,
                         grid: TileGrid) -> float:
    """Mean squared difference of two neighbouring tiles in their shared region."""
    sl = _overlap_slices(0, grid.stride, grid)
    if sl is None:
        return 0.0
    sa, sb = sl
    idx_a = [slice(None)] * 4
    idx_b = [slice(None)] * 4
    idx_a[axis + 1], idx_b[axis + 1] = sa, sb
    return float((a[tuple(idx_a)] - b[tuple(idx_b)]).pow(2).mean())


def joint_score(
    choice: Dict[Tuple[int, int, int], int],
    verifier: Dict[Tuple[int, int, int], Sequence[float]],
    disagreement: Dict[Tuple, np.ndarray],
    lam_overlap: float,
) -> float:
    """``sum_i S_verifier(x_i) + lam * sum_(i,j) ||x_i - x_j||^2_overlap``."""
    total = sum(float(verifier[s][choice[s]]) for s in choice)
    for (s, t), mat in disagreement.items():
        total += lam_overlap * float(mat[choice[s], choice[t]])
    return float(total)


def select_tiles_coordinate_descent(
    verifier: Dict[Tuple[int, int, int], Sequence[float]],
    disagreement: Dict[Tuple, np.ndarray],
    lam_overlap: float,
    max_sweeps: int = 8,
    init: Optional[Dict[Tuple[int, int, int], int]] = None,
) -> Tuple[Dict[Tuple[int, int, int], int], List[float]]:
    """Minimise :func:`joint_score` by iterated conditional modes.

    The objective is a pairwise MRF over tiles with a small label set (``K``
    candidates), so exact inference is unnecessary and coordinate descent from
    the independent-argmin initialisation converges in a couple of sweeps. Each
    sweep is monotone, so the trajectory is a sufficient convergence check.
    """
    choice = dict(init) if init else {s: int(np.argmin(v)) for s, v in verifier.items()}
    neighbours: Dict[Tuple, List[Tuple[Tuple, bool]]] = {s: [] for s in verifier}
    for (s, t) in disagreement:
        neighbours[s].append((t, False))
        neighbours[t].append((s, True))

    traj = [joint_score(choice, verifier, disagreement, lam_overlap)]
    for _ in range(int(max_sweeps)):
        changed = False
        for s in sorted(verifier):
            k = len(verifier[s])
            costs = np.asarray(verifier[s], dtype=float).copy()
            for t, flipped in neighbours[s]:
                mat = disagreement[(t, s)] if flipped else disagreement[(s, t)]
                costs += lam_overlap * (mat[:, choice[t]] if not flipped
                                        else mat[choice[t], :])
            best = int(np.argmin(costs))
            if best != choice[s]:
                choice[s] = best
                changed = True
            del k
        traj.append(joint_score(choice, verifier, disagreement, lam_overlap))
        if not changed:
            break
    return choice, traj


def stitch_overlapping(
    tiles: Dict[Tuple[int, int, int], torch.Tensor],
    grid: TileGrid,
    mode: str = "blend",
) -> torch.Tensor:
    """Assemble selected tiles into a full periodic ``(6, N, N, N)`` HR box.

    ``"crop"`` keeps each tile's central ``stride * scale`` block (no averaging,
    so no variance is lost); ``"blend"`` cosine-blends the overlap. Both run
    *after* selection.
    """
    if mode not in ("crop", "blend"):
        raise ValueError(f"mode must be 'crop' or 'blend', got {mode!r}")
    any_tile = next(iter(tiles.values()))
    c = any_tile.shape[0]
    n_hr = grid.ng * grid.scale
    dev = any_tile.device
    out = torch.zeros((c, n_hr, n_hr, n_hr), device=dev, dtype=torch.float32)

    if mode == "crop":
        keep = grid.stride * grid.scale
        off = (grid.tile_hr - keep) // 2
        for start, tile in tiles.items():
            h = [s * grid.scale + off for s in start]
            idx = [(torch.arange(keep, device=dev) + hi) % n_hr for hi in h]
            sel = tile[..., off:off + keep, off:off + keep, off:off + keep]
            out[:, idx[0].view(-1, 1, 1), idx[1].view(1, -1, 1), idx[2].view(1, 1, -1)] = sel
        return out

    ramp = grid.overlap_lr * grid.scale
    w1 = blend_window(grid.tile_hr, ramp, device=dev)
    weight = torch.zeros((1, n_hr, n_hr, n_hr), device=dev, dtype=torch.float32)
    w3 = w1.view(-1, 1, 1) * w1.view(1, -1, 1) * w1.view(1, 1, -1)
    for start, tile in tiles.items():
        h = [s * grid.scale for s in start]
        ix = ((torch.arange(grid.tile_hr, device=dev) + h[0]) % n_hr).view(-1, 1, 1)
        iy = ((torch.arange(grid.tile_hr, device=dev) + h[1]) % n_hr).view(1, -1, 1)
        iz = ((torch.arange(grid.tile_hr, device=dev) + h[2]) % n_hr).view(1, 1, -1)
        out[:, ix, iy, iz] += tile * w3
        weight[:, ix, iy, iz] += w3
    return out / weight.clamp_min(1e-12)


@torch.no_grad()
def tiled_inference(
    generator: ControlledG,
    lr_field,
    grid: TileGrid,
    seed: int,
    device=None,
    noise_mode: str = "global",
    mode: str = "blend",
    starts: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> torch.Tensor:
    """One full-box realisation from overlapping, globally coherent tiles."""
    tiles = generate_tiles(generator, lr_field, grid, seed, device=device,
                           noise_mode=noise_mode, starts=starts)
    return stitch_overlapping(tiles, grid, mode=mode)
