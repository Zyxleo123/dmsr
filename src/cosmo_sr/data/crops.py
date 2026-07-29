"""Periodic cropping, padding and stitching for cubic cosmology boxes.

Cosmology boxes are periodic, so crops near a boundary must wrap around rather
than zero-pad. All functions operate on channel-first fields ``(C, N, N, N)`` and
work for both NumPy arrays and PyTorch tensors (NumPy is preferred for
preprocessing, Torch for training).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

try:  # torch is optional at import time for pure-numpy preprocessing
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def _is_torch(x) -> bool:
    return _HAS_TORCH and torch.is_tensor(x)


def _as_triplet(value, name: str) -> Tuple[int, int, int]:
    if np.isscalar(value):
        return (int(value), int(value), int(value))
    seq = tuple(int(v) for v in value)
    if len(seq) != 3:
        raise ValueError(f"{name} must be a scalar or length-3, got {value!r}")
    return seq  # type: ignore[return-value]


def _check_field(field) -> Tuple[int, int, int, int]:
    if field.ndim != 4:
        raise ValueError(
            f"Expected channel-first field (C, N, N, N), got shape {tuple(field.shape)}"
        )
    c, nx, ny, nz = field.shape
    if not (nx == ny == nz):
        raise ValueError(f"Only cubic fields are supported; got {(nx, ny, nz)}")
    return int(c), int(nx), int(ny), int(nz)


def _wrapped_index(start: int, size: int, N: int, is_torch: bool, device=None):
    if is_torch:
        ar = torch.arange(size, device=device)
        return (start + ar) % N
    return (start + np.arange(size)) % N


def _axis_runs(start: int, size: int, n: int) -> List[slice]:
    """Contiguous runs covering ``[start, start+size)`` mod ``n`` -- at most two."""
    s0 = start % n
    if s0 + size <= n:
        return [slice(s0, s0 + size)]
    return [slice(s0, n), slice(0, s0 + size - n)]


def periodic_crop(field, start: Sequence[int], crop_size, pad=0):
    """Crop a ``crop_size`` block starting at ``start`` with periodic wrapping.

    Parameters
    ----------
    field:
        ``(C, N, N, N)`` NumPy array or Torch tensor.
    start:
        Length-3 spatial start indices.
    crop_size:
        Scalar or length-3 crop size (before padding).
    pad:
        Scalar or length-3 padding added on *both* sides of each axis, taken
        periodically. Output size along axis ``d`` is ``crop_size[d] + 2*pad[d]``.

    Returns
    -------
    A crop of shape ``(C, cs0+2p0, cs1+2p1, cs2+2p2)`` of the same backend/dtype.

    Notes
    -----
    Implemented with **basic slicing**, not per-axis fancy indexing. The obvious
    version (``np.take`` along each axis in turn) reduces one axis per call, so
    the first call materialises a ``(C, crop, N, N)`` intermediate -- 805 MB for
    a 128^3 crop of a 6x512^3 box, to produce a 50 MB result. That measured
    ~1s per crop (3 channels; ~2s at 6), and at 32 crops per optimizer step it
    dominated training completely.

    A wrapped index range is at most two contiguous runs per axis, so the crop
    is at most 2^3 = 8 rectangular blocks. Copying those straight into the
    output touches only ``crop**3`` elements. Measured ~58x faster, and the
    no-wrap case (the common one) becomes a single strided copy.
    """
    _c, nx, ny, nz = _check_field(field)
    spatial = (nx, ny, nz)
    start = _as_triplet(start, "start")
    crop_size = _as_triplet(crop_size, "crop_size")
    pad = _as_triplet(pad, "pad")

    sizes = []
    runs = []
    for axis in range(3):
        size = crop_size[axis] + 2 * pad[axis]
        if size > spatial[axis]:
            raise ValueError(
                f"Padded crop size {size} exceeds box size {spatial[axis]} along "
                f"axis {axis}; periodic crop would self-overlap."
            )
        sizes.append(size)
        runs.append(_axis_runs(start[axis] - pad[axis], size, spatial[axis]))

    # Fast path: nothing wraps -> one contiguous strided copy, no assembly.
    if all(len(r) == 1 for r in runs):
        sub = field[:, runs[0][0], runs[1][0], runs[2][0]]
        if _is_torch(field):
            return sub.contiguous()
        return np.ascontiguousarray(sub)

    is_torch = _is_torch(field)
    shape = (field.shape[0], sizes[0], sizes[1], sizes[2])
    if is_torch:
        out = torch.empty(shape, dtype=field.dtype, device=field.device)
    else:
        out = np.empty(shape, dtype=field.dtype)

    ox = 0
    for rx in runs[0]:
        oy = 0
        for ry in runs[1]:
            oz = 0
            for rz in runs[2]:
                block = field[:, rx, ry, rz]
                bx, by, bz = block.shape[1], block.shape[2], block.shape[3]
                out[:, ox:ox + bx, oy:oy + by, oz:oz + bz] = block
                oz += bz
            oy += by
        ox += bx
    return out


def make_crop_grid(field_shape: Sequence[int], crop_size, stride) -> List[Tuple[int, int, int]]:
    """Enumerate crop start coordinates tiling a box with the given stride.

    ``field_shape`` may be ``(C, N, N, N)`` or ``(N, N, N)``. Returns a list of
    ``(x, y, z)`` start tuples covering ``range(0, N, stride)`` on each axis.
    """
    shape = tuple(int(s) for s in field_shape)
    if len(shape) == 4:
        spatial = shape[1:]
    elif len(shape) == 3:
        spatial = shape
    else:
        raise ValueError(f"field_shape must have length 3 or 4, got {field_shape!r}")
    if not (spatial[0] == spatial[1] == spatial[2]):
        raise ValueError(f"Only cubic fields are supported; got {spatial}")
    crop_size = _as_triplet(crop_size, "crop_size")
    stride = _as_triplet(stride, "stride")

    starts_per_axis = [list(range(0, spatial[d], stride[d])) for d in range(3)]
    grid: List[Tuple[int, int, int]] = []
    for x in starts_per_axis[0]:
        for y in starts_per_axis[1]:
            for z in starts_per_axis[2]:
                grid.append((x, y, z))
    return grid


def stitch_crops(crops, starts, full_shape, mode: str = "overwrite"):
    """Stitch unpadded crops back into a full periodic field.

    Parameters
    ----------
    crops:
        Sequence of ``(C, cs, cs, cs)`` crops (same backend/dtype).
    starts:
        Sequence of length-3 start coordinates (matching ``crops``).
    full_shape:
        ``(C, N, N, N)`` shape of the output field.
    mode:
        ``"overwrite"`` (later crops win) or ``"average"`` (mean over all crops
        covering each voxel). For identical source crops both reconstruct exactly.
    """
    if mode not in ("overwrite", "average"):
        raise ValueError(f"mode must be 'overwrite' or 'average', got {mode!r}")
    crops = list(crops)
    starts = list(starts)
    if len(crops) != len(starts):
        raise ValueError("crops and starts must have equal length")
    if len(crops) == 0:
        raise ValueError("no crops to stitch")

    full_shape = tuple(int(s) for s in full_shape)
    if len(full_shape) != 4 or not (full_shape[1] == full_shape[2] == full_shape[3]):
        raise ValueError(f"full_shape must be cubic (C, N, N, N), got {full_shape}")
    N = full_shape[1]

    is_torch = _is_torch(crops[0])
    if is_torch:
        out = torch.zeros(full_shape, dtype=crops[0].dtype, device=crops[0].device)
        count = torch.zeros(full_shape, dtype=crops[0].dtype, device=crops[0].device)
    else:
        out = np.zeros(full_shape, dtype=crops[0].dtype)
        count = np.zeros(full_shape, dtype=np.float64)

    for crop, start in zip(crops, starts):
        start = _as_triplet(start, "start")
        cs = _as_triplet(crop.shape[1:], "crop.shape")
        device = crop.device if is_torch else None
        i0 = _wrapped_index(start[0], cs[0], N, is_torch, device)
        i1 = _wrapped_index(start[1], cs[1], N, is_torch, device)
        i2 = _wrapped_index(start[2], cs[2], N, is_torch, device)
        if is_torch:
            ix0 = i0.view(-1, 1, 1)
            ix1 = i1.view(1, -1, 1)
            ix2 = i2.view(1, 1, -1)
        else:
            ix0 = i0.reshape(-1, 1, 1)
            ix1 = i1.reshape(1, -1, 1)
            ix2 = i2.reshape(1, 1, -1)

        if mode == "overwrite":
            out[:, ix0, ix1, ix2] = crop
        else:  # average
            out[:, ix0, ix1, ix2] = out[:, ix0, ix1, ix2] + crop
            count[:, ix0, ix1, ix2] = count[:, ix0, ix1, ix2] + 1

    if mode == "average":
        if is_torch:
            count = torch.clamp(count, min=1)
            out = out / count
        else:
            count = np.maximum(count, 1)
            out = (out / count).astype(crops[0].dtype)
    return out
