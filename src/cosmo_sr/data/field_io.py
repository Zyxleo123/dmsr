"""Canonical field I/O and layout helpers.

Canonical field format::

    field.shape == (C, N, N, N)   # channel-first, cubic

For SRS-style data ``C == 6`` where channels ``0:3`` are displacement and
``3:6`` are velocity. Fields are stored as ``float32`` by default.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np

DEFAULT_DTYPE = np.float32


def assert_channel_first_3d(
    array: np.ndarray,
    allowed_channels: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Assert ``array`` is a channel-first cubic 3D field ``(C, N, N, N)``.

    Parameters
    ----------
    array:
        Candidate field.
    allowed_channels:
        If given, the channel count ``C`` must be one of these values.

    Raises
    ------
    TypeError
        If ``array`` is not a numpy array.
    ValueError
        If the array is not 4D, not cubic, or has a disallowed channel count.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected a numpy.ndarray, got {type(array)!r}")
    if array.ndim != 4:
        raise ValueError(
            f"Expected a 4D channel-first field (C, N, N, N), got shape {array.shape}"
        )
    c, nx, ny, nz = array.shape
    if not (nx == ny == nz):
        raise ValueError(
            "Only cubic fields are supported; got non-cubic spatial dims "
            f"{(nx, ny, nz)}. Non-cubic arrays are explicitly rejected."
        )
    if nx == 0:
        raise ValueError("Field has zero spatial extent.")
    if allowed_channels is not None and c not in set(allowed_channels):
        raise ValueError(
            f"Channel count {c} not in allowed set {sorted(set(allowed_channels))}."
        )
    return array


def load_field(path: str | os.PathLike, mmap: bool = False) -> np.ndarray:
    """Load a canonical ``(C, N, N, N)`` field from a ``.npy`` file.

    Set ``mmap=True`` to memory-map large fields (e.g. Ng=512 HR fields are
    ~3 GB each); cropping then reads only the needed slices from disk.
    """
    array = np.load(os.fspath(path), mmap_mode="r" if mmap else None)
    assert_channel_first_3d(array)
    return array


def save_field(path: str | os.PathLike, array: np.ndarray) -> None:
    """Save a canonical ``(C, N, N, N)`` field to a ``.npy`` file as float32.

    The array is validated before saving. Values are cast to ``float32`` (the
    canonical dtype); a ``float32`` input is therefore preserved exactly.
    """
    assert_channel_first_3d(array)
    out = np.ascontiguousarray(array, dtype=DEFAULT_DTYPE)
    path = os.fspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.save(path, out)


def split_disp_vel(field6: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a 6-channel field into ``(disp3, vel3)`` views.

    ``disp = field6[0:3]`` and ``vel = field6[3:6]``.
    """
    assert_channel_first_3d(field6, allowed_channels=[6])
    disp3 = field6[0:3]
    vel3 = field6[3:6]
    return disp3, vel3


def merge_disp_vel(disp3: np.ndarray, vel3: np.ndarray) -> np.ndarray:
    """Concatenate displacement and velocity into a 6-channel field."""
    assert_channel_first_3d(disp3, allowed_channels=[3])
    assert_channel_first_3d(vel3, allowed_channels=[3])
    if disp3.shape != vel3.shape:
        raise ValueError(
            f"disp/vel spatial shapes must match, got {disp3.shape} vs {vel3.shape}"
        )
    return np.concatenate([disp3, vel3], axis=0)
