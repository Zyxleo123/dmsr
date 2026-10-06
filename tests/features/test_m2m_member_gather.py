"""Geometry of the map2map member-gather term (``scripts/m2m_callback/srlosses``).

These are the checks that cannot wait for a GPU: if the LR window is off by a
cell, or the tile-id convention disagrees with ``tile_particles``, the loss
supervises the wrong particles and every number it reports is still finite and
plausible. The expensive half (``build_member_sets`` on a 537 MB owner array,
and the forward itself) runs in the shakeout job.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "m2m_callback"))
sys.path.insert(0, str(_ROOT / "src"))

from srlosses import MemberGather, SCALE_FACTOR, TILE, NG_HR  # noqa: E402


def _bare(pad: int = 3) -> MemberGather:
    """An instance without ``__init__`` -- the data loading is the slow part."""
    mg = object.__new__(MemberGather)
    mg.pad = int(pad)
    return mg


def test_tile_origin_matches_tile_particles_convention():
    """``tile_particles`` derives origins as tx = tid // n^2, ty, tz = ...

    Both must agree or the gathered rows address another tile entirely.
    """
    n = NG_HR // TILE
    for tid in (0, 1, n, n * n, n * n * n - 1, 173):
        tx, ty, tz = tid // (n * n), (tid // n) % n, tid % n
        assert MemberGather._tile_origin(tid) == (tx * TILE, ty * TILE, tz * TILE)


def test_lr_window_puts_the_tile_in_the_centre(tmp_path):
    """The tile's own 8 LR cells must land at local ``[pad, pad + 8)``.

    Built so each LR cell carries its own global index, so a shifted window is
    visible as a value, not just a shape.
    """
    n_lr = NG_HR // SCALE_FACTOR
    ramp = np.arange(n_lr, dtype=np.float32)
    # channel 0 = x index, 1 = y, 2 = z
    d = np.stack([np.broadcast_to(ramp[:, None, None], (n_lr,) * 3),
                  np.broadcast_to(ramp[None, :, None], (n_lr,) * 3),
                  np.broadcast_to(ramp[None, None, :], (n_lr,) * 3)]).copy()
    dp, vp = tmp_path / "disp.npy", tmp_path / "vel.npy"
    np.save(dp, d)
    np.save(vp, np.zeros_like(d))

    mg = _bare()
    tiles = [0, 173, NG_HR // TILE - 1]
    win = mg._lr_windows(str(dp), str(vp), tiles).numpy()

    assert win.shape == (len(tiles), 6, 14, 14, 14)
    for b, tid in enumerate(tiles):
        ox, oy, oz = MemberGather._tile_origin(tid)
        for ax, o in enumerate((ox, oy, oz)):
            lo = o // SCALE_FACTOR
            # the central 8 of 14 are the tile's own LR cells, in order
            sl = [slice(None)] * 3
            sl[ax] = slice(3, 11)
            vals = win[b, ax][tuple(sl)]
            axis_vals = np.moveaxis(vals, ax, 0).reshape(8, -1)[:, 0]
            assert np.array_equal(axis_vals, (np.arange(8) + lo) % (NG_HR // SCALE_FACTOR))


def test_lr_window_wraps_periodically(tmp_path):
    """Tile 0 needs cells -3..-1, which must come from the far edge."""
    n_lr = NG_HR // SCALE_FACTOR
    ramp = np.arange(n_lr, dtype=np.float32)
    d = np.stack([np.broadcast_to(ramp[:, None, None], (n_lr,) * 3)] * 3).copy()
    dp, vp = tmp_path / "d.npy", tmp_path / "v.npy"
    np.save(dp, d)
    np.save(vp, np.zeros_like(d))

    win = _bare()._lr_windows(str(dp), str(vp), [0]).numpy()
    got = win[0, 0, :, 0, 0]
    assert np.array_equal(got[:3], [n_lr - 3, n_lr - 2, n_lr - 1])
    assert got[3] == 0


def test_generator_size_relation_inverts_to_64():
    """out(n) = 8n - 42, so a 64^3 tile needs a 14^3 window and narrow_by(3)."""
    out = lambda n: 8 * n - 42          # noqa: E731
    assert out(15) == 78                # the training run's own printed shape
    assert out(14) == 70
    assert 70 - 2 * 3 == TILE


def test_slice_tile_takes_six_channels():
    box = np.arange(6 * 64 ** 3, dtype=np.float32).reshape(6, 64, 64, 64)
    big = np.zeros((6, NG_HR, NG_HR, NG_HR), dtype=np.float32)
    big[:, :64, :64, :64] = box
    got = MemberGather._slice_tile(big, 0)
    assert got.shape == (6, TILE, TILE, TILE)
    assert np.array_equal(got, box)
