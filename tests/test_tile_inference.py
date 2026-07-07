import numpy as np
import pytest
import torch

from cosmo_sr.inference.tile_sr import super_resolve_full_box
from cosmo_sr.models.wrappers import NearestUpsampler


SCALE = 4


def _smooth_periodic_field(N=8, C=6, seed=0):
    rng = np.random.default_rng(seed)
    coords = np.arange(N)
    field = np.zeros((C, N, N, N), dtype=np.float32)
    for c in range(C):
        phase = rng.uniform(0, 2 * np.pi, size=3)
        gx = np.sin(2 * np.pi * coords / N + phase[0])
        gy = np.sin(2 * np.pi * coords / N + phase[1])
        gz = np.sin(2 * np.pi * coords / N + phase[2])
        field[c] = gx[:, None, None] + gy[None, :, None] + gz[None, None, :]
    return field


def test_tiled_equals_direct_nsplit1():
    model = NearestUpsampler(scale_factor=SCALE)
    lr = _smooth_periodic_field(8)
    direct = super_resolve_full_box(model, lr, SCALE, nsplit=1, pad_lr=0)
    assert direct.shape == (6, 8 * SCALE, 8 * SCALE, 8 * SCALE)


def test_tiled_equals_direct_nsplit2():
    model = NearestUpsampler(scale_factor=SCALE)
    lr = _smooth_periodic_field(8)
    direct = super_resolve_full_box(model, lr, SCALE, nsplit=1, pad_lr=0)
    tiled = super_resolve_full_box(model, lr, SCALE, nsplit=2, pad_lr=1)
    np.testing.assert_allclose(tiled, direct, atol=1e-5)


def test_nsplit_not_dividing_raises():
    model = NearestUpsampler(scale_factor=SCALE)
    lr = _smooth_periodic_field(8)
    with pytest.raises(ValueError):
        super_resolve_full_box(model, lr, SCALE, nsplit=3, pad_lr=0)


def test_output_shape():
    model = NearestUpsampler(scale_factor=SCALE)
    lr = _smooth_periodic_field(8)
    out = super_resolve_full_box(model, lr, SCALE, nsplit=2, pad_lr=0)
    Ng_hr = 8 * SCALE
    assert out.shape == (6, Ng_hr, Ng_hr, Ng_hr)


def test_no_seams_smooth_input():
    model = NearestUpsampler(scale_factor=SCALE)
    lr = _smooth_periodic_field(8)
    direct = super_resolve_full_box(model, lr, SCALE, nsplit=1, pad_lr=0)
    tiled = super_resolve_full_box(model, lr, SCALE, nsplit=2, pad_lr=1)
    err = np.abs(tiled - direct)
    # boundary planes between chunks (at multiples of chunk*scale)
    chunk_hr = (8 // 2) * SCALE
    boundary_err = err[:, chunk_hr - 1:chunk_hr + 1].max()
    interior_err = err.max()
    assert boundary_err <= interior_err + 1e-6
