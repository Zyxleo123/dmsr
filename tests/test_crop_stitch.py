import numpy as np
import pytest
import torch

from cosmo_sr.data.crops import periodic_crop, make_crop_grid, stitch_crops


def _field(N=8, C=2):
    rng = np.random.default_rng(0)
    return rng.standard_normal((C, N, N, N)).astype(np.float32)


def test_crop_inside_matches_slicing():
    field = _field(8)
    crop = periodic_crop(field, start=(2, 1, 3), crop_size=3, pad=0)
    ref = field[:, 2:5, 1:4, 3:6]
    np.testing.assert_array_equal(crop, ref)


def test_crop_wraps_right_boundary():
    field = _field(8)
    # start near the right edge so the crop wraps to the left boundary
    crop = periodic_crop(field, start=(6, 0, 0), crop_size=4, pad=0)
    # axis 0 wraps [6,7,0,1]; axes 1,2 are the plain [0:4] slice
    expected = np.concatenate([field[:, 6:8], field[:, 0:2]], axis=1)[:, :, 0:4, 0:4]
    np.testing.assert_array_equal(crop, expected)


def test_nonoverlapping_reconstruction_numpy():
    field = _field(8)
    cs = 4
    starts = make_crop_grid(field.shape, crop_size=cs, stride=cs)
    crops = [periodic_crop(field, s, cs, pad=0) for s in starts]
    recon = stitch_crops(crops, starts, field.shape, mode="overwrite")
    np.testing.assert_array_equal(recon, field)


def test_nonoverlapping_reconstruction_torch():
    field = torch.from_numpy(_field(8))
    cs = 4
    starts = make_crop_grid(tuple(field.shape), crop_size=cs, stride=cs)
    crops = [periodic_crop(field, s, cs, pad=0) for s in starts]
    recon = stitch_crops(crops, starts, tuple(field.shape), mode="overwrite")
    assert torch.equal(recon, field)


def test_overlapping_average_reconstructs_identical_source():
    field = _field(8)
    cs = 4
    starts = make_crop_grid(field.shape, crop_size=cs, stride=2)
    crops = [periodic_crop(field, s, cs, pad=0) for s in starts]
    recon = stitch_crops(crops, starts, field.shape, mode="average")
    np.testing.assert_allclose(recon, field, atol=1e-6)


def test_padded_crop_size():
    field = _field(8)
    crop = periodic_crop(field, start=(0, 0, 0), crop_size=4, pad=1)
    assert crop.shape == (2, 6, 6, 6)


def test_make_crop_grid_count():
    starts = make_crop_grid((2, 8, 8, 8), crop_size=4, stride=4)
    assert len(starts) == 2 ** 3
