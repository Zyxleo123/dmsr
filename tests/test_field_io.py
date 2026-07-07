import numpy as np
import pytest

from cosmo_sr.data.field_io import (
    load_field,
    save_field,
    assert_channel_first_3d,
    split_disp_vel,
    merge_disp_vel,
)


def test_save_load_roundtrip_exact(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((6, 16, 16, 16)).astype(np.float32)
    path = tmp_path / "field.npy"
    save_field(path, arr)
    loaded = load_field(path)
    assert loaded.shape == (6, 16, 16, 16)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, arr)


def test_assert_channel_last_fails():
    with pytest.raises(ValueError):
        assert_channel_first_3d(np.zeros((16, 16, 16, 6)))


def test_assert_allowed_channels():
    assert_channel_first_3d(np.zeros((6, 4, 4, 4)), allowed_channels=[6])
    with pytest.raises(ValueError):
        assert_channel_first_3d(np.zeros((5, 4, 4, 4)), allowed_channels=[6])


def test_split_merge_reconstructs_exactly():
    rng = np.random.default_rng(1)
    disp = rng.standard_normal((3, 8, 8, 8)).astype(np.float32)
    vel = rng.standard_normal((3, 8, 8, 8)).astype(np.float32)
    field6 = merge_disp_vel(disp, vel)
    assert field6.shape == (6, 8, 8, 8)
    d2, v2 = split_disp_vel(field6)
    np.testing.assert_array_equal(d2, disp)
    np.testing.assert_array_equal(v2, vel)


def test_noncubic_rejected():
    with pytest.raises(ValueError):
        assert_channel_first_3d(np.zeros((6, 16, 16, 8)))
