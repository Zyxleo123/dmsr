"""Tests for FieldCropDataset overfit / fixed-crop memorization mode."""
from __future__ import annotations

import numpy as np
import torch

from cosmo_sr.data.datasets import FieldCropDataset
from cosmo_sr.data.field_io import save_field


def _write_pair(tmp_path, ng_lr: int = 8, scale: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    lr = rng.standard_normal((6, ng_lr, ng_lr, ng_lr)).astype(np.float32)
    hr = rng.standard_normal((6, ng_lr * scale, ng_lr * scale, ng_lr * scale)).astype(
        np.float32
    )
    lr_path = tmp_path / "lr.npy"
    hr_path = tmp_path / "hr.npy"
    save_field(lr_path, lr)
    save_field(hr_path, hr)
    return str(lr_path), str(hr_path)


def test_fixed_crops_cycle_identical_samples(tmp_path):
    lr_path, hr_path = _write_pair(tmp_path)
    ds = FieldCropDataset(
        [lr_path], [hr_path], crop_lr=4, scale_factor=2, seed=0, fixed_crops=3, length=12
    )
    assert ds.fixed_crops == 3
    assert len(ds._fixed_samples) == 3

    a0 = ds[0]
    a3 = ds[3]  # same slot in the cycle
    assert torch.equal(a0["lr"], a3["lr"])
    assert torch.equal(a0["hr"], a3["hr"])

    # Distinct crops within the closed set (extremely unlikely all equal).
    diffs = [
        float((ds[i]["lr"] - ds[j]["lr"]).abs().sum())
        for i, j in ((0, 1), (0, 2), (1, 2))
    ]
    assert max(diffs) > 0.0


def test_fixed_crops_clone_is_safe(tmp_path):
    lr_path, hr_path = _write_pair(tmp_path, seed=1)
    ds = FieldCropDataset(
        [lr_path], [hr_path], crop_lr=4, scale_factor=2, seed=1, fixed_crops=1
    )
    s0 = ds[0]
    s0["lr"].zero_()
    s1 = ds[0]
    assert float(s1["lr"].abs().sum()) > 0.0


def test_same_seed_same_fixed_set(tmp_path):
    lr_path, hr_path = _write_pair(tmp_path, seed=2)
    kwargs = dict(
        lr_paths=[lr_path],
        hr_paths=[hr_path],
        crop_lr=4,
        scale_factor=2,
        seed=7,
        fixed_crops=2,
    )
    ds_a = FieldCropDataset(**kwargs)
    ds_b = FieldCropDataset(**kwargs)
    assert torch.equal(ds_a[0]["lr"], ds_b[0]["lr"])
    assert torch.equal(ds_a[1]["hr"], ds_b[1]["hr"])
