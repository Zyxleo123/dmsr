"""Section 2: paired residual targets, the cache, and split hygiene."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cosmo_sr.reward.targets import (BoxPaths, PairedResidualCrops, ResidualTargetCache,
                                     residual_stats, residual_target, resolve_boxes)


SF = 4
NG = 32


@pytest.fixture
def fake_boxes(tmp_path):
    """Two tiny paired boxes plus their cached 'SR2' base fields."""
    root = tmp_path / "data"
    (root / "lr").mkdir(parents=True)
    (root / "hr").mkdir(parents=True)
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    rng = np.random.default_rng(0)
    out = []
    for i, b in enumerate(("set0", "set1")):
        hr = rng.normal(0, 0.05, size=(6, NG, NG, NG)).astype(np.float32)
        base = hr + rng.normal(0, 0.005, size=hr.shape).astype(np.float32)
        lr = hr.reshape(6, NG // SF, SF, NG // SF, SF, NG // SF, SF).mean(axis=(2, 4, 6))
        np.save(root / "hr" / f"{b}.npy", hr)
        np.save(root / "lr" / f"{b}.npy", lr.astype(np.float32))
        np.save(base_dir / f"{b}_seed0_deadbeef.npy", base)
        out.append(b)
    return root, base_dir, out


def test_base_plus_residual_target_reconstructs_hr(fake_boxes):
    root, base_dir, boxes = fake_boxes
    bp = resolve_boxes(boxes, root, base_dir=base_dir)[0]
    hr = np.load(bp.hr)
    base = np.load(bp.base)
    assert np.allclose(base + residual_target(hr, base), hr, atol=1e-6)


def test_reconstruction_survives_normalize_denormalize(fake_boxes):
    from cosmo_sr.reward.diffusion import unwhiten, whiten

    root, base_dir, boxes = fake_boxes
    bp = resolve_boxes(boxes, root, base_dir=base_dir)[0]
    hr = torch.from_numpy(np.load(bp.hr)).unsqueeze(0)
    base = torch.from_numpy(np.load(bp.base)).unsqueeze(0)
    r = hr - base
    sigma = r.flatten(2).std(dim=2).squeeze(0).clamp_min(1e-8)
    back = unwhiten(whiten(r, sigma), sigma)
    assert torch.allclose(base + back, hr, atol=1e-6)


def test_cached_and_on_the_fly_residuals_agree(fake_boxes, tmp_path):
    root, base_dir, boxes = fake_boxes
    bp = resolve_boxes(boxes, root, base_dir=base_dir)[0]
    cache = ResidualTargetCache(tmp_path / "resid")
    p = cache.materialize(bp, "train", chunk=8)
    cached = np.load(p)
    live = np.load(bp.hr) - np.load(bp.base)
    assert np.allclose(cached, live, atol=1e-6)


def test_crops_are_lr_aligned_and_cover_the_same_region(fake_boxes):
    root, base_dir, boxes = fake_boxes
    bps = resolve_boxes(boxes, root, base_dir=base_dir)
    ds = PairedResidualCrops(bps, crop_hr=16, scale_factor=SF, length=8, seed=3)
    for i in range(8):
        item = ds[i]
        start = item["hr_start"].numpy()
        assert (start % SF == 0).all(), "crop start must sit on the LR lattice"
        assert item["y_lr"].shape[-1] * SF == item["psi_base"].shape[-1]
        assert item["residual"].shape == item["psi_base"].shape


def test_crop_residual_equals_hr_minus_base_on_the_same_crop(fake_boxes):
    from cosmo_sr.data.crops import periodic_crop

    root, base_dir, boxes = fake_boxes
    bps = resolve_boxes(boxes, root, base_dir=base_dir)
    ds = PairedResidualCrops(bps, crop_hr=16, scale_factor=SF, length=4, seed=1)
    item = ds[2]
    bp = bps[int(item["box_index"])]
    start = tuple(int(s) for s in item["hr_start"])
    hr = periodic_crop(np.load(bp.hr), start, 16, pad=0)
    base = periodic_crop(np.load(bp.base), start, 16, pad=0)
    assert np.allclose(item["residual"].numpy(), hr - base, atol=1e-6)
    assert np.allclose(item["psi_base"].numpy(), base, atol=1e-6)


def test_periodic_crop_wraps_rather_than_clipping(fake_boxes):
    from cosmo_sr.data.crops import periodic_crop

    root, base_dir, boxes = fake_boxes
    hr = np.load(root / "hr" / "set0.npy")
    # A crop starting near the far corner must wrap back to index 0.
    c = periodic_crop(hr, (NG - 4, NG - 4, NG - 4), 8, pad=0)
    assert c.shape[-1] == 8
    assert np.allclose(c[:, 4:, 4:, 4:], hr[:, 0:4, 0:4, 0:4])


def test_cache_manifest_detects_a_split_leak(fake_boxes, tmp_path):
    root, base_dir, boxes = fake_boxes
    bps = resolve_boxes(boxes, root, base_dir=base_dir)
    cache = ResidualTargetCache(tmp_path / "resid")
    cache.record(bps[0], "train", None)
    cache.record(bps[1], "val", None)
    cache.assert_split(["set0"], "train")
    with pytest.raises(RuntimeError, match="split leak"):
        cache.assert_split(["set1"], "train")


def test_dataset_refuses_boxes_without_a_cached_base(tmp_path, fake_boxes):
    root, base_dir, boxes = fake_boxes
    bps = resolve_boxes(["set0", "setMISSING"], root, base_dir=base_dir)
    with pytest.raises(FileNotFoundError):
        PairedResidualCrops(bps, crop_hr=16, scale_factor=SF)


def test_crop_size_must_be_a_multiple_of_the_scale_factor(fake_boxes):
    root, base_dir, boxes = fake_boxes
    bps = resolve_boxes(boxes, root, base_dir=base_dir)
    with pytest.raises(ValueError, match="multiple of"):
        PairedResidualCrops(bps, crop_hr=18, scale_factor=SF)


def test_residual_stats_report_the_bands_the_audit_needs(fake_boxes):
    root, base_dir, boxes = fake_boxes
    hr = np.load(root / "hr" / "set0.npy")
    base = np.load(base_dir / "set0_seed0_deadbeef.npy")
    st = residual_stats(hr, base, scale_factor=SF, n_bins=8, channels=(0,))
    assert len(st["residual_mean"]) == 6
    assert len(st["residual_std"]) == 6
    sp = st["spectra"]["ch0"]
    for k in ("P_hr", "P_base", "P_residual", "residual_power_low_k",
              "residual_power_transition_k", "residual_power_high_k",
              "frac_residual_power_below_lr_nyquist"):
        assert k in sp
    assert 0.0 <= sp["frac_residual_power_below_lr_nyquist"] <= 1.0
    assert st["residual_absmax_slab_distribution"]["p99"] > 0


def test_white_noise_residual_is_mostly_above_the_lr_nyquist(fake_boxes):
    # The audit's core check: a genuinely small-scale residual must NOT put most
    # of its power below the LR Nyquist. If it does, something is misaligned.
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.05, size=(6, NG, NG, NG)).astype(np.float32)
    hr = base + rng.normal(0, 0.005, size=base.shape).astype(np.float32)
    st = residual_stats(hr, base, scale_factor=SF, n_bins=8, channels=(0,))
    frac = st["spectra"]["ch0"]["frac_residual_power_below_lr_nyquist"]
    assert frac < 0.5
