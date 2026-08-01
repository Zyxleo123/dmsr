"""Rung 3: crops restricted to a handful of fixed massive-host chunks.

Restriction is the whole point of the rung -- if it silently fell back to
sampling the whole box, the run would look like a short prior fine-tune and the
"reward did not move" conclusion would be about the wrong experiment.
"""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.reward.geometry import ChunkGrid
from cosmo_sr.reward.targets import BoxPaths, PairedResidualCrops

NG = 64
SCALE = 4
CHUNK = 32


@pytest.fixture
def boxes(tmp_path):
    rng = np.random.default_rng(0)
    out = []
    for b in ("set0", "set1"):
        lr = (tmp_path / f"{b}_lr.npy")
        hr = (tmp_path / f"{b}_hr.npy")
        base = (tmp_path / f"{b}_base.npy")
        np.save(lr, rng.normal(0, 1, (6, NG // SCALE, NG // SCALE, NG // SCALE)).astype(np.float32))
        np.save(hr, rng.normal(0, 1, (6, NG, NG, NG)).astype(np.float32))
        np.save(base, rng.normal(0, 1, (6, NG, NG, NG)).astype(np.float32))
        out.append(BoxPaths(box=b, lr=lr, hr=hr, base=base))
    return out


def _starts(ds, n=64):
    return [tuple(int(v) for v in ds[i]["hr_start"]) for i in range(n)]


def test_unrestricted_crops_still_roam_the_whole_box(boxes):
    ds = PairedResidualCrops(boxes, crop_hr=16, scale_factor=SCALE, length=64, seed=0)
    starts = _starts(ds)
    assert len({s for s in starts}) > 8, "the default must stay random"


def test_fixed_host_crops_stay_inside_their_chunks(boxes):
    grid = ChunkGrid(ng_hr=NG, chunk_hr=CHUNK)
    picked = [("set0", 0), ("set0", 7)]
    ds = PairedResidualCrops(boxes, crop_hr=16, scale_factor=SCALE, length=64,
                             seed=0, host_chunks=picked, chunk_hr=CHUNK)

    allowed = []
    for _, cid in picked:
        o = grid.origin(cid)
        allowed.append((o, tuple(v + CHUNK - 16 for v in o)))

    for s in _starts(ds):
        assert any(
            all(lo[d] <= s[d] <= hi[d] for d in range(3)) for lo, hi in allowed
        ), f"crop at {s} escaped the fixed chunks"


def test_fixed_host_crops_only_come_from_the_named_box(boxes):
    ds = PairedResidualCrops(boxes, crop_hr=16, scale_factor=SCALE, length=64,
                             seed=0, host_chunks=[("set1", 3)], chunk_hr=CHUNK)
    assert {int(ds[i]["box_index"]) for i in range(32)} == {1}


def test_crops_stay_aligned_to_the_lr_lattice(boxes):
    """Misalignment would offset the conditioning window by a sub-cell shift."""
    ds = PairedResidualCrops(boxes, crop_hr=16, scale_factor=SCALE, length=32,
                             seed=0, host_chunks=[("set0", 5)], chunk_hr=CHUNK)
    for s in _starts(ds, n=32):
        assert all(v % SCALE == 0 for v in s)


def test_a_crop_larger_than_the_chunk_is_refused(boxes):
    with pytest.raises(ValueError, match="exceeds chunk_hr"):
        PairedResidualCrops(boxes, crop_hr=48, scale_factor=SCALE, length=8,
                            seed=0, host_chunks=[("set0", 0)], chunk_hr=CHUNK)


def test_fixed_hosts_naming_an_unloaded_box_is_refused(boxes):
    with pytest.raises(ValueError, match="none of the fixed host chunks"):
        PairedResidualCrops(boxes, crop_hr=16, scale_factor=SCALE, length=8,
                            seed=0, host_chunks=[("set9", 0)], chunk_hr=CHUNK)
