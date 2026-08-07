"""A training crop must be bit-identical to its slice of a full-box inference.

If it is not, every per-tile catalog label in this line is attached to a tensor
the box was not made of, and no amount of downstream care recovers from that.
So the central test here is an *exact* equality, not a tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

# Fixtures live in a uniquely-named module rather than a conftest.py; see
# tests/train/sr2_fixtures.py for why a second conftest.py breaks the suite.
from sr2_fixtures import (  # noqa: F401
    chan_kwargs, generator, geom, hr_field, lr_field, model_path,
)

from cosmo_sr.reward.tiles import TileGrid
from cosmo_sr.tts.sampling import super_resolve_srs_seeded, tile_starts
from cosmo_sr.train.sr2_finetune_data import (
    SR2TileDataset, collate_tiles, fold_draws, frozen_tile_forward,
    lr_start_of_tile_id, tile_id_of_lr_start, tile_lr_crop, tile_noise_stack,
    trim_to_tile,
)


def test_tile_id_and_lr_start_are_inverses(geom):
    starts = tile_starts(geom.ng_lr, geom.nsplit)
    assert len(starts) == geom.n_tiles
    for tile_id, start in enumerate(starts):
        # tile_starts walks (ix, iy, iz) in C order and TileGrid.index maps the
        # same triple the same way; this pins that the two agree.
        assert tile_id_of_lr_start(start, geom) == tile_id
        assert lr_start_of_tile_id(tile_id, geom) == tuple(start)


def test_tile_grid_slices_match_lr_start(geom):
    grid = geom.tile_grid()
    assert isinstance(grid, TileGrid)
    for tile_id in range(geom.n_tiles):
        sx, sy, sz = grid.slices(tile_id)
        lr_start = lr_start_of_tile_id(tile_id, geom)
        assert (sx.start, sy.start, sz.start) == tuple(
            s * geom.scale_factor for s in lr_start
        )
        assert sx.stop - sx.start == geom.tile_hr


def test_crop_matches_full_box_inference_exactly(geom, generator, lr_field):
    """The whole reason this module exists."""
    seed = 1234
    full = super_resolve_srs_seeded(
        generator, lr_field, seed, scale_factor=geom.scale_factor,
        nsplit=geom.nsplit, pad=geom.pad, device=torch.device("cpu"),
        noise_mode="per_tile",
    )
    grid = geom.tile_grid()
    for tile_id in range(geom.n_tiles):
        crop = tile_lr_crop(lr_field, tile_id, geom)
        assert crop.shape == (6, geom.lr_size, geom.lr_size, geom.lr_size)
        noise = tile_noise_stack([seed], tile_id, geom)
        z = {k: v[0:1] for k, v in noise.items()}
        out = frozen_tile_forward(
            generator, torch.from_numpy(crop).unsqueeze(0), z, geom
        ).squeeze(0).numpy()
        sx, sy, sz = grid.slices(tile_id)
        expected = full[:, sx, sy, sz]
        assert out.shape == expected.shape
        # Exact: same weights, same input, same noise, same trim.
        assert np.array_equal(out, expected), f"tile {tile_id} differs"


def test_noise_stack_is_reproducible_and_seed_dependent(geom):
    a = tile_noise_stack([5], 3, geom)
    b = tile_noise_stack([5], 3, geom)
    c = tile_noise_stack([6], 3, geom)
    for site in a:
        assert torch.equal(a[site], b[site])
        assert not torch.equal(a[site], c[site])


def test_noise_site_shapes_match_generator_expectations(geom, generator, lr_field):
    sizes = geom.site_sizes()
    noise = tile_noise_stack([0], 0, geom)
    assert set(noise) == set(sizes)
    for site, size in sizes.items():
        assert tuple(noise[site].shape) == (1, 1, size, size, size)
    crop = torch.from_numpy(tile_lr_crop(lr_field, 0, geom)).unsqueeze(0)
    out = generator(crop, noise={k: v for k, v in noise.items()})
    assert trim_to_tile(out, geom).shape[-1] == geom.tile_hr


def _dataset(geom, generator, lr_field, hr_field, frozen_field, **kw):
    return SR2TileDataset(
        boxes=["setT"], lr_fields={"setT": lr_field}, hr_fields={"setT": hr_field},
        geom=geom, frozen_fields={"setT": frozen_field},
        frozen_generator=generator, **kw,
    )


@pytest.fixture
def frozen_field(geom, generator, lr_field):
    return super_resolve_srs_seeded(
        generator, lr_field, 0, scale_factor=geom.scale_factor,
        nsplit=geom.nsplit, pad=geom.pad, device=torch.device("cpu"),
    )


def test_dataset_item_shapes_and_frozen_slice(geom, generator, lr_field, hr_field,
                                              frozen_field):
    ds = _dataset(geom, generator, lr_field, hr_field, frozen_field)
    assert len(ds) == geom.n_tiles
    item = ds[5]
    assert item["lr"].shape == (6, geom.lr_size, geom.lr_size, geom.lr_size)
    assert item["hr"].shape == (6, geom.tile_hr, geom.tile_hr, geom.tile_hr)
    assert item["frozen"].shape == (1, 6, geom.tile_hr, geom.tile_hr, geom.tile_hr)
    grid = geom.tile_grid()
    sx, sy, sz = grid.slices(5)
    assert np.array_equal(item["frozen"][0].numpy(), frozen_field[:, sx, sy, sz])


def test_frozen_generated_draw_matches_cached_slice(geom, generator, lr_field,
                                                    hr_field, frozen_field):
    """Draw 0 taken from the cache and draw 0 recomputed must agree exactly."""
    cached = _dataset(geom, generator, lr_field, hr_field, frozen_field)
    live = SR2TileDataset(
        boxes=["setT"], lr_fields={"setT": lr_field}, hr_fields={"setT": hr_field},
        geom=geom, frozen_generator=generator,
    )
    assert np.array_equal(cached[2]["frozen"][0].numpy(), live[2]["frozen"][0].numpy())


def test_two_draws_share_the_lr_tile_and_differ_only_in_noise(
        geom, generator, lr_field, hr_field, frozen_field):
    ds = _dataset(geom, generator, lr_field, hr_field, frozen_field, noise_draws=2)
    item = ds[1]
    assert item["frozen"].shape[0] == 2
    assert item["seeds"].tolist() == ds.seeds_for(1)
    for site, z in item["noise"].items():
        assert z.shape[0] == 2
        assert not torch.equal(z[0], z[1]), f"{site} identical across draws"
    # Seed diversity, not tile diversity: the two draws are the SAME LR tile.
    assert not torch.equal(item["frozen"][0], item["frozen"][1])


def test_collate_and_fold_draws_shapes(geom, generator, lr_field, hr_field,
                                       frozen_field):
    ds = _dataset(geom, generator, lr_field, hr_field, frozen_field, noise_draws=2)
    batch = collate_tiles([ds[0], ds[3]])
    assert batch["box"] == ["setT", "setT"]
    assert batch["lr"].shape == (2, 6, geom.lr_size, geom.lr_size, geom.lr_size)
    assert batch["frozen"].shape == (2, 2, 6, geom.tile_hr, geom.tile_hr, geom.tile_hr)
    lr, noise, b, d = fold_draws(batch)
    assert (b, d) == (2, 2)
    assert lr.shape[0] == 4
    for site, z in noise.items():
        assert z.shape[0] == 4 and z.shape[1] == 1
    # The repeated LR rows must be identical copies of their example's tile.
    assert torch.equal(lr[0], lr[1])
    assert torch.equal(lr[2], lr[3])
    assert not torch.equal(lr[0], lr[2])


def test_multi_draw_without_generator_is_refused(geom, lr_field, hr_field):
    with pytest.raises(ValueError, match="frozen_generator"):
        SR2TileDataset(
            boxes=["setT"], lr_fields={"setT": lr_field},
            hr_fields={"setT": hr_field}, geom=geom, noise_draws=2,
        )


def test_grid_mismatch_is_refused(geom, lr_field, hr_field):
    with pytest.raises(ValueError, match="HR grid"):
        SR2TileDataset(
            boxes=["setT"], lr_fields={"setT": lr_field},
            hr_fields={"setT": hr_field[:, :32, :32, :32]}, geom=geom,
        )
