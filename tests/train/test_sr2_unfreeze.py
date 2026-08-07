"""The unfreezing rungs are a safety argument, so they are tested as one.

Two claims are pinned: each rung names *exactly* the parameters it claims to,
and after a real optimizer step nothing outside that set has moved by a single
bit.
"""
from __future__ import annotations

import pytest
import torch

# Fixtures live in a uniquely-named module rather than a conftest.py; see
# tests/train/sr2_fixtures.py for why a second conftest.py breaks the suite.
from sr2_fixtures import (  # noqa: F401
    chan_kwargs, generator, geom, hr_field, lr_field, model_path,
)

from cosmo_sr.train import sr2_unfreeze as U


PROJ_NOISE_EXPECTED = sorted([
    *[f"blocks.{b}.proj.0.{w}" for b in range(3) for w in ("weight", "bias")],
    *[f"blocks.{b}.conv.{i}.std" for b in range(3) for i in (0, 4)],
])
FINE_CONVS = sorted(f"blocks.2.conv.{i}.{w}" for i in (2, 5)
                    for w in ("weight", "bias"))
MIDDLE_CONVS = sorted(f"blocks.1.conv.{i}.{w}" for i in (2, 5)
                      for w in ("weight", "bias"))
COARSE_CONVS = sorted(f"blocks.0.conv.{i}.{w}" for i in (2, 5)
                      for w in ("weight", "bias"))
BLOCK0 = ["block0.0.bias", "block0.0.weight"]


@pytest.mark.parametrize("rung,expected", [
    ("proj_noise", PROJ_NOISE_EXPECTED),
    ("fine", PROJ_NOISE_EXPECTED + FINE_CONVS),
    ("middle_fine", PROJ_NOISE_EXPECTED + FINE_CONVS + MIDDLE_CONVS),
    ("all_blocks", PROJ_NOISE_EXPECTED + FINE_CONVS + MIDDLE_CONVS + COARSE_CONVS),
    ("full", PROJ_NOISE_EXPECTED + FINE_CONVS + MIDDLE_CONVS + COARSE_CONVS + BLOCK0),
])
def test_rung_names_are_exact(generator, rung, expected):
    assert U.rung_names(generator, rung) == sorted(expected)


def test_rungs_are_nested_and_ordered(generator):
    sets = [set(U.rung_names(generator, r)) for r in U.RUNG_ORDER]
    for a, b in zip(sets[:-1], sets[1:]):
        assert a < b, "each rung must strictly contain the previous one"
    assert sets[-1] == {n for n, _ in generator.named_parameters()}


def test_unknown_rung_is_refused(generator):
    with pytest.raises(ValueError, match="unknown rung"):
        U.rung_names(generator, "everything")


def test_set_trainable_freezes_the_rest(generator):
    names = U.set_trainable(generator, "fine")
    assert U.trainable_names(generator) == sorted(names)
    frozen = [n for n, p in generator.named_parameters() if not p.requires_grad]
    assert "block0.0.weight" in frozen
    assert "blocks.1.conv.2.weight" in frozen


def test_set_trainable_is_not_cumulative(generator):
    """Entering a *lower* rung after a higher one must shrink the set."""
    U.set_trainable(generator, "full")
    U.set_trainable(generator, "proj_noise")
    assert U.trainable_names(generator) == sorted(PROJ_NOISE_EXPECTED)


def test_learning_rate_groups(generator):
    groups = U.parameter_groups(generator, "middle_fine", {"proj_noise": 1e-4})
    by_name = {g["name"]: g for g in groups}
    assert set(by_name) == {"proj_noise", "fine", "middle"}
    assert by_name["proj_noise"]["lr"] == pytest.approx(1e-4)
    assert by_name["fine"]["lr"] == pytest.approx(U.DEFAULT_GROUP_LR["fine"])
    assert by_name["middle"]["lr"] == pytest.approx(U.DEFAULT_GROUP_LR["middle"])
    # No empty group survives: an empty group in a log reads as a trained one.
    assert all(g["params"] for g in groups)


def test_unknown_lr_group_is_refused(generator):
    with pytest.raises(KeyError, match="unknown learning-rate group"):
        U.parameter_groups(generator, "fine", {"deep": 1e-9})


def test_group_of_parameter_assignments():
    assert U.group_of_parameter("blocks.0.conv.0.std") == "proj_noise"
    assert U.group_of_parameter("blocks.1.proj.0.weight") == "proj_noise"
    assert U.group_of_parameter("blocks.2.conv.2.weight") == "fine"
    assert U.group_of_parameter("blocks.1.conv.5.bias") == "middle"
    assert U.group_of_parameter("blocks.0.conv.2.weight") == "coarse"
    assert U.group_of_parameter("block0.0.weight") == "coarse"


def test_describe_trainable_counts(generator):
    d = U.describe_trainable(generator, "proj_noise")
    assert d["n_trainable_tensors"] == len(PROJ_NOISE_EXPECTED)
    assert [r["name"] for r in d["parameters"]] == sorted(PROJ_NOISE_EXPECTED)
    assert d["n_trainable_params"] < d["n_total_params"]
    assert 0.0 < d["trainable_fraction"] < 1.0


@pytest.mark.parametrize("rung", list(U.RUNG_ORDER))
def test_one_optimizer_step_moves_only_the_rung(generator, rung, lr_field, geom):
    from cosmo_sr.train.sr2_finetune_data import (
        tile_lr_crop, tile_noise_stack, trim_to_tile,
    )

    groups = U.parameter_groups(generator, rung)
    # A large LR so a real update is unambiguous rather than lost in float noise.
    for g in groups:
        g["lr"] = 1e-2
    opt = torch.optim.SGD(groups)
    before = U.snapshot_parameters(generator)

    crop = torch.from_numpy(tile_lr_crop(lr_field, 0, geom)).unsqueeze(0)
    noise = {k: v[0:1] for k, v in tile_noise_stack([0], 0, geom).items()}
    out = trim_to_tile(generator(crop, noise=noise), geom)
    out.pow(2).mean().backward()
    opt.step()

    expected = U.rung_names(generator, rung)
    deltas = U.assert_only_trainable_changed(generator, before, expected)
    moved = [n for n in expected if deltas[n] > 0]
    assert moved, f"rung {rung} took a step but nothing in it moved"


def test_assert_only_trainable_changed_catches_a_leak(generator):
    before = U.snapshot_parameters(generator)
    with torch.no_grad():
        generator.block0[0].weight.add_(1.0)
    with pytest.raises(AssertionError, match=r"block0\.0\.weight"):
        U.assert_only_trainable_changed(generator, before, ["blocks.0.conv.0.std"])


def test_next_rung_walks_and_stops():
    assert U.next_rung("proj_noise") == "fine"
    assert U.next_rung("all_blocks") == "full"
    assert U.next_rung("full") is None
