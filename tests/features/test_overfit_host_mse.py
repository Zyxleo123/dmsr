"""Pins for the capacity-vs-incentive run's bookkeeping and its verdict.

The training loop itself needs a GPU and a 3.2 GiB box, so what is pinned here
is everything that would make the result *wrong rather than absent*: the tile
indexing that decides which HR cube is used as the target for which generator
forward pass, the diagnostics that the verdict reads, and the verdict's own
branches -- which are a decision about the design and so are written down like
one.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load():
    for p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    path = PROJECT_ROOT / "scripts" / "features" / "overfit_host_mse.py"
    spec = importlib.util.spec_from_file_location("overfit_host_mse", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()


# --------------------------------------------------------------------------- #
# Tile indexing: the target must be the cube the forward pass produced
# --------------------------------------------------------------------------- #
def test_hr_target_slice_matches_the_generator_tiling():
    """``tile_hr_target`` and ``lr_start_of_tile_id`` must name the same cube.

    A mismatch here trains the generator against a different region of the box
    and still converges to *something*, so nothing downstream would flag it.
    """
    from cosmo_sr.train.sr2_finetune_data import SR2TileGeometry, lr_start_of_tile_id

    geom = SR2TileGeometry()
    for t in (0, 1, 8, 64, 73, 511):
        start = lr_start_of_tile_id(t, geom)
        hr_start = tuple(int(s) * geom.scale_factor for s in start)
        n = M.NG_HR // M.TILE
        ix, iy, iz = t // (n * n), (t // n) % n, t % n
        assert hr_start == (ix * M.TILE, iy * M.TILE, iz * M.TILE)


def test_hr_target_returns_the_right_cube():
    n = M.NG_HR // M.TILE
    field = np.arange(6 * 8 ** 3, dtype=np.float32).reshape(6, 8, 8, 8)
    # shrink the geometry to something a test can hold
    old_ng, old_tile = M.NG_HR, M.TILE
    try:
        M.NG_HR, M.TILE = 8, 4
        got = M.tile_hr_target(field, 7, None)      # (1,1,1) in a 2^3 tiling
        assert got.shape == (6, 4, 4, 4)
        assert np.array_equal(got, field[:, 4:8, 4:8, 4:8])
    finally:
        M.NG_HR, M.TILE = old_ng, old_tile


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
class _Args:
    n_bins = 8
    k_split = 4.0


def _tile(rng, n=32, scale=1e-3):
    return torch.from_numpy(
        (rng.normal(size=(1, 6, n, n, n)) * scale).astype(np.float32))


def test_field_report_scores_a_perfect_copy_at_ratio_one():
    from cosmo_sr.reward.soft_structure import SoftStructureConfig

    rng = np.random.default_rng(0)
    hr = _tile(rng)
    rep = M.field_report(hr.clone(), hr, hr.clone(), SoftStructureConfig(), _Args())
    assert rep["out"]["highk_power_ratio"] == pytest.approx(1.0, rel=1e-4)
    r = np.array(rep["out"]["r"], dtype=float)
    assert np.allclose(r[np.isfinite(r)], 1.0, atol=1e-4)


def test_field_report_sees_a_blurred_output_as_lost_high_k_power():
    """The signature the design predicts an MSE fine-tune will produce."""
    from scipy.ndimage import gaussian_filter
    from cosmo_sr.reward.soft_structure import SoftStructureConfig

    rng = np.random.default_rng(1)
    hr = _tile(rng)
    blur = hr.clone().numpy()
    for c in range(6):
        blur[0, c] = gaussian_filter(blur[0, c], sigma=1.5, mode="wrap")
    rep = M.field_report(torch.from_numpy(blur), hr, hr.clone(),
                         SoftStructureConfig(), _Args())
    assert rep["out"]["highk_power_ratio"] < 0.5
    assert rep["frozen"]["highk_power_ratio"] == pytest.approx(1.0, rel=1e-4)


def test_to_mpc_undoes_the_catnorm_scaling_once():
    from cosmo_sr.data.preprocess_srs import disnorm

    x = torch.ones((1, 6, 4, 4, 4))
    got = M.to_mpc(x)
    assert got.shape == (1, 3, 4, 4, 4)
    assert float(got[0, 0, 0, 0, 0]) == pytest.approx(
        float(disnorm(np.array([1.0]), z=0.0, undo=True)[0] * 1e-3))


# --------------------------------------------------------------------------- #
# The verdict is a decision about the design, so its branches are pinned
# --------------------------------------------------------------------------- #
def _hist(mse_ratio, hk, hk0):
    return [{"step": 3000, "mse_ratio": mse_ratio, "highk_power_ratio": hk,
             "highk_power_ratio_frozen": hk0}]


class _VArgs:
    rung = "fine"
    capacity_max = 0.8
    highk_recovered = 0.8
    memorise_ratio = 1.0


def test_a_blurred_underparameterised_run_is_called_ambiguous():
    """The reading the whole step turns on.

    At the deployed rung on four tiles there are ~0.05 trainable parameters per
    target value, so the network cannot memorise the region and a squared loss
    over patches whose realisation it cannot predict is minimised by averaging.
    Blurring there is therefore NOT evidence about capacity, and the verdict has
    to say so rather than claiming section 6.1 was measured.
    """
    v = M.verdict(_hist(0.4, 0.35, 0.55), 1.0, _VArgs(), params_per_target=0.053)
    assert v["text"].startswith("AMBIGUOUS")
    assert not v["memorisable"]
    assert "all_blocks" in v["text"]          # it names the fix


def test_the_same_blur_with_room_to_memorise_is_the_incentive_result():
    v = M.verdict(_hist(0.4, 0.35, 0.55), 1.0, _VArgs(), params_per_target=4.4)
    assert v["text"].startswith("BLURRED WITH ROOM TO SPARE")
    assert v["memorisable"]


def test_a_plateau_is_only_a_capacity_statement_when_it_could_have_memorised():
    a = _VArgs()
    over = M.verdict(_hist(0.98, 0.3, 0.5), 1.0, a, params_per_target=4.4)
    under = M.verdict(_hist(0.98, 0.3, 0.5), 1.0, a, params_per_target=0.053)
    assert over["text"].startswith("CAPACITY IS THE LIMIT")
    assert under["text"].startswith("INCONCLUSIVE")


def test_recovering_high_k_refuses_to_claim_anything_about_regression():
    """Fitting one region says the class CAN express substructure -- no more.

    An earlier version read this branch as "a plain regressor recovers
    substructure, so the flow-matching design is unnecessary". That inference
    does not hold: one fixed region is not the conditional distribution, and
    whether E[HR|SR2] is empty is measured elsewhere.
    """
    v = M.verdict(_hist(0.2, 0.95, 0.5), 1.0, _VArgs(), params_per_target=4.4)
    assert v["text"].startswith("CAPACITY IS NOT THE LIMIT")
    assert v["highk_recovered"]
    assert "measure_conditional_spread" in v["text"]
    assert "overturn" not in v["text"] and "unnecessary" not in v["text"]


def test_the_structure_block_is_a_second_axis_and_is_exact_on_a_copy():
    """Why `peak_contrast` was added after the first run.

    That run found frozen SR2 at 0.87 of HR's high-k displacement power and
    slightly *above* HR's fraction of cells over delta=100, while section 6.1's
    table puts its cluster interiors at ~7% of HR's clump count. Both amplitude
    statistics are close to blind to the deficit, so a statistic that asks about
    coherence -- mass in cells beating their own smoothed neighbourhood -- is
    worth carrying alongside them.

    What is pinned is only what is known: it reads exactly 1 on an identical
    field, so a ratio away from 1 is signal rather than estimator noise, and it
    moves under a blur. Whether it is *more* sensitive than the power ratio on
    real cluster fields is not asserted here -- an earlier version of this test
    claimed it and was wrong on synthetic noise, where a structureless field has
    no coherence to lose. That comparison is a measurement, and the rerun makes
    it.
    """
    from scipy.ndimage import gaussian_filter
    from cosmo_sr.reward.soft_structure import SoftStructureConfig

    rng = np.random.default_rng(3)
    hr = _tile(rng, n=32, scale=3e-3)
    blur = hr.clone().numpy()
    for c in range(6):
        blur[0, c] = gaussian_filter(blur[0, c], sigma=1.0, mode="wrap")
    rep = M.field_report(torch.from_numpy(blur), hr, hr.clone(),
                         SoftStructureConfig(), _Args())

    st = rep["structure"]
    assert st["frozen"]["peak_contrast_s1_over_hr"] == pytest.approx(1.0, rel=1e-3)
    assert st["out"]["peak_contrast_s1_over_hr"] < 1.0
    # reported for out and frozen against the same HR reference, on every scale
    for name in ("out", "frozen"):
        for scale in (1, 2, 4):
            assert f"peak_contrast_s{scale}_over_hr" in st[name]
