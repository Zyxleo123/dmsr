"""Pin the high-k decomposition against the guard it is meant to explain.

The point of the diagnostic is to say where ``all_blocks_self``'s 1.70x lives.
A decomposition that does not reproduce the scalar it decomposes is describing a
different quantity, so the first test here is that identity; the rest pin the
two claims the fix will be argued from -- that the guard's unweighted mode mean
is dominated by the outermost shells, and that the spatial concentration measure
separates ringing from structure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "features"))

from cosmo_sr.features.field_guards import highk_power_ratio_torch  # noqa: E402
from highk_spectrum_diagnostic import (  # noqa: E402
    DX, K_SPLIT, _fft, analyse_arm, corr, guard_ratio, half_power_volume,
    highpass_map,
)
from cosmo_sr.features.cond_spread import hann_window, wavenumbers  # noqa: E402

N = 16


def _field(seed: int, t: int = 4, c: int = 6, n: int = N) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(t, c, n, n, n)).astype(np.float32)


def _write_export(d: Path, t: int = 8) -> None:
    """A two-host export whose tiles are the ones the npz carries."""
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / "tiles.npz", tiles=np.arange(t, dtype=np.int64),
        out=_field(1, t), frozen=_field(2, t), hr=_field(3, t))
    (d / "export.json").write_text(json.dumps({
        "ok": True, "box": "setX",
        "per_host": [
            {"key": "setX:h1", "halo_id": 1, "log_mvir": 14.5,
             "tiles": [0, 1, 2, 3]},
            {"key": "setX:h2", "halo_id": 2, "log_mvir": 14.2,
             "tiles": [4, 5, 6, 7]},
        ]}))


def test_guard_ratio_reproduces_the_torch_guard():
    """The identity the whole decomposition rests on."""
    a, b = _field(11)[:, 0:3], _field(12)[:, 0:3]
    win = hann_window(N).astype(np.float32)
    kmag = wavenumbers(N, DX)
    mine = guard_ratio(_fft(a, win), _fft(b, win), kmag >= K_SPLIT)
    theirs = float(highk_power_ratio_torch(
        torch.from_numpy(_field(11)), torch.from_numpy(_field(12)),
        dx=DX, k_split=K_SPLIT))
    assert mine == pytest.approx(theirs, rel=1e-5)


def test_guard_sensitivity_is_proportional_to_mode_count():
    """The claim the fix is argued from: ``sel.mean()`` weights by mode count.

    Against a flat reference, bumping a set of modes by a factor ``f`` moves the
    guard scalar by ``(f - 1) * count / total`` exactly. So the guard's opinion
    of a band is set by how many modes it holds, not by how much that band
    matters physically -- which is what makes one scalar unable to separate
    "built subhalos" from "rang the grid".
    """
    kmag = wavenumbers(N, DX)
    mask = kmag >= K_SPLIT
    total = int(mask.sum())
    ref = np.ones((1,) + kmag.shape, dtype=np.float64)

    for band in (mask & (kmag < 1.5 * K_SPLIT), kmag >= 0.85 * kmag[mask].max()):
        bumped = ref.copy()
        bumped[:, band] *= 3.0
        assert guard_ratio(bumped, ref, mask) == pytest.approx(
            1.0 + 2.0 * int(band.sum()) / total, rel=1e-9)


def test_production_tile_puts_most_modes_above_half_nyquist():
    """On the 64^3 tile the guard actually runs on, the mean is an outer-shell mean.

    ``k >= 4`` admits nearly every mode of the tile, and because a shell's count
    grows as ``k^2`` the great majority of them sit above half Nyquist. This is a
    property of the production geometry, so it is asserted at that geometry and
    not at the small ``N`` the rest of this file uses for speed.
    """
    kmag = wavenumbers(64, DX)
    mask = kmag >= K_SPLIT
    k_ny = np.pi / DX
    assert int(mask.sum()) / kmag.size > 0.98
    assert int((mask & (kmag >= 0.5 * k_ny)).sum()) / int(mask.sum()) > 0.75


def test_half_power_volume_separates_flat_noise_from_a_spike():
    flat = np.ones((1, N, N, N), dtype=np.float32)
    spike = np.full((1, N, N, N), 1e-6, dtype=np.float32)
    spike[0, N // 2, N // 2, N // 2] = 1.0
    assert half_power_volume(flat) == pytest.approx(0.5, abs=0.02)
    assert half_power_volume(spike) < 0.01


def test_highpass_map_keeps_only_the_high_modes():
    """A pure low-k mode must leave no high-pass power at all."""
    n = N
    x = np.arange(n)
    k_low = 2.0 * np.pi / (n * DX)          # the fundamental, far below K_SPLIT
    f = np.zeros((1, 3, n, n, n), dtype=np.float32)
    f[0, 0] = np.sin(k_low * x * DX)[:, None, None]
    m = highpass_map(f, wavenumbers(n, DX), K_SPLIT)
    assert float(m.max()) < 1e-8 * float((f ** 2).sum(axis=1).max())


def test_corr_is_a_pearson_coefficient():
    a = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    assert corr(a, a) == pytest.approx(1.0)
    assert corr(a, -a) == pytest.approx(-1.0)


def test_analyse_arm_reproduces_its_own_guard_scalar(tmp_path):
    """End to end: the per-host row must equal the guard on the same tiles."""
    d = tmp_path / "all_blocks_x" / "holdout_setX"
    _write_export(d)
    out = analyse_arm(d, n_bins=6, k_sweep=[K_SPLIT, 8.0])

    assert out["n_hosts"] == 2
    z = np.load(d / "tiles.npz")
    for h in out["hosts"]:
        idx = [0, 1, 2, 3] if h["halo_id"] == 1 else [4, 5, 6, 7]
        want = float(highk_power_ratio_torch(
            torch.from_numpy(z["out"][idx]), torch.from_numpy(z["hr"][idx]),
            dx=DX, k_split=K_SPLIT))
        assert h["guard_dis"] == pytest.approx(want, rel=1e-5)
        # The k-sweep must agree with the guard at the guard's own split.
        assert h["ksweep_dis"][f"{K_SPLIT:g}"] == pytest.approx(want, rel=1e-5)
        # Shares are a partition of the scalar, so they sum to one.
        assert sum(h["share_cand_dis"]) == pytest.approx(1.0, rel=1e-6)
        assert sum(h["share_hr_dis"]) == pytest.approx(1.0, rel=1e-6)


def test_analyse_arm_skips_a_host_whose_tiles_are_not_exported(tmp_path):
    d = tmp_path / "all_blocks_y" / "holdout_setX"
    _write_export(d, t=4)
    e = json.loads((d / "export.json").read_text())
    e["per_host"][1]["tiles"] = [4, 5, 6, 7]        # not in the npz
    (d / "export.json").write_text(json.dumps(e))
    assert analyse_arm(d, n_bins=6, k_sweep=[K_SPLIT])["n_hosts"] == 1
