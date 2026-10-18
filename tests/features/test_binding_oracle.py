"""Binding oracle: the escape-speed clip, the aperture rule, the HR-rate selection."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts" / "reward", ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bo = pytest.importorskip("binding_oracle")

M_P = 6.2e8          # Msun/h, the 100 Mpc/h 512^3 particle mass


def clump(n=128, r=0.03, sigma=0.0, bulk=(300.0, -50.0, 20.0), seed=0):
    rng = np.random.default_rng(seed)
    dx = rng.normal(scale=r, size=(n, 3))
    v = np.asarray(bulk) + rng.normal(scale=sigma, size=(n, 3))
    return dx, v


def test_cold_clump_is_untouched():
    dx, v = clump(sigma=1.0)
    w, st = bo.escape_clip(dx, v, M_P, f=0.9)
    assert st["f_unbound"] == 0.0
    assert st["ke_removed"] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(w, v)


def test_hot_clump_ends_bound_about_its_own_bulk():
    dx, v = clump(sigma=500.0)
    w, st = bo.escape_clip(dx, v, M_P, f=0.9, passes=6)
    assert st["f_unbound"] > 0.9 and st["ke_removed"] > 0.9
    vesc = np.sqrt(2.0 * np.abs(bo.self_potential(dx, M_P)))
    u = w - w.mean(axis=0)
    # within escape speed about the FINAL bulk (the clip moves the bulk a little)
    assert np.all(np.linalg.norm(u, axis=1) <= vesc * 1.0 + 1e-9)
    assert st["tu_after"] < st["tu_before"]
    # bulk motion is kept: the clump still moves with the flow it was in
    np.testing.assert_allclose(w.mean(axis=0), v.mean(axis=0), atol=60.0)


def test_clip_keeps_direction_and_only_slows():
    dx, v = clump(sigma=200.0, seed=3)
    w, _ = bo.escape_clip(dx, v, M_P, f=0.9, passes=1)
    u0, u1 = v - v.mean(axis=0), w - v.mean(axis=0)
    s0, s1 = np.linalg.norm(u0, axis=1), np.linalg.norm(u1, axis=1)
    assert np.all(s1 <= s0 + 1e-9)
    cos = np.einsum("ij,ij->i", u0, u1) / np.maximum(s0 * s1, 1e-30)
    assert np.all(cos[s1 > 0] > 1 - 1e-9)


def test_core_frame_ignores_a_streaming_background():
    # 32 cold object particles at the peak + 96 host particles streaming past at
    # 400 km/s: the aperture mean sits on neither, the core frame on the object
    rng = np.random.default_rng(5)
    obj = rng.normal(scale=0.01, size=(32, 3))
    bg = rng.normal(scale=0.08, size=(96, 3))
    dx = np.concatenate([obj, bg])
    v = np.concatenate([np.tile([100.0, 0, 0], (32, 1)) + rng.normal(scale=2.0, size=(32, 3)),
                        np.tile([500.0, 0, 0], (96, 1)) + rng.normal(scale=50.0, size=(96, 3))])
    w, st = bo.escape_clip(dx, v, M_P, f=0.9, core=16)
    np.testing.assert_allclose(w[:32], v[:32])            # the object is untouched
    assert np.all(np.abs(w[32:, 0] - 100.0) < np.abs(v[32:, 0] - 100.0) + 1e-9)
    # the aperture-mean frame drags the object: it gets clipped toward the stream
    w_mean, _ = bo.escape_clip(dx, v, M_P, f=0.9, core=0)
    assert np.max(np.abs(w_mean[:32] - v[:32])) > 50.0


def test_cold_selection_puts_nan_last():
    band = np.zeros(4, dtype=int)
    score = np.array([np.nan, 0.2, 0.1, 0.5])
    ch = bo.select_bumps(band, score, np.array([0.75, 0, 0, 0]), "cold",
                         np.random.default_rng(0))
    assert ch.tolist() == [False, True, True, True]


def test_potential_matches_two_body():
    dx = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    phi = bo.self_potential(dx, M_P, soft=0.0)
    np.testing.assert_allclose(phi, -bo.G_MPC * M_P / 0.1)


def test_aperture_is_the_overdense_extent():
    # uniform density n0 inside r=0.1 (n=1000), then n0/10 outside: the enclosed
    # mean is ~0.83 n0 at k=1024 and ~0.18 n0 at k=2048, so a 0.5 n0 threshold
    # (3 x n0/6) keeps 1024 and drops 2048
    rng = np.random.default_rng(0)
    r_in = 0.1 * rng.random(1000) ** (1 / 3)
    n0 = 1000 / (4 / 3 * np.pi * 0.1 ** 3)
    n_out = 4000
    r_out = (0.1 ** 3 + 3 * n_out / (4 * np.pi * n0 / 10) * rng.random(n_out)) ** (1 / 3)
    dist = np.sort(np.concatenate([[0.0], r_in, r_out]))
    k = bo.aperture_k(dist, n_shell=n0 / 6, ladder=[32, 64, 128, 256, 512, 1024, 2048],
                      ap_contrast=3.0)
    assert k == 1024
    assert bo.aperture_k(dist, n_shell=n0, ladder=[32, 64], ap_contrast=3.0) == 32


def test_select_hits_the_rate_per_band_and_prefers_easy():
    band = np.array([0] * 10 + [1] * 4 + [3] * 2)
    score = np.arange(band.size, dtype=float)[::-1]       # later rows easier
    rate = np.array([0.3, 0.5, np.nan, 1.0])
    ch = bo.select_bumps(band, score, rate, "easy", np.random.default_rng(0))
    assert ch[band == 0].sum() == 3 and ch[band == 1].sum() == 2 and ch[band == 3].sum() == 2
    assert np.all(ch[7:10]) and not ch[:7].any()           # the 3 easiest of band 0
    rnd = bo.select_bumps(band, score, rate, "random", np.random.default_rng(1))
    assert rnd[band == 0].sum() == 3


def test_auc():
    y = np.array([True, True, False, False])
    assert bo._auc(np.array([3.0, 4.0, 1.0, 2.0]), y) == 1.0
    assert bo._auc(np.array([1.0, 2.0, 3.0, 4.0]), y) == 0.0
    assert bo._auc(np.array([1.0, 2.0]), np.array([True, True])) is None
