"""The host-centric temperature term charges what it says, and only through velocity.

The term exists because the density field is already right (placement
forensics, 2026-10-15); a gradient leaking into positions would let it move the
one thing the oracle showed must stay put. Pinned here along with the
statistic's scale behaviour, the bin mask and the neighbour search.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from cosmo_sr.features.host_temperature import (
    TempConfig, local_log_sigma, neighbour_rows, temperature_loss,
    temperature_profile, target_profile,
)


def _host(n=40000, seed=0, sigma=300.0, centre=(50.0, 50.0, 50.0)):
    g = np.random.default_rng(seed)
    r = np.cbrt(g.uniform(0, 1, n)) * 3.3
    u = g.normal(size=(n, 3))
    d = u / np.linalg.norm(u, axis=1, keepdims=True) * r[:, None]
    # a dense core so the shrinking sphere has something to find
    d = np.vstack([d, g.normal(0, 0.05, (4000, 3))])
    v = g.normal(0, sigma, (d.shape[0], 3))
    pos = torch.tensor(d + np.asarray(centre), dtype=torch.float64)
    return pos, torch.tensor(v, dtype=torch.float64)


def test_neighbour_rows_exclude_self_and_are_distinct():
    pos, _ = _host(n=3000)
    q = torch.arange(0, 200)
    nb = neighbour_rows(pos - pos.mean(0), q, k=16)
    assert nb.shape == (200, 16)
    assert not bool((nb == q[:, None]).any())
    assert all(len(set(row.tolist())) == 16 for row in nb)


def test_identical_fields_cost_nothing_and_doubled_velocities_cost_log2_squared():
    pos, vel = _host()
    cfg = TempConfig(k=16, queries_per_bin=300, min_queries=20)
    guess = torch.tensor([50.0, 50.0, 50.0], dtype=torch.float64)
    tgt = target_profile(pos, vel, guess, 1.0, cfg, queries_per_bin=300, seed=1)
    # same particles, same seed: the same queries, so exactly zero
    prof = temperature_profile(pos, vel, guess, 1.0, cfg,
                               generator=torch.Generator().manual_seed(1),
                               queries_per_bin=300)
    loss, _ = temperature_loss(prof, tgt["mean"], tgt["n"], cfg)
    assert float(loss) < 1e-12
    prof2 = temperature_profile(pos, 2.0 * vel, guess, 1.0, cfg,
                                generator=torch.Generator().manual_seed(1),
                                queries_per_bin=300)
    loss2, info = temperature_loss(prof2, tgt["mean"], tgt["n"], cfg)
    assert float(loss2) == pytest.approx(math.log(2.0) ** 2, rel=1e-3)
    assert info["temp_out_ratio"] == pytest.approx(2.0, rel=1e-3)


def test_gradient_reaches_velocity_and_never_position():
    pos, vel = _host(n=20000)
    pos = pos.clone().requires_grad_(True)
    vel = vel.clone().requires_grad_(True)
    cfg = TempConfig(k=16, queries_per_bin=200, min_queries=20)
    guess = torch.tensor([50.0, 50.0, 50.0], dtype=torch.float64)
    tgt = target_profile(pos.detach(), 0.5 * vel.detach(), guess, 1.0, cfg)
    prof = temperature_profile(pos, vel, guess, 1.0, cfg,
                               generator=torch.Generator().manual_seed(0))
    loss, _ = temperature_loss(prof, tgt["mean"], tgt["n"], cfg)
    loss.backward()
    assert vel.grad is not None and float(vel.grad.abs().sum()) > 0
    assert pos.grad is None or float(pos.grad.abs().sum()) == 0.0


def test_core_only_arm_charges_only_the_core_bins():
    pos, vel = _host()
    guess = torch.tensor([50.0, 50.0, 50.0], dtype=torch.float64)
    full = TempConfig(k=16, queries_per_bin=300, min_queries=20)
    core = TempConfig(k=16, queries_per_bin=300, min_queries=20, max_x=0.5)
    tgt = target_profile(pos, vel, guess, 1.0, full, queries_per_bin=300, seed=2)
    v2 = vel.clone()
    r = torch.linalg.vector_norm(pos - guess, dim=1)
    v2[r >= 0.75] *= 3.0                    # only the outskirts are wrong
    for cfg, expect_zero in ((core, True), (full, False)):
        prof = temperature_profile(pos, v2, guess, 1.0, cfg,
                                   generator=torch.Generator().manual_seed(2),
                                   queries_per_bin=300)
        loss, info = temperature_loss(prof, tgt["mean"], tgt["n"], cfg)
        assert (float(loss) < 1e-3) == expect_zero, (cfg.max_x, float(loss))
        if not expect_zero:
            assert info["temp_n_bins"] > 6


def test_local_log_sigma_is_the_neighbourhood_rms():
    vel = torch.tensor([[0.0, 0, 0], [3.0, 0, 0], [0.0, 4, 0], [10.0, 10, 10]])
    nbr = torch.tensor([[0, 1, 2]])
    vn = vel[nbr[0]] - vel[nbr[0]].mean(0)
    want = 0.5 * math.log(float((vn * vn).sum(1).mean()) + 1.0)
    assert float(local_log_sigma(vel, nbr)[0]) == pytest.approx(want)
