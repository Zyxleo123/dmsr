"""The `x0_clip` fraction, and the claim it exists to falsify.

An earlier revision of the design document said a zero-initialised output head
makes a freshly built model reproduce the frozen SR2 field. It does not: with
``eps_hat = 0`` the DDIM update returns ``u_0 = u_t / alpha(t)``, i.e. the
initial noise rescaled, which at ``t_max`` is enormous. Only
``residual_scale = 0`` gives bit-exact SR2.

These tests pin both halves of that: the clip is genuinely load-bearing for a
zero-init model, and its per-step fraction is recorded in standardized residual
units so a run can be judged on how much of the residual amplitude was the model
and how much was the bound.
"""
from __future__ import annotations

import numpy as np
import torch

from cosmo_sr.reward.diffusion import DiffusionConfig, ddim_sample
from cosmo_sr.reward.model import build_residual_denoiser
from cosmo_sr.reward.sampling import TileSpec, sample_residual_box


def tiny(sigma=0.02):
    return build_residual_denoiser({
        "channels": 6, "scale_factor": 4, "width": 6, "num_levels": 1,
        "blocks_per_level": 1, "embed_dim": 16, "num_groups": 2,
        "sigma_res": [sigma] * 6,
    })


def _cond(n=8, batch=1):
    return {
        "y_lr": torch.zeros(batch, 6, n // 4, n // 4, n // 4),
        "psi_base": torch.zeros(batch, 6, n, n, n),
        "redshift": 0.0,
    }


def test_a_zero_init_model_does_not_sample_a_zero_residual():
    """The correction itself: zero-init head != frozen SR2 output."""
    model = tiny().eval()
    cfg = DiffusionConfig(n_steps=4, x0_clip=4.0)
    g = torch.Generator().manual_seed(0)
    u = ddim_sample(model, (1, 6, 8, 8, 8), _cond(8), cfg,
                    device="cpu", generator=g)
    assert float(u.abs().max()) > 1e-3, (
        "a zero-initialised head produced an all-zero sample; if this ever "
        "passes, the sampler is not stochastic and the doc's correction is moot"
    )


def test_the_clip_log_records_one_entry_per_step_in_sigma_units():
    model = tiny().eval()
    cfg = DiffusionConfig(n_steps=5, x0_clip=4.0)
    log: list = []
    ddim_sample(model, (1, 6, 8, 8, 8), _cond(8), cfg, device="cpu",
                generator=torch.Generator().manual_seed(0), clip_log=log)
    assert len(log) == cfg.n_steps
    assert [r["step"] for r in log] == list(range(cfg.n_steps))
    for r in log:
        assert 0.0 <= r["clip_fraction"] <= 1.0
        assert len(r["clip_fraction_per_channel"]) == 6
        assert r["x0_clip"] == cfg.x0_clip
        assert np.isfinite(r["abs_max_sigma"]) and np.isfinite(r["rms_sigma"])
    # t descends from t_max toward t_min.
    assert all(a["t"] > b["t"] for a, b in zip(log, log[1:]))


def test_the_clip_is_load_bearing_for_a_zero_init_model():
    """The first step clips heavily; that is the blow-up being contained.

    ``u_0 = u_t / alpha(t_max)`` with ``alpha -> 0``, so essentially every voxel
    exceeds 4 sigma at step 0. A run where this were *not* true would mean the
    schedule had changed.
    """
    model = tiny().eval()
    cfg = DiffusionConfig(n_steps=6, x0_clip=4.0)
    log: list = []
    ddim_sample(model, (1, 6, 8, 8, 8), _cond(8), cfg, device="cpu",
                generator=torch.Generator().manual_seed(0), clip_log=log)
    assert log[0]["clip_fraction"] > 0.5
    assert log[0]["abs_max_sigma"] > cfg.x0_clip


def test_disabling_the_clip_disables_the_log():
    model = tiny().eval()
    log: list = []
    ddim_sample(model, (1, 6, 8, 8, 8), _cond(8),
                DiffusionConfig(n_steps=3, x0_clip=0.0), device="cpu",
                generator=torch.Generator().manual_seed(0), clip_log=log)
    assert log == [], "no clip means there is no clip fraction to report"


def test_the_full_box_sampler_logs_the_same_way():
    """The tiled path is a separate implementation and must not silently differ."""
    model = tiny().eval()
    n = 16
    base = np.zeros((6, n, n, n), dtype=np.float32)
    lr = np.zeros((6, n // 4, n // 4, n // 4), dtype=np.float32)
    log: list = []
    cfg = DiffusionConfig(n_steps=3, x0_clip=4.0)
    sample_residual_box(model, base, lr, seed=0, cfg=cfg,
                        spec=TileSpec(n, core=8, margin=4, scale_factor=4),
                        device="cpu", verify_margin=False, clip_log=log)
    assert len(log) == cfg.n_steps
    assert all(0.0 <= r["clip_fraction"] <= 1.0 for r in log)
    assert all(len(r["clip_fraction_per_channel"]) == 6 for r in log)
