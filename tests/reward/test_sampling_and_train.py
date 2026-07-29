"""Tiled full-box sampling, and a tiny CPU run of both training modes."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmo_sr.reward.diffusion import DiffusionConfig, denoising_loss
from cosmo_sr.reward.model import build_residual_denoiser
from cosmo_sr.reward.sampling import (TileSpec, check_margin, receptive_field_halfwidth,
                                      sample_residual_box)


def tiny(width=6, levels=2, sigma=0.02, **kw):
    return build_residual_denoiser({
        "channels": 6, "scale_factor": 4, "width": width, "num_levels": levels,
        "blocks_per_level": 1, "embed_dim": 16, "num_groups": 2,
        "sigma_res": [sigma] * 6, **kw,
    })


def test_receptive_field_is_measured_not_assumed():
    small = tiny(levels=1)
    big = tiny(levels=3)
    rf_small = receptive_field_halfwidth(small, size=32, channels=6, scale_factor=4)
    rf_big = receptive_field_halfwidth(big, size=64, channels=6, scale_factor=4)
    assert rf_small > 0
    assert rf_big >= rf_small, "a deeper U-Net cannot see less"


def test_the_receptive_field_does_not_depend_on_channel_width():
    """Why an expensive model may be measured on a cheap stand-in."""
    a = receptive_field_halfwidth(tiny(width=6, levels=1), size=64, channels=6,
                                  scale_factor=4)
    b = receptive_field_halfwidth(tiny(width=24, levels=1), size=64, channels=6,
                                  scale_factor=4)
    assert a == b


def test_the_zero_init_head_does_not_report_a_zero_receptive_field():
    """A fresh model outputs eps = 0 everywhere; the probe must measure anyway."""
    m = tiny(levels=1).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 6, 16, 16, 16), torch.full((1,), 0.5),
                y_lr=torch.randn(1, 6, 4, 4, 4),
                psi_base=torch.randn(1, 6, 16, 16, 16))
    assert float(out.abs().max()) == 0.0, "zero_init_tail should give eps = 0"
    assert receptive_field_halfwidth(m, size=64, channels=6, scale_factor=4) > 0


def test_a_too_small_margin_is_refused():
    m = tiny(levels=2)
    with pytest.raises(ValueError, match="receptive-field"):
        check_margin(m, margin=0, probe_size=32, channels=6, scale_factor=4)


def test_the_required_margin_rounds_up_and_carries_slack():
    from cosmo_sr.reward.sampling import RF_SAFETY_CELLS, tile_margin_for

    assert tile_margin_for(41, 8) == 48
    assert tile_margin_for(40, 8) == 48        # 40 + safety crosses to the next 8
    assert tile_margin_for(65, 8) == 72
    # Never smaller than the measurement itself, whatever the scale factor.
    for rf in range(1, 80):
        for sf in (1, 4, 8):
            assert tile_margin_for(rf, sf) >= rf + RF_SAFETY_CELLS


def test_tile_spec_rejects_geometry_that_would_misalign_the_lr_window():
    with pytest.raises(ValueError, match="does not divide"):
        TileSpec(32, core=12, margin=4, scale_factor=4)
    with pytest.raises(ValueError, match="margin"):
        TileSpec(32, core=16, margin=2, scale_factor=4)


def _perturbed(levels=1):
    """A model whose zero-init head has been broken, so it is a real function."""
    torch.manual_seed(0)
    m = tiny(levels=levels)
    with torch.no_grad():
        for p in m.unet.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m.eval()


def test_the_written_core_does_not_depend_on_the_tiling():
    """The whole justification for valid-core tiling: same answer, less memory.

    Compared tiling-to-tiling rather than to a whole-box pass: torch picks a
    different convolution algorithm for a 64^3 input than for a 40^3 one, and
    that float32 difference (~5e-4 here) is not a seam.
    """
    m = _perturbed(levels=1)
    ng = 64
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.05, (6, ng, ng, ng)).astype(np.float32)
    lr = base.reshape(6, ng // 4, 4, ng // 4, 4, ng // 4, 4).mean(axis=(2, 4, 6))
    lr = lr.astype(np.float32)

    rf = receptive_field_halfwidth(m, size=32, channels=6, scale_factor=4)
    margin = int(np.ceil(rf / 4.0)) * 4
    assert 0 < margin <= 16

    def run(core, extra_margin=0):
        return _eps_tiled(m, base, lr,
                          TileSpec(ng, core=core, margin=margin + extra_margin,
                                   scale_factor=4))

    small_tiles = run(16)
    big_tiles = run(32)
    wide_margin = run(16, extra_margin=8)
    assert np.abs(small_tiles - big_tiles).max() < 1e-5
    assert np.abs(small_tiles - wide_margin).max() < 1e-5


def test_a_margin_below_the_receptive_field_changes_the_answer():
    """The margin check is not ceremony: too small a margin really does leak."""
    m = _perturbed(levels=1)
    ng = 64
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.05, (6, ng, ng, ng)).astype(np.float32)
    lr = base.reshape(6, 16, 4, 16, 4, 16, 4).mean(axis=(2, 4, 6)).astype(np.float32)
    rf = receptive_field_halfwidth(m, size=32, channels=6, scale_factor=4)
    good = int(np.ceil(rf / 4.0)) * 4
    short = good - 4
    assert short > 0
    ok = _eps_tiled(m, base, lr, TileSpec(ng, core=16, margin=good, scale_factor=4))
    bad = _eps_tiled(m, base, lr, TileSpec(ng, core=16, margin=short, scale_factor=4))
    assert np.abs(ok - bad).max() > 1e-3


def _eps_tiled(model, base, lr, spec):
    """One tiled ``eps`` evaluation, written valid-core."""
    from cosmo_sr.reward.sampling import _crop_pair, _write_core

    g = torch.Generator().manual_seed(7)
    u = torch.randn(base.shape, generator=g)
    b = torch.from_numpy(base)
    y = torch.from_numpy(lr)
    t = torch.full((1,), 0.5)
    out = torch.empty_like(u)
    with torch.no_grad():
        for o in spec.origins():
            uc, bc, yc = _crop_pair(u, b, y, o, spec)
            _write_core(out, model(uc, t, y_lr=yc, psi_base=bc)[0], o, spec)
    return out.numpy()


def test_full_box_sampling_is_reproducible_and_seed_dependent():
    m = tiny(levels=1)
    ng = 16
    base = np.zeros((6, ng, ng, ng), dtype=np.float32)
    lr = np.zeros((6, ng // 4, ng // 4, ng // 4), dtype=np.float32)
    cfg = DiffusionConfig(n_steps=2)
    spec = TileSpec(ng, core=ng, margin=0, scale_factor=4)
    a = sample_residual_box(m, base, lr, seed=1, cfg=cfg, spec=spec,
                            device=torch.device("cpu"), verify_margin=False)
    b = sample_residual_box(m, base, lr, seed=1, cfg=cfg, spec=spec,
                            device=torch.device("cpu"), verify_margin=False)
    c = sample_residual_box(m, base, lr, seed=2, cfg=cfg, spec=spec,
                            device=torch.device("cpu"), verify_margin=False)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sampled_residual_is_in_physical_units():
    """Output scale is set by ``sigma_res`` alone, and ``x0_clip`` bounds it."""
    ng = 16
    cfg = DiffusionConfig(n_steps=2, x0_clip=4.0)
    spec = TileSpec(ng, core=ng, margin=0, scale_factor=4)

    def run(sigma):
        return sample_residual_box(
            tiny(levels=1, sigma=sigma),
            np.zeros((6, ng, ng, ng), np.float32),
            np.zeros((6, 4, 4, 4), np.float32), seed=0, cfg=cfg, spec=spec,
            device=torch.device("cpu"), verify_margin=False,
        )

    a, b = run(0.03), run(0.3)
    assert np.allclose(b, 10.0 * a, rtol=1e-5, atol=1e-9)
    # A zero-init head predicts eps = 0, so DDIM divides by alpha(t_max) ~ 0 and
    # only the clip keeps the residual finite: |dPsi| <= x0_clip * sigma_res.
    assert np.abs(a).max() <= 4.0 * 0.03 + 1e-6
    assert a.std() > 0.05 * 0.03


def test_per_sample_weights_scale_the_denoising_loss():
    m = tiny(levels=1)
    u0 = torch.randn(2, 6, 8, 8, 8)
    cond = {"y_lr": torch.randn(2, 6, 2, 2, 2), "psi_base": torch.randn(2, 6, 8, 8, 8),
            "redshift": 0.0}
    cfg = DiffusionConfig()
    g1 = torch.Generator().manual_seed(0)
    l1, _ = denoising_loss(m, u0, cond, cfg, generator=g1)
    g2 = torch.Generator().manual_seed(0)
    l2, _ = denoising_loss(m, u0, cond, cfg, generator=g2,
                           per_sample_weight=torch.tensor([2.0, 2.0]))
    assert float(l2.detach()) == pytest.approx(2 * float(l1.detach()), rel=1e-5)


def _smoke_cfg(tmp_path, mode):
    cfg = {
        "split": {"train_boxes": ["set0"], "val_boxes": ["set1"]},
        "data": {"root": str(tmp_path / "nonexistent"), "crop_hr": 16,
                 "smoke_crop_hr": 16, "scale_factor": 4, "redshift": 0.0},
        "model": {"channels": 6, "width": 8, "num_levels": 2, "blocks_per_level": 1,
                  "embed_dim": 16, "num_groups": 2, "sigma_res": [0.02] * 6},
        "diffusion": {"n_steps": 2},
        "train": {"steps": 4, "batch_size": 1, "num_workers": 0, "device": "cpu",
                  "val_every": 2, "log_every": 1, "val_batches": 1, "amp": False,
                  "diag_samples": 2},
        "wandb": {"mode": "disabled"},
    }
    if mode == "distill":
        cfg["distill"] = {"lambda_elite": 1.0, "lambda_ref": 0.1,
                          "lambda_elite_warmup_steps": 1}
    return cfg


def test_cpu_smoke_prior_training_runs_and_writes_a_checkpoint(tmp_path):
    from cosmo_sr.reward.train import run_training

    out = run_training(_smoke_cfg(tmp_path, "prior"), mode="prior", smoke=True,
                       run_dir=str(tmp_path / "prior"))
    assert (out / "ckpt_last.pt").is_file()
    ck = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert "ema" in ck["extra"] and "sigma_res" in ck["extra"]


def test_cpu_smoke_distillation_runs_all_three_loss_terms(tmp_path):
    from cosmo_sr.reward.train import run_training

    cfg = _smoke_cfg(tmp_path, "distill")
    out = run_training(cfg, mode="distill", smoke=True, run_dir=str(tmp_path / "d"))
    rows = (out / "metrics.csv").read_text()
    assert "loss_sup" in rows and "loss_elite" in rows
    assert not (out / "ABORTED.json").exists()


def test_abort_conditions_trip_on_an_exploding_residual():
    from cosmo_sr.reward.train import _check_abort

    bad = _check_abort({"diag_residual_rms_pred": 10.0, "diag_residual_rms_true": 1.0},
                       {}, {"residual_rms_ratio_max": 3.0})
    assert bad and "residual RMS ratio" in bad[0]
    assert _check_abort({"diag_sample_diversity": 0.001}, {},
                        {"diversity_min": 0.02})
    assert _check_abort({"diag_low_k_change": 0.5}, {}, {"low_k_change_max": 0.05})
    assert _check_abort({}, {"elite_weight_max": 50.0, "elite_weight_mean": 1.0},
                        {"elite_weight_ratio_max": 5.0})
    assert _check_abort({"diag_residual_rms_pred": 1.0, "diag_residual_rms_true": 1.0},
                        {}, {"residual_rms_ratio_max": 3.0}) == []
