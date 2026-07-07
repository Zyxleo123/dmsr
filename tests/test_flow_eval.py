import torch

from cosmo_sr.models.residual_flow import ResidualFlowModel
from cosmo_sr.operators.multiscale import MultiScaleOperators
from cosmo_sr.operators.base_upscaler import IdentityUpscaler
from cosmo_sr.eval.flow_eval import (
    consistency_error,
    highk_power_ratio,
    z_diversity,
    evaluate_cascade,
    sr2_power_summary,
)
from cosmo_sr.eval.sr2_stats import (
    velocity_statistics,
    two_point_correlation,
    equilateral_bispectrum,
)


def _setup():
    model = ResidualFlowModel(channels=6, width=8, depth=2, embed_dim=32)
    ops = MultiScaleOperators(2)
    B = IdentityUpscaler(2)
    return model, ops, B


def test_consistency_error_zero_for_consistent_field():
    ops = MultiScaleOperators(2)
    y = torch.randn(1, 6, 4, 4, 4)
    x = ops.U(y)  # A(U(y)) == y
    err = consistency_error(ops, x, y)
    assert err["consistency_mse"] < 1e-6


def test_highk_power_ratio_runs():
    a = torch.randn(6, 8, 8, 8)
    b = torch.randn(6, 8, 8, 8)
    out = highk_power_ratio(a, b)
    assert "highk_power_ratio" in out


def test_z_diversity_positive():
    model, ops, B = _setup()
    y = torch.randn(1, 6, 4, 4, 4)
    out = z_diversity(model, ops, B, y, 64.0, n_samples=3, n_steps=2)
    assert out["z_voxel_std_mean"] >= 0.0


def test_evaluate_cascade_runs():
    model, ops, B = _setup()
    # ground-truth pyramid: build from a random finest field
    x512 = torch.randn(1, 6, 16, 16, 16)
    x256 = ops.A(x512)
    x128 = ops.A(x256)
    x64 = ops.A(x128)
    pyramid = {512: x512, 256: x256, 128: x128, 64: x64}
    out = evaluate_cascade(model, ops, B, pyramid, resolutions=(64, 128), n_steps=2,
                           diversity_samples=2)
    assert 128 in out["octaves"] and 256 in out["octaves"]
    assert "consistency_mse" in out["octaves"][128]


def test_sr2_stats_run():
    field = torch.randn(6, 8, 8, 8)
    vs = velocity_statistics(field)
    assert "speed_rms" in vs
    r, xi = two_point_correlation(field[0])
    assert r.shape == xi.shape
    k, bk = equilateral_bispectrum(field[0].numpy(), n_bins=4)
    assert k.shape == bk.shape


def test_sr2_power_summary_runs():
    a = torch.randn(6, 8, 8, 8)
    b = torch.randn(6, 8, 8, 8)
    out = sr2_power_summary(a, b)
    assert "power_ratio" in out and "cross_corr_mean_per_channel" in out
