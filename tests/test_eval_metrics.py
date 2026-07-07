import json

import numpy as np
import torch

from cosmo_sr.eval.spectra import power_spectrum, cross_correlation_coefficient
from cosmo_sr.eval.metrics import voxel_mse, lr_reconstruction_mse, compute_metrics
from cosmo_sr.eval.run_eval import evaluate
from cosmo_sr.operators.degrader import FixedDegrader
from cosmo_sr.models.wrappers import build_generator, NearestUpsampler


def test_power_spectrum_finite_random():
    rng = np.random.default_rng(0)
    field = rng.standard_normal((16, 16, 16))
    k, pk = power_spectrum(field)
    assert k.size > 0 and pk.size == k.size
    assert np.all(np.isfinite(k)) and np.all(np.isfinite(pk))
    assert np.all(pk >= 0)


def test_constant_field_power_at_dc():
    field = np.full((8, 8, 8), 2.0)
    # default excludes DC -> non-DC power ~ 0
    k, pk = power_spectrum(field)
    if pk.size:
        assert np.allclose(pk, 0.0, atol=1e-9)
    # with DC included, the k=0 bin carries the power
    k2, pk2 = power_spectrum(field, include_dc=True)
    assert np.all(np.isfinite(pk2))
    assert pk2[0] > 0


def test_cross_correlation_self_is_one():
    rng = np.random.default_rng(1)
    field = rng.standard_normal((16, 16, 16))
    _, r = cross_correlation_coefficient(field, field)
    assert np.allclose(r, 1.0, atol=1e-6)


def test_lr_reconstruction_mse_zero_when_consistent():
    A = FixedDegrader(4)
    x_hr = torch.randn(6, 16, 16, 16)
    y_lr = A(x_hr.unsqueeze(0)).squeeze(0)
    assert lr_reconstruction_mse(A, x_hr, y_lr) < 1e-10


def test_compute_metrics_with_and_without_hr():
    A = FixedDegrader(4)
    x_hat = np.random.default_rng(0).standard_normal((6, 16, 16, 16)).astype(np.float32)
    y_lr = A(torch.from_numpy(x_hat).unsqueeze(0)).squeeze(0).numpy()

    m_no_hr = compute_metrics(A, x_hat, y_lr, x_hr=None)
    assert "lr_recon_mse" in m_no_hr and "hr_mse" not in m_no_hr

    x_hr = np.random.default_rng(1).standard_normal((6, 16, 16, 16)).astype(np.float32)
    m_hr = compute_metrics(A, x_hat, y_lr, x_hr=x_hr)
    assert "hr_mse" in m_hr and "hr_relative_mse" in m_hr


def test_evaluate_writes_outputs(tmp_path):
    scale = 4
    model = NearestUpsampler(scale_factor=scale)
    A = FixedDegrader(scale)
    rng = np.random.default_rng(2)
    lr = rng.standard_normal((6, 8, 8, 8)).astype(np.float32)
    out = evaluate(model, A, lr, str(tmp_path / "ev"), scale, hr_field=None, nsplit=1)
    assert (tmp_path / "ev" / "metrics.json").exists()
    assert (tmp_path / "ev" / "spectra.npz").exists()
    assert (tmp_path / "ev" / "slices.png").exists()
    metrics = json.loads((tmp_path / "ev" / "metrics.json").read_text())
    assert "lr_recon_mse" in metrics
    assert "hr_mse" not in metrics


def test_evaluate_with_hr(tmp_path):
    scale = 4
    model = NearestUpsampler(scale_factor=scale)
    A = FixedDegrader(scale)
    rng = np.random.default_rng(3)
    lr = rng.standard_normal((6, 8, 8, 8)).astype(np.float32)
    hr = rng.standard_normal((6, 32, 32, 32)).astype(np.float32)
    out = evaluate(model, A, lr, str(tmp_path / "ev2"), scale, hr_field=hr, nsplit=2)
    metrics = json.loads((tmp_path / "ev2" / "metrics.json").read_text())
    assert "hr_mse" in metrics
