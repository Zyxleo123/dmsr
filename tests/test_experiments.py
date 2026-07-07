"""Phase 11 sanity experiments as fast tests.

Experiment A: degrader sanity (A(x_hr) reproduces stored LR to ~machine eps).
Experiment C: ambient-only one-sample overfit (>=80% reduction, output finite).
Experiment B (supervised overfit) is covered by tests/test_model.py.
"""
import numpy as np
import torch

from cosmo_sr.operators.degrader import FixedDegrader
from cosmo_sr.models.wrappers import build_generator
from cosmo_sr.losses.ambient import compute_ambient
from cosmo_sr.eval.metrics import relative_mse


def test_experiment_a_degrader_sanity():
    scale = 8
    A = FixedDegrader(scale)
    rng = np.random.default_rng(0)
    x_hr = torch.from_numpy(rng.standard_normal((1, 6, 32, 32, 32)).astype(np.float32))
    y_lr = A(x_hr)
    y_recon = A(x_hr)
    rel = relative_mse(y_recon.numpy(), y_lr.numpy())
    assert rel < 1e-7, f"degrader not reproducible: rel MSE={rel}"


def test_experiment_c_ambient_only_overfit():
    torch.manual_seed(0)
    scale = 4
    A = FixedDegrader(scale)
    G = build_generator("SimpleSRGenerator", scale_factor=scale, width=16, depth=2)
    # one LR crop; a valid HR exists (LR = A(HR)) so the ambient objective is
    # satisfiable
    coarse = torch.randn(1, 6, 4, 4, 4)
    x_hr = torch.nn.functional.interpolate(
        coarse, scale_factor=scale, mode="trilinear", align_corners=False
    )
    y_lr = A(x_hr)

    opt = torch.optim.Adam(G.parameters(), lr=2e-3)
    with torch.no_grad():
        init, _, _ = compute_ambient(G, A, y_lr)
        init = init.item()
    final = init
    for _ in range(300):
        opt.zero_grad()
        loss, x_hat, _ = compute_ambient(G, A, y_lr)
        loss.backward()
        opt.step()
        final = loss.item()

    assert final < 0.2 * init, f"ambient MSE did not drop >=80%: init={init}, final={final}"
    with torch.no_grad():
        assert torch.isfinite(G(y_lr)).all()
