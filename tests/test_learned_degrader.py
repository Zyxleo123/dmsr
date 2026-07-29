import torch

from cosmo_sr.models.learned_degrader import LearnedDegrader
from cosmo_sr.operators.multiscale import MultiScaleOperators
from cosmo_sr.operators.base_upscaler import IdentityUpscaler, consistent_base


def _deg():
    return LearnedDegrader(channels=6, width=8, depth=2, use_res_embed=True, embed_dim=32)


def test_output_shape():
    deg = _deg()
    x_2R = torch.randn(2, 6, 8, 8, 8)
    R = torch.full((2,), 64.0)
    y = deg(x_2R, R)
    assert y.shape == (2, 6, 4, 4, 4)


def test_zero_init_equals_average():
    deg = _deg()
    ops = MultiScaleOperators(2)
    x_2R = torch.randn(1, 6, 8, 8, 8)
    y = deg(x_2R, torch.tensor([64.0]))
    assert torch.allclose(y, ops.A(x_2R), atol=1e-5)


def test_constant_field_preserved_initially():
    deg = _deg()
    ops = MultiScaleOperators(2)
    x_2R = torch.full((1, 6, 8, 8, 8), 0.7)
    y = deg(x_2R, torch.tensor([64.0]))
    assert torch.allclose(y, ops.A(x_2R), atol=1e-5)
    assert torch.allclose(y, torch.full((1, 6, 4, 4, 4), 0.7), atol=1e-5)


def test_gradients_flow():
    deg = _deg()
    x_2R = torch.randn(1, 6, 8, 8, 8)
    y = deg(x_2R, torch.tensor([64.0]))
    y.pow(2).mean().backward()
    grads = [p.grad for p in deg.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)


def test_training_reduces_val_mse():
    torch.manual_seed(0)
    deg = _deg()
    ops = MultiScaleOperators(2)

    # A cleanly generalizable correction: target = A(x) + constant per-channel bias.
    # The (zero-initialised) final conv bias can represent this exactly, so val MSE
    # provably drops -- this checks the machinery is trainable, not that it beats A.
    bias = torch.tensor([0.5, -0.3, 0.2, 0.4, -0.1, 0.25]).view(1, 6, 1, 1, 1)

    def make_target(x_2R):
        return ops.A(x_2R) + bias

    x_train = torch.randn(16, 6, 8, 8, 8)
    x_val = torch.randn(8, 6, 8, 8, 8)
    y_train = make_target(x_train)
    y_val = make_target(x_val)
    R_tr = torch.full((x_train.shape[0],), 64.0)
    R_va = torch.full((x_val.shape[0],), 64.0)
    opt = torch.optim.Adam(deg.parameters(), lr=2e-3)

    with torch.no_grad():
        val0 = float(torch.mean((deg(x_val, R_va) - y_val) ** 2))
    for _ in range(200):
        loss = torch.mean((deg(x_train, R_tr) - y_train) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        val1 = float(torch.mean((deg(x_val, R_va) - y_val) ** 2))
    assert val1 < 0.9 * val0


def test_hard_consistency_independent_of_degrader():
    ops = MultiScaleOperators(2)
    B = IdentityUpscaler(2)
    y = torch.randn(1, 6, 4, 4, 4)
    base = consistent_base(B, ops, y)
    r = torch.randn(1, 6, 8, 8, 8)  # arbitrary, "bad" residual
    x = base + ops.P_null(r)
    assert torch.allclose(ops.A(x), y, atol=1e-4)


def test_frozen_degrader_no_gradients_in_latent_flow(tmp_path):
    from cosmo_sr.train import train_latent_flow

    cfg = {
        "factor": 2,
        "resolutions": [64, 128],
        "data": {"crop_hr": 32, "n_levels": 4, "full_res": 512},
        "ae": {"channels": 6, "width": 8, "ch_mults": [1, 2, 2], "latent_channels": 4},
        "degrader": {"channels": 6, "width": 8, "depth": 1},
        "model": {"latent_channels": 4, "width": 8, "depth": 1, "embed_dim": 32},
        "loss": {"n_bands": 4, "lambda_D": 0.5},
        "train": {"steps": 4, "lr": 1e-3, "seed": 0, "log_every": 2, "save_every": 0},
        "output": {"run_dir": str(tmp_path / "lf")},
    }
    res = train_latent_flow.train(cfg, smoke=True)
    deg = res["degrader"]
    assert all(p.grad is None for p in deg.parameters())
