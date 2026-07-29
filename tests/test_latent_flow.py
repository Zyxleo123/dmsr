import torch

from cosmo_sr.models.residual_autoencoder import ResidualAutoencoder
from cosmo_sr.models.latent_flow import LatentFlowModel
from cosmo_sr.operators.multiscale import MultiScaleOperators
from cosmo_sr.operators.base_upscaler import IdentityUpscaler
from cosmo_sr.inference.latent_flow_sample import (
    cfg_velocity,
    latent_shape_from_cond,
    sample_latent_step,
    super_resolve_latent_cascade,
)


def _ae():
    return ResidualAutoencoder(channels=6, width=8, ch_mults=(1, 2, 2),
                               latent_channels=4, n_res=1)


def _flow():
    return LatentFlowModel(latent_channels=4, cond_channels=6, width=8, depth=2, embed_dim=32)


def test_forward_conditional_and_null():
    ae, flow = _ae(), _flow()
    ops = MultiScaleOperators(2)
    x_R = torch.randn(1, 6, 8, 8, 8)
    ls = latent_shape_from_cond(ae, ops, x_R)
    z = torch.randn(*ls)
    t = torch.rand(1)
    R = torch.tensor([64.0])
    v_cond = flow(z, t, x_R, R)
    v_null = flow(z, t, torch.zeros_like(x_R), R)
    assert v_cond.shape == z.shape and v_null.shape == z.shape
    assert torch.isfinite(v_cond).all() and torch.isfinite(v_null).all()


def test_condition_dropout_produces_both():
    torch.manual_seed(0)
    p_uncond = 0.5
    dropped_any = False
    kept_any = False
    for _ in range(50):
        drop = (torch.rand(4) < p_uncond)
        if drop.any():
            dropped_any = True
        if (~drop).any():
            kept_any = True
    assert dropped_any and kept_any


def test_loss_terms_finite():
    from cosmo_sr.train import train_latent_flow

    cfg = {
        "factor": 2,
        "resolutions": [64, 128],
        "data": {"crop_hr": 32, "n_levels": 4, "full_res": 512},
        "ae": {"channels": 6, "width": 8, "ch_mults": [1, 2, 2], "latent_channels": 4},
        "degrader": {"channels": 6, "width": 8, "depth": 1},
        "model": {"latent_channels": 4, "width": 8, "depth": 1, "embed_dim": 32},
        "loss": {"n_bands": 4, "lambda_clean": 0.1, "lambda_D": 0.1,
                 "lambda_x": 0.1, "lambda_band": 0.1},
        "train": {"steps": 6, "lr": 1e-3, "seed": 0, "log_every": 1, "save_every": 0},
        "output": {},
    }
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cfg["output"] = {"run_dir": td}
        res = train_latent_flow.train(cfg, smoke=True)
    last = res["last"]
    for k in ("flow/loss_fm", "flow/loss_clean_z", "flow/loss_D",
              "flow/loss_x", "flow/loss_band"):
        assert k in last
        import math
        assert math.isfinite(last[k])


def test_cfg_formula():
    ae, flow = _ae(), _flow()
    ops = MultiScaleOperators(2)
    x_R = torch.randn(1, 6, 8, 8, 8)
    ls = latent_shape_from_cond(ae, ops, x_R)
    z = torch.randn(*ls)
    t = torch.rand(1)
    R = torch.tensor([64.0])
    null = torch.zeros_like(x_R)
    with torch.no_grad():
        v0 = cfg_velocity(flow, z, t, x_R, null, R, 0.0)
        v1 = cfg_velocity(flow, z, t, x_R, null, R, 1.0)
        assert torch.allclose(v0, flow(z, t, null, R), atol=1e-6)
        assert torch.allclose(v1, flow(z, t, x_R, R), atol=1e-6)


def test_sampling_shape_and_consistency():
    ae, flow = _ae(), _flow()
    ops = MultiScaleOperators(2)
    B = IdentityUpscaler(2)
    x_R = torch.randn(1, 6, 8, 8, 8)
    with torch.no_grad():
        x_2R = sample_latent_step(flow, ae, ops, B, x_R, 64.0, n_steps=3, cfg_scale=1.0)
    assert x_2R.shape == (1, 6, 16, 16, 16)
    assert torch.allclose(ops.A(x_2R), x_R, atol=1e-4)


def test_cascade_keys_and_consistency():
    ae, flow = _ae(), _flow()
    ops = MultiScaleOperators(2)
    B = IdentityUpscaler(2)
    y64 = torch.randn(1, 6, 4, 4, 4)
    out = super_resolve_latent_cascade(flow, ae, ops, B, y64,
                                       resolutions=(64, 128, 256), n_steps=2, cfg_scale=1.0)
    assert set(out.keys()) == {128, 256, 512}
    assert torch.allclose(ops.A(out[128]), y64, atol=1e-4)
    assert torch.allclose(ops.A(out[256]), out[128], atol=1e-4)
    assert torch.allclose(ops.A(out[512]), out[256], atol=1e-4)
