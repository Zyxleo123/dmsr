import torch

from cosmo_sr.models.residual_autoencoder import ResidualAutoencoder
from cosmo_sr.operators.multiscale import MultiScaleOperators
from cosmo_sr.operators.base_upscaler import IdentityUpscaler, consistent_base
from cosmo_sr.train.common import save_checkpoint, load_checkpoint


def _ae():
    return ResidualAutoencoder(channels=6, width=8, ch_mults=(1, 2, 2),
                               latent_channels=4, n_res=1)


def test_encode_decode_shapes_two_sizes():
    ae = _ae()
    for n in (8, 16):
        x = torch.randn(2, 6, n, n, n)
        z = ae.encode(x)
        assert z.shape == (2, 4, n // 4, n // 4, n // 4)
        rec = ae.decode(z)
        assert rec.shape == x.shape


def test_decoded_residual_null_space():
    ae = _ae()
    ops = MultiScaleOperators(2)
    z = torch.randn(1, 4, 4, 4, 4)
    r = ops.P_null(ae.decode(z))
    assert ops.A(r).abs().max() < 1e-4


def test_reconstructed_field_hard_consistency():
    ae = _ae()
    ops = MultiScaleOperators(2)
    B = IdentityUpscaler(2)
    x_R = torch.randn(1, 6, 8, 8, 8)
    x_2R = torch.randn(1, 6, 16, 16, 16)
    base = consistent_base(B, ops, x_R)
    r_star = ops.P_null(x_2R - base)
    z = ae.encode(r_star)
    x_recon = base + ops.P_null(ae.decode(z))
    assert torch.allclose(ops.A(x_recon), x_R, atol=1e-4)


def test_smoke_training_decreases_recon_loss():
    torch.manual_seed(0)
    ae = _ae()
    ops = MultiScaleOperators(2)
    r_star = ops.P_null(torch.randn(2, 6, 16, 16, 16))
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    losses = []
    for _ in range(60):
        r_recon = ops.P_null(ae.decode(ae.encode(r_star)))
        loss = torch.mean((r_recon - r_star) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.9 * losses[0]


def test_save_load_checkpoint_identical(tmp_path):
    ae = _ae()
    ae.eval()
    x = torch.randn(1, 6, 16, 16, 16)
    with torch.no_grad():
        out_a = ae(x)
    save_checkpoint(tmp_path / "ae.pt", ae)
    ae2 = _ae()
    load_checkpoint(tmp_path / "ae.pt", ae2, map_location="cpu")
    ae2.eval()
    with torch.no_grad():
        out_b = ae2(x)
    assert torch.equal(out_a, out_b)


def test_frozen_ae_no_gradients_in_latent_flow(tmp_path):
    from cosmo_sr.train import train_latent_flow

    cfg = {
        "factor": 2,
        "resolutions": [64, 128],
        "data": {"crop_hr": 32, "n_levels": 4, "full_res": 512},
        "ae": {"channels": 6, "width": 8, "ch_mults": [1, 2, 2], "latent_channels": 4},
        "degrader": {"channels": 6, "width": 8, "depth": 1},
        "model": {"latent_channels": 4, "width": 8, "depth": 1, "embed_dim": 32},
        "loss": {"n_bands": 4, "p_uncond": 0.2},
        "train": {"steps": 4, "lr": 1e-3, "seed": 0, "log_every": 2, "save_every": 0},
        "output": {"run_dir": str(tmp_path / "lf")},
    }
    res = train_latent_flow.train(cfg, smoke=True)
    ae = res["ae"]
    assert all(p.grad is None for p in ae.parameters())
