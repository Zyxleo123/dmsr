import pytest
import torch

from cosmo_sr.operators.degrader import FixedDegrader
from cosmo_sr.models.wrappers import build_generator
from cosmo_sr.losses import compute_losses
from cosmo_sr.losses.ambient import ambient_loss, compute_ambient


SCALE = 2


def _gen():
    return build_generator("SimpleSRGenerator", in_channels=6, out_channels=6,
                           scale_factor=SCALE, width=8, depth=1)


def test_ambient_zero_when_consistent():
    A = FixedDegrader(SCALE)
    x_hr = torch.randn(1, 6, 8, 8, 8)
    y_lr = A(x_hr)
    y_recon = A(x_hr)
    loss = ambient_loss(y_recon, y_lr)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_ambient_positive_for_mismatch():
    A = FixedDegrader(SCALE)
    y_lr = torch.randn(1, 6, 4, 4, 4)
    x_hr = torch.randn(1, 6, 8, 8, 8)
    loss = ambient_loss(A(x_hr), y_lr)
    assert loss.item() > 0


def test_ambient_gradients_to_generator():
    A = FixedDegrader(SCALE)
    G = _gen()
    y_lr = torch.randn(1, 6, 4, 4, 4)
    loss, _, _ = compute_ambient(G, A, y_lr)
    loss.backward()
    grads = [p.grad for p in G.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)


def test_compute_losses_dict_keys_ambient_only():
    A = FixedDegrader(SCALE)
    G = _gen()
    y_lr = torch.randn(1, 6, 4, 4, 4)
    out = compute_losses(G, A, y_lr_unpaired=y_lr, lambda_ambient=1.0, lambda_pair=0.0)
    assert "loss" in out and "loss_ambient" in out
    assert "loss_pair" not in out


def test_compute_losses_dict_keys_mixed_with_reg():
    A = FixedDegrader(SCALE)
    G = _gen()
    y_lr = torch.randn(1, 6, 4, 4, 4)
    y_lr_p = torch.randn(1, 6, 4, 4, 4)
    x_hr_p = torch.randn(1, 6, 8, 8, 8)
    out = compute_losses(
        G, A,
        y_lr_unpaired=y_lr, y_lr_paired=y_lr_p, x_hr_paired=x_hr_p,
        lambda_ambient=1.0, lambda_pair=1.0, lambda_reg=1.0,
        reg_cfg={"lambda_tv": 0.1},
    )
    for key in ("loss", "loss_ambient", "loss_pair", "loss_reg"):
        assert key in out
    assert torch.isfinite(out["loss"]).all()


def test_nonfinite_raises():
    A = FixedDegrader(SCALE)
    G = _gen()
    # force a NaN input to produce a non-finite loss
    y_lr = torch.full((1, 6, 4, 4, 4), float("nan"))
    with pytest.raises(ValueError):
        compute_losses(G, A, y_lr_unpaired=y_lr, lambda_ambient=1.0, lambda_pair=0.0)


def test_lambda_ambient_zero_needs_no_unpaired():
    A = FixedDegrader(SCALE)
    G = _gen()
    y_lr_p = torch.randn(1, 6, 4, 4, 4)
    x_hr_p = torch.randn(1, 6, 8, 8, 8)
    out = compute_losses(G, A, y_lr_paired=y_lr_p, x_hr_paired=x_hr_p,
                         lambda_ambient=0.0, lambda_pair=1.0)
    assert "loss_pair" in out and "loss_ambient" not in out
