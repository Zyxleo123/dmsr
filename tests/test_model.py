import pytest
import torch

from cosmo_sr.models.wrappers import build_generator
from cosmo_sr.models.unet_baseline import SimpleSRGenerator


def test_output_shape_scale8():
    G = SimpleSRGenerator(scale_factor=8, width=8, depth=1)
    y = torch.randn(2, 6, 8, 8, 8)
    x = G(y)
    assert x.shape == (2, 6, 64, 64, 64)


def test_dtype_device_match():
    G = SimpleSRGenerator(scale_factor=4, width=8, depth=1)
    y = torch.randn(1, 6, 4, 4, 4)
    x = G(y)
    assert x.dtype == y.dtype
    assert x.device == y.device


def test_backprop():
    G = SimpleSRGenerator(scale_factor=4, width=8, depth=1)
    y = torch.randn(1, 6, 4, 4, 4)
    G(y).pow(2).mean().backward()
    grads = [p.grad for p in G.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_overfit_one_pair():
    torch.manual_seed(0)
    # smaller volume (LR 2 -> HR 16) than the 8->64 spec to keep CPU test fast,
    # but exercises the same scale_factor=8 path and overfit criterion.
    G = build_generator("SimpleSRGenerator", scale_factor=8, width=16, depth=2)
    y = torch.randn(1, 6, 2, 2, 2)
    coarse = torch.randn(1, 6, 2, 2, 2)
    target = torch.nn.functional.interpolate(
        coarse, scale_factor=8, mode="trilinear", align_corners=False
    )
    opt = torch.optim.Adam(G.parameters(), lr=2e-3)
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        init = loss_fn(G(y), target).item()
    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn(G(y), target)
        loss.backward()
        opt.step()
    final = loss.item()
    assert final < 0.2 * init, f"MSE did not drop >=80%: init={init}, final={final}"
