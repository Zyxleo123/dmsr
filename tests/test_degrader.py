import pytest
import torch

from cosmo_sr.operators.degrader import FixedDegrader


def test_output_shape():
    A = FixedDegrader(scale_factor=8)
    x = torch.randn(2, 6, 64, 64, 64)
    y = A(x)
    assert y.shape == (2, 6, 8, 8, 8)


def test_constant_field_preserved():
    A = FixedDegrader(scale_factor=4)
    x = torch.full((1, 6, 16, 16, 16), 3.5)
    y = A(x)
    assert torch.allclose(y, torch.full_like(y, 3.5))


def test_matches_manual_block_average():
    A = FixedDegrader(scale_factor=2)
    x = torch.arange(2 * 2 * 2, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
    y = A(x)
    assert y.shape == (1, 1, 1, 1, 1)
    assert y.item() == pytest.approx(x.mean().item())


def test_nondivisible_raises():
    A = FixedDegrader(scale_factor=8)
    with pytest.raises(ValueError):
        A(torch.randn(1, 6, 63, 63, 63))


def test_deterministic():
    A = FixedDegrader(scale_factor=4)
    x = torch.randn(1, 6, 16, 16, 16)
    y1 = A(x)
    y2 = A(x)
    assert torch.equal(y1, y2)


def test_invalid_mode():
    with pytest.raises(ValueError):
        FixedDegrader(scale_factor=2, mode="fourier")


def test_gradients_flow():
    A = FixedDegrader(scale_factor=4)
    x = torch.randn(1, 6, 16, 16, 16, requires_grad=True)
    loss = A(x).pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda():
    A = FixedDegrader(scale_factor=4)
    x = torch.randn(1, 6, 16, 16, 16, device="cuda")
    y = A(x)
    assert y.is_cuda
    assert y.shape == (1, 6, 4, 4, 4)
