"""Masked-SSL pretraining: masking semantics and device discipline.

The device tests exist because of a real failure: ``block_mask`` called
``torch.rand(..., device="cuda", generator=cpu_generator)``, which raises
``RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'``.
Every CPU-only test passed, and the bug only surfaced on the GPU node after the
job had queued, allocated and initialised W&B.

CPU-only tests structurally cannot catch device-placement bugs, so
:func:`test_seeded_generator_is_never_passed_to_a_device_tensor_factory`
asserts the *invariant* instead of the outcome: a seeded generator is only ever
used to create CPU tensors. That runs anywhere, including here.
"""
from __future__ import annotations

import torch

from cosmo_sr.dmsr.encoder import LRMaskedAutoencoder
from cosmo_sr.dmsr.ssl import (
    augment_lr,
    block_mask,
    channel_mask,
    masked_reconstruction_loss,
)


# --------------------------------------------------------------------------- #
# Masking semantics
# --------------------------------------------------------------------------- #
def test_block_mask_shape_and_blockiness():
    m = block_mask((2, 3, 8, 8, 8), block_size=2, mask_ratio=0.5)
    assert m.shape == (2, 1, 8, 8, 8)
    assert m.dtype == torch.bool
    # Every 2^3 block must be uniformly masked or uniformly visible.
    blocks = m[:, :, ::2, ::2, ::2]
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                assert torch.equal(m[:, :, dx::2, dy::2, dz::2], blocks)


def test_block_mask_ratio_is_approximately_honoured():
    m = block_mask((64, 3, 8, 8, 8), block_size=2, mask_ratio=0.5)
    assert 0.4 < float(m.float().mean()) < 0.6


def test_block_mask_rejects_indivisible_grid():
    import pytest

    with pytest.raises(ValueError, match="not divisible"):
        block_mask((1, 3, 7, 7, 7), block_size=2)


def test_channel_mask_hides_whole_vector_triples():
    m = channel_mask(16, 6, p=0.5)
    assert m.shape == (16, 6, 1, 1, 1)
    flat = m.view(16, 6)
    for t in (0, 3):
        tri = flat[:, t : t + 3]
        assert torch.equal(tri[:, 0], tri[:, 1]) and torch.equal(tri[:, 1], tri[:, 2]), (
            "a 3-vector triple was only partially masked"
        )


def test_channel_mask_rejects_non_multiple_of_three():
    import pytest

    with pytest.raises(ValueError, match="multiple of 3"):
        channel_mask(2, 4, p=0.5)


def test_loss_is_computed_only_on_masked_voxels():
    """A model that is perfect on masked voxels scores 0 regardless of the rest."""
    torch.manual_seed(0)
    y = torch.randn(2, 3, 8, 8, 8)

    class PerfectOnMasked(torch.nn.Module):
        def forward(self, y_in):
            # y_in has masked voxels zeroed; return the true field everywhere it
            # was masked, and garbage where it was visible.
            return torch.where(y_in == 0.0, PerfectOnMasked.truth, torch.full_like(y_in, 99.0))

    gen = torch.Generator().manual_seed(3)
    # Re-derive the exact augmented field the loss will use.
    PerfectOnMasked.truth = augment_lr(y, generator=torch.Generator().manual_seed(3))
    loss, _ = masked_reconstruction_loss(
        PerfectOnMasked(), y, block_size=2, mask_ratio=0.5, channel_mask_p=0.0,
        generator=gen,
    )
    assert float(loss) < 1e-8, f"loss {float(loss):.3e} should ignore visible voxels"


def test_masked_reconstruction_loss_runs_and_is_differentiable():
    torch.manual_seed(0)
    model = LRMaskedAutoencoder(in_channels=3, width=8, cond_channels=4)
    y = torch.randn(2, 3, 8, 8, 8)
    loss, metrics = masked_reconstruction_loss(
        model, y, block_size=2, mask_ratio=0.5, channel_mask_p=0.15,
        lambda_fourier=0.1, generator=torch.Generator().manual_seed(0),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.encoder.parameters())
    assert {"ssl_loss", "ssl_loss_voxel", "ssl_mask_frac", "ssl_loss_fourier"} <= set(metrics)


def test_seeded_run_is_reproducible():
    torch.manual_seed(0)
    model = LRMaskedAutoencoder(in_channels=3, width=8, cond_channels=4)
    y = torch.randn(2, 3, 8, 8, 8)

    def run(seed):
        loss, _ = masked_reconstruction_loss(
            model, y, generator=torch.Generator().manual_seed(seed))
        return float(loss)

    assert run(5) == run(5)
    assert run(5) != run(6)


# --------------------------------------------------------------------------- #
# Device discipline (the regression that actually bit)
# --------------------------------------------------------------------------- #
def test_masks_land_on_the_requested_device_with_a_cpu_generator():
    gen = torch.Generator().manual_seed(0)
    m = block_mask((2, 3, 8, 8, 8), device=torch.device("cpu"), generator=gen)
    c = channel_mask(2, 3, device=torch.device("cpu"), generator=gen)
    assert m.device.type == "cpu" and c.device.type == "cpu"


def test_generator_and_device_are_never_passed_to_the_same_factory(monkeypatch):
    """REGRESSION for the CPU-generator / CUDA-device crash.

    Asserting the *actual* torch rule ("generator.device must equal device") is
    untestable without a GPU: on a CPU-only host the two always match, so a test
    written that way passes against the broken code (verified). ``device="meta"``
    does not enforce the rule either, and ``device="cuda"`` fails earlier with
    "No CUDA GPUs are available".

    So this asserts the stronger, host-independent invariant that the fix
    actually establishes: **never hand ``device`` and ``generator`` to the same
    tensor factory.** Draw on CPU with the seeded generator, then ``.to(device)``.
    That is what makes the masking code portable *and* keeps one seeded CPU
    generator reproducible across devices. It fails against the original
    implementation and passes against the fixed one.
    """
    violations = []
    for name in ("rand", "randint", "randn", "randperm"):
        original = getattr(torch, name)

        def checked(*args, _orig=original, _name=name, **kwargs):
            if kwargs.get("generator") is not None and kwargs.get("device") is not None:
                violations.append(f"torch.{_name}(device=..., generator=...)")
            return _orig(*args, **kwargs)

        monkeypatch.setattr(torch, name, checked)

    model = LRMaskedAutoencoder(in_channels=3, width=8, cond_channels=4)
    y = torch.randn(2, 3, 8, 8, 8)
    masked_reconstruction_loss(
        model, y, block_size=2, mask_ratio=0.5, channel_mask_p=0.15,
        generator=torch.Generator().manual_seed(0),
    )
    assert not violations, (
        "generator passed together with device (crashes when the model is on GPU): "
        + "; ".join(sorted(set(violations)))
    )
