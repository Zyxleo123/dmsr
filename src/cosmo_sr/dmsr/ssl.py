"""Masked 3D reconstruction pretraining for the LR condition encoder.

Stage B's premise is that the ~350 LR-only boxes carry usable structure about the
*conditioning* distribution even though they can say nothing about ``ker A`` (the
fixed-operator identifiability limit -- unpaired LR under a single fixed ``A``
carries zero information about the null space; see ``docs/`` and the project
notes). Pretraining the condition encoder is the honest way to use them at this
stage: it improves how well the model *reads* ``y``, and makes no claim about
recovering unresolved detail.

**Split discipline.** Every validation and test box is excluded from pretraining,
even though LR fields exist for them. Otherwise the encoder would have seen the
held-out conditions and the comparison would be worthless. The pretraining
manifest is written to the run directory so this is auditable after the fact.

Augmentations, per the design:

* random 3D **block** masking (contiguous cubes, not scattered voxels -- scattered
  masking is trivially solved by local interpolation and teaches nothing);
* occasional **channel** masking (whole 3-vector triples, so a masked field stays
  a coherent physical configuration);
* periodic **translations** (the boxes are periodic, so this is exact);
* proper **cubic rotations** from :mod:`cosmo_sr.dmsr.cubic`, which rotate voxel
  axes *and* vector components together.

The loss is computed **only on masked regions** -- scoring visible voxels lets the
model win by learning the identity.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .cubic import sample_cubic_rotation


def block_mask(
    shape: Tuple[int, ...],
    block_size: int = 2,
    mask_ratio: float = 0.5,
    device=None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Boolean mask ``(B, 1, n, n, n)``; ``True`` marks **masked** (hidden) voxels.

    Built at block resolution then broadcast up, so masked regions are contiguous
    ``block_size^3`` cubes.

    The random draw happens on the **CPU** and the result is moved to ``device``.
    ``torch.rand`` requires the generator's device to match the tensor's, so
    passing a CPU generator together with ``device="cuda"`` raises. Drawing on CPU
    keeps one seeded CPU generator valid for every call site here (the cube
    rotation and translation draws are inherently CPU-side), which also makes a
    run bit-reproducible from its seed regardless of which device it lands on.
    """
    b, _, n = shape[0], shape[1], shape[-1]
    if n % block_size != 0:
        raise ValueError(f"grid {n} not divisible by block_size {block_size}")
    nb = n // block_size
    probs = torch.rand(b, 1, nb, nb, nb, generator=generator)
    coarse = (probs < float(mask_ratio)).to(device)
    return coarse.repeat_interleave(block_size, -3).repeat_interleave(
        block_size, -2
    ).repeat_interleave(block_size, -1)


def channel_mask(
    b: int,
    channels: int,
    p: float = 0.15,
    device=None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Boolean ``(B, C, 1, 1, 1)``; ``True`` marks masked channels.

    Masks whole 3-vector triples together: hiding one component of a displacement
    while showing the other two is not a physically meaningful corruption.
    """
    if channels % 3 != 0:
        raise ValueError(f"channels must be a multiple of 3, got {channels}")
    n_tri = channels // 3
    # Drawn on CPU then moved -- see block_mask for why.
    tri = (torch.rand(b, n_tri, generator=generator) < float(p)).to(device)
    return tri.repeat_interleave(3, dim=1).view(b, channels, 1, 1, 1)


def augment_lr(
    y: torch.Tensor,
    translate: bool = True,
    rotate: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Periodic translation + proper cubic rotation of an LR batch."""
    if translate:
        n = y.shape[-1]
        shifts = torch.randint(0, n, (3,), generator=generator).tolist()
        y = torch.roll(y, shifts=shifts, dims=(-3, -2, -1))
    if rotate:
        y = sample_cubic_rotation(generator).apply(y)
    return y


def masked_reconstruction_loss(
    model: torch.nn.Module,
    y: torch.Tensor,
    block_size: int = 2,
    mask_ratio: float = 0.5,
    channel_mask_p: float = 0.15,
    translate: bool = True,
    rotate: bool = True,
    lambda_fourier: float = 0.0,
    n_bands: int = 6,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Voxel-space masked reconstruction loss (+ optional weak Fourier term).

    Returns ``(loss, metrics)``. Masked voxels are zeroed in the input; the loss
    averages squared error over masked positions only.
    """
    y = augment_lr(y, translate=translate, rotate=rotate, generator=generator)
    b, c = y.shape[0], y.shape[1]

    spatial = block_mask(
        y.shape, block_size=block_size, mask_ratio=mask_ratio,
        device=y.device, generator=generator,
    )
    chan = channel_mask(b, c, p=channel_mask_p, device=y.device, generator=generator)
    masked = spatial | chan  # (B, C, n, n, n) by broadcast

    y_in = y.masked_fill(masked, 0.0)
    pred = model(y_in)

    n_masked = masked.expand_as(y).sum().clamp_min(1)
    err = (pred - y) ** 2
    loss = (err * masked).sum() / n_masked

    metrics = {
        "ssl_loss_voxel": float(loss.detach()),
        "ssl_mask_frac": float(masked.expand_as(y).float().mean().detach()),
    }

    if lambda_fourier > 0.0:
        # Weak spectral term on the *whole* field: keeps the reconstruction from
        # matching masked voxels while drifting in overall power.
        from ..losses.flow import band_power

        band_pred = band_power(pred, n_bands=n_bands, log=True)
        band_true = band_power(y, n_bands=n_bands, log=True)
        fourier = F.mse_loss(band_pred, band_true)
        loss = loss + float(lambda_fourier) * fourier
        metrics["ssl_loss_fourier"] = float(fourier.detach())

    metrics["ssl_loss"] = float(loss.detach())
    return loss, metrics
