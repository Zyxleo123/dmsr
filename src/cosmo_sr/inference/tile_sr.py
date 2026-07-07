"""Full-box tiled super-resolution inference (SRS-map2map style).

Split the LR box into ``nsplit**3`` chunks, add periodic padding, run ``G`` on
each padded chunk, trim the padded border from the generated HR chunk, and stitch
the trimmed chunks into the full HR field.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch

from ..data.crops import periodic_crop


def _as_torch(field) -> "tuple[torch.Tensor, bool]":
    was_numpy = not torch.is_tensor(field)
    if was_numpy:
        field = torch.from_numpy(np.ascontiguousarray(field)).float()
    return field, was_numpy


def super_resolve_full_box(
    model: torch.nn.Module,
    lr_field: Union[np.ndarray, torch.Tensor],
    scale_factor: int,
    nsplit: int,
    pad_lr: int,
):
    """Super-resolve a full ``(C, Ng, Ng, Ng)`` LR box by tiling.

    Parameters
    ----------
    model:
        Generator ``G`` mapping ``(B, C, n, n, n)`` LR chunks to
        ``(B, C, n*scale, n*scale, n*scale)`` HR chunks.
    lr_field:
        ``(C, Ng, Ng, Ng)`` LR field (numpy or torch).
    scale_factor:
        Upsampling ratio.
    nsplit:
        Number of chunks per axis. Must divide ``Ng``.
    pad_lr:
        Periodic LR padding (in LR voxels) added to each chunk before SR; the
        corresponding ``pad_lr*scale_factor`` HR border is trimmed after SR.

    Returns
    -------
    ``(C, Ng*scale, Ng*scale, Ng*scale)`` HR field, same backend as input.
    """
    if lr_field.ndim != 4:
        raise ValueError(
            f"lr_field must be (C, Ng, Ng, Ng), got shape {tuple(lr_field.shape)}"
        )
    C, nx, ny, nz = lr_field.shape
    if not (nx == ny == nz):
        raise ValueError(f"Only cubic LR boxes are supported; got {(nx, ny, nz)}")
    Ng = nx
    scale_factor = int(scale_factor)
    nsplit = int(nsplit)
    pad_lr = int(pad_lr)
    if nsplit < 1:
        raise ValueError("nsplit must be >= 1")
    if Ng % nsplit != 0:
        raise ValueError(
            f"nsplit={nsplit} does not divide Ng_lr={Ng}; choose a divisor."
        )

    field_t, was_numpy = _as_torch(lr_field)
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    field_t = field_t.to(device)

    chunk = Ng // nsplit
    Ng_hr = Ng * scale_factor
    border = pad_lr * scale_factor

    out = torch.zeros((C, Ng_hr, Ng_hr, Ng_hr), dtype=field_t.dtype, device=device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for ix in range(nsplit):
            for iy in range(nsplit):
                for iz in range(nsplit):
                    start = (ix * chunk, iy * chunk, iz * chunk)
                    lr_chunk = periodic_crop(field_t, start, chunk, pad=pad_lr)
                    hr_chunk = model(lr_chunk.unsqueeze(0)).squeeze(0)
                    expected = (chunk + 2 * pad_lr) * scale_factor
                    if hr_chunk.shape[1] != expected:
                        raise ValueError(
                            f"Model output size {hr_chunk.shape[1]} != expected "
                            f"{expected} for padded chunk; check scale_factor."
                        )
                    if border > 0:
                        hr_chunk = hr_chunk[
                            :,
                            border:border + chunk * scale_factor,
                            border:border + chunk * scale_factor,
                            border:border + chunk * scale_factor,
                        ]
                    hs = (ix * chunk * scale_factor,
                          iy * chunk * scale_factor,
                          iz * chunk * scale_factor)
                    cs = chunk * scale_factor
                    out[:, hs[0]:hs[0] + cs, hs[1]:hs[1] + cs, hs[2]:hs[2] + cs] = hr_chunk
    if was_training:
        model.train()

    if was_numpy:
        return out.cpu().numpy()
    return out
