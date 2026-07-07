"""2D central-slice visualisations for evaluation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from .spectra import _to_numpy


def central_slice(field, channel: int = 0) -> np.ndarray:
    """Return the central 2D slice ``field[channel, N//2]`` of ``(C, N, N, N)``."""
    field = _to_numpy(field)
    if field.ndim != 4:
        raise ValueError(f"central_slice expects (C, N, N, N), got {field.shape}")
    N = field.shape[1]
    return field[channel, N // 2]


def save_eval_slices(
    out_path: str | os.PathLike,
    y_lr,
    x_hat_hr,
    y_recon=None,
    x_hr=None,
    channel: int = 0,
) -> str:
    """Save a panel of central slices: LR input, generated HR, degraded-gen LR,
    HR target (if given) and the HR error slice (if target given)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("LR input", central_slice(y_lr, channel)),
              ("Generated HR", central_slice(x_hat_hr, channel))]
    if y_recon is not None:
        panels.append(("A(gen) LR", central_slice(y_recon, channel)))
    if x_hr is not None:
        panels.append(("HR target", central_slice(x_hr, channel)))
        err = central_slice(x_hat_hr, channel) - central_slice(x_hr, channel)
        panels.append(("HR error", err))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        im = ax.imshow(img, origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return str(out_path)
