"""Held-out clean-HR denoising evaluation for the operator-conditioned prior.

Runs clean HR crops through the **identity** branch at several noise levels and
scores ``x0_hat`` against ``x`` with normalised field MSE (overall + per channel)
and Fourier structure/amplitude agreement:

    r(k) = P_cross(k) / sqrt(P_pred(k) P_true(k))     (phase/structure agreement)
    T(k) = sqrt(P_pred(k) / P_true(k))                (amplitude agreement)

These are the P0/P4/P6 decision-gate metrics (plan section 7.1); lower MSE and
r(k)/T(k) nearer 1 are better.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from ..models.operator_denoiser import CosineSchedule


def _radial_bins(n: int, device) -> tuple[torch.Tensor, int]:
    kx = torch.fft.fftfreq(n, device=device) * n
    kz = torch.fft.rfftfreq(n, device=device) * n
    KX, KY, KZ = torch.meshgrid(kx, kx, kz, indexing="ij")
    kb = torch.sqrt(KX ** 2 + KY ** 2 + KZ ** 2).round().long().flatten()
    return kb, int(kb.max()) + 1


def _rk_tk(pred: torch.Tensor, true: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Batched r(k), T(k) averaged over batch+channels. Inputs ``(B, C, N, N, N)``."""
    n = pred.shape[-1]
    kb, nbins = _radial_bins(n, pred.device)
    fp = torch.fft.rfftn(pred, dim=(-3, -2, -1))
    ft = torch.fft.rfftn(true, dim=(-3, -2, -1))
    p_pred = (fp.real ** 2 + fp.imag ** 2).reshape(-1, kb.numel())
    p_true = (ft.real ** 2 + ft.imag ** 2).reshape(-1, kb.numel())
    p_cross = (fp.real * ft.real + fp.imag * ft.imag).reshape(-1, kb.numel())

    def binsum(x):  # (M, K) -> (nbins,)
        out = torch.zeros(nbins, device=x.device)
        out.index_add_(0, kb, x.sum(0))
        return out

    Pp, Pt, Pc = binsum(p_pred), binsum(p_true), binsum(p_cross)
    r = (Pc / torch.sqrt(Pp * Pt).clamp_min(1e-30)).cpu().numpy()
    T = torch.sqrt(Pp / Pt.clamp_min(1e-30)).cpu().numpy()
    return r, T


@torch.no_grad()
def clean_denoise_metrics(
    denoiser: torch.nn.Module,
    val_iter,
    device,
    schedule: Optional[CosineSchedule] = None,
    t_levels: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    n_batches: int = 2,
    t_spectrum: float = 0.5,
    field_key: str = "lr",
) -> Dict[str, float]:
    """Identity-branch denoising metrics, averaged over ``t_levels`` and batches."""
    if val_iter is None:
        return {}
    schedule = schedule or CosineSchedule()
    was_training = denoiser.training
    denoiser.eval()
    mse_sum, ch_sum, count = 0.0, None, 0
    rk_list, tk_list = [], []
    for _ in range(n_batches):
        x = next(val_iter)[field_key].to(device)
        for t_val in t_levels:
            t = torch.full((x.shape[0],), float(t_val), device=device)
            a, s = schedule.broadcast(t, ndim=x.dim())
            x_t = a * x + s * torch.randn_like(x)
            x0 = denoiser(x_t, t, shift=(0, 0, 0), kind="identity")
            denom = x.pow(2).mean().clamp_min(1e-12)
            mse_sum += (x0 - x).pow(2).mean().item() / denom.item()
            ch = ((x0 - x).pow(2).mean(dim=(0, 2, 3, 4))
                  / x.pow(2).mean(dim=(0, 2, 3, 4)).clamp_min(1e-12)).cpu().numpy()
            ch_sum = ch if ch_sum is None else ch_sum + ch
            count += 1
        # spectra at a single representative noise level
        t = torch.full((x.shape[0],), float(t_spectrum), device=device)
        a, s = schedule.broadcast(t, ndim=x.dim())
        x0 = denoiser(a * x + s * torch.randn_like(x), t, shift=(0, 0, 0), kind="identity")
        r, T = _rk_tk(x0, x)
        rk_list.append(r); tk_list.append(T)
    if was_training:
        denoiser.train()
    r = np.mean(rk_list, axis=0); T = np.mean(tk_list, axis=0)
    ch = ch_sum / max(count, 1)
    out = {
        "val_denoise_mse": mse_sum / max(count, 1),
        "val_rk_mean": float(np.nanmean(r[1:])),          # skip k=0 (DC)
        "val_tk_mean": float(np.nanmean(T[1:])),
        "val_tk_highk": float(np.nanmean(T[len(T) // 2:])),  # amplitude at high k
    }
    for c, v in enumerate(ch):
        out[f"val_denoise_mse_ch{c}"] = float(v)
    return out
