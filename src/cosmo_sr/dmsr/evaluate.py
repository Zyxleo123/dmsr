"""Held-out evaluation for the DMSR stage.

Everything here is computed on **completely held-out simulation boxes** (see
:func:`cosmo_sr.dmsr.data.resolve_split`), and every summary is reported in three
``k`` ranges, because the whole point of the method lives in one of them:

``low``
    ``k <= low_frac * k_LR`` -- reproduced *exactly* by ``A_plus(y)``, so this is a
    regression guard, not a measure of success. Stage D must not degrade it.
``transition``
    ``low_frac * k_LR < k <= high_frac * k_LR`` -- straddling the LR resolution
    limit. This is where conditioning on ``y`` can genuinely be checked: the
    model has partial information and must use it.
``high``
    ``k > high_frac * k_LR`` -- unresolved. Only the prior and the critic can put
    power here; ``r(k)`` is expected to be low and ``T(k) ~ 1`` is the target.

``k_LR = N_hr / (2 * s)`` in integer mode units is the LR Nyquist frequency.

Metric conventions
------------------
``r(k) = P_xy / sqrt(P_xx P_yy)`` measures phase/structural agreement; ``T(k) =
sqrt(P_xx / P_yy)`` measures amplitude. They are reported separately on purpose:
a mean-collapsed model has good ``r`` and bad ``T``, an unconditional prior has
good ``T`` and bad ``r``, and a single blended score would hide both failures.
The project's earlier flow runs failed exactly this way (``respow ~ 0.47`` with
acceptable correlation), which is why MSE is *not* the headline number here.

The bispectrum estimators are the standard shell-filtered form: band-pass each
field to a ``k``-shell, multiply in configuration space, and normalise by the
triangle count obtained the same way from the shell indicators.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .density import HighPassDensity
from .env import environment_descriptors


# --------------------------------------------------------------------------- #
# Spectral helpers
# --------------------------------------------------------------------------- #
def _kmag(n: int, device) -> torch.Tensor:
    kx = torch.fft.fftfreq(n, device=device) * n
    kz = torch.fft.rfftfreq(n, device=device) * n
    KX, KY, KZ = torch.meshgrid(kx, kx, kz, indexing="ij")
    return torch.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)


def _shell_bins(n: int, n_bins: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear radial bin index and bin centres over ``[1, n/2]``."""
    kmag = _kmag(n, device)
    edges = torch.linspace(1.0, n / 2.0, n_bins + 1, device=device)
    idx = torch.bucketize(kmag, edges[1:-1].contiguous()).clamp(0, n_bins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return idx, centres


def auto_cross_power(
    a: torch.Tensor, b: torch.Tensor, n_bins: int = 24
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shell-averaged ``(P_aa, P_bb, P_ab, k_centres)`` for two ``(B, C, N,N,N)`` fields.

    Channels are averaged (the fields here are stacked 3-vectors or a single
    scalar density), matching the convention in ``losses/flow.band_power``.
    """
    n = a.shape[-1]
    fa = torch.fft.rfftn(a.float(), dim=(-3, -2, -1))
    fb = torch.fft.rfftn(b.float(), dim=(-3, -2, -1))
    paa = (fa.real ** 2 + fa.imag ** 2) / n ** 3
    pbb = (fb.real ** 2 + fb.imag ** 2) / n ** 3
    pab = (fa.real * fb.real + fa.imag * fb.imag) / n ** 3

    idx, centres = _shell_bins(n, n_bins, a.device)
    flat_idx = idx.reshape(-1)
    counts = torch.zeros(n_bins, device=a.device).index_add_(
        0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32)
    )
    out = []
    for p in (paa, pbb, pab):
        m = p.mean(dim=(0, 1)).reshape(-1)
        s = torch.zeros(n_bins, device=a.device).index_add_(0, flat_idx, m)
        out.append(s / counts.clamp_min(1.0))
    return out[0], out[1], out[2], centres


@dataclass
class BandEdges:
    """``k`` band boundaries in units of the LR Nyquist frequency."""

    low_frac: float = 0.5
    high_frac: float = 1.5

    def masks(self, centres: torch.Tensor, k_lr: float) -> Dict[str, torch.Tensor]:
        return {
            "low": centres <= self.low_frac * k_lr,
            "transition": (centres > self.low_frac * k_lr) & (centres <= self.high_frac * k_lr),
            "high": centres > self.high_frac * k_lr,
        }


def rk_tk_summary(
    pred: torch.Tensor,
    true: torch.Tensor,
    factor: int,
    n_bins: int = 24,
    bands: Optional[BandEdges] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """Band-summarised ``r(k)`` and ``|T(k) - 1|`` (the ``T`` *error*, so lower is better)."""
    bands = bands or BandEdges()
    p_pp, p_tt, p_pt, centres = auto_cross_power(pred, true, n_bins)
    rk = p_pt / (p_pp * p_tt).clamp_min(1e-30).sqrt()
    tk = (p_pp / p_tt.clamp_min(1e-30)).clamp_min(0.0).sqrt()

    k_lr = pred.shape[-1] / (2.0 * factor)
    out: Dict[str, float] = {}
    for name, mask in bands.masks(centres, k_lr).items():
        if not bool(mask.any()):
            out[f"{prefix}rk_{name}"] = float("nan")
            out[f"{prefix}Tk_error_{name}"] = float("nan")
            continue
        out[f"{prefix}rk_{name}"] = float(rk[mask].mean())
        out[f"{prefix}Tk_error_{name}"] = float((tk[mask] - 1.0).abs().mean())
    return out


def power_error(pred: torch.Tensor, true: torch.Tensor, n_bins: int = 24) -> float:
    """Mean ``|log(P_pred / P_true)|`` -- a scale-free power-spectrum error."""
    p_pp, p_tt, _, _ = auto_cross_power(pred, true, n_bins)
    return float(torch.log((p_pp / p_tt.clamp_min(1e-30)).clamp_min(1e-12)).abs().mean())


# --------------------------------------------------------------------------- #
# PDFs
# --------------------------------------------------------------------------- #
def pdf_error(
    pred: torch.Tensor, true: torch.Tensor, n_bins: int = 50, clip: float = 8.0
) -> float:
    """L1 distance between normalised histograms, on a shared range.

    Bin edges come from the **true** field so the comparison is not silently
    rescaled by a prediction with the wrong dynamic range.
    """
    t = true.detach().float().reshape(-1)
    p = pred.detach().float().reshape(-1)
    lo = float(t.mean() - clip * t.std())
    hi = float(t.mean() + clip * t.std())
    ht = torch.histc(t.clamp(lo, hi), bins=n_bins, min=lo, max=hi)
    hp = torch.histc(p.clamp(lo, hi), bins=n_bins, min=lo, max=hi)
    ht = ht / ht.sum().clamp_min(1.0)
    hp = hp / hp.sum().clamp_min(1.0)
    return float((hp - ht).abs().sum())


def divergence_field(vec: torch.Tensor) -> torch.Tensor:
    """Periodic central-difference divergence of a ``(B, 3, N, N, N)`` field."""
    d = torch.zeros_like(vec[:, :1])
    for i, dim in enumerate((-3, -2, -1)):
        c = vec[:, i : i + 1]
        d = d + 0.5 * (torch.roll(c, -1, dims=dim) - torch.roll(c, 1, dims=dim))
    return d


# --------------------------------------------------------------------------- #
# Bispectra (shell-filtered estimators)
# --------------------------------------------------------------------------- #
def _shell_filter(fft: torch.Tensor, kmag: torch.Tensor, klo: float, khi: float, n: int):
    mask = ((kmag >= klo) & (kmag < khi)).to(fft.real.dtype)
    return torch.fft.irfftn(fft * mask, s=(n, n, n), dim=(-3, -2, -1)), mask


def equilateral_bispectrum(
    field: torch.Tensor, k_centres: Sequence[float], width: float
) -> torch.Tensor:
    """``B(k, k, k)`` for each ``k`` in ``k_centres``, triangle-count normalised."""
    n = field.shape[-1]
    x = field.float().mean(dim=1, keepdim=True)  # scalar-ise
    fx = torch.fft.rfftn(x, dim=(-3, -2, -1))
    kmag = _kmag(n, field.device)
    ones = torch.ones_like(fx)

    out = []
    for k in k_centres:
        dk, mask = _shell_filter(fx, kmag, k - width / 2, k + width / 2, n)
        ik = torch.fft.irfftn(ones * mask, s=(n, n, n), dim=(-3, -2, -1))
        num = (dk ** 3).mean()
        den = (ik ** 3).mean()
        out.append(num / den.clamp_min(1e-30) if abs(float(den)) > 1e-30 else torch.zeros((), device=field.device))
    return torch.stack([o.reshape(()) for o in out])


def squeezed_cross_bispectrum(
    pred: torch.Tensor,
    true_long: torch.Tensor,
    k_long: float,
    k_short: float,
    width: float,
) -> torch.Tensor:
    """``<delta_true(k_L) delta_pred(k_S) delta_pred(k_S)>``, triangle-count normalised.

    This is the sharpest *conditional* statistic available to us: it asks whether
    the small-scale power the model invents is modulated by the **true**
    large-scale environment. A model that produces marginally realistic residuals
    while ignoring ``y`` scores near zero here even when its power spectrum is
    perfect -- which is exactly the failure the condition-shuffle diagnostic in
    :func:`condition_shuffle_gap` is designed to catch.
    """
    n = pred.shape[-1]
    xs = pred.float().mean(dim=1, keepdim=True)
    xl = true_long.float().mean(dim=1, keepdim=True)
    fs = torch.fft.rfftn(xs, dim=(-3, -2, -1))
    fl = torch.fft.rfftn(xl, dim=(-3, -2, -1))
    kmag = _kmag(n, pred.device)
    ones = torch.ones_like(fs)

    d_long, m_long = _shell_filter(fl, kmag, k_long - width / 2, k_long + width / 2, n)
    d_short, m_short = _shell_filter(fs, kmag, k_short - width / 2, k_short + width / 2, n)
    i_long = torch.fft.irfftn(ones * m_long, s=(n, n, n), dim=(-3, -2, -1))
    i_short = torch.fft.irfftn(ones * m_short, s=(n, n, n), dim=(-3, -2, -1))
    num = (d_long * d_short ** 2).mean()
    den = (i_long * i_short ** 2).mean()
    return num / den.clamp_min(1e-30)


# --------------------------------------------------------------------------- #
# Stochasticity
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_diversity(
    flow, y: torch.Tensor, n_samples: int = 4, n_steps: int = 20
) -> Dict[str, float]:
    """Residual variance and mean pairwise distance across samples for one ``y``.

    Near-zero diversity means mode collapse: the flow has become a deterministic
    regressor and the ``z`` input is being ignored.
    """
    if n_samples < 2:
        raise ValueError("need at least 2 samples to measure diversity")
    residuals = [flow.sample_residual(y, n_steps=n_steps) for _ in range(int(n_samples))]
    stack = torch.stack(residuals)                       # (S, B, C, N, N, N)
    var = stack.var(dim=0, unbiased=True).mean()
    scale = stack.pow(2).mean().sqrt().clamp_min(1e-12)

    dists = []
    for i in range(len(residuals)):
        for j in range(i + 1, len(residuals)):
            dists.append((residuals[i] - residuals[j]).pow(2).mean().sqrt())
    return {
        "sample_diversity": float(var.sqrt() / scale),
        "pairwise_distance": float(torch.stack(dists).mean() / scale),
        "residual_rms": float(scale),
    }


@torch.no_grad()
def condition_shuffle_gap(
    flow,
    y: torch.Tensor,
    x_hr: torch.Tensor,
    factor: int,
    n_steps: int = 20,
    n_bins: int = 24,
    seed: int = 0,
) -> Dict[str, float]:
    """Correct-condition vs shuffled-condition performance.

    Both outputs are built on the **same** base ``A_plus(y_i)``, so both remain
    exactly consistent with ``y_i``; only the residual's conditioning differs.
    A gap near zero means the residual generator is producing marginally
    realistic detail while ignoring the LR structure it was given.
    """
    b = y.shape[0]
    if b < 2:
        return {"condition_shuffle_gap": float("nan")}
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(b, generator=g).to(y.device)
    if bool((perm == torch.arange(b, device=y.device)).all()):
        perm = torch.roll(perm, 1)

    z = torch.randn(
        b, y.shape[1], y.shape[-1] * factor, y.shape[-1] * factor, y.shape[-1] * factor,
        device=y.device, dtype=y.dtype,
    )
    base = flow.operator.A_plus(y)
    x_correct = base + flow.sample_residual(y, n_steps=n_steps, z=z)
    x_shuffled = base + flow.sample_residual(y[perm], n_steps=n_steps, z=z)

    corr = rk_tk_summary(x_correct, x_hr, factor, n_bins=n_bins, prefix="correct_")
    shuf = rk_tk_summary(x_shuffled, x_hr, factor, n_bins=n_bins, prefix="shuffled_")
    out = {**corr, **shuf}
    out["condition_shuffle_gap"] = float(
        corr["correct_rk_transition"] - shuf["shuffled_rk_transition"]
    )
    return out


# --------------------------------------------------------------------------- #
# Full held-out evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_batch(
    flow,
    y: torch.Tensor,
    x_hr: torch.Tensor,
    factor: int,
    highpass: HighPassDensity,
    n_steps: int = 20,
    n_bins: int = 24,
    bands: Optional[BandEdges] = None,
    disp_channels: Sequence[int] = (0, 1, 2),
    vel_channels: Optional[Sequence[int]] = (3, 4, 5),
) -> Dict[str, float]:
    """All field / distributional metrics for one held-out batch."""
    x_hat = flow.generate(y, n_steps=n_steps)
    out: Dict[str, float] = {}

    abs_err, rel_err = flow.operator.consistency_error(x_hat, y)
    out["exact_consistency_abs"] = abs_err
    out["exact_consistency_rel"] = rel_err
    out["mse"] = float((x_hat - x_hr).pow(2).mean())
    for c in range(x_hr.shape[1]):
        out[f"mse_ch{c}"] = float((x_hat[:, c] - x_hr[:, c]).pow(2).mean())

    out.update(rk_tk_summary(x_hat, x_hr, factor, n_bins=n_bins, bands=bands))

    # Eulerian density
    rho_hat, rho_true = highpass.density(x_hat), highpass.density(x_hr)
    out["density_power_error"] = power_error(rho_hat, rho_true, n_bins)
    out["density_pdf_error"] = pdf_error(rho_hat, rho_true)
    out.update(rk_tk_summary(rho_hat, rho_true, factor, n_bins=n_bins, bands=bands, prefix="density_"))

    # Velocity
    if vel_channels is not None and max(vel_channels) < x_hr.shape[1]:
        idx = torch.tensor(list(vel_channels), device=x_hr.device)
        v_hat, v_true = x_hat.index_select(1, idx), x_hr.index_select(1, idx)
        out["velocity_power_error"] = power_error(v_hat, v_true, n_bins)
        out["velocity_divergence_pdf_error"] = pdf_error(
            divergence_field(v_hat), divergence_field(v_true)
        )

    # Bispectra
    n = x_hat.shape[-1]
    k_lr = n / (2.0 * factor)
    width = max(2.0, k_lr / 4.0)
    ks = [0.5 * k_lr, k_lr, 1.5 * k_lr]
    b_hat = equilateral_bispectrum(rho_hat, ks, width)
    b_true = equilateral_bispectrum(rho_true, ks, width)
    out["bispectrum_error"] = float(
        ((b_hat - b_true).abs() / b_true.abs().clamp_min(1e-20)).mean()
    )
    sq_hat = squeezed_cross_bispectrum(rho_hat, rho_true, 0.3 * k_lr, 1.5 * k_lr, width)
    sq_true = squeezed_cross_bispectrum(rho_true, rho_true, 0.3 * k_lr, 1.5 * k_lr, width)
    out["squeezed_cross_bispectrum_error"] = float(
        (sq_hat - sq_true).abs() / sq_true.abs().clamp_min(1e-20)
    )
    return out


def environment_bin(
    y: torch.Tensor,
    standardizer,
    paired_z_mean: np.ndarray,
    paired_z_cov_inv: np.ndarray,
    edges: Sequence[float],
    descriptor_kwargs: Optional[Dict] = None,
) -> np.ndarray:
    """Bin crops by Mahalanobis distance from the *paired training* environment.

    The main theoretical benefit of extra LR data is broader condition coverage,
    so Stage D should improve most in bins that are under-represented in the
    paired pool but still inside its support. Bin 0 is the paired core; higher
    bins are progressively further out.
    """
    desc = environment_descriptors(y, **(descriptor_kwargs or {})).cpu().numpy()
    z = standardizer.transform(desc)
    d = z - paired_z_mean[None, :]
    m = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, paired_z_cov_inv, d), 0.0))
    return np.digitize(m, np.asarray(edges))
