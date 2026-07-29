"""Part 1 -- explain and verify the null space of block-average-and-decimate.

Two claims are made concrete here:

1. The synthetic 1D factor-two null space ``r = [a0,-a0,a1,-a1,...]`` is annihilated
   by ``A`` for *any* envelope ``a`` (constant or slowly varying), and its Fourier
   spectrum is a *structured* combination of modes -- a single Nyquist spike when
   ``a`` is constant, but a spread of modes (reaching well below Nyquist) when ``a``
   varies -- NOT one perfectly separated frequency interval.

2. The real 3D operator satisfies the projector identities, ``P_A`` removes the
   blockwise mean, projected random fields keep a *non-uniform* (structured)
   spectrum, and an arbitrary Fourier mode is generally **not** an eigenvector of
   ``P_A`` (unlike an ideal high-pass, for which every mode is an eigenvector).
"""
from __future__ import annotations

import pytest
import torch

from cosmo_sr.dmsr.nullspace_spectral import (
    alternating_null_vector,
    block_average_1d,
    block_zero_mean_vector,
    null_projection_1d,
    radial_power_spectrum_3d,
    rfft_power,
)
from cosmo_sr.dmsr.operator import NullSpaceOperator

TOL = 1e-6


# --------------------------------------------------------------------------- #
# 1D aliasing demonstration
# --------------------------------------------------------------------------- #
def test_alternating_vector_is_annihilated_for_constant_envelope():
    a = torch.ones(8)
    r = alternating_null_vector(a)
    assert float(block_average_1d(r, 2).abs().max()) < TOL


def test_alternating_vector_is_annihilated_for_slowly_varying_envelope():
    j = torch.arange(64, dtype=torch.float64)
    for env in (torch.cos(2 * torch.pi * j / 64),          # one slow cycle
                1.0 + 0.1 * j,                              # linear ramp
                torch.exp(-((j - 32) ** 2) / 200.0)):       # gaussian bump
        r = alternating_null_vector(env)
        assert float(block_average_1d(r, 2).abs().max()) < 1e-9, "A(r) != 0"


def test_general_block_zero_mean_vector_is_annihilated_for_factor_s():
    torch.manual_seed(0)
    for factor in (2, 3, 4):
        coeffs = torch.randn(10, factor, dtype=torch.float64)
        r = block_zero_mean_vector(coeffs, factor)
        assert float(block_average_1d(r, factor).abs().max()) < 1e-9


def test_constant_envelope_is_a_single_nyquist_mode():
    """Constant ``a`` -> pure Nyquist carrier -> power in exactly the last rfft bin."""
    a = torch.ones(32)
    p = rfft_power(alternating_null_vector(a))       # length N/2+1 = 33
    peak = int(p.argmax())
    assert peak == p.numel() - 1, "constant envelope should peak at the Nyquist bin"
    # essentially all power in that one bin
    assert float(p[-1] / p.sum()) > 0.999


def test_slowly_varying_envelope_gives_a_structured_multimode_spectrum():
    """A varying envelope is a *modulation product*, not a single high band.

    Multiplying a low-frequency envelope ``a`` (coarse frequency ``f``) by the
    Nyquist carrier shifts ``a``'s spectrum to sit around Nyquist (component at
    ``N/2 - f``) while the block-broadcast boxcar replica leaves a genuine
    *low-frequency* sideband (component near ``f``). The spectrum therefore
    straddles BOTH a low band and a near-Nyquist band simultaneously -- something
    an ideal high-pass at ``k_Ny/2`` (one contiguous interval) can never do.
    """
    N = 64
    j = torch.arange(N // 2, dtype=torch.float64)
    a = torch.cos(2 * torch.pi * 2.0 * j / (N // 2))   # 2 cycles across the envelope
    r = alternating_null_vector(a)
    p = rfft_power(r)                                   # bins 0..N/2 (=32)
    p = p / p.max()

    active = (p > 1e-3).nonzero().reshape(-1)
    assert active.numel() >= 2, "expected a spread of modes, not a single spike"

    nyq = p.numel() - 1                                # = N/2 = 32
    has_low = bool((active < nyq // 4).any())          # a bin below N/8 in rfft units
    has_high = bool((active > (3 * nyq) // 4).any())   # and a bin near Nyquist
    assert has_low and has_high, (
        f"expected a low + near-Nyquist mode pair (modulation), got active bins "
        f"{active.tolist()}; an ideal high-pass band cannot straddle both"
    )


def test_nullspace_is_not_the_ideal_highpass_band():
    """P_A(x) and an ideal Fourier high-pass of x differ substantially in 1D."""
    torch.manual_seed(1)
    N, factor = 128, 2
    x = torch.randn(N, dtype=torch.float64)
    p_null = null_projection_1d(x, factor)

    # ideal high-pass: drop the lower half of rfft modes
    f = torch.fft.rfft(x)
    cut = f.numel() // 2
    f_hi = f.clone()
    f_hi[:cut] = 0
    x_hi = torch.fft.irfft(f_hi, n=N)

    rel = float((p_null - x_hi).norm() / p_null.norm())
    assert rel > 0.3, f"P_A and ideal high-pass are nearly identical (rel={rel:.3f})"


# --------------------------------------------------------------------------- #
# 3D operator on production-like shapes
# --------------------------------------------------------------------------- #
@pytest.fixture(params=[2, 4, 8])
def op(request):
    return NullSpaceOperator(factor=request.param)


def _fields(op, seed=0):
    torch.manual_seed(seed)
    n_lr = 4
    n_hr = n_lr * op.factor
    x = torch.randn(2, 3, n_hr, n_hr, n_hr)
    y = torch.randn(2, 3, n_lr, n_lr, n_lr)
    return x, y


def test_a_a_plus_identity(op):
    _, y = _fields(op)
    assert float((op.A(op.A_plus(y)) - y).norm() / y.norm()) <= 1e-5


def test_a_of_p_a_is_zero(op):
    x, _ = _fields(op)
    assert float(op.A(op.P_A(x)).norm() / op.A(x).norm()) <= 1e-5


def test_p_a_idempotent(op):
    x, _ = _fields(op)
    p = op.P_A(x)
    assert float((op.P_A(p) - p).norm() / p.norm()) <= 1e-5


def test_p_a_removes_the_blockwise_mean(op):
    """The defining property: every block of P_A(x) averages to zero."""
    x, _ = _fields(op)
    block_means = op.A(op.P_A(x))
    assert float(block_means.abs().max()) < 1e-5


def test_projected_random_field_keeps_a_nonuniform_spectrum(op):
    """P_A(white noise) is neither white nor confined to one band: its radial
    per-mode spectrum varies strongly across shells (low k suppressed)."""
    x, _ = _fields(op, seed=3)
    p = op.P_A(x)
    shell_k, per_mode, counts = radial_power_spectrum_3d(p)
    valid = counts > 0
    pm = per_mode[valid]
    # non-uniform: coefficient of variation across shells is large
    cov = float(pm.std() / pm.mean().clamp_min(1e-30))
    assert cov > 0.2, f"projected spectrum looks flat (cov={cov:.3f})"
    # low-k strongly suppressed relative to high-k (block means removed)
    n_shells = pm.numel()
    lo = float(pm[: max(1, n_shells // 4)].mean())
    hi = float(pm[-max(1, n_shells // 4):].mean())
    assert lo < hi, "expected low-k power below high-k power after projection"


def test_arbitrary_fourier_mode_is_not_an_eigenvector_of_p_a(op):
    """An ideal high-pass has every Fourier mode as an eigenvector (0 or 1). P_A
    does not: a generic low-frequency mode is mapped to something not proportional
    to itself (block-broadcast staircase subtracted)."""
    n_lr = 4
    n = n_lr * op.factor
    ax = torch.arange(n, dtype=torch.float32)
    # a generic low mode along x (k=1), constant in y,z -- not aligned with the
    # block structure, so P_A must distort it.
    mode = torch.cos(2 * torch.pi * 1.0 * ax / n).view(1, 1, n, 1, 1).expand(1, 3, n, n, n).contiguous()
    pm = op.P_A(mode)
    # best scalar multiple lam of `mode` explaining pm, then the residual fraction
    lam = float((pm * mode).sum() / (mode * mode).sum().clamp_min(1e-30))
    residual = float((pm - lam * mode).norm() / pm.norm().clamp_min(1e-30))
    assert residual > 0.1, (
        f"P_A acted like an eigen-projector on this mode (residual={residual:.3f}); "
        "it should distort a generic Fourier mode"
    )


def test_block_aligned_modes_ARE_eigenvectors_the_point_is_arbitrary_modes_are_not(op):
    """Complement to the previous test: DC and the block-Nyquist carrier *are*
    eigenvectors (eigenvalues 0 and 1), so the claim is specifically about
    *arbitrary* modes, not all of them."""
    n_lr = 4
    n = n_lr * op.factor
    dc = torch.ones(1, 3, n, n, n)
    assert float(op.P_A(dc).abs().max()) < 1e-5           # eigenvalue 0

    ax = torch.arange(n, dtype=torch.float32)
    carrier = torch.cos(torch.pi * ax).view(1, 1, n, 1, 1).expand(1, 3, n, n, n).contiguous()
    # (+1,-1,+1,-1,...) averages to zero over any even block -> P_A fixes it
    if op.factor % 2 == 0:
        assert float((op.P_A(carrier) - carrier).norm() / carrier.norm()) < 1e-5  # eigenvalue 1
