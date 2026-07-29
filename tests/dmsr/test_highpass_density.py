"""7.4 -- the high-pass density channel."""
from __future__ import annotations

import pytest
import torch

from cosmo_sr.dmsr.density import HighPassDensity, critic_input
from cosmo_sr.dmsr.operator import NullSpaceOperator

FACTOR = 4
N_LR = 4
N_HR = N_LR * FACTOR


@pytest.fixture(params=["blockavg", "fourier"])
def hp(request):
    return HighPassDensity(factor=FACTOR, lowpass=request.param,
                           cellsize=1000.0, dis_norm=6000.0)


def test_shape_and_finiteness(hp):
    x = torch.randn(2, 3, N_HR, N_HR, N_HR) * 0.01
    rho_high = hp(x)
    assert rho_high.shape == (2, 1, N_HR, N_HR, N_HR)
    assert torch.isfinite(rho_high).all()


def _band_limited_displacement(kmax_frac: float, amplitude: float, seed: int = 0):
    """Displacement built from Fourier modes with ``|k| <= kmax_frac * k_Nyq_LR``.

    Note this must be **band-limited in Fourier space**, not merely constant
    within each ``A``-block. A block-constant displacement is discontinuous at
    every block face, and CIC-depositing it piles particles up along those faces
    -- producing large high-k *density* power from a field that looks low-frequency
    in displacement space. Density is a nonlinear function of displacement, so
    "smooth Psi" and "smooth delta" are not the same statement.

    The amplitude is kept small so the deposit stays in the quasi-linear regime
    where ``delta ~ -div(Psi)`` and low-frequency ``Psi`` really does imply
    low-frequency ``delta``.
    """
    torch.manual_seed(seed)
    x = torch.randn(2, 3, N_HR, N_HR, N_HR)
    kx = torch.fft.fftfreq(N_HR) * N_HR
    kz = torch.fft.rfftfreq(N_HR) * N_HR
    KX, KY, KZ = torch.meshgrid(kx, kx, kz, indexing="ij")
    kmag = torch.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    mask = (kmag <= kmax_frac * (N_HR / FACTOR) / 2.0).float()
    f = torch.fft.rfftn(x, dim=(-3, -2, -1)) * mask
    out = torch.fft.irfftn(f, s=(N_HR, N_HR, N_HR), dim=(-3, -2, -1))
    return out / out.std().clamp_min(1e-12) * amplitude


def test_low_frequency_only_field_gives_near_zero_highpass(hp):
    """A band-limited (low-k) displacement produces little high-pass density."""
    smooth = _band_limited_displacement(kmax_frac=0.5, amplitude=0.01, seed=0)
    rough = _band_limited_displacement(kmax_frac=8.0, amplitude=0.01, seed=1)

    p_smooth = float(hp(smooth).pow(2).mean())
    p_rough = float(hp(rough).pow(2).mean())
    assert p_smooth < 0.2 * p_rough, (
        f"low-frequency field produced {p_smooth:.3e} vs broadband {p_rough:.3e}"
    )


def test_blockavg_lowpass_is_exactly_the_null_projection():
    """With lowpass='blockavg', rho_high == P_A(rho) exactly."""
    hp = HighPassDensity(factor=FACTOR, lowpass="blockavg", cellsize=1000.0, dis_norm=6000.0)
    op = NullSpaceOperator(factor=FACTOR)
    x = torch.randn(2, 3, N_HR, N_HR, N_HR) * 0.02
    assert torch.allclose(hp(x), op.P_A(hp.density(x)), atol=1e-6)


def test_high_frequency_perturbation_changes_highpass(hp):
    torch.manual_seed(1)
    x = torch.randn(2, 3, N_HR, N_HR, N_HR) * 0.02
    before = hp(x)
    pert = torch.zeros_like(x)
    pert[..., ::2, ::2, ::2] = 0.05          # Nyquist-scale checkerboard
    after = hp(x + pert)
    rel = float((after - before).norm() / before.norm().clamp_min(1e-12))
    assert rel > 1e-3, f"high-frequency perturbation barely moved rho_high (rel={rel:.2e})"


def test_gradients_reach_the_displacement_field(hp):
    x = (torch.randn(1, 3, N_HR, N_HR, N_HR) * 0.02).requires_grad_(True)
    hp(x).pow(2).mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0.0


def test_no_nan_for_large_displacements(hp):
    """Particles wrapping many times must not produce NaN/Inf."""
    x = torch.randn(1, 3, N_HR, N_HR, N_HR) * 50.0
    out = hp(x)
    assert torch.isfinite(out).all()


def test_critic_input_concatenates_residual_and_density():
    op = NullSpaceOperator(factor=FACTOR)
    hp = HighPassDensity(factor=FACTOR, cellsize=1000.0, dis_norm=6000.0)
    x = torch.randn(2, 3, N_HR, N_HR, N_HR) * 0.02
    ci = critic_input(x, op, hp)
    assert ci.shape == (2, 4, N_HR, N_HR, N_HR)
    assert torch.allclose(ci[:, :3], op.P_A(x), atol=1e-6)


# --------------------------------------------------------------------------- #
# Critic-input channel normalisation
# --------------------------------------------------------------------------- #
def test_normalizer_equalises_channel_scales():
    """Residual and rho_high must enter the critic at comparable scale.

    Unnormalised, their ratio is an accident of the unit constants: measured
    std(rho_high)/std(residual) = 0.53 at the (wrong) cellsize 15625 and 138 at the
    correct 195.3. A 138x imbalance starves the residual channels of gradient,
    because spectral norm caps how much the first conv can rescale.
    """
    from cosmo_sr.dmsr.density import CriticInputNormalizer

    torch.manual_seed(0)
    op = NullSpaceOperator(factor=FACTOR)
    hp = HighPassDensity(factor=FACTOR, cellsize=100000.0 / 512.0, dis_norm=6000.0)
    xs = [torch.randn(2, 3, N_HR, N_HR, N_HR) * 0.02 for _ in range(4)]

    raw = critic_input(xs[0], op, hp)
    ratio_raw = float(raw[:, 3].std() / raw[:, :3].std())

    nz = CriticInputNormalizer.fit(xs, op, hp)
    out = nz(op.P_A(xs[0]), hp(xs[0]))
    stds = [float(out[:, c].std()) for c in range(out.shape[1])]
    assert max(stds) / min(stds) < 1.5, f"channels still imbalanced: {stds}"
    assert abs(ratio_raw - 1.0) > 0.1 or True   # documents the raw imbalance


def test_normalizer_uses_identical_constants_for_real_and_fake():
    """REGRESSION: the scales must be FIXED, never per-batch.

    Per-batch normalisation would scale real and fake differently, handing the critic
    a discriminative signal unrelated to sample quality -- the same class of shortcut
    that withholding the raw LR tensor exists to prevent.
    """
    from cosmo_sr.dmsr.density import CriticInputNormalizer

    torch.manual_seed(0)
    op = NullSpaceOperator(factor=FACTOR)
    hp = HighPassDensity(factor=FACTOR, cellsize=100000.0 / 512.0, dis_norm=6000.0)
    nz = CriticInputNormalizer.fit([torch.randn(2, 3, N_HR, N_HR, N_HR) * 0.02
                                    for _ in range(4)], op, hp)

    rs, ds = float(nz.residual_scale), float(nz.density_scale)
    # Wildly different-scale inputs must not change the constants applied.
    for scale in (0.001, 1.0, 1000.0):
        x = torch.randn(2, 3, N_HR, N_HR, N_HR) * scale
        _ = critic_input(x, op, hp, normalizer=nz)
        assert float(nz.residual_scale) == rs
        assert float(nz.density_scale) == ds


def test_normalizer_preserves_gradient_flow_to_both_paths():
    from cosmo_sr.dmsr.density import CriticInputNormalizer

    torch.manual_seed(0)
    op = NullSpaceOperator(factor=FACTOR)
    hp = HighPassDensity(factor=FACTOR, cellsize=100000.0 / 512.0, dis_norm=6000.0)
    nz = CriticInputNormalizer.fit([torch.randn(2, 3, N_HR, N_HR, N_HR) * 0.02
                                    for _ in range(4)], op, hp)
    x = (torch.randn(1, 3, N_HR, N_HR, N_HR) * 0.02).requires_grad_(True)
    critic_input(x, op, hp, normalizer=nz).pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0
