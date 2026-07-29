"""Part 3/4 -- FFT diagnostic correctness and the imbalance decision gate.

The FFT-diagnostic tests verify the properties the Part-5 spectral loss will later
depend on: exact Parseval under ``norm='ortho'``, that unit spectral weights reduce
to spatial MSE, correct shell assignment and mode counts, consistent Hermitian
counting, and that the diagnostic never back-propagates. The gate tests drive
synthetic diagnostics with known structure through :func:`gate_decision`.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmo_sr.dmsr.fourier_diag import (
    BandDiagnosticAccumulator,
    GateThresholds,
    ShellDiagnostic,
    band_of_shells,
    fft_ortho,
    gate_decision,
    parseval_residual,
    shell_index,
)
from cosmo_sr.dmsr.operator import NullSpaceOperator


# --------------------------------------------------------------------------- #
# FFT correctness
# --------------------------------------------------------------------------- #
def test_fft_ifft_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8, 8)
    X = fft_ortho(x)
    recon = torch.fft.ifftn(X, dim=(-3, -2, -1), norm="ortho").real
    assert torch.allclose(recon, x, atol=1e-5)


def test_parseval_equality_under_ortho_norm():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 16, 16, 16)
    assert parseval_residual(x) < 1e-5


def test_unit_weights_reproduce_spatial_mse():
    """mean(|dv|^2) == mean(|FFT(dv)|^2): unit spectral weights == spatial MSE."""
    torch.manual_seed(2)
    dv = torch.randn(3, 3, 12, 12, 12)
    spatial = float(dv.pow(2).mean())
    DV = fft_ortho(dv)
    spectral = float((DV.real ** 2 + DV.imag ** 2).mean())
    assert abs(spatial - spectral) / spatial < 1e-5


def test_single_mode_lands_in_correct_shell():
    n = 16
    kx, ky, kz = 3, 0, 0                      # |k| = 3 -> shell 3
    ax = torch.arange(n, dtype=torch.float32)
    X, Y, Z = torch.meshgrid(ax, ax, ax, indexing="ij")
    field = torch.cos(2 * torch.pi * (kx * X + ky * Y + kz * Z) / n).view(1, 1, n, n, n)
    P = (fft_ortho(field).abs() ** 2)[0, 0]
    idx, n_modes, n_shells = shell_index(n, field.device)
    shell_energy = torch.zeros(n_shells).index_add_(0, idx.reshape(-1), P.reshape(-1))
    assert int(shell_energy.argmax()) == 3
    # essentially all energy in shell 3 (mode + its conjugate)
    assert float(shell_energy[3] / shell_energy.sum()) > 0.999


def test_shell_mode_counts_small_grid():
    for n in (4, 6, 8):
        idx, n_modes, n_shells = shell_index(n, torch.device("cpu"))
        # every voxel-mode is assigned to exactly one shell
        assert float(n_modes.sum()) == n ** 3
        # DC (|k|=0) is a single mode in shell 0
        assert int(n_modes[0]) == 1
        # independent numpy recomputation of round(|k|) histogram must match exactly
        f = np.fft.fftfreq(n) * n
        KX, KY, KZ = np.meshgrid(f, f, f, indexing="ij")
        kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
        ref = np.bincount(np.rint(kmag).astype(int).clip(max=n_shells - 1).ravel(),
                          minlength=n_shells)
        assert np.array_equal(n_modes.numpy().astype(int), ref)


def test_per_mode_times_count_equals_shell_total():
    torch.manual_seed(3)
    op = NullSpaceOperator(factor=4)
    vt = op.P_A(torch.randn(2, 3, 16, 16, 16))
    vp = op.P_A(torch.randn(2, 3, 16, 16, 16))
    acc = BandDiagnosticAccumulator(factor=4)
    acc.add(vp, vt, op)
    d = acc.result()
    recon = d.error_power_per_mode * np.maximum(d.n_modes[None, :], 1.0)
    assert np.allclose(recon, d.error_shell_energy, rtol=1e-4, atol=1e-8)


def test_hermitian_counting_is_consistent():
    """A mode and its conjugate share a shell (round(|k|) is even under negation)."""
    n = 8
    idx, _, _ = shell_index(n, torch.device("cpu"))
    for q in [(1, 2, 3), (0, 4, 1), (3, 3, 3)]:
        neg = tuple((-c) % n for c in q)
        assert int(idx[q]) == int(idx[neg])


def test_dc_and_nyquist_are_handled_explicitly():
    n = 8
    idx, n_modes, n_shells = shell_index(n, torch.device("cpu"))
    assert int(idx[0, 0, 0]) == 0                        # DC -> shell 0
    # Nyquist along one axis: fftfreq index n/2 -> |k| = n/2 = 4 -> shell 4
    assert int(idx[n // 2, 0, 0]) == n // 2


def test_diagnostic_does_not_backpropagate_or_mutate():
    op = NullSpaceOperator(factor=4)
    vt = torch.randn(1, 3, 16, 16, 16)
    vp = torch.randn(1, 3, 16, 16, 16, requires_grad=True)
    before = vp.detach().clone()
    acc = BandDiagnosticAccumulator(factor=4)
    acc.add(vp, vt, op)
    _ = acc.result()
    assert vp.grad is None, "diagnostic created gradients"
    assert torch.equal(vp.detach(), before), "diagnostic mutated its input"


def test_band_assignment_matches_evaluation_semantics():
    # 64^3 grid, factor 8 -> k_LR = 4. low <= 2, transition (2,6], high > 6.
    shell_k = np.arange(33, dtype=np.float64)            # HR Nyquist 32
    bands = band_of_shells(shell_k, factor=8, low_frac=0.5, high_frac=1.5)
    assert bands[0] == "low" and bands[2] == "low"
    assert bands[3] == "transition" and bands[6] == "transition"
    assert bands[7] == "high" and bands[32] == "high"


# --------------------------------------------------------------------------- #
# Gate decision
# --------------------------------------------------------------------------- #
def _fake_diag(err_low, err_trans, err_high, tgt_low, tgt_trans, tgt_high,
               n_shells=33, factor=8):
    """Construct a ShellDiagnostic from per-BAND error/target energy totals, each
    spread uniformly across the shells of its band. This controls concentration
    (c1), relative error (c2), gradient share (c3) and target power (c5) directly."""
    shell_k = np.arange(n_shells, dtype=np.float64)
    bands = band_of_shells(shell_k, factor, 0.5, 1.5)
    counts = {b: bands.count(b) for b in ("low", "transition", "high")}
    band_err = {"low": err_low, "transition": err_trans, "high": err_high}
    band_tgt = {"low": tgt_low, "transition": tgt_trans, "high": tgt_high}
    err = np.zeros((1, n_shells))
    tgt = np.zeros((1, n_shells))
    for j, b in enumerate(bands):
        err[0, j] = band_err[b] / counts[b]
        tgt[0, j] = band_tgt[b] / counts[b]
    return ShellDiagnostic(
        factor=factor, grid=64, n_samples=1, n_batches_channels=1, flow_time_bin="all",
        channel_names=["ch0"], shell_k=shell_k, band_of_shell=bands, n_modes=np.ones(n_shells),
        target_power_per_mode=tgt, target_shell_energy=tgt,
        error_power_per_mode=err, error_shell_energy=err,
        loss_fraction=err / err.sum(), relative_shell_error=err / (tgt + 1e-30),
        shell_cosine=np.zeros((1, n_shells)),
    )


def test_gate_no_imbalance_when_balanced():
    # relative error ~uniform across bands, loss spread out -> no imbalance
    diags = [_fake_diag(err_low=0.4, err_trans=0.3, err_high=0.3,
                        tgt_low=1.0, tgt_trans=1.0, tgt_high=1.0) for _ in range(3)]
    rep = gate_decision(diags)
    assert rep.verdict == "NO_IMBALANCE", rep.human_summary()


def test_gate_supported_when_high_k_starved_with_power():
    # low-k learned well (large target, tiny relative error, most of the loss);
    # transition/high poorly learned (much larger relative error) yet carry little
    # loss share -- and they hold real target power. The imbalance signature.
    diags = [_fake_diag(err_low=0.90, err_trans=0.06, err_high=0.04,
                        tgt_low=20.0, tgt_trans=0.05, tgt_high=0.05) for _ in range(3)]
    rep = gate_decision(diags)
    assert rep.verdict == "IMBALANCE_SUPPORTED", rep.human_summary()
    assert all(rep.criteria.values())


def test_gate_inconclusive_when_artifact():
    # identical error shape, but transition/high target power is ~zero: the large
    # relative error is an artifact of dividing by ~0, not a learnable imbalance.
    diags = [_fake_diag(err_low=0.90, err_trans=0.06, err_high=0.04,
                        tgt_low=20.0, tgt_trans=1e-6, tgt_high=1e-6) for _ in range(3)]
    rep = gate_decision(diags)
    assert rep.verdict == "INCONCLUSIVE", rep.human_summary()
    assert not rep.criteria["c5_not_artifact"]


def test_gate_single_diagnostic_is_not_supported():
    diags = [_fake_diag(err_low=0.90, err_trans=0.06, err_high=0.04,
                        tgt_low=20.0, tgt_trans=0.05, tgt_high=0.05)]
    rep = gate_decision(diags)
    assert rep.verdict != "IMBALANCE_SUPPORTED"
    assert any("stability" in n or "one diagnostic" in n for n in rep.notes)


def test_gate_empty_is_inconclusive():
    assert gate_decision([]).verdict == "INCONCLUSIVE"
