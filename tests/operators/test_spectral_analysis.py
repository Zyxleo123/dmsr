"""Coverage / rank tests -- reproduces the plan's analytic Gate-1 numbers."""
import numpy as np

from cosmo_sr.operators.spectral_analysis import (
    stacked_rank_1d,
    stacked_eigs_1d,
    coverage_spectrum_3d,
    coverage_summary,
    effective_rank,
)


def test_1d_fixed_rank():
    # width-8 box average of a length-512 signal: 64 measurements.
    assert stacked_rank_1d(512, 8, [0]) == 64


def test_1d_all_subcell_shifts_rank_505():
    # The plan's key analytic prediction: nullity collapses 448 -> 7.
    assert stacked_rank_1d(512, 8, range(8)) == 505


def test_1d_factor2_shifts_rank():
    # factor-2, shifts {0,1}: nullity 256 -> 1.
    assert stacked_rank_1d(512, 2, [0]) == 256
    assert stacked_rank_1d(512, 2, [0, 1]) == 511


def test_3d_coverage_monotone_in_diversity():
    # more shifted axes -> more nominal coverage (never fewer identifiable modes).
    ef = stacked_eigs_1d(64, 8, [0])
    ea = stacked_eigs_1d(64, 8, range(8))
    fixed = coverage_summary(coverage_spectrum_3d([ef, ef, ef]))
    xyz = coverage_summary(coverage_spectrum_3d([ea, ea, ea]))
    assert xyz["rank"] > fixed["rank"]
    assert xyz["nullity"] < fixed["nullity"]
    assert fixed["rank"] == 512  # 8^3 LR measurements


def test_eta_identifiable_split():
    # Cross-case eta-identifiability uses a SHARED single-operator reference
    # (fixed-A max, raw units) so the count is monotone in diversity (C2 bound).
    ef = stacked_eigs_1d(64, 8, [0])       # raw units
    ea = stacked_eigs_1d(64, 8, range(8))  # raw units, same normalization
    fixed = coverage_spectrum_3d([ef, ef, ef])
    xyz = coverage_spectrum_3d([ea, ea, ea])
    ref = fixed.max()  # fully-measured single-operator eigenvalue
    fixed_vel = coverage_summary(fixed, eta_frac=0.60, ref=ref)["eta_identifiable"]
    disp = coverage_summary(xyz, eta_frac=0.0083, ref=ref)["eta_identifiable"]
    vel = coverage_summary(xyz, eta_frac=0.60, ref=ref)["eta_identifiable"]
    assert fixed_vel == 512                 # fixed A: 512 fully-measured modes
    assert vel > fixed_vel                  # diversity only adds identifiable modes
    assert disp > vel                       # displacement floor keeps far more
    assert 80000 < disp < 100000 and 15000 < vel < 22000  # matches Gate-1 report


def test_effective_rank_bounds():
    ef = stacked_eigs_1d(64, 8, [0])
    lam = coverage_spectrum_3d([ef, ef, ef])
    er = effective_rank(lam)
    assert 500 < er < 520  # fixed A: 512 equal nonzero eigenvalues
