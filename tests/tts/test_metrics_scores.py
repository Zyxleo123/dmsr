"""Stage 1: per-candidate metrics, composite oracles, best-of-K statistics."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmo_sr.tts.bootstrap import best_of_k, bootstrap_ci, paired_bootstrap, subset_draws
from cosmo_sr.tts.metrics import (
    DensityGeometry,
    MomentAccumulator,
    boundary_discontinuity,
    candidate_metrics,
    cic_density_slabs,
    density_profile,
)
from cosmo_sr.tts.scores import (
    PHASE_ORACLE_COMPONENTS,
    STATISTICAL_ORACLE_COMPONENTS,
    ScoreNormalizer,
    composite_score,
    derive_metrics,
    oracle_selection,
)

GEO = DensityGeometry(boxsize=100000.0, ng=16, dis_norm=6000.0)


# --------------------------------------------------------------------------- #
# CIC density
# --------------------------------------------------------------------------- #
def test_zero_displacement_gives_a_uniform_density():
    d = cic_density_slabs(torch.zeros(1, 3, 8, 8, 8), GEO.cellsize, GEO.dis_norm, slab=4)
    assert d.shape == (1, 1, 8, 8, 8)
    assert float(d.abs().max()) < 1e-5


def test_matches_the_reference_cic_implementation():
    from cosmo_sr.eval.density import cic_density

    disp = torch.randn(1, 3, 8, 8, 8) * 0.02
    a = cic_density_slabs(disp, GEO.cellsize, GEO.dis_norm, slab=3)
    b = cic_density(disp, GEO.cellsize, GEO.dis_norm)
    assert torch.allclose(a, b, atol=1e-4), float((a - b).abs().max())


def test_density_uses_all_three_displacement_components():
    """Not channel 0 alone: perturbing any component must move the density."""
    base = torch.zeros(1, 3, 8, 8, 8)
    d0 = cic_density_slabs(base, GEO.cellsize, GEO.dis_norm, slab=4)
    for c in range(3):
        disp = base.clone()
        disp[0, c] = torch.randn(8, 8, 8) * 0.05
        d = cic_density_slabs(disp, GEO.cellsize, GEO.dis_norm, slab=4)
        assert float((d - d0).abs().max()) > 1e-3, c


def test_converging_flow_raises_the_density_contrast():
    """Displacements pointing inward must clump material."""
    n = 16
    q = torch.arange(n, dtype=torch.float32) + 0.5
    grid = torch.stack(torch.meshgrid(q, q, q, indexing="ij"))
    centre = n / 2.0
    toward = (centre - grid) * 0.25 * (GEO.cellsize / GEO.dis_norm)
    flat = cic_density_slabs(torch.zeros(1, 3, n, n, n), GEO.cellsize, GEO.dis_norm)
    clumped = cic_density_slabs(toward.unsqueeze(0), GEO.cellsize, GEO.dis_norm)
    assert float(clumped.std()) > float(flat.std()) + 0.1


# --------------------------------------------------------------------------- #
# Field diagnostics
# --------------------------------------------------------------------------- #
def test_boundary_discontinuity_detects_an_injected_seam():
    """A per-tile offset -- what independent tile noise actually looks like."""
    torch.manual_seed(0)
    n, tile = 32, 8
    smooth = torch.nn.functional.avg_pool3d(
        torch.randn(1, 6, n, n, n), 3, stride=1, padding=1
    )
    clean = boundary_discontinuity(smooth, tile)
    assert clean == pytest.approx(1.0, abs=0.35), clean

    nt = n // tile
    offsets = torch.randn(1, 1, nt, nt, nt) * 3.0
    blocky = smooth + offsets.repeat_interleave(tile, 2).repeat_interleave(
        tile, 3).repeat_interleave(tile, 4)
    assert boundary_discontinuity(blocky, tile) > 5 * clean


def test_moment_accumulator_matches_the_direct_computation():
    torch.manual_seed(1)
    xs = [torch.randn(1, 2, 4, 4, 4) for _ in range(5)]
    acc = MomentAccumulator()
    for x in xs:
        acc.add(x)
    got = acc.summary()
    stack = torch.stack(xs)
    var = stack.var(dim=0, unbiased=True).mean()
    rms = stack.pow(2).mean().sqrt()
    assert got["diversity"] == pytest.approx(float(var.sqrt() / rms), rel=1e-4)
    assert got["pairwise_rms"] == pytest.approx(float(np.sqrt(2)) * got["diversity"], rel=1e-6)


def test_candidate_metrics_returns_the_documented_groups():
    torch.manual_seed(2)
    n = 16
    hr = torch.randn(1, 6, n, n, n) * 0.05
    sr = hr + torch.randn(1, 6, n, n, n) * 0.01
    lr = torch.nn.functional.avg_pool3d(hr, 8)
    m = candidate_metrics(sr, hr, lr, factor=8, geometry=GEO, n_bins=6, tile_size=8)
    for key in ("disp_rk_high", "density_rk_high", "density_power_error", "density_pdf_error",
                "bispectrum_equilateral_error", "bispectrum_squeezed_error",
                "velocity_power_error", "velocity_divergence_pdf_error",
                "lr_recon_rel_disp", "boundary_ratio"):
        assert key in m and np.isfinite(m[key]), key


def test_metrics_without_hr_only_return_test_time_quantities():
    sr = torch.randn(1, 6, 16, 16, 16) * 0.05
    lr = torch.nn.functional.avg_pool3d(sr, 8)
    m = candidate_metrics(sr, None, lr, factor=8, geometry=GEO, n_bins=6)
    assert "lr_recon_mse" in m
    assert not any(k.startswith("disp_rk") for k in m)


def test_density_profile_histogram_is_normalised():
    rho = cic_density_slabs(torch.randn(1, 3, 8, 8, 8) * 0.02, GEO.cellsize, GEO.dis_norm)
    prof = density_profile(rho, n_bins=6, pdf_bins=20)
    assert prof["log_density_pdf"].sum() == pytest.approx(1.0, abs=1e-5)
    assert prof["density_pk"].shape == (6,)


# --------------------------------------------------------------------------- #
# Composite scores
# --------------------------------------------------------------------------- #
def _rows(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {
            "density_power_error": float(rng.normal(0.1, 0.02)),
            "density_pdf_error": float(rng.normal(0.05, 0.01)),
            "density_sigma_ratio": float(rng.normal(1.0, 0.05)),
            # deliberately a different order of magnitude
            "bispectrum_equilateral_error": float(rng.normal(50.0, 10.0)),
            "bispectrum_squeezed_error": float(rng.normal(2.0, 0.4)),
            "velocity_power_error": float(rng.normal(0.2, 0.03)),
            "velocity_divergence_pdf_error": float(rng.normal(0.3, 0.05)),
            "density_rk_high": float(rng.normal(0.4, 0.05)),
        }
        for _ in range(n)
    ]


def test_normalisation_stops_a_large_scale_metric_from_dominating():
    rows = _rows()
    norm = ScoreNormalizer.fit(rows)
    scores = np.array([composite_score(r, STATISTICAL_ORACLE_COMPONENTS, norm) for r in rows])
    # correlation of the composite with each component should be comparable;
    # without normalisation the bispectrum term (scale 50) would swamp the rest.
    corrs = []
    for c in STATISTICAL_ORACLE_COMPONENTS:
        vals = np.array([derive_metrics(r)[c] for r in rows])
        corrs.append(abs(np.corrcoef(vals, scores)[0, 1]))
    assert max(corrs) / max(min(corrs), 1e-6) < 6.0, corrs


def test_higher_is_better_metrics_are_flipped():
    norm = ScoreNormalizer.fit(_rows())
    good = {"density_rk_high": 0.9}
    bad = {"density_rk_high": 0.1}
    assert norm.z(good, "density_rk_high") < norm.z(bad, "density_rk_high")


def test_a_constant_component_cannot_blow_up_the_composite():
    rows = [{"density_power_error": 0.1, "density_pdf_error": v} for v in np.linspace(0, 1, 10)]
    norm = ScoreNormalizer.fit(rows)
    assert norm.std["density_power_error"] == 1.0
    assert np.isfinite(composite_score(rows[0], ("density_power_error",), norm))


def test_oracle_selection_picks_the_best_row():
    rows = _rows(12, seed=3)
    norm = ScoreNormalizer.fit(rows)
    idx, best, all_scores = oracle_selection(rows, STATISTICAL_ORACLE_COMPONENTS, norm)
    assert best == pytest.approx(min(all_scores))
    assert idx == int(np.argmin(all_scores))


def test_the_two_oracles_use_disjoint_information():
    assert not set(PHASE_ORACLE_COMPONENTS) & set(STATISTICAL_ORACLE_COMPONENTS)
    assert not any("rk" in c or "Tk" in c for c in STATISTICAL_ORACLE_COMPONENTS)


# --------------------------------------------------------------------------- #
# Best-of-K and bootstrap
# --------------------------------------------------------------------------- #
def test_oracle_selection_beats_random_at_every_k():
    rng = np.random.default_rng(0)
    values = rng.normal(size=32)
    for k in (2, 4, 8, 16):
        draws = subset_draws(k, 32, 200, rng)
        oracle, _ = best_of_k(values, values, k, draws=draws)
        random_mean = float(values[draws].mean())
        assert oracle < random_mean


def test_best_of_k_reports_the_value_the_selector_did_not_optimise():
    """A selector minimising an unrelated score must not look like an oracle."""
    rng = np.random.default_rng(1)
    values = rng.normal(size=16)
    unrelated = rng.normal(size=16)
    draws = subset_draws(8, 16, 300, rng)
    mean, _ = best_of_k(values, unrelated, 8, draws=draws)
    assert abs(mean - values[draws].mean()) < 0.4


def test_full_k_has_a_single_subset():
    assert subset_draws(8, 8, 100, np.random.default_rng(0)).shape == (1, 8)


def test_paired_bootstrap_finds_a_real_shift_and_not_a_null_one():
    rng = np.random.default_rng(0)
    base = rng.normal(size=12)
    assert paired_bootstrap(base - 1.0, base, rng=rng)["significant"]
    assert not paired_bootstrap(base + rng.normal(0, 3, size=12), base, rng=rng)["significant"]


def test_bootstrap_ci_brackets_the_mean():
    ci = bootstrap_ci(np.random.default_rng(0).normal(5.0, 1.0, size=20))
    assert ci["lo"] < ci["mean"] < ci["hi"] and ci["n_boxes"] == 20
