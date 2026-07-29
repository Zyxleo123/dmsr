"""Held-out catalog statistics and box-level bootstrap intervals."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import synthetic_catalog

from cosmo_sr.reward.heldout import (bootstrap_ci, held_out_metrics,
                                     isolated_vs_sub_counts, subhalo_relative_velocity,
                                     two_halo_correlation)


def test_two_halo_correlation_of_a_poisson_field_is_near_zero():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 100.0, size=(3000, 3))
    cat = synthetic_catalog(hosts=[(p, 1e13) for p in pos], subs_per_host=[0] * len(pos))
    r, xi = two_halo_correlation(cat, 100.0, r_min=2.0, r_max=20.0, n_bins=5)
    assert np.all(np.abs(xi) < 0.3), xi


def test_two_halo_correlation_is_positive_for_a_clustered_field():
    rng = np.random.default_rng(1)
    centres = rng.uniform(0, 100.0, size=(30, 3))
    pos = np.concatenate([c + rng.normal(0, 1.0, size=(100, 3)) for c in centres]) % 100.0
    cat = synthetic_catalog(hosts=[(p, 1e13) for p in pos], subs_per_host=[0] * len(pos))
    r, xi = two_halo_correlation(cat, 100.0, r_min=1.0, r_max=10.0, n_bins=5)
    assert xi[0] > 1.0, xi


def test_relative_velocity_recovers_the_injected_dispersion():
    cat = synthetic_catalog(hosts=[((50.0, 50.0, 50.0), 1e14)], subs_per_host=[500],
                            seed=2)
    v, pdf, mom = subhalo_relative_velocity(cat)
    # conftest draws sub velocities from N(0, 100) per axis, hosts at rest.
    assert mom["n_pairs"] == 500
    assert 150.0 < mom["rms"] < 200.0
    assert np.isfinite(pdf).all()


def test_isolated_and_subhalo_counts_split_the_population():
    cat = synthetic_catalog(hosts=[((10.0, 10.0, 10.0), 1e13),
                                   ((50.0, 50.0, 50.0), 1e13)],
                            subs_per_host=[3, 5])
    c = isolated_vs_sub_counts(cat)
    assert c["n_isolated"] == 2 and c["n_subhalo"] == 8
    assert c["sub_per_host"] == pytest.approx(4.0)
    assert c["subhalo_fraction"] == pytest.approx(0.8)


def test_the_resolution_cut_applies_to_held_out_counts_too():
    cat = synthetic_catalog(hosts=[((10.0, 10.0, 10.0), 1e13)], subs_per_host=[4],
                            sub_num_p=5)
    c = isolated_vs_sub_counts(cat, min_particles=20)
    assert c["n_subhalo"] == 0


def test_held_out_metrics_are_all_present_and_finite():
    cat = synthetic_catalog(
        hosts=[((10.0 * i, 20.0, 30.0), 1e13) for i in range(1, 8)],
        subs_per_host=[2, 3, 1, 4, 0, 2, 5],
    )
    m = held_out_metrics(cat, boxsize_mpc_h=100.0)
    for key in ("host_mass_function", "subhalo_mass_function", "occupation",
                "radial_profile", "one_halo", "two_halo", "relative_velocity",
                "counts"):
        assert key in m
    assert np.isfinite(m["counts"]["sub_per_host"])


def test_an_empty_catalog_does_not_crash_the_held_out_metrics():
    cat = synthetic_catalog(hosts=[], subs_per_host=[])
    m = held_out_metrics(cat, boxsize_mpc_h=100.0)
    assert m["counts"]["n_isolated"] == 0
    assert np.isnan(m["counts"]["sub_per_host"])


def test_bootstrap_resamples_boxes_and_brackets_the_mean():
    v = [10.0, 12.0, 9.0, 11.0, 13.0]
    ci = bootstrap_ci(v, n_boot=500, seed=0)
    assert ci["n"] == 5
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert ci["mean"] == pytest.approx(np.mean(v))


def test_bootstrap_reports_no_interval_for_a_single_box():
    ci = bootstrap_ci([3.0])
    assert ci["n"] == 1 and np.isnan(ci["lo"])


def test_bootstrap_ignores_non_finite_entries():
    ci = bootstrap_ci([1.0, float("nan"), 3.0], n_boot=100, seed=0)
    assert ci["n"] == 2 and ci["mean"] == pytest.approx(2.0)
