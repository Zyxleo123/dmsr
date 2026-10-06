"""The two discriminators have to actually discriminate.

SUBFIND refuses an object for one of two reasons, and the whole forensic rests
on each having a number that moves the right way:

*Boundness.* ``virial_ratio`` must sit near 1 for a clump in equilibrium and run
far above it for one too hot to hold together. If it did not, "the arms' objects
are unbound" would be unfalsifiable.

*Compactness.* ``r_half_over_r`` must separate a centrally peaked object from a
uniform patch, and the uniform value is not a matter of opinion -- it is
``2^(-1/3) = 0.794`` exactly, which the test asserts against.

Both are measured from particles alone, never from a halo finder's columns, so
the tests build the particles and know the answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cosmo_sr.features.subhalo_forensics import (  # noqa: E402
    G_MPC_KMS2_MSUN, UNIFORM_R_HALF_OVER_R, clump_properties, summarize_split,
)

BOX = 20.0
M_P = 5.8e8
CENTRE = np.array([10.0, 10.0, 10.0])


def _uniform_sphere(n: int, r: float, seed: int = 0) -> np.ndarray:
    """``n`` points uniform in VOLUME inside radius ``r`` (so r = R u^(1/3))."""
    rng = np.random.default_rng(seed)
    u = rng.random(n) ** (1.0 / 3.0) * r
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return CENTRE[None, :] + v * u[:, None]


def _velocities(n: int, sigma: float, bulk=(0.0, 0.0, 0.0), seed: int = 1):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=(n, 3)) + np.asarray(bulk)[None, :]


def test_uniform_sphere_hits_the_analytic_half_mass_radius():
    """0.794 is arithmetic, not a fitted constant -- so assert on it."""
    n, r = 20_000, 0.3
    pos = _uniform_sphere(n, r)
    p = clump_properties(pos, _velocities(n, 100.0), CENTRE, r, M_P, BOX)
    assert p["n_in"] == n
    assert p["r_half_over_r"] == pytest.approx(UNIFORM_R_HALF_OVER_R, abs=0.01)
    assert UNIFORM_R_HALF_OVER_R == pytest.approx(0.7937, abs=1e-4)


def test_a_centrally_peaked_clump_reads_far_below_the_uniform_value():
    rng = np.random.default_rng(3)
    n, r = 20_000, 0.3
    # Gaussian core well inside the aperture: a real subhalo's shape.
    pos = CENTRE[None, :] + rng.normal(0.0, 0.06, size=(n, 3))
    p = clump_properties(pos, _velocities(n, 100.0), CENTRE, r, M_P, BOX)
    assert p["r_half_over_r"] < 0.45
    assert p["r_half_over_r"] < UNIFORM_R_HALF_OVER_R - 0.3


def test_virial_ratio_is_near_one_for_a_clump_in_equilibrium():
    """Build the dispersion FROM the mass, then check the ratio recovers it."""
    n, r = 20_000, 0.3
    mass = n * M_P
    v_circ = np.sqrt(G_MPC_KMS2_MSUN * mass / r)
    # sigma per axis = v_circ/sqrt(3) gives a 3-D dispersion of v_circ.
    pos = _uniform_sphere(n, r)
    vel = _velocities(n, v_circ / np.sqrt(3.0))
    p = clump_properties(pos, vel, CENTRE, r, M_P, BOX)
    assert p["v_circ"] == pytest.approx(v_circ, rel=1e-6)
    assert p["virial_ratio"] == pytest.approx(1.0, abs=0.05)


def test_a_hot_clump_reads_far_above_one():
    """The signature of something unbinding would delete."""
    n, r = 20_000, 0.3
    v_circ = np.sqrt(G_MPC_KMS2_MSUN * n * M_P / r)
    pos = _uniform_sphere(n, r)
    p = clump_properties(pos, _velocities(n, 5.0 * v_circ), CENTRE, r, M_P, BOX)
    assert p["virial_ratio"] > 4.0


def test_orbital_motion_does_not_count_against_boundness():
    """A subhalo crossing the host at 1500 km/s is not thereby unbound.

    The dispersion is taken about the clump's own bulk velocity. Subtracting the
    host frame instead would scale the ratio with orbital speed and report every
    fast satellite as unbound -- a systematic that would fall hardest on exactly
    the population under study.
    """
    n, r = 20_000, 0.3
    v_circ = np.sqrt(G_MPC_KMS2_MSUN * n * M_P / r)
    pos = _uniform_sphere(n, r)
    still = clump_properties(pos, _velocities(n, v_circ / np.sqrt(3.0)),
                             CENTRE, r, M_P, BOX)
    moving = clump_properties(
        pos, _velocities(n, v_circ / np.sqrt(3.0), bulk=(1500.0, -800.0, 300.0)),
        CENTRE, r, M_P, BOX)
    assert moving["virial_ratio"] == pytest.approx(still["virial_ratio"], rel=1e-9)


def test_properties_are_periodic():
    """An object on the box face must not be measured as half an object."""
    n, r = 5_000, 0.3
    edge = np.array([0.05, 10.0, 10.0])
    rng = np.random.default_rng(5)
    pos = (edge[None, :] + rng.normal(0.0, 0.06, size=(n, 3))) % BOX
    p = clump_properties(pos, _velocities(n, 100.0), edge, r, M_P, BOX)
    assert p["n_in"] > 0.98 * n


def test_too_few_particles_returns_nan_not_a_number():
    pos = _uniform_sphere(5, 0.3)
    p = clump_properties(pos, _velocities(5, 100.0), CENTRE, 0.3, M_P, BOX,
                         min_particles=10)
    assert not p["usable"]
    assert np.isnan(p["virial_ratio"]) and np.isnan(p["r_half_over_r"])


def test_summarize_splits_and_reports_the_gap():
    rows = [
        {"confirmed": True, "virial_ratio": 1.0, "r_half_over_r": 0.30,
         "num_p": 300},
        {"confirmed": True, "virial_ratio": 1.2, "r_half_over_r": 0.34,
         "num_p": 400},
        {"confirmed": False, "virial_ratio": 4.0, "r_half_over_r": 0.70,
         "num_p": 320},
        {"confirmed": False, "virial_ratio": 6.0, "r_half_over_r": 0.76,
         "num_p": 380},
    ]
    s = summarize_split(rows)
    assert s["n_confirmed"] == 2 and s["n_unconfirmed"] == 2
    assert s["virial_ratio__confirmed"] == pytest.approx(1.1)
    assert s["virial_ratio__unconfirmed"] == pytest.approx(5.0)
    assert s["virial_ratio__ratio"] == pytest.approx(1.1 / 5.0)
    # Matched in mass, so the split cannot be explained by particle count.
    assert s["num_p__confirmed"] == pytest.approx(350.0)
    assert s["num_p__unconfirmed"] == pytest.approx(350.0)


def test_nan_rows_do_not_poison_the_medians():
    rows = [{"confirmed": True, "virial_ratio": 1.0},
            {"confirmed": True, "virial_ratio": float("nan")},
            {"confirmed": False, "virial_ratio": 5.0}]
    s = summarize_split(rows)
    assert s["virial_ratio__confirmed"] == pytest.approx(1.0)
