"""The velocity-temperature oracle must change exactly what it claims to.

The verdict it feeds is a halo-finder count on a field whose velocities were
rescaled. A scale computed from a statistic other than the target's, or a
mask that leaks into the bins it should leave alone, would still produce a
plausible catalog -- so both are pinned here on toy spheres with known answers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import host_phase_profile as hpp  # noqa: E402
import velocity_profile_oracle as vpo  # noqa: E402


def _sphere(n=40000, seed=0, sigma=300.0):
    rng = np.random.default_rng(seed)
    r = np.cbrt(rng.uniform(0, 1, n)) * 3.3
    u = rng.normal(size=(n, 3))
    d = u / np.linalg.norm(u, axis=1, keepdims=True) * r[:, None]
    v = rng.normal(0, sigma, size=(n, 3))
    return d, v


def test_dispersion_matches_the_profile_estimator():
    from scipy.spatial import cKDTree
    d, v = _sphere()
    tree = cKDTree(d)
    q = np.arange(0, d.shape[0], 7)
    _, sig = vpo.local_mean_and_dispersion(tree, d, v, q)
    prof = hpp.host_profile(d, v, rvir=1.0, vvir=1.0, mean_n=1.0,
                            rng=np.random.default_rng(0), max_query_per_bin=10 ** 6)
    x = np.linalg.norm(d[q], axis=1)
    i = np.digitize([1.1], hpp.EDGES_X)[0] - 1          # the 1-1.25 bin
    m = (x >= hpp.EDGES_X[i]) & (x < hpp.EDGES_X[i + 1])
    # host_profile subtracts the inner mean velocity first: a constant, which a
    # neighbourhood dispersion does not see
    assert abs(np.median(sig[m]) / prof["sigma_loc"][i] - 1.0) < 0.03


def test_bin_scale_modes_and_clipping():
    edges = np.array([0.0, 0.25, 0.5, 1.0, 2.0])
    x = np.repeat([0.1, 0.3, 0.7, 1.5], 100)
    sig = np.repeat([1.0, 1.0, 1.0, 1.0], 100)
    target = np.array([2.0, 2.0, 0.5, 100.0])
    s_all = vpo.bin_scale(x, sig, edges, target, 1.0, "all", 0.5, 0.25, 4.0)
    assert np.allclose(s_all, [2.0, 2.0, 0.5, 4.0])            # last bin clipped
    s_core = vpo.bin_scale(x, sig, edges, target, 1.0, "core", 0.5, 0.25, 4.0)
    assert np.allclose(s_core, [2.0, 2.0, 1.0, 1.0])
    s_out = vpo.bin_scale(x, sig, edges, target, 1.0, "out", 0.5, 0.25, 4.0)
    assert np.allclose(s_out, [1.0, 1.0, 0.5, 4.0])


def test_rescaling_reaches_the_target_dispersion():
    from scipy.spatial import cKDTree
    d, v = _sphere(seed=1)
    tree = cKDTree(d)
    q = np.arange(d.shape[0])
    vloc, sig = vpo.local_mean_and_dispersion(tree, d, v, q)
    edges = hpp.EDGES_X
    x = np.linalg.norm(d, axis=1)
    # cool by 0.7 or heat by 1.6: the oracle's real range. Cooling leaves a floor
    # (the spread of the neighbours' own local means is not rescaled), heating
    # does not, so cooling gets the looser tolerance.
    for factor, tol in ((0.7, 0.12), (1.6, 0.05)):
        target = np.full(edges.size - 1, factor * float(np.median(sig)))
        s = vpo.bin_scale(x, sig, edges, target, 1.0, "all", 0.5, 0.25, 4.0)
        b = np.clip(np.digitize(x, edges) - 1, 0, s.size - 1)
        v2 = vloc + (v - vloc) * s[b][:, None]
        _, sig2 = vpo.local_mean_and_dispersion(tree, d, v2, q[::5])
        xb = np.digitize(x[::5], edges) - 1
        # a uniform toy sphere leaves the inner bins under bin_scale's 50-particle
        # floor (deliberately unscaled); check the bins it does rescale
        full = [i for i in range(edges.size - 1) if np.sum(b == i) >= 200]
        assert len(full) >= 6
        for i in full:
            assert abs(np.median(sig2[xb == i]) / target[i] - 1.0) < tol, (factor, i)


def test_knot_scale_only_cools_and_hits_the_target():
    import knot_velocity_oracle as kvo
    rng = np.random.default_rng(3)
    hot_core = rng.normal(0, 300.0, (64, 3))
    bg = rng.normal(0, 300.0, (400, 3))
    s = kvo.knot_scale(hot_core, bg, 0.3)
    assert 0.2 < s < 0.4                                   # ~0.3 for equal dispersions
    cold_core = rng.normal(0, 30.0, (64, 3))
    assert kvo.knot_scale(cold_core, bg, 0.3) == 1.0       # already colder: untouched


def test_saved_field_round_trips_the_new_velocities():
    from cosmo_sr.eval.particles import field_to_particles
    rng = np.random.default_rng(11)
    field = rng.normal(0, 0.01, size=(6, 8, 8, 8)).astype(np.float32)
    pb = field_to_particles(field, boxsize_kpc_h=100000.0)
    v_new = pb.vel_kms * 0.5 + 7.0
    f2 = vpo.field_with_velocities(field, v_new)
    pb2 = field_to_particles(f2, boxsize_kpc_h=100000.0)
    assert np.allclose(pb2.vel_kms, v_new, rtol=1e-5, atol=1e-3)
    assert np.array_equal(pb2.pos_mpc_h, pb.pos_mpc_h)      # displacement untouched
