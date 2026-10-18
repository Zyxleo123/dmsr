"""Pure helpers behind the placement forensics and the host phase-space profiles.

The verdicts these feed are counts per radial bin, and every one of them is
plausible whatever the bookkeeping does -- so the linking rule, the parent
walk, the bins and the sphere gather are pinned on cases with known answers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import placement_forensics as pf  # noqa: E402
import host_phase_profile as hpp  # noqa: E402


def test_link_structures_centre_inside_sphere_links():
    # a big sphere holding two small ones' centres, plus an isolated object
    pos = np.array([[10.0, 10, 10], [10.2, 10, 10], [10.0, 10.25, 10],
                    [30.0, 30, 30]])
    rad = np.array([0.3, 0.05, 0.05, 0.05])
    lab = pf.link_structures(pos, rad)
    assert lab[0] == lab[1] == lab[2]
    assert lab[3] != lab[0]


def test_link_structures_needs_one_centre_inside_the_other():
    # spheres overlap (0.15 + 0.15 > 0.25) but neither centre is inside the
    # other: NOT linked. The rule is max(r_i, r_j), not r_i + r_j.
    pos = np.array([[5.0, 5, 5], [5.25, 5, 5]])
    lab = pf.link_structures(pos, np.array([0.15, 0.15]))
    assert lab[0] != lab[1]


def test_link_structures_is_periodic():
    pos = np.array([[0.05, 50, 50], [99.95, 50, 50]])
    lab = pf.link_structures(pos, np.array([0.2, 0.01]))
    assert lab[0] == lab[1]


def test_representatives_pick_the_heaviest_member():
    lab = np.array([0, 0, 1, 1, 1, 2])
    w = np.array([5.0, 9.0, 1.0, 3.0, 2.0, 7.0])
    rep = pf.representatives(lab, w)
    assert sorted(rep.tolist()) == [1, 3, 5]


def test_parent_and_top_rows_walk_to_the_root():
    ids = np.array([10, 11, 12, 13, 14])
    par = np.array([-1, 10, 11, 99, -1])      # 12 is a sub-sub; 13's parent missing
    prow = pf.parent_rows(ids, par)
    assert prow.tolist() == [-1, 0, 1, -1, -1]
    top = pf.top_rows(prow)
    assert top.tolist() == [0, 0, 0, 3, 4]


def test_r_bin_matches_the_gate_convention():
    x = np.array([0.0, 0.2499, 0.25, 0.99, 1.0, 4.99, 5.0, 12.0])
    assert pf.r_bin(x).tolist() == [0, 0, 1, 3, 4, 7, 8, 8]


def test_m_bin_edges():
    n = np.array([19, 20, 49, 50, 199, 200, 999, 1000, 10 ** 7])
    assert pf.m_bin(n, pf.SUB_MASS_EDGES).tolist() == [-1, 0, 0, 1, 2, 3, 3, 4, 4]


def test_rvir_from_mass_cluster_scale():
    # Bryan-Norman Delta_c ~ 98.6 at Om 0.2814; 1e14 Msun/h -> ~0.955 Mpc/h,
    # consistent with the measured 1.007 Mpc/h of a ~1.2e14 cluster.
    assert pf.bryan_norman_delta_c(0.2814) == pytest.approx(98.59, abs=0.05)
    assert pf.rvir_from_mass(np.array([1e14]))[0] == pytest.approx(0.955, abs=0.01)


def test_cumulative_counts():
    n = np.array([20, 40, 50, 60, 1000])
    assert pf.cumulative_counts(n, (20, 50, 100)) == [5, 3, 1]


def test_radial_table_counts_and_total():
    x = np.array([0.1, 0.1, 0.6, 1.2, 6.0])
    n = np.array([25, 60, 300, 2000, 45])
    t = pf.radial_table(x, n)
    c = np.asarray(t["counts"])
    assert c.shape == (9, 5)
    assert t["total_by_r"] == [2, 0, 1, 0, 1, 0, 0, 0, 1]
    assert c[0, 0] == 1 and c[0, 1] == 1 and c[4, 4] == 1 and c[8, 0] == 1


def test_cell_index_gathers_every_particle_in_the_sphere_across_the_edge():
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 100, size=(20000, 3))
    ci = hpp.CellIndex(pos, cell=2.0)
    centre, rad = np.array([0.5, 99.0, 50.0]), 3.0
    got = set(ci.gather(centre, rad).tolist())
    d = pf.min_image(pos - centre)
    want = set(np.where(np.linalg.norm(d, axis=1) < rad)[0].tolist())
    assert want <= got


def test_shrinking_centre_finds_an_offset_core():
    rng = np.random.default_rng(2)
    core = rng.normal(0.0, 0.02, size=(3000, 3)) + np.array([0.1, 0.0, 0.0])
    halo = rng.uniform(-1, 1, size=(3000, 3))
    c = hpp.shrinking_centre(np.vstack([core, halo]), r0=0.5, r_min=0.02)
    assert np.linalg.norm(c - np.array([0.1, 0.0, 0.0])) < 0.01


def test_host_profile_uniform_hot_sphere_is_not_clumpy():
    # isotropic Gaussian velocities, uniform density: coldness ~1, no clumps
    rng = np.random.default_rng(3)
    n = 60000
    r = np.cbrt(rng.uniform(0, 1, n)) * 3.3
    u = rng.normal(size=(n, 3))
    d = u / np.linalg.norm(u, axis=1, keepdims=True) * r[:, None]
    v = rng.normal(0, 300.0, size=(n, 3))
    prof = hpp.host_profile(d, v, rvir=1.0, vvir=500.0, mean_n=1.0,
                            rng=np.random.default_rng(0), max_query_per_bin=2000)
    ok = np.isfinite(prof["coldness"]) & (prof["n"] > 500)
    assert np.all(np.abs(prof["coldness"][ok] - 1.0) < 0.15)
    assert np.nanmax(prof["f_clump10"]) < 0.01
    assert np.nanmax(prof["f_coldclump"]) < 0.01


def test_host_profile_sees_a_cold_dense_clump():
    rng = np.random.default_rng(4)
    n = 60000
    r = np.cbrt(rng.uniform(0, 1, n)) * 3.3
    u = rng.normal(size=(n, 3))
    d = u / np.linalg.norm(u, axis=1, keepdims=True) * r[:, None]
    v = rng.normal(0, 300.0, size=(n, 3))
    clump = rng.normal(0, 0.01, size=(800, 3)) + np.array([0.6, 0.0, 0.0])
    d = np.vstack([d, clump])
    v = np.vstack([v, rng.normal(0, 10.0, size=(800, 3))])
    prof = hpp.host_profile(d, v, rvir=1.0, vvir=500.0, mean_n=1.0,
                            rng=np.random.default_rng(0), max_query_per_bin=100000)
    i = np.digitize([0.6], hpp.EDGES_X)[0] - 1        # the 0.5-0.75 bin
    assert prof["f_coldclump"][i] > 0.05
    others = [j for j in range(len(prof["n"])) if j != i and prof["n"][j] > 500]
    assert np.nanmax(prof["f_coldclump"][others]) < 0.01


def _hot_sphere(n=60000, seed=5):
    rng = np.random.default_rng(seed)
    r = np.cbrt(rng.uniform(0, 1, n)) * 3.3
    u = rng.normal(size=(n, 3))
    d = u / np.linalg.norm(u, axis=1, keepdims=True) * r[:, None]
    return d, rng.normal(0, 300.0, size=(n, 3)), rng


def test_peak_profile_counts_planted_clumps_and_reads_their_shape():
    d, v, rng = _hot_sphere()
    centres = [np.array([1.2, 0.0, 0.0]), np.array([0.0, 1.3, 0.0]),
               np.array([0.0, 0.0, -1.25])]
    for c in centres:                                   # three round cold clumps
        d = np.vstack([d, rng.normal(0, 0.01, size=(300, 3)) + c])
        v = np.vstack([v, rng.normal(0, 5.0, size=(300, 3))])
    out = hpp.peak_profile(d, v, rvir=1.0, m_p=5.8e8)
    i = np.digitize([1.25], hpp.EDGES_X)[0] - 1          # the 1.25-1.5 bin
    j = np.digitize([1.1], hpp.EDGES_X)[0] - 1           # the 1-1.25 bin
    assert out["n_peaks10"][i] + out["n_peaks10"][j] >= 3
    assert out["n_peaks10"][i] + out["n_peaks10"][j] <= 6
    assert out["peak_c_over_a"][j] > 0.5                  # round
    assert out["peak_tu"][j] < 1.0                        # cold + compact: bound-like
    far = [k for k in range(len(out["n_peaks10"])) if k not in (i, j)]
    assert np.nansum(out["n_peaks10"][far]) == 0          # smooth hot halo: none


def test_peak_profile_reads_a_sheet_as_flat():
    d, v, rng = _hot_sphere(seed=6)
    sheet = np.column_stack([rng.uniform(-0.15, 0.15, 3000), rng.uniform(-0.15, 0.15, 3000),
                             rng.normal(0, 0.002, 3000)]) + np.array([0.0, 0.0, 1.1])
    d = np.vstack([d, sheet])
    v = np.vstack([v, rng.normal(0, 5.0, size=(3000, 3))])
    out = hpp.peak_profile(d, v, rvir=1.0, m_p=5.8e8)
    j = np.digitize([1.1], hpp.EDGES_X)[0] - 1
    assert out["n_peaks10"][j] >= 1
    assert out["peak_c_over_a"][j] < 0.3                   # flat


def test_peak_lagrangian_spread_and_kinematic_distinctness():
    d, v, rng = _hot_sphere(seed=7)
    n0 = d.shape[0]
    ids = rng.choice(np.arange(10 ** 6, 10 ** 6 + 10 ** 7), size=n0, replace=False)
    # A: one contiguous Lagrangian patch, moving at +800 km/s through the halo
    ii, jj, kk = np.meshgrid(np.arange(100, 107), np.arange(100, 107),
                             np.arange(100, 107), indexing="ij")
    patch = np.ravel_multi_index((ii.ravel(), jj.ravel(), kk.ravel()), (512,) * 3)[:300]
    d = np.vstack([d, rng.normal(0, 0.01, (300, 3)) + np.array([1.2, 0.0, 0.0])])
    v = np.vstack([v, rng.normal(0, 5.0, (300, 3)) + np.array([800.0, 0.0, 0.0])])
    ids = np.concatenate([ids, patch])
    # B: assembled from random lattice sites, co-moving with the background
    d = np.vstack([d, rng.normal(0, 0.01, (300, 3)) + np.array([0.0, 0.0, -1.2])])
    v = np.vstack([v, rng.normal(0, 5.0, (300, 3))])
    ids = np.concatenate([ids, rng.choice(512 ** 3, 300, replace=False)])
    out = hpp.peak_profile(d, v, rvir=1.0, m_p=5.8e8, ids=ids, records=True)
    rec = out["_records"]
    rows = rec["row"].astype(int)
    a = (rows >= n0) & (rows < n0 + 300)
    b = (rows >= n0 + 300) & (rows < n0 + 600)
    assert a.any() and b.any()
    assert np.max(rec["lag"][a]) < 6.0            # a 7^3 patch: ~3.5 cells rms
    assert np.min(rec["lag"][b]) > 50.0           # scattered over the lattice
    assert np.min(rec["dvbg"][a]) > 1.0           # moves apart from its surroundings
    assert np.max(rec["dvbg"][b]) < 0.5           # a knot in the flow
    assert np.max(rec["coldbg"][a]) < 0.1         # far colder than its surroundings


def test_peak_halo_match_radius_nearest_and_periodic():
    import peak_halo_match as phm
    peaks = np.array([[10.0, 10, 10], [50.0, 50, 50], [99.99, 5.0, 5.0], [70.0, 70, 70]])
    objs = np.array([[10.02, 10, 10], [10.2, 10, 10], [50.5, 50, 50], [0.01, 5.0, 5.0]])
    rad = np.array([0.05, 0.5, 0.1, 0.05])
    m = phm.match(peaks, objs, rad, min_match=0.03)
    assert m[0] == 0          # inside both spheres: the nearest wins
    assert m[1] == -1         # 0.5 away, radius 0.1
    assert m[2] == 3          # across the periodic boundary
    assert m[3] == -1         # nothing near


def test_bound_count_keeps_a_cold_clump_and_drops_a_hot_one():
    rng = np.random.default_rng(8)
    pos = rng.normal(0, 0.01, (200, 3))
    cold = rng.normal(0, 5.0, (200, 3))
    hot = rng.normal(0, 3000.0, (200, 3))
    assert hpp.bound_count(pos, cold, m_p=5.8e8) >= 190
    assert hpp.bound_count(pos, hot, m_p=5.8e8) <= 20


def test_bound_count_ignores_a_fast_background():
    rng = np.random.default_rng(9)
    pos = np.vstack([rng.normal(0, 0.01, (100, 3)), rng.uniform(-0.2, 0.2, (400, 3))])
    vel = np.vstack([rng.normal(0, 5.0, (100, 3)), rng.normal(0, 800.0, (400, 3))])
    nb = hpp.bound_count(pos, vel, m_p=5.8e8)
    assert 80 <= nb <= 140


def test_shuffle_directions_keeps_radii_and_erases_a_clump():
    import bump_cutouts as bc
    d, v, rng = _hot_sphere(seed=12)
    clump = rng.normal(0, 0.01, size=(800, 3)) + np.array([0.7, 0.0, 0.0])
    d = np.vstack([d, clump])
    v = np.vstack([v, rng.normal(0, 5.0, size=(800, 3))])
    ds = bc.shuffle_directions(d, 1.0, 0.35, 1.15, np.random.default_rng(0))
    r0, r1 = np.linalg.norm(d, axis=1), np.linalg.norm(ds, axis=1)
    assert np.allclose(r0, r1)
    out = (r0 < 0.35) | (r0 >= 1.15)
    assert np.array_equal(ds[out], d[out])                # outside the shell untouched
    before = hpp.peak_profile(d, v, 1.0, 5.8e8, min_ratio=10.0)
    after = hpp.peak_profile(ds, v, 1.0, 5.8e8, min_ratio=10.0)
    i = np.digitize([0.7], hpp.EDGES_X)[0] - 1
    assert before["n_peaks10"][i] >= 1 and after["n_peaks10"][i] == 0
