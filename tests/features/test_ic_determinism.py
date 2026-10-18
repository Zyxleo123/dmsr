"""Pinned cases for the IC-determinism test.

Its answer is a match rate against a null and a correlation against a control,
and every one of those numbers looks plausible whatever the bookkeeping does --
so the spectra, the host-relative matching, the periodic IC-patch centroid, the
unit handling and the pre-registered verdict rule are pinned on known answers.
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

import ic_determinism as icd  # noqa: E402

HOSTS = np.array([[20.0, 20, 20], [20, 70, 70], [70, 20, 70], [70, 70, 20]])


def _cat(pos, npart, rvir_mpc, parent, ratio=1.0):
    n = len(pos)
    return {"id": np.arange(n) + 1, "parent": np.asarray(parent),
            "pos": icd.wrap(np.asarray(pos, float)), "rvir": np.asarray(rvir_mpc, float),
            "np": np.asarray(npart, float), "np_ratio": ratio, "path": "synthetic"}


def _system(rng, n_obj=40, rv_host=0.8):
    """Four well-separated hosts, each with n_obj objects at 0.15-2.8 R_vir."""
    pos, npart, rv, par = list(HOSTS), [30000.0] * 4, [rv_host] * 4, [-1] * 4
    for h in HOSTS:
        u = rng.normal(size=(n_obj, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        r = rng.uniform(0.15, 2.8, n_obj) * rv_host
        pos.extend(h + r[:, None] * u)
        npart.extend(rng.uniform(400, 3000, n_obj))
        rv.extend([0.1] * n_obj)
        par.extend([-1] * n_obj)
    return _cat(np.array(pos), npart, rv, par)


def _selection(hr):
    hosts = icd.massive_hosts(hr, 20000)
    rows = np.setdiff1d(np.arange(hr["np"].size), hosts)
    return (hosts,) + icd.assign_to_hosts(hr, hosts, rows, 0.1, 3.0)


# --------------------------------------------------------------- spectra

def test_block_mean_and_corner():
    a = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    m = icd.block_mean(a, 2)
    assert m.shape == (2, 2, 2)
    assert m[1, 0, 1] == pytest.approx(a[2:4, 0:2, 2:4].mean())
    assert icd.corner(a, 2)[1, 1, 1] == a[2, 2, 2]
    with pytest.raises(ValueError):
        icd.block_mean(np.zeros((3, 3, 3)), 2)


def test_cross_r_identical_independent_and_half_shared():
    rng = np.random.default_rng(0)
    n = 32
    f = [rng.normal(size=(n, n, n)) for _ in range(3)]
    g = [rng.normal(size=(n, n, n)) for _ in range(3)]
    edges = icd.k_edges(n, 8)
    r_same, nm = icd.cross_r(f, [3.0 * x + 5.0 for x in f], edges)  # amplitude, offset irrelevant
    assert np.allclose(r_same[nm > 0], 1.0)
    r_ind, _ = icd.cross_r(f, g, edges)
    assert np.all(np.abs(r_ind[nm > 100]) < 0.2)
    r_half, _ = icd.cross_r(f, [x + y for x, y in zip(f, g)], edges)
    assert icd.band_mean(r_half, nm, edges, 0, np.inf) == pytest.approx(2 ** -0.5, abs=0.03)


def test_every_mode_but_k0_is_counted_once():
    n = 16
    _, nm = icd.cross_r([np.ones((n, n, n))], [np.ones((n, n, n))],
                        np.array([0.999 * 2 * np.pi / icd.BOX, 1e9]))
    assert nm.sum() == pytest.approx(n ** 3 - 1)


# ------------------------------------------------------------- positions

def test_particle_positions_raw_kpc_and_catnorm_units(tmp_path):
    ng = 4
    cell = icd.BOX / ng
    d = tmp_path / "dmo"
    d.mkdir()
    disp = np.zeros((3, ng, ng, ng), np.float32)
    disp[0, 1, 2, 3] = 500.0                                   # kpc/h
    np.save(d / "disp.npy", disp)
    pos, n = icd.particle_positions(str(d), "raw")
    assert n == ng
    flat = np.ravel_multi_index((1, 2, 3), (ng, ng, ng))
    assert pos[flat] == pytest.approx([1.5 * cell + 0.5, 2.5 * cell, 3.5 * cell])
    f = np.zeros((6, ng, ng, ng), np.float32)
    f[2, 0, 0, 0] = 1.0 / 6.0                                  # catnorm unit = 6000 kpc/h
    f[0, 0, 0, 1] = -3.0                                       # -18 Mpc/h wraps the box
    np.save(tmp_path / "f.npy", f)
    pos2, _ = icd.particle_positions(str(tmp_path / "f.npy"), "catnorm")
    assert pos2[0] == pytest.approx([0.5 * cell, 0.5 * cell, 0.5 * cell + 1.0])
    assert pos2[1, 0] == pytest.approx(0.5 * cell - 18.0 + icd.BOX)
    assert np.all((pos2 >= 0) & (pos2 < icd.BOX))


def test_lagrangian_centroid_wraps_the_box():
    ng = 8
    cell = icd.BOX / ng
    ii = np.stack(np.meshgrid(*(np.arange(ng),) * 3, indexing="ij"), -1).reshape(-1, 3)
    pos = (ii + 0.5) * cell
    block = np.all(np.isin(ii, [ng - 1, 0]), axis=1)           # the 2^3 block on the corner
    target = np.array([33.0, 33.0, 33.0])
    pos[block] = target + np.random.default_rng(1).normal(scale=0.01, size=(8, 3))
    tree = icd.particle_tree(icd.wrap(pos))
    q = icd.lagrangian_centroids(tree, ng, target[None], np.array([8]))
    assert np.allclose(icd.min_image(q[0]), 0.0, atol=1e-6)    # IC centre = box corner
    assert icd.lagrangian_radius(np.array([1.0]))[0] == pytest.approx(
        (3 / (4 * np.pi)) ** (1 / 3) * icd.CELL_HR)


# -------------------------------------------------------------- matching

def test_assign_to_hosts_uses_rvir_units():
    pos = np.array([[10.0, 10, 10], [20.0, 10, 10], [14.0, 10, 10]])
    cat = _cat(pos, [3e4, 3e4, 500], [1.0, 4.0, 0.1], [-1, -1, -1])
    rows, who, x = icd.assign_to_hosts(cat, np.array([0, 1]), np.array([2]), 0.0, 10.0)
    # 4 R_vir from host 0 but 1.5 R_vir from host 1: host 1 wins
    assert who[0] == 1 and x[0] == pytest.approx(1.5)


def test_shifted_copy_matches_everything_and_null_stays_low():
    hr = _system(np.random.default_rng(0))
    shift = np.array([0.17, 0.0, 0.0])                          # SR2-like frame offset
    cand = dict(hr, pos=icd.wrap(hr["pos"] + shift))
    hosts, rows, hidx, _ = _selection(hr)
    m = icd.match_catalogs(hr, hosts, rows, hidx, cand, mass_tol=3, match_rvir=1.0,
                           match_min_mpc=0.1, n_null=10, rng=np.random.default_rng(1))
    assert np.allclose(m["offset"], shift, atol=1e-6)
    assert (m["host_match"] >= 0).all()
    assert np.all(cand["id"][m["match"]] == hr["id"][rows])     # each finds its own copy
    assert m["null_frac"].mean() < 0.05
    rx, rm, rnull = icd.reverse_match(
        hr, hosts, cand, m["host_match"], obj_min_np=400, host_min_np=20000,
        r_min=0.1, r_max=3.0, mass_tol=3, match_rvir=1.0, match_min_mpc=0.1,
        n_null=5, rng=np.random.default_rng(2))
    assert rx.size == rows.size and rm.all() and rnull.mean() < 0.05


def test_independent_directions_match_at_chance():
    hr = _system(np.random.default_rng(0))
    cand = _system(np.random.default_rng(99))                  # same hosts, new placements
    hosts, rows, hidx, _ = _selection(hr)
    m = icd.match_catalogs(hr, hosts, rows, hidx, cand, mass_tol=3, match_rvir=1.0,
                           match_min_mpc=0.1, n_null=20, rng=np.random.default_rng(1))
    frac = (m["match"] >= 0).mean()
    assert frac < 0.1
    assert abs(frac - m["null_frac"].mean()) < 0.08


def test_mass_window_rejects_a_wrong_mass_object_at_the_right_place():
    hr = _system(np.random.default_rng(0))
    cand = dict(hr, np=np.where(hr["np"] < 20000, hr["np"] * 10.0, hr["np"]))
    hosts, rows, hidx, _ = _selection(hr)
    m = icd.match_catalogs(hr, hosts, rows, hidx, cand, mass_tol=3, match_rvir=1.0,
                           match_min_mpc=0.1, n_null=2, rng=np.random.default_rng(1))
    assert (m["match"] < 0).all()


# ------------------------------------------------------------ summaries

def test_radial_table_last_bin_includes_its_edge():
    x = np.array([0.1, 0.2, 3.0, 2.5])
    t = icd.radial_table(x, {"match": np.array([1.0, 0.0, 1.0, 1.0])})
    assert t[0]["n"] == 2 and t[0]["match"] == pytest.approx(0.5)
    assert t[-1]["n"] == 2 and t[-1]["match"] == pytest.approx(1.0)
    p = icd.pooled(x, {"match": np.array([1.0, 0.0, 1.0, 1.0])})
    assert p["outskirts_1_3"]["n"] == 2 and p["inner_0.1_0.5"]["n"] == 2


def test_verdict_rule():
    def res(match, null, patch):
        return {"pooled": {"outskirts_1_3": {"n": 100, "match": match, "null": null,
                                             "same_patch": patch}}}
    gate = {"shared": True}
    ctrl = res(0.10, 0.05, 0.0)
    call = lambda g, r: icd.verdict(g, {"dmo": r, "sr2": ctrl}, "dmo", "sr2")["call"]
    assert call(gate, res(0.70, 0.05, 0.60)) == "IC CARRIES PLACEMENT"
    assert call(gate, res(0.12, 0.05, 0.10)) == "IC DOES NOT CARRY PLACEMENT"
    assert call(gate, res(0.70, 0.05, 0.20)) == "INCONCLUSIVE"   # right place, wrong patch
    assert call({"shared": False}, res(0.70, 0.05, 0.60)).startswith("NOT A TEST")
    assert call(None, res(0.70, 0.05, 0.60)).startswith("NOT A TEST")


# ------------------------------------------------------------------ figure

def _fake_outputs(tmp_path):
    import json
    edges = icd.k_edges(256, 32)
    kc = np.sqrt(edges[:-1] * edges[1:])
    rip = {"k_centre": kc.tolist(), "band": [2.5, 6.0], "lr_nyquist": icd.LR_NYQUIST,
           "dmo_nyquist": np.pi * 256 / icd.BOX,
           "r": {"disp": {"dmo_mean": np.exp(-kc / 8).tolist(),
                          "dmo_corner": np.exp(-kc / 6).tolist(),
                          "sr2": np.exp(-(kc / 2.3) ** 2).tolist()}},
           "gate": {"shared": True, "variant": "mean",
                    "reason": "r = 0.62 at 2.5-6.0 h/Mpc vs frozen SR2 0.05"}}
    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 3.0, 2000)
    cands = {}
    for name, p in (("dmo256", 0.7), ("sr2", 0.08)):
        hit = (rng.uniform(size=x.size) < p * np.clip(x, 0.3, 1)).astype(float)
        flags = {"match": hit, "null": np.full(x.size, 0.04),
                 "same_patch": hit * (rng.uniform(size=x.size) < 0.8)}
        cands[name] = {"radial": icd.radial_table(x, flags)}
    match = {"primary": "dmo256", "candidates": cands,
             "verdict": {"call": "IC CARRIES PLACEMENT"}}
    (tmp_path / "rip.json").write_text(json.dumps(rip))
    (tmp_path / "match.json").write_text(json.dumps(match))
    return tmp_path / "rip.json", tmp_path / "match.json"


def test_figure_draws_from_the_jsons_alone(tmp_path, monkeypatch):
    import ic_determinism_plot as plot
    rip, match = _fake_outputs(tmp_path)
    for args, name in (([f"--ripples={rip}", f"--match={match}"], "both.png"),
                       ([f"--ripples={rip}", f"--match={tmp_path / 'none.json'}"], "rip.png")):
        monkeypatch.setattr(sys, "argv", ["plot"] + args + [f"--out={tmp_path / name}"])
        assert plot.main() == 0
        assert (tmp_path / name).stat().st_size > 10_000


# --------------------------------------------------------------- tracking

def test_map_to_grid_finds_the_coarse_cell():
    p = np.ravel_multi_index((2, 3, 5), (8, 8, 8))
    assert icd.map_to_grid(np.array([p]), 8, 4)[0] == np.ravel_multi_index((1, 1, 2), (4, 4, 4))
    assert icd.map_to_grid(np.array([p]), 8, 8)[0] == p
    with pytest.raises(ValueError):
        icd.map_to_grid(np.array([p]), 8, 3)


def test_matter_stats_wraps_and_measures_compactness():
    rng = np.random.default_rng(0)
    tight = icd.wrap(rng.normal(scale=0.01, size=(50, 3)))     # straddles the corner
    loose = 30.0 + rng.normal(scale=1.0, size=(50, 3))
    pos = np.vstack([tight, loose])
    c, s, f = icd.matter_stats(pos, [np.arange(50), np.arange(50, 100)],
                               np.array([0.1, 0.1]))
    assert np.allclose(icd.min_image(c[0]), 0.0, atol=0.01)
    assert s[0] < 0.05 and f[0] == 1.0
    assert s[1] > 1.0 and f[1] < 0.05


def test_radial_geometry_radius_and_direction():
    centre = np.array([[50.0, 50, 50]] * 2)
    dvec = np.array([[1.0, 0, 0], [1.0, 0, 0]])
    x = centre + np.array([[1.0, 0, 0], [0, -2.0, 0]])         # same place; rotated + farther
    r, cos = icd.radial_geometry(x, centre, dvec, np.array([0.5, 0.5]))
    assert r == pytest.approx([2.0, 4.0]) and cos == pytest.approx([1.0, 0.0])


def test_same_patch_object_found_anywhere_with_mass_and_tolerance():
    cand = _cat(np.zeros((3, 3)), [1000, 1000, 50], [0.1] * 3, [-1] * 3)
    rows_c = np.array([0, 1, 2])
    q_c = np.array([[10.0, 10, 10], [10.5, 10, 10], [20.0, 20, 20]])
    q_t = np.array([[10.1, 10, 10], [20.0, 20, 20], [40.0, 40, 40]])
    out = icd.same_patch_objects(cand, rows_c, q_c, q_t, np.array([0.3, 0.3, 0.3]),
                                 np.array([900.0, 900.0, 900.0]), 1.0, 3.0)
    # nearest patch; right patch but 18x too light; nothing near
    assert list(out) == [0, -1, -1]


def test_shuffled_radius_null_is_low_when_radii_are_spread():
    rng = np.random.default_rng(0)
    x = rng.uniform(1.0, 3.0, 400)
    null = icd.shuffled_within_bins(x, x, x.copy(), 1.05, rng, n_perm=20)
    # identical radii would match exactly; re-paired within a bin they rarely do
    assert np.nanmean(null) < 0.3


def test_track_figure_draws_from_its_json(tmp_path, monkeypatch):
    import json
    import ic_determinism_plot as plot
    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 3.0, 1000)

    def fake(scale):
        return lambda sel: {
            "matter_offset_mpc": 0.1 * scale, "matter_compact": 0.8 / scale,
            "hr_compact": 0.9, "rvir_obj_mpc": 0.12, "patch_found": 0.6 / scale,
            "patch_same_r": 0.7, "patch_same_r_null": 0.3, "patch_same_dir": 0.5,
            "patch_same_dir_null": 0.05}
    track = {"primary": "dmo256", "params": {"radius_tol": 1.25, "dir_cos": 0.9},
             "candidates": {"dmo256": {"radial": icd.per_bin(x, fake(1.0))},
                            "sr2": {"radial": icd.per_bin(x, fake(5.0))}}}
    (tmp_path / "track.json").write_text(json.dumps(track))
    monkeypatch.setattr(sys, "argv", ["plot", f"--track={tmp_path / 'track.json'}",
                                      f"--track-out={tmp_path / 'track.png'}"])
    assert plot.main() == 0
    assert (tmp_path / "track.png").stat().st_size > 10_000


# ---------------------------------------------------------------- scaling

def test_scaling_table_bins_by_coarse_particles_and_radius():
    n = np.array([3.0, 4.0, 60.0, 60.0, 60.0])
    x = np.array([1.5, 2.0, 1.2, 0.5, 2.9])
    t = icd.scaling_table(n, x, {"match": np.array([0.0, 1.0, 1.0, 1.0, 0.0])}, 1.0, 3.0)
    b2 = next(r for r in t if r["n_lo"] == 2)
    b50 = next(r for r in t if r["n_lo"] == 50)
    assert b2["n"] == 2 and b2["match"] == pytest.approx(0.5)
    assert b50["n"] == 2 and b50["match"] == pytest.approx(0.5)   # x=0.5 is out of range


def test_scaling_reads_match_npz_and_figure_draws(tmp_path, monkeypatch):
    import json
    import ic_determinism_plot as plot
    rng = np.random.default_rng(0)
    m = 600
    np_hr = 10 ** rng.uniform(np.log10(400), np.log10(50000), m)
    for tag in ("a", "b"):
        np.savez(tmp_path / f"{tag}.npz", np=np_hr, r_over_rvir=rng.uniform(0.25, 3, m),
                 dmo256_match_id=np.where(rng.uniform(size=m) < 0.6, 1, -1),
                 dmo256_null_frac=np.full(m, 0.01),
                 dmo256_dq_over_rl=np.where(rng.uniform(size=m) < 0.5, 0.2, np.nan),
                 sr2_match_id=np.where(rng.uniform(size=m) < 0.05, 1, -1),
                 sr2_null_frac=np.full(m, 0.005), sr2_dq_over_rl=np.full(m, np.nan))
    out = tmp_path / "scaling.json"
    assert icd.main(["scaling", "--run", "dmo256", str(tmp_path / "a.npz"), "dmo256", "256",
                     "--run", "sr2", str(tmp_path / "b.npz"), "sr2", "64",
                     "--out", str(out)]) == 0
    sc = json.loads(out.read_text())
    assert sc["runs"]["sr2"]["finest_ripple_mpc"] == pytest.approx(2 * icd.BOX / 64)
    # SR2's sizes are counted in LR particles: 512x fewer than HR's
    tot = sum(r["n"] for r in sc["runs"]["sr2"]["outskirts_1_3"])
    assert tot > 0
    monkeypatch.setattr(sys, "argv", ["plot", f"--scaling={out}",
                                      f"--scaling-out={tmp_path / 'scaling.png'}"])
    assert plot.main() == 0 and (tmp_path / "scaling.png").stat().st_size > 10_000


def test_run_labels_follow_the_resolution():
    import ic_determinism_plot as plot
    assert plot._label("dmo128") == "128³ run" and plot._label("sr2") == "frozen SR2"
