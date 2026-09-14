"""Round-2 kNN diagnostic on a 32^3 box: classes, membership, boundary modes.

What must hold for the verdict to be readable: the owner classes are the
catalog's; an identical field reads ratio 1 in every class; and the neighbour
modes on the SAME field recover the full-box distances exactly at a generous
margin -- if they did not, a nonzero error in the real run could be the code.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "features"))

_spec = importlib.util.spec_from_file_location(
    "knn_cdf_split_diagnostic", ROOT / "scripts/features/knn_cdf_split_diagnostic.py")
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)

NG, TILE = 32, 8


def _cat():
    return SimpleNamespace(ids=np.array([0, 1, 2, 3, 7]),
                           parent_ids=np.array([-1, 0, 0, 0, -1]),
                           num_p=np.array([5000, 120, 40, 900, 300]))


def test_classify_matches_the_catalog():
    owner = np.array([-1, 0, 1, 2, 3, 7, 1, -1], dtype=np.int32)
    got = [sd.CLASSES[i] for i in sd.classify(owner, _cat())]
    assert got == ["unbound", "host", "sub_50_200", "sub_lt50", "sub_ge200",
                   "host", "sub_50_200", "unbound"]


def test_classify_refuses_a_foreign_owner_array():
    import pytest
    with pytest.raises(ValueError, match="different runs"):
        sd.classify(np.array([5], dtype=np.int32), _cat())


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(sd.v1, "NG", NG)
    monkeypatch.setattr(sd.v1, "TILE", TILE)
    monkeypatch.setattr(sd.v1, "CELL", 100.0 / NG)
    rng = np.random.default_rng(0)
    paths = {}
    base = (0.3 * rng.standard_normal((6, NG, NG, NG))).astype(np.float32)
    for name, f in (("hr", base), ("frozen", base * 0.5), ("tuned", base * 0.9)):
        paths[name] = tmp_path / f"{name}.npy"
        np.save(paths[name], f)
    owner = rng.integers(-1, 5, size=NG ** 3).astype(np.int32)
    owner[owner == 4] = 7
    np.save(tmp_path / "owner.npy", owner)
    monkeypatch.setattr(sd, "load_rockstar_ascii", lambda p: _cat())
    out = tmp_path / "o.json"
    sd.main(["--fields", f"hr={paths['hr']}", f"frozen={paths['frozen']}",
             f"selfhostguard={paths['tuned']}", f"copy={paths['hr']}",
             "--catalog", "x", "--owner", str(tmp_path / "owner.npy"),
             "--ks", "4", "8", "--n-per-class", "400", "--n-quantiles", "8",
             "--sizes", "0", "1", "6", "--boundary-tiles", "2",
             "--boundary-queries", "64", "--out", str(out)])
    r = json.loads(out.read_text())
    assert set(r["membership"]) == {"hr", "frozen", "selfhostguard", "copy"}
    for c, e in r["membership"]["copy"]["classes"].items():
        assert np.allclose(e["paired_ratio_med"], 1.0)
        assert not e["outside_dead_zone"]
    assert r["membership"]["frozen"]["classes"]["host"]["outside_dead_zone"]
    b = r["boundary"]
    for mode in ("hr_lagrangian", "hr_centroid", "hr_centroid_nbr",
                 "tuned_centroid_nbr", "tuned_centroid_frozen"):
        assert mode in b, mode
    # Same-field neighbours at a 6-cell margin: exactly the full-box answer.
    wide = [x for x in b["hr_centroid_nbr"] if x["size_cells"] == 6.0][0]
    assert wide["incomplete_frac"] == 0.0 and wide["rel_kmax_p95"] < 1e-9
    assert wide["candidate_multiplier"] > 1.0
    # A truncated set can only overestimate.
    for x in b["hr_lagrangian"]:
        assert x["rel_kmax_median"] >= 0.0


def test_loss_forms_separate_a_cancelling_swap():
    """The round-2 finding as a test: clumping moved between classes, pooled
    distribution unchanged. The unpaired form must read ~0; the class-conditioned
    and paired forms must not."""
    rng = np.random.default_rng(3)
    n = 4000
    # HR: class 0 (subhalo-like) tight, class 4 (unbound-like) loose.
    h0 = rng.normal(-3.0, 0.2, size=(n, 2))
    h4 = rng.normal(-1.0, 0.2, size=(n, 2))
    # Candidate: the two classes SWAP values -- same pooled multiset.
    z_str = np.concatenate([h4, h0])
    h_str = np.concatenate([h0, h4])
    slices = {c: slice(0, 0) for c in sd.CLASSES}
    slices["sub_50_200"] = slice(0, n)
    slices["unbound"] = slice(n, 2 * n)
    cls_uni = np.array([0] * n + [4] * n)
    lf = sd.loss_forms(z_str, h_str, cls_uni, z_str, h_str, slices,
                       {"sub_50_200": 1, "unbound": 1}, n_q=32, dead_zone=0.02)
    assert lf["unpaired_total"] < 1e-12
    assert lf["paired_total"] > 1.0
    assert lf["class_cond_mass_total"] > 1.0
    assert lf["class_cond_equal_total"] > 1.0


def test_region_restricts_samples_and_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(sd.v1, "NG", NG)
    monkeypatch.setattr(sd.v1, "TILE", TILE)
    monkeypatch.setattr(sd.v1, "CELL", 100.0 / NG)
    rng = np.random.default_rng(1)
    base = (0.3 * rng.standard_normal((6, NG, NG, NG))).astype(np.float32)
    tuned = base.copy()
    region = [5, 21]
    from cosmo_sr.features.knn_cdf import tile_rows_flat
    n_side = NG // TILE
    for t in region:                      # only the region differs from frozen
        tx, ty, tz = t // n_side ** 2, (t // n_side) % n_side, t % n_side
        tuned[:3, tx * TILE:(tx + 1) * TILE, ty * TILE:(ty + 1) * TILE,
              tz * TILE:(tz + 1) * TILE] *= 1.5
    for name, f in (("hr", base * 1.2), ("frozen", base), ("tuned", tuned)):
        np.save(tmp_path / f"{name}.npy", f)
    owner = rng.integers(-1, 5, size=NG ** 3).astype(np.int32)
    owner[owner == 4] = 7
    np.save(tmp_path / "owner.npy", owner)
    (tmp_path / "field.json").write_text(json.dumps({"tiles": region}))
    monkeypatch.setattr(sd, "load_rockstar_ascii", lambda p: _cat())
    out = tmp_path / "o.json"
    sd.main(["--fields", f"hr={tmp_path / 'hr.npy'}",
             f"frozen={tmp_path / 'frozen.npy'}",
             f"selfhostguard={tmp_path / 'tuned.npy'}",
             "--catalog", "x", "--owner", str(tmp_path / "owner.npy"),
             "--region-json", str(tmp_path / "field.json"),
             "--ks", "4", "8", "--n-per-class", "300", "--n-uniform", "600",
             "--n-quantiles", "8", "--sizes", "0", "2",
             "--boundary-tiles", "16", "--boundary-queries", "64",
             "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["region_tiles"] == region
    assert sorted(r["boundary_tiles"]) == region          # all region tiles
    assert sum(r["class_particle_counts_region"].values()) == 2 * TILE ** 3
    assert r["n_uniform"] == 600
    lf = r["loss_forms"]
    assert set(lf) == {"frozen", "selfhostguard"}
    for key in ("unpaired_total", "paired_total", "class_cond_mass_total",
                "class_cond_equal_total"):
        assert lf["frozen"][key] >= 0.0
    # Same-field neighbours stay exact; and inside the region the tuned field
    # really differs from frozen, so the stale-neighbour mode is no longer the
    # same measurement as the same-field one (round 2's confound: it was).
    def at(mode, m):
        return [x for x in r["boundary"][mode] if x["size_cells"] == m][0]
    assert at("tuned_centroid_nbr", 2.0)["rel_kmax_p95"] < 1e-9
    f0, t0 = at("tuned_centroid_frozen", 0.0), at("tuned_centroid_nbr", 0.0)
    assert (f0["incomplete_frac"], f0["rel_kmax_p95"]) != (
        t0["incomplete_frac"], t0["rel_kmax_p95"])


# --------------------------------------------------------------------------- #
# Round 5: phase space
# --------------------------------------------------------------------------- #
def test_load_points_6d_scales_velocity(tmp_path, monkeypatch):
    monkeypatch.setattr(sd.v1, "NG", 8)
    monkeypatch.setattr(sd.v1, "CELL", 100.0 / 8)
    f = np.random.default_rng(0).standard_normal((6, 8, 8, 8)).astype(np.float32)
    np.save(tmp_path / "f.npy", f)
    p3 = sd.load_points(str(tmp_path / "f.npy"), 0.0)
    p6 = sd.load_points(str(tmp_path / "f.npy"), 500.0)
    assert p3.shape == (512, 3) and p6.shape == (512, 6)
    assert np.array_equal(p6[:, :3], p3)
    want = f[4].astype(np.float64).reshape(-1) * sd.VEL_NORM_KM_S / 500.0
    assert np.allclose(p6[:, 4], want)


def test_6d_tree_is_periodic_in_x_and_open_in_v():
    rng = np.random.default_rng(1)
    L = 10.0
    pts = np.concatenate([rng.uniform(0, L, (300, 3)),
                          rng.normal(0, 3.0, (300, 3))], axis=1)
    d = pts[:, None, :] - pts[None, :, :]
    d[..., :3] -= L * np.round(d[..., :3] / L)          # minimum image, x only
    brute = np.sort(np.sqrt((d ** 2).sum(-1)), axis=1)[:, 1:9]
    got, _ = sd.rk_query(sd.make_tree(pts, L), pts, np.arange(300), 8)
    assert np.allclose(got, brute, atol=1e-10)


def test_phase_space_sees_a_subhalo_that_position_cannot():
    """The round-5 premise in miniature. 60 particles sit inside a 3000-particle
    host, no denser than the host around them: in 3D their r_32 equals nearby
    host particles'. In 6D the same particles collapse when they share a bulk
    velocity ("coherent") and stay loose when they carry host-like velocities
    ("dissolved") -- the distinction Rockstar makes and 3D kNN cannot."""
    rng = np.random.default_rng(2)
    L, lam = 100.0, 1000.0
    host_x = 50 + rng.normal(0, 0.3, (3000, 3))
    host_v = rng.normal(0, 300.0, (3000, 3))
    sub_x = 50.2 + rng.normal(0, 0.05, (60, 3))
    x = np.concatenate([host_x, sub_x])
    rows = np.arange(3000, 3060)
    near = np.flatnonzero(np.linalg.norm(host_x - 50.2, axis=1) < 0.1)
    tree3 = sd.make_tree(x, L)
    r3_sub, _ = sd.rk_query(tree3, x[rows], rows, 32)
    r3_host, _ = sd.rk_query(tree3, x[near], near, 32)
    assert abs(np.median(r3_sub[:, -1]) / np.median(r3_host[:, -1]) - 1) < 0.1

    def r6(sub_v):
        p6 = np.concatenate([x, np.concatenate([host_v, sub_v]) / lam], axis=1)
        d, _ = sd.rk_query(sd.make_tree(p6, L), p6[rows], rows, 32)
        return np.median(d[:, -1])

    coherent = r6(600.0 + rng.normal(0, 30.0, (60, 3)))
    dissolved = r6(rng.normal(0, 300.0, (60, 3)))
    assert coherent < 0.5 * dissolved


def _field_set(tmp_path, rng):
    base = (0.3 * rng.standard_normal((6, NG, NG, NG))).astype(np.float32)
    for name, f in (("hr", base), ("frozen", base * 0.5), ("tuned", base * 0.9)):
        np.save(tmp_path / f"{name}.npy", f)
    owner = rng.integers(-1, 5, size=NG ** 3).astype(np.int32)
    owner[owner == 4] = 7
    np.save(tmp_path / "owner.npy", owner)


def _argv(tmp_path, out, fields, extra=()):
    return (["--fields"] + [f"{k}={tmp_path / (v + '.npy')}" for k, v in fields]
            + ["--catalog", "x", "--owner", str(tmp_path / "owner.npy"),
               "--ks", "4", "8", "--n-per-class", "400", "--n-quantiles", "8",
               "--sizes", "0", "6", "--boundary-tiles", "2",
               "--boundary-queries", "100", "--phase-lambda", "1000",
               "--boundary-by-class", "--out", str(out)] + list(extra))


def test_end_to_end_6d_by_class_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(sd.v1, "NG", NG)
    monkeypatch.setattr(sd.v1, "TILE", TILE)
    monkeypatch.setattr(sd.v1, "CELL", 100.0 / NG)
    monkeypatch.setattr(sd, "load_rockstar_ascii", lambda p: _cat())
    _field_set(tmp_path, np.random.default_rng(4))
    allf = [("hr", "hr"), ("frozen", "frozen"), ("selfhostguard", "tuned"),
            ("copy", "hr")]
    one = tmp_path / "one.json"
    sd.main(_argv(tmp_path, one, allf))
    r = json.loads(one.read_text())
    assert r["space"] == "6d" and r["phase_lambda_kms_per_mpch"] == 1000.0
    for e in r["membership"]["copy"]["classes"].values():
        assert np.allclose(e["paired_ratio_med"], 1.0)
    # 6D r_k differs from 3D r_k on the same field: the velocity axes are used.
    three = tmp_path / "three.json"
    sd.main([x for x in _argv(tmp_path, three, allf[:2])
             if x not in ("--phase-lambda", "1000")])
    r3 = json.loads(three.read_text())
    assert r3["space"] == "3d"
    assert not np.allclose(r3["membership"]["hr"]["classes"]["host"]["median_r"],
                           r["membership"]["hr"]["classes"]["host"]["median_r"])
    # Same-field neighbours at a wide margin stay exact in 6D, in every class.
    wide = [x for x in r["boundary"]["hr_centroid_nbr"] if x["size_cells"] == 6.0][0]
    assert wide["queries_stratified_by_class"]
    assert wide["rel_kmax_p95"] < 1e-9 and wide["incomplete_frac"] == 0.0
    assert set(wide["by_class"]) == set(sd.CLASSES)
    for c in wide["by_class"].values():
        assert c["rel_kmax_p95"] < 1e-9
    assert "tuned_centroid_frozen" in r["boundary"]

    # Resume: stop after frozen, then finish -- membership and loss forms match.
    part = tmp_path / "part.json"
    sd.main(_argv(tmp_path, part, allf[:2]))
    sd.main(_argv(tmp_path, part, allf, extra=["--resume"]))
    p = json.loads(part.read_text())
    assert set(p["membership"]) == set(r["membership"])
    for lab in r["loss_forms"]:
        assert np.allclose(p["loss_forms"][lab]["class_cond_equal"],
                           r["loss_forms"][lab]["class_cond_equal"])
    assert "tuned_centroid_frozen" in p["boundary"]
    import pytest
    with pytest.raises(SystemExit, match="different"):
        argv = _argv(tmp_path, part, allf, extra=["--resume"])
        argv[argv.index("--phase-lambda") + 1] = "500"
        sd.main(argv)
