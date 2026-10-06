"""Tests for the membership-solve figures that are pure functions of the npz.

The plotting itself is smoke-tested (it must write a file), but the index maths
behind the radial figure is checked against hand-computed answers: a segmented
max over a `rows` array that repeats a particle once per candidate pool, and a
periodic image that a box-straddling pool depends on.
"""
import os

import numpy as np
import pytest

import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    "viz_membership_solve",
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "scripts", "features", "viz_membership_solve.py"))
viz = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(viz)


def _dump(rows, setid, w, x, centre_host, is_member=None, host_rvir=None,
          host_key="set9:h165513"):
    """A minimal npz-shaped dict: one sweep, mult 1, so P == w."""
    rows = np.asarray(rows, dtype=np.int64)
    setid = np.asarray(setid, dtype=np.int64)
    n_sets = int(setid.max()) + 1
    return {
        "host_key": np.array(host_key),
        "n_sets": np.array(n_sets),
        "rows": rows,
        "setid": setid,
        "mult": np.ones(rows.size, dtype=np.float32),
        "sweep_tags": np.array(["init"]),
        "sweep_w": np.asarray(w, dtype=np.float32)[None, :],
        "sweep_c": np.zeros((1, n_sets, 3), dtype=np.float32),
        "sweep_mass": np.full((1, n_sets), 100.0, dtype=np.float32),
        "sweep_hr_kept": np.full((1, n_sets), 50.0, dtype=np.float32),
        "x": np.asarray(x, dtype=np.float32),
        "centre_host": np.asarray(centre_host, dtype=np.float32),
        "is_member": (np.zeros(rows.size, dtype=bool) if is_member is None
                      else np.asarray(is_member, dtype=bool)),
        "host_rvir": (np.zeros(n_sets) if host_rvir is None
                      else np.asarray(host_rvir, dtype=np.float64)),
        "target_mass": np.full(n_sets, 100.0, dtype=np.float32),
    }


def test_max_membership_is_a_segmented_max_over_rows():
    # particle 7 appears in two pools (0.2, 0.8); particle 9 in one (0.5).
    d = _dump(rows=[7, 9, 7], setid=[0, 1, 1], w=[0.2, 0.5, 0.8],
              x=[[0, 0, 0]] * 3, centre_host=[[0, 0, 0], [0, 0, 0]])
    P, r, ent = viz._max_membership_per_particle(d)
    assert P.size == 2                                  # distinct particles
    assert sorted(np.round(P, 6)) == [0.5, 0.8]


def test_radius_uses_the_winning_candidates_own_parent():
    # Both candidates hold the particle; candidate 1 wins, so the radius must
    # be measured from candidate 1's host, 3 Mpc/h away, not candidate 0's.
    d = _dump(rows=[4, 4], setid=[0, 1], w=[0.1, 0.9],
              x=[[10.0, 0, 0], [10.0, 0, 0]],
              centre_host=[[10.0, 0, 0], [7.0, 0, 0]])
    P, r, _ = viz._max_membership_per_particle(d)
    assert P.size == 1
    assert P[0] == pytest.approx(0.9)
    assert r[0] == pytest.approx(3.0)


def test_radius_takes_the_periodic_image():
    # A pool straddling the edge: 1.0 from a host at 99.5 in a 100 Mpc/h box.
    d = _dump(rows=[0], setid=[0], w=[1.0], x=[[0.5, 0, 0]],
              centre_host=[[99.5, 0, 0]])
    _, r, _ = viz._max_membership_per_particle(d)
    assert r[0] == pytest.approx(1.0)


def test_host_rvir_prefers_explicit_then_npz():
    d = _dump(rows=[0], setid=[0], w=[1.0], x=[[0, 0, 0]],
              centre_host=[[0, 0, 0]], host_rvir=[1.25])
    assert viz.host_rvir_mpc_h(d, explicit=2.0) == pytest.approx(2.0)
    assert viz.host_rvir_mpc_h(d) == pytest.approx(1.25)


def test_host_rvir_falls_back_to_the_rockstar_ascii(tmp_path):
    asc = tmp_path / "halos_0.0.ascii"
    asc.write_text(
        "#id num_p mvir mbound_vir rvir vmax\n"
        "12 27 1.2e10 1.2e10 48.15 34.3\n"
        "165513 222380 2.696e14 2.6e14 1328.903687 1500.0\n")
    d = _dump(rows=[0], setid=[0], w=[1.0], x=[[0, 0, 0]],
              centre_host=[[0, 0, 0]])
    # kpc/h in the file, Mpc/h out.
    assert viz.host_rvir_mpc_h(d, ascii_path=str(asc)) == pytest.approx(1.328903687)
    assert viz.host_rvir_mpc_h(d, ascii_path=str(tmp_path / "nope")) == 0.0


def test_host_rvir_is_zero_when_nothing_supplies_it():
    d = _dump(rows=[0], setid=[0], w=[1.0], x=[[0, 0, 0]],
              centre_host=[[0, 0, 0]])
    assert viz.host_rvir_mpc_h(d) == 0.0


@pytest.mark.parametrize("rvir", [0.0, 1.5])
def test_figure_radial_writes_a_file(tmp_path, rvir):
    rng = np.random.default_rng(0)
    n = 500
    d = _dump(rows=np.arange(n), setid=rng.integers(0, 3, n),
              w=rng.random(n), x=rng.random((n, 3)) * 4.0,
              centre_host=np.zeros((3, 3)),
              is_member=rng.random(n) < 0.3)
    out = tmp_path / "r.png"
    viz.figure_radial(d, str(out), rvir=rvir)
    assert out.stat().st_size > 0


def test_figure_panel_writes_a_file(tmp_path):
    rng = np.random.default_rng(1)
    n = 800
    d = _dump(rows=np.arange(n), setid=rng.integers(0, 3, n),
              w=rng.random(n), x=rng.random((n, 3)) * 20.0,
              centre_host=np.full((3, 3), 10.0))
    out = tmp_path / "p.png"
    viz.figure_heatmap_panel(d, str(out), half_width=10.0)
    assert out.stat().st_size > 0


def _hr_ascii(tmp_path):
    """A 3-halo HR ascii: host 100 at origin (Rvir 1000 kpc/h = 1 Mpc/h), one
    subhalo 200 at 1.5 Mpc/h (r/Rvir 1.5), one subhalo 300 at 0.3 Mpc/h.
    Rockstar parent is encoded via the i_so column pointing at the host's idx.
    """
    # columns: id num_p mvir mbound rvir vmax rvmax vrms x y z vx vy vz ...
    # load_rockstar_ascii uses id=0, num_p=1, rvir=4, x/y/z=8/9/10, idx=-5, i_so=-4.
    # Give 20 columns so idx=-5 -> col15, i_so=-4 -> col16.
    def row(hid, x, idxc, iso):
        c = [0.0] * 20
        c[0] = hid; c[1] = 500; c[4] = 1000.0
        c[8] = x; c[9] = 0.0; c[10] = 0.0
        c[15] = idxc; c[16] = iso
        return " ".join(repr(v) for v in c)
    asc = tmp_path / "halos_0.0.ascii"
    asc.write_text(
        "#id num_p mvir mbound rvir vmax rvmax vrms x y z vx vy vz Jx idx i_so a b c\n"
        + row(100, 0.0, 0, -1) + "\n"     # host, idx 0, no parent
        + row(200, 1.5, 1, 0) + "\n"      # sub -> parent idx 0 = id 100
        + row(300, 0.3, 2, 0) + "\n")     # sub -> parent idx 0
    return str(asc)


def test_hr_parent_radii_uses_each_candidates_own_parent(tmp_path):
    asc = _hr_ascii(tmp_path)
    # two candidates, ids 200 and 300; centre_ref at the HR positions, and an
    # optimized centre pushed 0.5 Mpc/h further out for candidate 200.
    d = _dump(rows=[0, 1], setid=[0, 1], w=[1.0, 1.0],
              x=[[1.5, 0, 0], [0.3, 0, 0]],
              centre_host=[[9, 9, 9], [9, 9, 9]])   # deliberately WRONG pool ref
    d["halo_id"] = np.array([200, 300], dtype=np.int64)
    d["centre_ref"] = np.array([[1.5, 0, 0], [0.3, 0, 0]], dtype=np.float32)
    d["sweep_c"] = np.array([[[1.5, 0, 0], [0.3, 0, 0]],
                             [[2.0, 0, 0], [0.3, 0, 0]]], dtype=np.float32)
    r_ref, r_opt, n, ntot = viz._hr_parent_radii(d, asc)
    assert (n, ntot) == (2, 2)
    # host Rvir is 1 Mpc/h, so radius == distance; parent is the host, not the
    # bogus centre_host.
    assert sorted(np.round(r_ref, 3)) == [0.3, 1.5]
    assert sorted(np.round(r_opt, 3)) == [0.3, 2.0]   # candidate 200 pushed out


def test_hr_parent_radii_drops_hosts_without_a_parent(tmp_path):
    asc = _hr_ascii(tmp_path)
    d = _dump(rows=[0], setid=[0], w=[1.0], x=[[0, 0, 0]],
              centre_host=[[0, 0, 0]])
    d["halo_id"] = np.array([100], dtype=np.int64)     # the host itself
    d["centre_ref"] = np.array([[0, 0, 0]], dtype=np.float32)
    d["sweep_c"] = np.zeros((2, 1, 3), dtype=np.float32)
    r_ref, r_opt, n, ntot = viz._hr_parent_radii(d, asc)
    assert (n, ntot) == (0, 1)


def test_figure_centre_radial_writes_a_file(tmp_path):
    asc = _hr_ascii(tmp_path)
    d = _dump(rows=[0, 1], setid=[0, 1], w=[1.0, 1.0],
              x=[[1.5, 0, 0], [0.3, 0, 0]], centre_host=[[0, 0, 0], [0, 0, 0]])
    d["halo_id"] = np.array([200, 300], dtype=np.int64)
    d["centre_ref"] = np.array([[1.5, 0, 0], [0.3, 0, 0]], dtype=np.float32)
    d["sweep_c"] = np.array([[[1.5, 0, 0], [0.3, 0, 0]],
                             [[1.6, 0, 0], [0.3, 0, 0]]], dtype=np.float32)
    out = tmp_path / "cr.png"
    viz.figure_centre_radial(d, str(out), asc, sr_median=0.35)
    assert out.stat().st_size > 0


def test_figure_centre_radial_bars_writes_a_file(tmp_path):
    asc = _hr_ascii(tmp_path)
    d = _dump(rows=[0, 1], setid=[0, 1], w=[1.0, 1.0],
              x=[[1.5, 0, 0], [0.3, 0, 0]], centre_host=[[0, 0, 0], [0, 0, 0]])
    d["halo_id"] = np.array([200, 300], dtype=np.int64)
    d["centre_ref"] = np.array([[1.5, 0, 0], [0.3, 0, 0]], dtype=np.float32)
    d["sweep_c"] = np.array([[[1.5, 0, 0], [0.3, 0, 0]],
                             [[1.6, 0, 0], [0.3, 0, 0]]], dtype=np.float32)
    out = tmp_path / "bars.png"
    viz.figure_centre_radial_bars(d, str(out), asc, sr_median=0.35)
    assert out.stat().st_size > 0
    # and with no SR reference line
    viz.figure_centre_radial_bars(d, str(tmp_path / "b2.png"), asc)
    assert (tmp_path / "b2.png").stat().st_size > 0
