"""The two questions owed before a host-profile arm is trained.

*The shell counts must be counts.* ``radial_counts`` is the whole supply
verdict, and a wrong ``R_vir`` normalisation or a wrong periodic wrap would
move the 1.0-1.5 shell without changing anything that looks broken. Pinned
against a configuration whose answer is known by construction.

*The drift must be signed.* An unsigned recentring says nothing about
placement: the question is whether the M step pulls candidates toward the host
or away from it, and the sign is the entire content. Pinned by feeding
snapshots that walk a known direction.

*The LR gate must drop what the LR catalog cannot see.* ``host_table`` is what
decides which supervised hosts get an NFW and which fall back to the flat
background, and a gate that silently passed everything would hand 43% of the
catalog a profile built from a row Rockstar never resolved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from build_lr_host_table import host_table  # noqa: E402
from host_profile_probe import (  # noqa: E402
    drift_for_dump, parent_geometry, radial_counts,
)


class _Cat:
    """The three columns ``gate_hosts`` reads, plus what ``host_table`` copies."""

    def __init__(self, parent_ids, num_p, rvir, mvir=None, pos=None, ids=None):
        n = len(num_p)
        self.parent_ids = np.asarray(parent_ids)
        self.num_p = np.asarray(num_p)
        self.rvir = np.asarray(rvir, dtype=np.float64)
        self.mvir = np.asarray(mvir if mvir is not None else np.full(n, 1e13))
        self.pos = np.asarray(pos if pos is not None else np.zeros((n, 3)))
        self.ids = np.asarray(ids if ids is not None else np.arange(n))
        self.n = n


# --------------------------------------------------------------------------- #
# The supply
# --------------------------------------------------------------------------- #
def test_radial_counts_puts_each_particle_in_its_own_shell():
    """One particle per shell, at the shell's midpoint, in R_vir units."""
    shells = (0.0, 1.0, 2.0, 3.0)
    rvir = np.array([2.0])
    centre = np.array([[50.0, 50.0, 50.0]])
    mids = np.array([0.5, 1.5, 2.5]) * rvir[0]
    pos = centre + np.stack([mids, np.zeros(3), np.zeros(3)], axis=1)
    got = radial_counts(pos, centre, rvir, boxsize=100.0, shells=shells)
    assert got.tolist() == [[1, 1, 1]]


def test_radial_counts_wraps_across_the_box_face():
    """A host at the face must count material on the other side of it."""
    shells = (0.0, 1.0)
    rvir = np.array([1.0])
    centre = np.array([[99.5, 50.0, 50.0]])
    # 0.5 Mpc/h away THROUGH the face, i.e. at x = 0.0 in box coordinates.
    pos = np.array([[0.0, 50.0, 50.0]])
    assert radial_counts(pos, centre, rvir, boxsize=100.0,
                         shells=shells).tolist() == [[1]]


def test_radial_counts_scales_with_rvir_not_with_mpc():
    """Two hosts of different size must bin the SAME physical radius
    differently -- the shells are in R_vir, which is the whole point."""
    shells = (0.0, 1.0, 2.0)
    centres = np.array([[50.0, 50.0, 50.0], [20.0, 20.0, 20.0]])
    rvir = np.array([1.0, 4.0])
    pos = np.array([[51.5, 50.0, 50.0], [21.5, 20.0, 20.0]])
    got = radial_counts(pos, centres, rvir, boxsize=100.0, shells=shells)
    assert got[0].tolist() == [0, 1]    # 1.5 R_vir for the small host
    assert got[1].tolist() == [1, 0]    # 0.375 R_vir for the big one


# --------------------------------------------------------------------------- #
# The drift
# --------------------------------------------------------------------------- #
def _dump(tmp_path, centres):
    p = tmp_path / "dump.npz"
    s = np.asarray(centres, dtype=np.float32)
    np.savez(p, sweep_c=s, sweep_tags=np.asarray(
        ["init"] + [f"sweep{i}" for i in range(1, s.shape[0])]),
        centre_host=np.zeros((s.shape[1], 3), dtype=np.float32),
        centre_scale=np.ones(s.shape[1], dtype=np.float32),
        halo_id=np.arange(s.shape[1]))
    return p


def test_drift_reports_no_rvir_rather_than_the_wrong_one(tmp_path):
    """`centre_scale` is the SET's own R_vir floored at 0.15 Mpc/h, not the
    host's. Dividing by it turned a placement of ~1 R_vir into 25 on the first
    real dump, which reads as a broken run rather than as wrong units -- so a
    dump with no host size must report None, not a number."""
    c = [[[10.0, 0.0, 0.0]], [[8.0, 0.0, 0.0]]]
    rep = drift_for_dump(_dump(tmp_path, c))
    assert rep["sweeps"][0]["median_r_over_host_rvir"] is None
    assert rep["sweeps"][0]["median_r_mpc_h"] == pytest.approx(10.0)


def test_drift_uses_host_rvir_when_the_dump_carries_it(tmp_path):
    p = tmp_path / "withrvir.npz"
    np.savez(p, sweep_c=np.array([[[10.0, 0.0, 0.0]]], dtype=np.float32),
             sweep_tags=np.asarray(["init"]),
             centre_host=np.zeros((1, 3), dtype=np.float32),
             centre_scale=np.full(1, 0.15, dtype=np.float32),
             host_rvir=np.array([2.0]), halo_id=np.arange(1))
    rep = drift_for_dump(p)
    assert rep["sweeps"][0]["median_r_over_host_rvir"] == pytest.approx(5.0)


def test_drift_reports_inward_when_the_centres_fall_toward_the_host(tmp_path):
    c = [[[10.0, 0.0, 0.0]], [[8.0, 0.0, 0.0]], [[6.0, 0.0, 0.0]]]
    rep = drift_for_dump(_dump(tmp_path, c))
    assert rep["net_radial_drift_mpc_h"] == pytest.approx(-4.0)
    assert rep["verdict"].startswith("INWARD")
    assert rep["sweeps"][1]["frac_moving_inward"] == 1.0


def test_drift_reports_outward_when_they_climb(tmp_path):
    c = [[[6.0, 0.0, 0.0]], [[8.0, 0.0, 0.0]]]
    rep = drift_for_dump(_dump(tmp_path, c))
    assert rep["net_radial_drift_mpc_h"] == pytest.approx(2.0)
    assert rep["verdict"].startswith("STABLE/OUTWARD")


def test_drift_refuses_a_dump_with_no_host(tmp_path):
    p = tmp_path / "nohost.npz"
    np.savez(p, sweep_c=np.zeros((2, 1, 3), dtype=np.float32),
             sweep_tags=np.asarray(["init", "sweep1"]), halo_id=np.arange(1))
    with pytest.raises(SystemExit, match="no centre_host"):
        drift_for_dump(p)


# --------------------------------------------------------------------------- #
# The LR gate
# --------------------------------------------------------------------------- #
def test_host_table_drops_subhalos_and_small_hosts_and_converts_rvir():
    # floor = 20000 / (512/64)^3 = 20000 / 512 = 39.06 LR particles
    cat = _Cat(parent_ids=[-1, -1, 7, -1],
               num_p=[100, 10, 100, 40],
               rvir=[500.0, 400.0, 300.0, 250.0],   # kpc/h
               mvir=[1e14, 1e12, 5e13, 6e13],
               pos=np.array([[1.0, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]))
    t = host_table(cat, min_hr_num_p=20000, ng_lr=64, ng_hr=512)
    assert t["host_id"].tolist() == [0, 3]          # subhalo and 10p host gone
    assert t["host_rvir_mpc_h"].tolist() == [0.5, 0.25]   # kpc/h -> Mpc/h
    assert t["host_mvir_msun_h"].tolist() == [1e14, 6e13]


def test_parent_geometry_resolves_each_set_to_its_own_host():
    """A task is named for one host but its sets are selected by home tile,
    and a tile is 12.5 Mpc/h a side. On the first real dump 108 of 159 sets
    belonged to OTHER parents, and signing against the task host inverted the
    verdict (-0.21 R_vir INWARD read as +0.08 STABLE). So each set must be
    resolved to its own parent, and a set with no resolvable parent dropped."""
    cat = _Cat(parent_ids=[-1, 10, 20, -1, 999],
               num_p=[1000, 100, 100, 1000, 100],
               rvir=[1200.0, 50.0, 50.0, 800.0, 50.0],
               pos=np.array([[5.0, 5, 5], [5.4, 5, 5], [9.2, 9, 9],
                             [9.0, 9, 9], [1, 1, 1]]),
               ids=[10, 11, 21, 20, 22])
    g = parent_geometry(np.array([11, 21, 22]), cat)
    assert g["parent_id"].tolist() == [10, 20, -1]
    assert g["parent_pos"][0].tolist() == [5.0, 5.0, 5.0]   # not the task host
    assert g["parent_pos"][1].tolist() == [9.0, 9.0, 9.0]
    assert g["parent_rvir"][:2].tolist() == [1.2, 0.8]       # kpc/h -> Mpc/h
    assert not np.isfinite(g["parent_rvir"][2])              # parent 999 absent


def test_drift_signed_against_parent_beats_the_task_host(tmp_path):
    """The two readings must actually differ, or the fix is cosmetic: one set
    falls toward its OWN parent while moving AWAY from the task's host."""
    p = tmp_path / "d.npz"
    # Set 11 sits between two hosts and falls toward its parent (10, at x=0)
    # while receding from the task host, which the dump records at x=20.
    np.savez(p, sweep_c=np.array([[[10.0, 0, 0]], [[8.0, 0, 0]]], dtype=np.float32),
             sweep_tags=np.asarray(["init", "sweep1"]),
             centre_host=np.array([[20.0, 0, 0]], dtype=np.float32),
             centre_scale=np.ones(1, dtype=np.float32),
             halo_id=np.array([11]))
    cat = _Cat(parent_ids=[-1, 10], num_p=[1000, 100], rvir=[1000.0, 50.0],
               pos=np.array([[0.0, 0, 0], [10.0, 0, 0]]), ids=[10, 11])

    naive = drift_for_dump(p)
    assert naive["signed_against"] == "task_host"
    assert naive["net_radial_drift_mpc_h"] == pytest.approx(+2.0)

    fixed = drift_for_dump(p, cat=cat)
    assert fixed["signed_against"] == "parent"
    assert fixed["net_radial_drift_mpc_h"] == pytest.approx(-2.0)
    assert fixed["n_with_parent"] == 1
