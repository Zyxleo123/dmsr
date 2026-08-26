"""Pins for the bound-halo discriminator.

This measurement decides which term gets added to the gather objective next, so
the two things that would silently move every number are pinned against values
computed by hand: the energetics (a factor of two in ``W`` turns "unbound" into
"virialised") and the periodic unwrap (a subhalo on a box face would otherwise
report a radius of order the box and score as unbound for a bookkeeping reason).

The controls in the report -- ``r_rms`` and ``sigma_v`` -- only mean something if
the candidate statistics really are independent of them, so the sign tests below
use configurations that share a size and differ only in temperature.
"""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.features.bound_discriminator import (
    G_MPC_KMS2_PER_MSUN, mann_whitney_auc, member_ids, particles_at,
    set_statistics, specific_potential, unwrap_periodic,
)

L = 100.0
M_P = 5.81881e8


# --------------------------------------------------------------------------- #
# Periodic unwrap
# --------------------------------------------------------------------------- #
def test_unwrap_puts_a_clump_straddling_the_face_back_together():
    # Four particles within 0.2 Mpc/h of each other, across x = 0.
    pos = np.array([[99.95, 50.0, 50.0], [0.05, 50.0, 50.0],
                    [99.90, 50.0, 50.0], [0.10, 50.0, 50.0]])
    out = unwrap_periodic(pos, L)
    assert np.ptp(out[:, 0]) == pytest.approx(0.2, abs=1e-9)
    # and the wrapped centroid is the true one, not the box centre
    assert (out.mean(axis=0)[0] % L) == pytest.approx(0.0, abs=0.03)


def test_unwrap_is_a_no_op_on_a_clump_in_the_interior():
    rng = np.random.default_rng(0)
    pos = 50.0 + 0.1 * rng.standard_normal((32, 3))
    assert unwrap_periodic(pos, L) == pytest.approx(pos, abs=1e-9)


def test_a_second_pass_survives_an_outlier_reference_particle():
    # The first particle is the odd one out; a single pass about it still works,
    # but the centroid must land on the bulk, not between the two groups.
    pos = np.array([[0.30, 50.0, 50.0]] + [[99.98, 50.0, 50.0]] * 20)
    out = unwrap_periodic(pos, L)
    assert (out.mean(axis=0)[0] % L) == pytest.approx(99.995, abs=0.02)


# --------------------------------------------------------------------------- #
# Energetics -- the factor-of-two pins
# --------------------------------------------------------------------------- #
def test_two_body_potential_is_the_analytic_value():
    d = 0.5
    pos = np.array([[10.0, 10.0, 10.0], [10.0 + d, 10.0, 10.0]])
    phi = specific_potential(pos, M_P, softening_mpc_h=1e-6)
    want = -G_MPC_KMS2_PER_MSUN * M_P / d
    assert phi == pytest.approx([want, want], rel=1e-12)


def test_softening_floors_the_potential_of_coincident_particles():
    pos = np.zeros((2, 3))
    eps = 0.01
    phi = specific_potential(pos, M_P, softening_mpc_h=eps)
    assert phi == pytest.approx([-G_MPC_KMS2_PER_MSUN * M_P / eps] * 2, rel=1e-12)


def test_self_term_is_excluded():
    # Three coincident-in-pairs particles: each sees exactly the other two.
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    phi = specific_potential(pos, M_P, softening_mpc_h=1e-9, chunk=2)
    g = G_MPC_KMS2_PER_MSUN * M_P
    assert phi[0] == pytest.approx(-g * (1 / 1.0 + 1 / 2.0), rel=1e-12)
    assert phi[1] == pytest.approx(-g * (1 / 1.0 + 1 / 1.0), rel=1e-12)


def test_chunking_does_not_change_the_potential():
    rng = np.random.default_rng(1)
    pos = rng.standard_normal((37, 3))
    a = specific_potential(pos, M_P, softening_mpc_h=0.01, chunk=4)
    b = specific_potential(pos, M_P, softening_mpc_h=0.01, chunk=1024)
    assert a == pytest.approx(b, rel=1e-12)


def test_a_cold_pair_is_fully_bound_with_zero_virial_ratio():
    d = 0.5
    pos = np.array([[10.0, 10.0, 10.0], [10.0 + d, 10.0, 10.0]])
    vel = np.zeros((2, 3))
    st = set_statistics(pos, vel, particle_mass_msun_h=M_P, boxsize_mpc_h=L,
                        softening_mpc_h=1e-6)
    assert st.bound_frac == 1.0
    assert st.virial_ratio == pytest.approx(0.0, abs=1e-12)
    # W = 0.5 m sum phi = -G m^2 / d : the half undoes the double count
    assert st.r_rms == pytest.approx(d / 2.0, rel=1e-12)


def test_a_hot_set_is_unbound_and_super_virial_at_the_same_size():
    rng = np.random.default_rng(2)
    pos = 10.0 + 0.05 * rng.standard_normal((64, 3))
    cold = set_statistics(pos, np.zeros((64, 3)), particle_mass_msun_h=M_P,
                          boxsize_mpc_h=L, softening_mpc_h=0.001)
    hot = set_statistics(pos, 5e3 * rng.standard_normal((64, 3)),
                         particle_mass_msun_h=M_P, boxsize_mpc_h=L,
                         softening_mpc_h=0.001)
    # identical geometry, so the CONTROLS agree and the candidates do not
    assert hot.r_rms == pytest.approx(cold.r_rms, rel=1e-12)
    assert hot.bound_frac == 0.0 and cold.bound_frac == 1.0
    assert hot.virial_ratio > 100.0 * max(cold.virial_ratio, 1e-9)
    assert hot.coldness > cold.coldness


def test_virial_ratio_is_one_for_a_circular_binary():
    # Two masses in a circular orbit: v = sqrt(G m / (2 d)) each about the
    # centre of mass, giving 2T = |W| exactly.
    d = 0.4
    v = np.sqrt(G_MPC_KMS2_PER_MSUN * M_P / (2.0 * d))
    pos = np.array([[10.0, 10.0, 10.0], [10.0 + d, 10.0, 10.0]])
    vel = np.array([[0.0, v, 0.0], [0.0, -v, 0.0]])
    st = set_statistics(pos, vel, particle_mass_msun_h=M_P, boxsize_mpc_h=L,
                        softening_mpc_h=1e-9)
    assert st.virial_ratio == pytest.approx(1.0, rel=1e-9)


def test_hubble_like_expansion_shows_up_in_vr_corr():
    rng = np.random.default_rng(3)
    pos = 10.0 + 0.05 * rng.standard_normal((256, 3))
    dx = pos - pos.mean(axis=0)
    hubble_like = 4000.0 * dx                      # v proportional to r
    scrambled = 200.0 * rng.standard_normal((256, 3))
    a = set_statistics(pos, hubble_like, particle_mass_msun_h=M_P,
                       boxsize_mpc_h=L, softening_mpc_h=0.001)
    b = set_statistics(pos, scrambled, particle_mass_msun_h=M_P,
                       boxsize_mpc_h=L, softening_mpc_h=0.001)
    assert a.vr_corr > 0.9          # a radial velocity gradient
    assert abs(b.vr_corr) < 0.3     # no radial ordering
    assert a.r_rms == pytest.approx(b.r_rms, rel=1e-12)


def test_constant_speed_outflow_is_invisible_to_vr_corr_and_caught_by_vr_mean():
    """The reason both halves are reported.

    A set drifting outward at one speed has no radial velocity *gradient*, so
    the Pearson statistic reads ~0 and would call it settled. The net radial
    velocity does not.
    """
    rng = np.random.default_rng(5)
    pos = 10.0 + 0.05 * rng.standard_normal((256, 3))
    dx = pos - pos.mean(axis=0)
    r = np.linalg.norm(dx, axis=1)
    outflow = 300.0 * dx / r[:, None]              # same speed, all outward
    st = set_statistics(pos, outflow, particle_mass_msun_h=M_P,
                        boxsize_mpc_h=L, softening_mpc_h=0.001)
    assert abs(st.vr_corr) < 0.3                   # the blind spot
    assert st.vr_mean > 0.9                        # caught here


def test_d6_is_nan_without_a_host_reference_and_finite_with_one():
    pos = np.zeros((4, 3)); pos[:, 0] = np.arange(4) * 0.01
    vel = np.zeros((4, 3))
    assert np.isnan(set_statistics(pos, vel, particle_mass_msun_h=M_P,
                                   boxsize_mpc_h=L).d6)
    st = set_statistics(pos, vel, particle_mass_msun_h=M_P, boxsize_mpc_h=L,
                        host_sigma_x=1.0, host_sigma_v=500.0)
    assert np.isfinite(st.d6) and st.d6 > 0


# --------------------------------------------------------------------------- #
# Field access
# --------------------------------------------------------------------------- #
def test_particles_at_matches_field_to_particles_on_the_same_ids(tmp_path):
    from cosmo_sr.eval.particles import field_to_particles

    rng = np.random.default_rng(4)
    ng = 8
    fld = (1e-4 * rng.standard_normal((6, ng, ng, ng))).astype(np.float32)
    ref = field_to_particles(fld, boxsize_kpc_h=100000.0, redshift=0.0)
    ids = np.array([0, 1, ng, ng * ng, ng ** 3 - 1, 17, 260], dtype=np.int64)
    pos, vel = particles_at(fld, ids)
    assert pos == pytest.approx(ref.pos_mpc_h[ids], abs=1e-5)
    assert vel == pytest.approx(ref.vel_kms[ids], abs=1e-4)


def test_particles_at_reads_a_memmap_without_loading_the_box(tmp_path):
    ng = 8
    fld = np.zeros((6, ng, ng, ng), dtype=np.float32)
    p = tmp_path / "f.npy"
    np.save(p, fld)
    mm = np.load(p, mmap_mode="r")
    pos, vel = particles_at(mm, np.array([0, 5], dtype=np.int64))
    cell = 100.0 / ng                                     # Mpc/h
    assert pos[0] == pytest.approx([0.5 * cell] * 3, rel=1e-6)
    assert vel == pytest.approx(np.zeros((2, 3)), abs=0.0)


def test_member_ids_recovers_the_sets_in_one_pass(tmp_path):
    owner = np.full(64, -1, dtype=np.int32)
    owner[[3, 9, 40]] = 7
    owner[[1, 2]] = 11
    p = tmp_path / "owner.npy"
    np.save(p, owner)
    got = member_ids(str(p), [7, 11, 99], chunk=7)     # chunk < array on purpose
    assert got[7].tolist() == [3, 9, 40]
    assert got[11].tolist() == [1, 2]
    assert got[99].size == 0


# --------------------------------------------------------------------------- #
# Discrimination
# --------------------------------------------------------------------------- #
def test_auc_is_zero_one_and_half_in_the_three_obvious_cases():
    assert mann_whitney_auc([1, 2, 3], [4, 5, 6]) == pytest.approx(0.0)
    assert mann_whitney_auc([4, 5, 6], [1, 2, 3]) == pytest.approx(1.0)
    assert mann_whitney_auc([1, 2, 3], [1, 2, 3]) == pytest.approx(0.5)


def test_auc_averages_ties_rather_than_inventing_separation():
    # bound_frac saturates, so a fully tied block must read as no separation.
    assert mann_whitney_auc([0.0] * 8, [0.0] * 8) == pytest.approx(0.5)
    assert mann_whitney_auc([1.0, 0.0], [0.0, 0.0]) == pytest.approx(0.75)


def test_auc_ignores_non_finite_values():
    assert mann_whitney_auc([1, np.nan, 3], [4, 5, np.inf]) == pytest.approx(0.0)
