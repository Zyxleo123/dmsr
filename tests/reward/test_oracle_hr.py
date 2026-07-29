"""Experiment 1: the HR-residual oracle must edit where it says it edits.

The failure this file is mostly guarding against is silent and cheap to make:
building the mask in the wrong space. The residual is indexed by *Lagrangian*
site and the target is at an *Eulerian* position, so a mask built around the
position would land on unrelated material and produce a clean, wrong, negative
result. The tests pin the mask to the site set it was built from and pin
alpha = 0 to bit-exact SR2.
"""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.eval.rockstar import HaloCatalog
from cosmo_sr.reward.oracle_hr import (
    apply_intervention,
    channel_slice,
    ids_to_lattice,
    lagrangian_mask,
    random_control_sites,
    recovery_report,
    select_targets,
)
from conftest import synthetic_catalog

NG = 32
BOX = 100.0


def _field(seed=0, ng=NG):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(6, ng, ng, ng)).astype(np.float32)


# ------------------------------------------------------------------ masks


def test_mask_is_localised_on_its_own_sites():
    sites = np.array([[4, 4, 4], [4, 4, 5], [5, 4, 4]])
    m = lagrangian_mask(sites, NG, dilate=2, smooth=1.0)
    assert m.shape == (NG, NG, NG)
    assert m[4, 4, 4] > 0.5
    # Far from the site set the mask must be negligible, or the "localised"
    # claim in the composition formula is false.
    assert m[20, 20, 20] < 1e-6


def test_mask_peak_is_one_so_alpha_one_means_full_hr():
    m = lagrangian_mask(np.array([[8, 8, 8]]), NG, dilate=2, smooth=1.5)
    assert float(m.max()) == pytest.approx(1.0, abs=1e-6)


def test_mask_is_periodic():
    """A site at the box edge must wrap, not be clipped."""
    m = lagrangian_mask(np.array([[0, 0, 0]]), NG, dilate=2, smooth=1.0)
    assert m[NG - 1, 0, 0] > 0
    assert m[0, NG - 1, NG - 1] > 0


def test_mask_of_empty_site_set_is_zero():
    m = lagrangian_mask(np.zeros((0, 3), dtype=np.int64), NG)
    assert not m.any()


def test_mask_is_smooth_no_hard_edge():
    """A step in displacement is a delta in density; the low-k constraint would
    then be measuring the mask rather than the physics."""
    m = lagrangian_mask(np.array([[16, 16, 16]]), NG, dilate=2, smooth=2.0)
    line = m[16, 16, :]
    jumps = np.abs(np.diff(line))
    assert jumps.max() < 0.5 * float(m.max())


def test_ids_to_lattice_inverts_the_flat_index():
    coords = np.array([[0, 0, 0], [3, 7, 11], [NG - 1, NG - 1, NG - 1]])
    ids = (coords[:, 0] * NG + coords[:, 1]) * NG + coords[:, 2]
    np.testing.assert_array_equal(ids_to_lattice(ids, NG), coords)


# ---------------------------------------------------------- interventions


def test_alpha_zero_is_bit_exact_sr2():
    """The recovery-vs-alpha curve may only be anchored on an exact baseline."""
    base, hr = _field(0), _field(1)
    m = lagrangian_mask(np.array([[8, 8, 8]]), NG)
    out = apply_intervention(base, hr, m, 0.0, "both")
    assert np.array_equal(out, base)


def test_alpha_one_replaces_the_field_at_the_mask_peak():
    base, hr = _field(0), _field(1)
    sites = np.array([[8, 8, 8]])
    m = lagrangian_mask(sites, NG, dilate=0, smooth=0.0)
    out = apply_intervention(base, hr, m, 1.0, "both")
    assert out[0, 8, 8, 8] == pytest.approx(hr[0, 8, 8, 8], abs=1e-5)


def test_intervention_is_linear_in_alpha():
    """The interpretation of the alpha curve assumes the field interpolates."""
    base, hr = _field(0), _field(1)
    m = lagrangian_mask(np.array([[8, 8, 8]]), NG, dilate=1, smooth=1.0)
    half = apply_intervention(base, hr, m, 0.5, "both")
    full = apply_intervention(base, hr, m, 1.0, "both")
    np.testing.assert_allclose(half, 0.5 * (base + full), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode,touched", [
    ("disp", (0, 1, 2)), ("vel", (3, 4, 5)), ("both", (0, 1, 2, 3, 4, 5)),
])
def test_channel_modes_leave_the_other_channels_frozen(mode, touched):
    """'position works but velocity does not' is only readable if this holds."""
    base, hr = _field(0), _field(1)
    m = lagrangian_mask(np.array([[8, 8, 8]]), NG, dilate=1, smooth=1.0)
    out = apply_intervention(base, hr, m, 1.0, mode)
    for c in range(6):
        if c in touched:
            assert not np.array_equal(out[c], base[c])
        else:
            assert np.array_equal(out[c], base[c])


def test_intervention_only_changes_the_masked_region():
    base, hr = _field(0), _field(1)
    m = lagrangian_mask(np.array([[8, 8, 8]]), NG, dilate=1, smooth=1.0)
    out = apply_intervention(base, hr, m, 1.0, "both")
    far = np.abs(out[:, 24, 24, 24] - base[:, 24, 24, 24])
    assert far.max() < 1e-5


def test_mask_shape_mismatch_is_an_error():
    base, hr = _field(0), _field(1)
    with pytest.raises(ValueError, match="does not match field"):
        apply_intervention(base, hr, np.ones((8, 8, 8), np.float32), 1.0, "both")


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        channel_slice("displacement")


# --------------------------------------------------------------- controls


def test_control_has_the_same_site_count_and_excludes_the_target():
    rng = np.random.default_rng(0)
    host = rng.integers(0, NG, size=(500, 3))
    target = host[:40]
    ctrl = random_control_sites(host, target, seed=1)
    assert ctrl.shape[0] == 40
    key = lambda a: set(map(tuple, a.tolist()))  # noqa: E731
    assert not (key(ctrl) & key(target))


def test_control_is_deterministic_given_a_seed():
    rng = np.random.default_rng(0)
    host = rng.integers(0, NG, size=(200, 3))
    target = host[:20]
    a = random_control_sites(host, target, seed=7)
    b = random_control_sites(host, target, seed=7)
    np.testing.assert_array_equal(a, b)


def test_control_shrinks_rather_than_repeating_when_the_host_is_small():
    host = np.array([[1, 1, 1], [1, 1, 2], [1, 1, 3]])
    ctrl = random_control_sites(host, host[:2], seed=0)
    assert ctrl.shape[0] == 1          # only one free site remains
    assert len(set(map(tuple, ctrl.tolist()))) == ctrl.shape[0]


# ------------------------------------------------------ target selection


def _hr_sr_pair():
    """HR with rich hosts; SR identical but with every subhalo of host 1 removed."""
    hosts = [((20.0, 20.0, 20.0), 2e13), ((60.0, 60.0, 60.0), 5e13)]
    hr = synthetic_catalog(hosts, [3, 4], boxsize=BOX, seed=0,
                           sub_num_p=80, sub_mass=8e10)
    keep = ~((hr.parent_ids == hr.ids[1]))
    sr = HaloCatalog(
        ids=hr.ids[keep], parent_ids=hr.parent_ids[keep], mvir=hr.mvir[keep],
        rvir=hr.rvir[keep], vmax=hr.vmax[keep], pos=hr.pos[keep],
        vel=hr.vel[keep], num_p=hr.num_p[keep],
    )
    return hr, sr


def test_select_targets_finds_the_missing_subhalos():
    hr, sr = _hr_sr_pair()
    edges = np.logspace(12.0, 14.5, 6)
    t = select_targets(hr, sr, host_mass_edges=edges, host_bins=(2, 3),
                       boxsize_mpc_h=BOX, min_sub_particles=20,
                       min_separation_mpc_h=1.0, one_per_host=False)
    assert t, "the removed subhalos of host 1 should be selectable"
    assert all(x.hr_host_id == int(hr.ids[1]) for x in t)
    assert all(x.host_bin in (2, 3) for x in t)


def test_select_targets_enforces_separation():
    """Batching targets into one box is only sound if their edits cannot interact."""
    hosts = [((10.0, 10.0, 10.0), 2e13), ((11.0, 10.0, 10.0), 2e13)]
    # rvir small enough that the offset subhalos sit well outside 0.25 Rvir:
    # a central missing subhalo is classified merged_into_host, not missing,
    # and select_targets deliberately only takes the latter.
    hr = synthetic_catalog(hosts, [2, 2], boxsize=BOX, seed=1, sub_num_p=80,
                           rvir_kpc=50.0)
    sr = HaloCatalog(
        ids=hr.ids[:2], parent_ids=hr.parent_ids[:2], mvir=hr.mvir[:2],
        rvir=hr.rvir[:2], vmax=hr.vmax[:2], pos=hr.pos[:2], vel=hr.vel[:2],
        num_p=hr.num_p[:2],
    )
    edges = np.logspace(12.0, 14.5, 6)
    t = select_targets(hr, sr, host_mass_edges=edges, host_bins=(2, 3),
                       boxsize_mpc_h=BOX, min_sub_particles=20,
                       min_separation_mpc_h=6.0)
    assert len(t) == 1, "hosts 1 Mpc/h apart must not both be selected"


def test_select_targets_respects_max_and_resolution_cut():
    hr, sr = _hr_sr_pair()
    edges = np.logspace(12.0, 14.5, 6)
    t = select_targets(hr, sr, host_mass_edges=edges, host_bins=(2, 3),
                       boxsize_mpc_h=BOX, min_sub_particles=20,
                       min_separation_mpc_h=1.0, one_per_host=False,
                       max_targets=2)
    assert len(t) <= 2
    none = select_targets(hr, sr, host_mass_edges=edges, host_bins=(2, 3),
                          boxsize_mpc_h=BOX, min_sub_particles=10_000,
                          min_separation_mpc_h=1.0)
    assert none == []


def test_select_targets_excludes_the_sparse_bin_by_default():
    """Bin 4 (1e14) is evaluation-only; an oracle target there proves nothing
    Gate B can use."""
    hosts = [((30.0, 30.0, 30.0), 2e14)]
    hr = synthetic_catalog(hosts, [3], boxsize=BOX, seed=2, sub_num_p=80)
    sr = HaloCatalog(
        ids=hr.ids[:1], parent_ids=hr.parent_ids[:1], mvir=hr.mvir[:1],
        rvir=hr.rvir[:1], vmax=hr.vmax[:1], pos=hr.pos[:1], vel=hr.vel[:1],
        num_p=hr.num_p[:1],
    )
    edges = np.logspace(12.0, 14.5, 6)
    assert select_targets(hr, sr, host_mass_edges=edges, host_bins=(2, 3),
                          boxsize_mpc_h=BOX, min_sub_particles=20) == []


# --------------------------------------------------------------- recovery


def test_recovery_detects_a_restored_subhalo():
    hr, sr = _hr_sr_pair()
    edges = np.logspace(12.0, 14.5, 6)
    targets = select_targets(hr, sr, host_mass_edges=edges, host_bins=(2, 3),
                             boxsize_mpc_h=BOX, min_sub_particles=20,
                             min_separation_mpc_h=1.0, one_per_host=False)
    none = recovery_report(targets, hr, sr, boxsize_mpc_h=BOX)
    assert not any(r["recovered"] for r in none)
    # "After the intervention" the HR catalog itself is the perfect outcome.
    full = recovery_report(targets, hr, hr, boxsize_mpc_h=BOX)
    assert all(r["recovered"] for r in full)


def test_recovery_requires_the_object_to_be_a_subhalo():
    """A target reappearing as an independent host is reported, not counted."""
    hr, sr = _hr_sr_pair()
    edges = np.logspace(12.0, 14.5, 6)
    targets = select_targets(hr, sr, host_mass_edges=edges, host_bins=(2, 3),
                             boxsize_mpc_h=BOX, min_sub_particles=20,
                             min_separation_mpc_h=1.0, one_per_host=False)[:1]
    promoted = HaloCatalog(
        ids=hr.ids.copy(), parent_ids=np.full(hr.n, -1, dtype=np.int64),
        mvir=hr.mvir.copy(), rvir=hr.rvir.copy(), vmax=hr.vmax.copy(),
        pos=hr.pos.copy(), vel=hr.vel.copy(), num_p=hr.num_p.copy(),
    )
    rep = recovery_report(targets, hr, promoted, boxsize_mpc_h=BOX)
    assert rep[0]["recovered"] is False
    assert rep[0]["as_host"] is not None


def test_recovery_rejects_a_mass_mismatched_object():
    hr, sr = _hr_sr_pair()
    edges = np.logspace(12.0, 14.5, 6)
    targets = select_targets(hr, sr, host_mass_edges=edges, host_bins=(2, 3),
                             boxsize_mpc_h=BOX, min_sub_particles=20,
                             min_separation_mpc_h=1.0, one_per_host=False)[:1]
    tiny = HaloCatalog(
        ids=hr.ids.copy(), parent_ids=hr.parent_ids.copy(),
        mvir=hr.mvir * 1e-3, rvir=hr.rvir.copy(), vmax=hr.vmax.copy(),
        pos=hr.pos.copy(), vel=hr.vel.copy(), num_p=hr.num_p.copy(),
    )
    assert not recovery_report(targets, hr, tiny, boxsize_mpc_h=BOX)[0]["recovered"]
