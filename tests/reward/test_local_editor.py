"""The local editor's invariants -- the ones a silent violation would hide.

Every failure mode pinned here produces a *plausible* result rather than a
crash: an editor that moves particles it did not claim, a "no-op" that perturbs
the field at the last bit, a claim that depends on pool ordering, an edit that
behaves differently near a box face. All of those would show up downstream as a
reward number, and none of them as an error.
"""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.data.preprocess_srs import disnorm, velnorm
from cosmo_sr.eval.particles import field_to_particles
from cosmo_sr.reward.local_editor import (
    ACTION_PARAMS, ActionCodec, EditorAction, HostPool, ParamSpec, SEARCH_PARAMS,
    SubhaloToken, action_from_values, add_displacement_mpc, apply_edits,
    build_host_pool, check_norm_convention, edge_window, min_image,
    n_particles_for_token, particle_positions_mpc, particle_velocities_kms,
    plan_edit, plan_edits, proposal_center_mpc, search_codec, token_from_values,
)

# A small periodic box with a fine enough lattice that an edit radius of a few
# tenths of Rvir still contains tens of particles: cell = 0.5 Mpc/h, Rvir = 4,
# so source_radius_rvir = 0.3 spans ~2.4 cells. The real box is 512^3 over 100
# Mpc/h (cell 0.195) with Rvir ~ 0.5-0.9, which is the same ratio.
NG = 20
BOX = 10.0
RVIR = 4.0


def make_field(seed=0, ng=NG):
    rng = np.random.default_rng(seed)
    return (rng.normal(0.0, 0.05, size=(6, ng, ng, ng))).astype(np.float32)


def make_pool(field, ids=None, *, ng=NG, center=(5.0, 5.0, 5.0), rvir=RVIR,
              mvir=1e13):
    ids = np.arange(ng ** 3, dtype=np.int64) if ids is None else np.asarray(ids)
    return HostPool(
        host_id=1, center_mpc=np.asarray(center, dtype=np.float64), rvir_mpc=rvir,
        mvir=mvir, vmax=200.0, n_members=int(ids.size), ids=ids,
        pos_mpc=particle_positions_mpc(field, ids, boxsize_mpc_h=BOX),
        vel_kms=particle_velocities_kms(field, ids),
        host_mean_vel_kms=np.zeros(3), boxsize_mpc_h=BOX,
    )


# ---------------------------------------------------------------------------
# Coordinate round trips
# ---------------------------------------------------------------------------


def test_gathered_positions_match_the_canonical_particle_builder():
    """The memory-light gather must be the same arithmetic, not merely similar.

    ``field_to_particles`` materialises the whole box; this module gathers a
    subset. If they ever disagree, every distance the editor computes is wrong
    in a way that still produces plausible-looking numbers.
    """
    f = make_field(1)
    ref = field_to_particles(f, boxsize_kpc_h=BOX * 1000.0, redshift=0.0)
    ids = np.array([0, 1, NG, NG * NG, NG ** 3 - 1, 137], dtype=np.int64)
    got = particle_positions_mpc(f, ids, boxsize_mpc_h=BOX)
    assert np.allclose(got, ref.pos_mpc_h[ids], atol=1e-4)
    v = particle_velocities_kms(f, ids)
    assert np.allclose(v, ref.vel_kms[ids], rtol=1e-5, atol=1e-3)


def test_displacement_write_back_round_trips():
    f = make_field(2)
    ids = np.array([3, 40, 900], dtype=np.int64)
    p0 = particle_positions_mpc(f, ids, boxsize_mpc_h=BOX)
    delta = np.array([[0.01, -0.02, 0.03]] * 3)
    add_displacement_mpc(f, ids, delta)
    p1 = particle_positions_mpc(f, ids, boxsize_mpc_h=BOX)
    assert np.allclose(min_image(p1 - p0, BOX), delta, atol=1e-5)


def test_a_config_that_disagrees_with_disnorm_is_refused():
    check_norm_convention(6000.0, 0.0)
    with pytest.raises(ValueError, match="do not re-derive"):
        check_norm_convention(3000.0, 0.0)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_minimum_image_wraps_the_short_way():
    d = np.array([[9.0, -9.0, 1.0]])
    assert np.allclose(min_image(d, BOX), [[-1.0, 1.0, 1.0]])


def test_contraction_across_a_box_face_moves_particles_the_short_way():
    """A proposal at x = 0.5 must pull a particle at x = 99.5 *forwards*, not
    99 Mpc/h backwards through the box."""
    center = np.array([0.5, 5.0, 5.0])
    p = np.array([[9.5, 5.0, 5.0]])
    dx = -0.5 * min_image(p - center, BOX)
    assert dx[0, 0] == pytest.approx(0.5)     # +0.5, i.e. towards 10 == 0


def test_the_window_is_one_in_the_core_and_exactly_zero_at_the_edge():
    u = np.array([0.0, 0.3, 0.5, 0.9, 1.0, 1.5])
    w = edge_window(u, 0.5)
    assert w[0] == 1.0 and w[1] == 1.0        # inside 1 - softness
    assert w[4] == 0.0 and w[5] == 0.0        # exactly zero at and beyond r = R
    assert 0.0 < w[3] < 1.0
    assert np.all(np.diff(w) <= 1e-12)        # monotone


def test_a_softer_edge_never_raises_the_window():
    u = np.linspace(0, 1, 51)
    assert np.all(edge_window(u, 1.0) <= edge_window(u, 0.2) + 1e-12)


# ---------------------------------------------------------------------------
# The codec
# ---------------------------------------------------------------------------


def test_every_real_vector_decodes_inside_the_bounds():
    codec = search_codec("both")
    rng = np.random.default_rng(0)
    for z in rng.normal(0.0, 40.0, size=(200, codec.dim)):
        vals = codec.decode(z)
        for p in codec.params:
            assert p.lo - 1e-12 <= vals[p.name] <= p.hi + 1e-12, p.name


def test_encode_is_the_inverse_of_decode_on_the_interior():
    codec = search_codec("both")
    rng = np.random.default_rng(1)
    z = rng.normal(0.0, 2.0, size=codec.dim)
    assert np.allclose(codec.encode(codec.decode(z)), z, atol=1e-6)


def test_encoding_a_value_at_a_bound_stays_finite():
    """An infinity in a manifest poisons every mean computed from it."""
    codec = search_codec("both")
    vals = {p.name: p.lo for p in codec.params}
    z = codec.encode(vals)
    assert np.all(np.isfinite(z))


def test_a_pinned_parameter_ignores_its_coordinate():
    codec = ActionCodec((ParamSpec("k", 0.0, 0.0), ParamSpec("j", 1.0, 2.0)))
    assert codec.decode([1e6, 0.0])["k"] == 0.0
    assert codec.encode({"k": 0.0, "j": 1.5})[0] == 0.0


@pytest.mark.parametrize("mode,zero", [("disp", "velocity_cooling"),
                                       ("vel", "contraction")])
def test_a_mode_pins_the_channel_it_is_not_allowed_to_touch(mode, zero):
    codec = search_codec(mode)
    rng = np.random.default_rng(2)
    for z in rng.normal(0.0, 5.0, size=(20, codec.dim)):
        assert codec.decode(z)[zero] == 0.0


def test_both_mode_is_displacement_dominant():
    codec = search_codec("both")
    cur = {p.name: p for p in codec.params}
    assert cur["velocity_cooling"].hi < cur["contraction"].hi


def test_the_search_vector_is_token_then_action():
    """The flow slices the action block off by index; the ordering is load-bearing."""
    assert SEARCH_PARAMS[-len(ACTION_PARAMS):] == ACTION_PARAMS
    assert len(ACTION_PARAMS) == 8


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------


def test_the_pool_excludes_subhalo_members_and_their_neighbourhoods():
    f = make_field(3)
    host = np.arange(500, dtype=np.int64)
    sub = np.arange(0, 50, dtype=np.int64)
    pos = particle_positions_mpc(f, host, boxsize_mpc_h=BOX)
    pool = build_host_pool(
        f, host_id=7, host_member_ids=host, subhalo_member_ids=[sub],
        subhalo_centers_mpc=pos[100:101], subhalo_radii_mpc=np.array([0.3]),
        center_mpc=(5.0, 5.0, 5.0), rvir_mpc=RVIR, mvir=1e13,
        boxsize_mpc_h=BOX)
    assert not np.isin(pool.ids, sub).any(), "subhalo members stayed in the pool"
    assert 100 not in set(pool.ids.tolist()), "a particle at a subhalo centre stayed"
    assert pool.n_excluded_sub_members == 50
    assert pool.n_excluded_near_sub >= 1
    assert pool.n_members == 500


def test_particle_count_follows_the_mass_ratio():
    f = make_field(4)
    pool = make_pool(f)
    t = SubhaloToken(1, log_mass_ratio=-2.0, radius_rvir=0.5, direction=(0, 0, 1))
    # 8000 members * 1e-2 = 80, inside the clamp
    assert n_particles_for_token(t, pool, n_min=10, n_max=400) == 80


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def test_exactly_the_requested_number_of_particles_is_claimed():
    f = make_field(5)
    pool = make_pool(f)
    t = SubhaloToken(1, -2.0, 0.2, (0.0, 0.0, 1.0))
    a = EditorAction((0, 0, 0), 0.30, 0.5, 0.0, 0.0, 0.5)
    plan = plan_edit(pool, t, a, n_particles=137)
    assert plan.ids.size == 137
    assert np.unique(plan.ids).size == 137


def test_the_claim_does_not_depend_on_pool_ordering():
    """Ties are broken by particle id, so a permuted pool claims the same set."""
    f = make_field(6)
    pool = make_pool(f)
    t = SubhaloToken(1, -2.0, 0.2, (0.0, 0.0, 1.0))
    a = EditorAction((0, 0, 0), 0.30, 0.5, 0.0, 0.0, 0.5)
    first = plan_edit(pool, t, a, n_particles=50).ids

    perm = np.random.default_rng(0).permutation(pool.ids.size)
    shuffled = HostPool(**{**pool.__dict__, "ids": pool.ids[perm],
                           "pos_mpc": pool.pos_mpc[perm],
                           "vel_kms": pool.vel_kms[perm]})
    assert set(plan_edit(shuffled, t, a, n_particles=50).ids.tolist()) == \
        set(first.tolist())


def test_a_periodic_translation_of_the_whole_problem_claims_the_same_particles():
    """Shifting host, centre and every particle by the same vector must not
    change which particles the proposal takes, including across the box face."""
    f = make_field(7)
    pool = make_pool(f)
    t = SubhaloToken(1, -2.0, 0.2, (0.3, 0.4, 0.86))
    a = EditorAction((0.01, 0.0, 0.0), 0.30, 0.5, 0.0, 0.0, 0.5)
    base = plan_edit(pool, t, a, n_particles=40)

    shift = np.array([60.0, -70.0, 33.0])
    moved = HostPool(**{**pool.__dict__,
                        "center_mpc": (pool.center_mpc + shift) % BOX,
                        "pos_mpc": (pool.pos_mpc + shift) % BOX})
    got = plan_edit(moved, t, a, n_particles=40)
    assert set(got.ids.tolist()) == set(base.ids.tolist())
    assert np.allclose(np.sort(got.weights), np.sort(base.weights))


def test_two_proposals_never_claim_the_same_particle():
    f = make_field(8)
    pools = {1: make_pool(f)}
    t1 = SubhaloToken(1, -2.0, 0.10, (0.0, 0.0, 1.0))
    t2 = SubhaloToken(1, -2.0, 0.12, (0.0, 0.0, 1.0))   # deliberately overlapping
    a = EditorAction((0, 0, 0), 0.30, 0.5, 0.0, 0.0, 0.5)
    plans = plan_edits(pools, [(t1, a), (t2, a)], n_max=60)
    assert set(plans[0].ids.tolist()).isdisjoint(plans[1].ids.tolist())
    assert plans[0].ids.size + plans[1].ids.size == \
        np.unique(np.concatenate([p.ids for p in plans])).size


def test_disjoint_proposals_are_order_independent():
    f = make_field(9)
    pools = {1: make_pool(f)}
    far = SubhaloToken(1, -2.5, 0.8, (1.0, 0.0, 0.0))
    near = SubhaloToken(1, -2.5, 0.1, (-1.0, 0.0, 0.0))
    a = EditorAction((0, 0, 0), 0.15, 0.5, 0.0, 0.0, 0.5)
    ab = plan_edits(pools, [(far, a), (near, a)], n_max=20)
    ba = plan_edits(pools, [(near, a), (far, a)], n_max=20)
    assert set(ab[0].ids.tolist()).isdisjoint(ab[1].ids.tolist())
    assert set(ab[0].ids.tolist()) == set(ba[1].ids.tolist())
    assert set(ab[1].ids.tolist()) == set(ba[0].ids.tolist())


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------


def test_zero_contraction_and_zero_cooling_return_sr2_bit_for_bit():
    """Not "close to": identical. A no-op that perturbs the last bit would make
    the frozen anchor a different box from the frozen baseline."""
    f = make_field(10)
    pools = {1: make_pool(f)}
    t = SubhaloToken(1, -2.0, 0.3, (0.0, 0.0, 1.0))
    a = EditorAction((0.05, 0.0, 0.0), 0.30, 0.0, 0.0, 0.5, 0.5)
    out, _, stats = apply_edits(f, pools, [(t, a)], boxsize_mpc_h=BOX)
    assert np.array_equal(out.view(np.uint8), f.view(np.uint8))
    assert stats[0]["n_moved"] == 0 and stats[0]["n_cooled"] == 0


def test_only_claimed_particles_change():
    f = make_field(11)
    pools = {1: make_pool(f)}
    t = SubhaloToken(1, -2.0, 0.2, (0.0, 0.0, 1.0))
    a = EditorAction((0, 0, 0), 0.30, 0.6, 0.3, 0.5, 0.5)
    out, plans, _ = apply_edits(f, pools, [(t, a)], boxsize_mpc_h=BOX)
    changed = np.nonzero(np.any(out.reshape(6, -1) != f.reshape(6, -1), axis=0))[0]
    active = set(plans[0].ids[plans[0].active].tolist())
    assert set(changed.tolist()) <= active
    assert len(active) > 0


def test_displacement_only_leaves_the_velocity_channels_untouched():
    f = make_field(12)
    pools = {1: make_pool(f)}
    t = SubhaloToken(1, -2.0, 0.2, (0.0, 0.0, 1.0))
    a = EditorAction((0, 0, 0), 0.30, 0.7, 0.0, 0.0, 0.5)
    out, _, _ = apply_edits(f, pools, [(t, a)], boxsize_mpc_h=BOX)
    assert np.array_equal(out[3:6], f[3:6])
    assert not np.array_equal(out[0:3], f[0:3])


def test_velocity_only_leaves_the_displacement_channels_untouched():
    f = make_field(13)
    pools = {1: make_pool(f)}
    t = SubhaloToken(1, -2.0, 0.2, (0.0, 0.0, 1.0))
    a = EditorAction((0, 0, 0), 0.30, 0.0, 0.7, 0.0, 0.5)
    out, _, _ = apply_edits(f, pools, [(t, a)], boxsize_mpc_h=BOX)
    assert np.array_equal(out[0:3], f[0:3])
    assert not np.array_equal(out[3:6], f[3:6])


def test_contraction_actually_makes_the_claimed_set_smaller():
    f = make_field(14)
    pools = {1: make_pool(f)}
    t = SubhaloToken(1, -2.0, 0.2, (0.0, 0.0, 1.0))
    a = EditorAction((0, 0, 0), 0.30, 0.8, 0.0, 0.0, 0.3)
    out, plans, _ = apply_edits(f, pools, [(t, a)], boxsize_mpc_h=BOX)
    ids = plans[0].ids[plans[0].active]
    before = plans[0].pos_mpc[plans[0].active]
    after = particle_positions_mpc(out, ids, boxsize_mpc_h=BOX)
    c = plans[0].center_mpc
    r0 = np.linalg.norm(min_image(before - c, BOX), axis=1)
    r1 = np.linalg.norm(min_image(after - c, BOX), axis=1)
    assert r1.mean() < r0.mean()
    assert np.all(r1 <= r0 + 1e-9)


def test_cooling_reduces_the_claimed_set_velocity_dispersion():
    f = make_field(15)
    pools = {1: make_pool(f)}
    t = SubhaloToken(1, -2.0, 0.2, (0.0, 0.0, 1.0))
    a = EditorAction((0, 0, 0), 0.30, 0.0, 0.8, 0.0, 0.3)
    out, plans, _ = apply_edits(f, pools, [(t, a)], boxsize_mpc_h=BOX)
    ids = plans[0].ids[plans[0].active]
    v0 = plans[0].vel_kms[plans[0].active]
    v1 = particle_velocities_kms(out, ids)
    assert v1.std(axis=0).mean() < v0.std(axis=0).mean()


def test_the_output_is_still_a_catnorm_float32_field():
    f = make_field(16)
    pools = {1: make_pool(f)}
    t = SubhaloToken(1, -2.0, 0.2, (0.0, 0.0, 1.0))
    a = EditorAction((0, 0, 0), 0.30, 0.5, 0.5, 0.5, 0.5)
    out, _, _ = apply_edits(f, pools, [(t, a)], boxsize_mpc_h=BOX)
    assert out.dtype == np.float32 and out.shape == f.shape
    assert np.isfinite(out).all()


def test_a_decoded_action_and_token_reproduce_the_proposal_centre():
    codec = search_codec("both")
    vals = codec.decode(np.zeros(codec.dim))
    t = token_from_values(1, vals)
    a = action_from_values(vals)
    pool = make_pool(make_field(17))
    c = proposal_center_mpc(pool, t, a)
    assert c.shape == (3,) and np.all((c >= 0) & (c < BOX))
    assert np.isclose(np.linalg.norm(np.asarray(t.direction)), 1.0)
