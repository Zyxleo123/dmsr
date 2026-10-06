"""Re-choosing each group's particles from the SR field (arms B and C).

What has to hold, and why each one would be invisible if it did not:

* **the targets do not move.** The replacement particles form no subhalo in HR,
  so the moment a run recomputes ``sets.ref`` on their ids it is supervising the
  statistics of a blob and every metric still looks fine.
* **the radius is what was requested.** The whole difference between arm B and
  arm C is one scalar per set. If the achieved centroid does not sit where it
  was asked to, the two arms are the same run twice.
* **no particle serves two groups**, or two candidates share material and the
  loss pulls it both ways.
* **a starved set keeps its HR membership and is flagged**, so the three arms
  supervise the same catalog and a B/C eval can still exclude the rows that were
  really arm A.
* **the loss runs on the result unchanged** -- the point of doing this as a pass
  over a finished pool rather than a branch inside the loss.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cosmo_sr.features.member_gather import (  # noqa: E402
    MemberGatherConfig, member_gather_loss, tile_particles,
)
from cosmo_sr.features.relocate_sets import (  # noqa: E402
    fibonacci_directions, relocate_member_sets,
)
from test_member_gather import BOX, DIS, HOST, NG, TILE, VEL, _sets  # noqa: E402


def _cfg(**kw):
    base = dict(min_num_p=4, min_purity=0.5, min_live_frac=0.5, bg_k=0,
                softening_kind="clamp", pot_chunk=64, relocate="frozen",
                relocate_dirs=6, relocate_eps=TOY_EPS)
    base.update(kw)
    return MemberGatherConfig(**base)


#: The toy host's R_vir in Mpc/h (catalog row 0 carries rvir 1000 kpc/h).
R_HOST = 1.0

#: The toy's tolerance, and it is NOT the production default of 0.05.
#: Achievable centroid precision is about ``r_rms / sqrt(N)``: the real run has
#: 0.195 Mpc/h between particles, r_rms ~ 0.4 and N ~ 200, so ~0.03 Mpc/h
#: against a cluster ``R_vir`` of ~2.1 -- comfortably inside 0.05 R_vir. The toy
#: puts 4096 particles on a 6.25 Mpc/h lattice across a 100 Mpc/h box, so the
#: same arithmetic gives ~0.3 R_host and a 0.05 tolerance would be testing the
#: toy's resolution rather than the selector.
TOY_EPS = 0.3


def _built(mode="frozen", split=False, **kw):
    cfg = _cfg(relocate=mode, **kw)
    ms, _, hr, fz, tiles = _sets(cfg, host_pos=HOST, split=split)
    fp, fv = tile_particles(fz, tiles, ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX,
                            dis_scale_mpc_h=DIS, vel_scale_kms=VEL)
    rep = {}
    out = relocate_member_sets(ms, fp, fv, cfg, host_rvir_mpc_h=R_HOST,
                               report=rep)
    return ms, out, rep, cfg, (hr, fz, tiles), (fp, fv)


# --------------------------------------------------------------------------- #
# Directions
# --------------------------------------------------------------------------- #
def test_fibonacci_directions_are_unit_and_spread():
    d = fibonacci_directions(32, seed=7)
    assert d.shape == (32, 3)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    # Near-uniform on the sphere: the centroid of 32 of them is near the origin,
    # which `n` random draws would not reliably satisfy at this count.
    assert np.linalg.norm(d.mean(axis=0)) < 0.15


def test_fibonacci_directions_are_deterministic_in_seed():
    assert np.allclose(fibonacci_directions(8, seed=3),
                       fibonacci_directions(8, seed=3))
    assert not np.allclose(fibonacci_directions(8, seed=3),
                           fibonacci_directions(8, seed=4))


# --------------------------------------------------------------------------- #
# The targets are HR's, untouched
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["frozen", "hr"])
def test_hr_targets_survive_relocation(mode):
    """Every reference is HR's own, on the ORIGINAL member set.

    The exception is ``bound_scale``, which is a temperature (``tau G m N /
    r_rms``) and not a target: ``N`` changes, so carrying the old one would
    scale the unbinding surrogate by the wrong particle count.
    """
    ms, out, _, _, _, _ = _built(mode, split=True)
    for k in ("r_rms", "sigma_v", "virial", "bound_soft", "bound_hard",
              "d6", "bg_occ"):
        assert torch.allclose(out.ref[k], ms.ref[k], equal_nan=True), k
    # `bound_scale` is exactly the old one rescaled by the particle-count ratio.
    ratio = (torch.as_tensor(out.n_live, dtype=torch.float64)
             / torch.as_tensor(ms.n_live, dtype=torch.float64))
    assert torch.allclose(out.ref["bound_scale"].double(),
                          ms.ref["bound_scale"].double() * ratio, rtol=1e-6)


def test_the_bound_temperature_follows_the_new_particle_count():
    """``bound_scale`` is ``tau G m N / r_rms`` and ``N`` changes when a set
    carried stragglers: the relocated group is wholly live, so it holds all
    ``N`` itself. Carrying the old temperature would scale the unbinding
    surrogate by the wrong particle count -- a silent 30% error in the one term
    that is Rockstar's own unbinding rule.

    The toy's ``build_member_sets`` runs with ``frozen_box=None``, so no set
    ever has stragglers and this has to be injected rather than arranged.
    """
    ms, _, hr, fz, tiles = _sets(_cfg(), host_pos=HOST)
    fp, fv = tile_particles(fz, tiles, ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX,
                            dis_scale_mpc_h=DIS, vel_scale_kms=VEL)
    n_extra = 17
    ms.fixed_pos[0] = ms.centre_ref[0][None, :].repeat(n_extra, 1).clone()
    ms.fixed_vel[0] = torch.zeros(n_extra, 3, dtype=ms.fixed_pos[0].dtype)
    before = int(ms.n_live[0])
    out = relocate_member_sets(ms, fp, fv, _cfg(), host_rvir_mpc_h=R_HOST)
    assert int(out.n_live[0]) == before + n_extra
    assert out.fixed_pos[0] is None
    assert float(out.ref["bound_scale"][0]) == pytest.approx(
        float(ms.ref["bound_scale"][0]) * (before + n_extra) / before, rel=1e-6)


@pytest.mark.parametrize("mode", ["frozen", "hr"])
def test_the_same_objects_in_the_same_order(mode):
    """Arms A, B and C must supervise one catalog, or their gates cannot be
    read against each other."""
    ms, out, _, _, _, _ = _built(mode)
    assert np.array_equal(out.halo_id, ms.halo_id)
    assert np.array_equal(out.num_p, ms.num_p)
    assert np.array_equal(out.parent_id, ms.parent_id)
    assert np.array_equal(out.iso_radius, ms.iso_radius)
    assert out.n_sets == ms.n_sets


# --------------------------------------------------------------------------- #
# The groups themselves
# --------------------------------------------------------------------------- #
def test_groups_hold_the_requested_particle_count():
    """``N_j`` is the size of the collection that produced the reference --
    live members plus the out-of-tile stragglers -- and the relocated group is
    wholly live, so it must hold all of them and carry no stragglers."""
    ms, out, _, _, _, _ = _built("frozen", split=True, min_live_frac=0.5)
    n_out = np.array([0 if f is None else int(f.shape[0])
                      for f in ms.fixed_pos])
    want = np.asarray(ms.n_live) + n_out
    for s in range(out.n_sets):
        if not out.relocated[s]:
            continue
        assert int(out.live_rows[s].numel()) == int(want[s])
        assert out.fixed_pos[s] is None and out.fixed_vel[s] is None
        assert int(out.n_live[s]) == int(want[s])


def test_no_particle_serves_two_groups():
    _, out, _, _, _, _ = _built("frozen")
    taken = np.concatenate([out.live_rows[s].numpy()
                            for s in range(out.n_sets) if out.relocated[s]])
    assert taken.size == np.unique(taken).size


def test_rows_are_sorted_and_inside_the_particle_table():
    _, out, _, _, _, (fp, _) = _built("frozen")
    for s in range(out.n_sets):
        r = out.live_rows[s].numpy()
        assert np.all(np.diff(r) > 0)             # sorted, no duplicates
        assert r.min() >= 0 and r.max() < int(fp.shape[0])


# --------------------------------------------------------------------------- #
# The constraint the arms differ by
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["frozen", "hr"])
def test_achieved_centroid_sits_at_the_requested_radius(mode):
    """The proposal's step 3, checked on the SELECTED PARTICLES rather than on
    the seed: a correct seed can carry a displaced centroid, and that is
    precisely the failure the check exists to catch."""
    ms, out, rep, cfg, _, (fp, _) = _built(mode)
    src = ms.centre_self if mode == "frozen" else ms.centre_target
    for s in range(out.n_sets):
        if not out.relocated[s]:
            continue
        host = ms.centre_host[s].numpy()
        rho = float(np.linalg.norm(src[s].numpy() - host)) / R_HOST
        # Recompute the centroid from the rows, not from the stored field: this
        # must be a statement about the particles, not about the bookkeeping.
        p = fp[out.live_rows[s]].numpy().astype(np.float64)
        ctr = out.centre_ref[s].numpy()
        p = ctr + np.remainder(p - ctr + 0.5 * BOX, BOX) - 0.5 * BOX
        got = float(np.linalg.norm(p.mean(axis=0) - host)) / R_HOST
        assert abs(got - rho) <= cfg.relocate_eps
    assert rep["frac_within_eps"] == pytest.approx(1.0)


def test_the_two_arms_request_different_radii():
    """Arm B and arm C are one scalar per set apart. If the toy's frozen and HR
    radii coincided the parametrised tests above would pass vacuously."""
    ms, _, rep_b, _, _, _ = _built("frozen")
    _, _, rep_c, _, _, _ = _built("hr")
    assert rep_b["median_requested_radius_radii"] != pytest.approx(
        rep_c["median_requested_radius_radii"])
    assert rep_b["relocate"] == "frozen" and rep_c["relocate"] == "hr"


def test_centre_self_is_the_achieved_centroid_and_target_is_the_sphere():
    """``centre_mode="self"`` -- the mode every working arm runs -- anchors the
    group where it was put, so its term must be exactly zero at step 0 for a
    relocated set exactly as it is for an HR one."""
    _, out, _, _, _, (fp, _) = _built("frozen")
    for s in range(out.n_sets):
        if not out.relocated[s]:
            continue
        ctr = out.centre_ref[s].numpy()
        p = fp[out.live_rows[s]].numpy().astype(np.float64)
        p = ctr + np.remainder(p - ctr + 0.5 * BOX, BOX) - 0.5 * BOX
        assert np.allclose(p.mean(axis=0), out.centre_self[s].numpy(), atol=1e-6)
    # The target sits on the requested sphere, so the residual radius error is
    # exactly |self - target| and nothing else.
    host = out.centre_host.numpy()
    tgt = out.centre_target.numpy()
    slf = out.centre_self.numpy()
    rel = out.relocated
    assert np.allclose(np.linalg.norm(tgt[rel] - host[rel], axis=1),
                       np.linalg.norm(tgt[rel] - host[rel], axis=1))
    assert np.all(np.linalg.norm(slf[rel] - tgt[rel], axis=1)
                  <= TOY_EPS * R_HOST + 1e-9)


def test_rhat_is_a_unit_vector_from_the_host():
    _, out, _, _, _, _ = _built("hr")
    rel = out.relocated
    n = np.linalg.norm(out.centre_rhat.numpy()[rel], axis=1)
    assert np.allclose(n, 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# Starvation is recorded, never silent
# --------------------------------------------------------------------------- #
def test_a_starved_set_keeps_its_hr_membership_and_is_flagged():
    """Asking for more particles than the tile holds starves every direction.

    The set must survive as an arm-A row rather than vanish: a dropped object
    would make the three arms supervise different catalogs, and their Rockstar
    gates would then differ for a reason that has nothing to do with the loss.
    """
    ms, _, hr, fz, tiles = _sets(_cfg(), host_pos=HOST)
    fp, fv = tile_particles(fz, tiles, ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX,
                            dis_scale_mpc_h=DIS, vel_scale_kms=VEL)
    # One particle in the table: every group is bigger than the whole pool.
    rep = {}
    out = relocate_member_sets(ms, fp[:1], fv[:1], _cfg(),
                               host_rvir_mpc_h=R_HOST, report=rep)
    assert rep["n_starved"] == ms.n_sets
    assert rep["n_relocated"] == 0
    assert not out.relocated.any()
    for s in range(out.n_sets):
        assert torch.equal(out.live_rows[s], ms.live_rows[s])
    assert torch.allclose(out.ref["bound_scale"], ms.ref["bound_scale"])


def test_report_carries_the_pre_training_check():
    _, _, rep, _, _, _ = _built("frozen")
    for k in ("relocate", "n_sets", "n_relocated", "n_starved",
              "frac_within_eps", "radius_err_radii", "rrms_over_hr",
              "angular_move_deg", "kept_hr_member_frac", "n_particles_claimed"):
        assert k in rep, k
    assert set(rep["radius_err_radii"]) == {"p50", "p90"}


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_relocation_refuses_a_pool_with_no_host():
    cfg = _cfg()
    ms, _, _, fz, tiles = _sets(cfg)          # host_pos=None -> centre_host 0
    fp, fv = tile_particles(fz, tiles, ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX,
                            dis_scale_mpc_h=DIS, vel_scale_kms=VEL)
    with pytest.raises(ValueError, match="centre_host is all zero"):
        relocate_member_sets(ms, fp, fv, cfg, host_rvir_mpc_h=R_HOST)


def test_relocation_refuses_a_nonpositive_host_radius():
    ms, _, _, fz, tiles = _sets(_cfg(), host_pos=HOST)
    fp, fv = tile_particles(fz, tiles, ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX,
                            dis_scale_mpc_h=DIS, vel_scale_kms=VEL)
    with pytest.raises(ValueError, match="host_rvir_mpc_h"):
        relocate_member_sets(ms, fp, fv, _cfg(), host_rvir_mpc_h=0.0)


def test_relocation_refuses_mode_off():
    ms, _, _, fz, tiles = _sets(_cfg(), host_pos=HOST)
    fp, fv = tile_particles(fz, tiles, ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX,
                            dis_scale_mpc_h=DIS, vel_scale_kms=VEL)
    with pytest.raises(ValueError, match="'off' means the caller"):
        relocate_member_sets(ms, fp, fv, _cfg(relocate="off"),
                             host_rvir_mpc_h=R_HOST)


def test_config_refuses_relocation_with_soft_membership():
    with pytest.raises(ValueError, match="both replace the member set"):
        MemberGatherConfig(relocate="hr", soft_membership=True, soft_lambda=1.0)


# --------------------------------------------------------------------------- #
# The loss runs on it unchanged
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["frozen", "hr"])
def test_the_existing_loss_runs_on_relocated_sets(mode):
    """The whole point of doing this as a pass over a finished pool: nothing in
    the loss knows the membership was re-chosen."""
    _, out, _, cfg, (hr, fz, tiles), _ = _built(mode, bg_k=32,
                                                bg_radius_factor=8.0)
    pos, vel = tile_particles(fz, tiles, ng_hr=NG, tile_hr=TILE,
                              boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                              vel_scale_kms=VEL)
    pos = pos.detach().requires_grad_(True)
    loss, diag = member_gather_loss(pos, vel, out, cfg)
    assert np.isfinite(float(loss.detach()))
    loss.backward()
    g = pos.grad
    assert torch.isfinite(g).all()
    # The gradient must reach the RE-SELECTED particles, not HR's.
    for s in range(out.n_sets):
        if out.relocated[s]:
            assert float(g[out.live_rows[s]].abs().sum()) > 0.0


def test_centre_self_term_starts_at_zero_on_relocated_sets():
    """``centre_mode="self"`` anchors a group where relocation put it, so the
    term is zero at step 0 by construction -- the same property the HR-member
    arm has, and the reason the run is an anchor rather than a pull."""
    _, out, _, cfg, (_, fz, tiles), _ = _built(
        "hr", centre_mode="self", w_centre=1.0)
    import dataclasses
    cfg = dataclasses.replace(cfg, centre_mode="self", w_centre=1.0,
                              w_virial=0.0, w_bound=0.0, w_d6=0.0,
                              w_rrms=0.0, w_sigmav=0.0)
    pos, vel = tile_particles(fz, tiles, ng_hr=NG, tile_hr=TILE,
                              boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                              vel_scale_kms=VEL)
    loss, diag = member_gather_loss(pos, vel, out, cfg)
    assert float(loss) == pytest.approx(0.0, abs=1e-10)


# --------------------------------------------------------------------------- #
# The gate has to look in the right place
# --------------------------------------------------------------------------- #
def test_diag_rows_carry_the_relocated_target_position():
    """``compare_gather_catalog`` searches around the HR subhalo's catalog
    centre. That is the right address for arm A and the WRONG one for B and C,
    which put the group somewhere else on purpose -- so the run has to say where
    it put it, and only for the sets it actually moved."""
    _, out, _, cfg, (_, fz, tiles), _ = _built("hr")
    pos, vel = tile_particles(fz, tiles, ng_hr=NG, tile_hr=TILE,
                              boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                              vel_scale_kms=VEL)
    _, diag = member_gather_loss(pos, vel, out, cfg)
    assert len(diag["rows"]) == out.n_sets
    for r, s in zip(diag["rows"], range(out.n_sets)):
        assert r["relocated"] is bool(out.relocated[s])
        assert ("target_pos" in r) is bool(out.relocated[s])
        if r["relocated"]:
            assert len(r["target_pos"]) == 3
            assert all(0.0 <= float(x) < BOX for x in r["target_pos"])
            assert np.allclose(
                np.remainder(out.centre_target[s].numpy(), BOX),
                np.asarray(r["target_pos"]), atol=1e-6)


def test_arm_a_rows_name_no_target_position():
    """Arm A must stay bit-identical: an emitted ``target_pos`` would move its
    hit test off the HR catalog centre and silently restate every number on
    disk."""
    cfg = _cfg(relocate="off")
    ms, _, _, fz, tiles = _sets(cfg, host_pos=HOST)
    pos, vel = tile_particles(fz, tiles, ng_hr=NG, tile_hr=TILE,
                              boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                              vel_scale_kms=VEL)
    _, diag = member_gather_loss(pos, vel, ms, cfg)
    for r in diag["rows"]:
        assert r["relocated"] is False
        assert "target_pos" not in r


def test_target_hit_rate_searches_the_relocated_position():
    """The comparator's override, end to end on a two-halo toy catalog."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                           / "scripts" / "reward"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from compare_gather_catalog import target_hit_rate
    from types import SimpleNamespace

    hr = SimpleNamespace(ids=np.array([7]), pos=np.array([[10.0, 10.0, 10.0]]),
                         rvir=np.array([200.0]), num_p=np.array([100]))
    # The only halo in the candidate catalog sits 5 Mpc/h away from HR's centre.
    cand = SimpleNamespace(ids=np.array([1]),
                           pos=np.array([[15.0, 10.0, 10.0]]),
                           rvir=np.array([200.0]), num_p=np.array([100]))
    miss = target_hit_rate(cand, hr, [7])
    assert miss["hits"] == 0
    assert miss["rows"][0]["searched_at"] == "hr_catalog"
    hit = target_hit_rate(cand, hr, [7], target_pos={7: [15.0, 10.0, 10.0]})
    assert hit["hits"] == 1
    assert hit["rows"][0]["searched_at"] == "relocated"


def test_the_local_swaps_improve_the_radius_error():
    """The proposal's step 3 in one number: nearest-neighbour selection alone
    leaves a displaced centroid, and the swap passes are what pull it back.

    Pinned because the correction is a Newton step on the QUERY point
    (``q <- q + (aim - achieved)``), not a re-query at the aim point. The two
    read the same in a diff and only the first removes the local-density bias,
    so a silent regression to the second would show up here and nowhere else.
    """
    _, _, raw, _, _, _ = _built("frozen", relocate_refine_iters=0)
    _, _, ref, _, _, _ = _built("frozen", relocate_refine_iters=3)
    assert ref["radius_err_radii"]["p50"] < 0.75 * raw["radius_err_radii"]["p50"]
