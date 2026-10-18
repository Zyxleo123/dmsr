"""The host as a background component: does the solve stop preferring the core?

What has to hold, and why each one is here.

*The flat background must survive inside it.* Every arm on disk ran with
``logbg = log(1 - claimed)``. ``soft_host_weight=0`` has to reproduce that
solve to the bit, or none of those results are comparable with the ones this
term produces, and the dose ladder has no zero.

*The profile must be a profile.* NFW's two asymptotic slopes (``-1`` inside
``r_s``, ``-3`` outside) and the empirical estimator's recovery of a known
power law are the only checks that the thing being subtracted is the density
and not, say, the enclosed mass -- which is also monotone and would be wrong
in exactly the direction that is hard to notice.

*The term must move material outward.* That is the claim, and it is the one
thing no amount of correct interpolation implies. Asserted as a monotone
response to ``beta``, because ``radial-placement-is-inverted`` is a statement
about a median radius and a single-``beta`` improvement is what the shell
family already produced and could not repeat on held-out data.

*An unknown host must be left alone.* The LR catalog is incomplete below ~20k
HR-equivalent particles (``lr-host-channel-fidelity``). A set whose host misses
that gate has to fall back to the flat background exactly, not to an NFW of
zero mass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cosmo_sr.features.host_profile import (  # noqa: E402
    build_host_profile, empirical_log_density, nfw_log_density,
)
from cosmo_sr.features.member_gather import MemberGatherConfig  # noqa: E402
from cosmo_sr.features.soft_membership import (  # noqa: E402
    build_soft_pools, solve_memberships,
)

from test_member_gather import BOX, HOST, NG, TILE, _cfg, _particles, _sets  # noqa: E402
from test_member_gather import _toy as _mg_toy  # noqa: E402
from test_soft_membership import F, _scfg  # noqa: E402


def _host_cfg(**kw):
    base = dict(soft_pool_ratio=1.0, soft_steps=2, soft_mass_steps=4)
    base.update(kw)
    return _scfg(**base)


def _toy(cfg, rvir=6.0, mvir=1e14, host_pos=HOST):
    """Sets + pools + the frozen particle table, with a host of known size.

    ``build_member_sets`` takes the host's ``R_vir``/``M_vir`` as arguments, but
    the toy catalog has no host row worth reading, so they are written on
    afterwards -- the field is plain data and the solve reads it through
    ``getattr``.
    """
    sets, _, _, fz, tiles = _sets(cfg, host_pos=host_pos)
    sets.host_rvir = np.full(sets.n_sets, float(rvir))
    sets.host_mvir = np.full(sets.n_sets, float(mvir))
    # `parent` reads the same size off the set's own parent; the toy's parent
    # IS the task host, so the two sources see one host here.
    sets.parent_rvir = np.full(sets.n_sets, float(rvir))
    sets.parent_mvir = np.full(sets.n_sets, float(mvir))
    sets.centre_parent = sets.centre_host.clone()
    pools = build_soft_pools(sets, tiles, cfg, ng_hr=NG, tile_hr=TILE,
                             factor=F, seed=0)
    sets.pools = pools
    pos, vel = _particles(fz, tiles)
    return sets, pools, pos, vel


def _weighted_radius(pos, sets, w, pools, idx, centre):
    """Weighted mean host-centric radius of the solved membership, Mpc/h."""
    num = den = 0.0
    c = torch.as_tensor(centre, dtype=pos.dtype)
    for k, s in enumerate(idx):
        p = pos[pools.rows[s]]
        r = torch.sqrt(((p - c) ** 2).sum(1) + 1e-12)
        num += float((w[k] * r).sum())
        den += float(w[k].sum())
    return num / max(den, 1e-12)


# --------------------------------------------------------------------------- #
# The profile is a density
# --------------------------------------------------------------------------- #
def test_nfw_has_the_two_asymptotic_slopes():
    """``-1`` well inside ``r_s``, ``-3`` well outside.

    ``conc`` is deliberately tiny here, not the 5 the runs use: the inner
    asymptote lives at ``r << r_s``, and ``R_FLOOR_FRAC`` clamps at
    ``0.02 R_vir = 0.02 c r_s``, so at ``c = 5`` the smallest radius the
    function will ever evaluate is already ``0.1 r_s`` and the true ``-1``
    slope is unreachable by construction. That clamp is wanted -- an
    unclamped NFW is an infinity inside a softmax -- so the asymptote is
    checked where it exists rather than the clamp being weakened to suit
    the test.
    """
    rv, c = 1.0, 0.5
    rs = rv / c
    r = np.array([0.02 * rv, 0.04 * rv, 20.0 * rs, 40.0 * rs])
    y = nfw_log_density(r, rvir_mpc_h=rv, mvir_msun_h=1e14, conc=c)
    inner = (y[1] - y[0]) / (np.log(r[1]) - np.log(r[0]))
    outer = (y[3] - y[2]) / (np.log(r[3]) - np.log(r[2]))
    assert inner == pytest.approx(-1.0, abs=0.05)
    assert outer == pytest.approx(-3.0, abs=0.15)


def test_nfw_is_strictly_decreasing_and_scales_with_mass():
    r = np.geomspace(0.05, 4.0, 40)
    y = nfw_log_density(r, rvir_mpc_h=1.0, mvir_msun_h=1e14, conc=5.0)
    assert np.all(np.diff(y) < 0.0)
    y2 = nfw_log_density(r, rvir_mpc_h=1.0, mvir_msun_h=1e15, conc=5.0)
    assert np.allclose(y2 - y, np.log(10.0))


def test_empirical_recovers_a_known_power_law():
    """An ``r^-2`` cloud must come back with slope ``-2`` in the fitted range."""
    rng = np.random.default_rng(0)
    n = 400_000
    # r ~ U^(1/1) for rho ~ r^-2 (since dN/dr ~ r^2 rho = const)
    r = rng.uniform(0.02, 4.0, size=n)
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    pos = u * r[:, None] + 50.0
    nodes, lg = empirical_log_density(
        pos, np.array([50.0, 50.0, 50.0]), boxsize_mpc_h=100.0,
        r_min_mpc_h=0.05, r_max_mpc_h=3.0, n_bins=20)
    slope = np.polyfit(np.log(nodes), lg, 1)[0]
    assert slope == pytest.approx(-2.0, abs=0.1)


def test_interpolation_hits_the_nodes_and_continues_the_end_slope():
    cfg = _host_cfg(soft_host_profile="lr", soft_host_weight=1.0,
                    soft_host_bins=16)
    sets, _, _, _ = _toy(cfg)
    prof = build_host_profile(sets, [0], cfg=cfg)
    r = prof.r_nodes[0]
    got = prof.log_rho(r, torch.zeros(r.numel(), dtype=torch.long))
    assert torch.allclose(got, prof.log_rho_nodes[0], atol=1e-5)
    # Outside the table the last slope continues, so a far particle is still
    # ranked as underdense rather than clamped to the edge value.
    far = torch.tensor([float(r[-1]) * 4.0])
    assert float(prof.log_rho(far, torch.zeros(1, dtype=torch.long))) < float(
        prof.log_rho_nodes[0, -1])


# --------------------------------------------------------------------------- #
# The zero of the dose ladder
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("src", ["lr", "self", "parent"])
def test_beta_zero_reproduces_the_flat_background_exactly(src):
    off = _host_cfg()
    sets, pools, pos, vel = _toy(off)
    idx = list(range(sets.n_sets))
    w_off, d_off = solve_memberships(pos, vel, sets, pools, idx, off)

    on = _host_cfg(soft_host_profile=src, soft_host_weight=0.0)
    w_on, d_on = solve_memberships(pos, vel, sets, pools, idx, on)
    for a, b in zip(w_off, w_on):
        assert torch.equal(a, b)
    assert d_on["soft_mass_ratio"] == d_off["soft_mass_ratio"]


def test_an_ungated_host_falls_back_to_the_flat_background():
    """``rvir = 0`` is "the LR catalog does not have this host", not a host."""
    cfg = _host_cfg(soft_host_profile="lr", soft_host_weight=1.0)
    sets, pools, pos, vel = _toy(cfg, rvir=0.0, mvir=0.0)
    idx = list(range(sets.n_sets))
    w_on, diag = solve_memberships(pos, vel, sets, pools, idx, cfg)
    assert diag["soft_host_valid_frac"] == 0.0

    off = _host_cfg()
    w_off, _ = solve_memberships(pos, vel, sets, pools, idx, off)
    for a, b in zip(w_off, w_on):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# The claim: material moves outward
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("src", ["lr", "self", "parent"])
def test_beta_moves_the_membership_outward(src):
    """The median host-centric radius of the solved membership rises with beta.

    This is the whole point of the term and the only assertion here that a
    correct-but-useless implementation would fail.
    """
    # The host sits ON one of the toy's subhalos, so that set's pool spans a
    # real range of host-centric radius. At the toy's default geometry every
    # pool sits 17 Mpc/h from the host and subtends almost no radial range at
    # all -- there is then nothing for a radial term to move, and the test
    # would be measuring rounding.
    #
    # Measured PER SET, not pooled. Pooling the two sets measures the mass
    # SPLIT between a pool at r ~ 1.5 and a pool at r ~ 34, which the dual
    # moves around for reasons that have nothing to do with the radial term,
    # and that split swamps the within-pool redistribution this term is.
    centre = np.array([10.0, 10.0, 10.0])
    rs = []
    for beta in (0.0, 1.0, 3.0):
        cfg = _host_cfg(soft_host_profile=src, soft_host_weight=beta)
        sets, pools, pos, vel = _toy(cfg, rvir=1.0, host_pos=centre)
        w, _ = solve_memberships(pos, vel, sets, pools,
                                 list(range(sets.n_sets)), cfg)
        rs.append(_weighted_radius(pos, sets, [w[0]], pools, [0], centre))
    assert rs[-1] > rs[0], rs
    # Non-decreasing rather than strictly increasing: `"self"` SATURATES. Its
    # empirical profile of a toy blob spans ~34 nats peak-to-trough against
    # NFW's ~10, so beta=1 has already pushed every core particle onto the
    # background and beta=3 has nothing left to move. That is also why beta is
    # NOT comparable between the two sources, and why `soft_host_logrho_p10/p90`
    # are reported per solve: a dose ladder on this term has to be read in
    # nats, not in beta.
    assert all(b >= a - 1e-9 for a, b in zip(rs, rs[1:])), rs


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_profile_without_soft_membership_is_refused():
    with pytest.raises(ValueError, match="only has a meaning"):
        MemberGatherConfig(soft_membership=False, soft_host_profile="lr")


def test_negative_weight_is_refused():
    with pytest.raises(ValueError, match="negative"):
        MemberGatherConfig(soft_membership=True, soft_host_profile="lr",
                           soft_host_weight=-1.0)


def test_unknown_source_is_refused():
    with pytest.raises(ValueError, match="not 'off', 'lr', 'self' or 'parent'"):
        MemberGatherConfig(soft_membership=True, soft_host_profile="nfw")


def test_no_host_position_is_refused():
    """Host-preservation sets are built ``host_pos=None``; the profile would
    otherwise be measured about the origin of the box."""
    cfg = _host_cfg(soft_host_profile="self", soft_host_weight=1.0)
    sets, _, _, _, tiles = _sets(cfg, host_pos=None)
    with pytest.raises(ValueError, match="every centre_host is zero"):
        build_host_profile(sets, [0], cfg=cfg,
                           pos_mpc_h=np.zeros((4, 3)))


def test_the_self_profile_is_built_once_per_host(monkeypatch):
    """The solve runs every training step and the empirical estimator is O(N)
    over the whole particle table. A task supervises ONE host, so rebuilding
    the identical profile per set would make the arm pay S times the cost of
    the thing it measures -- invisible in every correctness check, which is
    why it is pinned here."""
    import cosmo_sr.features.host_profile as hp

    calls = []
    real = hp.empirical_log_density

    def counted(*a, **kw):
        calls.append(kw.get("r_max_mpc_h"))
        return real(*a, **kw)

    monkeypatch.setattr(hp, "empirical_log_density", counted)
    cfg = _host_cfg(soft_host_profile="self", soft_host_weight=1.0)
    sets, _, pos, _ = _toy(cfg, host_pos=np.array([10.0, 10.0, 10.0]))
    idx = list(range(sets.n_sets))
    prof = hp.build_host_profile(sets, idx, cfg=cfg,
                                 pos_mpc_h=pos.detach().cpu().numpy())
    assert bool(prof.valid.all())
    # Both toy sets share the host; the two ranges differ only if their
    # centre_scale does, so at most one histogram per distinct range.
    assert len(calls) == len(set(calls)), calls
    assert len(calls) < len(idx) + 1


def test_the_host_guard_config_survives_a_host_profile_arm():
    """`--w-host-sets` derives a `soft_membership=False` config from the arm's
    own, because a PRESERVATION constraint must be an identity statement and
    cannot be satisfied by a different collection of particles. That derived
    config must not trip the refusal aimed at a user who set the profile and
    forgot `--soft-membership`.

    This is not hypothetical: jobs 56560 and 56562 both died here five minutes
    in, at the first eval, on the recommended baseline (which turns the host
    guard ON)."""
    import dataclasses

    from cosmo_sr.features.member_gather import host_guard_config

    # The trainer's OWN derivation, not a copy of it: a copy is how job 84183
    # died -- the radial prior was added to the arm but not to the guard.
    arm = _host_cfg(soft_host_profile="parent", soft_host_weight=0.5,
                    soft_radial_weight=1.0, soft_radial_width=0.2)
    guard = host_guard_config(arm)
    assert guard.soft_membership is False
    assert guard.soft_host_profile == "off"
    assert guard.soft_radial_weight == 0.0
    # And the refusal still fires for the combination a person can type.
    with pytest.raises(ValueError, match="only has a meaning"):
        dataclasses.replace(arm, soft_membership=False)


# --------------------------------------------------------------------------- #
# `parent`: each set's OWN host, not the task's
# --------------------------------------------------------------------------- #
def test_build_member_sets_records_each_sets_own_parent():
    """The toy catalog's subhalos all have parent id 1 (row 0): the recorded
    parent centre must be that row's position on the set's branch, and its
    R_vir/M_vir that row's -- independent of the task host passed in."""
    cfg = _host_cfg()
    far = np.array([90.0, 90.0, 90.0])
    sets, *_ = _sets(cfg, host_pos=far)
    cat = _mg_toy()[0]
    assert sets.centre_parent.shape == (sets.n_sets, 3)
    row = int(np.flatnonzero(np.asarray(cat.ids) == 1)[0])
    want = np.asarray(cat.pos[row], dtype=np.float64)
    got = sets.centre_parent.cpu().numpy()
    d = got - want[None, :]
    d -= BOX * np.round(d / BOX)
    assert np.allclose(d, 0.0, atol=1e-6)
    assert np.allclose(sets.parent_rvir, float(cat.rvir[row]) / 1000.0)
    assert np.allclose(sets.parent_mvir, float(cat.mvir[row]))
    # ...and NOT the task host, which is what `lr`/`self` are centred on.
    assert not np.allclose(got, sets.centre_host.cpu().numpy())


def test_parent_profile_ignores_the_task_host():
    """Moving the TASK host must not change a `parent` solve at all; it must
    change an `lr` solve (which is centred on it)."""
    cfg = _host_cfg(soft_host_profile="parent", soft_host_weight=1.0)
    lr = _host_cfg(soft_host_profile="lr", soft_host_weight=1.0)
    centre = np.array([10.0, 10.0, 10.0])
    sets, pools, pos, vel = _toy(cfg, rvir=1.0, host_pos=centre)
    idx = list(range(sets.n_sets))
    w_a, _ = solve_memberships(pos, vel, sets, pools, idx, cfg)
    w_lr_a, _ = solve_memberships(pos, vel, sets, pools, idx, lr)
    sets.centre_host = sets.centre_host + 5.0      # the task host moves away
    w_b, _ = solve_memberships(pos, vel, sets, pools, idx, cfg)
    w_lr_b, _ = solve_memberships(pos, vel, sets, pools, idx, lr)
    for a, b in zip(w_a, w_b):
        assert torch.equal(a, b)
    assert any(not torch.equal(a, b) for a, b in zip(w_lr_a, w_lr_b))


def test_a_set_without_a_parent_falls_back_to_flat():
    cfg = _host_cfg(soft_host_profile="parent", soft_host_weight=1.0)
    sets, pools, pos, vel = _toy(cfg)
    sets.parent_rvir = np.zeros(sets.n_sets)
    sets.parent_mvir = np.zeros(sets.n_sets)
    idx = list(range(sets.n_sets))
    w_on, diag = solve_memberships(pos, vel, sets, pools, idx, cfg)
    assert diag["soft_host_valid_frac"] == 0.0
    w_off, _ = solve_memberships(pos, vel, sets, pools, idx, _host_cfg())
    for a, b in zip(w_off, w_on):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# The per-set radial prior (soft_radial_weight)
# --------------------------------------------------------------------------- #
def _radial_dev(pos, sets, w, pools, idx):
    """Weighted mean |r - r_HR| of the solved membership, r from the parent."""
    num = den = 0.0
    for k, s in enumerate(idx):
        cp = sets.centre_parent[s]
        r_t = float(torch.linalg.vector_norm(sets.centre_ref[s] - cp))
        p = pos[pools.rows[s]]
        r = torch.sqrt(((p - cp.to(p.dtype)) ** 2).sum(1) + 1e-12)
        num += float((w[k] * (r - r_t).abs()).sum())
        den += float(w[k].sum())
    return num / max(den, 1e-12)


def _radial_toy(cfg, offset=1.0):
    """Each set's parent sits `offset` Mpc/h from its HR centre, along +x, so
    the HR radius is exactly `offset` and the pool straddles it."""
    sets, pools, pos, vel = _toy(cfg, rvir=1.0)
    shift = torch.zeros_like(sets.centre_ref)
    shift[:, 0] = float(offset)
    sets.centre_parent = (sets.centre_ref + shift).clone()
    sets.parent_rvir = np.full(sets.n_sets, 1.0)
    return sets, pools, pos, vel


def test_radial_weight_zero_is_bit_identical():
    off = _host_cfg()
    sets, pools, pos, vel = _radial_toy(off)
    idx = list(range(sets.n_sets))
    w_off, d_off = solve_memberships(pos, vel, sets, pools, idx, off)
    on = _host_cfg(soft_radial_weight=0.0, soft_radial_width=0.1)
    w_on, _ = solve_memberships(pos, vel, sets, pools, idx, on)
    for a, b in zip(w_off, w_on):
        assert torch.equal(a, b)


def test_radial_prior_pulls_material_to_the_hr_radius():
    """The claim: the solved material's distance from its own HR radius falls
    as gamma rises, at the real solve's step counts and a width the toy's
    pool can actually supply, without giving up the mass constraint."""
    devs, ratios = [], []
    for gam in (0.0, 1.0, 4.0):
        cfg = _host_cfg(soft_radial_weight=gam, soft_radial_width=0.5,
                        soft_steps=3, soft_mass_steps=10)
        sets, pools, pos, vel = _radial_toy(cfg)
        idx = list(range(sets.n_sets))
        w, diag = solve_memberships(pos, vel, sets, pools, idx, cfg)
        devs.append(_radial_dev(pos, sets, w, pools, idx))
        ratios.append(diag["soft_mass_ratio"])
    assert devs[-1] < devs[0], devs
    assert all(b <= a + 1e-9 for a, b in zip(devs, devs[1:])), devs
    # A moderate prior keeps the mass; a strong one TRADES it (measured on the
    # toy: 0.97 / 0.93 / 0.77 at gamma 0 / 1 / 4). That trade is real -- the
    # pool runs short of material near the HR radius -- so only gamma=1 is
    # held to the baseline, and the real field's ratio is a pre-test reading.
    assert ratios[1] > ratios[0] - 0.1, ratios


def test_a_too_narrow_radial_prior_starves_and_says_so():
    """When the pool holds too little material within `width` of the HR
    radius, the mass target cannot be met. The solve must report that in
    `soft_mass_ratio` rather than hide it -- it is what the pre-test reads."""
    wide = _host_cfg(soft_radial_weight=10.0, soft_radial_width=0.5,
                     soft_mass_steps=10)
    narrow = _host_cfg(soft_radial_weight=10.0, soft_radial_width=0.05,
                       soft_mass_steps=10)
    out = []
    for cfg in (wide, narrow):
        sets, pools, pos, vel = _radial_toy(cfg)
        _, diag = solve_memberships(pos, vel, sets, pools,
                                    list(range(sets.n_sets)), cfg)
        out.append(diag["soft_mass_ratio"])
    assert out[1] < out[0], out


def test_radial_prior_ignores_the_task_host():
    cfg = _host_cfg(soft_radial_weight=5.0, soft_radial_width=0.2)
    sets, pools, pos, vel = _radial_toy(cfg)
    idx = list(range(sets.n_sets))
    w_a, _ = solve_memberships(pos, vel, sets, pools, idx, cfg)
    sets.centre_host = sets.centre_host + 7.0
    w_b, _ = solve_memberships(pos, vel, sets, pools, idx, cfg)
    for a, b in zip(w_a, w_b):
        assert torch.equal(a, b)


def test_radial_prior_leaves_a_parentless_set_alone():
    cfg = _host_cfg(soft_radial_weight=5.0, soft_radial_width=0.2)
    sets, pools, pos, vel = _radial_toy(cfg)
    sets.parent_rvir = np.zeros(sets.n_sets)
    idx = list(range(sets.n_sets))
    w_on, diag = solve_memberships(pos, vel, sets, pools, idx, cfg)
    assert diag["soft_radial_valid_frac"] == 0.0
    w_off, _ = solve_memberships(pos, vel, sets, pools, idx, _host_cfg())
    for a, b in zip(w_off, w_on):
        assert torch.equal(a, b)


def test_radial_prior_refusals():
    with pytest.raises(ValueError, match="negative"):
        _host_cfg(soft_radial_weight=-1.0)
    with pytest.raises(ValueError, match="positive"):
        _host_cfg(soft_radial_width=0.0)
    with pytest.raises(ValueError, match="soft_membership is False"):
        MemberGatherConfig(soft_radial_weight=1.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("width", [0.5, 0.05])
def test_a_starved_radial_set_keeps_the_gradient_finite(dtype, width):
    """Job 84238 went NaN inside 250 steps. The radial prior leaves ~5% of
    sets with <0.5x their target mass (down to 0.005x); a set that small has a
    near-zero potential, and the loss must still hand back finite gradients."""
    from cosmo_sr.features.member_gather import member_gather_loss
    cfg = _host_cfg(soft_radial_weight=10.0, soft_radial_width=width,
                    soft_steps=3, soft_mass_steps=10, soft_lambda=0.3)
    # Sets built from the SAME-dtype field the loss sees, as training does
    # (float32 there); casting only the field would mix dtypes in the solve.
    from cosmo_sr.features.member_gather import build_member_sets
    from test_member_gather import DIS, M_P, VEL
    cat, oidx, tiles, hr, fz = _mg_toy()
    hr, fz = hr.to(dtype), fz.to(dtype)
    sets = build_member_sets(cat, oidx, tiles, hr, fz, cfg,
                             particle_mass_msun_h=M_P, ng_hr=NG, tile_hr=TILE,
                             boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                             vel_scale_kms=VEL, host_pos=HOST)
    shift = torch.zeros_like(sets.centre_ref)
    shift[:, 0] = 1.0
    sets.centre_parent = (sets.centre_ref + shift).clone()
    sets.parent_rvir = np.full(sets.n_sets, 1.0)
    sets.parent_mvir = np.full(sets.n_sets, 1e14)
    sets.pools = build_soft_pools(sets, tiles, cfg, ng_hr=NG, tile_hr=TILE,
                                  factor=F, seed=0)
    f = fz.clone().requires_grad_(True)
    pos, vel = _particles(f, tiles)
    loss, diag = member_gather_loss(pos, vel, sets, cfg)
    print(f"\n[{dtype} w={width}] loss={float(loss):.4g} "
          f"mass_min={diag.get('soft_mass_ratio_min')} "
          + " ".join(f"{k}={diag[k]:.4g}" for k in sorted(diag)
                     if k.startswith("term_raw_") and isinstance(diag[k], float)))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(f.grad).all(), (
        f"non-finite grad: {int((~torch.isfinite(f.grad)).sum())} entries")


@pytest.mark.parametrize("gam,width,beta", [(0.0, 0.2, 0.0), (1.0, 0.2, 0.0),
                                            (50.0, 0.02, 0.0), (1.0, 0.2, 0.5)])
def test_radial_prior_free_field_optimisation_stays_finite(gam, width, beta):
    """Many steps, not one: Adam on the toy field directly, as job 84238 did
    on the generator, under a prior that starves sets. Prints where the first
    non-finite value appears."""
    from cosmo_sr.features.member_gather import build_member_sets, member_gather_loss
    from test_member_gather import DIS, M_P, VEL
    cfg = _host_cfg(soft_radial_weight=gam, soft_radial_width=width,
                    soft_steps=3, soft_mass_steps=10, soft_lambda=1.0,
                    soft_host_profile=("parent" if beta else "off"),
                    soft_host_weight=beta)
    cat, oidx, tiles, hr, fz = _mg_toy()
    hr, fz = hr.float(), fz.float()
    sets = build_member_sets(cat, oidx, tiles, hr, fz, cfg,
                             particle_mass_msun_h=M_P, ng_hr=NG, tile_hr=TILE,
                             boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                             vel_scale_kms=VEL, host_pos=HOST)
    shift = torch.zeros_like(sets.centre_ref)
    shift[:, 0] = 1.0
    sets.centre_parent = (sets.centre_ref + shift).clone()
    sets.parent_rvir = np.full(sets.n_sets, 1.0)
    sets.parent_mvir = np.full(sets.n_sets, 1e14)
    sets.pools = build_soft_pools(sets, tiles, cfg, ng_hr=NG, tile_hr=TILE,
                                  factor=F, seed=0)
    f = fz.clone().requires_grad_(True)
    opt = torch.optim.Adam([f], lr=1e-3)
    for step in range(40):
        opt.zero_grad()
        pos, vel = _particles(f, tiles)
        loss, diag = member_gather_loss(pos, vel, sets, cfg)
        if not torch.isfinite(loss):
            pytest.fail(f"loss non-finite at step {step}: mass_min="
                        f"{diag.get('soft_mass_ratio_min')} " + " ".join(
                            f"{k}={v}" for k, v in sorted(diag.items())
                            if k.startswith("term_raw_")))
        loss.backward()
        if not torch.isfinite(f.grad).all():
            pytest.fail(f"grad non-finite at step {step}, loss={float(loss)}, "
                        f"mass_min={diag.get('soft_mass_ratio_min')}")
        opt.step()
    print(f"\n[gam={gam} w={width} beta={beta}] 40 steps finite, final loss "
          f"{float(loss):.4g}, mass_min {diag.get('soft_mass_ratio_min')}")


@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e-12, 1e-15, 1e-18])
def test_a_near_massless_set_has_a_finite_virial_gradient(scale):
    """The NaN of jobs 84238/84296: a starved set's weights are tiny, so |W| is
    tiny, and `2T/|W|` differentiated in float32 formed `T/W**2` = inf. The
    gradient must stay finite down to weights ~1e-12."""
    from cosmo_sr.features.member_gather import _log2
    from cosmo_sr.features.soft_membership import weighted_set_statistics
    g = torch.Generator().manual_seed(0)
    pos = (torch.randn(64, 3, generator=g) * 0.1).float().requires_grad_(True)
    vel = (torch.randn(64, 3, generator=g) * 100.0).float()
    w = torch.full((64,), float(scale), dtype=torch.float32)
    cfg = _host_cfg()
    st = weighted_set_statistics(pos, vel, w, particle_mass_msun_h=5.8e8,
                                 cfg=cfg)
    loss = _log2(st.virial, torch.tensor(2.0)) + _log2(st.r_rms,
                                                      torch.tensor(0.05))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(pos.grad).all(), int((~torch.isfinite(pos.grad)).sum())


# --------------------------------------------------------------------------- #
# centre_mode="self_tan_parent": the anchor that does not fight the radial prior
# --------------------------------------------------------------------------- #
def _centre_after_move(mode, direction, host_pos=HOST):
    """term_centre after moving set 0's live members 0.3 Mpc/h along
    `direction` ("radial" about its OWN parent, or "tangential")."""
    from cosmo_sr.features.member_gather import member_gather_loss
    cfg = _cfg(centre_mode=mode)
    sets, _r, _h, fz, tiles = _sets(cfg, host_pos=host_pos)
    pos, vel = _particles(fz, tiles)
    ray = sets.centre_target[0] - sets.centre_parent[0]
    rh = ray / torch.linalg.vector_norm(ray)
    if direction == "radial":
        u = rh
    else:
        a = torch.tensor([1.0, 0.0, 0.0], dtype=rh.dtype)
        if abs(float(a @ rh)) > 0.9:
            a = torch.tensor([0.0, 1.0, 0.0], dtype=rh.dtype)
        u = a - (a @ rh) * rh
        u = u / torch.linalg.vector_norm(u)
    pos = pos.clone()
    pos[sets.live_rows[0]] = pos[sets.live_rows[0]] + 0.3 * u.to(pos.dtype)
    _, diag = member_gather_loss(pos, vel, sets, cfg)
    return diag["term_centre"]


def test_self_tan_parent_frees_the_radial_move_and_anchors_the_rest():
    radial = _centre_after_move("self_tan_parent", "radial")
    tangential = _centre_after_move("self_tan_parent", "tangential")
    plain = _centre_after_move("self", "radial")
    assert radial < 1e-6 * max(plain, 1e-12) + 1e-9, (radial, plain)
    assert tangential > 0.0
    assert plain > 0.0          # `self` charges the very move the prior makes


def test_self_tan_parent_ignores_the_task_host():
    a = _centre_after_move("self_tan_parent", "tangential")
    b = _centre_after_move("self_tan_parent", "tangential",
                           host_pos=np.array([70.0, 5.0, 40.0]))
    assert a == pytest.approx(b, rel=1e-6)
