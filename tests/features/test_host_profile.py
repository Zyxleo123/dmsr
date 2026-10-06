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
@pytest.mark.parametrize("src", ["lr", "self"])
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
@pytest.mark.parametrize("src", ["lr", "self"])
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
    with pytest.raises(ValueError, match="not 'off', 'lr' or 'self'"):
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

    arm = _host_cfg(soft_host_profile="self", soft_host_weight=1.0)
    guard = dataclasses.replace(arm, centre_mode="self", w_radius=0.0,
                                soft_membership=False,
                                soft_host_profile="off", soft_host_weight=0.0)
    assert guard.soft_membership is False
    assert guard.soft_host_profile == "off"
    # And the refusal still fires for the combination a person can type.
    with pytest.raises(ValueError, match="only has a meaning"):
        dataclasses.replace(arm, soft_membership=False)
