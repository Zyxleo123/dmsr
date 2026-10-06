"""Membership as a latent variable: the relaxation of the id-gathered loss.

What has to hold, and why each one is here.

*The hard arm must survive inside it.* The whole claim of the construction is
that ``P`` fixed at the HR mask recovers the method already on disk, so a
weighted statistic at ``w == 1`` must equal the unweighted one and the weighted
pair potential must equal the unweighted pair potential -- not approximately,
and including its gradient, which is the thing that actually trains.

*The two constraints must bind.* They are the only reason the relaxation cannot
be satisfied by nonsense: without the per-particle budget several candidates
spend the same particles, and without the mass target a candidate keeps a tiny
cold core and scores perfectly on every moment. Both are asserted directly,
because both are the failure the advisor named.

*The prior must carry the mass it claims.* A coarse cell is 1.56 Mpc/h a side
in the real run and the pool subsamples its non-members, so the multiplicity
bookkeeping is what makes "every particle in the cell gets n_c/512" add up to
``n_c`` particles rather than to a number nobody checked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cosmo_sr.features.member_gather import (  # noqa: E402
    member_gather_loss, set_statistics_torch, specific_potential_torch,
)
from cosmo_sr.features.soft_membership import (  # noqa: E402
    build_soft_pools, rows_to_coarse_cells, solve_memberships,
    weighted_set_statistics,
)

from test_member_gather import (  # noqa: E402
    BOX, DIS, M_P, NG, TILE, VEL, _cfg, _particles, _sets,
)

#: Coarse factor for the toy: 4^3 = 64 HR sites per LR cell. Chosen so the
#: toy's cells are genuinely MIXED -- a cell holds both members and
#: non-members, which is the only configuration in which the relaxation has
#: anything to decide. At factor 2 the toy's contiguous member block fills
#: whole cells and every pool is its own mask again.
F = 4


def _scfg(**kw):
    base = dict(soft_membership=True, soft_pool_ratio=1.0, soft_steps=2,
                soft_mass_steps=4)
    base.update(kw)
    return _cfg(**base)


def _pools(cfg, sets, tiles, **kw):
    rep = {}
    return build_soft_pools(sets, tiles, cfg, ng_hr=NG, tile_hr=TILE, factor=F,
                            seed=0, report=rep, **kw), rep


# --------------------------------------------------------------------------- #
# The weighted estimator reduces to the one already on disk
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["clamp", "plummer"])
def test_weighted_potential_at_unit_weight_is_the_unweighted_one(kind):
    torch.manual_seed(0)
    pos = torch.randn(60, 3, dtype=torch.float64) * 0.3
    cfg = _cfg(softening_kind=kind)
    a = specific_potential_torch(pos, M_P, softening_mpc_h=0.01, kind=kind,
                                 chunk=7)
    b = specific_potential_torch(pos, M_P, softening_mpc_h=0.01, kind=kind,
                                 chunk=7, weights=torch.ones(60,
                                                             dtype=torch.float64))
    assert torch.equal(a, b) or torch.allclose(a, b, rtol=0, atol=1e-9)


def test_weighted_potential_gradient_matches_autograd_on_a_dense_form():
    """The analytic backward is hand-written; this is the only thing that
    catches a wrong weight on one of its two terms.

    The reference is the same sum built with plain autograd ops, which is
    correct by construction and unaffordable at scale -- which is why the
    chunked Function exists in the first place.
    """
    torch.manual_seed(1)
    n = 40
    pos = (torch.randn(n, 3, dtype=torch.float64) * 0.2).requires_grad_(True)
    w = torch.rand(n, dtype=torch.float64) + 0.05
    g = torch.randn(n, dtype=torch.float64)
    eps, gm = 0.01, 1.0

    phi = specific_potential_torch(pos, gm, softening_mpc_h=eps,
                                   kind="plummer", chunk=9, weights=w)
    (phi * g).sum().backward()
    got = pos.grad.clone()

    p2 = pos.detach().clone().requires_grad_(True)
    d = p2[:, None, :] - p2[None, :, :]
    u = torch.rsqrt((d * d).sum(-1) + eps * eps)
    from cosmo_sr.features.member_gather import G_MPC_KMS2_PER_MSUN
    ref_phi = -G_MPC_KMS2_PER_MSUN * gm * ((u * w[None, :]).sum(1) - w / eps)
    (ref_phi * g).sum().backward()
    assert torch.allclose(got, p2.grad, rtol=1e-8, atol=1e-10)


def test_weighted_statistics_at_unit_weight_match_the_hard_ones():
    torch.manual_seed(2)
    pos = torch.randn(80, 3, dtype=torch.float64) * 0.2 + 5.0
    vel = torch.randn(80, 3, dtype=torch.float64) * 50.0
    cfg = _cfg()
    a = set_statistics_torch(pos, vel, particle_mass_msun_h=M_P, cfg=cfg)
    b = weighted_set_statistics(pos, vel, torch.ones(80, dtype=torch.float64),
                                particle_mass_msun_h=M_P, cfg=cfg)
    for k in ("r_rms", "sigma_v", "virial", "bound_soft", "bound_hard"):
        assert float(getattr(b, k)) == pytest.approx(float(getattr(a, k)),
                                                     rel=1e-9)


def test_a_zero_weight_particle_is_the_same_as_a_deleted_one():
    """The property that makes ``P`` a membership rather than a reweighting:
    a particle the solve rejects must leave no trace in any statistic."""
    torch.manual_seed(3)
    pos = torch.randn(50, 3, dtype=torch.float64) * 0.2
    vel = torch.randn(50, 3, dtype=torch.float64) * 40.0
    cfg = _cfg()
    w = torch.ones(50, dtype=torch.float64)
    w[10:20] = 0.0
    keep = torch.cat([torch.arange(10), torch.arange(20, 50)])
    a = weighted_set_statistics(pos, vel, w, particle_mass_msun_h=M_P, cfg=cfg)
    b = set_statistics_torch(pos[keep], vel[keep], particle_mass_msun_h=M_P,
                             cfg=cfg)
    for k in ("r_rms", "sigma_v", "virial", "bound_soft"):
        assert float(getattr(a, k)) == pytest.approx(float(getattr(b, k)),
                                                     rel=1e-9)


# --------------------------------------------------------------------------- #
# Lagrangian bookkeeping
# --------------------------------------------------------------------------- #
def test_coarse_cells_partition_the_tile_into_full_blocks():
    tiles = [0, 3]
    rows = np.arange(2 * TILE ** 3, dtype=np.int64)
    cells = rows_to_coarse_cells(rows, tiles, ng_hr=NG, tile_hr=TILE, factor=F)
    uc, cnt = np.unique(cells, return_counts=True)
    # Every occupied cell is FULL: a coarse cell never straddles a tile, so a
    # trained tile contributes all factor^3 of each of its cells or none.
    assert set(cnt.tolist()) == {F ** 3}
    assert uc.size == 2 * (TILE // F) ** 3


def test_a_factor_that_straddles_tiles_is_refused():
    with pytest.raises(ValueError, match="straddle"):
        rows_to_coarse_cells(np.array([0]), [0], ng_hr=NG, tile_hr=TILE,
                             factor=TILE * 2)


# --------------------------------------------------------------------------- #
# The prior
# --------------------------------------------------------------------------- #
def test_the_coarse_prior_carries_exactly_the_member_count():
    """Step 1's arithmetic: ``n_c`` of ``f^3`` members in a cell puts prior
    ``n_c / f^3`` on every particle of it, and the pool's multiplicities make
    those fractions sum back to ``n_c`` real particles -- for every set."""
    cfg = _scfg()
    sets, _rep, _hr, _fz, tiles = _sets(cfg)
    pools, rep = _pools(cfg, sets, tiles)
    for s in range(sets.n_sets):
        mass = float((pools.mult[s] * torch.exp(pools.logpi[s])).sum())
        assert mass == pytest.approx(float(sets.n_live[s]), rel=2e-3)
    assert rep["pool_over_live"] > 1.1          # the pool is genuinely wider
    assert rep["soft_prior"] == "coarse"


def test_every_hr_member_stays_in_the_pool():
    """The sampler thins non-members only. Dropping a member would take the
    decision away from the solve and hand it to a random seed."""
    cfg = _scfg(soft_pool_ratio=0.1)
    sets, _r, _h, _f, tiles = _sets(cfg)
    pools, _ = _pools(cfg, sets, tiles)
    for s in range(sets.n_sets):
        live = set(sets.live_rows[s].tolist())
        assert live <= set(pools.rows[s].tolist())
        assert int(pools.is_member[s].sum()) == len(live)


def test_the_hard_prior_puts_its_mass_on_the_hr_members():
    cfg = _scfg(soft_prior="hard")
    sets, _r, _h, _f, tiles = _sets(cfg)
    pools, _ = _pools(cfg, sets, tiles)
    for s in range(sets.n_sets):
        pi = torch.exp(pools.logpi[s])
        assert float(pi[pools.is_member[s]].min()) > 0.9
        assert float(pi[~pools.is_member[s]].max()) < 0.01


# --------------------------------------------------------------------------- #
# The solve, and the two constraints that make it honest
# --------------------------------------------------------------------------- #
def _solved(cfg, field=None, **kw):
    sets, _r, hr, fz, tiles = _sets(cfg, **kw)
    pools, _ = _pools(cfg, sets, tiles)
    pos, vel = _particles(fz if field is None else field, tiles)
    idx = list(range(sets.n_sets))
    w, diag = solve_memberships(pos.double(), vel.double(), sets, pools, idx,
                                cfg)
    return sets, pools, idx, w, diag


def test_no_particle_spends_more_than_one_unit_of_itself():
    """The per-particle budget, asserted on the RAW rows -- the thing two
    overlapping candidates would otherwise double-count."""
    cfg = _scfg(soft_pool_ratio=4.0, soft_lambda=0.5)
    sets, pools, idx, w, _ = _solved(cfg)
    tot = {}
    for si, s in enumerate(idx):
        # `w` carries the multiplicity; the BUDGET is on the probability, so
        # divide it back out before summing over the candidates.
        p = (w[si] / pools.mult[s]).tolist()
        for r, pv in zip(pools.rows[s].tolist(), p):
            tot[r] = tot.get(r, 0.0) + pv
    assert max(tot.values()) <= 1.0 + 1e-6


def test_the_mass_constraint_is_met():
    cfg = _scfg(soft_lambda=0.3, soft_steps=6, soft_mass_steps=20)
    sets, pools, idx, w, diag = _solved(cfg)
    for si, s in enumerate(idx):
        assert float(w[si].sum()) == pytest.approx(float(sets.n_live[s]),
                                                   rel=0.05)
    assert diag["soft_mass_ratio"] == pytest.approx(1.0, rel=0.05)


def test_without_the_mass_dual_a_sharp_window_starves():
    """Why the Sinkhorn dual is not optional. At small lambda the phase-space
    window is narrow and the prior alone cannot fill it; the dual is what
    widens it until the candidate holds the mass it is supposed to hold."""
    cfg = _scfg(soft_lambda=0.1, soft_mass_steps=0, soft_steps=1)
    _s, _p, idx, w, diag = _solved(cfg)
    starved = diag["soft_mass_ratio"]
    cfg2 = _scfg(soft_lambda=0.1, soft_mass_steps=40, soft_steps=8)
    _s2, _p2, _i2, _w2, diag2 = _solved(cfg2)
    assert starved < 0.5
    assert diag2["soft_mass_ratio"] == pytest.approx(1.0, rel=0.05)


def test_a_window_too_sharp_to_fill_says_so_rather_than_lying():
    """The dual has a finite iteration budget and a per-step cap, so a lambda
    small enough to make the window hopeless will NOT reach the mass target.
    That is a configuration error, and the thing that makes it survivable is
    that ``soft_mass_ratio`` reports it every step -- an arm whose candidates
    hold 60% of the mass they claim is not the arm anyone meant to run."""
    _s, _p, _i, _w, diag = _solved(_scfg(soft_lambda=0.02, soft_steps=4,
                                         soft_mass_steps=10))
    assert diag["soft_mass_ratio"] < 0.9
    assert diag["soft_mass_ratio_min"] <= diag["soft_mass_ratio"]


def test_a_large_lambda_pins_the_membership_at_the_prior():
    """``lambda -> inf`` is the limit the boxed objective promises: the KL term
    dominates and ``P`` is the prior. With ``soft_prior='hard'`` that is the
    arm already on disk, which is what makes the sweep's control a control."""
    cfg = _scfg(soft_prior="hard", soft_lambda=1e6, soft_mass_steps=0,
                soft_steps=1, soft_prior_eps=1e-6)
    sets, pools, idx, w, _ = _solved(cfg)
    for si, s in enumerate(idx):
        mem = pools.is_member[s]
        assert float(w[si][mem].min()) > 0.99
        assert float(w[si][~mem].max()) < 0.01


def _place(field, rows, x, v):
    """Put the given table rows at Eulerian ``x`` (Mpc/h) and ``v`` (km/s).

    The toy's tile 0 is the identity map from local to global site, so the row
    arithmetic here is :func:`tile_particles`'s, inverted by hand -- which is
    the point: a helper that called the module's own inverse could not catch
    the module getting it wrong.
    """
    cell = BOX / NG
    f = field.clone()
    for i, r in enumerate(np.atleast_1d(rows)):
        r = int(r)
        li, lj, lk = (r // (TILE * TILE)) % TILE, (r // TILE) % TILE, r % TILE
        q = (np.array([li, lj, lk]) + 0.5) * cell
        xi = np.atleast_2d(x)[i % len(np.atleast_2d(x))]
        vi = np.atleast_2d(v)[i % len(np.atleast_2d(v))]
        f[0, 0:3, li, lj, lk] = torch.tensor((xi - q) / DIS)
        f[0, 3:6, li, lj, lk] = torch.tensor(vi / VEL)
    return f


def test_the_solve_prefers_the_phase_space_neighbourhood_over_the_prior():
    """The point of the whole exercise, and the property no hard mask has: a
    particle HR filed under a DIFFERENT object, which the SR field has placed
    on this one, is recruited into it.

    The toy is built to make the question decidable. Twenty pool non-members --
    HR members of the other subhalo, 28 Mpc/h away, so untouched they score
    zero -- are moved onto this set's centre in the *frozen field only*. HR's
    reference is unchanged, so nothing about the target moved; only where the
    SR field put the material did.
    """
    from cosmo_sr.features.member_gather import unwrap_about
    cfg = _scfg(soft_lambda=0.3, soft_steps=4, soft_mass_steps=20)
    sets, _r, _h, fz, tiles = _sets(cfg)
    pools, _ = _pools(cfg, sets, tiles)
    s = 0
    nm = pools.rows[s][~pools.is_member[s]].numpy()
    assert nm.size >= 20, "the toy's cells must be MIXED or there is nothing to decide"
    moved = nm[:20]

    rng = np.random.default_rng(0)
    c = sets.centre_self[s].numpy()
    fz2 = _place(fz, moved, c + rng.normal(scale=0.3, size=(20, 3)),
                 rng.normal(scale=200.0, size=(20, 3)))

    out = {}
    for name, field in (("before", fz), ("after", fz2)):
        pos, vel = _particles(field, tiles)
        w, _d = solve_memberships(pos.double(), vel.double(), sets, pools, [s],
                                  cfg)
        prob = (w[0] / pools.mult[s])[~pools.is_member[s]]
        out[name] = prob[:20].clone()

    assert float(out["before"].max()) < 1e-3      # 28 Mpc/h away: not members
    assert float(out["after"].mean()) > 0.5       # on the object: members now


def test_the_record_hook_snapshots_each_sweep_without_changing_the_answer():
    """The visualisation instrumentation is guarded and faithful: passing
    `record` returns the SAME weights and fills one snapshot per sweep plus the
    initial state, each aligned with the returned per-entry weight vector."""
    cfg = _scfg(soft_lambda=0.3, soft_steps=4, soft_mass_steps=20)
    sets, _r, _h, fz, tiles = _sets(cfg)
    pools, _ = _pools(cfg, sets, tiles)
    pos, vel = _particles(fz, tiles)
    idx = list(range(sets.n_sets))
    w_plain, _ = solve_memberships(pos.double(), vel.double(), sets, pools, idx, cfg)
    rec = {}
    w_rec, _ = solve_memberships(pos.double(), vel.double(), sets, pools, idx, cfg,
                                 record=rec)
    # Same answer with the hook on.
    for a, b in zip(w_plain, w_rec):
        assert torch.allclose(a, b)
    # init + one per sweep.
    assert len(rec["sweeps"]) == int(cfg.soft_steps) + 1
    assert rec["sweeps"][0]["tag"] == "init"
    n_entries = sum(int(pools.rows[s].numel()) for s in idx)
    assert rec["x"].shape == (n_entries, 3)
    assert rec["is_member"].shape == (n_entries,)
    # The last snapshot's weights are the returned ones.
    assert np.allclose(rec["sweeps"][-1]["w"], torch.cat(w_rec).numpy())
# --------------------------------------------------------------------------- #
# End to end, through the loss
# --------------------------------------------------------------------------- #
def test_the_loss_refuses_soft_membership_without_pools():
    cfg = _scfg()
    sets, _r, _h, fz, tiles = _sets(cfg)
    pos, vel = _particles(fz, tiles)
    with pytest.raises(ValueError, match="no pools"):
        member_gather_loss(pos, vel, sets, cfg)


def test_the_soft_loss_at_the_hard_limit_reproduces_the_hard_loss():
    """The strongest statement available without a GPU run: at
    ``soft_prior='hard'`` and large lambda every term of the relaxed loss
    matches the hard arm's, so the relaxation is a superset of it and any
    difference measured later is the membership moving, not a rewrite."""
    hard = _cfg()
    soft = _scfg(soft_prior="hard", soft_lambda=1e7, soft_prior_eps=1e-9,
                 soft_bg_floor=1e-9, soft_mass_steps=0, soft_steps=1,
                 soft_pool_ratio=1.0)
    sets, _r, _h, fz, tiles = _sets(hard)
    pos, vel = _particles(fz, tiles)
    l_hard, d_hard = member_gather_loss(pos, vel, sets, hard)

    ssets, _r2, _h2, _f2, _t2 = _sets(soft)
    pools, _ = _pools(soft, ssets, tiles)
    ssets.pools = pools
    l_soft, d_soft = member_gather_loss(pos, vel, ssets, soft)
    assert float(l_soft) == pytest.approx(float(l_hard), rel=2e-3)
    for k in ("term_virial", "term_rrms", "term_sigmav", "term_centre"):
        assert d_soft[k] == pytest.approx(d_hard[k], rel=5e-3, abs=1e-6)
    assert d_soft["soft_membership"] is True
    assert d_hard["soft_membership"] is False


def test_the_soft_loss_has_gradient_reaching_non_members():
    """The relaxation's whole purpose, stated as a gradient: a particle HR
    never named must be able to receive force from this loss."""
    cfg = _scfg(soft_lambda=0.3, soft_steps=3)
    sets, _r, _h, fz, tiles = _sets(cfg)
    pools, _ = _pools(cfg, sets, tiles)
    sets.pools = pools
    f = fz.clone().requires_grad_(True)
    pos, vel = _particles(f, tiles)
    loss, diag = member_gather_loss(pos, vel, sets, cfg)
    loss.backward()
    g = f.grad.reshape(1, 6, -1)[0, 0:3].abs().sum(0).reshape(-1)
    members = torch.zeros(TILE ** 3, dtype=torch.bool)
    for s in range(sets.n_sets):
        members[sets.live_rows[s]] = True
    nonmember_pool = torch.zeros(TILE ** 3, dtype=torch.bool)
    for s in range(sets.n_sets):
        nonmember_pool[pools.rows[s][~pools.is_member[s]]] = True
    assert torch.isfinite(f.grad).all()
    assert float(g[nonmember_pool & ~members].max()) > 0.0
    assert diag["soft_hr_fraction"] > 0.0


def test_the_soft_loss_still_falls_toward_hr():
    """A sanity floor: whatever the membership does, moving the field from
    frozen to HR must not make the objective worse."""
    cfg = _scfg(soft_lambda=0.3)
    sets, _r, hr, fz, tiles = _sets(cfg)
    pools, _ = _pools(cfg, sets, tiles)
    sets.pools = pools
    vals = []
    for t in (0.0, 0.5, 1.0):
        pos, vel = _particles(fz * (1 - t) + hr * t, tiles)
        vals.append(float(member_gather_loss(pos, vel, sets, cfg)[0]))
    assert vals[0] > vals[1] > vals[2]
