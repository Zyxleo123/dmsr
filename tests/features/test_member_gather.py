"""The id-gathered member-set loss.

Two things need pinning and one needs proving.

Pinned: the estimator must agree with
:mod:`cosmo_sr.features.bound_discriminator`'s numpy on identical inputs, or the
HR reference this module builds is not comparable to the section 8.2 table that
motivated it; and the flat particle table must invert exactly, or the loss
supervises the wrong particles silently.

Proved: the anti-gaming properties. ``docs/sr2_gather_finetune.md`` section 6
records that the window loss was satisfied by a raised pedestal, so a
replacement is only worth running if the cheap cheats provably cost it -- and
the specific new one is that a boundness objective is cheapest to satisfy by
cooling the whole neighbourhood.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cosmo_sr.features.bound_discriminator import (  # noqa: E402
    set_statistics, specific_potential,
)
from cosmo_sr.features import member_gather  # noqa: E402
from cosmo_sr.features.member_gather import (  # noqa: E402
    MemberGatherConfig, build_member_sets, member_gather_loss,
    set_statistics_torch, specific_potential_torch, tile_particles,
    unwrap_about, _rows_for_ids,
)

NG, TILE, BOX = 16, 8, 100.0
M_P = 5.8188e8
DIS = 6.0        # disnorm(1, undo=True) * 1e-3 at z=0, Mpc/h per unit
VEL = 313.42     # velnorm(1, undo=True) at z=0, km/s per unit -- the real scale,
                 # so a learning rate that works here is in the right ballpark
                 # for the cluster run rather than an artifact of the toy


def _cfg(**kw):
    base = dict(min_num_p=4, min_purity=0.5, min_live_frac=0.5, bg_k=0,
                softening_kind="clamp", pot_chunk=64)
    base.update(kw)
    return MemberGatherConfig(**base)


# --------------------------------------------------------------------------- #
# The particle table
# --------------------------------------------------------------------------- #
def test_tile_particles_matches_the_lattice_by_hand():
    n_side = NG // TILE
    tiles = [0, n_side * n_side]                  # (0,0,0) and (1,0,0)
    f = torch.zeros(2, 6, TILE, TILE, TILE)
    pos, vel = tile_particles(f, tiles, ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX,
                              dis_scale_mpc_h=DIS, vel_scale_kms=VEL)
    cell = BOX / NG
    assert pos.shape == (2 * TILE ** 3, 3)
    # tile 0, local (0,0,0) -> global site (0,0,0) -> half-cell offset
    assert pos[0].tolist() == pytest.approx([0.5 * cell] * 3)
    # tile n_side^2 == (1,0,0), local (0,0,0) -> global (TILE,0,0)
    off = TILE ** 3
    assert pos[off].tolist() == pytest.approx([(TILE + 0.5) * cell,
                                               0.5 * cell, 0.5 * cell])
    assert torch.count_nonzero(vel) == 0


def test_rows_for_ids_inverts_the_table_ordering():
    tiles = [0, NG // TILE, 3]
    f = torch.zeros(3, 6, TILE, TILE, TILE)
    # A displacement that is different in every cell, so a mis-indexed row
    # cannot coincidentally match.
    f[:, 0:3] = torch.arange(3 * 3 * TILE ** 3, dtype=torch.float32).reshape(
        3, 3, TILE, TILE, TILE) * 1e-6
    pos, _ = tile_particles(f, tiles, ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX,
                            dis_scale_mpc_h=DIS, vel_scale_kms=VEL)

    ids = np.arange(NG ** 3, dtype=np.int64)
    rows, keep = _rows_for_ids(ids, tiles, ng_hr=NG, tile_hr=TILE)
    assert keep.size == 3 * TILE ** 3          # exactly the three tiles' sites
    assert np.unique(rows).size == rows.size   # a bijection, no collisions

    cell = BOX / NG
    for pick in (0, 17, 511, keep.size - 1):
        pid = int(ids[keep][pick])
        gx, gy, gz = pid // (NG * NG), (pid // NG) % NG, pid % NG
        q = (np.array([gx, gy, gz]) + 0.5) * cell
        b = tiles.index(((gx // TILE) * (NG // TILE) + (gy // TILE))
                        * (NG // TILE) + (gz // TILE))
        loc = (((gx % TILE) * TILE) + (gy % TILE)) * TILE + (gz % TILE)
        d = f[b, 0:3].reshape(3, -1)[:, loc].numpy() * DIS
        assert pos[int(rows[pick])].numpy() == pytest.approx(q + d, abs=1e-5)


def test_rows_for_ids_keeps_only_trained_tiles():
    rows, keep = _rows_for_ids(np.arange(NG ** 3, dtype=np.int64), [0],
                               ng_hr=NG, tile_hr=TILE)
    assert keep.size == TILE ** 3
    assert rows.min() == 0 and rows.max() == TILE ** 3 - 1


# --------------------------------------------------------------------------- #
# The estimator, against the numpy it must stay comparable to
# --------------------------------------------------------------------------- #
def test_potential_clamp_matches_numpy_exactly():
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.4, size=(300, 3))
    a = specific_potential(x, M_P, softening_mpc_h=0.01)
    b = specific_potential_torch(torch.tensor(x, dtype=torch.float64), M_P,
                                 softening_mpc_h=0.01, kind="clamp", chunk=64)
    assert b.numpy() == pytest.approx(a, rel=1e-12)


def test_plummer_agrees_with_clamp_away_from_the_softening():
    rng = np.random.default_rng(1)
    x = torch.tensor(rng.normal(scale=0.5, size=(200, 3)), dtype=torch.float64)
    c = specific_potential_torch(x, M_P, softening_mpc_h=1e-4, kind="clamp")
    p = specific_potential_torch(x, M_P, softening_mpc_h=1e-4, kind="plummer")
    assert p.numpy() == pytest.approx(c.numpy(), rel=1e-3)


def test_plummer_has_gradient_inside_the_softening_and_clamp_does_not():
    x = torch.tensor([[0.0, 0.0, 0.0], [0.002, 0.0, 0.0]], requires_grad=True,
                     dtype=torch.float64)
    for kind, expect in (("clamp", 0.0), ("plummer", None)):
        g, = torch.autograd.grad(
            specific_potential_torch(x, M_P, softening_mpc_h=0.01,
                                     kind=kind).sum(), x)
        if expect == 0.0:
            assert float(g.abs().max()) == 0.0
        else:
            assert float(g.abs().max()) > 0.0


def test_set_statistics_matches_numpy():
    rng = np.random.default_rng(2)
    x = rng.normal(scale=0.5, size=(400, 3)) + 50.0
    v = rng.normal(scale=300.0, size=(400, 3))
    a = set_statistics(x, v, particle_mass_msun_h=M_P, boxsize_mpc_h=BOX,
                       softening_mpc_h=0.01)
    b = set_statistics_torch(torch.tensor(x), torch.tensor(v),
                             particle_mass_msun_h=M_P, cfg=_cfg())
    assert float(b.r_rms) == pytest.approx(a.r_rms, rel=1e-10)
    assert float(b.sigma_v) == pytest.approx(a.sigma_v, rel=1e-10)
    assert float(b.virial) == pytest.approx(a.virial_ratio, rel=1e-10)
    assert float(b.bound_hard) == pytest.approx(a.bound_frac, rel=1e-12)


def test_unwrap_puts_a_split_clump_back_together():
    x = torch.tensor([[99.8, 1.0, 1.0], [0.2, 1.0, 1.0]], dtype=torch.float64)
    ref = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    u = unwrap_about(x, ref, BOX)
    assert float((u[0] - u[1]).norm()) == pytest.approx(0.4, abs=1e-9)


def test_unwrap_is_differentiable_with_gradient_one():
    x = torch.tensor([[99.8, 1.0, 1.0]], dtype=torch.float64, requires_grad=True)
    g, = torch.autograd.grad(unwrap_about(x, torch.zeros(3, dtype=torch.float64),
                                          BOX).sum(), x)
    assert g.numpy() == pytest.approx(np.ones((1, 3)))


# --------------------------------------------------------------------------- #
# The soft unbinding test
# --------------------------------------------------------------------------- #
def test_adaptive_temperature_is_alive_where_a_fixed_one_is_dead():
    """Section 8.2's sets are saturated, not merely low -- so this matters."""
    rng = np.random.default_rng(3)
    x = torch.tensor(rng.normal(scale=1.1, size=(600, 3)), requires_grad=True)
    v = torch.tensor(rng.normal(scale=1004.0, size=(600, 3)), requires_grad=True)

    st = set_statistics_torch(x, v, particle_mass_msun_h=M_P, cfg=_cfg())
    assert float(st.bound_hard) == 0.0            # the frozen-like starting point
    assert 0.0 < float(st.bound_soft.detach()) < 0.5       # but the surrogate is not saturated
    g, = torch.autograd.grad(st.bound_soft, v, retain_graph=True)
    assert float(g.abs().max()) > 0.0

    # A fixed temperature at the monopole binding scale: exactly the dead case.
    scale = torch.tensor(0.5 * 4.30091e-9 * M_P * 600 / 1.1)
    dead = set_statistics_torch(x, v, particle_mass_msun_h=M_P, cfg=_cfg(),
                                bound_scale=scale)
    assert float(dead.bound_soft.detach()) < 1e-9
    gd, = torch.autograd.grad(dead.bound_soft, v)
    # The claim is about the gradient, not the value: a fixed temperature
    # does not merely report a small number, it reports one no optimiser
    # can use.
    assert float(gd.abs().max()) < 1e-4 * float(g.abs().max())


def test_cooling_a_set_raises_its_bound_fraction():
    rng = np.random.default_rng(4)
    x = torch.tensor(rng.normal(scale=0.3, size=(400, 3)))
    v = torch.tensor(rng.normal(scale=50.0, size=(400, 3)))
    hot = set_statistics_torch(x, v * 8.0, particle_mass_msun_h=M_P, cfg=_cfg())
    cold = set_statistics_torch(x, v, particle_mass_msun_h=M_P, cfg=_cfg())
    assert float(cold.bound_hard) > float(hot.bound_hard)
    assert float(cold.virial) < float(hot.virial)


def test_spreading_a_set_raises_its_virial_ratio():
    """Blurring is the worst available move, as it was for the window loss."""
    rng = np.random.default_rng(5)
    x = torch.tensor(rng.normal(scale=0.3, size=(400, 3)))
    v = torch.tensor(rng.normal(scale=200.0, size=(400, 3)))
    tight = set_statistics_torch(x, v, particle_mass_msun_h=M_P, cfg=_cfg())
    loose = set_statistics_torch(x * 3.0, v, particle_mass_msun_h=M_P, cfg=_cfg())
    assert float(loose.virial) > float(tight.virial)
    assert float(loose.r_rms) > float(tight.r_rms)


def test_d6_against_a_local_background_resists_a_global_cooling():
    """The specific new cheat: cool everything, and the contrast term must not move.

    ``r_rms`` and ``sigma_v`` are absolute and a uniform velocity rescale moves
    ``sigma_v``; ``d6`` normalises by the background's own dispersions measured
    on the same field, so the same rescale leaves it invariant. That is the term
    a window statistic cannot express.
    """
    rng = np.random.default_rng(6)
    x = torch.tensor(rng.normal(scale=0.3, size=(300, 3)))
    v = torch.tensor(rng.normal(scale=200.0, size=(300, 3)))
    bx = torch.tensor(rng.normal(scale=1.5, size=(2000, 3)))
    bv = torch.tensor(rng.normal(scale=900.0, size=(2000, 3)))

    a = set_statistics_torch(x, v, particle_mass_msun_h=M_P, cfg=_cfg(),
                             bg_pos=bx, bg_vel=bv)
    b = set_statistics_torch(x, v * 0.1, particle_mass_msun_h=M_P, cfg=_cfg(),
                             bg_pos=bx, bg_vel=bv * 0.1)
    assert float(b.d6) == pytest.approx(float(a.d6), rel=1e-9)
    assert float(b.sigma_v) < float(a.sigma_v)      # the absolute term did move


# --------------------------------------------------------------------------- #
# End to end on a toy box
# --------------------------------------------------------------------------- #
def _toy(seed: int = 7, tight: float = 0.15, loose: float = 0.9,
         cold: float = 30.0, hot: float = 1000.0,
         split: bool = False, bg_scale: float = 1.2):
    """A 16^3 box, one trained tile, two subhalos: HR tight, frozen diffuse.

    The frozen field is built to look like the real failure -- the same
    particles, spread ~6x wider and ~4x hotter -- so the toy exercises the
    branch the real run takes rather than a near-converged one.
    """
    rng = np.random.default_rng(seed)
    n_side = NG // TILE
    tiles = [0]
    cell = BOX / NG

    site = np.arange(TILE ** 3)
    gx, gy, gz = site // (TILE * TILE), (site // TILE) % TILE, site % TILE
    pid = gx * NG * NG + gy * NG + gz                     # tile 0's global ids
    q = (np.stack([gx, gy, gz], axis=1) + 0.5) * cell

    owner = np.full(NG ** 3, -1, dtype=np.int64)
    owner[pid[:200]] = 10
    owner[pid[200:350]] = 11
    if split:
        # 90 sites in the trained tile, 10 in an untrained neighbour: home tile
        # is still 0 at purity 0.9, but only 90% of it is movable.
        owner[pid[350:440]] = 12
        nb = (np.arange(10) // (TILE * TILE)) * NG * NG \
            + ((np.arange(10) // TILE) % TILE) * NG + (np.arange(10) % TILE) + TILE
        owner[nb] = 12

    centres = {10: np.array([10.0, 10.0, 10.0]), 11: np.array([30.0, 30.0, 30.0]),
               12: np.array([15.0, 15.0, 15.0])}
    hr = np.zeros((1, 6, TILE, TILE, TILE), dtype=np.float64)
    fz = np.zeros_like(hr)
    spans = [(10, slice(0, 200)), (11, slice(200, 350))]
    if split:
        spans.append((12, slice(350, 440)))
    for hid, sl in spans:
        n = site[sl].size
        c = centres[hid]
        for f, s, vs in ((hr, tight, cold), (fz, loose, hot)):
            x = c + rng.normal(scale=s, size=(n, 3))
            v = rng.normal(scale=vs, size=(n, 3))
            f[0, 0:3, gx[sl], gy[sl], gz[sl]] = (x - q[sl]) / DIS
            f[0, 3:6, gx[sl], gy[sl], gz[sl]] = v / VEL
    # Non-members: a warm cloud around EACH subhalo, identical in both fields.
    # It has to be local -- the background is selected inside a few reference
    # radii of the target, so a cloud sitting elsewhere in the box leaves d6
    # undefined and the contrast term silently absent.
    rest = site[440:] if split else site[350:]
    half = rest.size // 2
    bx = np.concatenate([
        rng.normal(scale=bg_scale, size=(half, 3)) + centres[10],
        rng.normal(scale=bg_scale, size=(rest.size - half, 3))
        + centres[11]])
    bv = rng.normal(scale=900.0, size=(rest.size, 3))
    for f in (hr, fz):
        f[0, 0:3, gx[rest], gy[rest], gz[rest]] = (bx - q[rest]) / DIS
        f[0, 3:6, gx[rest], gy[rest], gz[rest]] = bv / VEL

    ids = [1, 10, 11] + ([12] if split else [])
    cat = SimpleNamespace(
        n=len(ids),
        ids=np.array(ids), parent_ids=np.array([-1] + [1] * (len(ids) - 1)),
        num_p=np.array([TILE ** 3, 200, 150] + ([100] if split else [])),
        mvir=np.array([1e14, 1e12, 8e11] + ([5e11] if split else [])),
        rvir=np.array([1000.0, 200.0, 180.0] + ([150.0] if split else [])),
        vmax=np.zeros(len(ids)), vel=np.zeros((len(ids), 3)),
        pos=np.stack([np.array([20.0, 20.0, 20.0])]
                     + [centres[h] for h in ids[1:]]))

    from cosmo_sr.eval.particle_identity import build_owner_index
    return (cat, build_owner_index(owner), tiles,
            torch.tensor(hr), torch.tensor(fz))


def _sets(cfg, host_pos=None, **kw):
    cat, oidx, tiles, hr, fz = _toy(**kw)
    rep = {}
    ms = build_member_sets(cat, oidx, tiles, hr, fz, cfg,
                           particle_mass_msun_h=M_P, ng_hr=NG, tile_hr=TILE,
                           boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                           vel_scale_kms=VEL, host_pos=host_pos, report=rep)
    return ms, rep, hr, fz, tiles


#: The toy's host (catalog id 1). Its two subhalos sit at [10,10,10] and
#: [30,30,30], so `rhat` is +-1/sqrt(3) on every axis and the radial/transverse
#: split can be written down by hand rather than read back off the code.
HOST = np.array([20.0, 20.0, 20.0])


def _particles(f, tiles):
    return tile_particles(f, tiles, ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX,
                          dis_scale_mpc_h=DIS, vel_scale_kms=VEL)


def test_build_member_sets_finds_both_subhalos():
    ms, rep, _, _, _ = _sets(_cfg())
    assert ms.n_sets == 2
    assert sorted(ms.halo_id.tolist()) == [10, 11]
    assert ms.n_live.tolist() == [200, 150]      # wholly inside the trained tile
    assert rep["outside_dropped"] == 0
    assert rep["median_live_frac"] == pytest.approx(1.0)
    # HR is tight and cold, so its reference must be bound where frozen is not.
    assert rep["reference_median"]["bound_hard"] > 0.0


def test_the_reference_field_scores_exactly_zero():
    """The loss must not charge for reaching HR. Every term is at its reference."""
    cfg = _cfg(bg_k=32, bg_radius_factor=8.0)
    ms, _, hr, _, tiles = _sets(cfg)
    pos, vel = tile_particles(hr, tiles, ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX,
                              dis_scale_mpc_h=DIS, vel_scale_kms=VEL)
    loss, diag = member_gather_loss(pos, vel, ms, cfg)
    assert float(loss) == pytest.approx(0.0, abs=1e-12)
    for k in ("term_virial", "term_bound", "term_d6", "term_rrms", "term_sigmav"):
        assert diag[k] == pytest.approx(0.0, abs=1e-12)


def test_the_weighted_terms_sum_to_the_loss():
    """The invariant the training run's ``term_*`` logging rests on.

    ``finetune_member_gather`` charts each term weighted, so that the six series
    add up to the ``gather`` series and the reader can see which one owns the
    budget. That is only true if the diagnostics are the *same* accumulators the
    loss sums -- if a term is ever weighted inside ``acc`` instead, the charts
    silently stop adding up.
    """
    cfg = _cfg(bg_k=32, bg_radius_factor=8.0, w_virial=1.0, w_bound=2.0,
               w_d6=0.5, w_rrms=0.3, w_sigmav=0.3, w_centre=4.0)
    ms, _, _, fz, tiles = _sets(cfg)
    pos, vel = tile_particles(fz, tiles, ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX,
                              dis_scale_mpc_h=DIS, vel_scale_kms=VEL)
    loss, diag = member_gather_loss(pos, vel, ms, cfg)
    total = sum(getattr(cfg, f"w_{k}") * diag[f"term_{k}"]
                for k in ("virial", "bound", "d6", "rrms", "sigmav", "centre"))
    assert total == pytest.approx(float(loss), rel=1e-6)


def test_the_frozen_field_is_charged_on_every_term():
    cfg = _cfg(bg_k=32, bg_radius_factor=8.0)
    ms, _, _, fz, tiles = _sets(cfg)
    pos, vel = tile_particles(fz, tiles, ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX,
                              dis_scale_mpc_h=DIS, vel_scale_kms=VEL)
    loss, diag = member_gather_loss(pos, vel, ms, cfg)
    assert float(loss) > 0.0
    for k in ("term_virial", "term_bound", "term_d6", "term_rrms", "term_sigmav"):
        assert diag[k] > 0.0, k
    # The failure mode this whole line is about: diffuse, hot, unbound.
    assert diag["median_r_rms_over_hr"] > 2.0
    assert diag["median_sigma_v_over_hr"] > 2.0
    assert diag["median_bound_hard"] == 0.0
    assert len(diag["rows"]) == 2


def test_the_loss_falls_monotonically_along_the_path_to_hr():
    cfg = _cfg(bg_k=32, bg_radius_factor=8.0)
    ms, _, hr, fz, tiles = _sets(cfg)
    vals = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        f = (1.0 - a) * fz + a * hr
        pos, vel = tile_particles(f, tiles, ng_hr=NG, tile_hr=TILE,
                                  boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                                  vel_scale_kms=VEL)
        vals.append(float(member_gather_loss(pos, vel, ms, cfg)[0]))
    assert all(b < a for a, b in zip(vals, vals[1:])), vals
    assert vals[-1] == pytest.approx(0.0, abs=1e-12)


def test_gradients_reach_the_field_and_are_finite():
    cfg = _cfg(bg_k=32, bg_radius_factor=8.0)
    ms, _, _, fz, tiles = _sets(cfg)
    f = fz.clone().requires_grad_(True)
    pos, vel = tile_particles(f, tiles, ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX,
                              dis_scale_mpc_h=DIS, vel_scale_kms=VEL)
    g, = torch.autograd.grad(member_gather_loss(pos, vel, ms, cfg)[0], f)
    assert torch.isfinite(g).all()
    assert float(g[:, 0:3].abs().max()) > 0.0      # displacement channels move
    assert float(g[:, 3:6].abs().max()) > 0.0      # and so do the velocities


def test_gradient_is_confined_to_members_and_their_background():
    """Nothing else in the tile is touched, so collateral damage is attributable."""
    cfg = _cfg(bg_k=0)
    ms, _, _, fz, tiles = _sets(cfg)
    f = fz.clone().requires_grad_(True)
    pos, vel = tile_particles(f, tiles, ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX,
                              dis_scale_mpc_h=DIS, vel_scale_kms=VEL)
    g, = torch.autograd.grad(member_gather_loss(pos, vel, ms, cfg)[0], f)
    touched = torch.zeros(TILE ** 3, dtype=torch.bool)
    for r in ms.live_rows:
        touched[r] = True
    flat = g.permute(0, 2, 3, 4, 1).reshape(-1, 6)
    assert float(flat[~touched].abs().max()) == 0.0
    assert float(flat[touched].abs().max()) > 0.0


def test_a_straddling_set_is_kept_or_dropped_by_the_live_fraction():
    """Its outside members are unreachable, so the cut is about what we control."""
    loose_ms, loose_rep, _, _, _ = _sets(_cfg(min_live_frac=0.5), split=True)
    assert sorted(loose_ms.halo_id.tolist()) == [10, 11, 12]
    assert loose_rep["outside_dropped"] == 10       # no frozen_box was passed
    assert "outside_dropped_warning" in loose_rep

    tight_ms, tight_rep, _, _, _ = _sets(_cfg(min_live_frac=0.95), split=True)
    assert sorted(tight_ms.halo_id.tolist()) == [10, 11]
    assert tight_rep["n_dropped_live_frac"] == 1


def test_an_impossible_selection_says_which_cut_emptied_it():
    with pytest.raises(ValueError, match="no supervised sets survived"):
        _sets(_cfg(min_num_p=10 ** 6))


# --------------------------------------------------------------------------- #
# The host preservation guard (--w-host-sets)
# --------------------------------------------------------------------------- #
def _toy_host(seed=3, tight=0.15, loose=0.9, cold=30.0, hot=1000.0, n=220):
    """One trained tile holding one TOP-LEVEL host that owns particles in it.

    Mirrors ``_toy`` but the object is a host (``parent_ids < 0``), so
    ``build_member_sets(top_level=True)`` selects it. ``hr`` is compact+cold and
    ``fz`` diffuse+hot, as in the real failure -- but the preservation guard uses
    the FROZEN field as its reference, so the interesting cases are built against
    ``fz`` and perturbations away from it, not toward ``hr``.
    """
    rng = np.random.default_rng(seed)
    tiles = [0]
    cell = BOX / NG
    site = np.arange(TILE ** 3)
    gx, gy, gz = site // (TILE * TILE), (site // TILE) % TILE, site % TILE
    q = (np.stack([gx, gy, gz], axis=1) + 0.5) * cell
    pid = gx * NG * NG + gy * NG + gz

    owner = np.full(NG ** 3, -1, dtype=np.int64)
    owner[pid[:n]] = 1                                # host id 1 owns n sites
    centre = np.array([12.0, 12.0, 12.0])
    hr = np.zeros((1, 6, TILE, TILE, TILE), dtype=np.float64)
    fz = np.zeros_like(hr)
    sl = slice(0, n)
    for f, s, vs in ((hr, tight, cold), (fz, loose, hot)):
        x = centre + rng.normal(scale=s, size=(n, 3))
        v = rng.normal(scale=vs, size=(n, 3))
        f[0, 0:3, gx[sl], gy[sl], gz[sl]] = (x - q[sl]) / DIS
        f[0, 3:6, gx[sl], gy[sl], gz[sl]] = v / VEL
    cat = SimpleNamespace(
        ids=np.array([1]), parent_ids=np.array([-1]), num_p=np.array([n]),
        mvir=np.array([1e14]), rvir=np.array([300.0]), vmax=np.zeros(1),
        vel=np.zeros((1, 3)), pos=np.stack([centre]))
    from cosmo_sr.eval.particle_identity import build_owner_index
    return cat, build_owner_index(owner), tiles, torch.tensor(hr), torch.tensor(fz)


def _host_sets(cfg, ref, other):
    """Build host preservation sets with ``ref`` as the reference field."""
    cat, oidx, tiles, hr, fz = _toy_host()
    fields = {"hr": hr, "fz": fz}
    rep = {}
    ms = build_member_sets(cat, oidx, tiles, fields[ref], fields[other], cfg,
                           particle_mass_msun_h=M_P, ng_hr=NG, tile_hr=TILE,
                           boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                           vel_scale_kms=VEL, host_pos=None, top_level=True,
                           report=rep)
    return ms, rep, tiles, hr, fz


def test_top_level_build_selects_the_host():
    cfg = _cfg(min_num_p=50, centre_mode="self")
    ms, rep, _, _, _ = _host_sets(cfg, ref="fz", other="fz")
    assert ms.n_sets == 1
    assert ms.halo_id.tolist() == [1]
    assert rep["top_level"] is True


def test_the_host_guard_is_silent_at_the_frozen_reference():
    """Referenced on the frozen field, the guard is exactly zero on it.

    This is the property that makes it a PRESERVATION guard rather than a second
    objective: at step 0 the candidate IS the frozen field, so every hinged and
    two-sided term sits at its reference and contributes nothing.
    """
    cfg = _cfg(min_num_p=50, centre_mode="self", bg_k=32, bg_radius_factor=8.0)
    ms, _, tiles, _, fz = _host_sets(cfg, ref="fz", other="fz")
    pos, vel = _particles(fz, tiles)
    loss, diag = member_gather_loss(pos, vel, ms, cfg)
    assert float(loss) == pytest.approx(0.0, abs=1e-10)
    for k in ("term_virial", "term_bound", "term_d6", "term_rrms",
              "term_sigmav", "term_centre"):
        assert diag[k] == pytest.approx(0.0, abs=1e-10)


def test_the_host_guard_fires_when_a_host_is_destroyed():
    """Spreading and heating the host past its frozen state is charged.

    The guard's whole reason to exist: a run that fragments a resolved host --
    diffuses its particles and scrambles their velocities -- must pay for it. The
    frozen field is the reference, so this perturbation is unambiguously 'worse
    than where it started'.
    """
    cfg = _cfg(min_num_p=50, centre_mode="self", bg_k=32, bg_radius_factor=8.0)
    ms, _, tiles, _, fz = _host_sets(cfg, ref="fz", other="fz")
    # Diffuse the displacement channels and heat the velocity channels: exactly
    # the destruction the Rockstar A/B measured (a host coming apart).
    wrecked = fz.clone()
    wrecked[:, 0:3] = wrecked[:, 0:3] * 3.0
    wrecked[:, 3:6] = wrecked[:, 3:6] * 3.0
    pos, vel = _particles(wrecked, tiles)
    loss, diag = member_gather_loss(pos, vel, ms, cfg)
    assert float(loss) > 1e-3
    # A puffed-up host both unbinds (bound falls below frozen) and grows in
    # size, so at least the bound and r_rms guards must be positive.
    assert diag["term_bound"] > 0.0
    assert diag["term_rrms"] > 0.0


def test_descent_on_the_loss_actually_binds_a_set():
    """The miniature of the whole feasibility question, and the reason it is cheap.

    ``scripts/features/free_field_gather.py`` asks exactly this on the real
    cluster: optimise the field directly, with no network, and see whether the
    objective admits bound objects. If plain Adam cannot drive ``bound_frac``
    off zero on a toy where the answer is reachable by construction, the loss is
    wrong and no amount of GPU time on a generator fixes it -- which is the
    lesson ``docs/sr2_gather_finetune.md`` paid four Rockstar gates to learn.
    """
    cfg = _cfg(bg_k=32, bg_radius_factor=8.0)
    ms, rep, _, fz, tiles = _sets(cfg)
    assert rep["reference_median"]["bound_hard"] > 0.2      # reachable, by build

    delta = torch.zeros_like(fz, requires_grad=True)
    # 1e-2 on the normalised field is ~3 km/s and ~60 kpc/h a step; the
    # velocity gap is the long one, at ~1000 km/s to close.
    opt = torch.optim.Adam([delta], lr=1e-2)

    def probe(f):
        pos, vel = tile_particles(f, tiles, ng_hr=NG, tile_hr=TILE,
                                  boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                                  vel_scale_kms=VEL)
        return member_gather_loss(pos, vel, ms, cfg)

    start_loss, start = probe(fz)
    assert start["median_bound_hard"] == 0.0

    for _ in range(600):
        opt.zero_grad(set_to_none=True)
        loss, _ = probe(fz + delta)
        loss.backward()
        opt.step()

    end_loss, end = probe(fz + delta)
    assert float(end_loss.detach()) < 0.5 * float(start_loss.detach())
    assert end["median_bound_hard"] > 0.0
    assert end["median_r_rms_over_hr"] < start["median_r_rms_over_hr"]
    assert end["median_sigma_v_over_hr"] < start["median_sigma_v_over_hr"]
    assert abs(np.log(end["median_virial_over_hr"])) \
        < abs(np.log(start["median_virial_over_hr"]))


# --------------------------------------------------------------------------- #
# The centroid term -- added after the 2026-08-21 free-field run
# --------------------------------------------------------------------------- #
def test_the_centre_term_is_zero_at_the_reference_and_grows_with_offset():
    """Every other term is an internal moment; without this one nothing says WHERE.

    The run that exposed the gap built genuinely bound objects -- 156 subhalos in
    R_vir against a base of 11 -- a median 0.414 Mpc/h from their targets, with a
    0.150 Mpc/h search radius and 96% of misses holding no halo of any mass. The
    objects were right and the addresses were wrong.
    """
    cfg = _cfg(bg_k=0)
    ms, _, hr, _, tiles = _sets(cfg)

    def terms(f):
        pos, vel = tile_particles(f, tiles, ng_hr=NG, tile_hr=TILE,
                                  boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                                  vel_scale_kms=VEL)
        return member_gather_loss(pos, vel, ms, cfg)[1]

    at_ref = terms(hr)
    assert at_ref["term_centre"] == pytest.approx(0.0, abs=1e-12)
    assert at_ref["median_centre_offset_mpc_h"] == pytest.approx(0.0, abs=1e-9)

    # Translate every particle by a known amount: the term must see exactly that.
    shift = 0.30
    moved = hr.clone()
    moved[:, 0] += shift / DIS
    out = terms(moved)
    assert out["median_centre_offset_mpc_h"] == pytest.approx(shift, abs=1e-6)
    assert out["term_centre"] > 0.0
    # ... and nothing else, because a rigid translation changes no internal moment.
    for k in ("term_virial", "term_rrms", "term_sigmav"):
        assert out[k] == pytest.approx(0.0, abs=1e-9), k


def test_the_centre_term_is_measured_in_gate_search_radii():
    """One unit of the term is one Rockstar search radius, so it is readable."""
    cfg = _cfg(bg_k=0)
    ms, rep, hr, _, tiles = _sets(cfg)
    scale = float(ms.centre_scale[0])
    assert scale == pytest.approx(max(200.0 / 1000.0, 0.15))   # toy r_vir, kpc/h
    assert rep["median_centre_scale_mpc_h"] == pytest.approx(
        float(np.median(ms.centre_scale.numpy())))

    moved = hr.clone()
    moved[:, 0] += scale / DIS      # one search radius for set 0, by construction
    pos, vel = tile_particles(moved, tiles, ng_hr=NG, tile_hr=TILE,
                              boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                              vel_scale_kms=VEL)
    d = member_gather_loss(pos, vel, ms, cfg)[1]
    # Each set carries its own r_vir, so the same shift is a different number of
    # radii for each -- which is the point of normalising per set rather than by
    # one global length.
    per_set = scale / ms.centre_scale.numpy()
    assert d["median_centre_offset_radii"] == pytest.approx(
        float(np.median(per_set)), rel=1e-4)
    assert d["term_centre"] == pytest.approx(float((per_set ** 2).mean()),
                                             rel=1e-4)
    assert per_set.min() == pytest.approx(1.0, rel=1e-9)


def test_descent_closes_the_centre_offset():
    cfg = _cfg(bg_k=0)
    ms, _, hr, _, tiles = _sets(cfg)
    start = hr.clone()
    start[:, 0] += 0.5 / DIS
    delta = torch.zeros_like(start, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=1e-2)

    def probe(f):
        pos, vel = tile_particles(f, tiles, ng_hr=NG, tile_hr=TILE,
                                  boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                                  vel_scale_kms=VEL)
        return member_gather_loss(pos, vel, ms, cfg)

    before = probe(start)[1]["median_centre_offset_mpc_h"]
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        probe(start + delta)[0].backward()
        opt.step()
    after = probe(start + delta)[1]["median_centre_offset_mpc_h"]
    assert after < 0.2 * before, (before, after)


def test_masking_the_gradient_keeps_untouched_particles_bit_identical():
    """The guard's gradient is not confined, and Adam amplifies it to a full step.

    Without the mask the 2026-08-21 run moved 99.56% of the tile at a median of
    0.54 Mpc/h. The loss alone is surgical; the LR-scale guard is what leaks, and
    the mask is what puts it back.
    """
    cfg = _cfg(bg_k=0)
    ms, _, _, fz, tiles = _sets(cfg)
    touched = torch.zeros(len(tiles) * TILE ** 3, dtype=torch.bool)
    for r in ms.live_rows:
        touched[r] = True
    mask = touched.view(len(tiles), TILE, TILE, TILE).unsqueeze(1).to(fz.dtype)

    def run(masked: bool):
        delta = torch.zeros_like(fz, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=1e-2)
        for _ in range(30):
            opt.zero_grad(set_to_none=True)
            pos, vel = tile_particles(fz + delta, tiles, ng_hr=NG, tile_hr=TILE,
                                      boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                                      vel_scale_kms=VEL)
            loss, _ = member_gather_loss(pos, vel, ms, cfg)
            # A block-averaged guard, the same shape as the run's: it couples
            # every cell of a block to every other, which is the leak.
            b, nb = 2, TILE // 2        # block-average over 2^3 cells
            shp = (len(tiles), 6, nb, b, nb, b, nb, b)
            blk = (fz + delta).reshape(shp).mean(dim=(3, 5, 7))
            ref = fz.reshape(shp).mean(dim=(3, 5, 7))
            (loss + 100.0 * (blk - ref).pow(2).mean()).backward()
            if masked:
                delta.grad.mul_(mask)
            opt.step()
        d = delta.detach().permute(0, 2, 3, 4, 1).reshape(-1, 6)
        return float(d[~touched].abs().max())

    assert run(masked=False) > 1e-3        # the leak, reproduced
    assert run(masked=True) == 0.0         # and closed, exactly


# --------------------------------------------------------------------------
# The max_sets cap, which started binding when the tiling widened
#
# At four tiles the selection was 154 sets and the cap never fired. At sixteen
# it is 625 and the cap keeps 256, so a run that reports only "n_sets" reads as
# though the live-fraction cut removed 369 objects. The cap must be visible in
# the report, or the coverage a wider tiling bought is silently thrown away.
# --------------------------------------------------------------------------

def test_the_max_sets_cap_is_reported_and_not_hidden_in_the_live_cut():
    ms, rep, _, _, _ = _sets(_cfg(max_sets=1))
    assert rep["cap_binds"] is True
    assert rep["max_sets"] == 1
    assert rep["n_after_purity"] == 2
    assert rep["n_dropped_by_cap"] == 1
    assert ms.n_sets == 1
    # and it keeps the LARGEST set, which is also the costliest pair sum
    assert ms.halo_id.tolist() == [10]


def test_a_cap_that_does_not_bind_says_so():
    ms, rep, _, _, _ = _sets(_cfg())
    assert rep["cap_binds"] is False
    assert rep["n_dropped_by_cap"] == 0
    assert ms.n_sets == rep["n_after_purity"]


# --------------------------------------------------------------------------- #
# The pair sums: memory, and the estimator that survives being made cheap
#
# The 2026-08-23 rung ladder died on three GPUs inside the potential, and the
# reason was not the model: chunking bounded the FORWARD block and left the tape
# at O(sum_s N_s^2), because autograd saves every chunk's (c, N, 3) difference
# until backward. The fix recomputes each block in an analytic backward, so what
# needs pinning is that it is the SAME function it was -- value and gradient --
# and that the block is now bounded in elements rather than in rows.
# --------------------------------------------------------------------------- #
def _phi_reference(pos, m, eps, kind):
    """The pre-2026-08-23 implementation, written out. The thing to match."""
    d = pos[:, None, :] - pos[None, :, :]
    r2 = (d * d).sum(-1)
    if kind == "clamp":
        inv = 1.0 / torch.clamp(torch.sqrt(r2.clamp_min(1e-30)), min=eps)
    else:
        inv = torch.rsqrt(r2 + eps * eps)
    n = pos.shape[0]
    inv = inv.clone()
    inv[torch.arange(n), torch.arange(n)] = 0.0
    return -4.30091e-9 * m * inv.sum(1)


@pytest.mark.parametrize("kind", ["plummer", "clamp"])
@pytest.mark.parametrize("chunk", [4096, 7, 1])
def test_the_potential_is_unchanged_by_the_rewrite(kind, chunk):
    x = torch.randn(41, 3, dtype=torch.float64) * 0.3
    got = specific_potential_torch(x, M_P, softening_mpc_h=0.02, kind=kind,
                                   chunk=chunk)
    want = _phi_reference(x, M_P, 0.02, kind)
    assert torch.allclose(got, want, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("kind", ["plummer", "clamp"])
def test_the_analytic_backward_matches_autograd(kind):
    """The gradient is written out, not differentiated, so it needs pinning."""
    x0 = torch.randn(29, 3, dtype=torch.float64) * 0.25
    g = torch.randn(29, dtype=torch.float64)

    a = x0.clone().requires_grad_(True)
    (specific_potential_torch(a, M_P, softening_mpc_h=0.02, kind=kind,
                              chunk=5) * g).sum().backward()
    b = x0.clone().requires_grad_(True)
    (_phi_reference(b, M_P, 0.02, kind) * g).sum().backward()

    assert torch.allclose(a.grad, b.grad, rtol=1e-9, atol=1e-12)


def test_the_pair_block_is_bounded_in_elements_not_rows():
    """`pot_chunk` alone was not a memory bound, and that is what OOMed."""
    from cosmo_sr.features.member_gather import _pair_rows

    # A 118,000-particle satellite -- the set the traceback of job 35748 decodes
    # to -- at the old chunk of 2048 is a 2048 x 118000 x 3 block: 2.7 GiB, and
    # it is the allocation that failed on a 48 GB card.
    assert _pair_rows(118_000, 2048, 1 << 24) == 142
    assert 142 * 118_000 * 3 * 4 < 250e6          # bytes, per block
    # Small sets are untouched: the budget never makes anything slower than the
    # chunk it was asked for.
    assert _pair_rows(200, 2048, 1 << 24) == 2048
    assert _pair_rows(0, 2048, 1 << 24) == 1


def test_subsampling_leaves_the_virial_ratio_unbiased():
    """The cap is a TIME knob, and it may not move the statistic it prices."""
    torch.manual_seed(3)
    n, k = 1200, 200
    x = torch.randn(n, 3, dtype=torch.float64) * 0.2
    v = torch.randn(n, 3, dtype=torch.float64) * 40.0
    cfg = _cfg(softening_kind="plummer")
    full = set_statistics_torch(x, v, particle_mass_msun_h=M_P, cfg=cfg)

    got = []
    for trial in range(24):
        torch.manual_seed(100 + trial)
        take = torch.randperm(n)[:k]
        st = set_statistics_torch(x[take], v[take], particle_mass_msun_h=M_P,
                                  cfg=cfg, pot_mass_factor=(n - 1) / (k - 1))
        got.append(float(st.virial))
    mean = float(np.mean(got))
    # 24 draws of a sixth of the set: the mean must sit on the full-set value,
    # and WITHOUT the (N-1)/(K-1) rescaling it would be ~6x off, so the
    # tolerance is nowhere near able to hide a missing correction.
    assert mean == pytest.approx(float(full.virial), rel=0.05)


def test_the_uncorrected_pair_sum_would_be_wrong_by_the_sampling_ratio():
    """Guards the rescaling itself: drop it and the virial ratio moves ~N/K."""
    torch.manual_seed(4)
    n, k = 900, 150
    x = torch.randn(n, 3, dtype=torch.float64) * 0.2
    v = torch.randn(n, 3, dtype=torch.float64) * 40.0
    cfg = _cfg(softening_kind="plummer")
    take = torch.randperm(n)[:k]
    bad = set_statistics_torch(x[take], v[take], particle_mass_msun_h=M_P,
                              cfg=cfg, pot_mass_factor=1.0)
    good = set_statistics_torch(x[take], v[take], particle_mass_msun_h=M_P,
                                cfg=cfg, pot_mass_factor=(n - 1) / (k - 1))
    assert float(bad.virial) / float(good.virial) == pytest.approx(
        (n - 1) / (k - 1), rel=0.02)


def test_the_cap_is_off_by_default_and_changes_nothing():
    cfg = _cfg(bg_k=32, bg_radius_factor=8.0)
    ms, _, _, fz, tiles = _sets(cfg)
    pos, vel = tile_particles(fz, tiles, ng_hr=NG, tile_hr=TILE,
                              boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                              vel_scale_kms=VEL)
    a, _ = member_gather_loss(pos, vel, ms, cfg)
    b, _ = member_gather_loss(pos, vel, ms, cfg)
    assert float(a) == float(b)               # deterministic, no draw happened


# --------------------------------------------------------------------------- #
# The centre term's shape
# --------------------------------------------------------------------------- #
def test_the_centre_shaping_defaults_to_the_term_that_scored_72_of_154():
    from cosmo_sr.features.member_gather import _centre_cost
    u = torch.linspace(0.0, 6.0, 13, dtype=torch.float64)
    assert torch.allclose(_centre_cost(u, _cfg()), u ** 2)


def test_the_dead_zone_costs_nothing_and_pulls_nothing_inside_it():
    from cosmo_sr.features.member_gather import _centre_cost
    cfg = _cfg(centre_dead_zone=0.5)
    u = torch.tensor([0.0, 0.2, 0.49], dtype=torch.float64, requires_grad=True)
    c = _centre_cost(u, cfg)
    assert float(c.sum().detach()) == 0.0
    c.sum().backward()
    # Zero VALUE with a live gradient would be the worst of both: the gate is a
    # threshold at one radius and a run must be free to spend its capacity
    # elsewhere once it is inside.
    assert float(u.grad.abs().max()) == 0.0


def test_the_huber_arm_is_continuous_in_value_and_slope():
    from cosmo_sr.features.member_gather import _centre_cost
    cfg = _cfg(centre_dead_zone=0.3, centre_huber_radii=2.0)
    e = 1e-6
    u = torch.tensor([2.0 - e, 2.0, 2.0 + e], dtype=torch.float64,
                     requires_grad=True)
    c = _centre_cost(u, cfg)
    assert float(c[0]) == pytest.approx(float(c[2]), abs=1e-4)
    c.sum().backward()
    assert float(u.grad[0]) == pytest.approx(float(u.grad[2]), rel=1e-4)


def test_the_huber_arm_still_pulls_a_hopeless_set_but_does_not_own_the_batch():
    """The measured frozen offset is a median 5.6 radii. That is the case."""
    from cosmo_sr.features.member_gather import _centre_cost
    cfg = _cfg(centre_dead_zone=0.3, centre_huber_radii=2.0)
    far = torch.tensor([5.6], dtype=torch.float64, requires_grad=True)
    c = _centre_cost(far, cfg)
    c.backward()
    assert float(c) < 0.6 * 5.6 ** 2          # the tail no longer sets the scale
    assert float(far.grad) > 0.0              # but it is still pulled inward
    assert _centre_cost(torch.tensor([5.6]), cfg) > \
        _centre_cost(torch.tensor([2.0]), cfg)   # and monotone, so the gate's
                                                 # ordering of sets is preserved


def test_the_shaping_never_charges_more_than_the_quadratic_did():
    from cosmo_sr.features.member_gather import _centre_cost
    cfg = _cfg(centre_dead_zone=0.3, centre_huber_radii=2.0)
    u = torch.linspace(0.0, 12.0, 61, dtype=torch.float64)
    assert bool((_centre_cost(u, cfg) <= u ** 2 + 1e-12).all())


def test_a_huber_knee_inside_the_dead_zone_is_refused():
    with pytest.raises(ValueError, match="discontinuous"):
        _cfg(centre_dead_zone=1.0, centre_huber_radii=0.5)


# --------------------------------------------------------------------------- #
# Minibatching over sets
# --------------------------------------------------------------------------- #
def test_a_set_subset_is_a_mean_over_that_subset_only():
    cfg = _cfg(bg_k=32, bg_radius_factor=8.0)
    ms, _, _, fz, tiles = _sets(cfg)
    pos, vel = tile_particles(fz, tiles, ng_hr=NG, tile_hr=TILE,
                              boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                              vel_scale_kms=VEL)
    both, dboth = member_gather_loss(pos, vel, ms, cfg)
    one, d0 = member_gather_loss(pos, vel, ms, cfg, [0])
    two, d1 = member_gather_loss(pos, vel, ms, cfg, [1])
    assert d0["n_sets"] == 1 and d0["n_sets_total"] == 2
    assert dboth["n_sets"] == 2
    # A mean, so the gradient scale does not depend on how many were drawn.
    assert float(both) == pytest.approx(0.5 * (float(one) + float(two)), rel=1e-5)


def test_a_set_subset_reports_that_subset_s_diagnostics():
    """The per-set medians must be over the sets evaluated, not over all of them."""
    cfg = _cfg(bg_k=32, bg_radius_factor=8.0)
    ms, _, _, fz, tiles = _sets(cfg)
    pos, vel = tile_particles(fz, tiles, ng_hr=NG, tile_hr=TILE,
                              boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                              vel_scale_kms=VEL)
    _, d0 = member_gather_loss(pos, vel, ms, cfg, [0])
    _, d1 = member_gather_loss(pos, vel, ms, cfg, [1])
    assert d0["rows"][0]["halo_id"] != d1["rows"][0]["halo_id"]
    # median_x_over_hr divides by the reference of the SAME set; indexing the
    # reference with the wrong subset is silent and would corrupt every logged
    # ratio, so it is pinned against the per-row arithmetic.
    for d in (d0, d1):
        s = int(np.flatnonzero(ms.halo_id == d["rows"][0]["halo_id"])[0])
        assert d["median_r_rms_over_hr"] == pytest.approx(
            d["rows"][0]["r_rms"] / float(ms.ref["r_rms"][s]), rel=1e-6)


def test_an_empty_subset_is_zero_rather_than_a_nan():
    cfg = _cfg(bg_k=32, bg_radius_factor=8.0)
    ms, _, _, fz, tiles = _sets(cfg)
    pos, vel = tile_particles(fz, tiles, ng_hr=NG, tile_hr=TILE,
                              boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                              vel_scale_kms=VEL)
    loss, diag = member_gather_loss(pos, vel, ms, cfg, [])
    assert float(loss) == 0.0 and diag["n_sets"] == 0


# --------------------------------------------------------------------------- #
# What the centre term charges for: the three arms
#
# `centre_offset/pool/offsets.json`, 7,560 supervised sets, measured the target
# the `full` term asks for: 62.9% of its squared magnitude is radial against an
# isotropic null of 1/3, `o_par` is negative for 70% of sets -- a signed,
# systematic infall deficit -- and yet a linear fit on the features a generator
# can condition on explains only 11.5% of it out of sample. So the direction is
# a rule and the address is not. `radial` charges for the rule alone; `self`
# charges for no address at all. Both are strictly weaker than `full`, which is
# why the defaults do not move and why these tests pin that they do not.
# --------------------------------------------------------------------------- #
def test_the_centre_mode_defaults_to_the_full_offset():
    """An unset run is the objective that scored 72/154, not a new one."""
    assert MemberGatherConfig().centre_mode == "full"
    cfg = _cfg(bg_k=0)
    ms, _, hr, _, tiles = _sets(cfg, host_pos=HOST)
    pos, vel = _particles(hr, tiles)
    d = member_gather_loss(pos, vel, ms, cfg)[1]
    assert d["centre_mode"] == "full"
    # In `full` the penalised quantity and the gated quantity are the same
    # thing, which is exactly what the other two arms give up.
    assert d["median_centre_penalised_radii"] == pytest.approx(
        d["median_centre_offset_radii"], abs=1e-12)


def test_an_unknown_centre_mode_is_refused():
    with pytest.raises(ValueError, match="centre_mode"):
        _cfg(centre_mode="tangential")


def test_rhat_points_from_the_host_to_the_target_and_is_a_unit_vector():
    ms, rep, _, _, _ = _sets(_cfg(bg_k=0), host_pos=HOST)
    assert rep["host_pos_supplied"] is True
    rh = ms.centre_rhat.numpy()
    assert np.allclose(np.linalg.norm(rh, axis=1), 1.0, atol=1e-9)
    # Subhalo 10 lies at [10,10,10] and the host at [20,20,20], so the outward
    # direction is -1/sqrt(3) on each axis; subhalo 11 at [30,30,30] is +.
    order = {int(h): i for i, h in enumerate(ms.halo_id)}
    third = 1.0 / np.sqrt(3.0)
    assert np.allclose(rh[order[10]], -third, atol=0.02)
    assert np.allclose(rh[order[11]], +third, atol=0.02)


def test_radial_mode_is_blind_to_a_purely_transverse_displacement():
    """The 37% of the offset with no direction to predict from costs nothing.

    This is the whole point of the arm and also its whole cost: a set pushed
    sideways by three search radii satisfies the term completely while missing
    the gate, so the diagnostic keeps both numbers.
    """
    cfg = _cfg(bg_k=0, centre_mode="radial", w_virial=0.0, w_bound=0.0,
               w_d6=0.0, w_rrms=0.0, w_sigmav=0.0, w_centre=1.0)
    ms, _, hr, _, tiles = _sets(cfg, host_pos=HOST)
    pos, vel = _particles(hr, tiles)

    # One set at a time: `rhat` runs from the host to the set's own TARGET
    # centroid, which is scattered a little off the catalog centre, so the
    # perpendicular is per set rather than one direction shared by both.
    for i in range(ms.n_sets):
        rh = ms.centre_rhat[i]
        perp = torch.linalg.cross(
            rh, torch.tensor([0.0, 0.0, 1.0], dtype=rh.dtype))
        perp = perp / torch.linalg.vector_norm(perp)
        d = member_gather_loss(pos + 0.9 * perp, vel, ms, cfg,
                               set_indices=[i])[1]
        assert d["term_centre"] == pytest.approx(0.0, abs=1e-12)
        assert d["median_centre_penalised_radii"] == pytest.approx(0.0, abs=1e-9)
        # ...while the offset the gate scores is fully 0.9 Mpc/h away.
        assert d["median_centre_offset_mpc_h"] == pytest.approx(0.9, abs=1e-6)
        assert d["frac_centre_within_1_radius"] == 0.0


def test_radial_mode_charges_a_radial_displacement_exactly_as_full_does():
    ms, _, hr, _, tiles = _sets(_cfg(bg_k=0), host_pos=HOST)
    pos, vel = _particles(hr, tiles)

    def term(mode, shift, i):
        cfg = _cfg(bg_k=0, centre_mode=mode, w_virial=0.0, w_bound=0.0,
                   w_d6=0.0, w_rrms=0.0, w_sigmav=0.0, w_centre=1.0)
        return member_gather_loss(pos + shift, vel, ms, cfg,
                                  set_indices=[i])[1]

    for i in range(ms.n_sets):
        # Purely radial for THIS set: the two arms then see the same vector and
        # must charge the same number, which is the other half of the claim.
        shift = 0.4 * ms.centre_rhat[i]
        full, rad = term("full", shift, i), term("radial", shift, i)
        assert rad["term_centre"] == pytest.approx(full["term_centre"], rel=1e-9)
        assert rad["median_centre_offset_mpc_h"] == pytest.approx(
            full["median_centre_offset_mpc_h"], abs=1e-9)


def test_radial_mode_without_a_host_position_is_refused():
    """Silently charging zero would make this a w_centre=0 run under a new name."""
    cfg = _cfg(bg_k=0, centre_mode="radial")
    ms, rep, hr, _, tiles = _sets(cfg)          # no host_pos
    assert rep["host_pos_supplied"] is False
    pos, vel = _particles(hr, tiles)
    with pytest.raises(ValueError, match="host_pos"):
        member_gather_loss(pos, vel, ms, cfg)


def test_self_mode_is_exactly_zero_on_the_frozen_field():
    """"Concentrate where SR2 already put you": no address, and no step-0 pull.

    The anchor is built from the same collection `_gather_one` assembles -- live
    rows plus the frozen stragglers -- so this is zero by construction rather
    than by luck, and a non-zero value here would mean the two disagree.
    """
    cfg = _cfg(bg_k=0, centre_mode="self", w_virial=0.0, w_bound=0.0,
               w_d6=0.0, w_rrms=0.0, w_sigmav=0.0, w_centre=1.0)
    ms, rep, _, fz, tiles = _sets(cfg, host_pos=HOST)
    pos, vel = _particles(fz, tiles)
    d = member_gather_loss(pos, vel, ms, cfg)[1]
    assert d["term_centre"] == pytest.approx(0.0, abs=1e-12)
    assert d["median_centre_penalised_radii"] == pytest.approx(0.0, abs=1e-9)
    # And the offset the gate scores is emphatically NOT zero -- the toy's
    # frozen subhalos are diffuse and displaced, as the real ones are.
    assert d["median_centre_offset_radii"] > 0.2
    assert rep["median_frozen_centre_offset_radii"] == pytest.approx(
        d["median_centre_offset_radii"], rel=1e-6)


def test_self_mode_still_anchors_against_drift():
    """It is an anchor, not an absence: leaving the frozen centroid costs."""
    cfg = _cfg(bg_k=0, centre_mode="self", w_virial=0.0, w_bound=0.0,
               w_d6=0.0, w_rrms=0.0, w_sigmav=0.0, w_centre=1.0)
    ms, _, _, fz, tiles = _sets(cfg, host_pos=HOST)
    pos, vel = _particles(fz, tiles)
    scale = ms.centre_scale.numpy()
    shift = torch.tensor([0.3, 0.0, 0.0])
    d = member_gather_loss(pos + shift, vel, ms, cfg)[1]
    assert d["term_centre"] == pytest.approx(
        float(((0.3 / scale) ** 2).mean()), rel=1e-5)


def test_every_mode_reports_the_offset_the_gate_will_actually_score():
    """The gap between the two columns IS the result of the weaker arms."""
    ms, _, hr, _, tiles = _sets(_cfg(bg_k=0), host_pos=HOST)
    pos, vel = _particles(hr, tiles)
    perp = torch.tensor([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    for mode in ("full", "radial", "self"):
        cfg = _cfg(bg_k=0, centre_mode=mode)
        d = member_gather_loss(pos + 0.5 * perp, vel, ms, cfg)[1]
        assert d["centre_mode"] == mode
        # Independent of the mode, because it is measured from centre_target.
        assert d["median_centre_offset_mpc_h"] == pytest.approx(0.5, abs=1e-6)


def test_the_shaping_knobs_compose_with_every_mode():
    """Huber and the dead zone shape whatever the mode selected, not just `full`."""
    ms, _, hr, _, tiles = _sets(_cfg(bg_k=0), host_pos=HOST)
    pos, vel = _particles(hr, tiles)
    third = 1.0 / np.sqrt(3.0)
    far = torch.tensor([third, third, third]) * 1.5      # ~10 radii, radial

    def term(**kw):
        cfg = _cfg(bg_k=0, w_virial=0.0, w_bound=0.0, w_d6=0.0, w_rrms=0.0,
                   w_sigmav=0.0, w_centre=1.0, **kw)
        return member_gather_loss(pos + far, vel, ms, cfg)[1]["term_centre"]

    for mode in ("full", "radial"):
        quad = term(centre_mode=mode)
        huber = term(centre_mode=mode, centre_huber_radii=2.0)
        assert huber < quad, mode
    # Inside the dead zone nothing is charged, in any mode.
    near = torch.tensor([third, third, third]) * 0.02
    for mode in ("full", "radial"):
        cfg = _cfg(bg_k=0, centre_mode=mode, centre_dead_zone=1.0,
                   w_virial=0.0, w_bound=0.0, w_d6=0.0, w_rrms=0.0,
                   w_sigmav=0.0, w_centre=1.0)
        d = member_gather_loss(pos + near, vel, ms, cfg)[1]
        assert d["term_centre"] == pytest.approx(0.0, abs=1e-12), mode


# --------------------------------------------------------------------------- #
# The loss BUDGET
#
# *Measured*, step 0 of the 2026-08-24 generator pool (holdout, weighted):
# d6 111.78, virial 14.05, bound 0.31, rrms 0.61, sigmav 0.36. So `d6` held 88%
# of the gradient and `bound` -- the term that IS Rockstar's decision rule --
# held 0.24%, and after 500 steps that collapsed d6 32x it had moved 0.31->0.21.
# That split is not a design decision about importance. It is decided by which
# side of its reference the frozen field starts on, because `_hinge_below` is
# capped at 1 for any x >= 0 while `_hinge_above` is unbounded.
# --------------------------------------------------------------------------- #
def test_the_bound_hinge_is_capped_at_one_and_the_log_form_is_not():
    """The diagnosis, as arithmetic. This cap is the whole budget problem."""
    from cosmo_sr.features.member_gather import _hinge_below
    ref = torch.tensor(0.534)
    for frac in (0.5, 0.1, 0.01, 1e-4, 1e-8):
        x = ref * frac
        # Capped: no deficit, however total, can charge more than 1.
        assert float(_hinge_below(x, ref, "hinge")) <= 1.0
        # The log form is unbounded, and past a factor of e it is already over
        # the hinge's entire range.
        if frac < 0.3:
            assert float(_hinge_below(x, ref, "log")) > 1.0
    # The measured starting point: bound_soft at ~1.3% of HR's.
    x = ref * 0.0131
    assert float(_hinge_below(x, ref, "hinge")) == pytest.approx(0.974, abs=1e-3)
    assert float(_hinge_below(x, ref, "log")) == pytest.approx(18.79, rel=1e-2)
    # ~19x the charge, which is the point of the change.
    assert (float(_hinge_below(x, ref, "log"))
            / float(_hinge_below(x, ref, "hinge"))) > 15.0


def test_both_bound_penalties_are_exactly_zero_for_beating_the_reference():
    """The anti-over-sharpening property is why the hinge exists. It survives.

    `pilot_steps_2_4.md` step 4 over-sharpened because a two-sided term asked
    for more than HR had. Neither form may reintroduce that.
    """
    from cosmo_sr.features.member_gather import _hinge_below
    ref = torch.tensor(0.534)
    for kind in ("hinge", "log"):
        assert float(_hinge_below(ref, ref, kind)) == pytest.approx(0.0, abs=1e-12)
        for over in (1.001, 1.5, 10.0):
            assert float(_hinge_below(ref * over, ref, kind)) == 0.0, kind


def test_the_log_penalty_has_gradient_where_the_hinge_is_nearly_flat():
    from cosmo_sr.features.member_gather import _hinge_below
    ref = torch.tensor(0.534)
    g = {}
    for kind in ("hinge", "log"):
        x = (ref * 0.0131).clone().requires_grad_(True)
        _hinge_below(x, ref, kind).backward()
        g[kind] = abs(float(x.grad))
    assert g["log"] > 10.0 * g["hinge"]


def test_the_bound_penalty_defaults_to_the_form_every_run_used():
    assert MemberGatherConfig().bound_penalty == "hinge"
    with pytest.raises(ValueError, match="bound_penalty"):
        _cfg(bound_penalty="huber")


def test_the_log_penalty_raises_the_bound_term_in_the_real_loss():
    cfg_h = _cfg(bg_k=0, bound_penalty="hinge")
    cfg_l = _cfg(bg_k=0, bound_penalty="log")
    ms, _, _, fz, tiles = _sets(cfg_h, host_pos=HOST)
    pos, vel = _particles(fz, tiles)
    a = member_gather_loss(pos, vel, ms, cfg_h)[1]
    b = member_gather_loss(pos, vel, ms, cfg_l)[1]
    # The frozen toy is unbound, so the deficit is real and log charges more.
    assert b["term_bound"] > a["term_bound"]
    # And only that term: nothing else in the loss is touched.
    for k in ("virial", "rrms", "sigmav", "centre"):
        assert b[f"term_{k}"] == pytest.approx(a[f"term_{k}"], rel=1e-9), k


def test_term_norm_makes_the_declared_weights_the_actual_budget():
    """Without it the split is the terms' dynamic ranges; with it, the weights."""
    cfg = _cfg(bg_k=0)
    ms, _, _, fz, tiles = _sets(cfg, host_pos=HOST)
    pos, vel = _particles(fz, tiles)
    raw = member_gather_loss(pos, vel, ms, cfg)[1]
    scale = {k: raw[f"term_{k}"] for k in ("virial", "bound", "d6", "rrms",
                                           "sigmav", "centre")
             if raw[f"term_{k}"] > 1e-6}
    loss, d = member_gather_loss(pos, vel, ms, cfg, term_scale=scale)
    # Every scaled term starts at exactly its own weight.
    weights = {"virial": cfg.w_virial, "bound": cfg.w_bound, "d6": cfg.w_d6,
               "rrms": cfg.w_rrms, "sigmav": cfg.w_sigmav,
               "centre": cfg.w_centre}
    for k in scale:
        assert d[f"term_eff_{k}"] == pytest.approx(weights[k], rel=1e-6), k
    # `term_*` stays RAW so it is still comparable across runs...
    for k in scale:
        assert d[f"term_{k}"] == pytest.approx(raw[f"term_{k}"], rel=1e-9), k
    # ...and `term_eff_*` is what summed to the loss.
    assert float(loss) == pytest.approx(
        sum(d[f"term_eff_{k}"] for k in weights), rel=1e-6)
    assert d["term_scale_active"] is True


def test_term_norm_is_off_by_default_and_changes_nothing():
    cfg = _cfg(bg_k=0)
    ms, _, _, fz, tiles = _sets(cfg, host_pos=HOST)
    pos, vel = _particles(fz, tiles)
    a, da = member_gather_loss(pos, vel, ms, cfg)
    b, db = member_gather_loss(pos, vel, ms, cfg, term_scale=None)
    assert float(a) == pytest.approx(float(b), rel=1e-12)
    assert da["term_scale_active"] is False
    # With no scales, `term_eff_*` is just the weighted term.
    assert da["term_eff_virial"] == pytest.approx(
        cfg.w_virial * da["term_virial"], rel=1e-9)


def test_a_term_scale_for_an_unknown_term_is_refused():
    cfg = _cfg(bg_k=0)
    ms, _, _, fz, tiles = _sets(cfg, host_pos=HOST)
    pos, vel = _particles(fz, tiles)
    with pytest.raises(ValueError, match="unknown terms"):
        member_gather_loss(pos, vel, ms, cfg, term_scale={"boundness": 1.0})


def test_a_zero_scale_cannot_blow_up_the_loss():
    """A term already satisfied on the frozen field must not divide by ~0."""
    cfg = _cfg(bg_k=0)
    ms, _, _, fz, tiles = _sets(cfg, host_pos=HOST)
    pos, vel = _particles(fz, tiles)
    loss, d = member_gather_loss(pos, vel, ms, cfg,
                                 term_scale={"virial": 0.0})
    assert np.isfinite(float(loss))
    assert d["term_scale"]["virial"] == pytest.approx(1e-6)


# --------------------------------------------------------------------------- #
# The shell: the right distance from the host, any direction (`w_radius`)
# --------------------------------------------------------------------------- #
def _shell_cfg(**kw):
    """Only the placement terms live, so a number IS the term under test."""
    base = dict(bg_k=0, w_virial=0.0, w_bound=0.0, w_d6=0.0, w_rrms=0.0,
                w_sigmav=0.0, w_centre=0.0, w_radius=1.0, radius_dead_zone=0.0)
    base.update(kw)
    return _cfg(**base)


def test_the_shell_term_is_off_by_default():
    """Every arm in docs/sr2_member_gather_training.md ran without it."""
    assert MemberGatherConfig().w_radius == 0.0
    cfg = _cfg(bg_k=0)
    ms, _, hr, _, tiles = _sets(cfg, host_pos=HOST)
    pos, vel = _particles(hr, tiles)
    d = member_gather_loss(pos, vel, ms, cfg)[1]
    assert d["term_radius"] == 0.0
    assert d["term_eff_radius"] == 0.0


def _on_the_sphere(ms, i, a):
    """A displacement of exactly ``|a|`` sideways that keeps the radius EXACT.

    With ``d = c - h = r rhat``, a shift ``a perp + b rhat`` lands at radius
    ``sqrt((r + b)^2 + a^2)``, so ``b = sqrt(r^2 - a^2) - r`` puts it back on the
    sphere. Second order in ``a/r`` and therefore small, but it is not zero, and
    a test that ignored it would be asserting the dead zone rather than the
    direction-blindness this term exists for.
    """
    rhat = ms.centre_rhat[i]
    perp = torch.linalg.cross(rhat, torch.tensor([0.0, 0.0, 1.0],
                                                 dtype=rhat.dtype))
    perp = perp / torch.linalg.vector_norm(perp)
    r = float(torch.linalg.vector_norm(ms.centre_target[i] - ms.centre_host[i]))
    b = float(np.sqrt(r * r - a * a) - r)
    return a * perp + b * rhat


def test_the_shell_term_is_blind_to_a_move_along_the_sphere():
    """The whole point: wherever it is, as long as the distance is right.

    The transverse direction is the 37% of the centre offset that no input
    determines, so a term that charged for it would be asking the generator for
    realisation noise. Here a set is walked a full search radius sideways and
    the term stays at zero -- while `full` charges the whole thing, which is the
    same fact stated as the cost.
    """
    cfg = _shell_cfg()
    ms, _, hr, _, tiles = _sets(cfg, host_pos=HOST)
    pos, vel = _particles(hr, tiles)
    for i in range(ms.n_sets):
        shift = _on_the_sphere(ms, i, a=1.0)
        d = member_gather_loss(pos + shift, vel, ms, cfg, set_indices=[i])[1]
        assert d["term_radius"] == pytest.approx(0.0, abs=1e-9)
        assert d["median_radius_ratio"] == pytest.approx(1.0, abs=1e-9)
        # ...and it really did move: the address term charges every metre of
        # it, which is the cost this arm is choosing not to pay. Staying on the
        # sphere costs a second-order radial step, so the displacement is
        # sqrt(a^2 + b^2) rather than `a` -- assert what was actually applied.
        moved = float(torch.linalg.vector_norm(shift))
        assert moved == pytest.approx(1.0, abs=1e-3)
        full = member_gather_loss(pos + shift, vel, ms,
                                  _shell_cfg(w_centre=1.0, w_radius=0.0),
                                  set_indices=[i])[1]
        assert full["median_centre_offset_mpc_h"] == pytest.approx(moved, abs=1e-6)
        assert full["term_centre"] > 1.0


def test_the_shell_term_charges_a_radial_move_and_is_sign_symmetric():
    """Too close and too far cost the same; only the distance is specified."""
    cfg = _shell_cfg()
    ms, _, hr, _, tiles = _sets(cfg, host_pos=HOST)
    pos, vel = _particles(hr, tiles)
    for i in range(ms.n_sets):
        s = float(ms.centre_scale[i])
        out = member_gather_loss(pos + 2.0 * s * ms.centre_rhat[i], vel, ms,
                                 cfg, set_indices=[i])[1]
        inn = member_gather_loss(pos - 2.0 * s * ms.centre_rhat[i], vel, ms,
                                 cfg, set_indices=[i])[1]
        # Two search radii out, dead zone off: (2)^2 = 4, on both sides.
        assert out["term_radius"] == pytest.approx(4.0, rel=1e-4)
        assert inn["term_radius"] == pytest.approx(4.0, rel=1e-4)
        # ...and the SIGN survives into the diagnostic, which is how a run says
        # "radially collapsed" rather than merely "misplaced".
        assert out["median_radius_err_radii"] == pytest.approx(2.0, rel=1e-4)
        assert out["frac_radius_too_close"] == 0.0
        assert inn["frac_radius_too_close"] == 1.0
        assert inn["median_radius_ratio"] < 1.0


def test_the_shell_dead_zone_is_free_inside_and_exact_outside():
    """"Not too strict": one search radius of slack, then the usual quadratic."""
    ms, _, hr, _, tiles = _sets(_shell_cfg(), host_pos=HOST)
    pos, vel = _particles(hr, tiles)
    cfg = _shell_cfg(radius_dead_zone=1.0)
    for i in range(ms.n_sets):
        s = float(ms.centre_scale[i])
        near = member_gather_loss(pos + 0.9 * s * ms.centre_rhat[i], vel, ms,
                                  cfg, set_indices=[i])[1]
        far = member_gather_loss(pos + 3.0 * s * ms.centre_rhat[i], vel, ms,
                                 cfg, set_indices=[i])[1]
        assert near["term_radius"] == pytest.approx(0.0, abs=1e-12)
        assert far["term_radius"] == pytest.approx(4.0, rel=1e-4)   # (3-1)^2


def test_self_tan_drops_the_radial_direction_and_keeps_the_transverse_one():
    """The half of `self` that does not fight the shell term."""
    cfg = _cfg(bg_k=0, centre_mode="self_tan", w_virial=0.0, w_bound=0.0,
               w_d6=0.0, w_rrms=0.0, w_sigmav=0.0, w_centre=1.0)
    ms, _, _, fz, tiles = _sets(cfg, host_pos=HOST)
    pos, vel = _particles(fz, tiles)          # cbar == centre_self exactly
    slf = dataclasses.replace(cfg, centre_mode="self")
    for i in range(ms.n_sets):
        rhat = ms.centre_rhat[i]
        perp = torch.linalg.cross(rhat, torch.tensor([0.0, 0.0, 1.0],
                                                     dtype=rhat.dtype))
        perp = perp / torch.linalg.vector_norm(perp)
        s = float(ms.centre_scale[i])

        rad = member_gather_loss(pos + 2.0 * s * rhat, vel, ms, cfg,
                                 set_indices=[i])[1]
        assert rad["term_centre"] == pytest.approx(0.0, abs=1e-9)
        # `self` would have charged the full (2)^2 for the same move.
        assert member_gather_loss(pos + 2.0 * s * rhat, vel, ms, slf,
                                  set_indices=[i])[1]["term_centre"] == \
            pytest.approx(4.0, rel=1e-4)

        tan = member_gather_loss(pos + 2.0 * s * perp, vel, ms, cfg,
                                 set_indices=[i])[1]
        assert tan["term_centre"] == pytest.approx(
            member_gather_loss(pos + 2.0 * s * perp, vel, ms, slf,
                               set_indices=[i])[1]["term_centre"], rel=1e-9)


def test_self_fights_the_shell_term_and_self_tan_does_not():
    """The composition claim, as a number rather than an argument.

    `self` anchors the set's whole position, radius included, so the radial step
    the shell term asks for is a step `self` charges for -- at equal weights the
    two settle at half the correction. `self_tan` gives that one coordinate up.
    """
    ms, _, _, fz, tiles = _sets(_cfg(bg_k=0), host_pos=HOST)
    pos, vel = _particles(fz, tiles)

    def centre_term(mode, shift, i):
        cfg = _cfg(bg_k=0, centre_mode=mode, w_virial=0.0, w_bound=0.0,
                   w_d6=0.0, w_rrms=0.0, w_sigmav=0.0, w_centre=1.0)
        return member_gather_loss(pos + shift, vel, ms, cfg,
                                  set_indices=[i])[1]["term_centre"]

    for i in range(ms.n_sets):
        step = 1.5 * float(ms.centre_scale[i]) * ms.centre_rhat[i]
        assert centre_term("self", step, i) > 1.0          # charged for moving
        assert centre_term("self_tan", step, i) == pytest.approx(0.0, abs=1e-9)


def test_the_shell_term_without_a_host_position_is_refused():
    """Silently measuring from the origin of the box is the failure mode."""
    cfg = _shell_cfg()
    ms, rep, hr, _, tiles = _sets(cfg)                    # no host_pos
    assert rep["host_pos_supplied"] is False
    pos, vel = _particles(hr, tiles)
    with pytest.raises(ValueError, match="w_radius"):
        member_gather_loss(pos, vel, ms, cfg)


def test_self_tan_without_a_host_position_is_refused():
    """It would silently be a plain `self` run -- the exact mode it is not."""
    cfg = _cfg(bg_k=0, centre_mode="self_tan")
    ms, _, hr, _, tiles = _sets(cfg)
    pos, vel = _particles(hr, tiles)
    with pytest.raises(ValueError, match="centre_rhat"):
        member_gather_loss(pos, vel, ms, cfg)


def test_the_radius_diagnostic_is_nan_rather_than_zero_without_a_host():
    """A zero here would read as "the shell is already right"."""
    cfg = _cfg(bg_k=0)
    ms, _, hr, _, tiles = _sets(cfg)                      # no host_pos
    pos, vel = _particles(hr, tiles)
    d = member_gather_loss(pos, vel, ms, cfg)[1]
    assert d["has_host"] is False
    assert np.isnan(d["median_radius_err_radii"])
    assert np.isnan(d["median_radius_ratio"])


def test_the_shell_is_satisfied_by_the_hr_field_it_was_measured_from():
    """Zero cost at the answer, so the term adds nothing to a solved set."""
    cfg = _shell_cfg()
    ms, _, hr, _, tiles = _sets(cfg, host_pos=HOST)
    pos, vel = _particles(hr, tiles)
    loss, d = member_gather_loss(pos, vel, ms, cfg)
    assert float(loss) == pytest.approx(0.0, abs=1e-9)
    assert d["median_radius_ratio"] == pytest.approx(1.0, abs=1e-9)


def test_a_negative_shell_weight_is_refused():
    with pytest.raises(ValueError, match="w_radius"):
        _cfg(w_radius=-1.0)


def test_a_shell_huber_inside_the_dead_zone_is_refused():
    with pytest.raises(ValueError, match="radius_huber_radii"):
        _cfg(radius_dead_zone=2.0, radius_huber_radii=1.0)


def test_the_shell_term_has_a_gradient_where_it_charges():
    """A term with no gradient is a report, not an objective."""
    cfg = _shell_cfg()
    ms, _, hr, _, tiles = _sets(cfg, host_pos=HOST)
    pos, vel = _particles(hr, tiles)
    shift = torch.zeros_like(pos, requires_grad=True)
    loss, _ = member_gather_loss(pos + shift + 2.0 * float(ms.centre_scale[0])
                                 * ms.centre_rhat[0], vel, ms, cfg,
                                 set_indices=[0])
    loss.backward()
    g = shift.grad[ms.live_rows[0]]
    assert torch.isfinite(g).all()
    assert float(g.abs().sum()) > 0.0
    # Pushed out along +rhat, so the DESCENT direction must point back in.
    descent = -g.sum(dim=0)
    descent = descent / torch.linalg.vector_norm(descent)
    assert float(descent @ ms.centre_rhat[0]) == pytest.approx(-1.0, abs=1e-4)


# --------------------------------------------------------------------------- #
# The SOFT shell (`w_shell`): the radius DISTRIBUTION, never an address
# --------------------------------------------------------------------------- #
#: The toy's two subhalos sit at [10,10,10] and [30,30,30], symmetrically about
#: `HOST`, so from it they have the SAME true radius -- and every
#: address-vs-distribution test below would be vacuous, because an exchange of
#: two equal numbers is not an exchange. Displacing the host 1 Mpc/h along the
#: [1,1,1] diagonal moves it toward one and away from the other, so the two
#: radii differ by exactly 2 Mpc/h and a swap is a real swap.
HOST_OFF = HOST + 1.0 / np.sqrt(3.0)


def _soft_cfg(**kw):
    """Only the soft shell is live, so a number IS the term under test."""
    base = dict(bg_k=0, w_virial=0.0, w_bound=0.0, w_d6=0.0, w_rrms=0.0,
                w_sigmav=0.0, w_centre=0.0, w_radius=0.0, w_shell=1.0,
                shell_dead_zone=0.0)
    base.update(kw)
    return _cfg(**base)


def _radii(ms):
    return [float(torch.linalg.vector_norm(ms.centre_target[i]
                                           - ms.centre_host[i]))
            for i in range(ms.n_sets)]


def _move_each(pos, ms, shifts):
    """``pos`` with each set's OWN member rows shifted by its own vector.

    The existing shell tests shift the whole field and evaluate one set with
    ``set_indices``; that cannot express the state this term is about, which is
    two sets at different radii at the same time.
    """
    p = pos.clone()
    for i, d in enumerate(shifts):
        p[ms.live_rows[i]] = p[ms.live_rows[i]] + d
    return p


def _scaled(pos, ms, factors):
    """Every set moved radially to ``factors[i]`` times its true radius."""
    r = _radii(ms)
    return _move_each(pos, ms, [(float(factors[i]) - 1.0) * r[i]
                                * ms.centre_rhat[i] for i in range(ms.n_sets)])


def test_the_soft_shell_is_off_by_default():
    """Every arm recorded so far ran without it, `all_blocks_selfshell` included."""
    assert MemberGatherConfig().w_shell == 0.0
    assert MemberGatherConfig().shell_dead_zone == 0.02
    cfg = _cfg(bg_k=0)
    ms, _, hr, _, tiles = _sets(cfg, host_pos=HOST_OFF)
    pos, vel = _particles(hr, tiles)
    d = member_gather_loss(pos, vel, ms, cfg)[1]
    assert d["term_shell"] == 0.0
    assert d["term_eff_shell"] == 0.0


def test_the_soft_shell_is_zero_on_the_field_it_was_measured_from():
    """The module's standing property: the loss never asks for more than HR."""
    cfg = _soft_cfg()
    ms, _, hr, _, tiles = _sets(cfg, host_pos=HOST_OFF)
    pos, vel = _particles(hr, tiles)
    d = member_gather_loss(pos, vel, ms, cfg)[1]
    assert d["term_shell"] == pytest.approx(0.0, abs=1e-9)
    assert d["median_shell_resid"] == pytest.approx(0.0, abs=1e-5)


def test_the_soft_shell_is_blind_to_an_exchange_of_radii():
    """The defining property, and the whole reason the term exists.

    Two sets are put on each OTHER's true radius. The multiset of radii is
    unchanged, so this term is unchanged -- while `w_radius`, which names every
    set against its own target, charges the entire swap. That difference is the
    point: the address `w_radius` charges for has an out-of-sample R^2 of 0.022
    on the 7,560-set pool, and `all_blocks_selfshell` paid 55% of the abundance
    (host subhalos 367 -> 167 against HR's 374) memorising it.
    """
    cfg = _soft_cfg()
    ms, _, hr, _, tiles = _sets(cfg, host_pos=HOST_OFF)
    pos, vel = _particles(hr, tiles)
    assert ms.n_sets == 2
    r = _radii(ms)
    # The toy's design, to within the member scatter: `centre_target` is the
    # reachable HR CENTROID of each set, not the catalog centre the 1 Mpc/h host
    # displacement was measured against, so 2.03 rather than 2.00.
    assert abs(r[0] - r[1]) == pytest.approx(2.0, rel=0.05)
    swap = _move_each(pos, ms, [(r[1] - r[0]) * ms.centre_rhat[0],
                                (r[0] - r[1]) * ms.centre_rhat[1]])
    assert member_gather_loss(swap, vel, ms, cfg)[1]["term_shell"] == \
        pytest.approx(0.0, abs=1e-5)
    # ...and the address term charges every bit of it: each set now sits |dr| =
    # |r0 - r1| from its own target, which in its own search radii is the cost.
    got = member_gather_loss(swap, vel, ms, _shell_cfg())[1]["term_radius"]
    dr = abs(r[0] - r[1])
    want = float(np.mean([(dr / float(ms.centre_scale[i])) ** 2
                          for i in range(2)]))
    assert got == pytest.approx(want, rel=1e-3)
    assert got > 100.0            # the term it replaces, at its worst


def test_the_soft_shell_charges_a_uniform_radial_scaling_in_log_space():
    """The component it exists to keep: every set the same fraction too close.

    That is the frozen field's defect in miniature -- 58.3% of sets sit inside
    their true radius, median ratio 0.974 -- and it is exactly what survives the
    sort (signed median -0.026 -> -0.041 on train), so it must be charged at
    exactly log(f)^2 and stay signed in the diagnostic.
    """
    cfg = _soft_cfg()
    ms, _, hr, _, tiles = _sets(cfg, host_pos=HOST_OFF)
    pos, vel = _particles(hr, tiles)
    for f in (0.9, 1.1):
        d = member_gather_loss(_scaled(pos, ms, [f, f]), vel, ms, cfg)[1]
        assert d["term_shell"] == pytest.approx(float(np.log(f)) ** 2, rel=1e-3)
        assert d["median_shell_resid_signed"] == pytest.approx(
            float(np.log(f)), rel=1e-3)


def test_the_soft_shell_dead_zone_is_a_fractional_radius_error():
    """`log1p(x)`, so the knob means what it says: 5% is 5% of the radius."""
    cfg = _soft_cfg(shell_dead_zone=0.05)
    ms, _, hr, _, tiles = _sets(cfg, host_pos=HOST_OFF)
    pos, vel = _particles(hr, tiles)
    inside = member_gather_loss(_scaled(pos, ms, [1.04, 1.04]), vel, ms, cfg)[1]
    assert inside["term_shell"] == pytest.approx(0.0, abs=1e-9)
    assert inside["frac_shell_outside_dead_zone"] == 0.0
    outside = member_gather_loss(_scaled(pos, ms, [1.10, 1.10]), vel, ms, cfg)[1]
    assert outside["term_shell"] == pytest.approx(
        (float(np.log(1.10)) - float(np.log1p(0.05))) ** 2, rel=1e-3)
    assert outside["frac_shell_outside_dead_zone"] == 1.0


def test_the_soft_shell_has_no_plateau_below_its_floor():
    """`sqrt(|d|^2 + eps^2)` on the candidate, NOT `clamp_min(eps)`.

    A clamp is a flat region below the floor, and collapse toward the host
    centre is the one failure this term must never stop pushing back on. Two
    candidate radii well inside the floor must cost DIFFERENT amounts; under a
    clamp they would be identical and the gradient there would be zero.
    """
    cfg = _soft_cfg(shell_eps_mpc_h=0.5)
    ms, _, hr, _, tiles = _sets(cfg, host_pos=HOST_OFF)
    pos, vel = _particles(hr, tiles)
    deep = float(member_gather_loss(_scaled(pos, ms, [0.01, 0.01]), vel, ms,
                                    cfg)[1]["term_shell"])
    less = float(member_gather_loss(_scaled(pos, ms, [0.02, 0.02]), vel, ms,
                                    cfg)[1]["term_shell"])
    assert deep > less                      # strictly worse further in
    # ...and the gradient survives a set sitting ON the host centre, where a
    # bare `vector_norm` is not differentiable at all.
    at = _scaled(pos, ms, [0.0, 0.0]).detach().requires_grad_(True)
    loss, _ = member_gather_loss(at, vel, ms, cfg)
    loss.backward()
    assert torch.isfinite(at.grad).all()


def test_the_sorted_pairing_never_costs_more_than_the_address_pairing():
    """The rearrangement inequality, which is why this term can only ask LESS.

    Rank pairing minimises a convex cost of the difference over every pairing,
    so the soft shell is a lower bound on the identity-paired cost of the same
    configuration -- for a monotone-preserving move the two coincide, and for a
    rank-scrambling one the soft term forgives the scramble.
    """
    cfg = _soft_cfg()
    ms, _, hr, _, tiles = _sets(cfg, host_pos=HOST_OFF)
    pos, vel = _particles(hr, tiles)
    rng = np.random.default_rng(0)
    saw_strict = False
    for _ in range(12):
        f = rng.uniform(0.4, 2.0, size=2)
        got = float(member_gather_loss(_scaled(pos, ms, f), vel, ms,
                                       cfg)[1]["term_shell"])
        address = float(np.mean(np.log(f) ** 2))     # the identity pairing
        assert got <= address + 1e-6
        saw_strict |= got < address - 1e-6
    assert saw_strict, "no rank-scrambling draw: the bound was never exercised"


def test_the_soft_shell_without_a_host_position_is_refused():
    """It would otherwise match a distribution of distances from the box origin.

    Same guard, and the same reason, as `w_radius`: the host preservation sets
    are built `host_pos=None` by design, and silently measuring them from the
    origin is the failure mode this refuses.
    """
    cfg = _soft_cfg()
    ms, _, hr, _, tiles = _sets(cfg)                  # host_pos=None
    pos, vel = _particles(hr, tiles)
    with pytest.raises(ValueError, match="w_shell > 0"):
        member_gather_loss(pos, vel, ms, cfg)


def test_the_soft_shell_knobs_refuse_nonsense():
    with pytest.raises(ValueError, match="w_shell"):
        _cfg(w_shell=-1.0)
    with pytest.raises(ValueError, match="shell_dead_zone"):
        _cfg(shell_dead_zone=-0.1)
    with pytest.raises(ValueError, match="shell_eps_mpc_h"):
        _cfg(shell_eps_mpc_h=0.0)


def test_the_shell_residual_is_reported_at_w_shell_zero():
    """Same standing property as `radius_ratio` at `w_radius=0`.

    An arm that cannot report the number cannot be read against one that
    optimises it, and this is the number to read for this term:
    `radius_err_radii` barely moves even on a total win, because a perfect
    distribution match still leaves 79% (train) / 85% (holdout) of the per-set
    address error standing.
    """
    cfg = _cfg(bg_k=0)                                # w_shell = 0
    ms, _, hr, _, tiles = _sets(cfg, host_pos=HOST_OFF)
    pos, vel = _particles(hr, tiles)
    d = member_gather_loss(_scaled(pos, ms, [0.9, 0.9]), vel, ms, cfg)[1]
    assert d["term_shell"] == 0.0
    assert d["median_shell_resid"] == pytest.approx(abs(float(np.log(0.9))),
                                                    rel=1e-3)
    # NaN, not 0, where there is no host: a plotted zero would read as
    # "the distribution is already right".
    ms2, _, _, _, _ = _sets(cfg)
    assert np.isnan(
        member_gather_loss(pos, vel, ms2, cfg)[1]["median_shell_resid"])


def test_the_two_terms_settle_at_half_the_correction_under_self():
    """The conflict priced, by minimising the actual loss rather than arguing.

    A radial step ``delta`` from the frozen centroid costs
    ``w_centre (delta/s)^2 + w_radius ((|dr| - delta)/s)^2``, minimised at
    ``delta* = |dr| w_radius / (w_centre + w_radius)`` -- half at equal weights.
    Under ``self_tan`` the anchor charges nothing radially and the minimum is the
    full correction. Both come out on the nose, which is why the submitter warns
    rather than leaving this to be discovered in a metrics file.
    """
    def cfg(mode):
        return _cfg(bg_k=0, centre_mode=mode, w_virial=0.0, w_bound=0.0,
                    w_d6=0.0, w_rrms=0.0, w_sigmav=0.0, w_centre=1.0,
                    w_radius=1.0, radius_dead_zone=0.0)

    ms, _, _, fz, tiles = _sets(cfg("self"), host_pos=HOST)
    pos, vel = _particles(fz, tiles)
    for i in range(ms.n_sets):
        rhat = ms.centre_rhat[i]
        # The signed step that would put this set exactly on its true shell.
        dr0 = member_gather_loss(pos, vel, ms, cfg("self"),
                                 set_indices=[i])[1]["median_radius_err_mpc_h"]
        full = -dr0
        for mode, want in (("self", 0.5), ("self_tan", 1.0)):
            grid = np.linspace(-0.5 * abs(full), 1.5 * abs(full), 201) * np.sign(full)
            losses = [float(member_gather_loss(pos + float(d) * rhat, vel, ms,
                                               cfg(mode), set_indices=[i])[0])
                      for d in grid]
            assert grid[int(np.argmin(losses))] / full == pytest.approx(
                want, abs=0.02), mode


# --------------------------------------------------------------------------- #
# The cap policy: which sets survive --max-sets
# --------------------------------------------------------------------------- #
# `largest` is right for a subhalo run and is a trap for a mass-window one: the
# pool of a cluster region IS a mass function, so keeping the top of it drops
# exactly the small halos a small-halo arm exists to build. These pin that the
# three policies differ in the way the docstring claims, and that all three
# return ascending rows -- a reordering alone would move every
# `--sets-per-step` sample and make two runs incomparable for a reason that has
# nothing to do with the selection.
def test_largest_cap_keeps_the_top_of_the_mass_function():
    from cosmo_sr.features.member_gather import _apply_cap

    rows = np.arange(100, 110)
    n_sites = np.arange(10, 20) ** 3          # strictly increasing
    got = _apply_cap(rows, n_sites, 3, "largest", seed=0)
    assert got.tolist() == [107, 108, 109]


def test_stratified_cap_spans_the_mass_range_the_largest_cap_truncates():
    from cosmo_sr.features.member_gather import _apply_cap

    # A realistic pool: 200 small sets and 5 huge ones, four log2 decades apart.
    n_sites = np.concatenate([np.full(200, 60), np.full(5, 40_000)])
    rows = np.arange(n_sites.size)
    largest = _apply_cap(rows, n_sites, 10, "largest", seed=0)
    strat = _apply_cap(rows, n_sites, 10, "stratified", seed=0)

    # `largest` spends half its budget on the five giants; `stratified` splits
    # the budget between the two occupied log2 bins, so the small population
    # survives the cap in both -- but the giants do not crowd it out.
    assert set(largest.tolist()) >= set(range(200, 205))
    assert sum(1 for r in strat if r >= 200) == 5
    assert sum(1 for r in strat if r < 200) == 5
    assert list(strat) == sorted(strat)


def test_random_cap_is_deterministic_in_the_seed_and_ascending():
    from cosmo_sr.features.member_gather import _apply_cap

    rows = np.arange(50)
    n_sites = np.full(50, 100)
    a = _apply_cap(rows, n_sites, 8, "random", seed=3)
    b = _apply_cap(rows, n_sites, 8, "random", seed=3)
    c = _apply_cap(rows, n_sites, 8, "random", seed=4)
    assert a.tolist() == b.tolist() == sorted(a.tolist())
    assert a.tolist() != c.tolist()
    assert len(set(a.tolist())) == 8          # no repeats


def test_config_refuses_an_empty_window_and_an_unknown_population():
    from cosmo_sr.features.member_gather import MemberGatherConfig

    with pytest.raises(ValueError, match="population"):
        MemberGatherConfig(population="field")
    with pytest.raises(ValueError, match="max_num_p"):
        MemberGatherConfig(min_num_p=200, max_num_p=200)
    with pytest.raises(ValueError, match="set_select"):
        MemberGatherConfig(set_select="smallest")
    # The window is a window, not a floor: 200 <= num_p < 2000 is legal.
    assert MemberGatherConfig(min_num_p=200, max_num_p=2000).max_num_p == 2000


# --------------------------------------------------------------------------- #
# The environment axis: WHICH surroundings the supervised sets come from
# --------------------------------------------------------------------------- #
# Why a third selection axis exists at all. `--population all` selected on SIZE
# and the set9 census measured what that bought: in the edited region the arm
# built 3,374 net objects and Rockstar filed every one of them as a subhalo
# (subs +4,260, top-level -886), overshooting HR in `field` (subhalos 3,316
# against HR's 2,466) while closing 249 of the 6,557-object `near` deficit --
# 3.8% of the bin that holds 56% of everything missing. Selecting by size cannot
# reach that, because a cluster region's pool is dominated by field halos and
# `stratified` stratifies on mass.
def _env_cat(pos, num_p, rvir, parent_ids=None):
    n = len(num_p)
    return SimpleNamespace(
        n=n, ids=np.arange(n), pos=np.asarray(pos, dtype=np.float64),
        num_p=np.asarray(num_p, dtype=np.int64),
        rvir=np.asarray(rvir, dtype=np.float64),
        parent_ids=(np.full(n, -1, dtype=np.int64) if parent_ids is None
                    else np.asarray(parent_ids, dtype=np.int64)))


def test_halo_environment_bins_by_distance_in_host_rvir():
    from cosmo_sr.features.subhalo_gather import ENV_LABELS, halo_environment

    # One 1 Mpc/h host at the origin, three halos at 0.5, 2 and 10 of its R_vir.
    cat = _env_cat(
        pos=[[50.0, 50.0, 50.0], [50.5, 50.0, 50.0],
             [52.0, 50.0, 50.0], [60.0, 50.0, 50.0]],
        num_p=[10000, 100, 100, 100], rvir=[1000.0, 50.0, 50.0, 50.0])
    env = [ENV_LABELS[e] for e in halo_environment(cat, host_floor=200,
                                                   boxsize_mpc_h=BOX)]
    # The host itself is measured against hosts BIGGER than it, of which there
    # are none, so it is `field` and the label is never self-referential.
    assert env == ["field", "inside", "near", "field"]


def test_halo_environment_is_periodic():
    from cosmo_sr.features.subhalo_gather import ENV_LABELS, halo_environment

    # The satellite is 0.5 Mpc/h from the host THROUGH the box face. A
    # non-periodic distance would put it 99.5 away and call it `field`.
    cat = _env_cat(pos=[[0.25, 50.0, 50.0], [99.75, 50.0, 50.0]],
                   num_p=[10000, 100], rvir=[1000.0, 50.0])
    env = [ENV_LABELS[e] for e in halo_environment(cat, host_floor=200,
                                                   boxsize_mpc_h=BOX)]
    assert env[1] == "inside"


def test_env_mask_any_is_the_identity_and_a_typo_raises():
    from cosmo_sr.features.subhalo_gather import env_mask

    e = np.array([0, 1, 2, 1])
    assert env_mask(e, "any").all()
    assert env_mask(e, "near").tolist() == [False, True, False, True]
    assert env_mask(e, "near,field").tolist() == [False, True, True, True]
    # A typo must not quietly select nothing: that would surface four steps
    # later as the unrelated "no supervised sets survived".
    with pytest.raises(ValueError, match="unknown environments"):
        env_mask(e, "nearby")


def test_env_select_filters_the_pool_and_records_the_mix():
    # The toy's two subhalos sit 10 and 30 Mpc/h from the host, whose R_vir is
    # 1 Mpc/h -- so both are `field` by the census's own bins. Asking for `near`
    # must therefore empty the pool, and asking for `field` must keep it whole.
    cat, oidx, tiles, hr, fz = _toy()
    from cosmo_sr.features.subhalo_gather import halo_environment
    env = halo_environment(cat, host_floor=100, boxsize_mpc_h=BOX)

    def build(sel):
        rep = {}
        ms = build_member_sets(cat, oidx, tiles, hr, fz, _cfg(env_select=sel),
                               particle_mass_msun_h=M_P, ng_hr=NG, tile_hr=TILE,
                               boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                               vel_scale_kms=VEL, env=env, report=rep)
        return ms, rep

    ms, rep = build("field")
    assert ms.n_sets == 2
    assert rep["n_by_env"] == {"inside": 0, "near": 0, "field": 2}
    assert rep["n_after_env"] == 2
    with pytest.raises(ValueError, match="no supervised sets survived"):
        build("near")

    # `any` records the same mix without cutting: an untargeted run's row is
    # what makes a targeted run's row readable as a change.
    ms, rep = build("any")
    assert ms.n_sets == 2 and rep["n_by_env"]["field"] == 2


def test_env_select_refuses_to_run_without_the_labels():
    # A selection silently not applied is the failure mode `--set-select
    # largest` under `--population all` already paid for once.
    cat, oidx, tiles, hr, fz = _toy()
    with pytest.raises(ValueError, match="no `env` array was passed"):
        build_member_sets(cat, oidx, tiles, hr, fz, _cfg(env_select="near"),
                          particle_mass_msun_h=M_P, ng_hr=NG, tile_hr=TILE,
                          boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                          vel_scale_kms=VEL, report={})


def test_the_cap_stratifies_over_mass_AND_environment():
    from cosmo_sr.features.member_gather import _apply_cap

    # 40 sets: 20 small field, 20 large near. Mass-only stratification has two
    # bins that happen to coincide with the environments here, so the test that
    # bites is the OTHER way round -- one mass bin spanning both environments.
    rows = np.arange(40)
    n_sites = np.full(40, 100)             # one mass bin for all of them
    env = np.array([2] * 20 + [1] * 20)    # 20 field, 20 near
    keep = _apply_cap(rows, n_sites, 10, "stratified", seed=0, env=env)
    got = env[keep]
    assert keep.size == 10
    assert int((got == 1).sum()) == 5 and int((got == 2).sum()) == 5
    # Without the env axis the single mass bin is drained in whatever order the
    # shuffle produced, and the balance is not guaranteed.
    plain = _apply_cap(rows, n_sites, 10, "stratified", seed=0)
    assert plain.size == 10
    # Ascending rows in both cases: the cap changes WHICH sets, never the order.
    assert np.all(np.diff(keep) > 0) and np.all(np.diff(plain) > 0)


# --------------------------------------------------------------------------- #
# The isolation term: the only INTER-set statement in the loss
# --------------------------------------------------------------------------- #
def _iso_setup(**cfgkw):
    """Sets with a background, plus the HR particle table as the candidate.

    Candidate == HR means every statistic sits exactly at its reference, so any
    non-zero isolation cost in what follows is caused by the perturbation under
    test and not by the toy's starting point.
    """
    kw = dict(bg_k=256, bg_radius_factor=4.0, iso_min_count=1.0,
              w_isolation=1.0, w_virial=0.0, w_bound=0.0, w_d6=0.0,
              w_rrms=0.0, w_sigmav=0.0, w_centre=0.0)
    kw.update(cfgkw)
    cfg = _cfg(**kw)
    # A TIGHT non-member cloud, unlike the default toy's: the isolation
    # count is a count, and at the default 1.2 Mpc/h scale the background
    # sphere holds ten particles and the soft count is half of one -- which
    # is precisely the regime `iso_min_count` exists to refuse to score.
    ms, rep, hr, fz, tiles = _sets(cfg, bg_scale=0.35)
    pos, vel = _particles(hr, tiles)
    return cfg, ms, pos, vel


def _squeeze_background(pos, ms, s, factor):
    """Move set ``s``'s background particles radially by ``factor`` about the
    set's own centroid. < 1 is a pedestal growing on the object; > 1 is a
    neighbourhood cleaner than HR's."""
    out = pos.clone()
    bg = ms.bg_rows[s]
    c = pos[ms.live_rows[s]].mean(dim=0, keepdim=True)
    out[bg] = c + factor * (pos[bg] - c)
    return out


def test_isolation_is_exactly_zero_when_the_candidate_is_hr():
    cfg, ms, pos, vel = _iso_setup()
    loss, diag = member_gather_loss(pos, vel, ms, cfg)
    assert diag["n_iso"] == ms.n_sets
    assert diag["term_isolation"] == pytest.approx(0.0, abs=1e-9)
    assert diag["median_iso_ratio"] == pytest.approx(1.0, rel=1e-6)
    assert float(loss) == pytest.approx(0.0, abs=1e-9)


def test_isolation_charges_a_pedestal_growing_on_the_object():
    # THE failure this term exists for. Every other term is a moment of the
    # MEMBERS about their own centroid, so pulling non-member mass onto the
    # object costs nothing anywhere else in the loss -- and that is exactly the
    # state that makes Rockstar file the object as somebody's subhalo, which is
    # what `--population all` measured itself doing 4,260 times.
    cfg, ms, pos, vel = _iso_setup()
    piled = _squeeze_background(pos, ms, 0, 0.35)
    _, diag = member_gather_loss(piled, vel, ms, cfg)
    assert diag["median_iso_ratio"] > 1.0
    assert diag["term_isolation"] > 0.0
    # The members did not move, so no other term noticed. That IS the finding.
    for k in ("virial", "rrms", "sigmav", "bound", "centre"):
        assert diag[f"term_{k}"] == pytest.approx(0.0, abs=1e-9)


def test_isolation_is_hinged_a_cleaner_neighbourhood_is_free():
    # The module's standing property: the loss may never ask for more than HR
    # has. A candidate whose surroundings are emptier than HR's is not charged.
    cfg, ms, pos, vel = _iso_setup()
    swept = _squeeze_background(pos, ms, 0, 2.5)
    _, diag = member_gather_loss(swept, vel, ms, cfg)
    assert diag["median_iso_ratio"] < 1.0
    assert diag["term_isolation"] == pytest.approx(0.0, abs=1e-9)


def test_isolation_gradient_reaches_the_background_and_not_the_members():
    # The centroid is DETACHED on purpose: the term must push the surrounding
    # mass out to where HR has it, not let the set run away from the pile. A
    # live centroid would permit the second, and would put this term in the same
    # tug-of-war with `centre_mode="self"` that `w_radius` lost (docs s.14).
    cfg, ms, pos, vel = _iso_setup()
    x = _squeeze_background(pos, ms, 0, 0.35).requires_grad_(True)
    loss, _ = member_gather_loss(x, vel, ms, cfg)
    loss.backward()
    g = x.grad
    assert float(g[ms.bg_rows[0]].abs().sum()) > 0.0
    assert float(g[ms.live_rows[0]].abs().sum()) == pytest.approx(0.0, abs=1e-12)


def test_isolation_skips_sets_whose_hr_background_is_shot_noise():
    # A ratio taken on three particles is the background sampler's seed, not a
    # measurement, and hinging on it would charge the run for that seed.
    cfg, ms, pos, vel = _iso_setup(iso_min_count=1e9)
    piled = _squeeze_background(pos, ms, 0, 0.2)
    _, diag = member_gather_loss(piled, vel, ms, cfg)
    assert diag["n_iso"] == 0
    assert diag["term_isolation"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(diag["median_iso_ratio"])


def test_the_isolation_ratio_is_measured_even_at_weight_zero():
    # The four velocity-guard arms failed on the one quantity none of them
    # logged. The count is a sum over <=bg_k particles against a step dominated
    # by O(N^2) pair blocks, so there is no reason to be blind here.
    cfg, ms, pos, vel = _iso_setup()
    off = dataclasses.replace(cfg, w_isolation=0.0)
    piled = _squeeze_background(pos, ms, 0, 0.35)
    _, diag = member_gather_loss(piled, vel, ms, off)
    assert diag["median_iso_ratio"] > 1.0
    assert diag["term_eff_isolation"] == pytest.approx(0.0, abs=1e-12)


def test_the_counting_radius_comes_from_hr_not_from_the_candidate():
    # If the radius tracked the candidate's own size the run could satisfy the
    # term by SHRINKING the sphere instead of clearing it -- which is the
    # gaming failure `docs/sr2_gather_finetune.md` s.6 records for the window
    # loss, in a new costume. The radius is a stored constant of the data.
    cfg, ms, pos, vel = _iso_setup()
    assert np.all(ms.iso_radius > 0.0)
    stored = ms.iso_radius.copy()
    collapsed = pos.clone()
    for s in range(ms.n_sets):
        c = pos[ms.live_rows[s]].mean(dim=0, keepdim=True)
        collapsed[ms.live_rows[s]] = c + 0.01 * (pos[ms.live_rows[s]] - c)
    member_gather_loss(collapsed, vel, ms, cfg)
    assert np.allclose(ms.iso_radius, stored)


# --------------------------------------------------------------------------- #
# The splits: WHICH population and WHICH environment moved
# --------------------------------------------------------------------------- #
def test_the_loss_splits_its_diagnostics_by_population_and_environment():
    # `MemberSets.parent_id` was added for exactly this and nothing read it, so
    # `all_blocks_smallhalo` spent 6.9 GPU-hours on a top-level hypothesis and
    # reported one number pooled over both populations. A pooled median cannot
    # tell "the objective is wrong" from "the objective got no budget", and
    # those need different next runs.
    cat, oidx, tiles, hr, fz = _toy()
    from cosmo_sr.features.subhalo_gather import halo_environment
    env = halo_environment(cat, host_floor=100, boxsize_mpc_h=BOX)
    rep = {}
    ms = build_member_sets(cat, oidx, tiles, hr, fz, _cfg(bg_k=128),
                           particle_mass_msun_h=M_P, ng_hr=NG, tile_hr=TILE,
                           boxsize_mpc_h=BOX, dis_scale_mpc_h=DIS,
                           vel_scale_kms=VEL, env=env, report=rep)
    pos, vel = _particles(fz, tiles)
    _, diag = member_gather_loss(pos, vel, ms, _cfg(bg_k=128))

    pop, by_env = diag["by_population"], diag["by_env"]
    # The toy's two sets are both satellites of catalog id 1, and both `field`.
    assert pop["sub"]["n"] == 2.0 and pop["top"]["n"] == 0.0
    assert by_env["field"]["n"] == 2.0
    assert by_env["near"]["n"] == 0.0 and by_env["inside"]["n"] == 0.0
    # Counts sum to the sets evaluated, in both splits: a set that fell out of
    # every bin would be a bookkeeping bug that a pooled number would hide.
    assert sum(v["n"] for v in pop.values()) == float(diag["n_sets"])
    assert sum(v["n"] for v in by_env.values()) == float(diag["n_sets"])
    # An empty split reports n = 0 with NaN medians rather than vanishing: a
    # missing key reads as a bug, a zero reads as a fact.
    assert np.isnan(pop["top"]["bound_hard"])
    assert pop["sub"]["bound_hard"] == pytest.approx(diag["median_bound_hard"])


def test_the_host_guard_keeps_every_environment_whatever_the_arm_selects():
    """A run that supervises only the infall region still damages the hosts it
    is not being asked to build. Narrowing the guard with the objective would
    remove the protection exactly where the new work is happening."""
    import sys as _sys
    from pathlib import Path as _P
    for _p in (_P(__file__).resolve().parents[2] / "scripts" / "features",
               _P(__file__).resolve().parents[2] / "scripts" / "reward"):
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))
    from finetune_member_gather import host_member_config

    mcfg = _cfg(env_select="near", w_radius=1.0, centre_mode="radial")
    args = SimpleNamespace(host_min_num_p=200, host_max_sets=256)
    h = host_member_config(mcfg, args)
    assert h.env_select == "any"
    assert h.population == "top" and h.max_num_p == 0
    assert h.centre_mode == "self" and h.w_radius == 0.0
    # The isolation weight IS inherited: for a host, "more non-member mass
    # inside my radius than the FROZEN field had" is the accretion that
    # re-parents it, which is the damage the guard exists to charge for.
    assert h.w_isolation == mcfg.w_isolation


# --------------------------------------------------------------------------- #
# `shell_quantiles`: ask only for the part of the distribution that transfers
# --------------------------------------------------------------------------- #
def test_shell_quantiles_is_off_by_default_and_rejects_negatives():
    """Every arm through 2026-09-08 ran the full rank-by-rank pairing."""
    assert MemberGatherConfig().shell_quantiles == 0
    with pytest.raises(ValueError, match="shell_quantiles"):
        _cfg(shell_quantiles=-1)


def test_shell_reduce_at_q_zero_is_the_sorted_log_pairing():
    """Q=0 must be the pre-2026-09-09 term exactly, or no arm is comparable."""
    cfg = _soft_cfg(shell_quantiles=0)
    rc = torch.tensor([3.0, 1.0, 2.0])
    rt = torch.tensor([2.0, 4.0, 1.0])
    lc, lt = member_gather.shell_reduce(rc, rt, cfg)
    assert torch.allclose(lc, torch.log(torch.tensor([1.0, 2.0, 3.0])))
    assert torch.allclose(lt, torch.log(torch.tensor([1.0, 2.0, 4.0])))


def test_shell_quantiles_bins_by_equal_count_and_means_in_log():
    """Six sorted radii into three bins is three pairs, averaged in LOG space.

    Equal-COUNT, not equal-radius: each bin must carry the same weight however
    the radii pile up, or a dense inner shell would outvote the outskirts.
    """
    cfg = _soft_cfg(shell_quantiles=3)
    r = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    lc, _lt = member_gather.shell_reduce(r, r, cfg)
    want = torch.tensor([np.log([1.0, 2.0]).mean(), np.log([3.0, 4.0]).mean(),
                         np.log([5.0, 6.0]).mean()], dtype=lc.dtype)
    assert lc.shape == (3,)
    assert torch.allclose(lc, want, atol=1e-6)


def test_shell_quantiles_at_or_above_n_is_the_full_pairing():
    """A small host must never be charged on emptier bins than it has sets."""
    r = torch.tensor([1.0, 2.0, 3.0])
    for q in (3, 4, 99):
        lc, _ = member_gather.shell_reduce(r, r, _soft_cfg(shell_quantiles=q))
        assert lc.shape == (3,), q


def test_shell_quantiles_hides_a_rank_wiggle_that_the_full_pairing_charges():
    """The whole point: an equal-and-opposite rank swap inside a bin is free.

    That swap is the per-host detail the 2026-09-08 sweep memorised -- train
    residual 7.3x against holdout 1.4x. Q must be blind to it while staying
    sensitive to a shift of the distribution itself.
    """
    cfg0, cfg2 = _soft_cfg(shell_quantiles=0), _soft_cfg(shell_quantiles=2)
    rt = torch.tensor([1.0, 2.0, 4.0, 8.0])
    # Same bin means in log, different rank-by-rank values: sqrt(2)*(1,2) has
    # the same log mean as (1,2)... build it explicitly.
    inner = torch.tensor([np.exp(np.log([1.0, 2.0]).mean() - 0.3),
                          np.exp(np.log([1.0, 2.0]).mean() + 0.3)])
    outer = torch.tensor([np.exp(np.log([4.0, 8.0]).mean() - 0.3),
                          np.exp(np.log([4.0, 8.0]).mean() + 0.3)])
    rc = torch.cat([inner, outer]).float()
    q2 = float(member_gather.shell_charge(
        *member_gather.shell_reduce(rc, rt, cfg2), cfg2))
    q0 = float(member_gather.shell_charge(
        *member_gather.shell_reduce(rc, rt, cfg0), cfg0))
    assert q2 == pytest.approx(0.0, abs=1e-10), "bin means match; Q=2 is free"
    assert q0 > 1e-3, "the full pairing still charges the within-bin wiggle"


def test_shell_quantiles_still_charges_a_shift_of_the_distribution():
    """Blind to the wiggle, not blind to the signal the term exists for."""
    cfg = _soft_cfg(shell_quantiles=2)
    rt = torch.tensor([1.0, 2.0, 4.0, 8.0])
    lc, lt = member_gather.shell_reduce(0.5 * rt, rt, cfg)
    assert float(member_gather.shell_charge(lc, lt, cfg)) > 0.4


def test_shell_quantiles_keeps_a_gradient_on_every_set_in_the_bin():
    """A bin mean must not park the gradient on one representative member."""
    cfg = _soft_cfg(shell_quantiles=2)
    rc = torch.tensor([1.0, 2.0, 4.0, 8.0], requires_grad=True)
    rt = torch.tensor([2.0, 3.0, 6.0, 9.0])
    member_gather.shell_charge(*member_gather.shell_reduce(rc, rt, cfg),
                               cfg).backward()
    assert torch.all(rc.grad != 0.0)


def test_the_soft_shell_at_q_zero_is_unchanged_end_to_end():
    """The full loss at Q=0 must equal the default config's, on real sets."""
    ms, _, hr, _, tiles = _sets(_soft_cfg(), host_pos=HOST_OFF)
    pos, vel = _particles(hr, tiles)
    moved = _scaled(pos, ms, [1.4, 0.7])
    a = member_gather_loss(moved, vel, ms, _soft_cfg())[1]["term_shell"]
    b = member_gather_loss(moved, vel, ms,
                           _soft_cfg(shell_quantiles=0))[1]["term_shell"]
    assert a == pytest.approx(b, rel=1e-12)


def test_shell_radii_matches_the_in_loop_radii():
    """The cheap path must measure the same thing as the loss's own shell.

    `shell_radii` exists so a noise-averaged shell can add a draw for the price
    of a forward pass instead of the O(N^2) statistics. If it disagreed with the
    in-loop radii the average would be of two different quantities.
    """
    cfg = _soft_cfg(max_set_particles=0)
    ms, _, hr, _, tiles = _sets(cfg, host_pos=HOST_OFF)
    pos, vel = _particles(hr, tiles)
    moved = _scaled(pos, ms, [1.4, 0.7])
    rc, rt = member_gather.shell_radii(moved, ms, cfg)
    lc, lt = member_gather.shell_reduce(rc, rt, cfg)
    charged = float(member_gather.shell_charge(lc, lt, cfg))
    assert charged == pytest.approx(
        member_gather_loss(moved, vel, ms, cfg)[1]["term_shell"], rel=1e-6)


# --------------------------------------------------------------------------- #
# The noise-averaged shell: mean-then-charge, not charge-then-mean
# --------------------------------------------------------------------------- #
def test_averaging_identical_draws_leaves_the_charge_unchanged():
    """The swap `gather - charge(primary) + charge(mean)` must be a no-op when
    the draws agree, or every arm's step-0 value would shift for no reason."""
    cfg = _soft_cfg()
    rc = torch.tensor([1.0, 2.0, 5.0]), torch.tensor([1.0, 2.0, 5.0])
    rt = torch.tensor([1.5, 2.5, 4.0])
    curves = [member_gather.shell_reduce(r, rt, cfg)[0] for r in rc]
    lt = member_gather.shell_reduce(rc[0], rt, cfg)[1]
    mean_c = float(member_gather.shell_charge(
        torch.stack(curves).mean(dim=0), lt, cfg))
    assert mean_c == pytest.approx(
        float(member_gather.shell_charge(curves[0], lt, cfg)), rel=1e-12)


def test_mean_then_charge_is_not_charge_then_mean():
    """The whole reason the averaging is done on the CURVE. Two draws that
    straddle the target average to it exactly; charging each draw first and
    averaging the costs keeps both penalties and asks for a distribution every
    single realisation must hit -- which is the per-realisation objective the
    2026-09-08 arm was already gaming."""
    cfg = _soft_cfg(shell_dead_zone=0.0)
    rt = torch.tensor([2.0, 4.0])
    lo, hi = 0.5 * rt, 2.0 * rt              # straddle in log space
    lt = member_gather.shell_reduce(rt, rt, cfg)[1]
    curves = [member_gather.shell_reduce(r, rt, cfg)[0] for r in (lo, hi)]
    mean_then_charge = float(member_gather.shell_charge(
        torch.stack(curves).mean(dim=0), lt, cfg))
    charge_then_mean = float(np.mean([
        float(member_gather.shell_charge(c, lt, cfg)) for c in curves]))
    assert mean_then_charge == pytest.approx(0.0, abs=1e-10)
    assert charge_then_mean > 0.4


# --------------------------------------------------------------------------- #
# The reference de-bias (--stats-members)
# --------------------------------------------------------------------------- #
#
# `reference-inflated-by-stragglers`, MEASURED on the 16 holdout hosts: at a
# median live fraction of 0.979 the hybrid reference sits at 2.18x HR's own
# `r_rms` and 1.99x its `2T/|W|`, with corr(1 - live_frac, inflation) = 0.76
# held out and 0.84 on the 40 training hosts. `r_rms` is an RMS, so a ~2% tail
# at ~2 Mpc/h carries ~78% of `r_rms^2`.
#
# The toy reproduces the SHAPE of that (10 of 100 members parked 2 Mpc/h away
# from a 0.15 Mpc/h object), not the magnitude -- the point of these tests is
# that the policy is applied to both sides of every comparison at once, which
# is the property a half-applied fix would break silently.


def _straggler_box(offset_mpc_h: float = 2.0, vel_kms: float = 400.0):
    """A 16^3 frozen box whose halo-12 out-of-tile members sit far from it.

    `build_member_sets` reads the stragglers through
    `bound_discriminator.particles_at`, which wants the whole box rather than
    the trained tiles, so the toy's `split=True` neighbour sites are given
    displacements that put them `offset_mpc_h` from halo 12's centre. Every
    other site is left at its lattice point and is never read.

    They also get a bulk `vel_kms`, and that is not decoration: a straggler is
    a particle the tiling cut away, not a member at rest, so it does not share
    the subhalo's frame. Leaving the velocity channels at zero makes the tail
    inflate `r_rms` while *lowering* `2T/|W|`, which is the opposite of the
    measured defect and would let a broken policy pass.
    """
    cell = BOX / NG
    box = np.zeros((6, NG, NG, NG), dtype=np.float64)
    nb = (np.arange(10) // (TILE * TILE)) * NG * NG \
        + ((np.arange(10) // TILE) % TILE) * NG + (np.arange(10) % TILE) + TILE
    iz, iy, ix = nb % NG, (nb // NG) % NG, nb // (NG * NG)
    q = np.stack([(ix + 0.5) * cell, (iy + 0.5) * cell, (iz + 0.5) * cell])
    want = np.array([15.0, 15.0, 15.0])[:, None] + np.array(
        [[offset_mpc_h], [0.0], [0.0]])
    box[0:3, ix, iy, iz] = (want - q) / DIS
    box[3, ix, iy, iz] = vel_kms / VEL
    return box


def _split_sets(stats_members: str, offset_mpc_h: float = 2.0):
    cat, oidx, tiles, hr, fz = _toy(split=True)
    rep = {}
    ms = build_member_sets(
        cat, oidx, tiles, hr, fz,
        _cfg(min_live_frac=0.5, stats_members=stats_members),
        particle_mass_msun_h=M_P, ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX,
        dis_scale_mpc_h=DIS, vel_scale_kms=VEL,
        frozen_box=_straggler_box(offset_mpc_h), report=rep)
    s = int(np.flatnonzero(ms.halo_id == 12)[0])
    return ms, rep, s, tiles, fz


def test_stats_members_defaults_to_the_hybrid_every_arm_trained_on():
    assert MemberGatherConfig().stats_members == "all"
    with pytest.raises(ValueError, match="stats_members"):
        _cfg(stats_members="core")


def test_the_straggler_tail_inflates_the_reference_r_rms():
    """The defect, reproduced: 10% of members 2 Mpc/h out, and the reference
    the loss optimises against is several times HR's own object.

    Read PER SET, not off the medians: only halo 12 straddles a tile face, so
    a median over the three sets is 2/3 made of rows the policy cannot move
    and would report the defect as absent.
    """
    a_ms, rep, s, _, _ = _split_sets("all")
    l_ms, _, sl, _, _ = _split_sets("live")
    assert rep["outside_dropped"] == 0            # a frozen box WAS supplied
    assert a_ms.fixed_pos[s] is not None and a_ms.fixed_pos[s].shape[0] == 10
    assert float(a_ms.ref["r_rms"][s]) > 2.0 * float(l_ms.ref["r_rms"][sl])
    assert float(a_ms.ref["virial"][s]) > 2.0 * float(l_ms.ref["virial"][sl])


def test_live_policy_makes_the_reference_hrs_own_object():
    ms, rep, s, _, _ = _split_sets("live")
    assert ms.fixed_pos[s] is None and ms.fixed_vel[s] is None
    assert rep["outside_excluded_by_policy"] == 10
    assert rep["outside_dropped"] == 0
    for k in ("r_rms", "sigma_v", "virial", "bound_hard"):
        assert rep["reference_median"][k] == pytest.approx(
            rep["pure_hr_median"][k], rel=1e-9)


def test_the_inflation_stays_visible_under_the_live_policy():
    """A run that cannot see its own reference move is how this went unnoticed
    for 56 arms, so `hybrid_reference_median` is reported under BOTH policies
    and always means the same thing."""
    all_rep = _split_sets("all")[1]
    live_rep = _split_sets("live")[1]
    assert all_rep["stats_members"] == "all"
    assert live_rep["stats_members"] == "live"
    for k in ("r_rms", "virial"):
        assert live_rep["hybrid_reference_median"][k] == pytest.approx(
            all_rep["reference_median"][k], rel=1e-9)
        assert live_rep["pure_hr_median"][k] == pytest.approx(
            all_rep["pure_hr_median"][k], rel=1e-9)


def test_the_policy_tightens_what_the_loss_asks_for():
    """The point of the change. Same field, same members moved, smaller target:
    the rrms term at step 0 must be LARGER under the live policy, because the
    candidate is being compared to HR's real compactness instead of to a
    reference the stragglers had already loosened."""
    def rrms_term(policy):
        ms, _, _, tiles, fz = _split_sets(policy)
        cfg = _cfg(min_live_frac=0.5, stats_members=policy,
                   w_virial=0.0, w_bound=0.0, w_d6=0.0, w_sigmav=0.0,
                   w_centre=0.0, w_rrms=1.0)
        pos, vel = _particles(fz, tiles)
        return member_gather_loss(pos, vel, ms, cfg)[1]["term_rrms"]
    assert rrms_term("live") > rrms_term("all")


def test_centre_self_is_still_exactly_zero_at_step_zero():
    """The invariant a half-applied policy would break. `centre_mode="self"`
    anchors the set where the frozen field already put it, so the term must be
    zero before the first step under EITHER policy -- which is only true if
    `centre_self` and what `_gather_one` builds see the same collection."""
    for policy in ("all", "live"):
        ms, _, _, tiles, fz = _split_sets(policy)
        cfg = _cfg(min_live_frac=0.5, stats_members=policy,
                   w_virial=0.0, w_bound=0.0, w_d6=0.0, w_rrms=0.0,
                   w_sigmav=0.0, w_centre=1.0, centre_mode="self")
        pos, vel = _particles(fz, tiles)
        assert member_gather_loss(pos, vel, ms, cfg)[1]["term_centre"] \
            == pytest.approx(0.0, abs=1e-9)


def test_a_set_with_no_stragglers_is_untouched_by_the_policy():
    """Halos 10 and 11 are wholly inside the trained tile, so the flag must be
    a no-op for them -- otherwise the arms this is compared against move for a
    reason that has nothing to do with the reference."""
    a_ms, _, _, _, _ = _split_sets("all")
    l_ms, _, _, _, _ = _split_sets("live")
    for hid in (10, 11):
        ia = int(np.flatnonzero(a_ms.halo_id == hid)[0])
        il = int(np.flatnonzero(l_ms.halo_id == hid)[0])
        assert a_ms.fixed_pos[ia] is None and l_ms.fixed_pos[il] is None
        for k in ("r_rms", "sigma_v", "virial"):
            assert float(a_ms.ref[k][ia]) == pytest.approx(
                float(l_ms.ref[k][il]), rel=1e-12)
