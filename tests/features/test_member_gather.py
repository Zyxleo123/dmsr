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
         split: bool = False):
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
        rng.normal(scale=1.2, size=(half, 3)) + centres[10],
        rng.normal(scale=1.2, size=(rest.size - half, 3)) + centres[11]])
    bv = rng.normal(scale=900.0, size=(rest.size, 3))
    for f in (hr, fz):
        f[0, 0:3, gx[rest], gy[rest], gz[rest]] = (bx - q[rest]) / DIS
        f[0, 3:6, gx[rest], gy[rest], gz[rest]] = bv / VEL

    ids = [1, 10, 11] + ([12] if split else [])
    cat = SimpleNamespace(
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
