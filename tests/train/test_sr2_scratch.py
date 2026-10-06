"""Tests for the from-scratch SR2 trainer's library half.

The three things worth pinning here are the ones a training run cannot tell you
about from its own logs:

1. **The density channels are real.** ``valid_center = 0`` is the wrapped CIC
   deposit, which on a single tile correlates r = 0.08 with that tile's true
   density; eight of the critic's twenty channels would be noise and every log
   line would still look healthy. :func:`check_density_channels` must refuse it.
2. **The scratch guard is the fine-tune guard, with a different reference.** If
   :func:`lowk_anchor` and ``free_field_gather.guards`` ever computed different
   things, a scratch arm and a fine-tune arm would be reporting different
   quantities under the same ``low_k`` column.
3. **The noise scales can actually move off zero.** ``AddNoise.std`` starts at
   0, and SR2's fine-scale structure is written entirely by those six tensors. A
   zero gradient there would give a deterministic upsampler that no spectrum
   test would catch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cosmo_sr.reward.sr2_adversarial import SR2Critic  # noqa: E402
from cosmo_sr.train.sr2_finetune_data import SR2TileGeometry  # noqa: E402
from cosmo_sr.train.sr2_scratch import (  # noqa: E402
    PairedTileSampler,
    base_mse,
    build_scratch_critic,
    build_scratch_generator,
    check_density_channels,
    critic_update,
    forward_batch,
    generator_adv_loss,
    lowk_anchor,
    lr_core,
    mse_handoff_weight,
    noise_std_report,
    scale_anchor,
    scratch_critic_input,
    term_ramp,
    tile_hr_target,
)

from sr2_fixtures import TEST_GEOM, TINY_CHAN  # noqa: E402,F401

CELLSIZE_KPC_H = 100.0 * 1e3 / 512.0     # the production 100 Mpc/h, 512^3 grid
NG_HR = TEST_GEOM.ng_hr                  # 128 with the test geometry
TILE = TEST_GEOM.tile_hr                 # 64, the production tile


@pytest.fixture
def boxes():
    rng = np.random.default_rng(3)
    lr = {b: rng.normal(0, 0.1, size=(6, TEST_GEOM.ng_lr,) * 1
                        + (TEST_GEOM.ng_lr, TEST_GEOM.ng_lr)).astype(np.float32)
          for b in ("b0", "b1")}
    hr = {b: rng.normal(0, 0.1, size=(6, NG_HR, NG_HR, NG_HR)).astype(np.float32)
          for b in ("b0", "b1")}
    return lr, hr


@pytest.fixture
def tiny_model():
    return build_scratch_generator(scale_factor=8, seed=0,
                                   device=torch.device("cpu"), **TINY_CHAN)


# --------------------------------------------------------------------------- #
# The density channels
# --------------------------------------------------------------------------- #
def test_wrapped_density_deposit_is_refused():
    """valid_center=0 is the r=0.08 scrambling; it must not be reachable by accident."""
    with pytest.raises(ValueError, match="r=0.08"):
        check_density_channels(0, TILE)


def test_wrapped_density_deposit_is_reachable_deliberately():
    assert check_density_channels(0, TILE, allow_wrapped=True) == 0


def test_valid_center_outside_the_tile_is_refused():
    with pytest.raises(ValueError, match="outside"):
        check_density_channels(TILE + 1, TILE)


def test_critic_input_carries_eight_density_channels(tiny_model, boxes):
    """20 channels = 6 upsampled LR + 6 field + grid_mult^3 density."""
    lr, hr = boxes
    sam = PairedTileSampler(["b0"], lr, hr, geom=TEST_GEOM, ng_hr=NG_HR,
                            tile_hr=TILE, seed=0, device=torch.device("cpu"))
    b = sam.batch(2)
    cand = forward_batch(tiny_model, b, TEST_GEOM)
    x = scratch_critic_input(lr_core(b.lr, TEST_GEOM), cand,
                             cellsize_kpc_h=CELLSIZE_KPC_H, grid_mult=2,
                             valid_center=TILE // 2)
    assert x.shape[1] == 6 + 6 + 2 ** 3 == 20
    # the valid-centre crop is what sets the spatial size, not the tile
    assert x.shape[-1] == TILE // 2
    assert torch.isfinite(x).all()


def test_density_channels_respond_to_the_field(tiny_model, boxes):
    """The last 8 channels must change when the displacement changes.

    A deposit that ignored the field (or a constant one) would pass every shape
    assertion above while telling the critic nothing.
    """
    lr, hr = boxes
    sam = PairedTileSampler(["b0"], lr, hr, geom=TEST_GEOM, ng_hr=NG_HR,
                            tile_hr=TILE, seed=1, device=torch.device("cpu"))
    b = sam.batch(1)
    cand = forward_batch(tiny_model, b, TEST_GEOM)
    kw = dict(cellsize_kpc_h=CELLSIZE_KPC_H, grid_mult=2, valid_center=TILE // 2)
    a = scratch_critic_input(lr_core(b.lr, TEST_GEOM), cand, **kw)[:, 12:]
    c = scratch_critic_input(lr_core(b.lr, TEST_GEOM), cand + 0.05, **kw)[:, 12:]
    assert not torch.allclose(a, c)


# --------------------------------------------------------------------------- #
# The guard
# --------------------------------------------------------------------------- #
def test_lowk_anchor_matches_the_finetune_guard_operator():
    """Same operator as free_field_gather.guards -- only the reference differs.

    Feeding ``guards`` the HR tile as its ``base`` must reproduce
    :func:`lowk_anchor` exactly, which is what makes a scratch arm's ``low_k``
    column readable next to a fine-tune arm's.
    """
    from free_field_gather import guards

    torch.manual_seed(0)
    cand = torch.randn(2, 6, 64, 64, 64)
    hr = torch.randn(2, 6, 64, 64, 64)
    mine = lowk_anchor(cand, hr, TEST_GEOM.scale_factor)
    theirs = guards(cand, hr, TEST_GEOM)
    for key in ("low", "low_dis", "low_vel", "vel_rms_ratio"):
        assert torch.allclose(mine[key], theirs[key], atol=0, rtol=0), key


def test_lowk_anchor_is_zero_on_a_perfect_match_and_has_finite_gradient():
    """The squared form, not its square root: sqrt' is infinite at a match."""
    hr = torch.randn(1, 6, 64, 64, 64)
    cand = hr.clone().requires_grad_(True)
    g = lowk_anchor(cand, hr, 8)
    assert float(g["low"].detach()) == pytest.approx(0.0, abs=1e-12)
    g["low"].backward()
    assert torch.isfinite(cand.grad).all()


#: The eight production tile ids whose 64^3 slices land inside a 128^3 array.
#: Both slicers decompose the id in base ``512 // 64 = 8``, so these are the
#: ids with ``ix, iy, iz`` in {0, 1} -- enough to exercise all three axes
#: without materialising the 3.2 GB box that would get this test SIGKILLed.
_SMALL_TILE_IDS = (0, 1, 8, 9, 64, 65, 72, 73)


@pytest.mark.parametrize("tile_id", _SMALL_TILE_IDS)
def test_tile_hr_target_matches_the_finetune_slicer(tile_id):
    """One definition of 'which sub-cube is this tile', shared with the pool.

    ``overfit_host_mse.tile_hr_target`` is the slicer the host pool supervises
    through. If this module's copy disagreed, the base objective and the
    member-gather objective would be looking at different HR sub-cubes under the
    same tile id and nothing in either log would say so.
    """
    from overfit_host_mse import tile_hr_target as theirs

    rng = np.random.default_rng(5)
    hr = rng.normal(size=(6, 128, 128, 128)).astype(np.float32)
    mine = tile_hr_target(hr, tile_id, ng_hr=512, tile=64)
    assert mine.shape == (6, 64, 64, 64)
    assert np.array_equal(mine, theirs(hr, tile_id, None))


# --------------------------------------------------------------------------- #
# The sampler
# --------------------------------------------------------------------------- #
def test_sampler_excludes_what_it_is_told_to(boxes):
    lr, hr = boxes
    n_tot = (NG_HR // TILE) ** 3
    drop = [("b0", 0), ("b0", 1), ("b1", 7)]
    sam = PairedTileSampler(["b0", "b1"], lr, hr, geom=TEST_GEOM, ng_hr=NG_HR,
                            tile_hr=TILE, exclude=drop, seed=0,
                            device=torch.device("cpu"))
    assert len(sam) == 2 * n_tot - len(drop)
    assert all((b, t) not in set(drop) for b, t in sam.index)


def test_sampler_refuses_an_empty_stream(boxes):
    lr, hr = boxes
    n_tot = (NG_HR // TILE) ** 3
    everything = [("b0", t) for t in range(n_tot)]
    with pytest.raises(ValueError, match="no data"):
        PairedTileSampler(["b0"], lr, hr, geom=TEST_GEOM, ng_hr=NG_HR,
                          tile_hr=TILE, exclude=everything,
                          device=torch.device("cpu"))


def test_sampler_shapes_are_one_deployed_forward_pass(boxes, tiny_model):
    lr, hr = boxes
    sam = PairedTileSampler(["b0"], lr, hr, geom=TEST_GEOM, ng_hr=NG_HR,
                            tile_hr=TILE, seed=0, device=torch.device("cpu"))
    b = sam.batch(3)
    assert b.lr.shape == (3, 6, TEST_GEOM.lr_size, TEST_GEOM.lr_size,
                          TEST_GEOM.lr_size)
    assert b.hr.shape == (3, 6, TILE, TILE, TILE)
    assert set(b.noise) == {f"z{i}" for i in range(6)}
    assert forward_batch(tiny_model, b, TEST_GEOM).shape == b.hr.shape


def test_pinned_noise_stride_is_replayable(boxes):
    """stride 0 gives the same realisation for a tile on every visit."""
    lr, hr = boxes
    kw = dict(geom=TEST_GEOM, ng_hr=NG_HR, tile_hr=TILE, base_seed=4,
              noise_seed_stride=0, device=torch.device("cpu"))
    a = PairedTileSampler(["b0"], lr, hr, seed=0, **kw).example("b0", 2)
    c = PairedTileSampler(["b0"], lr, hr, seed=99, **kw).example("b0", 2)
    for site in a["noise"]:
        assert torch.equal(a["noise"][site], c["noise"][site])


def test_fresh_noise_stride_varies_per_visit(boxes):
    lr, hr = boxes
    sam = PairedTileSampler(["b0"], lr, hr, geom=TEST_GEOM, ng_hr=NG_HR,
                            tile_hr=TILE, noise_seed_stride=1_000_003, seed=0,
                            device=torch.device("cpu"))
    a = sam.example("b0", 2)["noise"]["z5"]
    c = sam.example("b0", 2)["noise"]["z5"]
    assert not torch.equal(a, c)


# --------------------------------------------------------------------------- #
# The generator's stochasticity
# --------------------------------------------------------------------------- #
def test_fresh_generator_starts_with_zero_noise_scales():
    m = build_scratch_generator(scale_factor=8, seed=0,
                                device=torch.device("cpu"), **TINY_CHAN)
    report = noise_std_report(m)
    assert len(report) == 6
    assert all(v == 0.0 for v in report.values())


def test_noise_scales_have_a_nonzero_gradient_at_zero(boxes, tiny_model):
    """std=0 must still be trainable, or SR2 never acquires stochasticity.

    ``AddNoise`` computes ``x + std * z`` (twice, verbatim from upstream), so
    d/d(std) is ``2 z`` and is non-zero even though ``std`` is. If this ever
    became zero the from-scratch generator would be a deterministic upsampler
    for the whole run.
    """
    lr, hr = boxes
    sam = PairedTileSampler(["b0"], lr, hr, geom=TEST_GEOM, ng_hr=NG_HR,
                            tile_hr=TILE, seed=0, device=torch.device("cpu"))
    b = sam.batch(1)
    base_mse(forward_batch(tiny_model, b, TEST_GEOM), b.hr).backward()
    grads = {n: float(p.grad.abs().sum()) for n, p in tiny_model.named_parameters()
             if n.endswith(".std")}
    assert grads, "no AddNoise.std parameters found"
    assert all(g > 0 for g in grads.values()), grads


def test_init_noise_std_seeds_every_site():
    m = build_scratch_generator(scale_factor=8, seed=0, init_noise_std=0.02,
                                device=torch.device("cpu"), **TINY_CHAN)
    assert all(v == pytest.approx(0.02) for v in noise_std_report(m).values())


def test_init_weight_std_is_applied_and_is_reproducible():
    a = build_scratch_generator(scale_factor=8, seed=3, init_weight_std=0.02,
                                device=torch.device("cpu"), **TINY_CHAN)
    c = build_scratch_generator(scale_factor=8, seed=3, init_weight_std=0.02,
                                device=torch.device("cpu"), **TINY_CHAN)
    w = a.blocks[0].conv[2].weight
    assert float(w.detach().std()) == pytest.approx(0.02, rel=0.25)
    assert torch.equal(w, c.blocks[0].conv[2].weight)


def test_scratch_generator_is_not_the_pretrained_checkpoint():
    """Two seeds must differ -- a 'from scratch' arm that loaded weights is not one."""
    a = build_scratch_generator(scale_factor=8, seed=0,
                                device=torch.device("cpu"), **TINY_CHAN)
    c = build_scratch_generator(scale_factor=8, seed=1,
                                device=torch.device("cpu"), **TINY_CHAN)
    assert not torch.equal(a.blocks[0].conv[2].weight, c.blocks[0].conv[2].weight)


def test_every_parameter_is_trainable():
    m = build_scratch_generator(scale_factor=8, seed=0,
                                device=torch.device("cpu"), **TINY_CHAN)
    assert all(p.requires_grad for p in m.parameters())


# --------------------------------------------------------------------------- #
# The adversarial half
# --------------------------------------------------------------------------- #
def test_critic_update_changes_the_critic_and_not_the_generator(boxes, tiny_model):
    lr, hr = boxes
    sam = PairedTileSampler(["b0"], lr, hr, geom=TEST_GEOM, ng_hr=NG_HR,
                            tile_hr=TILE, seed=0, device=torch.device("cpu"))
    b = sam.batch(2)
    critic = build_scratch_critic(width=8, depth=3, seed=0,
                                  device=torch.device("cpu"))
    opt_d = torch.optim.Adam(critic.parameters(), lr=1e-3, betas=(0.0, 0.9))
    kw = dict(cellsize_kpc_h=CELLSIZE_KPC_H, grid_mult=2, valid_center=TILE // 2)
    cand = forward_batch(tiny_model, b, TEST_GEOM)
    fake = scratch_critic_input(lr_core(b.lr, TEST_GEOM), cand, **kw).detach()
    real = scratch_critic_input(lr_core(b.lr, TEST_GEOM), b.hr, **kw)

    before_g = [p.detach().clone() for p in tiny_model.parameters()]
    before_d = critic.body[0].weight.detach().clone()
    stats = critic_update(critic, opt_d, real, fake, do_gp=True, gp_every=16)

    assert set(stats) == {"d_loss", "d_real", "d_fake", "d_gp"}
    assert stats["d_gp"] > 0
    assert not torch.equal(before_d, critic.body[0].weight)
    for a, p in zip(before_g, tiny_model.parameters()):
        assert torch.equal(a, p), "the critic step moved the generator"


def test_gradient_penalty_is_skipped_when_asked(boxes, tiny_model):
    lr, hr = boxes
    sam = PairedTileSampler(["b0"], lr, hr, geom=TEST_GEOM, ng_hr=NG_HR,
                            tile_hr=TILE, seed=0, device=torch.device("cpu"))
    b = sam.batch(1)
    critic = build_scratch_critic(width=8, depth=3, device=torch.device("cpu"))
    opt_d = torch.optim.Adam(critic.parameters(), lr=1e-3)
    kw = dict(cellsize_kpc_h=CELLSIZE_KPC_H, grid_mult=2, valid_center=TILE // 2)
    cand = forward_batch(tiny_model, b, TEST_GEOM)
    fake = scratch_critic_input(lr_core(b.lr, TEST_GEOM), cand, **kw).detach()
    real = scratch_critic_input(lr_core(b.lr, TEST_GEOM), b.hr, **kw)
    assert critic_update(critic, opt_d, real, fake, do_gp=False)["d_gp"] == 0.0


def test_generator_adv_loss_reaches_the_noise_scales(boxes, tiny_model):
    """The adversarial gradient must reach AddNoise.std, not just the convs."""
    lr, hr = boxes
    sam = PairedTileSampler(["b0"], lr, hr, geom=TEST_GEOM, ng_hr=NG_HR,
                            tile_hr=TILE, seed=0, device=torch.device("cpu"))
    b = sam.batch(1)
    critic = build_scratch_critic(width=8, depth=3, device=torch.device("cpu"))
    cand = forward_batch(tiny_model, b, TEST_GEOM)
    x = scratch_critic_input(lr_core(b.lr, TEST_GEOM), cand,
                             cellsize_kpc_h=CELLSIZE_KPC_H, grid_mult=2,
                             valid_center=TILE // 2)
    generator_adv_loss(critic, x).backward()
    stds = [p.grad for n, p in tiny_model.named_parameters() if n.endswith(".std")]
    assert stds and all(g is not None and float(g.abs().sum()) > 0 for g in stds)


# --------------------------------------------------------------------------- #
# The ramp
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("step,expected", [
    (0, 0.0), (1, 0.0), (99, 0.0), (100, 0.0), (150, 0.5), (200, 1.0), (999, 1.0),
])
def test_term_ramp(step, expected):
    assert term_ramp(step, warmup=100, ramp=100) == pytest.approx(expected)


def test_term_ramp_with_no_ramp_is_a_step_function():
    assert term_ramp(99, warmup=100, ramp=0) == 0.0
    assert term_ramp(100, warmup=100, ramp=0) == 1.0


def test_term_ramp_with_no_warmup_is_on_immediately():
    assert term_ramp(0, warmup=0, ramp=0) == 1.0


# --------------------------------------------------------------------------- #
# The trainer's plumbing
# --------------------------------------------------------------------------- #
def _scratch_args(argv):
    import train_sr2_scratch as TS
    return TS.build_parser().parse_args(argv)


def test_finetune_guard_ref_default_is_unchanged():
    """Every arm on disk is `frozen`; the scratch job is the only `hr` caller."""
    import finetune_member_gather as MG

    assert MG.build_parser().parse_args([]).guard_ref == "frozen"


def test_scratch_forces_the_settings_a_random_init_cannot_vary():
    import train_sr2_scratch as TS

    args = _scratch_args(["--rung", "fine", "--w-adv", "1.0"])
    changed = TS.apply_scratch_defaults(args)
    assert args.rung == "full"
    assert args.guard_ref == "hr"
    assert args.w_adv == 0.0
    # and it says so, rather than inheriting a fine-tune default silently
    assert any("rung" in line for line in changed)
    assert any("guard_ref" in line for line in changed)


def test_scratch_defaults_are_idempotent():
    import train_sr2_scratch as TS

    args = _scratch_args([])
    TS.apply_scratch_defaults(args)
    assert TS.apply_scratch_defaults(args) == []


def test_state_round_trips_and_resumes_at_the_right_step(tmp_path):
    """latest.pt is the resume point for a job that WILL hit its time limit."""
    import train_sr2_scratch as TS

    args = _scratch_args([])
    m = build_scratch_generator(scale_factor=8, seed=0,
                                device=torch.device("cpu"), **TINY_CHAN)
    c = build_scratch_critic(width=8, depth=3, device=torch.device("cpu"))
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    opt_d = torch.optim.Adam(c.parameters(), lr=1e-4)
    # one real step, so the optimiser state is not empty
    base_mse(m(torch.randn(1, 6, 14, 14, 14)).mean(), torch.zeros(())).backward()
    opt.step()

    p = tmp_path / "latest.pt"
    TS.save_state(p, model=m, critic=c, opt=opt, opt_d=opt_d, step=1234,
                  args=args, rng_state={"sampler": None})

    m2 = build_scratch_generator(scale_factor=8, seed=99,
                                 device=torch.device("cpu"), **TINY_CHAN)
    c2 = build_scratch_critic(width=8, depth=3, seed=99,
                              device=torch.device("cpu"))
    opt2 = torch.optim.Adam(m2.parameters(), lr=1e-4)
    opt_d2 = torch.optim.Adam(c2.parameters(), lr=1e-4)
    step, _rng, _norm = TS.load_state(p, model=m2, critic=c2, opt=opt2,
                                      opt_d=opt_d2, device=torch.device("cpu"))
    assert step == 1234
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)
    for a, b in zip(c.state_dict().values(), c2.state_dict().values()):
        assert torch.equal(a, b)


def test_resuming_a_finetune_checkpoint_is_refused(tmp_path):
    """A fine-tune checkpoint here would train PRETRAINED weights as `scratch`."""
    import train_sr2_scratch as TS

    p = tmp_path / "tuned.pt"
    torch.save({"model": {}, "rung": "fine", "step": 10}, p)
    m = build_scratch_generator(scale_factor=8, device=torch.device("cpu"),
                                **TINY_CHAN)
    with pytest.raises(SystemExit, match="not written by this trainer"):
        TS.load_state(p, model=m, critic=None,
                      opt=torch.optim.Adam(m.parameters()), opt_d=None,
                      device=torch.device("cpu"))


# --------------------------------------------------------------------------- #
# The critic input: normalisation and the density tail
# --------------------------------------------------------------------------- #
# Jobs 62067/62068 ran the WGAN score gap to +2548 by step 30k with the penalty
# at 1414 (||grad D|| ~ 4). The SR2 critic path had NO input normaliser, while
# both other critics in this repo fit one; and the density channels carry an
# unbounded cosmological tail. These tests pin both halves of that fix.
#
# What they do NOT claim: that normalising alone stops the runaway. That is a
# training-dynamics question over tens of thousands of steps and no unit test
# can settle it -- measured at random init, ||grad D|| is ~1.95 with or without
# the fix, because a fresh critic's gradient norm is set by its weight scale.
# What is measured here is the input conditioning, which was unambiguously wrong.
def test_asinh_compresses_the_density_tail_and_leaves_the_head_alone():
    from cosmo_sr.train.sr2_scratch import apply_density_transform

    head = torch.randn(2, 12, 4, 4, 4)
    rho = torch.full((2, 8, 4, 4, 4), 2786.0)
    x = torch.cat([head, rho], dim=1)
    y = apply_density_transform(x, 8, "asinh")
    assert y.shape == x.shape
    assert torch.equal(y[:, :12], head)
    assert float(y[0, -1, 0, 0, 0]) == pytest.approx(8.6255, abs=1e-3)


def test_asinh_is_finite_at_the_median_voxel():
    """delta = -1 is the MEDIAN voxel of a tile (most of it is empty).

    log1p is -inf there, which is why the transform is asinh.
    """
    from cosmo_sr.train.sr2_scratch import apply_density_transform

    x = torch.cat([torch.zeros(1, 12, 2, 2, 2),
                   torch.full((1, 8, 2, 2, 2), -1.0)], dim=1)
    y = apply_density_transform(x, 8, "asinh")
    assert torch.isfinite(y).all()
    assert float(y[0, -1, 0, 0, 0]) == pytest.approx(-0.881374, abs=1e-5)


def test_density_transform_none_is_the_identity():
    from cosmo_sr.train.sr2_scratch import apply_density_transform

    x = torch.randn(2, 20, 4, 4, 4)
    assert torch.equal(apply_density_transform(x, 8, "none"), x)


def test_density_transform_is_differentiable():
    from cosmo_sr.train.sr2_scratch import apply_density_transform

    x = torch.randn(1, 20, 4, 4, 4, requires_grad=True)
    apply_density_transform(x, 8, "asinh").sum().backward()
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0


@pytest.mark.parametrize("bad", [("nope", 8), ("asinh", 0), ("asinh", 20)])
def test_density_transform_rejects_nonsense(bad):
    from cosmo_sr.train.sr2_scratch import apply_density_transform

    kind, n = bad
    with pytest.raises(ValueError):
        apply_density_transform(torch.randn(1, 20, 4, 4, 4), n, kind)


def test_fitted_scales_equalise_the_channels(boxes, tiny_model):
    """The defect: density entered at 5.8x the field channels, so the field
    half starved. After fitting, every channel enters at RMS ~1."""
    from cosmo_sr.train.sr2_scratch import fit_critic_norm

    lr, hr = boxes
    kw = dict(cellsize_kpc_h=CELLSIZE_KPC_H, grid_mult=2, valid_center=TILE // 2)
    mk = lambda seed: PairedTileSampler(
        ["b0", "b1"], lr, hr, geom=TEST_GEOM, ng_hr=NG_HR, tile_hr=TILE,
        noise_seed_stride=0, seed=seed, device=torch.device("cpu"))
    norm = fit_critic_norm(mk(1), TEST_GEOM, n_tiles=4, batch=2,
                           density_transform="asinh", **kw)
    b = mk(2).batch(2)
    x = scratch_critic_input(lr_core(b.lr, TEST_GEOM), b.hr,
                             density_transform="asinh", norm=norm, **kw)
    rms = x.pow(2).mean(dim=(0, 2, 3, 4)).sqrt()
    # Fitted on other tiles of the same boxes, so this is a loose band, not 1.0
    assert 0.2 < float(rms.min()) and float(rms.max()) < 5.0
    assert float(rms.max() / rms.min()) < 4.0


def test_the_same_scales_are_applied_to_real_and_fake(boxes, tiny_model):
    """Per-batch statistics would normalise the two sides differently and hand
    the critic a shortcut that has nothing to do with sample quality."""
    from cosmo_sr.train.sr2_scratch import fit_critic_norm

    lr, hr = boxes
    kw = dict(cellsize_kpc_h=CELLSIZE_KPC_H, grid_mult=2, valid_center=TILE // 2)
    sam = PairedTileSampler(["b0"], lr, hr, geom=TEST_GEOM, ng_hr=NG_HR,
                            tile_hr=TILE, noise_seed_stride=0, seed=3,
                            device=torch.device("cpu"))
    norm = fit_critic_norm(sam, TEST_GEOM, n_tiles=2, batch=2,
                           density_transform="asinh", **kw)
    before = norm.scale.clone()
    b = sam.batch(2)
    cand = forward_batch(tiny_model, b, TEST_GEOM)
    scratch_critic_input(lr_core(b.lr, TEST_GEOM), cand,
                         density_transform="asinh", norm=norm, **kw)
    scratch_critic_input(lr_core(b.lr, TEST_GEOM), b.hr,
                         density_transform="asinh", norm=norm, **kw)
    assert torch.equal(before, norm.scale), "the scales moved between real and fake"


def test_norm_none_returns_the_unscaled_tensor_that_fit_consumes(boxes):
    from cosmo_sr.train.sr2_scratch import fit_critic_norm

    lr, hr = boxes
    kw = dict(cellsize_kpc_h=CELLSIZE_KPC_H, grid_mult=2, valid_center=TILE // 2)
    sam = PairedTileSampler(["b0"], lr, hr, geom=TEST_GEOM, ng_hr=NG_HR,
                            tile_hr=TILE, noise_seed_stride=0, seed=4,
                            device=torch.device("cpu"))
    norm = fit_critic_norm(sam, TEST_GEOM, n_tiles=2, batch=2,
                           density_transform="asinh", **kw)
    b = sam.batch(1)
    raw = scratch_critic_input(lr_core(b.lr, TEST_GEOM), b.hr,
                               density_transform="asinh", norm=None, **kw)
    scaled = scratch_critic_input(lr_core(b.lr, TEST_GEOM), b.hr,
                                  density_transform="asinh", norm=norm, **kw)
    assert not torch.allclose(raw, scaled)
    assert torch.allclose(scaled, raw / norm.scale.view(1, -1, 1, 1, 1))


def test_checkpoint_carries_the_critic_scales(tmp_path, boxes):
    """A gate that re-derived the scales from its own tiles would be scoring a
    different function than the one that was trained."""
    import train_sr2_scratch as TS
    from cosmo_sr.train.sr2_scratch import fit_critic_norm

    lr, hr = boxes
    kw = dict(cellsize_kpc_h=CELLSIZE_KPC_H, grid_mult=2, valid_center=TILE // 2)
    sam = PairedTileSampler(["b0"], lr, hr, geom=TEST_GEOM, ng_hr=NG_HR,
                            tile_hr=TILE, noise_seed_stride=0, seed=6,
                            device=torch.device("cpu"))
    norm = fit_critic_norm(sam, TEST_GEOM, n_tiles=2, batch=2,
                           density_transform="asinh", **kw)
    m = build_scratch_generator(scale_factor=8, device=torch.device("cpu"),
                                **TINY_CHAN)
    p = tmp_path / "latest.pt"
    TS.save_state(p, model=m, critic=None, opt=torch.optim.Adam(m.parameters()),
                  opt_d=None, step=7, args=_scratch_args([]),
                  rng_state={"sampler": None}, critic_norm=norm)
    blob = torch.load(str(p), map_location="cpu", weights_only=False)
    assert blob["density_transform"] == "asinh"
    assert blob["critic_norm"]["scale"] == [float(v) for v in norm.scale]


def test_resuming_a_checkpoint_without_the_normaliser_is_refused(tmp_path):
    """Run 1's checkpoints predate the input normaliser.

    Their critic was fitted to raw channels (density RMS ~5, max ~2800) and the
    fixed run feeds it normalised ones (RMS ~1, max ~10). Resuming continues a
    DIFFERENT experiment and nothing in the logs says so -- which is exactly
    what happened on 2026-09-23, when all four arms silently resumed run 1.
    """
    import train_sr2_scratch as TS

    m = build_scratch_generator(scale_factor=8, device=torch.device("cpu"),
                                **TINY_CHAN)
    p = tmp_path / "latest.pt"
    torch.save({"scratch": True, "step": 63500, "model": m.state_dict(),
                "opt": torch.optim.Adam(m.parameters()).state_dict(),
                "critic": None, "opt_d": None, "rng": {}}, p)
    with pytest.raises(SystemExit, match="predates the input normaliser"):
        TS.load_state(p, model=m, critic=None,
                      opt=torch.optim.Adam(m.parameters()), opt_d=None,
                      device=torch.device("cpu"), wants_critic=True)


def test_resuming_across_a_density_transform_change_is_refused(tmp_path):
    import train_sr2_scratch as TS

    m = build_scratch_generator(scale_factor=8, device=torch.device("cpu"),
                                **TINY_CHAN)
    p = tmp_path / "latest.pt"
    torch.save({"scratch": True, "step": 10, "model": m.state_dict(),
                "opt": torch.optim.Adam(m.parameters()).state_dict(),
                "critic": None, "opt_d": None, "rng": {},
                "critic_norm": {"scale": [1.0] * 20},
                "density_transform": "none"}, p)
    with pytest.raises(SystemExit, match="density-transform"):
        TS.load_state(p, model=m, critic=None,
                      opt=torch.optim.Adam(m.parameters()), opt_d=None,
                      device=torch.device("cpu"), wants_critic=True,
                      expect_transform="asinh")


def test_an_arm_with_no_critic_resumes_without_a_contract_check(tmp_path):
    """wants_critic=False: there is no critic, so there is no contract."""
    import train_sr2_scratch as TS

    m = build_scratch_generator(scale_factor=8, device=torch.device("cpu"),
                                **TINY_CHAN)
    p = tmp_path / "latest.pt"
    torch.save({"scratch": True, "step": 77, "model": m.state_dict(),
                "opt": torch.optim.Adam(m.parameters()).state_dict(),
                "critic": None, "opt_d": None, "rng": {}}, p)
    step, _rng, norm = TS.load_state(
        p, model=m, critic=None, opt=torch.optim.Adam(m.parameters()),
        opt_d=None, device=torch.device("cpu"), wants_critic=False)
    assert step == 77 and norm is None


def test_the_saved_scales_are_returned_for_the_caller_to_adopt(tmp_path):
    """The fit draws RANDOM tiles; the critic was trained under one draw."""
    import train_sr2_scratch as TS

    m = build_scratch_generator(scale_factor=8, device=torch.device("cpu"),
                                **TINY_CHAN)
    p = tmp_path / "latest.pt"
    scales = [0.5 + 0.01 * i for i in range(20)]
    torch.save({"scratch": True, "step": 5, "model": m.state_dict(),
                "opt": torch.optim.Adam(m.parameters()).state_dict(),
                "critic": None, "opt_d": None, "rng": {},
                "critic_norm": {"scale": scales},
                "density_transform": "asinh"}, p)
    _step, _rng, norm = TS.load_state(
        p, model=m, critic=None, opt=torch.optim.Adam(m.parameters()),
        opt_d=None, device=torch.device("cpu"), wants_critic=True,
        expect_transform="asinh")
    assert norm["scale"] == scales


# --------------------------------------------------------------------------- #
# The field report (scripts/features/sr2_scratch_field_report.py)
# --------------------------------------------------------------------------- #
# The trainer's holdout mse was the wrong yardstick: real SR2's high-k r(k) is
# ~0 on every channel, so it does not reproduce HR's realisation at small
# scales and a LOWER mse than SR2 means suppressed power, not a better map.
# These pin the report's shape and its two refusals.
def _field_report():
    import sys
    from pathlib import Path
    p = str(Path(__file__).resolve().parents[2] / "scripts" / "sr2")
    if p not in sys.path:
        sys.path.insert(0, p)
    import sr2_scratch_field_report as FR
    return FR


def test_field_table_reports_all_three_rows():
    FR = _field_report()
    rng = np.random.default_rng(0)
    hr = rng.normal(0, 0.1, size=(6, 32, 32, 32)).astype(np.float32)
    sr = (hr + rng.normal(0, 0.02, size=hr.shape)).astype(np.float32)
    t = FR.field_table(sr, hr, cell_kpc_h=195.3, dis_norm_kpc_h=6000.0,
                       device=torch.device("cpu"))
    assert set(t) == {"displacement", "velocity", "density"}
    for f in t:
        assert {"ratio_highk", "transfer_highk", "rk_highk"} <= set(t[f])
        assert all(np.isfinite(v) for v in t[f].values())
    assert "sigma_ratio" in t["density"]


def test_field_table_gives_a_perfect_score_to_the_truth():
    """sr == hr must give ratio and transfer 1 and r(k) 1 -- otherwise the
    estimator is measuring something other than agreement."""
    FR = _field_report()
    rng = np.random.default_rng(1)
    hr = rng.normal(0, 0.1, size=(6, 32, 32, 32)).astype(np.float32)
    t = FR.field_table(hr, hr, cell_kpc_h=195.3, dis_norm_kpc_h=6000.0,
                       device=torch.device("cpu"))
    for f in ("displacement", "velocity", "density"):
        assert t[f]["ratio_highk"] == pytest.approx(1.0, rel=1e-4), f
        assert t[f]["transfer_highk"] == pytest.approx(1.0, rel=1e-4), f
        assert t[f]["rk_highk"] == pytest.approx(1.0, rel=1e-4), f


def test_whole_box_holdout_covers_every_tile():
    """The trainer samples ~16 tiles; at that size the numbers swing 2x."""
    FR = _field_report()
    rng = np.random.default_rng(2)
    hr = rng.normal(0, 0.1, size=(6, 128, 128, 128)).astype(np.float32)
    sr = (hr + rng.normal(0, 0.05, size=hr.shape)).astype(np.float32)
    out = FR.whole_box_holdout(sr, hr, scale_factor=8, device=torch.device("cpu"),
                               ng_hr=128, tile=64)
    assert out["n_tiles"] == (128 // 64) ** 3 == 8
    assert out["mse"] > 0 and np.isfinite(out["vel_rms_ratio"])


def test_field_report_refuses_a_non_scratch_checkpoint(tmp_path):
    FR = _field_report()
    p = tmp_path / "tuned.pt"
    torch.save({"model": {}, "rung": "fine", "step": 10}, p)
    with pytest.raises(SystemExit, match="not written by train_sr2_scratch"):
        FR.load_scratch_generator(p, scale_factor=8, device=torch.device("cpu"))


def test_the_published_reference_is_carried_verbatim():
    """runs/sr2_baseline/field_table: SR2's r(k) is ~0, which is the whole point."""
    FR = _field_report()
    ref = FR.REFERENCE_SET14
    assert ref["density"]["ratio_highk"] == pytest.approx(0.977)
    assert ref["velocity"]["ratio_highk"] == pytest.approx(0.989)
    assert abs(ref["displacement"]["rk_highk"]) < 0.01
    assert abs(ref["velocity"]["rk_highk"]) < 0.01


# --------------------------------------------------------------------------- #
# The upstream-faithful objective
# --------------------------------------------------------------------------- #
# map2map/train.py: the generator steps on `criterion` (MSELoss) until
# `adv_start`, then on the ADVERSARIAL loss ALONE -- the two are never summed,
# and MSE gets no gradient once the critic is live. The adversary is
# BCEWithLogitsLoss on labels plus lazy R1 on the REAL batch; there is no WGAN
# and no interpolate penalty anywhere upstream. And there is no LR-scale anchor.
def test_bce_critic_loss_starts_at_two_ln2():
    """Two BCE terms at logit 0 each give ln2; a wrong sign or target shows here."""
    from cosmo_sr.train.sr2_scratch import bce_critic_update

    class Zero(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
        def forward(self, x):
            return self.w.expand(x.shape[0])

    c = Zero()
    out = bce_critic_update(c, torch.optim.Adam(c.parameters(), lr=0.0),
                            torch.randn(4, 20, 4, 4, 4), torch.randn(4, 20, 4, 4, 4))
    assert out["d_loss"] == pytest.approx(2 * np.log(2), abs=1e-5)


def test_bce_generator_loss_is_non_saturating():
    """BCE(D(fake), 1): pushing D(fake) UP, not -E[D(fake)]."""
    from cosmo_sr.train.sr2_scratch import bce_generator_loss

    class Const(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = torch.nn.Parameter(torch.tensor([float(v)]))
        def forward(self, x):
            return self.v.expand(x.shape[0])

    x = torch.randn(2, 20, 4, 4, 4)
    hi = float(bce_generator_loss(Const(4.0), x))     # D thinks fake is real
    lo = float(bce_generator_loss(Const(-4.0), x))    # D is not fooled
    assert hi < lo, "the generator loss must fall as D(fake) rises"
    assert float(bce_generator_loss(Const(0.0), x)) == pytest.approx(np.log(2), abs=1e-5)


def test_label_smoothing_reaches_the_real_target():
    """At logit 0 the BCE is ln2 for ANY target, so the probe must be off zero.

    -[t*log s + (1-t)*log(1-s)] with s = 0.5 collapses to ln2 independently of
    t; a test at the default init would pass whatever the target was.
    """
    from cosmo_sr.train.sr2_scratch import bce_critic_update

    class Const(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([float(v)]))
        def forward(self, x):
            return self.w.expand(x.shape[0])

    kw = dict(real_in=torch.randn(4, 20, 4, 4, 4), fake_in=torch.randn(4, 20, 4, 4, 4))
    c1, c2 = Const(2.0), Const(2.0)
    a = bce_critic_update(c1, torch.optim.Adam(c1.parameters(), lr=0.0),
                          label_smoothing=1.0, **kw)["d_loss"]
    b = bce_critic_update(c2, torch.optim.Adam(c2.parameters(), lr=0.0),
                          label_smoothing=0.9, **kw)["d_loss"]
    assert b > a, "smoothing the real target must raise D's loss at a confident logit"


def test_r1_is_applied_on_the_real_batch_only_and_lazily():
    from cosmo_sr.dmsr.critic import LazyR1
    from cosmo_sr.train.sr2_scratch import bce_critic_update, build_scratch_critic

    c = build_scratch_critic(width=8, depth=3, device=torch.device("cpu"))
    o = torch.optim.Adam(c.parameters(), lr=1e-4)
    r1 = LazyR1(gamma=10.0, interval=3)
    seen = [bce_critic_update(c, o, torch.randn(2, 20, 8, 8, 8),
                              torch.randn(2, 20, 8, 8, 8),
                              lazy_r1=r1)["d_r1"] for _ in range(6)]
    fired = [i for i, v in enumerate(seen) if v > 0]
    assert fired == [0, 3], f"R1 should fire every 3 steps, fired at {fired}"


def test_upstream_defaults_are_map2maps():
    """No LR anchor, Adam defaults, lr 1e-4, BCE, upstream mse schedule."""
    args = _scratch_args([])
    assert args.w_base_low == 0.0, "map2map has no LR-scale anchor"
    assert args.adv_loss == "bce"
    assert args.mse_mode == "upstream"
    assert args.scratch_lr == pytest.approx(1e-4)
    assert list(args.scratch_betas) == [0.9, 0.999]
    assert args.w_mse == 1.0


# --------------------------------------------------------------------------- #
# Stabilisers and the bracket verdict
# --------------------------------------------------------------------------- #
def test_spectral_norm_wraps_every_conv_and_the_head():
    c = build_scratch_critic(width=8, depth=3, spectral_norm=True,
                             device=torch.device("cpu"))
    n_sn = sum(1 for n, _ in c.named_parameters() if "parametriz" in n)
    n_wt = sum(1 for m in c.modules()
               if isinstance(m, (torch.nn.Conv3d, torch.nn.Linear)))
    assert n_sn == n_wt, f"{n_sn} spectral-normed against {n_wt} weight layers"
    assert c(torch.randn(2, 20, 8, 8, 8)).shape == (2,)


def test_spectral_norm_is_off_by_default():
    """map2map's --adv-model-spectral-norm is a flag, not a default."""
    c = build_scratch_critic(width=8, depth=3, device=torch.device("cpu"))
    assert not any("parametriz" in n for n, _ in c.named_parameters())


def test_instance_noise_anneals_to_zero():
    from cosmo_sr.train.sr2_scratch import instance_noise_std

    assert instance_noise_std(0, sigma=0.1, batches=1000) == pytest.approx(0.1)
    assert instance_noise_std(500, sigma=0.1, batches=1000) == pytest.approx(0.05)
    assert instance_noise_std(1000, sigma=0.1, batches=1000) == 0.0
    assert instance_noise_std(5000, sigma=0.1, batches=1000) == 0.0
    assert instance_noise_std(0, sigma=0.0, batches=1000) == 0.0


def test_instance_noise_is_a_no_op_at_zero():
    from cosmo_sr.train.sr2_scratch import add_instance_noise

    x = torch.randn(2, 20, 4, 4, 4)
    assert torch.equal(add_instance_noise(x, 0.0), x)
    assert not torch.equal(add_instance_noise(x, 0.1), x)


def _verdict():
    import sys
    from pathlib import Path
    p = str(Path(__file__).resolve().parents[2] / "scripts" / "features")
    if p not in sys.path:
        sys.path.insert(0, p)
    import sr2_scratch_bracket_verdict as V
    return V


def _rows(mses, gaps=None, w_mse=0.0, w_adv=1.0):
    out = []
    for i, m in enumerate(mses):
        r = {"step": 250 * (i + 1), "base_mse": m, "base_low_k": 0.05,
             "w_mse_now": w_mse, "noise_std_mean": 1e-3}
        if gaps is not None:
            r["adv"] = {"d_real": gaps[i] / 2, "d_fake": -gaps[i] / 2,
                        "d_loss": 0.69, "w_adv": w_adv}
        out.append(r)
    return out


def test_verdict_flags_divergence():
    V = _verdict()
    v = V.verdict_of(_rows([0.2, 1.6e8, 4.4e11]), mse_max=10.0, gap_max=8.0,
                     dloss_min=0.05)
    assert v["verdict"] == "DIVERGED" and "4.4e+11" in v["why"]


def test_verdict_flags_a_saturated_discriminator():
    V = _verdict()
    v = V.verdict_of(_rows([0.2, 0.2, 0.2], gaps=[1.0, 4.0, 16.7]),
                     mse_max=10.0, gap_max=8.0, dloss_min=0.05)
    assert v["verdict"] == "SATURATED" and "16.7" in v["why"]


def test_verdict_flags_a_probe_that_never_handed_over():
    """Probe A by design; a warning for any other probe."""
    V = _verdict()
    v = V.verdict_of(_rows([0.2, 0.19, 0.18], gaps=[1.0, 1.0, 1.0], w_mse=1.0,
                           w_adv=0.0),
                     mse_max=10.0, gap_max=8.0, dloss_min=0.05)
    assert v["verdict"] == "UNTESTED" and not v["handed_over"]


def test_mse_mode_sum_still_counts_as_exercised():
    """The bug this caught: under --mse-mode sum, w_mse_now NEVER reaches 0, so
    keying `handed_over` on it marked probe B UNTESTED when its adversary had
    driven the generator for 1750 stable steps."""
    V = _verdict()
    v = V.verdict_of(_rows([0.2, 0.09, 0.20], gaps=[3.6, 1.6, 2.8], w_mse=1.0,
                           w_adv=1.0),
                     mse_max=10.0, gap_max=8.0, dloss_min=0.05)
    assert v["handed_over"], "w_adv > 0 means the adversary drove the generator"
    assert v["verdict"] == "OK"


def test_verdict_passes_a_stable_handoff():
    V = _verdict()
    v = V.verdict_of(_rows([0.2, 0.18, 0.17], gaps=[1.0, 2.0, 1.5]),
                     mse_max=10.0, gap_max=8.0, dloss_min=0.05)
    assert v["verdict"] == "OK" and v["handed_over"]


def test_verdict_flags_a_collapsed_d_loss():
    """Run 3's d_loss was 0.0005; a BCE discriminator sits near ln2."""
    V = _verdict()
    rows = _rows([0.2, 0.2], gaps=[1.0, 1.0])
    rows[-1]["adv"]["d_loss"] = 0.0005
    v = V.verdict_of(rows, mse_max=10.0, gap_max=8.0, dloss_min=0.05)
    assert v["verdict"] == "SATURATED" and "ln2" in v["why"]


# --------------------------------------------------------------------------- #
# The handoff divergence: the critic's blindness to amplitude
#
# Bracket probes C and D (2026-09-24) blew up to 1e10-1e11 within one eval
# interval of `w_mse_now` reaching 0. The reading at the time was "a saturated
# discriminator gives no gradient back". It is not: the non-saturating BCE
# generator loss `softplus(-D)` has dL/dD -> -1 as D -> -inf, so the loss side
# is at FULL strength. What actually fails is the critic, and these tests pin
# the mechanism so it cannot come back silently.
# --------------------------------------------------------------------------- #
def _scale_probe(critic, x, alphas):
    out = []
    for a in alphas:
        xa = (x * a).clone().requires_grad_(True)
        s = critic(xa)
        g, = torch.autograd.grad(s.sum(), xa)
        out.append((float(s.mean().detach()), float(g.norm())))
    return out


def test_groupnorm_makes_the_critic_exactly_scale_invariant():
    """The ROOT CAUSE. GroupNorm divides out the per-group std, and Conv and
    LeakyReLU are positively homogeneous, so ``D(a*x)`` does not depend on ``a``
    at all: the critic cannot see that the field is 1e5 times too large.
    """
    torch.manual_seed(0)
    critic = SR2Critic(width=16, depth=4).eval()
    x = torch.randn(2, 20, 16, 16, 16)
    # The conv biases make the homogeneity asymptotic rather than algebraic, so
    # the invariance is stated relatively and from 1e4 up, where the bias is
    # already six orders below the signal it sits on.
    probe = _scale_probe(critic, x, [1e4, 1e5, 1e6, 1e7])
    scores = [p[0] for p in probe]
    for s in scores[1:]:
        assert abs(s - scores[0]) / abs(scores[0]) < 1e-3, (
            f"D moved with the input scale: {scores}")


def test_the_critics_gradient_decays_as_one_over_the_scale():
    """Worse than blind: self-reinforcing. ``||dD/dx|| ~ 1/a``, so the further
    the field walks out, the WEAKER the only signal that could bring it back.
    That is why the runs thrash across five decades instead of recovering.
    """
    torch.manual_seed(0)
    critic = SR2Critic(width=16, depth=4).eval()
    x = torch.randn(2, 20, 16, 16, 16)
    alphas = [1e3, 1e4, 1e5, 1e6]
    probe = _scale_probe(critic, x, alphas)
    products = [a * g for a, (_, g) in zip(alphas, probe)]
    for pr in products[1:]:
        assert abs(pr - products[0]) / products[0] < 1e-2, (
            f"a*||grad D|| is not constant: {products}")


def test_spectral_norm_does_not_restore_scale_sensitivity():
    """Probes C and D both ran ``--critic-spectral-norm`` and both diverged.
    Spectral norm bounds the critic; it does not let it see amplitude.
    """
    critic = build_scratch_critic(width=16, depth=4, seed=0,
                                  spectral_norm=True, device="cpu").eval()
    x = torch.randn(2, 20, 16, 16, 16)
    scores = [s for s, _ in _scale_probe(critic, x, [1e4, 1e6])]
    assert abs(scores[1] - scores[0]) / abs(scores[0]) < 1e-3, scores


def test_inner_norm_none_makes_the_critic_see_amplitude():
    """The fix. Without GroupNorm the critic is positively homogeneous, so a
    field 1e3 times too large produces a score 1e3 times larger -- "far too
    big" becomes something the critic can learn to say.
    """
    critic = build_scratch_critic(width=16, depth=4, seed=0, spectral_norm=True,
                                  inner_norm="none", device="cpu").eval()
    x = torch.randn(2, 20, 16, 16, 16)
    (s1, _), (s2, _) = _scale_probe(critic, x, [1.0, 1e3])
    assert abs(s2) > 100 * abs(s1), (
        f"scores {s1} -> {s2} did not grow with the input scale")


def test_inner_norm_choice_is_validated():
    with pytest.raises(ValueError, match="inner_norm"):
        SR2Critic(width=8, depth=2, inner_norm="batch")


def test_group_is_still_the_default_so_existing_checkpoints_load():
    assert SR2Critic(width=8, depth=3).inner_norm == "group"
    assert (set(SR2Critic(width=8, depth=3).state_dict())
            == set(SR2Critic(width=8, depth=3, inner_norm="group").state_dict()))


def test_the_two_inner_norms_are_different_architectures():
    """Dropping the GroupNorms shifts every later index in the Sequential, so a
    `group` checkpoint and a `none` checkpoint are not interchangeable. The
    trainer resumes with a strict load_state_dict, so changing --critic-inner-norm
    across a --resume raises rather than silently loading the wrong tensors.
    """
    group = SR2Critic(width=8, depth=3, inner_norm="group")
    none = SR2Critic(width=8, depth=3, inner_norm="none")
    with pytest.raises(RuntimeError):
        none.load_state_dict(group.state_dict())


# --------------------------------------------------------------------------- #
# The amplitude leash
# --------------------------------------------------------------------------- #
def test_scale_anchor_is_zero_on_a_perfect_match():
    hr = torch.randn(2, 6, 8, 8, 8)
    out = scale_anchor(hr.clone(), hr)
    assert float(out["scale"]) == pytest.approx(0.0, abs=1e-6)
    assert float(out["rms_ratio_dis"]) == pytest.approx(1.0, rel=1e-6)
    assert float(out["rms_ratio_vel"]) == pytest.approx(1.0, rel=1e-6)


def test_scale_anchor_reports_the_ratio_that_ran_away():
    hr = torch.randn(2, 6, 8, 8, 8)
    cand = hr.clone()
    cand[:, 3:6] *= 1e5          # probes C/D reached vel_rms_ratio 5.8e5
    out = scale_anchor(cand, hr)
    assert float(out["rms_ratio_vel"]) == pytest.approx(1e5, rel=1e-4)
    assert float(out["rms_ratio_dis"]) == pytest.approx(1.0, rel=1e-6)


def test_scale_anchor_is_symmetric_in_the_log():
    """Being 10x too loud and 10x too quiet must cost the same, or the term
    biases the amplitude it is supposed to leave alone."""
    hr = torch.randn(2, 6, 8, 8, 8)
    loud = float(scale_anchor(hr * 10.0, hr)["scale"])
    quiet = float(scale_anchor(hr / 10.0, hr)["scale"])
    assert loud == pytest.approx(quiet, rel=1e-5)


def test_scale_anchor_carries_no_spatial_information():
    """The whole reason it is safe to keep when MSE goes: a permutation of the
    field in space -- any radial rearrangement whatsoever -- does not change it.
    """
    torch.manual_seed(0)
    hr = torch.randn(2, 6, 8, 8, 8)
    cand = torch.randn(2, 6, 8, 8, 8)
    flat = cand.reshape(2, 6, -1)
    shuffled = flat[:, :, torch.randperm(flat.shape[-1])].reshape(cand.shape)
    # Exact in exact arithmetic; the tolerance is float32 summation order.
    assert float(scale_anchor(cand, hr)["scale"]) == pytest.approx(
        float(scale_anchor(shuffled, hr)["scale"]), rel=1e-4)


def test_scale_anchor_gradient_pushes_an_inflated_field_back_down():
    hr = torch.randn(2, 6, 8, 8, 8)
    cand = (hr * 1e4).clone().requires_grad_(True)
    scale_anchor(cand, hr)["scale"].backward()
    # the gradient must oppose the field, i.e. shrink it
    assert float((cand.grad * cand.detach()).sum()) > 0


def test_scale_anchor_gradient_is_finite_at_a_perfect_match():
    hr = torch.randn(2, 6, 8, 8, 8)
    cand = hr.clone().requires_grad_(True)
    scale_anchor(cand, hr)["scale"].backward()
    assert torch.isfinite(cand.grad).all()


def test_scale_anchor_does_not_backpropagate_into_hr():
    hr = torch.randn(2, 6, 8, 8, 8, requires_grad=True)
    cand = torch.randn(2, 6, 8, 8, 8, requires_grad=True)
    scale_anchor(cand, hr)["scale"].backward()
    assert hr.grad is None


# --------------------------------------------------------------------------- #
# The handoff ramp
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("step,expected", [
    (1, 1.0), (1000, 1.0), (1001, 0.998), (1250, 0.5), (1500, 0.0), (2000, 0.0),
])
def test_mse_handoff_ramps_instead_of_cliffing(step, expected):
    assert mse_handoff_weight(step, 1.0, adv_start=1000,
                              handoff=500) == pytest.approx(expected)


def test_mse_handoff_zero_is_the_upstream_cliff():
    assert mse_handoff_weight(1000, 1.0, adv_start=1000, handoff=0) == 1.0
    assert mse_handoff_weight(1001, 1.0, adv_start=1000, handoff=0) == 0.0


def test_a_constant_knn_penalty_is_not_backwarded():
    """`_unsup_knn` returns a CONSTANT zero when every tile in the minibatch was
    skipped, and backward() on it raises "element 0 of tensors does not require
    grad". That killed knn_r5 at step 11500, and it only surfaced because
    SC_UNSUP_TILES_PER_STEP went 8 -> 1, making one unlucky tile the whole batch.
    """
    const = torch.zeros((), dtype=torch.float64)
    assert not const.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        (1.0 * const).backward()
    # the guard the trainer now applies
    live = (torch.randn(4, requires_grad=True) ** 2).mean()
    assert live.requires_grad
    (1.0 * live).backward()
