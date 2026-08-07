"""The Gaussian residual policy: exact log-probability, init, and tiling.

Three things have to be true for Phase 2 to mean anything, and all three are
cheap to check on CPU:

* the log probability is the *actual* Gaussian density of the sampled
  coefficients -- reward-weighted likelihood on a wrong density optimises
  nothing in particular;
* exploration starts nonzero and at the configured value, or the support gate is
  measuring the initialisation rather than the reward;
* a tiled full-box sample equals an untiled one, or every full-box number is a
  property of the tiling.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from cosmo_sr.reward.correction import CorrectionConfig, CorrectionScales
from cosmo_sr.reward.gaussian_policy import (SCALE_NAMES, GaussianPolicyConfig,
                                             analytic_receptive_field,
                                             MultiScaleGaussianPolicy,
                                             build_gaussian_policy, diag_gaussian_kl,
                                             diag_gaussian_log_prob, global_noise,
                                             noise_shapes, policy_receptive_field,
                                             sample_policy_box, scale_specs)
from cosmo_sr.reward.sampling import TileSpec

C, SF = 6, 4


def scales() -> CorrectionScales:
    return CorrectionScales(fine_disp=0.02, fine_vel=0.05, coarse_disp=0.004,
                            coarse_vel=0.01, calibrated=True, source="test")


def policy(*, scale_factor=SF, **kw) -> MultiScaleGaussianPolicy:
    opts = dict(channels=C, scale_factor=scale_factor, width=8, num_levels=2,
                blocks_per_level=1, num_groups=4, use_checkpoint=False,
                sigma_init={"coarse": 0.05, "middle": 0.15, "fine": 0.15})
    opts.update(kw)
    corr = CorrectionConfig(mode="block_leaky", scale_factor=scale_factor, channels=C,
                            alpha_disp=0.25, alpha_vel=0.5, scales=scales())
    return MultiScaleGaussianPolicy(GaussianPolicyConfig(**opts), corr).eval()


def inputs(n=16, batch=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, C, n, n, n, generator=g),
            torch.randn(batch, C, n // SF, n // SF, n // SF, generator=g))


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def test_scale_specs_are_coarse_to_fine_with_powers_of_two():
    specs = scale_specs(2)
    assert [s.name for s in specs] == list(SCALE_NAMES)
    assert [s.stride for s in specs] == [4, 2, 1]


def test_distribution_has_one_head_per_scale_at_the_right_resolution():
    p = policy()
    base, lr = inputs(16)
    dist = p.distribution(base, lr)
    assert set(dist) == set(SCALE_NAMES)
    for s in p.specs:
        mu, sigma = dist[s.name]
        assert mu.shape == (1, C, 16 // s.stride, 16 // s.stride, 16 // s.stride)
        assert sigma.shape == mu.shape


def test_input_is_base_plus_upsampled_lr():
    p = policy()
    assert p.trunk.enc_stages[0][0].conv1.in_channels == 2 * C


def test_heads_are_pointwise():
    """A head with spatial extent would widen the receptive field and break tiling."""
    p = policy()
    for s in p.specs:
        assert p.mu_heads[s.name].kernel_size == (1, 1, 1)
        assert p.logsig_heads[s.name].kernel_size == (1, 1, 1)


def test_trunk_uses_circular_padding_and_pointwise_norm():
    from cosmo_sr.models.flow_unet import ChannelGroupNorm3d

    p = policy()
    convs = [m for m in p.trunk.modules() if isinstance(m, torch.nn.Conv3d)
             and m.padding != (0, 0, 0)]
    assert convs and all(m.padding_mode == "circular" for m in convs)
    assert any(isinstance(m, ChannelGroupNorm3d) for m in p.trunk.modules())


def test_crop_size_must_suit_the_number_of_levels():
    """padding='same' needs a size divisible by 2**num_levels, or the decoder's
    skip concatenation silently center-crops and the output is not the input's
    grid."""
    p = policy(scale_factor=2)
    with pytest.raises(ValueError, match="multiple of 4"):
        p.distribution(torch.zeros(1, C, 6, 6, 6), torch.zeros(1, C, 3, 3, 3))


# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #
def test_mean_heads_start_at_exactly_zero():
    p = policy()
    dist = p.distribution(*inputs(16))
    for name, (mu, _) in dist.items():
        assert torch.count_nonzero(mu) == 0, f"{name} mu is not zero at init"


def test_sigma_starts_at_the_configured_value_per_scale():
    p = policy()
    dist = p.distribution(*inputs(16))
    for s in p.specs:
        _, sigma = dist[s.name]
        want = p.cfg.sigma_init[s.name]
        assert torch.allclose(sigma, torch.full_like(sigma, want), atol=1e-6)


def test_exploration_is_nonzero_everywhere():
    p = policy()
    for _, sigma in p.distribution(*inputs(16)).values():
        assert float(sigma.detach().min()) > 0.0


def test_coarse_exploration_starts_smaller_than_the_finer_scales():
    p = policy()
    dist = p.distribution(*inputs(16))
    coarse = float(dist["coarse"][1].detach().mean())
    assert coarse < float(dist["middle"][1].detach().mean())
    assert coarse < float(dist["fine"][1].detach().mean())


def test_a_config_with_zero_exploration_is_refused():
    with pytest.raises(ValueError, match="NONZERO"):
        GaussianPolicyConfig(sigma_init={"coarse": 0.0, "middle": 0.1, "fine": 0.1})


def test_a_config_with_coarse_exploration_larger_than_fine_is_refused():
    with pytest.raises(ValueError, match="coarse exploration"):
        GaussianPolicyConfig(sigma_init={"coarse": 0.5, "middle": 0.1, "fine": 0.1})


def test_displacement_and_velocity_get_separate_initial_scales():
    p = policy(sigma_init_disp_scale=1.0, sigma_init_vel_scale=0.2)
    _, sigma = p.distribution(*inputs(16))["fine"]
    assert float(sigma[:, 0:3].mean()) == pytest.approx(0.15, rel=1e-4)
    assert float(sigma[:, 3:6].mean()) == pytest.approx(0.03, rel=1e-4)


def test_sigma_is_clamped_to_its_bounds():
    p = policy()
    with torch.no_grad():
        p.logsig_heads["fine"].bias.fill_(50.0)
        p.logsig_heads["coarse"].bias.fill_(-50.0)
    dist = p.distribution(*inputs(16))
    assert float(dist["fine"][1].max()) == pytest.approx(p.cfg.sigma_max, rel=1e-5)
    assert float(dist["coarse"][1].min()) == pytest.approx(p.cfg.sigma_min, rel=1e-5)


def test_config_rejects_unknown_options():
    with pytest.raises(ValueError, match="unknown gaussian policy option"):
        GaussianPolicyConfig.from_dict({"widht": 48})


# --------------------------------------------------------------------------- #
# The log probability is exact
# --------------------------------------------------------------------------- #
def test_log_prob_matches_torch_distributions():
    a = torch.randn(3, 4, 5)
    mu = torch.randn(3, 4, 5)
    sigma = torch.rand(3, 4, 5) + 0.1
    want = torch.distributions.Normal(mu, sigma).log_prob(a).flatten(1).sum(1)
    assert torch.allclose(diag_gaussian_log_prob(a, mu, sigma), want, atol=1e-5)


def test_policy_log_prob_is_the_sum_over_scales():
    p = policy()
    base, lr = inputs(16)
    res = p(base, lr, seed=7, grid_hr=16)
    manual = sum(
        torch.distributions.Normal(*res["dist"][s.name])
        .log_prob(res["actions"][s.name]).flatten(1).sum(1)
        for s in p.specs
    )
    assert torch.allclose(res["log_prob"], manual, rtol=1e-5)


def test_log_prob_responds_to_the_parameters_it_should():
    """Re-evaluating a FIXED action under a changed mean must change log pi."""
    p = policy()
    base, lr = inputs(16)
    res = p(base, lr, seed=1, grid_hr=16)
    a = {k: v.detach().clone() for k, v in res["actions"].items()}
    before = float(p(base, lr, actions=a)["log_prob"])
    with torch.no_grad():
        p.mu_heads["fine"].bias.fill_(0.5)
    after = float(p(base, lr, actions=a)["log_prob"])
    assert not math.isclose(before, after, rel_tol=1e-6)


def test_log_prob_is_maximised_at_the_mean():
    p = policy()
    base, lr = inputs(16)
    dist = p.distribution(base, lr)
    at_mean = {s.name: dist[s.name][0].clone() for s in p.specs}
    off = {k: v + 0.3 for k, v in at_mean.items()}
    assert float(p.log_prob(dist, at_mean)) > float(p.log_prob(dist, off))


def test_kl_is_zero_against_itself_and_positive_otherwise():
    mu = torch.randn(2, 3, 4)
    sig = torch.rand(2, 3, 4) + 0.1
    assert float(diag_gaussian_kl(mu, sig, mu.clone(), sig.clone()).abs().max()) < 1e-6
    assert float(diag_gaussian_kl(mu, sig, mu + 1.0, sig).min()) > 0.0


def test_closed_form_kl_matches_a_monte_carlo_estimate():
    torch.manual_seed(0)
    mu_p, sig_p = torch.zeros(1, 4000), torch.full((1, 4000), 0.7)
    mu_q, sig_q = torch.full((1, 4000), 0.2), torch.full((1, 4000), 1.1)
    exact = float(diag_gaussian_kl(mu_p, sig_p, mu_q, sig_q))
    x = mu_p + sig_p * torch.randn(1, 4000)
    mc = float(diag_gaussian_log_prob(x, mu_p, sig_p) - diag_gaussian_log_prob(x, mu_q, sig_q))
    assert mc == pytest.approx(exact, rel=0.15)


def test_policy_kl_against_a_frozen_reference():
    p = policy()
    ref = policy()
    base, lr = inputs(16)
    d, r = p.distribution(base, lr), ref.distribution(base, lr)
    assert float(p.kl_to(d, r).abs().max()) < 1e-5
    with torch.no_grad():
        p.mu_heads["fine"].bias.fill_(1.0)
    assert float(p.kl_to(p.distribution(base, lr), r).min()) > 0.0


# --------------------------------------------------------------------------- #
# Actions, combination, correction
# --------------------------------------------------------------------------- #
def test_actions_are_mu_plus_sigma_times_noise():
    p = policy()
    base, lr = inputs(16)
    dist = p.distribution(base, lr)
    noise = {s.name: torch.ones(1, C, *(16 // s.stride,) * 3) for s in p.specs}
    a = p.sample_actions(dist, noise)
    for s in p.specs:
        mu, sigma = dist[s.name]
        assert torch.allclose(a[s.name], mu + sigma, atol=1e-6)


def test_combine_sums_upsampled_coefficient_fields():
    p = policy()
    a = {s.name: torch.full((1, C, *(16 // s.stride,) * 3), float(i + 1))
         for i, s in enumerate(p.specs)}
    h = p.combine(a, 16)
    assert h.shape == (1, C, 16, 16, 16)
    assert torch.allclose(h, torch.full_like(h, 1.0 + 2.0 + 3.0), atol=1e-6)


def test_noise_shapes_match_what_the_policy_asks_for():
    p = policy()
    want = noise_shapes(16, C, num_levels=2)
    noise = p.window_noise(0, origin=(0, 0, 0), n_hr=16, grid_hr=16)
    assert {k: tuple(v.shape) for k, v in noise.items()} == want


def test_mismatched_noise_shape_is_refused():
    p = policy()
    base, lr = inputs(16)
    dist = p.distribution(base, lr)
    bad = {s.name: torch.zeros(1, C, 3, 3, 3) for s in p.specs}
    with pytest.raises(ValueError, match="expected"):
        p.sample_actions(dist, bad)


def test_delta_is_the_bounded_projected_correction():
    from cosmo_sr.operators.multiscale import block_average

    p = policy()
    base, lr = inputs(16)
    res = p(base, lr, seed=3, grid_hr=16)
    delta = res["delta"]
    # alpha_disp = 0.25, so the LR-visible displacement part is scaled, not free.
    coarse_h = block_average(res["h"], SF)
    coarse_d = block_average(delta, SF)
    from cosmo_sr.reward.correction import bounded_action
    u = bounded_action(res["h"], p.correction.fine_scale)
    assert torch.allclose(coarse_d[:, 0:3], 0.25 * block_average(u, SF)[:, 0:3], atol=1e-6)
    assert float(delta[:, 0:3].abs().max()) <= 2 * 0.02 + 1e-6
    assert coarse_h.shape == coarse_d.shape


def test_zero_amplitude_gives_an_exactly_zero_edit():
    corr = CorrectionConfig(mode="block_leaky", scale_factor=SF, channels=C,
                            amplitude=0.0, scales=scales())
    p = MultiScaleGaussianPolicy(
        GaussianPolicyConfig(channels=C, scale_factor=SF, width=8, num_levels=2,
                             blocks_per_level=1, num_groups=4, use_checkpoint=False,
                             sigma_init={"coarse": 0.05, "middle": 0.15, "fine": 0.15}),
        corr).eval()
    res = p(*inputs(16), seed=0, grid_hr=16)
    assert torch.count_nonzero(res["delta"]) == 0
    # The action still exists and still has a log probability: a disabled edit is
    # not a degenerate distribution.
    assert torch.isfinite(res["log_prob"]).all()


def test_stats_report_saturation_and_sigma_pinning():
    p = policy()
    res = p(*inputs(16), seed=0, grid_hr=16)
    st = res["stats"]
    for s in p.specs:
        assert f"sigma_{s.name}_mean" in st
        assert st[f"sigma_{s.name}_at_bound_fraction"] == pytest.approx(0.0)
    assert "tanh_saturated_fraction" in st
    assert st["log_prob_sum"] == pytest.approx(float(res["log_prob"].mean()), rel=1e-5)


# --------------------------------------------------------------------------- #
# Global coordinate-aligned noise
# --------------------------------------------------------------------------- #
def test_noise_is_reproducible_from_the_seed():
    a = global_noise(11, "fine", 2, 32, (4, 8, 12), 8, block=8)
    b = global_noise(11, "fine", 2, 32, (4, 8, 12), 8, block=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, global_noise(12, "fine", 2, 32, (4, 8, 12), 8, block=8))


def test_overlapping_windows_agree_in_their_overlap():
    """The property that makes tiled sampling equal untiled sampling."""
    full = global_noise(5, "fine", 3, 32, (0, 0, 0), 32, block=8)
    win = global_noise(5, "fine", 3, 32, (8, 8, 8), 16, block=8)
    assert torch.equal(win, full[:, 8:24, 8:24, 8:24])


def test_windows_wrap_periodically():
    """A window running off the far edge continues at index 0, as the box does."""
    full = global_noise(6, "middle", 2, 16, (0, 0, 0), 16, block=8)
    win = global_noise(6, "middle", 2, 16, (12, 0, 0), 8, block=8)
    assert win.shape == (2, 8, 8, 8)
    assert torch.equal(win[:, 0:4], full[:, 12:16, 0:8, 0:8])
    assert torch.equal(win[:, 4:8], full[:, 0:4, 0:8, 0:8])


def test_a_window_that_straddles_a_block_boundary_is_still_consistent():
    full = global_noise(7, "fine", 1, 32, (0, 0, 0), 32, block=8)
    win = global_noise(7, "fine", 1, 32, (5, 13, 2), 12, block=8)
    assert torch.equal(win, full[:, 5:17, 13:25, 2:14])


def test_different_scales_use_independent_noise():
    a = global_noise(9, "fine", 1, 16, (0, 0, 0), 16, block=8)
    b = global_noise(9, "coarse", 1, 16, (0, 0, 0), 16, block=8)
    assert not torch.equal(a, b)


def test_noise_is_standard_normal():
    x = global_noise(3, "fine", 2, 64, (0, 0, 0), 64, block=32)
    assert abs(float(x.mean())) < 0.02
    assert abs(float(x.std()) - 1.0) < 0.02


def test_a_window_larger_than_its_grid_is_refused():
    with pytest.raises(ValueError, match="larger than"):
        global_noise(0, "fine", 1, 8, (0, 0, 0), 16, block=8)


def test_window_noise_refuses_a_misaligned_origin():
    """An origin off the coarse lattice would make two tiles disagree."""
    p = policy()
    with pytest.raises(ValueError, match="not aligned"):
        p.window_noise(0, origin=(2, 0, 0), n_hr=16, grid_hr=32)


# --------------------------------------------------------------------------- #
# Gradients
# --------------------------------------------------------------------------- #
def test_gradients_reach_both_head_families():
    p = policy()
    p.train()
    base, lr = inputs(16)
    res = p(base, lr, seed=2, grid_hr=16)
    a = {k: v.detach() for k, v in res["actions"].items()}     # detached replay action
    (-p.log_prob(p.distribution(base, lr), a).mean()).backward()
    for s in p.specs:
        assert p.mu_heads[s.name].weight.grad is not None
        assert float(p.logsig_heads[s.name].bias.grad.abs().max()) > 0.0


def test_no_gradient_flows_into_a_detached_replay_action():
    p = policy()
    base, lr = inputs(16)
    res = p(base, lr, seed=2, grid_hr=16)
    a = {k: v.detach().requires_grad_(True) for k, v in res["actions"].items()}
    p.log_prob(p.distribution(base, lr), a).sum().backward()
    # The action IS a leaf here, so it receives grad; what matters is that the
    # training path detaches it -- checked in test_gaussian_train.py. Here we
    # only pin that log_prob is differentiable in both arguments.
    assert all(v.grad is not None for v in a.values())


def test_gradient_checkpointing_does_not_change_the_value():
    torch.manual_seed(0)
    off = policy(use_checkpoint=False)
    on = policy(use_checkpoint=True)
    on.load_state_dict(off.state_dict())
    off.train(), on.train()
    base, lr = inputs(16)
    noise = off.window_noise(0, origin=(0, 0, 0), n_hr=16, grid_hr=16)
    a = off(base, lr, noise=noise)
    b = on(base, lr, noise=noise)
    assert torch.allclose(a["delta"], b["delta"], atol=1e-6)


# --------------------------------------------------------------------------- #
# Curriculum and provenance
# --------------------------------------------------------------------------- #
def test_rescale_exploration_scales_every_sigma():
    p = policy()
    base, lr = inputs(16)
    before = {k: float(v[1].mean()) for k, v in p.distribution(base, lr).items()}
    p.rescale_exploration(2.0)
    after = {k: float(v[1].mean()) for k, v in p.distribution(base, lr).items()}
    for k in before:
        assert after[k] == pytest.approx(2.0 * before[k], rel=1e-4)


def test_the_variance_floor_stops_exploration_collapsing():
    p = policy()
    p.rescale_exploration(1e-6, floor=0.01)
    base, lr = inputs(16)
    for _, sigma in p.distribution(base, lr).values():
        assert float(sigma.min()) >= 0.01 - 1e-9


def test_parameter_hash_is_provenance():
    """A replay row records this, so it has to be stable AND change-detecting."""
    p = policy()
    h = p.parameter_hash()
    assert p.parameter_hash() == h                  # stable across calls
    with torch.no_grad():
        p.mu_heads["fine"].bias.add_(1.0)
    assert p.parameter_hash() != h                  # any weight change shows up


def test_build_from_config_dicts():
    p = build_gaussian_policy(
        {"channels": C, "scale_factor": SF, "width": 8, "num_levels": 2,
         "blocks_per_level": 1, "num_groups": 4, "use_checkpoint": False,
         "sigma_init": {"coarse": 0.05, "middle": 0.15, "fine": 0.15}},
        {"mode": "block_null", "scales": scales().to_dict()},
    )
    assert p.correction.cfg.mode == "block_null"
    assert p.correction.cfg.scales.calibrated


# --------------------------------------------------------------------------- #
# Full-box tiling
# --------------------------------------------------------------------------- #
def test_tiled_sampling_matches_a_single_pass_sample():
    """Different tilings must agree, or a full-box number describes the tiling."""
    p = policy()
    with torch.no_grad():                 # a nontrivial, non-zero mean field
        p.mu_heads["fine"].weight.normal_(0.0, 0.05)
        p.mu_heads["middle"].weight.normal_(0.0, 0.05)
    ng = 32
    g = torch.Generator().manual_seed(4)
    base = torch.randn(C, ng, ng, ng, generator=g)
    lr = torch.randn(C, ng // SF, ng // SF, ng // SF, generator=g)

    a, _ = sample_policy_box(p, base, lr, seed=13,
                             spec=TileSpec(ng, core=16, margin=8, scale_factor=SF),
                             verify_margin=False)
    b, _ = sample_policy_box(p, base, lr, seed=13,
                             spec=TileSpec(ng, core=32, margin=0, scale_factor=SF),
                             verify_margin=False)
    assert np.abs(a - b).max() < 1e-5


def test_tiled_sampling_is_reproducible_and_seed_dependent():
    p = policy()
    ng = 32
    g = torch.Generator().manual_seed(5)
    base = torch.randn(C, ng, ng, ng, generator=g)
    lr = torch.randn(C, ng // SF, ng // SF, ng // SF, generator=g)
    spec = TileSpec(ng, core=16, margin=8, scale_factor=SF)

    a, st = sample_policy_box(p, base, lr, seed=1, spec=spec, verify_margin=False)
    b, _ = sample_policy_box(p, base, lr, seed=1, spec=spec, verify_margin=False)
    c, _ = sample_policy_box(p, base, lr, seed=2, spec=spec, verify_margin=False)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert st["n_tiles"] == 8


def test_returned_action_fields_tile_into_a_whole_box():
    p = policy()
    ng = 32
    base = torch.zeros(C, ng, ng, ng)
    lr = torch.zeros(C, ng // SF, ng // SF, ng // SF)
    _, st = sample_policy_box(p, base, lr, seed=1,
                              spec=TileSpec(ng, core=16, margin=8, scale_factor=SF),
                              verify_margin=False, return_actions=("coarse", "middle"))
    acts = st["actions"]
    assert acts["coarse"].shape == (C, ng // 4, ng // 4, ng // 4)
    assert acts["middle"].shape == (C, ng // 2, ng // 2, ng // 2)
    # mu = 0 at init, so a = sigma * eps: the assembled field must be the global
    # noise scaled, i.e. have the configured standard deviation.
    assert float(np.std(acts["coarse"])) == pytest.approx(0.05, rel=0.25)


def test_a_margin_smaller_than_the_receptive_field_is_refused():
    p = policy()
    with torch.no_grad():
        for s in p.specs:
            p.mu_heads[s.name].weight.normal_(0.0, 0.1)
    ng = 32
    with pytest.raises(ValueError, match="tile margin"):
        sample_policy_box(p, torch.zeros(C, ng, ng, ng),
                          torch.zeros(C, ng // SF, ng // SF, ng // SF), seed=0,
                          spec=TileSpec(ng, core=16, margin=SF, scale_factor=SF),
                          verify_margin=True)


def test_core_and_margin_must_respect_the_coarsest_stride():
    """A margin the coarse lattice does not divide would put two tiles' coarse
    noise windows at different offsets, and their overlap would disagree."""
    p = policy(scale_factor=2)          # coarsest stride 4, LR alignment only 2
    ng = 32
    with pytest.raises(ValueError, match="coarsest scale stride"):
        sample_policy_box(p, torch.zeros(C, ng, ng, ng),
                          torch.zeros(C, ng // 2, ng // 2, ng // 2), seed=0,
                          spec=TileSpec(ng, core=16, margin=2, scale_factor=2),
                          verify_margin=False)


def test_receptive_field_is_measured_and_positive():
    rf = policy_receptive_field(policy(), size=32)
    assert rf > 0


# --------------------------------------------------------------------------- #
# Analytic receptive field
# --------------------------------------------------------------------------- #
def _analytic(levels, blocks, sf=8):
    return analytic_receptive_field(GaussianPolicyConfig(
        num_levels=levels, blocks_per_level=blocks, kernel_size=3, scale_factor=sf,
        sigma_init={n: 0.05 if n == "coarse" else 0.15
                    for n in [s.name for s in scale_specs(levels)]}))


#: The measured table in configs/reward/residual_prior.yaml. That probe perturbs
#: all three inputs, so these values ALREADY include the block-upsampled LR path.
MEASURED = {(1, 1): 16, (1, 2): 24, (2, 1): 26, (2, 2): 41, (3, 1): 42, (3, 2): 65}


def test_the_recurrence_is_the_textbook_one():
    a = _analytic(2, 2)
    assert a["trunk_full"] == 84 and a["trunk_halfwidth"] == 41
    assert a["policy_halfwidth"] == 49          # + scale_factor for the LR block


def test_analytic_upper_bounds_every_measured_value():
    """The probe truncates at the float32 noise floor, so it can only under-report.

    An analytic value BELOW a measured one would mean the formula is missing a
    path -- or that some layer is not spatially local, in which case no margin
    makes tiling exact.
    """
    for (levels, blocks), rf in MEASURED.items():
        a = _analytic(levels, blocks)
        assert a["policy_halfwidth"] >= rf, (
            f"levels={levels} blocks={blocks}: analytic "
            f"{a['policy_halfwidth']} < measured {rf}; the formula is missing "
            f"a path, or a layer reduces over space")


def test_analytic_and_measured_agree_exactly_where_truncation_is_negligible():
    """The shallow configurations pin the formula: too few layers for the outer
    shells to decay below float32, so the bound is attained."""
    for key in ((1, 1), (1, 2)):
        assert _analytic(*key)["policy_halfwidth"] == MEASURED[key]


def test_the_analytic_measured_gap_grows_with_depth():
    """Which is why the analytic value is what a margin should be set from."""
    gaps = [_analytic(l, 2)["policy_halfwidth"] - MEASURED[(l, 2)] for l in (1, 2, 3)]
    assert gaps == sorted(gaps) and gaps[0] == 0 and gaps[-1] > 0


def test_the_lr_path_widens_the_reach_by_one_block():
    """One LR cell drives a whole scale_factor^3 block before the trunk sees it."""
    kw = dict(num_levels=2, blocks_per_level=2, scale_factor=8,
              sigma_init={"coarse": 0.05, "middle": 0.15, "fine": 0.15})
    with_lr = analytic_receptive_field(GaussianPolicyConfig(condition_on_lr=True, **kw))
    without = analytic_receptive_field(GaussianPolicyConfig(condition_on_lr=False, **kw))
    assert with_lr["policy_halfwidth"] == without["policy_halfwidth"] + 8
    assert without["policy_halfwidth"] == with_lr["trunk_halfwidth"]


def test_the_committed_config_has_a_sufficient_tile_margin():
    """Catches a margin edited below what the architecture needs, without a GPU."""
    import yaml
    from cosmo_sr.reward.sampling import tile_margin_for

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "configs" / "reward" /
         "gaussian_policy.yaml").read_text())
    a = analytic_receptive_field(GaussianPolicyConfig.from_dict(cfg["model"]))
    need = tile_margin_for(a["policy_halfwidth"], int(cfg["model"]["scale_factor"]))
    for key, value in (("sampling.tile_margin", cfg["sampling"]["tile_margin"]),
                       ("train.context_margin", cfg["train"]["context_margin"])):
        assert value >= need, (
            f"{key}={value} is below the analytic requirement {need} "
            f"(receptive-field half-width {a['policy_halfwidth']}); valid-core "
            f"tiling would leak tile padding into the written core"
        )
