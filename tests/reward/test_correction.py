"""The unified correction transform: projections, bounds, and independence.

These are algebraic identities, so they are tested as identities rather than as
tolerances-on-a-trained-model. If ``P_N + P_R = I`` stops holding, every
statement the projection oracle makes about a coarse allowance is meaningless.
"""
from __future__ import annotations

import json

import pytest
import torch

from cosmo_sr.operators.multiscale import block_average, block_upsample
from cosmo_sr.reward.base import compose
from cosmo_sr.reward.correction import (MODES, CorrectionConfig, CorrectionScales,
                                        CorrectionTransform, bounded_action,
                                        coarse_projection, leaky_transform,
                                        load_correction_scales, null_projection,
                                        remove_group_mean, require_calibrated_scales,
                                        saturation_fraction)

F = 4          # scale factor used throughout; small enough for CPU, > 1 blockwise
N = 16         # HR grid size (4 blocks per axis)
C = 6


def scales(**kw) -> CorrectionScales:
    d = dict(fine_disp=0.02, fine_vel=0.05, coarse_disp=0.004, coarse_vel=0.01,
             calibrated=True, source="test")
    d.update(kw)
    return CorrectionScales(**d)


def cfg(**kw) -> CorrectionConfig:
    d = dict(mode="block_leaky", scale_factor=F, channels=C, scales=scales())
    d.update(kw)
    return CorrectionConfig(**d)


def field(seed=0, n=N, c=C, dtype=torch.float32, batch=2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, c, n, n, n, generator=g, dtype=torch.float64).to(dtype)


# --------------------------------------------------------------------------- #
# Projection algebra
# --------------------------------------------------------------------------- #
def test_projections_are_complementary():
    """``P_N + P_R = I`` exactly."""
    u = field(1, dtype=torch.float64)
    assert torch.allclose(null_projection(u, F) + coarse_projection(u, F), u, atol=1e-12)


def test_null_projection_is_invisible_to_the_degrader():
    """``A P_N u = 0``: the whole point of the null component."""
    u = field(2, dtype=torch.float64)
    assert block_average(null_projection(u, F), F).abs().max() < 1e-12


def test_projections_are_idempotent_and_orthogonal():
    u = field(3, dtype=torch.float64)
    pn, pr = null_projection(u, F), coarse_projection(u, F)
    assert torch.allclose(null_projection(pn, F), pn, atol=1e-12)
    assert torch.allclose(coarse_projection(pr, F), pr, atol=1e-12)
    assert torch.allclose(coarse_projection(pn, F), torch.zeros_like(pn), atol=1e-12)
    # Orthogonal in the L2 inner product, so the two parts' energies add.
    assert abs(float((pn * pr).sum())) < 1e-9


def test_coarse_projection_recovers_the_block_mean():
    u = field(4, dtype=torch.float64)
    assert torch.allclose(block_average(coarse_projection(u, F), F),
                          block_average(u, F), atol=1e-12)


# --------------------------------------------------------------------------- #
# The alpha family
# --------------------------------------------------------------------------- #
def test_leaky_alpha_zero_is_the_null_projection():
    u = field(5, dtype=torch.float64)
    assert torch.equal(leaky_transform(u, 0.0, F), null_projection(u, F))


def test_leaky_alpha_one_is_the_identity():
    u = field(6, dtype=torch.float64)
    assert torch.allclose(leaky_transform(u, 1.0, F), u, atol=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.25, 0.5, 1.0])
def test_leaky_scales_only_the_coarse_component(alpha):
    u = field(7, dtype=torch.float64)
    out = leaky_transform(u, alpha, F)
    assert torch.allclose(block_average(out, F), alpha * block_average(u, F), atol=1e-12)
    assert torch.allclose(null_projection(out, F), null_projection(u, F), atol=1e-12)


def test_transform_modes_match_their_definitions():
    u_raw = field(8)
    t_none = CorrectionTransform(cfg(mode="none"))
    t_null = CorrectionTransform(cfg(mode="block_null"))
    t_leak0 = CorrectionTransform(cfg(mode="block_leaky", alpha_disp=0.0, alpha_vel=0.0))
    t_leak1 = CorrectionTransform(cfg(mode="block_leaky", alpha_disp=1.0, alpha_vel=1.0))

    d_none, _ = t_none(u_raw)
    d_null, _ = t_null(u_raw)
    d_leak0, _ = t_leak0(u_raw)
    d_leak1, _ = t_leak1(u_raw)

    # block_leaky(alpha=0) == block_null, and block_leaky(alpha=1) == unprojected.
    assert torch.allclose(d_leak0, d_null, atol=1e-6)
    assert torch.allclose(d_leak1, d_none, atol=1e-6)
    assert not torch.allclose(d_null, d_none, atol=1e-4)


# --------------------------------------------------------------------------- #
# split mode
# --------------------------------------------------------------------------- #
def test_split_mode_gives_A_delta_equals_c():
    """``A delta = c``, where ``c`` is the BOUNDED coarse action."""
    t = CorrectionTransform(cfg(mode="split"))
    h = field(9)
    c_raw = field(10, n=N // F)
    delta, _ = t(h, c_raw)
    c = bounded_action(c_raw, t.coarse_scale)
    assert torch.allclose(block_average(delta, F), c, atol=1e-6)


def test_split_without_a_coarse_head_is_the_null_projection():
    t = CorrectionTransform(cfg(mode="split"))
    h = field(11)
    delta, _ = t(h, None)
    assert block_average(delta, F).abs().max() < 1e-6


def test_split_rejects_a_wrongly_shaped_coarse_head():
    t = CorrectionTransform(cfg(mode="split"))
    with pytest.raises(ValueError, match="LR grid"):
        t(field(12), field(13, n=N))


# --------------------------------------------------------------------------- #
# Displacement / velocity independence
# --------------------------------------------------------------------------- #
def test_alpha_acts_independently_on_displacement_and_velocity():
    t = CorrectionTransform(cfg(mode="block_leaky", alpha_disp=0.0, alpha_vel=1.0))
    h = field(14)
    delta, _ = t(h)
    a = block_average(delta, F)
    # Displacement is hard-projected; velocity keeps its full coarse component.
    assert a[:, 0:3].abs().max() < 1e-6
    assert float(a[:, 3:6].abs().max()) > 1e-4


def test_amplitude_bounds_are_separate_per_group():
    sc = scales(fine_disp=1e-3, fine_vel=1.0)
    t = CorrectionTransform(cfg(mode="none", scales=sc))
    delta, _ = t(100.0 * field(15))          # deep in saturation, so |u| -> s
    assert float(delta[:, 0:3].abs().max()) <= 1e-3 + 1e-9
    assert float(delta[:, 3:6].abs().max()) > 0.5


def test_mean_removal_is_per_group_and_optional():
    h = field(16)
    plain, _ = CorrectionTransform(cfg(mode="none"))(h)
    demeaned, _ = CorrectionTransform(
        cfg(mode="none", remove_mean_disp=True, remove_mean_vel=False))(h)

    assert demeaned[:, 0:3].mean(dim=(-3, -2, -1)).abs().max() < 1e-6
    assert torch.equal(demeaned[:, 3:6], plain[:, 3:6])
    # The default leaves both means alone.
    assert float(plain[:, 0:3].mean(dim=(-3, -2, -1)).abs().max()) > 0.0


def test_remove_group_mean_leaves_other_channels_untouched():
    u = field(17)
    out = remove_group_mean(u, (0, 1, 2))
    assert torch.equal(out[:, 3:6], u[:, 3:6])


# --------------------------------------------------------------------------- #
# Bounds and saturation reporting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", MODES)
def test_delta_never_exceeds_the_bound(mode):
    """No mode can produce an edit larger than the calibrated amplitude.

    ``block_leaky`` and ``block_null`` are averages/differences of bounded
    quantities, so the bound survives projection; the test states that rather
    than assuming it.
    """
    sc = scales(fine_disp=0.02, fine_vel=0.05, coarse_disp=0.004, coarse_vel=0.01)
    t = CorrectionTransform(cfg(mode=mode, scales=sc))
    c_raw = field(19, n=N // F) * 50.0 if mode == "split" else None
    delta, st = t(50.0 * field(18), c_raw)
    # |P_N u| <= 2 max|u| and |A^dagger c| <= max|c|; the loose factor is
    # deliberate -- the claim is boundedness, not a tight constant.
    assert float(delta[:, 0:3].abs().max()) <= 2 * 0.02 + 0.004 + 1e-9
    assert float(delta[:, 3:6].abs().max()) <= 2 * 0.05 + 0.01 + 1e-9
    assert st["delta_absmax"] == pytest.approx(float(delta.abs().max()), rel=1e-6)


def test_saturation_fraction_is_measured_and_reported():
    t = CorrectionTransform(cfg(mode="none"))
    saturated, st_sat = t(torch.full((1, C, N, N, N), 10.0))
    _, st_lin = t(torch.zeros(1, C, N, N, N))

    assert st_sat["tanh_saturated_fraction"] == pytest.approx(1.0)
    assert st_lin["tanh_saturated_fraction"] == pytest.approx(0.0)
    for k in ("tanh_saturated_fraction_disp", "tanh_saturated_fraction_vel"):
        assert st_sat[k] == pytest.approx(1.0)


def test_saturation_fraction_helper_matches_threshold():
    h = torch.atanh(torch.tensor([[[[[0.5, 0.995]]]]]))
    assert saturation_fraction(h, None, 0.99) == pytest.approx(0.5)


def test_stats_report_the_coarse_share():
    t_null = CorrectionTransform(cfg(mode="block_null"))
    _, st = t_null(field(20))
    assert st["coarse_fraction"] == pytest.approx(0.0, abs=1e-6)

    t_open = CorrectionTransform(cfg(mode="none"))
    _, st_open = t_open(field(20))
    assert st_open["coarse_fraction"] > 0.0


# --------------------------------------------------------------------------- #
# dtype, geometry, gradients
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(dtype):
    t = CorrectionTransform(cfg(mode="block_leaky", alpha_disp=0.3, alpha_vel=0.3))
    delta, _ = t(field(21, dtype=dtype))
    assert delta.dtype == dtype


def test_periodic_shift_by_a_whole_block_commutes_with_the_transform():
    """The transform is block-local, so it commutes with a block-aligned roll.

    That is what makes it safe on a periodic box: the answer cannot depend on
    where the box's origin was placed, as long as the shift respects the LR
    lattice the blocks are defined on.
    """
    t = CorrectionTransform(cfg(mode="block_leaky", alpha_disp=0.4, alpha_vel=0.7))
    h = field(22, dtype=torch.float64)
    shift = (F, 2 * F, -F)
    dims = (-3, -2, -1)

    a, _ = t(torch.roll(h, shifts=shift, dims=dims))
    b, _ = t(h)
    assert torch.allclose(a, torch.roll(b, shifts=shift, dims=dims), atol=1e-12)


def test_non_block_aligned_shift_does_not_commute():
    """The complement of the test above: the block lattice is real, not cosmetic."""
    t = CorrectionTransform(cfg(mode="block_null"))
    h = field(23, dtype=torch.float64)
    dims = (-3, -2, -1)
    a, _ = t(torch.roll(h, shifts=(1, 0, 0), dims=dims))
    b, _ = t(h)
    assert not torch.allclose(a, torch.roll(b, shifts=(1, 0, 0), dims=dims), atol=1e-8)


def test_rejects_a_size_that_is_not_a_whole_number_of_blocks():
    t = CorrectionTransform(cfg(mode="block_null"))
    with pytest.raises(ValueError, match="multiple of scale_factor"):
        t(field(24, n=N + 1))


@pytest.mark.parametrize("mode", MODES)
def test_gradients_reach_the_raw_action(mode):
    t = CorrectionTransform(cfg(mode=mode))
    h = field(25).requires_grad_(True)
    c_raw = field(26, n=N // F).requires_grad_(True) if mode == "split" else None
    delta, _ = t(h, c_raw)
    delta.pow(2).sum().backward()

    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert float(h.grad.abs().max()) > 0.0
    if mode == "split":
        assert c_raw.grad is not None
        assert float(c_raw.grad.abs().max()) > 0.0


def test_stats_do_not_carry_gradients():
    t = CorrectionTransform(cfg(mode="block_leaky"))
    h = field(27).requires_grad_(True)
    delta, st = t(h)
    assert delta.requires_grad
    assert all(not torch.is_tensor(v) for v in st.values())


def test_block_null_gradient_is_orthogonal_to_the_coarse_direction():
    """No gradient can push a hard-projected policy toward the LR-visible part."""
    t = CorrectionTransform(cfg(mode="block_null"))
    h = torch.zeros(1, C, N, N, N, requires_grad=True)
    delta, _ = t(h)
    # A purely coarse target: its overlap with the achievable set is zero.
    target = coarse_projection(torch.ones(1, C, N, N, N), F)
    (delta * target).sum().backward()
    assert float(h.grad.abs().max()) < 1e-6


# --------------------------------------------------------------------------- #
# Zero-amplitude fallback
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", MODES)
def test_zero_amplitude_is_exactly_zero(mode):
    t = CorrectionTransform(cfg(mode=mode, amplitude=0.0))
    c_raw = field(29, n=N // F) if mode == "split" else None
    delta, st = t(field(28) * 1e6, c_raw)
    assert torch.count_nonzero(delta) == 0
    assert st["zero_amplitude"] is True
    assert st["delta_rms"] == 0.0


def test_zero_amplitude_composes_bit_exactly_to_the_base():
    """The end-to-end fallback: SR2 in, SR2 out, bit for bit."""
    base = field(30)
    t = CorrectionTransform(cfg(mode="block_leaky", amplitude=0.0))
    delta, _ = t(field(31))
    out = compose(base, delta, residual_scale=1.0)
    assert torch.equal(out, base)
    # And the other route to the same guarantee.
    assert compose(base, field(32), residual_scale=0.0) is base


def test_zero_amplitude_survives_a_nan_action():
    """A diverged network must not be able to poison the disabled path."""
    t = CorrectionTransform(cfg(mode="none", amplitude=0.0))
    delta, _ = t(torch.full((1, C, N, N, N), float("nan")))
    assert torch.isfinite(delta).all()
    assert torch.count_nonzero(delta) == 0


# --------------------------------------------------------------------------- #
# Config and calibrated scales
# --------------------------------------------------------------------------- #
def test_block_null_is_not_the_default():
    """It is a measured choice, not a prior; the oracle decides it."""
    assert CorrectionConfig().mode == "block_leaky"
    assert CorrectionConfig().mode != "block_null"
    assert CorrectionConfig().alpha_disp == 1.0
    assert CorrectionConfig().alpha_vel == 1.0


def test_uncalibrated_scales_are_refused_by_the_guard():
    assert require_calibrated_scales(scales()) is None
    why = require_calibrated_scales(CorrectionScales())
    assert why and "calibrated" in why


def test_scales_round_trip_through_json(tmp_path):
    p = tmp_path / "correction_scales.json"
    p.write_text(json.dumps({"scales": scales(boxes=("set0", "set1")).to_dict()}))
    got = load_correction_scales(p)
    assert got.calibrated and got.fine_disp == pytest.approx(0.02)
    assert got.boxes == ("set0", "set1")


def test_config_rejects_inconsistent_channel_groups():
    with pytest.raises(ValueError, match="overlap"):
        CorrectionConfig(disp_channels=(0, 1, 2), vel_channels=(2, 3, 4))
    with pytest.raises(ValueError, match="out of range"):
        CorrectionConfig(channels=6, vel_channels=(3, 4, 9))


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_config_rejects_alpha_outside_the_unit_interval(bad):
    with pytest.raises(ValueError, match="alpha"):
        CorrectionConfig(alpha_disp=bad)


def test_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown correction mode"):
        CorrectionConfig(mode="null_space")


def test_config_from_dict_reads_a_scales_file(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"scales": scales().to_dict()}))
    c = CorrectionConfig.from_dict({"mode": "split", "scale_factor": F,
                                    "scales_path": str(p)})
    assert c.mode == "split" and c.scales.calibrated
    assert c.uses_coarse_head


def test_alpha_and_scale_vectors_have_the_right_layout():
    c = cfg(alpha_disp=0.25, alpha_vel=0.75, amplitude=0.5)
    assert torch.equal(c.alpha_vector(),
                       torch.tensor([0.25, 0.25, 0.25, 0.75, 0.75, 0.75]))
    assert torch.allclose(c.fine_scale_vector(),
                          torch.tensor([0.01, 0.01, 0.01, 0.025, 0.025, 0.025]))
