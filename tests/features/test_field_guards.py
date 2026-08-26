"""Pin the high-k hinge against the numpy spectra the rest of the line reports.

The guard and ``field_report``'s ``highk_power_ratio`` must be the same
quantity. If they drift apart, a run that behaved would be indistinguishable
from one that did not -- and this line has already paid once for a verdict that
read a statistic the objective was not constraining
(``docs/sr2_gather_finetune.md`` section 3.4).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmo_sr.features.cond_spread import hann_window, wavenumbers
from cosmo_sr.features.field_guards import (
    banded_highk_hinge, banded_power_ratio_torch, hann_window_torch,
    highk_hinge, highk_power_ratio_torch, wavenumbers_torch,
)

DX = 100.0 / 512.0          # the HR cell, Mpc/h -- overfit_host_mse.DX
N = 16                      # small enough to be fast, large enough for bins


def _field(seed: int, n: int = N, c: int = 6, b: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(b, c, n, n, n)).astype(np.float32)


def test_hann_window_matches_numpy():
    got = hann_window_torch(N, torch.device("cpu"), torch.float64).numpy()
    assert np.allclose(got, hann_window(N), atol=1e-12)


def test_wavenumbers_match_numpy():
    got = wavenumbers_torch(N, DX, torch.device("cpu"), torch.float64).numpy()
    assert np.allclose(got, wavenumbers(N, DX), rtol=1e-12, atol=1e-12)


def test_hann_window_is_not_the_periodic_one():
    """torch.hann_window's default differs; the guard must not silently use it."""
    ours = hann_window_torch(N, torch.device("cpu"), torch.float64)[:, 0, 0]
    periodic = torch.hann_window(N, periodic=True, dtype=torch.float64)
    assert not torch.allclose(ours / ours.max(), periodic / periodic.max())


def _numpy_ratio(a: np.ndarray, b: np.ndarray, k_split: float) -> float:
    """The ratio computed the way cond_spread does it, independently."""
    w = hann_window(a.shape[-1]).astype(np.float64)
    kmag = wavenumbers(a.shape[-1], DX)
    sel = kmag >= k_split

    def p(x):
        f = np.fft.fftn(x[:, 0:3].astype(np.float64) * w, axes=(-3, -2, -1))
        return (np.abs(f) ** 2).sum(axis=1)[:, sel].mean()

    return float(p(a) / p(b))


@pytest.mark.parametrize("k_split", [4.0, 8.0])
def test_ratio_matches_numpy(k_split):
    a, b = _field(0), _field(1)
    got = float(highk_power_ratio_torch(
        torch.from_numpy(a), torch.from_numpy(b), dx=DX, k_split=k_split))
    assert got == pytest.approx(_numpy_ratio(a, b, k_split), rel=1e-5)


def test_ratio_is_scale_invariant_in_the_units_of_both_fields():
    """Both sides in on-disk units or both in Mpc/h give the same ratio."""
    a, b = _field(2), _field(3)
    ta, tb = torch.from_numpy(a), torch.from_numpy(b)
    plain = float(highk_power_ratio_torch(ta, tb, dx=DX))
    scaled = float(highk_power_ratio_torch(ta * 3.7, tb * 3.7, dx=DX))
    assert plain == pytest.approx(scaled, rel=1e-5)


def test_identical_fields_give_ratio_one_and_zero_hinge():
    a = torch.from_numpy(_field(4))
    pen, ratio = highk_hinge(a.clone().requires_grad_(True), a, dx=DX)
    assert float(ratio.detach()) == pytest.approx(1.0, rel=1e-6)
    assert float(pen.detach()) == 0.0


def test_hinge_is_exactly_zero_with_zero_gradient_below_hr():
    """The property every hinged term in member_gather has, and for the reason:
    the guard must never charge a run for having *less* small-scale power."""
    ref = torch.from_numpy(_field(5))
    cand = (ref * 0.5).clone().requires_grad_(True)
    pen, ratio = highk_hinge(cand, ref, dx=DX)
    assert float(ratio.detach()) < 1.0
    assert float(pen.detach()) == 0.0
    pen.backward()
    assert cand.grad is not None
    assert float(cand.grad.abs().max()) == 0.0


def test_hinge_penalises_and_has_gradient_above_hr():
    ref = torch.from_numpy(_field(6))
    cand = (ref * 2.0).clone().requires_grad_(True)
    pen, ratio = highk_hinge(cand, ref, dx=DX)
    assert float(ratio.detach()) > 1.0
    assert float(pen.detach()) == pytest.approx((float(ratio.detach()) - 1.0) ** 2, rel=1e-6)
    pen.backward()
    assert float(cand.grad.abs().max()) > 0.0


def test_hinge_gradient_points_downhill_in_power():
    """One Adam-free step against the gradient must lower the ratio."""
    ref = torch.from_numpy(_field(7))
    cand = (ref * 2.0).clone().requires_grad_(True)
    pen, before = highk_hinge(cand, ref, dx=DX)
    pen.backward()
    with torch.no_grad():
        stepped = cand - 1e-3 * cand.grad
    _, after = highk_hinge(stepped, ref, dx=DX)
    assert float(after.detach()) < float(before.detach())


def test_reference_is_detached():
    """A shared-tensor reference must not receive gradient: HR is data."""
    ref = torch.from_numpy(_field(8)).requires_grad_(True)
    cand = torch.from_numpy(_field(9) * 3.0).requires_grad_(True)
    pen, _ = highk_hinge(cand, ref, dx=DX)
    pen.backward()
    assert ref.grad is None or float(ref.grad.abs().max()) == 0.0


def test_velocity_channels_are_selectable():
    a, b = _field(10), _field(11)
    a[:, 3:6] *= 4.0
    ta, tb = torch.from_numpy(a), torch.from_numpy(b)
    dis = float(highk_power_ratio_torch(ta, tb, dx=DX, channels=slice(0, 3)))
    vel = float(highk_power_ratio_torch(ta, tb, dx=DX, channels=slice(3, 6)))
    assert vel > 4.0 * dis


def test_inert_k_split_raises_rather_than_silently_passing():
    """A guard that selects no modes reports 'held' forever. Fail loudly."""
    a = torch.from_numpy(_field(12))
    nyquist = np.pi / DX
    with pytest.raises(ValueError, match="selects no modes"):
        highk_power_ratio_torch(a, a, dx=DX, k_split=10.0 * nyquist)


def test_shape_mismatch_raises():
    a = torch.from_numpy(_field(13))
    b = torch.from_numpy(_field(14, n=N // 2))
    with pytest.raises(ValueError, match="matching"):
        highk_power_ratio_torch(a, b, dx=DX)


def test_cache_returns_consistent_tensors_across_calls():
    k1 = wavenumbers_torch(N, DX, torch.device("cpu"))
    k2 = wavenumbers_torch(N, DX, torch.device("cpu"))
    assert k1 is k2
    assert torch.equal(k1, k2)


# --------------------------------------------------------------------------- #
# Band-resolved guard
# --------------------------------------------------------------------------- #
def test_one_band_reproduces_the_scalar_guard():
    """The banded form must be the same quantity, just resolved.

    A single band over the same mask is the original ``sel.mean()`` by
    construction; if it is not, the banded penalty is charging for something the
    reported ``highk_ratio`` does not measure, which is the exact confusion
    ``highk_hinge``'s docstring exists to prevent.
    """
    a, b = torch.from_numpy(_field(31)), torch.from_numpy(_field(32))
    ratio, _, counts = banded_power_ratio_torch(a, b, dx=DX, k_split=4.0,
                                                n_bins=1)
    want = highk_power_ratio_torch(a, b, dx=DX, k_split=4.0)
    assert float(counts[0]) > 0
    assert float(ratio[0]) == pytest.approx(float(want), rel=1e-5)


def test_bands_partition_the_masked_modes():
    """Every mode above the split lands in exactly one live band."""
    n = 32
    kmag = wavenumbers(n, DX)
    _, _, counts = banded_power_ratio_torch(
        torch.from_numpy(_field(33, n=n)), torch.from_numpy(_field(34, n=n)),
        dx=DX, k_split=4.0, n_bins=6)
    assert int(counts.sum()) == int((kmag >= 4.0).sum())


def test_a_localised_excess_is_compressed_by_the_scalar_average():
    """The whole point: an average cannot say WHERE.

    The scalar is a fair power-weighted mean -- measured spread of HR's per-bin
    share is only 2.3x -- so a large excess confined to one octave is diluted
    into a middling number, which is exactly how `all_blocks_self`'s +3.4x at
    the subhalo scale and 9x deficit at the grid scale averaged to 1.59.
    """
    n = 64
    rng = np.random.default_rng(7)
    b = rng.normal(size=(2, 6, n, n, n)).astype(np.float32)
    ref = torch.from_numpy(b)

    kmag = wavenumbers_torch(n, DX, "cpu")
    inner = (kmag >= 4.0) & (kmag < 8.0)
    fa = torch.fft.fftn(ref[:, 0:3].clone(), dim=(-3, -2, -1))
    fa[:, :, inner] *= 4.0                     # 16x power, subhalo scale only
    bumped = ref.clone()
    bumped[:, 0:3] = torch.fft.ifftn(fa, dim=(-3, -2, -1)).real

    scalar = float(highk_power_ratio_torch(bumped, ref, dx=DX, k_split=4.0))
    banded, ratio, centres = banded_highk_hinge(bumped, ref, dx=DX,
                                                k_split=4.0, n_bins=6)
    live = ~torch.isnan(ratio)
    # The scalar barely notices: the bumped modes are a few percent of its mean.
    assert scalar < 2.0
    # The banded ratios locate it -- the bins below 8 h/Mpc carry the excess.
    assert float(ratio[live][centres[live] < 8.0].max()) > 8.0
    assert float(ratio[live][centres[live] > 10.0].max()) == pytest.approx(1.0, abs=0.05)
    assert float(banded) > 0.0


def test_dead_zone_switches_the_term_off_only_inside_the_band():
    a = torch.from_numpy(_field(41))
    pen_at_par, ratio, _ = banded_highk_hinge(a, a, dx=DX, k_split=4.0,
                                              n_bins=4, tol=0.25)
    assert float(pen_at_par) == pytest.approx(0.0)
    live = ~torch.isnan(ratio)
    assert torch.allclose(ratio[live], torch.ones_like(ratio[live]), atol=1e-5)

    # 2x over HR in every band is outside a 1.25x dead zone and must be charged.
    scaled = a.clone()
    scaled[:, 0:3] *= float(np.sqrt(2.0))
    pen_over, _, _ = banded_highk_hinge(scaled, a, dx=DX, k_split=4.0,
                                        n_bins=4, tol=0.25)
    assert float(pen_over) > 0.0


def test_two_sided_charges_a_deficit_and_one_sided_does_not():
    """``all_blocks_nocentre`` took high-k to 0.026 and the hinge rated it perfect."""
    a = torch.from_numpy(_field(51))
    quiet = a.clone()
    quiet[:, 0:3] *= 0.1                        # 100x power deficit
    one, _, _ = banded_highk_hinge(quiet, a, dx=DX, k_split=4.0, n_bins=4)
    two, _, _ = banded_highk_hinge(quiet, a, dx=DX, k_split=4.0, n_bins=4,
                                   two_sided=True)
    assert float(one) == pytest.approx(0.0)
    assert float(two) > 1.0


def test_k_max_drops_the_corner_modes():
    """Above Nyquist a shell is only the cube's corners -- 48% of the guard's modes."""
    n = 64
    kmag = wavenumbers(n, DX)
    k_ny = float(np.pi / DX)
    a, b = torch.from_numpy(_field(61, n=n)), torch.from_numpy(_field(62, n=n))
    _, _, all_counts = banded_power_ratio_torch(a, b, dx=DX, k_split=4.0,
                                                n_bins=6)
    _, _, capped = banded_power_ratio_torch(a, b, dx=DX, k_split=4.0, n_bins=6,
                                            k_max=k_ny)
    assert int(all_counts.sum()) == int((kmag >= 4.0).sum())
    assert int(capped.sum()) == int(((kmag >= 4.0) & (kmag < k_ny * 1.001)).sum())
    assert int(capped.sum()) < 0.6 * int(all_counts.sum())


def test_gradient_flows_to_the_candidate_only():
    a = torch.from_numpy(_field(71)).requires_grad_(True)
    b = torch.from_numpy(_field(72)).requires_grad_(True)
    pen, _, _ = banded_highk_hinge(a, b, dx=DX, k_split=4.0, n_bins=4,
                                   tol=0.0, two_sided=True)
    pen.backward()
    assert a.grad is not None and float(a.grad.abs().sum()) > 0
    assert b.grad is None or float(b.grad.abs().sum()) == 0.0
