"""The projection argument, stated as tests.

The claim these figures rest on is that *not seeing* a subhalo in a whole-host
panel is uninformative, because the full Lagrangian column buries a real object
at exactly the rate a slab cut removes. So the tests are not about plotting:
they pin the three behaviours the conclusion needs.

*A real clump must decay towards the background as the slab opens.* A planted
overdensity in an otherwise uniform column has to score high in 3-D, high in a
thin slab, and far lower through the whole column -- otherwise "I can't see it"
would genuinely mean "it isn't there" and the whole diagnostic is pointless.

*A phantom must sit at 1 everywhere, including the thinnest slab.* This is the
discriminating case. At ``d = r`` there is no projection left to blame, so a
reported centre with no density behind it has nowhere to hide.

*The signal-to-noise must be the thing that tracks visibility*, not the ratio.
A 5% excess is invisible at snr 1 and obvious at snr 40, and the figures caption
themselves with it, so it has to be the Poisson number it claims to be.

A note on the test box. ``BOX = 20`` with ~40 particles per (Mpc/h)^3 is not the
real configuration -- the real one is 100 Mpc/h and 134M particles, where an
aperture the size of a subhalo holds ~1 background particle and no ratio is
measurable at all in a unit test. The numbers here are chosen so every count is
large enough to assert on. The *direction and ordering* of the effects are what
transfers, and the burial assertion is written as a ratio between depths rather
than an absolute, because the absolute depends on a column length the test box
does not have.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cosmo_sr.features.subhalo_zoom import (  # noqa: E402
    Window, contrast_2d, contrast_3d, contrast_host_shell, depth_ladder,
    periodic_delta, project_window, select_window, shell_counts,
)

BOX = 20.0
AXIS = 1                      # project along y, exactly as the host figures do
N_BG = 300_000                # ~37.5 particles per (Mpc/h)^3


def _uniform(n: int = N_BG, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.0, BOX, size=(n, 3))


def _clump(centre, n: int, sigma: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (np.asarray(centre)[None, :] + rng.normal(0.0, sigma, size=(n, 3))) % BOX


CENTRE = (10.0, 10.0, 10.0)


def test_periodic_delta_takes_the_short_way_round():
    d = periodic_delta(np.array([[19.0, 1.0, 10.0]]), np.array([[1.0, 19.0, 10.0]]),
                       BOX)
    assert np.allclose(d, [[-2.0, 2.0, 0.0]])


def test_window_selection_is_periodic_and_square():
    # A window centred on the box face must collect particles from both sides.
    pos = np.array([[0.5, 10.0, 10.0], [19.5, 10.0, 10.0], [3.0, 10.0, 10.0]])
    win = Window(centre=(0.0, 10.0, 10.0), half_xy=1.0, half_depth=-1.0, axis=AXIS)
    assert set(select_window(pos, win, BOX).tolist()) == {0, 1}


def test_projection_conserves_particles_and_orients_the_axes():
    pos = _uniform(5000, seed=3)
    win = Window(centre=CENTRE, half_xy=2.0, half_depth=2.0, axis=AXIS)
    counts = project_window(pos, win, BOX, bins=16)
    assert counts.sum() == select_window(pos, win, BOX).size

    # One particle offset along x only must land right of centre, on the row
    # axis; the column axis (z) must stay at the middle.
    one = np.array([[11.0, 10.0, 10.0]])
    c = project_window(one, win, BOX, bins=10)
    row, col = np.unravel_index(int(np.argmax(c)), c.shape)
    assert row > 4 and col in (4, 5)


def test_shell_counts_match_a_brute_force_radius():
    pos = _uniform(20_000, seed=5)
    r = np.sqrt((periodic_delta(pos, np.asarray(CENTRE)[None, :], BOX) ** 2)
                .sum(axis=1))
    got = shell_counts(pos, CENTRE, BOX, [0.0, 2.0, 4.0, 6.0])
    assert got[0] == int((r < 2.0).sum())
    assert got[1] == int(((r >= 2.0) & (r < 4.0)).sum())


def test_uniform_field_has_no_excess_in_three_dimensions():
    """~160 particles in the aperture, ~15k in the shell: enough to assert on."""
    c = contrast_3d(_uniform(seed=7), CENTRE, 1.0, BOX)
    assert c.ratio == pytest.approx(1.0, abs=0.15)
    assert abs(c.snr) < 4.0


def test_planted_clump_is_obvious_in_three_dimensions():
    pos = np.concatenate([_uniform(seed=9), _clump(CENTRE, 500, 0.10, seed=10)])
    c = contrast_3d(pos, CENTRE, 0.2, BOX)
    assert c.ratio > 50.0
    assert c.snr > 10.0


def test_the_column_buries_what_a_thin_slab_shows():
    """The figure's whole premise, on a clump that is real by construction."""
    pos = np.concatenate([_uniform(seed=11), _clump(CENTRE, 500, 0.10, seed=12)])
    r = 0.2
    thin, mid, full = depth_ladder(pos, CENTRE, r, BOX, [1.0, 10.0, -1.0], axis=AXIS)
    assert thin.half_depth == pytest.approx(r)
    assert full.half_depth == -1.0
    # Monotone burial: each wider slab dilutes the same object further.
    assert thin.ratio > mid.ratio > full.ratio
    # And the size of the effect is the point: the same real object is an order
    # of magnitude less prominent once the column is projected through it. In
    # the real box the column is 25 Mpc/h rather than this test's 20 and the
    # aperture is 4x smaller, so the true burial is far heavier than this.
    assert thin.ratio / full.ratio > 10.0


def test_a_phantom_stays_at_one_even_in_the_thinnest_slab():
    """The discriminating case: no clump planted, only uniform background."""
    pos = _uniform(seed=13)
    for c in depth_ladder(pos, CENTRE, 0.5, BOX, [1.0, 4.0, -1.0], axis=AXIS):
        assert c.ratio == pytest.approx(1.0, abs=0.4)
        assert abs(c.snr) < 4.0


def test_snr_is_the_poisson_excess_it_claims_to_be():
    pos = np.concatenate([_uniform(seed=15), _clump(CENTRE, 300, 0.08, seed=16)])
    c = contrast_2d(pos, CENTRE, 0.2, BOX, axis=AXIS, half_depth=0.2)
    assert c.n_excess == pytest.approx(c.n_in - c.n_expected, rel=1e-12)
    assert c.snr == pytest.approx(c.n_excess / np.sqrt(c.n_expected), rel=1e-12)
    assert c.ratio > 1.0 and c.snr > 0.0


def test_ratio_and_visibility_can_disagree():
    """Why both numbers are reported: they are not the same statement.

    This is the case the whole investigation turns on -- a real, small clump
    seen through a full column. As a *ratio* it is a tint no colour map will
    show; as an *excess over Poisson noise* it is unambiguous. A figure
    captioned only with the ratio would call this object absent, and a
    diagnostic that stopped at the ratio would agree with the eye and be wrong.
    """
    pos = np.concatenate([_uniform(seed=17), _clump(CENTRE, 60, 0.10, seed=18)])
    c = contrast_2d(pos, CENTRE, 0.2, BOX, axis=AXIS, half_depth=-1.0)
    assert c.ratio < 2.0        # a tint the eye cannot find ...
    assert c.snr > 4.0          # ... that is nonetheless a real detection


def test_empty_window_does_not_divide_by_zero():
    pos = np.array([[2.0, 2.0, 2.0]])
    win = Window(centre=CENTRE, half_xy=1.0, half_depth=1.0, axis=AXIS)
    assert project_window(pos, win, BOX, bins=8).sum() == 0
    c = contrast_3d(pos, CENTRE, 0.5, BOX)
    assert c.n_in == 0 and c.ratio == 0.0


# --------------------------------------------------------------------------- #
# The host-gradient bug the HR control caught on the first real run.
# --------------------------------------------------------------------------- #
def _host_with_subhalo(host_centre, sub_centre, *, n_host=400_000, n_sub=400,
                       r_host=4.0, sigma_sub=0.06, seed=21):
    """A centrally-concentrated host (rho ~ 1/r^2) with one real subhalo in it.

    The gradient is the point. Sampling it by inverting the enclosed mass
    (``M(<r) ~ r`` so ``r = R_host * u``) gives the profile exactly, with no
    rejection loop.

    The geometry is chosen to sit in the regime that broke the first run: with
    ``r_ap = 0.30`` and the subhalo at ``R = 1.2``, the background annulus
    ``3r..5r = 0.9..1.5`` **contains the host centre**, because ``3r <= R <=
    5r``. That is the configuration in which the local background is not a
    background at all. In the real catalog it catches any subhalo between 0.56
    and 0.93 Mpc/h of the host centre, at ``r_vir ~ 0.19``.

    The subhalo is deliberately weak -- 400 particles, so its mean density
    inside the aperture is comparable to the host's local density, which is what
    tidal truncation guarantees for a real one.
    """
    rng = np.random.default_rng(int(seed))
    r = r_host * rng.uniform(0.0, 1.0, n_host)
    u = rng.normal(size=(n_host, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    host = np.asarray(host_centre)[None, :] + u * r[:, None]
    sub = np.asarray(sub_centre)[None, :] + rng.normal(0.0, sigma_sub,
                                                       size=(n_sub, 3))
    return np.concatenate([host, sub]) % BOX


HOST_C = np.array([10.0, 10.0, 10.0])
SUB_C = HOST_C + np.array([1.2, 0.0, 0.0])
EMPTY_C = HOST_C + np.array([-1.2, 0.0, 0.0])     # same radius, nothing there
R_AP = 0.30


def test_local_annulus_cannot_see_a_real_subhalo_in_a_host():
    """A REAL subhalo scores ~1 when the background is taken locally.

    This is not hypothetical. On the first real run HR's own subhalos 168212,
    168577 and 168461 returned 1.2x, 0.4x and 0.6x, with signal-to-noise as
    negative as -105 -- the control saying the measurement, not the objects, was
    wrong. Here the same configuration gives ~1.1x at snr ~3: present, and
    indistinguishable from nothing.
    """
    pos = _host_with_subhalo(HOST_C, SUB_C)
    local = contrast_3d(pos, SUB_C, R_AP, BOX)
    assert local.ratio < 1.5
    assert local.snr < 10.0


def test_local_annulus_reports_a_DEFICIT_where_the_host_is_smooth():
    """And the same background turns smooth host material into a hole.

    The annulus contains the host centre, so it is denser than the patch it is
    supposed to be the background for, and the ratio goes below 1 with large
    negative significance. Any threshold on the local ratio would mark this
    position as evacuated; it is ordinary host.
    """
    pos = _host_with_subhalo(HOST_C, SUB_C)
    c = contrast_3d(pos, EMPTY_C, R_AP, BOX)
    assert c.ratio < 0.8
    assert c.snr < -5.0


def test_host_shell_recovers_the_subhalo_the_local_annulus_missed():
    """Same object, background taken from the host profile at its own radius."""
    pos = _host_with_subhalo(HOST_C, SUB_C)
    local = contrast_3d(pos, SUB_C, R_AP, BOX)
    shell = contrast_host_shell(pos, SUB_C, R_AP, HOST_C, BOX)
    assert shell.ratio > 1.5 * local.ratio
    assert shell.snr > 5.0 * max(local.snr, 1.0)


def test_host_shell_reports_nothing_where_there_is_nothing():
    """The other half: removing the gradient must not invent detections.

    A measurement that merely raised every ratio would pass the test above and
    be useless, because it would pass phantoms too.
    """
    pos = _host_with_subhalo(HOST_C, SUB_C)
    c = contrast_host_shell(pos, EMPTY_C, R_AP, HOST_C, BOX)
    assert c.ratio == pytest.approx(1.0, abs=0.2)
    assert abs(c.snr) < 4.0


def test_host_shell_excludes_the_object_from_its_own_background():
    """The background must not move when the object gets 20x heavier.

    If the object leaked into its own background the ratio would be damped by an
    amount growing with mass -- a bias that reads exactly like "big subhalos are
    less overdense", which is a conclusion someone might believe.
    """
    small = _host_with_subhalo(HOST_C, SUB_C, n_sub=400)
    big = _host_with_subhalo(HOST_C, SUB_C, n_sub=8_000)
    c_small = contrast_host_shell(small, SUB_C, R_AP, HOST_C, BOX)
    c_big = contrast_host_shell(big, SUB_C, R_AP, HOST_C, BOX)
    assert c_big.rho_bg == pytest.approx(c_small.rho_bg, rel=1e-6)
    assert c_big.ratio > 5.0 * c_small.ratio


def test_excluded_volume_matches_the_closed_form_when_it_can():
    """The Monte-Carlo volume is only trustworthy if it agrees where algebra does.

    For a sphere lying wholly inside the shell the answer is exactly
    ``1 - V_sphere / V_shell``; the estimator has to hit it, or every background
    density it normalises is quietly wrong by an unknown factor.
    """
    from cosmo_sr.features.subhalo_zoom import _shell_minus_sphere_fraction

    lo, hi, big_r, r_excl = 1.0, 3.0, 2.0, 0.5
    v_shell = 4.0 / 3.0 * np.pi * (hi ** 3 - lo ** 3)
    v_sphere = 4.0 / 3.0 * np.pi * r_excl ** 3
    got = _shell_minus_sphere_fraction(lo, hi, big_r, r_excl)
    assert got == pytest.approx(1.0 - v_sphere / v_shell, abs=0.002)
    assert _shell_minus_sphere_fraction(lo, hi, big_r, 0.0) == 1.0


def test_unmeasurable_aperture_is_flagged_not_reported_as_zero():
    """A 50-particle subhalo's core is smaller than the interparticle spacing.

    In the real catalog that object has ``r_vir = 0.022`` Mpc/h, so a 0.3 R_vir
    core is 0.0066 -- an eighth of a 512^3 cell. It holds no particles and the
    ratio came back ``0.00x``, which reads as "measured, and empty" when the
    truth is "not measurable". The flag is the difference between a null result
    and a missing one.
    """
    pos = _host_with_subhalo(HOST_C, SUB_C)
    c = contrast_host_shell(pos, SUB_C, 0.002, HOST_C, BOX)
    assert not c.usable
    assert np.isnan(c.ratio) and np.isnan(c.snr)
    assert "resolution" in c.why


def test_a_big_subhalo_near_the_host_centre_keeps_a_background():
    """The exclusion sphere must not swallow the shell it is cut from.

    The tuned arms put 0.7 Mpc/h subhalos at 0.33 host R_vir, where
    ``3 * r_vir`` exceeds the host-centric radius itself. Uncapped, the
    background volume went to zero and the ratio came back ``inf`` -- for the
    largest and most important objects in the sample.
    """
    big_sub = HOST_C + np.array([0.5, 0.0, 0.0])
    pos = _host_with_subhalo(HOST_C, big_sub, n_sub=20_000, sigma_sub=0.15)
    c = contrast_host_shell(pos, big_sub, 0.4, HOST_C, BOX)
    assert np.isfinite(c.ratio)
    assert c.n_bg > 0
