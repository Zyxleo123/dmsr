"""Per-subhalo slab projections: is there a bump where the halo finder says?

The whole-host figures project the full ~25 Mpc/h Lagrangian column of the
host's tiles onto one (x, z) plane. A 200-particle subhalo is ~0.1 Mpc/h across,
so that column stacks ~250 subhalo-diameters of unrelated foreground and
background on top of it. The contrast the eye is offered is then

    (subhalo + column) / column

which for a real subhalo of a few hundred particles in a column of a few
thousand is a percent-level tint, not a visible knot -- *even when the subhalo
is perfectly real*. Nothing about "I cannot see it" distinguishes that case from
a halo finder reporting a phantom.

This module makes the distinction measurable. Around one reported centre it
computes the same excess three ways:

* **3-D**, ``rho(<r) / rho(3r..5r)`` -- no projection at all, the honest
  overdensity;
* **2-D at a depth**, the same ratio after projecting a slab of half-thickness
  ``d`` along the line of sight;
* **the depth ladder**, that 2-D ratio as ``d`` runs from ``~r`` out to the full
  column.

The ladder is the experiment: a real subhalo washed out by projection starts
high and decays towards 1 as ``d`` grows, while a phantom sits at 1 at every
depth including ``d = r``, where no projection is left to blame.

Alongside the ratio each measurement carries the number that actually predicts
whether a human sees the knot -- the excess over the local background in units
of its own Poisson noise::

    N_exc = n_in - rho_bg * A_in,    snr = N_exc / sqrt(rho_bg * A_in)

A ratio of 1.05 is invisible at any signal-to-noise; a ratio of 1.05 at
snr = 40 is a real object the colour map is hiding. Both numbers are reported
because they answer different halves of the question.

Everything here is pure array work on particle positions, so the same functions
serve the frozen field, the tuned field and HR, and the panels a figure draws
are binned from exactly the particles the statistics were measured on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Contrast",
    "Window",
    "contrast_2d",
    "contrast_3d",
    "contrast_host_shell",
    "depth_ladder",
    "periodic_delta",
    "project_window",
    "select_window",
    "shell_counts",
]

#: Background annulus / shell, in units of the object's own radius. It starts
#: outside any plausible tidal extent and stops before the next object.
BG_LO = 3.0
BG_HI = 5.0


def periodic_delta(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    """Minimum-image ``a - b`` in a periodic box of side ``box``."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return (d + 0.5 * box) % box - 0.5 * box


def _plane_axes(axis: int) -> Tuple[int, int]:
    """The two axes that survive a projection along ``axis``, in order."""
    return tuple(i for i in (0, 1, 2) if i != int(axis))  # type: ignore[return-value]


@dataclass(frozen=True)
class Window:
    """A square sky window with a line-of-sight slab, about one centre.

    ``half_depth <= 0`` means *no* depth cut: the full column, which is what the
    whole-host figures project.
    """

    centre: Tuple[float, float, float]
    half_xy: float
    half_depth: float
    axis: int = 1

    @property
    def plane_axes(self) -> Tuple[int, int]:
        return _plane_axes(self.axis)

    def extent(self) -> Tuple[float, float, float, float]:
        """``(lo_u, hi_u, lo_v, hi_v)`` as offsets from the centre, Mpc/h."""
        h = float(self.half_xy)
        return (-h, h, -h, h)


def select_window(pos: np.ndarray, win: Window, box: float) -> np.ndarray:
    """Indices of the particles inside ``win`` (periodic, square in the plane).

    The slab cut is applied first because it is the one that usually removes
    most of the particles, but the result is order-independent.
    """
    pos = np.asarray(pos, dtype=np.float64)
    centre = np.asarray(win.centre, dtype=np.float64)
    d = periodic_delta(pos, centre[None, :], box)
    keep = np.ones(pos.shape[0], dtype=bool)
    if win.half_depth > 0:
        keep &= np.abs(d[:, int(win.axis)]) <= float(win.half_depth)
    for ax in win.plane_axes:
        keep &= np.abs(d[:, ax]) <= float(win.half_xy)
    return np.flatnonzero(keep)


def project_window(pos: np.ndarray, win: Window, box: float,
                   bins: int) -> np.ndarray:
    """Particle counts of ``win`` on a ``bins x bins`` grid of the plane.

    Returns raw counts, not a density: the mean count per bin is a property of
    the window (side, depth, and which particles were supplied), and a figure
    that divides by it silently rescales panels that are not comparable. Divide
    downstream, with the divisor you meant.

    Row index runs over the *first* surviving axis and column index over the
    second, so ``imshow(counts.T, origin="lower")`` puts the first axis on the
    horizontal -- the convention the host figures already use.
    """
    pos = np.asarray(pos, dtype=np.float64)
    idx = select_window(pos, win, box)
    centre = np.asarray(win.centre, dtype=np.float64)
    au, av = win.plane_axes
    h = float(win.half_xy)
    if idx.size == 0:
        return np.zeros((int(bins), int(bins)), dtype=np.float64)
    d = periodic_delta(pos[idx], centre[None, :], box)
    counts, _, _ = np.histogram2d(
        d[:, au], d[:, av], bins=int(bins), range=[[-h, h], [-h, h]])
    return counts.astype(np.float64)


def shell_counts(pos: np.ndarray, centre: Sequence[float], box: float,
                 radii: Sequence[float], *, axis: Optional[int] = None,
                 half_depth: float = -1.0) -> np.ndarray:
    """Counts in the radial bins ``radii``; 3-D when ``axis is None``.

    With ``axis`` set the radius is the *projected* one and only particles
    inside the slab ``|d_axis| <= half_depth`` are counted (``half_depth <= 0``
    keeps the whole column).
    """
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape[0] == 0:
        return np.zeros(len(radii) - 1, dtype=np.int64)
    d = periodic_delta(pos, np.asarray(centre, dtype=np.float64)[None, :], box)
    if axis is None:
        r = np.sqrt((d ** 2).sum(axis=1))
    else:
        au, av = _plane_axes(int(axis))
        if half_depth > 0:
            keep = np.abs(d[:, int(axis)]) <= float(half_depth)
            d = d[keep]
            if d.shape[0] == 0:
                return np.zeros(len(radii) - 1, dtype=np.int64)
        r = np.sqrt(d[:, au] ** 2 + d[:, av] ** 2)
    hist, _ = np.histogram(r, bins=np.asarray(radii, dtype=np.float64))
    return hist.astype(np.int64)


@dataclass(frozen=True)
class Contrast:
    """One excess measurement: the ratio, and whether it is visible.

    ``kind`` is ``"3d"`` or ``"2d"``; ``half_depth`` is the slab half-thickness
    the 2-D measurement used (``-1`` = the full column), in Mpc/h.
    """

    kind: str
    r: float
    half_depth: float
    n_in: int
    n_bg: int
    rho_in: float
    rho_bg: float
    ratio: float
    n_expected: float
    n_excess: float
    snr: float
    #: False when the aperture or the background held too few particles to mean
    #: anything -- a 0.022 Mpc/h subhalo whose 0.3 R_vir core is smaller than the
    #: 0.195 Mpc/h interparticle spacing, say. ``why`` names which end failed.
    usable: bool = True
    why: str = ""

    def to_dict(self) -> Dict[str, float]:
        d = {k: getattr(self, k) for k in
             ("kind", "r", "half_depth", "n_in", "n_bg", "rho_in", "rho_bg",
              "ratio", "n_expected", "n_excess", "snr")}
        out = {k: (v if isinstance(v, str) else float(v)) for k, v in d.items()}
        out["usable"] = bool(self.usable)
        out["why"] = str(self.why)
        return out


def _with_usability(c: "Contrast", shell_frac: float = 1.0, *,
                    min_in: int = 10, min_bg: int = 50) -> "Contrast":
    """Mark a contrast unusable rather than letting it report 0 or inf."""
    why = ""
    if c.n_in < int(min_in):
        why = (f"aperture holds {c.n_in} particles (< {min_in}): r = {c.r:.4f} "
               "Mpc/h is at or below the resolution of this field")
    elif c.n_bg < int(min_bg):
        why = f"background holds {c.n_bg} particles (< {min_bg})"
    elif shell_frac < 0.1:
        why = (f"the exclusion sphere removes {100 * (1 - shell_frac):.0f}% of "
               "the background shell")
    if not why:
        return c
    return Contrast(kind=c.kind, r=c.r, half_depth=c.half_depth, n_in=c.n_in,
                    n_bg=c.n_bg, rho_in=c.rho_in, rho_bg=c.rho_bg,
                    ratio=float("nan"), n_expected=c.n_expected,
                    n_excess=c.n_excess, snr=float("nan"),
                    usable=False, why=why)


def _contrast(kind: str, r: float, half_depth: float, n_in: int, n_bg: int,
              vol_in: float, vol_bg: float) -> Contrast:
    """Assemble a :class:`Contrast` from two counts and their two volumes."""
    rho_in = n_in / vol_in if vol_in > 0 else 0.0
    rho_bg = n_bg / vol_bg if vol_bg > 0 else 0.0
    ratio = rho_in / rho_bg if rho_bg > 0 else float("inf") if n_in else 0.0
    n_exp = rho_bg * vol_in
    n_exc = n_in - n_exp
    # Poisson noise of the background *as realised in the aperture*. The
    # background estimate itself is measured from many more particles, so its
    # own error is subdominant and deliberately not folded in.
    snr = n_exc / np.sqrt(n_exp) if n_exp > 0 else float("inf") if n_in else 0.0
    return Contrast(kind=kind, r=float(r), half_depth=float(half_depth),
                    n_in=int(n_in), n_bg=int(n_bg), rho_in=float(rho_in),
                    rho_bg=float(rho_bg), ratio=float(ratio),
                    n_expected=float(n_exp), n_excess=float(n_exc),
                    snr=float(snr))


def contrast_3d(pos: np.ndarray, centre: Sequence[float], r: float, box: float,
                *, bg_lo: float = BG_LO, bg_hi: float = BG_HI) -> Contrast:
    """``rho(<r) / rho(bg_lo*r .. bg_hi*r)`` with no projection anywhere."""
    r = float(r)
    counts = shell_counts(pos, centre, box, [0.0, r, bg_lo * r, bg_hi * r])
    n_in = int(counts[0])
    n_bg = int(counts[2])
    vol_in = 4.0 / 3.0 * np.pi * r ** 3
    vol_bg = 4.0 / 3.0 * np.pi * ((bg_hi * r) ** 3 - (bg_lo * r) ** 3)
    return _contrast("3d", r, -1.0, n_in, n_bg, vol_in, vol_bg)




def _shell_minus_sphere_fraction(lo: float, hi: float, big_r: float,
                                 r_excl: float, *, seed: int = 0,
                                 n: int = 200_000) -> float:
    """Fraction of a spherical shell that lies outside an off-centre sphere.

    Monte Carlo rather than algebra: the sphere may lie wholly inside the shell,
    straddle either face, or contain part of the hole, and the closed forms for
    those cases are four different expressions with easy sign errors. The
    estimate is deterministic (fixed seed) so a rerun reproduces the number, and
    at 2e5 points its own error is ~0.2%, far below the Poisson error on the
    counts it normalises.

    Points are drawn uniformly **in volume** -- ``r = (lo^3 + u (hi^3 - lo^3))^(1/3)``
    -- not uniformly in radius, which would over-weight the inner face.
    """
    if r_excl <= 0.0:
        return 1.0
    rng = np.random.default_rng(int(seed))
    u = rng.random(int(n))
    rad = np.cbrt(lo ** 3 + u * (hi ** 3 - lo ** 3))
    v = rng.normal(size=(int(n), 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pts = v * rad[:, None]
    # The excluded sphere sits at distance big_r along an arbitrary axis; put it
    # on +z, which costs nothing because the shell is isotropic.
    d2 = pts[:, 0] ** 2 + pts[:, 1] ** 2 + (pts[:, 2] - big_r) ** 2
    return float((d2 > r_excl ** 2).mean())


def contrast_host_shell(pos: np.ndarray, centre: Sequence[float], r: float,
                        host_centre: Sequence[float], box: float, *,
                        dr_frac: float = 0.1, excl_mult: float = 3.0,
                        mc_points: int = 200_000, mc_seed: int = 0,
                        min_in: int = 10, min_bg: int = 50) -> Contrast:
    """Overdensity against the HOST's own profile at the same host-centric radius.

    Why this exists, and why :func:`contrast_3d` is not enough. That function
    takes its background from a shell ``3r..5r`` about the *subhalo*. For a
    subhalo sitting inside a cluster at host-centric radius ``R``, with
    ``r ~ 0.2`` and ``R ~ 0.5`` Mpc/h, that shell reaches a third of the way to
    the host centre, where the host's own density is markedly higher. The
    "background" is then the host's gradient, and a perfectly real subhalo
    scores **below 1** -- which is exactly what HR's own subhalos did when this
    diagnostic was first run (ids 168212, 168577, 168461: 1.2x, 0.4x, 0.6x with
    signal-to-noise as negative as -105).

    The fix is to compare the subhalo with the host at *its own radius*: the
    background is the spherical shell ``|R_i - R| <= dr`` about the **host**
    centre, which is the azimuthal average of the host profile there and so
    carries no gradient by construction. The subhalo's own neighbourhood --
    everything within ``excl_mult * r`` of it -- is cut out of that shell, counts
    and volume together, so an object cannot supply its own background.

    ``dr`` is ``dr_frac * R``, floored only at the aperture ``r`` so the shell
    cannot be degenerate. It is deliberately NOT floored at the exclusion radius:
    doing that made ``dr`` larger than ``R`` itself for a well-resolved subhalo,
    so the "shell at this radius" swallowed the host centre and the measurement
    read 2x on an empty patch of smooth host. The excluded volume is instead
    computed by Monte Carlo -- uniform points in the shell, rejected inside the
    exclusion sphere -- which is correct for any overlap, partial or total.

    Note what this measurement still is not: a subhalo's mean density inside its
    own ``R_vir`` is, by the definition of tidal truncation, comparable to the
    host's local density. Even with a clean background, ``r = R_vir`` gives
    order-unity for a real object -- pass a core aperture (a fraction of
    ``R_vir``) to ask whether there is a dense core in there.
    """
    pos = np.asarray(pos, dtype=np.float64)
    c = np.asarray(centre, dtype=np.float64)
    h = np.asarray(host_centre, dtype=np.float64)
    r = float(r)

    d_host_sub = periodic_delta(c[None, :], h[None, :], box)[0]
    big_r = float(np.sqrt((d_host_sub ** 2).sum()))
    # The exclusion sphere must not swallow the shell. For a big subhalo close
    # to the host centre -- 0.7 Mpc/h R_vir at R = 0.33, which the tuned arms
    # produce routinely -- ``excl_mult * r`` exceeds R itself, the background
    # volume goes to zero and the ratio comes back ``inf``. Capping at half the
    # host-centric radius keeps a background to measure against; the
    # ``usable`` flag below says when even that was not enough.
    r_excl = min(float(excl_mult) * r, 0.5 * big_r)
    dr = max(float(dr_frac) * big_r, r)

    # Aperture: the subhalo itself.
    d_sub = periodic_delta(pos, c[None, :], box)
    r_sub = np.sqrt((d_sub ** 2).sum(axis=1))
    n_in = int((r_sub <= r).sum())
    vol_in = 4.0 / 3.0 * np.pi * r ** 3

    # Background: the host shell at this radius, minus the subhalo's own patch.
    d_h = periodic_delta(pos, h[None, :], box)
    r_h = np.sqrt((d_h ** 2).sum(axis=1))
    lo, hi = max(big_r - dr, 0.0), big_r + dr
    in_shell = (r_h >= lo) & (r_h <= hi)
    n_bg = int((in_shell & (r_sub > r_excl)).sum())

    vol_shell = 4.0 / 3.0 * np.pi * (hi ** 3 - lo ** 3)
    frac = _shell_minus_sphere_fraction(lo, hi, big_r, r_excl,
                                        seed=mc_seed, n=mc_points)
    vol_bg = vol_shell * frac

    out = _contrast("host_shell", r, -1.0, n_in, n_bg, vol_in, max(vol_bg, 1e-12))
    # A ratio is only a measurement if both ends of it were measured. Without
    # this, an aperture below the resolution reads 0.00x and an evaporated
    # background reads inf -- two numbers that look like results and are not.
    return _with_usability(out, frac, min_in=min_in, min_bg=min_bg)


def contrast_2d(pos: np.ndarray, centre: Sequence[float], r: float, box: float,
                *, axis: int = 1, half_depth: float = -1.0,
                bg_lo: float = BG_LO, bg_hi: float = BG_HI) -> Contrast:
    """The same excess after projecting a slab of half-thickness ``half_depth``.

    Areal densities, so the slab thickness cancels out of the *ratio* -- what it
    changes is how much uncorrelated column is added to both numerator and
    denominator, which is exactly the effect under test.
    """
    r = float(r)
    counts = shell_counts(pos, centre, box, [0.0, r, bg_lo * r, bg_hi * r],
                          axis=int(axis), half_depth=float(half_depth))
    n_in = int(counts[0])
    n_bg = int(counts[2])
    area_in = np.pi * r ** 2
    area_bg = np.pi * r ** 2 * (bg_hi ** 2 - bg_lo ** 2)
    return _contrast("2d", r, half_depth, n_in, n_bg, area_in, area_bg)


def depth_ladder(pos: np.ndarray, centre: Sequence[float], r: float, box: float,
                 depths_in_r: Sequence[float], *, axis: int = 1,
                 bg_lo: float = BG_LO, bg_hi: float = BG_HI) -> List[Contrast]:
    """:func:`contrast_2d` at a series of slab depths, in units of ``r``.

    A non-positive entry means the full column and is passed through as such, so
    ``[1, 2, 5, -1]`` reads "one radius, two, five, everything".
    """
    out: List[Contrast] = []
    for d in depths_in_r:
        half = float(d) * float(r) if float(d) > 0 else -1.0
        out.append(contrast_2d(pos, centre, r, box, axis=axis, half_depth=half,
                               bg_lo=bg_lo, bg_hi=bg_hi))
    return out
