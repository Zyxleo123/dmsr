"""Why does SUBFIND refuse an object Rockstar calls a subhalo?

The measurement this serves
---------------------------
Inside the trained host, Rockstar reports 366-419 subhalos for the tuned arms
against HR's 369, and SUBFIND finds 22-40 against HR's 146. Resolved by particle
count the gap is not uniform:

===========  ============  ===========
bin          HR confirmed  arms
===========  ============  ===========
50-100p      7%            0-1%
100-200p     40%           1-4%
200-500p     86%           1-5%
500-2000p    100%          12-29%
2000+p       100%          86-100%
===========  ============  ===========

Below 200 particles HR fails too, so that end is SUBFIND's own resolution limit
and says nothing about the arms. Between 200 and 2000 particles HR confirms
86-100% while the arms confirm 1-29%: SUBFIND demonstrably *can* find objects of
that size, and refuses these. This module measures the properties that would
explain the refusal, so the answer is evidence rather than a story.

SUBFIND requires two things Rockstar does not, and each has a signature:

**A density peak delimited by a saddle.** An object that is diffuse, or that
sits so deep in the host that no saddle separates it, is never a candidate. The
signatures are ``r_half_over_r`` (compactness: ~0.3-0.4 for an NFW-like cusp,
0.79 for a uniform sphere -- a diffuse blob approaches the uniform value) and
``host_radius_over_rvir`` (placement).

**Survival of iterative unbinding.** A concentration of particles moving too
fast for its own mass unbinds to nothing. The signature is ``virial_ratio`` =
``sigma_v / v_circ``: a bound, virialised clump sits near 1, and a hot or
expanding one runs well above it.

Rockstar asks neither question in the same way -- it links in 6-D, so a
kinematically coherent group is a halo to it whether or not it is a bound
spatial peak. That is exactly the gap these numbers probe.

Units are the project's throughout: positions Mpc/h, velocities km/s, masses
Msun/h. ``G`` in those units is the plain physical value, because the ``h`` in
Msun/h and the ``h`` in Mpc/h cancel in ``G m / r``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

__all__ = ["G_MPC_KMS2_MSUN", "clump_properties", "summarize_split"]

#: Newton's constant in Mpc (km/s)^2 / Msun -- unchanged in h-units.
G_MPC_KMS2_MSUN = 4.30091e-9


def _periodic_delta(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return (d + 0.5 * box) % box - 0.5 * box


def clump_properties(pos: np.ndarray, vel: np.ndarray, centre: Sequence[float],
                     r: float, m_p: float, box: float, *,
                     host_centre: Optional[Sequence[float]] = None,
                     min_particles: int = 10) -> Dict[str, float]:
    """Compactness and boundness of whatever sits within ``r`` of ``centre``.

    Everything is computed from the particles actually there, never from the
    halo finder's own columns: the question is whether Rockstar's object exists
    as a bound compact peak, so taking its mass or radius on trust would assume
    the answer.

    Returned, with the reason each one is here:

    ``n_in``                particles within ``r``. Compared against Rockstar's
                            ``num_p``, a large excess means the aperture is
                            dominated by host material rather than by an object.
    ``sigma_v``             3-D rms velocity dispersion about the local mean,
                            km/s.
    ``v_circ``              ``sqrt(G M(<r) / r)`` for the mass actually present.
    ``virial_ratio``        ``sigma_v / v_circ``. Near 1 for a bound virialised
                            clump; well above 1 for something too hot to hold
                            together, which is what unbinding deletes.
    ``r_half_over_r``       radius containing half the enclosed particles, in
                            units of ``r``. ~0.3-0.4 for a centrally peaked
                            profile, 0.794 (= 2^-1/3) for a uniform sphere, so
                            values near 0.79 mean "no peak, just a patch".
    ``host_radius_over_rvir``  where the object sits in the host, when a host
                            centre is supplied.

    ``n_in < min_particles`` returns NaNs rather than numbers computed from a
    handful of particles.
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    c = np.asarray(centre, dtype=np.float64)
    r = float(r)

    d = _periodic_delta(pos, c[None, :], box)
    rad = np.sqrt((d ** 2).sum(axis=1))
    inside = rad <= r
    n_in = int(inside.sum())

    out: Dict[str, float] = {"n_in": n_in, "r_mpc_h": r}
    if host_centre is not None:
        dh = _periodic_delta(c[None, :], np.asarray(host_centre, float)[None, :],
                             box)[0]
        out["host_radius_mpc_h"] = float(np.sqrt((dh ** 2).sum()))

    if n_in < int(min_particles):
        out.update(sigma_v=float("nan"), v_circ=float("nan"),
                   virial_ratio=float("nan"), r_half_over_r=float("nan"),
                   usable=False,
                   why=f"only {n_in} particles within r = {r:.4f} Mpc/h")
        return out

    v = vel[inside]
    # Dispersion about the clump's OWN bulk motion: a subhalo orbiting at
    # 1000 km/s is not thereby unbound, and subtracting the host frame instead
    # of the object's would say it was.
    sigma_v = float(np.sqrt(((v - v.mean(axis=0)[None, :]) ** 2).sum(axis=1).mean()))
    mass = float(n_in) * float(m_p)
    v_circ = float(np.sqrt(G_MPC_KMS2_MSUN * mass / r))

    rs = np.sort(rad[inside])
    r_half = float(rs[max(n_in // 2 - 1, 0)])

    out.update(
        sigma_v=sigma_v, v_circ=v_circ,
        virial_ratio=float(sigma_v / v_circ) if v_circ > 0 else float("nan"),
        r_half_over_r=float(r_half / r) if r > 0 else float("nan"),
        mass_msun_h=mass, usable=True, why="")
    return out


#: A uniform sphere puts half its mass inside 2^(-1/3) of its radius. Any
#: measured value at or above this has no central concentration at all.
UNIFORM_R_HALF_OVER_R = 2.0 ** (-1.0 / 3.0)


def summarize_split(rows: Sequence[Dict], key: str = "confirmed") -> Dict:
    """Median of every numeric property, split by ``key``, plus the gap.

    Medians rather than means: these distributions have tails (an aperture that
    caught a neighbouring structure, an object at the resolution limit) and a
    single outlier should not decide what the population looks like.
    """
    # n_in_over_num_p belongs here and was missing from the first run: it turned
    # out to be THE discriminator (HR 3.8 vs the arms 86-137 at matched host
    # radius), and it was computed per object but never summarised, so the
    # printed table showed a blank column where the answer was.
    fields = ("n_in", "n_in_over_num_p", "sigma_v", "v_circ", "virial_ratio",
              "r_half_over_r", "host_radius_over_rvir", "num_p", "rvir_mpc_h")
    groups = {True: [r for r in rows if r.get(key)],
              False: [r for r in rows if not r.get(key)]}
    out: Dict = {"n_confirmed": len(groups[True]),
                 "n_unconfirmed": len(groups[False])}
    for f in fields:
        for flag, label in ((True, "confirmed"), (False, "unconfirmed")):
            vals = [float(r[f]) for r in groups[flag]
                    if f in r and r[f] is not None and np.isfinite(r[f])]
            out[f"{f}__{label}"] = float(np.median(vals)) if vals else None
        a, b = out[f"{f}__confirmed"], out[f"{f}__unconfirmed"]
        out[f"{f}__ratio"] = (a / b) if (a is not None and b) else None
    return out
