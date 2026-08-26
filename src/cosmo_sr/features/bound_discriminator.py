"""Which statistic separates a bound halo from a field that only matches its moments?

``docs/sr2_gather_finetune.md`` sections 5-6: the gather objective hit every
target it was given -- compact mass 1.06 of HR, velocity dispersion 1.02, bulk
offset 0.05 sigma -- and Rockstar recovered **0 of 43** supervised subhalos. The
ceiling run then pushed the *true HR tiles* through the identical splice harness
and recovered **42 of 43** (227 subhalos in ``R_vir`` against base's 11), so the
gate is sound and the failure belongs to the objective.

That pair is why this module exists. The loss constrained three numbers per
subhalo; the HR field and the tuned field **agree on all three** and disagree
completely at the halo finder. Everything separating them therefore lives in
statistics the loss never touched -- and those are enumerable, cheap, and need
no halo finder to evaluate.

The point is to *name the missing term* rather than guess it. Section 7.4
guesses the virial ratio; section 6 guesses "the position-velocity correlation
that makes a halo findable, scrambled rather than built". Both are testable here
against a verified positive and a verified negative in identical geometry.

The statistics, ordered by how close they sit to Rockstar's actual decision
--------------------------------------------------------------------------
Rockstar links particles in 6-D with an adaptive metric
``d^2 = |dx|^2/sigma_x^2 + |dv|^2/sigma_v^2`` taken from the enclosing group,
seeds halos at the deepest level of that hierarchy, and then **unbinds**:
particles with positive energy relative to the candidate are discarded and the
object survives only if enough remain. So:

``bound_frac``
    The unbinding test itself, computed on the set as an isolated system --
    the fraction of members with ``0.5|v - <v>|^2 + phi < 0``. This is the
    closest differentiable-in-principle relative of Rockstar's decision rule
    that exists, and nothing in the gather loss constrains it.
``virial_ratio``
    ``2T/|W|``, section 7.4's candidate. Unlike every statistic in the gather
    loss it is a function of the **pair** structure (``W`` is a sum over pairs),
    not a moment of the field, so its level set is far smaller.
``d6``
    The set's 6-D compactness in Rockstar's own metric, normalised by the
    *host's* ``sigma_x`` and ``sigma_v`` measured on the same field. Small means
    "a tight phase-space clump against this host's background", which is the
    linking criterion rather than a density statement.
``vr_corr`` and ``vr_mean``
    Section 6's hypothesis, in its two independent halves. ``vr_corr`` is the
    Pearson correlation of radial position with radial velocity -- a *gradient*,
    which catches a Hubble-like expansion and is blind to a set drifting outward
    at one speed. ``vr_mean`` is the net radial velocity in units of
    ``sigma_v``, which catches exactly that blind spot. A settled object sits
    near zero on both; either alone can be fooled, so both are reported.
``coldness``
    ``sigma_v / v_circ`` with ``v_circ = sqrt(G M_set / r_rms)`` -- a monopole
    reading of the same question ``virial_ratio`` asks exactly.

``r_rms`` and ``sigma_v`` are the **controls**, not findings. The gather loss
constrained a compact-mass statistic and the window dispersion, so these two
should come out similar between the HR and tuned fields. If they do, and a
statistic above separates them, the separation is attributable rather than
generic -- that contrast is the whole design, and a run where the controls also
separate has measured something other than what it set out to.

Conventions and one honest limit
--------------------------------
Positions are comoving Mpc/h, velocities peculiar km/s, masses Msun/h, so
``G = 4.30091e-9`` works directly and the ``h`` cancels in ``G m / r``. Member
sets are the **HR** subhalos' particle ids, identical across fields because SR2
and HR share the Lagrangian lattice -- there is no matching anywhere here.

``virial_ratio``, ``bound_frac`` and ``coldness`` all depend on the softening
used for ``phi``: a set whose particles the candidate field has piled on top of
each other is softening-limited, not physical. The absolute values are therefore
softening-dependent and are **not** to be read as physics. What is read is the
comparison *between fields on the same sets at the same softening*, which is
what the discrimination table reports. Everything in this module is a pure
function of arrays so ``tests/features/test_bound_discriminator.py`` can pin it;
the I/O lives in ``scripts/features/measure_bound_discriminator.py``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

import numpy as np

G_MPC_KMS2_PER_MSUN = 4.30091e-9

__all__ = [
    "G_MPC_KMS2_PER_MSUN",
    "SetStats",
    "mann_whitney_auc",
    "member_ids",
    "particles_at",
    "set_statistics",
    "specific_potential",
    "unwrap_periodic",
]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def unwrap_periodic(pos: np.ndarray, boxsize: float, *,
                    ref: Optional[np.ndarray] = None,
                    passes: int = 2) -> np.ndarray:
    """Put a periodic clump on one continuous branch.

    A subhalo may straddle the box face, and every statistic below is a moment
    about the set's own centroid, so an un-unwrapped set would report a radius
    of order the box. Each pass re-centres on the previous mean, which fixes the
    case where the first reference particle is itself an outlier.

    Valid only while the set's true extent is under half a box; here the largest
    is a few Mpc/h against ``L = 100``, so this is not a live constraint.
    """
    p = np.asarray(pos, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected (N, 3), got {p.shape}")
    L = float(boxsize)
    r = (np.asarray(ref, dtype=np.float64) if ref is not None
         else p[0].astype(np.float64))
    out = p
    for _ in range(max(1, int(passes))):
        out = r + ((p - r + 0.5 * L) % L) - 0.5 * L
        r = out.mean(axis=0)
    return out


# --------------------------------------------------------------------------- #
# Energetics
# --------------------------------------------------------------------------- #
def specific_potential(pos: np.ndarray, particle_mass_msun_h: float, *,
                       softening_mpc_h: float = 0.01,
                       chunk: int = 512) -> np.ndarray:
    """``phi_i = -G m sum_{j != i} 1 / max(r_ij, eps)`` in ``(km/s)^2``.

    Chunked over rows so the ``N x N`` distance block never materialises: the
    largest supervised set is ~3k particles, which is small, but the same code
    is what a host-sized set would need and 611k^2 is not an option.
    """
    x = np.asarray(pos, dtype=np.float64)
    n = int(x.shape[0])
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    eps = float(softening_mpc_h)
    gm = G_MPC_KMS2_PER_MSUN * float(particle_mass_msun_h)
    phi = np.empty(n, dtype=np.float64)
    for s in range(0, n, int(chunk)):
        e = min(s + int(chunk), n)
        d = x[s:e, None, :] - x[None, :, :]
        r = np.sqrt((d * d).sum(axis=-1))
        np.maximum(r, eps, out=r)
        inv = 1.0 / r
        inv[np.arange(e - s), np.arange(s, e)] = 0.0     # drop the self term
        phi[s:e] = -gm * inv.sum(axis=1)
    return phi


@dataclass
class SetStats:
    """One member set, evaluated on one field. Lengths Mpc/h, speeds km/s."""

    n: int
    r_rms: float          # control: size about the set's own centroid
    sigma_v: float        # control: internal dispersion about its own mean
    virial_ratio: float   # 2T/|W|
    bound_frac: float     # Rockstar's unbinding test, set treated as isolated
    vr_corr: float        # radial position vs radial velocity (gradient)
    vr_mean: float        # net radial velocity in units of sigma_v (drift)
    coldness: float       # sigma_v / v_circ
    d6: float             # 6-D compactness against the host's own sigmas

    def to_dict(self) -> Dict:
        return asdict(self)


def set_statistics(pos: np.ndarray, vel: np.ndarray, *,
                   particle_mass_msun_h: float,
                   boxsize_mpc_h: float,
                   softening_mpc_h: float = 0.01,
                   host_sigma_x: Optional[float] = None,
                   host_sigma_v: Optional[float] = None) -> SetStats:
    """Every statistic for one member set on one field.

    ``T = sum 0.5 m |v - <v>|^2`` and ``W = 0.5 m sum_i phi_i`` -- the half
    undoes the double count in ``sum_i phi_i``, so ``2T/|W| = 1`` is the virial
    equilibrium the ratio is read against.
    """
    x = unwrap_periodic(pos, boxsize_mpc_h)
    v = np.asarray(vel, dtype=np.float64)
    n = int(x.shape[0])
    m = float(particle_mass_msun_h)
    nan = float("nan")
    if n == 0:
        return SetStats(n=0, r_rms=nan, sigma_v=nan, virial_ratio=nan,
                        bound_frac=nan, vr_corr=nan, vr_mean=nan,
                        coldness=nan, d6=nan)

    dx = x - x.mean(axis=0)
    dv = v - v.mean(axis=0)
    r = np.linalg.norm(dx, axis=1)
    r_rms = float(np.sqrt((r ** 2).mean()))
    ke = 0.5 * (dv ** 2).sum(axis=1)                     # (km/s)^2 per unit mass
    sigma_v = float(np.sqrt(2.0 * ke.mean()))

    phi = specific_potential(x, m, softening_mpc_h=softening_mpc_h)
    w = 0.5 * m * float(phi.sum())
    t = m * float(ke.sum())
    virial = float(2.0 * t / abs(w)) if w != 0.0 else nan
    bound = float((ke + phi < 0.0).mean())

    if n > 1:
        rhat = dx / np.maximum(r, 1e-12)[:, None]
        vrad = (dv * rhat).sum(axis=1)
        vr_mean = float(vrad.mean() / sigma_v) if sigma_v > 0 else nan
        vr = (float(np.corrcoef(r, vrad)[0, 1])
              if n > 2 and r.std() > 0 and vrad.std() > 0 else nan)
    else:
        vr = vr_mean = nan

    vcirc = np.sqrt(G_MPC_KMS2_PER_MSUN * n * m / max(r_rms, 1e-12))
    cold = float(sigma_v / vcirc) if vcirc > 0 else nan

    d6 = nan
    if host_sigma_x and host_sigma_v:
        d6 = float(np.sqrt((r_rms / float(host_sigma_x)) ** 2
                           + (sigma_v / float(host_sigma_v)) ** 2))
    return SetStats(n=n, r_rms=r_rms, sigma_v=sigma_v, virial_ratio=virial,
                    bound_frac=bound, vr_corr=vr, vr_mean=vr_mean,
                    coldness=cold, d6=d6)


# --------------------------------------------------------------------------- #
# Field and catalog access
# --------------------------------------------------------------------------- #
def particles_at(field, ids: np.ndarray, *,
                 boxsize_kpc_h: float = 100000.0,
                 redshift: float = 0.0):
    """``(pos Mpc/h, vel km/s)`` for flat Lagrangian ``ids``, from a memmap.

    The same reconstruction as :func:`cosmo_sr.eval.particles.field_to_particles`
    -- ``x = (q + Psi) mod L`` on the half-cell-offset lattice -- restricted to
    the ids asked for, because the whole box is 134M particles and the sets here
    total ~28k. Reading a memmap by fancy index touches only those pages.
    """
    from ..data.preprocess_srs import disnorm, velnorm

    ng = int(field.shape[1])
    idx = np.asarray(ids, dtype=np.int64)
    iz = idx % ng
    iy = (idx // ng) % ng
    ix = idx // (ng * ng)
    cell = float(boxsize_kpc_h) / ng

    ch = np.asarray(field[:, ix, iy, iz], dtype=np.float64)      # (6, n)
    disp = disnorm(ch[0:3], z=redshift, undo=True)               # kpc/h
    vel = velnorm(ch[3:6], z=redshift, undo=True)                # km/s
    q = np.stack([(ix + 0.5) * cell, (iy + 0.5) * cell, (iz + 0.5) * cell])
    pos_kpc = (q + disp) % float(boxsize_kpc_h)
    return (np.ascontiguousarray(pos_kpc.T * 1e-3),
            np.ascontiguousarray(vel.T))


def member_ids(owner_path: str, halo_ids: Sequence[int], *,
               chunk: int = 1 << 24) -> Dict[int, np.ndarray]:
    """``{halo_id: particle ids}`` from one chunked pass over the owner array.

    ``owner[particle_id]`` is the deepest catalog object binding that particle
    (``-1`` when unbound), so a set is just the positions where it equals the
    id. One pass rather than one pass per halo: the array is 134M int32 and the
    43 supervised sets would otherwise be 43 scans.
    """
    want = np.asarray(sorted({int(h) for h in halo_ids}), dtype=np.int64)
    owner = np.load(owner_path, mmap_mode="r")
    parts: Dict[int, list] = {int(h): [] for h in want}
    if want.size == 0:
        return {}
    for s in range(0, int(owner.size), int(chunk)):
        blk = np.asarray(owner[s:s + int(chunk)], dtype=np.int64)
        loc = np.searchsorted(want, blk)
        np.clip(loc, 0, want.size - 1, out=loc)
        hit = np.flatnonzero(want[loc] == blk)
        if hit.size == 0:
            continue
        for h in np.unique(blk[hit]):
            sel = hit[blk[hit] == h]
            parts[int(h)].append(sel.astype(np.int64) + s)
    return {h: (np.concatenate(v) if v else np.empty(0, dtype=np.int64))
            for h, v in parts.items()}


# --------------------------------------------------------------------------- #
# Discrimination
# --------------------------------------------------------------------------- #
def mann_whitney_auc(a: Sequence[float], b: Sequence[float]) -> float:
    """``P(random a > random b) + 0.5 P(tie)``; 0.5 means no separation.

    Ties get average ranks, which matters here: ``bound_frac`` saturates at 0
    and 1, so a naive rank would manufacture separation out of a tied block.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    both = np.concatenate([x, y])
    order = np.argsort(both, kind="mergesort")
    srt = both[order]
    ranks = np.empty(both.size, dtype=np.float64)
    i = 0
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    ux = ranks[:x.size].sum() - x.size * (x.size + 1) / 2.0
    return float(ux / (x.size * y.size))
