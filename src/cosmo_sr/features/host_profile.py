"""The host's own density profile, as a background component for the solve.

Why this exists
---------------
``soft_membership`` decides membership by ABSOLUTE phase-space proximity to a
candidate centre, against a background prior that is flat in space
(``logbg = log(1 - claimed)``, no position in it). Two consequences, both
pointing the same way:

* the coarse prior is ``n_c / f^3`` -- proportional to the LOCAL COUNT -- so a
  candidate overlapping a dense cell gets a higher prior on every particle in
  it, and the host core is the densest cell in the region;
* a candidate sitting in the core meets its Sinkhorn mass target at tiny
  ``d^2`` and near-zero dual, while the same candidate at ``1.1 R_vir`` must
  gather from a thin field and drives ``alpha`` up.

So the solve has a systematic radial cost gradient pointing INWARD, and the
gather loss then trains the generator to build coherent bound objects wherever
the solve found it cheapest to name them. Nothing in the loss ever contradicts
that: ``virial``, ``bound``, ``r_rms``, ``sigma_v`` and ``d6`` are all computed
from within-set differences and are exactly translation-invariant, so 7 of the
8 terms CANNOT express where a subhalo sits, and the 8th (``centre``) explains
13.4% of the radial offset (``centre-offset-is-an-address``). The measured
result is ``radial-placement-is-inverted``: HR's median 1.13 R_vir against
0.30-0.35 for every tuned arm.

What this module does
---------------------
It supplies ``log rho_host(r)`` so the background becomes a real component:

.. math::
    \\mathrm{logit}_{ij} - \\mathrm{logbg}_i
      = \\log \\Pi_{ij} + \\alpha_j - \\frac{d^2_{ij}}{2\\lambda}
        - \\beta \\log \\frac{\\rho_\\mathrm{host}(r_i)}{\\bar\\rho}

which is the log OVERDENSITY of candidate ``j`` over the host at that
particle's location. A lump in the core must then be denser than the core to
win its particles; a lump at ``1.1 R_vir`` wins them cheaply. The radial
preference changes sign through SELECTION, and no per-object radial target is
stated anywhere -- which is the point, because one-to-one, rank and quantile
matching of HR's radial distribution have all already failed on held-out data
(``shell-levers-built``, ``shell-target-is-real-not-noise``): the per-host
radial distribution is real but is not reproducible by a shared local operator.

The evidence that a term of THIS class can work is already on disk: the only
lever that ever moved radial placement is ``isolation``
(``radial-placement-is-inverted``: 0.718, beating frozen on both axes), and
``isolation`` is ``bg_occ``, the soft count of non-member mass sitting on top
of the object. That is a crude one-radius density contrast, applied in the
loss. This moves the same idea into the selection, where it acts per particle
and with a profile instead of a single sphere.

Two sources, deliberately kept apart
------------------------------------
``"lr"``   -- an analytic NFW from the LR-only Rockstar catalog's ``(M_vir,
             R_vir)`` for the host. Inference-honest: the LR catalog exists for
             any box the generator can be run on. It is also INCOMPLETE --
             ``lr-host-channel-fidelity`` measured the LR catalog usable only
             for hosts above ~20k HR-equivalent particles, 57% of region
             subhalos -- so sets whose host misses the gate get ``valid=False``
             and are left on the flat background rather than given a guess.
``"self"`` -- an empirical profile, binned straight off the CANDIDATE field's
             own particles around the host centre. Needs no catalog at all, so
             it covers every host including the ones the LR gate drops. It is
             partly circular by construction: a core-heavy field reports a
             core-heavy profile and so partly forgives itself. It cannot
             forgive itself completely -- the profile is a smooth function of
             radius only, while a subhalo is a LOCALISED excess, so a lump
             still has to beat the azimuthal average at its radius -- but the
             circularity is real and is why the two are run as separate arms
             rather than one being chosen here.

``beta = 0`` reproduces the flat background exactly, which
``test_host_profile.py`` pins against the unmodified solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

__all__ = ["HostProfile", "nfw_log_density", "empirical_log_density",
           "build_host_profile"]

#: Radii below this fraction of ``R_vir`` (or of the smallest bin, for
#: ``"self"``) are clamped before the profile is evaluated. NFW diverges as
#: ``1/r``, which is physically right and is exactly the penalty the core
#: needs, but an unclamped evaluation at ``r = 0`` is an infinity in a softmax.
R_FLOOR_FRAC = 0.02


@dataclass(frozen=True)
class HostProfile:
    """``log rho_host`` for each set, evaluable at arbitrary radii.

    ``log_rho_nodes`` is tabulated on ``r_nodes`` (Mpc/h, ascending, per set)
    and evaluated by linear interpolation in ``(log r, log rho)``, with the end
    slopes continued outside the table. A table rather than a closure because
    both sources end up as one -- NFW is sampled onto the same nodes -- so the
    solve has exactly one evaluation path to test.
    """

    r_nodes: torch.Tensor        # (S, B) Mpc/h, ascending
    log_rho_nodes: torch.Tensor  # (S, B) log density, arbitrary units
    valid: torch.Tensor          # (S,) bool -- False where no profile was built
    source: str

    def to(self, device) -> "HostProfile":
        return HostProfile(r_nodes=self.r_nodes.to(device),
                           log_rho_nodes=self.log_rho_nodes.to(device),
                           valid=self.valid.to(device), source=self.source)

    def log_rho(self, r: torch.Tensor, setid: torch.Tensor) -> torch.Tensor:
        """``log rho_host(r)`` for entries whose set is ``setid``.

        Zero where the set has no profile, so an invalid set contributes
        nothing and its particles keep the flat background.
        """
        rn = self.r_nodes[setid]                       # (E, B)
        ln = self.log_rho_nodes[setid]
        lr = torch.log(r.clamp_min(1e-12))[:, None]
        lrn = torch.log(rn.clamp_min(1e-12))
        b = int(rn.shape[1])
        if b < 2:
            return torch.zeros_like(r)
        # The bin whose LEFT edge is the last node at or below `r`, clamped so
        # the two end bins provide the extrapolation slopes.
        j = torch.searchsorted(lrn.contiguous(), lr.contiguous())
        j = (j - 1).clamp(0, b - 2)
        j = j.squeeze(1)
        ar = torch.arange(r.shape[0], device=r.device)
        x0, x1 = lrn[ar, j], lrn[ar, j + 1]
        y0, y1 = ln[ar, j], ln[ar, j + 1]
        t = (lr.squeeze(1) - x0) / (x1 - x0).clamp_min(1e-12)
        out = y0 + t * (y1 - y0)
        return torch.where(self.valid[setid], out, torch.zeros_like(out))


def nfw_log_density(r: np.ndarray, *, rvir_mpc_h: float, mvir_msun_h: float,
                    conc: float) -> np.ndarray:
    """``log`` of the NFW density at ``r`` (Mpc/h), up to an additive constant.

    .. math:: \\rho(r) \\propto \\frac{M_\\mathrm{vir}}
              {r/r_s\\,(1 + r/r_s)^2},\\qquad r_s = R_\\mathrm{vir}/c

    The mass enters only as an overall factor, so it matters not at all for a
    solve over ONE host -- the dual absorbs any per-candidate constant -- and
    matters for a pool spanning several, where two hosts' profiles must be on
    a common scale. Keeping it costs one term and removes that footgun.
    """
    if not (rvir_mpc_h > 0.0):
        raise ValueError(f"rvir_mpc_h must be > 0, got {rvir_mpc_h}")
    if not (conc > 0.0):
        raise ValueError(f"conc must be > 0, got {conc}")
    rs = float(rvir_mpc_h) / float(conc)
    x = np.maximum(np.asarray(r, dtype=np.float64), R_FLOOR_FRAC * rvir_mpc_h) / rs
    log_m = np.log(max(float(mvir_msun_h), 1.0))
    return log_m - np.log(x) - 2.0 * np.log1p(x)


def empirical_log_density(
    pos_mpc_h: np.ndarray, centre_mpc_h: np.ndarray, *,
    boxsize_mpc_h: float, r_min_mpc_h: float, r_max_mpc_h: float,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(r_nodes, log_rho)`` from raw particle counts in log-spaced shells.

    The counts come from the CANDIDATE field, which is the honest thing to bin:
    the background a particle is competing against is the material that is
    actually there, not the material HR would have put there. Empty shells are
    floored at half a particle rather than dropped, so the table stays monotone
    in radius and the interpolation has no holes; a genuinely empty shell is
    then merely very underdense, which is what it is.
    """
    if not (r_max_mpc_h > r_min_mpc_h > 0.0):
        raise ValueError(f"need 0 < r_min < r_max, got {r_min_mpc_h}, {r_max_mpc_h}")
    nb = int(n_bins)
    if nb < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    box = float(boxsize_mpc_h)
    d = np.asarray(pos_mpc_h, dtype=np.float64) - np.asarray(
        centre_mpc_h, dtype=np.float64)[None, :]
    d -= box * np.round(d / box)
    r = np.sqrt((d * d).sum(axis=1))

    edges = np.exp(np.linspace(np.log(r_min_mpc_h), np.log(r_max_mpc_h), nb + 1))
    cnt = np.histogram(r, bins=edges)[0].astype(np.float64)
    vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho = np.maximum(cnt, 0.5) / np.maximum(vol, 1e-30)
    # Geometric bin centres: the table is interpolated in log r, so the node
    # belongs at the log midpoint, not the arithmetic one.
    nodes = np.sqrt(edges[1:] * edges[:-1])
    return nodes, np.log(rho)


def build_host_profile(
    sets, idx, *, cfg, pos_mpc_h: Optional[np.ndarray] = None,
    device=None, dtype=torch.float32,
) -> HostProfile:
    """The profile for each set in ``idx``, from ``cfg.soft_host_profile``.

    ``pos_mpc_h`` is the candidate field's particle table in Mpc/h and is
    required by ``"self"`` only. ``"lr"`` reads ``sets.host_rvir`` and
    ``sets.host_mvir``, which ``build_member_sets`` fills from the LR catalog
    and leaves at zero for a host the LR gate dropped -- those sets come back
    ``valid=False``.
    """
    src = str(cfg.soft_host_profile)
    if src not in ("lr", "self"):
        raise ValueError(f"soft_host_profile {src!r} is not 'lr' or 'self'")
    sel = [int(i) for i in idx]
    S = len(sel)
    nb = int(cfg.soft_host_bins)
    box = float(sets.boxsize_mpc_h)
    ch = sets.centre_host.detach().cpu().numpy().astype(np.float64)
    if float(np.abs(ch[sel]).sum()) == 0.0:
        raise ValueError(
            "soft_host_profile is on, but every centre_host is zero -- "
            "build_member_sets was called without host_pos, so the profile "
            "would be measured about the origin of the box. Host-preservation "
            "sets are built host_pos=None by design (a host IS the centre) and "
            "must run soft_host_profile='off'.")

    nodes = np.zeros((S, nb), dtype=np.float64)
    logr = np.zeros((S, nb), dtype=np.float64)
    valid = np.zeros(S, dtype=bool)

    if src == "lr":
        rv = np.asarray(getattr(sets, "host_rvir", np.zeros(sets.n_sets)),
                        dtype=np.float64)
        mv = np.asarray(getattr(sets, "host_mvir", np.zeros(sets.n_sets)),
                        dtype=np.float64)
        for k, s in enumerate(sel):
            if not (rv[s] > 0.0 and mv[s] > 0.0):
                continue   # host below the LR gate: no guess, flat background
            rmin = R_FLOOR_FRAC * float(rv[s])
            rmax = float(cfg.soft_host_r_max_rvir) * float(rv[s])
            nodes[k] = np.exp(np.linspace(np.log(rmin), np.log(rmax), nb))
            logr[k] = nfw_log_density(nodes[k], rvir_mpc_h=float(rv[s]),
                                      mvir_msun_h=float(mv[s]),
                                      conc=float(cfg.soft_host_conc))
            valid[k] = True
    else:
        if pos_mpc_h is None:
            raise ValueError("soft_host_profile='self' needs the particle "
                             "table; pos_mpc_h was None")
        p = np.asarray(pos_mpc_h, dtype=np.float64)
        # The radial range, in absolute Mpc/h so the arm needs no catalog at
        # all: from the softening to the sphere the sets actually occupy.
        scale = np.asarray(sets.centre_scale.detach().cpu().numpy(),
                           dtype=np.float64)
        # ONE histogram per distinct (host centre, range), not one per set.
        # The solve runs every training step and the estimator is O(N) over the
        # whole tile particle table; a task supervises one host, so without this
        # the identical profile is rebuilt `S` times a step and the arm pays
        # ~S x the cost of the thing it is measuring. Keyed on the rounded
        # centre and range so two sets that genuinely have different hosts still
        # get different profiles.
        cache: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
        for k, s in enumerate(sel):
            rmax = float(cfg.soft_host_r_max_rvir) * max(float(scale[s]), 0.15)
            rmin = max(R_FLOOR_FRAC * rmax, 1e-3)
            key = (round(float(ch[s][0]), 6), round(float(ch[s][1]), 6),
                   round(float(ch[s][2]), 6), round(rmin, 9), round(rmax, 9))
            if key not in cache:
                cache[key] = empirical_log_density(
                    p, ch[s], boxsize_mpc_h=box, r_min_mpc_h=rmin,
                    r_max_mpc_h=rmax, n_bins=nb)
            n, lg = cache[key]
            nodes[k], logr[k], valid[k] = n, lg, True

    return HostProfile(
        r_nodes=torch.as_tensor(nodes, dtype=dtype, device=device),
        log_rho_nodes=torch.as_tensor(logr, dtype=dtype, device=device),
        valid=torch.as_tensor(valid, device=device), source=src)


# --------------------------------------------------------------------------- #
# The LR host table
# --------------------------------------------------------------------------- #
def load_lr_host_table(path) -> dict:
    """The ``.npz`` written by ``scripts/features/build_lr_host_table.py``."""
    z = np.load(str(path))
    return {k: z[k] for k in z.files}


def lr_host_for(pos_mpc_h: np.ndarray, table: dict, *, boxsize_mpc_h: float,
                r_max_rvir: float = 2.0) -> Tuple[float, float, float]:
    """``(rvir_mpc_h, mvir_msun_h, r_over_rvir)`` of the LR host owning a point.

    The owner is the gated LR host minimising ``|x - x_host| / R_vir``, which
    is ``lr_host_channel.assign_hosts``' rule and is deliberately the SCALED
    distance: a point 3 Mpc from a cluster belongs to the cluster, not to the
    ``10^12`` group 2 Mpc away it is nowhere near virialised with.

    Returns zeros when nothing owns the point within ``r_max_rvir`` -- the LR
    catalog is incomplete by construction below its gate, and a supervised host
    with no LR owner must fall back to the flat background rather than be given
    the nearest cluster's profile, which would be a different object's density.
    """
    hp = np.asarray(table["host_pos_mpc_h"], dtype=np.float64)
    if hp.shape[0] == 0:
        return 0.0, 0.0, float("inf")
    rv = np.asarray(table["host_rvir_mpc_h"], dtype=np.float64)
    mv = np.asarray(table["host_mvir_msun_h"], dtype=np.float64)
    box = float(boxsize_mpc_h)
    d = np.asarray(pos_mpc_h, dtype=np.float64)[None, :] - hp
    d -= box * np.round(d / box)
    u = np.sqrt((d * d).sum(axis=1)) / np.maximum(rv, 1e-12)
    j = int(np.argmin(u))
    if not (u[j] <= float(r_max_rvir)):
        return 0.0, 0.0, float(u[j])
    return float(rv[j]), float(mv[j]), float(u[j])
