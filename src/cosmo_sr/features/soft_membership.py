"""Particle membership as a LATENT VARIABLE, not a fixed HR mask.

What this changes, in one sentence: :mod:`member_gather` supervises a set of
particles named by HR's own owner array, and this module replaces that hard
naming with a weight ``P_ij`` -- "how much of particle ``i`` belongs to
candidate ``j``" -- that is re-solved from the *current SR field* between
generator steps.

Why
---
The member-gather arm works (``gather-holdout-rockstar-gate``: tuned ``self``
reproduces HR's subhalo mass function on held-out set9, 20 -> 366 against HR's
369) and its remaining defects are placement ones -- radially over-concentrated
subhalos, host damage that is ~88% re-parenting. Every loss term is a statistic
of a particle collection that the loss is **not allowed to choose**: if SR2's
material for a given subhalo is not the material HR used, the loss charges the
generator for failing to bind particles that were never going to be there, and
rewards it for dragging the wrong ones in.

The relaxation keeps what made the hard arm work and drops only the part that
was never justified -- the exact identities. HR still supplies *which region the
object's material comes from* (the prior) and *what the object must look like*
(``sets.ref``, unchanged); the SR field supplies *which particles participate*.

The failure this must not repeat
--------------------------------
``gather-loss-never-touched-member-sets``: Gaussian windows failed because their
statistics described the surrounding host material rather than the object. A
prior that is uniform over a coarse cell is exactly such a window -- one LR cell
is ``100/64 = 1.5625`` Mpc/h a side, against a search radius of 0.15 -- so the
prior is **never used as the membership**. It is used as the support and as the
KL anchor; the membership itself is solved with a 6-D phase-space term at HR's
own ``r_rms`` and ``sigma_v``, which is the same quantity Rockstar groups on.
Two constraints keep that solve honest, and both are load-bearing:

* **a per-particle budget of one**, shared with a background/host category, so
  several candidates cannot spend the same convenient particles; and
* **a per-candidate mass target**, so a candidate cannot throw everything away
  and keep a tiny cold core that scores perfectly on every moment.

The mass constraint is imposed with a Sinkhorn dual (one scalar per candidate),
which is also what stops the phase-space window from being too narrow to fill:
if the window is starved the dual widens it until the mass is there.

Recovering the current method
-----------------------------
``soft_lambda -> inf`` freezes ``P`` at the prior, and ``soft_prior="hard"``
makes that prior the HR mask itself. That combination is the hard arm, and it is
the control the sweep must include -- ``soft_prior="coarse"`` at large lambda is
the *other* control, the one that says whether the coarse window alone is
already destructive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = ["SoftPools", "build_soft_pools", "solve_memberships",
           "weighted_set_statistics", "rows_to_coarse_cells"]


# --------------------------------------------------------------------------- #
# Lagrangian bookkeeping
# --------------------------------------------------------------------------- #
def rows_to_coarse_cells(rows: np.ndarray, tiles: Sequence[int], *,
                         ng_hr: int, tile_hr: int, factor: int) -> np.ndarray:
    """Coarse (LR) cell index for each particle-table row.

    Inverse of :func:`member_gather.tile_particles`'s ordering composed with the
    ``factor``-fold Lagrangian coarsening: row -> global HR site -> LR cell,
    flattened over the ``(ng_hr // factor)^3`` coarse grid.

    ``factor`` divides ``tile_hr`` in every configuration this project runs
    (8 into 64), so a coarse cell never straddles two tiles -- which is what
    lets :func:`_coarse_cell_rows` answer with rows of a single tile.
    """
    r = np.asarray(rows, dtype=np.int64)
    th, ng, f = int(tile_hr), int(ng_hr), int(factor)
    if th % f:
        raise ValueError(f"factor {f} does not divide tile_hr {th}; a coarse "
                         "cell would straddle tiles and the pool would be "
                         "built from rows that are not all trained")
    n_side = ng // th
    b, loc = r // (th ** 3), r % (th ** 3)
    li, lj, lk = loc // (th * th), (loc // th) % th, loc % th
    t = np.asarray(tiles, dtype=np.int64)[b]
    tx, ty, tz = t // (n_side * n_side), (t // n_side) % n_side, t % n_side
    gx, gy, gz = tx * th + li, ty * th + lj, tz * th + lk
    ngc = ng // f
    return ((gx // f) * ngc + (gy // f)) * ngc + (gz // f)


def _coarse_cell_rows(cells: np.ndarray, tile_of: np.ndarray, *,
                      ng_hr: int, tile_hr: int, factor: int) -> np.ndarray:
    """``(C, factor^3)`` table rows of every HR site in each coarse cell.

    ``tile_of`` maps a tile id to its batch slot in the particle table, ``-1``
    for an untrained tile. Cells whose tile is untrained are the caller's to
    drop: their rows come back as ``-1``.
    """
    c = np.asarray(cells, dtype=np.int64)
    th, ng, f = int(tile_hr), int(ng_hr), int(factor)
    ngc, n_side = ng // f, ng // th
    cx, cy, cz = c // (ngc * ngc), (c // ngc) % ngc, c % ngc
    ar = np.arange(f, dtype=np.int64)
    ox, oy, oz = np.meshgrid(ar, ar, ar, indexing="ij")
    gx = cx[:, None] * f + ox.reshape(-1)[None, :]
    gy = cy[:, None] * f + oy.reshape(-1)[None, :]
    gz = cz[:, None] * f + oz.reshape(-1)[None, :]
    tid = ((gx // th) * n_side + (gy // th)) * n_side + (gz // th)
    b = tile_of[tid]
    loc = (((gx % th) * th) + (gy % th)) * th + (gz % th)
    out = b * (th ** 3) + loc
    out[b < 0] = -1
    return out


def _dilate(cells: np.ndarray, ng_coarse: int) -> np.ndarray:
    """The 26 face/edge/corner neighbours of each cell, periodic, deduplicated."""
    c = np.asarray(cells, dtype=np.int64)
    n = int(ng_coarse)
    cx, cy, cz = c // (n * n), (c // n) % n, c % n
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                out.append((((cx + dx) % n) * n + (cy + dy) % n) * n
                           + (cz + dz) % n)
    return np.unique(np.concatenate(out))


# --------------------------------------------------------------------------- #
# The pools
# --------------------------------------------------------------------------- #
@dataclass
class SoftPools:
    """Per-set candidate pools: which particles MAY belong, and how much.

    One entry per (particle, set) pair. ``rows`` indexes the particle table
    exactly as ``MemberSets.live_rows`` does, so the two are interchangeable
    inputs to the same statistics.

    ``mult`` is the stratified-sampling multiplicity: a pool keeps every HR
    member of a cell but only a sample of the cell's non-members, and ``mult``
    is how many real particles each sampled one stands for. It is what makes
    the prior's total mass over a cell equal that cell's member count exactly,
    which in turn is what makes ``soft_lambda -> inf`` reproduce the hard arm's
    mass rather than a subsample of it.
    """

    rows: List[torch.Tensor]        # (S,) long, table rows in the pool
    logpi: List[torch.Tensor]       # (S,) log prior membership of each row
    mult: List[torch.Tensor]        # (S,) particles each pool row stands for
    is_member: List[torch.Tensor]   # (S,) bool, was this an HR member
    target_mass: torch.Tensor       # (S,) particles the candidate must hold
    factor: int
    prior: str

    def to(self, device) -> "SoftPools":
        import dataclasses
        return dataclasses.replace(
            self,
            rows=[t.to(device) for t in self.rows],
            logpi=[t.to(device) for t in self.logpi],
            mult=[t.to(device) for t in self.mult],
            is_member=[t.to(device) for t in self.is_member],
            target_mass=self.target_mass.to(device),
        )


def build_soft_pools(
    sets, tiles: Sequence[int], cfg, *,
    ng_hr: int = 512, tile_hr: int = 64, factor: int = 8,
    seed: int = 0, report: Optional[Dict] = None,
) -> SoftPools:
    """The coarse prior, built from an existing :class:`MemberSets`.

    Deliberately a **separate pass** over a finished pool rather than a branch
    inside ``build_member_sets``: the pool file costs a 537 MB owner-array load
    and a CSR inversion per box and is cached on disk, and this needs none of
    that -- only row arithmetic and the set's own ``live_rows``. So a soft arm
    reuses the same cached pool every hard arm ran on, and the two are
    guaranteed to supervise the same objects.

    Step 1 of the relaxation, verbatim: a set with ``n_c`` HR members among the
    ``factor^3`` sites of one coarse cell gives **every** particle of that cell
    a prior membership of ``n_c / factor^3``, the original members included and
    not otherwise privileged (``cfg.soft_prior == "coarse"``). ``"hard"`` keeps
    the HR mask as the prior instead and lets only ``soft_lambda`` open it --
    a strictly more conservative start, and the one whose ``lambda -> inf``
    limit is the arm already on disk.
    """
    rep: Dict = report if report is not None else {}
    f = int(factor)
    ngc = int(ng_hr) // f
    n_side = int(ng_hr) // int(tile_hr)
    tile_of = np.full(n_side ** 3, -1, dtype=np.int64)
    for b, t in enumerate(tiles):
        tile_of[int(t)] = b

    rng = np.random.default_rng(int(seed))
    ratio = float(cfg.soft_pool_ratio)
    dil = int(cfg.soft_cell_dilate)
    eps = float(cfg.soft_prior_eps)
    mode = str(cfg.soft_prior)
    if mode not in ("coarse", "hard"):
        raise ValueError(f"soft_prior {mode!r} is not 'coarse' or 'hard'")

    rows_l, logpi_l, mult_l, mem_l = [], [], [], []
    n_pool, n_cells = [], []
    for s in range(sets.n_sets):
        live = sets.live_rows[s].detach().cpu().numpy().astype(np.int64)
        cell = rows_to_coarse_cells(live, tiles, ng_hr=ng_hr, tile_hr=tile_hr,
                                    factor=f)
        uc, cnt = np.unique(cell, return_counts=True)
        cells, counts = uc, cnt.astype(np.float64)
        if dil:
            extra = np.setdiff1d(_dilate(uc, ngc), uc)
            if extra.size:
                cells = np.concatenate([uc, extra])
                counts = np.concatenate([counts, np.zeros(extra.size)])
        blk = _coarse_cell_rows(cells, tile_of, ng_hr=ng_hr, tile_hr=tile_hr,
                                factor=f)
        mset = set(live.tolist())

        pr, pp, pm, pb = [], [], [], []
        for ci in range(cells.size):
            all_rows = blk[ci]
            all_rows = all_rows[all_rows >= 0]
            if all_rows.size == 0:
                continue
            n_c = float(counts[ci])
            is_m = np.isin(all_rows, live, assume_unique=False)
            mem, non = all_rows[is_m], all_rows[~is_m]
            # The prior. `n_c / f^3` for both strata under "coarse" -- the
            # original members receive no special treatment, which is the whole
            # point of step 1 -- and the HR mask under "hard".
            if mode == "coarse":
                pi_m = pi_n = max(n_c / float(f ** 3), eps)
            else:
                pi_m, pi_n = 1.0 - eps, eps
            # Stratified: every member kept, the cell's non-members sampled.
            # A member is never dropped from the POOL (dropping it would take
            # the decision away from the solve and hand it to the sampler);
            # the solve is free to give it weight zero.
            k = int(np.clip(round(ratio * max(n_c, 1.0)), 1, non.size)) if non.size else 0
            if k:
                take = non if k >= non.size else rng.choice(non, size=k,
                                                            replace=False)
                pr.append(take)
                pp.append(np.full(take.size, np.log(pi_n)))
                pm.append(np.full(take.size, float(non.size) / float(take.size)))
                pb.append(np.zeros(take.size, dtype=bool))
            if mem.size:
                pr.append(mem)
                pp.append(np.full(mem.size, np.log(pi_m)))
                pm.append(np.ones(mem.size))
                pb.append(np.ones(mem.size, dtype=bool))
        if not pr:
            # Cannot happen for a set that passed `min_live_frac`, but a pool
            # of zero entries would make the solve's scatter silently empty and
            # the set would be supervised on its stragglers alone.
            raise RuntimeError(f"set {s} (halo {int(sets.halo_id[s])}) built an "
                               "empty soft pool")
        r = np.concatenate(pr)
        order = np.argsort(r, kind="stable")
        rows_l.append(torch.as_tensor(r[order], dtype=torch.long))
        logpi_l.append(torch.as_tensor(np.concatenate(pp)[order],
                                       dtype=torch.float32))
        mult_l.append(torch.as_tensor(np.concatenate(pm)[order],
                                      dtype=torch.float32))
        mem_l.append(torch.as_tensor(np.concatenate(pb)[order]))
        n_pool.append(int(r.size))
        n_cells.append(int(cells.size))

    # The mass each candidate must hold: its LIVE member count. Not `num_p` --
    # the out-of-tile members are carried separately as frozen stragglers at
    # weight 1, exactly as the hard arm carries them, so asking for `num_p`
    # here would ask the live pool to supply them twice.
    tgt = torch.as_tensor(np.asarray(sets.n_live, dtype=np.float32))
    rep["n_pool_total"] = int(np.sum(n_pool))
    rep["n_pool_median"] = float(np.median(n_pool)) if n_pool else 0.0
    rep["n_pool_max"] = int(np.max(n_pool)) if n_pool else 0
    rep["n_cells_median"] = float(np.median(n_cells)) if n_cells else 0.0
    rep["pool_over_live"] = float(np.sum(n_pool)
                                  / max(float(np.sum(sets.n_live)), 1.0))
    # The pair bill this pool commits a step to, against the hard arm's. The
    # estimator is O(pool^2) and the pool is a few times the member count, so
    # this is the number that decides whether the arm fits in the time limit.
    rep["sum_pool_squared"] = float(np.sum(np.asarray(n_pool, dtype=np.float64) ** 2))
    rep["soft_prior"] = mode
    rep["soft_pool_ratio"] = ratio
    rep["soft_cell_dilate"] = dil
    return SoftPools(rows=rows_l, logpi=logpi_l, mult=mult_l, is_member=mem_l,
                     target_mass=tgt, factor=f, prior=mode)


# --------------------------------------------------------------------------- #
# The membership solve
# --------------------------------------------------------------------------- #
def _scatter_lse(logit: torch.Tensor, group: torch.Tensor, n_groups: int,
                 extra: torch.Tensor) -> torch.Tensor:
    """``log(exp(extra_g) + sum_{e in g} exp(logit_e))`` per group, stably."""
    mx = torch.full((n_groups,), -float("inf"), device=logit.device,
                    dtype=logit.dtype)
    mx = mx.scatter_reduce(0, group, logit, reduce="amax", include_self=True)
    mx = torch.maximum(mx, extra)
    acc = torch.exp(extra - mx)
    acc = acc.index_add(0, group, torch.exp(logit - mx[group]))
    return mx + torch.log(acc.clamp_min(1e-30))


@torch.no_grad()
def solve_memberships(
    pos: torch.Tensor, vel: torch.Tensor, sets, pools: SoftPools,
    idx: Sequence[int], cfg, *,
    centres: Optional[torch.Tensor] = None,
    vbars: Optional[torch.Tensor] = None,
    record: Optional[Dict] = None,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """Solve ``P`` for the sets in ``idx`` on the CURRENT field. Detached.

    Returns one weight vector per entry of ``idx``, aligned with
    ``pools.rows[s]``, already multiplied by ``pools.mult`` -- so the caller
    treats it as "how many real particles this pool row contributes".

    The solve is a few EM sweeps of a phase-space mixture whose components are
    the candidates plus one background category, with the prior as the mixture
    weight and a Sinkhorn dual per candidate for the mass constraint:

    .. math::
        P_{ij} \\propto \\Pi_{ij}\\,
            e^{\\alpha_j}\\,e^{-\\tfrac{1}{2\\lambda}\\left(
            \\frac{|x_i-c_j|^2}{\\sigma_{r,j}^2}
            + \\frac{|v_i-\\bar v_j|^2}{\\sigma_{v,j}^2}\\right)}

    normalised per particle against a background term of ``Pi_{i,bg}``. That is
    exactly the stationary point of ``min_P [cost + lambda KL(P||Pi)]`` under a
    per-particle budget of one, so ``lambda`` is the KL weight of the boxed
    objective and nothing else.

    **The dispersions are HR's, not the candidate's** (``cfg.soft_sigma="hr"``).
    Using the candidate's own would let the window contract onto whatever is
    already cold and call that the object -- the "tiny cold core" failure the
    mass budget exists to block, re-entering through the metric instead of
    through the weights. ``"cand"`` is available for an ablation and is not the
    default.

    Everything here runs under ``no_grad``: the generator step sees ``P`` as a
    constant, which is the alternating scheme (a few membership updates, then a
    generator update) and is what keeps the gradient the familiar gathering and
    binding one.
    """
    from .member_gather import unwrap_about

    dev, dt = pos.device, pos.dtype
    S = len(idx)
    lam = float(cfg.soft_lambda)
    L = float(sets.boxsize_mpc_h)

    rows = torch.cat([pools.rows[s] for s in idx])
    setid = torch.cat([torch.full((int(pools.rows[s].numel()),), j,
                                  device=dev, dtype=torch.long)
                       for j, s in enumerate(idx)])
    logpi = torch.cat([pools.logpi[s] for s in idx]).to(dt)
    mult = torch.cat([pools.mult[s] for s in idx]).to(dt)

    # Each entry's particle, on ITS OWN set's unwrap branch. Two sets that share
    # a particle across a box face see it at different coordinates, which is
    # correct: each set's statistics live on its own branch.
    ctr_ref = sets.centre_ref[torch.as_tensor([int(i) for i in idx],
                                              device=dev)]
    x = unwrap_about(pos[rows], ctr_ref[setid], L)
    v = vel[rows]

    # The per-particle budget. Entries are grouped by the RAW table row, which
    # is the identity of the particle -- the unwrap branch above changes its
    # coordinates for one set's benefit, never which particle it is.
    part, pinv = torch.unique(rows, return_inverse=True)
    n_part = int(part.numel())

    sel = torch.as_tensor([int(i) for i in idx], device=dev)

    # The background prior: whatever the candidates have not claimed. Floored,
    # so a particle every candidate claims outright still has somewhere to go
    # and the softmax stays a proper distribution.
    bg_floor = float(cfg.soft_bg_floor)
    claimed = torch.zeros(n_part, device=dev, dtype=dt).index_add(
        0, pinv, torch.exp(logpi))
    over = (claimed > 1.0 - bg_floor)
    if bool(over.any()):
        scale = torch.where(over, (1.0 - bg_floor) / claimed.clamp_min(1e-12),
                            torch.ones_like(claimed))
        logpi = logpi + torch.log(scale[pinv])
        claimed = claimed * scale
    logbg = torch.log((1.0 - claimed).clamp_min(bg_floor))

    # --- the host as a real component, not a flat residual -------------------
    # `logbg` above has no position in it, so membership is decided by ABSOLUTE
    # proximity to a candidate centre and the densest place in the region is
    # the cheapest place to be a subhalo. Adding `beta * log rho_host(r)` to the
    # BACKGROUND -- equivalently, subtracting it from `logit - logbg` -- turns
    # the comparison into a log OVERDENSITY over the host, so a lump must beat
    # the material that would be at its radius anyway. See
    # `cosmo_sr.features.host_profile`. `beta = 0` leaves `logbg` untouched
    # bit-for-bit, which is what keeps every arm already on disk reproducible.
    beta = float(getattr(cfg, "soft_host_weight", 0.0))
    host_src = str(getattr(cfg, "soft_host_profile", "off"))
    if host_src != "off" and beta != 0.0:
        from .host_profile import build_host_profile
        prof = build_host_profile(
            sets, idx, cfg=cfg,
            pos_mpc_h=(pos.detach().cpu().numpy() if host_src == "self"
                       else None),
            device=dev, dtype=dt)
        # Radius per POOL ENTRY first: `x` is already unwrapped on its own
        # set's branch and `centre_host` is on that same branch, so the
        # difference is the true host-centric vector and never a box-face
        # artefact. Per-particle afterwards.
        # `parent` measures radius from each set's OWN host; `lr`/`self` keep
        # the task host they were built and gated with.
        hc = (sets.centre_parent if host_src == "parent"
              else sets.centre_host)[sel].to(dt)
        r_ent = torch.sqrt(((x - hc[setid]) ** 2).sum(1) + 1e-12)
        lrho_ent = prof.log_rho(r_ent, setid)
        # A particle in reach of two hosts competes against the DENSER one:
        # the question the background answers is "how much material would be
        # here anyway", and the answer is the largest claim, not the average.
        lrho = torch.full((n_part,), -float("inf"), device=dev, dtype=dt)
        lrho = lrho.scatter_reduce(0, pinv, lrho_ent, reduce="amax",
                                   include_self=True)
        lrho = torch.where(torch.isfinite(lrho), lrho, torch.zeros_like(lrho))
        # Centred on the pool's own mean, so `beta` changes the SHAPE of the
        # background in radius and not the overall weight of background against
        # candidates -- those are two different knobs and conflating them would
        # make a dose ladder unreadable.
        lrho = lrho - lrho.mean()
        # PLUS, not minus. The background must get MORE attractive where the
        # host is dense, so that `logit - logbg` is the log overdensity and a
        # core particle is harder to claim. The other sign makes the core
        # cheaper still, which is the defect, and it is invisible in every
        # check except a radial one -- hence the behavioural test.
        logbg = logbg + beta * lrho
        rep_host = {
            "soft_host_profile": host_src,
            "soft_host_weight": beta,
            "soft_host_valid_frac": float(prof.valid.to(dt).mean()),
            "soft_host_logrho_p10": float(torch.quantile(lrho, 0.10)),
            "soft_host_logrho_p90": float(torch.quantile(lrho, 0.90)),
        }
    else:
        rep_host = {"soft_host_profile": host_src, "soft_host_weight": beta}

    # --- the per-set RADIAL prior (cfg.soft_radial_weight) -------------------
    # A fixed cost per pool entry: how far the particle sits from its set's HR
    # radius, measured from the set's OWN parent (never the task host), in
    # units of `width * R_vir,parent`. Direction is free. Static over the solve
    # because the field is fixed while P is solved; the M-step then recentres
    # the window on the material at the right radius. Sets with no parent row
    # are left uncharged. gamma = 0 adds nothing, so the solve is unchanged.
    gam = float(getattr(cfg, "soft_radial_weight", 0.0))
    radcost = None
    rep_rad = {"soft_radial_weight": gam}
    if gam > 0.0:
        cp = getattr(sets, "centre_parent", None)
        prv = getattr(sets, "parent_rvir", None)
        if cp is None or prv is None:
            raise ValueError(
                "soft_radial_weight > 0 needs MemberSets.centre_parent and "
                "parent_rvir, which this pool predates -- rebuild it.")
        cpj = cp[sel].to(dt)
        rv = torch.as_tensor(np.asarray(prv, dtype=np.float64)[sel.cpu().numpy()],
                             device=dev, dtype=dt)
        ok = rv > 0.0
        r_t = torch.linalg.vector_norm(ctr_ref.to(dt) - cpj, dim=1)
        r_e = torch.sqrt(((x - cpj[setid]) ** 2).sum(1) + 1e-12)
        width = (float(cfg.soft_radial_width) * rv).clamp_min(1e-3)
        u = (r_e - r_t[setid]) / width[setid]
        # In the solve's dtype: `x` is promoted by the float64 set centres, and
        # a float64 cost would promote every logit after it.
        radcost = torch.where(ok[setid], 0.5 * gam * u * u,
                              torch.zeros_like(u)).to(dt)
        # Zero each set's cheapest entry. A per-set constant is exactly a shift
        # of that set's dual `alpha`, so the fixed point is unchanged -- but
        # left in, the dual has to climb it within `soft_mass_steps` capped
        # steps, and measured on the toy it could not (mass ratio 0.56 at
        # gamma=1, 0.23 at gamma=10).
        rmin = torch.full((S,), float("inf"), device=dev,
                          dtype=radcost.dtype).scatter_reduce(
            0, setid, radcost, reduce="amin", include_self=True)
        rmin = torch.where(torch.isfinite(rmin), rmin, torch.zeros_like(rmin))
        radcost = radcost - rmin[setid]
        rep_rad.update({
            "soft_radial_width": float(cfg.soft_radial_width),
            "soft_radial_valid_frac": float(ok.to(dt).mean()),
            "soft_radial_cost_p50": float(radcost.median()),
        })

    hr_r = sets.ref["r_rms"][sel].to(dt).clamp_min(1e-4)
    hr_v = sets.ref["sigma_v"][sel].to(dt).clamp_min(1.0)
    # The candidate's CURRENT dispersions, over its HR member rows. Needed
    # because the window is a selector, not a target: SR2's material for a set
    # starts ~6x wider and ~20x hotter than HR (measured on the toy, and the
    # same ratio the section 8.2 table reports for the real field), so a window
    # at HR's sigma has every candidate particle 400 sigma^2 out and selects
    # NOTHING. The dual would then have to carry the whole discrepancy, which
    # is a numerically hopeless way to say something simple.
    cand_r = torch.stack([
        (lambda q: torch.sqrt(((q - q.mean(0)) ** 2).sum(1).mean() + 1e-12))(
            unwrap_about(pos[sets.live_rows[s]], sets.centre_ref[s], L))
        for s in idx]).to(dt)
    cand_v = torch.stack([
        (lambda u: torch.sqrt((u.var(0, unbiased=False)).sum() + 1e-12))(
            vel[sets.live_rows[s]]) for s in idx]).to(dt)
    mode = str(cfg.soft_sigma)
    if mode == "hr":
        sr, sv = hr_r, hr_v
    elif mode == "cand":
        sr, sv = cand_r.clamp_min(1e-4), cand_v.clamp_min(1.0)
    else:
        # "adaptive", the default: never narrower than the object actually is
        # right now, so the window always selects a sensible fraction, and it
        # tightens toward HR's as the object does. The same continuation
        # argument as `member_gather._bound_temperature`, and the same reason:
        # a fixed relaxation that starts saturated has no gradient and no
        # content. It cannot contract onto a cold core, because it is bounded
        # BELOW by HR's dispersion, which is the failure `"cand"` invites.
        sr = torch.maximum(hr_r, cand_r)
        sv = torch.maximum(hr_v, cand_v)
    sr = sr * float(cfg.soft_sigma_scale)
    sv = sv * float(cfg.soft_sigma_scale)

    # The window's starting centre is the candidate's OWN current centroid over
    # the HR member rows -- not the prior-weighted one, which is the mean of a
    # 1.56 Mpc/h cell and is the Gaussian-window failure by another route.
    if centres is None:
        centres = torch.stack([
            unwrap_about(pos[sets.live_rows[s]], sets.centre_ref[s], L).mean(0)
            for s in idx])
    if vbars is None:
        vbars = torch.stack([vel[sets.live_rows[s]].mean(0) for s in idx])
    c, vb = centres.to(dt).clone(), vbars.to(dt).clone()
    alpha = torch.zeros(S, device=dev, dtype=dt)
    tgt = pools.target_mass[sel].to(dt).clamp_min(1.0)

    logmult = torch.log(mult.clamp_min(1e-30))

    def _logp() -> torch.Tensor:
        d2 = (((x - c[setid]) ** 2).sum(1) / (sr[setid] ** 2)
              + ((v - vb[setid]) ** 2).sum(1) / (sv[setid] ** 2))
        logit = logpi + alpha[setid] - 0.5 * d2 / lam
        if radcost is not None:
            logit = logit - radcost
        lse = _scatter_lse(logit, pinv, n_part, logbg)
        return logit - lse[pinv]

    def _weights() -> torch.Tensor:
        return mult * torch.exp(_logp())

    def _logmass(lp: torch.Tensor) -> torch.Tensor:
        """``log sum_i mult_i P_ij`` per candidate, in log space.

        Not ``log(sum(w))``. The dual's whole job is to rescue a window that
        currently selects almost nothing, and at the start it selects
        ``exp(-700)`` -- which is 0 in float64, so a linear-space sum hands the
        dual no information about HOW far off it is and the iteration crawls
        by a fixed step. In log space the same state reports -700 and one step
        lands on the constraint.
        """
        m = torch.full((S,), -float("inf"), device=dev, dtype=dt)
        z = logmult + lp
        m = m.scatter_reduce(0, setid, z, reduce="amax", include_self=True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        acc = torch.zeros(S, device=dev, dtype=dt).index_add(
            0, setid, torch.exp(z - m[setid]))
        return m + torch.log(acc.clamp_min(1e-300))

    eta = float(cfg.soft_mass_eta)
    step_cap = float(cfg.soft_mass_step_cap)
    lp = _logp()

    # Optional instrumentation (visualisation only): snapshot the per-entry
    # weight and the window state after each sweep, plus the static per-entry
    # arrays a plot needs. Guarded, so a training run pays nothing for it.
    _mem = torch.cat([pools.is_member[s] for s in idx]).to(dev) if record is not None else None
    if record is not None:
        record.update({
            "idx": [int(i) for i in idx],
            "halo_id": [int(sets.halo_id[int(i)]) for i in idx],
            "x": x.detach().cpu().numpy(),
            "v": v.detach().cpu().numpy(),
            "rows": rows.detach().cpu().numpy(),
            "setid": setid.detach().cpu().numpy(),
            "is_member": _mem.detach().cpu().numpy(),
            "mult": mult.detach().cpu().numpy(),
            "logpi": logpi.detach().cpu().numpy(),
            "target_mass": tgt.detach().cpu().numpy(),
            "sigma_r": sr.detach().cpu().numpy(),
            "sigma_v": sv.detach().cpu().numpy(),
            "lam": lam,
            "sweeps": [],
        })

        def _snap(tag: str) -> None:
            w_ = _weights()
            mass_ = torch.zeros(S, device=dev, dtype=dt).index_add(0, setid, w_)
            kept_ = torch.zeros(S, device=dev, dtype=dt).index_add(
                0, setid, torch.where(_mem, w_, torch.zeros_like(w_)))
            record["sweeps"].append({
                "tag": tag,
                "w": w_.detach().cpu().numpy(),
                "c": c.detach().cpu().numpy(),
                "vb": vb.detach().cpu().numpy(),
                "alpha": alpha.detach().cpu().numpy(),
                "mass": mass_.detach().cpu().numpy(),
                "hr_kept": kept_.detach().cpu().numpy(),
            })

        _snap("init")
    w = _weights()
    # The order inside the sweep is M-step THEN mass, not the reverse, so the
    # solve always ends on the constraint. Ending on the M-step leaves the
    # membership at whatever mass the last recentring happened to produce --
    # measured on the toy, a 9% shortfall that no amount of extra sweeps
    # removes, because every sweep reintroduced it.
    for it in range(max(int(cfg.soft_steps), 0)):
        if it:
            # --- M step: the window follows the material it selected ------
            tot = torch.zeros(S, device=dev, dtype=dt).index_add(0, setid, w)
            tot = tot.clamp_min(1e-6)
            c = (torch.zeros(S, 3, device=dev, dtype=dt)
                 .index_add(0, setid, w[:, None] * x) / tot[:, None])
            vb = (torch.zeros(S, 3, device=dev, dtype=dt)
                  .index_add(0, setid, w[:, None] * v) / tot[:, None])
            lp = _logp()
        # --- Sinkhorn on the mass constraint ------------------------------
        # An exact dual ascent would be `alpha += log(tgt) - logmass`, which is
        # the Sinkhorn row scaling; it is damped by `eta` and capped per step
        # because the per-particle budget makes the map non-separable (two
        # candidates competing for the same particle can otherwise trade
        # overshoots back and forth forever).
        for _m in range(max(int(cfg.soft_mass_steps), 0)):
            step = (torch.log(tgt) - _logmass(lp)).clamp(-step_cap, step_cap)
            alpha = alpha + eta * step
            lp = _logp()
        w = _weights()
        if record is not None:
            _snap(f"sweep{it + 1}")

    mass = torch.zeros(S, device=dev, dtype=dt).index_add(0, setid, w)
    # How much of the solved object is HR's own material. 1.0 is the hard arm;
    # a number near 0 means the relaxation has walked off the object entirely,
    # which the mass constraint permits and the Rockstar gate would be the
    # first to notice -- so it is reported every step, not measured afterwards.
    mem = torch.cat([pools.is_member[s] for s in idx]).to(dev)
    kept = torch.zeros(S, device=dev, dtype=dt).index_add(
        0, setid, torch.where(mem, w, torch.zeros_like(w)))
    w2 = torch.zeros(S, device=dev, dtype=dt).index_add(0, setid, w * w)
    diag = {
        "soft_mass_ratio": float((mass / tgt).median()),
        "soft_mass_ratio_min": float((mass / tgt).min()),
        "soft_hr_fraction": float((kept / mass.clamp_min(1e-6)).median()),
        "soft_n_eff_over_target": float(
            ((mass ** 2 / w2.clamp_min(1e-12)) / tgt).median()),
        "soft_lambda": lam,
    }
    diag.update(rep_host)
    diag.update(rep_rad)
    sizes = [int(pools.rows[s].numel()) for s in idx]
    return list(torch.split(w, sizes)), diag


# --------------------------------------------------------------------------- #
# Weighted statistics
# --------------------------------------------------------------------------- #
def weighted_set_statistics(
    pos: torch.Tensor, vel: torch.Tensor, w: torch.Tensor, *,
    particle_mass_msun_h: float, cfg,
    bound_scale: Optional[torch.Tensor] = None,
    bg_pos: Optional[torch.Tensor] = None,
    bg_vel: Optional[torch.Tensor] = None,
    pot_mass_factor: float = 1.0,
    iso_radius: float = 0.0,
):
    """:func:`member_gather.set_statistics_torch` with membership weights.

    Every reduction that was a mean over members becomes a ``w``-weighted mean,
    and the pair sum weights pair ``(i, k)`` by ``w_i w_k`` -- so a particle
    that half belongs binds half as much and is half as bound. Setting ``w`` to
    all ones reproduces the unweighted function exactly, which is what the unit
    test pins.

    ``w`` is a constant here (the solve is a separate, ``no_grad`` pass), so the
    gradient reaching the generator has the same shape it always had: move the
    particles, not the labels.
    """
    from .member_gather import SetTerms, specific_potential_torch

    m = float(particle_mass_msun_h)
    wsum = w.sum().clamp_min(1e-12)
    n = int(pos.shape[0])

    cbar = (w[:, None] * pos).sum(0) / wsum
    vbar = (w[:, None] * vel).sum(0) / wsum
    dx, dv = pos - cbar, vel - vbar
    r_rms = torch.sqrt((w * (dx * dx).sum(1)).sum() / wsum + 1e-12)
    ke = 0.5 * (dv * dv).sum(1)
    sigma_v = torch.sqrt(2.0 * (w * ke).sum() / wsum + 1e-12)

    phi = specific_potential_torch(
        pos, m * float(pot_mass_factor), softening_mpc_h=cfg.softening_mpc_h,
        kind=cfg.softening_kind, chunk=int(cfg.pot_chunk),
        max_elems=int(cfg.pot_max_elems), weights=w)
    # W = 0.5 m sum_i w_i phi_i and T = m sum_i w_i ke_i: the weights enter the
    # potential twice (once as the source in phi, once as the field point here)
    # and the kinetic energy once, exactly as the masses would.
    wpot = 0.5 * m * (w * phi).sum()
    t = m * (w * ke).sum()
    # In LOG space, not `2t / |W|`. A set the radial prior starves has tiny
    # weights and a tiny |W| (quadratic in w), and autograd's quotient rule
    # forms `t / W**2`, which overflows float32 to inf -- the potential's
    # backward then multiplies inf by its zero self-pair and returns NaN (job
    # 84238 went NaN inside 250 steps; anomaly mode named
    # `_SpecificPotentialBackward`, job 84372). The log form only ever divides
    # by W once. Capped at e^60 so a degenerate set cannot overflow exp().
    # The particle mass cancels in the ratio (2T/|W| = 4 sum w ke / |sum w phi|)
    # and is left OUT: kept in, it multiplied the 1/W gradient by 5.8e8 after
    # the division and still overflowed float32 for the lightest sets (job
    # 84387: 3 non-finite steps in 250 with the first log form).
    s_ke = (w * ke).sum()
    s_phi = (w * phi).sum()
    log_virial = (torch.log((4.0 * s_ke).clamp_min(1e-30))
                  - torch.log(s_phi.abs().clamp_min(1e-30))).clamp(max=60.0)
    virial = torch.exp(log_virial)

    energy = ke + phi
    if bound_scale is None:
        # The weighted twin of `member_gather._bound_temperature`: the width is
        # the set's own energy spread, and a particle the solve rejected is not
        # part of the set, so it must not set the scale either. Detached, for
        # the same reason as there.
        tau = (float(cfg.bound_tau) * (w * energy.abs()).sum()
               / wsum).detach().clamp_min(1e-12)
    else:
        tau = bound_scale.clamp_min(1e-12)
    bound_soft = (w * torch.sigmoid(-energy / tau)).sum() / wsum
    bound_hard = (w * (energy < 0).to(pos.dtype)).sum() / wsum

    d6 = None
    if bg_pos is not None and bg_pos.shape[0] > 1:
        bdx = bg_pos - bg_pos.mean(dim=0, keepdim=True)
        bdv = bg_vel - bg_vel.mean(dim=0, keepdim=True)
        sx = torch.sqrt((bdx * bdx).sum(dim=1).mean() + 1e-12)
        sv = torch.sqrt((bdv * bdv).sum(dim=1).mean() + 1e-12)
        d6 = torch.sqrt((r_rms / sx) ** 2 + (sigma_v / sv) ** 2 + 1e-12)

    bg_occ = None
    if bg_pos is not None and bg_pos.shape[0] > 0 and float(iso_radius) > 0.0:
        r_iso = float(iso_radius)
        cdet = cbar.detach()
        d = torch.sqrt(((bg_pos - cdet) ** 2).sum(dim=1) + 1e-12)
        width = max(float(cfg.iso_softness) * r_iso, 1e-9)
        bg_occ = torch.sigmoid((r_iso - d) / width).sum()

    return SetTerms(n=n, r_rms=r_rms, sigma_v=sigma_v, virial=virial,
                    bound_soft=bound_soft, bound_hard=bound_hard, d6=d6,
                    bg_occ=bg_occ)
