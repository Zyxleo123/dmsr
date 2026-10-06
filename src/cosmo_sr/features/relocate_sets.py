"""Choose each supervised group's particles from the SR field, not from HR.

What this changes, in one sentence: :func:`member_gather.build_member_sets`
names a group by HR's owner array and lets the loss find those exact particles
wherever SR2 put them; this module keeps the HR object's **targets** and
**size** but re-selects *which* particles carry it, from compact material that
already exists in the frozen SR field at a requested host-centric radius.

Why
---
``gather-holdout-rockstar-gate`` is a real win -- the tuned ``self`` arm
reproduces HR's subhalo mass function on held-out set9 -- and its residual
defects are placement ones (``radial-profile-is-flat-in-host-mass``,
``host-damage-is-mostly-reparenting``). ``docs/sr2_member_gather_training.md``
records the specific symptom this module is aimed at: the **supervised** centres
barely move while the detected catalog concentrates centrally. If SR2's material
for a given subhalo is not where HR's was, the objective spends itself dragging
distant particles together instead of binding material that is already there.

The relaxation is the hard-set one, deliberately. ``soft_membership`` made
membership a latent variable re-solved every step and failed its own control
(``soft-membership-relaxation``). Here membership is re-chosen **once, before
training, from the frozen field**, then frozen for the run: one selection, fully
inspectable on disk before a GPU-hour is spent, and the existing loss runs on it
unchanged.

The three arms
--------------
======  ==========================  ==========================
arm     particle groups             initial group radii
======  ==========================  ==========================
A       HR memberships              frozen-SR radii (``relocate="off"``)
B       new local SR groups         frozen-SR radii (``relocate="frozen"``)
C       new local SR groups         HR radii (``relocate="hr"``)
======  ==========================  ==========================

B measures the effect of replacing membership at the placement the frozen field
already has; C adds the placement change. Same targets, same directions, same
training setup -- the two differ in one scalar per set, the requested radius.

What is deliberately NOT recomputed
-----------------------------------
``sets.ref`` -- every HR target -- is carried through untouched. The replacement
particles form no subhalo in HR, so recomputing HR's statistics on their ids
would produce the statistics of a random blob and the run would be supervised to
reproduce it. The one exception is ``ref["bound_scale"]``, which is not a target
but a temperature: it is ``tau G m N / r_rms`` and ``N`` changes, so leaving it
would scale the unbinding surrogate by the wrong particle count.

The radius is measured from ``centre_host`` -- the HR host's catalog position on
this set's unwrap branch -- and normalised by the host's ``R_vir``. There is no
SR halo catalog at pool-build time, so ``c_h^SR`` and ``R_h^SR`` are taken as
``c_h^HR`` and ``R_h^HR``; every other radial quantity in this line
(``w_radius``, ``shell_radii``) already measures from that same point, and a
second definition of "where the host is" would make the arms unreadable against
the results already on disk.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = ["relocate_member_sets", "fibonacci_directions"]


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def fibonacci_directions(n: int, *, seed: int = 0) -> np.ndarray:
    """``(n, 3)`` near-uniform unit vectors, deterministic in ``seed``.

    A Fibonacci spiral rather than ``n`` random draws: at the counts this runs
    at (4-16 proposals) random directions clump, and a set whose proposals all
    landed on one side of the sphere would report "no material at this radius"
    when the failure was the sampler's. The per-set random rotation keeps two
    neighbouring sets from proposing the same points and colliding on the same
    particles in the same order.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"fibonacci_directions needs n >= 1, got {n}")
    i = np.arange(n, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    d = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    if seed:
        rng = np.random.default_rng(int(seed))
        # A uniformly random rotation (QR of a Gaussian matrix, sign-fixed so
        # the map is a rotation and not a reflection composed with one).
        q, rr = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q * np.sign(np.diag(rr))[None, :]
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        d = d @ q.T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _unwrap_np(p: np.ndarray, ref: np.ndarray, L: float) -> np.ndarray:
    """:func:`member_gather.unwrap_about` for numpy arrays."""
    return ref + np.remainder(p - ref + 0.5 * L, L) - 0.5 * L


def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else fallback


def _spread(p: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """``(centroid, r_rms, sigma_v)`` of an already-unwrapped group."""
    c = p.mean(axis=0)
    r = float(np.sqrt(((p - c) ** 2).sum(1).mean()))
    s = float(np.sqrt(((v - v.mean(axis=0)) ** 2).sum(1).mean()))
    return c, r, s


# --------------------------------------------------------------------------- #
# Candidate formation
# --------------------------------------------------------------------------- #
def _nearest_available(tree, pos: np.ndarray, claimed: np.ndarray,
                       q: np.ndarray, n: int, *, oversample: float,
                       ) -> Optional[np.ndarray]:
    """The ``n`` unclaimed particles nearest ``q``, or ``None`` if starved.

    The oversample is grown rather than fixed because "claimed" is not uniform:
    the first sets processed take the densest material, so a later set at the
    same radius can find its first ``oversample * n`` neighbours entirely spoken
    for. Growing is bounded, and exhausting the bound is a **recorded failure**
    -- never a silent fallback to particles from far away, which is the failure
    mode that would make a relocated set a random blob wearing an HR target.
    """
    p_total = int(pos.shape[0])
    k = min(p_total, max(int(np.ceil(float(oversample) * n)), n))
    for _ in range(4):
        _, idx = tree.query(q, k=k)
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        idx = idx[idx < p_total]                  # scipy pads misses with n
        free = idx[~claimed[idx]]
        if free.size >= n:
            return free[:n]
        if k >= p_total:
            break
        k = min(p_total, k * 4)
    return None


def _score(r_rms: float, sigma_v: float, rad_err: float, *,
           r_ref: float, s_ref: float, eps: float,
           w_vel: float, w_rad: float) -> float:
    """Lower is better. Compactness, velocity coherence, radius error.

    Both spread terms are **one-sided in log**: a candidate already at or below
    HR's own ``r_rms`` scores zero on it, because the loss that follows is
    two-sided and hinged and will handle the rest. Making the selector minimise
    spread outright would have it prefer an over-collapsed core over the object
    HR actually had, which the ``rrms`` term would then have to undo.
    """
    a = max(np.log(max(r_rms, 1e-9) / max(r_ref, 1e-9)), 0.0)
    b = max(np.log(max(sigma_v, 1e-9) / max(s_ref, 1e-9)), 0.0)
    return a + float(w_vel) * b + float(w_rad) * (rad_err / max(eps, 1e-9))


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #
def relocate_member_sets(
    sets, frozen_pos: torch.Tensor, frozen_vel: torch.Tensor, cfg, *,
    host_rvir_mpc_h: float,
    report: Optional[Dict] = None,
):
    """A new :class:`member_gather.MemberSets` on re-selected SR particles.

    ``frozen_pos`` / ``frozen_vel`` are
    :func:`member_gather.tile_particles` of the **frozen** field on the trained
    tiles -- the same table ``sets.live_rows`` indexes, so the returned rows are
    interchangeable with the ones they replace and every downstream consumer
    (the loss, the splice, ``export_gather_tiles``) needs no change.

    ``cfg.relocate`` picks the arm: ``"frozen"`` requests each group at the
    radius its frozen-SR centroid already sits at (arm B), ``"hr"`` at the
    radius HR's reachable centroid sits at (arm C). ``"off"`` is a programming
    error here -- the caller decides whether to relocate at all.

    Sets are processed **largest first** and particles are claimed as they are
    taken, so no particle serves two groups. Largest first because the big sets
    are the ones with no slack: a 10,000-particle group has one or two viable
    pieces of material at a given radius and a 200-particle one has many.
    """
    from .member_gather import MemberSets

    mode = str(cfg.relocate)
    if mode not in ("frozen", "hr"):
        raise ValueError(
            f"relocate={mode!r} is not 'frozen' or 'hr'. 'off' means the caller "
            "should not have called relocate_member_sets at all.")
    if getattr(sets, "pools", None) is not None:
        raise ValueError(
            "relocation and soft membership both replace the member set, one "
            "before training and one every step. Running both would supervise "
            "a re-solved relaxation of a re-selected group and no result could "
            "be attributed to either. Use --relocate off or --soft-membership.")
    if not (float(host_rvir_mpc_h) > 0.0):
        raise ValueError(
            f"host_rvir_mpc_h {host_rvir_mpc_h} is not positive; it is the "
            "denominator of the radius constraint and of every number in the "
            "pre-training check, so a default of 1.0 would silently reinterpret "
            "the tolerance.")
    if not bool(sets.centre_host.abs().sum() > 0):
        raise ValueError(
            "centre_host is all zero: the pool was built with host_pos=None, so "
            "there is no host to measure a radius from. Relocation needs the "
            "host position that --centre-mode radial and --w-radius need.")

    from scipy.spatial import cKDTree

    rep: Dict = report if report is not None else {}
    dev = sets.centre_host.device
    dt = sets.centre_host.dtype
    L = float(sets.boxsize_mpc_h)
    R_h = float(host_rvir_mpc_h)
    eps = float(cfg.relocate_eps)

    pos = frozen_pos.detach().cpu().numpy().astype(np.float64)
    vel = frozen_vel.detach().cpu().numpy().astype(np.float64)
    if pos.shape[0] != int(frozen_vel.shape[0]):
        raise ValueError("frozen_pos and frozen_vel disagree on particle count")
    tree = cKDTree(np.remainder(pos, L), boxsize=L)
    claimed = np.zeros(pos.shape[0], dtype=bool)

    host_np = sets.centre_host.detach().cpu().numpy().astype(np.float64)
    self_np = sets.centre_self.detach().cpu().numpy().astype(np.float64)
    tgt_np = sets.centre_target.detach().cpu().numpy().astype(np.float64)
    ref_r = sets.ref["r_rms"].detach().cpu().numpy().astype(np.float64)
    ref_s = sets.ref["sigma_v"].detach().cpu().numpy().astype(np.float64)

    # N_j is the size of the collection that produced `sets.ref` -- the live
    # members PLUS the out-of-tile stragglers that entered the hybrid reference
    # at frozen coordinates. Using `n_live` alone would ask the group to match a
    # reference measured on more particles than it holds; using `num_p` would
    # ask it to match one measured on fewer.
    n_out = np.array([0 if fp is None else int(fp.shape[0])
                      for fp in sets.fixed_pos], dtype=np.int64)
    want = np.asarray(sets.n_live, dtype=np.int64) + n_out
    order = np.argsort(-want, kind="stable")

    rows_new: List[Optional[np.ndarray]] = [None] * sets.n_sets
    ctr_new = np.zeros((sets.n_sets, 3), dtype=np.float64)
    tgt_new = np.zeros((sets.n_sets, 3), dtype=np.float64)
    rad_err = np.full(sets.n_sets, np.nan)
    ang_move = np.full(sets.n_sets, np.nan)
    rrms_ratio = np.full(sets.n_sets, np.nan)
    keep_frac = np.full(sets.n_sets, np.nan)
    starved: List[int] = []

    rng = np.random.default_rng(int(cfg.bg_seed))
    n_dirs = int(cfg.relocate_dirs)
    for s in order:
        n = int(want[s])
        host = host_np[s]
        # rho_j: the requested host-centric radius, in R_h. The ONLY thing that
        # differs between arm B and arm C.
        anchor_pt = self_np[s] if mode == "frozen" else tgt_np[s]
        rho = float(np.linalg.norm(anchor_pt - host)) / R_h
        r_want = rho * R_h

        d0 = _unit(self_np[s] - host, np.array([0.0, 0.0, 1.0]))
        dirs = np.concatenate(
            [d0[None, :],
             fibonacci_directions(max(n_dirs - 1, 1),
                                  seed=int(cfg.bg_seed) + int(s) + 1)],
            axis=0)[:max(n_dirs, 1)]

        def score_of(c: np.ndarray, r: float, sig: float) -> Tuple[float, float]:
            """``(score, radius error)`` of a group with these moments."""
            err = abs(float(np.linalg.norm(c - host)) - r_want) / R_h
            return _score(r, sig, err, r_ref=ref_r[s], s_ref=ref_s[s], eps=eps,
                          w_vel=float(cfg.relocate_w_vel),
                          w_rad=float(cfg.relocate_w_radius)), err

        best = None
        for d in dirs:
            q = host + r_want * d
            sel = _nearest_available(tree, pos, claimed, q, n,
                                     oversample=float(cfg.relocate_oversample))
            if sel is None:
                continue
            # Refinement, and it is a Newton step on the query point rather
            # than a re-query at the aim point. Selecting the nearest N about
            # `q` lands a centroid at `q + bias`, where the bias is the local
            # density gradient; re-querying at the aim would carry that same
            # bias along, so the correction that actually removes it is
            # `q <- q + (aim - achieved)`. Each pass is a re-selection of the
            # nearest N, i.e. a local swap of the members.
            cand, cand_q = sel, q
            cand_c, cand_r, cand_s = _spread(_unwrap_np(pos[sel], q, L), vel[sel])
            cand_score, _ = score_of(cand_c, cand_r, cand_s)
            for _ in range(int(cfg.relocate_refine_iters)):
                aim = host + r_want * _unit(cand_c - host, d)
                q2 = cand_q + (aim - cand_c)
                sel2 = _nearest_available(
                    tree, pos, claimed, q2, n,
                    oversample=float(cfg.relocate_oversample))
                if sel2 is None:
                    break
                c2, r2, s2 = _spread(_unwrap_np(pos[sel2], q2, L), vel[sel2])
                sc2, _ = score_of(c2, r2, s2)
                if sc2 >= cand_score:
                    break
                cand, cand_q, cand_score = sel2, q2, sc2
                cand_c, cand_r, cand_s = c2, r2, s2
            sc, e = score_of(cand_c, cand_r, cand_s)
            if best is None or sc < best[0]:
                best = (sc, cand, cand_c, cand_r, cand_s, e)

        if best is None:
            # Every direction starved. Recorded, and the set keeps its ORIGINAL
            # HR membership -- an arm-A set inside a B/C run, flagged, rather
            # than a dropped object that would make the three arms supervise
            # different catalogs and their Rockstar gates incomparable.
            starved.append(int(s))
            continue

        _, sel, c_new, r_new, _, e = best
        claimed[sel] = True
        rows_new[s] = np.sort(sel)
        ctr_new[s] = c_new
        tgt_new[s] = host + r_want * _unit(c_new - host, d0)
        rad_err[s] = e
        rrms_ratio[s] = r_new / max(ref_r[s], 1e-9)
        ang_move[s] = float(np.degrees(np.arccos(np.clip(
            float(np.dot(_unit(c_new - host, d0), d0)), -1.0, 1.0))))
        old = sets.live_rows[s].detach().cpu().numpy()
        keep_frac[s] = (float(np.intersect1d(np.sort(sel), np.sort(old)).size)
                        / max(old.size, 1))

    # --- rebuild the per-set fields ----------------------------------------
    live_rows, fixed_pos, fixed_vel, bg_rows = [], [], [], []
    centre_ref, centre_tgt, centre_slf, centre_hst, centre_rh = [], [], [], [], []
    n_live_new = np.array(sets.n_live, dtype=np.int64).copy()
    relocated = np.zeros(sets.n_sets, dtype=bool)

    for s in range(sets.n_sets):
        if rows_new[s] is None:
            live_rows.append(sets.live_rows[s])
            fixed_pos.append(sets.fixed_pos[s])
            fixed_vel.append(sets.fixed_vel[s])
            bg_rows.append(sets.bg_rows[s])
            centre_ref.append(sets.centre_ref[s])
            centre_tgt.append(sets.centre_target[s])
            centre_slf.append(sets.centre_self[s])
            centre_hst.append(sets.centre_host[s])
            centre_rh.append(sets.centre_rhat[s])
            continue
        relocated[s] = True
        rows = rows_new[s]
        n_live_new[s] = int(rows.size)
        live_rows.append(torch.as_tensor(rows, device=dev, dtype=torch.long))
        # No stragglers by construction: every chosen particle came out of the
        # table of trained tiles, so the whole group is movable and the splice
        # will carry all of it.
        fixed_pos.append(None)
        fixed_vel.append(None)

        c_new = ctr_new[s]
        # The local non-member background, rebuilt about the NEW centroid at the
        # same physical radius the original used (bg_radius_factor * HR's own
        # r_rms). Keeping the old rows would compute `d6` against a background
        # a megaparsec from the object it normalises.
        bg_t = None
        if int(cfg.bg_k) > 0:
            r_ball = float(cfg.bg_radius_factor) * float(ref_r[s])
            near = np.asarray(tree.query_ball_point(
                np.remainder(c_new, L), r_ball), dtype=np.int64)
            near = np.setdiff1d(near, rows, assume_unique=False)
            if near.size > int(cfg.bg_k):
                near = rng.choice(near, size=int(cfg.bg_k), replace=False)
            if near.size > 1:
                bg_t = torch.as_tensor(np.sort(near), device=dev,
                                       dtype=torch.long)
        bg_rows.append(bg_t)

        host = _unwrap_np(host_np[s], c_new, L)
        centre_ref.append(torch.as_tensor(c_new, device=dev, dtype=dt))
        centre_slf.append(torch.as_tensor(c_new, device=dev, dtype=dt))
        centre_tgt.append(torch.as_tensor(tgt_new[s], device=dev, dtype=dt))
        centre_hst.append(torch.as_tensor(host, device=dev, dtype=dt))
        rh = _unit(tgt_new[s] - host, np.zeros(3))
        centre_rh.append(torch.as_tensor(rh, device=dev, dtype=dt))

    ref = {k: v.clone() for k, v in sets.ref.items()}
    # `bound_scale` is a TEMPERATURE, not a target: tau G m N / r_rms. N changed.
    from .member_gather import G_MPC_KMS2_PER_MSUN
    ref["bound_scale"] = (
        float(cfg.bound_tau) * G_MPC_KMS2_PER_MSUN
        * float(sets.particle_mass_msun_h)
        * torch.as_tensor(n_live_new.astype(np.float64), device=dev, dtype=dt)
        / ref["r_rms"].clamp_min(1e-12))

    # --- the pre-training check --------------------------------------------
    # Seeds at the correct radii are not enough: the nearest-particle group
    # around a correct seed can have a displaced centroid. This is the
    # distribution of what was ACTUALLY built, written before a GPU-hour is
    # spent, and `relocate_check_only` stops the run right after it.
    def _q(a: np.ndarray, qs=(0.5, 0.9)) -> Dict[str, float]:
        good = a[np.isfinite(a)]
        if good.size == 0:
            return {f"p{int(x * 100)}": float("nan") for x in qs}
        return {f"p{int(x * 100)}": float(np.quantile(good, x)) for x in qs}

    ok = np.isfinite(rad_err) & (rad_err <= eps)
    rep["relocate"] = mode
    rep["host_rvir_mpc_h"] = R_h
    rep["relocate_eps"] = eps
    rep["relocate_dirs"] = n_dirs
    rep["n_sets"] = int(sets.n_sets)
    rep["n_relocated"] = int(relocated.sum())
    rep["n_starved"] = len(starved)
    rep["starved_halo_ids"] = [int(sets.halo_id[i]) for i in starved[:32]]
    rep["n_within_eps"] = int(ok.sum())
    rep["frac_within_eps"] = float(ok.sum() / max(int(relocated.sum()), 1))
    rep["radius_err_radii"] = _q(rad_err)
    rep["rrms_over_hr"] = _q(rrms_ratio)
    rep["angular_move_deg"] = _q(ang_move)
    # How much of the HR membership survived. ~1 would mean the selector found
    # the same particles and the arm is arm A under another name; ~0 means the
    # groups are made of entirely different material, which is the intent.
    rep["kept_hr_member_frac"] = _q(keep_frac)
    rep["n_particles_claimed"] = int(claimed.sum())
    rep["particle_table_size"] = int(pos.shape[0])
    rep["median_requested_radius_radii"] = float(np.median([
        float(np.linalg.norm((self_np[s] if mode == "frozen" else tgt_np[s])
                             - host_np[s])) / R_h
        for s in range(sets.n_sets)]))
    rep["median_n_want"] = float(np.median(want))
    rep["n_live_sum_before"] = int(np.sum(sets.n_live))
    rep["n_live_sum_after"] = int(np.sum(n_live_new))
    rep["sum_num_p_squared"] = float(
        np.sum(n_live_new.astype(np.float64) ** 2))

    out = replace(
        sets,
        n_live=n_live_new,
        live_rows=live_rows, fixed_pos=fixed_pos, fixed_vel=fixed_vel,
        bg_rows=bg_rows,
        centre_ref=torch.stack(centre_ref),
        centre_target=torch.stack(centre_tgt),
        centre_self=torch.stack(centre_slf),
        centre_host=torch.stack(centre_hst),
        centre_rhat=torch.stack(centre_rh),
        ref=ref,
        relocated=relocated,
    )
    return out
