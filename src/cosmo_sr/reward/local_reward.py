"""Object-level Rockstar reward: did this proposal create a *new* subhalo?

Why not the catalog reward
--------------------------
``R_cat`` / ``R_occ`` (:mod:`cosmo_sr.reward.reward`) compare whole-box binned
statistics against an HR target. They are the right objective for a model that
rewrites the entire field, and they are hopeless as a *search* signal here: one
extra subhalo in one host changes a full-box occupation curve by well under its
sampling noise. Searching on it would mean ranking candidates by Rockstar
jitter. So the search reward is asked at the level the edit actually operates
on -- one proposal, one object -- and the full-box statistics are still measured
and reported on every candidate, as evidence rather than as the objective.

The question, precisely
-----------------------
For proposal ``j``, targeted at frozen-SR2 host ``H``:

1. does the candidate catalog contain a subhalo that the frozen catalog did
   *not* contain (matched object-by-object, not by counting);
2. is it a subhalo **of the host that matches ``H``**, not of a neighbour and
   not an independent halo;
3. is it where and roughly what mass the proposal asked for;
4. did ``H`` itself survive -- same mass, same place, same ``Vmax``;
5. did the edit leave debris elsewhere?

Step 1 is the one that is easy to get wrong in a flattering direction. Counting
subhalos before and after would reward an edit that destroys one object and
creates another; matching only against the *host's* current children would
reward a pre-existing subhalo that merely moved. So new-ness is decided by a
one-to-one periodic match of the whole candidate subhalo population against the
whole frozen one, and an object that matched anything is permanently ineligible.

The score
---------
    r_j = r_detected + w_m r_mass + w_p r_position
          - lambda_h r_host_damage - lambda_a r_artifacts

``r_detected`` is the only term that can be positive on its own; the shaping
terms are gated on detection, so a candidate cannot accumulate reward for being
"nearly right" without Rockstar ever finding an object. That is deliberate: the
whole point of this pipeline is that the halo finder, not a proxy, decides.

The proxy, and its quarantine
-----------------------------
:func:`compactness_proxy` measures whether the claimed particles ended up denser
and colder in phase space. It exists for one situation only -- an entire CEM
round in which no candidate creates anything, where a reward that is identically
zero gives the search no gradient at all. It is reported separately, it never
enters ``r_j``, and :func:`is_scientific_success` (which gates the replay set the
flow trains on) ignores it completely.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..eval.halo_match import match_hosts
from ..eval.rockstar import HaloCatalog

__all__ = [
    "LocalRewardConfig",
    "ProposalOutcome",
    "compactness_proxy",
    "evaluate_candidate",
    "full_box_scores",
    "gate1_verdict",
    "host_damage",
    "is_scientific_success",
    "load_local_reward_config",
    "match_by_position_mass",
    "new_subhalo_mask",
]


@dataclass(frozen=True)
class LocalRewardConfig:
    """Every tolerance the object-level reward depends on, in one place."""

    # --- "is this candidate object the same as one the frozen catalog had?" ---
    base_match_pos_mpc: float = 0.20
    base_match_mass_dex: float = 0.60
    # --- "is this new object the one proposal j asked for?" ------------------
    proposal_pos_tol_rvir: float = 0.35      # fraction of the HOST's Rvir
    proposal_mass_tol_dex: float = 0.80
    # --- shaping -------------------------------------------------------------
    w_detected: float = 1.0
    w_mass: float = 0.5
    w_position: float = 0.5
    sigma_mass_dex: float = 0.35
    # --- host preservation ---------------------------------------------------
    lambda_host: float = 1.0
    host_dmass_tol_dex: float = 0.05
    host_dpos_tol_rvir: float = 0.10
    host_dvmax_tol: float = 0.10
    host_lost_penalty: float = 3.0
    # --- artifacts -----------------------------------------------------------
    lambda_artifact: float = 0.5
    artifact_radius_rvir: float = 3.0
    artifact_scale: float = 4.0              # new objects that saturate the term
    # --- proxy tie-break (never a success) -----------------------------------
    proxy_weight: float = 0.0

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


def load_local_reward_config(cfg: Dict) -> LocalRewardConfig:
    """Build from the ``reward`` block of ``configs/reward/local_editor.yaml``."""
    c = dict(cfg or {})
    known = LocalRewardConfig().to_dict()
    return LocalRewardConfig(**{k: type(v)(c.get(k, v)) for k, v in known.items()})


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _min_image(d: np.ndarray, box: float) -> np.ndarray:
    return d - float(box) * np.round(d / float(box))


def _periodic_dist(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    return np.linalg.norm(_min_image(np.asarray(a) - np.asarray(b), box), axis=-1)


def match_by_position_mass(
    ref_pos: np.ndarray,
    ref_mass: np.ndarray,
    qry_pos: np.ndarray,
    qry_mass: np.ndarray,
    *,
    boxsize_mpc_h: float = 100.0,
    pos_tol_mpc: float = 0.20,
    mass_tol_dex: float = 0.60,
) -> np.ndarray:
    """``(n_qry,)`` index into ``ref`` for each query object, ``-1`` if unmatched.

    Greedy, one-to-one, nearest-first over the whole population. Greedy is not a
    compromise here: the alternative (global optimal assignment) can trade a
    tight pair for two loose ones, and "this object had a counterpart" should
    not depend on the presence of unrelated objects elsewhere in the box.
    """
    from scipy.spatial import cKDTree

    ref_pos = np.asarray(ref_pos, dtype=np.float64).reshape(-1, 3)
    qry_pos = np.asarray(qry_pos, dtype=np.float64).reshape(-1, 3)
    out = np.full(qry_pos.shape[0], -1, dtype=np.int64)
    if ref_pos.shape[0] == 0 or qry_pos.shape[0] == 0:
        return out

    box = float(boxsize_mpc_h)
    tree = cKDTree(ref_pos % box, boxsize=box)
    pairs = tree.query_ball_point(qry_pos % box, r=float(pos_tol_mpc))

    lref = np.log10(np.maximum(np.asarray(ref_mass, dtype=np.float64), 1e-30))
    lqry = np.log10(np.maximum(np.asarray(qry_mass, dtype=np.float64), 1e-30))

    cand: List[Tuple[float, int, int]] = []
    for q, hits in enumerate(pairs):
        for r in hits:
            if abs(lqry[q] - lref[r]) > float(mass_tol_dex):
                continue
            d = float(np.linalg.norm(_min_image(qry_pos[q] - ref_pos[r], box)))
            cand.append((d, int(q), int(r)))
    cand.sort()
    used_r, used_q = set(), set()
    for _, q, r in cand:
        if q in used_q or r in used_r:
            continue
        used_q.add(q)
        used_r.add(r)
        out[q] = r
    return out


def new_subhalo_mask(
    base: HaloCatalog,
    cand: HaloCatalog,
    cfg: LocalRewardConfig,
    *,
    boxsize_mpc_h: float = 100.0,
) -> np.ndarray:
    """``(n_cand_sub,)`` bool: candidate subhalos with no frozen counterpart.

    The reference population is *every* frozen object, hosts included, not just
    frozen subhalos. An edit that splits a frozen host into a host plus a
    satellite has not created new structure; matching against subhalos alone
    would call the satellite new and pay for it.
    """
    cs = cand.subhalos()
    if cs.n == 0:
        return np.zeros(0, dtype=bool)
    m = match_by_position_mass(
        base.pos, base.mvir, cs.pos, cs.mvir,
        boxsize_mpc_h=boxsize_mpc_h,
        pos_tol_mpc=cfg.base_match_pos_mpc,
        mass_tol_dex=cfg.base_match_mass_dex,
    )
    return m < 0


# ---------------------------------------------------------------------------
# Host preservation
# ---------------------------------------------------------------------------


def host_damage(
    base: HaloCatalog,
    cand: HaloCatalog,
    base_host_id: int,
    cand_host_id: int,
    cfg: LocalRewardConfig,
    *,
    boxsize_mpc_h: float = 100.0,
) -> Dict:
    """How far the targeted host moved, in units of its own tolerances.

    Each channel is normalised by its tolerance and the penalty is their sum, so
    ``r_host_damage = 1`` means "one tolerance's worth of damage" regardless of
    which channel it came from. A host that failed to match at all is charged
    ``host_lost_penalty``: losing the parent is the worst outcome the editor can
    produce and must dominate any object it created.
    """
    if cand_host_id < 0:
        return {"host_matched": False, "r_host_damage": float(cfg.host_lost_penalty),
                "dlog10_mvir": float("nan"), "dpos_rvir": float("nan"),
                "dlog10_vmax": float("nan")}
    bi = int(np.nonzero(base.ids == int(base_host_id))[0][0])
    ci = int(np.nonzero(cand.ids == int(cand_host_id))[0][0])
    rvir_mpc = max(float(base.rvir[bi]) * 1e-3, 1e-9)
    dm = float(abs(np.log10(max(cand.mvir[ci], 1e-30) / max(base.mvir[bi], 1e-30))))
    dx = float(_periodic_dist(cand.pos[ci], base.pos[bi], boxsize_mpc_h)) / rvir_mpc
    dv = float(abs(np.log10(max(cand.vmax[ci], 1e-30) / max(base.vmax[bi], 1e-30))))
    r = (dm / max(cfg.host_dmass_tol_dex, 1e-9)
         + dx / max(cfg.host_dpos_tol_rvir, 1e-9)
         + dv / max(cfg.host_dvmax_tol, 1e-9)) / 3.0
    return {"host_matched": True, "r_host_damage": float(r),
            "dlog10_mvir": dm, "dpos_rvir": dx, "dlog10_vmax": dv}


# ---------------------------------------------------------------------------
# The proxy (quarantined)
# ---------------------------------------------------------------------------


def compactness_proxy(
    pos_pre: np.ndarray, vel_pre: np.ndarray,
    pos_post: np.ndarray, vel_post: np.ndarray,
    *, boxsize_mpc_h: float = 100.0,
) -> float:
    """``log10`` change in the coarse phase-space density of the claimed pool.

    ``Q ~ N / (sigma_x^3 sigma_v^3)``; the return value is ``log10(Q_post/Q_pre)``,
    so positive means denser and colder. This is a *weak* statement about
    boundedness and nothing at all about detectability: Rockstar's own criterion
    is an unbinding calculation this does not attempt.

    Use only to break ties among candidates that all scored exactly zero. It
    never enters ``r_j`` and never marks a success.
    """
    def _q(p, v):
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
        if p.shape[0] < 2:
            return np.nan
        # Centre periodically on the mean of the minimum-image offsets from the
        # first particle, so a pool straddling a box face is not measured as
        # box-sized.
        rel = _min_image(p - p[0], boxsize_mpc_h)
        sx = float(np.sqrt(np.mean(np.sum((rel - rel.mean(0)) ** 2, axis=1)) / 3.0))
        sv = float(np.sqrt(np.mean(np.sum((v - v.mean(0)) ** 2, axis=1)) / 3.0))
        if sx <= 0 or sv <= 0:
            return np.nan
        return p.shape[0] / (sx ** 3 * sv ** 3)

    a, b = _q(pos_pre, vel_pre), _q(pos_post, vel_post)
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return float("nan")
    return float(np.log10(b / a))


# ---------------------------------------------------------------------------
# Per-proposal outcome
# ---------------------------------------------------------------------------


@dataclass
class ProposalOutcome:
    """Everything the search, the replay set and the report need about one edit."""

    proposal_index: int
    base_host_id: int
    cand_host_id: int = -1
    detected: bool = False
    new_sub_id: int = -1
    new_sub_mvir: float = float("nan")
    new_sub_num_p: int = 0
    requested_mvir: float = float("nan")
    dlog10_mass: float = float("nan")
    dpos_rvir: float = float("nan")
    r_detected: float = 0.0
    r_mass: float = 0.0
    r_position: float = 0.0
    r_host_damage: float = 0.0
    r_artifacts: float = 0.0
    reward: float = 0.0
    host_matched: bool = False
    host_damage: Dict = dc_field(default_factory=dict)
    n_artifacts: float = 0.0
    # Particles the edit actually moved. Zero means the window contained none of
    # the claimed particles, so the candidate was a no-op that still cost a
    # Rockstar run -- a *different* failure from an edit that was applied and did
    # not produce an object, and one the search must be able to tell apart.
    n_active_particles: int = -1
    compactness_proxy: float = float("nan")
    feasible_field: bool = True
    violations: Tuple[str, ...] = ()
    reason: str = ""
    meta: Dict = dc_field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = dict(self.__dict__)
        d["violations"] = list(self.violations)
        return d

    @staticmethod
    def from_dict(d: Dict) -> "ProposalOutcome":
        d = dict(d)
        d["violations"] = tuple(d.get("violations", ()))
        known = set(ProposalOutcome(0, 0).__dict__)
        return ProposalOutcome(**{k: v for k, v in d.items() if k in known})


def is_scientific_success(o: ProposalOutcome) -> bool:
    """Gate for the replay set the conditional flow trains on.

    Three independent conditions, all necessary: the halo finder found a *new*
    object, it is a subhalo of the host that was asked for, and the candidate
    field passed the feasibility filter. The compactness proxy is deliberately
    absent -- a proxy-only "success" in the training set would teach the flow to
    produce actions that look good to the proxy and not to Rockstar, which is
    the exact failure the object-level reward exists to avoid.
    """
    return bool(o.detected and o.host_matched and o.feasible_field)


def evaluate_candidate(
    base: HaloCatalog,
    cand: HaloCatalog,
    proposals: Sequence[Dict],
    cfg: LocalRewardConfig,
    *,
    boxsize_mpc_h: float = 100.0,
    feasible_field: bool = True,
    violations: Sequence[str] = (),
) -> List[ProposalOutcome]:
    """Score every proposal in one candidate box against the frozen catalog.

    ``proposals`` carries, per entry: ``base_host_id``, ``center_mpc``,
    ``requested_mvir``, ``host_rvir_mpc``. That is all the reward needs -- the
    action itself is recorded by the caller, not scored here, so no tolerance in
    this module can be tuned against a particular action parameterisation.

    An infeasible field short-circuits: every proposal scores exactly zero with
    ``reason='infeasible_field'``. It does not score *negative*, because a
    negative reward would let CEM learn to avoid a region of action space for a
    reason unrelated to whether an object appeared there.
    """
    n = len(proposals)
    out = [ProposalOutcome(proposal_index=i,
                           base_host_id=int(p["base_host_id"]),
                           requested_mvir=float(p.get("requested_mvir", np.nan)),
                           feasible_field=bool(feasible_field),
                           violations=tuple(violations))
           for i, p in enumerate(proposals)]
    if not feasible_field:
        for o in out:
            o.reason = "infeasible_field"
        return out
    if n == 0:
        return out

    # --- host correspondence -------------------------------------------------
    hm = match_hosts(base, cand, float(boxsize_mpc_h))
    base_to_cand = {int(h): int(s) for h, s in zip(hm.hr_ids, hm.sr_ids)}

    # --- which candidate subhalos are new -----------------------------------
    cs = cand.subhalos()
    is_new = new_subhalo_mask(base, cand, cfg, boxsize_mpc_h=boxsize_mpc_h)
    new_rows = np.nonzero(is_new)[0]

    # --- proposal <-> new subhalo, one to one -------------------------------
    # Ranked by a single scalar (normalised distance plus a tenth of the mass
    # residual) so a proposal never claims a far object just because its mass is
    # a better fit; position is what identifies an object.
    claims: List[Tuple[float, int, int]] = []
    for i, p in enumerate(proposals):
        bh = int(p["base_host_id"])
        ch = base_to_cand.get(bh, -1)
        out[i].cand_host_id = ch
        dmg = host_damage(base, cand, bh, ch, cfg, boxsize_mpc_h=boxsize_mpc_h)
        out[i].host_matched = bool(dmg["host_matched"])
        out[i].host_damage = dmg
        out[i].r_host_damage = float(dmg["r_host_damage"])
        if ch < 0:
            out[i].reason = "host_lost"
            continue
        rvir = max(float(p["host_rvir_mpc"]), 1e-9)
        tol = float(cfg.proposal_pos_tol_rvir) * rvir
        want_m = float(p.get("requested_mvir", np.nan))
        center = np.asarray(p["center_mpc"], dtype=np.float64)
        for r in new_rows:
            if int(cs.parent_ids[r]) != ch:
                continue          # a new object in the wrong host is an artifact
            d = float(_periodic_dist(cs.pos[r], center, boxsize_mpc_h))
            if d > tol:
                continue
            dm = (abs(float(np.log10(max(cs.mvir[r], 1e-30) / max(want_m, 1e-30))))
                  if np.isfinite(want_m) and want_m > 0 else 0.0)
            if dm > float(cfg.proposal_mass_tol_dex):
                continue
            claims.append((d / tol + 0.1 * dm, i, int(r)))

    claims.sort()
    used_p, used_r = set(), set()
    for _, i, r in claims:
        if i in used_p or r in used_r:
            continue
        used_p.add(i)
        used_r.add(r)
        p = proposals[i]
        rvir = max(float(p["host_rvir_mpc"]), 1e-9)
        tol = float(cfg.proposal_pos_tol_rvir) * rvir
        center = np.asarray(p["center_mpc"], dtype=np.float64)
        d = float(_periodic_dist(cs.pos[r], center, boxsize_mpc_h))
        want_m = float(p.get("requested_mvir", np.nan))
        dm = (float(np.log10(max(cs.mvir[r], 1e-30) / max(want_m, 1e-30)))
              if np.isfinite(want_m) and want_m > 0 else float("nan"))
        o = out[i]
        o.detected = True
        o.new_sub_id = int(cs.ids[r])
        o.new_sub_mvir = float(cs.mvir[r])
        o.new_sub_num_p = int(cs.num_p[r])
        o.dlog10_mass = dm
        o.dpos_rvir = d / rvir
        o.r_detected = float(cfg.w_detected)
        o.r_mass = float(cfg.w_mass) * (
            1.0 if not np.isfinite(dm)
            else float(np.exp(-0.5 * (dm / max(cfg.sigma_mass_dex, 1e-9)) ** 2)))
        o.r_position = float(cfg.w_position) * float(
            np.exp(-0.5 * (d / tol) ** 2))
        o.reason = "detected"

    for o in out:
        if not o.detected and not o.reason:
            o.reason = "no_new_subhalo_in_host"

    # --- artifacts -----------------------------------------------------------
    # Every new object the proposals did not claim: unclaimed new subhalos, plus
    # candidate hosts with no frozen counterpart. Attributed to the nearest
    # proposal host when it is close enough to plausibly be that edit's debris,
    # otherwise shared evenly -- a box-wide change nobody can be blamed for
    # individually still has to cost something, or the search learns to make it.
    unclaimed = [int(r) for r in new_rows if int(r) not in used_r]
    ch_pos, ch_m = cand.hosts().pos, cand.hosts().mvir
    hb = match_by_position_mass(
        base.pos, base.mvir, ch_pos, ch_m, boxsize_mpc_h=boxsize_mpc_h,
        pos_tol_mpc=cfg.base_match_pos_mpc, mass_tol_dex=cfg.base_match_mass_dex)
    new_host_rows = [int(r) for r in np.nonzero(hb < 0)[0]]

    debris_pos = ([cs.pos[r] for r in unclaimed]
                  + [ch_pos[r] for r in new_host_rows])
    per = np.zeros(n, dtype=np.float64)
    shared = 0.0
    if debris_pos:
        hosts_c = np.asarray([np.asarray(p["center_mpc"], dtype=np.float64)
                              for p in proposals])
        radii = np.asarray([float(cfg.artifact_radius_rvir)
                            * float(p["host_rvir_mpc"]) for p in proposals])
        for q in debris_pos:
            d = _periodic_dist(hosts_c, np.asarray(q, dtype=np.float64), boxsize_mpc_h)
            near = d <= radii
            if np.any(near):
                per[int(np.argmin(np.where(near, d, np.inf)))] += 1.0
            else:
                shared += 1.0
    per = per + shared / max(n, 1)
    for i, o in enumerate(out):
        o.n_artifacts = float(per[i])
        o.r_artifacts = float(min(per[i] / max(cfg.artifact_scale, 1e-9), 1.0))
        o.reward = float(
            o.r_detected + o.r_mass + o.r_position
            - float(cfg.lambda_host) * o.r_host_damage
            - float(cfg.lambda_artifact) * o.r_artifacts
        )
    return out


# ---------------------------------------------------------------------------
# Gate 1: does the action space have anything in it?
# ---------------------------------------------------------------------------


def gate1_verdict(
    rows: Sequence[Dict],
    *,
    min_successes: int = 5,
    min_hosts: int = 3,
    min_boxes: int = 2,
    forbidden_boxes: Sequence[str] = (),
) -> Dict:
    """Stage 3's gate, computed from candidate rows. Nothing proceeds without it.

    A success must be a *legitimate* one, which is the same three conditions
    :func:`is_scientific_success` uses. The gate then asks for breadth as well as
    count -- five successes in one host of one box is a property of that host,
    not of the editor -- and for the targeted arm to beat the equal-count random
    control, which is what separates "this works" from "compressing any few
    hundred particles works".

    Returns a verdict rather than raising, and reports every component, so a
    failure says *which* clause failed. The plan's instruction if it does fail is
    explicit: expand the editor representation, do not implement the flow.
    """
    forb = set(forbidden_boxes)
    seen_boxes = sorted({str(r.get("box", "")) for r in rows})
    leak = sorted(set(seen_boxes) & forb)

    succ: List[Dict] = []
    by_arm: Dict[str, List[float]] = {}
    for r in rows:
        arm = f"{r.get('arm', '?')}/{r.get('control', 'none')}"
        for o in r.get("outcomes", []):
            oo = ProposalOutcome.from_dict(o)
            by_arm.setdefault(arm, []).append(float(oo.reward))
            if is_scientific_success(oo):
                succ.append({"box": str(r.get("box", "")), "arm": arm,
                             "host_id": int(oo.base_host_id),
                             "reward": float(oo.reward),
                             "new_sub_id": int(oo.new_sub_id)})

    targeted = [s for s in succ if s["arm"].endswith("/none")]
    hosts = sorted({s["host_id"] for s in targeted})
    boxes = sorted({s["box"] for s in targeted})

    def _rate(key: str) -> float:
        v = [s for s in succ if s["arm"] == key]
        n = len(by_arm.get(key, []))
        return (len(v) / n) if n else float("nan")

    rates = {k: _rate(k) for k in sorted(by_arm)}
    ctrl = [k for k in rates if "random_particles" in k]
    beats_control = (
        all(np.isnan(rates[k]) or rates[k] < max(
            (rates[t] for t in rates if t.endswith("/none")), default=0.0)
            for k in ctrl) if ctrl else False)

    checks = {
        "n_successes": len(targeted) >= int(min_successes),
        "n_hosts": len(hosts) >= int(min_hosts),
        "n_boxes": len(boxes) >= int(min_boxes),
        "beats_random_particle_control": bool(beats_control),
        "control_arm_present": bool(ctrl),
        "no_forbidden_box": not leak,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "n_successes": len(targeted),
        "hosts": hosts, "boxes": boxes,
        "success_rate_by_arm": rates,
        "n_scored_by_arm": {k: len(v) for k, v in sorted(by_arm.items())},
        "successes": targeted,
        "forbidden_boxes_touched": leak,
        "requirement": (f">= {min_successes} legitimate new subhalos across "
                        f">= {min_hosts} hosts and >= {min_boxes} non-final boxes, "
                        "hosts preserved, field feasible, and the targeted arm "
                        "ahead of the equal-count random-particle control"),
    }


# ---------------------------------------------------------------------------
# Full-box statistics, reported alongside every candidate
# ---------------------------------------------------------------------------


def full_box_scores(
    cand: HaloCatalog,
    bins,
    *,
    boxsize_mpc_h: float = 100.0,
    reward_model=None,
    reliable_host_bins: Optional[Sequence[int]] = None,
    box: str = "",
    tag: str = "",
) -> Dict:
    """``R_occ`` / ``R_abund`` / ``R_cat`` plus the raw occupation curve.

    Always computed, never optimised. One extra subhalo cannot move these
    numbers outside their sampling noise, so they are evidence that the editor
    is not wrecking the box -- not a search signal.
    """
    from .catalog import EnsembleSummary
    from .tiles import direct_full_box_stats

    stats = direct_full_box_stats(cand, bins)
    ens = EnsembleSummary(stats["n_sub"], stats["n_host"], stats["occ_numerator"],
                          float(boxsize_mpc_h) ** 3, (f"{box}/{tag}",))
    out = {
        "occupation": np.asarray(stats["occupation"]).tolist(),
        "n_host": np.asarray(stats["n_host"]).tolist(),
        "occ_numerator": np.asarray(stats["occ_numerator"]).tolist(),
        "n_sub": np.asarray(stats["n_sub"]).tolist(),
        "n_objects": int(cand.n),
    }
    if reward_model is not None:
        out.update(reward_model.scores(ens, reliable_host_bins))
        out["occupation_gap"] = np.asarray(reward_model.occupation_gap(ens)).tolist()
    return out
