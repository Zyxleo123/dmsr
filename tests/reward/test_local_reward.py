"""The object-level reward, and the ways it could flatter the editor.

The reward decides what the search optimises and what the flow trains on, so
every test here is about a *false positive*: paying for an object that was
already in the frozen catalog, paying twice for one object, paying for an object
in the wrong host, or paying at all for a candidate that destroyed its host.
"""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.eval.rockstar import HaloCatalog
from cosmo_sr.reward.local_reward import (
    LocalRewardConfig, ProposalOutcome, compactness_proxy, evaluate_candidate,
    gate1_verdict, host_damage, is_scientific_success, match_by_position_mass,
    new_subhalo_mask,
)

BOX = 100.0
CFG = LocalRewardConfig()


def cat(rows):
    """``rows`` = list of dicts with pos/mvir/parent; ids assigned in order."""
    n = len(rows)
    return HaloCatalog(
        ids=np.arange(n, dtype=np.int64),
        parent_ids=np.asarray([r.get("parent", -1) for r in rows], dtype=np.int64),
        mvir=np.asarray([r["mvir"] for r in rows], dtype=np.float64),
        rvir=np.asarray([r.get("rvir", 500.0) for r in rows], dtype=np.float64),
        vmax=np.asarray([r.get("vmax", 300.0) for r in rows], dtype=np.float64),
        pos=np.asarray([r["pos"] for r in rows], dtype=np.float64).reshape(-1, 3),
        vel=np.zeros((n, 3)),
        num_p=np.asarray([r.get("num_p", 200) for r in rows], dtype=np.int64),
    )


HOST = {"pos": [50.0, 50.0, 50.0], "mvir": 1e13, "rvir": 500.0, "num_p": 20000}


def base_and_candidate(extra_sub=None, host_shift=(0, 0, 0), host_mass=1e13):
    """A one-host box, optionally with one extra subhalo in the candidate."""
    base = cat([HOST, {"pos": [50.4, 50.0, 50.0], "mvir": 5e10, "parent": 0}])
    rows = [{**HOST, "pos": list(np.asarray(HOST["pos"]) + host_shift),
             "mvir": host_mass},
            {"pos": [50.4, 50.0, 50.0], "mvir": 5e10, "parent": 0}]
    if extra_sub is not None:
        rows.append({**extra_sub, "parent": 0})
    return base, cat(rows)


def proposal(center, mvir=8e10, rvir=0.5):
    return {"base_host_id": 0, "center_mpc": list(center),
            "host_rvir_mpc": rvir, "requested_mvir": mvir}


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def test_matching_is_one_to_one():
    ref_p = np.array([[10.0, 10.0, 10.0]])
    qry_p = np.array([[10.01, 10.0, 10.0], [10.02, 10.0, 10.0]])
    m = match_by_position_mass(ref_p, [1e11], qry_p, [1e11, 1e11],
                               boxsize_mpc_h=BOX, pos_tol_mpc=0.2)
    assert sorted(m.tolist()) == [-1, 0], "one reference matched two queries"


def test_matching_wraps_periodically():
    m = match_by_position_mass(np.array([[0.05, 0.0, 0.0]]), [1e11],
                               np.array([[99.95, 0.0, 0.0]]), [1e11],
                               boxsize_mpc_h=BOX, pos_tol_mpc=0.2)
    assert m[0] == 0


def test_a_mass_mismatch_blocks_a_match():
    m = match_by_position_mass(np.array([[10.0, 10.0, 10.0]]), [1e11],
                               np.array([[10.0, 10.0, 10.0]]), [1e14],
                               boxsize_mpc_h=BOX, pos_tol_mpc=0.2, mass_tol_dex=0.5)
    assert m[0] == -1


def test_an_unchanged_candidate_has_no_new_subhalos():
    base, cand = base_and_candidate()
    assert not new_subhalo_mask(base, cand, CFG, boxsize_mpc_h=BOX).any()


def test_a_genuinely_new_subhalo_is_detected_as_new():
    base, cand = base_and_candidate({"pos": [50.2, 50.1, 50.0], "mvir": 8e10})
    m = new_subhalo_mask(base, cand, CFG, boxsize_mpc_h=BOX)
    assert m.tolist() == [False, True]


def test_a_candidate_subhalo_matching_a_frozen_HOST_is_not_new():
    """Splitting a host into host + satellite creates no structure. Matching only
    against frozen *subhalos* would call the satellite new and pay for it."""
    base = cat([HOST])
    cand = cat([{**HOST, "pos": [50.05, 50.0, 50.0]},
                {"pos": [50.0, 50.0, 50.0], "mvir": 1e13, "parent": 0,
                 "num_p": 20000}])
    assert not new_subhalo_mask(base, cand, CFG, boxsize_mpc_h=BOX).any()


# ---------------------------------------------------------------------------
# Scoring one proposal
# ---------------------------------------------------------------------------


def test_a_new_subhalo_at_the_proposed_place_is_rewarded():
    base, cand = base_and_candidate({"pos": [50.1, 50.0, 50.0], "mvir": 8e10})
    out = evaluate_candidate(base, cand, [proposal([50.1, 50.0, 50.0])], CFG,
                             boxsize_mpc_h=BOX)
    o = out[0]
    assert o.detected and is_scientific_success(o)
    assert o.r_detected == pytest.approx(CFG.w_detected)
    assert o.reward > 1.0
    assert o.new_sub_mvir == pytest.approx(8e10)


def test_an_existing_frozen_subhalo_is_never_rewarded():
    """The reward's single most important negative: aiming at an object that was
    already there must score exactly zero detection."""
    base, cand = base_and_candidate()
    out = evaluate_candidate(base, cand, [proposal([50.4, 50.0, 50.0], mvir=5e10)],
                             CFG, boxsize_mpc_h=BOX)
    assert not out[0].detected
    assert out[0].r_detected == 0.0
    assert out[0].reason == "no_new_subhalo_in_host"


def test_the_mass_comparison_is_against_what_the_editor_could_build():
    """``requested_mvir`` must be the moved-particle mass, not the token's
    nominal ratio.

    On the hosts stage 1 selects (~1.4e5 members) a token at log_mass_ratio
    = -1.3 nominally asks for 4.5e12 Msun/h while the count clamp allows a few
    hundred particles, i.e. ~2.3e11 -- a 1.29 dex gap against a 0.8 dex
    tolerance. Judged against the nominal figure, a *genuine* new subhalo is
    rejected on mass grounds and recorded as a failure.
    """
    base, cand = base_and_candidate({"pos": [50.1, 50.0, 50.0], "mvir": 2.3e11})
    buildable = proposal([50.1, 50.0, 50.0], mvir=2.3e11)   # 400 particles' worth
    nominal = proposal([50.1, 50.0, 50.0], mvir=4.5e12)     # 10^-1.3 * 9e13
    assert evaluate_candidate(base, cand, [buildable], CFG,
                              boxsize_mpc_h=BOX)[0].detected
    assert not evaluate_candidate(base, cand, [nominal], CFG,
                                  boxsize_mpc_h=BOX)[0].detected


def test_a_new_object_too_far_from_the_proposal_is_an_artifact_not_a_success():
    base, cand = base_and_candidate({"pos": [50.0, 50.0, 53.0], "mvir": 8e10})
    out = evaluate_candidate(base, cand, [proposal([50.1, 50.0, 50.0])], CFG,
                             boxsize_mpc_h=BOX)
    assert not out[0].detected
    assert out[0].n_artifacts >= 1
    assert out[0].r_artifacts > 0


def test_one_new_object_cannot_pay_two_proposals():
    base, cand = base_and_candidate({"pos": [50.1, 50.0, 50.0], "mvir": 8e10})
    out = evaluate_candidate(
        base, cand,
        [proposal([50.10, 50.0, 50.0]), proposal([50.11, 50.0, 50.0])],
        CFG, boxsize_mpc_h=BOX)
    assert sum(1 for o in out if o.detected) == 1


def test_a_new_object_in_the_wrong_host_is_not_a_success():
    two_hosts = [HOST, {"pos": [70.0, 50.0, 50.0], "mvir": 1e13, "num_p": 20000}]
    base = cat(two_hosts)
    cand = cat(two_hosts + [{"pos": [70.1, 50.0, 50.0], "mvir": 8e10, "parent": 1}])
    out = evaluate_candidate(base, cand, [proposal([70.1, 50.0, 50.0])], CFG,
                             boxsize_mpc_h=BOX)
    # The object exists and is exactly where proposal 0 asked, but it is a child
    # of host 1 while the proposal named host 0.
    assert not out[0].detected


# ---------------------------------------------------------------------------
# Host preservation
# ---------------------------------------------------------------------------


def test_moving_the_host_costs_reward():
    base, cand = base_and_candidate({"pos": [50.1, 50.0, 50.0], "mvir": 8e10})
    clean = evaluate_candidate(base, cand, [proposal([50.1, 50.0, 50.0])], CFG,
                               boxsize_mpc_h=BOX)[0]
    base2, damaged = base_and_candidate({"pos": [50.15, 50.05, 50.0], "mvir": 8e10},
                                        host_shift=(0.1, 0.0, 0.0))
    hurt = evaluate_candidate(base2, damaged, [proposal([50.15, 50.05, 50.0])],
                              CFG, boxsize_mpc_h=BOX)[0]
    assert hurt.r_host_damage > clean.r_host_damage
    assert hurt.reward < clean.reward


def test_losing_the_host_dominates_anything_the_edit_created():
    base = cat([HOST])
    cand = cat([{"pos": [10.0, 10.0, 10.0], "mvir": 1e11, "num_p": 300}])
    out = evaluate_candidate(base, cand, [proposal([50.1, 50.0, 50.0])], CFG,
                             boxsize_mpc_h=BOX)[0]
    assert not out.host_matched
    assert out.r_host_damage == pytest.approx(CFG.host_lost_penalty)
    assert out.reward < 0
    assert not is_scientific_success(out)


def test_host_damage_channels_are_normalised_by_their_own_tolerances():
    base = cat([HOST])
    cand = cat([{**HOST, "mvir": 1e13 * 10 ** CFG.host_dmass_tol_dex}])
    d = host_damage(base, cand, 0, 0, CFG, boxsize_mpc_h=BOX)
    # One tolerance of mass error, nothing else: 1/3 of a "unit" of damage.
    assert d["r_host_damage"] == pytest.approx(1.0 / 3.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Feasibility and the proxy
# ---------------------------------------------------------------------------


def test_an_infeasible_field_scores_zero_and_not_negative():
    """Zero, because a negative would teach CEM to avoid that region of action
    space for a reason unrelated to whether an object appeared there."""
    base, cand = base_and_candidate({"pos": [50.1, 50.0, 50.0], "mvir": 8e10})
    out = evaluate_candidate(base, cand, [proposal([50.1, 50.0, 50.0])], CFG,
                             boxsize_mpc_h=BOX, feasible_field=False,
                             violations=["low_k_change=1>0.5"])[0]
    assert out.reward == 0.0
    assert not out.detected and not is_scientific_success(out)
    assert out.reason == "infeasible_field"


def test_the_compactness_proxy_never_makes_a_success():
    o = ProposalOutcome(0, 0, detected=False, host_matched=True,
                        compactness_proxy=99.0, feasible_field=True)
    assert not is_scientific_success(o)


def test_the_proxy_rises_when_the_pool_gets_denser_and_colder():
    rng = np.random.default_rng(0)
    p = rng.normal(50.0, 0.2, size=(200, 3))
    v = rng.normal(0.0, 100.0, size=(200, 3))
    tighter = 50.0 + 0.3 * (p - 50.0)
    colder = 0.3 * v
    assert compactness_proxy(p, v, tighter, colder, boxsize_mpc_h=BOX) > 0
    assert compactness_proxy(p, v, p, v, boxsize_mpc_h=BOX) == pytest.approx(0.0)


def test_the_proxy_is_not_fooled_by_a_pool_straddling_the_box_face():
    p = np.array([[99.9, 50.0, 50.0], [0.1, 50.0, 50.0], [0.0, 50.0, 50.0]])
    v = np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
    val = compactness_proxy(p, v, p, v, boxsize_mpc_h=BOX)
    assert val == pytest.approx(0.0)     # finite, not a box-sized sigma


# ---------------------------------------------------------------------------
# Periodic invariance of the whole reward
# ---------------------------------------------------------------------------


def test_translating_the_whole_box_does_not_change_the_reward():
    base, cand = base_and_candidate({"pos": [50.1, 50.0, 50.0], "mvir": 8e10})
    a = evaluate_candidate(base, cand, [proposal([50.1, 50.0, 50.0])], CFG,
                           boxsize_mpc_h=BOX)[0]

    shift = np.array([60.0, -33.0, 47.0])
    def moved(c):
        return HaloCatalog(**{**c.__dict__, "pos": (c.pos + shift) % BOX})
    b = evaluate_candidate(moved(base), moved(cand),
                           [proposal((np.array([50.1, 50.0, 50.0]) + shift) % BOX)],
                           CFG, boxsize_mpc_h=BOX)[0]
    assert b.detected == a.detected
    assert b.reward == pytest.approx(a.reward, rel=1e-9)


# ---------------------------------------------------------------------------
# Gate 1
# ---------------------------------------------------------------------------


def _row(box, arm, control, outcomes):
    return {"box": box, "arm": arm, "control": control,
            "outcomes": [o.to_dict() for o in outcomes]}


def _succ(host):
    return ProposalOutcome(0, host, detected=True, host_matched=True,
                           feasible_field=True, reward=1.5, new_sub_id=99)


def _fail(host):
    return ProposalOutcome(0, host, detected=False, host_matched=True,
                           feasible_field=True, reward=0.0)


def test_gate1_passes_only_with_breadth_and_a_beaten_control():
    rows = [_row("set8", "random", "none", [_succ(1), _succ(2), _succ(3)]),
            _row("set9", "random", "none", [_succ(4), _succ(5)]),
            _row("set8", "random", "random_particles", [_fail(1), _fail(2)])]
    v = gate1_verdict(rows)
    assert v["pass"], v["checks"]
    assert v["n_successes"] == 5 and len(v["hosts"]) == 5 and len(v["boxes"]) == 2


def test_gate1_fails_when_all_the_successes_are_in_one_host():
    rows = [_row("set8", "random", "none", [_succ(1)] * 6),
            _row("set8", "random", "random_particles", [_fail(1)])]
    v = gate1_verdict(rows)
    assert not v["pass"]
    assert not v["checks"]["n_hosts"] and not v["checks"]["n_boxes"]


def test_gate1_fails_when_the_random_particle_control_does_just_as_well():
    rows = [_row("set8", "random", "none", [_succ(1), _succ(2), _succ(3)]),
            _row("set9", "random", "none", [_succ(4), _succ(5)]),
            _row("set8", "random", "random_particles", [_succ(1), _succ(2)])]
    v = gate1_verdict(rows)
    assert not v["checks"]["beats_random_particle_control"]
    assert not v["pass"]


def test_gate1_reports_a_final_eval_box_as_a_failure():
    rows = [_row("set13", "random", "none", [_succ(1)])]
    v = gate1_verdict(rows, forbidden_boxes=["set13", "set14", "set15"])
    assert not v["pass"]
    assert v["forbidden_boxes_touched"] == ["set13"]
