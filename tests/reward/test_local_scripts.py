"""The script-level guards, which are the ones that stop a bad run from starting.

Cheap checks, but they cover the two ways this pipeline could produce numbers
that look fine and are not: scoring against placeholder feasibility thresholds,
and touching the held-out boxes before the final comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "reward"))

from _local_common import (  # noqa: E402
    DEFAULT_CONFIG, assert_no_final_boxes, assert_training_boxes, codec_for,
    mode_plan, require_calibrated_constraints,
)

CFG = yaml.safe_load(DEFAULT_CONFIG.read_text())


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


UNCALIBRATED = {**CFG, "constraints": {"calibrated": False,
                                       "low_k_change_max": None}}


def test_the_committed_config_carries_measured_thresholds():
    """Calibrated 2026-08-02 from 18 editor candidates, 18 successful HR-oracle
    interventions, 6 controls and 2 frozen anchors over set8/set9.

    The flag alone is not the point -- anyone can set a boolean -- so this also
    checks the thresholds are not the reward line's whole-field placeholders,
    which is the specific mistake the guard exists to prevent.
    """
    c = CFG["constraints"]
    assert c["calibrated"] is True
    assert c["low_k_change_max"] is not None
    assert c["low_k_change_max"] != 0.02, "that is reward.yaml's whole-field value"
    # HR-referenced constraints stay disabled: measuring them would make the
    # feasibility filter illegal for a deployment-legal editor.
    assert c["density_power_error_max"] is None
    assert c["displacement_power_error_max"] is None


def test_the_thresholds_still_accept_the_oracle_successes():
    """The plan's requirement on the low-k bound: an intervention already shown
    to restore a real subhalo may not be called infeasible. Measured oracle
    successes reached 5.835e-2."""
    assert CFG["constraints"]["low_k_change_max"] >= 5.835e-2


def test_a_scoring_script_refuses_to_run_uncalibrated():
    with pytest.raises(SystemExit, match="calibrated: false"):
        require_calibrated_constraints(UNCALIBRATED, script="test")


def test_the_guard_passes_once_thresholds_are_measured():
    require_calibrated_constraints(CFG, script="test")     # must not raise


def test_the_error_message_does_not_offer_the_reward_lines_thresholds():
    with pytest.raises(SystemExit) as e:
        require_calibrated_constraints(UNCALIBRATED, script="test")
    assert "Do NOT copy the thresholds from configs/reward/reward.yaml" in str(e.value)


@pytest.mark.parametrize("box", ["set13", "set14", "set15"])
def test_the_final_eval_boxes_are_refused(box):
    with pytest.raises(SystemExit, match="final-eval"):
        assert_no_final_boxes(CFG, ["set8", box], script="test")


def test_the_search_boxes_are_allowed():
    assert_no_final_boxes(CFG, CFG["search_boxes"], script="test")


def test_the_token_library_may_only_be_built_from_training_boxes():
    assert_training_boxes(CFG, CFG["tokens"]["library_boxes"], script="test")
    with pytest.raises(SystemExit, match="training boxes"):
        assert_training_boxes(CFG, ["set8"], script="test")


# ---------------------------------------------------------------------------
# Config <-> code
# ---------------------------------------------------------------------------


def test_every_search_parameter_has_a_bound_in_the_yaml():
    """A missing entry would silently fall back to the module default, so the
    committed bounds would not be the bounds that ran."""
    declared = set(CFG["editor"]["bounds"])
    assert declared == set(codec_for(CFG, "both").names)


def test_the_yaml_bounds_are_what_the_codec_uses():
    codec = codec_for(CFG, "both")
    lo, hi = CFG["editor"]["bounds"]["source_radius_rvir"]
    p = {q.name: q for q in codec.params}["source_radius_rvir"]
    assert (p.lo, p.hi) == (lo, hi)


def test_the_mode_budget_is_allocated_exactly_and_favours_the_joint_mode():
    for n in (8, 12, 28, 32):
        plan = mode_plan(CFG, n)
        assert len(plan) == n
        assert plan.count("both") == max(plan.count(m)
                                         for m in ("disp", "both", "vel"))


def test_every_mode_keeps_a_nonzero_budget():
    """Each mode answers a different question, so none may be starved to zero.

    ``disp`` is the control for the coherence argument: the measured selection
    hands the editor particles at the host's full velocity dispersion (~613 km/s
    against an ~89 km/s target), so displacement alone should not be able to
    bind an object. If it ever does, that reasoning is wrong and it matters more
    than the rest of the round.
    """
    for n in (12, 28, 32):
        plan = mode_plan(CFG, n)
        for m in ("disp", "both", "vel"):
            assert plan.count(m) > 0, (m, n, plan)


def test_the_joint_mode_can_reach_the_cooling_the_physics_requires():
    """kappa_v ~ 0.82-0.91 is needed to bring the selected set to a realistic
    subhalo dispersion. A cap below that puts the viable region outside the
    search space entirely, which is what the original 0.60 did."""
    cur = {p.name: p for p in codec_for(CFG, "both").params}
    assert cur["velocity_cooling"].hi >= 0.90


def test_the_joint_mode_still_cannot_become_velocity_only():
    """The surviving sense of 'displacement-dominant': cooling stays capped
    below the contraction ceiling, so `both` and `vel` remain distinguishable."""
    cur = {p.name: p for p in codec_for(CFG, "both").params}
    assert cur["velocity_cooling"].hi < cur["contraction"].hi


def test_reward_weights_config_matches_the_dataclass_fields():
    from cosmo_sr.reward.local_reward import LocalRewardConfig, load_local_reward_config
    known = set(LocalRewardConfig().to_dict())
    unknown = set(CFG["reward"]) - known
    # reliable_host_bins / upper_reliable_host_bins are reporting knobs read by
    # the scripts, not reward tolerances; anything else is a typo.
    assert unknown == {"reliable_host_bins", "upper_reliable_host_bins"}
    c = load_local_reward_config(CFG["reward"])
    assert c.lambda_host == CFG["reward"]["lambda_host"]


# ---------------------------------------------------------------------------
# The scripts import at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "select_editor_hosts", "extract_editor_members", "run_editor_candidates",
    "aggregate_cem_round", "train_action_flow", "evaluate_local_editor",
    "audit_local_editor_constraints",
])
def test_each_pipeline_script_imports_and_declares_its_arguments(name):
    import importlib
    m = importlib.import_module(name)
    assert hasattr(m, "main")
    with pytest.raises(SystemExit):
        m.main(["--help"])
