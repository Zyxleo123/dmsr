"""The projection oracle's arm plan, field construction and decision rule.

The expensive parts (Rockstar, 512^3 boxes) are not tested here -- they are the
job's whole content and belong to the CPU array. What is tested is everything
that could make the *conclusion* wrong while the job still ran to completion:
the arms enumerated, the identity ``alpha = 1 -> Psi_HR``, the pairing of the
bootstrap, and the rule that reads "damaged" off a CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from cosmo_sr.reward.projection import (DEFAULT_ALPHAS, PRIMARY_METRICS, MetricSpec,
                                        ProjectionArm, arm_plan, bootstrap_ci,
                                        choose_alpha, compare_to_reference,
                                        paired_bootstrap_diff, per_box_means,
                                        project_residual_field, reference_arm_name)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "reward"))

F, N, C = 4, 16, 6


def pair(seed=0, n=N):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(C, n, n, n)).astype(np.float32)
    hr = base + 0.1 * rng.normal(size=(C, n, n, n)).astype(np.float32)
    return hr, base


# --------------------------------------------------------------------------- #
# Arm plan
# --------------------------------------------------------------------------- #
def test_arm_plan_covers_the_three_sweeps_and_the_anchors():
    arms = arm_plan()
    names = [a.name for a in arms]
    assert "sr2" in names and "hr" in names
    assert {a.sweep for a in arms} == {"reference", "joint", "disp_only", "vel_only"}

    joint = [a for a in arms if a.sweep == "joint"]
    assert sorted(a.alpha_disp for a in joint) == sorted(DEFAULT_ALPHAS)
    assert all(a.alpha_disp == a.alpha_vel for a in joint)

    disp = [a for a in arms if a.sweep == "disp_only"]
    assert all(a.alpha_vel == 1.0 for a in disp)
    vel = [a for a in arms if a.sweep == "vel_only"]
    assert all(a.alpha_disp == 1.0 for a in vel)


def test_the_shared_alpha_one_point_is_emitted_once():
    """All three sweeps meet at Psi_HR; running it three times would waste a
    Rockstar pass and put three copies of the reference in the bootstrap."""
    arms = [a for a in arm_plan() if a.sweep != "reference"]
    ones = [a for a in arms if a.alpha_disp == 1.0 and a.alpha_vel == 1.0]
    assert len(ones) == 1
    assert reference_arm_name(arm_plan()) == ones[0].name


def test_arm_names_are_unique_and_filesystem_safe():
    arms = arm_plan()
    names = [a.name for a in arms]
    assert len(names) == len(set(names))
    assert all(set(n) <= set("abcdefghijklmnopqrstuvwxyz0123456789_") for n in names)


def test_arm_plan_rejects_bad_input():
    with pytest.raises(ValueError, match="unknown sweep"):
        arm_plan(sweeps=("everything",))
    with pytest.raises(ValueError, match="outside"):
        arm_plan(alphas=(0.0, 1.5))


def test_reference_arm_name_requires_an_alpha_one_arm():
    with pytest.raises(ValueError, match="no alpha=1 arm"):
        reference_arm_name(arm_plan(alphas=(0.0, 0.5)))


# --------------------------------------------------------------------------- #
# Field construction
# --------------------------------------------------------------------------- #
def test_alpha_one_reproduces_hr():
    """``T_1 = I``, so the upper reference IS the paired field."""
    hr, base = pair(1)
    x = project_residual_field(hr, base, alpha_disp=1.0, alpha_vel=1.0,
                               scale_factor=F, slab=8)
    assert np.abs(x - hr).max() < 1e-5


def test_alpha_zero_leaves_the_coarse_field_at_the_baseline():
    """A hard null projection cannot move anything the LR field sees."""
    hr, base = pair(2)
    x = project_residual_field(hr, base, alpha_disp=0.0, alpha_vel=0.0,
                               scale_factor=F, slab=8)
    from cosmo_sr.reward.fields import block_average
    assert np.abs(block_average(x, F) - block_average(base, F)).max() < 1e-5
    # ... while the within-block part is HR's, not the baseline's.
    assert np.abs(x - base).max() > 1e-3


def test_displacement_and_velocity_alphas_act_on_their_own_channels():
    hr, base = pair(3)
    from cosmo_sr.reward.fields import block_average
    x = project_residual_field(hr, base, alpha_disp=0.0, alpha_vel=1.0,
                               scale_factor=F, slab=8)
    a_x, a_b, a_hr = (block_average(v, F) for v in (x, base, hr))
    assert np.abs(a_x[0:3] - a_b[0:3]).max() < 1e-5      # displacement held
    assert np.abs(a_x[3:6] - a_hr[3:6]).max() < 1e-5     # velocity free


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.25, 0.5, 1.0])
def test_coarse_component_interpolates_linearly_in_alpha(alpha):
    hr, base = pair(4)
    from cosmo_sr.reward.fields import block_average
    x = project_residual_field(hr, base, alpha_disp=alpha, alpha_vel=alpha,
                               scale_factor=F, slab=8)
    want = block_average(base, F) + alpha * (block_average(hr, F) - block_average(base, F))
    assert np.abs(block_average(x, F) - want).max() < 1e-5


def test_slab_size_does_not_change_the_result():
    """The streaming assembly must not depend on how it was chunked."""
    hr, base = pair(5)
    a = project_residual_field(hr, base, alpha_disp=0.25, alpha_vel=0.5,
                               scale_factor=F, slab=4)
    b = project_residual_field(hr, base, alpha_disp=0.25, alpha_vel=0.5,
                               scale_factor=F, slab=N)
    assert np.array_equal(a, b)


def test_a_slab_that_splits_a_block_is_refused():
    hr, base = pair(6)
    with pytest.raises(ValueError, match="multiple of scale_factor"):
        project_residual_field(hr, base, alpha_disp=0.5, alpha_vel=0.5,
                               scale_factor=F, slab=F + 1)


def test_shape_mismatch_is_refused():
    hr, base = pair(7)
    with pytest.raises(ValueError, match="!="):
        project_residual_field(hr[:, :8], base, alpha_disp=1.0, alpha_vel=1.0,
                               scale_factor=F, slab=8)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def test_seeds_are_averaged_within_a_box_before_the_bootstrap():
    rows = [
        {"box": "set8", "m": 1.0}, {"box": "set8", "m": 3.0},   # -> 2.0
        {"box": "set9", "m": 10.0},
    ]
    assert per_box_means(rows, "m") == {"set8": 2.0, "set9": 10.0}


def test_per_box_means_drops_non_finite_values():
    rows = [{"box": "a", "m": float("nan")}, {"box": "a", "m": 2.0},
            {"box": "b", "m": float("inf")}]
    assert per_box_means(rows, "m") == {"a": 2.0}


def test_single_box_gets_no_confidence_interval():
    """One box says nothing about box-to-box scatter; a zero-width CI would lie."""
    d = bootstrap_ci({"set8": 1.0})
    assert d["n_boxes"] == 1 and np.isnan(d["lo"]) and np.isnan(d["hi"])


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    vals = {f"set{i}": float(i) for i in range(8)}
    a = bootstrap_ci(vals, n_boot=500, seed=3)
    b = bootstrap_ci(vals, n_boot=500, seed=3)
    assert a == b
    assert a["lo"] <= a["mean"] <= a["hi"]


def test_pairing_resolves_a_shift_that_unpaired_scatter_hides():
    """The reason comparisons are paired, as a test rather than a comment."""
    boxes = [f"set{i}" for i in range(8)]
    big = {b: 100.0 * i for i, b in enumerate(boxes)}          # box-to-box scatter
    arm = {b: v + 1.0 for b, v in big.items()}                 # constant +1 effect

    paired = paired_bootstrap_diff(arm, big, n_boot=500, seed=0)
    assert paired["lo"] > 0.0 and paired["hi"] < 2.0           # resolved

    unpaired_lo = bootstrap_ci(arm, n_boot=500, seed=0)["lo"]
    unpaired_hi = bootstrap_ci(big, n_boot=500, seed=0)["hi"]
    assert unpaired_lo < unpaired_hi                            # swamped


def test_paired_diff_uses_only_boxes_present_in_both():
    d = paired_bootstrap_diff({"a": 1.0, "b": 2.0, "c": 9.0}, {"a": 0.0, "b": 1.0})
    assert d["n_paired_boxes"] == 2


# --------------------------------------------------------------------------- #
# Verdicts and the decision rule
# --------------------------------------------------------------------------- #
def _rows(arm_values, metric="R_occ_reliable", n_boxes=8):
    out = []
    for arm, offset in arm_values.items():
        for i in range(n_boxes):
            out.append({"arm": arm, "box": f"set{i}", metric: 100.0 * i + offset})
    return out


def test_verdicts_read_the_right_side_of_the_interval():
    rows = _rows({"a": 0.0, "ref": 0.0, "worse": -5.0, "better": +5.0})
    m = MetricSpec("R_occ_reliable", True)
    assert compare_to_reference(rows, m, "a", "ref")["verdict"] == "indistinguishable"
    assert compare_to_reference(rows, m, "worse", "ref")["verdict"] == "damaged"
    assert compare_to_reference(rows, m, "better", "ref")["verdict"] == "improved"


def test_lower_is_better_metrics_flip_the_verdict():
    rows = _rows({"ref": 0.0, "high": +5.0}, metric="density_power_error")
    m = MetricSpec("density_power_error", False)
    assert compare_to_reference(rows, m, "high", "ref")["verdict"] == "damaged"


def test_one_box_is_undetermined_not_indistinguishable():
    """Too little data to see a difference is not evidence of no difference."""
    rows = _rows({"ref": 0.0, "a": 0.0}, n_boxes=1)
    v = compare_to_reference(rows, MetricSpec("R_occ_reliable", True), "a", "ref")
    assert v["verdict"] == "undetermined"


def _sweep_rows(by_alpha, metrics=PRIMARY_METRICS, n_boxes=8):
    """Rows for a joint sweep where every primary metric shifts by `offset`."""
    rows = []
    for alpha, offset in by_alpha.items():
        name = "joint_a" + f"{alpha:g}".replace(".", "p")
        for i in range(n_boxes):
            r = {"arm": name, "box": f"set{i}"}
            for m in metrics:
                r[m.name] = 100.0 * i + (offset if m.higher_is_better else -offset)
            rows.append(r)
    return rows


def test_hard_null_rejected_when_alpha_zero_damages_a_primary_metric():
    rows = _sweep_rows({0.0: -10.0, 0.1: 0.0, 0.25: 0.0, 0.5: 0.0, 1.0: 0.0})
    d = choose_alpha(rows, arm_plan(), sweep="joint", n_boot=500)
    assert d["hard_null_rejected"] is True
    assert set(d["hard_null_damaged_metrics"]) == {m.name for m in PRIMARY_METRICS}
    # The smallest allowance that is indistinguishable from alpha=1 is chosen.
    assert d["recommended"]["alpha"] == pytest.approx(0.1)


def test_smallest_indistinguishable_alpha_is_chosen():
    rows = _sweep_rows({0.0: -10.0, 0.1: -10.0, 0.25: 0.0, 0.5: 0.0, 1.0: 0.0})
    d = choose_alpha(rows, arm_plan(), sweep="joint", n_boot=500)
    assert d["recommended"]["alpha"] == pytest.approx(0.25)
    assert "joint_a0" in d["blocked_by"] and "joint_a0p1" in d["blocked_by"]


def test_hard_null_accepted_when_nothing_is_damaged():
    rows = _sweep_rows({0.0: 0.0, 0.1: 0.0, 0.25: 0.0, 0.5: 0.0, 1.0: 0.0})
    d = choose_alpha(rows, arm_plan(), sweep="joint", n_boot=500)
    assert d["hard_null_rejected"] is False
    assert d["recommended"]["alpha"] == pytest.approx(0.0)


def test_no_affordable_allowance_falls_back_to_alpha_one():
    """If every reduction damages something, none is recommended."""
    rows = _sweep_rows({0.0: -10.0, 0.1: -10.0, 0.25: -10.0, 0.5: -10.0, 1.0: 0.0})
    d = choose_alpha(rows, arm_plan(), sweep="joint", n_boot=500)
    assert d["recommended"]["alpha"] == 1.0
    assert "note" in d["recommended"]


def test_a_metric_that_is_undetermined_does_not_pass_the_gate():
    rows = _sweep_rows({0.0: 0.0, 1.0: 0.0}, n_boxes=1)
    d = choose_alpha(rows, arm_plan(alphas=(0.0, 1.0)), sweep="joint", n_boot=500)
    assert d["recommended"]["alpha"] == 1.0


def test_displacement_and_velocity_sweeps_are_decided_separately():
    """A velocity-only failure must not move the displacement recommendation."""
    rows = []
    for a, off in {0.0: 0.0, 0.1: 0.0, 0.25: 0.0, 0.5: 0.0, 1.0: 0.0}.items():
        for i in range(8):
            r = {"arm": "disp_a" + f"{a:g}".replace(".", "p"), "box": f"set{i}"}
            for m in PRIMARY_METRICS:
                r[m.name] = 100.0 * i + off
            rows.append(r)
    for a, off in {0.0: -20.0, 0.1: -20.0, 0.25: 0.0, 0.5: 0.0}.items():
        for i in range(8):
            r = {"arm": "vel_a" + f"{a:g}".replace(".", "p"), "box": f"set{i}"}
            for m in PRIMARY_METRICS:
                r[m.name] = 100.0 * i + (off if m.higher_is_better else -off)
            rows.append(r)
    for i in range(8):                                   # the shared alpha=1 arm
        r = {"arm": "joint_a1", "box": f"set{i}"}
        for m in PRIMARY_METRICS:
            r[m.name] = 100.0 * i
        rows.append(r)

    arms = arm_plan()
    d_disp = choose_alpha(rows, arms, sweep="disp_only", n_boot=500)
    d_vel = choose_alpha(rows, arms, sweep="vel_only", n_boot=500)
    assert d_disp["recommended"]["alpha"] == pytest.approx(0.0)
    assert d_vel["recommended"]["alpha"] == pytest.approx(0.25)
    assert d_vel["hard_null_rejected"] and not d_disp["hard_null_rejected"]


# --------------------------------------------------------------------------- #
# Script-level guards
# --------------------------------------------------------------------------- #
def test_the_audit_script_declares_it_writes_no_training_data():
    """The scope restriction is the point of the experiment, so it is pinned."""
    import audit_projection_oracle as aud

    doc = (aud.__doc__ or "").lower()
    assert "no training data" in doc or "writes no training" in doc


def test_report_primary_metrics_are_a_subset_of_what_is_reported():
    import projection_oracle_report as rep

    reported = {m.name for m in rep.REPORTED}
    assert {m.name for m in PRIMARY_METRICS} <= reported


def test_config_alphas_include_the_endpoints():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "reward" / "projection_oracle.yaml").read_text())
    alphas = cfg["projection_oracle"]["alphas"]
    assert 0.0 in alphas and 1.0 in alphas, (
        "the sweep needs both the hard null projection and the HR reference"
    )
    assert cfg["projection_oracle"]["sweeps"] == ["joint", "disp_only", "vel_only"]
    assert len(cfg["projection_oracle"]["base_seeds"]) >= 2, (
        "a conclusion from one SR2 seed is a statement about that realisation"
    )
