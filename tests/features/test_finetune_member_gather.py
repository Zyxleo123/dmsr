"""The fine-tune's verdict logic.

``docs/sr2_gather_finetune.md`` section 3.4 records a run that printed
*ALL THREE HELD* while structure outside the supervised windows sat at 0.52 of
frozen, because the verdict read one guard and not the others. These tests pin
the four branches so that cannot recur: a damaged field must never report
success, and an in-sample-only result must never be reported as generalisation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reward"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from finetune_member_gather import verdict  # noqa: E402


# highk_max_holdout=0 is the historical train-only verdict, so every test below
# that does not name it is asserting the behaviour the four finished arms saw.
ARGS = argparse.Namespace(low_k_max=0.02, highk_max=1.5, vel_rms_tol=0.10,
                          highk_max_holdout=0.0, vel_highk_min=0.0)


def _row(bound_train, bound_hold, *, low_k=0.01, highk_max=1.1, vel=1.0,
         n_hold=3, highk_max_hold=None, vel_highk=1.0):
    return {
        "train": {"bound_hard": bound_train, "low_k": low_k,
                  "highk_ratio_max": highk_max, "vel_rms_ratio": vel,
                  "velhighk_ratio_min": vel_highk},
        "holdout": {"bound_hard": bound_hold, "n_hosts": n_hold,
                    "highk_ratio_max": (highk_max if highk_max_hold is None
                                        else highk_max_hold)},
    }


def test_guard_failure_dominates_even_when_the_objective_moved():
    """A wrecked field is not a weak result; nothing is readable from it."""
    first = _row(0.00, 0.00)
    last = _row(0.60, 0.55, highk_max=5.50)      # the free field's measured value
    v = verdict(first, last, ARGS)
    assert v["guards_held"] is False
    assert "GUARD FAILED" in v["text"]


def test_low_k_breach_is_a_guard_failure():
    v = verdict(_row(0.0, 0.0), _row(0.6, 0.55, low_k=0.5), ARGS)
    assert v["guards_held"] is False


def test_velocity_rms_breach_is_a_guard_failure():
    """Boundness pressure attacks velocity; cooling the field is the cheat."""
    v = verdict(_row(0.0, 0.0), _row(0.6, 0.55, vel=0.5), ARGS)
    assert v["guards_held"] is False


def test_not_moved_when_the_objective_did_not_rise_on_training_hosts():
    v = verdict(_row(0.00, 0.00), _row(0.02, 0.01), ARGS)
    assert v["moved"] is False
    assert "NOT MOVED" in v["text"]
    assert "Gating is premature" in v["text"]


def test_in_sample_only_is_named_as_such():
    """The failure mode this whole run exists to detect must be unmissable."""
    v = verdict(_row(0.00, 0.00), _row(0.60, 0.01), ARGS)
    assert v["moved"] is True
    assert v["generalised_by_surrogate"] is False
    assert "IN-SAMPLE ONLY" in v["text"]
    assert "answered NO" in v["text"]


def test_success_branch_still_demands_the_halo_finder():
    """The surrogate is expected to be gameable; it may never close the case."""
    v = verdict(_row(0.00, 0.00), _row(0.60, 0.55), ARGS)
    assert v["guards_held"] and v["moved"] and v["generalised_by_surrogate"]
    assert "WHOLE-BOX ROCKSTAR GATE" in v["text"]
    assert "surrogate" in v["text"]


def test_no_holdout_hosts_cannot_report_generalisation():
    """An empty held-out pool is not evidence of anything."""
    v = verdict(_row(0.00, 0.00, n_hold=0), _row(0.60, 0.00, n_hold=0), ARGS)
    assert v["generalised_by_surrogate"] is False
    assert "IN-SAMPLE ONLY" in v["text"]


def test_the_move_threshold_is_not_satisfied_by_noise():
    """0.05 in bound_frac against a frozen start of 0.002; a 0.04 drift is not
    a result."""
    assert verdict(_row(0.00, 0.00), _row(0.04, 0.04), ARGS)["moved"] is False
    assert verdict(_row(0.00, 0.00), _row(0.06, 0.06), ARGS)["moved"] is True


@pytest.mark.parametrize("hk", [1.49, 1.5])
def test_highk_gate_boundary_is_inclusive(hk):
    v = verdict(_row(0.0, 0.0), _row(0.6, 0.55, highk_max=hk), ARGS)
    assert v["guards_held"] is True


# --------------------------------------------------------------------------- #
# wandb mirroring
# --------------------------------------------------------------------------- #
from finetune_member_gather import (  # noqa: E402
    _POOL_SCALARS, _TERM_NAMES, log_wandb, wandb_row)


def _wpool(n_hosts=2, **over):
    d = {"label": "train", "n_hosts": n_hosts, "n_sets_total": 300,
         "gather": 1.83, "bound_hard": 0.42, "virial": 6.1,
         "centre_offset_radii": 0.3,
         "r_rms_over_hr": 1.1, "sigma_v_over_hr": 1.05, "low_k": 0.004,
         "highk_ratio": 1.2, "highk_ratio_max": 1.4, "vel_rms_ratio": 0.99,
         "term_virial": 0.9, "term_bound": 0.4, "term_d6": 0.2,
         "term_rrms": 0.13, "term_sigmav": 0.1, "term_centre": 0.1,
         # `term_raw_*` is the unweighted term the --term-norm scales are
         # measured from; `term_eff_*` is what actually entered the loss, and
         # under normalisation it is `term_eff_*`, not `term_*`, that sums to
         # `gather`. All three are mirrored so a budget question can be answered
         # from the charts without reopening metrics.jsonl.
         **{f"term_raw_{k}": v for k, v in
            (("virial", 0.9), ("bound", 0.4), ("d6", 0.2),
             ("rrms", 0.43), ("sigmav", 0.33), ("centre", 0.1))},
         **{f"term_eff_{k}": v for k, v in
            (("virial", 1.0), ("bound", 1.0), ("d6", 1.0),
             ("rrms", 0.3), ("sigmav", 0.3), ("centre", 1.0))},
         "per_host": [{"key": "set3:h1", "bound_hard": 0.40,
                       "centre_offset_radii": 0.2, "highk_ratio": 1.1},
                      {"key": "set3:h2", "bound_hard": 0.44,
                       "centre_offset_radii": 0.4, "highk_ratio": 1.3}]}
    d.update(over)
    return d


def _wrow(step=250):
    return {"step": step, "wall_s": 12.5, "batch_gather": 0.31,
            "train": _wpool(), "holdout": _wpool(n_hosts=1, bound_hard=0.39)}


def test_every_pooled_scalar_reaches_wandb_for_both_sides():
    """A metric silently missing from the mirror is invisible until the run is
    over and the charts are read."""
    flat = wandb_row(_wrow())
    for side in ("train", "holdout"):
        for k in _POOL_SCALARS:
            assert f"{side}/{k}" in flat, f"{side}/{k} not mirrored"


def test_step_and_batch_scalars_are_carried():
    flat = wandb_row(_wrow(step=500))
    assert flat["step"] == 500
    assert flat["wall_s"] == 12.5
    assert flat["batch_gather"] == 0.31


def test_row_zero_without_batch_keys_does_not_crash():
    """Step 0 is written before the loop and has no batch_gather/wall_s."""
    flat = wandb_row({"step": 0, "train": _wpool(), "holdout": _wpool()})
    assert flat["step"] == 0
    assert "batch_gather" not in flat and "wall_s" not in flat


def test_empty_holdout_pool_is_tolerated():
    row = {"step": 10, "train": _wpool(), "holdout": {"label": "holdout",
                                                    "n_hosts": 0,
                                                    "per_host": []}}
    flat = wandb_row(row)
    assert flat["holdout/n_hosts"] == 0
    assert "holdout/bound_hard" not in flat
    assert "train/bound_hard" in flat


def test_missing_side_entirely_is_tolerated():
    flat = wandb_row({"step": 1, "train": _wpool()})
    assert "train/bound_hard" in flat
    assert not any(k.startswith("holdout/") for k in flat)


def test_values_are_not_mutated_or_rescaled():
    row = _wrow()
    flat = wandb_row(row)
    assert flat["train/bound_hard"] == row["train"]["bound_hard"]
    assert flat["holdout/bound_hard"] == row["holdout"]["bound_hard"]


def test_per_host_detail_is_not_flattened_into_series():
    """Per-host goes in as a histogram, not 56 separate keys."""
    flat = wandb_row(_wrow())
    assert not any("set3:h1" in k for k in flat)


def test_log_wandb_is_a_noop_when_disabled():
    log_wandb(_wrow(), False)          # must not raise, must not import wandb


def test_log_wandb_never_raises_on_a_malformed_row():
    """metrics.jsonl is already written by the time this runs; it may not fail."""
    log_wandb({"step": "not-an-int"}, True)


# --------------------------------------------------------------------------- #
# The held-out high-k gate
# --------------------------------------------------------------------------- #
HOLD_ARGS = argparse.Namespace(low_k_max=0.02, highk_max=1.5, vel_rms_tol=0.10,
                               highk_max_holdout=1.5, vel_highk_min=0.0)


def test_train_only_verdict_passes_the_run_that_failed_out_of_sample():
    """The measured `all_blocks_self` case: 0.89x on train, 3.87x held out.

    This is not a hypothetical -- it is the row the finished run wrote, and the
    reason it was reported as a guard pass.
    """
    v = verdict(_row(0.00, 0.00),
                _row(0.60, 0.55, highk_max=0.89, highk_max_hold=3.87), ARGS)
    assert "GUARD FAILED" not in v["text"]


def test_holdout_gate_catches_it_and_names_both_numbers():
    v = verdict(_row(0.00, 0.00),
                _row(0.60, 0.55, highk_max=0.89, highk_max_hold=3.87),
                HOLD_ARGS)
    assert "GUARD FAILED" in v["text"]
    assert "0.89" in v["text"] and "3.87" in v["text"]


def test_holdout_gate_does_not_fire_when_the_field_held():
    v = verdict(_row(0.00, 0.00),
                _row(0.60, 0.55, highk_max=0.89, highk_max_hold=1.10),
                HOLD_ARGS)
    assert "GUARD FAILED" not in v["text"]


def test_holdout_gate_is_inert_with_no_holdout_hosts():
    """No held-out pool is a missing measurement, not a breach."""
    v = verdict(_row(0.00, 0.00),
                _row(0.60, 0.55, highk_max=0.89, highk_max_hold=99.0,
                     n_hold=0),
                HOLD_ARGS)
    assert "GUARD FAILED" not in v["text"]


# --------------------------------------------------------------------------- #
# The velocity small-scale power gate
# --------------------------------------------------------------------------- #
VEL_ARGS = argparse.Namespace(low_k_max=0.02, highk_max=1.5, vel_rms_tol=0.10,
                              highk_max_holdout=0.0, vel_highk_min=0.5)


def test_velocity_power_collapse_passed_the_historical_verdict():
    """The measured case: frozen SR2 is 1.02x HR here and every arm hit ~0.05x.

    `vel_rms_ratio` is one global std over the tile and read 0.71 through all of
    it, so the collapse had no criterion anywhere in the verdict.
    """
    v = verdict(_row(0.00, 0.00),
                _row(0.60, 0.55, vel=0.95, vel_highk=0.053), ARGS)
    assert "GUARD FAILED" not in v["text"]


def test_velocity_power_gate_catches_the_collapse():
    v = verdict(_row(0.00, 0.00),
                _row(0.60, 0.55, vel=0.95, vel_highk=0.053), VEL_ARGS)
    assert "GUARD FAILED" in v["text"]
    assert "0.053" in v["text"]


def test_velocity_power_gate_passes_a_field_that_kept_its_velocities():
    v = verdict(_row(0.00, 0.00),
                _row(0.60, 0.55, vel=0.95, vel_highk=0.90), VEL_ARGS)
    assert "GUARD FAILED" not in v["text"]


# --------------------------------------------------------------------------- #
# The box-wide (unsupervised-tile) high-k gate
# --------------------------------------------------------------------------- #
# The supervised held-out HOST tiles are still 6.25% of the box, so a run can
# clear `highk_max_holdout` while the box at large is corrupt. This gate reads
# the worst band on random UNSUPERVISED held-out tiles instead.
UNSUP_ARGS = argparse.Namespace(low_k_max=0.02, highk_max=1.5, vel_rms_tol=0.10,
                                highk_max_holdout=0.0, vel_highk_min=0.0,
                                unsup_highk_max=1.5)


def _with_unsup(row, ratio_max):
    row = dict(row)
    row["unsup"] = {"holdout": {"ratio_max": ratio_max}}
    return row


def test_unsup_gate_absent_attr_is_inert():
    """Old configs and the replay path have no `unsup_highk_max`; getattr=0=off."""
    v = verdict(_row(0.00, 0.00),
                _with_unsup(_row(0.60, 0.55), 9.9), ARGS)
    assert "GUARD FAILED" not in v["text"]


def test_supervised_holdout_can_pass_while_the_box_is_corrupt():
    """The case the arm exists for: 1.10x on held-out HOST tiles, 3.9x box-wide."""
    v = verdict(_row(0.00, 0.00),
                _with_unsup(_row(0.60, 0.55, highk_max=0.9, highk_max_hold=1.1),
                            3.9),
                HOLD_ARGS)                       # no unsup_highk_max on HOLD_ARGS
    assert "GUARD FAILED" not in v["text"]


def test_unsup_gate_catches_the_box_wide_excess():
    v = verdict(_row(0.00, 0.00),
                _with_unsup(_row(0.60, 0.55, highk_max=0.9, highk_max_hold=1.1),
                            3.9),
                UNSUP_ARGS)
    assert "GUARD FAILED" in v["text"]
    assert "unsup high-k max 3.90" in v["text"]


def test_unsup_gate_passes_a_field_that_generalised():
    v = verdict(_row(0.00, 0.00),
                _with_unsup(_row(0.60, 0.55), 1.20), UNSUP_ARGS)
    assert "GUARD FAILED" not in v["text"]


@pytest.mark.parametrize("r", [1.49, 1.5])
def test_unsup_gate_boundary_is_inclusive(r):
    v = verdict(_row(0.0, 0.0), _with_unsup(_row(0.6, 0.55), r), UNSUP_ARGS)
    assert v["guards_held"] is True


def test_unsup_gate_is_inert_with_no_unsup_pool():
    """No unsupervised pool built is a missing measurement, not a breach."""
    v = verdict(_row(0.00, 0.00), _row(0.60, 0.55), UNSUP_ARGS)
    assert "GUARD FAILED" not in v["text"]
