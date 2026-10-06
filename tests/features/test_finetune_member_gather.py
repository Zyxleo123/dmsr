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
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reward"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from finetune_member_gather import (  # noqa: E402
    HostTask, draw_seeds, fresh_draw_index, verdict, _box_to_tiles,
)


def test_box_to_tiles_matches_tile_particles_origin_convention():
    """The dump's full-box slicer must invert tile_particles' tile ordering, or
    the solved membership would be plotted against the wrong particles."""
    import numpy as np
    import torch
    from cosmo_sr.features.member_gather import tile_particles
    ng, th = 32, 8            # a toy box: 4^3 tiles of 8^3
    n_side = ng // th
    # A full box whose displacement channels are 0, so tile_particles returns the
    # Lagrangian grid q -- then a tile's own particle positions pin its origin.
    box = np.zeros((6, ng, ng, ng), dtype=np.float32)
    tiles = [0, 5, 21, 63 % (n_side ** 3)]
    tiles = [t for t in tiles if t < n_side ** 3]
    sub = _box_to_tiles(box, tiles, ng_hr=ng, tile_hr=th)
    assert sub.shape == (len(tiles), 6, th, th, th)
    pos, _ = tile_particles(torch.as_tensor(sub), tiles, ng_hr=ng, tile_hr=th,
                            boxsize_mpc_h=float(ng), dis_scale_mpc_h=1.0,
                            vel_scale_kms=1.0)
    cell = 1.0
    per = th ** 3
    for b, tid in enumerate(tiles):
        tx = tid // (n_side * n_side); ty = (tid // n_side) % n_side; tz = tid % n_side
        first = pos[b * per].numpy()            # local cell (0,0,0) of this tile
        expected = (np.array([tx, ty, tz]) * th + 0.5) * cell
        assert np.allclose(first, expected), (tid, first, expected)


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
         "radius_err_radii": 2.8, "radius_ratio": 0.97,
         # The SOFT shell's residual: the sorted-pairing log ratio, which is the
         # number to read for --w-shell. It moves where `radius_err_radii` does
         # not, a perfect distribution match leaving 79%/85% of the per-set
         # address error standing by construction.
         "shell_resid": 0.079, "shell_resid_signed": -0.041,
         "r_rms_over_hr": 1.1, "sigma_v_over_hr": 1.05, "low_k": 0.004,
         "highk_ratio": 1.2, "highk_ratio_max": 1.4, "vel_rms_ratio": 0.99,
         # The pedestal: how much non-member mass sits on the supervised
         # objects, against HR on the same particle ids. > 1 is the state
         # that makes Rockstar file them as somebody's subhalos.
         "iso_ratio": 1.34, "iso_above_hr": 0.71,
         # Soft membership. `soft_mass_ratio` is the constraint that stops a
         # candidate keeping a tiny cold core, and `soft_hr_fraction` is how
         # much of the solved object is still HR's own material -- 1.0 on the
         # hard arm. Present in every row so both arms chart together.
         "soft_mass_ratio": 1.0, "soft_mass_ratio_min": 0.96,
         "soft_hr_fraction": 0.82, "soft_n_eff_over_target": 0.94,
         "term_virial": 0.9, "term_bound": 0.4, "term_d6": 0.2,
         "term_rrms": 0.13, "term_sigmav": 0.1, "term_centre": 0.1,
         "term_radius": 0.0, "term_shell": 0.0, "term_isolation": 0.6,
         # `term_raw_*` is the unweighted term the --term-norm scales are
         # measured from; `term_eff_*` is what actually entered the loss, and
         # under normalisation it is `term_eff_*`, not `term_*`, that sums to
         # `gather`. All three are mirrored so a budget question can be answered
         # from the charts without reopening metrics.jsonl.
         **{f"term_raw_{k}": v for k, v in
            (("virial", 0.9), ("bound", 0.4), ("d6", 0.2),
             ("rrms", 0.43), ("sigmav", 0.33), ("centre", 0.1),
             ("radius", 2.4), ("shell", 0.071), ("isolation", 1.7))},
         **{f"term_eff_{k}": v for k, v in
            (("virial", 1.0), ("bound", 1.0), ("d6", 1.0),
             ("rrms", 0.3), ("sigmav", 0.3), ("centre", 1.0),
             ("radius", 0.0), ("shell", 0.0), ("isolation", 0.5))},
         # The splits. One series per (split, bin, field): the chart
         # `all_blocks_smallhalo` needed and did not have, since a median
         # pooled over both populations could not say whether its top-level
         # sets -- its entire hypothesis -- were moving.
         "by_population": {"sub": {"n": 180.0, "bound_hard": 0.44,
                                   "virial": 5.8,
                                   "centre_offset_radii": 0.28,
                                   "iso_ratio": 1.21},
                           "top": {"n": 120.0, "bound_hard": 0.31,
                                   "virial": 7.2,
                                   "centre_offset_radii": 0.35,
                                   "iso_ratio": 1.55}},
         "by_env": {"near": {"n": 40.0, "bound_hard": 0.29, "virial": 8.0,
                             "centre_offset_radii": 0.4,
                             "iso_ratio": 1.9}},
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


def test_the_population_and_environment_splits_reach_wandb():
    """The chart `all_blocks_smallhalo` needed and did not have.

    Its hypothesis was entirely about TOP-LEVEL halos and its eval reported one
    median pooled over both populations, so "the objective is wrong" and "the
    objective never got any budget" were indistinguishable until a Rockstar gate
    answered it ten hours later. They need different next runs.
    """
    flat = wandb_row(_wrow())
    assert flat["train/top/n"] == 120.0
    assert flat["train/top/iso_ratio"] == 1.55
    assert flat["holdout/sub/bound_hard"] == 0.44
    assert flat["train/near/iso_ratio"] == 1.9


def test_pooling_a_split_sums_counts_and_medians_the_rest():
    """A host contributing zero sets to a split must not drag its medians toward
    NaN, and must not silently vanish from the count either -- that distinction
    is what makes "the near bin got 4 sets" readable as a SELECTION failure
    rather than as a learning one."""
    from finetune_member_gather import _pool_splits

    rows = [{"by_env": {"near": {"n": 4.0, "bound_hard": 0.2, "virial": 3.0,
                                 "centre_offset_radii": 0.5, "iso_ratio": 1.1},
                        "field": {"n": 0.0, "bound_hard": float("nan"),
                                  "virial": float("nan"),
                                  "centre_offset_radii": float("nan"),
                                  "iso_ratio": float("nan")}}},
            {"by_env": {"near": {"n": 6.0, "bound_hard": 0.4, "virial": 5.0,
                                 "centre_offset_radii": 0.7, "iso_ratio": 1.3},
                        "field": {"n": 10.0, "bound_hard": 0.9, "virial": 2.0,
                                  "centre_offset_radii": 0.1,
                                  "iso_ratio": 0.8}}}]
    out = _pool_splits(rows, "by_env")
    assert out["near"]["n"] == 10.0
    assert out["near"]["bound_hard"] == pytest.approx(0.3)
    # The empty host is counted (0) but does not enter the median.
    assert out["field"]["n"] == 10.0
    assert out["field"]["bound_hard"] == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# The early Rockstar gate
# --------------------------------------------------------------------------- #
def test_save_atomic_leaves_no_partial_file_for_a_concurrent_reader():
    """The whole point of a mid-run snapshot is that a gate may read the run
    directory WHILE the trainer is writing it. A plain torch.save to a live path
    is a multi-second non-atomic write, and the export spends ~138 s reading; the
    collision is a torch unpickling error whose message says nothing about a
    race. os.replace is atomic on POSIX, so a reader sees the old file or the new
    one and never half of either."""
    import os
    import tempfile
    import torch
    from finetune_member_gather import save_atomic

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tuned.pt"
        save_atomic({"model": {"a": torch.zeros(3)}, "step": 1}, p)
        first = torch.load(p, map_location="cpu", weights_only=False)
        save_atomic({"model": {"a": torch.ones(3)}, "step": 2}, p)
        second = torch.load(p, map_location="cpu", weights_only=False)
        assert first["step"] == 1 and second["step"] == 2
        # No temp file survives either write -- a stray .tmp<pid> in the run dir
        # would be picked up by nothing, but it would sit on the same quota that
        # has already killed three stages of this pipeline.
        assert [f for f in os.listdir(d) if ".tmp" in f] == []


def test_save_atomic_cleans_up_when_the_write_fails():
    import os
    import tempfile
    from finetune_member_gather import save_atomic

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("boom")

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tuned.pt"
        with pytest.raises(RuntimeError, match="boom"):
            save_atomic({"model": Unpicklable()}, p)
        # The real path was never created, and no temp file is left behind: a
        # half-written .tmp that later got renamed would be the exact corruption
        # this function exists to prevent.
        assert not p.exists()
        assert os.listdir(d) == []


@pytest.mark.parametrize("raw,want", [
    ("", []), ("2000", [2000]), ("1000,2000", [1000, 2000]),
    (" 500 2000 ", [500, 2000]), ("2000,2000", [2000]),
])
def test_gate_steps_parses_and_dedupes(raw, want):
    from finetune_member_gather import build_parser, gate_steps

    a = build_parser().parse_args(["--gate-at-steps", raw, "--steps", "8000"])
    assert gate_steps(a) == want


@pytest.mark.parametrize("raw", ["8000", "9000", "0", "-100", "abc"])
def test_gate_steps_refuses_a_snapshot_that_would_never_be_written(raw):
    """A gate chain queued against a snapshot outside (0, --steps) strands four
    jobs on DependencyNeverSatisfied with nothing to read. Catch it at startup,
    where the message can say why."""
    from finetune_member_gather import build_parser, gate_steps

    a = build_parser().parse_args(["--gate-at-steps", raw, "--steps", "8000"])
    with pytest.raises(SystemExit):
        gate_steps(a)


# --------------------------------------------------------------------------- #
# The noise realisation: draw bookkeeping
#
# MEASURED 2026-09-08: the w_shell=30 arm drove its train residual 7.3x while
# holdout moved 1.4x. A run holds (LR tile, noise tile) constant for 8000 steps,
# so the noise is an input channel carrying real per-tile detail and zero
# generalisable information -- a memorisation channel. These pin the machinery
# that removes it, and above all pin that the DEFAULT is the old single draw.
# --------------------------------------------------------------------------- #
def _nargs(**kw):
    base = dict(seed=0, noise_draws=1, noise_mode="fixed",
                fresh_noise_eval=False, shell_noise_draws=0)
    base.update(kw)
    return argparse.Namespace(**base)


def test_a_default_run_has_exactly_one_draw_and_no_reserved_one():
    """Byte-identical to every arm before 2026-09-09, or nothing is comparable."""
    a = _nargs()
    assert draw_seeds(a) == [0]
    assert fresh_draw_index(a) is None


def test_draw_zero_is_always_the_run_seed():
    """Entry 0 must be `--seed` itself so the rotating path starts where the
    fixed path is, and a rotating run can be read against a fixed one."""
    for kw in ({}, dict(noise_draws=4), dict(fresh_noise_eval=True),
               dict(noise_draws=3, fresh_noise_eval=True)):
        assert draw_seeds(_nargs(seed=7, **kw))[0] == 7


def test_the_fresh_draw_is_reserved_after_every_training_draw():
    """The reserved draw must sit past the last TRAINED index, or the fresh eval
    would be measured on a realisation the run had already optimised against and
    the 2x2 would say nothing."""
    a = _nargs(noise_draws=3, fresh_noise_eval=True)
    seeds = draw_seeds(a)
    assert len(seeds) == 4
    assert fresh_draw_index(a) == 3
    assert fresh_draw_index(a) not in range(3)


def test_seeds_are_distinct_so_the_draws_are_distinct():
    seeds = draw_seeds(_nargs(seed=5, noise_draws=4, fresh_noise_eval=True))
    assert len(set(seeds)) == len(seeds) == 5


def test_fresh_eval_alone_still_reserves_a_draw():
    """`--fresh-noise-eval` is a measurement, usable without any rotation: it
    asks what the single-draw run already on disk would score on fresh noise."""
    a = _nargs(fresh_noise_eval=True)
    assert draw_seeds(a) == [0, 1000]
    assert fresh_draw_index(a) == 1


def test_sets_at_falls_back_to_the_single_table():
    """A task built before the draw axis existed must still be readable."""
    t = SimpleNamespace(sets="S", sets_by_draw=None, host_sets="H",
                        host_sets_by_draw=None)
    assert HostTask.sets_at(t, 0) == "S"
    assert HostTask.sets_at(t, 3) == "S"
    assert HostTask.host_sets_at(t, 2) == "H"


def test_sets_at_selects_the_draws_own_table():
    """`centre_self` and the hybrid reference are functions of the frozen field,
    so a draw MUST read its own table; borrowing draw 0's anchor would charge
    the run for ordinary realisation scatter."""
    t = SimpleNamespace(sets="S0", sets_by_draw=["S0", "S1", "S2"],
                        host_sets="H0", host_sets_by_draw=["H0", "H1", "H2"])
    assert [HostTask.sets_at(t, d) for d in range(3)] == ["S0", "S1", "S2"]
    assert [HostTask.host_sets_at(t, d) for d in range(3)] == ["H0", "H1", "H2"]
    assert HostTask.n_draws.fget(t) == 3
