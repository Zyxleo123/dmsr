"""Script-level guards for the reward-only Gaussian line.

These are the checks that stop a bad run from starting or a good run from
being misread: the manifest contract with the shared scoring stage, the support
gate's statistics, the variance floor, and the config promises that "reward-only"
and "not block_null by default" are actually true of the committed files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "reward"))

from cosmo_sr.reward.correction import CorrectionConfig, CorrectionScales
from cosmo_sr.reward.gaussian_policy import (GaussianPolicyConfig,
                                             MultiScaleGaussianPolicy)

import gaussian_support_gate as gate            # noqa: E402
import sample_gaussian_candidates as sampler    # noqa: E402
import train_gaussian_reward as trainer         # noqa: E402

CFG_PATH = ROOT / "configs" / "reward" / "gaussian_policy.yaml"
CFG = yaml.safe_load(CFG_PATH.read_text())


# --------------------------------------------------------------------------- #
# The committed config
# --------------------------------------------------------------------------- #
def test_config_has_no_paired_hr_term():
    """'Reward-only' has to be a property of the file, not of the prose."""
    text = CFG_PATH.read_text().lower()
    for banned in ("hr_weight", "supervised_weight", "lambda_hr", "recon_weight",
                   "paired_weight"):
        assert banned not in text, f"{banned} would reintroduce paired supervision"
    assert "beta_kl" in CFG["train"] and "lambda_edit" in CFG["train"]


def test_block_null_is_not_the_committed_default():
    assert CFG["correction"]["mode"] == "block_leaky"
    assert CFG["correction"]["alpha_disp"] == 1.0
    assert CFG["correction"]["alpha_vel"] == 1.0


def test_amplitude_scales_come_from_a_calibration_artifact():
    """The bound may only ever be a measurement, never a literal in this file.

    ``scales_path`` is allowed to be null (nothing calibrated yet -- the sampler
    refuses to run) or a path to the calibration output. What is never allowed
    is an inline ``scales`` block: a number written here would silently set the
    size of every edit the policy can propose, and would drift from the
    measurement it was copied from.
    """
    assert "scales" not in CFG["correction"], (
        "inline scales in the config; point scales_path at "
        "calibrate_correction_scales.py's output instead"
    )
    p = CFG["correction"]["scales_path"]
    assert p is None or str(p).endswith(".json")


def test_exploration_starts_nonzero_and_coarse_is_smallest():
    s = CFG["model"]["sigma_init"]
    assert all(v > 0 for v in s.values())
    assert s["coarse"] < min(s["middle"], s["fine"])
    assert CFG["model"]["sigma_min"] > 0


def test_config_builds_a_policy():
    corr = dict(CFG["correction"])
    corr.pop("scales_path", None)
    corr["scales"] = CorrectionScales(calibrated=True).to_dict()
    p = sampler.build_policy({"model": CFG["model"], "correction": corr})
    assert isinstance(p, MultiScaleGaussianPolicy)
    assert p.cfg.width == 48 and p.cfg.num_levels == 2
    assert p.cfg.blocks_per_level == 2 and p.cfg.use_attention is False


def test_uncalibrated_scales_stop_the_sampler_before_it_runs():
    """The guard that makes the test above safe: no path, no run."""
    corr = dict(CFG["correction"])
    corr.pop("scales_path", None)
    with pytest.raises(SystemExit, match="calibrated"):
        sampler.build_policy({"model": CFG["model"], "correction": corr})


def test_the_committed_scales_path_resolves_to_a_calibrated_measurement():
    """Skipped until Phase 1 has run; asserts the artifact really is calibrated."""
    p = CFG["correction"]["scales_path"]
    if p is None:
        pytest.skip("no calibration run yet (scales_path is null)")
    if not Path(p).is_file():
        pytest.skip(f"calibration artifact not present on this host: {p}")
    from cosmo_sr.reward.correction import load_correction_scales

    s = load_correction_scales(p)
    assert s.calibrated, "scales_path points at an UNcalibrated file"
    assert s.boxes, "calibrated scales carry no provenance for which boxes fitted them"
    for v in (s.fine_disp, s.fine_vel, s.coarse_disp, s.coarse_vel):
        assert v > 0 and np.isfinite(v)
    # Coarse below fine in both groups -- the reason they are separate bounds.
    assert s.coarse_disp < s.fine_disp and s.coarse_vel < s.fine_vel


def test_support_gate_thresholds_match_the_spec():
    g = CFG["support_gate"]
    assert g["min_feasible_positive"] >= 5
    assert g["min_ess"] >= 8.0
    assert g["min_improved_reliable_bins"] >= 2
    assert g["min_improved_upper_bins"] >= 1


def test_amplitude_curriculum_has_small_medium_large():
    amps = CFG["sampling"]["amplitude_curriculum"]
    assert len(amps) >= 3 and amps == sorted(amps)
    assert amps[-1] <= 1.0


def test_train_config_keeps_a_variance_floor():
    assert CFG["train"]["sigma_floor"] > 0
    assert CFG["train"]["reference"] == "init"


# --------------------------------------------------------------------------- #
# The manifest contract with the shared scoring stage
# --------------------------------------------------------------------------- #
def test_candidate_rows_carry_exactly_what_score_oracle_reads():
    """score_oracle.py is shared with the supervised line and must not change."""
    ccfg = CorrectionConfig(scale_factor=8, channels=6, amplitude=0.5,
                            scales=CorrectionScales(calibrated=True))
    row = sampler._row("set8", 3, Path("/tmp/r.npy"), Path("/tmp/b.npy"), "fp",
                       0.5, ccfg, {"behavior_log_prob_sum": -1.0,
                                   "policy_hash": "abc"},
                       Path("/tmp/rec.json"), regenerated=True)
    for key in ("box", "seed", "residual", "base"):
        assert key in row, f"score_oracle.py reads {key!r}"
    assert isinstance(row["seed"], int)
    # ... and the Gaussian arm's own provenance rides alongside, ignored there.
    assert row["amplitude"] == 0.5
    assert row["projection_mode"] == "block_leaky"
    assert row["action_record"].endswith("rec.json")
    assert row["policy_hash"] == "abc"


# --------------------------------------------------------------------------- #
# Support-gate statistics
# --------------------------------------------------------------------------- #
def test_ess_is_n_for_equal_weights_and_one_for_a_spike():
    assert gate.effective_sample_size([1.0] * 8) == pytest.approx(8.0)
    assert gate.effective_sample_size([1.0, 1e-12, 1e-12]) == pytest.approx(1.0, rel=1e-3)
    assert gate.effective_sample_size([]) == 0.0


def test_ess_is_scale_invariant():
    w = [0.3, 1.2, 0.7, 2.0]
    assert gate.effective_sample_size(w) == pytest.approx(
        gate.effective_sample_size([5 * x for x in w]))


def test_ess_ignores_non_positive_weights():
    assert gate.effective_sample_size([1.0, 1.0, 0.0, -1.0]) == pytest.approx(2.0)


def test_relative_spread_is_zero_for_a_constant_and_nan_for_one_value():
    assert gate.relative_spread([3.0, 3.0, 3.0]) == pytest.approx(0.0)
    assert np.isnan(gate.relative_spread([3.0]))
    assert gate.relative_spread([1.0, 3.0]) > 0.0


def test_relative_spread_drops_non_finite_values():
    assert gate.relative_spread([1.0, 3.0, float("nan")]) == pytest.approx(
        gate.relative_spread([1.0, 3.0]))


# --------------------------------------------------------------------------- #
# The variance floor
# --------------------------------------------------------------------------- #
def _policy(sigma_min=1e-3) -> MultiScaleGaussianPolicy:
    cfg = GaussianPolicyConfig(
        channels=6, scale_factor=4, width=8, num_levels=2, blocks_per_level=1,
        num_groups=4, use_checkpoint=False, sigma_min=sigma_min,
        sigma_init={"coarse": 0.05, "middle": 0.15, "fine": 0.15})
    corr = CorrectionConfig(scale_factor=4, channels=6,
                            scales=CorrectionScales(calibrated=True))
    return MultiScaleGaussianPolicy(cfg, corr)


def test_sigma_floor_is_a_hard_pointwise_bound_not_a_bias_clamp():
    """After training the head weights are nonzero, so only clamping log sigma
    itself is still a floor."""
    p = _policy()
    trainer.apply_sigma_floor(p, 0.02)
    with torch.no_grad():
        p.logsig_heads["fine"].weight.normal_(0.0, 5.0)   # a trained, varying head
        p.logsig_heads["fine"].bias.fill_(-20.0)          # driven far below the floor
    base = torch.randn(1, 6, 16, 16, 16)
    lr = torch.randn(1, 6, 4, 4, 4)
    for _, sigma in p.distribution(base, lr).values():
        assert float(sigma.detach().min()) >= 0.02 - 1e-9


def test_sigma_floor_never_lowers_the_existing_minimum():
    p = _policy(sigma_min=0.05)
    assert trainer.apply_sigma_floor(p, 0.01) == pytest.approx(0.05)


def test_a_floor_above_sigma_max_is_refused():
    p = _policy()
    with pytest.raises(SystemExit, match="sigma_floor"):
        trainer.apply_sigma_floor(p, 100.0)


def test_zero_floor_is_a_no_op():
    p = _policy()
    assert trainer.apply_sigma_floor(p, 0.0) == pytest.approx(p.cfg.sigma_min)


# --------------------------------------------------------------------------- #
# Behaviour-policy identity
# --------------------------------------------------------------------------- #
class _Entry:
    def __init__(self, box, seed):
        self.box, self.residual_seed = box, seed


def test_a_wrong_behaviour_checkpoint_stops_the_run():
    """Reconstructing a_i under the wrong parameters would reinforce a different
    action than the one that earned w_i -- silently, and forever."""
    p = _policy()
    entries = [_Entry("set0", 1)]
    records = {"set0|1": {"policy_hash": "not-the-one"}}
    with pytest.raises(SystemExit, match="behaviour policy mismatch"):
        trainer.check_behavior_identity(p, records, entries)


def test_the_right_behaviour_checkpoint_passes():
    p = _policy()
    entries = [_Entry("set0", 1)]
    trainer.check_behavior_identity(
        p, {"set0|1": {"policy_hash": p.parameter_hash()}}, entries)


def test_missing_hashes_warn_rather_than_falsely_pass(capsys):
    p = _policy()
    trainer.check_behavior_identity(p, {"set0|1": {}}, [_Entry("set0", 1)])
    assert "cannot be verified" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Documented scope
# --------------------------------------------------------------------------- #
def test_the_trainer_declares_the_objective_it_implements():
    doc = (trainer.__doc__ or "").lower()
    assert "no hr reconstruction term" in doc
    assert "detached" in doc


def test_the_support_gate_declares_its_thresholds_are_not_truths():
    doc = (gate.__doc__ or "").lower()
    assert "report thresholds" in doc
    assert "diffusion" in doc      # the "don't assume diffusion fixes it" note


def test_the_sampler_reuses_the_shared_scoring_format():
    doc = (sampler.__doc__ or "").lower()
    assert "score_oracle" in doc and "reused" in doc
