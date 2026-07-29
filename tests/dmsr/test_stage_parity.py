"""Stage C and Stage D must differ in exactly one thing.

The headline claim of this stage rests on C-vs-D being a controlled comparison.
That control is easy to break by editing one config and forgetting the other, so
it is asserted here rather than trusted to review.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from cosmo_sr.train.train_dmsr import ALL_LR_STAGES, lambda_adv_at
from cosmo_sr.utils.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "dmsr"
C_CFG = CONFIG_DIR / "stage_c_critic_pairedlr.yaml"
D_CFG = CONFIG_DIR / "stage_d_critic_alllr.yaml"


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


@pytest.fixture(scope="module")
def configs():
    return load_config(C_CFG), load_config(D_CFG)


def test_c_and_d_differ_only_in_stage_and_run_dir(configs):
    """The ONLY permitted differences are the stage selector and output paths."""
    c, d = configs
    fc, fd = _flatten(c), _flatten(d)
    assert set(fc) == set(fd), f"config key sets differ: {set(fc) ^ set(fd)}"

    allowed = {"stage", "output.run_dir", "wandb.name"}
    differing = {k for k in fc if fc[k] != fd[k]}
    assert differing <= allowed, (
        f"Stage C and D differ in {sorted(differing - allowed)}; the comparison is "
        "only valid if every training-relevant setting is identical"
    )
    assert fc["stage"] == "c" and fd["stage"] == "d"


@pytest.mark.parametrize("key", [
    "train.steps", "train.seed", "train.batch_size", "train.lr", "train.betas",
    "train.ema_decay", "train.grad_clip",
    "adv.lambda_adv", "adv.start_step", "adv.ramp_steps", "adv.n_critic",
    "adv.gen_ode_steps", "adv.critic_warmup_steps",
    "critic.width", "critic.n_layers", "critic.lr", "critic.r1_gamma",
    "critic.r1_interval", "critic.lowpass",
    "data.crop_lr", "data.use_channels",
    "model.init_from", "model.condition_encoder_init", "model.encoder_ckpt",
    "eval.n_steps", "eval.best_metric",
])
def test_specific_settings_are_matched(configs, key):
    c, d = configs
    fc, fd = _flatten(c), _flatten(d)
    assert fc[key] == fd[key], f"{key}: C={fc[key]!r} vs D={fd[key]!r}"


def test_lambda_adv_schedule_is_identical(configs):
    c, d = configs
    for step in (0, 1, 100, 500, 1000, 2000, 5000, 20000):
        assert lambda_adv_at(step, c) == lambda_adv_at(step, d)


def test_only_stage_d_uses_lr_only_data(configs):
    c, d = configs
    assert c["stage"] not in ALL_LR_STAGES
    assert d["stage"] in ALL_LR_STAGES


def _audit(cfg_path):
    out = subprocess.run(
        [sys.executable, "-m", "cosmo_sr.train.train_dmsr",
         "--config", str(cfg_path), "--audit-compute"],
        capture_output=True, text=True, check=True,
        cwd=str(CONFIG_DIR.parents[1]),
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
    )
    import json
    return json.loads(out.stdout)


def test_compute_budgets_match():
    """Generator and critic update counts must be equal between C and D."""
    a_c, a_d = _audit(C_CFG), _audit(D_CFG)
    for key in ("steps", "generator_updates", "paired_generator_updates",
                "second_stream_generator_updates", "critic_updates",
                "batch_size", "crop_lr"):
        assert a_c[key] == a_d[key], f"{key}: C={a_c[key]} D={a_d[key]}"
    # ...and the one thing that must differ:
    assert a_c["second_stream_source"] == "paired_repeat"
    assert a_d["second_stream_source"] == "lr_only"
