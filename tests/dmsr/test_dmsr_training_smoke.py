"""Section 8 -- training smoke tests, run before any full job.

These deliberately exercise the **real** data path (on-disk boxes, box-level
split resolution, the LR crop pool, the balanced sampler) on miniature boxes,
rather than the trainer's ``--smoke`` synthetic shortcut. The Stage C/D test in
particular is worthless unless it actually loads files.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cosmo_sr.dmsr.critic import HRCritic, LazyR1, hinge_d_loss, hinge_g_loss
from cosmo_sr.dmsr.density import HighPassDensity, critic_input
from cosmo_sr.dmsr.flow import NullSpaceFlow, null_space_flow_loss

REPO = Path(__file__).resolve().parents[2]
FACTOR = 8
N_LR, N_HR = 8, 64
CHANNELS = 6


# --------------------------------------------------------------------------- #
# 8.1 Overfit a tiny paired batch
# --------------------------------------------------------------------------- #
def test_overfit_tiny_paired_batch():
    """Paired flow loss must drop strongly, and consistency must survive."""
    torch.manual_seed(0)
    flow = NullSpaceFlow(channels=3, factor=4, cond_channels=8, encoder_width=16,
                         width=16, num_levels=2, zero_init_tail=False)
    opt = torch.optim.Adam(flow.parameters(), lr=2e-3)
    y = torch.randn(2, 3, 4, 4, 4)
    x = torch.randn(2, 3, 16, 16, 16)

    # Fixed t and z so the target is deterministic -- otherwise the flow-matching
    # loss has irreducible variance and "overfitting" is not well defined.
    t = torch.full((2,), 0.5)
    torch.manual_seed(1)
    losses = []
    for _ in range(150):
        torch.manual_seed(1)  # same z every step
        loss, _ = null_space_flow_loss(flow, y, x, t=t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < 0.5 * losses[0], (
        f"flow loss did not decrease strongly: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    with torch.no_grad():
        _, rel = flow.operator.consistency_error(flow.generate(y, n_steps=4), y)
    assert rel <= 1e-5, f"exact consistency broke during overfitting (rel={rel:.2e})"


# --------------------------------------------------------------------------- #
# 8.2 Critic separates a fixed tiny real/fake set
# --------------------------------------------------------------------------- #
def test_critic_separates_fixed_real_and_fake_sets():
    torch.manual_seed(0)
    critic = HRCritic(in_channels=4, width=16, n_layers=2)
    opt = torch.optim.Adam(critic.parameters(), lr=2e-3, betas=(0.0, 0.99))
    lazy_r1 = LazyR1(gamma=10.0, interval=4)

    # Real: spatially correlated. Fake: white noise. Trivially separable.
    real = torch.randn(4, 4, 16, 16, 16)
    real = torch.nn.functional.avg_pool3d(real, 3, stride=1, padding=1)
    fake = torch.randn(4, 4, 16, 16, 16) * 0.3

    r1_values = []
    for _ in range(120):
        s_real, s_fake = critic(real), critic(fake)
        loss = hinge_d_loss(s_real, s_fake)
        pen, m = lazy_r1(critic, real)
        if pen is not None:
            loss = loss + pen
            r1_values.append(m["loss_R1"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), "critic loss became non-finite"

    with torch.no_grad():
        m_real, m_fake = float(critic(real).mean()), float(critic(fake).mean())
    assert m_real > m_fake, f"critic failed to separate: real={m_real:.3f} fake={m_fake:.3f}"
    assert all(np.isfinite(r1_values)), "R1 penalty went non-finite"
    assert r1_values[-1] < 1e4 * max(r1_values[0], 1e-6), "R1 penalty exploded"


# --------------------------------------------------------------------------- #
# 8.3 One LR-only generator adversarial update
# --------------------------------------------------------------------------- #
def test_lr_only_adversarial_update_moves_flow_and_keeps_consistency():
    torch.manual_seed(0)
    flow = NullSpaceFlow(channels=3, factor=4, cond_channels=8, encoder_width=16,
                         width=16, num_levels=2, zero_init_tail=False)
    critic = HRCritic(in_channels=4, width=16, n_layers=2)
    hp = HighPassDensity(factor=4, cellsize=1000.0, dis_norm=6000.0)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)

    y_lr_only = torch.randn(2, 3, 4, 4, 4)     # no HR counterpart, by design
    z = torch.randn(2, 3, 16, 16, 16)
    before = [p.detach().clone() for p in flow.velocity_net.parameters()]

    def adv_score():
        with torch.no_grad():
            x = flow.generate(y_lr_only, n_steps=2, z=z)
            return float(critic(critic_input(x, flow.operator, hp)).mean())

    s0 = adv_score()
    for _ in range(30):
        x_fake = flow.generate(y_lr_only, n_steps=2, z=z)
        loss = hinge_g_loss(critic(critic_input(x_fake, flow.operator, hp)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    s1 = adv_score()

    after = list(flow.velocity_net.parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(before, after)), (
        "an LR-only adversarial update left the flow parameters unchanged"
    )
    assert s1 > s0, f"critic score did not improve under repeated updates ({s0:.4f} -> {s1:.4f})"
    with torch.no_grad():
        _, rel = flow.operator.consistency_error(flow.generate(y_lr_only, n_steps=2, z=z), y_lr_only)
    assert rel <= 1e-5, f"LR consistency drifted during adversarial training (rel={rel:.2e})"


# --------------------------------------------------------------------------- #
# 8.4 Stage C vs Stage D short equal-compute run, on real files
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mini_dataset(tmp_path_factory):
    """Miniature on-disk boxes: LR 8^3, HR 64^3, factor 8. Exercises the real loaders."""
    root = tmp_path_factory.mktemp("mini")
    rng = np.random.default_rng(0)
    (root / "lr").mkdir(); (root / "hr").mkdir()

    def make(seed):
        hr = rng.standard_normal((CHANNELS, N_HR, N_HR, N_HR)).astype(np.float32) * 0.02
        hr_t = torch.from_numpy(hr).unsqueeze(0)
        hr_t = torch.nn.functional.avg_pool3d(hr_t, 3, stride=1, padding=1)  # correlate
        hr = hr_t.squeeze(0).numpy()
        lr = torch.nn.functional.avg_pool3d(
            torch.from_numpy(hr).unsqueeze(0), FACTOR, FACTOR).squeeze(0).numpy()
        return lr, hr

    for i in range(5):                       # set0-2 train, set3 val, set4 test
        lr, hr = make(i)
        np.save(root / "lr" / f"set{i}.npy", lr)
        np.save(root / "hr" / f"set{i}.npy", hr)

    lr_only = root / "lr_only"
    for i in range(6):                       # unpaired boxes, disjoint names
        d = lr_only / f"u{i}"
        d.mkdir(parents=True)
        lr, _ = make(100 + i)
        np.save(d / "catnorm.npy", lr)
    return root


def _mini_config(root: Path, run_dir: Path, stage: str) -> Path:
    cfg = {
        "base": str(REPO / "configs" / "dmsr" / "_base.yaml"),
        "stage": stage,
        "data": {
            "train_lr_glob": f"{root}/lr/set[0-2].npy",
            "train_hr_glob": f"{root}/hr/set[0-2].npy",
            "val_lr_glob": f"{root}/lr/set3.npy",
            "val_hr_glob": f"{root}/hr/set3.npy",
            "test_lr_glob": f"{root}/lr/set4.npy",
            "test_hr_glob": f"{root}/hr/set4.npy",
            "lr_only_glob": f"{root}/lr_only/*/catnorm.npy",
            "crop_lr": 4, "channels": 6, "use_channels": [0, 1, 2],
            "mmap": False, "epoch_length": 64,
        },
        "model": {"cond_channels": 8, "encoder_width": 8, "width": 8,
                  "num_levels": 2, "grad_checkpoint": False,
                  "condition_encoder_init": "random", "init_from": None},
        "critic": {"width": 8, "n_layers": 2},
        "adv": {"lambda_adv": 0.1, "ramp_steps": 1, "critic_warmup_steps": 0,
                "n_critic": 1, "gen_ode_steps": 2},
        "env": {"pool_size": 48, "n_dims": 1, "n_bins": 3},
        "train": {"batch_size": 2, "steps": 3, "log_every": 1,
                  "eval_every": 0, "save_every": 0, "num_workers": 0},
        "eval": {"n_steps": 2, "max_val_batches": 1, "max_val_crops": 4},
        "wandb": {"mode": "disabled"},
        "output": {"run_dir": str(run_dir)},
    }
    path = run_dir.parent / f"cfg_{stage}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def _run_stage(cfg_path: Path) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["WANDB_MODE"] = "disabled"
    return subprocess.run(
        [sys.executable, "-m", "cosmo_sr.train.train_dmsr", "--config", str(cfg_path)],
        capture_output=True, text=True, cwd=str(REPO), env=env, timeout=1800,
    )


@pytest.mark.slow
def test_stage_c_and_d_short_equal_compute_run(mini_dataset, tmp_path):
    audits = {}
    for stage in ("c", "d"):
        run_dir = tmp_path / f"run_{stage}"
        proc = _run_stage(_mini_config(mini_dataset, run_dir, stage))
        assert proc.returncode == 0, f"stage {stage} failed:\n{proc.stdout}\n{proc.stderr}"
        audits[stage] = json.loads((run_dir / "compute_audit.json").read_text())

        split = json.loads((run_dir / "split.json").read_text())
        assert len(split["train_hr"]) == 3
        assert len(split["val_hr"]) == 1 and len(split["test_hr"]) == 1

    # Identical update counts...
    for key in ("steps", "gen_paired", "gen_second", "critic", "n_critic"):
        assert audits["c"][key] == audits["d"][key], (
            f"{key} differs: C={audits['c'][key]} D={audits['d'][key]}"
        )
    # ...and exactly one difference: where the second stream's crops came from.
    assert audits["c"]["second_stream_source"] == "paired_repeat"
    assert audits["d"]["second_stream_source"] == "lr_only"

    # Stage D must actually have built the balanced sampler over LR-only boxes.
    balance = json.loads((tmp_path / "run_d" / "env_balance.json").read_text())
    assert balance["balance"]["n_unpaired"] > 0
    assert balance["balance"]["auc_after"] <= 0.60
    assert not (tmp_path / "run_c" / "env_balance.json").exists(), (
        "Stage C must not build an LR-only sampler"
    )
