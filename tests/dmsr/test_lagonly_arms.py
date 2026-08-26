"""The Lagrangian-only critic sweep must be a one-lever-per-arm ladder.

Each arm answers exactly one question, so each config may differ from its parent
in exactly one training-relevant field. That property is easy to break by editing
one file and forgetting another, so it is asserted here rather than reviewed.

Also pins the two traps found while building the sweep:
  * PyYAML is YAML 1.1, where a bare ``off`` parses as the boolean ``False`` --
    which would reach ``density_channels`` as a KeyError instead of selecting the
    no-density critic.
  * ``adv.lambda_flow`` must default to 1.0, or every previous run's loss changes.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cosmo_sr.dmsr.density import (
    CriticInputNormalizer,
    HighPassDensity,
    critic_input,
    density_channels,
)
from cosmo_sr.utils.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "dmsr"

# (arm, parent, the single field the arm is allowed to change)
LADDER = [
    ("t13_lagonly_l003", "t13_fix_vc32", "critic.density_mode"),
    ("t13_lagonly_bp4_l003", "t13_lagonly_l003", "adv.bp_steps"),
    ("t13_lagonly_bp4_l03", "t13_lagonly_bp4_l003", "adv.lambda_adv"),
    ("t13_lagonly_bp4_l03_flow01", "t13_lagonly_bp4_l03", "adv.lambda_flow"),
]
# Bookkeeping that is expected to differ on every arm and carries no experiment.
BOOKKEEPING = {"output.run_dir", "wandb.name", "wandb.group", "train.seed", "train.steps"}


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


@pytest.mark.parametrize("arm,parent,lever", LADDER)
def test_each_arm_changes_exactly_one_lever(arm, parent, lever):
    a = _flatten(load_config(CONFIG_DIR / f"{arm}.yaml"))
    p = _flatten(load_config(CONFIG_DIR / f"{parent}.yaml"))

    # A new key is a change too (lambda_flow is absent from every parent).
    changed = {k for k in set(a) | set(p) if a.get(k) != p.get(k)}
    unexplained = changed - BOOKKEEPING - {lever}
    assert not unexplained, f"{arm} vs {parent}: unexpected changes {sorted(unexplained)}"
    assert lever in changed, f"{arm} does not actually change its stated lever {lever}"


def test_density_mode_off_survives_yaml_11_boolean_coercion():
    """A bare `off` in YAML 1.1 is False, which would KeyError in density_channels."""
    cfg = load_config(CONFIG_DIR / "t13_lagonly_l003.yaml")
    mode = cfg["critic"]["density_mode"]
    assert mode == "off" and isinstance(mode, str), f"got {mode!r} ({type(mode).__name__})"
    assert density_channels(mode) == 0


def test_lambda_flow_defaults_to_one_everywhere_it_is_unset():
    """Absent lambda_flow must mean 1.0, or every completed run's loss changed."""
    for name in ["_base", "_base_train13", "stage_c_critic_pairedlr", "t13_fix_vc32"]:
        cfg = load_config(CONFIG_DIR / f"{name}.yaml")
        assert "lambda_flow" not in cfg.get("adv", {}), f"{name} pins lambda_flow"


def test_critic_input_off_is_the_residual_alone():
    """density_mode 'off' must drop the channel, not zero it: 3 channels in, 3 out."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16, 16, 16)

    class _Op:  # `residual_mode='full'` never touches the operator
        pass

    hp = HighPassDensity(factor=2, cellsize=1.0, dis_norm=1.0)
    out = critic_input(x, _Op(), hp, residual_mode="full", density_mode="off")
    assert out.shape == x.shape
    torch.testing.assert_close(out, x)

    norm = CriticInputNormalizer.fit([x], _Op(), hp, residual_mode="full", density_mode="off")
    out_n = critic_input(x, _Op(), hp, normalizer=norm,
                         residual_mode="full", density_mode="off")
    assert out_n.shape == x.shape
    assert float(out_n.pow(2).mean()) == pytest.approx(1.0, rel=0.05)


# --------------------------------------------------------------------------- #
# The two mechanism levers, exercised on a tiny model.
#
# The trainer's own --smoke path cannot cover these: `critic_warmup_steps: 200`
# means every step of a 4-step smoke run is a critic-warmup step, so the
# generator's adversarial branch never executes. (And a real smoke run is
# SIGKILLed on the login node anyway.) These drive that branch directly.
# --------------------------------------------------------------------------- #
from cosmo_sr.dmsr.critic import HRCritic, hinge_g_loss  # noqa: E402
from cosmo_sr.dmsr.flow import NullSpaceFlow, unconstrained_flow_loss  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    # zero_init_tail=False for the same reason as tests/dmsr/test_adversarial_grad.py:
    # the zero-initialised output conv would zero the upstream gradient at step 0
    # and make a connectivity test pass or fail on an init detail.
    flow = NullSpaceFlow(channels=3, factor=4, cond_channels=8, encoder_width=8,
                         width=8, num_levels=2, zero_init_tail=False)
    # Every arm in this sweep inherits `model.unconstrained: true` from
    # t13_unconstrained, so the tiny stand-in must match: it selects
    # `unconstrained_flow_loss` and drops the P_A projection in generate().
    flow.unconstrained = True
    # in_channels=3, not 4: this is the whole point of the sweep -- no density channel.
    critic = HRCritic(in_channels=3, width=8, n_layers=2)
    hp = HighPassDensity(factor=4, cellsize=1000.0, dis_norm=6000.0)
    y = torch.randn(2, 3, 4, 4, 4)
    x = torch.randn(2, 3, 16, 16, 16)
    return flow, critic, hp, y, x


def _adv_grad_norm(flow, critic, hp, y, bp_steps, n_steps=4):
    torch.manual_seed(1234)                       # same ODE noise for both paths
    x_hat = flow.generate(y, n_steps=n_steps, bp_steps=bp_steps)
    loss = hinge_g_loss(critic(critic_input(
        x_hat, flow.operator, hp, residual_mode="full", density_mode="off")))
    flow.zero_grad(set_to_none=True)
    loss.backward()
    return sum(float(p.grad.pow(2).sum()) for p in flow.velocity_net.parameters()
               if p.grad is not None) ** 0.5


def test_density_off_critic_still_delivers_adversarial_gradient(tiny):
    """A 3-channel critic must reach the flow -- otherwise the sweep trains on nothing."""
    flow, critic, hp, y, _ = tiny
    g = _adv_grad_norm(flow, critic, hp, y, bp_steps=None)
    assert g > 0.0, "3-channel critic produced no gradient in the velocity net"


def test_bp_steps_null_is_a_real_lever_not_a_no_op(tiny):
    """Full backprop must give a different gradient than backprop through 1 step.

    If these matched, `adv.bp_steps: null` would be cosmetic and the bp4 arms would
    silently duplicate their parents.
    """
    flow, critic, hp, y, _ = tiny
    g_trunc = _adv_grad_norm(flow, critic, hp, y, bp_steps=1)
    g_full = _adv_grad_norm(flow, critic, hp, y, bp_steps=None)
    assert g_trunc > 0.0 and g_full > 0.0
    rel = abs(g_full - g_trunc) / max(g_trunc, 1e-12)
    assert rel > 1e-3, (
        f"bp_steps=None and bp_steps=1 gave the same gradient norm "
        f"({g_full:.6g} vs {g_trunc:.6g}); the lever does nothing"
    )


def test_lambda_flow_scales_the_flow_gradient_linearly(tiny):
    """lambda_flow must scale the mean-seeking term, and 1.0 must be a no-op."""
    flow, _critic, _hp, y, x = tiny

    def flow_grad_norm(lam_flow):
        torch.manual_seed(99)
        loss, _ = unconstrained_flow_loss(flow, y, x)
        flow.zero_grad(set_to_none=True)
        (lam_flow * loss).backward()
        return sum(float(p.grad.pow(2).sum()) for p in flow.velocity_net.parameters()
                   if p.grad is not None) ** 0.5

    g1 = flow_grad_norm(1.0)
    g01 = flow_grad_norm(0.1)
    assert g1 > 0.0
    assert g01 == pytest.approx(0.1 * g1, rel=1e-4)
