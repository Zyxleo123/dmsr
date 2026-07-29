"""Part 2/7 -- deterministic mean + stochastic innovation decomposition.

Covers the mandatory operator tests (``A(m)=0``, ``A(e_gt)=0``, ``A(z_null)=0``,
``A(v_pred)=0``, ``A(x_hat)=y``, consistency through integration) and the
mean/innovation tests (mean independent of ``z``, different ``z`` -> different
innovation, frozen mean gets no gradient, flow loss updates the flow, adversarial
gradient follows the Stage C path and never the mean, output = base+mean+innovation,
save/load round-trip).
"""
from __future__ import annotations

import copy

import pytest
import torch

from cosmo_sr.dmsr.critic import HRCritic, hinge_g_loss
from cosmo_sr.dmsr.density import HighPassDensity, critic_input
from cosmo_sr.dmsr.flow import NullSpaceFlow
from cosmo_sr.dmsr.mean_innovation import (
    MeanInnovationFlow,
    innovation_flow_loss,
    mean_reconstruction_loss,
)

TOL = 1e-5


def _build(zero_init_tail=True, seed=0):
    torch.manual_seed(seed)
    kw = dict(channels=3, factor=4, cond_channels=8, encoder_width=8, width=8,
              num_levels=2, zero_init_tail=zero_init_tail)
    mean_model = NullSpaceFlow(**kw)
    innovation = NullSpaceFlow(**kw)
    return MeanInnovationFlow(mean_model, innovation)


@pytest.fixture
def model():
    return _build()


@pytest.fixture
def batch(model):
    torch.manual_seed(1)
    n_lr = 4
    n_hr = n_lr * model.factor
    y = torch.randn(2, 3, n_lr, n_lr, n_lr)
    x = torch.randn(2, 3, n_hr, n_hr, n_hr)
    return y, x


def _null_rel(op, f):
    f = f.detach()
    return float(op.A(f).norm() / f.norm().clamp_min(1e-12))


# --------------------------------------------------------------------------- #
# Operator identities
# --------------------------------------------------------------------------- #
def test_mean_is_in_null_space(model, batch):
    y, _ = batch
    assert _null_rel(model.operator, model.mean_residual(y)) <= TOL


def test_innovation_target_is_in_null_space(model, batch):
    y, x = batch
    op = model.operator
    e_gt = op.P_A(op.P_A(x) - model.mean_residual(y).detach())
    assert _null_rel(op, e_gt) <= TOL


def test_z_null_and_predicted_velocity_in_null_space(model, batch):
    y, x = batch
    op = model.operator
    z_null = op.P_A(torch.randn_like(x))
    assert _null_rel(op, z_null) <= TOL
    t = torch.rand(y.shape[0])
    v = model.velocity(z_null, t, y)
    assert _null_rel(op, v) <= TOL


def test_generated_field_is_exactly_consistent(model, batch):
    y, _ = batch
    with torch.no_grad():
        x_hat = model.generate(y, n_steps=4)
    _, rel = model.operator.consistency_error(x_hat, y)
    assert rel <= TOL


def test_consistency_holds_through_integration(model, batch):
    """A(x_hat) = y at *every* Euler step of the innovation ODE, not only the end."""
    y, _ = batch
    op = model.operator
    n_steps = 6
    b, c = y.shape[0], y.shape[1]
    n_hr = y.shape[-1] * model.factor
    m = model.mean_residual(y).detach()
    e = op.P_A(torch.randn(b, c, n_hr, n_hr, n_hr))
    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full((b,), i / n_steps)
            e = op.P_A(e + (1.0 / n_steps) * model.velocity(e, t, y))
            x_hat = op.A_plus(y) + op.P_A(m + e)
            assert op.consistency_error(x_hat, y)[1] <= TOL, f"drift at step {i}"


# --------------------------------------------------------------------------- #
# Mean / innovation behaviour
# --------------------------------------------------------------------------- #
def test_mean_is_independent_of_z(model, batch):
    y, _ = batch
    a = model.mean_residual(y)
    b = model.mean_residual(y)
    assert torch.allclose(a, b, atol=1e-7)               # deterministic, no z


def test_different_z_give_different_innovations(model, batch):
    y, x = batch
    z1 = torch.randn_like(x)
    z2 = torch.randn_like(x)
    with torch.no_grad():
        e1 = model.sample_innovation(y, n_steps=4, z=z1)
        e2 = model.sample_innovation(y, n_steps=4, z=z2)
    assert not torch.allclose(e1, e2, atol=1e-6)


def test_same_z_is_deterministic(model, batch):
    y, x = batch
    z = torch.randn_like(x)
    with torch.no_grad():
        a = model.generate(y, n_steps=4, z=z)
        b = model.generate(y, n_steps=4, z=z)
    assert torch.allclose(a, b, atol=1e-6)


def test_output_equals_base_plus_mean_plus_innovation(model, batch):
    y, x = batch
    z = torch.randn_like(x)
    op = model.operator
    with torch.no_grad():
        x_hat = model.generate(y, n_steps=4, z=z)
        m = model.mean_residual(y).detach()
        e = model.sample_innovation(y, n_steps=4, z=z)
        manual = op.A_plus(y) + op.P_A(m + e)
    assert torch.allclose(x_hat, manual, atol=1e-6)


def test_frozen_mean_receives_no_gradient(batch):
    model = _build(zero_init_tail=False)
    model.freeze_mean()
    y, x = batch
    loss, _ = innovation_flow_loss(model, y, x)
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert all(p.grad is None for p in model.mean_parameters()), "frozen mean got a gradient"


def test_flow_loss_updates_the_innovation_flow(batch):
    model = _build(zero_init_tail=False)
    model.freeze_mean()
    y, x = batch
    loss, metrics = innovation_flow_loss(model, y, x)
    assert torch.isfinite(loss)
    model.zero_grad(set_to_none=True)
    loss.backward()
    g = [p.grad for p in model.innovation.velocity_net.parameters() if p.requires_grad]
    assert any(gr is not None and gr.abs().sum() > 0 for gr in g)
    assert "target_innovation_rms" in metrics and "mean_residual_rms" in metrics


def test_adversarial_gradient_follows_stage_c_path_not_the_mean(batch):
    model = _build(zero_init_tail=False)
    model.freeze_mean()
    y, _ = batch
    critic = HRCritic(in_channels=4, width=8, n_layers=2)
    hp = HighPassDensity(factor=4, cellsize=1000.0, dis_norm=6000.0)
    x_hat = model.generate(y, n_steps=2, bp_steps=None)
    loss = hinge_g_loss(critic(critic_input(x_hat, model.operator, hp)))
    model.zero_grad(set_to_none=True)
    loss.backward()
    innov = sum(float(p.grad.abs().sum()) for p in model.innovation.parameters()
                if p.grad is not None)
    mean_grad = [p.grad for p in model.mean_parameters()]
    assert innov > 0.0, "adversarial gradient never reached the innovation flow"
    assert all(g is None for g in mean_grad), "adversarial gradient leaked into the mean"


def test_innovation_loss_does_not_backprop_into_mean_even_if_unfrozen(batch):
    """The mean is stop-gradient'd in e_gt, so even an *unfrozen* mean gets no flow
    gradient -- only its own reconstruction loss would train it."""
    model = _build(zero_init_tail=False)
    # deliberately NOT frozen
    y, x = batch
    loss, _ = innovation_flow_loss(model, y, x)
    model.zero_grad(set_to_none=True)
    loss.backward()
    mean_grad = sum(float(p.grad.abs().sum()) for p in model.mean_parameters()
                    if p.grad is not None)
    assert mean_grad == 0.0, "innovation loss trained the mean through e_gt"

    # ... but the dedicated reconstruction loss DOES reach the mean.
    loss_m, _ = mean_reconstruction_loss(model, y, x)
    model.zero_grad(set_to_none=True)
    loss_m.backward()
    got = sum(float(p.grad.abs().sum()) for p in model.mean_parameters()
              if p.grad is not None)
    assert got > 0.0


def test_save_load_preserves_all_components(batch):
    # zero_init_tail=False so two random inits genuinely differ (a zero-init tail
    # would make every fresh model emit the same field regardless of its weights).
    model = _build(zero_init_tail=False, seed=0)
    y, x = batch
    z = torch.randn_like(x)
    with torch.no_grad():
        before = model.generate(y, n_steps=4, z=z)

    sd = copy.deepcopy(model.state_dict())
    reload = _build(zero_init_tail=False, seed=123)       # different init
    with torch.no_grad():
        assert not torch.allclose(reload.generate(y, n_steps=4, z=z), before, atol=1e-4)
    reload.load_state_dict(sd)
    with torch.no_grad():
        after = reload.generate(y, n_steps=4, z=z)
    assert torch.allclose(after, before, atol=1e-6)
