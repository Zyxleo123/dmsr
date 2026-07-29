"""7.2 -- the flow stays inside ker A, including through ODE integration."""
from __future__ import annotations

import pytest
import torch

from cosmo_sr.dmsr.flow import NullSpaceFlow, null_space_flow_loss

TOL = 1e-5


@pytest.fixture(scope="module")
def flow():
    torch.manual_seed(0)
    return NullSpaceFlow(channels=3, factor=4, cond_channels=8, encoder_width=8,
                         width=8, num_levels=2, zero_init_tail=False)


@pytest.fixture
def batch(flow):
    torch.manual_seed(1)
    n_lr = 4
    y = torch.randn(2, 3, n_lr, n_lr, n_lr)
    x = torch.randn(2, 3, n_lr * flow.factor, n_lr * flow.factor, n_lr * flow.factor)
    return y, x


def _null_rel(op, field) -> float:
    """||A(field)|| relative to ||field||: 0 iff field is in ker A."""
    return float(op.A(field).norm() / field.norm().clamp_min(1e-12))


def test_z_null_is_in_null_space(flow, batch):
    """A(P_A(z)) == 0."""
    _, x = batch
    z_null = flow.operator.P_A(torch.randn_like(x))
    assert _null_rel(flow.operator, z_null) <= TOL


def test_predicted_velocity_is_in_null_space(flow, batch):
    """A(predicted_velocity) == 0 -- the model's output is projected."""
    y, x = batch
    t = torch.rand(y.shape[0])
    r_t = flow.operator.P_A(torch.randn_like(x))
    v = flow.velocity(r_t, t, y)
    assert _null_rel(flow.operator, v) <= TOL


def test_target_velocity_is_in_null_space(flow, batch):
    y, x = batch
    op = flow.operator
    v_target = op.P_A(x) - op.P_A(torch.randn_like(x))
    assert _null_rel(op, v_target) <= TOL


def test_residual_stays_in_null_space_through_integration(flow, batch):
    """Integrate the toy ODE and check ker A membership at *every* step."""
    y, x = batch
    op = flow.operator
    n_steps = 8
    b, c = y.shape[0], y.shape[1]
    n_hr = y.shape[-1] * flow.factor
    r = op.P_A(torch.randn(b, c, n_hr, n_hr, n_hr))

    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full((b,), i / n_steps)
            r = op.P_A(r + (1.0 / n_steps) * flow.velocity(r, t, y))
            assert _null_rel(op, r) <= TOL, f"left null space at step {i}"


def test_generated_field_is_exactly_consistent(flow, batch):
    y, _ = batch
    with torch.no_grad():
        x_hat = flow.generate(y, n_steps=4)
    _, rel = flow.operator.consistency_error(x_hat, y)
    assert rel <= TOL


def test_flow_loss_is_finite_and_differentiable(flow, batch):
    y, x = batch
    loss, metrics = null_space_flow_loss(flow, y, x)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in flow.velocity_net.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    assert "loss_flow" in metrics


def test_deterministic_mode_bypasses_the_ode(flow, batch):
    """REGRESSION: the `det` baseline must not be evaluated by ODE integration.

    `paired_deterministic` trains a one-shot regressor (`deterministic_regression_loss`
    fits `P_A(v_theta(0, t=1, y))` directly). If `generate` still integrated the ODE,
    that model would be scored on a trajectory it was never fitted to -- giving
    meaningless val metrics *and* meaningless best-checkpoint selection, while looking
    perfectly healthy in the logs.
    """
    y, x = batch
    z1 = torch.randn_like(x)
    z2 = torch.randn_like(x)

    flow.deterministic = True
    try:
        with torch.no_grad():
            a = flow.generate(y, n_steps=20, z=z1)
            b = flow.generate(y, n_steps=20, z=z2)
            # No z dependence and no n_steps dependence: it is not integrating.
            assert torch.allclose(a, b, atol=1e-6)
            assert torch.allclose(a, flow.generate(y, n_steps=3, z=z1), atol=1e-6)
            # It equals exactly what the training loss fits.
            expected = flow.operator.P_A(
                flow.velocity(torch.zeros_like(x), torch.ones(y.shape[0]), y))
            assert torch.allclose(flow.sample_residual(y), expected, atol=1e-6)
            # And it is still exactly LR-consistent.
            assert flow.operator.consistency_error(a, y)[1] <= TOL
    finally:
        flow.deterministic = False

    with torch.no_grad():
        c = flow.generate(y, n_steps=4, z=z1)
        d = flow.generate(y, n_steps=4, z=z2)
    assert not torch.allclose(c, d, atol=1e-6), "flow mode lost its z dependence"
