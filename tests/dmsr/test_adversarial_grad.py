"""7.3 -- the adversarial gradient is non-degenerate, and consistency is untouched.

The last two tests here are the important ones. They are a **regression guard
against silently reintroducing the degenerate consistency objective**: under this
parameterization ``A(x_hat) = y`` identically, so ``MSE(A(x_hat), y)`` is ~0 and
supplies ~0 gradient to the null-space residual. If someone later adds it as a
training loss believing it teaches consistency, these tests document that it
teaches nothing at all.
"""
from __future__ import annotations

import pytest
import torch

from cosmo_sr.dmsr.critic import HRCritic, hinge_g_loss
from cosmo_sr.dmsr.density import HighPassDensity, critic_input
from cosmo_sr.dmsr.flow import NullSpaceFlow


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    # zero_init_tail=False on purpose. Production uses True so the flow starts at
    # v=0, but that zero-initialised output conv also zeroes the gradient to every
    # layer upstream of it *at step 0 only* -- which would make these connectivity
    # tests vacuously pass-or-fail on an initialisation detail rather than on
    # whether the graph is actually wired end to end.
    flow = NullSpaceFlow(channels=3, factor=4, cond_channels=8, encoder_width=8,
                         width=8, num_levels=2, zero_init_tail=False)
    critic = HRCritic(in_channels=4, width=8, n_layers=2)
    hp = HighPassDensity(factor=4, cellsize=1000.0, dis_norm=6000.0)
    y = torch.randn(2, 3, 4, 4, 4)
    return flow, critic, hp, y


def _adv_backward(flow, critic, hp, y, n_steps=2):
    x_hat = flow.generate(y, n_steps=n_steps, bp_steps=None)
    loss = hinge_g_loss(critic(critic_input(x_hat, flow.operator, hp)))
    flow.zero_grad(set_to_none=True)
    loss.backward()
    return x_hat, loss


def test_adv_gradient_reaches_flow_parameters(setup):
    flow, critic, hp, y = setup
    _adv_backward(flow, critic, hp, y)
    grads = [p.grad for p in flow.velocity_net.parameters() if p.requires_grad]
    total = sum(float(g.abs().sum()) for g in grads if g is not None)
    assert total > 0.0, "loss_G_adv produced no gradient in the residual flow"
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_adv_gradient_reaches_condition_encoder(setup):
    flow, critic, hp, y = setup
    _adv_backward(flow, critic, hp, y)
    total = sum(float(p.grad.abs().sum()) for p in flow.encoder.parameters()
                if p.grad is not None)
    assert total > 0.0, "adversarial gradient never reached the condition encoder"


def test_gradient_flows_through_the_cic_path_alone(setup):
    """Isolate the Eulerian path: critic sees ONLY rho_high, residual zeroed.

    If CIC were non-differentiable (or accidentally wrapped in no_grad) this is
    the test that fails while the others still pass.
    """
    flow, _, hp, y = setup
    critic_rho = HRCritic(in_channels=1, width=8, n_layers=2)
    x_hat = flow.generate(y, n_steps=2)
    rho_high = hp(x_hat)
    assert rho_high.requires_grad, "rho_high is detached from the graph"
    loss = hinge_g_loss(critic_rho(rho_high))
    flow.zero_grad(set_to_none=True)
    loss.backward()
    total = sum(float(p.grad.abs().sum()) for p in flow.velocity_net.parameters()
                if p.grad is not None)
    assert total > 0.0, "no gradient reached the flow through the CIC density path"


def test_consistency_preserved_after_generator_step(setup):
    """A(x_hat) must still equal y after an optimizer step on the adversarial loss."""
    flow, critic, hp, y = setup
    opt = torch.optim.Adam(flow.parameters(), lr=1e-2)  # deliberately large
    z = torch.randn(2, 3, 16, 16, 16)

    with torch.no_grad():
        _, rel_before = flow.operator.consistency_error(
            flow.generate(y, n_steps=2, z=z), y)

    x_hat = flow.generate(y, n_steps=2, z=z)
    loss = hinge_g_loss(critic(critic_input(x_hat, flow.operator, hp)))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    with torch.no_grad():
        _, rel_after = flow.operator.consistency_error(
            flow.generate(y, n_steps=2, z=z), y)

    assert rel_before <= 1e-5 and rel_after <= 1e-5
    # Consistency is structural, so it cannot drift with the parameters.
    assert abs(rel_after - rel_before) < 1e-5


def test_degenerate_consistency_loss_is_identically_zero(setup):
    """REGRESSION: MSE(A(x_hat), y) ~ 0 -- it is not a usable training signal."""
    flow, _, _, y = setup
    x_hat = flow.generate(y, n_steps=2)
    loss_deg = torch.nn.functional.mse_loss(flow.operator.A(x_hat), y)
    scale = float(y.pow(2).mean().detach())
    assert float(loss_deg.detach()) / scale < 1e-10, (
        f"expected a vacuous consistency loss, got {float(loss_deg.detach()):.3e}"
    )


def test_degenerate_consistency_loss_gives_no_gradient(setup):
    """REGRESSION: and it back-propagates ~nothing to the null-space residual."""
    flow, critic, hp, y = setup
    x_hat = flow.generate(y, n_steps=2)

    loss_deg = torch.nn.functional.mse_loss(flow.operator.A(x_hat), y)
    flow.zero_grad(set_to_none=True)
    loss_deg.backward(retain_graph=True)
    deg_grad = sum(float(p.grad.abs().sum()) for p in flow.velocity_net.parameters()
                   if p.grad is not None)

    x_hat2 = flow.generate(y, n_steps=2)
    loss_adv = hinge_g_loss(critic(critic_input(x_hat2, flow.operator, hp)))
    flow.zero_grad(set_to_none=True)
    loss_adv.backward()
    adv_grad = sum(float(p.grad.abs().sum()) for p in flow.velocity_net.parameters()
                   if p.grad is not None)

    assert adv_grad > 0.0
    assert deg_grad < 1e-6 * max(adv_grad, 1e-12), (
        f"consistency-loss gradient {deg_grad:.3e} is not negligible vs "
        f"adversarial {adv_grad:.3e}; the degenerate objective may have become live"
    )
