"""Part 2 -- deterministic null-space mean + stochastic innovation flow.

Experiment **E**. The generator is decomposed into a *predictable* conditional
mean and a *stochastic* innovation, both living entirely in ``ker A``::

    r_gt   = P_A(x_hr)                       # full null-space target
    m      = P_A(mean_model(y))              # predictable conditional mean
    e_gt   = P_A(r_gt - stop_grad(m))        # innovation target
    x_hat  = A_plus(y) + P_A(m + u_theta(y, z))

``mean_model`` is the trained ``paired_deterministic`` network reused verbatim: it
is a :class:`~cosmo_sr.dmsr.flow.NullSpaceFlow` in ``deterministic`` mode, so
``P_A(mean_model(y))`` is exactly the ``x_mean = A_plus(y) + P_A(mean_model(y))``
regressor that project already trained and measured. In Phase 2 it is **frozen**;
the innovation flow ``u_theta`` is the only thing learning, and it learns the
*innovation* target ``e_gt`` with the identical straight-line rectified-flow
objective used by Stage C (only the target changes from ``r_gt`` to ``e_gt``).

Design invariants (all tested in ``tests/dmsr/test_mean_innovation.py``):

* **Exact consistency is preserved.** ``A(x_hat) = y`` still holds structurally,
  because every learned/added piece is projected and ``x_hat = A_plus(y) +
  (null-space field)``.
* **The mean is independent of ``z``**; different ``z`` give different innovations;
  the same ``z`` is deterministic.
* **No samplewise reconstruction loss** is applied to a stochastic ``x_hat`` --
  only the deterministic mean model ever sees a reconstruction loss (and in Phase 2
  it sees none, being frozen). The innovation is trained by flow-matching alone.
* The frozen mean receives **no gradient** during Phase 2, and the adversarial /
  flow path always uses a **detached** mean, so the intended Stage C generator
  gradient routing is unchanged.

The interface mirrors :class:`NullSpaceFlow` (``operator``, ``velocity``,
``sample_residual``, ``generate``, ``flow_parameters``, ``encoder_parameters``,
``set_encoder_trainable``) so the Stage C trainer drives it with minimal branching.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow import NullSpaceFlow, build_flow


class MeanInnovationFlow(nn.Module):
    """Frozen deterministic mean + trainable stochastic innovation flow."""

    def __init__(self, mean_model: NullSpaceFlow, innovation: NullSpaceFlow):
        super().__init__()
        if mean_model.factor != innovation.factor:
            raise ValueError("mean and innovation factors differ")
        if mean_model.channels != innovation.channels:
            raise ValueError("mean and innovation channel counts differ")
        self.mean_model = mean_model
        self.innovation = innovation
        # mean_model is a one-shot regressor, never integrated as an ODE.
        self.mean_model.deterministic = True
        self.operator = innovation.operator
        self.factor = innovation.factor
        self.channels = innovation.channels
        self.deterministic = False           # interface parity with NullSpaceFlow
        self._mean_frozen = False

    # -- mean / innovation freeze plumbing --------------------------------- #
    def freeze_mean(self) -> None:
        """Phase 2: the mean supplies a fixed offset and receives no gradients."""
        for p in self.mean_model.parameters():
            p.requires_grad_(False)
        self.mean_model.eval()
        self._mean_frozen = True

    def unfreeze_mean(self) -> None:
        for p in self.mean_model.parameters():
            p.requires_grad_(True)
        self._mean_frozen = False

    def train(self, mode: bool = True):  # keep a frozen mean in eval mode
        super().train(mode)
        if self._mean_frozen:
            self.mean_model.eval()
        return self

    # -- parameter groups (only the innovation flow trains in Phase 2) ----- #
    def flow_parameters(self):
        return self.innovation.flow_parameters()

    def encoder_parameters(self):
        return self.innovation.encoder_parameters()

    def mean_parameters(self):
        return self.mean_model.parameters()

    def set_encoder_trainable(self, trainable: bool) -> None:
        self.innovation.set_encoder_trainable(trainable)

    # -- components -------------------------------------------------------- #
    def mean_residual(self, y: torch.Tensor) -> torch.Tensor:
        """``m = P_A(mean_model(y))`` -- the predictable null-space conditional mean.

        Not detached here (Phase-1 / joint reconstruction needs the graph); the
        generative path detaches it explicitly.
        """
        return self.mean_model.sample_residual(y)      # deterministic one-shot, projected

    def x_mean(self, y: torch.Tensor) -> torch.Tensor:
        """``x_mean = A_plus(y) + P_A(mean_model(y))`` -- the deterministic recon."""
        return self.operator.A_plus(y) + self.mean_residual(y)

    def velocity(self, e_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Projected innovation velocity ``P_A(u_theta(e_t, t, y))``."""
        return self.innovation.velocity(e_t, t, y)

    forward = velocity

    def sample_innovation(self, y, n_steps=20, z=None, bp_steps=None) -> torch.Tensor:
        """Integrate the innovation ODE from ``z_null`` to ``e_hat`` (in ``ker A``)."""
        return self.innovation.sample_residual(y, n_steps=n_steps, z=z, bp_steps=bp_steps)

    def sample_residual(self, y, n_steps=20, z=None, bp_steps=None) -> torch.Tensor:
        """Full null-space residual ``P_A(m + e_hat)`` with a **detached** mean.

        The final projection is applied even though ``m`` and ``e_hat`` are each
        already in ``ker A`` (belt-and-braces, per spec).
        """
        m = self.mean_residual(y).detach()
        e = self.sample_innovation(y, n_steps=n_steps, z=z, bp_steps=bp_steps)
        return self.operator.P_A(m + e)

    def generate(self, y, n_steps=20, z=None, bp_steps=None) -> torch.Tensor:
        """``x_hat = A_plus(y) + P_A(m + e_hat)`` -- exactly consistent with ``y``."""
        return self.operator.A_plus(y) + self.sample_residual(
            y, n_steps=n_steps, z=z, bp_steps=bp_steps)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def innovation_flow_loss(
    model: MeanInnovationFlow,
    y: torch.Tensor,
    x_hr: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    return_fields: bool = False,
):
    """Phase-2 flow matching on the innovation target ``e_gt`` (frozen mean).

    Identical straight-line interpolation to :func:`null_space_flow_loss`; the only
    change is the target: ``e_gt = P_A(P_A(x_hr) - m.detach())`` instead of
    ``P_A(x_hr)``. ``z`` / ``return_fields`` are the same additive diagnostic hooks.
    """
    op = model.operator
    r_gt = op.P_A(x_hr)
    m = model.mean_residual(y).detach()
    e_gt = op.P_A(r_gt - m)
    z_null = op.P_A(torch.randn_like(x_hr) if z is None else z)

    b = x_hr.shape[0]
    if t is None:
        t = torch.rand(b, device=x_hr.device, dtype=x_hr.dtype)
    t_b = t.view(b, *([1] * (x_hr.dim() - 1)))

    e_t = (1.0 - t_b) * z_null + t_b * e_gt
    v_target = e_gt - z_null
    v_pred = model.velocity(e_t, t, y)

    loss = F.mse_loss(v_pred, v_target)
    metrics = {
        "loss_flow": float(loss.detach()),
        "target_residual_rms": float(r_gt.detach().pow(2).mean().sqrt()),
        "target_innovation_rms": float(e_gt.detach().pow(2).mean().sqrt()),
        "mean_residual_rms": float(m.pow(2).mean().sqrt()),
    }
    if return_fields:
        return loss, metrics, {
            "v_pred": v_pred.detach(), "v_target": v_target.detach(), "t": t.detach(),
        }
    return loss, metrics


def mean_reconstruction_loss(
    model: MeanInnovationFlow, y: torch.Tensor, x_hr: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Deterministic paired reconstruction ``MSE(P_A(mean_model(y)), P_A(x_hr))``.

    The **only** loss the mean model ever receives (Phase 1, or the disabled joint
    phase). Never applied to a stochastic ``x_hat``.
    """
    op = model.operator
    r_target = op.P_A(x_hr)
    m = model.mean_residual(y)
    loss = F.mse_loss(m, r_target)
    return loss, {
        "loss_mean_recon": float(loss.detach()),
        "mean_residual_rms": float(m.detach().pow(2).mean().sqrt()),
    }


# --------------------------------------------------------------------------- #
# Construction / checkpoint plumbing
# --------------------------------------------------------------------------- #
def _load_plain_flow_state(dst: NullSpaceFlow, ckpt_path, device) -> None:
    """Load a plain-``NullSpaceFlow`` checkpoint (``paired_det`` / Stage B) into
    a submodule. Raises on a mismatch rather than silently leaving it random."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state["model"] if "model" in state else state
    dst.load_state_dict(sd)


def build_mean_innovation(
    cfg: Dict[str, Any], channels: int, device, load_ckpts: bool = True
) -> MeanInnovationFlow:
    """Build experiment E's generator from a ``mean_innovation:`` config block.

    ``mean_innovation.mean_ckpt``  -> loaded into (and frozen) ``mean_model``
    ``model.init_from``            -> loaded into ``innovation`` (Stage B warm start)
    """
    mi = dict(cfg.get("mean_innovation", {}))
    mean_model = build_flow(cfg, channels).to(device)
    innovation = build_flow(cfg, channels).to(device)
    model = MeanInnovationFlow(mean_model, innovation).to(device)

    if load_ckpts:
        mean_ckpt = mi.get("mean_ckpt")
        if not mean_ckpt:
            raise ValueError("mean_innovation.enabled requires mean_innovation.mean_ckpt "
                             "(the trained paired_deterministic checkpoint)")
        _load_plain_flow_state(model.mean_model, mean_ckpt, device)

    freeze = bool(mi.get("freeze_mean", True))
    if freeze:
        model.freeze_mean()
    return model


# --------------------------------------------------------------------------- #
# Innovation diagnostics (Part 2) -- run at validation, never in the hot loop
# --------------------------------------------------------------------------- #
@torch.no_grad()
def innovation_diagnostics(
    model: MeanInnovationFlow,
    y: torch.Tensor,
    x_hr: torch.Tensor,
    factor: int,
    n_samples: int = 4,
    n_steps: int = 20,
    n_bins: int = 24,
) -> Dict[str, float]:
    """Ensemble diagnostics for the mean/innovation split on one conditioning ``y``.

    Reports whether mean and innovation power are being double-counted, how the
    ensemble mean compares to the deterministic mean, and ``r(k)`` for the mean,
    the ensemble mean and individual samples. Diagnostic only -- ``no_grad``.

    Note (per spec): no innovation-centering penalty is added. With one HR
    realisation per condition an aggressive per-condition zero-mean constraint
    could suppress valid conditional variation; we *measure* the bias first.
    """
    from .evaluate import rk_tk_summary

    op = model.operator
    m = model.mean_residual(y)                       # (B,C,...) predictable mean
    x_mean = op.A_plus(y) + m

    innovations = [model.sample_innovation(y, n_steps=n_steps) for _ in range(int(n_samples))]
    e_stack = torch.stack(innovations)               # (S,B,C,...)
    x_samples = [op.A_plus(y) + op.P_A(m + e) for e in innovations]
    x_stack = torch.stack(x_samples)

    innovation_mean = e_stack.mean(dim=0)            # mean_z e_hat
    innovation_rms = e_stack.pow(2).mean(dim=0).sqrt()
    sample_mean = x_stack.mean(dim=0)                # mean_z x_hat
    innovation_var = e_stack.var(dim=0, unbiased=True).mean()

    scale = e_stack.pow(2).mean().sqrt().clamp_min(1e-12)
    out: Dict[str, float] = {
        "innovation_mean_rms": float(innovation_mean.pow(2).mean().sqrt()),
        "innovation_rms": float(innovation_rms.mean()),
        "innovation_variance": float(innovation_var),
        # bias of the ensemble mean away from the deterministic mean, normalised
        "sample_mean_minus_x_mean_rms": float((sample_mean - x_mean).pow(2).mean().sqrt()),
        "innovation_mean_over_rms": float(innovation_mean.pow(2).mean().sqrt() / scale),
    }
    # r(k) of the deterministic mean, the ensemble mean, and individual samples
    out.update(rk_tk_summary(x_mean, x_hr, factor, n_bins=n_bins, prefix="xmean_"))
    out.update(rk_tk_summary(sample_mean, x_hr, factor, n_bins=n_bins, prefix="samplemean_"))
    ind = [rk_tk_summary(xs, x_hr, factor, n_bins=n_bins) for xs in x_samples]
    for key in ("rk_low", "rk_transition", "rk_high"):
        out[f"sample_{key}"] = float(sum(d[key] for d in ind) / len(ind))

    # power of mean vs innovation, and a double-counting check:
    # ||P_A(m + e)||^2  vs  ||m||^2 + ||e||^2  (cross term reveals correlation)
    p_mean = float(m.pow(2).mean())
    p_innov = float(e_stack.pow(2).mean())
    p_total = float(torch.stack([op.P_A(m + e).pow(2).mean() for e in innovations]).mean())
    out["power_mean"] = p_mean
    out["power_innovation"] = p_innov
    out["power_total_residual"] = p_total
    # ~0 => additive (no double counting); large => mean and innovation overlap
    out["power_cross_fraction"] = (p_total - p_mean - p_innov) / max(p_total, 1e-30)
    return out
