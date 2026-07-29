"""Exact-consistent stochastic null-space rectified flow.

The generator is

    x_hat = A_plus(y) + P_A(r_theta(y, z))

with the residual ``r`` produced by integrating a rectified-flow ODE that lives
**entirely inside** ``ker A``:

    r_target   = P_A(x_hr)
    z_null     = P_A(z),            z ~ N(0, I)
    r_t        = (1 - t) z_null + t r_target
    v_target   = r_target - z_null
    v_pred     = P_A(v_theta(r_t, t, y))
    loss_flow  = MSE(v_pred, v_target)

Every term is projected, so the state, the initial noise, the target velocity and
the predicted velocity are all null-space fields, and the integrated residual
stays in ``ker A`` for the whole trajectory (checked numerically in
``tests/dmsr/test_flow_nullspace.py``).

Two design notes worth keeping straight
---------------------------------------
**Why ``r_target = P_A(x_hr)`` and not ``P_A(x_hr - A_plus(y))``.** These coincide
only when ``y = A(x_hr)`` exactly. On this dataset they do not: the real LR
simulation differs from ``A(HR)`` by a measured residual ``eta`` (displacement
``eta_frac ~ 0.008``, velocity ``~0.60``) because LR runs have heavier particles
and coarser softening. Taking ``P_A(x_hr)`` makes the regression target depend on
the HR box alone, so ``eta`` never enters the flow objective -- it is absorbed
entirely by the ``A_plus(y)`` base, which is where it belongs. (The octave-cascade
code in ``losses/flow.py`` uses the other form, but its ``y`` is a synthetic
average-pool of the HR crop, where the two are identical.)

**Consistency is structural, not learned.** ``A(x_hat) = y`` holds by
construction. ``||A(x_hat) - y||`` is logged as a diagnostic and is never added to
the loss; doing so would be the known-degenerate objective, which supplies exactly
zero gradient to the null-space residual.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.flow_unet import UNetResidualFlowModel
from ..operators.multiscale import block_upsample
from .encoder import LRConditionEncoder
from .operator import NullSpaceOperator


class NullSpaceFlow(nn.Module):
    """Condition encoder + velocity network + the exact-consistency combine.

    The velocity backbone is the existing, tested
    :class:`~cosmo_sr.models.flow_unet.UNetResidualFlowModel`, instantiated at
    ``factor=s`` and given the encoder's features through its ``context`` input
    (block-upsampled from the LR grid to the HR grid).
    """

    def __init__(
        self,
        channels: int = 6,
        factor: int = 8,
        cond_channels: int = 32,
        encoder_width: int = 64,
        width: int = 64,
        num_levels: int = 3,
        blocks_per_level: int = 1,
        embed_dim: int = 128,
        use_resblocks: bool = True,
        use_attention: bool = False,
        attention_heads: int = 4,
        use_checkpoint: bool = False,
        zero_init_tail: bool = True,
        norm: str = "group",
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.channels = int(channels)
        self.factor = int(factor)
        # When True, `sample_residual` returns the single-shot prediction
        # `P_A(v_theta(0, t=1, y))` instead of integrating the ODE. Required by the
        # `paired_deterministic` baseline: that model is *trained* as a one-shot
        # regressor (see `deterministic_regression_loss`), so integrating it as a
        # velocity field would evaluate a network on a trajectory it was never
        # fitted to -- producing meaningless metrics AND meaningless best-checkpoint
        # selection. Set via `flow.deterministic = True`; it survives the deepcopy
        # into ModelEMA, so evaluation of the EMA model behaves identically.
        self.deterministic = False
        # When True, the null-space projection `P_A` is *removed* from the velocity
        # and the ODE integration, so the model may put content anywhere in the full
        # space and `x_hat = A_plus(y) + r` is no longer exactly consistent. This is
        # the diagnostic ablation that measures the price of the exact-consistency
        # constraint; it is trained by `unconstrained_flow_loss`. Set via
        # `flow.unconstrained = True` (also read from `model.unconstrained` in
        # `build_flow`). Default False keeps Stage C/D/E byte-identical.
        self.unconstrained = False
        # When True, the operator is removed ENTIRELY: no `A_plus(y)` base and no
        # projection. `generate` returns `x_hat = r` (the raw flow/regression
        # output), the target becomes the full HR field `x_hr` (not `x_hr -
        # A_plus(y)`), and the velocity is unprojected. `y` still conditions the
        # model through the encoder -- that is data, not the operator. This is the
        # "no A / A_plus at all" ablation; `a_free` implies unprojected velocity.
        # Set via `flow.a_free = True` (read from `model.a_free` in `build_flow`).
        self.a_free = False
        self.operator = NullSpaceOperator(factor=self.factor)
        self.encoder = LRConditionEncoder(
            in_channels=self.channels, width=encoder_width, cond_channels=cond_channels
        )
        self.velocity_net = UNetResidualFlowModel(
            channels=self.channels,
            width=width,
            num_levels=num_levels,
            blocks_per_level=blocks_per_level,
            padding="same",
            # `norm="channel"` swaps GroupNorm (which reduces over D,H,W and so
            # makes every output voxel depend on whole-crop statistics) for a
            # strictly pointwise channel-group norm. Measured: 92.7% of the
            # context-8 conditioning error is exactly this layer's per-channel
            # affine drift. Default "group" keeps existing runs byte-identical.
            norm=norm,
            # The boxes are periodic and crops are wrapped, so zero padding
            # manufactures a fictitious vacuum at every crop face. Default
            # "zeros" preserves existing behaviour; "circular" matches the LR
            # encoder, which already pads circularly.
            padding_mode=padding_mode,
            activation="silu",
            use_resblocks=use_resblocks,
            use_film=True,
            embed_dim=embed_dim,
            use_attention=use_attention,
            attention_heads=attention_heads,
            zero_init_tail=zero_init_tail,
            use_checkpoint=use_checkpoint,
            context_channels=int(cond_channels),
            factor=self.factor,
        )

    # -- parameter groups (Stage B wants a smaller LR on pretrained weights) -- #
    def encoder_parameters(self):
        return self.encoder.parameters()

    def flow_parameters(self):
        return self.velocity_net.parameters()

    def set_encoder_trainable(self, trainable: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(bool(trainable))

    # -- velocity ----------------------------------------------------------- #
    def velocity(
        self, r_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """``P_A(v_theta(r_t, t, y))`` -- the projected (null-space) velocity."""
        context = block_upsample(self.encoder(y), self.factor)
        R = torch.full(
            (r_t.shape[0],), float(y.shape[-1]), device=r_t.device, dtype=torch.float32
        )
        v_raw = self.velocity_net(r_t, t, y, R, context=context)
        return v_raw if (self.unconstrained or self.a_free) else self.operator.P_A(v_raw)

    forward = velocity

    # -- sampling ----------------------------------------------------------- #
    def sample_residual(
        self,
        y: torch.Tensor,
        n_steps: int = 20,
        z: Optional[torch.Tensor] = None,
        bp_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Integrate the ODE from ``t=0`` to ``t=1``; returns a null-space residual.

        Gradients flow through the integration (Stage C/D need that for
        ``loss_G_adv``). ``bp_steps`` bounds backprop depth to the final ``k``
        Euler steps so training memory does not scale with ``n_steps`` -- the same
        device already used by ``inference/flow_sample.integrate_residual``.
        """
        b, c = y.shape[0], y.shape[1]
        n_hr = y.shape[-1] * self.factor
        if self.deterministic:
            # One-shot regression baseline: no ODE, no z dependence (so
            # sample_diversity correctly reports 0 for it).
            zeros = torch.zeros(b, c, n_hr, n_hr, n_hr, device=y.device, dtype=y.dtype)
            t_one = torch.ones(b, device=y.device, dtype=y.dtype)
            return self.velocity(zeros, t_one, y)
        if z is None:
            z = torch.randn(b, c, n_hr, n_hr, n_hr, device=y.device, dtype=y.dtype)
        # `_proj` is the identity for the unconstrained / a_free ablations, so the
        # initial noise and every Euler step stay in the full space rather than `ker A`.
        _proj = (lambda u: u) if (self.unconstrained or self.a_free) else self.operator.P_A
        r = _proj(z)
        dt = 1.0 / int(n_steps)
        cutoff = 0 if bp_steps is None else max(int(n_steps) - int(bp_steps), 0)
        for i in range(int(n_steps)):
            t = torch.full((b,), i * dt, device=y.device, dtype=y.dtype)
            if i < cutoff:
                with torch.no_grad():
                    r = _proj(r + dt * self.velocity(r, t, y))
            else:
                r = _proj(r + dt * self.velocity(r, t, y))
        return r

    def generate(
        self,
        y: torch.Tensor,
        n_steps: int = 20,
        z: Optional[torch.Tensor] = None,
        bp_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Full generator ``x_hat = A_plus(y) + P_A(r)``. Exactly consistent with ``y``.

        With ``a_free`` the operator is gone: ``x_hat = r`` (no ``A_plus(y)`` base).
        """
        r = self.sample_residual(y, n_steps=n_steps, z=z, bp_steps=bp_steps)
        if self.a_free:
            return r
        return self.operator.A_plus(y) + r


# --------------------------------------------------------------------------- #
# Training loss
# --------------------------------------------------------------------------- #
def deterministic_regression_loss(
    model: NullSpaceFlow, y: torch.Tensor, x_hr: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Deterministic null-space regression baseline: ``MSE(P_A(f(y)), P_A(x_hr))``.

    The ``paired_deterministic`` control. It reuses the *identical* backbone with the
    stochastic input pinned (``r_t = 0``, ``t = 1``), so the comparison against Stage A
    isolates stochasticity rather than confounding it with architecture or capacity.

    Expected to under-disperse: regressing a near-white stochastic target to its
    conditional mean is exactly the mean-collapse this project already measured
    (generated residual carrying ~half the true power). That is the point of having
    it as a baseline.
    """
    op = model.operator
    r_target = op.P_A(x_hr)
    zeros = torch.zeros_like(x_hr)
    t_one = torch.ones(x_hr.shape[0], device=x_hr.device, dtype=x_hr.dtype)
    r_pred = model.velocity(zeros, t_one, y)
    loss = F.mse_loss(r_pred, r_target)
    return loss, {
        "loss_flow": float(loss.detach()),
        "target_residual_rms": float(r_target.detach().pow(2).mean().sqrt()),
        "pred_residual_rms": float(r_pred.detach().pow(2).mean().sqrt()),
    }


def null_space_flow_loss(
    model: NullSpaceFlow,
    y: torch.Tensor,
    x_hr: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    return_fields: bool = False,
):
    """Rectified-flow matching restricted to ``ker A``. Returns ``(loss, metrics)``.

    ``z`` and ``return_fields`` are additive, default-off hooks for the offline
    Fourier diagnostic (Part 3): with ``z=None`` and ``return_fields=False`` the
    control flow, RNG draws and numerics are **byte-identical** to the Stage C
    objective, so this remains a drop-in. When ``return_fields=True`` a third
    element ``{'v_pred', 'v_target', 't'}`` (all detached) is returned so the
    diagnostic can reuse the exact velocities rather than perturbing training RNG.
    """
    op = model.operator
    r_target = op.P_A(x_hr)
    z_null = op.P_A(torch.randn_like(x_hr) if z is None else z)

    b = x_hr.shape[0]
    if t is None:
        t = torch.rand(b, device=x_hr.device, dtype=x_hr.dtype)
    t_b = t.view(b, *([1] * (x_hr.dim() - 1)))

    r_t = (1.0 - t_b) * z_null + t_b * r_target
    v_target = r_target - z_null
    v_pred = model.velocity(r_t, t, y)

    loss = F.mse_loss(v_pred, v_target)
    metrics = {
        "loss_flow": float(loss.detach()),
        "target_residual_rms": float(r_target.detach().pow(2).mean().sqrt()),
    }
    if return_fields:
        return loss, metrics, {
            "v_pred": v_pred.detach(), "v_target": v_target.detach(), "t": t.detach(),
        }
    return loss, metrics


def unconstrained_flow_loss(
    model: NullSpaceFlow,
    y: torch.Tensor,
    x_hr: torch.Tensor,
    lambda_cons: float = 0.0,
    t: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    return_fields: bool = False,
):
    """Rectified-flow matching with the null-space projection **removed** -- the
    ablation for the exact-consistency constraint (question (b)).

    Differences from :func:`null_space_flow_loss`, and *only* these:

    * target is the **full** residual ``r_target = x_hr - A_plus(y)`` (range + null),
      not ``P_A(x_hr)``. On this dataset the two differ by ``A_plus(eta)``, the
      LR-sim mismatch, so the model is now *allowed to correct the LR large scales*
      instead of trusting them.
    * the initial noise ``z`` is full-space, not ``P_A(z)``.
    * ``model.velocity`` is unprojected (requires ``model.unconstrained is True``).

    ``x_hat = A_plus(y) + r`` is therefore only approximately consistent. The
    range-space leakage ``A(x_hat) - y = A(r_hat_1)`` (one-step endpoint estimate)
    is always logged as ``soft_consistency_rms``; with ``lambda_cons > 0`` its mean
    square is added to the loss as a soft penalty (reference only -- set 0 for a
    fully free generator).
    """
    if not model.unconstrained:
        raise ValueError("unconstrained_flow_loss requires model.unconstrained = True")
    op = model.operator
    r_target = x_hr - op.A_plus(y)                 # full residual, unprojected
    z_full = torch.randn_like(x_hr) if z is None else z

    b = x_hr.shape[0]
    if t is None:
        t = torch.rand(b, device=x_hr.device, dtype=x_hr.dtype)
    t_b = t.view(b, *([1] * (x_hr.dim() - 1)))

    r_t = (1.0 - t_b) * z_full + t_b * r_target
    v_target = r_target - z_full
    v_pred = model.velocity(r_t, t, y)             # unprojected (model.unconstrained)

    loss_flow = F.mse_loss(v_pred, v_target)
    # Euler one-step endpoint estimate r_hat(t=1) = r_t + (1 - t) v_pred; its range
    # component A(r_hat_1) is exactly A(x_hat) - y, since A(A_plus(y)) = y.
    r_hat_1 = r_t + (1.0 - t_b) * v_pred
    a_leak = op.A(r_hat_1)
    soft_cons = a_leak.pow(2).mean()

    loss = loss_flow + (float(lambda_cons) * soft_cons if lambda_cons else 0.0)
    metrics = {
        "loss_flow": float(loss_flow.detach()),
        "target_residual_rms": float(r_target.detach().pow(2).mean().sqrt()),
        "soft_consistency_rms": float(a_leak.detach().pow(2).mean().sqrt()),
    }
    if return_fields:
        return loss, metrics, {
            "v_pred": v_pred.detach(), "v_target": v_target.detach(), "t": t.detach(),
        }
    return loss, metrics


def deterministic_free_loss(
    model: NullSpaceFlow, y: torch.Tensor, x_hr: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Deterministic (one-shot MSE) regression with the projection removed.

    Two base modes, selected by the model flags -- neither uses ``P_A``:

    * ``model.unconstrained`` (base kept): ``x_hat = A_plus(y) + f(y)``,
      ``loss = MSE(f(y), x_hr - A_plus(y))``  -- "MSE with A_plus, but unconstrained".
    * ``model.a_free`` (operator gone): ``x_hat = f(y)``,
      ``loss = MSE(f(y), x_hr)``              -- "MSE with no A / A_plus at all".

    ``f(y) = velocity(0, t=1, y)`` is the *unprojected* one-shot prediction (the
    velocity net already returns the raw field when ``unconstrained or a_free``), so
    this mirrors :func:`deterministic_regression_loss` with the null-space
    restriction lifted. ``A(x_hat) - y`` is logged as a diagnostic only.
    """
    a_free = bool(getattr(model, "a_free", False))
    if not (model.unconstrained or a_free):
        raise ValueError("deterministic_free_loss requires model.unconstrained or model.a_free")
    op = model.operator
    base = torch.zeros_like(x_hr) if a_free else op.A_plus(y)
    target = x_hr - base
    zeros = torch.zeros_like(x_hr)
    t_one = torch.ones(x_hr.shape[0], device=x_hr.device, dtype=x_hr.dtype)
    r_pred = model.velocity(zeros, t_one, y)
    loss = F.mse_loss(r_pred, target)
    a_leak = op.A(base + r_pred) - y
    return loss, {
        "loss_flow": float(loss.detach()),
        "target_residual_rms": float(target.detach().pow(2).mean().sqrt()),
        "pred_residual_rms": float(r_pred.detach().pow(2).mean().sqrt()),
        "soft_consistency_rms": float(a_leak.detach().pow(2).mean().sqrt()),
    }


def a_free_flow_loss(
    model: NullSpaceFlow,
    y: torch.Tensor,
    x_hr: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    return_fields: bool = False,
):
    """Rectified-flow matching with **no operator at all** -- no ``A_plus`` base and
    no projection.

    The flow regresses the **full** HR field ``r_target = x_hr`` from full-space
    noise, conditioned on ``y`` only through the encoder, and ``x_hat = r`` (the
    integrated endpoint). Contrast :func:`unconstrained_flow_loss`, which keeps the
    ``A_plus(y)`` base and regresses ``x_hr - A_plus(y)``. Requires
    ``model.a_free is True``. ``A(x_hat) - y`` is logged as ``soft_consistency_rms``
    (diagnostic only; never added to the loss -- with no base there is nothing to
    anchor it to).
    """
    if not bool(getattr(model, "a_free", False)):
        raise ValueError("a_free_flow_loss requires model.a_free = True")
    op = model.operator
    r_target = x_hr                                    # full field; no base subtracted
    z_full = torch.randn_like(x_hr) if z is None else z

    b = x_hr.shape[0]
    if t is None:
        t = torch.rand(b, device=x_hr.device, dtype=x_hr.dtype)
    t_b = t.view(b, *([1] * (x_hr.dim() - 1)))

    r_t = (1.0 - t_b) * z_full + t_b * r_target
    v_target = r_target - z_full
    v_pred = model.velocity(r_t, t, y)                 # unprojected (a_free)

    loss = F.mse_loss(v_pred, v_target)
    r_hat_1 = r_t + (1.0 - t_b) * v_pred               # Euler one-step endpoint x_hat
    a_leak = op.A(r_hat_1) - y
    metrics = {
        "loss_flow": float(loss.detach()),
        "target_residual_rms": float(r_target.detach().pow(2).mean().sqrt()),
        "soft_consistency_rms": float(a_leak.detach().pow(2).mean().sqrt()),
    }
    if return_fields:
        return loss, metrics, {
            "v_pred": v_pred.detach(), "v_target": v_target.detach(), "t": t.detach(),
        }
    return loss, metrics


# --------------------------------------------------------------------------- #
# Checkpoint plumbing
# --------------------------------------------------------------------------- #
def load_pretrained_encoder(
    model: NullSpaceFlow, ckpt_path, strict: bool = True
) -> Dict[str, Any]:
    """Initialise ``model.encoder`` from an SSL checkpoint written by
    :meth:`cosmo_sr.dmsr.encoder.LRMaskedAutoencoder.save_encoder`.

    Raises on a shape/architecture mismatch rather than silently leaving the
    encoder random -- a Stage B run that quietly failed to load its pretrained
    weights would be an invalid ablation that looks fine in the logs.
    """
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "encoder" not in blob:
        raise KeyError(
            f"{ckpt_path} has no 'encoder' entry (keys: {sorted(blob)}); expected a "
            "checkpoint from LRMaskedAutoencoder.save_encoder"
        )
    missing, unexpected = model.encoder.load_state_dict(blob["encoder"], strict=strict)
    return {
        "encoder_ckpt": str(ckpt_path),
        "encoder_config": blob.get("encoder_config"),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def build_flow(cfg: Dict[str, Any], channels: int) -> NullSpaceFlow:
    """Construct a :class:`NullSpaceFlow` from a ``model:`` config block."""
    m = dict(cfg.get("model", {}))
    flow = NullSpaceFlow(
        channels=channels,
        factor=int(cfg.get("factor", 8)),
        cond_channels=int(m.get("cond_channels", 32)),
        encoder_width=int(m.get("encoder_width", 64)),
        width=int(m.get("width", 64)),
        num_levels=int(m.get("num_levels", 3)),
        blocks_per_level=int(m.get("blocks_per_level", 1)),
        embed_dim=int(m.get("embed_dim", 128)),
        use_resblocks=bool(m.get("use_resblocks", True)),
        use_attention=bool(m.get("use_attention", False)),
        attention_heads=int(m.get("attention_heads", 4)),
        use_checkpoint=bool(m.get("grad_checkpoint", False)),
        zero_init_tail=bool(m.get("zero_init_tail", True)),
        norm=str(m.get("norm", "group")),
        padding_mode=str(m.get("padding_mode", "zeros")),
    )
    # Ablation (b): remove the exact-consistency projection. Absent key -> False,
    # so Stage C/D/E resolve identically.
    flow.unconstrained = bool(m.get("unconstrained", False))
    # Ablation: remove the operator entirely (no A_plus base, no projection).
    flow.a_free = bool(m.get("a_free", False))
    if flow.unconstrained and flow.a_free:
        raise ValueError("model.unconstrained and model.a_free are mutually exclusive")
    return flow
