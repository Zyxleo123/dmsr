"""Direct reward-guided fine-tuning of the SR2 generator.

The objective, written out
--------------------------
::

    L = -Q_safe
        + lambda_P    * L_Pdelta
        + lambda_low  * L_low
        + lambda_prox * L_prox
        + lambda_div  * [d_min - d_struct]_+^2

``Q_safe``
    Ensemble lower confidence bound on the predicted reward improvement,
    ``mean_m Q_m - beta * std_m Q_m``, where ``Q_m`` is the swap-form
    ``dR = R(S_0 - s_{0,j} + s_hat_{theta,j}) - R(S_0)`` from proxy member ``m``.
    The proxy is *frozen but differentiable*: its parameters do not move, and the
    gradient passes through it into SR2.
``L_Pdelta``
    Squared log-density-power difference from the paired HR crop, on valid-centre
    CIC. The density guard, as a loss rather than only as a gate, because a term
    that only appears at checkpoint time gives the optimiser thousands of steps
    to walk somewhere it then has to be rejected from.
``L_low``
    Relative difference between the block-averaged fine-tuned and frozen outputs
    at the same ``(Y, z)``. Keeps the LR-visible scales -- the ones SR2 already
    gets right -- where they were.
``L_prox``
    Per-tensor distance from ``theta_0``. Per-tensor, not global: a global norm
    is dominated by the largest weight tensor and would leave the small ones
    (including every ``std``) effectively unconstrained.
``d_struct``
    Spread across noise seeds in displacement / density / soft-peak features.
    A one-sided hinge: exceeding the floor costs nothing.

There is deliberately **no** full-field L2 term, to HR or to frozen SR2. An L2 to
HR asks the generator to predict the unpredictable realisation of high-k modes
and is minimised by blurring -- the precise failure that makes SR2's occupation
curve flat. An L2 to frozen SR2 asks it not to change, which is the opposite of
the intent; ``L_prox`` and ``L_low`` already bound the change in the two places
where bounding it is meaningful.

Operational care
----------------
* **Reward warm-up.** The reward weight ramps from zero. The guards are exact
  and the reward is a surrogate, so the surrogate does not get to move anything
  until the guards are established.
* **Per-loss gradient norms.** Logged every ``grad_norm_every`` steps by
  differentiating each term separately. Without them "the reward term is not
  overwhelming the guards" is an assumption; with them it is a number.
* **EMA.** The evaluated checkpoint is the EMA one. Reward-following training is
  noisy and the last iterate is not the best estimate of where it has got to.
* **AMP** around the generator forward only. The reward head is float64 (see
  :mod:`cosmo_sr.reward.torch_reward`) and the guards are cheap; running them in
  half precision would buy nothing and cost the parity that makes them
  comparable to the gates.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..reward.arms import check_arm
from ..reward.catalog_proxy import ProxyEnsemble
from ..reward.phase_space import (
    PhaseSpaceConfig, arm_paired_features, phase_space_paired_grid,
)
from ..reward.soft_rockstar import SoftRockstarConfig, soft_rockstar_paired_tokens
from ..reward.sr2_adversarial import critic_input
from ..reward.soft_structure import (
    SoftStructureConfig, density_from_disp, paired_features, structural_diversity,
)
from ..reward.torch_reward import TorchRewardModel, TorchSummary
from ..tts.srs_noise import ControlledG, load_controlled_generator
from . import sr2_unfreeze
from .sr2_finetune_data import SR2TileGeometry, fold_draws, trim_to_tile

__all__ = [
    "DirectFinetuneConfig",
    "DirectFinetuneTrainer",
    "EMA",
    "attach_summaries",
    "block_average_torch",
    "log_density_power",
]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def block_average_torch(x: torch.Tensor, factor: int) -> torch.Tensor:
    """``A``: mean over non-overlapping ``factor^3`` blocks, ``(B, C, N, ...)``.

    The torch twin of :func:`cosmo_sr.reward.fields.block_average`, which is the
    same operator the LR-consistency constraint and the degrader use. Any
    difference between the two would make the training-time guard and the
    checkpoint-time gate measure different things.
    """
    if x.dim() != 5:
        raise ValueError(f"expected (B, C, N, N, N), got {tuple(x.shape)}")
    b, c, n = x.shape[0], x.shape[1], x.shape[-1]
    f = int(factor)
    if n % f:
        raise ValueError(f"block_average: {n} not divisible by {f}")
    m = n // f
    return x.reshape(b, c, m, f, m, f, m, f).mean(dim=(3, 5, 7))


def log_density_power(delta: torch.Tensor, n_bands: int = 8) -> torch.Tensor:
    """``(B, n_bands)`` log radial power of a ``(B, 1, R, R, R)`` overdensity."""
    from ..reward.soft_structure import _radial_log_power

    return _radial_log_power(delta, int(n_bands))


class EMA:
    """Exponential moving average of the trainable parameters.

    Only the trainable ones: the frozen tensors are identical to ``theta_0`` by
    construction, and averaging them would be a no-op that quietly doubles the
    checkpoint's memory.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: Mapping[str, torch.Tensor]) -> None:
        self.shadow = {k: torch.as_tensor(v).clone() for k, v in sd.items()}

    @contextlib.contextmanager
    def applied(self, model: nn.Module):
        """Temporarily swap the EMA weights in, e.g. to evaluate or to save."""
        backup = {n: p.detach().clone() for n, p in model.named_parameters()
                  if n in self.shadow}
        try:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in self.shadow:
                        p.copy_(self.shadow[n])
            yield model
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in backup:
                        p.copy_(backup[n])


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class DirectFinetuneConfig:
    """Everything the actor step needs. Read from ``sr2_direct_finetune.yaml``."""

    rung: str = "proj_noise"
    group_lr: Dict[str, float] = dc_field(
        default_factory=lambda: dict(sr2_unfreeze.DEFAULT_GROUP_LR))
    weight_decay: float = 0.0
    max_steps: int = 2000

    # -- objective weights --
    beta_uncertainty: float = 1.0
    w_joint_reward: float = 0.25
    w_occ_reward: float = 1.0
    lambda_reward: float = 1.0
    lambda_density_power: float = 10.0
    lambda_low_k: float = 100.0
    lambda_prox: float = 1.0
    lambda_diversity: float = 10.0
    d_struct_min: float = 0.05
    reward_warmup_steps: int = 200

    # -- numerics / bookkeeping --
    grad_clip: float = 1.0
    amp: bool = True
    ema_decay: float = 0.999
    grad_norm_every: int = 25
    log_every: int = 10
    checkpoint_every: int = 250
    seed: int = 0
    density_power_bands: int = 8

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["group_lr"] = {k: float(v) for k, v in self.group_lr.items()}
        return d

    @staticmethod
    def from_dict(cfg: Mapping) -> "DirectFinetuneConfig":
        c = dict(cfg or {})
        base = DirectFinetuneConfig()
        known = set(asdict(base))
        unknown = [k for k in c if k not in known]
        if unknown:
            # A silently ignored key in a config that controls a safety guard is
            # the worst possible failure mode: the run looks configured and is
            # not.
            raise KeyError(
                f"unknown actor config keys {sorted(unknown)}; expected "
                f"{sorted(known)}"
            )
        lrs = dict(base.group_lr)
        lrs.update({k: float(v) for k, v in dict(c.get("group_lr", {})).items()})
        out = DirectFinetuneConfig(**{k: v for k, v in c.items() if k != "group_lr"})
        out.group_lr = lrs
        return out


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class DirectFinetuneTrainer:
    """One actor: a trainable SR2, its frozen twin, and the guarded objective.

    The frozen twin is a *second copy of the same checkpoint*, not the same
    module with gradients off. Both are needed simultaneously in every step --
    the fine-tuned output and the baseline at the same ``(Y, z)`` -- so they
    cannot be the same object.
    """

    def __init__(
        self,
        model_path: str | Path,
        proxies: ProxyEnsemble,
        reward: TorchRewardModel,
        *,
        cfg: Optional[DirectFinetuneConfig] = None,
        geom: Optional[SR2TileGeometry] = None,
        soft_cfg: Optional[SoftStructureConfig] = None,
        device=None,
        actor: Optional[ControlledG] = None,
        frozen: Optional[ControlledG] = None,
        chan_kwargs: Optional[Mapping] = None,
        arm: str = "a",
        phase_cfg: Optional[PhaseSpaceConfig] = None,
        rockstar_cfg: Optional[SoftRockstarConfig] = None,
    ):
        self.cfg = cfg or DirectFinetuneConfig()
        self.geom = geom or SR2TileGeometry()
        self.soft_cfg = soft_cfg or SoftStructureConfig()
        # Which arm's differentiable extractor feeds the frozen proxy. The
        # extractors are the SAME functions the labelled features came from
        # (phase_space / soft_rockstar), so the proxy sees at fine-tuning time
        # exactly what it saw at training time.
        self.arm = check_arm(arm)
        self.phase_cfg = phase_cfg or PhaseSpaceConfig()
        self.rockstar_cfg = rockstar_cfg or SoftRockstarConfig()
        # Arm F's fine density is inverse-pixel-shuffled at this multiple; 2 is
        # the SR2 recipe and matches arm F's offline provider.
        self.f_grid_mult = int(getattr(self.cfg, "density_grid_mult", 2) or 2)
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(int(self.cfg.seed))

        # `eval_mode=False` deliberately. FrozenSR2Base is NOT used for the
        # actor: it exists to guarantee `requires_grad = False`, which is
        # exactly what an actor must not have.
        # `chan_kwargs` exists because the channel widths are a property of the
        # checkpoint, not a default: `load_controlled_generator` assumes the
        # production `chan_base=512`, and a mismatch surfaces as a wall of
        # size-mismatch errors rather than as "wrong widths".
        ck = dict(chan_kwargs or {})
        self.actor = actor if actor is not None else load_controlled_generator(
            model_path, scale_factor=self.geom.scale_factor,
            device=self.device, eval_mode=False, **ck)
        self.frozen = frozen if frozen is not None else load_controlled_generator(
            model_path, scale_factor=self.geom.scale_factor,
            device=self.device, eval_mode=True, **ck)
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        self.frozen.eval()

        self.proxies = proxies.to(self.device).freeze()
        self.reward = reward.to(self.device)

        self.theta0 = {n: p.detach().clone()
                       for n, p in self.actor.named_parameters()}
        self.groups = sr2_unfreeze.parameter_groups(
            self.actor, self.cfg.rung, self.cfg.group_lr,
            weight_decay=float(self.cfg.weight_decay))
        self.trainable = sr2_unfreeze.trainable_names(self.actor)
        self.opt = torch.optim.Adam(self.groups)
        self.ema = EMA(self.actor, decay=float(self.cfg.ema_decay))
        self.step_index = 0
        self.use_amp = bool(self.cfg.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # -- pieces ------------------------------------------------------------ #
    def reward_weight(self) -> float:
        """Linear warm-up of the reward term from 0 to ``lambda_reward``."""
        w = int(self.cfg.reward_warmup_steps)
        if w <= 0:
            return float(self.cfg.lambda_reward)
        return float(self.cfg.lambda_reward) * min(1.0, self.step_index / float(w))

    def _forward(self, lr: torch.Tensor, noise: Dict[str, torch.Tensor]):
        """``(candidate, frozen)`` tile outputs for the same ``(Y, z)``."""
        # float16 rather than bfloat16: the pool this runs on is mixed
        # (v100 / a5000 / a6000 / h200) and bf16 is not available on all of it.
        # float16 needs the GradScaler, which is why one is constructed.
        ctx = (torch.amp.autocast("cuda", dtype=torch.float16)
               if self.use_amp else contextlib.nullcontext())
        with ctx:
            cand = trim_to_tile(self.actor(lr, noise=noise), self.geom)
        cand = cand.float()
        with torch.no_grad():
            base = trim_to_tile(self.frozen(lr, noise=noise), self.geom).float()
        return cand, base

    def _prox_loss(self) -> torch.Tensor:
        """Mean over trainable tensors of ``||theta - theta_0||^2 / ||theta_0||^2``.

        Relative per tensor, so ``blocks.2.conv.5.weight`` (tens of thousands of
        entries, O(0.01) each) and ``blocks.0.conv.0.std`` (64 entries, O(1e-3)
        each) are held to the same *fractional* movement rather than the same
        absolute one.
        """
        terms: List[torch.Tensor] = []
        for n, p in self.actor.named_parameters():
            if not p.requires_grad:
                continue
            ref = self.theta0[n]
            den = ref.pow(2).sum().clamp_min(1e-12)
            terms.append((p - ref).pow(2).sum() / den)
        if not terms:
            return torch.zeros((), device=self.device)
        return torch.stack(terms).mean()

    def _density_power_loss(self, cand: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        d_c = density_from_disp(cand, self.soft_cfg)
        with torch.no_grad():
            d_h = density_from_disp(hr, self.soft_cfg)
        p_c = log_density_power(d_c, int(self.cfg.density_power_bands))
        p_h = log_density_power(d_h, int(self.cfg.density_power_bands))
        return ((p_c - p_h.detach()) ** 2).mean()

    def _low_k_loss(self, cand: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """``||A(Psi_theta) - A(Psi_0)||^2 / ||A(Psi_0)||^2`` -- the SQUARED ratio.

        The gate measures the un-squared ``low_k_change`` (an RMS ratio) and
        :meth:`low_k_change` reports it, but the *loss* must be the square.
        ``sqrt`` has infinite derivative at zero, and at step zero the actor is
        the frozen generator, so this term is exactly zero and its gradient was
        ``inf`` -- which propagated to NaN weights within three steps. The
        squared form is smooth, has zero gradient at zero, and is monotone in
        the same quantity, so it penalises exactly what the gate blocks on.
        """
        a_c = block_average_torch(cand, self.geom.scale_factor)
        a_b = block_average_torch(base, self.geom.scale_factor).detach()
        den = a_b.pow(2).mean().clamp_min(1e-30)
        return (a_c - a_b).pow(2).mean() / den

    @staticmethod
    def low_k_change(loss_value: float) -> float:
        """The gate-comparable RMS ratio from the squared loss term."""
        return float(max(float(loss_value), 0.0) ** 0.5)

    def _extract(self, cand: torch.Tensor, base: torch.Tensor,
                 lr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The trainable candidate's proxy features, arm-appropriately.

        Each is the EXACT function the labelled training features came from --
        the whole contract is that a proxy fine-tuned against features it was not
        trained on is answering a different question than the one it was gated on:

        * ``a`` density summaries, ``b`` + phase space (flat vectors);
        * ``c`` and ``d`` the SAME soft-Rockstar token grid (arm D differs only in
          the model that reads it, so its live extractor IS arm C's);
        * ``e`` the dense 32^3 phase-space grid;
        * ``f`` the 20-channel SR2 critic input, which needs the LR conditioning --
          the unpadded LR core (``lr[pad:pad+chunk]``) matching the output tile,
          exactly the region arm F's offline provider crops.
        """
        if self.arm == "a":
            return paired_features(cand, base, self.soft_cfg)
        if self.arm == "b":
            return arm_paired_features(cand, base, "b", self.soft_cfg,
                                       self.phase_cfg)
        if self.arm in ("c", "d"):
            return soft_rockstar_paired_tokens(cand, base, self.soft_cfg,
                                               self.phase_cfg, self.rockstar_cfg)
        if self.arm == "e":
            return phase_space_paired_grid(cand, base, self.soft_cfg, self.phase_cfg)
        # arm f: build the exact SR2 critic input from the generated tile and the
        # LR core. No candidate-minus-frozen block -- F keeps its original input.
        if lr is None:
            raise ValueError("arm 'f' needs the LR conditioning to build critic_input")
        p, c = int(self.geom.pad), int(self.geom.chunk)
        lr_core = lr[:, :, p:p + c, p:p + c, p:p + c]
        return critic_input(lr_core, cand, cellsize_kpc_h=self.soft_cfg.cellsize_kpc_h,
                            dis_norm_kpc_h=float(self.soft_cfg.dis_norm_kpc_h),
                            grid_mult=int(self.f_grid_mult))

    # -- the step ---------------------------------------------------------- #
    def loss_terms(self, batch: Mapping) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Every loss term, unweighted, plus diagnostics.

        Returned separately from the weighted sum so the per-term gradient norms
        can be taken without recomputing the forward pass, and so a log row shows
        what each guard actually measured rather than its contribution to a total.

        Draw 0 versus the other draws: the proxy reward and its catalog-context
        swap use **draw 0 only**. Draw 0 is the base seed, whose frozen catalog
        context is the one the labels were measured against; scoring the other
        draws against that context would ask the proxy about (seed, context)
        pairs it never saw. The second draw exists for the diversity term, and
        the exact field guards (density power, low-k) run on every draw --
        they are guards, they are cheap, and a draw that broke the field would
        otherwise be invisible.
        """
        lr, noise, b, d = fold_draws(
            {"lr": batch["lr"].to(self.device),
             "noise": {k: v.to(self.device) for k, v in batch["noise"].items()}}
        )
        hr = batch["hr"].to(self.device)
        cand, base = self._forward(lr, noise)

        # HR is per (box, tile); the draws share it.
        hr_rep = hr.unsqueeze(1).expand(b, d, *hr.shape[1:]).reshape(b * d, *hr.shape[1:])

        # fold_draws lays the batch out as (example, draw) -> example * d + draw,
        # so draw 0 is every d-th row.
        feats = self._extract(cand[::d], base[::d], lr=lr[::d])
        box = batch["box_summary"].to(self.device)
        frozen_tile = batch["frozen_tile_summary"].to(self.device)
        all_dr = self.proxies.delta_rewards_all(
            self.reward, feats.to(torch.float64), box, frozen_tile,
            w_joint=float(self.cfg.w_joint_reward),
            w_occ=float(self.cfg.w_occ_reward),
        )
        q = ProxyEnsemble.q_safe(all_dr["dR_combined"],
                                 beta=float(self.cfg.beta_uncertainty))

        terms: Dict[str, torch.Tensor] = {
            "reward": -q["q_safe"].mean().to(torch.float32),
            "density_power": self._density_power_loss(cand, hr_rep),
            "low_k": self._low_k_loss(cand, base),
            "prox": self._prox_loss(),
        }

        diag: Dict[str, float] = {
            "q_mean": float(q["mean"].detach().mean()),
            "q_std": float(q["std"].detach().mean()),
            "q_safe": float(q["q_safe"].detach().mean()),
            "n_draws": float(d),
            "n_reward_rows": float(b),
        }
        # All three rewards, always: the joint score can fall because abundance
        # improved while occupation stayed flat, and occupation is the target.
        for key in ("dR_cat", "dR_occ", "dR_abund", "clamped_fraction"):
            diag[key] = float(all_dr[key].detach().mean())

        if d >= 2:
            div = structural_diversity(
                cand.reshape(b, d, *cand.shape[1:]), self.soft_cfg)
            ds = div["d_struct"]
            terms["diversity"] = F.relu(float(self.cfg.d_struct_min) - ds).pow(2).mean()
            for k, v in div.items():
                diag[f"div_{k}"] = float(v.detach().mean())
        else:
            # Not silently zero: with one draw the collapse detector cannot fire
            # at all, and a run configured that way must say so in its log.
            terms["diversity"] = torch.zeros((), device=self.device)
            diag["div_d_struct"] = float("nan")
            diag["diversity_inactive"] = 1.0
        return terms, diag

    def weighted_sum(self, terms: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        w = {
            "reward": self.reward_weight(),
            "density_power": float(self.cfg.lambda_density_power),
            "low_k": float(self.cfg.lambda_low_k),
            "prox": float(self.cfg.lambda_prox),
            "diversity": float(self.cfg.lambda_diversity),
        }
        total = sum(w[k] * terms[k] for k in terms)
        return total, w

    def _per_term_grad_norms(self, terms: Mapping[str, torch.Tensor],
                             weights: Mapping[str, float]) -> Dict[str, float]:
        """Weighted gradient norm of each term w.r.t. the trainable parameters.

        This is the measurement that answers "is the reward overwhelming the
        guards". It costs one extra backward per term, so it runs every
        ``grad_norm_every`` steps rather than every step.
        """
        params = [p for p in self.actor.parameters() if p.requires_grad]
        out: Dict[str, float] = {}
        for k, t in terms.items():
            if not t.requires_grad:
                out[f"gradnorm_{k}"] = 0.0
                continue
            g = torch.autograd.grad(float(weights[k]) * t, params,
                                    retain_graph=True, allow_unused=True)
            n2 = sum(float(x.detach().pow(2).sum()) for x in g if x is not None)
            out[f"gradnorm_{k}"] = float(math.sqrt(n2))
        return out

    def step(self, batch: Mapping) -> Dict[str, float]:
        """One optimizer step. Returns a flat metrics row, ready for JSONL."""
        self.actor.train()
        self.opt.zero_grad(set_to_none=True)
        terms, diag = self.loss_terms(batch)
        total, weights = self.weighted_sum(terms)

        metrics: Dict[str, float] = {
            "step": float(self.step_index),
            "loss": float(total.detach()),
            "reward_weight": float(weights["reward"]),
            **{f"loss_{k}": float(v.detach()) for k, v in terms.items()},
            "low_k_change": self.low_k_change(float(terms["low_k"].detach())),
            **{f"weight_{k}": float(v) for k, v in weights.items()},
            **diag,
        }
        take_norms = (int(self.cfg.grad_norm_every) > 0
                      and self.step_index % int(self.cfg.grad_norm_every) == 0)
        if take_norms:
            metrics.update(self._per_term_grad_norms(terms, weights))

        if self.use_amp:
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.opt)
        else:
            total.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in self.actor.parameters() if p.requires_grad],
            float(self.cfg.grad_clip))
        metrics["grad_norm_total"] = float(gn)
        if self.use_amp:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()
        self.ema.update(self.actor)
        self.step_index += 1
        return metrics

    # -- checkpoints ------------------------------------------------------- #
    def state_dict(self) -> Dict:
        return {
            "step": int(self.step_index),
            "config": self.cfg.to_dict(),
            "rung": str(self.cfg.rung),
            "trainable": list(self.trainable),
            "model": {k: v.detach().cpu() for k, v in self.actor.state_dict().items()},
            "ema": {k: v.detach().cpu() for k, v in self.ema.state_dict().items()},
            "optimizer": self.opt.state_dict(),
        }

    def save(self, path: str | Path, extra: Optional[Mapping] = None) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = self.state_dict()
        if extra:
            blob["extra"] = dict(extra)
        tmp = p.with_suffix(p.suffix + ".tmp")
        torch.save(blob, tmp)
        tmp.replace(p)
        return p

    def save_ema_generator(self, path: str | Path,
                           extra: Optional[Mapping] = None) -> Path:
        """Write an EMA checkpoint in the ``{'model': state_dict}`` SR2 layout.

        Deliberately the *upstream* layout, so the result loads through
        :func:`cosmo_sr.tts.srs_noise.load_controlled_generator` exactly like
        ``G_z0.pt`` and every existing evaluation path can consume it with no
        special case.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with self.ema.applied(self.actor):
            sd = {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()}
        blob: Dict = {"model": sd, "epoch": int(self.step_index),
                      "source": "sr2_direct_finetune_ema"}
        if extra:
            blob["extra"] = dict(extra)
        tmp = p.with_suffix(p.suffix + ".tmp")
        torch.save(blob, tmp)
        tmp.replace(p)
        return p

    def load(self, path: str | Path, *, map_location=None) -> Dict:
        blob = torch.load(str(path), map_location=map_location or self.device,
                          weights_only=False)
        self.actor.load_state_dict(blob["model"])
        self.ema.load_state_dict(blob["ema"])
        self.opt.load_state_dict(blob["optimizer"])
        self.step_index = int(blob.get("step", 0))
        # requires_grad flags do not travel in a state_dict, so the rung has to
        # be re-applied or a resumed stage would train whatever the last one did.
        sr2_unfreeze.set_trainable(self.actor, self.cfg.rung)
        return blob


def attach_summaries(
    batch: Dict,
    box_summaries: Mapping[str, TorchSummary],
    frozen_tiles: Mapping[Tuple[str, int], TorchSummary],
) -> Dict:
    """Add ``box_summary`` and ``frozen_tile_summary`` to a collated batch.

    Kept out of the dataset because ``S_0`` is a property of a *box's whole
    catalog run*, not of a crop, and materialising it per example would copy the
    same eleven numbers into every one of 512 tiles.
    """
    boxes = list(batch["box"])
    tiles = [int(t) for t in batch["tile_id"].tolist()]
    missing_box = sorted({b for b in boxes if b not in box_summaries})
    missing_tile = sorted({(b, t) for b, t in zip(boxes, tiles)
                           if (b, t) not in frozen_tiles})
    if missing_box or missing_tile:
        raise KeyError(
            f"no pooled summary for boxes {missing_box[:5]} / no frozen tile "
            f"summary for {missing_tile[:5]}"
        )

    def stack(items: Sequence[TorchSummary]) -> TorchSummary:
        return TorchSummary(
            n_sub=torch.cat([s.n_sub for s in items], dim=0),
            n_host=torch.cat([s.n_host for s in items], dim=0),
            occ_numerator=torch.cat([s.occ_numerator for s in items], dim=0),
            volume_mpc3=torch.cat([s.volume_mpc3 for s in items], dim=0),
        )

    batch["box_summary"] = stack([box_summaries[b] for b in boxes])
    batch["frozen_tile_summary"] = stack(
        [frozen_tiles[(b, t)] for b, t in zip(boxes, tiles)])
    return batch
