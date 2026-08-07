"""One-step multiscale **Gaussian residual U-Net** -- not an MLP, not a diffusion.

The policy proposes a correction to the frozen SR2 field in a single forward
pass. It is deliberately the simplest stochastic model with an *exact* log
probability, because the question Phase 2 asks is whether the reward has any
usable support at all, and that question must not be confounded with whether a
sampler converged.

    input   [Psi_base, U(y_lr)]                     (2 * C = 12 channels)
    trunk   two-level 3D U-Net, width 48, 2 residual blocks per level,
            pointwise ChannelGroupNorm3d, SiLU, circular padding, no attention
    heads   1x1x1 convs at the coarse, middle and fine scales -> (mu_s, log sigma_s)
    action  a_s = mu_s + sigma_s * eps_s ,  eps_s ~ N(0, I)
    field   h   = sum_s upsample_s(a_s)             (fixed interpolation)
    edit    delta = CorrectionTransform(h)          (bounded, then projected)

**The sampled coefficient fields ``a_s`` are the stochastic action**, not the
final field ``h`` and not ``delta``. That is what buys an exact tractable log
probability,

    log pi_theta(a | B, Y) = sum_s log N(a_s; mu_s, sigma_s^2) ,

with no ELBO, no score estimate and no sampler to invert -- ``h`` and ``delta``
are deterministic functions of ``a``, so reward-weighted maximum likelihood on
``a`` is exact. A policy defined on ``delta`` instead would have no density at
all: ``P_N`` is a projection, so ``delta`` lives on a measure-zero subspace.

Initialisation
--------------
Every mean head is zero-initialised, so at step 0 ``mu = 0`` exactly and the
policy is a pure exploration distribution centred on "no correction". The
standard deviations start at small, **nonzero**, configurable values -- a zero
sigma is a policy with no support, and reward-weighted likelihood can never
recover from it. Coarse exploration starts smaller than middle/fine: the coarse
scales are the ones the LR field already constrains and the ones the projection
oracle is measuring an allowance for, so an untuned run should not spend its
budget there.

Note the contrast with the diffusion prior in :mod:`cosmo_sr.reward.model`: a
zero-initialised *mean* head here really does give a zero-mean action, because
this is one step and not a sampler that divides by ``alpha(t)``. What it does
not give is a zero *edit* -- sigma is nonzero by construction. Only
``amplitude = 0`` (or ``residual_scale = 0``) recovers SR2 bit for bit; see
``tests/reward/test_zero_init_claim.py``.

Tiled sampling
--------------
A 512^3 forward pass does not fit on one GPU, so full-box sampling tiles with a
valid core, exactly as :mod:`cosmo_sr.reward.sampling` does. For that to be
seed-reproducible and independent of the tiling, the noise must be a **global
coordinate-aligned field**: ``eps_s`` at global index ``i`` is the same number
whichever tile reads it. :func:`global_noise` provides that by seeding a
generator per fixed-size block of the global lattice -- the same principle as
the offset bookkeeping in :mod:`cosmo_sr.tts.srs_noise`, which is what makes
adjacent SR2 tiles agree in their overlap.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field as dc_field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt

from ..models.flow_unet import ConvBlock3d, Resample3d, center_crop_like
from ..operators.multiscale import block_upsample
from .correction import CorrectionConfig, CorrectionTransform

__all__ = [
    "SCALE_NAMES",
    "GaussianPolicyConfig",
    "analytic_receptive_field",
    "MultiScaleGaussianPolicy",
    "ScaleSpec",
    "build_gaussian_policy",
    "diag_gaussian_kl",
    "diag_gaussian_log_prob",
    "global_noise",
    "noise_shapes",
]

#: Coarse -> fine. Names match the SR2 noise-site stages in
#: :mod:`cosmo_sr.tts.srs_noise` so a log line means the same thing in both.
SCALE_NAMES: Tuple[str, str, str] = ("coarse", "middle", "fine")

#: Edge of the cube whose noise one RNG seed fills, on each scale's own lattice.
#: Bigger blocks mean fewer generator seeds per tile and more waste at the edges.
NOISE_BLOCK = 64


@dataclass(frozen=True)
class ScaleSpec:
    """One head's place in the trunk: its name and its stride vs the HR grid."""

    name: str
    stride: int          # HR cells per cell of this scale's lattice

    def size(self, n_hr: int) -> int:
        if n_hr % self.stride:
            raise ValueError(
                f"HR size {n_hr} is not divisible by the {self.name} stride {self.stride}"
            )
        return n_hr // self.stride


def scale_specs(num_levels: int = 2) -> List[ScaleSpec]:
    """``fine`` at stride 1 up to the bottleneck at stride ``2**num_levels``."""
    names = list(SCALE_NAMES)
    if num_levels + 1 > len(names):
        names = [f"level{i}" for i in range(num_levels + 1)]
    out = [ScaleSpec(names[-1], 1)]
    for i in range(1, num_levels + 1):
        out.append(ScaleSpec(names[-1 - i], 2 ** i))
    return list(reversed(out))          # coarse .. fine


# --------------------------------------------------------------------------- #
# Global, coordinate-aligned noise
# --------------------------------------------------------------------------- #
def _block_seed(seed: int, scale: str, block: Tuple[int, int, int]) -> int:
    payload = f"{int(seed)}|{scale}|{block[0]},{block[1]},{block[2]}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def global_noise(
    seed: int,
    scale: str,
    channels: int,
    grid: int,
    start: Sequence[int],
    size: int,
    *,
    block: int = NOISE_BLOCK,
    dtype=torch.float32,
) -> torch.Tensor:
    """``eps`` for one window of a global, periodic, coordinate-aligned lattice.

    The value at global index ``i`` depends on ``(seed, scale, i)`` alone, so two
    tiles whose windows overlap read *identical* numbers in the overlap and a
    tiled sample equals the untiled one. Windows wrap periodically, matching the
    box.

    Drawn on the CPU, deliberately: a realisation must depend on the seed and not
    on which device produced it -- the same guarantee as
    :func:`cosmo_sr.tts.sampling.super_resolve_srs_seeded`.
    """
    grid, size, block = int(grid), int(size), int(block)
    if size > grid:
        raise ValueError(f"window {size} is larger than the {scale} grid {grid}")
    start = [int(s) % grid for s in start]

    # Index range on the global lattice, then the blocks that cover it.
    idx = [np.arange(s, s + size) % grid for s in start]
    b0 = [int(i.min() // block) for i in idx]
    b1 = [int(i.max() // block) for i in idx]
    # A wrapped window touches every block on that axis; cheaper to say so than
    # to reason about two disjoint runs.
    wrapped = [bool((np.diff(i) < 0).any()) for i in idx]
    n_blocks = -(-grid // block)
    rng_lo = [0 if w else b0[d] for d, w in enumerate(wrapped)]
    rng_hi = [n_blocks - 1 if w else b1[d] for d, w in enumerate(wrapped)]

    span = [(rng_hi[d] - rng_lo[d] + 1) * block for d in range(3)]
    buf = torch.empty((channels, *span), dtype=dtype)
    for bx in range(rng_lo[0], rng_hi[0] + 1):
        for by in range(rng_lo[1], rng_hi[1] + 1):
            for bz in range(rng_lo[2], rng_hi[2] + 1):
                g = torch.Generator(device="cpu").manual_seed(
                    _block_seed(seed, scale, (bx, by, bz)) % (2 ** 63 - 1))
                chunk = torch.randn((channels, block, block, block), generator=g,
                                    dtype=torch.float32).to(dtype)
                ox = (bx - rng_lo[0]) * block
                oy = (by - rng_lo[1]) * block
                oz = (bz - rng_lo[2]) * block
                buf[:, ox:ox + block, oy:oy + block, oz:oz + block] = chunk
    # Gather the requested indices out of the block buffer.
    sel = [torch.as_tensor(idx[d] - rng_lo[d] * block, dtype=torch.long) for d in range(3)]
    out = buf[:, sel[0]][:, :, sel[1]][:, :, :, sel[2]]
    return out.contiguous()


def noise_shapes(
    n_hr: int, channels: int, num_levels: int = 2, batch: int = 1
) -> Dict[str, Tuple[int, ...]]:
    """``{scale: (B, C, n_s, n_s, n_s)}`` for an HR window of size ``n_hr``."""
    return {
        s.name: (int(batch), int(channels)) + (s.size(int(n_hr)),) * 3
        for s in scale_specs(num_levels)
    }


# --------------------------------------------------------------------------- #
# Diagonal Gaussian algebra
# --------------------------------------------------------------------------- #
_LOG_2PI = math.log(2.0 * math.pi)


def diag_gaussian_log_prob(
    a: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, *, reduce: str = "sum"
) -> torch.Tensor:
    """``log N(a; mu, sigma^2)``, summed (or averaged) over everything but batch.

    ``reduce='sum'`` is the true log density -- the quantity the reward-weighted
    likelihood is defined on, and the one in which the fine scale legitimately
    counts for more than the coarse one because it has more elements.
    ``reduce='mean'`` is per-element and exists only for logging, where a number
    of order 1e7 is unreadable.
    """
    z = (a - mu) / sigma
    per = -0.5 * (z ** 2 + _LOG_2PI) - torch.log(sigma)
    per = per.flatten(1)
    return per.sum(dim=1) if reduce == "sum" else per.mean(dim=1)


def diag_gaussian_kl(
    mu_p: torch.Tensor, sigma_p: torch.Tensor,
    mu_q: torch.Tensor, sigma_q: torch.Tensor, *, reduce: str = "sum",
) -> torch.Tensor:
    """``KL(p || q)`` for diagonal Gaussians -- closed form, no sampling."""
    vp, vq = sigma_p ** 2, sigma_q ** 2
    per = torch.log(sigma_q / sigma_p) + (vp + (mu_p - mu_q) ** 2) / (2.0 * vq) - 0.5
    per = per.flatten(1)
    return per.sum(dim=1) if reduce == "sum" else per.mean(dim=1)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class GaussianPolicyConfig:
    """Architecture, exploration scales, and the correction it composes with."""

    channels: int = 6
    scale_factor: int = 8
    width: int = 48
    num_levels: int = 2
    blocks_per_level: int = 2
    kernel_size: int = 3
    norm: str = "channel"           # pointwise: required for exact valid-core tiling
    num_groups: int = 8
    activation: str = "silu"
    padding_mode: str = "circular"
    use_attention: bool = False     # "no attention initially"
    use_checkpoint: bool = True
    condition_on_base: bool = True
    condition_on_lr: bool = True
    upsample: str = "nearest"       # fixed interpolation; "nearest" | "trilinear"

    # Exploration. Per scale and per channel group, in units of the calibrated
    # amplitude (the correction transform turns h into a bounded edit), so these
    # are dimensionless and comparable across runs.
    sigma_init: Dict[str, float] = dc_field(default_factory=lambda: {
        "coarse": 0.05, "middle": 0.15, "fine": 0.15})
    sigma_init_disp_scale: float = 1.0
    sigma_init_vel_scale: float = 1.0
    sigma_min: float = 1e-3
    sigma_max: float = 3.0

    def __post_init__(self) -> None:
        if self.num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {self.num_levels}")
        if self.upsample not in ("nearest", "trilinear"):
            raise ValueError(f"upsample must be nearest|trilinear, got {self.upsample!r}")
        if not 0.0 < self.sigma_min <= self.sigma_max:
            raise ValueError(
                f"need 0 < sigma_min <= sigma_max, got {self.sigma_min}, {self.sigma_max}")
        specs = scale_specs(self.num_levels)
        missing = [s.name for s in specs if s.name not in self.sigma_init]
        if missing:
            raise ValueError(f"sigma_init has no entry for scale(s) {missing}")
        for name, v in self.sigma_init.items():
            if float(v) <= 0.0:
                raise ValueError(
                    f"sigma_init[{name}] = {v}: exploration must start NONZERO. A "
                    f"policy with zero variance has no support, and reward-weighted "
                    f"likelihood cannot widen a distribution that never proposes "
                    f"anything."
                )
            if not self.sigma_min <= float(v) <= self.sigma_max:
                raise ValueError(
                    f"sigma_init[{name}] = {v} outside [{self.sigma_min}, {self.sigma_max}]")
        coarse = float(self.sigma_init.get("coarse", 0.0))
        finer = [float(self.sigma_init[s.name]) for s in specs if s.name != "coarse"]
        if finer and coarse > min(finer):
            raise ValueError(
                f"coarse exploration ({coarse}) exceeds the finest scales "
                f"({min(finer)}). The coarse scales are the ones the LR field "
                f"already constrains; spending the budget there first is how a "
                f"run fails the low_k_change constraint before it proposes "
                f"anything useful."
            )

    @property
    def divisor(self) -> int:
        return 2 ** int(self.num_levels)

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping) -> "GaussianPolicyConfig":
        d = {k: v for k, v in dict(d or {}).items() if k != "name"}
        known = set(GaussianPolicyConfig.__dataclass_fields__)
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown gaussian policy option(s): {unknown}")
        return GaussianPolicyConfig(**d)


# --------------------------------------------------------------------------- #
# The trunk
# --------------------------------------------------------------------------- #
class _Trunk(nn.Module):
    """The Map2MapUNet3D design, with the decoder's per-level output exposed.

    Built from the same :class:`~cosmo_sr.models.flow_unet.ConvBlock3d` and
    :class:`~cosmo_sr.models.flow_unet.Resample3d` pieces rather than wrapping
    :class:`~cosmo_sr.models.flow_unet.Map2MapUNet3D`, because the heads need the
    features *at each scale* and that class returns only its last layer. Nothing
    else differs: same widths, same norm, same activation, same padding, same
    checkpointing policy.
    """

    def __init__(self, cfg: GaussianPolicyConfig, in_channels: int):
        super().__init__()
        self.cfg = cfg
        w, nl = int(cfg.width), int(cfg.num_levels)
        self.use_checkpoint = bool(cfg.use_checkpoint)

        def stage(in_ch: int) -> nn.ModuleList:
            return nn.ModuleList([
                ConvBlock3d(in_ch if i == 0 else w, w, kernel_size=cfg.kernel_size,
                            padding="same", norm=cfg.norm, num_groups=cfg.num_groups,
                            activation=cfg.activation, cond_dim=None, residual=True,
                            padding_mode=cfg.padding_mode)
                for i in range(int(cfg.blocks_per_level))
            ])

        def resample(direction: str) -> Resample3d:
            return Resample3d(w, direction, norm=cfg.norm, num_groups=cfg.num_groups,
                              activation=cfg.activation)

        self.enc_stages = nn.ModuleList(
            [stage(in_channels if l == 0 else w) for l in range(nl)])
        self.downs = nn.ModuleList([resample("down") for _ in range(nl)])
        self.bottleneck = stage(w)
        self.ups = nn.ModuleList([resample("up") for _ in range(nl)])
        self.dec_stages = nn.ModuleList([stage(2 * w) for _ in range(nl)])

    def _run(self, blocks: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for blk in blocks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = _ckpt.checkpoint(blk, x, None, use_reentrant=False)
            else:
                x = blk(x, None)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``{scale_name: features}`` at every scale a head is attached to."""
        nl = int(self.cfg.num_levels)
        specs = scale_specs(nl)
        skips = []
        h = x
        for l in range(nl):
            h = self._run(self.enc_stages[l], h)
            skips.append(h)
            h = self.downs[l](h)
        h = self._run(self.bottleneck, h)
        feats = {specs[0].name: h}                    # bottleneck = coarsest
        for l in reversed(range(nl)):
            h = self.ups[l](h)
            h = torch.cat([center_crop_like(skips[l], h), h], dim=1)
            h = self._run(self.dec_stages[l], h)
            feats[specs[nl - l].name] = h
        return feats


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #
class MultiScaleGaussianPolicy(nn.Module):
    """``(Psi_base, y_lr) -> {mu_s, sigma_s}``, and the edit those actions imply."""

    def __init__(self, cfg: GaussianPolicyConfig,
                 correction: Optional[CorrectionConfig] = None):
        super().__init__()
        self.cfg = cfg
        self.specs = scale_specs(cfg.num_levels)
        c = int(cfg.channels)

        in_ch = c * (int(cfg.condition_on_base) + int(cfg.condition_on_lr))
        if in_ch == 0:
            raise ValueError("the policy must condition on Psi_base, y_lr or both")
        self.trunk = _Trunk(cfg, in_ch)

        # 1x1x1 heads: pointwise, so they cannot widen the receptive field and
        # valid-core tiling stays exact.
        self.mu_heads = nn.ModuleDict()
        self.logsig_heads = nn.ModuleDict()
        for s in self.specs:
            self.mu_heads[s.name] = nn.Conv3d(cfg.width, c, kernel_size=1)
            self.logsig_heads[s.name] = nn.Conv3d(cfg.width, c, kernel_size=1)

        corr = correction or CorrectionConfig(scale_factor=cfg.scale_factor,
                                              channels=c)
        self.correction = CorrectionTransform(corr)
        self.register_buffer("log_sigma_min",
                             torch.tensor(math.log(cfg.sigma_min)), persistent=False)
        self.register_buffer("log_sigma_max",
                             torch.tensor(math.log(cfg.sigma_max)), persistent=False)
        self.reset_parameters()

    # -- initialisation ----------------------------------------------------- #
    def reset_parameters(self) -> None:
        """Zero means; sigma exactly at its configured per-scale, per-group value.

        The log-sigma head's weights are zeroed too, so at initialisation sigma is
        spatially constant and *is* the configured number -- not "approximately,
        depending on the features". The bias carries it, which is also why a
        curriculum can rescale exploration by editing the bias alone.
        """
        corr = self.correction.cfg
        with torch.no_grad():
            for s in self.specs:
                mu = self.mu_heads[s.name]
                nn.init.zeros_(mu.weight)
                nn.init.zeros_(mu.bias)

                head = self.logsig_heads[s.name]
                nn.init.zeros_(head.weight)
                base = float(self.cfg.sigma_init[s.name])
                bias = torch.full((int(self.cfg.channels),), math.log(base))
                for ch in corr.disp_channels:
                    bias[int(ch)] = math.log(
                        max(base * float(self.cfg.sigma_init_disp_scale),
                            float(self.cfg.sigma_min)))
                for ch in corr.vel_channels:
                    bias[int(ch)] = math.log(
                        max(base * float(self.cfg.sigma_init_vel_scale),
                            float(self.cfg.sigma_min)))
                head.bias.copy_(bias.clamp(math.log(self.cfg.sigma_min),
                                           math.log(self.cfg.sigma_max)))

    # -- distribution ------------------------------------------------------- #
    def distribution(
        self, psi_base: Optional[torch.Tensor], y_lr: Optional[torch.Tensor]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """``{scale: (mu, sigma)}``; sigma is already clamped to its bounds."""
        feats = self.trunk(self._inputs(psi_base, y_lr))
        out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for s in self.specs:
            f = feats[s.name]
            mu = self.mu_heads[s.name](f)
            ls = self.logsig_heads[s.name](f).clamp(self.log_sigma_min, self.log_sigma_max)
            out[s.name] = (mu, torch.exp(ls))
        return out

    def _inputs(self, psi_base, y_lr) -> torch.Tensor:
        feats = []
        if self.cfg.condition_on_base:
            if psi_base is None:
                raise ValueError("condition_on_base=True but psi_base was not given")
            if psi_base.dim() != 5:
                raise ValueError(f"psi_base must be 5D, got {tuple(psi_base.shape)}")
            n = int(psi_base.shape[-1])
            if n % self.cfg.divisor:
                raise ValueError(
                    f"crop size {n} must be a multiple of {self.cfg.divisor} "
                    f"(2**num_levels) for padding='same'")
            feats.append(psi_base)
        if self.cfg.condition_on_lr:
            if y_lr is None:
                raise ValueError("condition_on_lr=True but y_lr was not given")
            up = block_upsample(y_lr, int(self.cfg.scale_factor))
            if feats and up.shape[-1] != feats[0].shape[-1]:
                raise ValueError(
                    f"U(y_lr) size {up.shape[-1]} != crop size {feats[0].shape[-1]}; "
                    f"the LR window must cover the same region at "
                    f"1/{self.cfg.scale_factor} scale")
            feats.append(up)
        return torch.cat(feats, dim=1)

    # -- actions ------------------------------------------------------------ #
    def sample_actions(
        self,
        dist: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
        noise: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """``a_s = mu_s + sigma_s * eps_s`` with caller-supplied global noise."""
        out = {}
        for name, (mu, sigma) in dist.items():
            if name not in noise:
                raise KeyError(f"no noise for scale {name!r}; got {sorted(noise)}")
            eps = noise[name].to(device=mu.device, dtype=mu.dtype)
            if tuple(eps.shape) != tuple(mu.shape):
                raise ValueError(
                    f"noise for {name} has shape {tuple(eps.shape)}, expected "
                    f"{tuple(mu.shape)}")
            out[name] = mu + sigma * eps
        return out

    def combine(self, actions: Mapping[str, torch.Tensor], n_hr: int) -> torch.Tensor:
        """``h = sum_s upsample_s(a_s)`` by fixed, parameter-free interpolation."""
        total = None
        for s in self.specs:
            a = actions[s.name]
            if s.stride == 1:
                up = a
            elif self.cfg.upsample == "nearest":
                up = block_upsample(a, s.stride)
            else:
                up = F.interpolate(a, scale_factor=s.stride, mode="trilinear",
                                   align_corners=False)
            if up.shape[-1] != n_hr:
                raise ValueError(
                    f"scale {s.name} upsampled to {up.shape[-1]}, expected {n_hr}")
            total = up if total is None else total + up
        return total

    def log_prob(
        self,
        dist: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
        actions: Mapping[str, torch.Tensor],
        *,
        reduce: str = "sum",
    ) -> torch.Tensor:
        """``log pi(a | B, Y) = sum_s log N(a_s; mu_s, sigma_s^2)``, per batch element.

        Exact. Nothing is marginalised and nothing is bounded from below.
        """
        total = None
        for s in self.specs:
            mu, sigma = dist[s.name]
            lp = diag_gaussian_log_prob(actions[s.name].to(mu.dtype), mu, sigma,
                                        reduce=reduce)
            total = lp if total is None else total + lp
        return total

    def kl_to(
        self,
        dist: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
        ref: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
        *,
        reduce: str = "sum",
    ) -> torch.Tensor:
        """``KL(pi_theta || pi_ref)``, summed over scales. Closed form."""
        total = None
        for s in self.specs:
            mu, sig = dist[s.name]
            mu_r, sig_r = ref[s.name]
            kl = diag_gaussian_kl(mu, sig, mu_r.detach(), sig_r.detach(), reduce=reduce)
            total = kl if total is None else total + kl
        return total

    def n_action_elements(self, n_hr: int) -> int:
        """Total dimensionality of ``a``; the constant that turns a summed
        log-probability into a per-element one."""
        c = int(self.cfg.channels)
        return sum(c * s.size(int(n_hr)) ** 3 for s in self.specs)

    # -- one full step ------------------------------------------------------ #
    def forward(
        self,
        psi_base: Optional[torch.Tensor],
        y_lr: Optional[torch.Tensor],
        *,
        noise: Optional[Mapping[str, torch.Tensor]] = None,
        seed: Optional[int] = None,
        origin: Sequence[int] = (0, 0, 0),
        grid_hr: Optional[int] = None,
        actions: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, object]:
        """Sample (or re-evaluate) one correction.

        ``actions`` re-evaluates a stored action under the current parameters --
        the replay path, where the action is fixed and only ``log pi_theta``
        changes. ``noise`` or ``seed`` sample a new one; ``seed`` builds global
        coordinate-aligned noise for the window at ``origin`` of a ``grid_hr``
        box, so a tiled sample matches an untiled one.
        """
        ref = psi_base if psi_base is not None else y_lr
        n_hr = int(ref.shape[-1]) * (1 if psi_base is not None
                                     else int(self.cfg.scale_factor))
        dist = self.distribution(psi_base, y_lr)

        if actions is None:
            if noise is None:
                if seed is None:
                    raise ValueError("pass actions, noise, or a seed")
                noise = self.window_noise(seed, origin=origin, n_hr=n_hr,
                                          grid_hr=grid_hr or n_hr,
                                          batch=int(ref.shape[0]))
            actions = self.sample_actions(dist, noise)

        h = self.combine(actions, n_hr)
        delta, stats = self.correction(h)
        logp = self.log_prob(dist, actions)
        stats = dict(stats)
        stats["log_prob_sum"] = float(logp.detach().mean())
        stats["log_prob_per_element"] = float(
            self.log_prob(dist, actions, reduce="mean").detach().mean())
        for s in self.specs:
            mu, sigma = dist[s.name]
            stats[f"sigma_{s.name}_mean"] = float(sigma.detach().mean())
            stats[f"mu_{s.name}_absmax"] = float(mu.detach().abs().max())
            stats[f"a_{s.name}_rms"] = float(
                actions[s.name].detach().float().pow(2).mean().sqrt())
            # A sigma pinned at either bound is a sigma the gradient cannot move.
            frac = ((sigma.detach() <= self.cfg.sigma_min * (1 + 1e-6)) |
                    (sigma.detach() >= self.cfg.sigma_max * (1 - 1e-6)))
            stats[f"sigma_{s.name}_at_bound_fraction"] = float(frac.float().mean())
        return {"dist": dist, "actions": actions, "h": h, "delta": delta,
                "log_prob": logp, "stats": stats}

    def window_noise(
        self, seed: int, *, origin: Sequence[int], n_hr: int, grid_hr: int,
        batch: int = 1, device=None,
    ) -> Dict[str, torch.Tensor]:
        """Global coordinate-aligned ``eps`` for the window at ``origin``.

        ``origin`` is the HR index of the window's first cell. Each scale reads
        its own lattice at ``origin / stride``, which is why the origin (and the
        tile margin) must be divisible by the coarsest stride -- otherwise two
        tiles would sample the same physical region at different lattice offsets
        and their overlap would disagree.
        """
        out: Dict[str, torch.Tensor] = {}
        c = int(self.cfg.channels)
        for s in self.specs:
            bad = [o for o in origin if int(o) % s.stride]
            if bad:
                raise ValueError(
                    f"tile origin {tuple(origin)} is not aligned to the {s.name} "
                    f"stride {s.stride}; a tiled sample would not match an "
                    f"untiled one")
            eps = global_noise(
                seed, s.name, c, s.size(int(grid_hr)),
                [int(o) // s.stride for o in origin], s.size(int(n_hr)),
            )
            out[s.name] = eps.unsqueeze(0).expand(int(batch), *eps.shape).contiguous()
            if device is not None:
                out[s.name] = out[s.name].to(device)
        return out

    # -- exploration curriculum -------------------------------------------- #
    @torch.no_grad()
    def rescale_exploration(self, factor: float, *, floor: Optional[float] = None) -> None:
        """Multiply every sigma by ``factor`` (the amplitude curriculum's knob).

        ``floor`` is the variance floor: exploration is never allowed below it,
        so the first elite cannot collapse the policy onto itself.
        """
        if factor <= 0:
            raise ValueError(f"factor must be > 0, got {factor}")
        lo = math.log(max(float(floor), self.cfg.sigma_min) if floor else self.cfg.sigma_min)
        for s in self.specs:
            b = self.logsig_heads[s.name].bias
            b.copy_((b + math.log(float(factor))).clamp(lo, math.log(self.cfg.sigma_max)))

    def parameter_hash(self) -> str:
        """Short digest of every parameter; the provenance a replay row stores."""
        h = hashlib.blake2b(digest_size=16)
        for name, p in sorted(self.named_parameters()):
            h.update(name.encode())
            h.update(p.detach().to(torch.float32).cpu().numpy().tobytes())
        return h.hexdigest()

    @property
    def divisor(self) -> int:
        return self.cfg.divisor


def build_gaussian_policy(
    model_cfg: Mapping, correction_cfg: Optional[Mapping] = None
) -> MultiScaleGaussianPolicy:
    """Construct from config dicts (the YAML path)."""
    cfg = GaussianPolicyConfig.from_dict(model_cfg)
    corr = None
    if correction_cfg is not None:
        corr = CorrectionConfig.from_dict({
            "scale_factor": cfg.scale_factor, "channels": cfg.channels,
            **dict(correction_cfg)})
    return MultiScaleGaussianPolicy(cfg, corr)


# --------------------------------------------------------------------------- #
# Full-box sampling by exact valid-core tiling
# --------------------------------------------------------------------------- #
def analytic_receptive_field(cfg: GaussianPolicyConfig) -> Dict[str, int]:
    """Closed-form receptive-field half-width -- an exact UPPER bound.

    The trunk is a stack of local operators, so the standard recurrence gives
    the true mathematical support: for a conv of kernel ``k`` and stride ``s``,
    ``r += (k-1)*j`` then ``j *= s``, with ``ConvTranspose(k=2, s=2)`` halving
    ``j`` and leaving ``r`` alone. The ``y_lr`` path adds one term the trunk does
    not have -- the LR field is block-upsampled by ``scale_factor`` before the
    trunk sees it, so one LR cell drives a whole ``scale_factor^3`` block and the
    influence reaches ``half + scale_factor`` from that block's first cell.

    Analytic vs measured
    --------------------
    This is an **upper** bound and
    :func:`cosmo_sr.reward.sampling.receptive_field_halfwidth` is a **lower** one.
    That probe uses all-positive weights of order ``1/fan_in``, so the outermost
    shells of the support decay below the float32 noise floor and truncate --
    increasingly with depth. Against the measured table in
    ``configs/reward/residual_prior.yaml`` (which already includes the LR path):

    =============  ==================  ========
    levels/blocks  analytic (this fn)  measured
    =============  ==================  ========
    1 / 1          16                  16
    1 / 2          24                  24
    2 / 1          29                  26
    2 / 2          49                  41
    3 / 2          107                 65
    =============  ==================  ========

    Exact agreement where truncation is negligible, divergence with depth. So
    **prefer this value when setting a tile margin**: the truncated tail is real
    support, merely small. (It also means the supervised line's ``tile_margin:
    48``, chosen from the measured 41, is a little under the rigorous 56 -- by
    tails at the float32 noise floor.)

    What the formula cannot see is a layer that is not spatially local.
    ``nn.GroupNorm`` reduces over the whole volume, so its true reach is the
    entire window while this function would still return 49. That is what the
    empirical probe is for: if the probe ever exceeds this bound, the locality
    assumption is broken and *no* finite margin makes tiling exact.
    """
    k = int(cfg.kernel_size)
    per_stage = 2 * int(cfg.blocks_per_level)      # ConvBlock3d = two k-convs
    r, j = 1, 1

    def conv(kk: int, s: int, n: int = 1) -> None:
        nonlocal r, j
        for _ in range(n):
            r += (kk - 1) * j
            j *= s

    conv(k, 1, per_stage)                          # enc0
    for _ in range(int(cfg.num_levels) - 1):
        conv(2, 2)                                 # down
        conv(k, 1, per_stage)                      # enc_l
    conv(2, 2)                                     # last down
    conv(k, 1, per_stage)                          # bottleneck
    for _ in range(int(cfg.num_levels)):
        j //= 2                                    # ConvTranspose(2, 2)
        conv(k, 1, per_stage)                      # dec_l

    half = (r - 1) // 2
    sf = int(cfg.scale_factor)
    return {
        "trunk_full": int(r),
        "trunk_halfwidth": int(half),
        # One LR cell drives an sf^3 block; the far edge of that block is then
        # dilated by the trunk's half-width.
        "policy_halfwidth": int(half + sf) if cfg.condition_on_lr else int(half),
        "scale_factor": sf,
    }


@torch.no_grad()
def policy_receptive_field(
    policy: MultiScaleGaussianPolicy, *, size: int = 64, device=None
) -> int:
    """Measured half-width of the policy's conditioning path, in HR cells.

    Same structural-probe argument as
    :func:`cosmo_sr.reward.sampling.receptive_field_halfwidth`: positive weights,
    zero biases, no spatially-reducing norm, so a nonzero output really is
    support and a zero-initialised head cannot report a half-width of zero and
    wave a too-small margin through.

    Written separately from the diffusion version because the call signature
    differs (no ``u_t``, no ``t``) and because only the *conditioning* inputs can
    couple neighbouring tiles here -- the noise is global by construction.
    """
    from .sampling import _structural_probe

    device = device or torch.device("cpu")
    probe = _structural_probe(policy).to(device).eval()
    c, sf = int(policy.cfg.channels), int(policy.cfg.scale_factor)
    n = int(size)
    align = sf * policy.divisor
    j = max(n // 2 // align, 1) * align
    base = torch.zeros(1, c, n, n, n, device=device)
    lr = torch.zeros(1, c, n // sf, n // sf, n // sf, device=device)

    def run(b, y):
        return torch.cat([probe.mu_heads[s.name](probe.trunk(probe._inputs(b, y))[s.name])
                          for s in probe.specs[-1:]], dim=1)

    y0 = run(base, lr)
    reach = 0
    for name in ("psi_base", "y_lr"):
        b, y = base.clone(), lr.clone()
        if name == "psi_base":
            b[:, :, j, j, j] = 1.0
        else:
            b, y = base, lr.clone()
            y[:, :, j // sf, j // sf, j // sf] = 1.0
        diff = (run(b, y) - y0).abs().amax(dim=(0, 1))
        if float(diff.max()) <= 0.0:
            continue
        idx = torch.nonzero(diff > 0)
        if idx.numel():
            reach = max(reach, int((idx - j).abs().max().item()))
    return reach


@torch.no_grad()
def sample_policy_box(
    policy: MultiScaleGaussianPolicy,
    psi_base: np.ndarray | torch.Tensor,
    y_lr: np.ndarray | torch.Tensor,
    *,
    seed: int,
    spec=None,
    device=None,
    tile_batch: int = 1,
    verify_margin: bool = True,
    return_actions: Sequence[str] = (),
) -> Tuple[np.ndarray, Dict[str, object]]:
    """One full-box correction ``delta`` in catnorm units, by valid-core tiling.

    Exact rather than blended: each tile is ``core + 2 * margin`` wide, only the
    central ``core`` is written, and the noise is read from the *global* lattice
    so overlapping tiles agree by construction. Provided the margin covers the
    conditioning receptive field, the written box does not depend on the tiling.

    One forward pass per tile -- there is no sampler loop -- which is the main
    practical reason to try the Gaussian policy before diffusion.

    ``return_actions`` names scales whose coefficient fields are assembled and
    returned, for a replay buffer that wants to store them rather than
    regenerate them from the seed.
    """
    from .sampling import TileSpec, tile_margin_for

    device = device or next(policy.parameters()).device
    base = torch.as_tensor(np.asarray(psi_base), dtype=torch.float32)
    lr = torch.as_tensor(np.asarray(y_lr), dtype=torch.float32)
    if base.dim() == 5:
        base = base.squeeze(0)
    if lr.dim() == 5:
        lr = lr.squeeze(0)
    ng = int(base.shape[-1])
    sf = int(policy.cfg.scale_factor)
    spec = spec or TileSpec(ng, core=min(128, ng), margin=min(48, ng // 4),
                            scale_factor=sf)

    if verify_margin:
        rf = policy_receptive_field(policy, size=max(32, 2 * spec.margin + 8),
                                    device=device)
        need = tile_margin_for(rf, sf)
        if spec.margin < need:
            raise ValueError(
                f"tile margin {spec.margin} < required {need} (measured "
                f"conditioning receptive field {rf}); valid-core tiling would "
                f"leak tile-boundary padding into the written core")

    coarsest = max(s.stride for s in policy.specs)
    if spec.margin % coarsest or spec.core % coarsest:
        raise ValueError(
            f"core={spec.core} and margin={spec.margin} must both be multiples "
            f"of the coarsest scale stride {coarsest}, or the per-tile noise "
            f"windows would not be aligned to the global lattice")

    from ..data.crops import periodic_crop

    out = torch.zeros_like(base)
    actions_out = {
        s.name: torch.zeros((int(policy.cfg.channels),) + (ng // s.stride,) * 3,
                            dtype=torch.float32)
        for s in policy.specs if s.name in set(return_actions)
    }
    agg: Dict[str, List[float]] = {}
    origins = spec.origins()
    for b0 in range(0, len(origins), max(1, int(tile_batch))):
        group = origins[b0:b0 + max(1, int(tile_batch))]
        for o in group:
            start_hr = tuple(int(v) - spec.margin for v in o)
            start_lr = tuple(v // sf for v in start_hr)
            bc = periodic_crop(base, start_hr, spec.size, pad=0).unsqueeze(0).to(device)
            yc = periodic_crop(lr, start_lr, spec.size // sf, pad=0).unsqueeze(0).to(device)
            noise = policy.window_noise(seed, origin=start_hr, n_hr=spec.size,
                                        grid_hr=ng, batch=1, device=device)
            res = policy(bc, yc, noise=noise)
            m, c = spec.margin, spec.core
            core = res["delta"][0, ..., m:m + c, m:m + c, m:m + c].to("cpu")
            ox, oy, oz = (int(v) % ng for v in o)
            out[..., ox:ox + c, oy:oy + c, oz:oz + c] = core
            for name, buf in actions_out.items():
                s = next(sp for sp in policy.specs if sp.name == name)
                mm, cc = m // s.stride, c // s.stride
                a = res["actions"][name][0, ..., mm:mm + cc, mm:mm + cc, mm:mm + cc]
                px, py, pz = (int(v) // s.stride for v in (ox, oy, oz))
                buf[..., px:px + cc, py:py + cc, pz:pz + cc] = a.to("cpu")
            for k, v in res["stats"].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    agg.setdefault(k, []).append(float(v))
            del bc, yc, noise, res

    stats: Dict[str, object] = {
        k: float(np.mean(v)) for k, v in agg.items() if np.isfinite(v).any()
    }
    stats["tanh_saturated_fraction_max"] = float(
        np.max(agg.get("tanh_saturated_fraction", [float("nan")])))
    stats["n_tiles"] = len(origins)
    stats["tile"] = {"core": spec.core, "margin": spec.margin}
    if actions_out:
        stats["actions"] = {k: v.numpy() for k, v in actions_out.items()}
    return out.numpy().astype(np.float32, copy=False), stats
