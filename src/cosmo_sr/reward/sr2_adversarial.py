"""Optional adversarial continuation of the *original* SR2 objective.

Status on this cluster, checked at import time by
:func:`find_discriminator_checkpoint`: **there is no discriminator checkpoint.**
``external/SRS-map2map/SRmodel/`` contains ``G_z0.pt`` and ``G_z2.pt`` and
nothing else, and no ``D_*`` file exists anywhere under the scratch tree. The
consequences are stated here rather than left implicit:

* the adversarial weight is **zero in the mainline** -- see
  ``configs/reward/sr2_direct_finetune.yaml``;
* a freshly initialised critic is **not** "ordinary SR2 continuation". A random
  critic defines a random objective, and generator steps taken against it in the
  first few hundred iterations are noise applied to a checkpoint this whole line
  is trying not to damage. Calling that "continuing the original training" would
  be a claim about provenance that is simply false;
* if a critic is ever trained here it is a **separate ablation**, entered only
  after a frozen-generator warm-up (the critic is trained alone until its loss
  stabilises, generator frozen), and reported as such.

Why not reuse ``dmsr.HRCritic``
-------------------------------
:class:`cosmo_sr.dmsr.critic.HRCritic` was designed for the high-pass *residual*
field of the DMSR line: different input channels, different normalisation,
different assumptions about what the interesting signal looks like. Handing it
SR2's inputs would produce a critic that runs and means nothing.
:class:`SR2Critic` below instead reproduces SR2's own discriminator input
contract exactly:

===================================================  ========
upsampled LR (trilinear to the SR grid)                     6
candidate / HR displacement and velocity                    6
inverse-pixel-shuffled fine CIC density (grid_mult 2)       8
                                                       ------
total                                                      20
===================================================  ========

with WGAN-GP, ``lambda = 10``, and the gradient penalty applied every 16 critic
batches (the upstream schedule -- the penalty is the expensive term and SR2 paid
for it once per 16 rather than every step).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..eval.density import cic_density, cic_density_valid_center

__all__ = [
    "AdversarialConfig",
    "SR2Critic",
    "critic_input",
    "find_discriminator_checkpoint",
    "gradient_penalty",
    "wgan_critic_loss",
    "wgan_generator_loss",
]

#: Places a real SR2 discriminator could plausibly live, most specific first.
_D_SEARCH_GLOBS: Tuple[str, ...] = (
    "external/SRS-map2map/SRmodel/D_*.pt",
    "external/SRS-map2map/SRmodel/*D*.pt",
    "external/map2map/**/D_*.pt",
)


def find_discriminator_checkpoint(
    project_root: Optional[str | os.PathLike] = None,
    extra: Sequence[str | os.PathLike] = (),
) -> Optional[Path]:
    """The original SR2 critic checkpoint, or ``None``.

    Returns ``None`` rather than raising, because "there is no discriminator" is
    the expected answer here and the correct response to it is to leave the
    adversarial term switched off -- not to fail the job.
    """
    root = Path(project_root or Path(__file__).resolve().parents[3])
    hits: List[Path] = []
    for pattern in _D_SEARCH_GLOBS:
        hits.extend(sorted(root.glob(pattern)))
    for e in extra:
        p = Path(e)
        if p.is_file():
            hits.append(p)
    hits = [h for h in hits if h.is_file() and not h.name.startswith("G_")]
    return hits[0] if hits else None


@dataclass(frozen=True)
class AdversarialConfig:
    """WGAN-GP settings, matching the SR2 recipe.

    ``weight = 0.0`` is the mainline. ``allow_fresh_critic`` must be set
    explicitly to train a new one, and even then the warm-up is not optional:
    ``critic_warmup_steps`` generator-frozen critic steps run first.
    """

    weight: float = 0.0
    gp_lambda: float = 10.0
    gp_every: int = 16
    critic_steps_per_generator_step: int = 1
    critic_lr: float = 1e-4
    critic_betas: Tuple[float, float] = (0.0, 0.9)
    critic_width: int = 64
    critic_depth: int = 4
    density_grid_mult: int = 2
    allow_fresh_critic: bool = False
    critic_warmup_steps: int = 1000
    checkpoint: str = ""

    def enabled(self) -> bool:
        return float(self.weight) != 0.0

    def to_dict(self) -> Dict:
        return {
            "weight": float(self.weight),
            "gp_lambda": float(self.gp_lambda),
            "gp_every": int(self.gp_every),
            "critic_steps_per_generator_step": int(self.critic_steps_per_generator_step),
            "critic_lr": float(self.critic_lr),
            "critic_betas": list(self.critic_betas),
            "critic_width": int(self.critic_width),
            "critic_depth": int(self.critic_depth),
            "density_grid_mult": int(self.density_grid_mult),
            "allow_fresh_critic": bool(self.allow_fresh_critic),
            "critic_warmup_steps": int(self.critic_warmup_steps),
            "checkpoint": str(self.checkpoint),
        }


def _inverse_pixel_shuffle(x: torch.Tensor, r: int) -> torch.Tensor:
    """``(B, C, rN, rN, rN) -> (B, C*r^3, N, N, N)``.

    The 3-D analogue of ``nn.PixelUnshuffle``, which torch only provides in 2-D.
    Folding the fine density mesh into channels rather than downsampling it is
    what lets the critic see sub-cell density -- the caustics and halo cores
    that CIC at the particle resolution smooths away -- at the coarse spatial
    resolution the rest of its input lives on.
    """
    r = int(r)
    b, c, d, h, w = x.shape
    if d % r or h % r or w % r:
        raise ValueError(f"{tuple(x.shape)} not divisible by r={r}")
    x = x.reshape(b, c, d // r, r, h // r, r, w // r, r)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    return x.reshape(b, c * r ** 3, d // r, h // r, w // r)


def _center_crop(x: torch.Tensor, region: int) -> torch.Tensor:
    """The central ``region^3`` cube of a ``(B, C, N, N, N)`` tensor."""
    n = x.shape[-1]
    r = int(region)
    if r >= n:
        return x
    lo = (n - r) // 2
    return x[..., lo:lo + r, lo:lo + r, lo:lo + r]


def critic_input(
    lr: torch.Tensor,
    field: torch.Tensor,
    *,
    cellsize_kpc_h: float,
    dis_norm_kpc_h: float = 6000.0,
    grid_mult: int = 2,
    valid_center: int = 0,
) -> torch.Tensor:
    """The 20-channel discriminator input SR2 was trained with.

    ``lr`` is the six-channel LR conditioning at its own resolution and is
    trilinearly upsampled to ``field``'s grid, so the critic is *conditional*:
    it judges "is this a plausible SR of this LR", not "is this a plausible
    universe".

    ``field`` is the six-channel displacement/velocity candidate (or HR).
    Density is CIC-deposited from its displacement channels on a ``grid_mult``x
    finer mesh and inverse-pixel-shuffled back, giving ``grid_mult^3 = 8``
    channels.

    ``valid_center``
        Density deposition mode. ``0`` (the default) keeps the historical
        wrapped ``% ng`` deposit -- which, on a single tile rather than a whole
        periodic box, is a *scrambling*: only ~10% of a tile's particles stay
        inside it (median displacement ~36 HR cells against a 64-cell tile), so
        the wrapped density correlates just ``r = 0.08`` with the tile's true
        density (see :func:`cosmo_sr.eval.density.cic_density_valid_center` and
        ``docs/density_collapse_investigation.md``). A positive value switches to
        the buffer-free valid-centre deposit: it follows the tile's bulk
        displacement, discards escaping particles rather than wrapping them, and
        scores the central ``valid_center^3`` Eulerian sub-cube, which the tile's
        own particles fill exactly (``r = 1.0`` at region = tile/2). The LR and
        field channels are centre-cropped to the same ``valid_center^3`` so all
        20 channels describe the one sub-region; the critic pools globally, so
        the smaller grid is not an obstacle. This is the same valid-centre CIC
        arm A already deposits through :func:`density_from_disp`.
    """
    if lr.dim() != 5 or field.dim() != 5:
        raise ValueError("critic_input wants (B, C, N, N, N) tensors")
    if lr.shape[1] != 6 or field.shape[1] != 6:
        raise ValueError(
            f"expected 6 LR and 6 field channels, got {lr.shape[1]} and {field.shape[1]}"
        )
    up = F.interpolate(lr, size=tuple(field.shape[-3:]), mode="trilinear",
                       align_corners=False)
    r = int(valid_center)
    if r > 0:
        rho = cic_density_valid_center(
            field[:, 0:3], float(cellsize_kpc_h), float(dis_norm_kpc_h),
            region=r, grid_mult=int(grid_mult))
        up = _center_crop(up, r)
        field = _center_crop(field, r)
    else:
        rho = cic_density(field[:, 0:3], float(cellsize_kpc_h),
                          float(dis_norm_kpc_h), grid_mult=int(grid_mult))
    rho = _inverse_pixel_shuffle(rho, int(grid_mult))
    return torch.cat([up, field, rho], dim=1)


class SR2Critic(nn.Module):
    """Strided 3-D convolutional critic over the 20-channel SR2 input.

    IMPORTANT -- what this is and is NOT. This reproduces the *input contract* of
    SR2's discriminator exactly (the 20 channels above), but the *architecture*
    here -- ``Conv3d(3, stride 2, padding 1)`` blocks, ``GroupNorm``, LeakyReLU,
    global mean pool, linear head -- is a reasonable SR2-STYLE reconstruction, not
    a verified copy of the original SR2 discriminator's layer stack. No original
    discriminator checkpoint or architecture spec exists in this checkout (see
    :func:`find_discriminator_checkpoint`), so there is nothing to match it
    against. A result obtained with this network -- in particular a negative arm-F
    proxy result -- is evidence about *an SR2-style discriminator architecture on
    these features*, NOT evidence that "the original SR2 discriminator failed".
    If the authentic discriminator is ever recovered, load it through
    :func:`build_critic` (which stamps the provenance) and re-run before drawing
    any conclusion about the original.

    No normalisation layers with batch statistics: WGAN-GP's gradient penalty is
    a per-sample constraint, and BatchNorm makes the critic's output for one
    sample depend on the others in the batch, which invalidates it. GroupNorm is
    fine and is what is used.
    """

    IN_CHANNELS = 20

    def __init__(self, width: int = 64, depth: int = 4, in_chan: int = IN_CHANNELS):
        super().__init__()
        layers: List[nn.Module] = []
        c = int(in_chan)
        for i in range(int(depth)):
            nxt = int(width) * (2 ** min(i, 3))
            layers += [nn.Conv3d(c, nxt, 3, stride=2, padding=1)]
            if i > 0:
                layers.append(nn.GroupNorm(min(8, nxt), nxt))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            c = nxt
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(c, 1)
        self.in_chan = int(in_chan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_chan:
            raise ValueError(
                f"SR2Critic expects {self.in_chan} channels (6 upsampled LR + 6 "
                f"field + {self.in_chan - 12} inverse-pixel-shuffled density), "
                f"got {x.shape[1]}"
            )
        h = self.body(x).mean(dim=(2, 3, 4))
        return self.head(h).squeeze(-1)


def wgan_critic_loss(real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
    """``E[D(fake)] - E[D(real)]``: the critic wants real high, fake low."""
    return fake_score.mean() - real_score.mean()


def wgan_generator_loss(fake_score: torch.Tensor) -> torch.Tensor:
    return -fake_score.mean()


def gradient_penalty(
    critic: nn.Module, real: torch.Tensor, fake: torch.Tensor,
    *, gp_lambda: float = 10.0,
) -> torch.Tensor:
    """``lambda * E[(||grad D(x_interp)|| - 1)^2]`` on the 20-channel input.

    Interpolation happens in *critic-input space*, which is the space the
    Lipschitz constraint is about. Interpolating displacements and then
    recomputing CIC would penalise the gradient of a different function.
    """
    if real.shape != fake.shape:
        raise ValueError(f"real {tuple(real.shape)} != fake {tuple(fake.shape)}")
    b = real.shape[0]
    eps = torch.rand(b, *([1] * (real.dim() - 1)), device=real.device, dtype=real.dtype)
    x = (eps * real + (1.0 - eps) * fake).detach().requires_grad_(True)
    score = critic(x)
    grad = torch.autograd.grad(
        outputs=score.sum(), inputs=x, create_graph=True, retain_graph=True,
    )[0]
    norm = grad.reshape(b, -1).norm(2, dim=1)
    return float(gp_lambda) * ((norm - 1.0) ** 2).mean()


def build_critic(
    cfg: AdversarialConfig,
    *,
    project_root: Optional[str | os.PathLike] = None,
    device=None,
) -> Tuple[Optional[SR2Critic], Dict]:
    """``(critic, provenance)``. ``None`` when no critic may legitimately be used.

    The provenance dict is the point: it records whether the weights came from
    the original SR2 run or from a fresh initialisation, and every consumer is
    expected to put it in its manifest. A run whose report does not say which of
    those two it was cannot be read.
    """
    path = Path(cfg.checkpoint) if cfg.checkpoint else find_discriminator_checkpoint(
        project_root)
    if path is not None and Path(path).is_file():
        critic = SR2Critic(width=cfg.critic_width, depth=cfg.critic_depth)
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        state = blob.get("model", blob) if isinstance(blob, dict) else blob
        missing, unexpected = critic.load_state_dict(state, strict=False)
        prov = {
            "source": "original_sr2_discriminator",
            "path": str(path),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
        if missing or unexpected:
            # Loud, and not fatal: a partial match means this checkpoint is a
            # different architecture, so calling the result "the original
            # critic" would be wrong even though it loaded.
            prov["source"] = "partial_match_NOT_original"
            print(f"WARNING: discriminator at {path} does not match SR2Critic "
                  f"({len(missing)} missing, {len(unexpected)} unexpected keys); "
                  "this is NOT the original critic", flush=True)
        return critic.to(device or "cpu"), prov

    if not cfg.allow_fresh_critic:
        return None, {
            "source": "none",
            "reason": (
                "no SR2 discriminator checkpoint found; a freshly initialised "
                "critic is not 'ordinary continuation' of the original training, "
                "so the adversarial term stays off. Set "
                "adversarial.allow_fresh_critic: true to run it as an explicit "
                "ablation instead."
            ),
        }
    critic = SR2Critic(width=cfg.critic_width, depth=cfg.critic_depth)
    return critic.to(device or "cpu"), {
        "source": "fresh_random_init_ABLATION_ONLY",
        "critic_warmup_steps": int(cfg.critic_warmup_steps),
        "note": "results from this arm are an ablation, not SR2 continuation",
    }
