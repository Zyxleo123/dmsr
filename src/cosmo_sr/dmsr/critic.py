"""3D PatchGAN critic over ``(residual, rho_high)``, with hinge loss and lazy R1.

What the critic sees, and what it deliberately does not
-------------------------------------------------------
Input is ``concat(P_A(x), rho_high(x))``: the null-space field residual plus the
high-pass Eulerian density. Both are quantities the LR simulation cannot resolve,
which is the point -- a critic fed the full field would spend most of its capacity
discriminating low-frequency structure that ``A_plus(y)`` already reproduces
*exactly and identically* in real and fake samples, and would learn nothing about
the unresolved detail we actually want.

**No raw LR tensor is an input** in this first implementation. Conditioning the
critic on ``y`` would hand it the paired-vs-unpaired shortcut directly: in Stage D
the real examples come only from the 3 paired boxes while fakes come from
hundreds of LR-only boxes, so any LR-distribution cue is a free win for the critic
that has nothing to do with sample quality. Environment balancing
(:mod:`cosmo_sr.dmsr.env`) closes that gap statistically; withholding ``y``
closes the most direct route to it structurally.

Stabilisation is deliberately belt-and-braces because the generator is a
multi-step ODE and its adversarial gradient is comparatively weak: spectral norm
on every conv, hinge loss, and lazy R1 on real examples.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class HRCritic(nn.Module):
    """Spectral-norm 3D PatchGAN.

    Patch-based (fully convolutional, no global pooling) so the score is local:
    with only 3 paired boxes, a whole-volume discriminator would memorise them.

    Parameters
    ----------
    in_channels:
        ``C_residual + 1`` (the density high-pass channel).
    width:
        Base width; doubles at each stride-2 stage up to ``max_width``.
    n_layers:
        Number of stride-2 stages.
    """

    def __init__(
        self,
        in_channels: int,
        width: int = 64,
        n_layers: int = 3,
        max_width: int = 256,
        negative_slope: float = 0.2,
        global_pool: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.global_pool = bool(global_pool)
        layers: list[nn.Module] = [
            spectral_norm(nn.Conv3d(self.in_channels, width, 4, stride=2, padding=1)),
            nn.LeakyReLU(negative_slope, inplace=True),
        ]
        ch = width
        for _ in range(int(n_layers) - 1):
            nxt = min(ch * 2, int(max_width))
            layers += [
                spectral_norm(nn.Conv3d(ch, nxt, 4, stride=2, padding=1)),
                nn.LeakyReLU(negative_slope, inplace=True),
            ]
            ch = nxt
        layers += [
            spectral_norm(nn.Conv3d(ch, ch, 3, stride=1, padding=1)),
            nn.LeakyReLU(negative_slope, inplace=True),
        ]
        # global_pool: collapse the spatial patch grid to a SINGLE score before the
        # final 1x1 (SR2's discriminator is global, ours is a local PatchGAN by
        # default). The pool sees the whole crop jointly, so it can police
        # large-scale statistics a patch critic is blind to. Kept off by default so
        # existing arms are byte-identical.
        if self.global_pool:
            layers.append(nn.AdaptiveAvgPool3d(1))
        layers.append(spectral_norm(nn.Conv3d(ch, 1, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"critic expects (B, C, N, N, N), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"critic built for {self.in_channels} channels, got {x.shape[1]}"
            )
        return self.net(x)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def hinge_d_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """``mean(relu(1 - D(real))) + mean(relu(1 + D(fake)))``.

    ``fake_scores`` must come from a **detached** fake sample.
    """
    return F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()


def hinge_g_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """``-mean(D(fake))`` -- the generator's adversarial objective."""
    return -fake_scores.mean()


def r1_penalty(critic: nn.Module, real_input: torch.Tensor) -> torch.Tensor:
    """``mean(||grad_x D(x)||^2)`` on real examples (Mescheder et al. R1).

    ``real_input`` is re-attached to the graph here, so callers may pass a plain
    detached tensor.
    """
    real_input = real_input.detach().requires_grad_(True)
    scores = critic(real_input)
    grad = torch.autograd.grad(
        outputs=scores.sum(), inputs=real_input, create_graph=True, retain_graph=True
    )[0]
    return grad.reshape(grad.shape[0], -1).pow(2).sum(dim=1).mean()


class LazyR1:
    """Apply R1 every ``interval`` critic steps, scaled by ``interval``.

    Scaling by the interval keeps the *effective* regularization strength equal
    to applying ``gamma`` every step, so ``interval`` can be tuned for cost
    without silently changing the amount of regularization -- and R1 is expensive
    here (a double-backward through a 3D conv stack).
    """

    def __init__(self, gamma: float = 10.0, interval: int = 16):
        self.gamma = float(gamma)
        self.interval = max(1, int(interval))
        self._step = 0

    def due(self) -> bool:
        return self._step % self.interval == 0

    def __call__(
        self, critic: nn.Module, real_input: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Returns ``(weighted_penalty_or_None, metrics)`` and advances the counter."""
        due = self.due()
        self._step += 1
        if not due or self.gamma <= 0.0:
            return None, {}
        raw = r1_penalty(critic, real_input)
        weighted = 0.5 * self.gamma * raw * self.interval
        return weighted, {"loss_R1": float(raw.detach())}


@torch.no_grad()
def critic_scores(critic: nn.Module, x: torch.Tensor) -> float:
    """Mean patch score, for logging."""
    return float(critic(x).mean())
