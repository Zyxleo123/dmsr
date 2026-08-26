"""An HR critic (PatchGAN) over the high-pass 6-channel field, for the member-gather fine-tune.

Why this critic exists, and what it sees
----------------------------------------
The member-gather objective (:mod:`cosmo_sr.features.member_gather`) supervises the
*bound moments* of member sets -- virial ratio, boundness, size, dispersion,
centre -- and nothing else. The held-out Rockstar gate
(``docs/sr2_member_gather_training.md`` section 11) showed exactly the gap that
leaves: the ``self`` arm recovered HR's subhalo mass function out of sample
(20 -> 366 vs 369), and simultaneously

  * collapsed VELOCITY small-scale power 19-30x (section 11.6-3 -- the frozen
    field had it right to 2%, and the tune destroyed it, with no loss term
    watching);
  * put its high-k DISPLACEMENT power in the *wrong places* (real-space
    correlation with HR fell 0.915 -> 0.318, section 11.6-2);
  * lost resolved hosts locally (section 11.4).

Section 11.5's prescription was a hand-crafted hinge per artifact -- a two-sided
velocity-power term, a lower arm on the high-k hinge. This module is the
*general* version of that idea: a critic that sees whole HR tiles against tuned
tiles learns to penalise precisely the artifacts a moment loss leaves free,
because those artifacts are what make a tuned tile separable from a real one.

What it sees is the **null-space (high-pass) of all six channels**::

    x_hp = x - U(A(x))          A = block-average by the degrade factor s
                                U = block (nearest) upsample by s

so displacement *and* velocity small-scale structure are both judged, and the
LR-resolvable coarse field -- which is *identical* in real and fake because both
share the same LR tile -- is withheld. Withholding it is the same argument the
DMSR critic makes (:mod:`cosmo_sr.dmsr.critic`): a critic fed the resolved field
spends its capacity discriminating structure ``A_plus(y)`` already reproduces
byte-for-byte in both, and learns nothing about the unresolved detail we want.

**Velocity is in the input by construction.** That is the whole reason this
critic is worth adding over the hand-crafted velocity hinge: it does not need to
be told *which* statistic of the velocity field to preserve.

The critic, the hinge losses and the lazy-R1 stabiliser are reused unchanged from
:mod:`cosmo_sr.dmsr.critic` -- this module only supplies the gather-specific
*input view* (the six-channel high-pass) and its normaliser.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from ..operators.multiscale import null_projection


def highpass_field(x: torch.Tensor, factor: int) -> torch.Tensor:
    """``x - U(A(x))``: the part of the field the LR grid cannot represent.

    This is :func:`cosmo_sr.operators.multiscale.null_projection` applied to the
    whole ``(B, 6, N, N, N)`` tensor, so both the displacement channels ``[0:3]``
    and the velocity channels ``[3:6]`` are high-passed with the *same* operator
    the degrader and the LR-consistency guard use. Fully differentiable, so the
    generator's adversarial gradient reaches its weights through this view.
    """
    if x.dim() != 5:
        raise ValueError(f"highpass_field expects (B, C, N, N, N), got {tuple(x.shape)}")
    return null_projection(x, int(factor))


class GatherCriticNorm(nn.Module):
    """Fixed per-channel scales so every channel enters the critic comparably.

    The high-pass field mixes displacement channels (Mpc/h) and velocity channels
    (km/s); their raw RMS differ by orders of magnitude, so without normalisation
    the critic's first spectral-normed conv cannot rescale them and one group
    starves the other of gradient -- the same failure the DMSR
    :class:`~cosmo_sr.dmsr.density.CriticInputNormalizer` was written for.

    The scales are constants estimated from **real HR high-pass tiles only** and
    applied identically to real and fake. Per-batch statistics must never be used:
    real and fake would then be normalised differently, handing the critic a
    discriminative signal that has nothing to do with sample quality.
    """

    def __init__(self, scale: torch.Tensor):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    @classmethod
    @torch.no_grad()
    def fit(cls, hp_tiles: Iterable[torch.Tensor], eps: float = 1e-8) -> "GatherCriticNorm":
        """Per-channel RMS over an iterable of **real** HR high-pass tensors.

        Each element is ``(B, C, N, N, N)`` -- the output of
        :func:`highpass_field` on a batch of real HR tiles. The RMS is pooled over
        batch and space, giving one scale per channel, held fixed thereafter.
        """
        acc: Optional[torch.Tensor] = None
        n = 0
        for hp in hp_tiles:
            s = hp.reshape(hp.shape[0], hp.shape[1], -1).pow(2).mean(dim=(0, 2))
            acc = s if acc is None else acc + s
            n += 1
        if acc is None or n == 0:
            raise ValueError("GatherCriticNorm.fit got no tiles")
        rms = (acc / float(n)).sqrt().clamp_min(eps)
        return cls(rms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.scale.numel():
            raise ValueError(
                f"norm built for {self.scale.numel()} channels, got {x.shape[1]}")
        return x / self.scale.to(x.dtype).view(1, -1, *([1] * (x.dim() - 2)))

    def to_dict(self) -> Dict[str, list]:
        return {"scale": [float(v) for v in self.scale]}

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return "scale=[" + ", ".join(f"{float(v):.4g}" for v in self.scale) + "]"


def gather_critic_input(
    x: torch.Tensor,
    factor: int,
    normalizer: Optional[GatherCriticNorm] = None,
) -> torch.Tensor:
    """The critic's view of a gather tile: the normalised six-channel high-pass.

    ``x`` is a candidate or HR tile ``(B, 6, N, N, N)``; ``factor`` is the degrade
    block factor (``geom.scale_factor``). Differentiable in ``x``, so the same
    call builds the fake view the generator's adversarial loss flows through and
    (detached) the real/fake views the critic is trained on.
    """
    hp = highpass_field(x, factor)
    return hp if normalizer is None else normalizer(hp)
