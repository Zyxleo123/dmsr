"""Conditional 3D residual denoiser ``eps_phi(dPsi_t, t, y, Psi_base, z)``.

Deliberately not a new architecture: the trunk is the existing
:class:`~cosmo_sr.models.flow_unet.Map2MapUNet3D` (3D U-Net, GroupNorm, SiLU,
FiLM conditioning, optional gradient checkpointing), with the residual-specific
plumbing on top:

* conditioning fields are concatenated channel-wise --
  ``[dPsi_t (C), Psi_base (C), U(y_lr) (C)]``, where ``U`` is the nearest
  block-broadcast upsample already used by every flow model here;
* ``t`` and the redshift ``z`` are sinusoidally embedded, summed, and injected
  as FiLM, so a single checkpoint can serve several redshifts once matched
  ``z != 0`` pairs exist;
* the output head is zero-initialised, so an untrained model predicts ``eps = 0``.
  That does **not** make the sampled residual zero, and it does not make the
  composed field equal the frozen SR2 output. A DDIM/DDPM sample starts from
  ``u_T ~ N(0, I)`` and each step is a function of that draw: with
  ``eps_hat = 0`` the update reduces to ``u_0_hat = u_t / alpha_t`` and
  ``u_next = alpha_next * u_0_hat`` (plus ``sigma_next * eps_hat = 0``), i.e. the
  initial noise *rescaled* -- and rescaled by a factor that diverges as
  ``alpha(t_max) -> 0``, which is the only reason ``x0_clip`` exists (see
  :class:`~cosmo_sr.reward.diffusion.DiffusionConfig`). A zero head therefore
  yields a residual of the clip's amplitude, not of zero amplitude.

  The single exact statement is: **only ``residual_scale = 0`` recovers the
  frozen SR2 output bit for bit**, because
  :func:`~cosmo_sr.reward.base.compose` short-circuits and returns ``psi_base``
  untouched rather than adding ``0 * dPsi``. Nothing about the initialisation of
  this module can substitute for that. ``tests/reward/test_zero_init_claim.py``
  pins this down so the claim cannot come back.
* normalisation defaults to ``"channel"``
  (:class:`~cosmo_sr.models.flow_unet.ChannelGroupNorm3d`), not ``nn.GroupNorm``.
  GroupNorm reduces over space, so a tile of size A and a tile of size B
  renormalise the same physical region differently -- which would make tiled
  full-box sampling depend on the tiling and reintroduce the train/eval window
  mismatch that layer was written to fix. Channel norm is strictly pointwise, so
  valid-core tiling is exact.

The model predicts ``eps`` on the **whitened** residual (see
:mod:`cosmo_sr.reward.diffusion`); ``sigma_res`` is carried as a buffer so a
checkpoint is self-describing.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..models.flow_unet import Map2MapUNet3D
from ..models.residual_flow import sinusoidal_embedding
from ..operators.multiscale import block_upsample

__all__ = ["ResidualDenoiser", "build_residual_denoiser"]


class ResidualDenoiser(nn.Module):
    """``(u_t, t, y_lr, psi_base, z) -> eps_hat`` on a cubic HR crop.

    ``u_t`` and ``psi_base`` are on the HR grid; ``y_lr`` is on the LR grid and is
    broadcast up by ``scale_factor``. All three must describe the *same* region.
    """

    def __init__(
        self,
        channels: int = 6,
        scale_factor: int = 8,
        width: int = 48,
        num_levels: int = 3,
        blocks_per_level: int = 2,
        embed_dim: int = 128,
        num_groups: int = 8,
        use_attention: bool = False,
        use_checkpoint: bool = False,
        padding_mode: str = "circular",
        norm: str = "channel",
        sigma_res: Optional[Sequence[float]] = None,
        condition_on_lr: bool = True,
        condition_on_base: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.scale_factor = int(scale_factor)
        self.embed_dim = int(embed_dim)
        self.condition_on_lr = bool(condition_on_lr)
        self.condition_on_base = bool(condition_on_base)

        in_ch = self.channels
        if self.condition_on_base:
            in_ch += self.channels
        if self.condition_on_lr:
            in_ch += self.channels

        self.t_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.z_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.unet = Map2MapUNet3D(
            in_channels=in_ch,
            out_channels=self.channels,
            width=width,
            num_levels=num_levels,
            blocks_per_level=blocks_per_level,
            kernel_size=3,
            padding="same",
            norm=norm,
            num_groups=num_groups,
            activation="silu",
            use_resblocks=True,
            cond_dim=embed_dim,
            use_attention=use_attention,
            zero_init_tail=True,
            use_checkpoint=use_checkpoint,
            global_bypass=False,
            padding_mode=padding_mode,
        )
        s = torch.ones(self.channels) if sigma_res is None else torch.as_tensor(
            sigma_res, dtype=torch.float32
        )
        if s.numel() != self.channels:
            raise ValueError(f"sigma_res must have {self.channels} entries, got {s.numel()}")
        self.register_buffer("sigma_res", s.float())

    @property
    def divisor(self) -> int:
        """Crop sizes must be a multiple of this for ``padding='same'`` U-Nets."""
        return 2 ** self.unet.num_levels

    def _cond(self, t: torch.Tensor, z: torch.Tensor, batch: int, device) -> torch.Tensor:
        t = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1)
        z = torch.as_tensor(z, device=device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(batch)
        if z.numel() == 1:
            z = z.expand(batch)
        return self.t_mlp(sinusoidal_embedding(t, self.embed_dim)) + self.z_mlp(
            sinusoidal_embedding(torch.log1p(z.clamp_min(0.0)), self.embed_dim)
        )

    def forward(
        self,
        u_t: torch.Tensor,
        t: torch.Tensor,
        *,
        y_lr: Optional[torch.Tensor] = None,
        psi_base: Optional[torch.Tensor] = None,
        redshift: float | torch.Tensor = 0.0,
    ) -> torch.Tensor:
        if u_t.dim() != 5:
            raise ValueError(f"u_t must be (B, C, D, H, W), got {tuple(u_t.shape)}")
        n = u_t.shape[-1]
        if n % self.divisor:
            raise ValueError(
                f"crop size {n} must be a multiple of {self.divisor} "
                f"(2**num_levels) for padding='same'"
            )
        feats = [u_t]
        if self.condition_on_base:
            if psi_base is None:
                raise ValueError("condition_on_base=True but psi_base was not given")
            if psi_base.shape != u_t.shape:
                raise ValueError(
                    f"psi_base {tuple(psi_base.shape)} != u_t {tuple(u_t.shape)}"
                )
            feats.append(psi_base)
        if self.condition_on_lr:
            if y_lr is None:
                raise ValueError("condition_on_lr=True but y_lr was not given")
            y_up = block_upsample(y_lr, self.scale_factor)
            if y_up.shape[-1] != n:
                raise ValueError(
                    f"U(y_lr) size {y_up.shape[-1]} != crop size {n}; the LR crop "
                    f"must cover exactly the same region at 1/{self.scale_factor} scale"
                )
            feats.append(y_up)
        x = torch.cat(feats, dim=1)
        cond = self._cond(t, torch.as_tensor(redshift), u_t.shape[0], u_t.device)
        return self.unet(x, cond)


def build_residual_denoiser(model_cfg: dict) -> ResidualDenoiser:
    """Construct from a config dict, ignoring a ``name`` key if present."""
    kwargs = {k: v for k, v in dict(model_cfg or {}).items() if k != "name"}
    return ResidualDenoiser(**kwargs)
