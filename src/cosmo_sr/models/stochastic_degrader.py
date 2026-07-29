"""Stochastic degrader: ``p(x_R | x_2R, R)`` via conditional flow matching.

Why this exists (and why the deterministic ``LearnedDegrader`` cannot be fixed
by making it bigger):

On genuinely independent paired N-body runs (Ng=512 HR vs Ng=64 LR, same ICs),
the LR field is *not* a deterministic function of the HR field. The two runs
have different particle mass, softening and timestepping, so inside collapsed
regions the LR particles' small-scale velocities chaotically decorrelate from
the HR run's. Measured on set0: the cross-correlation ``r(k)`` between
``A(x_HR)`` and ``x_LR`` falls to ~0.70 by 45% of Nyquist and ~0.31 near
Nyquist for the velocity channels, and the *optimal* linear shift-invariant
filter beats plain ``A`` by only 0.4%.

An MSE-trained deterministic model can therefore only ever learn
``E[x_R | x_2R]``, and its irreducible loss is the conditional variance. Worse,
regression to the conditional mean systematically *destroys power*: block
averaging already suppresses the velocity dispersion (``std(A_vel) ~ 0.79-0.84``
vs ``std(x_LR_vel) ~ 0.97-1.01``), and an MSE fit only smooths further. The
degraded fields it produces are not statistically valid LR fields.

This module models the whole conditional distribution instead. With

    r      = (x_R - A_R(x_2R)) / sigma          # sigma = per-channel resid std
    r_t    = (1 - t) * z + t * r,   z ~ N(0, I)
    v*     = r - z

we train ``v_theta(r_t, t, x_2R, R)`` to regress ``v*`` (rectified/linear
interpolant conditional flow matching -- same objective as ``losses/flow.py``),
and sample by integrating ``dr/dt = v_theta`` from ``z`` at ``t=0`` to ``t=1``:

    D_phi(x_2R, R; eps) = A_R(x_2R) + sigma * ODE_solve(z)

Each call is a *sample* from ``p(x_R | x_2R)``, so it carries the right velocity
power rather than the over-smoothed mean. Averaging ``K`` samples recovers the
conditional mean (i.e. reproduces the deterministic degrader) as a special case,
so nothing is lost -- see :meth:`StochasticDegrader.conditional_mean`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..operators.multiscale import MultiScaleOperators
from .learned_degrader import _FiLMResBlock3d, _groups
from .residual_flow import sinusoidal_embedding


class StochasticDegrader(nn.Module):
    """Conditional flow-matching degrader ``p(x_R | x_2R, R)``.

    Parameters
    ----------
    channels:
        Field channels (6: disp[0:3] + vel[3:6]).
    width, depth:
        Feature width and number of FiLM residual blocks in ``v_theta``.
    factor:
        Downsample factor (``2R -> R``; 8 for the 512->64 paired boxes).
    use_res_embed:
        Inject a resolution embedding (from ``log2(R)``) alongside the time
        embedding via FiLM, so one model can serve several octaves.
    embed_dim:
        Conditioning embedding width.
    """

    def __init__(
        self,
        channels: int = 6,
        width: int = 64,
        depth: int = 4,
        factor: int = 8,
        use_res_embed: bool = True,
        embed_dim: int = 64,
        groups: int = 8,
    ):
        super().__init__()
        self.channels = int(channels)
        self.factor = int(factor)
        self.use_res_embed = bool(use_res_embed)
        self.embed_dim = int(embed_dim)
        self.ops = MultiScaleOperators(self.factor)

        # Per-channel residual std, used to whiten the flow-matching target.
        # Registered as a buffer so it round-trips through the checkpoint --
        # sampling is wrong if it doesn't match what training used.
        self.register_buffer("residual_std", torch.ones(self.channels))

        # t (and optionally log2 R) -> FiLM conditioning vector
        self.t_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        if use_res_embed:
            self.r_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
            )

        # Conditioning encoder: x_2R (grid 2R) -> features on the LR grid R.
        # Same geometry as LearnedDegrader.head so each output cell sees the
        # full HR block it is responsible for (plus overlap).
        self.cond_head = nn.Conv3d(
            channels, width, kernel_size=2 * factor, stride=factor, padding=factor // 2
        )
        # Noisy-residual encoder: r_t already lives on the LR grid.
        self.r_head = nn.Conv3d(channels, width, kernel_size=3, padding=1)
        self.merge = nn.Conv3d(2 * width, width, kernel_size=1)

        self.blocks = nn.ModuleList(
            [_FiLMResBlock3d(width, embed_dim, groups) for _ in range(max(depth, 1))]
        )
        self.out_norm = nn.GroupNorm(_groups(groups, width), width)
        self.out_act = nn.SiLU()
        self.final = nn.Conv3d(width, channels, kernel_size=3, padding=1)
        # Zero-init the output so v_theta == 0 at init: the initial flow is the
        # identity map from z, i.e. samples start as pure noise of the right
        # scale rather than an arbitrary large field.
        nn.init.zeros_(self.final.weight)
        if self.final.bias is not None:
            nn.init.zeros_(self.final.bias)

    # ---------------------------------------------------------------- utils
    def set_residual_std(self, std: torch.Tensor) -> None:
        """Set the per-channel whitening scale (call once, before training)."""
        std = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        if std.numel() != self.channels:
            raise ValueError(
                f"residual_std must have {self.channels} entries, got {std.numel()}"
            )
        self.residual_std.copy_(std.clamp_min(1e-6).to(self.residual_std.device))

    def _sigma(self) -> torch.Tensor:
        return self.residual_std.view(1, -1, 1, 1, 1)

    def _cond(self, t: torch.Tensor, R, batch: int, device) -> torch.Tensor:
        cond = self.t_mlp(sinusoidal_embedding(t.reshape(-1), self.embed_dim))
        if self.use_res_embed:
            R = torch.as_tensor(R, device=device, dtype=torch.float32).reshape(-1)
            if R.numel() == 1:
                R = R.expand(batch)
            cond = cond + self.r_mlp(
                sinusoidal_embedding(torch.log2(R.clamp_min(1.0)), self.embed_dim)
            )
        return cond

    # ------------------------------------------------------------- forward
    def forward(self, r_t: torch.Tensor, t: torch.Tensor, x_2R: torch.Tensor, R) -> torch.Tensor:
        """Predicted flow-matching velocity ``v_theta(r_t, t, x_2R, R)``.

        ``r_t`` is the *whitened* noisy residual on the LR grid.
        """
        cond = self._cond(t, R, r_t.shape[0], r_t.device)
        h = torch.cat([self.r_head(r_t), self.cond_head(x_2R)], dim=1)
        h = self.merge(h)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.final(self.out_act(self.out_norm(h)))

    def target_residual(self, x_R: torch.Tensor, x_2R: torch.Tensor) -> torch.Tensor:
        """Whitened target residual ``(x_R - A_R(x_2R)) / sigma``."""
        return (x_R - self.ops.A(x_2R)) / self._sigma()

    # ------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(
        self,
        x_2R: torch.Tensor,
        R,
        n_steps: int = 50,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Draw one ``x_R ~ p(. | x_2R)`` by integrating the flow (midpoint ODE)."""
        a_base = self.ops.A(x_2R)
        r = torch.randn(a_base.shape, device=a_base.device, dtype=a_base.dtype,
                        generator=generator)
        dt = 1.0 / max(1, int(n_steps))
        for i in range(int(n_steps)):
            t0 = torch.full((r.shape[0],), i * dt, device=r.device, dtype=r.dtype)
            # midpoint (RK2): markedly more accurate than Euler at equal cost/2
            v0 = self(r, t0, x_2R, R)
            t_mid = t0 + 0.5 * dt
            v_mid = self(r + 0.5 * dt * v0, t_mid, x_2R, R)
            r = r + dt * v_mid
        return a_base + self._sigma() * r

    @torch.no_grad()
    def conditional_mean(
        self, x_2R: torch.Tensor, R, n_samples: int = 8, n_steps: int = 50
    ) -> torch.Tensor:
        """Monte-Carlo ``E[x_R | x_2R]`` -- what the deterministic degrader fits.

        Useful as an apples-to-apples check: this should reach roughly the same
        MSE as the MSE-trained ``LearnedDegrader``, while :meth:`sample` should
        instead match the *power spectrum* of the true LR field.
        """
        acc = None
        for _ in range(max(1, int(n_samples))):
            s = self.sample(x_2R, R, n_steps=n_steps)
            acc = s if acc is None else acc + s
        return acc / max(1, int(n_samples))
