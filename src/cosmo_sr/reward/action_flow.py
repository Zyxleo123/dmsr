"""Distil successful editor actions into a conditional policy.

Once CEM has found actions that make Rockstar produce a new subhalo, the search
itself is not the deliverable -- it costs a full-box halo run per candidate and
it starts from scratch on every host. What we want is ``q_theta(a | h, c)``: given
a frozen host's features ``h`` and a desired subhalo token ``c``, sample an
action likely to realise it, at zero Rockstar cost.

Two models, and the baseline is not optional
--------------------------------------------
:class:`GaussianMixturePolicy` is a conditional diagonal-Gaussian mixture trained
by reward-weighted maximum likelihood. :class:`ActionFlow` is conditional flow
matching. The flow is only justified if it beats the mixture at equal Rockstar
budget -- on reward, or on covering genuinely distinct action modes. A flow that
merely matches a 4-component GMM on an 8-dimensional action space is a more
expensive way to do the same thing, and the evaluation is set up to say so.

Both are MLPs. A 3D U-Net would be the reflex given the rest of this repo, and it
would be wrong: the action is 8 numbers, not a field. The field-scale model is
exactly what failed.

Reward weighting
----------------
    w_i = clip(exp((r_i - b(h_i)) / tau), 0, w_max),  then normalised to mean 1.

``b(h_i)`` is the host's own mean reward. Subtracting a *per-host* baseline is
what stops the objective from being dominated by whichever host happens to be
easiest -- without it the policy learns "act like you are editing host 7",
which generalises to nothing. Clipping and mean-normalisation keep the effective
batch size from collapsing onto one sample when one reward is an outlier.

Mode collapse and the reference penalty
---------------------------------------
Reward-weighted flow matching has an obvious attractor: put all mass on the
single best action ever seen. The countermeasure here is a penalty on the
squared velocity difference against a **frozen reference** copy of the network
(the initial, unweighted fit), which bounds how far the policy transports mass
away from the distribution it started from.

This follows the motivation of Wasserstein-regularised online reward-weighted
flow matching (ORW-CFM-W2, https://arxiv.org/abs/2502.06061) -- a W2 trust
region on the transported distribution -- but it is **not** a reproduction of
that paper's objective or training loop. The paper's regulariser is derived from
the W2 geometry of the flow; this is a first-order surrogate. Anything reporting
these results should say the former, not the latter.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ActionFlow",
    "GaussianMixturePolicy",
    "flow_matching_loss",
    "host_features",
    "reference_velocity_penalty",
    "reward_weights",
    "token_features",
]


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------

HOST_FEATURE_NAMES = (
    "log10_mvir", "log10_rvir_mpc", "log10_vmax",
    "n_sub_current", "smooth_fraction", "log10_n_members",
)
TOKEN_FEATURE_NAMES = (
    "log_mass_ratio", "radius_rvir", "dir_x", "dir_y", "dir_z",
)


def host_features(host: Dict) -> np.ndarray:
    """``h``: frozen host descriptors the action can legitimately depend on.

    Every entry is measurable on the frozen SR2 box alone -- mass, radius,
    ``Vmax``, how many subhalos it already has, and what fraction of its
    particles are smooth (i.e. how much material an edit has to work with). No
    HR quantity appears, which is what makes the trained policy deployable.
    """
    return np.asarray([
        np.log10(max(float(host["mvir"]), 1e-30)),
        np.log10(max(float(host["rvir_mpc"]), 1e-30)),
        np.log10(max(float(host.get("vmax", 1.0)), 1e-30)),
        float(host.get("n_sub_current", 0.0)),
        float(host.get("smooth_fraction", 1.0)),
        np.log10(max(float(host.get("n_members", 1)), 1.0)),
    ], dtype=np.float64)


def token_features(token: Dict) -> np.ndarray:
    d = token.get("direction", (0.0, 0.0, 1.0))
    return np.asarray([
        float(token["log_mass_ratio"]), float(token["radius_rvir"]),
        float(d[0]), float(d[1]), float(d[2]),
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------


def reward_weights(
    rewards: Sequence[float],
    host_ids: Sequence[int],
    *,
    tau: float = 0.5,
    w_max: float = 10.0,
    baseline: str = "mean",
) -> np.ndarray:
    """Bounded, host-baselined, mean-one weights.

    A host with a single sample gets ``r - b = 0`` and therefore weight 1: with
    nothing to compare against, the honest statement is "no evidence this action
    was better than typical for this host", not "this action is the best one".

    The bound is applied **symmetrically, to the exponent**, so weights land in
    ``[1/w_max, w_max]`` before normalisation. Clipping only the top would not
    do the job it is there for: with one reward far above the baseline, every
    other sample underflows to zero and the effective batch size collapses to
    one regardless of what the maximum is capped at. Bounding the exponent caps
    the *ratio* between the largest and smallest weight at ``w_max^2``, which is
    the quantity that controls how many samples the gradient actually sees.
    """
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    h = np.asarray(host_ids).reshape(-1)
    if r.shape != h.shape:
        raise ValueError(f"{r.shape} rewards but {h.shape} host ids")
    b = np.zeros_like(r)
    for hid in np.unique(h):
        m = h == hid
        b[m] = np.mean(r[m]) if baseline == "mean" else np.median(r[m])
    lim = float(np.log(max(float(w_max), 1.0 + 1e-12)))
    w = np.exp(np.clip((r - b) / max(float(tau), 1e-9), -lim, lim))
    s = float(w.mean())
    return w / s if s > 0 else np.ones_like(w)


# ---------------------------------------------------------------------------
# Conditional flow matching
# ---------------------------------------------------------------------------


def _mlp(din: int, dout: int, width: int, depth: int) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Linear(din, width), nn.SiLU()]
    for _ in range(max(0, depth - 1)):
        layers += [nn.Linear(width, width), nn.SiLU()]
    layers += [nn.Linear(width, dout)]
    return nn.Sequential(*layers)


class ActionFlow(nn.Module):
    """``v_theta(a_t, t, h, c)``: the conditional flow-matching velocity field."""

    def __init__(self, action_dim: int, cond_dim: int, *,
                 width: int = 128, depth: int = 3, time_freqs: int = 8):
        super().__init__()
        self.action_dim = int(action_dim)
        self.cond_dim = int(cond_dim)
        self.time_freqs = int(time_freqs)
        din = self.action_dim + 2 * self.time_freqs + self.cond_dim
        self.net = _mlp(din, self.action_dim, int(width), int(depth))

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Fourier features of ``t``. A raw scalar in ``[0, 1]`` is a weak input
        to a SiLU MLP; the endpoints of the path matter and need resolving."""
        f = torch.arange(self.time_freqs, device=t.device, dtype=t.dtype)
        ang = t.reshape(-1, 1) * (2.0 ** f).reshape(1, -1) * np.pi
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

    def forward(self, a: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([a, self.time_embedding(t), cond], dim=1))

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, *, n_steps: int = 64,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Integrate ``da/dt = v`` from ``z ~ N(0, I)`` at ``t=0`` to ``t=1``.

        Explicit Euler with a fixed step count: the path is a straight line by
        construction of the target, so the integrator's job is to not be the
        bottleneck rather than to be accurate about curvature.
        """
        n = cond.shape[0]
        a = torch.randn(n, self.action_dim, device=cond.device, dtype=cond.dtype,
                        generator=generator)
        dt = 1.0 / int(n_steps)
        for k in range(int(n_steps)):
            t = torch.full((n,), k * dt, device=cond.device, dtype=cond.dtype)
            a = a + dt * self.forward(a, t, cond)
        return a


def flow_matching_loss(
    net: ActionFlow,
    a1: torch.Tensor,
    cond: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Reward-weighted conditional flow matching on the straight path.

        a_t = (1 - t) z + t a_1,      v* = a_1 - z,
        L   = sum_i w_i || v_theta(a_t, t, h_i, c_i) - v* ||^2

    ``z`` is standard normal, so ``t = 0`` is the prior and ``t = 1`` is the data
    -- the same orientation :meth:`ActionFlow.sample` integrates in.
    """
    n = a1.shape[0]
    z = torch.randn(a1.shape, device=a1.device, dtype=a1.dtype, generator=generator)
    t = torch.rand(n, device=a1.device, dtype=a1.dtype, generator=generator)
    at = (1.0 - t).reshape(-1, 1) * z + t.reshape(-1, 1) * a1
    target = a1 - z
    pred = net(at, t, cond)
    per = ((pred - target) ** 2).mean(dim=1)
    if weights is None:
        loss = per.mean()
    else:
        loss = (weights.reshape(-1) * per).mean()
    return loss, {"cfm": float(loss.detach().cpu()),
                  "unweighted_cfm": float(per.mean().detach().cpu())}


def reference_velocity_penalty(
    net: ActionFlow, ref: ActionFlow, a1: torch.Tensor, cond: torch.Tensor,
    *, generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Mean squared velocity difference against a frozen reference network.

    Evaluated on the same interpolation path the loss uses, so the penalty and
    the objective disagree about the same points rather than about a resampled
    cloud. See the module docstring for what this is and is not.
    """
    n = a1.shape[0]
    z = torch.randn(a1.shape, device=a1.device, dtype=a1.dtype, generator=generator)
    t = torch.rand(n, device=a1.device, dtype=a1.dtype, generator=generator)
    at = (1.0 - t).reshape(-1, 1) * z + t.reshape(-1, 1) * a1
    with torch.no_grad():
        v_ref = ref(at, t, cond)
    return ((net(at, t, cond) - v_ref) ** 2).mean()


# ---------------------------------------------------------------------------
# The mandatory baseline
# ---------------------------------------------------------------------------


class GaussianMixturePolicy(nn.Module):
    """Conditional diagonal-Gaussian mixture, trained by weighted max-likelihood.

    Deliberately the simplest thing that can be multimodal. If the CEM data has
    two ways to make a subhalo in a given host, this will find them, and the
    flow has to do better than that to earn its place.
    """

    def __init__(self, action_dim: int, cond_dim: int, *, n_components: int = 4,
                 width: int = 128, depth: int = 2,
                 log_std_range: Tuple[float, float] = (-4.0, 1.5)):
        super().__init__()
        self.action_dim = int(action_dim)
        self.n_components = int(n_components)
        self.log_std_range = (float(log_std_range[0]), float(log_std_range[1]))
        self.net = _mlp(int(cond_dim), self.n_components * (1 + 2 * self.action_dim),
                        int(width), int(depth))

    def _params(self, cond: torch.Tensor):
        k, d = self.n_components, self.action_dim
        raw = self.net(cond)
        logits = raw[:, :k]
        mu = raw[:, k:k + k * d].reshape(-1, k, d)
        lo, hi = self.log_std_range
        log_std = raw[:, k + k * d:].reshape(-1, k, d)
        log_std = lo + (hi - lo) * torch.sigmoid(log_std)
        return logits, mu, log_std

    def log_prob(self, a: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        logits, mu, log_std = self._params(cond)
        x = a.unsqueeze(1)
        comp = (-0.5 * ((x - mu) / log_std.exp()) ** 2 - log_std
                - 0.5 * float(np.log(2.0 * np.pi))).sum(dim=2)
        return torch.logsumexp(F.log_softmax(logits, dim=1) + comp, dim=1)

    def loss(self, a: torch.Tensor, cond: torch.Tensor,
             weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        lp = self.log_prob(a, cond)
        return -(lp if weights is None else weights.reshape(-1) * lp).mean()

    @torch.no_grad()
    def sample(self, cond: torch.Tensor,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        logits, mu, log_std = self._params(cond)
        u = torch.rand(logits.shape, device=cond.device, dtype=cond.dtype,
                       generator=generator).clamp_min(1e-20)
        pick = torch.argmax(F.log_softmax(logits, dim=1) - torch.log(-torch.log(u)),
                            dim=1)
        idx = pick.reshape(-1, 1, 1).expand(-1, 1, self.action_dim)
        m = mu.gather(1, idx).squeeze(1)
        s = log_std.gather(1, idx).squeeze(1).exp()
        eps = torch.randn(m.shape, device=cond.device, dtype=cond.dtype,
                          generator=generator)
        return m + s * eps


# ---------------------------------------------------------------------------
# Diversity, for the "is the flow worth it" comparison
# ---------------------------------------------------------------------------


def action_diversity(samples: np.ndarray) -> Dict[str, float]:
    """Spread of a set of sampled actions, in unconstrained coordinates.

    ``mean_pairwise`` is the plain average distance; ``min_std`` is the smallest
    per-coordinate standard deviation, which is the one that detects a policy
    that is diverse in seven dimensions and collapsed in the eighth.
    """
    a = np.asarray(samples, dtype=np.float64).reshape(len(samples), -1)
    if a.shape[0] < 2:
        return {"mean_pairwise": float("nan"), "min_std": float("nan"),
                "mean_std": float("nan")}
    d = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2)
    iu = np.triu_indices(a.shape[0], k=1)
    sd = a.std(axis=0, ddof=1)
    return {"mean_pairwise": float(d[iu].mean()),
            "min_std": float(sd.min()), "mean_std": float(sd.mean())}
