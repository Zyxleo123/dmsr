"""A differentiable copy of :class:`cosmo_sr.reward.reward.RewardModel`.

The NumPy reward is the definition. This module is a *copy*, not a
reimplementation with the same spirit: the transforms, the empty-host-bin fill,
the active-dimension sub-block, the marginal (not conditional) block
Mahalanobis distances and the ridge are reproduced element for element, and
``tests/reward/test_torch_reward.py`` requires agreement to ``1e-5`` on real
fitted models -- including the case where a host bin is empty, which is the one
place the two could plausibly disagree and the one place a disagreement would be
invisible in the aggregate.

Why a copy at all
-----------------
Actor training needs ``dR/d(predicted tile summary)``. The NumPy model cannot
supply it, and approximating the reward with something differentiable-but-
different would mean the actor optimises a quantity no gate ever scores. So the
requirement is exact numerical agreement, and the test is written as an equality
rather than a correlation.

Precision, literally
--------------------
Everything here is ``float64`` by default. The quadratic form is 11-dimensional,
so the cost is irrelevant, while ``float32`` on a matrix whose condition number
is ~90 loses about three digits -- enough to fail a ``1e-5`` parity check for
reasons that have nothing to do with correctness. Gradients flow through
``float64`` perfectly well; the AMP region of the training step ends before the
reward head.

The swap form
-------------
Actor training never scores a tile in isolation, because the reward is a
whole-box statistic and a tile is 1/512 of a box. It scores the *box the tile
would have been part of*:

    dR_j = R(S_0 - s_{0,j} + s_hat_{theta,j}) - R(S_0)

with ``S_0`` the pooled frozen box summary and ``s_{0,j}`` the frozen tile's own
contribution -- both measured, both constants. :func:`delta_reward_swap`
implements exactly that, and only the predicted tile carries gradient.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .catalog import CatalogBins, EnsembleSummary
from .reward import RewardModel

__all__ = [
    "TorchRewardModel",
    "TorchSummary",
    "summary_from_ensemble",
    "summary_from_tiles",
]


@dataclass
class TorchSummary:
    """A batch of pooled catalog summaries, as tensors.

    Mirrors :class:`cosmo_sr.reward.catalog.EnsembleSummary` field for field.
    Shapes are ``(B, J)``, ``(B, I)``, ``(B, I)`` and ``(B,)``; a single summary
    is a batch of one, so nothing downstream needs a scalar special case.
    """

    n_sub: torch.Tensor
    n_host: torch.Tensor
    occ_numerator: torch.Tensor
    volume_mpc3: torch.Tensor

    def __post_init__(self) -> None:
        if self.n_sub.dim() != 2 or self.n_host.dim() != 2 or self.occ_numerator.dim() != 2:
            raise ValueError(
                "TorchSummary wants (B, J) / (B, I) / (B, I) tensors, got "
                f"{tuple(self.n_sub.shape)} / {tuple(self.n_host.shape)} / "
                f"{tuple(self.occ_numerator.shape)}"
            )
        if self.volume_mpc3.dim() != 1:
            raise ValueError(
                f"volume must be (B,), got {tuple(self.volume_mpc3.shape)}"
            )
        b = self.n_sub.shape[0]
        for name, t in (("n_host", self.n_host), ("occ_numerator", self.occ_numerator),
                        ("volume_mpc3", self.volume_mpc3)):
            if t.shape[0] != b:
                raise ValueError(f"{name} batch {t.shape[0]} != n_sub batch {b}")

    @property
    def batch(self) -> int:
        return int(self.n_sub.shape[0])

    def to(self, *args, **kwargs) -> "TorchSummary":
        return TorchSummary(
            self.n_sub.to(*args, **kwargs), self.n_host.to(*args, **kwargs),
            self.occ_numerator.to(*args, **kwargs),
            self.volume_mpc3.to(*args, **kwargs),
        )


def summary_from_ensemble(
    ens: EnsembleSummary, *, dtype=torch.float64, device=None
) -> TorchSummary:
    """Lift one NumPy :class:`EnsembleSummary` into a batch-of-one tensor summary."""
    def t(x):
        return torch.as_tensor(np.asarray(x, dtype=np.float64),
                               dtype=dtype, device=device).reshape(1, -1)
    return TorchSummary(
        n_sub=t(ens.n_sub), n_host=t(ens.n_host), occ_numerator=t(ens.occ_numerator),
        volume_mpc3=torch.as_tensor([float(ens.volume_mpc3)], dtype=dtype, device=device),
    )


def summary_from_tiles(
    tiles: Sequence, *, dtype=torch.float64, device=None
) -> TorchSummary:
    """Pool :class:`cosmo_sr.reward.tiles.TileSummary` objects into one summary.

    Same arithmetic as :func:`cosmo_sr.reward.tiles.pool_tiles` -- a sum of
    numerators, denominators and volumes -- kept here so a training job can pool
    without importing the NumPy pooling path.
    """
    items = list(tiles)
    if not items:
        raise ValueError("cannot pool an empty set of tiles")
    n_sub = np.sum([np.asarray(s.n_sub, dtype=np.float64) for s in items], axis=0)
    n_host = np.sum([np.asarray(s.n_host, dtype=np.float64) for s in items], axis=0)
    num = np.sum([np.asarray(s.occ_numerator, dtype=np.float64) for s in items], axis=0)
    vol = float(np.sum([float(s.volume_mpc3) for s in items]))
    return TorchSummary(
        n_sub=torch.as_tensor(n_sub, dtype=dtype, device=device).reshape(1, -1),
        n_host=torch.as_tensor(n_host, dtype=dtype, device=device).reshape(1, -1),
        occ_numerator=torch.as_tensor(num, dtype=dtype, device=device).reshape(1, -1),
        volume_mpc3=torch.as_tensor([vol], dtype=dtype, device=device),
    )


def _transform(x: torch.Tensor, kind: str, floor: torch.Tensor | float) -> torch.Tensor:
    """Element-for-element copy of :func:`cosmo_sr.reward.catalog._transform`."""
    if kind == "identity":
        return x
    if kind == "log10_floor":
        return torch.log10(x + floor)
    if kind == "sqrt":
        return torch.sqrt(x.clamp_min(0.0))
    raise ValueError(f"unknown transform {kind!r}")


class TorchRewardModel(nn.Module):
    """``R_cat``, ``R_occ`` and ``R_abund``, differentiable in the summary counts.

    Built from a fitted :class:`RewardModel` with :meth:`from_numpy`; there is no
    fitting path here on purpose. Fitting is a whole-box statistical procedure
    that belongs in one place, and a second implementation of it would be a
    second answer.
    """

    def __init__(
        self,
        mu: np.ndarray,
        cov_reg: np.ndarray,
        bins: CatalogBins,
        active_dims: Sequence[int],
        *,
        dtype=torch.float64,
        labels: Sequence[str] = (),
    ):
        super().__init__()
        mu = np.asarray(mu, dtype=np.float64)
        cov_reg = np.asarray(cov_reg, dtype=np.float64)
        d = int(mu.shape[0])
        if cov_reg.shape != (d, d):
            raise ValueError(f"cov_reg {cov_reg.shape} does not match mu {mu.shape}")
        self.bins = bins
        self.dtype_ = dtype
        self.labels = tuple(labels or bins.labels())
        self.active_dims = tuple(int(i) for i in active_dims)
        j, i = int(bins.n_sub_bins), int(bins.n_host_bins)
        if d != j + i:
            raise ValueError(f"mu dim {d} != n_sub_bins + n_host_bins = {j + i}")

        self.register_buffer("mu", torch.as_tensor(mu, dtype=dtype))
        self.register_buffer("cov_reg", torch.as_tensor(cov_reg, dtype=dtype))

        self.abundance_dims = tuple(k for k in self.active_dims if k < j)
        self.occupation_dims = tuple(k for k in self.active_dims if j <= k < j + i)
        for name, idx in (("active", self.active_dims),
                          ("occupation", self.occupation_dims),
                          ("abundance", self.abundance_dims)):
            self._register_block(name, idx, cov_reg)

    # -- construction ------------------------------------------------------ #
    def _register_block(self, name: str, idx: Sequence[int], cov_reg: np.ndarray) -> None:
        """Marginal precision of one block: invert the sub-block on its own.

        Deliberately not a slice of the joint precision and not the conditional
        (partial) form. Both of those mix the other block's residual in through
        the cross-covariance, and separating the two blocks is the entire point
        of having sub-rewards at all -- see the note in ``reward.py``.
        """
        k = np.asarray([int(v) for v in idx], dtype=np.int64)
        self.register_buffer(f"idx_{name}", torch.as_tensor(k, dtype=torch.int64))
        if k.size == 0:
            self.register_buffer(f"prec_{name}", torch.zeros((0, 0), dtype=self.dtype_))
            return
        p = np.linalg.inv(cov_reg[np.ix_(k, k)])
        p = 0.5 * (p + p.T)
        self.register_buffer(f"prec_{name}", torch.as_tensor(p, dtype=self.dtype_))

    @classmethod
    def from_numpy(cls, model: RewardModel, *, dtype=torch.float64) -> "TorchRewardModel":
        """Copy a fitted NumPy model, ridge included.

        ``cov_reg`` (``C + lambda I``) is taken rather than ``cov`` so the ridge
        cannot be applied twice or forgotten; ``active_dims`` is taken resolved
        rather than as the possibly-empty "all dimensions" convention, so the
        two models score the same sub-block by construction.
        """
        return cls(
            mu=model.mu, cov_reg=model.cov_reg, bins=model.bins,
            active_dims=model.active_dims, dtype=dtype,
            labels=model.labels or tuple(model.bins.labels()),
        )

    # -- summary vector ---------------------------------------------------- #
    def summary_vector(self, s: TorchSummary) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(s, valid)``: copy of :func:`cosmo_sr.reward.catalog.summary_vector`.

        Empty host bins are filled with ``mu`` -- so they contribute exactly zero
        to the distance rather than a fabricated penalty -- and are reported in
        ``valid`` so a caller can refuse to score a candidate that emptied a bin.
        Nothing here notices that emptying a bin is rewarded; that is the
        caller's job, exactly as in the NumPy model.
        """
        n_sub = s.n_sub.to(self.mu.dtype)
        n_host = s.n_host.to(self.mu.dtype)
        num = s.occ_numerator.to(self.mu.dtype)
        vol = s.volume_mpc3.to(self.mu.dtype).clamp_min(1e-30).unsqueeze(1)

        dens = n_sub / vol
        abundance = _transform(
            dens, self.bins.abundance_transform,
            float(self.bins.abundance_floor_halos) / vol,
        )
        empty = n_host <= 0
        safe_den = torch.where(empty, torch.ones_like(n_host), n_host)
        occ = torch.where(empty, torch.zeros_like(num), num / safe_den)
        occupation = _transform(
            occ, self.bins.occupation_transform, float(self.bins.occupation_floor)
        )
        vec = torch.cat([abundance, occupation], dim=1)
        valid = torch.cat([torch.ones_like(abundance, dtype=torch.bool), ~empty], dim=1)
        vec = torch.where(valid, vec, self.mu.unsqueeze(0).expand_as(vec))
        return vec, valid

    def occupation_curve(self, s: TorchSummary) -> torch.Tensor:
        """``<N_sub | M_host>`` per host bin, NaN where the bin has no hosts."""
        n_host = s.n_host.to(self.mu.dtype)
        num = s.occ_numerator.to(self.mu.dtype)
        empty = n_host <= 0
        safe = torch.where(empty, torch.ones_like(n_host), n_host)
        return torch.where(empty, torch.full_like(num, float("nan")), num / safe)

    # -- rewards ----------------------------------------------------------- #
    def _block_mahalanobis2(self, s: TorchSummary, name: str) -> torch.Tensor:
        idx = getattr(self, f"idx_{name}")
        prec = getattr(self, f"prec_{name}")
        if idx.numel() == 0:
            return torch.full((s.batch,), float("nan"),
                              dtype=self.mu.dtype, device=self.mu.device)
        vec, _ = self.summary_vector(s)
        d = (vec - self.mu.unsqueeze(0)).index_select(1, idx)
        return ((d @ prec) * d).sum(dim=1)

    def reward(self, s: TorchSummary) -> torch.Tensor:
        """``R_cat``: the joint score on the active sub-block."""
        return -self._block_mahalanobis2(s, "active")

    def reward_occupation(self, s: TorchSummary) -> torch.Tensor:
        return -self._block_mahalanobis2(s, "occupation")

    def reward_abundance(self, s: TorchSummary) -> torch.Tensor:
        return -self._block_mahalanobis2(s, "abundance")

    def reward_hosted_subs(self, s: TorchSummary, *,
                           floor: float = 0.5) -> torch.Tensor:
        """``log10`` of the box's total hosted-subhalo count.

        ``occ_numerator`` is the count of resolved subhalos binned by *host*
        mass, so summed over the host bins it is exactly "subhalos that live
        inside a resolved host" -- the literal target. A monotone functional,
        numerator-only (a candidate cannot raise it by deleting hosts, unlike the
        occupation ratio), and it sums exactly over tiles, so a tile swap is
        linear in the swapped counts before the log. The floor keeps an empty box
        finite and the gradient well defined; it matches the ``max(., 0.5)`` in
        the NumPy stability scan that selected this target.

        No covariance, no bin whitening: the proxy predicts one non-negative
        count per tile and this reads their sum. See ``reward_stability_scan``.
        """
        tot = s.occ_numerator.to(self.mu.dtype).sum(dim=1)
        return torch.log10(tot.clamp_min(float(floor)))

    def scores(self, s: TorchSummary) -> Dict[str, torch.Tensor]:
        """All three, always. Reporting one of them alone hides the trade."""
        return {
            "R_cat": self.reward(s),
            "R_occ": self.reward_occupation(s),
            "R_abund": self.reward_abundance(s),
        }

    def combined(
        self, s: TorchSummary, *, w_joint: float = 0.25, w_occ: float = 1.0
    ) -> torch.Tensor:
        """``w_occ * R_occ + w_joint * R_cat``.

        Occupation is primary by default because it is the scientific target and
        the statistic Gate B is decided on; the joint term is kept at a small
        weight so abundance cannot be traded away entirely to buy occupation.
        Both weights are configurable and both components are always logged
        separately, so the trade is visible rather than folded into one number.
        """
        return float(w_occ) * self.reward_occupation(s) \
            + float(w_joint) * self.reward(s)

    # -- the swap form ----------------------------------------------------- #
    def swap_summary(
        self,
        box: TorchSummary,
        frozen_tile: TorchSummary,
        predicted_tile: TorchSummary,
    ) -> TorchSummary:
        """``S_0 - s_{0,j} + s_hat_{theta,j}``, with counts floored at zero.

        The volume is **not** changed: a tile is replaced, not removed, so the
        box still covers the same volume. (Removal is a different operation with
        a different meaning -- see ``tiles.leave_one_out_credit`` -- and mixing
        the two silently rescales every number density.)

        The floor matters. A proxy is free to predict a tile contribution larger
        than the frozen box's total in some bin, and the resulting negative host
        count would make the occupation denominator negative and the reward
        nonsense. Clamping keeps the summary in the space real summaries live in;
        the clamp is reported by :meth:`swap_clamped_fraction` so a run where it
        binds often is visible rather than merely finite.
        """
        def mix(total, old, new):
            return (total - old + new).clamp_min(0.0)

        return TorchSummary(
            n_sub=mix(box.n_sub, frozen_tile.n_sub, predicted_tile.n_sub),
            n_host=mix(box.n_host, frozen_tile.n_host, predicted_tile.n_host),
            occ_numerator=mix(box.occ_numerator, frozen_tile.occ_numerator,
                              predicted_tile.occ_numerator),
            volume_mpc3=box.volume_mpc3,
        )

    @staticmethod
    def swap_clamped_fraction(
        box: TorchSummary, frozen_tile: TorchSummary, predicted_tile: TorchSummary
    ) -> torch.Tensor:
        """Fraction of summary entries the zero-floor had to move, per sample."""
        parts = []
        for total, old, new in (
            (box.n_sub, frozen_tile.n_sub, predicted_tile.n_sub),
            (box.n_host, frozen_tile.n_host, predicted_tile.n_host),
            (box.occ_numerator, frozen_tile.occ_numerator, predicted_tile.occ_numerator),
        ):
            parts.append((total - old + new) < 0)
        hit = torch.cat(parts, dim=1)
        return hit.to(torch.float64).mean(dim=1)

    def delta_reward_swap(
        self,
        box: TorchSummary,
        frozen_tile: TorchSummary,
        predicted_tile: TorchSummary,
        *,
        w_joint: float = 0.25,
        w_occ: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """``dR`` for every reward, from swapping one tile's summary.

        The baseline ``R(S_0)`` is computed under ``no_grad``: it is a constant
        of the box, and letting it carry gradient would add a term that cancels
        analytically but not numerically under AMP.
        """
        after = self.swap_summary(box, frozen_tile, predicted_tile)
        with torch.no_grad():
            before = self.scores(box)
            before_comb = self.combined(box, w_joint=w_joint, w_occ=w_occ)
        now = self.scores(after)
        out = {f"dR_{k[2:]}": now[k] - before[k].detach() for k in now}
        out["dR_combined"] = (
            self.combined(after, w_joint=w_joint, w_occ=w_occ) - before_comb.detach()
        )
        # The hosted-subhalo target (reward_stability_scan): numerator-only,
        # monotone, ungameable by host deletion. Baseline under no_grad like the
        # others -- it is a constant of the box.
        with torch.no_grad():
            hosted_before = self.reward_hosted_subs(box)
        out["dR_hosted_subs"] = self.reward_hosted_subs(after) - hosted_before.detach()
        out["clamped_fraction"] = self.swap_clamped_fraction(
            box, frozen_tile, predicted_tile)
        return out
