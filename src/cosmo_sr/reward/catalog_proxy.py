"""Learned surrogate for a tile's catalog sufficient statistics.

What it predicts, and why not a scalar
--------------------------------------
The obvious surrogate is "predict the reward". It is the wrong target here for
two reasons. The reward is a *whole-box* Mahalanobis distance, so a tile does
not have one; and a scalar throws away the structure that makes the prediction
checkable. What a tile does have, exactly, is its fractional contribution to the
box's sufficient statistics -- the three vectors
:class:`cosmo_sr.reward.tiles.TileSummary` already carries:

* ``N_j``  subhalo counts in six subhalo-mass bins;
* ``H_j``  host counts in five host-mass bins (the occupation denominator);
* ``S_j``  subhalo counts binned by their *host's* mass (the numerator).

Predict those and the reward follows through
:class:`cosmo_sr.reward.torch_reward.TorchRewardModel`, differentiably and
without a second approximation. Occupation is then ``O = sum_j S_j / sum_j H_j``
after pooling, which is a *ratio of sums*, not a sum of ratios -- predicting
occupation per tile directly would get that wrong for every tile with few hosts,
which is most of them.

Two losses, doing different jobs
--------------------------------
``count_loss``
    Huber on ``log1p`` counts. ``log1p`` because the bins span three orders of
    magnitude and a plain L2 would fit the 1e12 host bin and ignore the 3.16e13
    one, which is the bin Gate B is decided on. Huber because tile counts are
    fractional sums of a handful of objects and are genuinely heavy-tailed.
``ranking_loss``
    Pairwise, on the true baseline-relative catalog reward, and formed **only
    within the same (box, tile_id)**. This is the load-bearing constraint of the
    whole module: a proxy that ranks tiles across environments has learned where
    the hosts are, which the actor cannot change. The actor changes what happens
    *in a given tile*, so that is the comparison the proxy has to get right.

An ensemble, because the actor optimises against it
---------------------------------------------------
Three to five members with different seeds. The actor maximises a lower
confidence bound ``mean_m Q_m - beta * std_m Q_m``, so a direction the members
disagree about is worth less than one they agree on. That is the only defence
against the actor walking off the data manifold the proxy was fitted on, and it
requires the spread to be *measured* rather than assumed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_reward import TorchRewardModel, TorchSummary

__all__ = [
    "CatalogProxy",
    "ProxyConfig",
    "ProxyEnsemble",
    "bin_weights_from_counts",
    "count_loss",
    "make_within_tile_pairs",
    "pairwise_ranking_loss",
    "spearman",
    "split_indices_by_box",
]


@dataclass
class ProxyConfig:
    """Shape of one proxy. Small on purpose -- see the module docstring."""

    n_features: int = 26
    n_sub_bins: int = 6
    n_host_bins: int = 5
    hidden: Tuple[int, ...] = (128, 128)
    dropout: float = 0.0
    #: Predicted counts are ``softplus(raw) * scale``; the scale is set from the
    #: training set's mean counts so the network starts near the right magnitude
    #: instead of spending its first thousand steps learning the units.
    output_scale: Tuple[float, ...] = ()

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        d["output_scale"] = [float(x) for x in self.output_scale]
        return d

    @staticmethod
    def from_dict(d: Mapping) -> "ProxyConfig":
        return ProxyConfig(
            n_features=int(d["n_features"]),
            n_sub_bins=int(d["n_sub_bins"]),
            n_host_bins=int(d["n_host_bins"]),
            hidden=tuple(int(h) for h in d.get("hidden", (128, 128))),
            dropout=float(d.get("dropout", 0.0)),
            output_scale=tuple(float(x) for x in d.get("output_scale", ())),
        )

    @property
    def n_outputs(self) -> int:
        return int(self.n_sub_bins) + 2 * int(self.n_host_bins)


class CatalogProxy(nn.Module):
    """Features -> ``(N, H, S)``, nonnegative by construction.

    Nonnegativity through ``softplus`` rather than a clamp: a clamp has zero
    gradient on the wrong side, so a member that starts with a bin pushed
    negative can never recover it, and the ensemble spread then reports
    confidence about a bin one member has silently stopped modelling.
    """

    def __init__(self, cfg: Optional[ProxyConfig] = None, *, seed: Optional[int] = None):
        super().__init__()
        self.cfg = cfg or ProxyConfig()
        if seed is not None:
            torch.manual_seed(int(seed))
        dims = [int(self.cfg.n_features), *[int(h) for h in self.cfg.hidden]]
        layers: List[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.SiLU()]
            if self.cfg.dropout > 0:
                layers.append(nn.Dropout(float(self.cfg.dropout)))
        layers.append(nn.Linear(dims[-1], self.cfg.n_outputs))
        self.net = nn.Sequential(*layers)
        self.seed = None if seed is None else int(seed)

        scale = list(self.cfg.output_scale) or [1.0] * self.cfg.n_outputs
        if len(scale) != self.cfg.n_outputs:
            raise ValueError(
                f"output_scale has {len(scale)} entries, expected {self.cfg.n_outputs}"
            )
        self.register_buffer("output_scale",
                             torch.as_tensor(scale, dtype=torch.float64))
        self.register_buffer("feat_mean", torch.zeros(self.cfg.n_features,
                                                      dtype=torch.float64))
        self.register_buffer("feat_std", torch.ones(self.cfg.n_features,
                                                    dtype=torch.float64))

    # -- feature scaling --------------------------------------------------- #
    def fit_standardizer(self, features: torch.Tensor | np.ndarray) -> "CatalogProxy":
        """Set the input mean/std from the **training** rows only.

        Stored as buffers so they travel with the checkpoint. A constant feature
        keeps ``std = 1``: dividing by its (zero) spread would emit inf and, at
        best, silently disable that coordinate.
        """
        x = torch.as_tensor(np.asarray(features, dtype=np.float64))
        if x.dim() != 2 or x.shape[1] != self.cfg.n_features:
            raise ValueError(
                f"expected (n, {self.cfg.n_features}) features, got {tuple(x.shape)}"
            )
        self.feat_mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False)
        self.feat_std = torch.where(std < 1e-12, torch.ones_like(std), std)
        return self

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feat_mean.to(x.dtype)) / self.feat_std.to(x.dtype)

    # -- forward ----------------------------------------------------------- #
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``{'n_sub': (B,J), 'n_host': (B,I), 'occ_numerator': (B,I)}``."""
        x = self._standardize(features.to(self.feat_mean.dtype))
        raw = self.net(x.to(next(self.net.parameters()).dtype))
        counts = F.softplus(raw.to(torch.float64)) * self.output_scale
        j, i = int(self.cfg.n_sub_bins), int(self.cfg.n_host_bins)
        return {
            "n_sub": counts[:, :j],
            "n_host": counts[:, j:j + i],
            "occ_numerator": counts[:, j + i:j + 2 * i],
        }

    def summary(self, features: torch.Tensor, volume_mpc3: torch.Tensor) -> TorchSummary:
        out = self.forward(features)
        return TorchSummary(
            n_sub=out["n_sub"], n_host=out["n_host"],
            occ_numerator=out["occ_numerator"],
            volume_mpc3=volume_mpc3.to(torch.float64),
        )

    # -- persistence ------------------------------------------------------- #
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.cfg.to_dict(), "seed": self.seed,
                    "state_dict": self.state_dict()}, p)
        return p

    @staticmethod
    def load(path: str | Path, map_location="cpu") -> "CatalogProxy":
        blob = torch.load(str(path), map_location=map_location, weights_only=False)
        m = CatalogProxy(ProxyConfig.from_dict(blob["config"]), seed=blob.get("seed"))
        m.load_state_dict(blob["state_dict"])
        return m


class ProxyEnsemble(nn.Module):
    """``M`` proxies, and the lower-confidence reward the actor maximises."""

    def __init__(self, members: Sequence[CatalogProxy]):
        super().__init__()
        if not members:
            raise ValueError("an ensemble needs at least one member")
        if len(members) < 2:
            # Not an error: a single member is useful for debugging. But the
            # spread is then identically zero, so `beta` silently stops doing
            # anything, and that has to be loud rather than inferred.
            print("WARNING: ProxyEnsemble with 1 member -- the uncertainty "
                  "penalty is identically zero and beta has no effect",
                  flush=True)
        self.members = nn.ModuleList(list(members))

    def __len__(self) -> int:
        return len(self.members)

    def freeze(self) -> "ProxyEnsemble":
        """Parameters fixed; the graph through them stays live.

        This is the configuration actor training needs and it is easy to get
        half right: ``eval()`` alone leaves the proxy trainable, and
        ``no_grad`` alone kills the gradient the actor exists to follow. The
        proxy must be *frozen but differentiable*.
        """
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def summaries(self, features: torch.Tensor,
                  volume_mpc3: torch.Tensor) -> List[TorchSummary]:
        return [m.summary(features, volume_mpc3) for m in self.members]

    def delta_rewards_all(
        self,
        reward: TorchRewardModel,
        features: torch.Tensor,
        box: TorchSummary,
        frozen_tile: TorchSummary,
        *,
        w_joint: float = 0.25,
        w_occ: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """``{key: (M, B)}`` per-member ``dR`` for every reward, in one pass.

        One pass rather than one per key: the members are the expensive part, and
        recomputing them to log ``dR_occ`` alongside ``dR_combined`` would triple
        the proxy cost of every step for numbers that come free.
        """
        rows: Dict[str, List[torch.Tensor]] = {}
        for m in self.members:
            pred = m.summary(features, box.volume_mpc3)
            d = reward.delta_reward_swap(box, frozen_tile, pred,
                                         w_joint=w_joint, w_occ=w_occ)
            for k, v in d.items():
                rows.setdefault(k, []).append(v)
        return {k: torch.stack(v, dim=0) for k, v in rows.items()}

    def delta_rewards(
        self,
        reward: TorchRewardModel,
        features: torch.Tensor,
        box: TorchSummary,
        frozen_tile: TorchSummary,
        *,
        w_joint: float = 0.25,
        w_occ: float = 1.0,
        key: str = "dR_combined",
    ) -> torch.Tensor:
        """``(M, B)`` per-member ``dR`` from swapping the predicted tile in."""
        return self.delta_rewards_all(
            reward, features, box, frozen_tile, w_joint=w_joint, w_occ=w_occ
        )[key]

    @staticmethod
    def q_safe(per_member: torch.Tensor, beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """``mean_m Q_m - beta * std_m Q_m``, with the pieces kept separately.

        ``unbiased=False`` because ``M`` is 3-5 by design: the unbiased estimator
        divides by ``M - 1`` and is very noisy there, and the quantity wanted is
        a penalty proportional to disagreement, not an unbiased variance.
        """
        mean = per_member.mean(dim=0)
        std = (per_member.std(dim=0, unbiased=False)
               if per_member.shape[0] > 1 else torch.zeros_like(mean))
        return {"mean": mean, "std": std, "q_safe": mean - float(beta) * std}

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.members):
            m.save(d / f"member_{i:02d}.pt")
        (d / "ensemble.json").write_text(json.dumps(
            {"n_members": len(self.members),
             "seeds": [m.seed for m in self.members]}, indent=2))
        return d

    @staticmethod
    def load(directory: str | Path, map_location="cpu") -> "ProxyEnsemble":
        d = Path(directory)
        files = sorted(d.glob("member_*.pt"))
        if not files:
            raise FileNotFoundError(f"no proxy members under {d}")
        return ProxyEnsemble([CatalogProxy.load(f, map_location) for f in files])


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def bin_weights_from_counts(
    targets: np.ndarray, *, cap: float = 8.0, floor: float = 0.25
) -> np.ndarray:
    """Per-output weights that lift rare bins, **capped**.

    ``w_i = clip(sqrt(mean_count / mean_count_i), floor, cap)``. The upper host
    bins hold ~30 hosts per box against ~1300 in the lowest, so an uncapped
    inverse weighting would put ~40x the loss on a bin whose *label* is itself
    the noisiest -- fitting its sampling noise in preference to everything else.
    The cap is the whole point of this function; it is not a safety margin.

    A bin that is **identically zero** over the whole training set gets the
    ``floor``, not the cap. "Rare" and "absent" are different: a rare bin has
    information the weighting is meant to protect, whereas an absent one has a
    constant target that any model fits exactly, so weighting it up spends
    capacity on nothing. Rounding it up to the cap would look like caution and
    would be the opposite.
    """
    t = np.asarray(targets, dtype=np.float64)
    if t.ndim != 2:
        raise ValueError(f"expected (n, D) targets, got {t.shape}")
    mean_per_bin = t.mean(axis=0)
    present = mean_per_bin > 0
    overall = float(np.mean(mean_per_bin[present])) if np.any(present) else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.sqrt(overall / np.where(present, mean_per_bin, np.inf))
    w = np.where(present & np.isfinite(w), w, float(floor))
    return np.clip(w, float(floor), float(cap))


def count_loss(
    predicted: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    *,
    weights: Optional[torch.Tensor] = None,
    huber_delta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Huber on ``log1p`` counts, per output block and combined.

    ``log1p`` rather than ``log``: tile counts are fractional and frequently
    exactly zero (a tile with no 1e14 host), and ``log`` of that is not a number.
    """
    keys = ("n_sub", "n_host", "occ_numerator")
    parts: Dict[str, torch.Tensor] = {}
    stacked_p, stacked_t = [], []
    for k in keys:
        p = torch.log1p(predicted[k].clamp_min(0.0).to(torch.float64))
        t = torch.log1p(target[k].clamp_min(0.0).to(torch.float64))
        parts[f"loss_{k}"] = F.huber_loss(p, t, delta=float(huber_delta))
        stacked_p.append(p)
        stacked_t.append(t)
    p = torch.cat(stacked_p, dim=1)
    t = torch.cat(stacked_t, dim=1)
    per_elem = F.huber_loss(p, t, delta=float(huber_delta), reduction="none")
    if weights is not None:
        w = weights.to(per_elem.dtype).to(per_elem.device).reshape(1, -1)
        if w.shape[1] != per_elem.shape[1]:
            raise ValueError(
                f"weights has {w.shape[1]} entries, expected {per_elem.shape[1]}"
            )
        parts["loss"] = (per_elem * w).sum() / (w.sum() * per_elem.shape[0])
    else:
        parts["loss"] = per_elem.mean()
    return parts


def make_within_tile_pairs(
    box: Sequence[str],
    tile_id: Sequence[int],
    target: Sequence[float],
    *,
    max_pairs_per_group: int = 32,
    min_margin: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """``(P, 2)`` index pairs, ``pair[:, 0]`` the **better** (higher-reward) row.

    Grouping is by ``(box, tile_id)`` and nothing else. Pairing across tiles
    would let the proxy score well by learning which tiles are host-rich -- true,
    useless, and not something the actor can act on. Pairs whose true reward gap
    is below ``min_margin`` are dropped as ties: with a small edit most pairs
    carry no signal and training on them is pure label noise.
    """
    rng = rng or np.random.default_rng(0)
    t = np.asarray(target, dtype=np.float64)
    groups: Dict[Tuple[str, int], List[int]] = {}
    for i, (b, j) in enumerate(zip(box, tile_id)):
        groups.setdefault((str(b), int(j)), []).append(i)

    pairs: List[Tuple[int, int]] = []
    for key in sorted(groups):
        idx = np.asarray(groups[key])
        if idx.size < 2:
            continue
        cand = [(a, b) for m, a in enumerate(idx) for b in idx[m + 1:]]
        rng.shuffle(cand)
        kept = 0
        for a, b in cand:
            if not (np.isfinite(t[a]) and np.isfinite(t[b])):
                continue
            if abs(t[a] - t[b]) < float(min_margin):
                continue
            pairs.append((int(a), int(b)) if t[a] > t[b] else (int(b), int(a)))
            kept += 1
            if kept >= int(max_pairs_per_group):
                break
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def split_indices_by_box(
    boxes: Sequence[str],
    train_boxes: Sequence[str],
    val_boxes: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Row indices for a **box-level** split, with the overlap check up front.

    Crops from one box share its large-scale modes, so a row-level split leaks
    the answer: two tiles of the same box are not independent samples, and a
    proxy validated across such a split reports a generalisation it does not
    have. The box lists are therefore required to be disjoint, and a row whose
    box is in neither list is dropped loudly rather than silently.
    """
    tr, va = {str(b) for b in train_boxes}, {str(b) for b in val_boxes}
    overlap = sorted(tr & va)
    if overlap:
        raise ValueError(
            f"boxes {overlap} are in both the train and validation split; a "
            "box-level split is the only thing standing between this proxy and "
            "a leaked validation score"
        )
    rows = [str(b) for b in boxes]
    unknown = sorted({b for b in rows if b not in tr and b not in va})
    train_idx = np.asarray([i for i, b in enumerate(rows) if b in tr], dtype=np.int64)
    val_idx = np.asarray([i for i, b in enumerate(rows) if b in va], dtype=np.int64)
    return {
        "train": train_idx,
        "val": val_idx,
        "dropped_boxes": np.asarray(unknown, dtype=object),
    }


def pairwise_ranking_loss(better: torch.Tensor, worse: torch.Tensor) -> torch.Tensor:
    """RankNet loss with the convention **higher score = better**.

    The opposite of :func:`cosmo_sr.tts.verifier.pairwise_ranking_loss`, which
    scores candidates by an error (lower = better). Rewards are rewards, so the
    sign is flipped here rather than negating rewards at the call site, where the
    convention would be invisible.
    """
    return F.softplus(worse - better).mean()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(np.float64)
    rb = np.argsort(np.argsort(b[ok])).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")
