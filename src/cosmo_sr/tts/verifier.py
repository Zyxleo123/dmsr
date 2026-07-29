"""Stage 2: learned candidate verifiers, trained by pairwise ranking.

A verifier answers "of these two candidates for the *same* LR input, which is
better?" -- never "how good is this candidate in absolute terms". Three reasons
the problem is posed that way:

* Absolute quality is dominated by which box we are looking at (cosmic variance),
  which the verifier can neither change nor predict; the *within-input* ordering
  is the only part selection can act on.
* A ranking loss needs no calibration between boxes, so a single model works
  across boxes whose error scales differ by an order of magnitude.
* It matches the decision the selector actually makes at test time (an argmax
  over candidates).

The training target is the **statistical** oracle score, not the phase oracle:
the exact realisation of unresolved high-k modes is not a function of the LR
input, so training to predict it is training to fit noise. See
:mod:`cosmo_sr.tts.scores`.

Splits are **by simulation box**, never by crop -- crops from one box share its
large-scale modes, so a crop-level split leaks the answer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .features import FEATURE_KEYS, feature_matrix

__all__ = [
    "FeatureRanker",
    "PatchVerifier",
    "Standardizer",
    "make_pairs",
    "pairwise_ranking_loss",
    "rank_metrics",
    "train_feature_ranker",
]


# --------------------------------------------------------------------------- #
# Feature scaling
# --------------------------------------------------------------------------- #
@dataclass
class Standardizer:
    """Per-feature mean/std fitted on the **training** boxes only."""

    mean: np.ndarray
    std: np.ndarray
    keys: Tuple[str, ...] = FEATURE_KEYS

    @classmethod
    def fit(cls, x: np.ndarray, keys: Sequence[str] = FEATURE_KEYS) -> "Standardizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-12] = 1.0      # a constant feature carries no ordering signal
        return cls(mean=mean, std=std, keys=tuple(keys))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(
            {"mean": self.mean.tolist(), "std": self.std.tolist(), "keys": list(self.keys)}
        ))

    @classmethod
    def load(cls, path) -> "Standardizer":
        d = json.loads(Path(path).read_text())
        return cls(mean=np.asarray(d["mean"]), std=np.asarray(d["std"]),
                   keys=tuple(d["keys"]))


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class FeatureRanker(nn.Module):
    """Linear (``hidden=()``) or small-MLP scorer over summary features.

    Deliberately the first thing tried. With a few dozen boxes x K candidates
    there are thousands of *pairs* but only a handful of independent boxes, so a
    high-capacity model would memorise box identity long before it learned
    anything transferable.

    Output convention matches :mod:`cosmo_sr.tts.scores`: **lower is better**.
    """

    def __init__(self, n_features: int = len(FEATURE_KEYS),
                 hidden: Sequence[int] = (32,), dropout: float = 0.0):
        super().__init__()
        dims = [int(n_features), *[int(h) for h in hidden]]
        layers: List[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)
        self.n_features = int(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float()).squeeze(-1)


class PatchVerifier(nn.Module):
    """Conditional 3-D patch scorer: ``(LR context, SR displacement/velocity, CIC density)``.

    Only worth reaching for when the summary features leave oracle gain on the
    table -- it costs a forward pass over the field instead of a dot product with
    16 numbers. The LR input is trilinearly upsampled to the SR grid and
    concatenated, so the score is *conditional* on the input rather than a
    generic realism score.
    """

    def __init__(self, lr_chan: int = 6, sr_chan: int = 6, width: int = 32, depth: int = 3):
        super().__init__()
        in_chan = lr_chan + sr_chan + 1          # + CIC density
        blocks: List[nn.Module] = []
        c = in_chan
        for i in range(depth):
            nxt = width * (2 ** i)
            blocks += [nn.Conv3d(c, nxt, 3, stride=2, padding=1),
                       nn.GroupNorm(min(8, nxt), nxt),
                       nn.LeakyReLU(0.2, inplace=True)]
            c = nxt
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(c, 1)

    def forward(self, lr_up: torch.Tensor, sr: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        x = torch.cat([lr_up, sr, rho], dim=1)
        h = self.body(x).mean(dim=(2, 3, 4))
        return self.head(h).squeeze(-1)


# --------------------------------------------------------------------------- #
# Pairwise training
# --------------------------------------------------------------------------- #
def make_pairs(
    groups: Sequence[Sequence[int]],
    targets: Sequence[float],
    max_pairs_per_group: int = 64,
    min_margin: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """``(n_pairs, 2)`` index pairs drawn **within** groups (one group = one LR input).

    ``groups`` lists the row indices belonging to each input. Pairs are ordered so
    that ``pair[:, 0]`` is the better candidate (lower target). Pairs whose target
    gap is under ``min_margin`` are dropped: with SR2's small stochastic spread a
    large fraction of pairs are effectively ties, and training on them injects
    label noise with no signal.
    """
    rng = rng or np.random.default_rng(0)
    t = np.asarray(targets, dtype=float)
    pairs: List[Tuple[int, int]] = []
    for idx in groups:
        idx = np.asarray(idx)
        if idx.size < 2:
            continue
        cand = [(a, b) for i, a in enumerate(idx) for b in idx[i + 1:]]
        rng.shuffle(cand)
        kept = 0
        for a, b in cand:
            if not (np.isfinite(t[a]) and np.isfinite(t[b])):
                continue
            if abs(t[a] - t[b]) < min_margin:
                continue
            pairs.append((a, b) if t[a] < t[b] else (b, a))
            kept += 1
            if kept >= max_pairs_per_group:
                break
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def pairwise_ranking_loss(better: torch.Tensor, worse: torch.Tensor) -> torch.Tensor:
    """RankNet/logistic loss on the score gap; lower score = better candidate."""
    return torch.nn.functional.softplus(better - worse).mean()


def train_feature_ranker(
    x_train: np.ndarray,
    groups_train: Sequence[Sequence[int]],
    target_train: Sequence[float],
    x_val: Optional[np.ndarray] = None,
    groups_val: Optional[Sequence[Sequence[int]]] = None,
    target_val: Optional[Sequence[float]] = None,
    hidden: Sequence[int] = (32,),
    epochs: int = 300,
    lr: float = 3e-3,
    weight_decay: float = 1e-3,
    max_pairs_per_group: int = 64,
    min_margin: float = 0.0,
    seed: int = 0,
    device=None,
    verbose: bool = False,
) -> Tuple[FeatureRanker, Dict[str, List[float]]]:
    """Fit a :class:`FeatureRanker`, early-stopping on validation pair accuracy.

    Validation groups come from *held-out boxes*; the returned model is the one
    with the best validation accuracy, not the last epoch, because with this many
    pairs per box the training loss keeps falling long after generalisation stops.
    """
    device = device or torch.device("cpu")
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))

    model = FeatureRanker(x_train.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    xt = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    pairs = make_pairs(groups_train, target_train, max_pairs_per_group, min_margin, rng)
    if pairs.size == 0:
        raise ValueError("no training pairs survived; lower min_margin or add candidates")
    pt = torch.as_tensor(pairs, device=device)

    have_val = x_val is not None and groups_val is not None and target_val is not None
    if have_val:
        xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
        pv = torch.as_tensor(
            make_pairs(groups_val, target_val, max_pairs_per_group, min_margin, rng),
            device=device,
        )

    history: Dict[str, List[float]] = {"loss": [], "train_acc": [], "val_acc": []}
    best_state, best_acc = {k: v.detach().clone() for k, v in model.state_dict().items()}, -1.0
    for epoch in range(int(epochs)):
        model.train()
        s = model(xt)
        loss = pairwise_ranking_loss(s[pt[:, 0]], s[pt[:, 1]])
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            s = model(xt)
            tr_acc = float((s[pt[:, 0]] < s[pt[:, 1]]).float().mean())
            va_acc = float("nan")
            if have_val and pv.numel():
                sv = model(xv)
                va_acc = float((sv[pv[:, 0]] < sv[pv[:, 1]]).float().mean())
        history["loss"].append(float(loss.detach()))
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        score = va_acc if have_val and np.isfinite(va_acc) else tr_acc
        if score > best_acc:
            best_acc = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"  epoch {epoch:4d} loss {history['loss'][-1]:.4f} "
                  f"train_acc {tr_acc:.3f} val_acc {va_acc:.3f}", flush=True)

    model.load_state_dict(best_state)
    history["best_acc"] = [best_acc]
    return model, history


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def rank_metrics(
    predicted: Sequence[float],
    target: Sequence[float],
    groups: Sequence[Sequence[int]],
) -> Dict[str, float]:
    """Within-input Spearman, pairwise accuracy and selection regret.

    All three are computed *within* each group and then averaged over groups: a
    global correlation would be dominated by between-box differences the selector
    is not being asked to predict.
    """
    p = np.asarray(predicted, dtype=float)
    t = np.asarray(target, dtype=float)
    rho, acc, reg, top1 = [], [], [], []
    for idx in groups:
        idx = np.asarray(idx)
        if idx.size < 2:
            continue
        pi, ti = p[idx], t[idx]
        ok = np.isfinite(pi) & np.isfinite(ti)
        if ok.sum() < 2:
            continue
        pi, ti = pi[ok], ti[ok]
        rho.append(_spearman(pi, ti))
        agree = [
            (pi[a] < pi[b]) == (ti[a] < ti[b])
            for a in range(len(pi)) for b in range(a + 1, len(pi)) if ti[a] != ti[b]
        ]
        if agree:
            acc.append(float(np.mean(agree)))
        chosen = int(np.argmin(pi))
        reg.append(float(ti[chosen] - ti.min()))
        top1.append(float(chosen == int(np.argmin(ti))))
    return {
        "spearman": float(np.nanmean(rho)) if rho else float("nan"),
        "pairwise_accuracy": float(np.mean(acc)) if acc else float("nan"),
        "regret": float(np.mean(reg)) if reg else float("nan"),
        "top1_rate": float(np.mean(top1)) if top1 else float("nan"),
        "n_groups": len(rho),
    }


def score_rows(model: FeatureRanker, standardizer: Standardizer,
               rows: Iterable[Dict[str, float]]) -> np.ndarray:
    """Verifier score (lower = better) for a sequence of feature rows."""
    rows = list(rows)
    x = standardizer(feature_matrix(rows, standardizer.keys))
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
