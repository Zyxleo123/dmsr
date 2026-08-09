"""The three baselines every proxy arm has to beat before it means anything.

A within-tile Spearman of 0.55 sounds like a result until you ask what a model
that has learned *nothing* would score. These are the three "nothing"s worth
naming, and the gate compares against all of them:

``zero_change``
    Predict that the candidate tile is exactly the frozen tile. This is the
    null hypothesis of the entire line -- "fine-tuning changes nothing you can
    see" -- and it is a very strong count predictor, because most tiles really
    are almost unchanged. Its ``dR`` is identically zero, so it cannot rank at
    all, which is precisely the distinction the ranking metrics exist to draw.
``train_mean``
    Predict the training set's mean counts for every tile. Ignores the features
    entirely; anything that does not beat it has not used them.
``linear``
    Ordinary least squares from the arm's features to ``log1p`` counts. The
    honest question about a two-layer MLP is not "is it better than nothing" but
    "is it better than a line through the same features", and on 16 outputs from
    26 or 44 inputs a line is a serious competitor.

All three predict in the same space as :class:`cosmo_sr.reward.catalog_proxy.CatalogProxy`
-- the three count vectors of a :class:`cosmo_sr.reward.torch_reward.TorchSummary`
-- so every metric in :mod:`_proxy_data` applies to them unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from cosmo_sr.reward.torch_reward import TorchSummary

from _proxy_data import COUNT_KEYS

BASELINES = ("zero_change", "train_mean", "linear")


class Baseline:
    """One fitted baseline, callable like a proxy member.

    ``summary(features, volume, frozen)`` mirrors ``CatalogProxy.summary`` with
    one extra argument, because ``zero_change`` is a function of the frozen
    reference rather than of the features. Passing it to all three keeps the
    call site free of per-baseline branching.
    """

    def __init__(self, kind: str, *, shapes: Sequence[int],
                 mean: Optional[np.ndarray] = None,
                 coef: Optional[np.ndarray] = None,
                 intercept: Optional[np.ndarray] = None):
        if kind not in BASELINES:
            raise ValueError(f"unknown baseline {kind!r}; expected {list(BASELINES)}")
        self.kind = kind
        self.shapes = [int(s) for s in shapes]
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.coef = None if coef is None else np.asarray(coef, dtype=np.float64)
        self.intercept = (None if intercept is None
                          else np.asarray(intercept, dtype=np.float64))

    # -- prediction -------------------------------------------------------- #
    def _split(self, flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        j, i = self.shapes[0], self.shapes[1]
        return {"n_sub": flat[:, :j], "n_host": flat[:, j:j + i],
                "occ_numerator": flat[:, j + i:j + 2 * i]}

    def summary(self, features: torch.Tensor, volume_mpc3: torch.Tensor,
                frozen: TorchSummary) -> TorchSummary:
        if self.kind == "zero_change":
            return TorchSummary(frozen.n_sub, frozen.n_host, frozen.occ_numerator,
                                volume_mpc3.to(torch.float64))
        x = features.to(torch.float64)
        if self.kind == "train_mean":
            flat = torch.as_tensor(self.mean, dtype=torch.float64).reshape(1, -1) \
                .expand(x.shape[0], -1)
        else:
            flat = (x @ torch.as_tensor(self.coef.T, dtype=torch.float64)
                    + torch.as_tensor(self.intercept, dtype=torch.float64))
        # Fitted in log1p space; counts are non-negative by definition and a
        # negative prediction here would make `log1p` of it undefined downstream.
        counts = torch.expm1(flat.clamp(min=0.0, max=30.0))
        out = self._split(counts)
        return TorchSummary(out["n_sub"], out["n_host"], out["occ_numerator"],
                            volume_mpc3.to(torch.float64))

    # -- persistence ------------------------------------------------------- #
    def to_dict(self) -> Dict:
        return {
            "kind": self.kind, "shapes": self.shapes,
            "mean": None if self.mean is None else self.mean.tolist(),
            "coef": None if self.coef is None else self.coef.tolist(),
            "intercept": None if self.intercept is None else self.intercept.tolist(),
        }

    @staticmethod
    def from_dict(d: Mapping) -> "Baseline":
        return Baseline(str(d["kind"]), shapes=d["shapes"],
                        mean=d.get("mean"), coef=d.get("coef"),
                        intercept=d.get("intercept"))


def fit_baselines(features: np.ndarray, targets: Mapping[str, np.ndarray], *,
                  row_weight: Optional[np.ndarray] = None) -> Dict[str, Baseline]:
    """Fit all three on the training rows, in ``log1p`` count space.

    ``row_weight`` is the same weighting the proxy is trained under. Fitting the
    baselines on a differently-weighted sample would make the comparison a
    comparison of sample weights.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.concatenate([np.asarray(targets[k], dtype=np.float64)
                        for k in COUNT_KEYS], axis=1)
    ly = np.log1p(np.clip(y, 0.0, None))
    shapes = [np.asarray(targets[k]).shape[1] for k in COUNT_KEYS]
    w = (np.ones(x.shape[0]) if row_weight is None
         else np.asarray(row_weight, dtype=np.float64))
    w = w / max(float(w.sum()), 1e-12)

    mean = (ly * w[:, None]).sum(axis=0)
    design = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    sw = np.sqrt(w)[:, None]
    sol, *_ = np.linalg.lstsq(design * sw, ly * sw, rcond=None)
    return {
        "zero_change": Baseline("zero_change", shapes=shapes),
        "train_mean": Baseline("train_mean", shapes=shapes, mean=mean),
        "linear": Baseline("linear", shapes=shapes,
                           coef=sol[:-1].T, intercept=sol[-1]),
    }


def save_baselines(path: str | Path, arms: Mapping[str, Mapping[str, Baseline]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(
        {arm: {k: b.to_dict() for k, b in d.items()} for arm, d in arms.items()},
        indent=2))
    return p


def load_baselines(path: str | Path) -> Dict[str, Dict[str, Baseline]]:
    blob = json.loads(Path(path).read_text())
    return {arm: {k: Baseline.from_dict(v) for k, v in d.items()}
            for arm, d in blob.items()}


__all__: List[str] = [
    "BASELINES", "Baseline", "fit_baselines", "load_baselines", "save_baselines",
]
