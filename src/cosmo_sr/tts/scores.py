"""Composite quality scores and the two oracles.

Two very different questions get asked of a candidate set, and conflating them
is the main way a test-time-scaling study fools itself:

``phase`` oracle
    "Which candidate has the true small-scale *phases*?" -- uses ``r(k)`` and
    ``T(k)`` against the held-out HR box. This is the upper bound on what any
    selector could do, but much of it is **not inferable from LR**: the exact
    realisation of unresolved modes is not a function of the input. Reported as
    a ceiling, never as a target for the verifier.

``statistical`` oracle
    "Which candidate has the right *statistics*?" -- density power, density PDF,
    equilateral and squeezed bispectra, velocity power and divergence PDF. These
    are the properties a test-time selector can plausibly learn to judge, and
    they are what the science actually needs.

Composite scores are **z-normalised per component using validation-set
statistics** before averaging. Without that, ``density_pdf_error`` (an L1
distance, order 0.1) and ``bispectrum_squeezed_error`` (a relative error that can
be order 10) would enter the sum with a 100x weight ratio decided by nothing but
their units.

Convention throughout: **lower is better** for a composite score.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "METRIC_DIRECTION",
    "PHASE_ORACLE_COMPONENTS",
    "STATISTICAL_ORACLE_COMPONENTS",
    "ScoreNormalizer",
    "composite_score",
    "oracle_selection",
    "regret",
]

#: ``+1`` = larger is better, ``-1`` = smaller is better. Metrics not listed are
#: treated as smaller-is-better errors.
METRIC_DIRECTION: Dict[str, int] = {
    "disp_rk_low": +1, "disp_rk_transition": +1, "disp_rk_high": +1,
    "vel_rk_low": +1, "vel_rk_transition": +1, "vel_rk_high": +1,
    "density_rk_low": +1, "density_rk_transition": +1, "density_rk_high": +1,
}

#: Realisation-specific phase/amplitude agreement with the true HR box.
PHASE_ORACLE_COMPONENTS: Tuple[str, ...] = (
    "disp_rk_transition", "disp_rk_high",
    "density_rk_transition", "density_rk_high",
    "vel_rk_transition", "vel_rk_high",
    "disp_Tk_error_transition", "disp_Tk_error_high",
    "density_Tk_error_transition", "density_Tk_error_high",
)

#: Distributional / higher-order agreement, deliberately free of high-k phase.
STATISTICAL_ORACLE_COMPONENTS: Tuple[str, ...] = (
    "density_power_error",
    "density_pdf_error",
    "density_sigma_ratio_error",
    "bispectrum_equilateral_error",
    "bispectrum_squeezed_error",
    "velocity_power_error",
    "velocity_divergence_pdf_error",
)


def derive_metrics(row: Mapping[str, float]) -> Dict[str, float]:
    """Add the ``*_error`` forms of ratio-like metrics (target 1 -> target 0)."""
    out = dict(row)
    if "density_sigma_ratio" in out:
        out["density_sigma_ratio_error"] = abs(float(out["density_sigma_ratio"]) - 1.0)
    return out


@dataclass
class ScoreNormalizer:
    """Per-metric ``(mean, std)`` estimated on a validation set.

    Fit on **validation** rows only, then applied unchanged to test rows: fitting
    on the rows being scored would let the normalisation absorb exactly the
    between-candidate spread the study is trying to measure.
    """

    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, rows: Iterable[Mapping[str, float]],
            metrics: Optional[Sequence[str]] = None) -> "ScoreNormalizer":
        rows = [derive_metrics(r) for r in rows]
        if not rows:
            raise ValueError("cannot fit a ScoreNormalizer on zero rows")
        keys = list(metrics) if metrics is not None else sorted(
            {k for r in rows for k, v in r.items() if isinstance(v, (int, float))}
        )
        mean, std = {}, {}
        for k in keys:
            vals = np.asarray([float(r[k]) for r in rows if k in r and np.isfinite(r[k])])
            if vals.size == 0:
                continue
            mean[k] = float(vals.mean())
            s = float(vals.std())
            # A component with no spread carries no selection signal; a floor of
            # 0 would turn it into +-inf and let it dominate the composite.
            std[k] = s if s > 1e-12 else 1.0
        return cls(mean=mean, std=std)

    def z(self, row: Mapping[str, float], key: str) -> float:
        """Signed z-score, oriented so that **negative is better**."""
        if key not in row or key not in self.mean:
            return float("nan")
        value = float(row[key])
        if not np.isfinite(value):
            return float("nan")
        zz = (value - self.mean[key]) / self.std[key]
        return -zz if METRIC_DIRECTION.get(key, -1) > 0 else zz

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"mean": self.mean, "std": self.std}, indent=2))

    @classmethod
    def load(cls, path) -> "ScoreNormalizer":
        d = json.loads(Path(path).read_text())
        return cls(mean=d["mean"], std=d["std"])


def composite_score(
    row: Mapping[str, float],
    components: Sequence[str],
    normalizer: ScoreNormalizer,
) -> float:
    """Mean normalised component score; lower is better. NaN if nothing usable."""
    row = derive_metrics(row)
    zs = [normalizer.z(row, c) for c in components]
    zs = [z for z in zs if np.isfinite(z)]
    return float(np.mean(zs)) if zs else float("nan")


def oracle_selection(
    rows: Sequence[Mapping[str, float]],
    components: Sequence[str],
    normalizer: ScoreNormalizer,
) -> Tuple[int, float, List[float]]:
    """``(argmin index, best score, all scores)`` over a candidate set."""
    scores = [composite_score(r, components, normalizer) for r in rows]
    finite = [(i, s) for i, s in enumerate(scores) if np.isfinite(s)]
    if not finite:
        return 0, float("nan"), scores
    idx, best = min(finite, key=lambda t: t[1])
    return idx, best, scores


def regret(selected_score: float, oracle_score: float) -> float:
    """How much worse the selection is than the best available candidate."""
    return float(selected_score - oracle_score)
