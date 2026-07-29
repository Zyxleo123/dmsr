"""Best-of-K scaling curves and box-level paired bootstrap intervals.

Two statistical points drive the design:

1. **The unit of independence is the simulation box, not the candidate.**
   Candidates from one box share its cosmic variance, so bootstrapping over
   candidates would give intervals that are far too tight. Every interval here
   resamples *boxes* with replacement.
2. **Best-of-K must be averaged over subsets, not read off one ordering.**
   With ``K_max`` candidates in hand, "best of 4" is estimated by drawing many
   random size-4 subsets and averaging the selected candidate's quality --
   otherwise the answer depends on the arbitrary order the seeds were run in.
   Paired comparisons reuse the *same* subsets across methods, so the difference
   between two selectors is not contaminated by subset noise.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "best_of_k",
    "bootstrap_ci",
    "paired_bootstrap",
    "subset_draws",
]


def subset_draws(
    k: int, n_candidates: int, n_repeats: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_repeats, k)`` index matrix of size-``k`` subsets drawn without replacement.

    When ``k == n_candidates`` there is only one subset, so a single row is
    returned -- repeating it would only inflate the apparent sample size.
    """
    k = int(min(k, n_candidates))
    if k >= n_candidates:
        return np.arange(n_candidates)[None, :]
    return np.stack([rng.choice(n_candidates, size=k, replace=False) for _ in range(n_repeats)])


def best_of_k(
    values: Sequence[float],
    selector: Sequence[float],
    k: int,
    n_repeats: int = 200,
    rng: Optional[np.random.Generator] = None,
    draws: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """Expected quality when a selector picks the best of ``k`` candidates.

    ``values`` is the quality being *reported* (lower is better) and ``selector``
    is the score the method actually minimises. Passing ``selector = values``
    gives the oracle; a constant ``selector`` gives random choice; a verifier
    score gives the learned selector. Keeping the two separate is what makes
    "the selector optimised its own target while damaging the real metric"
    visible instead of hidden.

    Returns ``(mean over subsets, per-subset selected values)``.
    """
    v = np.asarray(values, dtype=float)
    s = np.asarray(selector, dtype=float)
    if v.shape != s.shape:
        raise ValueError(f"values {v.shape} and selector {s.shape} must match")
    rng = rng or np.random.default_rng(0)
    if draws is None:
        draws = subset_draws(k, len(v), n_repeats, rng)
    # NaN selector scores must never win an argmin.
    s = np.where(np.isfinite(s), s, np.inf)
    picked = draws[np.arange(len(draws)), np.argmin(s[draws], axis=1)]
    chosen = v[picked]
    return float(np.nanmean(chosen)), chosen


def bootstrap_ci(
    per_box: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Percentile bootstrap over boxes for a per-box statistic."""
    x = np.asarray([v for v in per_box if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boxes": 0}
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, x.size, size=(int(n_boot), x.size))
    means = x[idx].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n_boxes": int(x.size),
    }


def paired_bootstrap(
    a_per_box: Sequence[float],
    b_per_box: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """CI for ``mean(a - b)`` resampling boxes, keeping each box's pair together.

    ``significant`` is ``True`` when the interval excludes zero -- the decision
    gate used throughout the test-time-scaling stages.
    """
    a = np.asarray(a_per_box, dtype=float)
    b = np.asarray(b_per_box, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    out = bootstrap_ci(d, n_boot=n_boot, alpha=alpha, rng=rng)
    denom = float(np.abs(b[keep]).mean()) if keep.any() else 0.0
    out["relative"] = float(out["mean"] / denom) if denom > 1e-30 else float("nan")
    out["significant"] = bool(np.isfinite(out["lo"]) and (out["lo"] > 0 or out["hi"] < 0))
    return out


def scaling_curve(
    rows_by_box: Mapping[str, Sequence[Mapping[str, float]]],
    value_key: str,
    selector_fn: Callable[[Mapping[str, float]], float],
    k_values: Sequence[int],
    n_repeats: int = 200,
    seed: int = 0,
) -> Dict[int, Dict[str, float]]:
    """Quality vs ``K`` for one selector, with a box-bootstrap CI at each ``K``.

    ``selector_fn`` maps a candidate row to the score the method minimises.
    """
    rng = np.random.default_rng(seed)
    out: Dict[int, Dict[str, float]] = {}
    for k in k_values:
        per_box: List[float] = []
        for _box, rows in sorted(rows_by_box.items()):
            values = [float(r.get(value_key, np.nan)) for r in rows]
            sel = [float(selector_fn(r)) for r in rows]
            mean, _ = best_of_k(values, sel, k, n_repeats=n_repeats, rng=rng)
            per_box.append(mean)
        out[int(k)] = bootstrap_ci(per_box, rng=rng)
        out[int(k)]["per_box"] = per_box  # type: ignore[assignment]
    return out
