"""Checkpoint gates for **direct** SR2 fine-tuning: relative, not absolute.

The residual line's density threshold is
``density_power_error_max = 0.03751``, and it is the right number *for what it
was built to do*: it is 1.5x the worst frozen-SR2-vs-HR error over the eight
training boxes, i.e. "a candidate is infeasible when it is materially worse than
the baseline it is supposed to improve". Reused here it would be far too
permissive, because it admits every checkpoint that degrades density by up to
50% of SR2's own error while calling it a pass. "No degradation" and "no more
than 1.5x the baseline's error" are different claims, and this line has to make
the first one.

So the gate here is **paired and relative**:

    degradation(box, seed) = err_theta(box, seed) - err_0(box, seed)

measured on the *same box and the same seed*, against the frozen generator. That
cancels the box-to-box scatter (which is large and has nothing to do with the
fine-tune) and leaves the quantity the decision is actually about.

Calibrating the tolerance
-------------------------
How much degradation is indistinguishable from noise? Exactly the amount the
frozen generator already varies by when only its noise seed changes:
:func:`calibrate_from_frozen_seeds` measures the frozen per-box spread over
seeds and proposes ``mean`` and ``single_box`` tolerances from it. A checkpoint
inside that band has not been shown to hurt density; one outside it has.
Nothing here invents a number, and :attr:`RelativeDensityGate.calibrated`
carries whether the numbers came from a measurement, exactly as
``ConstraintSet.calibrated`` does for the residual line.

This module deliberately does **not** touch
:mod:`cosmo_sr.reward.constraints`. The residual pipeline's severities stay as
they are; the stricter rules live in ``configs/reward/sr2_direct_finetune.yaml``
and are read here.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "DirectGateResult",
    "RelativeDensityGate",
    "calibrate_from_frozen_seeds",
    "check_direct_gates",
    "load_direct_gate",
    "paired_degradation",
]


@dataclass(frozen=True)
class RelativeDensityGate:
    """Thresholds for one candidate checkpoint, against frozen SR2.

    ``None`` disables a bound (it is still measured and reported).
    """

    #: Mean over boxes of ``err_theta - err_0``. The headline "did density get
    #: worse on average" test.
    mean_degradation_max: Optional[float] = None
    #: Worst single box. A checkpoint that is fine on seven boxes and ruins the
    #: eighth has not preserved density; averaging would hide it.
    single_box_degradation_max: Optional[float] = None
    #: ``||A(Psi_theta) - A(Psi_0)|| / ||A(Psi_0)||``. Kept blocking: rewriting
    #: the LR-visible scales means the output is no longer a refinement of the
    #: thing that was validated, whatever the catalog says.
    low_k_change_max: Optional[float] = 0.139595
    #: Structural spread across noise seeds (displacement / density / soft-peak,
    #: NOT six-channel). A floor: collapse to a deterministic map scores well on
    #: every other line here and is a failure.
    d_struct_min: Optional[float] = 0.05
    #: Absolute backstop, inherited from the residual line's calibration. A
    #: checkpoint may not be worse than this even if it happened to start good.
    density_power_error_max: Optional[float] = 0.03751
    calibrated: bool = False
    meta: Mapping = dc_field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "mean_degradation_max": self.mean_degradation_max,
            "single_box_degradation_max": self.single_box_degradation_max,
            "low_k_change_max": self.low_k_change_max,
            "d_struct_min": self.d_struct_min,
            "density_power_error_max": self.density_power_error_max,
            "calibrated": bool(self.calibrated),
            "meta": dict(self.meta),
        }


def load_direct_gate(cfg: Mapping) -> RelativeDensityGate:
    """Build the gate from the ``gates`` block of ``sr2_direct_finetune.yaml``."""
    c = dict(cfg or {})

    def g(key, default):
        v = c.get(key, default)
        return None if v is None else float(v)

    return RelativeDensityGate(
        mean_degradation_max=g("mean_degradation_max", None),
        single_box_degradation_max=g("single_box_degradation_max", None),
        low_k_change_max=g("low_k_change_max", 0.139595),
        d_struct_min=g("d_struct_min", 0.05),
        density_power_error_max=g("density_power_error_max", 0.03751),
        calibrated=bool(c.get("calibrated", False)),
        meta=dict(c.get("meta", {})),
    )


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibrate_from_frozen_seeds(
    rows: Sequence[Mapping],
    *,
    mean_margin: float = 1.0,
    single_margin: float = 2.0,
    key: str = "density_power_error",
) -> Dict:
    """Propose tolerances from the frozen generator's own seed-to-seed spread.

    ``rows`` are per ``(box, seed)`` measurements of the **frozen** generator,
    each with ``box``, ``seed`` and ``key``. For every box the spread over seeds
    is the amount density power moves for reasons that are not the weights;
    a fine-tune that stays inside it has not been shown to have done anything to
    density.

    * ``mean_degradation_max = mean_margin * (mean over boxes of the per-box
      standard deviation over seeds)``;
    * ``single_box_degradation_max = single_margin * (max over boxes of the
      per-box peak-to-peak range over seeds)``.

    The two use different statistics on purpose. The mean bound is a statement
    about a *mean*, so it uses a standard deviation; the single-box bound has to
    survive the worst individual comparison, so it uses the observed range.

    Raises rather than guessing when a box has fewer than two seeds: a "spread"
    from one sample is zero, and a zero tolerance would reject every checkpoint
    including the frozen one.
    """
    by_box: Dict[str, List[float]] = {}
    for r in rows:
        v = float(r[key])
        if not np.isfinite(v):
            raise ValueError(f"non-finite {key} for box {r.get('box')} seed {r.get('seed')}")
        by_box.setdefault(str(r["box"]), []).append(v)
    thin = {b: len(v) for b, v in by_box.items() if len(v) < 2}
    if not by_box or thin:
        raise ValueError(
            "seed-to-seed calibration needs >= 2 seeds per box; got "
            f"{thin or 'no boxes at all'}"
        )
    stds = {b: float(np.std(v, ddof=1)) for b, v in by_box.items()}
    ranges = {b: float(np.max(v) - np.min(v)) for b, v in by_box.items()}
    mean_std = float(np.mean(list(stds.values())))
    max_range = float(np.max(list(ranges.values())))
    return {
        "key": key,
        "n_boxes": len(by_box),
        "seeds_per_box": {b: len(v) for b, v in by_box.items()},
        "per_box_mean": {b: float(np.mean(v)) for b, v in by_box.items()},
        "per_box_std": stds,
        "per_box_range": ranges,
        "mean_of_per_box_std": mean_std,
        "max_per_box_range": max_range,
        "mean_margin": float(mean_margin),
        "single_margin": float(single_margin),
        "proposal": {
            "mean_degradation_max": float(mean_margin) * mean_std,
            "single_box_degradation_max": float(single_margin) * max_range,
            "calibrated": True,
        },
    }


# --------------------------------------------------------------------------- #
# Checking
# --------------------------------------------------------------------------- #
def paired_degradation(
    candidate_rows: Sequence[Mapping],
    frozen_rows: Sequence[Mapping],
    *,
    key: str = "density_power_error",
) -> List[Dict]:
    """``err_theta - err_0`` matched on ``(box, seed)``.

    An unmatched candidate row is an error, not a skip. Comparing a candidate on
    seed 3 against a frozen baseline on seed 0 measures the seed, and the whole
    construction here exists to avoid exactly that.
    """
    base = {(str(r["box"]), int(r["seed"])): float(r[key]) for r in frozen_rows}
    out: List[Dict] = []
    missing: List[str] = []
    for r in candidate_rows:
        k = (str(r["box"]), int(r["seed"]))
        if k not in base:
            missing.append(f"{k[0]}/seed{k[1]}")
            continue
        out.append({
            "box": k[0], "seed": k[1],
            "candidate": float(r[key]), "frozen": base[k],
            "degradation": float(r[key]) - base[k],
        })
    if missing:
        raise KeyError(
            "no frozen baseline for " + ", ".join(missing[:8])
            + "; a candidate must be compared against the same box and seed"
        )
    if not out:
        raise ValueError("no candidate rows to compare")
    return out


@dataclass
class DirectGateResult:
    """Verdict plus every number behind it."""

    passed: bool
    violations: List[str]
    values: Dict[str, float]
    per_box: List[Dict]
    gate: Dict

    def to_dict(self) -> Dict:
        return {
            "passed": bool(self.passed),
            "violations": list(self.violations),
            "values": dict(self.values),
            "per_box": list(self.per_box),
            "gate": dict(self.gate),
        }


def check_direct_gates(
    candidate_rows: Sequence[Mapping],
    frozen_rows: Sequence[Mapping],
    gate: RelativeDensityGate,
    *,
    key: str = "density_power_error",
) -> DirectGateResult:
    """Accept or reject one checkpoint. Every bound is checked, none short-circuit.

    ``candidate_rows`` carry, per ``(box, seed)``: ``density_power_error``,
    optionally ``low_k_change`` and ``d_struct``. NaN never passes an enabled
    bound -- a measurement that failed is not a measurement that passed.
    """
    per_box = paired_degradation(candidate_rows, frozen_rows, key=key)
    deg = np.asarray([r["degradation"] for r in per_box], dtype=np.float64)
    cand = np.asarray([r["candidate"] for r in per_box], dtype=np.float64)

    values: Dict[str, float] = {
        "n_comparisons": float(len(per_box)),
        "mean_degradation": float(np.mean(deg)),
        "max_degradation": float(np.max(deg)),
        "mean_candidate_error": float(np.mean(cand)),
        "max_candidate_error": float(np.max(cand)),
    }
    for name, agg in (("low_k_change", np.max), ("d_struct", np.min)):
        vals = [float(r[name]) for r in candidate_rows if name in r]
        values[f"{'max' if name == 'low_k_change' else 'min'}_{name}"] = (
            float(agg(vals)) if vals else float("nan")
        )

    violations: List[str] = []

    def bound(value_key: str, limit: Optional[float], kind: str, label: str) -> None:
        if limit is None:
            return
        v = values.get(value_key, float("nan"))
        if not np.isfinite(v):
            violations.append(f"{label}=nan (bound {limit:.6g})")
        elif kind == "max" and v > limit:
            violations.append(f"{label}={v:.6g} > {limit:.6g}")
        elif kind == "min" and v < limit:
            violations.append(f"{label}={v:.6g} < {limit:.6g}")

    bound("mean_degradation", gate.mean_degradation_max, "max",
          "mean density-power degradation vs frozen")
    bound("max_degradation", gate.single_box_degradation_max, "max",
          "worst single-box density-power degradation vs frozen")
    bound("max_candidate_error", gate.density_power_error_max, "max",
          "absolute density-power error")
    bound("max_low_k_change", gate.low_k_change_max, "max", "low-k displacement change")
    bound("min_d_struct", gate.d_struct_min, "min", "structural diversity")

    if not gate.calibrated:
        violations.append(
            "gate.calibrated is false -- the tolerances are placeholders; run "
            "the frozen seed-to-seed calibration and paste the proposal into "
            "configs/reward/sr2_direct_finetune.yaml"
        )
    return DirectGateResult(
        passed=not violations, violations=violations, values=values,
        per_box=per_box, gate=gate.to_dict(),
    )
