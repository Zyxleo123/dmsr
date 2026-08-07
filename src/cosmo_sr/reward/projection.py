"""The projection oracle: how much does deleting the coarse residual cost?

The correction transform can be constrained to leave the LR-visible scales
alone (``block_null``), or allowed a bounded fraction ``alpha`` of them
(``block_leaky``). That is a choice about the action space, and it is made here
by measurement rather than by argument.

The construction uses the *paired* residual, which is why this is an oracle and
not a policy:

    B = Psi_SR2 ,  r = Psi_HR - B ,
    X(alpha_dis, alpha_vel) = B + T_{alpha_dis}(r_dis) + T_{alpha_vel}(r_vel) ,
    T_alpha(r) = P_N r + alpha * P_R r .

``alpha = 1`` reproduces ``Psi_HR`` exactly (``T_1 = I``), so the sweep
interpolates between the frozen baseline's coarse scales and HR's, holding the
within-block part fixed at HR's. Whatever structural metric collapses as
``alpha -> 0`` is a metric the coarse component is responsible for, and a policy
forbidden that component cannot recover it however well it is trained.

**This experiment chooses a constraint. It must not produce training examples.**
Nothing here writes a residual, a crop or a replay entry -- the deliverable is a
number for ``alpha_disp`` and ``alpha_vel``, and the fields it builds are scored
and discarded. See ``scripts/reward/audit_projection_oracle.py``.

Uncertainty is a **box bootstrap**: boxes are the independent cosmological unit,
seeds within a box are not, so seeds are averaged inside a box first and boxes
are resampled with replacement. Comparisons against ``alpha = 1`` are *paired*
on the box, which removes the box-to-box scatter that otherwise swamps the
effect being measured.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

__all__ = [
    "DEFAULT_ALPHAS",
    "PRIMARY_METRICS",
    "MetricSpec",
    "ProjectionArm",
    "arm_plan",
    "bootstrap_ci",
    "choose_alpha",
    "compare_to_reference",
    "paired_bootstrap_diff",
    "per_box_means",
    "project_residual_field",
    "reference_arm_name",
]

DEFAULT_ALPHAS: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)

SWEEPS: Tuple[str, ...] = ("joint", "disp_only", "vel_only")


@dataclass(frozen=True)
class MetricSpec:
    """A scalar the decision is allowed to depend on, and which way is better."""

    name: str
    higher_is_better: bool
    label: str = ""

    def sign(self) -> float:
        return 1.0 if self.higher_is_better else -1.0


#: The structural metrics the decision rule reads. Occupation is the primary
#: scientific target; host recovery and density are the two ways a candidate can
#: look better on occupation while being worse as a field.
PRIMARY_METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec("R_occ_reliable", True, "reliable-bin occupation reward"),
    MetricSpec("host_recovery_fraction", True, "fraction of HR hosts matched"),
    MetricSpec("density_power_error", False, "mean |log P_hat/P_hr| of delta"),
)


@dataclass(frozen=True)
class ProjectionArm:
    """One point of the sweep: a pair of coarse allowances with a stable name."""

    name: str
    alpha_disp: float
    alpha_vel: float
    sweep: str

    def to_dict(self) -> Dict:
        return {"arm": self.name, "alpha_disp": float(self.alpha_disp),
                "alpha_vel": float(self.alpha_vel), "sweep": self.sweep}


def _fmt(a: float) -> str:
    return f"{float(a):g}".replace(".", "p")


def arm_plan(
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    sweeps: Sequence[str] = SWEEPS,
    *,
    include_reference_arms: bool = True,
) -> List[ProjectionArm]:
    """Every arm of the sweep, deduplicated and in a deterministic order.

    * ``joint``     ``alpha_dis = alpha_vel = a``
    * ``disp_only`` ``alpha_dis = a``, ``alpha_vel = 1`` -- velocity untouched
    * ``vel_only``  ``alpha_vel = a``, ``alpha_dis = 1`` -- displacement untouched

    The three sweeps share their ``a = 1`` point (it is ``Psi_HR`` in all of
    them), so it is emitted once, as ``joint_a1``. ``include_reference_arms``
    adds ``sr2`` (the frozen baseline, ``r`` deleted entirely) and ``hr`` (the
    paired field itself, run through the same Rockstar settings) -- the two
    anchors every ratio in the report is quoted against.
    """
    bad = [s for s in sweeps if s not in SWEEPS]
    if bad:
        raise ValueError(f"unknown sweep(s) {bad}; use {SWEEPS}")
    for a in alphas:
        if not 0.0 <= float(a) <= 1.0:
            raise ValueError(f"alpha {a} outside [0, 1]")

    out: List[ProjectionArm] = []
    seen = set()

    def add(arm: ProjectionArm) -> None:
        key = (round(arm.alpha_disp, 6), round(arm.alpha_vel, 6))
        if key in seen:
            return
        seen.add(key)
        out.append(arm)

    if include_reference_arms:
        out.append(ProjectionArm("sr2", float("nan"), float("nan"), "reference"))
        out.append(ProjectionArm("hr", float("nan"), float("nan"), "reference"))

    for sweep in sweeps:
        for a in alphas:
            a = float(a)
            if sweep == "joint":
                add(ProjectionArm(f"joint_a{_fmt(a)}", a, a, "joint"))
            elif sweep == "disp_only":
                add(ProjectionArm(f"disp_a{_fmt(a)}", a, 1.0, "disp_only"))
            else:
                add(ProjectionArm(f"vel_a{_fmt(a)}", 1.0, a, "vel_only"))
    return out


def reference_arm_name(arms: Sequence[ProjectionArm]) -> str:
    """The ``alpha = 1`` arm every other arm is compared against."""
    for arm in arms:
        if arm.sweep != "reference" and arm.alpha_disp == 1.0 and arm.alpha_vel == 1.0:
            return arm.name
    raise ValueError(
        "no alpha=1 arm in the plan; the sweep has no upper reference and no "
        "comparison in this module is meaningful without one"
    )


# --------------------------------------------------------------------------- #
# Field construction
# --------------------------------------------------------------------------- #
def _block_coarse(r: np.ndarray, factor: int) -> np.ndarray:
    """``P_R r`` for a ``(C, nx, N, N)`` slab whose ``nx`` is block-aligned."""
    c, nx, ny, nz = r.shape
    f = int(factor)
    small = r.reshape(c, nx // f, f, ny // f, f, nz // f, f).mean(axis=(2, 4, 6))
    return np.repeat(np.repeat(np.repeat(small, f, axis=1), f, axis=2), f, axis=3)


def project_residual_field(
    hr: np.ndarray,
    base: np.ndarray,
    *,
    alpha_disp: float,
    alpha_vel: float,
    scale_factor: int = 8,
    disp_channels: Sequence[int] = (0, 1, 2),
    vel_channels: Sequence[int] = (3, 4, 5),
    slab: int = 64,
    dtype=np.float32,
) -> np.ndarray:
    """``X = B + T_{alpha_dis}(r_dis) + T_{alpha_vel}(r_vel)`` for a full box.

    Streams in ``slab``-thick Lagrangian slabs so peak memory is the output box
    plus one slab, not three boxes. ``slab`` must be a multiple of
    ``scale_factor`` -- a slab that splits a block would compute the block mean
    from part of the block.

    ``alpha_dis = alpha_vel = 1`` returns ``Psi_HR`` to float round-off, which
    the audit checks explicitly rather than assuming.
    """
    hr = np.asarray(hr)
    base = np.asarray(base)
    if hr.shape != base.shape:
        raise ValueError(f"HR {hr.shape} != base {base.shape}")
    f = int(scale_factor)
    if int(slab) % f:
        raise ValueError(f"slab={slab} must be a multiple of scale_factor={f}")
    n = int(hr.shape[1])
    if n % f:
        raise ValueError(f"grid {n} is not a whole number of {f}-blocks")

    alpha = np.ones(int(hr.shape[0]), dtype=np.float64)
    for c in disp_channels:
        alpha[int(c)] = float(alpha_disp)
    for c in vel_channels:
        alpha[int(c)] = float(alpha_vel)
    a = alpha.reshape(-1, 1, 1, 1).astype(np.float32)

    out = np.empty(hr.shape, dtype=dtype)
    for i in range(0, n, int(slab)):
        j = min(i + int(slab), n)
        b = np.asarray(base[:, i:j], dtype=np.float32)
        r = np.asarray(hr[:, i:j], dtype=np.float32) - b
        coarse = _block_coarse(r, f)
        # T_alpha(r) = (r - P_R r) + alpha * P_R r
        out[:, i:j] = (b + (r - coarse) + a * coarse).astype(dtype, copy=False)
        del b, r, coarse
    return out


# --------------------------------------------------------------------------- #
# Box bootstrap
# --------------------------------------------------------------------------- #
def per_box_means(
    rows: Iterable[Mapping], metric: str, *, box_key: str = "box"
) -> Dict[str, float]:
    """Average a metric over the seeds of each box.

    Seeds of one box are not independent samples of the universe -- they are
    repeats of the same realisation -- so they are collapsed before the box
    bootstrap ever sees them. Non-finite values are dropped, and a box whose
    values are all non-finite does not appear.
    """
    acc: Dict[str, List[float]] = {}
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        v = float(v)
        if not np.isfinite(v):
            continue
        acc.setdefault(str(r[box_key]), []).append(v)
    return {b: float(np.mean(vs)) for b, vs in acc.items() if vs}


def bootstrap_ci(
    values: Mapping[str, float] | Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Mean and percentile CI over a **box** bootstrap.

    Returns NaN bounds for a single box rather than a zero-width interval: one
    box carries no information about box-to-box scatter, and a zero-width CI
    would read as certainty.
    """
    v = np.asarray(list(values.values()) if isinstance(values, Mapping) else list(values),
                   dtype=np.float64)
    v = v[np.isfinite(v)]
    out = {"n_boxes": int(v.size), "mean": float(v.mean()) if v.size else float("nan")}
    if v.size < 2:
        out.update({"lo": float("nan"), "hi": float("nan"), "se": float("nan")})
        return out
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, v.size, size=(int(n_boot), v.size))
    means = v[draws].mean(axis=1)
    lo_q = 0.5 * (1.0 - float(ci))
    out.update({
        "lo": float(np.quantile(means, lo_q)),
        "hi": float(np.quantile(means, 1.0 - lo_q)),
        "se": float(means.std(ddof=1)),
    })
    return out


def paired_bootstrap_diff(
    arm: Mapping[str, float],
    reference: Mapping[str, float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Box bootstrap of ``arm - reference``, paired on the box.

    Pairing is what makes the sweep readable: the box-to-box scatter of
    occupation is far larger than the difference between two alphas, so an
    unpaired comparison would call everything indistinguishable.
    """
    boxes = sorted(set(arm) & set(reference))
    d = {b: float(arm[b]) - float(reference[b]) for b in boxes
         if np.isfinite(arm[b]) and np.isfinite(reference[b])}
    out = bootstrap_ci(d, n_boot=n_boot, seed=seed, ci=ci)
    out["n_paired_boxes"] = len(d)
    return out


def compare_to_reference(
    rows: Sequence[Mapping],
    metric: MetricSpec,
    arm_name: str,
    reference_name: str,
    *,
    arm_key: str = "arm",
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> Dict:
    """Paired comparison of one arm against the reference, on one metric.

    ``verdict`` is one of:

    ``indistinguishable``  the CI of the difference contains 0;
    ``damaged``            the CI lies entirely on the worse side;
    ``improved``           the CI lies entirely on the better side;
    ``undetermined``       fewer than two paired boxes.
    """
    arm_rows = [r for r in rows if str(r.get(arm_key)) == arm_name]
    ref_rows = [r for r in rows if str(r.get(arm_key)) == reference_name]
    a = per_box_means(arm_rows, metric.name)
    b = per_box_means(ref_rows, metric.name)
    d = paired_bootstrap_diff(a, b, n_boot=n_boot, seed=seed, ci=ci)

    verdict = "undetermined"
    if d["n_paired_boxes"] >= 2 and np.isfinite(d["lo"]) and np.isfinite(d["hi"]):
        signed_lo, signed_hi = metric.sign() * d["lo"], metric.sign() * d["hi"]
        if signed_lo > 0.0:
            verdict = "improved"
        elif signed_hi < 0.0:
            verdict = "damaged"
        else:
            verdict = "indistinguishable"
    return {
        "arm": arm_name, "reference": reference_name, "metric": metric.name,
        "higher_is_better": metric.higher_is_better,
        "arm_mean": bootstrap_ci(a, n_boot=n_boot, seed=seed, ci=ci)["mean"],
        "reference_mean": bootstrap_ci(b, n_boot=n_boot, seed=seed, ci=ci)["mean"],
        "diff": d, "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Decision rule
# --------------------------------------------------------------------------- #
def choose_alpha(
    rows: Sequence[Mapping],
    arms: Sequence[ProjectionArm],
    *,
    sweep: str,
    metrics: Sequence[MetricSpec] = PRIMARY_METRICS,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
    arm_key: str = "arm",
) -> Dict:
    """Smallest coarse allowance indistinguishable from ``alpha = 1``.

    The rule, in order:

    1. If ``alpha = 0`` is *damaged* on any primary metric, the hard null
       projection is rejected -- that is the question the experiment exists to
       answer, and it is answered by the ``alpha = 0`` arm alone.
    2. Among the remaining alphas, take the smallest whose difference from
       ``alpha = 1`` is ``indistinguishable`` (or ``improved``) on **every**
       primary metric.
    3. If none qualifies, the recommendation is ``alpha = 1``: no coarse
       allowance was shown to be affordable, so none is claimed.

    ``undetermined`` never counts as passing. Too few boxes to resolve a
    difference is not evidence that there is none, and the returned dict says so
    in ``blocked_by``.
    """
    ref = reference_arm_name(arms)
    in_sweep = [a for a in arms
                if a.sweep == sweep or (a.name == ref and sweep != "reference")]
    varying = "alpha_vel" if sweep == "vel_only" else "alpha_disp"
    ladder = sorted({(getattr(a, varying), a.name) for a in in_sweep})

    per_arm: Dict[str, Dict] = {}
    for alpha, name in ladder:
        per_arm[name] = {
            "alpha": float(alpha),
            "metrics": {
                m.name: compare_to_reference(rows, m, name, ref, arm_key=arm_key,
                                             n_boot=n_boot, seed=seed, ci=ci)
                for m in metrics
            },
        }

    def verdicts(name: str) -> List[str]:
        return [v["verdict"] for v in per_arm[name]["metrics"].values()]

    zero = [n for a, n in ladder if float(a) == 0.0]
    null_damaged, null_damage_on = False, []
    if zero:
        null_damage_on = [
            m for m, v in per_arm[zero[0]]["metrics"].items() if v["verdict"] == "damaged"
        ]
        null_damaged = bool(null_damage_on)

    chosen, blocked_by = None, {}
    for alpha, name in ladder:
        if name == ref:
            continue
        vs = verdicts(name)
        if all(v in ("indistinguishable", "improved") for v in vs):
            chosen = {"alpha": float(alpha), "arm": name}
            break
        blocked_by[name] = {
            m: v["verdict"] for m, v in per_arm[name]["metrics"].items()
            if v["verdict"] != "indistinguishable" and v["verdict"] != "improved"
        }
    if chosen is None:
        chosen = {"alpha": 1.0, "arm": ref,
                  "note": "no smaller allowance was statistically indistinguishable "
                          "from alpha=1 on every primary metric"}

    return {
        "sweep": sweep,
        "reference_arm": ref,
        "varying": varying,
        "hard_null_rejected": bool(null_damaged),
        "hard_null_damaged_metrics": null_damage_on,
        "recommended": chosen,
        "blocked_by": blocked_by,
        "arms": per_arm,
        "metrics": [m.name for m in metrics],
        "ci": float(ci),
        "n_boot": int(n_boot),
    }
