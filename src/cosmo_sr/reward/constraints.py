"""Field fidelity as a feasibility filter, not a second reward.

Folding field statistics into the objective as another weighted term invites the
optimiser to trade a little large-scale accuracy for a lot of catalog reward,
and then to report the average. Instead every candidate is measured against hard
thresholds and either **feasible** or **infeasible**; an infeasible candidate can
never become an elite, no matter how good its catalog reward is.

Measured per candidate ensemble:

``low_k_change``
    ``||A(Psi_hat) - A(Psi_base)|| / ||A(Psi_base)||`` -- how far the residual
    moved the scales SR2 already gets right. ``A`` is the same block-average
    degrader used everywhere else, so this is exactly the LR-visible part.
``displacement_power_error``
    Mean ``|T(k) - 1|`` of the displacement against HR over all bands,
    ``T = sqrt(P_hat / P_hr)``.
``density_power_error``
    Mean ``|log(P_delta_hat / P_delta_hr)|`` of the CIC overdensity.
``lr_consistency_error``
    ``||A(Psi_hat) - y_lr|| / ||y_lr||``. The frozen baseline's own value is not
    zero, so the threshold is set from it rather than from 0.
``diversity``
    RMS spread across residual samples divided by the mean residual RMS. A floor,
    not a ceiling: collapse to a deterministic corrector is a failure mode.

A NaN in any measured value makes the candidate infeasible. Silence is the one
outcome a filter must never produce.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import fields

__all__ = [
    "ConstraintSet",
    "check_feasible",
    "constraint_values",
    "diversity_value",
    "load_constraints",
]


@dataclass(frozen=True)
class ConstraintSet:
    """Hard thresholds. ``None`` disables a constraint (it is still reported)."""

    low_k_change_max: Optional[float] = 0.02
    displacement_power_error_max: Optional[float] = None
    density_power_error_max: Optional[float] = None
    lr_consistency_error_max: Optional[float] = None
    diversity_min: Optional[float] = 0.05
    # False = the committed thresholds are placeholders, not values derived from
    # the frozen baseline. Carried on the object (not just left in the YAML) so
    # a consumer cannot use the numbers without also seeing where they came from.
    calibrated: bool = False

    def to_dict(self) -> Dict:
        return {
            "low_k_change_max": self.low_k_change_max,
            "displacement_power_error_max": self.displacement_power_error_max,
            "density_power_error_max": self.density_power_error_max,
            "lr_consistency_error_max": self.lr_consistency_error_max,
            "diversity_min": self.diversity_min,
            "calibrated": self.calibrated,
        }

    def as_bounds(self) -> Dict[str, Tuple[str, Optional[float]]]:
        return {
            "low_k_change": ("max", self.low_k_change_max),
            "displacement_power_error": ("max", self.displacement_power_error_max),
            "density_power_error": ("max", self.density_power_error_max),
            "lr_consistency_error": ("max", self.lr_consistency_error_max),
            "diversity": ("min", self.diversity_min),
        }


def load_constraints(cfg: Dict) -> ConstraintSet:
    c = dict(cfg or {})
    def g(k, d):
        v = c.get(k, d)
        return None if v is None else float(v)
    return ConstraintSet(
        low_k_change_max=g("low_k_change_max", 0.02),
        displacement_power_error_max=g("displacement_power_error_max", None),
        density_power_error_max=g("density_power_error_max", None),
        lr_consistency_error_max=g("lr_consistency_error_max", None),
        diversity_min=g("diversity_min", 0.05),
        calibrated=bool(c.get("calibrated", False)),
    )


def diversity_value(residuals: Sequence[np.ndarray]) -> float:
    """RMS spread across residual samples, normalised by the mean residual RMS.

    ``< 1e-6`` means the sampler has collapsed to a deterministic map.
    """
    if len(residuals) < 2:
        return float("nan")
    stack = np.stack([np.asarray(r, dtype=np.float32) for r in residuals])
    var = stack.var(axis=0, ddof=1).mean()
    scale = np.sqrt(np.mean(stack.astype(np.float64) ** 2))
    return float(np.sqrt(var) / max(scale, 1e-30))


def constraint_values(
    psi_hat: np.ndarray,
    psi_base: np.ndarray,
    y_lr: np.ndarray,
    *,
    hr: Optional[np.ndarray] = None,
    scale_factor: int = 8,
    n_bins: int = 24,
    boxsize_mpc_h: float = 100.0,
    dis_norm_kpc_h: float = 6000.0,
    redshift: float = 0.0,
    disp_channels: Sequence[int] = (0, 1, 2),
    diversity: Optional[float] = None,
    compute_density: bool = True,
) -> Dict[str, float]:
    """All constraint quantities plus the diagnostics that explain them.

    ``hr`` is optional: on held-out boxes without a paired HR field the
    HR-referenced entries come back as NaN, which :func:`check_feasible` treats
    as a violation of any *enabled* threshold. Disable those thresholds
    explicitly rather than letting NaN pass.
    """
    hat = np.asarray(psi_hat)
    base = np.asarray(psi_base)
    lr = np.asarray(y_lr)
    if hat.shape != base.shape:
        raise ValueError(f"psi_hat {hat.shape} != psi_base {base.shape}")
    n = int(hat.shape[1])
    k_lr_nyq = n / (2.0 * float(scale_factor))
    out: Dict[str, float] = {}

    a_hat = fields.block_average(hat, scale_factor)
    a_base = fields.block_average(base, scale_factor)
    out["low_k_change"] = fields.rel_rms(a_hat, a_base)
    out["lr_consistency_error"] = fields.rel_rms(a_hat, lr)
    out["lr_consistency_error_base"] = fields.rel_rms(a_base, lr)
    del a_hat, a_base

    out["residual_rms"] = float(
        np.sqrt(np.mean((hat[list(disp_channels)].astype(np.float64)
                         - base[list(disp_channels)].astype(np.float64)) ** 2))
    )
    out["base_rms"] = float(np.sqrt(np.mean(base[list(disp_channels)].astype(np.float64) ** 2)))

    # --- displacement power / correlation, per channel then averaged ---------
    if hr is not None:
        t_err, r_low, t_low = [], [], []
        for ch in disp_channels:
            k, p_hat, p_hr, p_x = fields.cross_power(
                np.asarray(hat[ch]), np.asarray(hr[ch]), n_bins
            )
            tk = np.sqrt(np.maximum(p_hat, 0.0) / np.maximum(p_hr, 1e-30))
            rk = p_x / np.maximum(np.sqrt(np.maximum(p_hat * p_hr, 0.0)), 1e-30)
            masks = fields.band_masks(k, k_lr_nyq)
            t_err.append(float(np.mean(np.abs(tk - 1.0))))
            t_low.append(float(np.mean(np.abs(tk[masks["low"]] - 1.0))) if masks["low"].any() else np.nan)
            r_low.append(float(np.mean(rk[masks["low"]])) if masks["low"].any() else np.nan)
        out["displacement_power_error"] = float(np.mean(t_err))
        out["displacement_power_error_low_k"] = float(np.mean(t_low))
        out["displacement_rk_low_k"] = float(np.mean(r_low))
    else:
        out["displacement_power_error"] = float("nan")
        out["displacement_power_error_low_k"] = float("nan")
        out["displacement_rk_low_k"] = float("nan")

    # --- density -------------------------------------------------------------
    if compute_density:
        d_hat = fields.cic_density_box(
            hat[0:3], boxsize_mpc_h=boxsize_mpc_h,
            dis_norm_kpc_h=dis_norm_kpc_h, redshift=redshift,
        )
        if hr is not None:
            d_hr = fields.cic_density_box(
                np.asarray(hr[0:3]), boxsize_mpc_h=boxsize_mpc_h,
                dis_norm_kpc_h=dis_norm_kpc_h, redshift=redshift,
            )
            k, p_hat, p_hr, p_x = fields.cross_power(d_hat, d_hr, n_bins)
            ratio = np.maximum(p_hat, 1e-30) / np.maximum(p_hr, 1e-30)
            out["density_power_error"] = float(np.mean(np.abs(np.log(ratio))))
            out["density_sigma_ratio"] = float(d_hat.std() / max(d_hr.std(), 1e-30))
            _, pdf_hat = fields.density_pdf(d_hat)
            _, pdf_hr = fields.density_pdf(d_hr)
            out["density_pdf_l1"] = float(np.abs(pdf_hat - pdf_hr).sum())
            del d_hr
        else:
            out["density_power_error"] = float("nan")
            out["density_sigma_ratio"] = float("nan")
            out["density_pdf_l1"] = float("nan")
        out["density_sigma"] = float(d_hat.std())
        del d_hat
    else:
        out["density_power_error"] = float("nan")
        out["density_sigma_ratio"] = float("nan")
        out["density_pdf_l1"] = float("nan")
        out["density_sigma"] = float("nan")

    out["diversity"] = float("nan") if diversity is None else float(diversity)
    return out


def check_feasible(
    values: Dict[str, float], constraints: ConstraintSet
) -> Tuple[bool, List[str]]:
    """``(feasible, violations)``. Deterministic; NaN never passes an enabled bound."""
    violations: List[str] = []
    for name, (kind, bound) in constraints.as_bounds().items():
        if bound is None:
            continue
        v = values.get(name, float("nan"))
        if v is None or not np.isfinite(v):
            violations.append(f"{name}=nan")
            continue
        if kind == "max" and v > bound:
            violations.append(f"{name}={v:.6g}>{bound:.6g}")
        elif kind == "min" and v < bound:
            violations.append(f"{name}={v:.6g}<{bound:.6g}")
    return (len(violations) == 0), violations
