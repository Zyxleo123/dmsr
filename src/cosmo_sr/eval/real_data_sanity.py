"""Experiment 0: real-data operator sanity.

Loads a real HR field, builds the average-pool pyramid (levels 512, 256, 128,
64), and verifies the exact operator identities that the whole method relies on:

* ``A_R(U_R(x_R)) == x_R``                 (U is a right inverse of A)
* ``A_R(r_star) == 0``                     (r_star lives in ker A_R)
* ``A_R(x_recon_oracle) == x_R``           (hard LR consistency)
* ``x_recon_oracle == x_2R``               (oracle residual recovers the field)

where ``base = consistent_base(identity, ops, x_R)`` and
``r_star = P_null_R(x_2R - base)``.

Usage::

    python -m cosmo_sr.eval.real_data_sanity \
        --hr .../set15.npy --crop 128 --out runs/poc_real_data_sanity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from ..data.field_io import load_field
from ..data.crops import periodic_crop
from ..operators.multiscale import MultiScaleOperators
from ..operators.base_upscaler import IdentityUpscaler, consistent_base
from ..losses.flow import band_power

PASS_TOL = 1e-5


def build_pyramid(crop: torch.Tensor, full_res: int = 512, n_levels: int = 4) -> Dict[int, torch.Tensor]:
    """``crop`` is ``(1, C, N, N, N)``; returns {label: field} finest->coarsest."""
    levels = [crop]
    for _ in range(n_levels - 1):
        levels.append(F.avg_pool3d(levels[-1], kernel_size=2, stride=2))
    return {full_res // (2 ** lvl): t for lvl, t in enumerate(levels)}


def run(hr_path: str, crop_size: int, out_dir: str, full_res: int = 512,
        n_levels: int = 4, seed: int = 0) -> Dict[str, float]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ops = MultiScaleOperators(2)
    identity = IdentityUpscaler(2)

    field = load_field(hr_path, mmap=True)
    Ng = field.shape[1]
    rng = np.random.default_rng(seed)
    start = tuple(int(rng.integers(0, Ng)) for _ in range(3))
    crop_np = periodic_crop(field, start, crop_size, pad=0)
    crop = torch.from_numpy(np.ascontiguousarray(crop_np)).float().unsqueeze(0)

    pyramid = build_pyramid(crop, full_res=full_res, n_levels=n_levels)
    resolutions = [r for r in (64, 128, 256) if r in pyramid and 2 * r in pyramid]

    metrics: Dict[str, float] = {}
    worst = {"AU": 0.0, "null": 0.0, "cons": 0.0, "oracle": 0.0}
    for R in resolutions:
        x_R = pyramid[R]
        x_2R = pyramid[2 * R]
        au_err = float((ops.A(ops.U(x_R)) - x_R).abs().max())
        base = consistent_base(identity, ops, x_R)
        r_star = ops.P_null(x_2R - base)
        null_err = float(ops.A(r_star).abs().max())
        x_oracle = base + r_star
        cons_err = float((ops.A(x_oracle) - x_R).abs().max())
        oracle_mse = float(F.mse_loss(x_oracle, x_2R))
        metrics[f"sanity/max_abs_AU_error_R{R}"] = au_err
        metrics[f"sanity/max_abs_null_error_R{R}"] = null_err
        metrics[f"sanity/max_abs_consistency_error_R{R}"] = cons_err
        metrics[f"sanity/oracle_x_mse_R{R}"] = oracle_mse
        metrics[f"sanity/residual_var_R{R}"] = float(torch.var(r_star))
        metrics[f"sanity/residual_power_R{R}"] = float(
            band_power(r_star, n_bands=8, log=False).mean()
        )
        worst["AU"] = max(worst["AU"], au_err)
        worst["null"] = max(worst["null"], null_err)
        worst["cons"] = max(worst["cons"], cons_err)
        worst["oracle"] = max(worst["oracle"], oracle_mse)

    passed = (
        worst["AU"] < PASS_TOL and worst["null"] < PASS_TOL
        and worst["cons"] < PASS_TOL and worst["oracle"] < 1e-8
    )
    metrics["sanity/passed"] = float(passed)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[real_data_sanity] passed={passed} worst={worst}")
    for k in sorted(metrics):
        print(f"  {k}: {metrics[k]:.3e}")
    return metrics


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Real-data operator sanity (Experiment 0)")
    p.add_argument("--hr", required=True, type=str)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--out", type=str, default="runs/poc_real_data_sanity")
    p.add_argument("--full-res", type=int, default=512)
    p.add_argument("--n-levels", type=int, default=4)
    args = p.parse_args(argv)
    metrics = run(args.hr, args.crop, args.out, full_res=args.full_res, n_levels=args.n_levels)
    if metrics.get("sanity/passed", 0.0) < 1.0:
        raise SystemExit("real_data_sanity FAILED operator checks")


if __name__ == "__main__":
    main()
