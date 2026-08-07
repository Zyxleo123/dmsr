#!/usr/bin/env python
"""Measure the amplitude of the paired residual's coarse and fine components.

The correction transform bounds every proposal by ``u = s * tanh(h)``. ``s``
decides how large an edit any policy can ever make, so it is measured here from
paired data rather than written into a config by hand.

For each paired box,

    r = Psi_HR - Psi_SR2 ,   r_coarse = P_R r = A^dagger A r ,   r_fine = P_N r

and the reported scale is ``bound_sigma`` times the RMS of the corresponding
component, per channel group (displacement, velocity). The two components are
reported separately because they differ by roughly an order of magnitude, so a
single bound would be a real constraint on one and a formality on the other.

``bound_sigma`` is a stated convention, not a measurement, so its consequence is
measured too: the report carries the fraction of residual voxels that exceed the
proposed bound, taken from a histogram accumulated in the same pass. A bound
that excludes a large tail is a bound that will saturate.

**This is calibration, not training data.** Paired HR is read to size the action
space and for nothing else; no residual, crop or target is written.

    python scripts/reward/calibrate_correction_scales.py --boxes set0,set1
"""
from __future__ import annotations

import argparse

import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from _common import (add_common_args, assert_no_leak, banner, hr_path,
                     load_reward_config, parse_boxes, write_json)

from cosmo_sr.reward import paths
from cosmo_sr.reward.base import find_base_field

# Histogram of |value| over log10 bins, so any quantile or exceedance fraction
# can be derived afterwards without a second pass over 3.2 GB of field.
LOG_MIN, LOG_MAX, N_BINS = -8.0, 2.0, 500


class Accumulator:
    """Streaming moments plus a log-magnitude histogram, for one component."""

    def __init__(self) -> None:
        self.count = 0
        self.sum_sq = 0.0
        self.sum = 0.0
        self.absmax = 0.0
        self.hist = np.zeros(N_BINS, dtype=np.int64)

    def update(self, x: np.ndarray) -> None:
        v = np.asarray(x, dtype=np.float64).ravel()
        self.count += v.size
        self.sum += float(v.sum())
        self.sum_sq += float((v ** 2).sum())
        self.absmax = max(self.absmax, float(np.abs(v).max(initial=0.0)))
        a = np.abs(v)
        lg = np.log10(np.maximum(a, 10.0 ** LOG_MIN))
        idx = np.clip(
            ((lg - LOG_MIN) / (LOG_MAX - LOG_MIN) * N_BINS).astype(np.int64),
            0, N_BINS - 1,
        )
        self.hist += np.bincount(idx, minlength=N_BINS)

    @property
    def rms(self) -> float:
        return float(np.sqrt(self.sum_sq / max(self.count, 1)))

    @property
    def mean(self) -> float:
        return float(self.sum / max(self.count, 1))

    def quantile(self, q: float) -> float:
        """Approximate quantile of ``|x|`` from the histogram (~0.02 dex bins)."""
        if self.count == 0:
            return float("nan")
        c = np.cumsum(self.hist)
        j = int(np.searchsorted(c, q * self.count))
        j = min(max(j, 0), N_BINS - 1)
        edge = LOG_MIN + (j + 1) * (LOG_MAX - LOG_MIN) / N_BINS
        return float(10.0 ** edge)

    def exceedance(self, bound: float) -> float:
        """Fraction of ``|x|`` above ``bound``; the cost of the chosen bound."""
        if self.count == 0 or not np.isfinite(bound) or bound <= 0:
            return float("nan")
        lg = np.log10(bound)
        j = int(np.ceil((lg - LOG_MIN) / (LOG_MAX - LOG_MIN) * N_BINS))
        j = min(max(j, 0), N_BINS)
        return float(self.hist[j:].sum() / self.count)

    def to_dict(self, bound: float) -> Dict:
        return {
            "count": int(self.count),
            "mean": self.mean,
            "rms": self.rms,
            "absmax": self.absmax,
            "p99": self.quantile(0.99),
            "p999": self.quantile(0.999),
            "proposed_bound": float(bound),
            "fraction_above_bound": self.exceedance(bound),
        }

    def merge(self, other: "Accumulator") -> "Accumulator":
        out = Accumulator()
        out.count = self.count + other.count
        out.sum = self.sum + other.sum
        out.sum_sq = self.sum_sq + other.sum_sq
        out.absmax = max(self.absmax, other.absmax)
        out.hist = self.hist + other.hist
        return out


def block_parts(r: np.ndarray, factor: int) -> tuple:
    """``(P_N r, P_R r)`` for a ``(C, nx, N, N)`` slab whose ``nx`` is block-aligned."""
    c, nx, ny, nz = r.shape
    f = int(factor)
    coarse_small = r.reshape(c, nx // f, f, ny // f, f, nz // f, f).mean(axis=(2, 4, 6))
    coarse = np.repeat(np.repeat(np.repeat(coarse_small, f, axis=1), f, axis=2), f, axis=3)
    return r - coarse, coarse


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--boxes", default=None, help="comma list; default = train split")
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test", "dev", "final"])
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--bound-sigma", type=float, default=3.0,
                    help="amplitude bound in units of the component RMS")
    ap.add_argument("--slab", type=int, default=64,
                    help="HR cells per streaming slab (must be a multiple of "
                         "scale_factor so blocks are not split)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: $DMSR_REWARD_ROOT/audits/"
                         "correction_scales/correction_scales.json)")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    boxes = parse_boxes(args.boxes, cfg, args.split)
    # Calibration may read paired HR, but only from boxes that are allowed to be
    # opened -- reading a final-eval box here would burn it just as surely as
    # training on it would.
    assert_no_leak(cfg, boxes, ["train", "val"])

    f = int(cfg["data"]["scale_factor"])
    slab = int(args.slab)
    if slab % f:
        raise SystemExit(f"--slab {slab} must be a multiple of scale_factor={f}")

    disp = tuple(int(c) for c in cfg.get("correction", {}).get("disp_channels", (0, 1, 2)))
    vel = tuple(int(c) for c in cfg.get("correction", {}).get("vel_channels", (3, 4, 5)))

    groups = {"disp": disp, "vel": vel}
    pooled = {f"{part}_{g}": Accumulator() for g in groups for part in ("fine", "coarse")}
    per_box: List[Dict] = []

    for box in boxes:
        t0 = time.time()
        base_path = find_base_field(box, args.base_seed)
        if base_path is None:
            raise SystemExit(
                f"no cached SR2 base field for {box}; run "
                f"scripts/slurm/cache_sr2_base.sbatch first"
            )
        hr = np.load(hr_path(cfg, box), mmap_mode="r")
        base = np.load(base_path, mmap_mode="r")
        if hr.shape != base.shape:
            raise SystemExit(f"{box}: HR {hr.shape} != base {base.shape}")

        acc = {k: Accumulator() for k in pooled}
        n = int(hr.shape[1])
        for i in range(0, n, slab):
            j = min(i + slab, n)
            r = np.asarray(hr[:, i:j], dtype=np.float32) - \
                np.asarray(base[:, i:j], dtype=np.float32)
            fine, coarse = block_parts(r, f)
            for gname, chans in groups.items():
                acc[f"fine_{gname}"].update(fine[list(chans)])
                acc[f"coarse_{gname}"].update(coarse[list(chans)])
            del r, fine, coarse

        row = {"box": box, "base": str(base_path), "seconds": time.time() - t0}
        for k, a in acc.items():
            row[k] = a.to_dict(float(args.bound_sigma) * a.rms)
            pooled[k] = pooled[k].merge(a)
        per_box.append(row)
        print(f"[{box}] fine_disp_rms={acc['fine_disp'].rms:.4g} "
              f"coarse_disp_rms={acc['coarse_disp'].rms:.4g} "
              f"fine_vel_rms={acc['fine_vel'].rms:.4g} "
              f"coarse_vel_rms={acc['coarse_vel'].rms:.4g} "
              f"({row['seconds']:.0f}s)", flush=True)

    ks = float(args.bound_sigma)
    bounds = {k: ks * a.rms for k, a in pooled.items()}
    scales = {
        "fine_disp": bounds["fine_disp"],
        "fine_vel": bounds["fine_vel"],
        "coarse_disp": bounds["coarse_disp"],
        "coarse_vel": bounds["coarse_vel"],
        "calibrated": True,
        "source": "scripts/reward/calibrate_correction_scales.py",
        "boxes": list(boxes),
        "meta": {
            "bound_sigma": ks,
            "scale_factor": f,
            "disp_channels": list(disp),
            "vel_channels": list(vel),
            "base_seed": int(args.base_seed),
            "note": (
                "bound = bound_sigma * RMS of the component over the pooled "
                "boxes. fraction_above_bound in `pooled` is what that costs."
            ),
        },
    }

    out = Path(args.out) if args.out else \
        paths.AUDITS("correction_scales", create=True) / "correction_scales.json"
    write_json(out, {
        "scales": scales,
        "pooled": {k: a.to_dict(bounds[k]) for k, a in pooled.items()},
        "per_box": per_box,
        "coarse_over_fine_rms": {
            g: pooled[f"coarse_{g}"].rms / max(pooled[f"fine_{g}"].rms, 1e-30)
            for g in groups
        },
    })
    banner(f"correction scales -> {out}")
    for k in ("fine_disp", "coarse_disp", "fine_vel", "coarse_vel"):
        a = pooled[k]
        print(f"  {k:12s} rms={a.rms:.6g}  bound={bounds[k]:.6g}  "
              f"above={100 * a.exceedance(bounds[k]):.3f}%  p999={a.quantile(0.999):.6g}")
    print(f"\nPoint configs/reward/*.yaml at:\n  correction:\n    scales_path: {out}")


if __name__ == "__main__":
    main()
