#!/usr/bin/env python
"""CPU stage: read residual_diagnose.jsonl and say which failure it is.

Reads only what the GPU job wrote, so the table and the figure can be redrawn
without resampling. The verdict block is mechanical -- it applies the same three
tests the GPU job was built around and names the one the numbers support:

* **amplitude** -- coherence is real (``r(k)`` in the high band clearly above
  zero) and the amplitude ratio is far from 1. ``residual_scale`` is the fix and
  the sweep says which value.
* **direction** -- ``r(k) ~ 0``. The residual carries no information about the
  correction; shrinking it only reduces the damage, and at ``alpha = 0`` the
  model is exactly SR2. Retraining is required, not rescaling.
* **conditioning** -- the shuffled-conditioning arm scores the same as the
  matched one. The model is not using ``Psi_base``/``y_lr`` at all.

These are not exclusive; the point of printing all three is that the fix differs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from _common import (add_common_args, banner, constraints_of, load_reward_config,
                     read_jsonl, write_json)

from cosmo_sr.reward import paths

# Coherence below this in the high band is indistinguishable from noise for the
# crop counts this job runs; above it, the residual is carrying real signal.
RK_FLOOR = 0.10
# An amplitude ratio inside this band is "right size"; outside it, rescaling has
# somewhere to go.
AMP_OK = (0.7, 1.4)


def _vals(rows: List[Dict], key: str) -> List[float]:
    return [r[key] for r in rows if isinstance(r.get(key), (int, float))
            and math.isfinite(r[key])]


def _mean(rows: List[Dict], key: str) -> float:
    """MEDIAN over crops, despite the name kept for call-site stability.

    Not a style preference. The crop-level density sigma ratio is heavy-tailed:
    over 8 crops of the frozen baseline it reads
    [3.73, 1.38, 1.00, 1.04, 0.59, 0.92, 1.21, 0.60] -- one crop at 3.73 pulls
    the mean to 1.31 while the median is 1.02, and 1.02 is the number that
    agrees with the full-box calibration (0.99) and with train.py's own
    diagnostic (1.0011). Using the mean made the report claim SR2 itself sits
    outside the Gate A band and shifted the "best alpha" it recommends.
    """
    v = _vals(rows, key)
    return float(np.median(v)) if v else float("nan")


def _fmt(x: float, w: int = 9, p: int = 4) -> str:
    return " " * w if x is None or (isinstance(x, float) and math.isnan(x)) \
        else f"{x:{w}.{p}f}"


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--rows", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    cons = constraints_of(cfg)
    out_dir = Path(args.out) if args.out else paths.AUDITS("residual_diagnose",
                                                           create=True)
    rows_path = Path(args.rows) if args.rows else out_dir / "residual_diagnose.jsonl"
    if not rows_path.is_file():
        print(f">>> MISSING: {rows_path}")
        print(">>> produced by: scripts/slurm/residual_diagnose_gpu.sbatch")
        print(">>> nothing to report; exiting 0 so dependents say the same.")
        return

    rows = read_jsonl(rows_path)
    sampled = [r for r in rows if r.get("arm") == "sampled"]
    oracle = [r for r in rows if r.get("arm") == "oracle_true_residual"]
    shuffled = [r for r in rows if r.get("arm") == "shuffled_cond"]
    alphas = sorted({float(r["alpha"]) for r in sampled})
    n_crops = len({r["crop"] for r in rows})

    banner(f"{n_crops} crops, {len(alphas)} alphas, "
           f"{'with' if shuffled else 'no'} conditioning-shuffle arm")

    # --- 1. the amplitude sweep -------------------------------------------
    print("\n=== amplitude sweep: compose Psi_base + alpha * dPsi_hat ===")
    print(f"{'alpha':>7} {'dens_sigma':>11} {'low_k':>9} {'lr_cons':>9} "
          f"{'disp_rk_hi':>11} {'dens_pow_err':>13} {'disp_pow_err':>13}")
    sweep = []
    for a in alphas:
        rs = [r for r in sampled if float(r["alpha"]) == a]
        rec = {
            "alpha": a,
            "density_sigma_ratio": _mean(rs, "density_sigma_ratio"),
            "low_k_change": _mean(rs, "low_k_change"),
            "lr_consistency": _mean(rs, "lr_consistency"),
            "disp_rk_high": _mean(rs, "disp_rk_high"),
            "density_power_error": _mean(rs, "density_power_error"),
            "displacement_power_error": _mean(rs, "displacement_power_error"),
        }
        sweep.append(rec)
        print(f"{a:7.2f} {_fmt(rec['density_sigma_ratio'],11)} "
              f"{_fmt(rec['low_k_change'])} {_fmt(rec['lr_consistency'])} "
              f"{_fmt(rec['disp_rk_high'],11)} "
              f"{_fmt(rec['density_power_error'],13,5)} "
              f"{_fmt(rec['displacement_power_error'],13,5)}")

    orc = {
        "density_sigma_ratio": _mean(oracle, "density_sigma_ratio"),
        "low_k_change": _mean(oracle, "low_k_change"),
        "lr_consistency": _mean(oracle, "lr_consistency"),
        "disp_rk_high": _mean(oracle, "disp_rk_high"),
        "density_power_error": _mean(oracle, "density_power_error"),
        "displacement_power_error": _mean(oracle, "displacement_power_error"),
    }
    print(f"{'TRUE':>7} {_fmt(orc['density_sigma_ratio'],11)} "
          f"{_fmt(orc['low_k_change'])} {_fmt(orc['lr_consistency'])} "
          f"{_fmt(orc['disp_rk_high'],11)} "
          f"{_fmt(orc['density_power_error'],13,5)} "
          f"{_fmt(orc['displacement_power_error'],13,5)}   <- the true residual")
    print(f"{'limit':>7} {'0.8-1.25':>11} {_fmt(cons.low_k_change_max)} "
          f"{_fmt(cons.lr_consistency_error_max)} {'':>11} "
          f"{_fmt(cons.density_power_error_max,13,5)} "
          f"{_fmt(cons.displacement_power_error_max,13,5)}"
          f"   <- calibrated constraints")

    # The row that matters: does any alpha put density back in range?
    best = min(sweep, key=lambda r: abs(r["density_sigma_ratio"] - 1.0)
               if math.isfinite(r["density_sigma_ratio"]) else 1e9)

    # --- 2. coherence ------------------------------------------------------
    print("\n=== coherence: r(k) between dPsi_hat and the TRUE dPsi ===")
    amp = [r.get("coh_amp_ratio_per_channel") for r in sampled
           if r.get("coh_amp_ratio_per_channel")]
    amp_mean = np.mean(np.asarray(amp, dtype=float), axis=0) if amp else None
    coh = {b: _mean(sampled, f"coh_disp_rk_{b}") for b in
           ("low", "transition", "high")}
    print(f"  r(k) low={_fmt(coh['low'],7)} transition={_fmt(coh['transition'],7)}"
          f" high={_fmt(coh['high'],7)}      (1 = perfect, 0 = no information)")
    print(f"  cosine similarity = {_mean(sampled, 'coh_cosine'):+.4f}")
    if amp_mean is not None:
        print(f"  amplitude ratio per channel (pred/true) = "
              f"{[round(float(v), 2) for v in amp_mean]}")
        print(f"    disp {np.mean(amp_mean[:3]):.2f}x   vel {np.mean(amp_mean[3:]):.2f}x")

    # --- 3. conditioning ---------------------------------------------------
    sh = {}
    if shuffled:
        print("\n=== conditioning: same noise, conditioning from another crop ===")
        sh = {b: _mean(shuffled, f"coh_disp_rk_{b}") for b in
              ("low", "transition", "high")}
        print(f"  matched   r(k)_high = {_fmt(coh['high'],7)}")
        print(f"  shuffled  r(k)_high = {_fmt(sh['high'],7)}")
        print(f"  matched   cosine    = {_mean(sampled, 'coh_cosine'):+.4f}")
        print(f"  shuffled  cosine    = {_mean(shuffled, 'coh_cosine'):+.4f}")
        print(f"  sample-vs-matched cosine = "
              f"{_mean(shuffled, 'delta_vs_matched_cosine'):+.4f}"
              f"   (1.0 = the conditioning changed nothing)")

    # --- 4. the clip profile ----------------------------------------------
    prof = [r["clip_profile"] for r in sampled if r.get("clip_profile")]
    if prof:
        p = np.mean(np.asarray(prof, dtype=float), axis=0)
        ts = sampled[0].get("clip_t", list(range(len(p))))
        print("\n=== x0 clip fraction per DDIM step (mean over crops) ===")
        print("  t:    " + " ".join(f"{t:5.2f}" for t in ts[:12]))
        print("  clip: " + " ".join(f"{v:5.2f}" for v in p[:12]))
        n_hot = int((p > 0.5).sum())
        print(f"  steps with >50% of voxels clipped: {n_hot} of {len(p)}")

    # --- 5. verdict --------------------------------------------------------
    print("\n=== verdict ===")
    findings = []
    rk_hi = coh["high"]
    amp_d = float(np.mean(amp_mean[:3])) if amp_mean is not None else float("nan")
    if math.isfinite(rk_hi) and rk_hi < RK_FLOOR:
        findings.append(
            f"DIRECTION: r(k)_high={rk_hi:.3f} < {RK_FLOOR}. The residual carries "
            f"no information about the true correction. Rescaling cannot help -- "
            f"alpha=0 (exactly SR2) is then the best member of the sweep, which "
            f"is not an improvement. Retraining is required.")
    elif math.isfinite(amp_d) and not (AMP_OK[0] <= amp_d <= AMP_OK[1]):
        findings.append(
            f"AMPLITUDE: r(k)_high={rk_hi:.3f} (real signal) but the displacement "
            f"amplitude ratio is {amp_d:.2f}x. The shape is right and the size is "
            f"not. Set residual_scale ~ {1.0 / amp_d:.2f} and re-run Gate A.")
    elif math.isfinite(rk_hi):
        findings.append(
            f"NEITHER: r(k)_high={rk_hi:.3f} and amplitude ratio {amp_d:.2f}x are "
            f"both reasonable. The field damage is not explained by size or "
            f"direction at the crop level; look at the composed-field metrics.")
    if shuffled:
        d = _mean(shuffled, "delta_vs_matched_cosine")
        if math.isfinite(d) and abs(d) > 0.9:
            findings.append(
                f"CONDITIONING: shuffling Psi_base/y_lr changed the sample by "
                f"cosine {d:+.3f} -- i.e. barely at all. The model is not using "
                f"its conditioning; it has learned unconditional denoising.")
        elif math.isfinite(d):
            findings.append(
                f"conditioning IS used: shuffling it moved the sample "
                f"(cosine {d:+.3f}).")
    if math.isfinite(best["density_sigma_ratio"]):
        findings.append(
            f"best alpha for density is {best['alpha']:.2f} "
            f"(sigma ratio {best['density_sigma_ratio']:.3f}); "
            f"the true residual scores {orc['density_sigma_ratio']:.3f}.")
    if math.isfinite(orc["low_k_change"]) and cons.low_k_change_max is not None \
            and orc["low_k_change"] > cons.low_k_change_max:
        findings.append(
            f"NOTE: the TRUE residual scores low_k_change="
            f"{orc['low_k_change']:.3f}, above the calibrated ceiling of "
            f"{cons.low_k_change_max:.3f}. The constraint would reject ground "
            f"truth and must be re-derived before it gates anything.")
    for f in findings:
        print(f"  * {f}")

    summary = {"sweep": sweep, "oracle": orc, "coherence": coh,
               "coherence_shuffled": sh,
               "amp_ratio_per_channel": None if amp_mean is None
               else [float(v) for v in amp_mean],
               "cosine": _mean(sampled, "coh_cosine"),
               "cosine_shuffled": _mean(shuffled, "coh_cosine") if shuffled else None,
               "clip_profile": None if not prof else [float(v) for v in p],
               "best_alpha_for_density": best["alpha"],
               "n_crops": n_crops, "findings": findings,
               "constraints": {"low_k_change_max": cons.low_k_change_max,
                               "density_power_error_max": cons.density_power_error_max,
                               "lr_consistency_error_max": cons.lr_consistency_error_max}}
    write_json(out_dir / "residual_diagnose_summary.json", summary)
    banner(f"summary -> {out_dir / 'residual_diagnose_summary.json'}")

    if not args.no_figure:
        _figure(sweep, orc, coh, sh, p if prof else None, cons,
                out_dir / "residual_diagnose.png")
        print(f"=== figure -> {out_dir / 'residual_diagnose.png'}")


def _figure(sweep, orc, coh, sh, prof, cons, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    a = [r["alpha"] for r in sweep]

    ax[0].plot(a, [r["density_sigma_ratio"] for r in sweep], "o-", label="sampled")
    ax[0].axhline(orc["density_sigma_ratio"], color="k", ls="--",
                  label="true residual")
    ax[0].axhspan(0.8, 1.25, color="green", alpha=0.12, label="Gate A band")
    ax[0].set_xlabel("residual_scale alpha"); ax[0].set_ylabel("density sigma ratio")
    ax[0].set_title("Does shrinking restore the field?"); ax[0].legend(fontsize=8)

    ax[1].plot(a, [r["low_k_change"] for r in sweep], "o-", label="sampled")
    ax[1].axhline(orc["low_k_change"], color="k", ls="--", label="true residual")
    if cons.low_k_change_max:
        ax[1].axhline(cons.low_k_change_max, color="r", ls=":",
                      label="calibrated ceiling")
    ax[1].set_xlabel("residual_scale alpha"); ax[1].set_ylabel("low_k_change")
    ax[1].set_title("Large-scale contamination"); ax[1].legend(fontsize=8)

    bands = ["low", "transition", "high"]
    x = np.arange(len(bands))
    ax[2].bar(x - 0.2, [coh.get(b, np.nan) for b in bands], 0.4, label="matched")
    if sh:
        ax[2].bar(x + 0.2, [sh.get(b, np.nan) for b in bands], 0.4,
                  label="shuffled cond")
    ax[2].axhline(RK_FLOOR, color="r", ls=":", label="noise floor")
    ax[2].set_xticks(x); ax[2].set_xticklabels(bands)
    ax[2].set_ylabel("r(k), residual vs TRUE residual")
    ax[2].set_title("Is it pointing the right way?"); ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
