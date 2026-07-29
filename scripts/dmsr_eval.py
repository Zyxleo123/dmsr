#!/usr/bin/env python
"""Held-out evaluation and the Stage C-vs-D comparison, with plots.

Two modes:

``--mode evaluate``
    Score one or more checkpoints (plus the ``A_plus(y)`` baseline) on a held-out
    box, writing per-crop metrics to ``metrics.csv`` and a summary to
    ``summary.json``. Crops are also tagged with an environment bin so results can
    be stratified by distance from the paired-training environment.

``--mode compare``
    Take the per-crop metric files from several Stage C seeds and several Stage D
    seeds and produce the C-vs-D table with **paired bootstrap** confidence
    intervals on ``metric(D) - metric(C)``.

Statistical discipline (this is the part that is easy to get wrong):

* Bootstrap resampling is **by simulation box**, never by crop. Crops within a box
  share initial conditions and large-scale modes, so crop-level resampling would
  understate the variance by a large factor and manufacture significance.
* With only one held-out box, no independent-box claim is made at all. The script
  prints an explicit warning and labels any interval as a **spatial jackknife
  diagnostic**, not a significance test.
* The C-vs-D decision rule (>= 5% relative error reduction on >= 2 pre-registered
  metrics, at least one conditional, no >5% degradation of the guards) is applied
  mechanically by :func:`decision_rule` so the verdict is not eyeballed.

Examples::

    python scripts/dmsr_eval.py --mode evaluate \\
        --config configs/dmsr/stage_d_critic_alllr.yaml \\
        --ckpt stage_d:runs/dmsr/stage_d_s0/ckpt_best.pt \\
        --ckpt stage_c:runs/dmsr/stage_c_s0/ckpt_best.pt \\
        --baseline --split test --out runs/dmsr/eval_test

    python scripts/dmsr_eval.py --mode compare \\
        --c runs/dmsr/stage_c_s{0,1,2}/eval_test/metrics.csv \\
        --d runs/dmsr/stage_d_s{0,1,2}/eval_test/metrics.csv \\
        --out runs/dmsr/cvd
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cosmo_sr.data.datasets import finite_loader  # noqa: E402
from cosmo_sr.dmsr.data import build_val_dataset, resolve_split  # noqa: E402
from cosmo_sr.dmsr.density import (  # noqa: E402
    CriticInputNormalizer,
    HighPassDensity,
    cellsizes,
)
from cosmo_sr.dmsr.evaluate import (  # noqa: E402
    BandEdges,
    condition_shuffle_gap,
    environment_bin,
    evaluate_batch,
    rk_tk_summary,
    sample_diversity,
)
from cosmo_sr.dmsr.data import LRCropPool  # noqa: E402
from cosmo_sr.dmsr.env import DescriptorStandardizer  # noqa: E402
from cosmo_sr.dmsr.flow import build_flow  # noqa: E402
from cosmo_sr.train import common  # noqa: E402
from cosmo_sr.utils.config import load_config  # noqa: E402

# Pre-registered metrics for the C-vs-D decision (fixed BEFORE looking at results).
PRIMARY_METRICS = ("rk_transition", "squeezed_cross_bispectrum_error",
                   "density_power_error", "bispectrum_error")
CONDITIONAL_METRICS = ("rk_transition", "squeezed_cross_bispectrum_error")
GUARD_METRICS = ("rk_low", "Tk_error_low", "exact_consistency_rel", "sample_diversity")
# Metrics where LOWER is better (everything else is treated as higher-is-better).
LOWER_IS_BETTER = {
    "mse", "density_power_error", "velocity_power_error", "density_pdf_error",
    "velocity_divergence_pdf_error", "bispectrum_error",
    "squeezed_cross_bispectrum_error", "exact_consistency_rel", "exact_consistency_abs",
    "Tk_error_low", "Tk_error_transition", "Tk_error_high",
}


class _UpsampleBaseline:
    """``x_hat = A_plus(y)``: the parameter-free floor, with the flow's interface."""

    def __init__(self, operator):
        self.operator = operator

    def generate(self, y, n_steps=None, z=None, bp_steps=None):
        return self.operator.A_plus(y)

    def sample_residual(self, y, n_steps=None, z=None, bp_steps=None):
        return torch.zeros_like(self.operator.A_plus(y))



def build_env_reference(cfg, split, n_crops: int = 1024):
    """Reference for environment stratification, fitted on **paired training** crops.

    Returns ``(standardizer, mean, cov_inv, edges, descriptor_kwargs)``. Held-out crops
    are then binned by Mahalanobis distance from the paired-training environment
    distribution, with edges at the paired 50th/90th percentiles:

        bin 0 = paired core, bin 1 = paired periphery,
        bin 2 = beyond the paired 90th percentile -- "under-represented but still
                supported", which is where criterion 3 expects Stage D to gain most.

    Percentile edges (rather than fixed distances) guarantee every bin is populated.
    """
    dcfg = cfg.get("data", {})
    dk = dict(cfg.get("env", {}).get("descriptor", {}))
    dk["cellsize"] = cellsizes(dcfg, int(cfg.get("factor", 8)))[1]
    pool = LRCropPool(
        split.train_lr, crop_lr=int(dcfg.get("crop_lr", 8)), n_crops=int(n_crops),
        seed=0, channels=int(dcfg.get("channels", 6)),
        use_channels=dcfg.get("use_channels"), mmap=bool(dcfg.get("mmap", True)),
        descriptor_kwargs=dk,
    )
    desc = pool.descriptors()
    std = DescriptorStandardizer.fit(desc)
    z = std.transform(desc)
    mean = z.mean(axis=0)
    cov = np.cov(z, rowvar=False)
    cov = np.atleast_2d(cov) + 1e-8 * np.eye(z.shape[1])
    cov_inv = np.linalg.inv(cov)
    d = z - mean[None, :]
    m = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, cov_inv, d), 0.0))
    edges = [float(np.percentile(m, 50)), float(np.percentile(m, 90))]
    print(f"[env] stratification edges (paired p50/p90 Mahalanobis): "
          f"{edges[0]:.3f}, {edges[1]:.3f}; descriptors={std.kept_names}")
    return std, mean, cov_inv, edges, dk


def summarise_by_env(rows, keys=("rk_transition", "rk_high", "squeezed_cross_bispectrum_error",
                                 "density_power_error", "sample_diversity")):
    """Per-model, per-environment-bin means of the central metrics."""
    out = {}
    for m in sorted({r["model"] for r in rows}):
        for b in sorted({int(r["env_bin"]) for r in rows if "env_bin" in r}):
            sel = [r for r in rows if r["model"] == m and int(r.get("env_bin", -1)) == b]
            if not sel:
                continue
            out[(m, b)] = {"n": len(sel), **{
                k: float(np.mean([r[k] for r in sel
                                  if isinstance(r.get(k), float) and np.isfinite(r[k])]))
                if any(isinstance(r.get(k), float) and np.isfinite(r[k]) for r in sel)
                else float("nan")
                for k in keys}}
    return out


def load_flow(cfg, channels, ckpt_path, device, use_ema=True, deterministic=None):
    # Experiment E: the checkpoint is a MeanInnovationFlow (frozen deterministic
    # mean + trained innovation flow), whose state dict spans `mean_model.` and
    # `innovation.`. A plain NullSpaceFlow cannot load it, and evaluating without
    # the mean term would drop the predictable conditional mean from x_hat. Build
    # the mean+innovation shell and load the checkpoint's full state directly; the
    # EMA already contains the (unchanged, frozen) mean weights.
    if cfg.get("mean_innovation", {}).get("enabled"):
        from cosmo_sr.dmsr.mean_innovation import build_mean_innovation  # noqa: E402
        flow = build_mean_innovation(cfg, channels, device, load_ckpts=False)
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = blob.get("extra", {}).get("ema") if use_ema else None
        if state is None:
            state = blob["model"]
            print(f"  [{Path(ckpt_path).parent.name}] no EMA weights found; using raw model")
        flow.load_state_dict(state)
        flow.eval()
        return flow
    flow = build_flow(cfg, channels).to(device)
    # Must match how the checkpoint was trained; otherwise a `det` model gets
    # integrated as a velocity field and scores nonsense. Inferred from the
    # checkpoint's recorded stage when not passed explicitly.
    if deterministic is None:
        blob_stage = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        deterministic = blob_stage.get("extra", {}).get("stage") == "det"
    flow.deterministic = bool(deterministic)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = blob.get("extra", {}).get("ema") if use_ema else None
    if state is None:
        state = blob["model"]
        print(f"  [{Path(ckpt_path).parent.name}] no EMA weights found; using raw model")
    flow.load_state_dict(state)
    flow.eval()
    return flow


def evaluate_model(name, model, loader, cfg, device, factor, highpass, is_baseline=False,
                   env_ref=None):
    ecfg = cfg.get("eval", {})
    bands = BandEdges(float(ecfg.get("low_frac", 0.5)), float(ecfg.get("high_frac", 1.5)))
    n_steps = int(ecfg.get("n_steps", 20))
    rows: List[Dict[str, float]] = []

    for i, batch in enumerate(loader):
        y, x = batch["lr"].to(device), batch["hr"].to(device)
        box_idx = -1
        if "box" in batch:
            b = batch["box"]
            # evaluate_batch reduces the whole batch to ONE metric dict, so a batch
            # spanning two boxes would produce a number belonging to neither and
            # silently corrupt the box-level bootstrap. GridCropDataset is iterated
            # unshuffled, so this only happens at box boundaries with batch_size > 1.
            if int(b.min()) != int(b.max()):
                raise ValueError(
                    f"batch spans boxes {int(b.min())}..{int(b.max())}; use "
                    "--batch-size 1 for evaluation so each metric belongs to one box"
                )
            box_idx = int(b[0])
        with torch.no_grad():
            if is_baseline:
                x_hat = model.generate(y)
                op = model.operator
                abs_e, rel_e = op.consistency_error(x_hat, y)
                m = {"exact_consistency_abs": abs_e, "exact_consistency_rel": rel_e,
                     "mse": float((x_hat - x).pow(2).mean())}
                m.update(rk_tk_summary(x_hat, x, factor, bands=bands))
                rho_hat, rho_true = highpass.density(x_hat), highpass.density(x)
                from cosmo_sr.dmsr.evaluate import power_error, pdf_error
                m["density_power_error"] = power_error(rho_hat, rho_true)
                m["density_pdf_error"] = pdf_error(rho_hat, rho_true)
                m["sample_diversity"] = 0.0        # deterministic by construction
            else:
                m = evaluate_batch(model, y, x, factor, highpass, n_steps=n_steps, bands=bands)
                m.update(sample_diversity(model, y,
                                          n_samples=int(ecfg.get("diversity_samples", 3)),
                                          n_steps=n_steps))
                m.update(condition_shuffle_gap(model, y, x, factor, n_steps=n_steps))
        row = {"model": name, "crop": i, "box": box_idx, **m}
        if env_ref is not None:
            std, mean, cov_inv, edges, dk = env_ref
            row["env_bin"] = int(environment_bin(y, std, mean, cov_inv, edges, dk)[0])
        rows.append(row)
        print(f"  [{name}] crop {i}: rk_transition={m.get('rk_transition', float('nan')):.4f}")
    return rows


def cmd_evaluate(args):
    cfg = load_config(args.config)
    device = common.select_device(None)
    dcfg = cfg.get("data", {})
    use_channels = dcfg.get("use_channels")
    channels = len(use_channels) if use_channels else int(dcfg.get("channels", 6))
    factor = int(cfg.get("factor", 8))

    split = resolve_split(dcfg)
    ds = build_val_dataset(split, crop_lr=int(dcfg.get("crop_lr", 8)), scale_factor=factor,
                           channels=int(dcfg.get("channels", 6)), use_channels=use_channels,
                           mmap=bool(dcfg.get("mmap", True)),
                           max_crops=args.max_crops, which=args.split)
    loader = finite_loader(ds, args.batch_size)
    hr_cellsize, lr_cellsize = cellsizes(dcfg, factor)
    print(f"[geom] HR cellsize={hr_cellsize:.4f} kpc/h  LR cellsize={lr_cellsize:.4f} kpc/h")
    highpass = HighPassDensity(
        factor=factor, lowpass=str(cfg.get("critic", {}).get("lowpass", "blockavg")),
        cellsize=hr_cellsize,
        dis_norm=float(dcfg.get("dis_norm", 6000.0)),
    ).to(device)

    boxes = getattr(split, f"{args.split}_hr")
    print(f"[eval] split={args.split} boxes={[Path(b).stem for b in boxes]} "
          f"crops={len(ds)}")
    if len(boxes) < 2:
        print("[eval] WARNING: only one held-out box. Independent-box significance "
              "CANNOT be claimed; any interval below is a spatial-jackknife diagnostic.")

    env_ref = None
    if not args.no_env_strat:
        try:
            env_ref = build_env_reference(cfg, split)
        except Exception as e:
            print(f"[env] stratification unavailable ({e}); continuing without it")

    rows: List[Dict[str, float]] = []
    if args.baseline:
        from cosmo_sr.dmsr.operator import NullSpaceOperator
        base = _UpsampleBaseline(NullSpaceOperator(factor=factor).to(device))
        rows += evaluate_model("baseline_upsample", base, loader, cfg, device,
                               factor, highpass, is_baseline=True, env_ref=env_ref)
    for spec in args.ckpt or []:
        name, path = spec.split(":", 1)
        flow = load_flow(cfg, channels, path, device, use_ema=not args.no_ema)
        rows += evaluate_model(name, flow, loader, cfg, device, factor, highpass,
                               env_ref=env_ref)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "metrics.csv", rows)
    summary = _summarise(rows)
    by_env = summarise_by_env(rows) if env_ref is not None else {}
    with open(out / "summary.json", "w") as f:
        json.dump({"split": args.split,
                   "boxes": [Path(b).stem for b in boxes],
                   "n_crops": len(ds), "summary": summary,
                   "env_edges": env_ref[3] if env_ref else None,
                   "by_env": {f"{m}|bin{b}": v for (m, b), v in by_env.items()}}, f, indent=2)
    _print_table(summary)
    if by_env:
        print("Environment-stratified (bin 0 = paired core, 2 = under-represented "
              "but supported):")
        hdr = f"{'model':<22}{'bin':>5}{'n':>5}" + "".join(
            f"{k[:16]:>18}" for k in ("rk_transition", "rk_high",
                                      "squeezed_cross_bispectrum_error",
                                      "density_power_error", "sample_diversity"))
        print(hdr); print("-" * len(hdr))
        for (m, b), v in sorted(by_env.items()):
            print(f"{m:<22}{b:>5}{v['n']:>5}" + "".join(
                f"{v.get(k, float('nan')):>18.5g}" for k in
                ("rk_transition", "rk_high", "squeezed_cross_bispectrum_error",
                 "density_power_error", "sample_diversity")))
        print()
    try:
        _plots(rows, out)
        print(f"[eval] plots -> {out}")
    except Exception as e:
        print(f"[eval] plotting skipped ({e})")
    print(f"[eval] wrote {out/'metrics.csv'} and {out/'summary.json'}")


def _write_csv(path, rows):
    import csv

    if not rows:
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _metric_keys(rows) -> List[str]:
    """Union of metric keys across ALL rows, in first-seen order.

    Deriving the key set from ``rows[0]`` alone silently drops every metric that
    the first model does not emit. The ``A_plus(y)`` baseline skips the bispectra,
    velocity metrics, diversity and condition-shuffle, and it is listed first --
    so a first-row key set quietly deleted ``bispectrum_error`` and
    ``squeezed_cross_bispectrum_error`` from the summary. Those are 2 of the 4
    pre-registered primary metrics and 1 of the 2 conditional ones, and
    ``decision_rule`` skips non-finite entries, so the C-vs-D verdict would have
    been decided by a quietly reduced metric set.
    """
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen and k not in _NON_METRIC:
                seen.add(k)
                keys.append(k)
    return keys


_NON_METRIC = ("model", "crop", "env_bin", "box")


def _summarise(rows) -> Dict[str, Dict[str, float]]:
    models = sorted({r["model"] for r in rows})
    keys = _metric_keys(rows)
    out = {}
    for m in models:
        sel = [r for r in rows if r["model"] == m]
        summary = {}
        for k in keys:
            # Drop NaNs explicitly rather than leaning on nanmean, which warns
            # ("Mean of empty slice") when a model emits the key but never a
            # finite value -- noise that hides real warnings.
            vals = [r[k] for r in sel
                    if k in r and isinstance(r[k], float) and np.isfinite(r[k])]
            summary[k] = float(np.mean(vals)) if vals else float("nan")
        out[m] = summary
    return out


def _print_table(summary):
    cols = ["rk_low", "rk_transition", "rk_high", "Tk_error_transition",
            "density_power_error", "bispectrum_error",
            "squeezed_cross_bispectrum_error", "sample_diversity",
            "exact_consistency_rel"]
    header = f"{'model':<24}" + "".join(f"{c[:14]:>16}" for c in cols)
    print("\n" + header); print("-" * len(header))
    for m, vals in summary.items():
        line = f"{m:<24}" + "".join(f"{vals.get(c, float('nan')):>16.5g}" for c in cols)
        print(line)
    print()


# --------------------------------------------------------------------------- #
# C vs D comparison
# --------------------------------------------------------------------------- #
def _read_csv(path) -> List[Dict[str, float]]:
    import csv

    with open(path) as f:
        out = []
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = v
            out.append(row)
        return out


def box_level_stats(c_runs, d_runs, keys, n, seed, n_boot: int = 10000):
    """Paired bootstrap resampling **simulation boxes**, not crops or seeds.

    This is the estimator the design actually asks for. Crops inside one box share
    initial conditions and large-scale modes, so resampling crops understates the
    variance badly; and resampling seeds measures optimisation noise, not
    generalisation to new universes. Here each box contributes one paired
    ``D - C`` value (crops averaged within the box, then seeds averaged), and the
    bootstrap resamples boxes with replacement.

    Requires >= 2 held-out boxes; returns ``{}`` otherwise so the caller can fall
    back to the seed-level interval and say so.
    """
    boxes = sorted({int(r["box"]) for run in (c_runs[:n] + d_runs[:n]) for r in run
                    if int(r.get("box", -1)) >= 0})
    if len(boxes) < 2:
        return {}
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        def per_box(runs):
            vals = []
            for b in boxes:
                per_seed = []
                for run in runs[:n]:
                    v = [r[k] for r in run if int(r.get("box", -1)) == b
                         and isinstance(r.get(k), float) and np.isfinite(r[k])]
                    if v:
                        per_seed.append(float(np.mean(v)))
                vals.append(float(np.mean(per_seed)) if per_seed else np.nan)
            return np.array(vals, dtype=float)

        c_box, d_box = per_box(c_runs), per_box(d_runs)
        ok = np.isfinite(c_box) & np.isfinite(d_box)
        if ok.sum() < 2:
            continue
        diff = d_box[ok] - c_box[ok]
        idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
        boots = diff[idx].mean(axis=1)
        lower_better = k in LOWER_IS_BETTER
        denom = abs(float(c_box[ok].mean()))
        rel = ((-diff.mean() / denom if lower_better else diff.mean() / denom)
               if denom > 0 else np.nan)
        out[k] = {
            "n_boxes": int(ok.sum()),
            "C_mean": float(c_box[ok].mean()), "D_mean": float(d_box[ok].mean()),
            "diff_D_minus_C": float(diff.mean()),
            "diff_sd_across_boxes": float(diff.std(ddof=1)),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "rel_improvement": float(rel), "lower_is_better": lower_better,
        }
    return out


def paired_bootstrap(
    c_vals: np.ndarray, d_vals: np.ndarray, n_boot: int = 10000, seed: int = 0
):
    """Bootstrap CI for ``mean(D) - mean(C)`` over paired units (boxes or seeds)."""
    rng = np.random.default_rng(seed)
    diff = d_vals - c_vals
    n = len(diff)
    if n < 2:
        return float(diff.mean()), (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    return float(diff.mean()), (float(np.percentile(boots, 2.5)),
                                float(np.percentile(boots, 97.5)))


def decision_rule(stats: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    """Apply the pre-registered C-vs-D success criteria mechanically."""
    improved = []
    for m in PRIMARY_METRICS:
        s = stats.get(m)
        if not s or not np.isfinite(s["rel_improvement"]):
            continue
        if s["rel_improvement"] >= 0.05:
            improved.append(m)

    degraded = []
    for m in GUARD_METRICS:
        s = stats.get(m)
        if not s or not np.isfinite(s["rel_improvement"]):
            continue
        if s["rel_improvement"] <= -0.05:
            degraded.append(m)

    has_conditional = any(m in CONDITIONAL_METRICS for m in improved)
    core = len(improved) >= 2 and has_conditional and not degraded

    return {
        "improved_metrics": improved,
        "degraded_guards": degraded,
        "has_conditional_improvement": has_conditional,
        "verdict": ("core success" if core else
                    "failure of the LR-only hypothesis" if not improved else
                    "partial / inconclusive"),
        "note": ("Core success requires >=2 pre-registered metrics improved by >=5% "
                 "relative, at least one conditional, and no guard metric degraded "
                 "by >5%."),
    }



def _paired_stats(c_runs, d_runs, keys, n, seed, row_filter=None):
    """Per-metric seed-level paired stats. ``row_filter`` restricts to a subset
    of crops (used for the per-environment-bin breakdown)."""
    stats: Dict[str, Dict[str, float]] = {}
    for k in keys:
        def seed_means(runs):
            out = []
            for run in runs[:n]:
                sel = [r for run_r in [run] for r in run_r
                       if (row_filter is None or row_filter(r))
                       and isinstance(r.get(k), float) and np.isfinite(r[k])]
                out.append(np.mean(sel and [r[k] for r in sel]) if sel else np.nan)
            return np.array(out, dtype=float)

        c_seed, d_seed = seed_means(c_runs), seed_means(d_runs)
        if not (np.isfinite(c_seed).all() and np.isfinite(d_seed).all()):
            continue
        mean_diff, (lo, hi) = paired_bootstrap(c_seed, d_seed, seed=seed)
        lower_better = k in LOWER_IS_BETTER
        denom = abs(c_seed.mean())
        rel = ((-mean_diff / denom if lower_better else mean_diff / denom)
               if denom > 0 else np.nan)
        stats[k] = {
            "C_mean": float(c_seed.mean()),
            "C_std": float(c_seed.std(ddof=1)) if n > 1 else 0.0,
            "D_mean": float(d_seed.mean()),
            "D_std": float(d_seed.std(ddof=1)) if n > 1 else 0.0,
            "diff_D_minus_C": mean_diff, "ci_low": lo, "ci_high": hi,
            "rel_improvement": float(rel), "lower_is_better": lower_better,
            "n_seed_pairs": n,
        }
    return stats


def cmd_compare(args):
    c_runs = [_read_csv(p) for p in args.c]
    d_runs = [_read_csv(p) for p in args.d]
    if len(c_runs) != len(d_runs):
        print(f"[compare] WARNING: {len(c_runs)} C seeds vs {len(d_runs)} D seeds; "
              "pairing by index over the shorter list")
    n = min(len(c_runs), len(d_runs))
    if n < 3:
        print(f"[compare] WARNING: only {n} seed pair(s); the design asks for >= 3 "
              "for Stages C and D")

    # Union across every row of every run -- not just the first row of the first
    # run. See _metric_keys: a first-row key set silently drops metrics, and here
    # that would quietly shrink the pre-registered decision set.
    all_rows = [r for run in (c_runs[:n] + d_runs[:n]) for r in run]
    keys = [k for k in _metric_keys(all_rows)
            if any(isinstance(r.get(k), float) for r in all_rows)]

    missing = [k for k in PRIMARY_METRICS + GUARD_METRICS if k not in keys]
    if missing:
        print(f"[compare] WARNING: pre-registered metric(s) absent from the inputs: "
              f"{missing}. The decision rule will be evaluated on a REDUCED set.")
    stats = _paired_stats(c_runs, d_runs, keys, n, args.seed)

    print(f"\n{'metric':<38}{'C mean+-sd':>20}{'D mean+-sd':>20}"
          f"{'D-C':>12}{'95% CI':>26}{'rel':>9}")
    print("-" * 125)
    for k in PRIMARY_METRICS + GUARD_METRICS:
        if k not in stats:
            continue
        s = stats[k]
        print(f"{k:<38}{s['C_mean']:>12.5g}+-{s['C_std']:<6.3g}"
              f"{s['D_mean']:>12.5g}+-{s['D_std']:<6.3g}"
              f"{s['diff_D_minus_C']:>12.4g}"
              f"   [{s['ci_low']:>10.4g},{s['ci_high']:>10.4g}]"
              f"{100 * s['rel_improvement']:>8.1f}%")

    # --- BOX-LEVEL bootstrap: the estimator the design actually asks for ---------
    box_stats = box_level_stats(c_runs, d_runs, keys, n, args.seed)
    if box_stats:
        nb = max(v["n_boxes"] for v in box_stats.values())
        print(f"\n=== BOX-LEVEL paired bootstrap over {nb} held-out boxes "
              f"(resamples BOXES, not crops or seeds) ===")
        print(f"{'metric':<38}{'C mean':>12}{'D mean':>12}{'D-C':>12}"
              f"{'sd(boxes)':>11}{'95% CI':>26}{'rel':>9}")
        print("-" * 120)
        for k in PRIMARY_METRICS + GUARD_METRICS:
            if k not in box_stats:
                continue
            v = box_stats[k]
            print(f"{k:<38}{v['C_mean']:>12.5g}{v['D_mean']:>12.5g}"
                  f"{v['diff_D_minus_C']:>12.4g}{v['diff_sd_across_boxes']:>11.4g}"
                  f"   [{v['ci_low']:>10.4g},{v['ci_high']:>10.4g}]"
                  f"{100 * v['rel_improvement']:>8.1f}%")
        print("This interval supports an independent-box significance claim; the "
              "seed-level table above does not.")
    else:
        print("\n[compare] NOTE: fewer than 2 held-out boxes present in the inputs -- "
              "no box-level interval. Seed-level CIs only; per the design, do NOT "
              "claim independent-box significance.")

    # The verdict is decided on the BOX-level estimator when available, since that
    # is what generalisation to new universes means here.
    verdict = decision_rule(box_stats or stats)
    verdict["estimator"] = "box-level" if box_stats else "seed-level"
    print(f"\n[compare] improved (>=5%): {verdict['improved_metrics'] or 'none'}")
    print(f"[compare] degraded guards (> 5%): {verdict['degraded_guards'] or 'none'}")
    print(f"[compare] VERDICT: {verdict['verdict'].upper()} "
          f"(decided on the {verdict['estimator']} estimator)")
    print("[compare] NOTE: CIs above resample over SEEDS. Independent-box "
          "significance additionally requires >= 2 held-out boxes.")
    if n < 5:
        # With n resampling units there are only C(2n-1, n) distinct bootstrap
        # multisets -- 10 of them at n=3. The 2.5/97.5 percentiles are then
        # essentially the min/max of a handful of values: the interval comes out
        # far too narrow and does NOT have 95% coverage. Report it as a spread,
        # not as a significance statement.
        from math import comb
        n_multisets = comb(2 * n - 1, n)
        extra = f" (only {n_multisets} distinct resamples)"
        print(f"[compare] WARNING: n={n} resampling unit(s){extra}. A percentile "
              "bootstrap is UNRELIABLE at this size -- the intervals above are a "
              "spread indicator, not a 95% coverage claim. Treat the verdict as "
              "provisional and report mean+-sd alongside it.")

    # --- criterion 3: is the gain strongest in under-represented environments? ---
    by_env: Dict[str, Dict[str, Dict[str, float]]] = {}
    env_bins = sorted({int(r["env_bin"]) for r in all_rows if "env_bin" in r})
    if env_bins:
        focus = [k for k in CONDITIONAL_METRICS if k in stats]
        print(f"\nCriterion 3 -- D-C by environment bin "
              f"(0 = paired core, {max(env_bins)} = most under-represented):")
        print(f"{'bin':>5}  {'metric':<38}{'C mean':>12}{'D mean':>12}{'rel':>9}")
        print("-" * 78)
        for b in env_bins:
            st = _paired_stats(c_runs, d_runs, focus, n, args.seed,
                               row_filter=lambda r, _b=b: int(r.get("env_bin", -1)) == _b)
            by_env[str(b)] = st
            for k, v in st.items():
                print(f"{b:>5}  {k:<38}{v['C_mean']:>12.5g}{v['D_mean']:>12.5g}"
                      f"{100 * v['rel_improvement']:>8.1f}%")
        print("NOTE: per 9.0b, the paired and LR-only environment distributions are "
              "indistinguishable at production scale (AUC ~0.50), so criterion 3 is "
              "expected to be weak or vacuous on this dataset. A null here is NOT "
              "evidence against Stage D.")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "cvd.json", "w") as f:
        json.dump({"stats_seed_level": stats, "stats_box_level": box_stats,
                   "decision": verdict, "by_env": by_env}, f, indent=2)
    print(f"[compare] wrote {out / 'cvd.json'}")


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _plots(rows, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({r["model"] for r in rows})
    bands = ["low", "transition", "high"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        sel = [r for r in rows if r["model"] == m]
        rk = [np.nanmean([r.get(f"rk_{b}", np.nan) for r in sel]) for b in bands]
        tk = [np.nanmean([r.get(f"Tk_error_{b}", np.nan) for r in sel]) for b in bands]
        xs = np.arange(len(bands)) + i * width
        axes[0].bar(xs, rk, width, label=m)
        axes[1].bar(xs, tk, width, label=m)
    for ax, title, ylab in ((axes[0], "r(k) by band (higher better)", "r(k)"),
                            (axes[1], "|T(k)-1| by band (lower better)", "T(k) error")):
        ax.set_xticks(np.arange(len(bands)) + 0.4 - width / 2)
        ax.set_xticklabels(bands); ax.set_title(title); ax.set_ylabel(ylab)
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "rk_tk.png", dpi=130); plt.close(fig)

    keys = ["density_power_error", "velocity_power_error", "bispectrum_error",
            "squeezed_cross_bispectrum_error", "sample_diversity",
            "condition_shuffle_gap"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, k in zip(axes.ravel(), keys):
        vals = [np.nanmean([r.get(k, np.nan) for r in rows if r["model"] == m])
                for m in models]
        ax.bar(range(len(models)), vals, color="tab:blue")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
        ax.set_title(k, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "distributional.png", dpi=130); plt.close(fig)

    # correct- vs shuffled-condition
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(models):
        sel = [r for r in rows if r["model"] == m]
        c = np.nanmean([r.get("correct_rk_transition", np.nan) for r in sel])
        s = np.nanmean([r.get("shuffled_rk_transition", np.nan) for r in sel])
        ax.bar(i - 0.2, c, 0.4, color="tab:green", label="correct y" if i == 0 else None)
        ax.bar(i + 0.2, s, 0.4, color="tab:red", label="shuffled y" if i == 0 else None)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("transition-band r(k)"); ax.legend()
    ax.set_title("Condition-use diagnostic")
    fig.tight_layout(); fig.savefig(out / "condition_shuffle.png", dpi=130); plt.close(fig)

    # environment-stratified performance
    if any("env_bin" in r for r in rows):
        be = summarise_by_env(rows)
        bins = sorted({b for (_, b) in be})
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, key in zip(axes, ("rk_transition", "squeezed_cross_bispectrum_error")):
            w = 0.8 / max(len(models), 1)
            for i, m in enumerate(models):
                vals = [be.get((m, b), {}).get(key, np.nan) for b in bins]
                ax.bar(np.arange(len(bins)) + i * w, vals, w, label=m)
            ax.set_xticks(np.arange(len(bins)) + 0.4 - w / 2)
            ax.set_xticklabels([f"bin {b}" for b in bins])
            ax.set_title(f"{key} by environment bin", fontsize=9)
            ax.legend(fontsize=7)
        axes[0].set_xlabel("0 = paired core  ->  2 = under-represented but supported")
        fig.tight_layout(); fig.savefig(out / "env_stratified.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("evaluate", "compare"), default="evaluate")
    ap.add_argument("--config")
    ap.add_argument("--ckpt", action="append", help="name:path/to/ckpt.pt (repeatable)")
    ap.add_argument("--baseline", action="store_true", help="also score A_plus(y)")
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--max-crops", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--no-env-strat", action="store_true",
                    help="skip environment stratification (saves building the paired crop pool)")
    ap.add_argument("--c", nargs="*", default=[], help="Stage C metrics.csv files")
    ap.add_argument("--d", nargs="*", default=[], help="Stage D metrics.csv files")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.mode == "evaluate":
        if not args.config:
            ap.error("--mode evaluate requires --config")
        cmd_evaluate(args)
    else:
        if not args.c or not args.d:
            ap.error("--mode compare requires --c and --d")
        cmd_compare(args)


if __name__ == "__main__":
    main()
