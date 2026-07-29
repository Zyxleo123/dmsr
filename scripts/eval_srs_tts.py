#!/usr/bin/env python
"""Stage 1: oracle best-of-K audit for the pretrained SR2/SRS generator.

Asks one question: **does SR2's injected noise carry usable selection leverage?**
For each held-out box we draw ``K`` full-box realisations that differ only in
noise, score every one against the true HR box, and compare

  * ``random``      -- expected quality of a single draw (the K=1 baseline),
  * ``phase``       -- an oracle that peeks at HR ``r(k)`` / ``T(k)``,
  * ``statistical`` -- an oracle that uses only distributional / higher-order
    agreement (density power, PDF, bispectra, velocity), i.e. no realisation-
    specific high-k phase.

If the ``statistical`` curve is flat in ``K``, no verifier can help and the whole
programme stops here. That is a real possible outcome: the pretrained ``G_z0``
has ``|std|`` of order 1e-3 at five of its six noise sites.

Two phases, so the expensive part is done once and never repeated:

    generate : run the generator, write one JSONL row per (box, candidate)
    analyze  : fit the normaliser on val boxes, build curves, CIs and plots

Example (GPU node)::

    python scripts/eval_srs_tts.py \
        --lr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr \
        --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr \
        --k-values 1 2 4 8 16 32 --seeds 0 1 2 3 ... 31 \
        --val-boxes set8 set9 set10 set11 \
        --test-boxes set12 set13 set14 set15 \
        --out runs/tts_oracle
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def box_paths(lr_dir: str, hr_dir: str, boxes: Optional[Sequence[str]]) -> List[tuple]:
    lr_d, hr_d = Path(lr_dir), Path(hr_dir)
    if lr_d.is_file():
        return [(lr_d.stem, lr_d, Path(hr_dir))]
    names = sorted(p.stem for p in lr_d.glob("*.npy"))
    if boxes:
        missing = sorted(set(boxes) - set(names))
        if missing:
            raise SystemExit(f"boxes not found in {lr_d}: {missing}")
        names = [n for n in names if n in boxes]
    return [(n, lr_d / f"{n}.npy", hr_d / f"{n}.npy") for n in names]


def append_rows(path: Path, rows: Sequence[Dict]) -> None:
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _chan_kwargs(args) -> Dict[str, int]:
    """Channel widths, overridable so smoke tests can run a tiny generator."""
    return {"chan_base": args.chan_base, "chan_min": args.chan_min, "chan_max": args.chan_max}


def _peak_memory_gb(device) -> float:
    """Peak allocation since the last reset, in GB (CUDA), else host RSS."""
    import torch

    if getattr(device, "type", "") == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        torch.cuda.reset_peak_memory_stats(device)
        return round(float(peak), 3)
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)
    except Exception:  # pragma: no cover - platform dependent
        return float("nan")


# --------------------------------------------------------------------------- #
# Generation phase
# --------------------------------------------------------------------------- #
def _hr_reference(args, geometry, device, cache: Path):
    """Training-HR density power/PDF reference for the plausibility features.

    Fitted on **training** boxes only, so a candidate's plausibility score never
    sees statistics of the box it is being scored on.
    """
    import torch
    from cosmo_sr.tts.features import HRReference
    from cosmo_sr.tts.metrics import cic_density_slabs

    if cache.exists():
        return HRReference.load(cache)
    train = [b for b in (args.train_boxes or [])]
    if not train:
        return None
    densities = []
    for name, _lr_path, hr_path in box_paths(args.lr, args.hr, train):
        hr = torch.from_numpy(np.load(hr_path).astype(np.float32)).unsqueeze(0).to(device)
        geo = geometry.for_grid(hr.shape[-1])
        densities.append(cic_density_slabs(hr[:, 0:3], geo.cellsize, geo.dis_norm,
                                           slab=args.slab))
        del hr
        print(f"[reference] {name}", flush=True)
    ref = HRReference.fit(densities, n_bins=args.n_bins)
    ref.save(cache)
    return ref


def run_generate(args) -> None:
    import torch
    from cosmo_sr.tts.features import candidate_features, equivariance_features
    from cosmo_sr.tts.metrics import (
        DensityGeometry, MomentAccumulator, candidate_metrics,
        cic_density_slabs, density_profile,
    )
    from cosmo_sr.tts.sampling import iter_srs_candidates
    from cosmo_sr.tts.srs_noise import load_controlled_generator

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    profiles_path = out / "profiles.npz"
    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    G = load_controlled_generator(args.model, scale_factor=args.scale, device=device,
                                  **_chan_kwargs(args))
    geometry = DensityGeometry(boxsize=args.boxsize, ng=args.scale * 64, dis_norm=args.dis_norm)
    reference = _hr_reference(args, geometry, device, out / "hr_reference.npz")

    done = {(r["box"], r["seed"]) for r in read_rows(rows_path)}
    if done:
        print(f"[resume] {len(done)} (box, seed) rows already present", flush=True)
    profiles: Dict[str, np.ndarray] = {}
    if profiles_path.exists():
        profiles = dict(np.load(profiles_path))

    box_summary: Dict[str, Dict] = {}
    summary_path = out / "box_summary.json"
    if summary_path.exists():
        box_summary = json.loads(summary_path.read_text())

    for name, lr_path, hr_path in box_paths(args.lr, args.hr, args.boxes):
        lr_np = np.load(lr_path).astype(np.float32)
        hr_t = torch.from_numpy(np.load(hr_path).astype(np.float32)).unsqueeze(0)
        lr_t = torch.from_numpy(lr_np).unsqueeze(0)
        n_hr = hr_t.shape[-1]
        if n_hr != lr_np.shape[-1] * args.scale:
            raise SystemExit(f"{name}: HR {n_hr} != LR {lr_np.shape[-1]} x {args.scale}")

        hr_dev = hr_t.to(device)
        rho_hr = cic_density_slabs(
            hr_dev[:, 0:3], geometry.for_grid(n_hr).cellsize, geometry.dis_norm, slab=args.slab
        )
        lr_dev = lr_t.to(device)
        acc_disp, acc_vel, acc_rho = MomentAccumulator(), MomentAccumulator(), MomentAccumulator()

        seeds = [s for s in args.seeds if (name, int(s)) not in done]
        if not seeds:
            print(f"[{name}] all {len(args.seeds)} seeds done, skipping", flush=True)
        tile_hr = (lr_np.shape[-1] // args.nsplit) * args.scale
        t_box = time.time()
        for cand in iter_srs_candidates(
            G, lr_np, seeds, nsplit=args.nsplit, pad=args.pad, scale_factor=args.scale,
            device=device, noise_mode=args.noise_mode, box=name, model_path=args.model,
            progress=args.progress,
        ):
            t0 = time.time()
            sr = torch.from_numpy(cand.field).unsqueeze(0).to(device)
            rho_sr = cic_density_slabs(
                sr[:, 0:3], geometry.for_grid(n_hr).cellsize, geometry.dis_norm, slab=args.slab
            )
            m = candidate_metrics(
                sr, hr_dev, lr_dev, factor=args.scale, geometry=geometry,
                n_bins=args.n_bins, tile_size=tile_hr, rho_sr=rho_sr, rho_hr=rho_hr,
            )
            prof = density_profile(rho_sr, n_bins=args.n_bins)
            profiles[f"{name}|{cand.seed}|pk"] = prof["density_pk"]
            profiles[f"{name}|{cand.seed}|pdf"] = prof["log_density_pdf"]

            if args.features:
                extra = {}
                if args.equivariance:
                    extra.update(equivariance_features(
                        G, lr_np, cand.seed, args.nsplit, args.pad, args.scale,
                        device=device, n_probes=args.n_probes, noise_mode=args.noise_mode,
                    ))
                m.update(candidate_features(
                    sr, lr_dev, factor=args.scale, geometry=geometry, reference=reference,
                    rho=rho_sr, tile_size=tile_hr, n_bins=args.n_bins, extra=extra,
                    slab=args.slab,
                ))

            acc_disp.add(sr[:, 0:3]); acc_vel.add(sr[:, 3:6]); acc_rho.add(rho_sr)
            row = {"box": name, "seed": int(cand.seed), "wall_s": round(time.time() - t0, 2),
                   "peak_mem_gb": _peak_memory_gb(device)}
            row.update({k: float(v) for k, v in m.items()})
            row.update({"config_" + k: v for k, v in cand.config.items()})
            append_rows(rows_path, [row])
            del sr, rho_sr, prof
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[{name}] seed {cand.seed} done in {row['wall_s']}s", flush=True)

        if seeds:
            box_summary[name] = {
                **acc_disp.summary("disp_"), **acc_vel.summary("vel_"),
                **acc_rho.summary("density_"),
                "n_candidates": acc_disp.n, "wall_s": round(time.time() - t_box, 1),
            }
            summary_path.write_text(json.dumps(box_summary, indent=2))
            np.savez_compressed(profiles_path, **profiles)
        del hr_dev, lr_dev, rho_hr, acc_disp, acc_vel, acc_rho
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"wrote {rows_path} ({len(read_rows(rows_path))} rows)")


# --------------------------------------------------------------------------- #
# Analysis phase
# --------------------------------------------------------------------------- #
#: Headline metrics reported against K. Chosen to span the failure modes:
#: density morphology, higher order, velocity, and the phase ceiling.
REPORT_METRICS = (
    "density_power_error", "density_pdf_error", "bispectrum_equilateral_error",
    "bispectrum_squeezed_error", "velocity_power_error",
    "velocity_divergence_pdf_error", "density_rk_high", "disp_rk_high",
)


def run_analyze(args) -> None:
    from cosmo_sr.tts.bootstrap import best_of_k, bootstrap_ci, paired_bootstrap, subset_draws
    from cosmo_sr.tts.scores import (
        PHASE_ORACLE_COMPONENTS, STATISTICAL_ORACLE_COMPONENTS,
        ScoreNormalizer, composite_score, derive_metrics,
    )

    out = Path(args.out)
    rows = read_rows(out / "rows.jsonl")
    if not rows:
        raise SystemExit(f"no rows in {out / 'rows.jsonl'}; run --stage generate first")
    rows = [derive_metrics(r) for r in rows]

    by_box: Dict[str, List[Dict]] = {}
    for r in rows:
        by_box.setdefault(r["box"], []).append(r)
    for b in by_box:
        by_box[b].sort(key=lambda r: r["seed"])

    val = [b for b in by_box if b in set(args.val_boxes)] or sorted(by_box)[: max(1, len(by_box) // 2)]
    test = [b for b in by_box if b in set(args.test_boxes)] or [b for b in sorted(by_box) if b not in val]
    print(f"normaliser fitted on val boxes {val}; curves reported on test boxes {test}")

    norm = ScoreNormalizer.fit([r for b in val for r in by_box[b]])
    norm.save(out / "normalizer.json")

    scorers = {
        "random": lambda r: 0.0,
        "phase": lambda r: composite_score(r, PHASE_ORACLE_COMPONENTS, norm),
        "statistical": lambda r: composite_score(r, STATISTICAL_ORACLE_COMPONENTS, norm),
    }
    for b, rs in by_box.items():
        for r in rs:
            for name, fn in scorers.items():
                r[f"score_{name}"] = fn(r)

    rng = np.random.default_rng(args.seed)
    k_values = sorted(set(int(k) for k in args.k_values))
    n_cand = min(len(by_box[b]) for b in test)
    if max(k_values) > n_cand:
        print(f"[warn] only {n_cand} candidates per box; clipping K")
        k_values = [k for k in k_values if k <= n_cand] or [n_cand]

    results: Dict = {"val_boxes": val, "test_boxes": test, "n_candidates": n_cand,
                     "k_values": k_values, "curves": {}, "selected_seeds": {}}

    for k in k_values:
        # One subset draw per box, shared by every selector: the comparison
        # between selectors is then paired at the subset level too.
        draws = {b: subset_draws(k, len(by_box[b]), args.n_repeats, rng) for b in test}
        for metric in REPORT_METRICS:
            for sel in scorers:
                per_box = []
                for b in test:
                    rs = by_box[b]
                    vals = [r.get(metric, np.nan) for r in rs]
                    if sel == "random":
                        # every candidate equally likely -> mean over the subset
                        chosen = np.nanmean(np.asarray(vals, dtype=float)[draws[b]], axis=1)
                        per_box.append(float(np.nanmean(chosen)))
                    else:
                        sc = [r[f"score_{sel}"] for r in rs]
                        mean, _ = best_of_k(vals, sc, k, draws=draws[b])
                        per_box.append(mean)
                results["curves"].setdefault(metric, {}).setdefault(sel, {})[k] = {
                    **bootstrap_ci(per_box, n_boot=args.n_boot, rng=rng), "per_box": per_box,
                }

    # Improvement of each oracle over random, paired by box, at every K.
    gates: Dict = {}
    for metric in REPORT_METRICS:
        for sel in ("phase", "statistical"):
            for k in k_values:
                a = results["curves"][metric][sel][k]["per_box"]
                b = results["curves"][metric]["random"][k]["per_box"]
                cmp = paired_bootstrap(a, b, n_boot=args.n_boot, rng=rng)
                gates.setdefault(metric, {}).setdefault(sel, {})[k] = cmp
    results["improvement_vs_random"] = gates

    for b in test:
        rs = by_box[b]
        for sel in ("phase", "statistical"):
            sc = np.asarray([r[f"score_{sel}"] for r in rs], dtype=float)
            sc = np.where(np.isfinite(sc), sc, np.inf)
            results["selected_seeds"].setdefault(b, {})[sel] = int(rs[int(np.argmin(sc))]["seed"])

    # --- decision gate --------------------------------------------------- #
    k_gate = 16 if 16 in k_values else max(k_values)
    primary = ("density_power_error", "density_pdf_error",
               "bispectrum_equilateral_error", "bispectrum_squeezed_error")
    verdict = {"k": k_gate, "threshold_rel": args.gate_rel, "metrics": {}}
    passed = False
    for metric in primary:
        cmp = gates[metric]["statistical"][k_gate]
        rel = -cmp["relative"]        # improvement is a *decrease* in an error
        ok = bool(cmp["significant"] and cmp["mean"] < 0 and rel >= args.gate_rel)
        verdict["metrics"][metric] = {"rel_improvement": rel, "significant": cmp["significant"],
                                      "passes": ok}
        passed = passed or ok
    verdict["pass"] = passed
    results["decision_gate"] = verdict

    (out / "oracle_report.json").write_text(json.dumps(results, indent=2, default=float))
    _plot(out, results, k_values)

    from cosmo_sr.tts.scores import METRIC_DIRECTION
    print("\n=== best-of-K vs K (test boxes; 'better' is direction-aware) ===")
    for metric in REPORT_METRICS:
        base = results["curves"][metric]["random"][k_values[0]]["mean"]
        sign = METRIC_DIRECTION.get(metric, -1)     # +1 = larger is better
        line = [f"{metric:34s} K=1 {base:9.4g}"]
        for sel in ("statistical", "phase"):
            v = results["curves"][metric][sel][k_gate]["mean"]
            better = sign * (v - base) / abs(base or 1)
            line.append(f"| {sel}@K{k_gate} {v:9.4g} ({100 * better:+6.1f}% better)")
        print(" ".join(line))
    print(f"\nDECISION GATE (statistical oracle, K={k_gate}, "
          f">={100 * args.gate_rel:.0f}% relative): {'PASS' if passed else 'FAIL'}")
    for m, d in verdict["metrics"].items():
        print(f"  {m:34s} rel={100 * d['rel_improvement']:+6.2f}%  "
              f"significant={d['significant']}  passes={d['passes']}")
    if not passed:
        print("\n  -> SR2's noise gives no usable selection leverage on the primary\n"
              "     density / higher-order metrics. Stop before verifier development.")
    print(f"\nwrote {out}/oracle_report.json, oracle_scaling.png")


def _plot(out: Path, results: Dict, k_values: Sequence[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in REPORT_METRICS if m in results["curves"]]
    ncol = 4
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics):
        for sel, style in (("random", "k--"), ("statistical", "C0-o"), ("phase", "C3-s")):
            curve = results["curves"][metric][sel]
            ks = sorted(int(k) for k in curve)
            mean = [curve[k]["mean"] for k in ks]
            lo = [curve[k]["lo"] for k in ks]
            hi = [curve[k]["hi"] for k in ks]
            ax.plot(ks, mean, style, label=sel, ms=4)
            ax.fill_between(ks, lo, hi, alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K"); ax.set_title(metric, fontsize=9)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    fig.suptitle("SR2 test-time scaling: best-of-K oracle audit (95% box bootstrap)", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "oracle_scaling.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("generate", "analyze", "all"), default="all")
    ap.add_argument("--lr", default="/zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr")
    ap.add_argument("--hr", default="/zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr")
    ap.add_argument("--model", default=str(_ROOT / "external" / "SRS-map2map" / "SRmodel" / "G_z0.pt"))
    ap.add_argument("--boxes", nargs="*", default=None, help="subset of box stems to run")
    ap.add_argument("--train-boxes", nargs="*",
                    default=[f"set{i}" for i in range(8)],
                    help="boxes used for the HR plausibility reference and verifier training")
    ap.add_argument("--val-boxes", nargs="*", default=["set8", "set9", "set10", "set11"])
    ap.add_argument("--test-boxes", nargs="*", default=["set12", "set13", "set14", "set15"])
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(16)))
    ap.add_argument("--k-values", nargs="*", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--nsplit", type=int, default=8)
    ap.add_argument("--pad", type=int, default=3)
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--noise-mode", choices=("per_tile", "global"), default="per_tile")
    ap.add_argument("--boxsize", type=float, default=100000.0)
    ap.add_argument("--dis-norm", type=float, default=6000.0)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--slab", type=int, default=32)
    ap.add_argument("--n-repeats", type=int, default=200, help="subsets per K")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--gate-rel", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chan-base", type=int, default=512)
    ap.add_argument("--chan-min", type=int, default=64)
    ap.add_argument("--chan-max", type=int, default=512)
    ap.add_argument("--features", dest="features", action="store_true", default=True,
                    help="also record HR-free features (needed by scripts/train_srs_verifier.py)")
    ap.add_argument("--no-features", dest="features", action="store_false")
    ap.add_argument("--equivariance", action="store_true",
                    help="add rotation/flip consistency features (~7 extra tile passes each)")
    ap.add_argument("--n-probes", type=int, default=4)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--out", default="runs/tts_oracle")
    args = ap.parse_args()

    if args.stage in ("generate", "all"):
        run_generate(args)
    if args.stage in ("analyze", "all"):
        run_analyze(args)


if __name__ == "__main__":
    main()
