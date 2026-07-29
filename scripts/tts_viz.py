#!/usr/bin/env python
"""Figures for the SR2 test-time-scaling study.

Reads only the artefacts the pipeline already wrote (``rows.jsonl``,
``profiles.npz``, ``box_summary.json``, ``oracle_report.json``,
``final_table.json``), so it is cheap, rerunnable, and never has to regenerate a
box just to redraw a plot. That matters here: one 512^3 candidate is minutes of
GPU time, and figures get redrawn a lot more often than they get computed.

Figures produced in ``--out``:

``fig_scaling.png``
    Best-of-K against K for every selector and every headline metric, with the
    95% box-bootstrap band. The saturation point is the whole result.
``fig_diversity.png``
    Candidate diversity per box (displacement / velocity / density) before
    selection, next to the fraction of distinct candidates a selector actually
    picks. A selector that always picks the same kind of realisation shows up
    here and nowhere else.
``fig_ensemble_bias.png``
    Ensemble density power and log-density PDF, selected subset vs the full
    candidate pool. Any departure from unity is selection-induced distribution
    bias -- the failure mode where a metric improves because the output
    distribution moved, not because the field got better.
``fig_metric_corr.png``
    Within-box correlation between per-candidate metrics. Tells you whether the
    statistical oracle is chasing one direction of variation or several
    conflicting ones; if density and velocity errors anti-correlate, no single
    selector can win both.
``fig_score_spread.png``
    Per-box spread of candidate scores with the oracle pick marked, so the
    "is there anything to select between?" question is answerable by eye.

``--render-slices`` additionally regenerates the oracle-best and oracle-worst
candidate for one box and renders density slices plus their difference. That one
needs the generator and a GPU, hence the separate flag.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HEADLINE = (
    "density_power_error", "density_pdf_error", "bispectrum_equilateral_error",
    "bispectrum_squeezed_error", "velocity_power_error",
    "velocity_divergence_pdf_error", "density_rk_high", "disp_rk_high",
)
CORR_METRICS = (
    "density_power_error", "density_pdf_error", "density_sigma_ratio",
    "bispectrum_equilateral_error", "bispectrum_squeezed_error",
    "velocity_power_error", "velocity_divergence_pdf_error",
    "density_rk_transition", "disp_rk_high", "lr_recon_rel_disp", "boundary_ratio",
)
#: ``name -> (colour, line/marker style, label)``. Colour is kept separate from
#: the format string because single-K methods are drawn as horizontal reference
#: lines rather than curves, and they still have to be told apart.
SELECTOR_STYLE = {
    "random":      ("k",  "--",  "random"),
    "random_k":    ("k",  "--",  "random"),
    "sr2_single":  ("0.45", ":", "SR2 single"),
    "handcrafted": ("C2", "-^",  "hand-crafted"),
    "verifier":    ("C0", "-o",  "verifier"),
    "statistical": ("C0", "-o",  "statistical oracle"),
    "phase":       ("C3", "-s",  "phase oracle"),
    "oracle_k":    ("C3", "-s",  "oracle"),
    "refine":      ("C1", "-.",  "verifier + refinement"),
    "cem_refine":  ("C5", "-.",  "verifier + CEM"),
    "global_joint": ("C4", "--", "verifier + global tiling"),
}


def _style(name: str):
    return SELECTOR_STYLE.get(name, ("C6", "-", name))


def _load_json(path) -> Optional[Dict]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def _load_rows(path) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
def fig_scaling(out: Path, report: Optional[Dict], table: Optional[Dict]) -> None:
    """Best-of-K vs K. Prefers the final table (all selectors) over Stage 1."""
    curves, source = None, ""
    if table and table.get("table"):
        curves, source = table["table"], "final table"
    elif report and report.get("curves"):
        curves, source = report["curves"], "Stage-1 audit"
    if not curves:
        print("[skip] fig_scaling: no curves found")
        return

    metrics = [m for m in HEADLINE if m in curves]
    if not metrics:
        return
    ncol = min(4, len(metrics))
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.5 * nrow), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics):
        for sel, cells in curves[metric].items():
            colour, fmt, label = _style(sel)
            ks = sorted(int(k) for k in cells)
            get = lambda k: cells[str(k) if str(k) in cells else k]  # noqa: E731
            if len(ks) < 2:      # single-K methods: a horizontal reference line
                ls = fmt if not fmt.lstrip("-").startswith(("o", "^", "s")) else "-."
                ax.axhline(get(ks[0])["mean"], color=colour, ls=ls, lw=1.4, label=label)
                continue
            ax.plot(ks, [get(k)["mean"] for k in ks], fmt, color=colour, label=label, ms=4)
            ax.fill_between(ks, [get(k)["lo"] for k in ks], [get(k)["hi"] for k in ks],
                            color=colour, alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K (candidates)")
        ax.set_title(metric, fontsize=9)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=7)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    fig.suptitle(f"SR2 test-time scaling: best-of-K ({source}, 95% box bootstrap)", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "fig_scaling.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/fig_scaling.png")


def fig_diversity(out: Path, box_summary: Optional[Dict], table: Optional[Dict]) -> None:
    """Candidate spread before selection, and how much of it a selector uses."""
    if not box_summary and not table:
        print("[skip] fig_diversity: nothing to plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if box_summary:
        boxes = sorted(box_summary)
        keys = [("disp_diversity", "displacement"), ("vel_diversity", "velocity"),
                ("density_diversity", "density")]
        width = 0.8 / len(keys)
        for i, (key, label) in enumerate(keys):
            vals = [box_summary[b].get(key, np.nan) for b in boxes]
            axes[0].bar(np.arange(len(boxes)) + i * width, vals, width, label=label)
        axes[0].set_xticks(np.arange(len(boxes)) + 0.4 - width / 2)
        axes[0].set_xticklabels(boxes, rotation=45, ha="right", fontsize=7)
        axes[0].set_ylabel("across-candidate sd / rms")
        axes[0].set_title("Candidate diversity before selection", fontsize=10)
        axes[0].set_yscale("log")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="y")

    after = (table or {}).get("diversity_after_unique_frac") or {}
    if after:
        names = list(after)
        k_div = (table or {}).get("diversity_measured_at_k")
        axes[1].bar(np.arange(len(names)), [after[n] for n in names],
                    color=[_style(n)[0] for n in names])
        axes[1].set_xticks(np.arange(len(names)))
        axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("distinct candidates chosen / pool")
        axes[1].set_title(
            f"Selection diversity{f' at K={k_div}' if k_div else ''} "
            "(1.0 = uses the whole pool)", fontsize=10)
        axes[1].grid(alpha=0.3, axis="y")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out / "fig_diversity.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out}/fig_diversity.png")


def fig_ensemble_bias(out: Path, table: Optional[Dict], profiles_path) -> None:
    """Does selection move the output distribution rather than improve it?"""
    bias = (table or {}).get("distribution_bias") or {}
    if not bias:
        print("[skip] fig_ensemble_bias: no distribution_bias in the final table")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for sel, d in bias.items():
        pk_s = np.asarray(d.get("pk_selected", []), dtype=float)
        pk_a = np.asarray(d.get("pk_ensemble", []), dtype=float)
        if pk_s.size and pk_a.size:
            colour, _fmt, label = _style(sel)
            axes[0].plot(np.arange(len(pk_s)), pk_s / np.maximum(pk_a, 1e-30),
                         "-o", color=colour, label=label, ms=3)
    axes[0].axhline(1.0, color="k", lw=0.8)
    axes[0].set_xlabel("k bin (low -> high)")
    axes[0].set_ylabel(r"$P_{\rm selected}/P_{\rm ensemble}$")
    axes[0].set_title("Density power: selected subset vs full pool", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    names = list(bias)
    axes[1].bar(np.arange(len(names)), [bias[n].get("pdf_l1_shift", np.nan) for n in names])
    axes[1].set_xticks(np.arange(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel(r"$L_1$ shift of the $\log_{10}(1+\delta)$ PDF")
    axes[1].set_title("PDF shift induced by selection (0 = unbiased)", fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig_ensemble_bias.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out}/fig_ensemble_bias.png")


def fig_metric_corr(out: Path, rows: Sequence[Dict]) -> None:
    """Within-box correlation between per-candidate metrics.

    Correlations are computed after removing each box's mean, so this measures
    how metrics co-vary *across candidates of the same input* -- the only
    variation a selector can act on. A global correlation would mostly report
    differences between boxes.
    """
    if not rows:
        print("[skip] fig_metric_corr: no rows")
        return
    metrics = [m for m in CORR_METRICS if any(m in r for r in rows)]
    if len(metrics) < 2:
        return
    by_box: Dict[str, List[Dict]] = {}
    for r in rows:
        by_box.setdefault(r["box"], []).append(r)

    cols = []
    for b, rs in sorted(by_box.items()):
        x = np.asarray([[float(r.get(m, np.nan)) for m in metrics] for r in rs])
        if x.shape[0] < 3:
            continue
        cols.append(x - np.nanmean(x, axis=0, keepdims=True))   # de-mean per box
    if not cols:
        return
    x = np.vstack(cols)
    keep = np.isfinite(x).all(axis=1)
    x = x[keep]
    sd = x.std(axis=0)
    ok = sd > 1e-30
    corr = np.full((len(metrics), len(metrics)), np.nan)
    sub = np.corrcoef(x[:, ok], rowvar=False)
    idx = np.where(ok)[0]
    for a, ia in enumerate(idx):
        for bb, ib in enumerate(idx):
            corr[ia, ib] = sub[a, bb]

    fig, ax = plt.subplots(figsize=(1 + 0.55 * len(metrics), 1 + 0.5 * len(metrics)))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(metrics))); ax.set_yticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(metrics, fontsize=7)
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            if np.isfinite(corr[i, j]):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Within-box metric correlation across candidates", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_metric_corr.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out}/fig_metric_corr.png")


def fig_score_spread(out: Path, rows: Sequence[Dict], report: Optional[Dict],
                     val_boxes: Sequence[str]) -> None:
    """Per-box spread of the statistical-oracle score, with the oracle pick marked."""
    from cosmo_sr.tts.scores import (
        STATISTICAL_ORACLE_COMPONENTS, ScoreNormalizer, composite_score, derive_metrics,
    )

    if not rows:
        print("[skip] fig_score_spread: no rows")
        return
    rows = [derive_metrics(r) for r in rows]
    val = [r for r in rows if r["box"] in set(val_boxes)] or rows
    norm = ScoreNormalizer.fit(val)
    by_box: Dict[str, List[Dict]] = {}
    for r in rows:
        r["_score"] = composite_score(r, STATISTICAL_ORACLE_COMPONENTS, norm)
        by_box.setdefault(r["box"], []).append(r)

    boxes = sorted(by_box)
    fig, ax = plt.subplots(figsize=(1.2 + 0.7 * len(boxes), 4.2))
    for i, b in enumerate(boxes):
        s = np.asarray([r["_score"] for r in by_box[b]], dtype=float)
        s = s[np.isfinite(s)]
        if not s.size:
            continue
        ax.scatter(np.full(s.shape, i) + np.random.default_rng(i).normal(0, 0.06, s.size),
                   s, s=12, alpha=0.6, color="C0")
        ax.scatter([i], [s.min()], marker="*", s=140, color="C3", zorder=5,
                   label="oracle pick" if i == 0 else None)
    picks = (report or {}).get("selected_seeds", {})
    ax.set_xticks(range(len(boxes)))
    ax.set_xticklabels(
        [f"{b}\nseed {picks.get(b, {}).get('statistical', '-')}" for b in boxes],
        fontsize=7,
    )
    ax.set_ylabel("statistical-oracle score (lower = better)")
    ax.set_title("Per-box candidate spread -- is there anything to select between?",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig_score_spread.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out}/fig_score_spread.png")


# --------------------------------------------------------------------------- #
def render_slices(args, rows: Sequence[Dict]) -> None:
    """Regenerate the oracle-best and oracle-worst candidate and show them.

    The only figure that needs the generator. Pictures are how a selection
    result is sanity-checked: a candidate that wins on every number while
    looking obviously wrong is a bug, not a discovery.
    """
    import torch
    from cosmo_sr.tts.metrics import DensityGeometry, cic_density_slabs
    from cosmo_sr.tts.sampling import super_resolve_srs_seeded
    from cosmo_sr.tts.scores import (
        STATISTICAL_ORACLE_COMPONENTS, ScoreNormalizer, composite_score, derive_metrics,
    )
    from cosmo_sr.tts.srs_noise import load_controlled_generator

    rows = [derive_metrics(r) for r in rows if r["box"] == args.slice_box]
    if len(rows) < 2:
        print(f"[skip] render_slices: fewer than 2 candidates for {args.slice_box}")
        return
    norm = ScoreNormalizer.load(args.normalizer) if Path(args.normalizer).exists() \
        else ScoreNormalizer.fit(rows)
    scores = [composite_score(r, STATISTICAL_ORACLE_COMPONENTS, norm) for r in rows]
    order = np.argsort(np.where(np.isfinite(scores), scores, np.inf))
    best, worst = rows[int(order[0])], rows[int(order[-1])]
    print(f"[slices] best seed {best['seed']}, worst seed {worst['seed']}")

    device = torch.device(args.device)
    G = load_controlled_generator(args.model, scale_factor=args.scale, device=device)
    lr = np.load(Path(args.lr) / f"{args.slice_box}.npy").astype(np.float32)
    hr = torch.from_numpy(
        np.load(Path(args.hr) / f"{args.slice_box}.npy").astype(np.float32)
    ).unsqueeze(0)
    geo = DensityGeometry(boxsize=args.boxsize, ng=hr.shape[-1], dis_norm=args.dis_norm)

    panels = []
    with torch.no_grad():
        panels.append(("HR truth", cic_density_slabs(hr[:, 0:3].to(device), geo.cellsize,
                                                     geo.dis_norm, slab=args.slab)))
        for tag, row in (("oracle best", best), ("oracle worst", worst)):
            field = super_resolve_srs_seeded(
                G, lr, int(row["seed"]), scale_factor=args.scale, nsplit=args.nsplit,
                pad=args.pad, device=device,
            )
            t = torch.from_numpy(field).unsqueeze(0).to(device)
            panels.append((f"{tag} (seed {row['seed']})",
                           cic_density_slabs(t[:, 0:3], geo.cellsize, geo.dis_norm,
                                             slab=args.slab)))
            del t, field

    z = panels[0][1].shape[-1] // 2
    thick = max(1, args.slab_thickness)
    imgs = [(name, np.log10(
        1.0 + d[0, 0, :, :, z:z + thick].mean(dim=-1).clamp_min(-0.999).cpu().numpy()
    )) for name, d in panels]
    diff = imgs[1][1] - imgs[2][1]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    vmin = min(im.min() for _n, im in imgs)
    vmax = max(im.max() for _n, im in imgs)
    for ax, (name, im) in zip(axes[:3], imgs):
        h = ax.imshow(im.T, origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
        ax.set_title(name, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(h, ax=ax, fraction=0.046)
    lim = float(np.abs(diff).max()) or 1.0
    h = axes[3].imshow(diff.T, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim)
    axes[3].set_title("best - worst", fontsize=10)
    axes[3].set_xticks([]); axes[3].set_yticks([])
    fig.colorbar(h, ax=axes[3], fraction=0.046)
    fig.suptitle(f"{args.slice_box}: log10(1+delta) slice, oracle-selected vs rejected", y=1.0)
    fig.tight_layout()
    fig.savefig(Path(args.out) / f"fig_slices_{args.slice_box}.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}/fig_slices_{args.slice_box}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle-dir", default="runs/tts/oracle")
    ap.add_argument("--final-dir", default="runs/tts/final")
    ap.add_argument("--val-boxes", nargs="*", default=["set8", "set9", "set10", "set11"])
    ap.add_argument("--out", default="runs/tts/figures")
    # --render-slices only
    ap.add_argument("--render-slices", action="store_true")
    ap.add_argument("--slice-box", default="set14")
    ap.add_argument("--slab-thickness", type=int, default=4)
    ap.add_argument("--lr", default="/zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr")
    ap.add_argument("--hr", default="/zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr")
    ap.add_argument("--model", default=str(_ROOT / "external" / "SRS-map2map" /
                                           "SRmodel" / "G_z0.pt"))
    ap.add_argument("--normalizer", default="runs/tts/oracle/normalizer.json")
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--nsplit", type=int, default=8)
    ap.add_argument("--pad", type=int, default=3)
    ap.add_argument("--slab", type=int, default=32)
    ap.add_argument("--boxsize", type=float, default=100000.0)
    ap.add_argument("--dis-norm", type=float, default=6000.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    oracle = Path(args.oracle_dir)
    rows = _load_rows(oracle / "rows.jsonl")
    report = _load_json(oracle / "oracle_report.json")
    table = _load_json(Path(args.final_dir) / "final_table.json")
    box_summary = _load_json(oracle / "box_summary.json")

    fig_scaling(out, report, table)
    fig_diversity(out, box_summary, table)
    fig_ensemble_bias(out, table, oracle / "profiles.npz")
    fig_metric_corr(out, rows)
    fig_score_spread(out, rows, report, args.val_boxes)
    if args.render_slices:
        render_slices(args, rows)
    print(f"\nfigures in {out}")


if __name__ == "__main__":
    main()
