#!/usr/bin/env python
"""Collect the density-fix sweep into one table, from saved artefacts only.

Reads whatever the pipeline has produced so far and writes ``summary.md``,
``summary.json`` and ``summary.png``. Nothing here touches a model or a 512^3
field, so the table and the figure can be regenerated as often as wanted for the
cost of a few JSON reads.

Two things this script exists to keep straight:

**The density columns of ``metrics.csv`` are not comparable across the base/arm
boundary.** ``HighPassDensity`` is shared between the critic and validation, so
an arm with ``critic.valid_center: 32`` validates its density on a 32^3 offset
cube while the baseline validates on a 64^3 wrapped one -- different fields, on
different grids, with band edges that mean different physical scales. They are
comparable *among* the valid-center arms. The displacement columns
(``val_rk_*``, ``val_Tk_error_*``, ``val_mse``) involve no CIC deposit and are
comparable everywhere. The table marks each block accordingly.

**Only Stage 0 is decisive.** It generates every arm over the whole box and
scores them with one function against one truth, which is the only construction
in which ``density_highk_pk_ratio`` means the same thing for all four.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

# Reference points on the full-box ruler, from the earlier set14 comparison.
# They are printed alongside the new numbers so a result is never read without
# the two things that bound it.
REFERENCE = {
    "trilinear": {"highk_power_ratio": 0.036, "density_highk_pk_ratio": 0.706},
    "srs":       {"highk_power_ratio": 0.396, "density_highk_pk_ratio": 0.977},
    "baseline (as measured 07-25)":
                 {"highk_power_ratio": 0.487, "density_highk_pk_ratio": 0.282},
}

DISP_COLS = ["val_mse", "val_rk_transition", "val_Tk_error_transition",
             "val_rk_high", "val_Tk_error_high", "val_sample_diversity",
             "val_condition_shuffle_gap"]
DENS_COLS = ["val_density_power_error", "val_density_rk_high",
             "val_density_Tk_error_high", "val_density_pdf_error"]


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def tail_mean(rows: List[dict], col: str, n: int = 5) -> float:
    vals = [_f(r[col]) for r in rows if r.get(col) not in (None, "")]
    vals = [v for v in vals if not math.isnan(v)][-n:]
    return sum(vals) / len(vals) if vals else float("nan")


def read_metrics(run_dir: Path) -> Optional[List[dict]]:
    p = run_dir / "metrics.csv"
    if not p.exists():
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def fmt(v, width: int = 9, prec: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-".rjust(width)
    return f"{v:{width}.{prec}g}"


def md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the pipeline OUT directory")
    ap.add_argument("--arms", nargs="+", required=True,
                    help="label:run_dir pairs, in the order to display")
    ap.add_argument("--box", default="set14")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [(s.split(":", 1)[0], Path(s.split(":", 1)[1])) for s in args.arms]
    labels = [a for a, _ in arms]

    summary: Dict[str, dict] = {"box": args.box, "arms": {}}
    doc: List[str] = [f"# Density-fix sweep -- {args.box}", ""]

    # --- Stage 0: the decisive full-box ruler ------------------------------- #
    for tile, note in ((16, "32^3 LR windows"), (8, "16^3 LR windows, close to training")):
        cmp_json = read_json(out / f"ceiling_tile{tile}" / "comparison.json")
        doc += [f"## Stage 0 -- full-box ruler, tile {tile} ({note})", ""]
        if cmp_json is None:
            doc += [f"_not run: {out}/ceiling_tile{tile}/comparison.json missing._", ""]
            continue
        rows = []
        for name, rec in cmp_json.items():
            rows.append([name,
                         fmt(rec.get("highk_power_ratio")),
                         fmt(rec.get("density_highk_pk_ratio")),
                         fmt(rec.get("density_rk_highk")),
                         fmt(rec.get("cross_corr_mean"))])
        doc += [md_table(["field", "disp P/P_HR hi-k", "dens P/P_HR hi-k",
                          "dens r(k) hi-k", "cross corr"], rows), ""]
        summary[f"stage0_tile{tile}"] = cmp_json

    ref_rows = [[k, fmt(v["highk_power_ratio"]), fmt(v["density_highk_pk_ratio"])]
                for k, v in REFERENCE.items()]
    doc += ["### Reference points on the same ruler", "",
            md_table(["field", "disp P/P_HR hi-k", "dens P/P_HR hi-k"], ref_rows), "",
            "A density gain that costs displacement power is the mean-collapse",
            "failure returning, not progress: read the two columns together.", ""]

    both = [read_json(out / f"ceiling_tile{t}" / "comparison.json") for t in (16, 8)]
    if all(b is not None for b in both):
        rows = []
        for lab in labels:
            a, b = both[0].get(lab), both[1].get(lab)
            if not a or not b:
                continue
            rows.append([lab,
                         fmt(a.get("density_highk_pk_ratio")),
                         fmt(b.get("density_highk_pk_ratio")),
                         fmt(_f(a.get("density_highk_pk_ratio")) - _f(b.get("density_highk_pk_ratio")))])
        doc += ["### Window-size sensitivity (Stage 0 minus Stage 0b)", "",
                md_table(["arm", "tile 16", "tile 8", "difference"], rows), "",
                "This difference is the train/eval normalisation shift. Training used",
                "8^3 LR crops, so a model whose conditioning is genuinely local should",
                "show almost none -- that is the specific prediction `model.norm:",
                "channel` makes, and the honest way to check it.", ""]

    # --- training-time metrics --------------------------------------------- #
    doc += ["## Validation metrics (from metrics.csv, mean of last 5 points)", ""]
    metrics = {lab: read_metrics(d) for lab, d in arms}
    present = [lab for lab in labels if metrics[lab]]

    doc += ["### Displacement -- comparable across ALL arms", "",
            "No CIC deposit is involved, so these mean the same thing everywhere.", ""]
    rows = [[c] + [fmt(tail_mean(metrics[lab], c)) for lab in present] for c in DISP_COLS]
    doc += [md_table(["metric"] + present, rows), ""]

    doc += ["### Density -- comparable only AMONG valid-center arms", "",
            "`HighPassDensity` is shared by the critic and validation, so an arm with",
            "`critic.valid_center: 32` scores a 32^3 offset cube while the baseline",
            "scores a 64^3 wrapped one. Reading a baseline-to-arm difference off this",
            "block measures the change of instrument, not the change of model.", ""]
    rows = [[c] + [fmt(tail_mean(metrics[lab], c)) for lab in present] for c in DENS_COLS]
    doc += [md_table(["metric"] + present, rows), ""]

    for lab in present:
        summary["arms"][lab] = {c: tail_mean(metrics[lab], c) for c in DISP_COLS + DENS_COLS}

    # --- diagnostics -------------------------------------------------------- #
    for title, pattern, keys in (
        ("Stage 1 -- residual alpha", "stage1_alpha_*/residual_alpha.json", None),
        ("Stage 3 -- context oracle", "stage3_context_*/context_oracle.json", None),
        ("Stage 4 -- receptive field", "stage4_rf_*/receptive_field.json", None),
        ("Stage 6 -- best-of-K", "stage6_bok_*/best_of_k.json", None),
    ):
        found = sorted(out.glob(pattern))
        doc += [f"## {title}", ""]
        if not found:
            doc += [f"_not run: no `{pattern}` under {out}._", ""]
            continue
        for p in found:
            doc += [f"- `{p.relative_to(out)}`"]
        doc += ["", "See the per-stage `*.png` next to each JSON.", ""]
        summary[title.split(" --")[0].lower().replace(" ", "")] = [str(p) for p in found]

    (out / "summary.md").write_text("\n".join(doc) + "\n")
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n".join(doc))

    _plot(out, labels, both[0] if both else None, metrics, present)
    print(f"\n=== wrote {out}/summary.md, summary.json")


def _plot(out: Path, labels, cmp_json, metrics, present) -> None:
    """One figure: the decisive plane, plus the displacement metric over steps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib unavailable -- skipping the figure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    if cmp_json:
        for name, rec in cmp_json.items():
            x, y = rec.get("highk_power_ratio"), rec.get("density_highk_pk_ratio")
            if x is None or y is None:
                continue
            ax.scatter(x, y, s=60)
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    for name, v in REFERENCE.items():
        ax.scatter(v["highk_power_ratio"], v["density_highk_pk_ratio"],
                   marker="x", c="k", s=50)
        ax.annotate(name, (v["highk_power_ratio"], v["density_highk_pk_ratio"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=7, color="gray")
    ax.axhline(1.0, ls=":", c="gray", lw=0.8)
    ax.axvline(1.0, ls=":", c="gray", lw=0.8)
    ax.set_xlabel("displacement P/P_HR, high-k")
    ax.set_ylabel("density P/P_HR, high-k")
    ax.set_title("The decisive plane (Stage 0, full box)\nup-and-right is better")

    ax = axes[1]
    for lab in present:
        rows = metrics[lab]
        pts = [(int(r["step"]), _f(r["val_Tk_error_high"]))
               for r in rows if r.get("val_Tk_error_high") not in (None, "")]
        if pts:
            ax.plot(*zip(*pts), marker="o", ms=3, label=lab)
    ax.set_xlabel("step")
    ax.set_ylabel("val_Tk_error_high (lower is better)")
    ax.set_title("Displacement power error\n(comparable across all arms)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
