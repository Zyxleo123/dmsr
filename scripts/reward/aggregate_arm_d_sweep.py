#!/usr/bin/env python
"""Aggregate the arm-D LOBO sweep: pick a winner, draw the curves.

Reads only the sweep's JSONL rows, so the figures are redrawable without
refitting anything. Writes:

* ``arm_<arm>_<stage>_decision.json`` -- the winner and why, which the
  hyperparameter stage reads so it cannot sweep against a stale choice;
* ``arm_<arm>_<stage>_summary.csv`` -- per (variant, epoch) mean over folds;
* ``arm_<arm>_<stage>_curves.png`` -- validation Spearman vs epoch per variant,
  with the train curve dashed so the generalisation gap is visible.

Selection rule
--------------
Mean over LOBO folds of the validation within-tile Spearman, at the epoch where
that mean is highest (early stopping on validation, never on training loss). A
variant is only comparable if it ran every fold, so incomplete variants are
reported and excluded rather than winning on a lucky subset.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from _sr2_direct import banner, direct_root, write_json_atomic  # noqa: E402


def load_rows(path: Path) -> List[Dict]:
    if not path.is_file():
        raise SystemExit(f"{path} does not exist; run the sweep stage first")
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _key(r: Dict) -> str:
    """Variant identity including hyperparameters, so the two stages share code."""
    hp = r.get("hparams") or {}
    if not hp:
        return str(r["variant"])
    bits = ",".join(f"{k}={hp[k]}" for k in sorted(hp))
    return f"{r['variant']}|lr={r.get('lr')}|{bits}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="d")
    ap.add_argument("--stage", choices=["loss", "hparam"], default="loss")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--min-folds", type=int, default=0,
                    help="0 = require every fold present in the file")
    args = ap.parse_args(argv)

    root = direct_root("sweeps")
    path = Path(args.jsonl) if args.jsonl else root / f"arm_{args.arm}_{args.stage}.jsonl"
    rows = load_rows(path)
    if not rows:
        raise SystemExit(f"{path} is empty")

    all_folds = sorted({str(r["held_out"]) for r in rows})
    need = int(args.min_folds) or len(all_folds)

    # (variant_key, epoch) -> {fold: val_rho}
    val: Dict = defaultdict(dict)
    tr: Dict = defaultdict(dict)
    for r in rows:
        k = (_key(r), int(r["epoch"]))
        val[k][str(r["held_out"])] = float(r["val_within_tile_spearman"])
        tr[k][str(r["held_out"])] = float(r["train_within_tile_spearman"])

    variants = sorted({k[0] for k in val})
    epochs = sorted({k[1] for k in val})

    table: List[Dict] = []
    for v in variants:
        for e in epochs:
            f = val.get((v, e), {})
            if not f:
                continue
            vals = np.array([x for x in f.values() if np.isfinite(x)])
            trs = np.array([x for x in tr.get((v, e), {}).values() if np.isfinite(x)])
            table.append({
                "variant": v, "epoch": e, "n_folds": len(f),
                "val_mean": float(vals.mean()) if vals.size else float("nan"),
                "val_std": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
                "train_mean": float(trs.mean()) if trs.size else float("nan"),
                "gap": (float(trs.mean() - vals.mean())
                        if vals.size and trs.size else float("nan")),
            })

    complete = [t for t in table if t["n_folds"] >= need]
    incomplete = sorted({t["variant"] for t in table if t["n_folds"] < need})
    if not complete:
        raise SystemExit(
            f"no variant has all {need} folds yet; {len(rows)} rows so far. "
            f"Resubmit the sweep (it resumes) before aggregating.")

    best = max(complete, key=lambda t: (t["val_mean"] if np.isfinite(t["val_mean"])
                                        else -np.inf))
    per_variant = {}
    for v in sorted({t["variant"] for t in complete}):
        vt = [t for t in complete if t["variant"] == v]
        b = max(vt, key=lambda t: (t["val_mean"] if np.isfinite(t["val_mean"])
                                   else -np.inf))
        per_variant[v] = b

    decision = {
        "stage": args.stage, "arm": args.arm, "jsonl": str(path),
        "folds": all_folds, "folds_required": need,
        "winner": best["variant"].split("|")[0],
        "winner_key": best["variant"], "winner_epoch": best["epoch"],
        "winner_val_mean": best["val_mean"], "winner_val_std": best["val_std"],
        "winner_train_mean": best["train_mean"], "winner_gap": best["gap"],
        "per_variant_best": per_variant,
        "incomplete_variants_excluded": incomplete,
        "selection_rule": ("highest fold-mean validation within-tile Spearman, "
                           "early-stopped on validation and never on train loss"),
        "note": ("LOBO within set0-7. set8-11 were not read. A winner here is a "
                 "DIAGNOSTIC result for arm D and is not an arm-comparison claim: "
                 "re-fit all six arms through train_catalog_proxy.py to make one."),
    }
    write_json_atomic(root / f"arm_{args.arm}_{args.stage}_decision.json", decision)

    csv = root / f"arm_{args.arm}_{args.stage}_summary.csv"
    hdr = ["variant", "epoch", "n_folds", "val_mean", "val_std", "train_mean", "gap"]
    csv.write_text("\n".join([",".join(hdr)] + [
        ",".join(str(t[h]) for h in hdr)
        for t in sorted(table, key=lambda t: (t["variant"], t["epoch"]))]) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5.5))
        cmap = plt.get_cmap("tab10")
        for i, v in enumerate(sorted({t["variant"] for t in complete})):
            vt = sorted([t for t in complete if t["variant"] == v],
                        key=lambda t: t["epoch"])
            e = [t["epoch"] for t in vt]
            c = cmap(i % 10)
            ax.plot(e, [t["val_mean"] for t in vt], "-o", color=c, label=v, ms=4)
            ax.plot(e, [t["train_mean"] for t in vt], "--", color=c, alpha=0.45)
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.text(ax.get_xlim()[1], 0.5, " gate 0.5", va="center", fontsize=8)
        ax.set_xlabel("epoch (full-batch Adam steps)")
        ax.set_ylabel("within-tile Spearman (mean over LOBO folds)")
        ax.set_title(f"arm {args.arm} {args.stage} sweep -- solid: validation, "
                     f"dashed: train")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(root / f"arm_{args.arm}_{args.stage}_curves.png", dpi=140)
        plt.close(fig)
    except Exception as exc:   # a missing backend must not lose the decision
        print(f"  (figure skipped: {exc})", flush=True)

    banner(json.dumps({k: decision[k] for k in
                       ("winner", "winner_epoch", "winner_val_mean",
                        "winner_val_std", "winner_train_mean", "winner_gap",
                        "incomplete_variants_excluded")}, indent=2))
    print(f"  decision -> {root}/arm_{args.arm}_{args.stage}_decision.json")
    print(f"  curves   -> {root}/arm_{args.arm}_{args.stage}_curves.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
