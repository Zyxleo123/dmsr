#!/usr/bin/env python
"""Redraw the candidate/partition diagnostic from its JSONL rows -- no recompute.

Reads ``retrieval.jsonl`` and ``coverage.jsonl`` written by
:mod:`scripts/reward/diagnose_candidate_partition.py` and draws, per box:

* retrieval completeness vs ``R_c`` (in host-Rvir units) -- frozen SR2 against
  the HR ceiling, missing targets against present-subhalo controls. The gap
  between the SR2 curve and the ceiling is exactly the retrieval loss the
  fixed-id partition suffers by being built in the wrong (frozen) configuration.
* coverage: fraction of HR subhalos with a frozen-SR2 candidate within ``R``,
  sub-only against sub+peak, missing against present.

    python scripts/reward/plot_candidate_partition.py --boxes set8,set9
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import paths, read_jsonl  # noqa: E402


def _retrieval_curves(rows: List[Dict], kind: str, field: str):
    mults = sorted({r["r_mult"] for r in rows})
    med, lo, hi = [], [], []
    for mult in mults:
        vals = [r[field] for r in rows if r["kind"] == kind and r["r_mult"] == mult]
        if not vals:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        med.append(float(np.median(vals)))
        lo.append(float(np.quantile(vals, 0.25)))
        hi.append(float(np.quantile(vals, 0.75)))
    return np.array(mults), np.array(med), np.array(lo), np.array(hi)


def plot_retrieval(box: str, rows: List[Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plans = [
        ("missing_target", "completeness_sr2", "missing, frozen SR2", "C3", "-"),
        ("missing_target", "completeness_hr", "missing, HR ceiling", "C3", "--"),
        ("present_control", "completeness_sr2", "present, frozen SR2", "C0", "-"),
        ("present_control", "completeness_hr", "present, HR ceiling", "C0", "--"),
    ]
    for kind, field, label, color, ls in plans:
        if not any(r["kind"] == kind for r in rows):
            continue
        x, med, lo, hi = _retrieval_curves(rows, kind, field)
        ax.plot(x, med, ls, color=color, label=label, marker="o", ms=4)
        if ls == "-":
            ax.fill_between(x, lo, hi, color=color, alpha=0.15)
    ax.set_xlabel("collection radius $R_c$ / host $R_{vir}$")
    ax.set_ylabel("member completeness")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{box}: Q1 retrieval -- can the fixed bag reach the members?")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_coverage(box: str, rows: List[Dict], out: Path) -> None:
    thr = np.linspace(0.05, 3.0, 60)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, sel, color in (("missing", [r for r in rows if r["is_missing"]], "C3"),
                              ("present", [r for r in rows if not r["is_missing"]], "C0")):
        if not sel:
            continue
        for key, ls in (("nearest_sub_mpc_h", "--"), ("nearest_all_mpc_h", "-")):
            d = np.array([r[key] for r in sel])
            cov = [float(np.mean(d <= t)) for t in thr]
            axes[0].plot(thr, cov, ls, color=color,
                         label=f"{label}, {'sub+peak' if ls=='-' else 'sub only'}")
    axes[0].set_xlabel("match radius $R$ [Mpc/h]")
    axes[0].set_ylabel("fraction of HR subhalos covered")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title(f"{box}: Q2 coverage")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(alpha=0.3)

    mv = np.array([r["sub_mvir"] for r in rows])
    nd = np.array([r["nearest_all_mpc_h"] for r in rows])
    miss = np.array([r["is_missing"] for r in rows])
    axes[1].scatter(mv[~miss], nd[~miss], s=8, alpha=0.3, color="C0", label="present")
    axes[1].scatter(mv[miss], nd[miss], s=24, color="C3", label="missing", zorder=3)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("subhalo $M_{vir}$ [$M_\\odot/h$]")
    axes[1].set_ylabel("nearest sub+peak candidate [Mpc/h]")
    axes[1].set_title("distance to nearest candidate vs mass")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes", default="set8,set9")
    ap.add_argument("--out-name", default="candidate_partition")
    args = ap.parse_args(argv)
    root = paths.subdir("audits", args.out_name)
    for box in [b.strip() for b in args.boxes.split(",") if b.strip()]:
        d = root / box
        ret = d / "retrieval.jsonl"
        cov = d / "coverage.jsonl"
        if ret.is_file():
            plot_retrieval(box, read_jsonl(ret), d / "retrieval.png")
            print(f"wrote {d / 'retrieval.png'}", flush=True)
        if cov.is_file():
            plot_coverage(box, read_jsonl(cov), d / "coverage.png")
            print(f"wrote {d / 'coverage.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
