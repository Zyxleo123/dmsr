#!/usr/bin/env python
"""kNN round 5 verdict: does PHASE-SPACE kNN fix what killed position kNN?

Rounds 1-4 (``docs/sr2_knn_distance.md``) measured position kNN and stopped it as
a training objective for three reasons. Round 5 (``knn_cdf_split_diagnostic.py
--phase-lambda``) re-measures in 6D; this script applies one stop rule per
reason, to the 3D baseline and to every lambda, from saved JSON only.

``A_separation``  frozen's ``sub_50_200`` paired median r_k / HR, per k.
                  PASS when 6D's largest |log ratio| exceeds 3D's: the particles
                  that ARE the deficit must look MORE wrong, not less.
``B_ranking``     THE test 3D failed. Within the WORKING family (region 50-200p
                  subhalos > ``--working-min-sub50``; 20 set9 arms), Spearman of
                  (term - frozen) vs (gate count - frozen) for the class-
                  conditioned ``sub_50_200`` loss, per k and k-mean. A loss must
                  fall as the count rises, so PASS is rho <= ``--rho-pass`` (-0.6)
                  with p < 0.05 at the k-mean. 3D was |rho| <= 0.4.
``C_locality``    tile-local r_kmax error on ``sub_50_200`` query particles with
                  the tuned tile + FROZEN neighbours (``tuned_centroid_frozen``,
                  smallest margin) and with the tile alone (``tuned_centroid``,
                  buffer 0). PASS when the frozen-neighbour p95 < ``--c-pass``
                  (0.10): a trainer could then skip live neighbours for this
                  class. 3D, all classes pooled: p95 0.41.

Gate counts come from round 4's ``knn_gate_crosscheck_<box>.json`` (they do not
depend on the kNN space). A missing or partial lambda is reported, not raised.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr

SUB = "sub_50_200"
GATE_KEYS = ("sub_50_200", "sub_50_200_matched", "hr_recall_50_200", "lost_ge200")


def separation(knn: dict, frozen: str) -> Optional[Dict]:
    e = (knn.get("membership", {}).get(frozen, {}).get("classes", {}).get(SUB) or {})
    r = e.get("paired_ratio_med")
    if not r:
        return None
    return {"paired_ratio_med": r, "max_abs_log": float(np.max(np.abs(np.log(r))))}


def ranking(knn: dict, gate_rows: Dict[str, dict], frozen: str, min_sub50: int,
            term: str = f"class_cond[{SUB}]") -> Dict:
    lf = knn.get("loss_forms", {})
    if frozen not in lf or frozen not in gate_rows or term not in lf[frozen]:
        return {"ok": False, "why": "no frozen row"}
    f_term = np.asarray(lf[frozen][term], dtype=float)
    fam: Dict[str, List[str]] = {"working": [], "damaging": []}
    for lab in lf:
        if lab == frozen or lab not in gate_rows or term not in lf[lab]:
            continue
        g = gate_rows[lab]["sub_50_200"]
        fam["working" if g > min_sub50 else "damaging"].append(lab)
    out = {"ok": True, "term": term, "families": {}}
    for name, labs in fam.items():
        rec = {"n": len(labs), "arms": sorted(labs), "by_gate": {}}
        if len(labs) >= 5:
            x_k = np.array([np.asarray(lf[l][term], float) - f_term for l in labs])
            for gk in GATE_KEYS:
                y = np.array([gate_rows[l][gk] - gate_rows[frozen][gk] for l in labs],
                             dtype=float)
                if np.ptp(y) == 0 or np.ptp(x_k.mean(axis=1)) == 0:
                    continue
                per_k = [float(spearmanr(x_k[:, j], y)[0]) for j in range(x_k.shape[1])]
                rho, p = spearmanr(x_k.mean(axis=1), y)
                rec["by_gate"][gk] = {"rho_kmean": float(rho), "p_kmean": float(p),
                                      "rho_per_k": per_k}
        out["families"][name] = rec
    return out


def locality(knn: dict) -> Dict:
    b = knn.get("boundary", {})
    out = {}
    for mode, size_kind in (("tuned_centroid_frozen", "margin"),
                            ("tuned_centroid", "buffer"),
                            ("tuned_centroid_nbr", "margin")):
        recs = [r for r in b.get(mode, []) if r.get("by_class")]
        out[mode] = [{"size_cells": r["size_cells"],
                      "candidate_multiplier": r["candidate_multiplier"],
                      **{c: r["by_class"][c] for c in r["by_class"]}} for r in recs]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-json", required=True,
                    help="round 4 knn_gate_crosscheck_<box>.json (gate counts per arm)")
    ap.add_argument("--knn", nargs="+", required=True,
                    help="label=path of knn_cdf_split_diagnostic outputs; include the "
                         "3D all-arms JSON as label 3d")
    ap.add_argument("--frozen-label", default="frozen")
    ap.add_argument("--working-min-sub50", type=int, default=4500)
    ap.add_argument("--rho-pass", type=float, default=-0.6)
    ap.add_argument("--c-pass", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    gate_rows = {r["label"]: r for r in json.loads(Path(a.gate_json).read_text())["rows"]}
    res = {"gate_json": a.gate_json, "working_min_sub50": a.working_min_sub50,
           "rho_pass": a.rho_pass, "c_pass": a.c_pass, "spaces": {}}
    base_sep = None
    for item in a.knn:
        label, _, path = item.partition("=")
        rec = {"path": path}
        res["spaces"][label] = rec
        if not Path(path).exists():
            rec["missing"] = True
            print(f"[{label}] MISSING {path} -- job failed or has not run", flush=True)
            continue
        knn = json.loads(Path(path).read_text())
        rec["space"] = knn.get("space", "3d")
        rec["phase_lambda"] = knn.get("phase_lambda_kms_per_mpch", 0.0)
        rec["n_fields"] = len(knn.get("membership", {}))
        rec["A_separation"] = separation(knn, a.frozen_label)
        rec["B_ranking"] = ranking(knn, gate_rows, a.frozen_label, a.working_min_sub50)
        rec["C_locality"] = locality(knn)
        if label == "3d" and rec["A_separation"]:
            base_sep = rec["A_separation"]["max_abs_log"]
    for label, rec in res["spaces"].items():
        if rec.get("missing"):
            continue
        s = rec["A_separation"]
        v = {}
        if s is not None and base_sep is not None and label != "3d":
            v["A"] = bool(s["max_abs_log"] > base_sep)
        w = (rec["B_ranking"].get("families", {}).get("working", {})
             .get("by_gate", {}).get("sub_50_200"))
        if w:
            v["B"] = bool(w["rho_kmean"] <= a.rho_pass and w["p_kmean"] < 0.05)
        fz = rec["C_locality"].get("tuned_centroid_frozen") or []
        if fz and SUB in fz[0]:
            v["C"] = bool(fz[0][SUB]["rel_kmax_p95"] < a.c_pass)
        rec["verdict"] = v

        print(f"\n[{label}] space={rec['space']} lambda={rec['phase_lambda']} "
              f"fields={rec['n_fields']}", flush=True)
        if s:
            print(f"  A  frozen {SUB} paired r_k/HR {np.round(s['paired_ratio_med'], 3).tolist()}"
                  f"  max|log| {s['max_abs_log']:.3f}"
                  f"{'' if base_sep is None else f' (3d {base_sep:.3f})'}  -> {v.get('A')}",
                  flush=True)
        for fam, fr in rec["B_ranking"].get("families", {}).items():
            cells = "  ".join(f"{gk} {g['rho_kmean']:+.2f}(p={g['p_kmean']:.2g})"
                              for gk, g in fr["by_gate"].items())
            print(f"  B  {fam:<8} n={fr['n']:<3} {cells}", flush=True)
        print(f"  B  -> {v.get('B')}", flush=True)
        for mode, rows in rec["C_locality"].items():
            for r in rows[:3]:
                c = r.get(SUB)
                if c:
                    print(f"  C  {mode:<22} size={r['size_cells']:>4} {SUB} p95(r_kmax)="
                          f"{c['rel_kmax_p95']:.3f} inc={c['incomplete_frac']:.3f} "
                          f"x{r['candidate_multiplier']:.2f}", flush=True)
        print(f"  C  -> {v.get('C')}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\n=== wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
