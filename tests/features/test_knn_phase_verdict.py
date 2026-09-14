"""Round-5 verdict: stop rules read the right families, signs and modes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "knn_phase_verdict", ROOT / "scripts/features/knn_phase_verdict.py")
pv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pv)


def _write(tmp_path, rng):
    gate = [{"label": "frozen", "sub_50_200": 2762, "sub_50_200_matched": 100,
             "hr_recall_50_200": 0.01, "lost_ge200": 0}]
    lf3 = {"frozen": {"class_cond[sub_50_200]": [0.15, 0.15]}}
    lf6 = {"frozen": {"class_cond[sub_50_200]": [0.30, 0.30]}}
    for i in range(20):                                  # working: 4600..8400
        n = 4600 + 200 * i
        lab = f"w{i}"
        gate.append({"label": lab, "sub_50_200": n, "sub_50_200_matched": n // 20,
                     "hr_recall_50_200": n / 1e5, "lost_ge200": 100})
        lf3[lab] = {"class_cond[sub_50_200]": list(0.15 + 0.01 * rng.standard_normal(2))}
        lf6[lab] = {"class_cond[sub_50_200]": [0.30 - n * 1e-5] * 2}   # falls as count rises
    for i in range(6):                                   # damaging
        lab = f"d{i}"
        gate.append({"label": lab, "sub_50_200": 2400 + 50 * i, "sub_50_200_matched": 50,
                     "hr_recall_50_200": 0.005, "lost_ge200": 600})
        lf3[lab] = lf6[lab] = {"class_cond[sub_50_200]": [0.5 + i * 0.1] * 2}
    (tmp_path / "gate.json").write_text(json.dumps({"rows": gate}))
    mem = lambda r: {"frozen": {"classes": {"sub_50_200": {"paired_ratio_med": r}}}}  # noqa: E731
    by = {"sub_50_200": {"rel_kmax_p95": 0.05, "incomplete_frac": 0.01, "n_queries": 99}}
    k6 = {"space": "6d", "phase_lambda_kms_per_mpch": 1000.0, "membership": mem([2.5, 1.8]),
          "loss_forms": lf6,
          "boundary": {"tuned_centroid_frozen": [
              {"size_cells": 0.0, "candidate_multiplier": 3.0, "by_class": by}]}}
    k3 = {"membership": mem([1.5, 1.1]), "loss_forms": lf3, "boundary": {}}
    (tmp_path / "k6.json").write_text(json.dumps(k6))
    (tmp_path / "k3.json").write_text(json.dumps(k3))


def test_stop_rules(tmp_path):
    _write(tmp_path, np.random.default_rng(0))
    out = tmp_path / "v.json"
    assert pv.main(["--gate-json", str(tmp_path / "gate.json"),
                    "--knn", f"3d={tmp_path / 'k3.json'}", f"l1000={tmp_path / 'k6.json'}",
                    f"l500={tmp_path / 'missing.json'}", "--out", str(out)]) == 0
    r = json.loads(out.read_text())["spaces"]
    assert r["l500"]["missing"]
    fam = r["l1000"]["B_ranking"]["families"]
    assert fam["working"]["n"] == 20 and fam["damaging"]["n"] == 6
    assert r["l1000"]["verdict"] == {"A": True, "B": True, "C": True}
    assert r["3d"]["verdict"]["B"] is False
