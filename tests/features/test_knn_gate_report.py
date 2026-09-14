"""The kNN damage report must reproduce the set9 calibration it documents.

Replays every arm of the 32-arm crosscheck through ``report`` when that JSON is
on disk, so a threshold edit that stops separating the two families fails here;
and pins the flag logic on synthetic inputs so it runs anywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "features"))

import knn_gate_report as kr  # noqa: E402

REAL = Path("/zfsauton/scratch/yixiz/DMSR/dmsr_reward/member_gather/knn_cdf")


def _knn(arm_vals, frozen_vals, host128):
    def lf(v):
        return {f"class_cond[{c}]": [v[c], v[c]] for c in v}
    return {"loss_forms": {"frozen": lf(frozen_vals), "arm": lf(arm_vals)},
            "membership": {
                "frozen": {"classes": {"host": {"paired_ratio_med": [1.0, 1.05]}}},
                "arm": {"classes": {"host": {"paired_ratio_med": [1.0, host128]}}}}}


F = {"sub_50_200": 0.15, "sub_ge200": 0.10, "unbound": 0.20}


def test_damage_needs_both_subhalo_terms():
    both = kr.report(_knn({"sub_50_200": 0.3, "sub_ge200": 0.3, "unbound": 0.1}, F, 1.2),
                     "arm", box="set9")
    assert "DAMAGE" in both["flags"] and "hosts_diffused" in both["flags"]
    one = kr.report(_knn({"sub_50_200": 0.3, "sub_ge200": 0.1, "unbound": 0.1}, F, 1.0),
                    "arm", box="set9")
    assert "DAMAGE" not in one["flags"]


def test_advisory_flags_and_calibration_note():
    r = kr.report(_knn({"sub_50_200": 0.1, "sub_ge200": 0.05, "unbound": 0.5}, F, 0.7),
                  "arm", box="set10")
    assert set(r["flags"]) == {"hosts_overconcentrated", "unbound_crowded"}
    assert r["calibrated"] is False


def test_missing_arm_is_skipped_not_raised(tmp_path):
    out = tmp_path / "r.json"
    (tmp_path / "k.json").write_text(json.dumps(_knn(F, F, 1.0)))
    assert kr.main(["--knn-json", str(tmp_path / "k.json"), "--arm", "nope",
                    "--out", str(out)]) == 0
    assert json.loads(out.read_text())["ok"] is False


@pytest.mark.skipif(not (REAL / "knn_gate_crosscheck_set9.json").exists(),
                    reason="set9 calibration JSON not on this filesystem")
def test_thresholds_separate_the_set9_families():
    knn = json.loads((REAL / "knn_cdf_region_allarms_set9.json").read_text())
    gate = {r["label"]: r for r in json.loads(
        (REAL / "knn_gate_crosscheck_set9.json").read_text())["rows"]}
    for label in knn["loss_forms"]:
        if label == "frozen":
            continue
        r = kr.report(knn, label, box="set9")
        damaging = gate[label]["sub_50_200"] <= 4500
        assert ("DAMAGE" in r["flags"]) == damaging, (label, r)
