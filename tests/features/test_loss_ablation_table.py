import importlib.util
import json
from pathlib import Path

_P = Path(__file__).resolve().parents[2] / "scripts/features/loss_ablation_table.py"
_spec = importlib.util.spec_from_file_location("loss_ablation_table", _P)
lat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lat)


def _counts(subs, hosts, host_subs):
    return ({"in_tiles": {"subs_ge_50p": subs, "hosts_ge_200p": hosts, "n_halos": 0}},
            {"total": host_subs})


BINS = ["50-100p", "100-200p", "200-500p", "500-2000p", "2000-1000000000000p"]


def _rmf(sub):
    return {"sub": dict(zip(BINS, sub), total=sum(sub)),
            "host": dict(zip(BINS, [0] * 5), total=0)}


def _verdict(new, region_mf=None):
    rc, mf = {}, {}
    for who, (s, h, hs) in {"base": (4541, 3028, 20), "hr": (12367, 3775, 374),
                            "self": (9187, 2708, 367), "new": new}.items():
        rc[who], mf[who] = _counts(s, h, hs)
    d = {"region_counts": rc, "mass_function": mf,
         "q1_abundance_held": False, "q2_hosts_recovered": True}
    if region_mf is not None:
        d["region_mass_function"] = {"new": _rmf(region_mf), "hr": _rmf([99, 99, 99, 99, 99])}
        d["shape_score"] = {"new": {"sub": 0.25, "host": 0.10}}
        d["radial_score"] = {"new": 0.42}
        d["radial_distribution"] = {"new": {"median_r_over_rvir": 0.35}}
    return d


def test_build_reads_final_and_early_and_reports_missing(tmp_path):
    arm = tmp_path / "member_gather/all_blocks_abl_nobound"
    (arm / "holdout_set9").mkdir(parents=True)
    (arm / "holdout_set9/verdict_vs_self.json").write_text(
        json.dumps(_verdict((5000, 2900, 40))))
    rows = [{"step": s, "holdout": {"bound_hard": s / 1e4, "term_bound": 1.0}}
            for s in (0, 2000, 8000)]
    (arm / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    t = lat.build(tmp_path, "set9", ["nobound", "repro"], 2000)
    by = {(r["arm"], r["stage"]): r for r in t["rows"]}
    assert by[("nobound", "final")]["region_subs"] == 5000
    assert by[("nobound", "final")]["holdout_metrics"]["step"] == 8000
    assert not by[("nobound", "step2000")]["gated"]
    # the early row's surrogate is capped at the snapshot step, not the last row
    assert by[("nobound", "step2000")]["holdout_metrics"]["step"] == 2000
    assert not by[("repro", "final")]["gated"]
    assert t["references"]["self"]["region_subs"] == 9187

    md = lat.markdown(t)
    assert "| nobound | final | 40 | 5000 | -4187 | 2900 | 192 | - | - | - | - |" in md
    assert "| repro | final | not gated |" in md


def test_bin_order_is_numeric_not_alphabetical():
    # exactly how write_json stores it: sorted keys
    stored = {"100-200p": 1, "200-500p": 2, "2000-1000000000000p": 3,
              "50-100p": 4, "500-2000p": 5, "total": 15}
    assert lat.bin_order(stored) == BINS


def test_shape_columns_are_read_when_the_verdict_carries_them(tmp_path):
    arm = tmp_path / "member_gather/all_blocks_abl_shp"
    (arm / "holdout_set9").mkdir(parents=True)
    # sort_keys mirrors write_json, so the ratio column must not follow key order
    (arm / "holdout_set9/verdict_vs_self.json").write_text(
        json.dumps(_verdict((5000, 2900, 40), region_mf=[99, 99, 199, 49, 99]),
                   sort_keys=True))

    t = lat.build(tmp_path, "set9", ["shp"], 2000)
    row = [r for r in t["rows"] if r["stage"] == "final"][0]
    assert row["shape_sub"] == 0.25
    # (199+1)/(99+1) = 2.0 over, (49+1)/(99+1) = 0.5 under
    assert row["region_bin_ratio"] == [1.0, 1.0, 2.0, 0.5, 1.0]
    assert row["radial_sub"] == 0.42
    assert row["median_r_over_rvir"] == 0.35
    assert "| 0.250 | 0.420 | 0.350 | 1.00 1.00 2.00 0.50 1.00 |" in lat.markdown(t)
