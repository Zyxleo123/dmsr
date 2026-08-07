"""End-to-end wiring of the direct-line scripts, on synthetic data.

These are not physics tests: they run the actual ``main()`` of each script
against a tiny fabricated dataset and check that the pieces connect -- the proxy
trains, the gate reads what the trainer wrote, a failing gate stops the actor,
and the density gate refuses to pass anything while its tolerances are
placeholders. That last one is the property most likely to rot silently, since
nothing else in the pipeline notices a gate that always says yes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "reward"))

from cosmo_sr.reward import paths  # noqa: E402
from cosmo_sr.reward.catalog import CatalogBins, ChunkSummary  # noqa: E402
from cosmo_sr.reward.reward import fit_reward_model  # noqa: E402
from cosmo_sr.reward.soft_structure import SoftStructureConfig, feature_names  # noqa: E402

N_SUB_BINS, N_HOST_BINS = 6, 5


@pytest.fixture
def fitted_reward(reward_root):
    bins = CatalogBins(
        sub_mass_edges=tuple(np.logspace(10.1, 13.1, N_SUB_BINS + 1).tolist()),
        host_mass_edges=tuple(np.logspace(12.0, 14.5, N_HOST_BINS + 1).tolist()))
    rng = np.random.default_rng(0)
    chunks = []
    for b in range(6):
        off = float(rng.normal(0.0, 0.15))
        for c in range(8):
            s = float(np.exp(off + rng.normal(0.0, 0.05)))
            chunks.append(ChunkSummary(
                box=f"set{b}", chunk_id=c, source="hr",
                n_sub=np.round(np.array([400., 160., 60., 22., 8., 3.]) * s),
                n_host=np.round(np.array([160., 54., 16., 4., 1.]) * s) + 1,
                occ_numerator=np.round(np.array([200., 90., 40., 14., 5.]) * s),
                volume_mpc3=1562.5))
    model = fit_reward_model(chunks, bins,
                             active_dims=[i for i in range(11) if i != 10])
    out = paths.subdir("reward_model", create=True) / "reward_model.json"
    out.write_text(json.dumps(model.to_dict()))
    return out


def _write_rows(path: Path, *, boxes, tiles, n_feat, seed=0):
    """Frozen + two perturbed candidates per (box, tile), with a real signal.

    ``compact_mass`` (feature 0) drives the subhalo counts, so a proxy that
    learns anything at all can rank the three candidates of a tile. The point is
    that the *table shape* is right, not that the physics is.
    """
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for box in boxes:
            for t in tiles:
                base_h = rng.uniform(0.5, 3.0, size=N_HOST_BINS)
                for k, (source, alpha, gain) in enumerate([
                    ("frozen", None, 0.0),
                    ("intervention", 0.5, 0.5),
                    ("intervention", 1.0, 1.0),
                ]):
                    f = rng.normal(0.0, 0.2, size=n_feat)
                    f[0] = gain + rng.normal(0.0, 0.01)
                    f[n_feat // 2] = gain
                    fh.write(json.dumps({
                        "box": box, "tag": f"{source}_{k}", "source": source,
                        "seed": 0, "alpha": alpha, "tile_id": int(t),
                        "features": f.tolist(),
                        "n_sub": (np.array([4., 3., 2., 1., 0.5, 0.2])
                                  * (1.0 + gain)).tolist(),
                        "n_host": base_h.tolist(),
                        "occ_numerator": (base_h * (1.0 + gain)).tolist(),
                        "volume_mpc3": 1562.5 / 64.0,
                        "model_sha": "deadbeef", "lr_sha": "cafe",
                        "code_commit": "test",
                    }) + "\n")


@pytest.fixture
def direct_cfg(tmp_path, reward_root):
    """A copy of the real config, pointed at the tmp roots and made small."""
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "reward"
                          / "sr2_direct_finetune.yaml").read_text())
    cfg["proxy"].update({"n_members": 2, "epochs": 12, "hidden": [16]})
    cfg["split"]["proxy_fit_boxes"] = ["set0", "set1"]
    cfg["split"]["proxy_gate_boxes"] = ["set8"]
    p = tmp_path / "direct.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _n_features() -> int:
    return 2 * len(feature_names(SoftStructureConfig()))


def test_proxy_trains_and_gate_reads_it(tmp_path, direct_cfg, fitted_reward, capsys):
    import gate_catalog_proxy
    import train_catalog_proxy
    from _sr2_direct import direct_root, run_dir

    table = direct_root("proxy_data", create=True) / "rows.jsonl"
    _write_rows(table, boxes=["set0", "set1", "set8"], tiles=range(6),
                n_feat=_n_features())

    assert train_catalog_proxy.main(
        ["--config", str(direct_cfg), "--run-name", "t", "--device", "cpu"]) == 0
    run = run_dir("t")
    assert (run / "proxy" / "member_00.pt").is_file()
    report = json.loads((run / "proxy" / "train_report.json").read_text())
    # Pairs must exist, or the ranking loss silently did nothing.
    assert report["n_ranking_pairs"] > 0
    assert report["n_train_rows"] > 0 and report["n_val_rows"] > 0

    assert gate_catalog_proxy.main(
        ["--config", str(direct_cfg), "--run-name", "t"]) == 0
    verdict = json.loads((run / "proxy_gate.json").read_text())
    assert set(verdict) >= {"passed", "failures", "checks"}
    names = {c["name"] for c in verdict["checks"]}
    assert names == {
        "within_tile_spearman", "pairwise_accuracy", "selected_positive_fraction",
        "alpha_ordering_spearman", "single_feature_dependence",
        "occupation_curve_log_error_mean", "splice_sign_agreement",
        "n_splice_verifications",
    }


def test_gate_fails_without_the_real_splice_verifications(
        tmp_path, direct_cfg, fitted_reward):
    """Missing evidence makes a criterion unmet, never met."""
    import gate_catalog_proxy
    import train_catalog_proxy
    from _sr2_direct import direct_root, run_dir

    _write_rows(direct_root("proxy_data", create=True) / "rows.jsonl",
                boxes=["set0", "set1", "set8"], tiles=range(6), n_feat=_n_features())
    train_catalog_proxy.main(
        ["--config", str(direct_cfg), "--run-name", "t", "--device", "cpu"])
    gate_catalog_proxy.main(["--config", str(direct_cfg), "--run-name", "t"])
    verdict = json.loads((run_dir("t") / "proxy_gate.json").read_text())
    assert not verdict["passed"]
    assert any("splice" in f for f in verdict["failures"])


def test_actor_refuses_to_train_on_a_failed_proxy_gate(tmp_path, direct_cfg,
                                                       fitted_reward, capsys):
    import train_sr2_direct
    from _sr2_direct import run_dir

    run = run_dir("t", create=True)
    (run / "proxy_gate.json").write_text(json.dumps(
        {"passed": False, "failures": ["within_tile_spearman=0.1 < 0.5"]}))
    assert train_sr2_direct.main(
        ["--config", str(direct_cfg), "--run-name", "t"]) == 0
    out = capsys.readouterr().out
    assert "GATE FAILED" in out
    # And it says the one thing the plan is emphatic about.
    assert "unfreezing more of the generator" in out


def test_actor_exits_zero_when_the_gate_has_not_run(tmp_path, direct_cfg,
                                                    fitted_reward, capsys):
    """Exit 0, not non-zero: a non-zero exit strands every afterok dependent."""
    import train_sr2_direct

    assert train_sr2_direct.main(
        ["--config", str(direct_cfg), "--run-name", "never"]) == 0
    assert "MISSING INPUT" in capsys.readouterr().out


def test_density_calibration_then_gate(tmp_path, direct_cfg, fitted_reward, capsys):
    import score_sr2_direct
    from _sr2_direct import run_dir

    run = run_dir("t", create=True)
    rng = np.random.default_rng(0)
    frozen = [{"box": b, "seed": s, "tag": "frozen",
               "density_power_error": 0.020 + float(rng.normal(0, 0.001)),
               "low_k_change": 0.0, "d_struct": 0.3}
              for b in ("set8", "set9") for s in (0, 1, 2)]
    with open(run / "field_metrics_frozen.jsonl", "w") as fh:
        for r in frozen:
            fh.write(json.dumps(r) + "\n")

    assert score_sr2_direct.main(
        ["--config", str(direct_cfg), "--run-name", "t", "--stage", "calibrate"]) == 0
    cal = json.loads((run / "gate_calibration.json").read_text())
    assert cal["proposal"]["mean_degradation_max"] > 0.0
    assert "HUMAN STEP" in capsys.readouterr().out

    # Degradation well inside the frozen seed-to-seed spread: indistinguishable
    # from noise, so it must pass once the tolerance is real.
    tol = cal["proposal"]["mean_degradation_max"]
    assert tol > 0
    for tag, delta in (("cand", 0.2 * tol), ("bad", 5.0 * tol)):
        with open(run / f"field_metrics_{tag}.jsonl", "w") as fh:
            for r in frozen:
                fh.write(json.dumps({
                    **r, "tag": tag,
                    "density_power_error": r["density_power_error"] + delta,
                    "low_k_change": 0.01, "d_struct": 0.25}) + "\n")

    # calibrated: false in the committed config -> nothing may pass.
    assert score_sr2_direct.main(
        ["--config", str(direct_cfg), "--run-name", "t", "--stage", "gate",
         "--tag", "cand"]) == 0
    v = json.loads((run / "gate_cand.json").read_text())
    assert not v["passed"]
    assert any("calibrated is false" in x for x in v["violations"])

    # With the proposal pasted in, the same candidate is judged on its merits.
    cfg = yaml.safe_load(direct_cfg.read_text())
    cfg["gates"].update(cal["proposal"])
    direct_cfg.write_text(yaml.safe_dump(cfg))
    score_sr2_direct.main(
        ["--config", str(direct_cfg), "--run-name", "t", "--stage", "gate",
         "--tag", "cand"])
    v = json.loads((run / "gate_cand.json").read_text())
    assert v["passed"], v["violations"]
    assert len(v["per_box"]) == len(frozen)

    # And a candidate five times outside that spread is rejected, so the gate is
    # discriminating rather than merely permissive.
    score_sr2_direct.main(
        ["--config", str(direct_cfg), "--run-name", "t", "--stage", "gate",
         "--tag", "bad"])
    bad = json.loads((run / "gate_bad.json").read_text())
    assert not bad["passed"]
    assert any("degradation" in x for x in bad["violations"])


def test_score_exits_zero_without_a_frozen_baseline(tmp_path, direct_cfg,
                                                    fitted_reward, capsys):
    import score_sr2_direct

    assert score_sr2_direct.main(
        ["--config", str(direct_cfg), "--run-name", "empty",
         "--stage", "calibrate"]) == 0
    assert "MISSING INPUT" in capsys.readouterr().out


def test_proxy_data_index_warns_when_there_is_no_within_tile_variation(
        tmp_path, direct_cfg, fitted_reward, capsys):
    import collect_catalog_proxy_data as C

    assert C.main(["--config", str(direct_cfg), "--stage", "index"]) == 0
    out = capsys.readouterr().out
    assert "has_within_tile_variation" in out
    assert "easy binary distinction" in out
