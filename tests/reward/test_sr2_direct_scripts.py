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


N_TOKENS, N_TOKEN_FEATURES = 8, 4


def _write_rows(path: Path, *, boxes, tiles, seed=0, invalid_box=None):
    """Frozen + perturbed candidates per (box, tile), with ALL arms' features.

    ``compact_mass`` (feature 0) drives the counts, so a proxy that learns
    anything can rank a tile's candidates; the phase-space block is present and
    carries the same signal only on the ``vel`` rows, which is the shape the
    diagnostic slice needs. Arm C's tokens go into a ``tokens_c.npy`` sidecar
    joined by ``row_id``, exactly as the indexer writes them, with the same
    gain in token channel 0. The point is that the *table* is right, not that
    the physics is.
    """
    from cosmo_sr.reward.phase_space import arm_paired_feature_names
    from _sr2_direct import candidate_tag

    n_a = len(arm_paired_feature_names("a"))
    n_b = len(arm_paired_feature_names("b"))
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    cands = ([("hr", None, "both", 0), ("frozen", None, "both", 0),
              ("frozen_seed", None, "both", 1)]
             + [("intervention", a, m, 0)
                for m in ("both", "disp", "vel") for a in (0.25, 0.5, 1.0)])
    row_id, side_blocks = 0, []
    with open(path, "w") as fh:
        for box in boxes:
            for t in tiles:
                base_h = rng.uniform(0.5, 3.0, size=N_HOST_BINS)
                for source, alpha, mode, cseed in cands:
                    gain = {"hr": 1.0, "frozen": 0.0, "frozen_seed": 0.0}.get(source)
                    if gain is None:
                        gain = float(alpha)
                    half_b = rng.normal(0.0, 0.2, size=n_b // 2)
                    half_b[0] = gain
                    half_b[len(feature_names(SoftStructureConfig()))] = (
                        gain if mode in ("vel", "both") else 0.0)
                    diff = half_b.copy()
                    fb = np.concatenate([half_b, diff])
                    fa = np.concatenate([half_b[:n_a // 2], diff[:n_a // 2]])
                    tok = rng.normal(0.0, 0.2,
                                     size=(2, N_TOKENS, N_TOKEN_FEATURES))
                    tok[:, :, 0] = gain
                    side_blocks.append(tok.astype(np.float32))
                    fh.write(json.dumps({
                        "row_id": row_id, "box": box,
                        "tag": candidate_tag(source, seed=cseed, alpha=alpha,
                                             mode=mode),
                        "source": source, "seed": cseed, "alpha": alpha,
                        "mode": mode, "tile_id": int(t),
                        "features_a": fa.tolist(), "features_b": fb.tolist(),
                        "n_sub": (np.array([4., 3., 2., 1., 0.5, 0.2])
                                  * (1.0 + gain)).tolist(),
                        "n_host": base_h.tolist(),
                        "occ_numerator": (base_h * (1.0 + gain)).tolist(),
                        "volume_mpc3": 1562.5 / 64.0,
                        "model_sha": "deadbeef", "lr_sha": "cafe",
                        "field_sha": "f00d", "code_commit": "test",
                    }) + "\n")
                    row_id += 1
    np.save(path.parent / "tokens_c.npy", np.stack(side_blocks))


def _mark_labels_complete():
    from _sr2_direct import labels_complete_path, write_json_atomic
    from cosmo_sr.reward.arms import FEATURE_SCHEMA_VERSION

    return write_json_atomic(labels_complete_path(), {
        "complete": True,
        "feature_schema_version": int(FEATURE_SCHEMA_VERSION),
    })


@pytest.fixture
def direct_cfg(tmp_path, reward_root):
    """A copy of the real config, pointed at the tmp roots and made small."""
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "reward"
                          / "sr2_direct_finetune.yaml").read_text())
    cfg["proxy"].update({"n_members": 2, "epochs": 12, "hidden": [16]})
    cfg["proxy"]["arm_c"] = {"token_hidden": [8], "embed_dim": 4}
    assert not cfg["proxy_cv"].get("enabled") and not cfg["proxy_cv"].get("grid"), \
        "the committed config must not carry a hyperparameter sweep"
    # Four fit boxes, not two: the ensemble is a box-BOOTSTRAP, and a
    # bootstrap over two boxes is uniform a quarter of the time, which makes
    # the test that it resamples at all a coin flip.
    cfg["split"]["proxy_fit_boxes"] = ["set0", "set1", "set2", "set3"]
    cfg["split"]["proxy_gate_boxes"] = ["set8", "set9"]
    p = tmp_path / "direct.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.fixture
def trained(direct_cfg, fitted_reward):
    """A fitted three-arm ensemble on a synthetic table. Returns the run dir."""
    import train_catalog_proxy
    from _sr2_direct import direct_root, run_dir

    _write_rows(direct_root("proxy_data", create=True) / "rows.jsonl",
                boxes=["set0", "set1", "set2", "set3", "set8", "set9"],
                tiles=range(3))
    _mark_labels_complete()
    assert train_catalog_proxy.main(
        ["--config", str(direct_cfg), "--run-name", "t", "--device", "cpu",
         "--arms", "a", "b", "c"]) == 0
    return run_dir("t")


def test_trainer_refuses_a_partial_table(direct_cfg, fitted_reward):
    """No labels_complete.json means labelling is still in flight.

    This is the failure the whole label workflow was restructured around: a
    trainer that reads whatever has landed fits a different dataset every run,
    and its two arms may not even see the same one.
    """
    import train_catalog_proxy
    from _sr2_direct import direct_root

    _write_rows(direct_root("proxy_data", create=True) / "rows.jsonl",
                boxes=["set0", "set1", "set8"], tiles=range(3))
    with pytest.raises(SystemExit, match="labelling is not complete"):
        train_catalog_proxy.main(
            ["--config", str(direct_cfg), "--run-name", "t", "--device", "cpu"])


def test_all_three_arms_are_fitted_with_fixed_predeclared_configs(trained):
    from cosmo_sr.reward.catalog_proxy import ProxyEnsemble
    from cosmo_sr.reward.soft_rockstar import SoftRockstarProxy

    report = json.loads((trained / "train_report.json").read_text())
    assert report["arms"] == ["a", "b", "c"]
    assert report["model_selection"] == "fixed_predeclared"
    assert "cross_validation" not in report
    assert report["proxy_unit"] == "region" and report["region_width"] == 1
    for arm in ("a", "b", "c"):
        assert (trained / f"proxy_{arm}" / "member_00.pt").is_file()
        assert report["arm_reports"][arm]["members"]
    # Arm B really is the wider feature vector, or the comparison is vacuous.
    assert (report["arm_reports"]["b"]["feature_shape"]
            > report["arm_reports"]["a"]["feature_shape"])
    # Arm C is a token grid and its ensemble loads back as the DeepSets class.
    assert report["arm_reports"]["c"]["feature_shape"] == [
        2, N_TOKENS, N_TOKEN_FEATURES]
    ens = ProxyEnsemble.load(trained / "proxy_c")
    assert all(isinstance(m, SoftRockstarProxy) for m in ens.members)
    assert (trained / "proxy_baselines.json").is_file()


def test_the_sweep_is_refused_if_it_creeps_back(direct_cfg, fitted_reward):
    import train_catalog_proxy
    from _sr2_direct import direct_root

    _write_rows(direct_root("proxy_data", create=True) / "rows.jsonl",
                boxes=["set0", "set1"], tiles=range(2))
    _mark_labels_complete()
    cfg = yaml.safe_load(direct_cfg.read_text())
    cfg["proxy_cv"] = {"enabled": True,
                       "grid": [{"hidden": [16], "lr": 1e-3}]}
    direct_cfg.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="sweep was removed"):
        train_catalog_proxy.main(
            ["--config", str(direct_cfg), "--run-name", "t", "--device", "cpu"])


def test_ensemble_members_are_box_bootstrap_samples(trained):
    """Different data, not just different seeds.

    Five re-initialisations on identical rows measure optimisation noise; the
    spread the actor's uncertainty penalty needs is the spread over which boxes
    were seen.
    """
    report = json.loads((trained / "train_report.json").read_text())
    draws = report["bootstrap_draws"]
    assert len(draws) == len(report["arm_reports"]["a"]["members"])
    n_boxes = len(report["fit_boxes"])
    # A member that saw every box exactly once is not a bootstrap sample.
    assert any(sorted(d.values()) != [1] * n_boxes for d in draws), draws
    assert all(sum(d.values()) == n_boxes for d in draws)


def test_gate_reports_every_predeclared_criterion(trained, direct_cfg):
    import gate_catalog_proxy

    assert gate_catalog_proxy.main(
        ["--config", str(direct_cfg), "--run-name", "t", "--arm", "a"]) == 0
    v = json.loads((trained / "proxy_gate_a.json").read_text())
    assert {c["name"] for c in v["checks"]} == {
        "within_tile_spearman", "pairwise_accuracy", "selected_positive_fraction",
        "single_feature_dependence",
        "occupation_log_error_worst_reliable_bin",
        "pooled_margin_over_worst_baseline", "uncertainty_error_spearman",
    }
    # The actor-like Rockstar verification is its own check family, judged
    # separately: an arm can be offline-eligible while awaiting the probe.
    assert {c["name"] for c in v["actor_checks"]} == {
        "actor_sign_agreement", "n_actor_checks", "n_actor_boxes",
    }
    # Occupancy is gated per reliable bin and the sparse bin is reported only.
    assert v["occupation_by_bin"]["reliable_host_bins"] == [0, 1, 2, 3]
    assert v["occupation_by_bin"]["report_only_bins"] == [4]
    # The predeclared slices are all present, including the two that make an
    # arm-B win attributable; the alpha ladder is reported there, not gated.
    for name in ("near_sr2", "interventions_disp", "interventions_vel",
                 "alpha_ordering_both", "alpha_ordering_vel"):
        assert name in v["slices"], sorted(v["slices"])
    assert set(v["baseline_comparison"]) == {"zero_change", "train_mean", "linear"}


def test_gate_fails_without_the_real_actor_verifications(trained, direct_cfg):
    """Missing evidence makes a criterion unmet, never met."""
    import gate_catalog_proxy

    gate_catalog_proxy.main(
        ["--config", str(direct_cfg), "--run-name", "t", "--arm", "a"])
    v = json.loads((trained / "proxy_gate_a.json").read_text())
    assert not v["passed"]
    assert not v["actor_passed"]
    assert any("actor" in f for f in v["actor_failures"])
    # And full passage is the conjunction, never the offline screen alone.
    assert v["passed"] == (v["offline_passed"] and v["actor_passed"])


def test_gate_passes_only_with_landed_actor_checks(trained, direct_cfg):
    """Actor rows flip actor_passed; the offline verdict is untouched."""
    import gate_catalog_proxy
    from _sr2_direct import append_jsonl

    rng = np.random.default_rng(0)
    for i in range(12):
        pred = float(rng.normal(0.5, 0.1))
        append_jsonl(trained / "actor_verification_a.jsonl", {
            "stratum": ["predicted_positive", "high_uncertainty", "random"][i % 3],
            "box": ["set8", "set9", "set10"][i % 3], "tile_id": i,
            "predicted_dR": pred, "measured_dR": pred + float(rng.normal(0, 0.01)),
        })
    gate_catalog_proxy.main(
        ["--config", str(direct_cfg), "--run-name", "t", "--arm", "a"])
    v = json.loads((trained / "proxy_gate_a.json").read_text())
    assert v["actor_passed"], v["actor_failures"]
    assert v["actor"]["n_checks"] == 12
    assert v["actor"]["n_boxes"] == 3
    assert v["passed"] == v["offline_passed"]


def test_splice_plan_is_stratified_and_spread_over_boxes(trained, direct_cfg):
    import splice_verify
    from _sr2_direct import direct_root

    # The planner needs fields on disk to point at; empty stand-ins are enough
    # to exercise the selection, which never opens them.
    for box in ("set8", "set9"):
        for tag in ("frozen_seed0", "hr", "intervention_vel_a1.00_seed0"):
            d = direct_root("candidates", f"{box}__{tag}", create=True)
            np.save(d / "field.npy", np.zeros((6, 4, 4, 4), dtype=np.float32))

    assert splice_verify.main(
        ["--config", str(direct_cfg), "--run-name", "t", "--arm", "a",
         "--stage", "select"]) == 0
    plan = json.loads((trained / "splice_plan_a.json").read_text())
    assert plan["n_planned"] == 12
    assert set(plan["strata"]) == {"predicted_positive", "high_uncertainty", "random"}
    assert all(v > 0 for v in plan["strata"].values())
    assert len(plan["boxes"]) >= 2
    # A frozen donor spliced into the frozen box is the identity and would pass
    # the sign test for free.
    assert all(s["donor_tag"] != s["base_tag"] for s in plan["splices"])


def test_benchmark_writes_the_decision_and_is_immutable(trained, direct_cfg, capsys):
    import gate_catalog_proxy
    import proxy_benchmark

    for arm in ("a", "b", "c"):
        gate_catalog_proxy.main(
            ["--config", str(direct_cfg), "--run-name", "t", "--arm", arm])
    assert proxy_benchmark.main(
        ["--config", str(direct_cfg), "--run-name", "t",
         "--arms", "a", "b", "c", "--n-bootstrap", "50"]) == 0
    doc = json.loads((trained / "proxy_benchmark.json").read_text())
    assert set(doc["passed"]) == {"a", "b", "c"}
    # No arm can advance here: the actor-like Rockstar checks were never run,
    # and offline evidence alone must not fine-tune anything.
    assert doc["decision"]["decision"] in (
        "do_not_finetune", "actor_gate_incomplete_do_not_finetune")
    assert doc["decision"]["advance"] == []
    assert doc["arm_comparison"]["n_boxes"] == 2
    assert "pairwise" in doc["arm_comparison"]
    assert doc["provenance"]["table_sha"]
    assert doc["content_sha256"]

    # Written once. A record that can be rewritten after the fact is a draft.
    before = (trained / "proxy_benchmark.json").read_text()
    assert proxy_benchmark.main(
        ["--config", str(direct_cfg), "--run-name", "t",
         "--arms", "a", "b", "c", "--n-bootstrap", "50"]) == 0
    assert (trained / "proxy_benchmark.json").read_text() == before
    assert "immutable" in capsys.readouterr().out


def test_benchmark_refuses_without_every_arm_verdict(trained, direct_cfg, capsys):
    import gate_catalog_proxy
    import proxy_benchmark

    gate_catalog_proxy.main(
        ["--config", str(direct_cfg), "--run-name", "t", "--arm", "a"])
    assert proxy_benchmark.main(
        ["--config", str(direct_cfg), "--run-name", "t",
         "--arms", "a", "b", "c"]) == 0
    assert not (trained / "proxy_benchmark.json").is_file()
    assert "MISSING INPUT" in capsys.readouterr().out


def _verdict(offline: bool, actor: bool) -> dict:
    return {"passed": offline and actor, "offline_passed": offline,
            "actor_passed": actor}


@pytest.mark.parametrize("offline, actor, expected", [
    ({"a": False, "b": False, "c": False}, False, "do_not_finetune"),
    ({"a": True, "b": False, "c": False}, True, "advance_a"),
    ({"a": False, "b": True, "c": False}, True, "advance_b"),
    ({"a": True, "b": True, "c": True}, True, "advance_a_b_c"),
])
def test_the_advancement_rule_is_a_function_not_a_paragraph(offline, actor,
                                                            expected):
    import proxy_benchmark

    verdicts = {a: _verdict(p, actor and p) for a, p in offline.items()}
    out = proxy_benchmark._decide(verdicts, {"pairwise": {}}, {})
    assert out["decision"] == expected
    assert set(out["advance"]) == {a for a, p in offline.items() if p and actor}


def test_offline_alone_never_advances_an_arm():
    """The property the whole milestone hangs on: no Rockstar, no fine-tune."""
    import proxy_benchmark

    out = proxy_benchmark._decide(
        {"a": _verdict(True, False), "b": _verdict(False, False)},
        {"pairwise": {}}, {})
    assert out["decision"] == "actor_gate_incomplete_do_not_finetune"
    assert out["advance"] == []
    assert out["awaiting_actor_gate"] == ["a"]
    assert "Rockstar" in out["rationale"]


def test_equivalent_arms_prefer_the_simpler_one():
    import proxy_benchmark

    both = {"a": _verdict(True, True), "b": _verdict(True, True)}
    out = proxy_benchmark._decide(
        both, {"pairwise": {"b-a": {"verdict": "equivalent"}}}, {})
    assert out["preferred"] == "a"
    assert set(out["advance"]) == {"a", "b"}

    ahead = proxy_benchmark._decide(
        both, {"pairwise": {"b-a": {"verdict": "second_better"}}}, {})
    assert ahead["preferred"] == "b"
    assert set(ahead["advance"]) == {"a", "b"}


def test_actor_refuses_without_a_benchmark(direct_cfg, fitted_reward, capsys):
    """Exit 0, not non-zero: a non-zero exit strands every afterok dependent."""
    import train_sr2_direct

    assert train_sr2_direct.main(
        ["--config", str(direct_cfg), "--run-name", "never"]) == 0
    assert "MISSING INPUT" in capsys.readouterr().out


def test_actor_refuses_when_the_benchmark_says_do_not_finetune(
        direct_cfg, fitted_reward, capsys):
    import train_sr2_direct
    from _sr2_direct import run_dir

    run = run_dir("t", create=True)
    (run / "proxy_benchmark.json").write_text(json.dumps({
        "passed": {"a": False, "b": False},
        "failures": {"a": ["within_tile_spearman=0.1 < 0.5"], "b": []},
        "decision": {"decision": "do_not_finetune", "advance": [],
                     "rationale": "Neither arm cleared its criteria."}}))
    assert train_sr2_direct.main(
        ["--config", str(direct_cfg), "--run-name", "t"]) == 0
    out = capsys.readouterr().out
    assert "do_not_finetune" in out
    # And it says the one thing the plan is emphatic about.
    assert "unfreezing more of the generator" in out.lower()


def test_actor_refuses_an_arm_the_benchmark_did_not_advance(
        direct_cfg, fitted_reward, capsys):
    import train_sr2_direct
    from _sr2_direct import run_dir

    run = run_dir("t", create=True)
    (run / "proxy_benchmark.json").write_text(json.dumps({
        "decision": {"decision": "advance_a_only", "advance": ["a"],
                     "rationale": "Only the density-only arm passed."}}))
    assert train_sr2_direct.main(
        ["--config", str(direct_cfg), "--run-name", "t", "--arm", "b"]) == 0
    assert "not in the benchmark's advance list" in capsys.readouterr().out

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
