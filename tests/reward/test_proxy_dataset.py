"""Phases 1-3: the predeclared matrix, the completeness gate, the label identity.

These are the properties the label workflow was rebuilt around, and each one
corresponds to a way the previous version was wrong:

* the matrix covered only the fit boxes, so the held-out boxes had no rows;
* the index was rewritten by every labelling job, so it could be read torn;
* a candidate whose tile decomposition did not sum to its own whole-box catalog
  was reported and then used anyway.
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

from _proxy_matrix import candidate_matrix, candidate_tag  # noqa: E402

CONFIG = yaml.safe_load(
    (PROJECT_ROOT / "configs" / "reward" / "sr2_direct_finetune.yaml").read_text())


# --------------------------------------------------------------------------- #
# The predeclared matrix
# --------------------------------------------------------------------------- #
def test_the_matrix_covers_the_held_out_boxes_too():
    """96 core runs over TWELVE boxes, not eight.

    The fit boxes alone cannot gate anything: a proxy has to be scored on boxes
    it never saw, and those boxes need labels of their own.
    """
    m = candidate_matrix(CONFIG)
    boxes = {c["box"] for c in m}
    assert boxes == {f"set{i}" for i in range(12)}
    assert boxes >= set(CONFIG["split"]["proxy_gate_boxes"])
    # Eight core candidates per box: HR + four frozen seeds + three alphas.
    core = [c for c in m if c["mode"] == CONFIG["proxy_dataset"]["core_intervention_mode"]]
    assert len(core) == 8 * 12


def test_the_sealed_boxes_appear_nowhere():
    m = candidate_matrix(CONFIG)
    boxes = {c["box"] for c in m}
    assert not (boxes & set(CONFIG["split"]["sealed_boxes"]))
    assert not (boxes & set(CONFIG["split"]["preflight_boxes"]))


def test_the_diagnostic_subset_is_the_only_extra():
    """disp-only and vel-only ladders, on the predeclared four boxes only."""
    m = candidate_matrix(CONFIG)
    d = CONFIG["proxy_dataset"]
    extra = [c for c in m if c["mode"] != d["core_intervention_mode"]]
    assert {c["box"] for c in extra} == set(d["diagnostic_boxes"])
    assert {c["mode"] for c in extra} == set(d["diagnostic_modes"])
    assert len(extra) == len(d["diagnostic_boxes"]) * len(d["diagnostic_modes"]) \
        * len(d["alphas"])
    assert len(m) == 96 + len(extra)


def test_tags_are_unique_and_carry_the_channel_mode():
    """Without mode in the tag the three ladders collide on one directory.

    They differ only in which channels the intervention touched, so the last one
    written would silently become all three -- and the velocity-only diagnostic,
    the one thing that can attribute an arm-B win, would not exist.
    """
    m = candidate_matrix(CONFIG)
    keys = [(c["box"], c["tag"]) for c in m]
    assert len(set(keys)) == len(keys)
    assert candidate_tag("intervention", alpha=0.5, mode="vel") != \
        candidate_tag("intervention", alpha=0.5, mode="disp")
    assert candidate_tag("hr") == "hr"
    assert candidate_tag("frozen", seed=0) == candidate_tag("frozen_seed", seed=0)


def test_a_restricted_box_list_still_enumerates_the_same_way():
    m = candidate_matrix(CONFIG, boxes=["set2"])
    assert {c["box"] for c in m} == {"set2"}
    # set2 is not a diagnostic box, so it gets the eight core candidates only.
    assert len(m) == 8


# --------------------------------------------------------------------------- #
# Indexing and the completeness gate
# --------------------------------------------------------------------------- #
@pytest.fixture
def small_cfg(tmp_path, reward_root):
    cfg = yaml.safe_load(json.dumps(CONFIG))  # deep copy through json
    cfg["proxy_dataset"].update({"boxes": ["set0"], "frozen_seeds": [0, 1],
                                 "alphas": [1.0], "diagnostic_boxes": []})
    p = tmp_path / "direct.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _fake_candidate(box, tag, *, n_tiles=2, label_ok=True, additivity=0.0,
                    field_sha="f00d"):
    """A candidate directory with the files the indexer reads, and nothing else."""
    from _sr2_direct import direct_root, write_json_atomic
    from cosmo_sr.reward.phase_space import arm_paired_feature_names

    d = direct_root("candidates", f"{box}__{tag}", create=True)
    feats = {"tile_id": np.arange(n_tiles, dtype=np.int64)}
    for arm in ("a", "b"):
        n = len(arm_paired_feature_names(arm))
        feats[f"features_{arm}"] = np.arange(n_tiles * n,
                                             dtype=np.float32).reshape(n_tiles, n)
    np.savez_compressed(d / "features.npz", **feats)
    write_json_atomic(d / "manifest.json", {
        "box": box, "tag": tag, "source": ("hr" if tag == "hr" else "frozen"),
        "seed": 0, "alpha": None, "mode": "both",
        "features_path": str(d / "features.npz"), "field_path": str(d / "field.npy"),
        "field_sha": field_sha, "model_sha": "m", "lr_sha": "l",
        "code_commit": "test"})

    summaries = d / "tile_summaries.jsonl"
    with open(summaries, "w") as fh:
        for t in range(n_tiles):
            fh.write(json.dumps({
                "box": box, "tile_id": t, "source": tag,
                "n_sub": [1.0] * 6, "n_host": [1.0] * 5,
                "occ_numerator": [1.0] * 5, "volume_mpc3": 24.4}) + "\n")
    write_json_atomic(d / "label_report.json", {
        "box": box, "tag": tag, "field_sha": field_sha,
        "tile_summaries_path": str(summaries),
        "label_ok": bool(label_ok), "member_consistency_ok": True,
        "additivity_ok": additivity == 0.0,
        "tile_sum_vs_direct_max_abs_residual": {"n_host": additivity},
        "full_box": {"n_sub": [2.0] * 6, "n_host": [2.0] * 5,
                     "occ_numerator": [2.0] * 5, "occupation": [1.0] * 5}})
    return d


def _index(small_cfg):
    import collect_catalog_proxy_data as C
    return C.main(["--config", str(small_cfg), "--stage", "index"])


def test_no_completeness_marker_while_candidates_are_missing(small_cfg, capsys):
    from _sr2_direct import labels_complete_path

    _fake_candidate("set0", "hr")
    assert _index(small_cfg) == 0
    assert not labels_complete_path().is_file()
    out = capsys.readouterr().out
    assert "labels INCOMPLETE" in out
    assert "no proxy trainer will" in out


def test_the_marker_appears_only_when_the_whole_matrix_is_labelled(small_cfg):
    from _sr2_direct import direct_root, labels_complete_path

    for c in candidate_matrix(yaml.safe_load(small_cfg.read_text())):
        _fake_candidate(c["box"], c["tag"])
    assert _index(small_cfg) == 0
    marker = labels_complete_path()
    assert marker.is_file()
    doc = json.loads(marker.read_text())
    assert doc["complete"] and doc["rows"] > 0
    report = json.loads((direct_root("proxy_data") / "index_report.json").read_text())
    assert report["missing_from_predeclared_matrix"] == []
    assert report["n_arms"]["b"] > report["n_arms"]["a"]


def test_a_candidate_that_fails_the_additivity_identity_is_excluded(small_cfg):
    """Reported AND excluded. Its tile rows decompose something else.

    ``sum_t N_t = N`` is the identity every label rests on; a candidate that
    breaks it has a member table and a catalog describing different runs, and
    including it would put unattributable counts into the training targets.
    """
    from _sr2_direct import direct_root, labels_complete_path

    matrix = candidate_matrix(yaml.safe_load(small_cfg.read_text()))
    for i, c in enumerate(matrix):
        _fake_candidate(c["box"], c["tag"], label_ok=(i != 0),
                        additivity=(0.0 if i != 0 else 3.5))
    assert _index(small_cfg) == 0
    report = json.loads((direct_root("proxy_data") / "index_report.json").read_text())
    assert report["candidates_invalid"]
    assert matrix[0]["tag"] in report["missing_from_predeclared_matrix"][0]
    # One bad candidate is enough to withhold the marker: the dataset is the
    # predeclared matrix, not "whatever passed".
    assert not labels_complete_path().is_file()


def test_a_label_describing_a_stale_field_is_excluded(small_cfg):
    """The catalog must describe the field the features came from.

    A regenerated field with a stale label is the one failure that produces a
    perfectly well-formed, entirely meaningless training row.
    """
    from _sr2_direct import direct_root

    matrix = candidate_matrix(yaml.safe_load(small_cfg.read_text()))
    for c in matrix:
        _fake_candidate(c["box"], c["tag"])
    d = direct_root("candidates", f"{matrix[0]['box']}__{matrix[0]['tag']}")
    rep = json.loads((d / "label_report.json").read_text())
    rep["field_sha"] = "a-different-field"
    (d / "label_report.json").write_text(json.dumps(rep))

    assert _index(small_cfg) == 0
    report = json.loads((direct_root("proxy_data") / "index_report.json").read_text())
    assert any("stale field" in str(x.get("reason", ""))
               for x in report["candidates_invalid"])


def test_the_index_is_written_atomically(small_cfg):
    """No .tmp survives, so a reader sees a complete table or the previous one."""
    from _sr2_direct import direct_root

    for c in candidate_matrix(yaml.safe_load(small_cfg.read_text())):
        _fake_candidate(c["box"], c["tag"])
    _index(small_cfg)
    leftovers = list(direct_root("proxy_data").glob("*.tmp*"))
    assert leftovers == []
    rows = (direct_root("proxy_data") / "rows.jsonl").read_text().splitlines()
    assert rows and all(json.loads(r)["features_a"] for r in rows)


def test_index_warns_when_there_is_no_within_tile_variation(small_cfg, capsys):
    _fake_candidate("set0", "hr")
    _index(small_cfg)
    out = capsys.readouterr().out
    assert "has_within_tile_variation" in out
    assert "easy binary distinction" in out


# --------------------------------------------------------------------------- #
# Row weighting
# --------------------------------------------------------------------------- #
def test_source_balance_and_changed_tile_weighting():
    """HR and frozen are most of the rows and the easy ones.

    Unweighted, the loss is mostly "HR has more substructure than SR2" -- true,
    already known, and not a direction the actor can move in.
    """
    from _proxy_data import row_weights

    source = np.asarray(["hr"] * 1 + ["frozen"] * 1 + ["intervention"] * 8)
    feats = np.zeros((10, 4))
    feats[:, 2:] = np.asarray([[0.0, 0.0]] * 2 + [[3.0, 0.0]] * 8)
    out = row_weights(source, feats, balance_sources=True, changed_weight=3.0,
                      changed_threshold=0.5)
    w = out["weight"]
    assert np.isclose(w.mean(), 1.0)
    # The single hr row must not be worth less than one of eight interventions.
    assert w[0] > w[2]
    assert out["changed"][2:].all() and not out["changed"][:2].any()


def test_zero_weight_rows_drop_out_of_both_losses():
    """That is how a box the bootstrap did not draw is expressed."""
    import torch
    from cosmo_sr.reward.catalog_proxy import count_loss, pairwise_ranking_loss

    pred = {"n_sub": torch.ones(4, 2), "n_host": torch.ones(4, 2),
            "occ_numerator": torch.ones(4, 2)}
    target = {k: v.clone() for k, v in pred.items()}
    target["n_sub"][2:] = 99.0            # only the excluded rows are wrong
    rw = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    assert float(count_loss(pred, target, row_weights=rw)["loss"]) == 0.0
    assert float(count_loss(pred, target)["loss"]) > 0.0

    better, worse = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    unweighted = float(pairwise_ranking_loss(better, worse))
    weighted = float(pairwise_ranking_loss(better, worse, torch.tensor([1.0, 0.0])))
    assert weighted < unweighted
