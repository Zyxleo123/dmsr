"""Object forensics that decide whether kNN or the raw count is right.

The fragment and matching logic is what the verdict rests on, so it is pinned on
hand-built catalogs where the answer is known: a faithful arm, a fragmenting arm
(big halo gone, unmatched small pieces where it was), and an arm whose extra
subhalos are in the wrong place.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "scripts/features"):
    sys.path.insert(0, str(ROOT / p))

import knn_gate_crosscheck as kc  # noqa: E402
from cosmo_sr.eval.rockstar import HaloCatalog  # noqa: E402
from cosmo_sr.reward.tiles import TileGrid  # noqa: E402


def _cat(rows):
    """rows: (id, parent, num_p, rvir_kpc, x, y, z)"""
    r = np.array(rows, dtype=float)
    n = len(rows)
    return HaloCatalog(ids=r[:, 0].astype(int), parent_ids=r[:, 1].astype(int),
                       mvir=r[:, 2] * 1e9, rvir=r[:, 3], vmax=np.zeros(n),
                       pos=r[:, 4:7], vel=np.zeros((n, 3)),
                       num_p=r[:, 2].astype(int))


BOXES = kc.tile_boxes([0], TileGrid(ng_hr=512, tile_hr=64, boxsize_mpc_h=100.0))
HOST = (1, -1, 5000, 400, 6.0, 6.0, 6.0)
HR = _cat([HOST,
           (2, 1, 120, 60, 5.0, 5.0, 5.0),
           (3, 1, 80, 50, 7.0, 7.0, 7.0),
           (4, 1, 60, 40, 9.0, 3.0, 4.0),
           (5, -1, 900, 150, 3.0, 9.0, 9.0)])       # a >=200p halo HR has
FROZEN = _cat([HOST, (6, -1, 900, 150, 3.0, 9.0, 9.0)])


def test_faithful_arm_matches_hr():
    arm = _cat([HOST, (7, 1, 110, 60, 5.02, 5.0, 5.0), (8, 1, 90, 50, 7.0, 7.03, 7.0),
                (9, -1, 880, 150, 3.0, 9.0, 9.0)])
    row = kc.forensics(arm, HR, FROZEN, BOXES, 2.0, 2.0)
    assert row["sub_50_200"] == 2 and row["sub_50_200_matched"] == 2
    assert row["hr_recall_50_200"] == 2 / 3
    assert row["lost_ge200"] == 0 and row["fragment_candidates"] == 0


def test_fragmenting_arm_is_caught():
    # The 900p halo at (3,9,9) is gone; four 60p pieces sit inside its radius.
    pieces = [(10 + i, 1, 60, 30, 3.0 + 0.03 * i, 9.0, 9.0) for i in range(4)]
    arm = _cat([HOST] + pieces)
    row = kc.forensics(arm, HR, FROZEN, BOXES, 2.0, 2.0)
    assert row["sub_50_200"] == 4
    assert row["sub_50_200_matched"] == 0       # not HR objects
    assert row["lost_ge200"] == 1 and row["lost_ge200_hr_backed"] == 1
    assert row["fragment_candidates"] == 4


def test_misplaced_extras_are_unmatched_but_not_fragments():
    extras = [(20 + i, 1, 70, 40, 11.0, 1.0 + 0.5 * i, 11.0) for i in range(3)]
    arm = _cat([HOST, (9, -1, 880, 150, 3.0, 9.0, 9.0)] + extras)
    row = kc.forensics(arm, HR, FROZEN, BOXES, 2.0, 2.0)
    assert row["sub_50_200"] == 3 and row["sub_50_200_matched"] == 0
    assert row["sub_50_200_precision"] == 0.0
    assert row["lost_ge200"] == 0 and row["fragment_candidates"] == 0


def test_mass_factor_gates_a_match():
    arm = _cat([HOST, (7, 1, 60, 60, 5.0, 5.0, 5.0)])     # 60p vs HR's 120p: x2
    assert kc.forensics(arm, HR, FROZEN, BOXES, 2.0, 2.0)["sub_50_200_matched"] == 1
    assert kc.forensics(arm, HR, FROZEN, BOXES, 2.0, 1.5)["sub_50_200_matched"] == 0


def test_region_excludes_far_halos():
    arm = _cat([HOST, (7, 1, 110, 60, 50.0, 50.0, 50.0)])
    assert kc.forensics(arm, HR, FROZEN, BOXES, 2.0, 2.0)["sub_50_200"] == 0


def test_knn_terms_reads_split_json_labels():
    knn = {"loss_forms": {"all_blocks_x": {
        "unpaired_total": 1.0, "class_cond_mass_total": 2.0,
        "class_cond_equal_total": 3.0, "paired_total": 4.0,
        "class_cond[sub_50_200]": [0.1, 0.3], "paired[host]": [1.0, 3.0]}}}
    t = kc.knn_terms(knn, "all_blocks_x")
    assert t["knn_paired"] == 4.0 and t["knn_class_cond[sub_50_200]"] == 0.2
    assert t["knn_paired[host]"] == 2.0
    assert kc.knn_terms(knn, "missing") is None


def test_main_end_to_end(tmp_path, monkeypatch):
    import json
    root = tmp_path / "rr"
    (root / "halos" / "set9__hr__hr" / "hr_rockstar").mkdir(parents=True)
    (root / "halos" / "set9__hr__hr" / "hr_rockstar" / "halos_0.0.ascii").write_text("hr")
    arms = {"mgho_frozen_set9": FROZEN}
    rng = np.random.default_rng(0)
    for i in range(6):
        pieces = [(10 + j, 1, 60, 30, 3.0 + 0.03 * j, 9.0, 9.0) for j in range(i)]
        arms[f"mgho_all_blocks_a{i}_set9"] = _cat([HOST] + pieces) if i else FROZEN
    for tag in arms:
        d = root / "flow_rockstar" / "halos" / f"set9__candidate__{tag}" / f"{tag}_rockstar"
        d.mkdir(parents=True)
        (d / "halos_0.0.ascii").write_text(tag)
    def fake_load(p):
        txt = Path(p).read_text()
        return HR if txt == "hr" else arms[txt]
    monkeypatch.setattr(kc, "load_rockstar_ascii", fake_load)
    (tmp_path / "region.json").write_text(json.dumps({"tiles": [0]}))
    knn = {"loss_forms": {}}
    for i, tag in enumerate(["frozen"] + [f"all_blocks_a{i}" for i in range(6)]):
        v = float(i)
        knn["loss_forms"][tag] = {
            "unpaired_total": v, "class_cond_mass_total": v,
            "class_cond_equal_total": v, "paired_total": v,
            "class_cond[sub_50_200]": [v, v], "class_cond[host]": [0.0, 0.0]}
    (tmp_path / "knn.json").write_text(json.dumps(knn))
    out = tmp_path / "o.json"
    kc.main(["--reward-root", str(root), "--region-json", str(tmp_path / "region.json"),
             "--knn-json", str(tmp_path / "knn.json"), "--out", str(out)])
    r = json.loads(out.read_text())
    tags = {x["tag"] for x in r["rows"]}
    assert "hr" in tags and "mgho_frozen_set9" in tags and len(tags) == 8
    assert r["n_arms_with_knn"] == 6
    # More fragments <-> larger kNN loss, by construction: rho = +1.
    c = r["spearman"]["knn_class_cond[sub_50_200]"]["fragment_candidates"]
    assert c["rho"] > 0.99
    import knn_gate_crosscheck_plot as kp
    kp.main(["--json", str(out), "--out", str(tmp_path / "o.png")])
    assert (tmp_path / "o.png").stat().st_size > 10_000
