"""The go/no-go diagnostic must measure what the loss measures.

It is the reference the training term is read against, so two agreements are
pinned on a 32^3 box small enough to run anywhere: its particle table must be
``tile_particles``'s table (or its "same Lagrangian ids" are different particles),
and its scipy search must return the torch search's distances (or the verdict and
the loss disagree about the statistic, which is indistinguishable from a real
result).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cosmo_sr.features.knn_cdf import KnnConfig, knn_log_distances  # noqa: E402
from cosmo_sr.features.member_gather import tile_particles  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "knn_cdf_diagnostic", ROOT / "scripts/features/knn_cdf_diagnostic.py")
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)

NG, TILE = 32, 8


def _small_box(monkeypatch):
    monkeypatch.setattr(diag, "NG", NG)
    monkeypatch.setattr(diag, "TILE", TILE)
    monkeypatch.setattr(diag, "CELL", diag.BOX_MPC / NG)


def _field(tmp_path, name, amp, seed):
    rng = np.random.default_rng(seed)
    f = np.zeros((6, NG, NG, NG), dtype=np.float32)
    f[:3] = amp * rng.standard_normal((3, NG, NG, NG))
    path = tmp_path / f"{name}.npy"
    np.save(path, f)
    return str(path), f


def test_positions_match_tile_particles(tmp_path, monkeypatch):
    _small_box(monkeypatch)
    path, f = _field(tmp_path, "a", 0.05, 1)
    pos = diag.load_positions(path)
    tid = 13
    n_side = NG // TILE
    tx, ty, tz = tid // (n_side * n_side), (tid // n_side) % n_side, tid % n_side
    sub = f[:, tx * TILE:(tx + 1) * TILE, ty * TILE:(ty + 1) * TILE,
            tz * TILE:(tz + 1) * TILE]
    tp, _ = tile_particles(torch.tensor(sub[None], dtype=torch.float64), [tid],
                           ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=diag.BOX_MPC,
                           dis_scale_mpc_h=diag.DIS_SCALE_MPC_H)
    li = np.arange(TILE)
    gi, gj, gk = np.meshgrid(tx * TILE + li, ty * TILE + li, tz * TILE + li,
                             indexing="ij")
    rows = ((gi * NG + gj) * NG + gk).reshape(-1)
    # float32 on disk: agreement to float32 precision in Mpc/h.
    d = pos[rows] - tp.numpy()
    d -= diag.BOX_MPC * np.round(d / diag.BOX_MPC)
    assert np.abs(d).max() < 1e-4


def test_scipy_search_matches_torch_search(tmp_path, monkeypatch):
    _small_box(monkeypatch)
    path, _ = _field(tmp_path, "a", 0.3, 2)
    pos = diag.load_positions(path)
    ks = np.array([2, 8, 16])
    ids = np.arange(0, NG ** 3, 97)
    tree = diag.cKDTree(pos, boxsize=diag.BOX_MPC)
    r_scipy = diag.knn_full_box(tree, pos, ids, ks, workers=1)
    cfg = KnnConfig(ks=tuple(ks.tolist()), buffer_cells=0.0)
    z = knn_log_distances(torch.tensor(pos), torch.tensor(ids), cfg,
                          boxsize_mpc_h=diag.BOX_MPC)
    assert np.allclose(r_scipy, z.exp().numpy(), rtol=1e-8, atol=1e-5)


def test_end_to_end_json(tmp_path, monkeypatch):
    _small_box(monkeypatch)
    hr, _ = _field(tmp_path, "hr", 0.3, 3)
    cand, _ = _field(tmp_path, "cand", 0.1, 3)
    out = tmp_path / "out.json"
    diag.main(["--fields", f"hr={hr}", f"cand={cand}", "missing=/nope.npy",
               "--ks", "2", "8", "--n-sample", "2000", "--n-quantiles", "16",
               "--buffers", "0", "2", "--boundary-tiles", "4",
               "--boundary-queries", "64", "--workers", "1",
               "--out", str(out)])
    res = json.loads(out.read_text())
    assert set(res["fields"]) == {"hr", "cand"}
    c = res["fields"]["cand"]
    assert len(c["ratio_q10"]) == 2 and len(c["clumped_recall"]) == 2
    assert np.array(c["ratio_bins"]).shape == (16, 2)
    recs = [r for r in res["boundary"]["records"] if r.get("n_queries")]
    assert recs
    # A subset of the candidates can never find a CLOSER neighbour.
    for r in recs:
        assert r["rel_err_median"] >= 0.0 and r["rel_err_p95"] >= 0.0
        assert 0.0 <= r["incomplete_frac"] <= 1.0
