"""Section 7: marginal contributions, elite selection, and manifest hygiene."""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import make_chunk

from cosmo_sr.reward.catalog import pool
from cosmo_sr.reward.replay import (ReplayEntry, elite_weights, marginal_contributions,
                                    read_replay, select_elites, sha256_file, write_replay)
from cosmo_sr.reward.reward import fit_reward_model


@pytest.fixture
def model(hr_chunks, bins):
    return fit_reward_model(hr_chunks, bins, ensemble_size=8, n_draws=200, seed=0)


def _ens(counts, cid_offset=0):
    return {i: make_chunk(box="set0", cid=i + cid_offset, source="candidate",
                          n_sub=c, n_host=(6, 2), occ=(18, 6), volume=900.0)
            for i, c in enumerate(counts)}


def test_counterfactual_reward_matches_an_explicit_swap(model):
    ens = _ens([(50, 14, 3), (45, 13, 3), (40, 12, 3)])
    base = _ens([(20, 6, 1), (20, 6, 1), (20, 6, 1)])
    a = marginal_contributions(model, ens, base, score="R_cat")
    for k in ens:
        swapped = [base[k] if kk == k else ens[kk] for kk in ens]
        expect = model.reward(pool(list(ens.values()))) - model.reward(pool(swapped))
        assert a[k] == pytest.approx(expect)


def test_an_improving_chunk_has_a_positive_marginal_contribution(model, bins):
    # The frozen baseline under-produces subhalos, so a chunk closer to mu_HR
    # must earn positive credit against its own baseline version.
    good = 10.0 ** model.mu[:bins.n_sub_bins] * 900.0
    ens = _ens([tuple(good), tuple(good), tuple(good)])
    base = _ens([(5, 1, 0), (5, 1, 0), (5, 1, 0)])
    a = marginal_contributions(model, ens, base, score="R_cat")
    assert all(v > 0 for v in a.values()), a


def test_a_harmful_chunk_has_a_negative_marginal_contribution(model):
    ens = _ens([(400, 120, 40), (45, 13, 3), (40, 12, 3)])   # chunk 0 wildly high
    base = _ens([(40, 12, 3), (40, 12, 3), (40, 12, 3)])
    a = marginal_contributions(model, ens, base, score="R_cat")
    assert a[0] < 0


def test_marginals_are_invariant_to_chunk_ordering(model):
    ens = _ens([(50, 14, 3), (45, 13, 3), (40, 12, 3)])
    base = _ens([(20, 6, 1), (22, 7, 1), (18, 5, 1)])
    a = marginal_contributions(model, ens, base, score="R_cat")
    rev_ens = {k: ens[k] for k in reversed(list(ens))}
    rev_base = {k: base[k] for k in reversed(list(base))}
    b = marginal_contributions(model, rev_ens, rev_base, score="R_cat")
    for k in a:
        assert a[k] == pytest.approx(b[k])


def test_a_missing_baseline_summary_is_an_error_not_a_zero(model):
    ens = _ens([(50, 14, 3), (45, 13, 3)])
    with pytest.raises(KeyError):
        marginal_contributions(model, ens, {0: list(ens.values())[0]})


def test_credit_defaults_to_occupation_not_the_joint_reward(model):
    """Gate B selects on occupation, so credit must be measured in occupation.

    These ensembles differ only in ABUNDANCE: the occupation numerator and host
    counts are identical everywhere. Crediting them in R_cat would hand out
    large positive weights for a statistic nobody selected on.
    """
    ens = _ens([(50, 14, 3), (45, 13, 3), (40, 12, 3)])
    base = _ens([(20, 6, 1), (20, 6, 1), (20, 6, 1)])
    occ = marginal_contributions(model, ens, base)          # default
    cat = marginal_contributions(model, ens, base, score="R_cat")
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in occ.values()), occ
    assert any(abs(v) > 1e-6 for v in cat.values()), cat


def test_occupation_credit_tracks_the_occupation_numerator(model):
    """A chunk whose hosts gained subhalos earns positive occupation credit."""
    from conftest import make_chunk

    def chunk(cid, occ_num):
        return make_chunk(box="set0", cid=cid, source="candidate",
                          n_sub=(45, 13, 3), n_host=(6, 2), occ=(occ_num, 6),
                          volume=900.0)

    target = float(10.0 ** model.mu[model.bins.n_sub_bins] * 6.0)
    ens = {i: chunk(i, target) for i in range(3)}
    base = {i: chunk(i, 1.0) for i in range(3)}
    a = marginal_contributions(model, ens, base)
    assert all(v > 0 for v in a.values()), a


def test_an_unknown_credit_score_is_refused(model):
    ens = _ens([(50, 14, 3)])
    with pytest.raises(ValueError, match="unknown score"):
        marginal_contributions(model, ens, ens, score="R_made_up")


def test_selection_requires_both_a_top_ensemble_and_a_positive_marginal():
    rows = [
        {"ensemble_id": "e_best", "feasible": True, "ensemble_reward": -1.0,
         "marginal_contribution": 0.5},
        {"ensemble_id": "e_best", "feasible": True, "ensemble_reward": -1.0,
         "marginal_contribution": -0.2},          # good ensemble, bad chunk
        {"ensemble_id": "e_mid", "feasible": True, "ensemble_reward": -5.0,
         "marginal_contribution": 0.9},           # good chunk, bad ensemble
        {"ensemble_id": "e_bad", "feasible": True, "ensemble_reward": -50.0,
         "marginal_contribution": 5.0},
    ]
    kept = select_elites(rows, elite_quantile=0.4)
    ids = [(r["ensemble_id"], r["marginal_contribution"]) for r in kept]
    assert ("e_best", 0.5) in ids
    assert all(m > 0 for _, m in ids)
    assert "e_bad" not in [i for i, _ in ids]


def test_an_infeasible_ensemble_can_never_become_an_elite():
    rows = [
        {"ensemble_id": "e0", "feasible": False, "ensemble_reward": 0.0,
         "marginal_contribution": 10.0},
    ]
    assert select_elites(rows, elite_quantile=1.0) == []


def test_the_quantile_is_taken_over_ensembles_not_chunks():
    # One ensemble contributing many chunks must not drag the cut down.
    rows = [{"ensemble_id": "big", "feasible": True, "ensemble_reward": -10.0,
             "marginal_contribution": 1.0} for _ in range(50)]
    rows.append({"ensemble_id": "good", "feasible": True, "ensemble_reward": -1.0,
                 "marginal_contribution": 1.0})
    kept = select_elites(rows, elite_quantile=0.5)
    assert {r["ensemble_id"] for r in kept} == {"good"}


def test_weights_are_bounded_and_normalised():
    w = elite_weights([0.0, 1.0, 100.0], tau=1.0, a_max=5.0, w_max=10.0)
    assert w.max() <= 10.0
    assert np.all(w >= 0)
    assert w[2] == w.max()
    # a_max must bind, or one lucky ensemble owns every batch.
    assert w[2] / max(w[0], 1e-12) == pytest.approx(np.exp(5.0), rel=0.5)


def test_negative_marginals_get_the_baseline_weight():
    w = elite_weights([-3.0, 0.0], tau=1.0)
    assert w[0] == pytest.approx(w[1])


def test_replay_manifest_round_trips_and_verifies_checksums(tmp_path):
    field = tmp_path / "resid.npy"
    np.save(field, np.zeros((6, 4, 4, 4), dtype=np.float32))
    e = ReplayEntry(
        box="set0", chunk_id=3, ensemble_id="run:group0:set0_seed1",
        residual_path=str(field), residual_sha256=sha256_file(field),
        lr_id="set0", base_id="base.npy", base_seed=0, residual_seed=1,
        residual_scale=1.0, ensemble_reward=-1.0, marginal_contribution=0.4,
        constraint_values={"low_k_change": 0.001}, feasible=True,
        generation_checkpoint="ckpt.pt", hr_origin=(128, 0, 256), chunk_hr=128,
        weight=1.2,
    )
    p = write_replay(tmp_path / "replay.jsonl", [e])
    back = read_replay(p, verify=True)
    assert len(back) == 1
    assert back[0].hr_origin == (128, 0, 256)
    assert back[0].to_dict() == e.to_dict()


def test_a_corrupted_replay_field_is_caught(tmp_path):
    field = tmp_path / "resid.npy"
    np.save(field, np.zeros((6, 4, 4, 4), dtype=np.float32))
    e = ReplayEntry(
        box="set0", chunk_id=0, ensemble_id="x", residual_path=str(field),
        residual_sha256=sha256_file(field), lr_id="set0", base_id="b", base_seed=0,
        residual_seed=0, residual_scale=1.0, ensemble_reward=0.0,
        marginal_contribution=0.1,
    )
    p = write_replay(tmp_path / "replay.jsonl", [e])
    np.save(field, np.ones((6, 4, 4, 4), dtype=np.float32))
    with pytest.raises(RuntimeError, match="checksum"):
        read_replay(p, verify=True)


def test_a_missing_replay_field_is_caught(tmp_path):
    e = ReplayEntry(
        box="set0", chunk_id=0, ensemble_id="x", residual_path=str(tmp_path / "gone.npy"),
        residual_sha256="", lr_id="set0", base_id="b", base_seed=0, residual_seed=0,
        residual_scale=1.0, ensemble_reward=0.0, marginal_contribution=0.1,
    )
    p = write_replay(tmp_path / "replay.jsonl", [e])
    with pytest.raises(FileNotFoundError):
        read_replay(p, verify=True)


def test_training_refuses_a_replay_buffer_containing_held_out_boxes(tmp_path):
    """The one leak that would invalidate every downstream number."""
    from cosmo_sr.reward.train import run_training

    field = tmp_path / "resid.npy"
    np.save(field, np.zeros((6, 8, 8, 8), dtype=np.float32))
    e = ReplayEntry(
        box="set9", chunk_id=0, ensemble_id="x", residual_path=str(field),
        residual_sha256="", lr_id="set9", base_id="b", base_seed=0, residual_seed=0,
        residual_scale=1.0, ensemble_reward=0.0, marginal_contribution=0.1,
    )
    manifest = write_replay(tmp_path / "replay.jsonl", [e])
    cfg = {
        "split": {"train_boxes": ["set0"], "val_boxes": ["set9"]},
        "data": {"root": str(tmp_path), "crop_hr": 8, "scale_factor": 4},
        "model": {"channels": 6, "width": 4, "num_levels": 1, "blocks_per_level": 1,
                  "embed_dim": 8, "num_groups": 2},
        "train": {"steps": 1, "batch_size": 1, "num_workers": 0},
        "distill": {"replay_manifest": str(manifest)},
        "output": {"run_dir": str(tmp_path / "run")},
    }
    with pytest.raises(Exception) as ei:
        run_training(cfg, mode="distill", run_dir=str(tmp_path / "run"))
    assert "set9" in str(ei.value) or "non-training" in str(ei.value)
