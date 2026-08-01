"""Rung 2: the parts of the CEM search that decide things, on the CPU.

The generation step needs a GPU and a trained prior, so what is covered here is
the logic that would silently pick the wrong candidate: the perturbation's
amplitude behaviour, and the ranking rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reward"))

from cem_search import candidate_id, perturb  # noqa: E402


def test_a_child_stays_on_the_unit_shell():
    """Otherwise search would win by turning the residual amplitude up.

    ``elite + sigma * xi`` has variance ``1 + sigma^2``; left unnormalised the
    population's RMS would grow every iteration, and a bigger residual scores
    differently for reasons that have nothing to do with the search.
    """
    rng = np.random.default_rng(0)
    elite = rng.standard_normal((6, 16, 16, 16)).astype(np.float32)
    assert abs(float(elite.std()) - 1.0) < 0.02

    for sigma in (0.1, 0.3, 1.0):
        child = perturb(elite, sigma, rng)
        assert abs(float(child.std()) - 1.0) < 0.03, sigma


def test_a_child_is_nearer_its_parent_for_a_smaller_sigma():
    rng = np.random.default_rng(1)
    elite = rng.standard_normal((6, 8, 8, 8)).astype(np.float32)
    near = perturb(elite, 0.05, np.random.default_rng(2))
    far = perturb(elite, 1.0, np.random.default_rng(2))
    assert np.abs(near - elite).mean() < np.abs(far - elite).mean()


def test_candidate_ids_do_not_collide_across_iterations():
    """score_oracle.py keys on `seed`; a collision would overwrite a candidate."""
    seen = set()
    for it in range(6):
        for j in range(8):
            cid = candidate_id(it, j)
            assert cid not in seen
            seen.add(cid)


def _ranked(cands):
    """The ranking rule in cem_select_elites.py, applied to plain dicts."""
    out = list(cands)
    out.sort(key=lambda c: (c["feasible_field"], c["R_occ"]), reverse=True)
    return out


def test_ranking_puts_the_best_occupation_first():
    ranked = _ranked([
        {"seed": 0, "R_occ": -5.0, "R_cat": -1.0, "feasible_field": True},
        {"seed": 1, "R_occ": -1.0, "R_cat": -9.0, "feasible_field": True},
        {"seed": 2, "R_occ": -3.0, "R_cat": -2.0, "feasible_field": True},
    ])
    # Ranked on occupation, not the joint reward: candidate 1 has the worst
    # R_cat and must still win, because a run that fixes abundance and leaves
    # occupation flat is the informative failure, not a partial success.
    assert [c["seed"] for c in ranked] == [1, 2, 0]


def test_an_infeasible_candidate_never_outranks_a_feasible_one():
    ranked = _ranked([
        {"seed": 0, "R_occ": -9.0, "feasible_field": True},
        {"seed": 1, "R_occ": -0.1, "feasible_field": False},
    ])
    assert ranked[0]["seed"] == 0, "an infeasible field cannot be an elite"
