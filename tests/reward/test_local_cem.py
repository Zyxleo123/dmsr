"""The CEM search: determinism, that it actually searches, and that resume works.

A search is the easiest component to have quietly broken, because a broken one
still produces a "best" candidate every round. The three things pinned here are
the ones whose failure looks like a result: a distribution that never moves, one
that collapses onto a tie, and a resumed run that silently explores somewhere
else than the run it claims to continue.
"""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.reward.cem import CEMRun, CEMState, elite_threshold


def state(dim=4, **kw):
    kw.setdefault("n_samples", 32)
    return CEMState.initial(dim, seed=7, sigma=1.5, **kw)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_same_state_samples_the_same_population_every_time():
    s = state()
    assert np.array_equal(s.sample(), s.sample())


def test_different_rounds_sample_different_populations():
    s = state()
    t, _ = s.update(s.sample(), np.arange(32.0))
    assert not np.allclose(s.sample(), t.sample())


def test_the_seed_is_the_only_source_of_randomness():
    a = CEMState.initial(4, seed=1).sample()
    b = CEMState.initial(4, seed=1).sample()
    c = CEMState.initial(4, seed=2).sample()
    assert np.array_equal(a, b) and not np.allclose(a, c)


def test_the_exploration_draws_are_the_last_rows_and_come_from_the_initial_prior():
    s = state(explore_mix=0.25)
    s2, _ = s.update(s.sample(), np.arange(32.0))
    s2.mean = s2.mean + 50.0        # move the search far from the prior
    z = s2.sample()
    k = s2.n_explore()
    assert k == 8
    assert np.all(s2.is_explore()[-k:]) and not np.any(s2.is_explore()[:-k])
    # The exploration block must still be centred on the ORIGINAL mean, which is
    # the whole point of mixing it back in.
    assert abs(z[-k:].mean()) < 10.0 < abs(z[:-k].mean())


# ---------------------------------------------------------------------------
# Updating
# ---------------------------------------------------------------------------


def test_cem_improves_on_a_synthetic_objective():
    """Quadratic bowl centred away from the initial mean. Three rounds must move
    the mean towards it and shrink the spread."""
    target = np.array([2.0, -1.5, 0.5, 1.0])
    s = state(n_samples=64, sigma_floor=0.05)
    d0 = float(np.linalg.norm(s.mean - target))
    for _ in range(4):
        z = s.sample()
        r = -np.sum((z - target) ** 2, axis=1)
        s, info = s.update(z, r)
        assert info["updated"]
    assert float(np.linalg.norm(s.mean - target)) < 0.35 * d0
    assert float(s.std.mean()) < 1.5


def test_the_variance_floor_stops_a_collapse():
    s = state(n_samples=32, sigma_floor=0.4)
    for _ in range(6):
        z = s.sample()
        s, _ = s.update(z, -np.sum(z ** 2, axis=1))
    assert np.all(s.std >= 0.4 - 1e-12)


def test_a_round_with_no_feasible_candidate_does_not_update():
    s = state()
    z = s.sample()
    t, info = s.update(z, np.arange(32.0), feasible=np.zeros(32, dtype=bool))
    assert not info["updated"] and info["reason"] == "no_feasible_candidate"
    assert np.array_equal(t.mean, s.mean) and np.array_equal(t.std, s.std)
    assert t.round_index == s.round_index + 1


def test_a_round_where_every_candidate_ties_does_not_update():
    """The overwhelmingly likely early case: nothing creates an object and every
    reward is 0. Fitting to an arbitrary subset of ties is how a search convinces
    itself it is progressing."""
    s = state()
    z = s.sample()
    t, info = s.update(z, np.zeros(32))
    assert not info["updated"] and info["reason"] == "all_candidates_equivalent"
    assert np.array_equal(t.mean, s.mean)


def test_infeasible_candidates_cannot_become_elites():
    s = state(n_samples=8, min_elites=2, elite_frac=0.5)
    z = np.zeros((8, 4))
    z[0] = 100.0                       # the best reward, but infeasible
    r = np.array([9.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    feas = np.ones(8, dtype=bool)
    feas[0] = False
    t, info = s.update(z, r, feas)
    assert info["updated"] and info["n_feasible"] == 7
    assert np.all(np.abs(t.mean) < 1.0), "the infeasible outlier moved the mean"


def test_elites_are_the_top_fraction():
    r = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]
    assert elite_threshold(r, 0.2, min_elites=2) == 4.0
    assert elite_threshold(r, 0.5, min_elites=2) == 1.0


def test_the_history_records_why_a_round_did_not_update():
    s = state()
    s, _ = s.update(s.sample(), np.zeros(32))
    s, _ = s.update(s.sample(), np.arange(32.0))
    reasons = [h["reason"] for h in s.history]
    assert reasons == ["all_candidates_equivalent", "refit"]


def test_a_degenerate_configuration_is_rejected_at_construction():
    with pytest.raises(ValueError):
        CEMState.initial(4, elite_frac=0.0)
    with pytest.raises(ValueError):
        CEMState.initial(4, explore_mix=1.0)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_resume_reproduces_an_uninterrupted_run_exactly(tmp_path):
    """The property the whole manifest design exists for: a run that stopped
    after round 1 and was resumed emits bit-identical round-2 candidates."""
    def rewards(z):
        return -np.sum((z - 1.0) ** 2, axis=1)

    # (a) uninterrupted
    live = state(n_samples=32)
    run_a = CEMRun(root=tmp_path / "a", name="both")
    for _ in range(2):
        z = live.sample()
        run_a.write_round(live, z)
        run_a.record_rewards(live.round_index, rewards(z).tolist())
        live, _ = live.update(z, rewards(z))
    uninterrupted = live.sample()

    # (b) the same manifests, replayed from disk by a fresh process's state
    resumed, infos = CEMRun(root=tmp_path / "a", name="both").resume(state(n_samples=32))
    assert len(infos) == 2
    assert resumed.round_index == 2
    assert np.array_equal(resumed.sample(), uninterrupted)
    assert np.array_equal(resumed.mean, live.mean)


def test_resume_stops_at_the_first_unscored_round(tmp_path):
    s = state(n_samples=16)
    run = CEMRun(root=tmp_path, name="disp")
    z = s.sample()
    run.write_round(s, z)
    run.record_rewards(0, (-np.sum(z ** 2, axis=1)).tolist())
    nxt, _ = s.update(z, -np.sum(z ** 2, axis=1))
    run.write_round(nxt, nxt.sample())          # proposed, not yet scored

    resumed, infos = CEMRun(root=tmp_path, name="disp").resume(state(n_samples=16))
    assert resumed.round_index == 1 and len(infos) == 1


def test_a_missing_round_manifest_is_an_error_not_a_silent_skip(tmp_path):
    s = state(n_samples=8)
    run = CEMRun(root=tmp_path, name="both")
    z = s.sample()
    run.write_round(s, z)
    run.record_rewards(0, (-np.sum(z ** 2, axis=1)).tolist())
    # Fabricate round 2 without round 1.
    s2 = CEMState.from_dict({**s.to_dict(), "round_index": 2})
    run.write_round(s2, s2.sample())
    run.record_rewards(2, np.arange(8.0).tolist())
    with pytest.raises(ValueError, match="manifest gap"):
        CEMRun(root=tmp_path, name="both").resume(state(n_samples=8))


def test_state_survives_a_json_round_trip():
    s = state()
    s, _ = s.update(s.sample(), np.arange(32.0))
    t = CEMState.from_dict(s.to_dict())
    assert np.array_equal(t.sample(), s.sample())
    assert t.round_index == s.round_index and t.history == s.history
