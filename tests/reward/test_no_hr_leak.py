"""Explicit guard: the local-editor search and training path never reads HR.

The whole claim of this pipeline is that it is *deployment-legal* -- that the
same code could run on a box with no high-resolution counterpart. That claim is
worth exactly as much as the checking behind it, and it is the kind of property
that decays by accident: one debugging line that loads the paired HR box to
"just check", left in, and every subsequent number is an oracle result wearing a
deployment label.

Three independent checks, because each catches what the others miss:

1. **Source scan** -- no module on the search/training path names the HR data
   path, the HR oracle, or the paired-residual cache.
2. **Import graph** -- importing the local-editor modules does not pull in the
   modules that know how to read those things.
3. **Runtime poison** -- with ``np.load`` rigged to explode on any path under
   ``/hr/``, a full compose-and-score cycle completes.

Deliberately *not* covered: ``evaluate_local_editor.py`` and
``audit_local_editor_constraints.py`` read the Experiment-1 oracle's JSON row
summaries. That is legal and intended -- the oracle is reported as an
unattainable upper bound and as a calibration population, neither of which feeds
an action, a reward or a gradient. The distinction is enforced by listing those
two scripts separately below rather than by hoping nobody notices.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

# Everything that produces an action, a reward, or a model update.
TRAINING_PATH = [
    ROOT / "src/cosmo_sr/reward/local_editor.py",
    ROOT / "src/cosmo_sr/reward/local_reward.py",
    ROOT / "src/cosmo_sr/reward/cem.py",
    ROOT / "src/cosmo_sr/reward/action_flow.py",
    ROOT / "src/cosmo_sr/reward/token_bootstrap.py",
    ROOT / "scripts/reward/_local_common.py",
    ROOT / "scripts/reward/select_editor_hosts.py",
    ROOT / "scripts/reward/extract_editor_members.py",
    ROOT / "scripts/reward/run_editor_candidates.py",
    ROOT / "scripts/reward/aggregate_cem_round.py",
    ROOT / "scripts/reward/train_action_flow.py",
]

# Read the oracle's *reported numbers* only, and never to make an action.
REPORTING_ONLY = [
    ROOT / "scripts/reward/evaluate_local_editor.py",
    ROOT / "scripts/reward/audit_local_editor_constraints.py",
]

FORBIDDEN = {
    r"\bhr_path\b": "loads the paired HR field",
    r"\boracle_hr\b": "imports the HR-residual oracle",
    r"\bRESIDUAL_CACHE\b": "reads the paired-residual cache",
    r"\bcosmo_sr\.reward\.targets\b": "reads paired residual targets",
    r"""["']hr["']\s*,?\s*\)""": "passes 'hr' as a catalog/field source",
    r"\bsource\s*=\s*[\"']hr[\"']": "asks for the HR source",
}


@pytest.mark.parametrize("path", TRAINING_PATH, ids=lambda p: p.name)
def test_no_module_on_the_training_path_mentions_hr_data(path):
    text = path.read_text()
    # Strip docstrings and comments: this file's own prose, and the modules'
    # explanations of what they refuse to read, are not leaks.
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            d = ast.get_docstring(node, clean=False)
            if d:
                text = text.replace(d, "")
    text = re.sub(r"#.*", "", text)

    hits = [why for pat, why in FORBIDDEN.items() if re.search(pat, text)]
    assert not hits, f"{path.name} {hits}"


def test_the_reporting_only_scripts_are_listed_and_still_exist():
    """If one of these is deleted or renamed the exemption must be revisited,
    not silently inherited by whatever replaces it."""
    for p in REPORTING_ONLY:
        assert p.is_file(), p
    for p in REPORTING_ONLY:
        assert p not in TRAINING_PATH


def test_importing_the_editor_does_not_pull_in_the_hr_machinery():
    import subprocess
    import sys
    code = (
        "import sys;"
        "import cosmo_sr.reward.local_editor, cosmo_sr.reward.local_reward,"
        "cosmo_sr.reward.cem, cosmo_sr.reward.token_bootstrap;"
        "bad=[m for m in sys.modules if m.endswith('reward.oracle_hr')"
        " or m.endswith('reward.targets') or m.endswith('reward.diffusion')];"
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT),
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"pulled in {out.stdout.strip()}"


def test_a_full_compose_and_score_cycle_runs_with_hr_loading_poisoned(monkeypatch):
    """The strong version of the claim: rig ``np.load`` to raise on anything
    under an ``hr/`` directory, then do the whole thing."""
    from cosmo_sr.reward.local_editor import (
        EditorAction, HostPool, SubhaloToken, apply_edits, particle_positions_mpc,
        particle_velocities_kms, search_codec)
    from cosmo_sr.reward.local_reward import LocalRewardConfig, evaluate_candidate
    from cosmo_sr.reward.cem import CEMState

    real_load = np.load

    def poisoned(path, *a, **kw):
        s = str(path)
        if "/hr/" in s or s.endswith("_hr.npy") or "residual_targets" in s:
            raise AssertionError(f"the local editor tried to load HR data: {s}")
        return real_load(path, *a, **kw)

    monkeypatch.setattr(np, "load", poisoned)

    ng, box = 20, 10.0
    field = np.random.default_rng(0).normal(0, 0.05, (6, ng, ng, ng)).astype(np.float32)
    ids = np.arange(ng ** 3, dtype=np.int64)
    pool = HostPool(host_id=1, center_mpc=np.array([5.0, 5.0, 5.0]), rvir_mpc=4.0,
                    mvir=1e13, vmax=200.0, n_members=ids.size, ids=ids,
                    pos_mpc=particle_positions_mpc(field, ids, boxsize_mpc_h=box),
                    vel_kms=particle_velocities_kms(field, ids),
                    host_mean_vel_kms=np.zeros(3), boxsize_mpc_h=box)

    codec = search_codec("both")
    state = CEMState.initial(codec.dim, seed=0, n_samples=4)
    for z in state.sample():
        vals = codec.decode(z)
        t = SubhaloToken(1, vals["log_mass_ratio"], vals["radius_rvir"], (0, 0, 1.0))
        a = EditorAction((vals["center_offset_x"], vals["center_offset_y"],
                          vals["center_offset_z"]),
                         vals["source_radius_rvir"], vals["contraction"],
                         vals["velocity_cooling"], vals["bulk_velocity_mix"],
                         vals["edge_softness"])
        out, plans, _ = apply_edits(field, {1: pool}, [(t, a)], boxsize_mpc_h=box)
        assert out.shape == field.shape

    from conftest import synthetic_catalog
    base = synthetic_catalog([([5.0, 5.0, 5.0], 1e13)], [1], boxsize=box)
    cand = synthetic_catalog([([5.0, 5.0, 5.0], 1e13)], [2], boxsize=box)
    outcomes = evaluate_candidate(
        base, cand,
        [{"base_host_id": int(base.ids[0]), "center_mpc": [5.0, 5.0, 5.0],
          "host_rvir_mpc": 0.2, "requested_mvir": 5e10}],
        LocalRewardConfig(), boxsize_mpc_h=box)
    assert len(outcomes) == 1
    assert np.isfinite(outcomes[0].reward)
