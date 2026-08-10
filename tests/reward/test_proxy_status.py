"""The login-node status script: which jobs a candidate actually needs.

The submitter trusts ``candidate_action`` to spend GPU and Rockstar hours, so
the states are pinned one by one. The expensive mistakes it exists to prevent:
re-running Rockstar on a labelled candidate whose features are merely
schema-stale (that is a features-only backfill), and skipping the backfill
because the label looks done.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "reward"))

from cosmo_sr.reward.arms import FEATURE_SCHEMA_VERSION  # noqa: E402
from _proxy_status import candidate_action  # noqa: E402


def _candidate(tmp_path, *, manifest=None, features=False, label=None,
               field=False) -> Path:
    d = tmp_path / "setX__hr"
    d.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (d / "manifest.json").write_text(json.dumps(manifest))
    if features:
        np.savez(d / "features.npz", tile_id=np.arange(2))
    if label is not None:
        (d / "label_report.json").write_text(json.dumps(label))
    if field:
        np.save(d / "field.npy", np.zeros((6, 2, 2, 2), dtype=np.float32))
    return d


def test_nothing_on_disk_needs_the_full_pipeline(tmp_path):
    assert candidate_action(tmp_path / "missing") == "generate_label"


def test_current_schema_and_good_label_is_skipped(tmp_path):
    d = _candidate(tmp_path,
                   manifest={"feature_schema_version": FEATURE_SCHEMA_VERSION,
                             "field_sha": "f00d"},
                   features=True,
                   label={"label_ok": True, "field_sha": "f00d"})
    assert candidate_action(d) == "skip"


def test_stale_schema_with_a_good_label_backfills_features_only(tmp_path):
    """The whole point: a schema bump must never re-run Rockstar."""
    d = _candidate(tmp_path,
                   manifest={"feature_schema_version": 1, "field_sha": "f00d"},
                   features=True,
                   label={"label_ok": True, "field_sha": "f00d"})
    assert candidate_action(d) == "generate_only"


def test_unlabelled_at_current_schema_labels_only_if_the_field_is_kept(tmp_path):
    man = {"feature_schema_version": FEATURE_SCHEMA_VERSION, "field_sha": "f00d"}
    with_field = _candidate(tmp_path, manifest=man, features=True, field=True)
    assert candidate_action(with_field) == "label_only"
    dropped = _candidate(tmp_path / "sub", manifest=man, features=True)
    assert candidate_action(dropped) == "generate_label"


def test_a_label_against_a_different_field_is_not_a_label(tmp_path):
    """A field regenerated after labelling silently orphans the label."""
    d = _candidate(tmp_path,
                   manifest={"feature_schema_version": FEATURE_SCHEMA_VERSION,
                             "field_sha": "NEW"},
                   features=True, field=True,
                   label={"label_ok": True, "field_sha": "OLD"})
    assert candidate_action(d) == "label_only"


def test_a_failed_label_is_flagged_not_resubmitted(tmp_path):
    d = _candidate(tmp_path,
                   manifest={"feature_schema_version": FEATURE_SCHEMA_VERSION},
                   features=True,
                   label={"label_ok": False})
    assert candidate_action(d) == "invalid"
