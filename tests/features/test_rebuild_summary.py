import importlib.util
import json
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[2] / "scripts/features/rebuild_summary.py"
_spec = importlib.util.spec_from_file_location("rebuild_summary", _P)
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)


def _meta(tmp_path, argv):
    p = tmp_path / "wandb/latest-run/files/wandb-metadata.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"args": argv, "startedAt": "2026-09-14T09:57:45",
                             "host": "gpu24", "program": "finetune_member_gather.py"}))
    return p


def test_argv_round_trips_through_the_trainer_parser(tmp_path):
    _meta(tmp_path, ["--rung", "all_blocks", "--steps", "8000",
                     "--w-virial", "1.0", "--w-d6", "0.0", "--w-isolation", "1.0"])
    s = rs.rebuild(tmp_path)
    assert s["config"]["rung"] == "all_blocks"
    assert s["config"]["steps"] == 8000
    assert s["config"]["w_d6"] == 0.0 and s["config"]["w_isolation"] == 1.0
    # every other key is the parser's own default, not a guess
    assert s["config"]["w_centre"] == 1.0
    assert s["reconstructed"] is True and s["ok"] is False
    assert s["provenance"]["host"] == "gpu24"


def test_unknown_flag_fails_loudly(tmp_path):
    _meta(tmp_path, ["--rung", "all_blocks", "--w-renamed-since", "1.0"])
    with pytest.raises(SystemExit):
        rs.rebuild(tmp_path)


def test_missing_metadata_is_an_error_not_an_empty_config(tmp_path):
    with pytest.raises(SystemExit):
        rs.rebuild(tmp_path)
    (tmp_path / "wandb/latest-run/files").mkdir(parents=True)
    (tmp_path / "wandb/latest-run/files/wandb-metadata.json").write_text("{}")
    with pytest.raises(SystemExit):
        rs.rebuild(tmp_path)


def test_verdict_names_the_partial_step(tmp_path, monkeypatch):
    _meta(tmp_path, ["--rung", "all_blocks", "--steps", "8000"])
    monkeypatch.setattr(rs, "checkpoint_state",
                        lambda d, n="tuned.pt": {"checkpoint": "x", "step": 3750,
                                                 "partial": True})
    s = rs.rebuild(tmp_path)
    assert s["step"] == 3750 and s["partial"] is True
    assert "3750" in s["verdict"] and "NOT comparable" in s["verdict"]
