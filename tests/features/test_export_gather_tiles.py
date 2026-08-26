"""``export_gather_tiles.replay_args`` must reproduce the run it is gating.

The export rebuilds the member sets from scratch, so every setting that changes
what a set IS -- ``min_num_p``, ``min_purity``, ``min_live_frac``, ``max_sets``,
``n_tiles``, the softening, the background -- has to come back exactly as the run
had it. If one silently reverts to a parser default, the gate scores a different
pool than the trainer reported, and nothing in the output says so: every
statistic is still well formed and means something else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

export_gather_tiles = pytest.importorskip("export_gather_tiles")
finetune_member_gather = pytest.importorskip("finetune_member_gather")


def _write_run(tmp_path: Path, **overrides) -> Path:
    """A run directory whose summary.json holds a full parsed namespace."""
    args = finetune_member_gather.build_parser().parse_args([])
    cfg = vars(args).copy()
    cfg.update(overrides)
    run = tmp_path / "arm"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"config": cfg, "ok": True}))
    return run


# The settings that change what a member set is. A default-valued round trip
# would pass by accident, so every value here differs from the parser default.
POOL_KEYS = {
    "min_num_p": 50,
    "min_purity": 0.75,
    "min_live_frac": 0.25,
    "max_sets": 1024,
    "n_tiles": 16,
    "softening": 0.02,
    "bound_tau": 0.25,
    "bg_k": 2048,
    "bg_radius": 6.0,
    "centre_mode": "self",
    "w_centre": 0.0,
    "rung": "middle_fine",
    "max_hosts_per_box": 3,
}


def test_every_pool_setting_round_trips(tmp_path):
    run = _write_run(tmp_path, **POOL_KEYS)
    got = export_gather_tiles.replay_args(run, "set9", [])
    for k, want in POOL_KEYS.items():
        assert getattr(got, k) == want, f"{k} did not survive the replay"


def test_the_replay_pins_the_box_to_one_holdout(tmp_path):
    """build_pool loads a 537 MB owner array per box; exactly one is wanted."""
    run = _write_run(tmp_path, train_boxes="set3 set4", holdout_boxes="set9 set10",
                     holdout_hosts="set3:h1")
    got = export_gather_tiles.replay_args(run, "set10", [])
    assert got.holdout_boxes == "set10"
    assert got.train_boxes == ""
    assert got.holdout_hosts == ""


def test_a_store_true_flag_survives_both_ways(tmp_path):
    """``--term-norm`` and ``--bound-penalty log`` change the loss, not the sets,
    but a run that used them is a different run and the export must say so."""
    on = export_gather_tiles.replay_args(
        _write_run(tmp_path / "a", term_norm=True), "set9", [])
    assert on.term_norm is True
    off = export_gather_tiles.replay_args(
        _write_run(tmp_path / "b", term_norm=False), "set9", [])
    assert off.term_norm is False


def test_a_key_the_parser_no_longer_knows_is_reported_not_silent(tmp_path, capsys):
    """A renamed option must not fall back to a default without saying so."""
    run = _write_run(tmp_path, **{"a_key_that_was_removed": 7})
    got = export_gather_tiles.replay_args(run, "set9", [])
    assert not hasattr(got, "a_key_that_was_removed")
    assert "a_key_that_was_removed" in capsys.readouterr().out


def test_extra_argv_overrides_the_stored_config(tmp_path):
    """The escape hatch: a deliberate override on the command line wins."""
    run = _write_run(tmp_path, min_num_p=200)
    got = export_gather_tiles.replay_args(run, "set9", ["--min-num-p", "50"])
    assert got.min_num_p == 50


def test_a_repeatable_option_round_trips_element_by_element(tmp_path):
    """``--set key=value`` is an append action, so its value is a LIST.

    ``str([])`` is the literal ``"[]"``, which ``apply_overrides`` rejects as not
    key=value -- the first launch of this exporter died on exactly that, three
    seconds in, on all five jobs (36243-36256).
    """
    run = _write_run(tmp_path, overrides=["train.lr=1e-5", "model.embed_dim=64"])
    got = export_gather_tiles.replay_args(run, "set9", [])
    assert got.overrides == ["train.lr=1e-5", "model.embed_dim=64"]


def test_an_empty_repeatable_option_emits_nothing(tmp_path):
    run = _write_run(tmp_path, overrides=[])
    got = export_gather_tiles.replay_args(run, "set9", [])
    assert got.overrides == []


# --- the exporter's call into the trainer -----------------------------------
#
# `host_forward` lives in finetune_member_gather and the exporter unpacks its
# return. Nothing binds the two: when `--w-host-sets` added a sixth return
# (the host-preservation term) the exporter still unpacked five, and the only
# thing that noticed was a GPU job that had already spent 255 s rebuilding the
# pool. This is a static check precisely because reaching that line for real
# costs an a6000 and four minutes.

def _return_arity(fn) -> int:
    """How many values the LAST `return` of ``fn`` yields."""
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    returns = [n for n in ast.walk(tree)
               if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)]
    assert returns, "host_forward no longer returns a tuple"
    return len(returns[-1].value.elts)


def _unpack_arity(path: Path, callee: str) -> int:
    """How many names the assignment of a ``callee(...)`` call binds."""
    import ast
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt, val = node.targets[0], node.value
        if not isinstance(tgt, ast.Tuple) or not isinstance(val, ast.Call):
            continue
        fn = val.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
        if name == callee:
            return len(tgt.elts)
    raise AssertionError(f"no tuple-unpacking call to {callee} in {path}")


def test_the_exporter_unpacks_every_value_host_forward_returns():
    src = Path(export_gather_tiles.__file__)
    assert (_unpack_arity(src, "host_forward")
            == _return_arity(finetune_member_gather.host_forward))
