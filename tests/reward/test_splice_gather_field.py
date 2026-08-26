"""The splice must survive being killed after it has written the field.

On 2026-08-25 a node-level kill took out three splice jobs in their EPILOGUE:
every 3.2 GiB field was already complete and valid on disk, but the sidecar was
not written and the chain's next stage never fired. Redoing the whole assemble
to recover a 5 KB JSON is the failure this guards.

The box here is (6, 8, 8, 8) with 4^3 tiles -- the same code path at a size a
test can hold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

splice_gather_field = pytest.importorskip("splice_gather_field")

NG, TILE = 8, 4
SHAPE = (6, NG, NG, NG)


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    """A run dir, a cached base box, and argv that splices tile 0."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=SHAPE).astype(np.float32)
    base_p = tmp_path / "base.npy"
    np.save(base_p, base)

    # The run's frozen tile must MATCH the cache or the consistency check
    # rejects the splice; the `out` tile is what differs.
    frozen_tile = base[:, :TILE, :TILE, :TILE].copy()
    out_tile = frozen_tile + 5.0
    run = tmp_path / "run"
    run.mkdir()
    np.savez(run / "tiles.npz", tiles=np.array([0]),
             out=out_tile[None], frozen=frozen_tile[None], hr=frozen_tile[None])

    monkeypatch.setattr(splice_gather_field, "find_base_field",
                        lambda box, seed=0: base_p)
    out = tmp_path / "field.npy"
    argv = ["--run-dir", str(run), "--box", "setT", "--tag", "t",
            "--out", str(out)]
    return base, out, argv


def _meta(out: Path) -> dict:
    return json.loads(out.with_suffix(".json").read_text())


def test_a_fresh_splice_writes_the_field_and_the_sidecar(rig):
    base, out, argv = rig
    assert splice_gather_field.main(argv) == 0
    field = np.load(out)
    assert field.shape == base.shape
    # Tile 0 moved by 5, everything else is untouched frozen base.
    assert np.allclose(field[:, :TILE, :TILE, :TILE],
                       base[:, :TILE, :TILE, :TILE] + 5.0)
    assert np.array_equal(field[:, TILE:, TILE:, TILE:],
                          base[:, TILE:, TILE:, TILE:])
    m = _meta(out)
    assert m["ok"] and m["field_reused"] is False
    assert m["max_abs_change"] == pytest.approx(5.0)


def test_the_sidecar_is_recoverable_after_a_kill_that_spared_the_field(rig):
    """The exact 2026-08-25 failure: field on disk, sidecar gone."""
    base, out, argv = rig
    assert splice_gather_field.main(argv) == 0
    first = np.load(out).copy()
    out.with_suffix(".json").unlink()

    assert splice_gather_field.main(argv) == 0
    m = _meta(out)
    assert m["field_reused"] is True
    # Reused, not rebuilt -- and the recovered numbers are the same numbers.
    assert np.array_equal(np.load(out), first)
    assert m["max_abs_change"] == pytest.approx(5.0)
    assert m["n_tiles"] == 1


def test_force_rebuilds_even_when_a_field_is_present(rig):
    base, out, argv = rig
    assert splice_gather_field.main(argv) == 0
    np.save(out, np.zeros(SHAPE, dtype=np.float32))  # a corrupt "resume"
    assert splice_gather_field.main(argv + ["--force"]) == 0
    assert _meta(out)["field_reused"] is False
    assert np.allclose(np.load(out)[:, :TILE, :TILE, :TILE],
                       base[:, :TILE, :TILE, :TILE] + 5.0)


def test_a_field_of_the_wrong_shape_is_rebuilt_not_reused(rig):
    base, out, argv = rig
    np.save(out, np.zeros((6, 4, 4, 4), dtype=np.float32))
    assert splice_gather_field.main(argv) == 0
    assert _meta(out)["field_reused"] is False
    assert np.load(out).shape == SHAPE


def test_the_chunked_max_change_equals_the_whole_box_one(rig):
    """The epilogue was rewritten to chunk over channels; same number."""
    base, out, argv = rig
    assert splice_gather_field.main(argv) == 0
    whole = float(np.abs(np.load(out) - base).max())
    assert _meta(out)["max_abs_change"] == pytest.approx(whole)
