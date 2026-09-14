"""The R_vir zoom: is it the same picture at higher resolution, or a new one?

The zoom is rebuilt from an export's ``tiles.npz`` rather than cropped, which
buys resolution but opens two ways to be silently wrong: the tiles might not
reconstruct to the particles the stage-1 fields gave, and ``out`` might not be
the arm stage 1 called ``self``. Both are the same check -- re-derive stage 1's
panels from the tiles and demand an exact match -- so what is pinned here is
that the check passes when it should, fails when it should, and that a failure
writes nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cosmo_sr.reward.tiles import TileGrid  # noqa: E402

add = pytest.importorskip("gather_holdout_add_density")
zoom = pytest.importorskip("gather_holdout_zoom_data")
plot = pytest.importorskip("gather_holdout_figures_plot")

NG, TILE, BOX = 8, 4, 100.0
GRID = TileGrid(ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX)
SMALL = ["--ng-hr", str(NG), "--tile-hr", str(TILE)]
TILES = [0, 1]
CENTRE = np.array([20.0, 20.0, 20.0])   # inside tile 0, which spans [0, 50)
RVIR = 6.0
SXR, SZR = np.array([0.0, 100.0]), np.array([0.0, 100.0])
SBINS = 8


def blocks(seed: int, amp: float = 0.02) -> np.ndarray:
    """(n_tiles, 6, TILE, TILE, TILE), the tiles.npz layout."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, amp, size=(len(TILES), 6, TILE, TILE, TILE)
                      ).astype(np.float32)


def panels(blk: np.ndarray) -> np.ndarray:
    pos = np.concatenate([add.tile_block_positions(blk[i], t, GRID)
                          for i, t in enumerate(TILES)], axis=0)
    return add.bin_window(pos, SXR, SZR, SBINS)[0]


def make_run(tmp_path: Path, out=None, frozen=None, hr=None):
    """An export dir + a stage-1 stem whose panels agree with those tiles."""
    out = blocks(1) if out is None else out
    frozen = blocks(2) if frozen is None else frozen
    hr = blocks(3) if hr is None else hr
    run = tmp_path / "holdout"
    run.mkdir(exist_ok=True)
    np.savez_compressed(run / "tiles.npz", tiles=np.array(TILES, dtype=np.int64),
                        out=out, frozen=frozen, hr=hr)
    (run / "export.json").write_text(
        json.dumps({"per_host": [{"halo_id": 7, "tiles": TILES}]}))
    stem = tmp_path / "data_h7"
    np.savez_compressed(stem.with_suffix(".npz"),
                        host_centre=CENTRE, host_rvir_mpc=np.array([RVIR]),
                        density_xr=SXR, density_zr=SZR,
                        frozen_density=panels(frozen), self_density=panels(out),
                        overlay_hr_pos=np.tile(CENTRE, (3, 1)),
                        overlay_hr_np=np.array([50, 60, 70]),
                        overlay_frozen_pos=np.tile(CENTRE, (2, 1)),
                        overlay_frozen_np=np.array([50, 60]),
                        overlay_self_pos=np.tile(CENTRE, (2, 1)),
                        overlay_self_np=np.array([50, 60]))
    stem.with_suffix(".json").write_text(
        json.dumps({"host_id": 7, "box": "setX", "host_rvir_mpc": RVIR}))
    return run, stem


def run_zoom(run, stem, out, *extra):
    return zoom.main(["--run-dir", str(run), "--source-npz", str(stem),
                      "--host-id", "7", "--out", str(out)] + SMALL + list(extra))


def test_verify_passes_and_the_window_is_the_circle(tmp_path):
    run, stem = make_run(tmp_path)
    out = tmp_path / "zoom"
    assert run_zoom(run, stem, out, "--zoom-rvir", "1.15", "--bins", "16") == 0

    with np.load(out.with_suffix(".npz")) as z:
        assert {"self_host_density", "frozen_host_density", "hr_host_density"} \
            <= set(z.files)
        assert z["self_host_density"].shape == (16, 16)     # rebinned, not cropped
        half = 1.15 * RVIR
        assert np.allclose(z["host_density_xr"], [CENTRE[0] - half,
                                                  CENTRE[0] + half])
        assert float(z["zoom_rvir"][0]) == 1.15
        assert float(z["zoom_depth_mpc"][0]) == pytest.approx(half)  # 0 -> window
        # the overlays and identity ride along, so the plot job needs nothing else
        assert "overlay_hr_pos" in z.files and "host_centre" in z.files
    assert json.loads(out.with_suffix(".json").read_text())["host_id"] == 7


def test_mismatched_tiles_write_nothing(tmp_path):
    """Stage-1 panels that these tiles do not reproduce must stop the job."""
    run, stem = make_run(tmp_path)
    # Repack the export with a different tuned arm than the stem was built
    # from. amp=2 is ~12 Mpc/h of displacement: a small one moves no particle
    # across a 12.5 Mpc/h bin, so the two arms would histogram identically and
    # the test would pass for the wrong reason.
    np.savez_compressed(run / "tiles.npz", tiles=np.array(TILES, dtype=np.int64),
                        out=blocks(99, amp=2.0), frozen=blocks(2), hr=blocks(3))
    out = tmp_path / "zoom"
    assert run_zoom(run, stem, out) == 0        # exit 0: dependents report
    assert not out.with_suffix(".npz").exists()


def test_depth_cut_removes_material_and_can_be_turned_off(tmp_path):
    run, stem = make_run(tmp_path)
    thin, thick = tmp_path / "thin", tmp_path / "thick"
    run_zoom(run, stem, thin, "--zoom-rvir", "1.15", "--depth-rvir", "0.2")
    run_zoom(run, stem, thick, "--zoom-rvir", "1.15", "--depth-rvir", "-1")
    with np.load(thin.with_suffix(".npz")) as a, \
            np.load(thick.with_suffix(".npz")) as b:
        assert a["hr_host_density"].sum() < b["hr_host_density"].sum()
        assert float(b["zoom_depth_mpc"][0]) < 0        # no cut, and recorded


def test_plot_takes_the_zoom_from_the_data_file(tmp_path):
    """fig1 must frame the circle without being told the zoom on the CLI."""
    run, stem = make_run(tmp_path)
    out = tmp_path / "zoom"
    run_zoom(run, stem, out, "--zoom-rvir", "1.15", "--bins", "16")
    figs = tmp_path / "figs"
    assert plot.main(["--data", str(out), "--out-dir", str(figs),
                      "--figures", "fig1"]) == 0
    assert (figs / "fig1_host_density.png").stat().st_size > 0
    # --figures fig1 means exactly that: no duplicates of the shared figures
    assert sorted(p.name for p in figs.iterdir()) == ["fig1_host_density.png"]


def test_dotted_stem_keeps_its_name(tmp_path):
    """A stem like zoom_rvir1.15_h7 must not be truncated to zoom_rvir1.

    Path.with_suffix reads ".15_h7" as the extension and replaces it, which
    writes the outputs under a name the plot job does not look for -- the data
    job then "succeeds" and the render gates out as MISSING INPUT.
    """
    run, stem = make_run(tmp_path)
    out = tmp_path / "zoom_rvir1.15_h7"
    assert run_zoom(run, stem, out, "--bins", "16") == 0
    assert (tmp_path / "zoom_rvir1.15_h7.npz").is_file()
    assert (tmp_path / "zoom_rvir1.15_h7.json").is_file()
    assert not (tmp_path / "zoom_rvir1.npz").exists()

    figs = tmp_path / "figs"
    assert plot.main(["--data", str(out), "--out-dir", str(figs),
                      "--figures", "fig1"]) == 0
    assert (figs / "fig1_host_density.png").is_file()
