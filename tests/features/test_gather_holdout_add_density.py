"""The blank-panel repair: is the added density built like the panels beside it?

The failure this guards is silent and only visible as a wrong picture. Stage 1
does not bin a spatial window -- it selects the host's LAGRANGIAN tiles and bins
where that material landed -- so a panel built by binning everything in a box,
or binned at a different resolution, or transposed, still renders a convincing
figure that no longer means the same thing as its neighbours. What is pinned
here is the tile selection, the window and bin count taken from the npz, the
array orientation, the legacy-name normalisation, and that a failed
re-derivation of an existing panel writes nothing.
"""
from __future__ import annotations

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
plot = pytest.importorskip("gather_holdout_figures_plot")

NG, TILE = 8, 4          # a 2x2x2 cover of 4^3-particle tiles
BOX = 100.0
GRID = TileGrid(ng_hr=NG, tile_hr=TILE, boxsize_mpc_h=BOX)
SMALL = ["--ng-hr", str(NG), "--tile-hr", str(TILE)]
TILES = "0,1"            # ids 0 and 1: same x/y block, adjacent in z
XR, ZR = [0.0, 100.0], [0.0, 100.0]


def write_field(path: Path, amp: float = 0.0, seed: int = 0) -> str:
    """A catnorm-shaped field; ``amp=0`` leaves particles on the lattice."""
    rng = np.random.default_rng(seed)
    f = (rng.normal(0.0, amp, size=(6, NG, NG, NG)) if amp else
         np.zeros((6, NG, NG, NG))).astype(np.float32)
    np.save(path, f)
    return str(path)


def npz_with(path: Path, **arrays) -> Path:
    base = dict(host_centre=np.array([50.0, 50.0, 50.0]),
                host_rvir_mpc=np.array([1.55]),
                density_xr=np.array(XR), density_zr=np.array(ZR),
                **{f"overlay_{k}_pos": np.full((3, 3), 50.0)
                   for k in ("hr", "frozen", "self")},
                **{f"overlay_{k}_np": np.array([50, 60, 70])
                   for k in ("hr", "frozen", "self")})
    base.update(arrays)
    np.savez_compressed(path, **base)
    return path


def panel_of(field: str, tiles=(0, 1), bins: int = 8) -> np.ndarray:
    pos = add.tile_positions(field, tiles, GRID)
    return add.bin_window(pos, np.array(XR), np.array(ZR), bins)[0]


def test_canonical_key_renames_only_the_legacy_host_window():
    assert add.canonical_key("frozen_density") == "frozen_host_density"
    assert add.canonical_key("density_xr") == "host_density_xr"
    assert add.canonical_key("hr_host_density") == "hr_host_density"
    assert add.canonical_key("self_box_density") == "self_box_density"
    assert add.canonical_key("host_centre") == "host_centre"


def test_tile_positions_are_the_tile_and_nothing_else():
    """Undisplaced, a tile's particles sit on its own block of the lattice."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        pos = add.tile_positions(write_field(Path(d) / "f.npy"), [0], GRID)
    assert pos.shape == (TILE ** 3, 3)          # one tile, not the whole box
    # float32, like field_to_particles: binning float64 positions puts a handful
    # of particles one bin over and the panel no longer matches its neighbours.
    assert pos.dtype == np.float32
    # tile 0 is the [0, 50) octant in every axis, particles at cell centres
    assert pos.min() > 0.0 and pos.max() < 50.0
    assert np.allclose(np.unique(np.round(pos[:, 0], 6)),
                       [6.25, 18.75, 31.25, 43.75])


def test_panel_orientation_is_row_z_column_x():
    """Transposing the panel would put the material in the wrong place."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        # tile 4 = (ix,iy,iz) = (1,0,0): high x, low z
        pos = add.tile_positions(write_field(Path(d) / "f.npy"), [4], GRID)
    h, lost = add.bin_window(pos, np.array(XR), np.array(ZR), 2)
    assert lost == 0
    assert h[0, 1] == TILE ** 3                  # row = z (low), col = x (high)
    assert h[1, 0] == 0


def test_added_panel_matches_the_stored_window_and_bins(tmp_path):
    field = write_field(tmp_path / "hr.npy", amp=0.02, seed=1)
    npz = npz_with(tmp_path / "data.npz",
                   frozen_density=np.ones((8, 8)), self_density=np.ones((8, 8)))
    assert add.main(["--npz", str(npz), "--field", field, "--name", "hr",
                     "--tiles", TILES] + SMALL) == 0

    with np.load(npz) as z:
        keys, hr, xr = set(z.files), z["hr_host_density"], z["host_density_xr"]
        frozen = z["frozen_host_density"]
        assert list(z["hr_density_tiles"]) == [0, 1]   # what was selected
    assert hr.shape == frozen.shape == (8, 8)          # bins from the neighbours
    assert np.allclose(xr, XR)                         # window from the npz
    assert hr.sum() == 2 * TILE ** 3                   # both tiles, nothing lost
    assert "frozen_density" not in keys and "density_xr" not in keys
    assert {"host_centre", "overlay_hr_pos", "host_rvir_mpc"} <= keys


def test_verify_passes_when_the_panel_is_rederived_the_same_way(tmp_path):
    frozen_field = write_field(tmp_path / "frozen.npy", amp=0.02, seed=2)
    npz = npz_with(tmp_path / "data.npz", frozen_density=panel_of(frozen_field))
    rc = add.main(["--npz", str(npz), "--field", write_field(tmp_path / "hr.npy",
                                                             amp=0.02, seed=3),
                   "--tiles", TILES, "--verify-name", "frozen",
                   "--verify-field", frozen_field] + SMALL)
    assert rc == 0
    with np.load(npz) as z:
        assert "hr_host_density" in z.files


def test_verify_mismatch_writes_nothing(tmp_path):
    """A stored panel built some other way must stop the job, not be joined."""
    npz = npz_with(tmp_path / "data.npz", frozen_density=np.ones((8, 8)))
    rc = add.main(["--npz", str(npz), "--field", write_field(tmp_path / "hr.npy"),
                   "--tiles", TILES, "--verify-name", "frozen",
                   "--verify-field", write_field(tmp_path / "frozen.npy",
                                                 amp=0.02, seed=4)] + SMALL)
    assert rc == 0                       # exit 0: dependents report, not strand
    with np.load(npz) as z:
        assert "hr_host_density" not in z.files
        assert "density_xr" in z.files   # untouched, not even renamed


def test_second_run_is_a_no_op_unless_overwrite(tmp_path):
    npz = npz_with(tmp_path / "data.npz", frozen_density=np.ones((8, 8)))
    a = ["--npz", str(npz), "--field", write_field(tmp_path / "hr.npy", amp=0.02,
                                                   seed=5), "--tiles", TILES]
    add.main(a + SMALL)
    with np.load(npz) as z:
        first = z["hr_host_density"].copy()
    add.main(a + SMALL)                                      # no --overwrite
    with np.load(npz) as z:
        assert np.array_equal(z["hr_host_density"], first)
    b = ["--npz", str(npz), "--field", write_field(tmp_path / "o.npy", amp=0.4,
                                                   seed=6), "--tiles", TILES]
    add.main(b + SMALL + ["--overwrite"])
    with np.load(npz) as z:
        assert not np.array_equal(z["hr_host_density"], first)


def test_tiles_come_from_export_json_when_not_given(tmp_path):
    run_dir = tmp_path / "holdout"
    run_dir.mkdir()
    (run_dir / "export.json").write_text(
        '{"per_host": [{"halo_id": 168880, "tiles": [0, 1]}]}')
    npz = npz_with(tmp_path / "data.npz", frozen_density=np.ones((8, 8)))
    add.main(["--npz", str(npz), "--field", write_field(tmp_path / "hr.npy"),
              "--run-dir", str(run_dir), "--host-id", "168880"] + SMALL)
    with np.load(npz) as z:
        assert list(z["hr_density_tiles"]) == [0, 1]


def test_spatial_mode_refuses_an_offcentre_window(tmp_path):
    """--spatial on a tile-union npz would silently bin the wrong material."""
    npz = npz_with(tmp_path / "data.npz", frozen_density=np.ones((8, 8)),
                   density_xr=np.array([33.5, 66.5]),
                   density_zr=np.array([46.0, 79.0]))
    with pytest.raises(SystemExit):
        add.main(["--npz", str(npz), "--field", write_field(tmp_path / "hr.npy"),
                  "--spatial"] + SMALL)


def test_fig1_renders_with_and_without_the_hr_panel(tmp_path):
    """The point of the exercise: fig1 draws HR's background once it is there."""
    npz = npz_with(tmp_path / "data.npz", frozen_density=np.ones((8, 8)),
                   self_density=np.ones((8, 8)))
    S = {"host_id": 168880, "box": "set9", "host_rvir_mpc": 1.55}
    with np.load(npz) as z:
        plot.fig_host_density(z, S, tmp_path / "before.png")
    assert (tmp_path / "before.png").stat().st_size > 0

    add.main(["--npz", str(npz), "--field", write_field(tmp_path / "hr.npy",
                                                        amp=0.02, seed=8),
              "--tiles", TILES] + SMALL)
    with np.load(npz) as z:
        plot.fig_host_density(z, S, tmp_path / "after.png")
    assert (tmp_path / "after.png").stat().st_size > 0
