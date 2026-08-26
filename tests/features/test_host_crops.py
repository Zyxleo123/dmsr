"""The crop's index arithmetic, pinned numerically.

Everything here is periodic-lattice bookkeeping, which is exactly the kind of
code that runs, produces a plausible picture, and is wrong. The properties that
matter: a crop of a face-straddling host is the same crop as one of the same
host translated to the middle of the box, and the id a crop site carries is the
id the field is indexed by.
"""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.features import (
    auc, block_reduce, crop_frame, flat_to_sites, lagrangian_radius_sites,
    periodic_site_centre, resample_report, roc_curve, sites_to_flat,
    to_crop_coords,
)

NG = 32


def test_flat_and_sites_round_trip():
    pid = np.arange(NG ** 3)
    assert np.array_equal(sites_to_flat(flat_to_sites(pid, NG), NG), pid)


def test_sites_match_field_to_particles_convention():
    """Site ``(a,b,c)`` must be the C-order index the GADGET2 ids use."""
    vol = np.arange(NG ** 3).reshape(NG, NG, NG)
    for a, b, c in [(0, 0, 0), (1, 2, 3), (NG - 1, 0, NG - 1)]:
        pid = sites_to_flat(np.array([[a, b, c]]), NG)[0]
        assert vol[a, b, c] == pid


def test_periodic_centre_has_no_seam():
    """Sites at both faces average to the face, not to the middle of the box."""
    s = np.array([[1, 5, 5], [NG - 1, 5, 5], [0, 5, 5]])
    c = periodic_site_centre(s, NG)
    assert c[0] < 1.0 or c[0] > NG - 1.0
    assert c[1] == pytest.approx(5.0, abs=1e-6)


def test_lagrangian_radius_is_the_volume_equivalent_sphere():
    n = 4.0 / 3.0 * np.pi * 10.0 ** 3
    assert lagrangian_radius_sites(n) == pytest.approx(10.0, rel=1e-9)
    assert lagrangian_radius_sites(0) == 0.0


def _blob(centre, radius, ng=NG):
    g = np.arange(ng)
    d = ((g[:, None, None] - centre[0] + ng / 2) % ng - ng / 2) ** 2 \
        + ((g[None, :, None] - centre[1] + ng / 2) % ng - ng / 2) ** 2 \
        + ((g[None, None, :] - centre[2] + ng / 2) % ng - ng / 2) ** 2
    return np.argwhere(d <= radius ** 2)


def test_crop_of_a_wrapped_host_equals_the_translated_one():
    """A host on the box face and the same host in the middle crop identically."""
    mid, edge = np.array([16, 16, 16]), np.array([0, 16, 16])
    fa = crop_frame(_blob(mid, 5), NG, scale=1.5)
    fb = crop_frame(_blob(edge, 5), NG, scale=1.5)
    assert fa.side == fb.side

    # The same field, shifted by the same offset the hosts differ by, must be
    # gathered identically by the two frames.
    rng = np.random.default_rng(0)
    field = rng.normal(size=(NG, NG, NG))
    shifted = np.roll(field, shift=int(edge[0] - mid[0]), axis=0)
    ga = field[np.ix_(*fa.axes())]
    gb = shifted[np.ix_(*fb.axes())]
    assert np.allclose(ga, gb)


def test_flat_ids_index_the_same_cube_as_axes():
    f = crop_frame(_blob(np.array([2, 30, 15]), 4), NG, scale=1.5)
    field = np.arange(NG ** 3).reshape(NG, NG, NG)
    by_axes = field[np.ix_(*f.axes())].reshape(-1)
    by_ids = field.reshape(-1)[f.flat_ids()]
    assert np.array_equal(by_axes, by_ids)


def test_crop_coords_place_the_centre_at_the_middle():
    sites = _blob(np.array([4, 28, 16]), 3)
    f = crop_frame(sites, NG, scale=2.0)
    u = to_crop_coords(sites, f)
    assert np.all(u >= 0) and np.all(u < f.side)
    assert np.allclose(u.mean(axis=0), f.side / 2.0, atol=1.5)


def test_crop_coords_flag_the_outside_without_a_mask():
    f = crop_frame(_blob(np.array([16, 16, 16]), 3), NG, scale=1.0)
    far = np.array([[(16 + f.side) % NG, 16, 16]])
    u = to_crop_coords(far, f)
    assert not np.all((u >= 0) & (u < f.side))


def test_crop_side_is_even_and_clipped():
    f = crop_frame(_blob(np.array([16, 16, 16]), 2), NG, scale=1.0, min_side=10)
    assert f.side % 2 == 0 and f.side >= 10
    big = crop_frame(_blob(np.array([16, 16, 16]), 15), NG, scale=8.0)
    assert big.side <= NG


def test_resample_report_is_the_ratio_not_a_resample():
    f = crop_frame(_blob(np.array([16, 16, 16]), 6), NG, scale=1.0)
    r = resample_report(f, 96)
    assert r["ratio"] == pytest.approx(96.0 / f.side)
    assert r["native_sites"] == f.side ** 3


def test_block_reduce_max_keeps_an_isolated_peak():
    """A two-site clump must survive the downsample the page draws from.

    Deliberately smaller than the 4x4x4 block, which is the case that motivates
    ``how='max'``: the mean dilutes it by a factor of 8 and the clump the page
    exists to show stops being visible.
    """
    v = np.zeros((48, 48, 48), dtype=np.float32)
    v[21:23, 21:23, 21:23] = 5.0
    out, extent = block_reduce(v, 12, "max")
    assert out.shape == (12, 12, 12) and extent == 48
    assert out.max() == pytest.approx(5.0)
    assert block_reduce(v, 12, "mean")[0].max() == pytest.approx(5.0 * 8 / 64)


def test_block_reduce_passes_through_when_not_shrinking():
    v = np.arange(8 ** 3, dtype=np.float32).reshape(8, 8, 8)
    out, extent = block_reduce(v, 16, "max")
    assert np.array_equal(out, v) and extent == 8


def test_block_reduce_picks_the_side_that_minimises_padding():
    """The output side follows the factor; padding stays under one block.

    Fixing the output side instead and padding up to it is what fabricated a
    quarter of the drawn cube: n=148 into 48 pads 44 sites.
    """
    for n, max_side in [(148, 48), (114, 48), (106, 48), (100, 48), (50, 16)]:
        out, extent = block_reduce(np.ones((n, n, n), dtype=np.float32), max_side)
        f = extent // out.shape[0]
        assert out.shape[0] <= max_side
        assert extent >= n
        assert extent - n < f, f"padding {extent - n} is a whole block of {f}"


def test_block_reduce_does_not_replicate_the_boundary_row():
    """A bright last row must not become a band in the reduced cube.

    ``mode='edge'`` duplicates it across the padding, which is what produced
    the streaks running to the edges of the crop panel.
    """
    n = 50
    v = np.zeros((n, n, n), dtype=np.float32)
    v[n - 1, :, :] = 9.0                      # the last real slice, along axis 0
    out, extent = block_reduce(v, 16, "max")
    hot = np.flatnonzero(out.max(axis=(1, 2)) > 0)
    assert hot.size == 1, "the boundary row bled into more than one output block"
    assert hot[0] == out.shape[0] - 1


def test_block_reduce_padding_never_makes_a_whole_block_empty():
    """Padding is under one block, so no output cell is pure fill."""
    out, extent = block_reduce(np.ones((106, 106, 106), dtype=np.float32), 48)
    assert np.isfinite(out).all()


def test_auc_matches_the_definition_on_a_small_case():
    s = np.array([0.1, 0.4, 0.35, 0.8])
    y = np.array([0, 0, 1, 1])
    pairs = [(a, b) for a in s[y == 1] for b in s[y == 0]]
    want = np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs])
    assert auc(s, y) == pytest.approx(want)


def test_auc_of_a_constant_score_is_one_half():
    assert auc(np.ones(50), np.r_[np.ones(10), np.zeros(40)]) == pytest.approx(0.5)


def test_auc_is_invariant_to_monotone_rescaling():
    rng = np.random.default_rng(1)
    s = rng.normal(size=500)
    y = (rng.random(500) < 0.2 + 0.3 * (s > 0)).astype(int)
    assert auc(s, y) == pytest.approx(auc(np.exp(3 * s) - 7, y))


def test_auc_is_nan_when_a_class_is_empty():
    assert np.isnan(auc(np.arange(5.0), np.zeros(5)))


def test_roc_is_monotone_and_spans_the_unit_square():
    rng = np.random.default_rng(2)
    s = rng.normal(size=2000)
    y = (rng.random(2000) < 1 / (1 + np.exp(-2 * s))).astype(int)
    r = roc_curve(s, y, 32)
    assert np.all(np.diff(r["fpr"]) >= -1e-9)
    assert np.all(np.diff(r["tpr"]) >= -1e-9)
    assert r["fpr"][-1] == pytest.approx(1.0) and r["tpr"][-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The collector's per-crop physics
# ---------------------------------------------------------------------------

import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    for p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    path = PROJECT_ROOT / "scripts" / "features" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod(monkeypatch_module=None):
    m = _load("collect_host_crops")
    m.NG_HR, m.BOXSIZE = NG, 100.0
    return m


def _frame_at(centre, radius=4, scale=2.0):
    return crop_frame(_blob(np.array(centre), radius), NG, scale=scale,
                      min_side=8, max_side=NG)


def test_crop_positions_of_a_zero_field_are_the_lattice(mod):
    """Zero displacement must give the unperturbed lattice, spacing and all."""
    f = np.zeros((6, NG, NG, NG), dtype=np.float32)
    frame = _frame_at([16, 16, 16])
    pos, inner = mod.crop_positions(f, frame, pad=2)
    assert pos.shape == ((frame.side + 4) ** 3, 3)
    assert int(inner.sum()) == frame.side ** 3
    step = np.diff(np.unique(pos[:, 0].round(6)))
    assert np.allclose(step, 100.0 / NG, atol=1e-4)


def test_crop_positions_do_not_wrap_at_the_box_face(mod):
    """A crop straddling the face must be one contiguous cloud, not two."""
    f = np.zeros((6, NG, NG, NG), dtype=np.float32)
    frame = _frame_at([0, 16, 16])
    pad = 2
    pos, _ = mod.crop_positions(f, frame, pad=pad)
    extent = pos.max(axis=0) - pos.min(axis=0)
    # A wrapped crop would span nearly the whole box on the straddled axis;
    # an unwrapped one spans exactly its own padded width.
    want = (frame.side + 2 * pad - 1) * (100.0 / NG)
    assert np.allclose(extent, want, atol=1e-3)


def test_a_wrapped_and_an_interior_crop_give_the_same_cloud(mod):
    """The face is not a special case: same field, same crop, up to a shift."""
    rng = np.random.default_rng(3)
    base = rng.normal(scale=1e-3, size=(6, NG, NG, NG)).astype(np.float32)
    shifted = np.roll(base, shift=-16, axis=1)     # axis 1 of the (6,N,N,N) array
    pa, _ = mod.crop_positions(base, _frame_at([16, 16, 16]), pad=2)
    pb, _ = mod.crop_positions(shifted, _frame_at([0, 16, 16]), pad=2)
    assert np.allclose(np.sort(pa, axis=0), np.sort(pb, axis=0), atol=1e-4)


def test_local_density_is_higher_inside_a_collapsed_clump(mod):
    """The estimator must rank a collapsed region above an unperturbed one.

    Built by pulling one octant of the crop toward its centre, which is what a
    forming halo does to the Lagrangian sites it owns.
    """
    frame = _frame_at([16, 16, 16], radius=6, scale=2.0)
    side = frame.side
    f = np.zeros((6, NG, NG, NG), dtype=np.float32)
    pos, inner = mod.crop_positions(f, frame, pad=3)
    n_in = int(inner.sum())

    # Collapse: move the inner eighth of the crop 70% of the way to its centre.
    g = np.arange(side) - (side - 1) / 2.0
    r = np.sqrt(g[:, None, None] ** 2 + g[None, :, None] ** 2
                + g[None, None, :] ** 2)
    core = (r < side / 6.0)
    pos_in = pos[inner].reshape(side, side, side, 3).copy()
    c = pos_in.reshape(-1, 3).mean(axis=0)
    pos_in[core] = c + 0.3 * (pos_in[core] - c)
    moved = pos.copy()
    moved[inner] = pos_in.reshape(-1, 3)

    rho = mod.local_log_density(moved, inner, side)
    assert rho.shape == (side, side, side)
    assert rho[core].mean() > rho[~core].mean() + 0.5
    assert auc(rho.reshape(-1), core.reshape(-1)) > 0.95
    assert n_in == side ** 3


def test_crop_gather_keeps_the_channel_axis_first(mod):
    """``field[(slice(0,3),) + np.ix_(*axes)]`` must stay ``(3, w, w, w)``.

    NumPy moves the broadcast result of advanced indices to the *front* when
    they are separated by a slice. Here they are contiguous, so the channel axis
    stays where it is -- but that is a property of this expression, not a
    guarantee, and getting it wrong silently transposes displacement components
    into spatial axes.
    """
    f = np.arange(6 * 8 * 8 * 8).reshape(6, 8, 8, 8)
    ax = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    out = f[(slice(0, 3),) + np.ix_(*ax)]
    assert out.shape == (3, 2, 2, 2)
    assert out[1, 0, 1, 0] == f[1, 1, 4, 5]


def test_disnorm_undo_is_the_srs_displacement_scale(mod):
    """A catnorm value of 1 is 6000 kpc/h at z=0; the crop math assumes it."""
    from cosmo_sr.data.preprocess_srs import disnorm
    got = disnorm(np.ones((3, 2, 2, 2)), z=0.0, undo=True).ravel()[0]
    assert got == pytest.approx(6000.0, rel=1e-6)


def test_overlay_and_image_agree_after_the_block_reduction():
    """A voxel and a circle at the same native coordinate land on the same pixel.

    This is the registration the page's whole claim depends on, and it is the
    one that broke: the reduced cube spans ``extent`` native sites, so both the
    image and the overlay must divide by ``extent``. Dividing the overlay by the
    crop side instead is what put circles outside the data.
    """
    n, max_side, W = 106, 48, 560
    v = np.zeros((n, n, n), dtype=np.float32)
    site = (70, 30, 12)
    v[site] = 1.0
    out, extent = block_reduce(v, max_side, "max")
    g = out.shape[0]

    # where the image puts it
    hot = np.array(np.unravel_index(int(np.argmax(out)), out.shape))
    img_px = (hot + 0.5) * W / g

    good = (np.array(site) + 0.5) * W / extent
    assert np.all(np.abs(good - img_px) < W / g), "correct scaling misses its voxel"
    assert extent - n <= 2


def test_the_old_fixed_output_side_would_have_misregistered_by_cells():
    """Why the reduction had to change, stated as a number.

    Under the previous contract the output side was pinned to ``max_side`` and
    the cube padded up to ``f * max_side``. For the largest cluster crop that is
    148 -> 192: a quarter of the drawn cube was edge-replicated, and an overlay
    scaled by 148 drifted outward by 30% of the canvas at the far edge.
    """
    n, max_side, W = 148, 48, 560
    f = -(-n // max_side)
    old_extent = f * max_side                      # 192
    assert old_extent - n == 44

    far = n - 1
    drift = abs(far * W / n - far * W / old_extent)
    assert drift > 4 * (W / max_side), f"drift {drift:.0f}px is under four cells"

    # the new contract keeps the same factor but sizes the output to it
    out, extent = block_reduce(np.ones((n, n, n), dtype=np.float32), max_side)
    assert (out.shape[0], extent) == (37, 148)
