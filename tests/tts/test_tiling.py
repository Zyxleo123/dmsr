"""Stage 5: globally coherent tiled inference."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmo_sr.tts.sampling import GlobalNoiseField
from cosmo_sr.tts.srs_noise import ControlledG
from cosmo_sr.tts.tiling import (
    TileGrid,
    blend_window,
    generate_tiles,
    joint_score,
    overlap_pairs,
    select_tiles_coordinate_descent,
    stitch_overlapping,
    tiled_inference,
)

SMALL = dict(chan_base=16, chan_min=8, chan_max=16)
# ng must leave room for a padded tile: chunk + 2 * pad <= ng.
NG, SCALE, PAD = 16, 8, 3


def _generator(seed: int = 0) -> ControlledG:
    torch.manual_seed(seed)
    g = ControlledG(6, 6, SCALE, **SMALL).eval()
    with torch.no_grad():
        for name, p in g.named_parameters():
            if name.endswith(".std"):
                p.copy_(torch.randn_like(p) * 0.3)
    return g


def _lr(seed: int = 2) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(6, NG, NG, NG)).astype(np.float32)


def _grid(chunk=8, stride=4) -> TileGrid:
    return TileGrid(ng=NG, chunk=chunk, stride=stride, pad=PAD, scale=SCALE)


# --------------------------------------------------------------------------- #
def test_grid_rejects_geometries_that_break_the_two_tile_blend():
    with pytest.raises(ValueError):
        TileGrid(ng=NG, chunk=8, stride=2, pad=PAD, scale=SCALE)     # 4 tiles overlap
    with pytest.raises(ValueError):
        TileGrid(ng=NG, chunk=8, stride=3, pad=PAD, scale=SCALE)     # stride does not divide Ng
    with pytest.raises(ValueError):
        TileGrid(ng=NG, chunk=16, stride=16, pad=PAD, scale=SCALE)   # padded tile > box


def test_blend_window_is_complementary():
    """Two neighbours whose ramps coincide must sum to exactly one there."""
    size, ramp = 16, 4
    w = blend_window(size, ramp)
    assert torch.allclose(w[:ramp] + w[size - ramp:], torch.ones(ramp), atol=1e-6)
    assert torch.allclose(w[ramp:size - ramp], torch.ones(size - 2 * ramp))


def test_adjacent_tiles_with_global_noise_agree_in_their_overlap():
    """The core Stage-5 claim: shared noise + valid convs => identical overlap."""
    g, lr, grid = _generator(), _lr(), _grid()
    tiles = generate_tiles(g, lr, grid, seed=5, noise_mode="global")
    hr_overlap = grid.overlap_lr * grid.scale
    assert hr_overlap > 0
    a = tiles[(0, 0, 0)]
    b = tiles[(grid.stride, 0, 0)]
    d = grid.stride * grid.scale
    lhs = a[:, d:, :, :]
    rhs = b[:, : grid.tile_hr - d, :, :]
    assert torch.allclose(lhs, rhs, atol=1e-5), float((lhs - rhs).abs().max())


def test_per_tile_noise_does_not_agree_in_the_overlap():
    """Control: without coordinate-indexed noise the overlap genuinely differs."""
    g, lr, grid = _generator(), _lr(), _grid()
    tiles = generate_tiles(g, lr, grid, seed=5, noise_mode="per_tile")
    d = grid.stride * grid.scale
    a, b = tiles[(0, 0, 0)], tiles[(grid.stride, 0, 0)]
    diff = float((a[:, d:] - b[:, : grid.tile_hr - d]).abs().max())
    assert diff > 1e-4, diff


def test_opposite_faces_obey_periodicity():
    """The tile wrapping past Ng must agree with the tile at the origin."""
    g, lr, grid = _generator(), _lr(), _grid()
    last = grid.ng - grid.stride
    tiles = generate_tiles(g, lr, grid, seed=1, noise_mode="global",
                           starts=[(0, 0, 0), (last, 0, 0)])
    d = grid.stride * grid.scale
    wrap = grid.tile_hr - d                 # part of the last tile that wraps to 0
    if wrap > 0:
        lhs = tiles[(last, 0, 0)][:, grid.tile_hr - wrap:, :, :]
        rhs = tiles[(0, 0, 0)][:, :wrap, :, :]
        assert torch.allclose(lhs, rhs, atol=1e-5), float((lhs - rhs).abs().max())


def test_full_box_is_invariant_to_tile_processing_order():
    g, lr, grid = _generator(), _lr(), _grid()
    starts = grid.starts()
    shuffled = list(starts)
    np.random.default_rng(0).shuffle(shuffled)
    a = tiled_inference(g, lr, grid, seed=9, noise_mode="global", starts=starts)
    b = tiled_inference(g, lr, grid, seed=9, noise_mode="global", starts=shuffled)
    assert torch.allclose(a, b, atol=1e-6)


def test_stitching_produces_a_full_box_in_both_modes():
    g, lr, grid = _generator(), _lr(), _grid()
    tiles = generate_tiles(g, lr, grid, seed=0, noise_mode="global")
    for mode in ("crop", "blend"):
        box = stitch_overlapping(tiles, grid, mode=mode)
        assert box.shape == (6, NG * SCALE, NG * SCALE, NG * SCALE)
        assert torch.isfinite(box).all()


def test_blending_globally_coherent_tiles_is_a_no_op_in_the_overlap():
    """Coherent tiles are already equal there, so any weights reconstruct them."""
    g, lr, grid = _generator(), _lr(), _grid()
    tiles = generate_tiles(g, lr, grid, seed=0, noise_mode="global")
    blended = stitch_overlapping(tiles, grid, mode="blend")
    cropped = stitch_overlapping(tiles, grid, mode="crop")
    assert torch.allclose(blended, cropped, atol=1e-4), float((blended - cropped).abs().max())


# --------------------------------------------------------------------------- #
# Joint selection
# --------------------------------------------------------------------------- #
def _toy_problem(k=3, n=2):
    grid = TileGrid(ng=4, chunk=2, stride=2, pad=1, scale=2)
    starts = grid.starts()
    rng = np.random.default_rng(0)
    verifier = {s: rng.normal(size=k) for s in starts}
    pairs = [(s, t) for s in starts for t in starts if s != t][: n * len(starts)]
    disagreement = {p: rng.random((k, k)) for p in pairs}
    return verifier, disagreement


def test_coordinate_descent_monotonically_reduces_the_joint_score():
    verifier, dis = _toy_problem()
    choice, traj = select_tiles_coordinate_descent(verifier, dis, lam_overlap=2.0)
    assert all(b <= a + 1e-9 for a, b in zip(traj[:-1], traj[1:])), traj
    assert joint_score(choice, verifier, dis, 2.0) == pytest.approx(traj[-1])


def test_joint_selection_beats_independent_selection_on_the_joint_objective():
    verifier, dis = _toy_problem()
    lam = 5.0
    independent = {s: int(np.argmin(v)) for s, v in verifier.items()}
    joint, _ = select_tiles_coordinate_descent(verifier, dis, lam_overlap=lam)
    assert joint_score(joint, verifier, dis, lam) <= joint_score(independent, verifier, dis, lam)


def test_joint_selection_reduces_seam_energy_without_flattening_the_field():
    """Selection must cut the boundary discontinuity, not the small-scale variance."""
    from cosmo_sr.tts.metrics import boundary_discontinuity

    g, lr = _generator(), _lr()
    grid = _grid(chunk=8, stride=8)          # non-overlapping: seams are real
    k_seeds = [0, 1, 2, 3]
    per_seed = [generate_tiles(g, lr, grid, seed=s, noise_mode="per_tile") for s in k_seeds]

    starts = grid.starts()
    verifier = {s: np.zeros(len(k_seeds)) for s in starts}
    dis = {}
    for (s, t) in overlap_pairs(grid) or [(starts[0], starts[1])]:
        dis[(s, t)] = np.zeros((len(k_seeds), len(k_seeds)))
    # butt-joined tiles: score neighbours by the jump across the shared face
    axis_of = {}
    for (s, t) in dis:
        axis_of[(s, t)] = next(a for a in range(3) if s[a] != t[a])
    for (s, t), mat in dis.items():
        a = axis_of[(s, t)]
        for i in range(len(k_seeds)):
            for j in range(len(k_seeds)):
                face_i = per_seed[i][s].index_select(a + 1, torch.tensor([grid.tile_hr - 1]))
                face_j = per_seed[j][t].index_select(a + 1, torch.tensor([0]))
                mat[i, j] = float((face_i - face_j).pow(2).mean())

    choice, _ = select_tiles_coordinate_descent(verifier, dis, lam_overlap=1.0)
    chosen = {s: per_seed[choice[s]][s] for s in starts}
    baseline = {s: per_seed[0][s] for s in starts}
    box_sel = stitch_overlapping(chosen, grid, mode="crop").unsqueeze(0)
    box_base = stitch_overlapping(baseline, grid, mode="crop").unsqueeze(0)

    r_sel = boundary_discontinuity(box_sel, grid.tile_hr)
    r_base = boundary_discontinuity(box_base, grid.tile_hr)
    assert r_sel <= r_base + 1e-9, (r_sel, r_base)
    # and the field must not have been smoothed to achieve it
    assert float(box_sel.std()) > 0.9 * float(box_base.std())
