"""The differentiable kNN log-distance loss.

One thing needs proving and three need pinning.

Proved: the gradient. The search is a detached ``topk`` and the distances are
recomputed on the live tensor, so the claim "this is exactly the gradient of an
all-pairs differentiable ``topk``" is a claim that has to be checked against that
reference and against central finite differences. If it is wrong, a training run
descends a direction that is not the loss's, which is invisible in every metric.

Pinned: the geometry must agree with
:func:`cosmo_sr.features.member_gather.tile_particles` digit for digit, or the
buffer cuts a cube that is not the tile's; minimum-image periodicity must be real
(a neighbour across a box face is a neighbour); and the estimator must be blind to
a rigid translation, which is the module's stated limitation and therefore has to
be a test rather than a sentence.

Plus the one that is about the physics: the loss must SEE a clump that a matched
mean density hides -- the whole premise is that ``r_k`` reports substructure a
grid statistic cannot -- and must fall when the clump is rebuilt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cosmo_sr.features.knn_cdf import (  # noqa: E402
    EPS_MPC_H, KnnConfig, interior_queries, knn_cdf_loss, knn_log_distances,
    knn_neighbour_ids, knn_stratified_loss, knn_tile_loss, strata_partition,
    tile_geometry,
)
from cosmo_sr.features.member_gather import tile_particles  # noqa: E402

BOX = 100.0


def _cfg(**kw):
    base = dict(ks=(2, 4, 8), n_queries=24, n_quantiles=0, buffer_cells=0.0)
    base.update(kw)
    return KnnConfig(**base)


def _cloud(n=96, seed=3):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, BOX, size=(n, 3))


# --------------------------------------------------------------------------- #
# The gradient
# --------------------------------------------------------------------------- #
def _reference_loss(x, query, ks, target):
    """All-pairs distances with a fully differentiable ``topk``. No index tricks.

    This is the implementation the module is a memory-efficient rewrite of, so
    agreement between the two is the statement that the rewrite changed the cost
    and not the function.
    """
    delta = x[query][:, None, :] - x[None, :, :]
    delta = delta - BOX * torch.round(delta / BOX)
    r = torch.sqrt(delta.square().sum(-1) + EPS_MPC_H ** 2)
    mask = torch.arange(x.shape[0])[None, :] == torch.as_tensor(query)[:, None]
    r = r.masked_fill(mask, float("inf"))
    sel = r.topk(int(max(ks)), dim=1, largest=False, sorted=True).values
    z = sel[:, torch.as_tensor(ks) - 1].log()
    return (z.sort(dim=0).values - target).abs().mean()


def test_gradient_matches_all_pairs_topk():
    cfg = _cfg()
    a, b = _cloud(seed=11), _cloud(seed=12)
    query = torch.arange(0, 96, 4)
    target = knn_log_distances(torch.tensor(b, dtype=torch.float64), query, cfg,
                               boxsize_mpc_h=BOX).sort(dim=0).values

    x = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    z = knn_log_distances(x, query, cfg, boxsize_mpc_h=BOX)
    loss = (z.sort(dim=0).values - target).abs().mean()
    grad, = torch.autograd.grad(loss, x)

    xr = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    ref = _reference_loss(xr, query, cfg.ks, target)
    grad_ref, = torch.autograd.grad(ref, xr)

    assert float((loss - ref).abs()) < 1e-12
    assert float((grad - grad_ref).abs().max()) < 1e-10
    # Not a tautology: the gradient must be non-trivial for the agreement to mean
    # anything, and it must reach NEIGHBOURS as well as queries -- rows that are
    # not query rows have to carry gradient too, or only half of every pair does.
    assert float(grad.norm()) > 1e-6
    non_query = np.setdiff1d(np.arange(96), query.numpy())
    assert float(grad[non_query].abs().max()) > 1e-9


def test_gradient_matches_finite_differences():
    cfg = _cfg()
    a, b = _cloud(seed=21), _cloud(seed=22)
    query = torch.arange(0, 96, 4)
    target = knn_log_distances(torch.tensor(b, dtype=torch.float64), query, cfg,
                               boxsize_mpc_h=BOX).sort(dim=0).values

    def loss_at(arr):
        z = knn_log_distances(torch.tensor(arr, dtype=torch.float64), query, cfg,
                              boxsize_mpc_h=BOX)
        return float((z.sort(dim=0).values - target).abs().mean())

    x = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    z = knn_log_distances(x, query, cfg, boxsize_mpc_h=BOX)
    grad, = torch.autograd.grad((z.sort(dim=0).values - target).abs().mean(), x)
    g = grad.numpy().ravel()

    # The largest components only: a component whose exact derivative is ~0 has a
    # finite-difference estimate dominated by the |.| kink and says nothing.
    h = 1e-6
    fd, exact = [], []
    for i in np.argsort(np.abs(g))[-8:]:
        plus, minus = a.copy(), a.copy()
        plus.ravel()[i] += h
        minus.ravel()[i] -= h
        # The search is redone inside both forwards, so this also checks that a
        # step which CHANGES a neighbour ranking does not break the estimate.
        fd.append((loss_at(plus) - loss_at(minus)) / (2 * h))
        exact.append(g[i])
    rel = np.linalg.norm(np.subtract(fd, exact)) / np.linalg.norm(exact)
    assert rel < 1e-5, f"fd {fd} vs autograd {exact}"


def test_search_is_detached_and_distances_are_not():
    cfg = _cfg()
    x = torch.tensor(_cloud(), dtype=torch.float64, requires_grad=True)
    query = torch.arange(0, 96, 4)
    ids = knn_neighbour_ids(x, query, cfg, boxsize_mpc_h=BOX)
    assert ids.dtype == torch.long and not ids.requires_grad
    assert knn_log_distances(x, query, cfg, boxsize_mpc_h=BOX).requires_grad


def test_chunked_search_is_exact():
    """Element-bounded chunking must not change the answer, only the peak."""
    x = torch.tensor(_cloud(n=128, seed=5), dtype=torch.float64)
    query = torch.arange(0, 128, 2)
    whole = knn_neighbour_ids(x, query, _cfg(), boxsize_mpc_h=BOX)
    # 128 candidates x 300 elems => 2 rows per pass, i.e. 32 passes.
    chunked = knn_neighbour_ids(x, query, _cfg(search_max_elems=300),
                                boxsize_mpc_h=BOX)
    assert torch.equal(whole, chunked)


def test_self_match_detected_by_identity_not_by_rank():
    """Coincident particles must not be mistaken for the query itself."""
    base = _cloud(n=64, seed=7)
    twin = base.copy()
    twin[5] = twin[4]                      # exactly coincident with row 4
    x = torch.tensor(twin, dtype=torch.float64)
    ids = knn_neighbour_ids(x, torch.tensor([4]), _cfg(ks=(1, 2)),
                            boxsize_mpc_h=BOX)
    # Rank 1 for query 4 is its twin at distance 0 -- a real neighbour. A
    # "drop the first column" implementation would have deleted it and reported
    # rank 1 as the next particle out.
    assert int(ids[0, 0]) == 5
    z = knn_log_distances(x, torch.tensor([4]), _cfg(ks=(1, 2)),
                          boxsize_mpc_h=BOX)
    assert torch.isfinite(z).all()         # log(0) would be -inf without EPS


def test_query_rows_must_index_pos():
    cfg = _cfg()
    x = torch.tensor(_cloud(), dtype=torch.float64)
    with pytest.raises(ValueError, match="exactly once"):
        knn_neighbour_ids(x[:80], torch.tensor([90]), cfg, boxsize_mpc_h=BOX)


# --------------------------------------------------------------------------- #
# Geometry and periodicity
# --------------------------------------------------------------------------- #
def test_periodicity_is_real():
    """Two particles either side of a face are neighbours, not antipodes."""
    pts = np.array([[0.05, 50.0, 50.0], [BOX - 0.05, 50.0, 50.0],
                    [50.0, 50.0, 50.0], [30.0, 10.0, 70.0]])
    x = torch.tensor(pts, dtype=torch.float64)
    ids = knn_neighbour_ids(x, torch.tensor([0]), _cfg(ks=(1,)),
                            boxsize_mpc_h=BOX)
    assert int(ids[0, 0]) == 1
    r = float(knn_log_distances(x, torch.tensor([0]), _cfg(ks=(1,)),
                                boxsize_mpc_h=BOX).exp())
    assert r == pytest.approx(0.1, abs=1e-6)


def test_loss_is_invariant_under_rigid_translation():
    """The module's stated limitation, as a test: no position information.

    This is why the term cannot replace ``w_centre``. A run that used it alone
    to place objects would be optimising a quantity that is constant in the
    answer, which is the failure the 2026-08-21 free-field run measured.
    """
    cfg = _cfg(n_quantiles=4)
    a = torch.tensor(_cloud(seed=31), dtype=torch.float64)
    hr = torch.tensor(_cloud(seed=32), dtype=torch.float64)
    query = torch.arange(0, 96, 4)
    zh = knn_log_distances(hr, query, cfg, boxsize_mpc_h=BOX)
    here, _ = knn_cdf_loss(knn_log_distances(a, query, cfg, boxsize_mpc_h=BOX),
                           zh, cfg)
    shift = torch.tensor([9.0, -3.0, 17.0], dtype=torch.float64)
    moved = torch.remainder(a + shift, BOX)
    there, _ = knn_cdf_loss(
        knn_log_distances(moved, query, cfg, boxsize_mpc_h=BOX), zh, cfg)
    assert float(here) == pytest.approx(float(there), rel=1e-12)


def test_tile_geometry_matches_tile_particles():
    """The buffer's cube must be the tile's cube, on the same lattice."""
    ng, tile = 64, 16
    for tid in (0, 5, 37, 63):
        field = torch.zeros((1, 6, tile, tile, tile), dtype=torch.float64)
        pos, _ = tile_particles(field, [tid], ng_hr=ng, tile_hr=tile,
                                boxsize_mpc_h=BOX, dis_scale_mpc_h=1.0)
        centre, side, cell = tile_geometry(tid, ng_hr=ng, tile_hr=tile,
                                           boxsize_mpc_h=BOX)
        # Zero displacement: every particle sits on its Lagrangian site, so the
        # table's own bounding box must be the cube this module computes.
        assert side == pytest.approx(float(tile) * BOX / ng)
        assert cell == pytest.approx(BOX / ng)
        lo, hi = pos.amin(dim=0), pos.amax(dim=0)
        assert torch.allclose(0.5 * (lo + hi), centre, atol=1e-9)
        assert float((hi - lo).max()) == pytest.approx(side - cell, abs=1e-9)


def test_interior_queries_respect_the_buffer_in_eulerian_space():
    ng, tile = 64, 16
    centre, side, cell = tile_geometry(21, ng_hr=ng, tile_hr=tile,
                                       boxsize_mpc_h=BOX)
    field = torch.zeros((1, 6, tile, tile, tile), dtype=torch.float64)
    # Push one interior particle far out of the cube: a Lagrangian cut would keep
    # it (its SITE is interior), an Eulerian cut must drop it.
    field[0, 0].view(-1)[tile ** 3 // 2 + tile // 2] = 10.0
    pos, _ = tile_particles(field, [21], ng_hr=ng, tile_hr=tile,
                            boxsize_mpc_h=BOX, dis_scale_mpc_h=1.0)
    rows = torch.arange(pos.shape[0])
    cfg = _cfg(n_queries=10 ** 6, buffer_cells=2.0)
    kw = dict(tile_centre_mpc_h=centre, tile_side_mpc_h=side, cell_mpc_h=cell,
              boxsize_mpc_h=BOX)
    picked = interior_queries(pos, rows, cfg, **kw)
    d = (pos[picked] - centre).abs().amax(dim=1)
    assert float(d.max()) <= 0.5 * side - 2.0 * cell + 1e-9
    moved = int(tile ** 3 // 2 + tile // 2)
    assert moved not in set(int(i) for i in picked)
    # A buffer wider than the tile leaves nothing, and says so rather than
    # silently charging a term on zero queries.
    assert interior_queries(pos, rows, _cfg(buffer_cells=tile),
                            **kw).numel() == 0


# --------------------------------------------------------------------------- #
# What the loss is for
# --------------------------------------------------------------------------- #
def _clumped(n_clump, n_back, spread, seed=41):
    """``n_back`` uniform particles plus one Gaussian clump of ``n_clump``."""
    rng = np.random.default_rng(seed)
    back = rng.uniform(0.0, BOX, size=(n_back, 3))
    clump = rng.normal(50.0, spread, size=(n_clump, 3))
    return np.concatenate([clump, back], axis=0)


def test_loss_sees_a_clump_at_fixed_mean_density():
    """The premise. Same particle count, same box: only the clumping differs.

    A mean-density statistic is identical across these three fields by
    construction, so a non-trivial ordering here is the thing P(k) at the grid
    scale could not give us.
    """
    cfg = _cfg(ks=(2, 4, 8), n_queries=200, n_quantiles=8, buffer_cells=0.0)
    hr = torch.tensor(_clumped(120, 400, 0.3, seed=51), dtype=torch.float64)
    diffuse = torch.tensor(_clumped(120, 400, 2.5, seed=51), dtype=torch.float64)
    tight = torch.tensor(_clumped(120, 400, 0.3, seed=52), dtype=torch.float64)
    query = torch.arange(0, 520, 2)
    zh = knn_log_distances(hr, query, cfg, boxsize_mpc_h=BOX)

    def at(x):
        return knn_cdf_loss(
            knn_log_distances(x, query, cfg, boxsize_mpc_h=BOX), zh, cfg)

    l_diffuse, d_diffuse = at(diffuse)
    l_tight, d_tight = at(tight)
    assert float(l_diffuse) > 5.0 * float(l_tight)
    # Under-clumped reads as neighbours FURTHER away, which is the sign
    # convention the diagnostic key promises -- and it reads in the LOW
    # quantiles, where the clumped particles are.
    assert min(d_diffuse["ratio_q10"]) > 2.0
    assert all(abs(r - 1.0) < 0.2 for r in d_tight["ratio_q10"])
    # The finding that made ``ratio_q10`` exist: the median is background and
    # does not see a 120-particle clump at all. If this ever starts failing the
    # verdict scripts may be reading the median again and should be checked.
    assert all(abs(r - 1.0) < 0.05 for r in d_diffuse["ratio_med"])


def _fof_largest(arr, n_clump, link=0.2):
    """Largest friends-of-friends group among the clump's particles.

    The label is used for this diagnostic only, never by the loss.
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    c = arr[:n_clump]
    tree = cKDTree(c)
    _, lab = connected_components(tree.sparse_distance_matrix(tree, link),
                                  directed=False)
    return int(np.bincount(lab).max())


def _descend(ks, steps=400, seed=61):
    """Adam on this loss alone, from a diffuse clump toward HR's tight one."""
    hr = _clumped(150, 300, 0.3, seed=seed)
    start = _clumped(150, 300, 2.0, seed=seed)
    cfg = KnnConfig(ks=ks, n_queries=225, n_quantiles=8, buffer_cells=0.0,
                    dead_zone=0.0)
    query = torch.arange(0, 450, 2)
    zh = knn_log_distances(torch.tensor(hr), query, cfg, boxsize_mpc_h=BOX)
    x = torch.tensor(start, requires_grad=True)
    opt = torch.optim.Adam([x], lr=0.05)
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, _ = knn_cdf_loss(
            knn_log_distances(x, query, cfg, boxsize_mpc_h=BOX), zh, cfg)
        losses.append(float(loss.detach()))
        loss.backward()
        assert torch.isfinite(x.grad).all()
        opt.step()
    return hr, start, x.detach().numpy(), losses


def test_small_ranks_fragment_large_ranks_build():
    """THE design constraint of this module, as a regression test.

    Both rank sets drive the loss to (numerically) zero -- so the loss value
    cannot tell them apart. The object can: ranks capped well below the clump's
    particle count satisfy the distribution with a field of small knots, and
    ranks reaching the clump's size cannot. Measured when written: largest FOF
    group 13 vs 107 of 150 (HR 112). This is why ``KnnConfig.ks`` defaults to a
    128 cap; if a future change lowers it, this test says what that buys.
    """
    hr, start, small, l_small = _descend((2, 4, 8))
    _, _, large, l_large = _descend((8, 32, 64, 128))
    assert l_small[-1] < 1e-3 * l_small[0]
    assert l_large[-1] < 1e-3 * l_large[0]
    n_hr = _fof_largest(hr, 150)
    assert _fof_largest(start, 150) < 10
    assert _fof_largest(small, 150) < 0.3 * n_hr          # fragments
    assert _fof_largest(large, 150) > 0.8 * n_hr          # a halo

    def r50(arr):
        c = arr[:150]
        return float(np.median(np.linalg.norm(c - c.mean(0), axis=1)))
    assert r50(large) < 1.3 * r50(hr)
    assert r50(small) > 3.0 * r50(hr)


def test_dead_zone_and_one_sided_shapes():
    cfg = _cfg(n_queries=64, n_quantiles=8, dead_zone=0.05)
    z = knn_log_distances(torch.tensor(_cloud(seed=71), dtype=torch.float64),
                          torch.arange(0, 96, 2), cfg, boxsize_mpc_h=BOX)
    # Exactly on the reference: zero value AND zero gradient, the property every
    # hinged term in this line has.
    zc = z.clone().requires_grad_(True)
    loss, diag = knn_cdf_loss(zc, z, cfg)
    assert float(loss) == 0.0
    loss.backward()
    assert float(zc.grad.abs().max()) == 0.0
    assert all(r == pytest.approx(1.0) for r in diag["ratio_med"])
    # Inside the dead zone: a 3% error at a 5% free zone is still free.
    assert float(knn_cdf_loss(z + np.log(1.03), z, cfg)[0]) == 0.0
    assert float(knn_cdf_loss(z + np.log(1.30), z, cfg)[0]) > 0.0
    # One-sided charges the deficit (candidate distances too large) only.
    one = _cfg(n_queries=64, n_quantiles=8, dead_zone=0.05, one_sided=True)
    assert float(knn_cdf_loss(z + np.log(1.30), z, one)[0]) > 0.0
    assert float(knn_cdf_loss(z - np.log(1.30), z, one)[0]) == 0.0
    assert float(knn_cdf_loss(z - np.log(1.30), z, cfg)[0]) > 0.0


def test_quantile_reduction_is_equal_count_and_degenerates_safely():
    cfg_full = _cfg(n_queries=40, n_quantiles=0)
    cfg_bins = _cfg(n_queries=40, n_quantiles=5)
    z = knn_log_distances(torch.tensor(_cloud(seed=81), dtype=torch.float64),
                          torch.arange(0, 80, 2), cfg_full, boxsize_mpc_h=BOX)
    assert knn_cdf_loss(z, z, cfg_bins)[1]["n_quantiles"] == 5
    # More bins than queries is the identity, not an error and not empty bins.
    assert knn_cdf_loss(z, z, _cfg(n_queries=40, n_quantiles=10 ** 4)
                        )[1]["n_quantiles"] == z.shape[0]
    # Unequal query counts are legal with bins and refused without them.
    with pytest.raises(ValueError, match="equal query counts"):
        knn_cdf_loss(z[:30], z, cfg_full)
    knn_cdf_loss(z[:30], z, cfg_bins)


def test_tile_loss_runs_on_a_real_tile_table():
    """End to end on ``tile_particles`` output, with a displaced candidate."""
    ng, tile = 64, 16
    torch.manual_seed(0)
    hr_field = torch.zeros((1, 6, tile, tile, tile), dtype=torch.float64)
    hr_field[0, 0:3] = 0.02 * torch.randn_like(hr_field[0, 0:3])
    cand_field = hr_field.clone()
    cand_field[0, 0:3] *= 0.3                      # under-clustered candidate
    cand_field.requires_grad_(True)
    kw = dict(ng_hr=ng, tile_hr=tile, boxsize_mpc_h=BOX, dis_scale_mpc_h=6.0)
    cand_pos, _ = tile_particles(cand_field, [21], **kw)
    hr_pos, _ = tile_particles(hr_field, [21], **kw)
    cfg = KnnConfig(ks=(2, 4, 8), n_queries=256, n_quantiles=16,
                    buffer_cells=2.0)
    loss, diag = knn_tile_loss(cand_pos, hr_pos, torch.arange(cand_pos.shape[0]),
                               21, cfg, ng_hr=ng, tile_hr=tile,
                               boxsize_mpc_h=BOX)
    assert not diag["skipped"] and diag["tile"] == 21
    assert diag["n_queries_cand"] == diag["n_queries_hr"] == 256
    loss.backward()
    assert torch.isfinite(cand_field.grad).all()
    assert float(cand_field.grad.abs().max()) > 0.0
    # A buffer that swallows the tile skips rather than charging on nothing.
    wide = KnnConfig(ks=(2, 4, 8), n_queries=256, buffer_cells=tile)
    l0, d0 = knn_tile_loss(cand_pos, hr_pos, torch.arange(cand_pos.shape[0]),
                           21, wide, ng_hr=ng, tile_hr=tile, boxsize_mpc_h=BOX)
    assert d0["skipped"] and float(l0) == 0.0


def test_config_refuses_nonsense():
    with pytest.raises(ValueError, match="empty"):
        KnnConfig(ks=())
    with pytest.raises(ValueError, match="repeats"):
        KnnConfig(ks=(8, 8))
    with pytest.raises(ValueError, match="below 1"):
        KnnConfig(ks=(0, 4))
    with pytest.raises(ValueError, match="negative"):
        KnnConfig(dead_zone=-0.1)
    with pytest.raises(ValueError, match="negative"):
        KnnConfig(buffer_cells=-1.0)


def test_too_few_candidates_is_refused():
    x = torch.tensor(_cloud(n=6), dtype=torch.float64)
    with pytest.raises(ValueError, match="cannot supply"):
        knn_neighbour_ids(x, torch.tensor([0]), _cfg(ks=(8,)),
                          boxsize_mpc_h=BOX)


# --------------------------------------------------------------------------- #
# Neighbouring tiles and the recentred cube (the job-49259 fix)
# --------------------------------------------------------------------------- #
from cosmo_sr.features.knn_cdf import (  # noqa: E402
    box_neighbours, box_positions, periodic_centroid, tile_rows_flat,
)


def test_periodic_centroid_straddles_a_face():
    pts = torch.tensor([[0.5, 50.0, 50.0], [99.5, 50.0, 50.0],
                        [1.0, 50.0, 50.0], [99.0, 50.0, 50.0]], dtype=torch.float64)
    c = periodic_centroid(pts, BOX)
    # The arithmetic mean is 50 -- the far side of the box.
    d = float(c[0]) if float(c[0]) < 50 else float(c[0]) - BOX
    assert abs(d) < 1e-9
    assert float(c[1]) == pytest.approx(50.0)


def test_box_positions_and_tile_rows_match_tile_particles():
    ng, tile = 32, 8
    rng = np.random.default_rng(4)
    f = (0.3 * rng.standard_normal((6, ng, ng, ng))).astype(np.float32)
    pos = box_positions(f, ng=ng, boxsize_mpc_h=BOX, dis_scale_mpc_h=6.0,
                        dtype=np.float64)
    tid = 45
    n_side = ng // tile
    tx, ty, tz = tid // (n_side * n_side), (tid // n_side) % n_side, tid % n_side
    sub = f[:, tx * tile:(tx + 1) * tile, ty * tile:(ty + 1) * tile,
            tz * tile:(tz + 1) * tile]
    tp, _ = tile_particles(torch.tensor(sub[None], dtype=torch.float64), [tid],
                           ng_hr=ng, tile_hr=tile, boxsize_mpc_h=BOX,
                           dis_scale_mpc_h=6.0)
    d = pos[tile_rows_flat(tid, ng=ng, tile=tile)] - tp.numpy()
    d -= BOX * np.round(d / BOX)
    assert np.abs(d).max() < 1e-4


def test_box_neighbours_excludes_own_tile_and_respects_the_cube():
    ng, tile = 32, 8
    rng = np.random.default_rng(6)
    pos = rng.uniform(0, BOX, size=(ng ** 3, 3))
    centre = np.array([1.0, 50.0, 50.0])
    ids, xp = box_neighbours(pos, 3, centre, 6.0, ng=ng, tile=tile,
                             boxsize_mpc_h=BOX, chunk=1000)
    own = set(tile_rows_flat(3, ng=ng, tile=tile).tolist())
    assert not own & set(ids.tolist())
    d = xp - centre
    d -= BOX * np.round(d / BOX)
    assert np.abs(d).max() <= 6.0 + 1e-12
    # Periodic: particles just below L on the x axis are inside the cube.
    assert (xp[:, 0] > 90).any()
    # Exhaustive: nothing outside the own tile within the cube was missed.
    dd = pos - centre
    dd -= BOX * np.round(dd / BOX)
    inside = np.flatnonzero(np.abs(dd).max(axis=1) <= 6.0)
    assert set(ids.tolist()) == set(inside.tolist()) - own


def test_extras_complete_a_truncated_candidate_set():
    """Offering the rest of the field as constants recovers the full-box r_k,
    and the extras carry no gradient."""
    rng = np.random.default_rng(8)
    full = rng.uniform(0, BOX, size=(3000, 3))
    live = torch.tensor(full[:600], dtype=torch.float64, requires_grad=True)
    extra = torch.tensor(full[600:], dtype=torch.float64, requires_grad=True)
    cfg = _cfg(ks=(4, 16))
    q = torch.arange(0, 600, 7)
    z_full = knn_log_distances(torch.tensor(full), q, cfg, boxsize_mpc_h=BOX)
    z_live = knn_log_distances(live, q, cfg, boxsize_mpc_h=BOX)
    z_both = knn_log_distances(torch.cat([live, extra.detach()]), q, cfg,
                               boxsize_mpc_h=BOX)
    assert float((z_live - z_full).min()) >= -1e-12
    assert float((z_live - z_full).max()) > 0.1          # truncation biases up
    assert torch.allclose(z_both, z_full)
    z_both.sum().backward()
    assert extra.grad is None and live.grad is not None


def test_tile_loss_with_extras_and_centroid():
    ng, tile = 64, 16
    torch.manual_seed(1)
    hr_field = torch.zeros((1, 6, tile, tile, tile), dtype=torch.float64)
    hr_field[0, 0:3] = 0.02 * torch.randn_like(hr_field[0, 0:3])
    hr_field[0, 0] += 0.5            # a coherent 3 Mpc/h bulk shift
    cand_field = hr_field.clone().requires_grad_(True)
    kw = dict(ng_hr=ng, tile_hr=tile, boxsize_mpc_h=BOX, dis_scale_mpc_h=6.0)
    cp, _ = tile_particles(cand_field, [21], **kw)
    hp, _ = tile_particles(hr_field, [21], **kw)
    extra = torch.tensor(np.random.default_rng(2).uniform(0, BOX, (5000, 3)))
    cfg = KnnConfig(ks=(2, 8), n_queries=128, n_quantiles=8, buffer_cells=1.0)
    rows = torch.arange(cp.shape[0])
    loss, d = knn_tile_loss(cp, hp, rows, 21, cfg, ng_hr=ng, tile_hr=tile,
                            boxsize_mpc_h=BOX, cand_extra=extra, hr_extra=extra,
                            centre="centroid",
                            generator=torch.Generator().manual_seed(0))
    assert not d["skipped"] and d["n_extra_cand"] == 5000
    assert d["centre"] == "centroid"
    loss.backward()
    assert torch.isfinite(cand_field.grad).all()
    # The tile moved 3 Mpc/h in bulk (its side here is 25 Mpc/h): a cube on the
    # Lagrangian centre holds less of the tile than one on its centroid.
    from cosmo_sr.features.knn_cdf import interior_queries, tile_geometry
    lag, side, cell = tile_geometry(21, ng_hr=ng, tile_hr=tile, boxsize_mpc_h=BOX)
    big = KnnConfig(ks=(2, 8), n_queries=10 ** 6, buffer_cells=1.0)
    n_lag = interior_queries(hp, rows, big, tile_centre_mpc_h=lag,
                             tile_side_mpc_h=side, cell_mpc_h=cell,
                             boxsize_mpc_h=BOX).numel()
    n_cen = interior_queries(hp, rows, big,
                             tile_centre_mpc_h=periodic_centroid(hp, BOX),
                             tile_side_mpc_h=side, cell_mpc_h=cell,
                             boxsize_mpc_h=BOX).numel()
    assert n_cen > n_lag
    with pytest.raises(ValueError, match="centre"):
        knn_tile_loss(cp, hp, rows, 21, cfg, ng_hr=ng, tile_hr=tile,
                      boxsize_mpc_h=BOX, centre="middle")


# --------------------------------------------------------------------------- #
# Round 5b: 6D phase space and class conditioning
# --------------------------------------------------------------------------- #
from cosmo_sr.features import knn_cdf as kc  # noqa: E402


def _phase_pts(n=240, seed=0, lam=1000.0, L=100.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((n, 3), generator=g, dtype=torch.float64) * L
    v = torch.randn((n, 3), generator=g, dtype=torch.float64) * 300.0
    return torch.cat([x, v / lam], dim=1)


def test_6d_search_matches_brute_force_periodic_in_x_only():
    pts = _phase_pts()
    L, n = 100.0, pts.shape[0]
    d = pts[:, None, :] - pts[None, :, :]
    d = torch.cat([d[..., :3] - L * torch.round(d[..., :3] / L), d[..., 3:]], -1)
    brute = d.square().sum(-1).sqrt()
    brute.fill_diagonal_(float("inf"))
    cfg = kc.KnnConfig(ks=(1, 4, 16), n_queries=n, buffer_cells=0.0)
    rows = torch.arange(n)
    got = torch.exp(kc.knn_log_distances(pts, rows, cfg, boxsize_mpc_h=L))
    want = brute.sort(dim=1).values[:, [0, 3, 15]]
    assert torch.allclose(got, want, atol=1e-9)


def test_6d_gradient_reaches_velocity_and_matches_finite_differences():
    pts = _phase_pts(seed=2).requires_grad_(True)
    cfg = kc.KnnConfig(ks=(4,), n_queries=64, n_quantiles=8, buffer_cells=0.0)
    rows = torch.arange(pts.shape[0])
    z = kc.knn_log_distances(pts, rows, cfg, boxsize_mpc_h=100.0)
    z.sum().backward()
    g = pts.grad.clone()
    assert float(g[:, 3:].abs().max()) > 0.0          # velocity is optimised too

    def f(p):
        return float(kc.knn_log_distances(p, rows, cfg, boxsize_mpc_h=100.0).sum())
    eps = 1e-6
    for i, j in ((3, 4), (17, 0), (55, 5)):
        p = pts.detach().clone()
        p[i, j] += eps
        m = pts.detach().clone()
        m[i, j] -= eps
        assert (f(p) - f(m)) / (2 * eps) == pytest.approx(float(g[i, j]), rel=2e-4)


def test_6d_is_galilean_and_translation_invariant():
    pts = _phase_pts(seed=4)
    cfg = kc.KnnConfig(ks=(2, 8), n_queries=120, n_quantiles=10, buffer_cells=0.0)
    rows = torch.arange(pts.shape[0])
    base = kc.knn_log_distances(pts, rows, cfg, boxsize_mpc_h=100.0)
    moved = pts.clone()
    moved[:, :3] = torch.remainder(moved[:, :3] + 13.7, 100.0)   # translate x
    moved[:, 3:] += 0.42                                          # boost v
    assert torch.allclose(
        kc.knn_log_distances(moved, rows, cfg, boxsize_mpc_h=100.0), base,
        atol=1e-9)


def test_class_conditioning_follows_the_named_class():
    """Two spatially separate classes, only one of them damaged.

    The charged class must dominate the loss, the queries must be drawn from it
    alone, and charging both must average the two equally. Note what is NOT
    claimed: gradient reaching only that class. A charged query's NEIGHBOURS may
    belong to any class, and moving them is a legitimate way to fix its r_k.
    """
    g = torch.Generator().manual_seed(7)
    tid = 100
    centre, side, _ = kc.tile_geometry(tid)
    n = 1200
    half = n // 2
    off = torch.tensor([side * 0.25, 0.0, 0.0], dtype=torch.float64)
    blob = lambda c, m: c + (torch.rand((m, 3), generator=g,  # noqa: E731
                                        dtype=torch.float64) - 0.5) * side * 0.25
    hr = torch.cat([blob(centre - off, half), blob(centre + off, half)])
    cls = torch.zeros(n, dtype=torch.long)
    cls[half:] = 3                                    # 0 = sub_50_200, 3 = host
    cand = hr.clone()
    cand[:half] = (centre - off) + (cand[:half] - (centre - off)) * 2.0
    cand.requires_grad_(True)
    cfg = kc.KnnConfig(ks=(4, 16), n_queries=200, n_quantiles=8, buffer_cells=0.0)
    rows = torch.arange(n)
    # A fixed generator per call: queries are drawn at random, so without one
    # the three calls below would not be the same measurement.
    gen = lambda: torch.Generator().manual_seed(0)          # noqa: E731
    kw = dict(ng_hr=512, tile_hr=64, boxsize_mpc_h=100.0)
    wrong, d0 = kc.knn_tile_loss(cand, hr, rows, tid, cfg, classes=cls,
                                 class_ids=(0,), generator=gen(), **kw)
    right, _ = kc.knn_tile_loss(cand, hr, rows, tid, cfg, classes=cls,
                                class_ids=(3,), generator=gen(), **kw)
    assert float(wrong) > 5 * float(right)
    assert d0["class_ids"] == [0] and d0["n_classes_charged"] == 1
    assert d0["classes"]["class_0"]["n_queries_cand"] <= half
    wrong.backward()
    assert float(cand.grad[:half].abs().max()) > 0.0
    both, db = kc.knn_tile_loss(cand.detach(), hr, rows, tid, cfg, classes=cls,
                                class_ids=(0, 3), generator=gen(), **kw)
    # Equal weighting, to the precision the random query draw allows: the
    # combined call draws each class's queries from the same generator in
    # sequence, so the two runs are not the same sample.
    assert float(both) == pytest.approx(0.5 * (float(wrong) + float(right)),
                                        rel=0.02)
    assert float(right) < float(both) < float(wrong)
    assert db["n_classes_charged"] == 2


def test_a_class_too_small_to_fill_the_bins_is_skipped():
    g = torch.Generator().manual_seed(11)
    centre, side, _ = kc.tile_geometry(0)
    n = 400
    pts = centre + (torch.rand((n, 3), generator=g, dtype=torch.float64) - 0.5) * side * 0.8
    cfg = kc.KnnConfig(ks=(2,), n_queries=64, n_quantiles=16, buffer_cells=0.0)
    cls = torch.zeros(n, dtype=torch.long)
    cls[:3] = 1                                   # three particles: too few bins
    rows = torch.arange(n)
    loss, d = kc.knn_tile_loss(pts, pts, rows, 0, cfg, classes=cls, class_ids=(1,))
    assert d["skipped"] and float(loss) == 0.0
    # ... and a skipped class does not drag the average of the others down.
    _, d2 = kc.knn_tile_loss(pts, pts, rows, 0, cfg, classes=cls, class_ids=(0, 1))
    assert not d2["skipped"] and d2["n_classes_charged"] == 1


def test_class_labels_must_match_the_tile_rows():
    pts = _phase_pts(n=300, seed=1)[:, :3]
    cfg = kc.KnnConfig(ks=(2,), n_queries=50, n_quantiles=4, buffer_cells=0.0)
    with pytest.raises(ValueError, match="one label per row"):
        kc.knn_tile_loss(pts, pts, torch.arange(300), 0, cfg,
                         classes=torch.zeros(7, dtype=torch.long), class_ids=(0,))


def test_box_phase_appends_scaled_velocity():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((6, 8, 8, 8)).astype(np.float32)
    p3 = kc.box_positions(f, ng=8)
    p6 = kc.box_phase(f, ng=8, vel_scale_kms=313.4, phase_lambda=4000.0)
    assert p6.shape == (512, 6) and np.allclose(p6[:, :3], p3)
    assert np.allclose(p6[:, 3], f[3].reshape(-1) * (313.4 / 4000.0), atol=1e-6)
    assert kc.box_phase(f, ng=8).shape == (512, 3)


def test_classify_owner_uses_the_catalog_cuts():
    owner = np.array([-1, 0, 1, 2, 3, 7, -1], dtype=np.int32)
    got = [kc.OWNER_CLASSES[i] for i in kc.classify_owner(
        owner, np.array([0, 1, 2, 3, 7]), np.array([-1, 0, 0, 0, -1]),
        np.array([5000, 120, 40, 900, 300]))]
    assert got == ["unbound", "host", "sub_50_200", "sub_lt50", "sub_ge200",
                   "host", "unbound"]


# --------------------------------------------------------------------------- #
# Density strata: the catalog-free replacement for the HR owner-class mask
# --------------------------------------------------------------------------- #
# The point of these is NOT that the feature runs. It is that two of its
# plausible-sounding claims are false, both were believed while it was being
# written, and both are cheap to re-believe later. So they are tests.
N_SUB, N_HOST, N_BACK = 120, 1200, 600


def _diluted(sub_spread, seed=11):
    """A small tight subhalo inside a dominant host, plus background.

    The round-5b situation in miniature and the reason ``--knn-classes`` exists:
    the subhalo is 6% of the particles, so any statistic pooled over the whole
    interior is a statement about the host. Labels (the slice boundaries) are
    used for composition diagnostics only, never by the loss.
    """
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.normal(50.0, sub_spread, size=(N_SUB, 3)),
        rng.normal(50.0, 4.0, size=(N_HOST, 3)),
        rng.uniform(0.0, BOX, size=(N_BACK, 3)),
    ], axis=0)


def _diluted_z(sub_spread, cfg, seed=11):
    x = torch.tensor(_diluted(sub_spread, seed=seed))
    q = torch.arange(N_SUB + N_HOST + N_BACK)
    return knn_log_distances(x, q, cfg, boxsize_mpc_h=BOX)


def _scfg(**kw):
    base = dict(ks=(8, 32, 64, 128), n_queries=N_SUB + N_HOST + N_BACK,
                n_quantiles=16, buffer_cells=0.0, dead_zone=0.0)
    base.update(kw)
    return KnnConfig(**base)


def test_strata_partition_is_equal_count_and_dense_first():
    cfg = _scfg(strata=8)
    z = _diluted_z(1.5, cfg)
    groups = strata_partition(z, cfg)
    assert len(groups) == 8
    sizes = {int(g.numel()) for g in groups}
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes for sizes in (int(g.numel()) for g in groups)) == z.shape[0]
    # Dense first, and the split is on the stratifying column only.
    col = list(cfg.ks).index(min(cfg.ks))
    means = [float(z[g, col].mean()) for g in groups]
    assert means == sorted(means)
    # Every row lands in exactly one stratum.
    allrows = torch.cat(groups).sort().values
    assert torch.equal(allrows, torch.arange(z.shape[0]))


def test_strata_on_the_stratifying_rank_is_finer_binning():
    """The claim that keeps this feature honest, as exact arithmetic.

    With a SINGLE rank there is no conditional information to be had, and
    equal-count strata of Q bins each are the same equal-count bins as the
    pooled form with strata*Q. If this ever stops holding, the partition has
    started moving weight around and the docstring's "no new information in that
    column" is no longer true.
    """
    n = 480
    cfg_p = _scfg(ks=(8,), n_quantiles=32)
    cfg_s = _scfg(ks=(8,), n_quantiles=4, strata=8)
    zc = _diluted_z(1.5, cfg_p)[:n]
    zh = _diluted_z(0.25, cfg_p)[:n]
    pooled = float(knn_cdf_loss(zc, zh, cfg_p)[0])
    strat = float(knn_stratified_loss(zc, zh, cfg_s)[0])
    assert strat == pytest.approx(pooled, rel=1e-12)


def test_equal_count_strata_do_not_reweight_the_dense_tail():
    """MEASURED, and it is the finding that made ``strata_charge`` exist.

    Partitioning the queries by density and charging every stratum equally is
    NOT a fix for host domination: it is the pooled loss in finer bins, to
    within the small conditional term the other ranks contribute. The dense tail
    carries the same share of the gradient either way. Only restricting the
    charge changes that, and the ratio here is ~8x -- the number of strata.
    """
    cfg_p = _scfg(strata=0)
    cfg_e = _scfg(strata=8)
    cfg_d = _scfg(strata=8, strata_charge=(0,))
    zh = _diluted_z(0.25, cfg_p)
    zc = _diluted_z(1.5, cfg_p)
    pooled = float(knn_cdf_loss(zc, zh, cfg_p)[0])
    equal = float(knn_stratified_loss(zc, zh, cfg_e)[0])
    dense = float(knn_stratified_loss(zc, zh, cfg_d)[0])
    assert equal == pytest.approx(pooled, rel=0.15)     # measured 0.0912/0.0881
    assert dense > 5.0 * equal                          # measured 0.726/0.0912
    # The diagnostic, unlike the loss, IS sharpened by partitioning alone: the
    # pooled median cannot see the subhalo and the dense stratum's can.
    _, d_e = knn_stratified_loss(zc, zh, cfg_e)
    _, d_p = knn_cdf_loss(zc, zh, cfg_p)
    assert all(abs(r - 1.0) < 0.02 for r in d_p["ratio_med"])
    assert max(d_e["ratio_med_dense"]) > 1.5


def test_uncharged_strata_are_measured_but_carry_no_gradient():
    cfg = _scfg(strata=8, strata_charge=(0,))
    zh = _diluted_z(0.25, cfg)
    zc = _diluted_z(1.5, cfg).requires_grad_(True)
    loss, diag = knn_stratified_loss(zc, zh, cfg)
    assert diag["n_strata_charged"] == 1
    assert diag["strata_charge"] == [0]
    assert len(diag["strata"]) == 8                    # all eight measured
    assert [b["charged"] for b in diag["strata"]].count(True) == 1
    loss.backward()
    groups = strata_partition(zc, cfg)
    g = zc.grad
    assert float(g[groups[0]].abs().sum()) > 0.0
    for j in range(1, 8):
        assert float(g[groups[j]].abs().sum()) == 0.0


def test_the_dense_stratum_is_mostly_host_particles():
    """MEASURED, and it is why this arm has a diagnostic gate before any GPU.

    A density cut is NOT a subhalo cut. ``--knn-classes sub_50_200`` names
    subhalo particles; the densest stratum is 65-83% host on this fixture, and
    the fraction gets WORSE as the candidate's subhalo is made more diffuse --
    the damaged field has fewer of its own subhalo particles in its own dense
    tail. Host over-concentration is this line's measured damage
    (`host-damage-is-mostly-reparenting`), so charging the dense stratum is not
    a free substitution and the composition on the REAL box has to be measured
    (scripts/features/knn_strata_composition.py) before this is trained on.
    """
    cfg = _scfg(strata=8)
    label = np.array(["sub"] * N_SUB + ["host"] * N_HOST + ["back"] * N_BACK)

    def host_fraction(sub_spread):
        z = _diluted_z(sub_spread, cfg)
        dense = strata_partition(z, cfg)[0].numpy()
        return float((label[dense] == "host").mean()), float(
            (label[dense] == "sub").mean())

    h_mild, s_mild = host_fraction(1.5)
    h_bad, s_bad = host_fraction(3.0)
    assert h_mild > 0.5 and h_bad > 0.5
    # Worse when the field is worse: the selection follows the damage.
    assert s_bad < s_mild
    # HR's own dense stratum does hold its subhalo, which is what makes the
    # comparison meaningful at all even though the cut is impure.
    assert host_fraction(0.25)[1] > s_mild


def test_strata_reach_the_tile_loss_and_compose_with_classes():
    ng, tile = 64, 16
    rng = np.random.default_rng(5)
    hr = torch.tensor(rng.uniform(0.0, BOX, size=(ng ** 3, 3)))
    cand = hr + torch.tensor(rng.normal(0.0, 0.4, size=hr.shape))
    rows = torch.arange(hr.shape[0])
    cfg = KnnConfig(ks=(2, 4, 8), n_queries=512, n_quantiles=4, strata=4,
                    strata_charge=(0,), buffer_cells=0.0, dead_zone=0.0)
    loss, diag = knn_tile_loss(cand, hr, rows, 0, cfg, ng_hr=ng, tile_hr=ng,
                               boxsize_mpc_h=BOX)
    assert not diag["skipped"]
    assert diag["n_strata_charged"] == 1
    assert float(loss) >= 0.0


def test_a_tile_too_small_for_strata_times_bins_is_skipped():
    """The interior floor is per STRATUM, not per tile: 4 strata of 16 bins need
    64 draws, and charging 16 bins on 8 rows would silently compare quantiles
    that are not the same quantiles."""
    ng = 16
    rng = np.random.default_rng(6)
    hr = torch.tensor(rng.uniform(0.0, BOX, size=(ng ** 3, 3)))
    rows = torch.arange(hr.shape[0])
    cfg = KnnConfig(ks=(2, 4, 8), n_queries=40, n_quantiles=16, strata=4,
                    buffer_cells=0.0)
    _, diag = knn_tile_loss(hr + 0.01, hr, rows, 0, cfg, ng_hr=ng, tile_hr=ng,
                            boxsize_mpc_h=BOX)
    assert diag["skipped"]


def test_strata_config_refuses_nonsense():
    with pytest.raises(ValueError, match="not in ks"):
        KnnConfig(ks=(2, 4, 8), strata=4, strata_rank=16)
    with pytest.raises(ValueError, match="needs strata > 1"):
        KnnConfig(ks=(2, 4, 8), strata_charge=(0,))
    with pytest.raises(ValueError, match="outside"):
        KnnConfig(ks=(2, 4, 8), strata=4, strata_charge=(4,))
    with pytest.raises(ValueError, match="repeats"):
        KnnConfig(ks=(2, 4, 8), strata=4, strata_charge=(0, 0))
    # strata_rank 0 means min(ks) and is always legal.
    KnnConfig(ks=(2, 4, 8), strata=4, strata_rank=0)
