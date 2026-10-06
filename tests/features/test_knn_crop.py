"""The claim this module rests on is invariance, so that is what gets measured.

:mod:`cosmo_sr.features.knn_crop` exists because a kNN log-distance distribution
is invariant under map2map's ``--augment``, which is what lets the term run with
no tile id and no catalog. The argument is short enough to be convincing and
wrong: these tests drive map2map's OWN ``flip`` and ``perm`` rather than a local
reimplementation of them, because a reimplementation would be testing the
paragraph in the docstring instead of the code that will actually run.
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cosmo_sr.features.knn_cdf import KnnConfig
from cosmo_sr.features.knn_crop import (
    CELL_MPC_H,
    DIS_SCALE_MPC_H,
    OPEN_BOX_MPC_H,
    VEL_SCALE_KMS,
    crop_phase,
    crop_positions,
    crop_query_rows,
    knn_crop_loss,
)

M2M_REPO = Path("/zfsauton2/home/yixiz/DMSR/map2map_styled")

N = 16          # 4096 particles: enough for ks=(4, 8) with a self-match margin
CFG = KnnConfig(ks=(4, 8), n_queries=128, n_quantiles=8, buffer_cells=2.0)


def _field(seed: int, *, clump: float = 0.0) -> torch.Tensor:
    """A ``(6, N, N, N)`` catnorm crop. ``clump`` pulls disp toward the centre."""
    g = torch.Generator().manual_seed(seed)
    f = 0.01 * torch.randn(6, N, N, N, generator=g, dtype=torch.float64)
    if clump:
        ax = (torch.arange(N, dtype=torch.float64) + 0.5) * CELL_MPC_H
        mid = 0.5 * N * CELL_MPC_H
        for c in range(3):
            shape = [1, 1, 1]
            shape[c] = N
            # Toward the centre, in catnorm units.
            f[c] -= clump * ((ax.reshape(shape).expand(N, N, N) - mid)
                             / DIS_SCALE_MPC_H)
    return f


# --------------------------------------------------------------------------- #
# positions
# --------------------------------------------------------------------------- #
def test_undisplaced_crop_is_exactly_the_lattice():
    pos = crop_positions(torch.zeros(6, N, N, N, dtype=torch.float64))
    assert pos.shape == (N ** 3, 3)
    # Row (i * N + j) * N + k is site (i, j, k) -- the ordering every owner
    # array and knn_cdf.box_positions uses. A transposed build would still pass
    # a shape check and would silently pair the wrong particles.
    for (i, j, k) in ((0, 0, 0), (0, 0, 1), (1, 0, 0), (3, 5, 7), (N - 1,) * 3):
        row = (i * N + j) * N + k
        want = torch.tensor([(i + 0.5), (j + 0.5), (k + 0.5)],
                            dtype=torch.float64) * CELL_MPC_H
        assert torch.allclose(pos[row], want)


def test_displacement_enters_in_mpc_and_carries_the_gradient():
    f = torch.zeros(6, N, N, N, dtype=torch.float64, requires_grad=True)
    pos = crop_positions(f)
    pos.sum().backward()
    # d pos / d disp is the Mpc/h scale on the three displacement channels...
    assert torch.allclose(f.grad[:3], torch.full_like(f.grad[:3],
                                                      DIS_SCALE_MPC_H))
    # ...and exactly zero on the velocity channels, which positions do not use.
    assert torch.count_nonzero(f.grad[3:]) == 0


def test_no_wrapping_across_the_crop_faces():
    """A sub-cube is not periodic: the faces must stay far apart.

    If the builder applied a modulo, a particle just outside one face would
    reappear at the other and invent a close pair -- which is the one error this
    statistic cannot tolerate, since it moves r_k in the same direction as the
    clumping signal.
    """
    f = torch.zeros(6, N, N, N, dtype=torch.float64)
    pos = crop_positions(f)
    lo = pos[(0 * N + 0) * N + 0]
    hi = pos[((N - 1) * N + (N - 1)) * N + (N - 1)]
    span = float((hi - lo).norm())
    assert span == pytest.approx(math.sqrt(3) * (N - 1) * CELL_MPC_H)

    # And a displacement that carries a particle past a face keeps going.
    f[0, 0, 0, 0] = -1.0                       # -6 Mpc/h, far outside the crop
    assert float(crop_positions(f)[0, 0]) == pytest.approx(
        0.5 * CELL_MPC_H - DIS_SCALE_MPC_H)


def test_open_box_makes_minimum_image_the_identity():
    from cosmo_sr.features.knn_cdf import _minimum_image
    d = torch.tensor([[-12.5, 0.3, 7.0]], dtype=torch.float64)
    assert torch.allclose(_minimum_image(d, OPEN_BOX_MPC_H), d)


# --------------------------------------------------------------------------- #
# phase coordinates
# --------------------------------------------------------------------------- #
def test_phase_lambda_zero_is_position_only():
    f = _field(0)
    assert crop_phase(f, phase_lambda=0.0).shape == (N ** 3, 3)
    assert crop_phase(f, phase_lambda=4000.0).shape == (N ** 3, 6)


def test_velocity_columns_are_km_per_s_over_lambda():
    f = torch.zeros(6, N, N, N, dtype=torch.float64)
    f[3] = 1.0                                  # one catnorm velocity unit
    p = crop_phase(f, phase_lambda=4000.0)
    assert float(p[0, 3]) == pytest.approx(VEL_SCALE_KMS / 4000.0)
    assert float(p[0, 4]) == 0.0


# --------------------------------------------------------------------------- #
# the invariance the module rests on
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def m2m_aug():
    """map2map's own ``flip``/``perm``, or skip. Never a reimplementation."""
    if not (M2M_REPO / "map2map" / "data" / "fields.py").is_file():
        pytest.skip(f"no map2map checkout at {M2M_REPO}")
    sys.path.insert(0, str(M2M_REPO))
    from map2map.data.fields import flip, perm
    return flip, perm


def _augment_like_map2map(field, flip_fn, perm_fn, *, flip_axes, perm_axes):
    """Exactly what ``FieldDataset.__getitem__`` does to a sample.

    The split into two 3-channel fields is not cosmetic. Both helpers touch
    vector components only when ``x.shape[0] == ndim``, so a 6-channel tensor
    would have its voxels moved and its components left pointing the old way --
    which is precisely why ``m2m_split_data.py`` writes disp.npy and vel.npy
    separately and ``train.py`` concatenates them after augmentation.
    """
    fields = [field[:3].clone(), field[3:].clone()]
    flip_fn(fields, flip_axes, 3)
    perm_fn(fields, perm_axes, 3)
    return torch.cat(fields, dim=0)


@pytest.mark.parametrize("flip_axes,perm_axes", [
    ([0], [0, 1, 2]),                 # a single reflection
    ([0, 2], [0, 1, 2]),              # two reflections
    ([], [2, 0, 1]),                  # a pure cyclic permutation
    ([], [1, 0, 2]),                  # a pure transposition (odd)
    ([1], [2, 1, 0]),                 # both at once
])
def test_loss_is_invariant_under_flip_and_perm(m2m_aug, flip_axes, perm_axes):
    """The whole design claim, measured against map2map's own augmentation.

    flip maps ``q_c -> L - q_c`` and ``d_c -> -d_c``, so ``p_c -> L - p_c``; perm
    permutes coordinates. Both are isometries, the velocity block gets the same
    treatment, and so the 6-D phase metric -- not just the 3-D one -- is
    preserved. A kNN log-distance distribution is a function of that metric
    alone, so the loss must not move at all.
    """
    flip_fn, perm_fn = m2m_aug
    fa = torch.tensor(flip_axes, dtype=torch.long)
    pa = torch.tensor(perm_axes, dtype=torch.long)

    cand, tgt = _field(1, clump=0.15), _field(2)
    cand_a = _augment_like_map2map(cand, flip_fn, perm_fn,
                                   flip_axes=fa, perm_axes=pa)
    tgt_a = _augment_like_map2map(tgt, flip_fn, perm_fn,
                                  flip_axes=fa, perm_axes=pa)

    # Same generator seed both sides: the query DRAW is random, and comparing
    # two different draws would measure sampling noise instead of invariance.
    # n_queries above the interior count (12^3 = 1728) makes interior_queries
    # return ALL of it and skip the randperm entirely. That is what makes this
    # exact: a SUBSAMPLE would draw different particles either side of the
    # isometry, because randperm picks positions in the `inside` array and the
    # isometry relabels those rows -- measuring sampling noise, not invariance.
    exact = replace(CFG, n_queries=N ** 3)

    def run(c, t):
        return float(knn_crop_loss(c, t, exact)[0])

    plain, aug = run(cand, tgt), run(cand_a, tgt_a)
    assert plain > 0.0, "a clumped candidate must score above zero to be a test"
    assert aug == pytest.approx(plain, rel=1e-9, abs=1e-12)


def test_invariance_is_not_vacuous_because_augmentation_moves_the_field(m2m_aug):
    """Guard the guard: if augmentation were a no-op the test above is empty."""
    flip_fn, perm_fn = m2m_aug
    f = _field(3)
    a = _augment_like_map2map(f, flip_fn, perm_fn,
                              flip_axes=torch.tensor([0, 1]),
                              perm_axes=torch.tensor([2, 0, 1]))
    assert not torch.allclose(f, a)
    # And the positions really are a different point set, in the same order.
    assert not torch.allclose(crop_positions(f), crop_positions(a))


# --------------------------------------------------------------------------- #
# the loss
# --------------------------------------------------------------------------- #
def test_matched_fields_score_far_below_a_clumped_one():
    g = torch.Generator().manual_seed(11)
    same = float(knn_crop_loss(_field(4), _field(4), CFG, generator=g)[0])
    g = torch.Generator().manual_seed(11)
    clumped = float(knn_crop_loss(_field(4, clump=0.15), _field(4), CFG,
                                  generator=g)[0])
    assert same == pytest.approx(0.0, abs=1e-3)
    assert clumped > 20 * max(same, 1e-6)


def test_gradient_reaches_the_candidate_displacement():
    cand = _field(5, clump=0.1).requires_grad_(True)
    g = torch.Generator().manual_seed(13)
    loss, diag = knn_crop_loss(cand, _field(6), CFG, generator=g)
    loss.backward()
    assert diag["n_crops"] == 1
    assert torch.isfinite(cand.grad).all()
    assert torch.count_nonzero(cand.grad[:3]) > 0


def test_target_is_treated_as_a_constant():
    tgt = _field(7).requires_grad_(True)
    g = torch.Generator().manual_seed(17)
    cand = _field(8, clump=0.1).requires_grad_(True)
    knn_crop_loss(cand, tgt, CFG, generator=g)[0].backward()
    assert tgt.grad is None or torch.count_nonzero(tgt.grad) == 0


def test_batched_input_averages_over_crops():
    a, b = _field(9, clump=0.2), _field(10)
    batch_c = torch.stack([a, _field(11, clump=0.05)])
    batch_t = torch.stack([b, _field(12)])
    g = torch.Generator().manual_seed(19)
    loss, diag = knn_crop_loss(batch_c, batch_t, CFG, generator=g)
    assert diag["n_crops"] == 2
    assert torch.isfinite(loss)


def test_buffer_that_swallows_the_crop_returns_a_connected_zero():
    """Not an error, and the zero must still be differentiable.

    A bare ``torch.zeros(())`` would make the caller's ``.backward()`` fail with
    "does not require grad", turning a legitimately empty interior into a crash
    partway through a run.
    """
    cfg = KnnConfig(ks=(4, 8), n_queries=32, n_quantiles=4,
                    buffer_cells=float(N))          # half - buffer*cell < 0
    cand = _field(13).requires_grad_(True)
    loss, diag = knn_crop_loss(cand, _field(14), cfg)
    assert diag["n_crops"] == 0
    assert float(loss) == 0.0
    loss.backward()
    assert cand.grad is not None


def test_query_rows_stay_inside_the_buffered_interior():
    pos = crop_positions(torch.zeros(6, N, N, N, dtype=torch.float64))
    g = torch.Generator().manual_seed(23)
    rows = crop_query_rows(pos, CFG, n_cells=N, generator=g)
    assert rows.numel() > 0
    half = 0.5 * N * CELL_MPC_H - CFG.buffer_cells * CELL_MPC_H
    d = (pos[rows] - 0.5 * N * CELL_MPC_H).abs()
    assert float(d.max()) <= half + 1e-9


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        knn_crop_loss(_field(15), _field(15)[:, :N - 1], CFG)


def test_non_cubic_crop_is_rejected():
    with pytest.raises(ValueError, match="cubic"):
        crop_positions(torch.zeros(6, N, N, N - 1, dtype=torch.float64))
