"""The kNN-CDF term on a map2map training crop, with no catalog and no tile id.

:mod:`knn_cdf` was written for the fine-tune pipeline, where every statistic is
indexed by ``(box, tile)`` against a precomputed Rockstar catalog. Training SR2
from scratch with map2map's own loop offers neither of those:

* ``FieldDataset.__getitem__`` computes ``ifile, icrop = divmod(idx, ncrop)`` and
  returns ``target_relpath`` -- the BOX -- but never ``icrop``. The tile id is a
  local variable. ``train.py`` constructs ``FieldDataset`` directly rather than
  through ``import_attr``, so the class cannot be swapped via ``--callback-at``
  either: there is no hook that would carry a tile id into the loss.
* ``--augment`` applies a random flip and a random axis permutation per sample,
  and ``flip_axes``/``perm_axes`` are local too. So even a table keyed by
  ``(box, icrop)`` could not be applied: it would be in the original frame and
  the network's output is in the augmented one.

Both obstacles dissolve for THIS statistic, because of what it measures.
``map2map.data.fields.perm`` permutes the vector COMPONENTS (``x[axes]`` when
``x.shape[0] == ndim``) together with the spatial axes, and ``flip`` negates the
components it flips. Both are therefore proper isometries of the displacement
field: the point set is rotated or reflected, and every pairwise distance is
preserved. A kNN log-distance distribution is a function of pairwise distances
alone, so it is invariant under map2map's augmentation exactly, not
approximately. ``test_loss_is_invariant_under_flip_and_perm`` measures that
rather than trusting this paragraph.

That makes the term computable from ``(output, target)`` alone, in the batch, with
no tile id, no catalog and no constraint on ``--crop``/``--crop-step``/
``--aug-shift``. It also retires the one geometry deviation the from-scratch
baseline was carrying: crop 8 / step 8 with no ``--aug-shift`` was chosen so
these terms would stay indexable by ``(box, tile)``, and they do not need to be.

OPEN BOUNDARIES. A crop is a sub-cube, not a periodic box. :func:`knn_cdf.
_minimum_image` is ``delta - L * round(delta / L)``, so passing an ``L`` far
larger than any separation in the crop makes it the identity -- open boundaries
with no second code path to keep in step with the periodic one. That is what
:data:`OPEN_BOX_MPC_H` is for.

The sub-cube is also why ``cfg.buffer_cells`` is not optional here. A query near
a crop face has true neighbours in the adjacent crop, which is not in the
candidate set, so its ``r_k`` is OVERestimated -- the same sign as the clumping
deficit the term charges for, i.e. an unbuffered query set manufactures its own
signal. :func:`knn_cdf.interior_queries` already makes that cut; this module
only hands it the crop's own centre instead of a tile's.

PHASE SPACE IS THE DEFAULT, and the reason is measured, not stylistic. Round 5b
(jobs 52090-52094) found the POOLED position-only statistic ranks arms at
rho +0.91 against region subhalo counts -- the wrong sign -- because it is
dominated by host and unbound particles. Appending ``v / phase_lambda`` at
4000 km/s per Mpc/h flips it to rho -0.65 (p 0.002), and that form needs no
labels, which is the only kind of fix available on an unlabelled crop. So
:func:`crop_phase` is the coordinate builder and ``phase_lambda`` defaults to
the measured value in :func:`knn_crop_loss`, NOT to ``KnnConfig``'s 0.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Tuple

import torch

from .knn_cdf import (
    KnnConfig,
    interior_queries,
    knn_cdf_loss,
    knn_log_distances,
)

#: Fewer interior queries than this and the crop is skipped: a "distribution"
#: over a handful of draws is noise, and quantile bins of it are meaningless.
MIN_QUERIES: int = 16

#: Mpc/h. Stands in for "no periodic wrapping": ``_minimum_image`` reduces to the
#: identity once ``L`` exceeds every separation in the crop, because
#: ``round(delta / L)`` is then 0. A crop is at most ~12.5 Mpc/h across, so 1e6
#: is six orders of margin and still far from float32's precision limit on the
#: division.
OPEN_BOX_MPC_H: float = 1.0e6

#: Mpc/h per unit of a catnorm displacement channel. ``disnorm`` divides by
#: ``6000 * D(z)`` kpc/h and ``D(0) = 1``, so this is exact at z=0 and is the
#: same 6.0 that :func:`knn_cdf.box_positions` takes.
DIS_SCALE_MPC_H: float = 6.0

#: km/s per unit of a catnorm velocity channel: ``velnorm``'s
#: ``6 D(z) H(z) f(z) / (1 + z)`` evaluated at z=0 with Om=0.31. Computed from
#: ``preprocess_srs.velnorm``, not typed from memory.
VEL_SCALE_KMS: float = 313.42210244571896

#: Mpc/h. The HR grid is 512^3 over a 100 Mpc/h box.
CELL_MPC_H: float = 100.0 / 512.0

#: km/s per Mpc/h. See the module docstring: the pass is at 4000 and it falls
#: off on both sides (2000 -> -0.55, 8000 -> -0.60).
PHASE_LAMBDA_KMS_PER_MPC_H: float = 4000.0


def crop_positions(
    field: torch.Tensor, *, cell_mpc_h: float = CELL_MPC_H,
    dis_scale_mpc_h: float = DIS_SCALE_MPC_H,
) -> torch.Tensor:
    """``(n^3, 3)`` Eulerian positions of a ``(6|3, n, n, n)`` crop, in Mpc/h.

    **Differentiable in ``field``.** The Lagrangian lattice is a constant and the
    displacement carries the whole gradient.

    Coordinates are the CROP's own frame: site ``(i, j, k)`` sits at
    ``(i + 0.5) * cell`` along each axis, so the crop spans ``[0, n * cell)``
    before displacement. There is no modulo. Wrapping a sub-cube into itself
    would teleport a particle that legitimately left the crop across to the far
    face and invent a close pair out of it.

    Row ``(i * n + j) * n + k`` is site ``(i, j, k)`` -- C order, the same
    convention as :func:`knn_cdf.box_positions` and every owner array -- so a row
    index means the same particle in both modules.
    """
    if field.dim() != 4 or field.shape[0] < 3:
        raise ValueError(f"expected (>=3, n, n, n), got {tuple(field.shape)}")
    n = int(field.shape[1])
    if not (field.shape[1] == field.shape[2] == field.shape[3]):
        raise ValueError(f"crop must be cubic, got {tuple(field.shape)}")

    ax = (torch.arange(n, device=field.device, dtype=field.dtype) + 0.5) \
        * float(cell_mpc_h)
    cols = []
    for c in range(3):
        shape = [1, 1, 1]
        shape[c] = n
        q = ax.reshape(shape).expand(n, n, n)
        cols.append((field[c] * float(dis_scale_mpc_h) + q).reshape(-1))
    return torch.stack(cols, dim=1)


def crop_phase(
    field: torch.Tensor, *, cell_mpc_h: float = CELL_MPC_H,
    dis_scale_mpc_h: float = DIS_SCALE_MPC_H,
    vel_scale_kms: float = VEL_SCALE_KMS,
    phase_lambda: float = PHASE_LAMBDA_KMS_PER_MPC_H,
) -> torch.Tensor:
    """``(n^3, 3)`` positions, or ``(n^3, 6)`` phase coordinates when asked.

    The velocity columns are ``v_kms / phase_lambda``, which puts them in
    Mpc/h-equivalent units so a single Euclidean metric spans both halves --
    :mod:`knn_cdf`'s convention, and its neighbour search already leaves columns
    3+ unwrapped because velocity is not periodic.

    ``phase_lambda <= 0`` returns positions only, which is the position-only
    statistic of rounds 1-4 and is kept reachable so the two can be compared in
    one run rather than across two code paths.
    """
    pos = crop_positions(field, cell_mpc_h=cell_mpc_h,
                         dis_scale_mpc_h=dis_scale_mpc_h)
    if float(phase_lambda) <= 0.0:
        return pos
    if field.shape[0] < 6:
        raise ValueError("phase_lambda > 0 needs velocity channels; got "
                         f"{int(field.shape[0])} channels")
    scale = float(vel_scale_kms) / float(phase_lambda)
    vel = torch.stack([field[3 + c].reshape(-1) * scale for c in range(3)],
                      dim=1)
    return torch.cat([pos, vel], dim=1)


def crop_query_rows(
    pos: torch.Tensor, cfg: KnnConfig, *, n_cells: int,
    cell_mpc_h: float = CELL_MPC_H,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Query rows from the crop's Eulerian interior. See the module docstring.

    Reuses :func:`knn_cdf.interior_queries` rather than restating the cut: the
    only crop-specific parts are the centre (the crop's own middle, not a tile's
    place in the box) and the open boundary.
    """
    side = float(n_cells) * float(cell_mpc_h)
    centre = torch.full((3,), 0.5 * side, dtype=torch.float64)
    rows = torch.arange(int(pos.shape[0]), device=pos.device)
    return interior_queries(
        pos, rows, cfg, tile_centre_mpc_h=centre, tile_side_mpc_h=side,
        cell_mpc_h=float(cell_mpc_h), boxsize_mpc_h=OPEN_BOX_MPC_H,
        generator=generator,
    )


def knn_crop_loss(
    output: torch.Tensor, target: torch.Tensor, cfg: KnnConfig, *,
    cell_mpc_h: float = CELL_MPC_H,
    dis_scale_mpc_h: float = DIS_SCALE_MPC_H,
    vel_scale_kms: float = VEL_SCALE_KMS,
    phase_lambda: float = PHASE_LAMBDA_KMS_PER_MPC_H,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """``(loss, diagnostics)`` for a batch of crops. Differentiable in ``output``.

    ``output`` and ``target`` are ``(B, C, n, n, n)`` or ``(C, n, n, n)``, the
    shapes ``train.py`` holds after ``narrow_cast``. The loss is the mean over
    the batch: each crop is an independent draw of the statistic, and averaging
    the per-crop losses keeps one crop with an unusually sparse interior from
    dominating through a longer query list.

    The reference is the TARGET of the same batch element, detached. That is what
    makes the term index-free -- there is no stored per-tile reference to get out
    of step with the augmentation, because the reference went through the same
    flip and permutation as the candidate. It is also why the reference is noisy:
    it is one crop's empirical distribution, not a box's. ``cfg.n_quantiles``
    bins absorb that, and ``cfg.dead_zone`` is what keeps the bin noise from
    becoming a gradient.
    """
    if output.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(output.shape)} vs "
                         f"{tuple(target.shape)}")
    if output.dim() == 4:
        output, target = output[None], target[None]
    if output.dim() != 5:
        raise ValueError("expected (B, C, n, n, n) or (C, n, n, n), got "
                         f"{tuple(output.shape)}")

    n = int(output.shape[2])
    losses, diags = [], []
    for b in range(int(output.shape[0])):
        p_cand = crop_phase(output[b], cell_mpc_h=cell_mpc_h,
                            dis_scale_mpc_h=dis_scale_mpc_h,
                            vel_scale_kms=vel_scale_kms,
                            phase_lambda=phase_lambda)
        p_hr = crop_phase(target[b].detach(), cell_mpc_h=cell_mpc_h,
                          dis_scale_mpc_h=dis_scale_mpc_h,
                          vel_scale_kms=vel_scale_kms,
                          phase_lambda=phase_lambda)
        # Each field draws its queries from its OWN interior. The docstring of
        # knn_cdf_loss is explicit that the two query sets need not be the same
        # particles: the match is between DISTRIBUTIONS, and sharing a query set
        # would quietly turn this back into a per-particle address loss.
        q_cand = crop_query_rows(p_cand, cfg, n_cells=n,
                                 cell_mpc_h=cell_mpc_h, generator=generator)
        q_hr = crop_query_rows(p_hr, cfg, n_cells=n,
                               cell_mpc_h=cell_mpc_h, generator=generator)
        # The two fields do NOT yield the same number of queries, and that is
        # physical rather than incidental: interior_queries cuts on EULERIAN
        # position, so a particle whose displacement carries it out of the
        # shrunken cube stops being a query. At buffer_cells=8 on a 64-cell
        # crop the interior half-width is 24 cells = 4.69 Mpc/h, while real
        # z = 0 displacements run to several Mpc/h -- so a large fraction
        # legitimately leaves, and a COLLAPSED candidate loses a different
        # fraction than HR does. MEASURED, job 68938: 33 queries against 256.
        #
        # knn_cdf_loss needs both sides reduced to the SAME bins, so match the
        # counts here. Truncating the longer draw is unbiased -- both are
        # already random subsets of their own interior.
        # Clamp the BIN count, and do NOT truncate the query lists. Truncating
        # by row order is not isometry-invariant: flip/perm relabel the rows, so
        # `q[:m]` would select different particles either side of an
        # augmentation and the loss would move -- test_loss_is_invariant_under_
        # flip_and_perm catches exactly that. _quantile_reduce already clamps to
        # the query count, so equal `n_quantiles` is all that is needed for the
        # two sides to be the same quantiles, and each field keeps its full
        # interior draw.
        m = int(min(q_cand.numel(), q_hr.numel()))
        if m < MIN_QUERIES:
            continue
        cfg_b = (cfg if m >= cfg.n_quantiles
                 else replace(cfg, n_quantiles=m))
        z_cand = knn_log_distances(p_cand, q_cand, cfg_b,
                                   boxsize_mpc_h=OPEN_BOX_MPC_H)
        with torch.no_grad():
            z_hr = knn_log_distances(p_hr, q_hr, cfg_b,
                                     boxsize_mpc_h=OPEN_BOX_MPC_H)
        loss_b, diag_b = knn_cdf_loss(z_cand, z_hr, cfg_b)
        diag_b = dict(diag_b, n_queries_used=m,
                      n_quantiles_used=int(cfg_b.n_quantiles))
        losses.append(loss_b)
        diags.append(diag_b)

    if not losses:
        # Not an error: cfg.buffer_cells can legitimately swallow a small crop.
        # Returning a zero that is CONNECTED to output keeps .backward() valid
        # for the caller, which a bare scalar would not.
        zero = output.sum() * 0.0
        return zero, {"n_crops": 0, "per_crop": []}

    loss = torch.stack(losses).mean()
    return loss, {"n_crops": len(losses), "per_crop": diags,
                  "phase_lambda": float(phase_lambda)}
