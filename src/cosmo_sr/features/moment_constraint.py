"""Per-host affine-moment projection of the substructure module's added field.

This is the operator that replaces the fixed spectral high-pass of skeleton
item 4 in ``docs/sr2_substructure_module.md``. It is specified in
``docs/sr2_moment_constraint.md``; read that first. In one sentence: the module's
added residual field ``d(q)`` is split, inside each host's Lagrangian footprint,
into its best-fit affine part ``T + M @ xi`` (the twelve degrees of freedom that
translate, resize, rotate and shear the host) plus an anharmonic remainder that
carries substructure, and the affine part is projected out.

The projector ``Pi`` is linear and idempotent. Flow matching runs inside its
range: project the target residual, project the learned velocity field each ODE
step, and no host is moved at any integration time -- the guarantee becomes a
property of the parameterization rather than a soft loss term.

Nothing here is materialised densely. A cluster footprint is ~10^5 HR sites and
its projector ``P_h = Phi (Phi^T Phi)^+ Phi^T`` would be ``|Omega_h|**2``; only
the per-host site list and the ``4x4`` normal-equation inverse are stored, and
``(I - P_h) d`` is a gather, a ``4x4`` solve and a scatter -- ``O(|Omega_h|)``.

**Whole-box, at two points only (docs section 5.1).** The projection is defined
per host over its *full* footprint, which for 88.7% of hosts spans more than one
SR2 tile, so it is NOT applied inside the per-tile training step (that would fit
each host's affine part on a partial, ill-conditioned footprint). Since the
guarantee constrains only the emitted ``d`` -- the sole thing added to
``Psi_SR2`` -- ``Pi`` enters at exactly two whole-box points, both of which have
the whole box in hand:

1. target precompute (once per box): cache ``x_1 = Pi(Psi_HR - Psi_SR2)`` and
   sample tiles from the already-projected target during training;
2. inference (once per sample): assemble the emitted tiles into a whole-box ``d``
   and ``Psi_final = Psi_SR2 + projector.apply(d)``.

Training on projected targets teaches the network to stay in range(``Pi``); the
single final projection makes it exact. The per-tile loop and the ODE trajectory
are never projected -- unnecessary for the endpoint guarantee, and doing so would
reintroduce the partial-footprint problem for no benefit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from cosmo_sr.features.lagrangian_host import LagrangianGrid

__all__ = [
    "HostAffineBlock",
    "MomentProjector",
    "build_projector",
    "from_features",
]

# Basis column counts per mode. "affine" removes translation + the full
# deformation gradient (recommended, docs section 3); "translation" is the
# degenerate fallback that only pins the centre of mass (docs section 8.1).
_MODE_COLUMNS = {"affine": 4, "translation": 1}


def _periodic_delta(pos: np.ndarray, centre: np.ndarray, box: float) -> np.ndarray:
    """``pos - centre`` wrapped into ``[-box/2, box/2)`` per axis."""
    d = np.asarray(pos, dtype=np.float64) - np.asarray(centre, dtype=np.float64)
    return d - box * np.round(d / box)


def _hr_positions(flat_ids: np.ndarray, ng_hr: int, box: float) -> np.ndarray:
    """Cell-centre HR Lagrangian positions of C-order flat ids, ``(n, 3)`` Mpc/h.

    Mirrors ``lagrangian_lattice_positions`` at HR resolution but only for the
    handful of sites in one footprint, so the dense ``ng_hr**3`` array is never
    built.
    """
    ids = np.asarray(flat_ids, dtype=np.int64)
    a = ids // (ng_hr * ng_hr)
    b = (ids // ng_hr) % ng_hr
    c = ids % ng_hr
    return (np.stack([a, b, c], axis=1).astype(np.float64) + 0.5) * (box / ng_hr)


@dataclass(frozen=True)
class HostAffineBlock:
    """The projection data for one host: which HR sites, and how to project them.

    ``sites`` are C-order flat HR particle ids (``pid = (a*ng+b)*ng+c``), the same
    index ``field_to_particles`` uses. ``phi`` is the affine design matrix over
    those sites and ``ginv = pinv(phi^T phi)`` gives the orthogonal projector onto
    its column space even when the footprint is rank-deficient (a collinear or
    single-site host) -- ``P = phi @ ginv @ phi^T`` is the orthogonal projector
    for ``pinv`` regardless of rank.
    """

    row: int                 # row into the host table (metadata)
    sites: np.ndarray        # (n,) int64, C-order flat HR ids in Omega_h
    phi: np.ndarray          # (n, k) design matrix [1, xi_x, xi_y, xi_z][:, :k]
    ginv: np.ndarray         # (k, k) = pinv(phi^T phi)

    @property
    def n_sites(self) -> int:
        return int(self.sites.shape[0])

    def remove_affine(self, block_field: np.ndarray) -> np.ndarray:
        """``(I - P_h)`` applied to a ``(n, m)`` field over this footprint.

        ``m`` is the number of scalar components projected together (3 for a
        displacement, 6 for displacement+velocity): each column is an independent
        scalar field over the footprint and shares the one affine basis.
        """
        beta = self.ginv @ (self.phi.T @ block_field)      # (k, m)
        return block_field - self.phi @ beta               # (n, m)


class MomentProjector:
    """The whole-field projector ``Pi``: block-diagonal over disjoint footprints.

    Bound HR sites partition by top-level host (``host_index`` is remapped to
    roots, ``docs/lagrangian_host_features.md``), so the per-host blocks are
    disjoint and ``Pi`` is a genuine projector, ``Pi**2 == Pi``. Sites in no
    footprint -- the field and void, ~69% of the box -- are left untouched.
    """

    def __init__(self, grid: LagrangianGrid, blocks: Sequence[HostAffineBlock],
                 mode: str = "affine") -> None:
        self.grid = grid
        self.blocks: List[HostAffineBlock] = list(blocks)
        self.mode = mode
        self.n_hr = grid.ng_hr ** 3
        self._check_disjoint()

    # -- construction -----------------------------------------------------
    def _check_disjoint(self) -> None:
        seen = np.zeros(self.n_hr, dtype=bool)
        for blk in self.blocks:
            if seen[blk.sites].any():
                raise ValueError(
                    "host footprints overlap; host_index must be top-level "
                    "(remap_to_roots) so bound sites partition")
            seen[blk.sites] = True

    # -- application ------------------------------------------------------
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Project a residual field into range(Pi). Shape-preserving.

        Accepts ``(C, ng, ng, ng)`` or flat ``(C, n_hr)`` with ``C`` a multiple
        of 3; every consecutive 3-channel block (a displacement, a velocity) is
        projected independently against the same per-host bases. The input is not
        mutated.
        """
        arr = np.asarray(field)
        squeeze_shape = arr.shape
        flat = arr.reshape(arr.shape[0], -1)
        if flat.shape[1] != self.n_hr:
            raise ValueError(
                f"field has {flat.shape[1]} sites, grid has {self.n_hr}")
        if flat.shape[0] % 3:
            raise ValueError(f"channel count {flat.shape[0]} is not a multiple of 3")

        out = flat.astype(np.float64, copy=True)
        n_groups = flat.shape[0] // 3
        for blk in self.blocks:
            g = out[:, blk.sites]                       # (C, n)
            for gi in range(n_groups):
                sl = slice(3 * gi, 3 * gi + 3)
                # (n, 3): the three spatial components share one affine basis
                out[sl, blk.sites] = blk.remove_affine(g[sl].T).T
        return out.astype(arr.dtype, copy=False).reshape(squeeze_shape)

    # -- diagnostics ------------------------------------------------------
    def footprint_mask(self) -> np.ndarray:
        """``(n_hr,)`` bool: sites inside some host footprint (constrained)."""
        mask = np.zeros(self.n_hr, dtype=bool)
        for blk in self.blocks:
            mask[blk.sites] = True
        return mask

    def affine_moments(self, field: np.ndarray, row: int) -> np.ndarray:
        """``phi^T d`` for one host -- zero after :meth:`apply`. For tests."""
        flat = np.asarray(field).reshape(np.asarray(field).shape[0], -1)
        blk = next(b for b in self.blocks if b.row == row)
        return blk.phi.T @ flat[:, blk.sites].T          # (k, C)


def from_features(feat, mode: str = "affine") -> MomentProjector:
    """Build ``Pi`` straight from a :class:`LagrangianHostFeatures`.

    The one-call seam for the step-5 precompute: read
    ``<box>_lagrangian_host.npz`` with ``LagrangianHostFeatures.from_npz`` and
    pass the result here. Uses ``host_index`` for the footprints and the host
    table's ``center_lag`` / ``r_lag_mpc_h`` for the geometry.
    """
    return build_projector(
        feat.grid, feat.host_index,
        feat.table.center_lag, feat.table.r_lag_mpc_h, mode=mode)


def build_projector(
    grid: LagrangianGrid,
    host_index_lr: np.ndarray,
    centres_mpc_h: np.ndarray,
    r_lag_mpc_h: np.ndarray,
    mode: str = "affine",
    rows: Optional[Sequence[int]] = None,
) -> MomentProjector:
    """Build ``Pi`` from the LR host map and per-host geometry.

    ``host_index_lr`` is ``(ng_lr,)*3`` (or flat) with ``row`` into the host
    table and ``-1`` for unbound sites -- i.e. ``LagrangianHostFeatures.
    host_index``. ``centres_mpc_h`` is ``(H, 3)`` periodic Lagrangian centres and
    ``r_lag_mpc_h`` ``(H,)`` Lagrangian radii -- ``HostTable.center_lag`` and
    ``.r_lag_mpc_h``. The centred coordinate is ``xi = (q_hr - centre) / R_L``,
    recomputed at **HR** rather than broadcast from ``dq_over_rl`` so the gradient
    basis is not collapsed (docs section 6).
    """
    if mode not in _MODE_COLUMNS:
        raise ValueError(f"mode {mode!r} not in {sorted(_MODE_COLUMNS)}")
    k = _MODE_COLUMNS[mode]

    host_index = np.asarray(host_index_lr).reshape(-1)
    centres = np.asarray(centres_mpc_h, dtype=np.float64)
    radii = np.asarray(r_lag_mpc_h, dtype=np.float64)
    box = float(grid.boxsize_mpc_h)
    present = np.unique(host_index[host_index >= 0]) if rows is None \
        else np.asarray(sorted(set(int(r) for r in rows)), dtype=np.int64)

    blocks: List[HostAffineBlock] = []
    for row in present:
        lr_sites = np.flatnonzero(host_index == int(row)).astype(np.int64)
        if lr_sites.size == 0:
            continue
        # HR children of every bound LR site: Omega_h at HR (docs section 6).
        hr_sites = np.concatenate([grid.hr_children(int(s)) for s in lr_sites])
        pos = _hr_positions(hr_sites, grid.ng_hr, box)
        xi = _periodic_delta(pos, centres[int(row)], box) / float(radii[int(row)])

        phi = np.empty((hr_sites.size, 4), dtype=np.float64)
        phi[:, 0] = 1.0
        phi[:, 1:] = xi
        phi = phi[:, :k]
        ginv = np.linalg.pinv(phi.T @ phi)
        blocks.append(HostAffineBlock(row=int(row), sites=hr_sites,
                                      phi=phi, ginv=ginv))
    return MomentProjector(grid, blocks, mode=mode)
