"""An auxiliary loss that makes SR2's particles gather where HR put its subhalos.

``docs/pilot_steps_2_4.md`` section 2 closed the capacity question: at 4.4
trainable parameters per target value the generator could have memorised one
cluster and did not -- MSE plateaued at 0.39x frozen while high-k power fell.
**The generator can represent this cluster's substructure; the squared loss
declines to ask for it.** A squared error over a realisation the network cannot
predict is minimised by averaging, and averaging is the deficit.

This module is the other objective. It keeps the supervision (true HR subhalos,
which exist on disk for set8/set9) but stops asking *which particle goes where*
and asks only *is there a collapsed object here*:

    for each true HR subhalo s
        C(s) = sum_cells  K(|x_cell - x_s|)  *  w_compact(cell)  *  mass(cell)
        L(s) = [ 1 - C_theta(s) / C_HR(s) ]_+^2                  # position
             + w_vdisp * log(sigma_v(s) / sigma_v,HR(s))^2       # kinematics
             + w_vbulk * |v(s) - v_HR(s)|^2 / sigma_v,HR(s)^2

``w_compact`` is exactly the coordinate ``reward/soft_structure.py`` builds --
a sigmoid on ``u = log(1 + delta)`` rather than on ``delta``, which is the form
that module's docstring records as the one where "the gradient is well
conditioned across the whole range" (a sigmoid on ``delta`` at threshold 100 put
an *empty* cell at 0.054, a pedestal five times the signal). Nothing here
introduces a new statistic; it evaluates that one in a kernel window at a known
location instead of over a whole tile.

Why velocity is in the objective and not left to the anchors
------------------------------------------------------------
Rockstar is a **phase-space** finder -- it links particles in 6-D -- and the
channel swap measured that HR loses **65% of its subhalos** when handed SR2's
velocities (``docs/sr2_substructure_module.md`` section 2, item 3). So a clump
that is spatially indistinguishable from HR's and kinematically wrong is not a
halo, and a loss that constrained only ``disp`` would build exactly that: the
particles would be moved into place while keeping the velocities they had, which
is not a self-consistent bound object. The velocity terms are two-sided rather
than hinged -- too hot is unbound, too cold is the measured sub-virial defect,
and neither direction is safe.

Why this is not MSE in disguise
-------------------------------
Three properties, each of which MSE lacks:

1. **No per-particle identity.** ``C`` is a sum over cells, invariant to which
   particle landed in which cell and to a sub-kernel displacement of the whole
   clump. Blurring strictly lowers it, so averaging is no longer the minimiser --
   it is the worst move available.
2. **Hinged.** Once the candidate matches HR's compact mass at a subhalo the
   term is exactly zero with zero gradient. Step 4's over-parameterised runs
   drove ``peak_contrast`` to 1.09-1.22x HR while broadband power fell -- at high
   capacity L2 over-sharpens what it can predict and erases the rest. A one-sided
   loss cannot ask for more contrast than HR has.
3. **Per-subhalo normalisation.** Dividing by ``C_HR(s)`` gives a 50-particle
   subhalo and a 2000-particle one the same weight, which is
   ``docs/sr2_substructure_module.md`` section 4.2's "equalize the per-subhalo
   gradient" applied to the loss rather than to the input normalisation.

Where the gradient reaches
--------------------------
``mass`` is the differentiable valid-centre CIC deposit
(:func:`cosmo_sr.eval.density.cic_density_valid_center`), so ``dC/d(disp_i)`` is
non-zero for exactly those particles whose CIC stencil touches a cell the kernel
weights -- a neighbourhood of ~``radius_factor * sigma`` fine cells around the
true centre, and no further. The loss therefore *gathers local material into the
right place*; it cannot summon material from across the tile, which is the
correct scope (subhalos are Lagrangian-pure, median one tile of origin).

**No KDTree is used, and one would be slower.** A neighbour query per subhalo per
step is ``O(S N log N)`` over 262,144 particles a tile; the CIC deposit is one
``O(N)`` pass that every particle contributes to exactly once, after which each
subhalo costs a ``(2H+1)^3`` window read. The deposit is also the estimator the
rest of this repository already scores with, so the loss and the eval cannot
disagree about where the mass went.

Inference is unaffected
-----------------------
Every HR quantity here -- the catalog, the owner array, the reference statistics
-- enters through :class:`GatherTargets`, which is built once before training and
is *data*, not part of the model. The generator's forward pass is unchanged, so a
fine-tuned checkpoint runs exactly as the frozen one does, from ``(Y, z)`` alone.

Two things this cannot say
--------------------------
* **Whether the clumps are bound.** ``C`` is a density statistic. Only Rockstar
  on a reassembled box settles boundedness (``sr2_substructure_module.md``
  section 9 step 6), and that gate is not in this module.
* **Whether it generalises.** Trained on one host's tiles the supervision is
  in-sample by construction; a held-out box, not this loss, answers that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ..eval.rockstar import HaloCatalog
from ..reward.phase_space import PhaseSpaceConfig, deposit_phase_space
from ..reward.soft_structure import SoftStructureConfig, _smooth

__all__ = [
    "GatherConfig",
    "GatherTargets",
    "TileSubhalos",
    "deposit_for_gather",
    "gather_loss",
    "outside_weight_map",
    "preserve_loss",
    "preserve_statistic",
    "gather_statistics",
    "region_coordinates",
    "subhalo_home_tiles",
    "tile_subhalos",
    "velocity_statistics",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GatherConfig:
    """Every number is a modelling choice, so every number is named.

    The threshold family (``compact_delta``, ``tau_log``, ``contrast_margin``) is
    deliberately *taken from* :class:`~cosmo_sr.reward.soft_structure.SoftStructureConfig`
    by :meth:`from_soft` rather than re-chosen here: the loss must optimise the
    same statistic the eval reports, or the run is unreadable.
    """

    #: Overdensity above which a cell counts as collapsed, through a sigmoid on
    #: ``log(1 + delta)``. At ``grid_mult = 1`` a cell is one HR lattice cell,
    #: whose uniform mass is exactly one particle, so ``delta = 100`` is "about a
    #: hundred particles in a 195 kpc/h cell".
    compact_delta: float = 100.0
    #: Sigmoid width in natural-log units of ``1 + delta``.
    tau_log: float = 0.35
    #: Margin by which a cell must beat its own smoothed neighbourhood to count
    #: as a local peak, in the same units.
    contrast_margin: float = 1.5
    #: Smoothing scale (fine cells) of that neighbourhood.
    contrast_scale: int = 2
    #: Weight of the contrast term against the compact-mass term. The contrast
    #: coordinate is the one a global amplitude shift cannot game, so it is
    #: carried at equal weight rather than as a decoration.
    w_contrast: float = 1.0
    #: Weight of the velocity-dispersion match. NOT optional and NOT hinged:
    #: Rockstar links in 6-D and the channel swap measured HR losing 65% of its
    #: subhalos when handed SR2's velocities, so a spatially perfect clump with
    #: the wrong kinematics is not a halo.
    w_vdisp: float = 1.0
    #: Weight of the bulk-velocity match, in units of the subhalo's own HR
    #: dispersion.
    w_vbulk: float = 1.0
    #: Weight of the structure-preservation term on the UNSUPERVISED field. A
    #: convolutional generator applies one learned operator everywhere, so
    #: supervising 43 windows changes what it does at every other site too, and
    #: an L2 anchor cannot defend that: minimising ||Psi - Psi_0||^2 is satisfied
    #: by broad low-amplitude change, which is precisely what erases local peaks.
    #: Measured: raising the L2 anchor 100x moved the outside peak-contrast ratio
    #: 0.570 -> 0.517, i.e. the wrong way. This term is the same hinge as the
    #: gather term, on the same statistic, over the complement of the windows.
    w_preserve: float = 1.0

    #: Kernel width: ``max(sigma_floor_cells, sigma_rvir_factor * r_vir)``. A
    #: 366-particle subhalo (the HR median inside clusters) has
    #: ``r_vir ~ 0.77`` fine cells at ``grid_mult = 1``, so the floor is what
    #: actually sets the scale for all but the largest satellites, and it sets
    #: the radius within which material can be pulled.
    sigma_floor_cells: float = 1.0
    sigma_rvir_factor: float = 1.0
    #: Window half-width as a multiple of sigma. Beyond ~2.5 sigma the Gaussian
    #: contributes under 5% and the window is only cost.
    radius_factor: float = 2.5
    #: Hard cap on the half-width, so one big satellite cannot make every
    #: window in the batch expensive.
    max_half_width: int = 6

    #: Selection. A subhalo under ``min_num_p`` particles is at or below
    #: Rockstar's own resolution limit; one whose Lagrangian sites are spread
    #: over several tiles cannot be built by this tile's particles.
    min_num_p: int = 50
    min_purity: float = 0.5
    #: Reject a target whose HR reference statistic is below this many
    #: particles: the ratio's denominator would be noise.
    min_hr_compact: float = 5.0
    #: Keep at most this many subhalos per tile (most massive first). Bounds the
    #: window tensor, which is ``S * (2H+1)^3`` per tile.
    max_per_tile: int = 512

    @staticmethod
    def from_soft(soft: SoftStructureConfig, **overrides) -> "GatherConfig":
        """Inherit the thresholds from the scored soft-structure config."""
        base = dict(
            compact_delta=float(soft.compact_delta),
            tau_log=float(soft.tau_log),
            contrast_margin=float(soft.contrast_margin),
        )
        base.update(overrides)
        return GatherConfig(**base)

    def to_dict(self) -> Dict:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items()}

    def sigma_of(self, rvir_cells: np.ndarray) -> np.ndarray:
        return np.maximum(float(self.sigma_floor_cells),
                          float(self.sigma_rvir_factor) * np.asarray(rvir_cells))

    def half_width_of(self, sigma: np.ndarray) -> int:
        if np.size(sigma) == 0:
            return 1
        h = int(np.ceil(float(self.radius_factor) * float(np.max(sigma))))
        return int(max(1, min(int(self.max_half_width), h)))


# --------------------------------------------------------------------------- #
# Which subhalos belong to which tile
# --------------------------------------------------------------------------- #
def _empty_home(return_occupancy: bool) -> Dict[str, np.ndarray]:
    z = np.zeros(0, dtype=np.int64)
    out = {"halo_id": z, "row": z, "tile": z,
           "purity": np.zeros(0), "n_sites": z}
    if return_occupancy:
        out.update({"occ_row": z, "occ_tile": z, "occ_count": z})
    return out


def subhalo_home_tiles(
    cat: HaloCatalog, index, *, ng_hr: int = 512, tile_hr: int = 64,
    min_num_p: int = 50, return_occupancy: bool = False,
    top_level: bool = False,
) -> Dict[str, np.ndarray]:
    """``(halo_id, row, tile, purity, n_sites)`` for every resolved HR subhalo.

    The *home* tile is the tile holding the plurality of a subhalo's own bound
    Lagrangian sites, and ``purity`` is that plurality's share. Subhalos are
    Lagrangian-pure -- median one tile of origin -- so for almost all of them
    purity is ~1 and the notion is not a compromise; the few that straddle a tile
    face are dropped by ``min_purity`` at :func:`tile_subhalos`, because this
    tile's particles genuinely cannot build them.

    ``top_level`` flips the population from subhalos (``parent_ids >= 0``, the
    default and every prior caller's meaning) to **host halos** (``parent_ids <
    0``). The home-tile plurality and purity are computed identically; only which
    catalog rows enter changes. It is what lets the member-gather fine-tune add a
    *preservation* constraint on the resolved hosts a run would otherwise be free
    to fragment (``docs/sr2_member_gather_training.md``): a host's own particle
    set (Rockstar assigns each particle to its deepest halo, so this is the
    host's smooth component, not its subhalos' material) is a member set exactly
    as a subhalo's is, and the same loss keeps it bound. Hosts are less
    Lagrangian-pure than subhalos, so ``min_purity`` and the live-fraction cut at
    :func:`member_gather.build_member_sets` drop the spread-out ones -- which is
    correct: this tiling's particles genuinely cannot rebuild a host whose mass
    is mostly elsewhere.

    Computed once for a whole box in a single grouped pass rather than per tile:
    a box holds ~100k subhalos and a per-tile loop over all of them would repeat
    the same work 512 times.

    ``return_occupancy`` additionally returns the full sparse occupancy the
    plurality is taken from -- ``(occ_row, occ_tile, occ_count)``, one entry per
    (subhalo, tile) pair it has sites in, with ``occ_row`` indexing the arrays
    above. It is what a *live fraction* against an arbitrary trained-tile set is
    computed from, at no extra pass: ``sum(occ_count[occ_tile in tiles]) /
    n_sites`` is exactly the fraction :func:`member_gather.build_member_sets`
    keeps. Asking for it here rather than regrouping elsewhere keeps one
    definition of which tile a subhalo's material is in.
    """
    parent = (cat.parent_ids < 0) if top_level else (cat.parent_ids >= 0)
    rows = np.flatnonzero(parent & (cat.num_p >= int(min_num_p)))
    if rows.size == 0:
        return _empty_home(return_occupancy)

    n_side = int(ng_hr) // int(tile_hr)
    n_tiles = n_side ** 3

    # One flat member list with a parallel "which subhalo" label, so the whole
    # box is grouped in two sorts rather than in a python loop per halo.
    members: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for j, row in enumerate(rows):
        pid = index.members(int(cat.ids[row]))
        if pid.size == 0:
            continue
        members.append(pid.astype(np.int64))
        labels.append(np.full(pid.size, j, dtype=np.int64))
    if not members:
        return _empty_home(return_occupancy)
    pid = np.concatenate(members)
    lab = np.concatenate(labels)

    ng = int(ng_hr)
    ix = (pid // (ng * ng)) // int(tile_hr)
    iy = ((pid // ng) % ng) // int(tile_hr)
    iz = (pid % ng) // int(tile_hr)
    tid = (ix * n_side + iy) * n_side + iz

    key = lab * n_tiles + tid
    uk, counts = np.unique(key, return_counts=True)
    ulab, utile = uk // n_tiles, uk % n_tiles
    # Sort by (subhalo, -count) so the first row of each group is its plurality
    # tile; np.unique's return_index on the sorted labels then picks it out.
    order = np.lexsort((-counts, ulab))
    ulab, utile, counts = ulab[order], utile[order], counts[order]
    first = np.unique(ulab, return_index=True)[1]
    tot = np.bincount(lab, minlength=int(rows.size))

    best_lab = ulab[first]
    out = {
        "halo_id": cat.ids[rows[best_lab]].astype(np.int64),
        "row": rows[best_lab].astype(np.int64),
        "tile": utile[first].astype(np.int64),
        "purity": counts[first] / np.maximum(tot[best_lab], 1),
        "n_sites": tot[best_lab].astype(np.int64),
    }
    if return_occupancy:
        # `best_lab` is np.unique's output, so it is sorted ascending and
        # searchsorted maps every occupancy entry's label onto its row above.
        out["occ_row"] = np.searchsorted(best_lab, ulab).astype(np.int64)
        out["occ_tile"] = utile.astype(np.int64)
        out["occ_count"] = counts.astype(np.int64)
    return out


# --------------------------------------------------------------------------- #
# Where a subhalo sits in the deposit grid
# --------------------------------------------------------------------------- #
def region_coordinates(
    pos_mpc_h: np.ndarray, tile_id: int, bulk: np.ndarray, *,
    ng_hr: int = 512, tile_hr: int = 64, boxsize_mpc_h: float = 100.0,
    region: int = 32, grid_mult: int = 1,
) -> np.ndarray:
    """``(S, 3)`` positions in the tile's valid-centre deposit grid.

    The deposit's own convention, from
    :func:`cosmo_sr.eval.density._valid_center_corners`: a particle at
    tile-local position ``g`` (in coarse HR cells, ``g = q + Psi`` with ``q`` the
    Lagrangian site) lands at ``u = (g - origin) * grid_mult`` with
    ``origin = (crop - region) / 2 + bulk``, and CIC splits its weight between
    cells ``floor(u)`` and ``floor(u) + 1``. Cell ``j`` therefore *is* the point
    ``u = j``, which is where the kernel measures distances from.

    ``bulk`` must be the same ``(3,)`` rounded offset the deposit was given (see
    :func:`cosmo_sr.eval.density.valid_center_bulk`). Passing the frozen
    generator's bulk for the candidate, the frozen field and HR alike is what
    makes one window index a like-for-like comparison across the three.
    """
    pos = np.atleast_2d(np.asarray(pos_mpc_h, dtype=np.float64))
    cell = float(boxsize_mpc_h) / float(ng_hr)
    n_side = int(ng_hr) // int(tile_hr)
    t = int(tile_id)
    corner = np.array([t // (n_side * n_side), (t // n_side) % n_side, t % n_side],
                      dtype=np.float64) * float(tile_hr)

    origin = (float(tile_hr) - float(region)) / 2.0 + np.asarray(bulk, dtype=np.float64)
    centre = origin + float(region) / 2.0            # region centre, tile-local

    g = pos / cell - corner
    # A tile is a window on a periodic box, so take the periodic image nearest
    # the scored cube. Without this a subhalo one cell across the box face is
    # 512 cells away and is silently dropped.
    g = (g - centre + 0.5 * ng_hr) % float(ng_hr) - 0.5 * ng_hr + centre
    return (g - origin) * int(grid_mult)


@dataclass
class TileSubhalos:
    """The true HR subhalos one tile's particles are responsible for."""

    tile_id: int
    halo_id: np.ndarray            # (S,)
    num_p: np.ndarray              # (S,)
    mvir: np.ndarray               # (S,) Msun/h
    purity: np.ndarray             # (S,) share of sites in this tile
    centre: np.ndarray             # (S, 3) deposit-grid coordinates
    sigma: np.ndarray              # (S,) kernel width, fine cells
    half_width: int = 1

    @property
    def n(self) -> int:
        return int(self.halo_id.shape[0])


def tile_subhalos(
    cat: HaloCatalog, home: Mapping[str, np.ndarray], tile_id: int,
    bulk: np.ndarray, cfg: GatherConfig, soft: SoftStructureConfig,
    *, tile_hr: int = 64,
) -> TileSubhalos:
    """Select, place and size the targets of one tile. Pure numpy, run once."""
    region = int(soft.region_of(int(tile_hr)))
    m = int(soft.grid_mult)
    grid = region * m

    keep = np.flatnonzero((home["tile"] == int(tile_id))
                          & (home["purity"] >= float(cfg.min_purity)))
    rows = home["row"][keep]
    if rows.size == 0:
        return TileSubhalos(int(tile_id), *(np.zeros(0) for _ in range(4)),
                            np.zeros((0, 3)), np.zeros(0), 1)

    u = region_coordinates(
        cat.pos[rows], tile_id, bulk, ng_hr=int(soft.ng_hr), tile_hr=int(tile_hr),
        boxsize_mpc_h=float(soft.boxsize_mpc_h), region=region, grid_mult=m)
    rvir_cells = (cat.rvir[rows] / float(soft.cellsize_kpc_h)) * m
    sigma = cfg.sigma_of(rvir_cells)
    half = cfg.half_width_of(sigma)

    # A window that leaves the scored cube would be comparing a truncated
    # candidate statistic against a truncated HR one at a face where the deposit
    # is already discarding particles. Drop it rather than clamp it.
    inside = np.all((u >= half) & (u <= grid - 1 - half), axis=1)
    sel = np.flatnonzero(inside)
    if sel.size > int(cfg.max_per_tile):
        sel = sel[np.argsort(-cat.num_p[rows][sel])[: int(cfg.max_per_tile)]]
        sel = np.sort(sel)

    return TileSubhalos(
        tile_id=int(tile_id),
        halo_id=cat.ids[rows][sel].astype(np.int64),
        num_p=cat.num_p[rows][sel].astype(np.int64),
        mvir=cat.mvir[rows][sel].astype(np.float64),
        purity=home["purity"][keep][sel].astype(np.float64),
        centre=u[sel].astype(np.float64),
        sigma=sigma[sel].astype(np.float64),
        half_width=int(half),
    )


# --------------------------------------------------------------------------- #
# The differentiable statistic
# --------------------------------------------------------------------------- #
def _windows(centre: torch.Tensor, sigma: torch.Tensor, half_width: int,
             grid: int, dtype, device, mask: Optional[torch.Tensor] = None):
    """``(flat_idx, kern, w)`` -- ONE window geometry, shared by every statistic.

    Density and phase space must read *the same cells with the same weights*, or
    "this window has HR's mass but not HR's velocity dispersion" would be a
    statement about two different regions. Computing the indices once and handing
    them to both is what makes that impossible rather than merely intended.

    ``kern`` is a Gaussian at the true centre evaluated at each cell's true
    (fractional) offset, so a sub-cell move of a clump changes every statistic
    smoothly. Cells outside the grid get weight zero.
    """
    b, s = int(centre.shape[0]), int(centre.shape[1])
    h, g = int(half_width), int(grid)
    off = torch.arange(-h, h + 1, device=device, dtype=dtype)
    w = int(off.numel())
    base = torch.round(centre).to(torch.long)
    frac = centre - base.to(dtype)

    idx_axes, dist_axes, ok_axes = [], [], []
    for a in range(3):
        ia = base[..., a].unsqueeze(-1) + off.to(torch.long)
        idx_axes.append(ia.clamp(0, g - 1))
        dist_axes.append(off.view(1, 1, w) - frac[..., a].unsqueeze(-1))
        ok_axes.append((ia >= 0) & (ia < g))

    flat = ((idx_axes[0].view(b, s, w, 1, 1) * g
             + idx_axes[1].view(b, s, 1, w, 1)) * g
            + idx_axes[2].view(b, s, 1, 1, w)).reshape(b, -1)
    inside = (ok_axes[0].view(b, s, w, 1, 1)
              & ok_axes[1].view(b, s, 1, w, 1)
              & ok_axes[2].view(b, s, 1, 1, w))
    d2 = (dist_axes[0].view(b, s, w, 1, 1) ** 2
          + dist_axes[1].view(b, s, 1, w, 1) ** 2
          + dist_axes[2].view(b, s, 1, 1, w) ** 2)
    var = (sigma.to(dtype) ** 2).clamp_min(1e-6).view(b, s, 1, 1, 1)
    kern = torch.exp(-0.5 * d2 / var) * inside.to(dtype)
    if mask is not None:
        kern = kern * mask.to(dtype).view(b, s, 1, 1, 1)
    return flat, kern, w


def _pull(x: torch.Tensor, flat: torch.Tensor, s: int, w: int) -> torch.Tensor:
    """Gather one ``(B, 1, G, G, G)`` field into ``(B, S, w, w, w)`` windows."""
    b = int(x.shape[0])
    return torch.gather(x.reshape(b, -1), 1, flat).view(b, s, w, w, w)


@dataclass
class GatherDeposit:
    """One CIC pass, read by every statistic below.

    ``deposit_phase_space`` returns mass, mean velocity and intra-cell velocity
    dispersion from a *single* pass over the particles, so the density and the
    velocity field describe the same particles in the same cells by construction.
    ``delta`` is ``mass - 1`` exactly -- the same number
    :func:`cosmo_sr.reward.soft_structure.density_from_disp` would return, taken
    from this pass rather than from a second one.
    """

    delta: torch.Tensor       # (B, 1, G, G, G) overdensity
    mass: torch.Tensor        # (B, 1, G, G, G) 1 + delta
    vbar: torch.Tensor        # (B, 3, G, G, G) mass-weighted mean velocity, km/s
    sigma2: torch.Tensor      # (B, 1, G, G, G) intra-cell dispersion, (km/s)^2


def deposit_for_gather(
    field: torch.Tensor, soft: SoftStructureConfig,
    ps: Optional[PhaseSpaceConfig] = None, bulk: Optional[torch.Tensor] = None,
) -> GatherDeposit:
    """Deposit a six-channel ``(B, 6, N, N, N)`` field once. Differentiable in both."""
    if field.dim() != 5 or field.shape[1] < 6:
        raise ValueError(f"expected (B, 6, N, N, N), got {tuple(field.shape)}")
    m, vbar, sigma2 = deposit_phase_space(
        field[:, 0:3].float(), field[:, 3:6].float(), soft,
        ps or PhaseSpaceConfig(), bulk=bulk)
    return GatherDeposit(delta=m - 1.0, mass=m, vbar=vbar, sigma2=sigma2)


def gather_statistics(
    delta: torch.Tensor, centre: torch.Tensor, sigma: torch.Tensor,
    half_width: int, cfg: GatherConfig, *, grid_mult: int = 1,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(compact, contrast)``, each ``(B, S)``, in particle units.

    ``delta`` is ``(B, 1, G, G, G)`` valid-centre overdensity, ``centre`` is
    ``(B, S, 3)`` in that grid's coordinates and ``sigma`` is ``(B, S)``.
    Differentiable in ``delta`` (hence in the displacement that deposited it).

    ``compact``
        Kernel-weighted mass in cells dense enough to be a collapsed object:
        ``sum_c K_c * sigmoid((u_c - u_thr)/tau) * mass_c``. The "is there an
        object here" coordinate.
    ``contrast``
        The same sum with the *threshold-free* weight -- mass in cells that beat
        their own smoothed neighbourhood by ``contrast_margin``. A generator that
        raised every density by a constant factor would move the first and not
        the second.
    """
    if delta.dim() != 5 or delta.shape[1] != 1:
        raise ValueError(f"expected (B, 1, G, G, G), got {tuple(delta.shape)}")
    if centre.dim() != 3 or centre.shape[-1] != 3:
        raise ValueError(f"expected centre (B, S, 3), got {tuple(centre.shape)}")
    b, s = int(centre.shape[0]), int(centre.shape[1])
    if int(delta.shape[0]) != b:
        raise ValueError(f"delta batch {int(delta.shape[0])} != centre batch {b}")
    if s == 0:
        z = torch.zeros(b, 0, device=delta.device, dtype=delta.dtype)
        return z, z

    m3 = float(int(grid_mult) ** 3)
    unit = (1.0 + delta).clamp_min(0.0)
    u = torch.log(unit.clamp_min(1e-6))
    us = torch.log(_smooth(unit, int(cfg.contrast_scale)).clamp_min(1e-6))
    thr = float(np.log1p(float(cfg.compact_delta)))
    tau = float(cfg.tau_log)

    part = unit / m3
    c_field = torch.sigmoid((u - thr) / tau) * part
    p_field = torch.sigmoid((u - us - float(cfg.contrast_margin)) / tau) * part

    flat, kern, w = _windows(centre, sigma, half_width, int(delta.shape[-1]),
                             delta.dtype, delta.device, mask=mask)
    dims = (-3, -2, -1)
    return ((kern * _pull(c_field, flat, s, w)).sum(dim=dims),
            (kern * _pull(p_field, flat, s, w)).sum(dim=dims))


def outside_weight_map(
    centre: torch.Tensor, sigma: torch.Tensor, half_width: int, grid: int,
    mask: Optional[torch.Tensor] = None, *, dtype=torch.float32, device=None,
) -> torch.Tensor:
    """``(B, 1, G, G, G)``: 1 where no target's kernel reaches, 0 under a target.

    The complement of the supervision. It is a *constant* -- the centres and
    widths come from the HR catalog and never move -- so it is built once and
    reused, and it carries no gradient of its own.
    """
    b, s = int(centre.shape[0]), int(centre.shape[1])
    g = int(grid)
    dev = device if device is not None else centre.device
    cover = torch.zeros(b, g ** 3, device=dev, dtype=dtype)
    if s:
        flat, kern, w = _windows(centre.to(dev), sigma.to(dev), half_width, g,
                                 dtype, dev, mask=mask)
        cover.scatter_add_(1, flat, kern.reshape(b, -1))
    return (1.0 - cover.clamp(0.0, 1.0)).view(b, 1, g, g, g)


def preserve_statistic(
    delta: torch.Tensor, outside_w: torch.Tensor, cfg: GatherConfig,
) -> torch.Tensor:
    """``(B,)`` mass-weighted local-peak fraction OUTSIDE the supervised windows.

    The same coordinate ``peak_contrast`` reports -- mass in cells that beat
    their own smoothed neighbourhood by ``contrast_margin`` -- restricted to the
    part of the field the objective is blind to. Threshold-free, so a generator
    cannot satisfy it by raising the density everywhere.
    """
    m = (1.0 + delta).clamp_min(0.0)
    u = torch.log(m.clamp_min(1e-6))
    us = torch.log(_smooth(m, int(cfg.contrast_scale)).clamp_min(1e-6))
    w = torch.sigmoid((u - us - float(cfg.contrast_margin)) / float(cfg.tau_log))
    dims = (1, 2, 3, 4)
    return ((outside_w * w * m).sum(dim=dims)
            / (outside_w * m).sum(dim=dims).clamp_min(1e-8))


def preserve_loss(
    delta: torch.Tensor, targets: GatherTargets, cfg: GatherConfig,
) -> Tuple[torch.Tensor, float]:
    """``[1 - S_theta/S_0]_+^2`` on the unsupervised field, and the ratio.

    Hinged, like the gather term and for the same reason: losing local structure
    away from the targets is the failure, *gaining* it is not, and a two-sided
    term would forbid the generator from improving anywhere it was not told to.
    """
    if targets.outside_w is None or targets.frozen_preserve is None:
        return torch.zeros((), device=delta.device, dtype=delta.dtype), 1.0
    s_now = preserve_statistic(delta, targets.outside_w, cfg)
    ratio = s_now / targets.frozen_preserve.clamp_min(1e-8)
    loss = torch.relu(1.0 - ratio).pow(2).mean()
    return loss, float(ratio.mean().item())


def velocity_statistics(
    dep: GatherDeposit, centre: torch.Tensor, sigma: torch.Tensor,
    half_width: int, *, mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(vbulk, vdisp)`` per subhalo: ``(B, S, 3)`` km/s and ``(B, S)`` km/s.

    Rockstar is a **phase-space** finder: it links particles in 6-D, and the
    channel swap measured that HR loses 65% of its subhalos when given SR2's
    velocities (``docs/sr2_substructure_module.md`` section 2 item 3). A clump
    that is spatially right and kinematically wrong is not a halo, so the
    displacement statistics above are not sufficient on their own.

    ``vbulk``
        The clump's mass-weighted mean velocity -- does the object move with the
        right flow.
    ``vdisp``
        Its **total** internal velocity dispersion, by the law of total variance:
        the mass-weighted mean of each cell's own dispersion *plus* the scatter
        of the cell means about the clump mean. Taking only the first would
        measure sub-cell jitter and miss coherent shear across the object, which
        is most of what distinguishes a bound satellite from a passing stream.

    The ``+ 1e-8`` under the square root is not cosmetic: an empty window pools
    to exactly zero, where ``d/dx sqrt(x)`` is infinite and the backward pass
    returns NaN -- the same failure, and the same fix, as the arm-B/C velocity
    dispersions in ``reward/phase_space.py``.
    """
    b, s = int(centre.shape[0]), int(centre.shape[1])
    if s == 0:
        return (torch.zeros(b, 0, 3, device=dep.mass.device, dtype=dep.mass.dtype),
                torch.zeros(b, 0, device=dep.mass.device, dtype=dep.mass.dtype))
    flat, kern, w = _windows(centre, sigma, half_width, int(dep.mass.shape[-1]),
                             dep.mass.dtype, dep.mass.device, mask=mask)
    dims = (-3, -2, -1)

    weight = kern * _pull(dep.mass, flat, s, w)            # (B,S,w,w,w)
    tot = weight.sum(dim=dims).clamp_min(1e-8)             # (B,S)
    vw = torch.stack([_pull(dep.vbar[:, a: a + 1], flat, s, w) for a in range(3)],
                     dim=2)                                # (B,S,3,w,w,w)
    vbulk = ((weight.unsqueeze(2) * vw).sum(dim=dims)
             / tot.unsqueeze(-1))                          # (B,S,3)
    between = ((vw - vbulk[..., None, None, None]) ** 2).sum(dim=2)
    within = _pull(dep.sigma2, flat, s, w)
    vdisp2 = (weight * (within + between)).sum(dim=dims) / tot
    return vbulk, (vdisp2.clamp_min(0.0) + 1e-8).sqrt()


# --------------------------------------------------------------------------- #
# Targets and loss
# --------------------------------------------------------------------------- #
@dataclass
class GatherTargets:
    """One batch's padded windows and their HR reference values.

    Built once, on the HR field, with the frozen generator's bulk offset -- so it
    is *data*. Nothing in it is read at inference.
    """

    centre: torch.Tensor           # (B, S, 3)
    sigma: torch.Tensor            # (B, S)
    mask: torch.Tensor             # (B, S) 1 where the slot is a real subhalo
    hr_compact: torch.Tensor       # (B, S)
    hr_contrast: torch.Tensor      # (B, S)
    hr_vbulk: torch.Tensor         # (B, S, 3) km/s
    hr_vdisp: torch.Tensor         # (B, S) km/s
    half_width: int
    num_p: torch.Tensor            # (B, S)
    halo_id: torch.Tensor          # (B, S)
    tiles: List[int] = field(default_factory=list)
    #: ``(B, 1, G, G, G)``, 1 where no target's kernel reaches. Optional so a
    #: density-only caller need not build one; ``None`` means "no preservation
    #: term", not "preserve nothing".
    outside_w: Optional[torch.Tensor] = None
    #: ``(B,)`` the frozen generator's structure in that complement.
    frozen_preserve: Optional[torch.Tensor] = None

    @property
    def n_targets(self) -> int:
        return int(self.mask.sum().item())

    def to(self, device) -> "GatherTargets":
        return GatherTargets(
            centre=self.centre.to(device), sigma=self.sigma.to(device),
            mask=self.mask.to(device), hr_compact=self.hr_compact.to(device),
            hr_contrast=self.hr_contrast.to(device),
            hr_vbulk=self.hr_vbulk.to(device), hr_vdisp=self.hr_vdisp.to(device),
            half_width=self.half_width,
            num_p=self.num_p.to(device), halo_id=self.halo_id.to(device),
            tiles=list(self.tiles),
            outside_w=None if self.outside_w is None else self.outside_w.to(device),
            frozen_preserve=(None if self.frozen_preserve is None
                             else self.frozen_preserve.to(device)))

    def select(self, rows: Sequence[int]) -> "GatherTargets":
        """The sub-batch for ``rows`` of the current batch order."""
        r = torch.as_tensor(list(rows), dtype=torch.long, device=self.centre.device)
        return GatherTargets(
            centre=self.centre.index_select(0, r),
            sigma=self.sigma.index_select(0, r),
            mask=self.mask.index_select(0, r),
            hr_compact=self.hr_compact.index_select(0, r),
            hr_contrast=self.hr_contrast.index_select(0, r),
            hr_vbulk=self.hr_vbulk.index_select(0, r),
            hr_vdisp=self.hr_vdisp.index_select(0, r),
            half_width=self.half_width,
            num_p=self.num_p.index_select(0, r),
            halo_id=self.halo_id.index_select(0, r),
            tiles=[self.tiles[int(i)] for i in rows] if self.tiles else [],
            outside_w=(None if self.outside_w is None
                       else self.outside_w.index_select(0, r)),
            frozen_preserve=(None if self.frozen_preserve is None
                             else self.frozen_preserve.index_select(0, r)))


def stack_tile_subhalos(
    per_tile: Sequence[TileSubhalos], device=None,
) -> GatherTargets:
    """Pad a list of per-tile selections into one padded batch.

    The HR reference values are left at zero here; :func:`attach_hr_reference`
    fills them from the HR deposit, which needs a forward pass this module does
    not own.
    """
    b = len(per_tile)
    s = max([t.n for t in per_tile] + [1])
    half = max([t.half_width for t in per_tile] + [1])
    z = lambda *shape: torch.zeros(*shape, dtype=torch.float32)  # noqa: E731
    centre, sigma, mask = z(b, s, 3), torch.ones(b, s), z(b, s)
    num_p = torch.zeros(b, s, dtype=torch.long)
    halo_id = torch.full((b, s), -1, dtype=torch.long)
    for i, t in enumerate(per_tile):
        if t.n == 0:
            continue
        centre[i, : t.n] = torch.from_numpy(t.centre).float()
        sigma[i, : t.n] = torch.from_numpy(t.sigma).float()
        mask[i, : t.n] = 1.0
        num_p[i, : t.n] = torch.from_numpy(np.asarray(t.num_p)).long()
        halo_id[i, : t.n] = torch.from_numpy(np.asarray(t.halo_id)).long()
    out = GatherTargets(centre=centre, sigma=sigma, mask=mask,
                        hr_compact=z(b, s), hr_contrast=z(b, s),
                        hr_vbulk=z(b, s, 3), hr_vdisp=torch.ones(b, s),
                        half_width=int(half), num_p=num_p, halo_id=halo_id,
                        tiles=[int(t.tile_id) for t in per_tile])
    return out.to(device) if device is not None else out


def attach_hr_reference(
    targets: GatherTargets, hr: "GatherDeposit | torch.Tensor",
    cfg: GatherConfig, *, grid_mult: int = 1,
) -> GatherTargets:
    """Measure HR's reference values and drop the targets HR itself cannot show.

    ``hr`` is the HR field's :class:`GatherDeposit` -- the *same* deposit the
    candidate is compared against, on the frozen generator's bulk origin. A bare
    overdensity tensor is still accepted for density-only use, in which case the
    velocity references are left neutral and the velocity terms contribute
    nothing.

    A subhalo whose HR compact mass is under ``min_hr_compact`` particles is a
    denominator made of noise -- it is masked out here rather than clipped later,
    so the reported target count is the number of subhalos actually being asked
    for.
    """
    delta = hr.delta if isinstance(hr, GatherDeposit) else hr
    with torch.no_grad():
        c, p = gather_statistics(delta, targets.centre, targets.sigma,
                                 targets.half_width, cfg, grid_mult=grid_mult,
                                 mask=targets.mask)
        keep = targets.mask * (c >= float(cfg.min_hr_compact)).to(targets.mask.dtype)
        if isinstance(hr, GatherDeposit):
            vb, vd = velocity_statistics(hr, targets.centre, targets.sigma,
                                         targets.half_width, mask=targets.mask)
            targets.hr_vbulk, targets.hr_vdisp = vb, vd
    targets.hr_compact, targets.hr_contrast, targets.mask = c, p, keep
    return targets


def gather_loss(
    delta: "GatherDeposit | torch.Tensor", targets: GatherTargets,
    cfg: GatherConfig, *, grid_mult: int = 1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """The per-subhalo objective, position **and** velocity, with its diagnostics.

        L = mean_s  [1 - C/C_HR]_+^2  +  w_contrast [1 - P/P_HR]_+^2
                  +  w_vdisp  log(sigma_v / sigma_v,HR)^2
                  +  w_vbulk  |v - v_HR|^2 / sigma_v,HR^2

    The two density terms are **one-sided**: a candidate that already matches HR
    at a subhalo contributes zero, and the loss can never ask for more contrast
    than HR has. The two velocity terms are **two-sided**, and deliberately: too
    hot is unbound and a phase-space finder discards it, too cold is the measured
    sub-virial defect. There is no direction in which being wrong about velocity
    is safe, so there is no hinge.

    Both velocity terms are normalised by the subhalo's *own* HR dispersion, so a
    50 km/s dwarf and a 600 km/s cluster satellite are held to the same
    *fractional* accuracy -- the same per-subhalo equalisation the density ratio
    does.

    Passing a bare overdensity tensor runs the density half alone.
    """
    dep = delta if isinstance(delta, GatherDeposit) else None
    d = dep.delta if dep is not None else delta
    c, p = gather_statistics(d, targets.centre, targets.sigma,
                             targets.half_width, cfg, grid_mult=grid_mult,
                             mask=targets.mask)
    mask = targets.mask
    n = mask.sum().clamp_min(1.0)
    eps = 1e-6
    r_c = c / targets.hr_compact.clamp_min(eps)
    r_p = p / targets.hr_contrast.clamp_min(eps)
    per = torch.relu(1.0 - r_c).pow(2) + float(cfg.w_contrast) * torch.relu(1.0 - r_p).pow(2)

    vel: Dict[str, float] = {}
    if dep is not None and (cfg.w_vdisp or cfg.w_vbulk):
        vb, vd = velocity_statistics(dep, targets.centre, targets.sigma,
                                     targets.half_width, mask=mask)
        hr_vd = targets.hr_vdisp.clamp_min(1.0)          # km/s; 1 km/s floor
        l_disp = torch.log(vd.clamp_min(1e-3) / hr_vd).pow(2)
        l_bulk = ((vb - targets.hr_vbulk) ** 2).sum(dim=-1) / hr_vd.pow(2)
        per = per + float(cfg.w_vdisp) * l_disp + float(cfg.w_vbulk) * l_bulk
        with torch.no_grad():
            vel = {
                "vdisp_ratio": float(((vd / hr_vd) * mask).sum().item() / float(n.item())),
                "vbulk_offset": float(
                    ((l_bulk.clamp_min(0.0).sqrt()) * mask).sum().item() / float(n.item())),
                "vdisp_hr_mean": float((hr_vd * mask).sum().item() / float(n.item())),
            }

    loss = (per * mask).sum() / n
    with torch.no_grad():
        def avg(x):
            return float((x * mask).sum().item() / float(n.item()))
        diag = {
            # The LIVE count, not the clamped denominator: with every target
            # masked out `n` is 1.0 and reporting it would claim a target that
            # is not being asked for.
            "n_targets": int(mask.sum().item()),
            "gather_loss": float(loss.item()),
            "compact_ratio": avg(r_c),
            "compact_ratio_median": float(
                torch.median(r_c[mask > 0]).item()) if int(mask.sum()) else 0.0,
            "contrast_ratio": avg(r_p),
            "compact_satisfied": avg((r_c >= 1.0).to(r_c.dtype)),
            "compact_hr_mean": avg(targets.hr_compact),
            "compact_pred_mean": avg(c),
            **vel,
        }
    return loss, diag


def per_subhalo_table(
    delta: "GatherDeposit | torch.Tensor", targets: GatherTargets,
    cfg: GatherConfig, *, grid_mult: int = 1,
) -> List[Dict[str, float]]:
    """One row per live target: what was asked for and what the field shows."""
    dep = delta if isinstance(delta, GatherDeposit) else None
    d = dep.delta if dep is not None else delta
    with torch.no_grad():
        c, p = gather_statistics(d, targets.centre, targets.sigma,
                                 targets.half_width, cfg, grid_mult=grid_mult,
                                 mask=targets.mask)
        if dep is not None:
            vb, vd = velocity_statistics(dep, targets.centre, targets.sigma,
                                         targets.half_width, mask=targets.mask)
        else:
            vb = vd = None
    rows: List[Dict[str, float]] = []
    mask = targets.mask.cpu().numpy()
    for i in range(mask.shape[0]):
        for j in np.flatnonzero(mask[i] > 0):
            row = {
                "tile": int(targets.tiles[i]) if targets.tiles else i,
                "halo_id": int(targets.halo_id[i, j].item()),
                "num_p": int(targets.num_p[i, j].item()),
                "hr_compact": float(targets.hr_compact[i, j].item()),
                "compact": float(c[i, j].item()),
                "hr_contrast": float(targets.hr_contrast[i, j].item()),
                "contrast": float(p[i, j].item()),
            }
            if vd is not None:
                row.update({
                    "hr_vdisp": float(targets.hr_vdisp[i, j].item()),
                    "vdisp": float(vd[i, j].item()),
                    "vbulk_err": float(torch.linalg.vector_norm(
                        vb[i, j] - targets.hr_vbulk[i, j]).item()),
                })
            rows.append(row)
    return rows
