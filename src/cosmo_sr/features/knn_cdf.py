"""Particle-centred k-nearest-neighbour distance distributions, differentiable.

What this is, and why it is not another spectrum
------------------------------------------------
Every field-level guard this line has is a **power spectrum**: ``field_guards``
hinges on binned ``P(k)``, ``gather_critic`` feeds a high-pass field to a CNN,
``cond_spread`` cross-correlates. Two measurements bound what that family can
ever say about subhalos.

*Measured*, ``density_power_set9.json`` (`density-power-cannot-see-subhalos`):
frozen SR2 already matches HR's density ``P(k)`` to 1-5% **at every resolvable
k**, and the fine-tune that won the held-out Rockstar gate *overshoots* it by
~35% at ``k ~ 8-16``. A 512^3 particle mesh has a 195 kpc/h cell; a 200-particle
subhalo is ~50-100 kpc/h across, i.e. ``k > 32``. The objects this line exists
to build live **below the grid**, so no grid statistic can see them: P(k) is not
a weak signal here, it is the wrong instrument.

*Measured*, `highk-guard-is-blind`: the ``self`` arm sat at 3.4x HR's high-k
amplitude while its cross-correlation with HR at those k fell 0.915 -> 0.318 --
it put the right amount of power in the wrong places. An amplitude guard cannot
charge for that, and ``build_tile_pool``'s own comment says so in as many words:
"It only constrains the AMOUNT of small-scale power, never its placement."

This module is the other kind of statistic. For a set of query particles, take
the distance to the ``k``-th nearest **other particle**, for several ``k``, and
compare the *distribution* of those distances to HR's. It is:

* **grid-free.** ``r_k`` is a length between two particles; nothing is deposited,
  so there is no cell size and no Nyquist. At the 512^3 mean interparticle
  spacing of 195 kpc/h, a particle inside a 200p subhalo has its 8th neighbour
  at a few tens of kpc/h -- the regime the density mesh cannot represent is the
  regime the SMALL ranks measure best. (The large ranks are there for a
  different reason, and it is the most important fact in this module: see
  "The ranks set what gets built".)
* **real-space and placement-sensitive.** ``r_8`` is small only if *that
  particle* actually has 8 neighbours nearby. Smearing the same small-scale
  power over the wrong sites does not produce close neighbours, so the failure
  the high-k hinge is blind to is the failure this charges for. (It is still not
  a *position* statistic -- see "What this cannot do".)
* **catalog-free.** No ``owner`` array, no halo matching, no parent split, no
  ``num_p`` window, no purity cut. The inputs are particle positions and a
  choice of query set, so unlike every term in :mod:`member_gather` it can be
  charged on the **unsupervised box-wide tile pool**, where the supervised loss
  has no support at all and where the 6x train/holdout gap of
  `selfvel-arm-failed-the-gate` was measured. One caveat, and it is the whole
  reason ``KnnConfig.strata`` exists: ``--knn-classes`` reintroduces a label by
  the back door, masking the queries by HR's owner class. ``strata`` is the
  catalog-free way to make the same split -- see "Density strata" below.
* **more than 2-point.** The kNN-distance CDF at a fixed ``k`` is a sum over all
  orders of the correlation hierarchy (it is ``1 - P(0 | sphere)``-style void
  probability, complementary to counts-in-cells), which is why a field can match
  ``P(k)`` exactly and miss this.

How the gradient survives a kNN search
--------------------------------------
The search is combinatorial and the distances are not. So:

1. Under ``no_grad``, find the neighbour **identities** -- integer row indices --
   by a chunked minimum-image ``topk``. Nothing differentiable happens here and
   nothing needs to: the output is an index array.
2. Re-index the *live* position tensor with those integers and recompute the
   distances in torch. The query and its neighbours both keep their connection to
   the generator's output, so ``d loss / d Psi`` flows through both ends of every
   pair.

This is **exactly** the gradient of a fully differentiable
``topk``-over-all-pairs implementation, not an approximation of it, because
``topk`` is itself a gather: its backward is a scatter through the selected
indices, which is what step 2 reconstructs. The only thing dropped is the
derivative of the *selection*, which is zero almost everywhere and undefined on
the measure-zero set where two neighbours tie.
``tests/features/test_knn_cdf.py`` pins this against an all-pairs differentiable
reference to 1e-10 and against central finite differences.

The form of the loss
--------------------
Per ``k``, both fields' ``log r_k`` are **sorted across queries** and compared by
rank -- a 1-D optimal-transport (W1) match, reduced to ``n_quantiles`` equal-count
bins. Three deliberate choices, each one a lesson already paid for:

*Sorted, not paired by particle id.* The per-particle value is an address, and
`shell-target-is-real-not-noise` measured what happens when this line asks a
shared operator for an address it cannot predict: train residual 7.3x, holdout
1.4x, 55% of the abundance spent. Sorting asks for the distribution and says
nothing about which particle sits where.

*Quantile-binned.* The ``w_shell`` post-mortem again (``shell_quantiles``): full
rank-by-rank pairing hands the loss one degree of freedom per query, and the
high-rank wiggle is the part a shared operator can only memorise. ``Q`` is a
direct dial on "ask only for the low-order shape".

*Two-sided by default, in log.* Unlike the high-k hinge, the failure mode here is
**not** one-directional. ``host-damage-is-mostly-reparenting`` and
``radial-profile-is-flat-in-host-mass`` both measured central over-concentration
as this line's characteristic damage, and over-concentration shows up here as
``r_k`` *below* HR's. A hinge would charge the deficit and pay for the excess.
``one_sided=True`` is available for the case where a rung is known to be
resolution-limited at small ``k``.

The ranks set what gets built
-----------------------------
*Measured*, ``tests/features/test_knn_cdf.py::test_small_ranks_fragment``: one
HR clump of 150 particles in 300 background, a candidate whose clump starts
diffuse, Adam on this loss alone for 400 steps. **Every** rank set drives the
loss to exactly zero. What differs is the object (largest FOF group of the clump
at b = 0.2; HR's is 112):

==============  ==========  ===================  =============
ks              final loss  largest FOF of 150   r50 (HR 0.44)
==============  ==========  ===================  =============
(2, 4, 8)       0.0         13  -- fragments     2.57
(8, 16, 32)     0.0         40                   1.40
(8, 32, 64)     0.0         90                   0.48
(16, 64, 128)   0.0         107                  0.49
==============  ==========  ===================  =============

A distribution of ``r_k`` only asks "does each particle have ``k`` neighbours
this close". A field of ``k+1``-particle knots answers yes for every particle and
never forms the halo, so **the minimum of this loss is not unique below
``k_max ~ N_object``**, and the optimiser finds the fragmented minimum first
because it is closer. The rule: the largest rank must be comparable to the
particle count of the smallest object the run is meant to build.

For this line that is not a detail. The deficit is at 50-200 particles --
`subhalo-count-is-hr-bound-is-not` measured 46.5% of HR's subhalos at 50-100p --
so ranks capped at 32 would build knots **below Rockstar's 50-particle floor**:
invisible to the gate at best, a pedestal of unresolved clutter at worst. The
default is therefore ``(8, 32, 64, 128)``: 8 for the sub-grid inner profile, 64
and 128 to make fragmentation cost something at the floor.

What this cannot do
-------------------
It is **translation-invariant** -- and not as an aside, it is provable and is
proven in the tests: rigidly shifting the whole field leaves every pair distance
unchanged and so leaves this loss exactly unchanged. So it cannot place an
object, and it is not a replacement for ``member_gather``'s ``w_centre``, whose
absence was measured as the dominant failure of the 2026-08-21 free-field run
(median 0.414 Mpc/h from target against a 0.150 search radius). It is a
**support** term: it says the field must have HR's small-scale clumping
everywhere, including the 93.75% of tiles no supervised set touches.

It is also, like every differentiable surrogate this line has built, gameable in
principle: a filamentary or a uniformly-compressed field can carry a matched
``r_k`` distribution without carrying bound halos. Per
`occupation-ratio-is-gameable` and `tile-overfit-proxy-exploitation` the
expectation is that it *will* be gamed unless the gate stays real Rockstar, and
the gate stays real Rockstar.

Units and conventions: positions in comoving Mpc/h on the full periodic box, the
same ones :func:`cosmo_sr.features.member_gather.tile_particles` returns, so a
query set built there and a distance measured here agree to float precision.
Distances are logged against a 1 Mpc/h reference, i.e. ``log r`` with ``r`` in
Mpc/h -- a constant offset that cancels in every difference this module takes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

#: Added inside the square root before the log. Two distinct jobs, both load
#: bearing: ``d/dr sqrt`` is infinite at ``r = 0``, which is exactly the NaN
#: recorded in `proxy-reward-gradient-nan` (and fixed there the same way); and
#: ``log 0`` is ``-inf``. 1e-5 Mpc/h = 10 pc/h is five orders below the softening
#: length the potential terms use, so it cannot bias a distance that means
#: anything.
EPS_MPC_H = 1e-5


@dataclass(frozen=True)
class KnnConfig:
    """Estimator and penalty shape. Frozen so a run's config is a value."""

    #: Which neighbour ranks to measure. The small ones are the sub-grid scales
    #: no spectrum reaches; the LARGE one decides whether the loss's minimum is
    #: a halo or a field of knots. See "The ranks set what gets built" in the
    #: module docstring -- measured: ``(2, 4, 8)`` reaches zero loss with the
    #: clump's largest FOF group at 13 of 150 particles, ``(16, 64, 128)`` at 107
    #: against HR's 112. Keep ``max(ks)`` at or above the particle count of the
    #: smallest object the run is meant to build; for the 50-200p deficit that
    #: is 64 at the very least.
    ks: Tuple[int, ...] = (8, 32, 64, 128)
    #: Query particles drawn per tile, per evaluation. The cost is
    #: ``O(n_queries * N)`` for the search and ``O(n_queries * max(ks))`` for the
    #: tape, so this is the only knob that matters for step time; 1024 of a
    #: tile's 262,144 particles is a 0.4% sample, and the statistic is a
    #: distribution over a thousand draws rather than a per-particle value, so
    #: the sampling error enters as noise in a mean, not as bias.
    n_queries: int = 1024
    #: Equal-count bins the sorted pairing is reduced to. 0 is the full
    #: rank-by-rank pairing. See the module docstring: the default is NOT the
    #: full pairing, because the full pairing is the mistake ``w_shell`` made.
    n_quantiles: int = 64
    #: Charge only for ``r_k`` ABOVE HR's (a clumping deficit), not below.
    #: Default off: central over-concentration is this line's measured damage
    #: and a one-sided term would pay for it.
    one_sided: bool = False
    #: Free zone, as a FRACTIONAL error in ``r_k``: the term costs nothing while
    #: a quantile's candidate radius is within this of HR's. Applied in log
    #: space as ``log1p(dead_zone)`` so the knob means what it says. 0.02 is the
    #: ``shell_dead_zone`` value and the same reasoning: wide enough to ignore
    #: the sampling noise of a 1024-draw quantile, narrow enough not to eat a
    #: signal that starts at tens of percent.
    dead_zone: float = 0.02
    #: Cells of Eulerian buffer a query must sit inside the tile's own cube by.
    #: This is the ONLY source of bias in the estimator and it is not optional:
    #: a query near a tile face has true neighbours in the adjacent tile, which
    #: is not in the candidate set, so its ``r_k`` is **overestimated** -- in the
    #: same direction as the deficit this term exists to charge, i.e. it would
    #: manufacture the signal. ``scripts/features/knn_cdf_diagnostic.py
    #: --boundary`` measures the value rather than assuming it; set it from that
    #: measurement, never by taste.
    buffer_cells: float = 8.0
    #: Cap on one search block's element count, the ``pot_max_elems`` convention
    #: of :mod:`member_gather` and for the same reason: rows are not a memory
    #: bound, elements are. Two ``(rows, N)`` temporaries live at once, so
    #: 1 << 24 is ~128 MiB of fp32 peak.
    search_max_elems: int = 1 << 24
    #: Phase space (MEASURED, round 5b, jobs 52090-52094): with v/phase_lambda
    #: appended to x, the class-conditioned 50-200p subhalo term ranks the 20
    #: working set9 arms at rho -0.65 (p 0.002) against region subhalo counts,
    #: where the same statistic in position only ranks them at -0.23 (p 0.32).
    #: km/s per Mpc/h; the pass is at 4000 and falls off on both sides (2000
    #: -0.55, 8000 -0.60). 0 keeps the position-only statistic of rounds 1-4.
    #: Positions stay periodic; the velocity axes are open.
    phase_lambda: float = 0.0
    #: Equal-count density strata, computed by each field from ITS OWN
    #: particles. 0 or 1 is the pooled form. This is the catalog-free
    #: replacement for ``--knn-classes``: the class mask exists because the
    #: pooled statistic is dominated by host and unbound particles (round 5b,
    #: rho +0.91 with subhalo count -- backwards), and the fix it applied was an
    #: HR Rockstar label in the gradient. Stratifying on each field's own
    #: ``r_{strata_rank}`` splits the query population by local density with no
    #: catalog at all, and charges the loss separately in each stratum.
    #:
    #: What this does and does not buy, because the arithmetic is exact and the
    #: claim would otherwise be overstated: for the stratifying column ITSELF,
    #: equal-count strata of the sorted values, each matched by sorted rank in
    #: ``n_quantiles`` bins, is *identically* the pooled match in
    #: ``strata * n_quantiles`` bins -- pinned in
    #: ``test_strata_on_the_stratifying_rank_is_finer_binning``. There is no new
    #: information in that column. ALL of the gain is in the OTHER ranks: the
    #: term becomes the CONDITIONAL distribution ``P(r_k | r_strat decile)``
    #: rather than the marginal of each, and it is the conditional that separates
    #: a halo from a field of knots -- a knot has HR's small ``r_8`` and nothing
    #: like HR's ``r_128`` at that same ``r_8``, which no marginal charges for.
    strata: int = 0
    #: Neighbour rank whose distance defines the strata. Must be one of ``ks``
    #: (the search is shared, so a rank outside ``ks`` would cost a second one).
    #: 0 means ``min(ks)`` -- the smallest rank is the most local density proxy
    #: and the one least contaminated by the halo a particle sits in.
    strata_rank: int = 0
    #: Which strata enter the loss, dense-first (``0`` is the densest). Empty
    #: charges all of them with equal weight.
    #:
    #: This knob is NOT optional garnish, and the measurement that says so is
    #: ``test_equal_count_strata_do_not_reweight_the_dense_tail``: equal-count
    #: strata of ``Q`` bins each partition the query population into exactly the
    #: same ``strata * Q`` equal-count bins the pooled form already uses, so the
    #: dense tail's share of the loss is IDENTICAL either way. Measured on the
    #: dilution fixture: pooled 0.0881, 8 equal-weight strata 0.0912, while the
    #: densest stratum alone carries 0.726 of a per-stratum budget in which the
    #: other seven are 0.004 or less. Partitioning alone changes the diagnostic,
    #: not the gradient; ``strata_charge=(0,)`` is what actually reweights, and
    #: it is the catalog-free analogue of ``--knn-classes sub_50_200`` -- a
    #: query mask by density the field reads off itself instead of by owner
    #: class read off HR's Rockstar catalog.
    #:
    #: Read ``docs/sr2_knn_strata.md`` before choosing it: MEASURED on the same
    #: fixture, the densest stratum is 65-83% HOST particles, not subhalo ones,
    #: and host over-concentration is this line's characteristic damage
    #: (`host-damage-is-mostly-reparenting`). Whether a density cut is a usable
    #: stand-in for the class mask on the REAL box is what
    #: ``scripts/features/knn_strata_composition.py`` measures, and that
    #: diagnostic gates this arm.
    strata_charge: Tuple[int, ...] = ()
    #: Deterministic query draw. The queries are redrawn every evaluation (a
    #: fixed set would supervise a fixed sub-sample, which is a weaker and
    #: memorisable statement); this seeds that draw so a diagnostic is
    #: reproducible.
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.ks:
            raise ValueError("ks is empty; there is nothing to measure")
        if min(self.ks) < 1:
            raise ValueError(f"ks {self.ks} contains a rank below 1")
        if len(set(self.ks)) != len(self.ks):
            raise ValueError(f"ks {self.ks} repeats a rank")
        if self.n_queries < 1:
            raise ValueError(f"n_queries {self.n_queries} is below 1")
        if self.n_quantiles < 0:
            raise ValueError(
                f"n_quantiles {self.n_quantiles} is negative; use 0 for the "
                "full rank-by-rank pairing")
        if self.dead_zone < 0.0:
            raise ValueError(f"dead_zone {self.dead_zone} is negative")
        if self.buffer_cells < 0.0:
            raise ValueError(f"buffer_cells {self.buffer_cells} is negative")
        if self.phase_lambda < 0.0:
            raise ValueError(f"phase_lambda {self.phase_lambda} is negative")
        if self.strata < 0:
            raise ValueError(f"strata {self.strata} is negative; 0 is pooled")
        if self.strata_rank < 0:
            raise ValueError(f"strata_rank {self.strata_rank} is negative")
        if self.strata_charge:
            if self.strata <= 1:
                raise ValueError(
                    f"strata_charge {self.strata_charge} needs strata > 1")
            bad = [j for j in self.strata_charge
                   if j < 0 or j >= int(self.strata)]
            if bad:
                raise ValueError(
                    f"strata_charge {bad} outside [0, {self.strata})")
            if len(set(self.strata_charge)) != len(self.strata_charge):
                raise ValueError(
                    f"strata_charge {self.strata_charge} repeats a stratum")
        if self.strata > 1 and self.strata_rank and (
                self.strata_rank not in self.ks):
            raise ValueError(
                f"strata_rank {self.strata_rank} is not in ks {self.ks}; the "
                "stratifying distance is read from the shared search, so it "
                "must be a rank the search already returns")
        if self.search_max_elems < 1:
            raise ValueError(
                f"search_max_elems {self.search_max_elems} is below 1")


# --------------------------------------------------------------------------- #
# The search: integers out, no tape
# --------------------------------------------------------------------------- #
def _minimum_image(delta: torch.Tensor, boxsize_mpc_h: float) -> torch.Tensor:
    """Wrap a separation into ``[-L/2, L/2)``. Differentiable, slope 1.

    ``round`` has zero derivative almost everywhere, so this is the identity to
    autograd except on the measure-zero branch cut half a box away -- the same
    argument :func:`member_gather.unwrap_about` makes, and it holds here for the
    same reason: the cut sits at ``L/2`` and a neighbour distance is ``< 1``.
    """
    L = float(boxsize_mpc_h)
    return delta - L * torch.round(delta / L)


@torch.no_grad()
def knn_neighbour_ids(
    pos: torch.Tensor, query_rows: torch.Tensor, cfg: KnnConfig, *,
    boxsize_mpc_h: float = 100.0,
) -> torch.Tensor:
    """``(n_queries, len(ks))`` row indices of the ``k``-th nearest neighbours.

    Detached by construction -- the decorator is the contract, not a precaution.
    Self-matches are removed **by row identity**, not by dropping the nearest
    neighbour: at a collapsed site two distinct particles can sit closer to each
    other than float precision distinguishes from zero, and "drop the first
    column" would then silently delete a real neighbour and report a true pair as
    a self-match. The identity test cannot be fooled by coincident positions.

    Chunked over queries with both temporaries bounded in elements, so the cost
    of a tile is linear in its particle count rather than a step function at the
    OOM (the ``_pair_rows`` lesson from the 2026-08-23 rung ladder).
    """
    n = int(pos.shape[0])
    k_max = int(max(cfg.ks))
    if n < k_max + 2:
        raise ValueError(
            f"{n} candidate particles cannot supply {k_max} neighbours plus a "
            "self-match with a margin; lower ks or widen the candidate set")
    q = query_rows.to(pos.device, dtype=torch.long)
    if q.numel() and (int(q.min()) < 0 or int(q.max()) >= n):
        # Checked, not left to the indexing: an out-of-range row is an
        # IndexError but a NEGATIVE one indexes from the END and would measure a
        # different particle than the caller named, silently.
        raise ValueError(
            f"query rows must index `pos` exactly once; got the range "
            f"[{int(q.min())}, {int(q.max())}] against {n} candidates")
    rows_per_pass = max(1, min(int(q.numel()),
                               int(cfg.search_max_elems) // max(n, 1)))
    ks = torch.as_tensor(cfg.ks, device=pos.device, dtype=torch.long)

    out = []
    for start in range(0, int(q.numel()), rows_per_pass):
        qi = q[start:start + rows_per_pass]
        d2 = torch.zeros((qi.numel(), n), device=pos.device, dtype=pos.dtype)
        for ax in range(int(pos.shape[1])):
            # One axis at a time: the (rows, N, d) delta is d times the peak of
            # this for no benefit, and only the squared sum is needed. Columns
            # 3+ are velocity/phase_lambda and are NOT wrapped: velocity is not
            # periodic.
            dax = pos[qi, ax][:, None] - pos[None, :, ax]
            if ax < 3:
                dax = _minimum_image(dax, boxsize_mpc_h)
            d2 += dax * dax
        # k_max + 1 because exactly one of the returned rows is the query itself.
        idx = d2.topk(k_max + 1, dim=1, largest=False, sorted=True).indices
        keep = idx != qi[:, None]
        n_self = (~keep).sum(dim=1)
        if not bool((n_self == 1).all()):
            raise ValueError(
                "every query must appear in the candidate set exactly once; "
                f"{int((n_self != 1).sum())} of {qi.numel()} queries did not. "
                "query_rows must index `pos` itself.")
        idx = idx[keep].reshape(qi.numel(), k_max)
        out.append(idx.gather(1, (ks - 1).expand(qi.numel(), ks.numel())))
    return torch.cat(out, dim=0)


def knn_log_distances(
    pos: torch.Tensor, query_rows: torch.Tensor, cfg: KnnConfig, *,
    boxsize_mpc_h: float = 100.0,
    neighbour_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``(n_queries, len(ks))`` of ``log r_k``. **Differentiable in ``pos``.**

    The search returns integers; this recomputes the selected distances on the
    live tensor, so the gradient reaches the query particle and each of its
    chosen neighbours. Pass ``neighbour_ids`` to reuse a search -- e.g. to
    measure a frozen field on the candidate's neighbour selection -- and leave it
    ``None`` to search ``pos`` itself, which is what a training step wants
    (the selection must be re-made as the field moves, or the loss would
    optimise a stale graph).
    """
    ids = (knn_neighbour_ids(pos, query_rows, cfg, boxsize_mpc_h=boxsize_mpc_h)
           if neighbour_ids is None else
           neighbour_ids.to(pos.device, dtype=torch.long))
    qi = query_rows.to(pos.device, dtype=torch.long)
    raw = pos[qi][:, None, :] - pos[ids]
    delta = _minimum_image(raw[..., :3], boxsize_mpc_h)
    if raw.shape[-1] > 3:      # velocity axes: open, so no minimum image
        delta = torch.cat([delta, raw[..., 3:]], dim=-1)
    r = torch.sqrt(delta.square().sum(dim=-1) + EPS_MPC_H ** 2)
    return torch.log(r)


# --------------------------------------------------------------------------- #
# Which particles may be queries
# --------------------------------------------------------------------------- #
def interior_queries(
    pos: torch.Tensor, tile_rows: Sequence[int] | torch.Tensor,
    cfg: KnnConfig, *, tile_centre_mpc_h: torch.Tensor,
    tile_side_mpc_h: float, cell_mpc_h: float, boxsize_mpc_h: float = 100.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw query rows from the tile's Eulerian **interior**, never its faces.

    ``tile_rows`` says which particles may be QUERIES -- one tile's rows of the
    table. It does not restrict the candidates: the neighbour search offers every
    particle in ``pos``, so a batch of tiles helps rather than hinders. What the
    buffer guards against is the material that is in **no** tile of the table. A
    candidate set that is a sub-cube of the box is missing every neighbour
    outside it, and the error that makes is one-signed: ``r_k`` too LARGE,
    because a closer neighbour was simply not offered. That is the same sign as
    the clumping deficit this term charges for, so an unbuffered query set
    manufactures its own signal. Restricting queries to the cube shrunk by
    ``buffer_cells`` leaves every one of them surrounded by real candidates out
    to that radius.

    The cut is in EULERIAN position, not on the Lagrangian lattice: the
    tile-particle table is indexed by Lagrangian site and a particle's
    displacement can carry it clean out of its own tile (and inside a cluster,
    far out). A Lagrangian-interior cut would leave a query that has physically
    moved to a face unbuffered while excluding one that moved inward.
    `subhalos-are-Lagrangian-pure` says MEMBERSHIP is Lagrangian-local; it does
    not say displacement is small.

    So the sampled population is "particles of these tiles now sitting in the
    shrunken cube", which is the right population for a spatial statistic and is
    the same population in both fields. Its one incompleteness is stated rather
    than hidden: a particle that moved INTO the cube from a tile outside the
    table is a valid candidate but never a query. That is a missing sample, not a
    biased distance, and it affects both fields the same way.

    Returns at most ``cfg.n_queries`` rows, drawn without replacement, and fewer
    only if the interior holds fewer. Empty means the buffer swallowed the tile.
    """
    rows = torch.as_tensor(tile_rows, device=pos.device, dtype=torch.long)
    half = 0.5 * float(tile_side_mpc_h) - float(cfg.buffer_cells) * float(
        cell_mpc_h)
    if half <= 0.0:
        return rows[:0]
    c = tile_centre_mpc_h.to(pos.device, dtype=pos.dtype).reshape(1, 3)
    d = _minimum_image(pos[rows].detach()[:, :3] - c, boxsize_mpc_h).abs()
    inside = rows[(d.amax(dim=1) <= half)]
    if inside.numel() <= int(cfg.n_queries):
        return inside
    pick = torch.randperm(int(inside.numel()), device=pos.device,
                          generator=generator)[:int(cfg.n_queries)]
    return inside[pick]


def tile_geometry(
    tile_id: int, *, ng_hr: int = 512, tile_hr: int = 64,
    boxsize_mpc_h: float = 100.0,
) -> Tuple[torch.Tensor, float, float]:
    """``(centre Mpc/h, tile side Mpc/h, cell Mpc/h)`` for a tile id.

    The tile decomposition is :func:`member_gather.tile_particles`'s, digit for
    digit -- ``tid // n_side^2, (tid // n_side) % n_side, tid % n_side`` on the
    half-cell-offset lattice -- so a tile's particles and a tile's bounding cube
    cannot drift apart.
    """
    n_side = int(ng_hr) // int(tile_hr)
    cell = float(boxsize_mpc_h) / float(ng_hr)
    side = float(tile_hr) * cell
    tx = int(tile_id) // (n_side * n_side)
    ty = (int(tile_id) // n_side) % n_side
    tz = int(tile_id) % n_side
    centre = torch.tensor([(tx + 0.5) * side, (ty + 0.5) * side,
                           (tz + 0.5) * side], dtype=torch.float64)
    return centre, side, cell


# --------------------------------------------------------------------------- #
# The loss
# --------------------------------------------------------------------------- #
def _quantile_reduce(z_sorted: torch.Tensor, n_quantiles: int) -> torch.Tensor:
    """Mean ``log r`` per equal-COUNT bin of an already-sorted column stack.

    Equal count, not equal radius, so every bin carries the same weight however
    the distances pile up -- ``shell_reduce``'s rule, and the same reason: a
    bin's weight must not be set by how many particles happened to land in it.
    ``n_quantiles >= n`` is the identity, so a short query list is never charged
    on emptier bins than it has draws.
    """
    n = int(z_sorted.shape[0])
    q = int(n_quantiles)
    if q <= 0 or q >= n:
        return z_sorted
    # Bin b spans rows [b*n//q, (b+1)*n//q): contiguous, exhaustive, and as
    # equal as integers allow without dropping a row.
    edges = [(b * n) // q for b in range(q + 1)]
    return torch.stack([z_sorted[a:b].mean(dim=0)
                        for a, b in zip(edges[:-1], edges[1:])], dim=0)


def knn_cdf_loss(
    z_cand: torch.Tensor, z_hr: torch.Tensor, cfg: KnnConfig,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """``(loss, diagnostics)`` from two ``(n_queries, len(ks))`` log-distance sets.

    Both are sorted down the query axis and reduced to quantile bins, then the
    penalty is ``[|dlog| - log1p(dead_zone)]_+^2`` averaged over bins and ``k``
    (one-sided in the deficit direction when ``cfg.one_sided``). Squared so the
    term is smooth at the dead-zone edge, in log so it is scale-free and a 10%
    error at ``r_8`` weighs the same as a 10% error at ``r_32``.

    ``z_hr`` is a **constant** reference: it is HR's own distribution on HR's own
    neighbour selection, which is the only thing HR's field can be asked for.
    The two query sets need not be the same particles and usually are not -- the
    match is between distributions, and forcing a shared query set would quietly
    turn this back into the per-particle address loss the module docstring
    rejects.

    Diagnostics report, per ``k``, the whole per-bin ratio curve and two
    summaries of it -- and the reason both exist is MEASURED, in
    ``test_loss_sees_a_clump_at_fixed_mean_density``. A field with 120 of 520
    particles in a clump scores a loss of 0.95 against a matched field's 0.0006
    while its MEDIAN log ratio reads 0.002: the clumping lives in the LOW
    quantiles -- the particles that have close neighbours -- and the median bin is
    background. So ``ratio_med`` is reported because it is the number every other
    diagnostic in this line quotes, and ``ratio_q10`` (the mean of the bottom
    decile of bins) is reported because it is the one that moves. A verdict that
    reads only the median of this statistic will call a win a no-op.

    Signs: a ratio above 1 means the candidate's ``k``-th neighbours are FURTHER
    away than HR's, i.e. it is under-clumped at that scale. The loss itself is
    dead-zoned and squared and so is not comparable across configs.
    """
    if z_cand.dim() != 2 or z_hr.dim() != 2:
        raise ValueError("log-distance sets must be (n_queries, len(ks)), got "
                         f"{tuple(z_cand.shape)} and {tuple(z_hr.shape)}")
    if z_cand.shape[1] != z_hr.shape[1]:
        raise ValueError(f"k columns disagree: {z_cand.shape[1]} vs "
                         f"{z_hr.shape[1]}")
    if z_cand.shape[1] != len(cfg.ks):
        raise ValueError(f"{z_cand.shape[1]} columns for {len(cfg.ks)} ks")
    if z_cand.shape[0] < 1 or z_hr.shape[0] < 1:
        raise ValueError("empty query set")

    q = int(cfg.n_quantiles)
    if q <= 0 and z_cand.shape[0] != z_hr.shape[0]:
        raise ValueError(
            f"the full rank-by-rank pairing (n_quantiles=0) needs equal query "
            f"counts, got {z_cand.shape[0]} and {z_hr.shape[0]}. Either bin "
            "into quantiles or draw the same number from both fields.")
    # With binning the two may differ in length: equal-count bins are a
    # quantile estimate, which is what makes the comparison a distribution
    # comparison rather than a rank-by-rank one.
    a = _quantile_reduce(z_cand.sort(dim=0).values, q)
    b = _quantile_reduce(z_hr.sort(dim=0).values.detach(), q)
    if a.shape != b.shape:
        raise ValueError(
            f"quantile reduction gave {tuple(a.shape)} vs {tuple(b.shape)}; "
            "one field has fewer queries than n_quantiles, so its bins are not "
            "the same quantiles. Raise n_queries or lower n_quantiles.")

    dlog = a - b
    dead = torch.log1p(torch.tensor(float(cfg.dead_zone), device=a.device,
                                    dtype=a.dtype))
    excess = (dlog - dead if cfg.one_sided
              else dlog.abs() - dead).clamp_min(0.0)
    loss = excess.square().mean()

    with torch.no_grad():
        med = dlog.median(dim=0).values
        amed = dlog.abs().median(dim=0).values
        # The bottom decile of BINS, which (bins being equal-count) is the bottom
        # decile of queries by r_k: the clumped ones. At least one bin always.
        n_low = max(1, int(dlog.shape[0]) // 10)
        low = dlog[:n_low].mean(dim=0)
    diag = {
        "ks": list(cfg.ks),
        "n_quantiles": int(a.shape[0]),
        "n_queries_cand": int(z_cand.shape[0]),
        "n_queries_hr": int(z_hr.shape[0]),
        # Ratios rather than logs: easier to read, and the sign convention is in
        # the docstring -- >1 is under-clumped at that k.
        "ratio_med": [float(torch.exp(v)) for v in med],
        "ratio_q10": [float(torch.exp(v)) for v in low],
        "abs_dlog_med": [float(v) for v in amed],
        # (n_bins, n_k): the shape of the disagreement, which is what a plot
        # wants and what the two summaries above are lossy reductions of.
        "ratio_bins": torch.exp(dlog).tolist(),
        "loss": float(loss.detach()),
    }
    return loss, diag


# --------------------------------------------------------------------------- #
# Density strata: splitting the query population without a catalog
# --------------------------------------------------------------------------- #
# WHY THIS EXISTS. ``--knn-classes`` is the one place a Rockstar label reaches
# this term's gradient: it restricts the queries to HR's ``sub_50_200``
# particles. It was not a preference, it was measured -- round 5b, the pooled
# form ranks the working arms BACKWARDS (rho +0.91 with subhalo count) because
# host and unbound particles are ~95% of any interior and the working arms
# over-concentrate hosts. The mask fixed the ranking by naming the particles
# that matter.
#
# The label is doing one job there and it is not membership: it is picking out
# the DENSE tail of the query population. That is a property of the particles,
# and each field can read it off its own positions. Stratifying on each field's
# own ``r_{strata_rank}`` does the same split with no owner array, no catalog,
# no parent ids -- which puts this term back to the catalog-free statement the
# module docstring claims for it.
#
# Read ``KnnConfig.strata`` before using it: for the stratifying column itself
# this is provably nothing but finer binning, and every bit of what it adds is
# in the CONDITIONAL distribution of the other ranks.
def strata_partition(z: torch.Tensor, cfg: KnnConfig) -> list:
    """Equal-count query groups, ordered dense-first, from ``z``'s own values.

    ``z`` is a field's ``(n_queries, len(ks))`` log-distance block and the split
    is read from its ``strata_rank`` column, so each field defines its own
    strata from its own particles -- the point of the whole construction. The
    ordering is **detached**, exactly as the neighbour search is and for the same
    reason: an assignment of rows to groups is combinatorial, its derivative is
    zero almost everywhere, and the distances inside each group stay live.

    Equal COUNT, not equal radius: every stratum then carries the same weight
    whatever the density distribution does, which is ``_quantile_reduce``'s rule
    one level up. Group 0 is the densest (smallest ``r``).
    """
    n = int(z.shape[0])
    m = max(int(cfg.strata), 1)
    col = list(cfg.ks).index(int(cfg.strata_rank) or int(min(cfg.ks)))
    order = z[:, col].detach().argsort()
    edges = [(b * n) // m for b in range(m + 1)]
    return [order[a:b] for a, b in zip(edges[:-1], edges[1:])]


def knn_stratified_loss(
    z_cand: torch.Tensor, z_hr: torch.Tensor, cfg: KnnConfig,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """:func:`knn_cdf_loss` charged separately in each self-defined stratum.

    ``cfg.strata <= 1`` is :func:`knn_cdf_loss` itself, so this is the only call
    site the tile loss needs. The charged strata are averaged with equal weight.

    **Partitioning is not reweighting.** Every stratum is measured, but with
    ``strata_charge`` empty every stratum is also charged, and that is
    arithmetically the pooled loss in finer bins -- not an approximation of it,
    the same equal-count bins. See ``KnnConfig.strata_charge`` for the numbers.
    If the intent is to stop hosts and background dominating the term, the
    charge has to be restricted; partitioning alone only sharpens the readout.

    The diagnostics keep ``ratio_med`` / ``ratio_q10`` at their pooled shape
    (one value per ``k``, meaned over strata) so every existing reader keeps
    working, and add the two that are the reason for the term:
    ``ratio_med_dense`` (stratum 0, the densest tenth-or-so of queries) and the
    per-stratum blocks under ``strata``.
    """
    if int(cfg.strata) <= 1:
        return knn_cdf_loss(z_cand, z_hr, cfg)
    gc = strata_partition(z_cand, cfg)
    gh = strata_partition(z_hr, cfg)
    charged = (set(int(j) for j in cfg.strata_charge) if cfg.strata_charge
               else set(range(len(gc))))
    per, blocks = [], []
    for j, (a, b) in enumerate(zip(gc, gh)):
        if int(a.numel()) < 1 or int(b.numel()) < 1:
            blocks.append({"skipped": True, "n_queries_cand": int(a.numel()),
                           "n_queries_hr": int(b.numel())})
            continue
        # Every stratum is MEASURED; only the charged ones reach the gradient,
        # so the diagnostic stays whole whatever the loss is charged on.
        with torch.set_grad_enabled(j in charged and torch.is_grad_enabled()):
            l_s, d_s = knn_cdf_loss(z_cand[a], z_hr[b], cfg)
        d_s["charged"] = bool(j in charged)
        if j in charged:
            per.append(l_s)
        blocks.append(d_s)
    if not per:
        return z_cand.sum() * 0.0, {"ks": list(cfg.ks), "skipped": True,
                                    "strata": blocks}
    loss = torch.stack(per).mean()
    live = [d for d in blocks if not d.get("skipped")]
    diag = {
        "ks": list(cfg.ks),
        "n_quantiles": int(live[0]["n_quantiles"]),
        "n_queries_cand": int(z_cand.shape[0]),
        "n_queries_hr": int(z_hr.shape[0]),
        "n_strata_charged": len(per),
        "strata_charge": [int(j) for j in sorted(charged)],
        "strata_rank": int(cfg.strata_rank) or int(min(cfg.ks)),
        "loss": float(loss.detach()),
        # Stratum 0 is the dense tail -- the population `--knn-classes` was
        # naming with a catalog. If the term is doing anything, it is here.
        "ratio_med_dense": list(live[0]["ratio_med"]),
        "ratio_q10_dense": list(live[0]["ratio_q10"]),
        "strata": blocks,
    }
    for key in ("ratio_med", "ratio_q10", "abs_dlog_med"):
        diag[key] = [float(v) for v in
                     torch.tensor([d[key] for d in live]).mean(dim=0)]
    return loss, diag


# --------------------------------------------------------------------------- #
# Neighbouring material: the z=0 tile is not where its particles are
# --------------------------------------------------------------------------- #
# MEASURED, knn_cdf_set9.json (job 49259): with a tile's own 64^3 particles as
# the only candidates, 43-56% of queries miss one of their 128 nearest neighbours
# and the p95 overestimate of r_128 is +62-80% AT EVERY BUFFER from 0 to 16
# cells -- the buffer does not help. The cause is not the faces: the one HR tile
# checked (the box-centre tile) has median |Psi| 11.2 Mpc/h against a 12.5 Mpc/h
# tile side, so at z=0 a tile
# has moved about its own width and half of the material around any query came
# from OTHER tiles. Two fixes, both below: centre the query cube on where the
# tile's particles ARE (`periodic_centroid`), and offer the other tiles'
# particles as constant candidates (`box_neighbours`).
def periodic_centroid(pos: torch.Tensor, boxsize_mpc_h: float = 100.0
                      ) -> torch.Tensor:
    """Circular mean per axis. Detached: it places a cube, it is not optimised.

    An arithmetic mean of wrapped coordinates is wrong for any tile that
    straddles a box face -- half its particles near 0 and half near L average to
    L/2, the far side of the box. The circular mean is the standard fix and is
    exact for a clump narrower than half the box.
    """
    L = float(boxsize_mpc_h)
    ang = pos.detach()[:, :3].to(torch.float64) * (2.0 * torch.pi / L)
    c = torch.atan2(torch.sin(ang).mean(dim=0), torch.cos(ang).mean(dim=0))
    return torch.remainder(c * (L / (2.0 * torch.pi)), L)


def box_positions(field, *, ng: int = 512, boxsize_mpc_h: float = 100.0,
                  dis_scale_mpc_h: float = 6.0, dtype=np.float32) -> np.ndarray:
    """``(ng^3, 3)`` Eulerian positions of a whole ``(6, ng, ng, ng)`` field.

    Row ``(i * ng + j) * ng + k`` is Lagrangian site ``(i, j, k)``, the ordering
    of :func:`member_gather.tile_particles` and of every owner array. float32 by
    default: this is the source of CONSTANT neighbour candidates and a whole box
    of them is 1.6 GB, where float64 would be 3.2.
    """
    ax = ((np.arange(ng, dtype=np.float64) + 0.5) * boxsize_mpc_h / ng)
    pos = np.empty((ng ** 3, 3), dtype=dtype)
    for c in range(3):
        shape = [1, 1, 1]
        shape[c] = ng
        q = np.broadcast_to(ax.reshape(shape), (ng, ng, ng))
        x = np.asarray(field[c], dtype=np.float64) * dis_scale_mpc_h + q
        pos[:, c] = np.mod(x, boxsize_mpc_h).reshape(-1)
        del x
    pos[pos >= boxsize_mpc_h] -= boxsize_mpc_h
    return pos


#: HR owner classes, in the order ``classify_owner`` returns indices for. The
#: same names and cuts as scripts/features/knn_cdf_split_diagnostic.py, which is
#: where every calibration number for them was measured.
OWNER_CLASSES = ("sub_50_200", "sub_ge200", "sub_lt50", "host", "unbound")


def classify_owner(owner: np.ndarray, ids: np.ndarray, parent_ids: np.ndarray,
                   num_p: np.ndarray, *, chunk: int = 1 << 24) -> np.ndarray:
    """``int8`` index into :data:`OWNER_CLASSES` per particle of an owner array.

    ``owner[i]`` is the id of the innermost halo binding Lagrangian particle
    ``i``, or < 0 for unbound; the three catalog arrays are the HR Rockstar
    catalog's. Chunked, so a 512^3 owner array costs one int64 chunk at a time.
    """
    ids = np.asarray(ids, dtype=np.int64)
    lut_parent = np.full(int(ids.max()) + 1, -2, dtype=np.int64)
    lut_np = np.zeros(int(ids.max()) + 1, dtype=np.int64)
    lut_parent[ids] = np.asarray(parent_ids, dtype=np.int64)
    lut_np[ids] = np.asarray(num_p, dtype=np.int64)
    out = np.full(owner.shape[0], OWNER_CLASSES.index("unbound"), dtype=np.int8)
    for a in range(0, owner.shape[0], chunk):
        o = np.asarray(owner[a:a + chunk], dtype=np.int64)
        b = o >= 0
        if (lut_parent[o[b]] == -2).any():
            raise ValueError("owner array names a halo id absent from the "
                             "catalog; they are from different runs")
        par, npart = lut_parent[o[b]], lut_np[o[b]]
        c = np.full(int(b.sum()), OWNER_CLASSES.index("host"), dtype=np.int8)
        sub = par >= 0
        c[sub & (npart < 50)] = OWNER_CLASSES.index("sub_lt50")
        c[sub & (npart >= 50) & (npart < 200)] = OWNER_CLASSES.index("sub_50_200")
        c[sub & (npart >= 200)] = OWNER_CLASSES.index("sub_ge200")
        seg = out[a:a + chunk]
        seg[b] = c
    return out


def box_phase(field, *, ng: int = 512, boxsize_mpc_h: float = 100.0,
              dis_scale_mpc_h: float = 6.0, vel_scale_kms: float = 1.0,
              phase_lambda: float = 0.0, dtype=np.float32) -> np.ndarray:
    """``(ng^3, 3)`` positions, or ``(ng^3, 6)`` ``(x, v / phase_lambda)`` in Mpc/h.

    The 6D form of :func:`box_positions`, for the constant neighbour material a
    tile needs. float32 and 6 columns is 3.2 GB for a 512^3 box.
    """
    pos = box_positions(field, ng=ng, boxsize_mpc_h=boxsize_mpc_h,
                        dis_scale_mpc_h=dis_scale_mpc_h, dtype=dtype)
    if phase_lambda <= 0:
        return pos
    out = np.empty((ng ** 3, 6), dtype=dtype)
    out[:, :3] = pos
    del pos
    scale = float(vel_scale_kms) / float(phase_lambda)
    for c in range(3):
        out[:, 3 + c] = np.asarray(field[3 + c], dtype=np.float64).reshape(-1) * scale
    return out


def tile_rows_flat(tile_id: int, *, ng: int = 512, tile: int = 64) -> np.ndarray:
    """Flat Lagrangian ids of a tile, in :func:`member_gather.tile_particles` order."""
    n_side = ng // tile
    tx = int(tile_id) // (n_side * n_side)
    ty = (int(tile_id) // n_side) % n_side
    tz = int(tile_id) % n_side
    li = np.arange(tile)
    gi, gj, gk = np.meshgrid(tx * tile + li, ty * tile + li, tz * tile + li,
                             indexing="ij")
    return ((gi * ng + gj) * ng + gk).reshape(-1).astype(np.int64)


def box_neighbours(box_pos: np.ndarray, tile_id: int, centre: np.ndarray,
                   half_extent_mpc_h: float, *, ng: int = 512, tile: int = 64,
                   boxsize_mpc_h: float = 100.0, chunk: int = 1 << 24
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """``(ids, positions)`` of every OTHER tile's particle in a cube about ``centre``.

    The cube is Chebyshev ``half_extent`` under the minimum image. The tile's own
    particles are excluded because the caller holds them live; everything else is
    returned so it can be offered to the search as constants. Chunked so the
    temporary is bounded however large the box.
    """
    L = float(boxsize_mpc_h)
    c = np.asarray(centre, dtype=np.float64).reshape(1, 3)
    own = np.zeros(box_pos.shape[0], dtype=bool)
    own[tile_rows_flat(tile_id, ng=ng, tile=tile)] = True
    ids = []
    for a in range(0, box_pos.shape[0], chunk):
        d = box_pos[a:a + chunk].astype(np.float64) - c
        d -= L * np.round(d / L)
        hit = np.flatnonzero((np.abs(d).max(axis=1) <= half_extent_mpc_h)
                             & ~own[a:a + chunk])
        ids.append(hit + a)
    ids = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64)
    return ids, box_pos[ids]


def knn_tile_loss(
    cand_pos: torch.Tensor, hr_pos: torch.Tensor,
    tile_rows: Sequence[int] | torch.Tensor, tile_id: int, cfg: KnnConfig, *,
    ng_hr: int = 512, tile_hr: int = 64, boxsize_mpc_h: float = 100.0,
    generator: Optional[torch.Generator] = None,
    cand_extra: Optional[torch.Tensor] = None,
    hr_extra: Optional[torch.Tensor] = None,
    centre: str = "lagrangian",
    classes: Optional[torch.Tensor] = None,
    class_ids: Sequence[int] = (),
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """The loss for one tile, given the candidate and HR particle tables.

    Queries are drawn **independently** for the two fields, from each field's own
    Eulerian interior. That is deliberate and it is the buffer argument again: a
    particle that is interior in HR may have moved to a face in the candidate,
    and using HR's interior set on the candidate field would measure the
    candidate on queries it does not buffer. Both sides then estimate the same
    population statistic -- the distribution of ``r_k`` over particles in the
    tile's interior -- each with its own valid sample.

    ``tile_rows`` must index the same Lagrangian sites in both tables, which is
    what :func:`member_gather.tile_particles` guarantees for a shared tile list.

    ``cand_extra`` / ``hr_extra`` are the OTHER tiles' particles near this one
    (:func:`box_neighbours`), appended to each table as constants: candidates,
    never queries. ``centre="centroid"`` places each field's query cube on the
    circular mean of that field's own tile particles instead of the tile's
    Lagrangian centre. See the block comment above for the measurement that makes
    both necessary at z=0.

    ``cfg.strata > 1`` charges the loss per self-defined density stratum
    (:func:`knn_stratified_loss`) instead of once on the whole query set. It is
    the catalog-free alternative to ``classes`` below and composes with it: with
    both on, each named class is stratified inside itself.

    ``classes`` (one label per entry of ``tile_rows``, HR's owner class) with
    ``class_ids`` charges the loss SEPARATELY on each named class and averages
    the classes equally, instead of once on the pooled interior. MEASURED, round
    5b: pooled forms score the arms that make more subhalos as worse, because
    host particles dominate them and the working arms over-concentrate hosts in
    phase space (class term vs subhalo count, rho +0.91). The class-conditioned
    50-200p subhalo term is the one that ranks them correctly (rho -0.65). A
    class whose interior is too small to fill the quantile bins is skipped, and
    the loss is 0 with ``skipped`` only if every named class is.
    """
    if centre not in ("lagrangian", "centroid"):
        raise ValueError(f"centre {centre!r} is not 'lagrangian' or 'centroid'")
    lag_centre, side, cell = tile_geometry(tile_id, ng_hr=ng_hr, tile_hr=tile_hr,
                                           boxsize_mpc_h=boxsize_mpc_h)
    rows = torch.as_tensor(tile_rows, device=cand_pos.device, dtype=torch.long)

    def table(pos, extra):
        if extra is None or extra.shape[0] == 0:
            return pos
        return torch.cat([pos, extra.to(pos.device, pos.dtype).detach()], dim=0)

    def queries(pos, rws):
        # The cube is placed by the WHOLE tile, never by the class subset: a
        # class's own centroid would move the query cube with the thing being
        # measured.
        c = (periodic_centroid(pos[rows], boxsize_mpc_h) if centre == "centroid"
             else lag_centre)
        return interior_queries(pos, rws, cfg, generator=generator,
                                tile_centre_mpc_h=c, tile_side_mpc_h=side,
                                cell_mpc_h=cell, boxsize_mpc_h=boxsize_mpc_h)

    cand_table = table(cand_pos, cand_extra)
    hr_table = table(hr_pos, hr_extra)

    def charge(rws):
        """``(loss, diag)`` on one row subset, or ``None`` if it is too small."""
        qc, qh = queries(cand_pos, rws), queries(hr_pos, rws)
        n_min = int(min(qc.numel(), qh.numel()))
        # Under strata each group gets n_min/strata of the draws, so the bins a
        # STRATUM must fill are what the floor is about, not the tile's total.
        need = max(int(cfg.n_quantiles), 1) * max(int(cfg.strata), 1)
        if n_min < need or n_min < 2:
            return None, {"n_queries_cand": int(qc.numel()),
                          "n_queries_hr": int(qh.numel()), "skipped": True}
        # Equal counts keep the quantile bins identical between the two fields
        # and make n_quantiles=0 legal; the draw is already random, so
        # truncating it is still a uniform sample of the interior.
        qc, qh = qc[:n_min], qh[:n_min]
        # Extras go on AFTER the live rows, so a query row index is unchanged.
        z_c = knn_log_distances(cand_table, qc, cfg, boxsize_mpc_h=boxsize_mpc_h)
        with torch.no_grad():
            z_h = knn_log_distances(hr_table, qh, cfg,
                                    boxsize_mpc_h=boxsize_mpc_h)
        return knn_stratified_loss(z_c, z_h, cfg)

    zero = torch.zeros((), device=cand_pos.device, dtype=cand_pos.dtype)
    if classes is None or not len(class_ids):
        loss, diag = charge(rows)
        if loss is None:
            diag.update({"ks": list(cfg.ks), "skipped": True})
            return zero, diag
    else:
        cls = torch.as_tensor(classes, device=rows.device).reshape(-1)
        if int(cls.numel()) != int(rows.numel()):
            raise ValueError(
                f"classes has {int(cls.numel())} labels for {int(rows.numel())} "
                "tile rows; one label per row of `tile_rows` is required")
        per, names = [], {}
        for cid in class_ids:
            l_c, d_c = charge(rows[cls == int(cid)])
            names[f"class_{int(cid)}"] = d_c
            if l_c is not None:
                per.append(l_c)
        if not per:
            return zero, {"ks": list(cfg.ks), "skipped": True,
                          "class_ids": [int(c) for c in class_ids],
                          "classes": names}
        # Equal weight per class: the round-5b form. A mass weighting is the
        # pooled form again, since hosts and unbound carry nearly all the mass.
        loss = torch.stack(per).mean()
        diag = {"ks": list(cfg.ks), "loss": float(loss.detach()),
                "n_classes_charged": len(per)}
        for key in ("ratio_med", "ratio_q10", "n_queries_cand"):
            vals = [d[key] for d in names.values() if not d.get("skipped")]
            if vals and isinstance(vals[0], list):
                diag[key] = [float(v) for v in torch.tensor(vals).mean(dim=0)]
            elif vals:
                diag[key] = int(sum(vals))
        diag["class_ids"] = [int(c) for c in class_ids]
        diag["classes"] = names
    diag["tile"] = int(tile_id)
    diag["skipped"] = False
    diag["centre"] = centre
    diag["n_extra_cand"] = 0 if cand_extra is None else int(cand_extra.shape[0])
    diag["n_extra_hr"] = 0 if hr_extra is None else int(hr_extra.shape[0])
    return loss, diag
