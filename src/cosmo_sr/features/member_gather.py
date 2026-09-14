"""Gather SR2's particles into true HR subhalos **by particle id**, not in a window.

``docs/sr2_gather_finetune.md`` section 8.2 closed the window objective: the
gather loss matched every statistic it was given -- compact mass 1.06 of HR,
window dispersion 1.02, bulk offset 0.05 sigma -- and measured over the
particles HR actually binds, the tuned field was **indistinguishable from the
frozen generator** (Mann-Whitney AUC 0.47-0.51 on every statistic, median
``sigma_v`` change 1.6%). The loss never moved the objects. The cause is named
there: a Gaussian window at a subhalo inside a cluster holds mostly *host*
material, so the statistic tracks the host, not the satellite.

This module removes the window. Membership comes from HR's ``owner`` array,
which is indexed by **flat Lagrangian id**, and SR2 and HR share that lattice --
so the same id set addresses the same particles in every field and no halo
matching enters anywhere. Gathering is one fancy-index into the generator's
output, which is cheaper than the CIC deposit ``subhalo_gather`` argued for and,
unlike it, is exactly the object.

What the statistics are, and why these
--------------------------------------
Rockstar (a) FOFs at ``b ~ 0.28`` into groups, (b) recurses a 6-D phase-space
FOF inside each group with ``d^2 = |dx|^2/sigma_x^2 + |dv|^2/sigma_v^2`` and
``sigma_x, sigma_v`` **re-measured at every level**, (c) seeds halos at the
deepest level and assigns particles to the nearest seed in that seed's own
normalised metric, then (d) **unbinds** -- drops positive-energy particles and
deletes the candidate if too few remain.

Supervising with the true member sets hands us (a)-(c) for free: those are the
partition steps, they have no clean differentiable analogue, and we are given
their answer. What is left to make differentiable is (d), plus the *local
contrast* half of (b):

``virial``  ``2T/|W|`` in log, two-sided.
    A **pair** statistic -- ``W`` is a sum over all ``N(N-1)/2`` pairs, not a
    moment of a smoothed field. A raised pedestal can match a field moment; it
    cannot match a pair sum. This is the term that carries the optimisation:
    frozen sits 64x from HR here with a large, smooth, everywhere-defined
    gradient.
``bound``   soft unbinding fraction, hinged above HR.
    Rockstar's decision rule itself, ``0.5|v - <v>|^2 + phi < 0`` per particle,
    with the indicator replaced by a sigmoid. N coupled per-particle conditions
    through an ``N^2`` kernel -- by far the smallest level set here. It is the
    *target*, not the driver: see ``_bound_temperature`` for why a fixed
    temperature makes this term numerically dead at the frozen start.
``d6``      6-D compactness against a **local background**, hinged above HR.
    Step (b)'s linking criterion and step (c)'s competition, together. The
    background is real non-member particles near the set, so "members are close
    and cold" cannot be satisfied by cooling the whole neighbourhood -- that
    cools the normaliser too. A window statistic cannot express this at all.
``r_rms``, ``sigma_v``  in log, two-sided.
    Not findings -- they break the ``sigma_v^2 * r_rms = const`` degeneracy that
    ``virial`` alone leaves open. Two-sided deliberately: an over-collapsed,
    over-cooled set is as wrong as a diffuse one, and per
    ``sr2_substructure_module.md`` section 2 item 3 there is no direction in
    which being wrong about velocity is safe.

Every term is normalised to **HR's own value on the same set**, computed by this
same code on the HR tiles at build time, so the loss and its reference can never
disagree about the estimator. Hinged terms contribute exactly zero with zero
gradient once HR is reached, which is ``subhalo_gather``'s property 2 kept.

What this does not do
---------------------
It does not make the objective sufficient. Nothing built from set-level
statistics is: many configurations carry any finite list of them. The claim is
narrower and is about **arity** -- the window loss constrained three 1-particle
moments of a smoothed field, and this constrains N per-particle inequalities
coupled through an ``N^2`` potential plus a pair sum. The gate stays real
Rockstar (``docs/sr2_gather_finetune.md`` section 8.1: ceiling 227, noise +-9,
42/43 against 0/43), and per ``occupation-ratio-is-gameable`` and
``tile-overfit-proxy-exploitation`` the expectation is that any differentiable
form gets gamed unless it does.

Conventions match :mod:`cosmo_sr.features.bound_discriminator` exactly -- comoving
Mpc/h, peculiar km/s, Msun/h, ``G = 4.30091e-9`` -- so
``tests/features/test_member_gather.py`` can pin this module against that one's
numpy on identical inputs. Absolute energies are softening-dependent and are not
physics; what is read is the comparison between fields at one softening.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .subhalo_gather import ENV_LABELS

G_MPC_KMS2_PER_MSUN = 4.30091e-9

__all__ = [
    "MemberGatherConfig",
    "MemberSets",
    "SetTerms",
    "build_member_sets",
    "member_gather_loss",
    "set_statistics_torch",
    "specific_potential_torch",
    "tile_particles",
    "unwrap_about",
]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MemberGatherConfig:
    """Selection, estimator and weights. Frozen so a run's config is a value."""

    # --- which sets ---------------------------------------------------------
    min_num_p: int = 200
    #: Upper edge of the ``num_p`` window, exclusive. 0 is no ceiling and is the
    #: default -- every arm recorded so far ran without one, because ``"sub"``
    #: never selects the cluster host. It stops being optional the moment
    #: ``population`` admits top-level halos: see :data:`population`.
    max_num_p: int = 0
    #: WHICH halos this loss is responsible for -- ``"sub"`` (``parent_ids >=
    #: 0``, every arm so far), ``"top"``, or ``"all"``.
    #:
    #: ``"sub"`` encodes an assumption that was never tested: that the halos SR2
    #: is missing are the ones Rockstar files inside a bigger halo's radius. The
    #: held-out gate says otherwise -- in the edited region HR resolves 3,775
    #: top-level halos >=200p against base's 3,028, a **747-halo deficit that is
    #: entirely outside hosts** and that no supervised set in any arm has ever
    #: pointed at. ``"all"`` selects on size alone and lets the parent split fall
    #: where it falls, which is the right shape for that deficit if it turns out
    #: to be a small-halo deficit rather than a substructure one.
    #: ``scripts/features/halo_gap_census.py`` is the measurement that decides.
    #:
    #: With ``"top"`` or ``"all"``, :attr:`max_num_p` is what keeps the selection
    #: away from the cluster itself. The pair sums are ``O(N^2)`` and a 1e14 host
    #: is ~100k particles -- 1e10 pairs against a 200p set's 2e4 -- and its HR
    #: reference asks the generator to concentrate a cluster, which is a
    #: different objective from this one and is already handled, against the
    #: FROZEN reference, by the host preservation guard.
    population: str = "sub"
    min_purity: float = 0.5
    min_live_frac: float = 0.5
    max_sets: int = 256
    #: How :attr:`max_sets` chooses when it binds. ``"largest"`` keeps the
    #: biggest sets and is the default because it is what every recorded arm
    #: ran; it is also exactly wrong for a small-halo run, where it would drop
    #: the whole population the run was created to build. ``"stratified"``
    #: fills log-``n_sites`` bins round-robin, so the cap thins the mass function
    #: instead of truncating it; ``"random"`` is the unbiased-but-mass-weighted
    #: middle option (the pool is dominated by small sets, so it lands close to
    #: stratified without the binning). All three are deterministic in
    #: :attr:`bg_seed`.
    set_select: str = "largest"      # "largest" | "random" | "stratified"
    #: WHICH environment the supervised sets are drawn from -- ``"any"`` (the
    #: default and every recorded arm), or a comma list of
    #: ``subhalo_gather.ENV_LABELS``: ``inside`` (<1 R_vir of a bigger resolved
    #: host), ``near`` (1-3, the infall region), ``field`` (>3, isolated).
    #:
    #: Why a THIRD selection axis, after population and mass. *Measured*, set9
    #: census at region scope, the small-halo arm against base and HR:
    #:
    #: ====== ========== ========== ========== ==========
    #: env    base       smallhalo  HR         closed
    #: ====== ========== ========== ========== ==========
    #: near   1,899      2,148      8,456      249 of 6,557 (3.8%)
    #: field  11,119     12,382     14,046     overshoots HR in every sub bin
    #: ====== ========== ========== ========== ==========
    #:
    #: 56% of the region's deficit is in ``near`` and the arm put its budget in
    #: ``field``, where it then overran the truth (field subhalos 3,316 against
    #: HR's 2,466, and 2-6x too many above 200p). Selecting by size alone cannot
    #: fix that, because the pool of a cluster region is dominated by field
    #: halos and ``stratified`` stratifies on MASS. ``near`` sits inside a
    #: cluster's own Lagrangian tiles, so unlike ``field`` it is reachable by
    #: changing the selection rather than the tiling.
    env_select: str = "any"
    #: Particle floor for a halo to count as a "resolved host" when labelling
    #: another halo's environment. 200 is the census's, and the number every
    #: quoted host count in this line uses.
    env_host_floor: int = 200
    #: Subsample a set to at most this many particles per evaluation; 0 is off
    #: and is the default, because this changes the ESTIMATOR and a silent
    #: estimator change is invisible in every metric. On, it is unbiased in
    #: ``phi`` (the pair sum is rescaled by ``(N-1)/(K-1)``) and therefore in
    #: ``2T/|W|``, and only mildly biased in ``bound_soft``, which is a
    #: surrogate already. It exists because the pair sums are ``O(N^2)`` and a
    #: handful of massive satellites carry most of that cost while being the
    #: objects SR2 already builds correctly.
    max_set_particles: int = 0

    # --- estimator ----------------------------------------------------------
    softening_mpc_h: float = 0.01
    softening_kind: str = "plummer"      # "plummer" | "clamp"
    pot_chunk: int = 2048
    #: Cap on one pair block's element count. ``pot_chunk`` alone is NOT a
    #: memory bound -- see :func:`specific_potential_torch` -- and a single
    #: 118,000-particle satellite makes ``chunk=2048`` a 2.7 GiB allocation,
    #: which is what killed the 2026-08-23 rung ladder (jobs 35746-35748). The
    #: rows per pass are ``min(pot_chunk, pot_max_elems // N)``, so the block is
    #: bounded in elements rather than in rows and the cost of a set is linear
    #: in its size instead of a step function at the OOM.
    pot_max_elems: int = 1 << 24
    bound_tau: float = 0.5
    bound_temperature: str = "adaptive"  # "adaptive" | "hr"
    #: How a boundness DEFICIT is charged. ``hinge`` is ``[1 - x/ref]_+^2``, the
    #: form every run so far used, and it is **capped at 1 by construction**
    #: because ``x >= 0``. That cap is not a statement about importance -- it is
    #: an accident of which side of its reference the frozen field starts on --
    #: and it is measured: at step 0 of the 2026-08-24 pool the weighted terms
    #: were d6 111.8, virial 14.1, **bound 0.31**, rrms 0.61, sigmav 0.36. The
    #: term carrying Rockstar's own decision rule held **0.24%** of the budget,
    #: and after 500 steps that collapsed d6 32x and virial 4x it had moved only
    #: 0.31 -> 0.21.
    #:
    #: ``log`` is ``[log(ref/x)]_+^2``: unbounded as ``x -> 0``, still exactly
    #: zero for MEETING OR BEATING the reference -- the anti-over-sharpening
    #: property the hinge exists for (``pilot_steps_2_4.md`` step 4) -- and
    #: scale-free, so it is the same form ``virial``, ``r_rms`` and ``sigma_v``
    #: already use. At the measured starting ratio of 1.3% of reference it
    #: charges ~19 instead of ~0.97.
    bound_penalty: str = "hinge"         # "hinge" | "log"

    # --- local background ---------------------------------------------------
    bg_k: int = 4096
    bg_radius_factor: float = 4.0
    bg_seed: int = 0

    # --- isolation: how much NON-MEMBER mass sits on top of the object -------
    #: Weight on the local background OCCUPANCY, hinged above HR. 0 is off and
    #: is the default; every arm recorded in
    #: ``docs/sr2_member_gather_training.md`` ran without it.
    #:
    #: What it is for, and why no existing term can say it. Every other term in
    #: this module is a moment of the members about their OWN centroid, or that
    #: centroid's address. Rockstar's top-level/subhalo decision is not a
    #: property of the members at all: it is whether some *other*, more massive
    #: halo's virial radius contains this centre. So an intra-set loss pointed
    #: at top-level HR halos gets full marks for an object that a pedestal then
    #: swallows -- which is exactly what the small-halo arm did. *Measured*,
    #: set9 region scope, ``--population all`` against base: it built 3,374 net
    #: objects and Rockstar filed **every one of them as a subhalo** (subs
    #: +4,260, top-level -886), so the top-level deficit the arm exists to close
    #: went the wrong way, 3,872 -> 4,758.
    #:
    #: ``d6`` is the only existing term that looks outside the set and it does
    #: not close this: it normalises by the background's *dispersion*, not its
    #: density, so a set can be maximally compact against its neighbourhood's
    #: position and velocity spread while sitting in a growing mass pile.
    #:
    #: This term charges the non-member mass inside :attr:`iso_radius_frac` of
    #: the background sphere, against HR's value on the **same particle ids** --
    #: so the background subsample fraction cancels exactly and the comparison
    #: is id-matched rather than field-matched. Hinged ABOVE: costs nothing for
    #: matching or beating HR, which keeps the module's standing property that
    #: the loss can never ask for more than HR has.
    #:
    #: It is the same lever against the re-parenting damage measured in
    #: `host-damage-is-mostly-reparenting` (~88% of the ``self`` arm's host loss
    #: was objects crossing the parent split, not particles lost) and against
    #: the radial over-concentration that killed the shell arm -- those are this
    #: same emergent pedestal, seen from the host's side. A *placement* loss on
    #: the supervised sets could not reach it; a *density* loss on the material
    #: around them can.
    w_isolation: float = 0.0
    #: The counting radius, as a fraction of the background sphere
    #: (``bg_radius_factor * r_ref``, with ``r_ref`` the HR set's rms radius).
    #: 0.5 keeps the sphere comfortably inside the sampled one -- the background
    #: ids are chosen ONCE on the frozen field within ``bg_radius_factor``
    #: radii, so material arriving from further out is invisible to this term,
    #: and a counting radius at the sampling edge would be counting an edge.
    #: Widen ``bg_radius_factor`` rather than this if the sphere is too small.
    iso_radius_frac: float = 0.5
    #: Sigmoid width of the soft indicator, as a fraction of the counting
    #: radius. The count has to be differentiable in the background particles'
    #: positions or the term has no gradient at all; 0.15 is soft enough to
    #: carry one everywhere in the shell and sharp enough that the count means
    #: what it says.
    iso_softness: float = 0.15
    #: Minimum HR soft count for a set to be CHARGED at all. A set whose HR
    #: reference holds a handful of background particles has a ratio dominated
    #: by shot noise, and hinging on it would charge the run for the background
    #: sampler's seed. Such sets are skipped and counted in ``n_iso``, exactly
    #: as ``d6`` skips sets with no background at all.
    iso_min_count: float = 8.0

    # --- weights ------------------------------------------------------------
    w_virial: float = 1.0
    w_bound: float = 1.0
    w_d6: float = 1.0
    w_rrms: float = 0.3
    w_sigmav: float = 0.3
    #: Pin the set's centre of mass. NOT optional, and its absence was measured:
    #: every other term is an internal moment about the set's OWN centroid, so
    #: without this the loss says what the object must look like and nothing
    #: about where it must be. The 2026-08-21 free-field run built bound objects
    #: (156 subhalos in R_vir against a base of 11) that landed a median
    #: 0.414 Mpc/h from their targets, against a 0.150 Mpc/h search radius --
    #: 8/154 recovered, with 96% of misses having no halo of ANY mass inside the
    #: radius. Position, not mass, was the dominant failure.
    w_centre: float = 1.0
    #: Radii inside which the centre term costs exactly nothing. The gate is a
    #: THRESHOLD at one search radius (``compare_gather_catalog``'s
    #: ``max(r_vir, 0.15)``), so cost paid at 0.1 radii buys nothing there and
    #: competes with every other term for the same gradient. 0 keeps the pure
    #: quadratic the 72/154 run used.
    centre_dead_zone: float = 0.0
    #: Radii beyond which the centre term becomes linear rather than quadratic.
    #: Frozen sets sit a median 5.6 search radii out, where a quadratic charges
    #: 31 and the worst sets own the batch gradient. 0 keeps the pure quadratic.
    centre_huber_radii: float = 0.0
    #: WHICH offset the centre term charges for. ``full`` is the whole vector to
    #: the reachable HR centroid -- the term the 72/154 run used, and the only
    #: one of the three that specifies an address.
    #:
    #: The other two exist because the address may not be learnable. *Measured*,
    #: ``centre_offset/pool/offsets.json`` over 7,560 supervised sets: the offset
    #: is 62.9% radial by variance against an isotropic null of 1/3, with
    #: ``o_par`` negative for 70% of sets -- a signed, systematic infall deficit
    #: shared across objects, which is a rule. The remaining transverse 37% has
    #: no direction to predict from and a linear fit on (d_host, num_p, M_host)
    #: explains only 11.5% of the whole offset out of sample.
    #:
    #: ``radial`` charges only ``|o . rhat|``, dropping the part of the target no
    #: input determines. ``self`` re-anchors to the set's own FROZEN centroid --
    #: "concentrate where SR2 already put you" -- which costs exactly zero at
    #: step 0 and asks for no address at all.
    #:
    #: Both are strictly weaker specifications and the arithmetic bounds them
    #: before any GPU does: a PERFECT ``radial`` leaves the transverse residual,
    #: a median 2.92 search radii, so 12.3% of sets land inside the gate's one
    #: radius; a perfect ``self`` leaves the frozen offset, 5.65 radii and 3.6%.
    #: Against ``full``, which leaves zero by construction. They are worth
    #: running as a GENERATOR objective, where the address is unavailable; they
    #: cannot beat ``full`` in the free field, which sees it.
    #: ``self_tan`` is ``self`` with the RADIAL direction removed: it charges
    #: only the transverse part of the offset from the frozen centroid. It has
    #: no meaning on its own -- alone it is a weaker ``self`` -- and exists for
    #: exactly one composition, with ``w_radius`` below. ``self`` pins a set to
    #: where the frozen field put it, *including its host-centric radius*, which
    #: is the coordinate the radius term is trying to move; ``self_tan`` hands
    #: that one coordinate to the radius term and keeps the anchor on the two
    #: the radius term says nothing about. The two terms are then orthogonal by
    #: construction rather than competing (see ``w_radius``).
    centre_mode: str = "full"      # "full" | "radial" | "self" | "self_tan"

    # --- the shell: right distance from the host, any direction -------------
    #: Weight on ``| |x_bar - x_host| - |x_target - x_host| |`` -- the set's
    #: host-centric RADIUS, with the direction left free. 0 is off and is the
    #: default; every arm recorded in ``docs/sr2_member_gather_training.md`` ran
    #: without it.
    #:
    #: Why a radius and not an address: *measured*, on the 7,560-set pool in
    #: ``centre_offset/pool/offsets.json``, the radius error is **60% of the
    #: whole centre offset** by magnitude, and it is the half that is a rule
    #: rather than noise -- 58.3% of frozen sets sit too CLOSE to their host
    #: (median ratio 0.974, median error 2.80 search radii, p90 8.18), against
    #: a transverse residual whose direction a linear fit on (d_host, num_p,
    #: M_host) explains 11.5% of out of sample. This term asks for the part
    #: that is predictable and stays silent about the part that is not.
    #:
    #: What it is FOR: the tuned ``self`` arm's subhalos are radially collapsed.
    #: On held-out set9 (``subhalo_radial/set9/summary.json``) its subhalos sit
    #: at a median 0.35 host R_vir against HR's 1.13, with **6,496 inside
    #: 0.25 R_vir where HR has 593** and 932 in 0.75-1.0 R_vir where HR has
    #: 3,846. Too many in the middle, not enough anywhere else.
    #:
    #: It CONFLICTS with ``centre_mode="self"``, and not marginally. That mode's
    #: minimum is the frozen centroid, radius included; this term's minimum is
    #: the true shell, a median 2.80 radii away. Both quadratic, a radial step
    #: ``delta`` costs ``w_centre (delta/s)^2 + w_radius ((|dr| - delta)/s)^2``,
    #: so the two settle at ``delta* = |dr| w_radius / (w_centre + w_radius)``:
    #: at equal weights the run closes exactly HALF the radius error it was
    #: added to close, and pays the centre term for the half it does close.
    #: ``centre_mode="self_tan"`` removes the competition instead of outbidding
    #: it -- the anchor keeps the two transverse coordinates, this term owns the
    #: radial one, and ``delta* = |dr|``.
    w_radius: float = 0.0
    #: Search radii of slack: the term costs exactly nothing while the radius is
    #: within this of the truth. The objective is a SHELL, not an address, and
    #: it is not worth spending gradient on the last fraction of a radius when
    #: the gate is a threshold at one -- and the frozen error is a median 2.80
    #: radii, so a dead zone of 1 leaves 79% of sets still charged.
    radius_dead_zone: float = 1.0
    #: Search radii beyond which the radius term is linear rather than
    #: quadratic, continuous in value and slope. Same argument as
    #: ``centre_huber_radii``: the p90 set is 8.2 radii out, where a quadratic
    #: charges 67 and the tail owns the batch gradient.
    radius_huber_radii: float = 0.0

    # --- the soft shell: match the host's radius DISTRIBUTION ---------------
    #: Weight on a 1-D optimal-transport match between the candidate and target
    #: host-centric radii of one host's sets: sort both, pair by RANK, and
    #: charge the log ratio. 0 is off and is the default.
    #:
    #: Why, and it is the post-mortem of ``w_radius`` above. That term charges
    #: each set against *its own* target radius -- a per-object address. On the
    #: 7,560-set pool, a linear probe on (log ``d_host``, log ``num_p``, log
    #: ``Mvir``, log ``n_live``) fit on train predicts that address with an
    #: **out-of-sample R^2 of 0.022** in the term's own units (0.015 in log
    #: units). Section 14.1's "the radius is the half that is a rule rather than
    #: noise" is true of the AGGREGATE -- 58.3% of frozen sets sit too close,
    #: median ratio 0.974 -- and false per set. Those are different claims and
    #: ``w_radius`` was built on the wrong one.
    #:
    #: The ``all_blocks_selfshell`` run is that stated as a trajectory: train
    #: ``radius_err_radii`` 2.860 -> 1.318 (memorised), held-out 2.678 -> 2.808
    #: (the address is not there), held-out ``radius_ratio`` 0.9744 -> 0.9872 --
    #: so the ONLY component that transferred was the multiplicative one. It
    #: cost 55% of the abundance: host subhalos 367 (``self``) -> 167, against
    #: HR's 374.
    #:
    #: This term asks for that component and nothing else. Measured on the same
    #: pool, against the identity pairing (train / holdout):
    #:
    #: * median ``|log ratio|`` 0.164 / 0.143 -> **0.079 / 0.064** -- the sort
    #:   halves the magnitude but the SIGNED median, the inward bias that is the
    #:   whole signal, survives intact: -0.026 / -0.022 -> -0.041 / -0.026.
    #: * mean square 0.345 / 0.320 -> 0.071 / 0.061, i.e. the monotone map
    #:   removes 79% of it. A per-host constant rescale would remove **1.9%**
    #:   (the between-host share of the variance; 98.1% is within-host), so what
    #:   this term fixes is the SHAPE of the radial distribution, not its mean.
    #:
    #: What it deliberately does not buy: a PERFECT distribution match still
    #: leaves 79% (train) / 85% (holdout) of the per-set address error standing.
    #: That is the point -- see ``median_shell_resid``, which is the diagnostic
    #: to read, because ``radius_err_radii`` will barely move on total success.
    #:
    #: Pairs with ``centre_mode="self_tan"`` for the same reason ``w_radius``
    #: does: the anchor holds the two transverse coordinates and this term owns
    #: the radial one. Unlike ``w_radius`` it never names a set's own target, so
    #: its minimum is a DISTRIBUTION -- reachable by sets exchanging radii along
    #: their own radial lines. Per-host ``spearman(r_frozen, r_true)`` is 0.92
    #: median (0.62 worst) and 31% of sets sit more than 0.1N from their own
    #: target rank, so that freedom is real and is why both diagnostics are
    #: logged: distribution matched with the addresses rearranged is a different
    #: outcome from both of them being right, and only the pair can tell.
    w_shell: float = 0.0
    #: Free zone, as a FRACTIONAL radius error: the term costs nothing while
    #: the sorted candidate radius is within this of the sorted target. Applied
    #: in log space as ``log1p(shell_dead_zone)``, so the knob means what it
    #: says. 0.02 charges 81% / 78% of sets (train / holdout) against a sorted
    #: median of 0.079 / 0.064; the 0.05 that ``radius_dead_zone``'s reasoning
    #: would suggest charges only 61% / 55% and eats most of a signal half the
    #: size of the one that number was set against.
    shell_dead_zone: float = 0.02
    #: Floor on both radii before the log, Mpc/h. On the target it is a plain
    #: clamp (a constant; min ``r_true`` in the pool is 0.0047 and 0.119% sit
    #: below 0.05). On the CANDIDATE it is a softened norm, ``sqrt(|d|^2 +
    #: eps^2)`` rather than ``clamp_min``: a clamp is a gradient PLATEAU below
    #: the floor, and the failure this term exists to prevent is collapse
    #: toward the host centre -- the one place a plateau must not be. It does
    #: not bind on the frozen field (the smallest frozen radius in a host is a
    #: median 0.142, min 0.041 Mpc/h); it is there for where the run can go.
    shell_eps_mpc_h: float = 1e-2
    #: How many QUANTILE BINS the sorted pairing is reduced to before it is
    #: charged. 0 (default) keeps the full rank-by-rank pairing, which is what
    #: every arm through 2026-09-08 ran.
    #:
    #: Why the knob exists: the full pairing hands the loss one degree of
    #: freedom per set (~134 for a big host), and MEASURED on 2026-09-08 the
    #: w=30 arm drove the train residual 7.3x while holdout moved 1.4x -- it was
    #: fitting per-host detail that did not transfer. The low-order shape of the
    #: radius distribution is the part a shared operator can plausibly carry;
    #: the high-rank-resolution wiggle is the part it can only memorise. Binning
    #: the sorted arrays into Q bins and charging the bins' mean log radius
    #: keeps the first and drops the second, so Q is a direct dial on "ask only
    #: for what is predictable".
    #:
    #: Bins are equal-COUNT over the sorted arrays, not equal-radius, so each
    #: carries the same weight however the radii pile up. Q >= the number of
    #: sets is the full pairing again (one set per bin), so a small host is
    #: never charged on emptier bins than it has sets.
    shell_quantiles: int = 0

    def __post_init__(self) -> None:
        if self.softening_kind not in ("plummer", "clamp"):
            raise ValueError(f"softening_kind {self.softening_kind!r} "
                             "is not 'plummer' or 'clamp'")
        if self.bound_penalty not in ("hinge", "log"):
            raise ValueError(f"bound_penalty {self.bound_penalty!r} is not "
                             "'hinge' or 'log'")
        if self.bound_temperature not in ("adaptive", "hr"):
            raise ValueError(f"bound_temperature {self.bound_temperature!r} "
                             "is not 'adaptive' or 'hr'")
        if not 0.0 < self.min_live_frac <= 1.0:
            raise ValueError(f"min_live_frac {self.min_live_frac} not in (0, 1]")
        if self.max_set_particles and self.max_set_particles < 3:
            raise ValueError(
                f"max_set_particles {self.max_set_particles} is below 3; the "
                "pair-sum rescaling divides by (K - 1) and a 2-particle sample "
                "carries no information about a subhalo. Use 0 to disable.")
        if self.population not in ("sub", "top", "all"):
            raise ValueError(f"population {self.population!r} is not 'sub', "
                             "'top' or 'all'")
        if self.max_num_p and self.max_num_p <= self.min_num_p:
            raise ValueError(
                f"max_num_p {self.max_num_p} is not above min_num_p "
                f"{self.min_num_p}; the window is empty. Use 0 for no ceiling.")
        if self.set_select not in ("largest", "random", "stratified"):
            raise ValueError(f"set_select {self.set_select!r} is not 'largest', "
                             "'random' or 'stratified'")
        if self.centre_mode not in ("full", "radial", "self", "self_tan"):
            raise ValueError(f"centre_mode {self.centre_mode!r} is not "
                             "'full', 'radial', 'self' or 'self_tan'")
        if self.w_radius < 0.0:
            raise ValueError(f"w_radius {self.w_radius} is negative; the term "
                             "would PAY for landing on the wrong shell")
        if self.w_shell < 0.0:
            raise ValueError(f"w_shell {self.w_shell} is negative; the term "
                             "would PAY for the wrong radius distribution")
        if self.shell_dead_zone < 0.0:
            raise ValueError(
                f"shell_dead_zone {self.shell_dead_zone} is negative")
        if self.shell_eps_mpc_h <= 0.0:
            raise ValueError(
                f"shell_eps_mpc_h {self.shell_eps_mpc_h} is not positive; it "
                "floors a radius before a log and softens a norm at zero")
        if self.shell_quantiles < 0:
            raise ValueError(
                f"shell_quantiles {self.shell_quantiles} is negative; use 0 "
                "for the full rank-by-rank pairing")
        if self.radius_dead_zone < 0.0:
            raise ValueError(
                f"radius_dead_zone {self.radius_dead_zone} is negative")
        if self.radius_huber_radii and (
                self.radius_huber_radii <= self.radius_dead_zone):
            raise ValueError(
                f"radius_huber_radii {self.radius_huber_radii} is not above "
                f"radius_dead_zone {self.radius_dead_zone}; the linear arm "
                "would start inside the dead zone and the term would be "
                "discontinuous.")
        if self.centre_huber_radii and (
                self.centre_huber_radii <= self.centre_dead_zone):
            raise ValueError(
                f"centre_huber_radii {self.centre_huber_radii} is not above "
                f"centre_dead_zone {self.centre_dead_zone}; the linear arm "
                "would start inside the dead zone and the term would be "
                "discontinuous.")


# --------------------------------------------------------------------------- #
# Geometry: the generator's tiles as a flat particle table
# --------------------------------------------------------------------------- #
def tile_particles(
    field: torch.Tensor, tiles: Sequence[int], *,
    ng_hr: int = 512, tile_hr: int = 64, boxsize_mpc_h: float = 100.0,
    dis_scale_mpc_h: float = 6.0, vel_scale_kms: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(pos Mpc/h, vel km/s)``, each ``(B * tile_hr^3, 3)``, differentiable.

    Row ``b * tile_hr^3 + (i * tile_hr + j) * tile_hr + k`` is the particle whose
    Lagrangian site is local cell ``(i, j, k)`` of ``tiles[b]``. That ordering is
    what :func:`build_member_sets` indexes into, and the two must not drift
    apart, so both derive it from this one expression.

    The reconstruction is ``x = (q + Psi) mod L`` on the half-cell-offset lattice
    -- the same one :func:`cosmo_sr.eval.particles.field_to_particles` and
    ``bound_discriminator.particles_at`` use, so a set gathered here and the same
    set measured there land on the same positions to float precision.

    ``dis_scale_mpc_h`` and ``vel_scale_kms`` are the *undo* factors of
    ``preprocess_srs.disnorm`` / ``velnorm`` at the run's redshift; passing them
    in rather than importing keeps this a pure function of arrays. The caller
    gets them from ``disnorm(1.0, z, undo=True) * 1e-3`` and
    ``velnorm(1.0, z, undo=True)``.
    """
    if field.dim() != 5 or field.shape[1] < 6:
        raise ValueError(f"expected (B, 6, T, T, T), got {tuple(field.shape)}")
    b, t = int(field.shape[0]), int(field.shape[-1])
    if t != int(tile_hr):
        raise ValueError(f"field tile {t} != tile_hr {tile_hr}")
    if b != len(tiles):
        raise ValueError(f"field batch {b} != {len(tiles)} tiles")

    n_side = int(ng_hr) // int(tile_hr)
    cell = float(boxsize_mpc_h) / float(ng_hr)
    dev, dt = field.device, field.dtype

    ar = torch.arange(t, device=dev, dtype=dt)
    li, lj, lk = torch.meshgrid(ar, ar, ar, indexing="ij")
    local = torch.stack([li, lj, lk], dim=0)                    # (3, T, T, T)

    origins = []
    for tid in tiles:
        tid = int(tid)
        tx, ty, tz = tid // (n_side * n_side), (tid // n_side) % n_side, tid % n_side
        origins.append([tx * t, ty * t, tz * t])
    org = torch.tensor(origins, device=dev, dtype=dt)            # (B, 3)

    q = (org[:, :, None, None, None] + local[None] + 0.5) * cell  # (B,3,T,T,T)
    pos = q + field[:, 0:3] * float(dis_scale_mpc_h)
    pos = torch.remainder(pos, float(boxsize_mpc_h))
    vel = field[:, 3:6] * float(vel_scale_kms)

    flat = lambda x: x.permute(0, 2, 3, 4, 1).reshape(-1, 3)     # noqa: E731
    return flat(pos), flat(vel)


def _rows_for_ids(ids: np.ndarray, tiles: Sequence[int], *,
                  ng_hr: int, tile_hr: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(rows, keep)``: table rows for the ids inside ``tiles``, and which.

    Inverse of :func:`tile_particles`'s ordering, and the only other place that
    ordering is written down. ``keep`` indexes ``ids``, so the caller can tell
    which members are *live* (in a trained tile, hence movable) and which are
    not -- the rest sit at frozen positions and are constants in the loss.
    """
    pid = np.asarray(ids, dtype=np.int64)
    ng, th = int(ng_hr), int(tile_hr)
    n_side = ng // th
    gx, gy, gz = pid // (ng * ng), (pid // ng) % ng, pid % ng
    tid = ((gx // th) * n_side + (gy // th)) * n_side + (gz // th)

    order = {int(t): b for b, t in enumerate(tiles)}
    lut = np.full(n_side ** 3, -1, dtype=np.int64)
    for t, b in order.items():
        lut[t] = b
    bat = lut[tid]
    keep = np.flatnonzero(bat >= 0)
    loc = (((gx[keep] % th) * th) + (gy[keep] % th)) * th + (gz[keep] % th)
    return bat[keep] * (th ** 3) + loc, keep


# --------------------------------------------------------------------------- #
# Energetics
# --------------------------------------------------------------------------- #
def unwrap_about(pos: torch.Tensor, ref: torch.Tensor,
                 boxsize_mpc_h: float) -> torch.Tensor:
    """Put a periodic clump on the branch containing ``ref``. Differentiable.

    ``ref`` must be a **constant** -- here the subhalo's HR catalog centre, which
    is field-independent, so the candidate, the frozen and the HR fields all
    unwrap onto the same branch and their statistics stay comparable. A
    per-field centroid would put them on different branches near a box face; a
    centroid that moves with the optimisation would drag the branch cut with it
    and put a discontinuity in the loss. A fixed reference leaves the cut half a
    box from every particle in the set, where ``d/dx remainder(x, L) = 1`` holds
    unambiguously.
    """
    L = float(boxsize_mpc_h)
    return ref + torch.remainder(pos - ref + 0.5 * L, L) - 0.5 * L


def _pair_rows(n: int, chunk: int, max_elems: int) -> int:
    """Rows per pass. Bounded in ELEMENTS, which is what actually allocates."""
    if n <= 0:
        return 1
    return max(1, min(int(chunk), int(max_elems) // int(n)))


class _SpecificPotential(torch.autograd.Function):
    """``phi`` with an analytic backward that recomputes each pair block.

    **Why this is not a plain chunked loop.** It was one, and the loop's comment
    -- "chunked over rows so the ``N x N`` block never materialises" -- is true
    of the forward pass and false of the tape. Autograd saves ``d`` (``c x N x
    3``) and ``inv`` (``c x N``) for *every* chunk until backward, so peak memory
    was ``O(sum_s N_s^2)`` no matter what ``pot_chunk`` was set to: ~20 bytes a
    pair, which at the measured ``sum_n_squared = 8.8e8`` is ~18 GB, and 21 GB
    is what the 2026-08-23 rung ladder reported before dying on three GPUs
    (jobs 35746, 35747 on 24 GB; 35748 on 48 GB).

    Recomputing the block in backward makes the live memory one chunk instead of
    the whole tape, for one extra pass over the pairs. The backward is written
    out rather than left to autograd because the expression collapses:

        phi_i = -Gm sum_{j != i} u_ij,        u_ij = (r_ij^2 + eps^2)^{-1/2}

        dL/dx_a = Gm sum_j (g_a + g_j) u_aj^3 d_aj,      d_aj = x_a - x_j

    symmetric in the two ways ``x_a`` enters (as the field point of ``phi_a`` and
    as a source in every other ``phi_i``), which is why one pass over the rows
    computes the whole gradient. The ``a = j`` term vanishes because
    ``d_aa = 0``, so the self-pair needs no masking here -- unlike the forward,
    where ``u_aa = 1/eps`` is the largest entry in its row.
    """

    @staticmethod
    def forward(ctx, pos, gm, eps, clamp, chunk, max_elems):
        n = int(pos.shape[0])
        ctx.save_for_backward(pos)
        ctx.gm, ctx.eps, ctx.clamp = float(gm), float(eps), bool(clamp)
        ctx.chunk, ctx.max_elems = int(chunk), int(max_elems)
        out = pos.new_zeros(n)
        if n < 2:
            return out
        rows = _pair_rows(n, chunk, max_elems)
        for a in range(0, n, rows):
            b = min(a + rows, n)
            d = pos[a:b, None, :] - pos[None, :, :]
            r2 = (d * d).sum(dim=-1)
            del d
            if clamp:
                inv = 1.0 / torch.clamp(torch.sqrt(r2.clamp_min(1e-30)), min=eps)
            else:
                inv = torch.rsqrt(r2 + eps * eps)
            # The self term is exactly 1/eps under BOTH kinds (clamp sends r = 0
            # to eps; Plummer gives (0 + eps^2)^{-1/2}), so it is subtracted as a
            # constant instead of zeroing a diagonal -- which used to cost a
            # second N^2 tensor for the `.clone()` the in-place write needed.
            out[a:b] = -gm * (inv.sum(dim=1) - 1.0 / eps)
        return out

    @staticmethod
    def backward(ctx, grad_phi):
        pos, = ctx.saved_tensors
        n = int(pos.shape[0])
        grad = torch.zeros_like(pos)
        if n < 2:
            return grad, None, None, None, None, None
        gm, eps, clamp = ctx.gm, ctx.eps, ctx.clamp
        g = grad_phi
        rows = _pair_rows(n, ctx.chunk, ctx.max_elems)
        for a in range(0, n, rows):
            b = min(a + rows, n)
            d = pos[a:b, None, :] - pos[None, :, :]        # (c, N, 3)
            r2 = (d * d).sum(dim=-1)                        # (c, N)
            if clamp:
                r = torch.sqrt(r2.clamp_min(1e-30))
                # 1/max(r, eps) is FLAT inside the softening, so its gradient is
                # zero there -- which is exactly the pairs an optimiser is
                # trying to create, and the reason "clamp" is a pinning kind and
                # not the training default.
                k = torch.where(r > eps, r.pow(-3.0), torch.zeros_like(r))
            else:
                k = (r2 + eps * eps).pow(-1.5)
            w = (g[a:b, None] + g[None, :]) * k              # (c, N)
            # bmm, not `(w[..., None] * d).sum(1)`: the broadcast form allocates
            # another (c, N, 3) temporary, which is the largest tensor here.
            grad[a:b] = gm * torch.bmm(w.unsqueeze(1), d).squeeze(1)
        return grad, None, None, None, None, None


def specific_potential_torch(
    pos: torch.Tensor, particle_mass_msun_h: float, *,
    softening_mpc_h: float = 0.01, kind: str = "plummer", chunk: int = 2048,
    max_elems: int = 1 << 24,
) -> torch.Tensor:
    """``phi_i`` in ``(km/s)^2``, ``(N,)``. Differentiable in ``pos``.

    ``kind="clamp"`` reproduces :func:`bound_discriminator.specific_potential`
    bit for bit -- ``1/max(r, eps)`` -- and is what the tests pin against. It has
    **zero gradient** for any pair inside the softening, which is precisely the
    pairs an optimiser is trying to create, so it is not the training default.

    ``kind="plummer"`` uses ``1/sqrt(r^2 + eps^2)``: the same softening scale,
    smooth everywhere, and monotone in ``r`` all the way to zero. The two agree
    to better than 1% for pairs beyond ``3 eps`` and the sets here have
    ``r_rms ~ 50 eps``, so the choice moves absolute energies slightly and
    changes no comparison.

    Memory is one pair block -- ``min(chunk, max_elems // N)`` rows -- rather
    than the whole ``N x N`` tape; see :class:`_SpecificPotential` for why that
    distinction is not academic.
    """
    return _SpecificPotential.apply(
        pos, G_MPC_KMS2_PER_MSUN * float(particle_mass_msun_h),
        float(softening_mpc_h), kind == "clamp", int(chunk), int(max_elems))


# --------------------------------------------------------------------------- #
# Per-set statistics
# --------------------------------------------------------------------------- #
@dataclass
class SetTerms:
    """One set's statistics on one field. Tensors, each a 0-dim scalar."""

    n: int
    r_rms: torch.Tensor
    sigma_v: torch.Tensor
    virial: torch.Tensor
    bound_soft: torch.Tensor
    bound_hard: torch.Tensor
    d6: Optional[torch.Tensor] = None
    #: Soft COUNT of local non-member particles inside the isolation
    #: radius -- the only statistic here that is about what SURROUNDS the
    #: object rather than about the object. ``None`` when there is no
    #: background or no radius was supplied.
    bg_occ: Optional[torch.Tensor] = None

    def detached(self) -> Dict[str, float]:
        d = {"n": float(self.n)}
        for k in ("r_rms", "sigma_v", "virial", "bound_soft", "bound_hard",
                  "d6", "bg_occ"):
            v = getattr(self, k)
            d[k] = float("nan") if v is None else float(v.detach())
        return d


def set_statistics_torch(
    pos: torch.Tensor, vel: torch.Tensor, *,
    particle_mass_msun_h: float,
    cfg: MemberGatherConfig,
    bound_scale: Optional[torch.Tensor] = None,
    bg_pos: Optional[torch.Tensor] = None,
    bg_vel: Optional[torch.Tensor] = None,
    pot_mass_factor: float = 1.0,
    iso_radius: float = 0.0,
) -> SetTerms:
    """Every differentiable statistic for one **already unwrapped** member set.

    ``T = sum 0.5 m |v - <v>|^2`` and ``W = 0.5 m sum_i phi_i``; the half undoes
    the pair double count so ``2T/|W| = 1`` is virial equilibrium. Identical
    algebra to :func:`bound_discriminator.set_statistics`, which is what makes
    the HR reference built by this module comparable to that module's tables.

    ``bound_scale`` is the sigmoid temperature; ``None`` means derive it
    adaptively (see :func:`_bound_temperature`). ``bg_pos``/``bg_vel`` are the
    local non-member background; without them ``d6`` is ``None``.

    ``iso_radius`` (Mpc/h) turns on ``bg_occ``, the soft count of background
    particles inside that radius of the set's centroid. The centroid is
    **detached**: the term is meant to push the surrounding non-member mass
    out to where HR has it, not to let the set run away from the pile, which
    is what a live centroid would also permit and what would put it in a
    tug-of-war with the centre term (the mistake ``w_radius`` made against
    ``centre_mode='self'``, docs section 14).

    ``pot_mass_factor`` scales the mass **inside the pair sum only**. It is
    ``(N - 1) / (K - 1)`` when the set has been subsampled to ``K`` of its ``N``
    particles, which makes ``phi_i`` an unbiased estimate of the full set's
    ``phi_i`` -- and since ``T`` and ``W`` are then both sums over the same ``K``
    samples, the ``K/N`` cancels in ``2T/|W|`` and the virial ratio is unbiased
    too. It must NOT touch ``T``, or the ratio would be rescaled twice.
    """
    n = int(pos.shape[0])
    m = float(particle_mass_msun_h)

    dx = pos - pos.mean(dim=0, keepdim=True)
    dv = vel - vel.mean(dim=0, keepdim=True)
    # The +1e-12 is not cosmetic: a degenerate set pools to exactly zero, where
    # d/dx sqrt(x) is infinite and the backward pass returns NaN -- the same
    # failure, and the same fix, as `phase_space.py`'s dispersions.
    r_rms = torch.sqrt((dx * dx).sum(dim=1).mean() + 1e-12)
    ke = 0.5 * (dv * dv).sum(dim=1)
    sigma_v = torch.sqrt(2.0 * ke.mean() + 1e-12)

    phi = specific_potential_torch(
        pos, m * float(pot_mass_factor), softening_mpc_h=cfg.softening_mpc_h,
        kind=cfg.softening_kind, chunk=int(cfg.pot_chunk),
        max_elems=int(cfg.pot_max_elems))
    w = 0.5 * m * phi.sum()
    t = m * ke.sum()
    virial = 2.0 * t / w.abs().clamp_min(1e-30)

    energy = ke + phi
    tau = (_bound_temperature(energy, cfg) if bound_scale is None
           else bound_scale.clamp_min(1e-12))
    bound_soft = torch.sigmoid(-energy / tau).mean()
    bound_hard = (energy < 0).to(pos.dtype).mean()

    d6 = None
    if bg_pos is not None and bg_pos.shape[0] > 1:
        bdx = bg_pos - bg_pos.mean(dim=0, keepdim=True)
        bdv = bg_vel - bg_vel.mean(dim=0, keepdim=True)
        sx = torch.sqrt((bdx * bdx).sum(dim=1).mean() + 1e-12)
        sv = torch.sqrt((bdv * bdv).sum(dim=1).mean() + 1e-12)
        d6 = torch.sqrt((r_rms / sx) ** 2 + (sigma_v / sv) ** 2 + 1e-12)

    bg_occ = None
    if bg_pos is not None and bg_pos.shape[0] > 0 and float(iso_radius) > 0.0:
        r_iso = float(iso_radius)
        # The sphere is centred where the object actually is, so the term
        # tracks the object rather than a fixed point in space -- but
        # detached, so the gradient reaches the BACKGROUND particles only.
        c = pos.mean(dim=0).detach()
        d = torch.sqrt(((bg_pos - c) ** 2).sum(dim=1) + 1e-12)
        w = max(float(cfg.iso_softness) * r_iso, 1e-9)
        bg_occ = torch.sigmoid((r_iso - d) / w).sum()

    return SetTerms(n=n, r_rms=r_rms, sigma_v=sigma_v, virial=virial,
                    bound_soft=bound_soft, bound_hard=bound_hard, d6=d6,
                    bg_occ=bg_occ)


def _bound_temperature(energy: torch.Tensor,
                       cfg: MemberGatherConfig) -> torch.Tensor:
    """The sigmoid width for the soft unbinding test. Detached, by design.

    A **fixed** temperature makes this term numerically dead at the start, and
    not marginally: frozen SR2's supervised sets carry ``ke ~ 5e5 (km/s)^2``
    against a monopole binding scale ``G N m / r_rms ~ 3e3``, so ``energy/tau``
    is O(300) and ``sigmoid(-300)`` is exactly 0 in float32 -- zero value, zero
    gradient, for every particle in every set. That is the whole reason
    ``bound_frac`` reads 0.000 in the section 8.2 table rather than something
    small: it is saturated, not merely low.

    So the default scales the width to the set's *current* energy spread and
    detaches it. The term always has gradient, and as the set tightens the
    temperature shrinks with it, sharpening the surrogate toward the true
    indicator -- a continuation, not a fixed relaxation. ``bound_hard`` is
    reported alongside at every eval so the surrogate is never mistaken for it.
    """
    return (float(cfg.bound_tau) * energy.abs().mean()).detach().clamp_min(1e-12)


# --------------------------------------------------------------------------- #
# The supervised sets
# --------------------------------------------------------------------------- #
@dataclass
class MemberSets:
    """The true HR subhalos these tiles' particles are responsible for.

    Built **once**, before the loop, from the HR catalog, HR's owner array and
    the HR field. Data in the loss exactly as HR displacements are data in an
    MSE: the generator's inputs are unchanged and nothing here is read at
    inference. The supervision is in-sample by construction and no claim about
    generalisation is available from a run that uses it.
    """

    halo_id: np.ndarray                  # (S,)
    num_p: np.ndarray                    # (S,) catalog size
    #: (S,) the catalog's ``parent_id`` -- ``< 0`` for a top-level halo. Carried
    #: so an eval can split "did the arm build this" by population: with
    #: ``population="all"`` the pool holds both, and an aggregate that mixes a
    #: satellite with an isolated field halo cannot say which of the two the run
    #: actually learned. ``-1`` everywhere for a pool built before this field
    #: existed is not possible -- it is always written by
    #: :func:`build_member_sets`.
    parent_id: np.ndarray                # (S,)
    #: (S,) index into ``subhalo_gather.ENV_LABELS`` -- which environment
    #: each set's halo sits in, measured on the HR catalog at build time.
    #: Carried for the same reason as ``parent_id``: an aggregate that
    #: pools an infall-region satellite with an isolated field halo cannot
    #: say which of the two the run learned, and the set9 census says those
    #: two bins are where the arm succeeded and failed respectively.
    #: All ``-1`` when no environment was supplied.
    env: np.ndarray                      # (S,)
    #: (S,) the isolation term's counting radius, Mpc/h. Derived from each
    #: set's own background sphere at build time and therefore a constant of
    #: the data, not of the optimisation. 0 where the term cannot apply.
    iso_radius: np.ndarray               # (S,)
    n_live: np.ndarray                   # (S,) members inside the trained tiles
    live_rows: List[torch.Tensor]        # (S,) long, rows of the particle table
    fixed_pos: List[Optional[torch.Tensor]]   # (S,) frozen positions, detached
    fixed_vel: List[Optional[torch.Tensor]]
    bg_rows: List[Optional[torch.Tensor]]     # (S,) local non-member background
    centre_ref: torch.Tensor             # (S, 3) unwrap branch, HR catalog centre
    centre_target: torch.Tensor          # (S, 3) reachable centroid, on that branch
    centre_scale: torch.Tensor           # (S,) max(r_vir, 0.15) Mpc/h
    #: (S, 3) unit vector from the HOST to this set's target, on the same
    #: unwrap branch. The direction ``centre_mode="radial"`` projects onto.
    #: All-zero when no host position was supplied, which the loss refuses.
    centre_rhat: torch.Tensor
    #: (S, 3) the set's centroid under the FROZEN field -- the same particle
    #: collection ``_gather_one`` builds, live rows plus frozen stragglers, so
    #: ``centre_mode="self"`` costs exactly zero at step 0 by construction.
    centre_self: torch.Tensor
    #: (S, 3) the HOST's centre, on this set's unwrap branch. ``centre_rhat`` is
    #: its direction and loses its length; the radius term needs the length, so
    #: the point is kept rather than the ray. All-zero when no host position was
    #: supplied -- which the loss refuses for ``w_radius > 0``, exactly as it
    #: refuses ``centre_mode="radial"`` for an all-zero ``centre_rhat``.
    centre_host: torch.Tensor
    ref: Dict[str, torch.Tensor]         # HR's value of every statistic
    particle_mass_msun_h: float
    boxsize_mpc_h: float

    @property
    def n_sets(self) -> int:
        return int(self.halo_id.shape[0])

    def to(self, device) -> "MemberSets":
        """A copy with every tensor on ``device``; numpy fields are shared.

        Needed once the fine-tune supervises many hosts: building the sets costs
        one 537 MB owner-array load and a CSR inversion per box, so a multi-host
        run builds them once, keeps the pool on the CPU, and moves one host's
        sets to the GPU per step. ``torch.save`` of a CPU-side ``MemberSets`` is
        then a valid on-disk cache, which is what makes the pool build a separate
        resumable job rather than a preamble to every training run.

        Cheap and non-destructive: tensors already on ``device`` are returned by
        ``Tensor.to`` unchanged, so re-moving a host that is already resident
        allocates nothing.
        """
        import dataclasses

        def _t(x):
            return None if x is None else x.to(device)

        return dataclasses.replace(
            self,
            live_rows=[t.to(device) for t in self.live_rows],
            fixed_pos=[_t(t) for t in self.fixed_pos],
            fixed_vel=[_t(t) for t in self.fixed_vel],
            bg_rows=[_t(t) for t in self.bg_rows],
            centre_ref=self.centre_ref.to(device),
            centre_target=self.centre_target.to(device),
            centre_scale=self.centre_scale.to(device),
            centre_rhat=self.centre_rhat.to(device),
            centre_self=self.centre_self.to(device),
            centre_host=self.centre_host.to(device),
            ref={k: v.to(device) for k, v in self.ref.items()},
        )


def _gather_one(pos: torch.Tensor, vel: torch.Tensor, sets: MemberSets, s: int,
                *, cap: int = 0,
                ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """One set's phase-space coordinates: live rows plus frozen stragglers.

    A member whose Lagrangian site is outside the trained tiles cannot be moved
    by this run and **will not be moved by the splice either** -- the spliced box
    keeps the frozen field everywhere outside those tiles. Dropping such members
    would compute the potential of a fraction of the object and overstate how
    bound it is; including them at frozen coordinates as constants is what the
    halo finder will actually see, so that is what the loss sees.

    ``cap`` subsamples the set uniformly to at most that many particles and
    returns the ``(N - 1) / (K - 1)`` factor the pair sum needs to stay unbiased.
    The draw is fresh on every call by design: a fixed subsample would supervise
    a fixed sub-object, which is a different and much weaker statement than
    supervising the object with a noisy estimator.
    """
    p = pos[sets.live_rows[s]]
    v = vel[sets.live_rows[s]]
    fp, fv = sets.fixed_pos[s], sets.fixed_vel[s]
    if fp is not None and fp.shape[0] > 0:
        p = torch.cat([p, fp.to(p.dtype)], dim=0)
        v = torch.cat([v, fv.to(v.dtype)], dim=0)
    factor = 1.0
    n = int(p.shape[0])
    k = int(cap)
    if k and n > k >= 3:
        take = torch.randperm(n, device=p.device)[:k]
        p, v = p[take], v[take]
        factor = (n - 1.0) / (k - 1.0)
    return unwrap_about(p, sets.centre_ref[s], sets.boxsize_mpc_h), v, factor


#: The six terms, in the order they are weighted. Used by the budget
#: normalisation so a caller cannot pass a scale for a term that does not exist.
TERM_NAMES = ("virial", "bound", "d6", "rrms", "sigmav", "centre",
              "radius", "shell", "isolation")


def member_gather_loss(
    pos: torch.Tensor, vel: torch.Tensor, sets: MemberSets,
    cfg: MemberGatherConfig, set_indices: Optional[Sequence[int]] = None,
    *, term_scale: Optional[Mapping[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """``(loss, diagnostics)`` over every supervised set.

    Five moment terms, each normalised to HR's own value on the same set, and
    two placement terms:

    ==========  ========  ================================================
    term        shape     rationale
    ==========  ========  ================================================
    virial      2-sided   the driver; a pair sum, 64x from HR at the start
    bound       hinged    Rockstar's decision rule; the target
    d6          hinged    6-D contrast against real local non-members
    r_rms       2-sided   breaks the sigma_v^2 * r_rms degeneracy
    sigma_v     2-sided   likewise, and velocity is never safe to get wrong
    centre      softened  WHERE the object is; `centre_mode` picks the offset
    radius      softened  how far from the HOST it is; direction left free
    shell       softened  the DISTRIBUTION of those distances; no addresses
    isolation   hinged    non-member mass ON it; the only INTER-set term
    ==========  ========  ================================================

    Hinged terms are ``[1 - x/x_HR]_+^2`` (bound) or ``[x/x_HR - 1]_+^2`` (d6):
    reaching HR contributes exactly zero with zero gradient, so the loss cannot
    ask for more than HR has and the step-4 over-sharpening failure of
    ``docs/pilot_steps_2_4.md`` is unreachable. Two-sided terms are
    ``log(x/x_HR)^2``, scale-free and symmetric in over- and under-shoot.

    The python loop over sets is deliberate. There are tens of them, each op is
    tiny, and the alternative -- padding to the largest set for a batched
    ``N^2`` -- wastes an order of magnitude of memory on a 200-particle set
    padded to 3000. Cost is ~1% of a generator step.

    ``term_scale`` divides each term by a constant before the weights are
    applied, so the DECLARED weights become the ACTUAL budget. Without it the
    split is set by the terms' incidental dynamic ranges: *measured*, step 0 of
    the 2026-08-24 pool ran d6 111.8, virial 14.1, bound 0.31, rrms 0.61,
    sigmav 0.36 at weights 1, 1, 1, 0.3, 0.3 -- so ``d6`` held 88% of the budget
    and ``bound``, which is the gate's own criterion, held 0.24%. Passing each
    term's step-0 value makes every term start at 1.0 and the weights mean what
    they say. It also removes ``d6``'s head start, which is the same fact stated
    the other way round: d6 collapses 32x in 250 steps because it starts 11x
    above its reference, and those are the steps ``bound`` most needs.

    Scales are constants, not parameters: they are measured once on the frozen
    field and never updated, so the objective stays fixed during the run.

    ``set_indices`` restricts the loop to a subset, which is how a training step
    minibatches over sets rather than taking all of a host's ~134 every time.
    The loss is a MEAN over whatever it evaluates, so the gradient scale does not
    depend on how many were drawn and the learning rate stays comparable. Leave
    it ``None`` for an eval: a median over a random third of the sets is not the
    same number twice.
    """
    dev = pos.device
    zero = torch.zeros((), device=dev, dtype=pos.dtype)
    if sets.n_sets == 0:
        return zero, {"n_sets": 0, "rows": []}

    if term_scale is not None:
        bad = sorted(set(term_scale) - set(TERM_NAMES))
        if bad:
            raise ValueError(f"term_scale has unknown terms {bad}; "
                             f"expected a subset of {list(TERM_NAMES)}")

    acc = {k: zero.clone() for k in TERM_NAMES}
    offsets: List[float] = []
    penalised: List[float] = []
    radius_err: List[float] = []
    radius_ratio: List[float] = []
    # The shell's two sides. Floats always, for the diagnostic -- it is reported
    # at ``w_shell=0`` too, exactly as ``radius_ratio`` is at ``w_radius=0``, so
    # the finished arms can be re-read against this one. Live tensors only when
    # the term is on, which is what keeps ``w_shell=0`` byte-identical to them.
    shell_now: List[float] = []
    shell_tgt: List[float] = []
    shell_now_t: List[torch.Tensor] = []
    shell_tgt_t: List[torch.Tensor] = []
    shell_curves: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    rows: List[Dict[str, float]] = []
    iso_ratio: List[float] = []
    n_d6 = 0
    n_iso = 0

    if cfg.centre_mode in ("radial", "self_tan") and float(
            sets.centre_rhat.abs().sum()) == 0.0:
        raise ValueError(
            f"centre_mode={cfg.centre_mode!r} splits the offset along the "
            "clustercentric direction, but every centre_rhat is zero -- "
            "build_member_sets was called without host_pos. Pass it, or the "
            "term would silently charge for something other than what it "
            "names (for 'radial', identically zero -- a w_centre=0 run under "
            "another name; for 'self_tan', the full self offset).")
    has_host = float(sets.centre_host.abs().sum()) != 0.0
    if float(cfg.w_radius) > 0.0 and not has_host:
        raise ValueError(
            "w_radius > 0 charges for the set's distance from its HOST, but "
            "every centre_host is zero -- build_member_sets was called without "
            "host_pos. Pass it, or the term would measure distance from the "
            "origin of the box. Host preservation sets are built host_pos=None "
            "by design (a host IS the centre) and must run w_radius=0.")
    if float(cfg.w_shell) > 0.0 and not has_host:
        raise ValueError(
            "w_shell > 0 matches the DISTRIBUTION of the sets' distances from "
            "their HOST, but every centre_host is zero -- build_member_sets "
            "was called without host_pos. Pass it, or the term would match a "
            "distribution of distances from the origin of the box. Host "
            "preservation sets are built host_pos=None by design and must run "
            "w_shell=0.")

    idx = (list(range(sets.n_sets)) if set_indices is None
           else [int(i) for i in set_indices])
    if not idx:
        return zero, {"n_sets": 0, "n_sets_total": sets.n_sets, "rows": []}

    for s in idx:
        p, v, pot_factor = _gather_one(pos, vel, sets, s,
                                       cap=int(cfg.max_set_particles))
        bg = sets.bg_rows[s]
        bp = bv = None
        if bg is not None and bg.numel() > 1:
            bp = unwrap_about(pos[bg], sets.centre_ref[s], sets.boxsize_mpc_h)
            bv = vel[bg]

        scale = (sets.ref["bound_scale"][s]
                 if cfg.bound_temperature == "hr" else None)
        st = set_statistics_torch(
            p, v, particle_mass_msun_h=sets.particle_mass_msun_h, cfg=cfg,
            bound_scale=scale, bg_pos=bp, bg_vel=bv,
            pot_mass_factor=pot_factor,
            # ALWAYS measured, whatever is being penalised. The count is a
            # sum over <=bg_k particles against a step dominated by O(N^2)
            # pair blocks, so the diagnostic is free -- and an arm run at
            # w_isolation=0 that cannot report its pedestal is exactly the
            # blind spot the four velocity-guard arms had.
            iso_radius=float(sets.iso_radius[s]))

        acc["virial"] = acc["virial"] + _log2(st.virial, sets.ref["virial"][s])
        acc["rrms"] = acc["rrms"] + _log2(st.r_rms, sets.ref["r_rms"][s])
        acc["sigmav"] = acc["sigmav"] + _log2(st.sigma_v, sets.ref["sigma_v"][s])
        acc["bound"] = acc["bound"] + _hinge_below(
            st.bound_soft, sets.ref["bound_soft"][s], cfg.bound_penalty)
        # Where the object is. Normalised by the same radius the Rockstar gate
        # matches on -- max(r_vir, 0.15 Mpc/h), `compare_gather_catalog`'s
        # `min_radius` -- so a unit of this term is exactly one search radius and
        # driving it below 1 is driving the target into the hit criterion.
        cbar = p.mean(dim=0)
        # `off` is ALWAYS the full offset to the reachable HR centroid, in every
        # mode, because that is the quantity the Rockstar gate scores. What the
        # loss CHARGES for is `pen`, which the mode chooses. Reporting the two
        # separately is the point: a `radial` run that drives its own term to
        # zero while `off` stays at 3 radii has satisfied its objective and
        # missed the gate, and only a diagnostic that keeps both can say so.
        off = torch.linalg.vector_norm(cbar - sets.centre_target[s])
        if cfg.centre_mode == "radial":
            pen = torch.abs((cbar - sets.centre_target[s])
                            @ sets.centre_rhat[s])
        elif cfg.centre_mode == "self":
            pen = torch.linalg.vector_norm(cbar - sets.centre_self[s])
        elif cfg.centre_mode == "self_tan":
            # The self offset with its radial component projected out, so the
            # anchor holds the two coordinates `w_radius` says nothing about and
            # leaves the one it owns free. `centre_rhat` is taken at the TARGET
            # and is a constant, so this is a fixed plane, not a projection onto
            # a direction that moves with the optimisation variable.
            doff = cbar - sets.centre_self[s]
            pen = torch.linalg.vector_norm(
                doff - (doff @ sets.centre_rhat[s]) * sets.centre_rhat[s])
        else:
            pen = off
        acc["centre"] = acc["centre"] + _centre_cost(pen / sets.centre_scale[s],
                                                     cfg)
        # The SHELL: right distance from the host, direction free. Charged on
        # the radius error rather than on a vector, so a set is free to sit
        # anywhere on the sphere its truth defines -- the transverse direction
        # is the 37% of the offset that no input predicts (`w_radius`), and a
        # term that demanded it would be asking for realisation noise.
        if has_host:
            r_now = torch.linalg.vector_norm(cbar - sets.centre_host[s])
            r_true = torch.linalg.vector_norm(sets.centre_target[s]
                                              - sets.centre_host[s])
            dr = r_now - r_true
            if float(cfg.w_radius) > 0.0:
                acc["radius"] = acc["radius"] + _radius_cost(
                    dr.abs() / sets.centre_scale[s], cfg)
            radius_err.append(float(dr.detach()))
            radius_ratio.append(float((r_now / r_true.clamp_min(1e-12)).detach()))
            # The shell's candidate radius is a SOFTENED norm, not `r_now`
            # clamped: `sqrt(|d|^2 + eps^2)` has no plateau, and a plateau at
            # small radius is a zero gradient exactly where this term's whole
            # purpose is to push back (see `shell_eps_mpc_h`).
            eps2 = float(cfg.shell_eps_mpc_h) ** 2
            r_soft = torch.sqrt(((cbar - sets.centre_host[s]) ** 2).sum() + eps2)
            r_tgt = r_true.clamp_min(float(cfg.shell_eps_mpc_h))
            shell_now.append(float(r_soft.detach()))
            shell_tgt.append(float(r_tgt.detach()))
            if float(cfg.w_shell) > 0.0:
                shell_now_t.append(r_soft)
                shell_tgt_t.append(r_tgt)
        else:
            # Host preservation sets, and the unit tests that pass no host. NaN
            # rather than a number measured from the box origin: a plotted zero
            # would read as "the shell is already right".
            radius_err.append(float("nan"))
            radius_ratio.append(float("nan"))
        offsets.append(float(off.detach()))
        penalised.append(float(pen.detach()))
        if st.d6 is not None and torch.isfinite(sets.ref["d6"][s]):
            acc["d6"] = acc["d6"] + _hinge_above(st.d6, sets.ref["d6"][s])
            n_d6 += 1
        # ISOLATION. Hinged above HR's own count on the SAME background ids,
        # so the subsample fraction cancels and "as clean as HR" costs zero.
        # Sets whose HR reference is too sparse to be a ratio are skipped
        # rather than charged, and `n_iso` says how many survived -- an arm
        # whose `n_iso` is a fraction of `n_sets` is not running this term
        # on the pool it thinks it is.
        iso_ref = sets.ref["bg_occ"][s]
        iso_r = float("nan")
        if (st.bg_occ is not None and torch.isfinite(iso_ref)
                and float(iso_ref) >= float(cfg.iso_min_count)):
            acc["isolation"] = acc["isolation"] + _hinge_above(st.bg_occ,
                                                              iso_ref)
            iso_r = float((st.bg_occ / iso_ref).detach())
            n_iso += 1
        # NaN, not skipped: the list stays aligned with `rows`, which is what
        # lets the by-population and by-environment splits below be taken on
        # the same objects the per-set table reports.
        iso_ratio.append(iso_r)

        r = st.detached()
        r["halo_id"] = int(sets.halo_id[s])
        r["num_p"] = int(sets.num_p[s])
        r["n_live"] = int(sets.n_live[s])
        # Population and environment on every row, so an aggregate can be
        # SPLIT rather than only pooled. `MemberSets.parent_id` existed for
        # exactly this and nothing read it, which is why the small-halo arm
        # could not say whether its TOP-LEVEL sets -- the whole point of it
        # -- had improved, and had to wait ten hours for a Rockstar gate to
        # find out that they had not.
        r["parent_id"] = int(sets.parent_id[s])
        r["env"] = int(sets.env[s])
        r["iso_ratio"] = iso_r
        r["centre_offset_radii"] = float(off.detach()
                                         / sets.centre_scale[s].detach())
        rows.append(r)

    # THE SOFT SHELL. One sort per call, over this host's sets in this step:
    # pair the sorted candidate radii with the sorted target radii by RANK --
    # the 1-D optimal transport map, exact and O(N log N) -- and charge the log
    # ratio. No set is ever named against its OWN target, so the term asks for
    # the radius distribution and stays silent about the address `w_radius`
    # could not predict (out-of-sample R^2 0.022; see `w_shell`).
    #
    # Pooling the sets of one host is safe because they are ID-GATHERED: the
    # candidate set `i` and target set `i` hold the same particle rows and so
    # the same `num_p`. The mass distribution is identical on both sides by
    # construction, and a rank pairing cannot put the wrong masses at the wrong
    # radii the way it could on two independently selected populations.
    if shell_now_t:
        lc, lt = shell_reduce(torch.stack(shell_now_t),
                              torch.stack(shell_tgt_t), cfg)
        acc["shell"] = shell_charge(lc, lt, cfg)
        # The candidate side of the reduction, kept live for the caller. A
        # noise-AVERAGED shell (`--shell-noise-draws`) must charge the mean of
        # these curves over the draws, which is a different quantity from the
        # mean of the per-draw charges and is the one that cannot be satisfied
        # by any single realisation -- so the caller needs the curve, not the
        # scalar. `lt` rides along because it is the same target either way.
        shell_curves = (lc, lt)

    ns = float(len(idx))
    # `shell` is absent: it is already a mean over the sorted pairs.
    for k in ("virial", "bound", "rrms", "sigmav", "centre", "radius"):
        acc[k] = acc[k] / ns
    acc["d6"] = acc["d6"] / max(n_d6, 1)
    acc["isolation"] = acc["isolation"] / max(n_iso, 1)

    weight = {"virial": float(cfg.w_virial), "bound": float(cfg.w_bound),
              "d6": float(cfg.w_d6), "rrms": float(cfg.w_rrms),
              "sigmav": float(cfg.w_sigmav), "centre": float(cfg.w_centre),
              "radius": float(cfg.w_radius), "shell": float(cfg.w_shell),
              "isolation": float(cfg.w_isolation)}
    # A term already at zero would divide by ~0 and dominate everything after
    # it, so the scale is floored. Reaching the floor means the term was already
    # satisfied on the frozen field and has nothing to contribute either way.
    scale = {k: max(float((term_scale or {}).get(k, 1.0)), 1e-6)
             for k in TERM_NAMES}
    loss = zero.clone()
    for k in TERM_NAMES:
        loss = loss + weight[k] * acc[k] / scale[k]

    # The live (differentiable) reduced shell curve, for a caller that wants to
    # average it over noise draws before charging. None when the term is off.
    # Not a float: this is the one diag entry that is deliberately a tensor.
    diag_curves = shell_curves
    diag: Dict[str, object] = {"n_sets": len(idx), "n_sets_total": sets.n_sets,
                               "n_d6": n_d6, "n_iso": n_iso,
                               "rows": rows}
    # The isolation ratio, reported at w_isolation=0 too: it is the number
    # that says whether the pedestal is there, and an arm that cannot see
    # it cannot be read against one that can. > 1 is "more non-member mass
    # sitting on this object than HR has", which is the state that makes
    # Rockstar file it as somebody's subhalo.
    _iso = np.asarray(iso_ratio, dtype=np.float64)
    _isog = _iso[np.isfinite(_iso)]
    diag["median_iso_ratio"] = float(np.median(_isog)) if _isog.size else float("nan")
    diag["frac_iso_above_hr"] = (float(np.mean(_isog > 1.0)) if _isog.size
                                 else float("nan"))
    diag["by_population"] = _split_rows(rows, "parent_id",
                                        {"sub": lambda v: v >= 0,
                                         "top": lambda v: v < 0})
    diag["by_env"] = _split_rows(
        rows, "env",
        {lab: (lambda v, i=i: v == i) for i, lab in enumerate(ENV_LABELS)})
    # `term_*` stays the RAW unweighted term in every run, normalised or not:
    # it is the quantity the scales are derived from and the only one comparable
    # across runs. `term_eff_*` is what actually entered the loss, so
    # `sum(term_eff_*) == loss` is a standing self-check on the decomposition.
    for k, t in acc.items():
        diag[f"term_{k}"] = float(t.detach())
        diag[f"term_eff_{k}"] = weight[k] * float(t.detach()) / scale[k]
    diag["term_scale"] = {k: scale[k] for k in TERM_NAMES}
    diag["term_scale_active"] = term_scale is not None
    sel = np.asarray(idx, dtype=np.int64)
    for k in ("r_rms", "sigma_v", "virial", "bound_soft", "bound_hard"):
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        ref = sets.ref[k].detach().cpu().numpy().astype(np.float64)[sel]
        diag[f"median_{k}"] = float(np.nanmedian(vals))
        with np.errstate(divide="ignore", invalid="ignore"):
            diag[f"median_{k}_over_hr"] = float(np.nanmedian(vals / ref))
    diag["median_bound_hard_hr"] = float(
        np.nanmedian(sets.ref["bound_hard"].detach().cpu().numpy()[sel]))
    scl = sets.centre_scale.detach().cpu().numpy()[sel]
    diag["centre_mode"] = cfg.centre_mode
    diag["median_centre_offset_mpc_h"] = float(np.median(offsets))
    diag["median_centre_offset_radii"] = float(
        np.median(np.asarray(offsets) / scl))
    # What the term actually charged for, and what the gate will actually see.
    diag["median_centre_penalised_radii"] = float(
        np.median(np.asarray(penalised) / scl))
    diag["frac_centre_within_1_radius"] = float(
        np.mean(np.asarray(offsets) / scl < 1.0))
    # Reported at w_radius=0 too: it is the number that says whether the radial
    # collapse is happening, and a run that cannot see it cannot be read. Signed
    # median, so "too close" (negative, 58.3% of frozen sets) stays visible
    # after the absolute value the term charges on.
    rerr = np.asarray(radius_err)
    nan = float("nan")
    diag["has_host"] = bool(has_host)
    diag["median_radius_err_mpc_h"] = float(np.median(rerr)) if has_host else nan
    diag["median_radius_err_radii"] = (
        float(np.median(np.abs(rerr) / scl)) if has_host else nan)
    diag["median_radius_ratio"] = (
        float(np.median(radius_ratio)) if has_host else nan)
    diag["frac_radius_too_close"] = float(np.mean(rerr < 0.0)) if has_host else nan
    # The SHELL's own diagnostic, and the one to read for it: a perfect
    # distribution match still leaves 79% (train) / 85% (holdout) of the per-set
    # address error standing, so `median_radius_err_radii` above barely moves
    # even on total success. This is the residual the term actually charges --
    # the sorted pairing's log ratio -- and it goes to zero when it wins.
    if has_host and shell_now:
        _rc = np.sort(np.asarray(shell_now, dtype=np.float64))
        _rt = np.sort(np.asarray(shell_tgt, dtype=np.float64))
        _u = np.log(_rc / _rt)
        diag["median_shell_resid"] = float(np.median(np.abs(_u)))
        # Signed, so the inward bias the term exists to close stays visible.
        diag["median_shell_resid_signed"] = float(np.median(_u))
        diag["frac_shell_outside_dead_zone"] = float(
            np.mean(np.abs(_u) > float(np.log1p(float(cfg.shell_dead_zone)))))
    else:
        diag["median_shell_resid"] = nan
        diag["median_shell_resid_signed"] = nan
        diag["frac_shell_outside_dead_zone"] = nan
    diag["frac_radius_within_1_radius"] = (
        float(np.mean(np.abs(rerr) / scl < 1.0)) if has_host else nan)
    # Under `_shell_curves` rather than a plain name: it is a live tensor pair,
    # not a float, and every other consumer of `diag` serialises it to JSON. The
    # leading underscore is the marker that it must be popped, not logged.
    diag["_shell_curves"] = diag_curves
    return loss, diag


#: What a by-population / by-environment split reports. Deliberately short: the
#: point of the split is to say WHICH population moved, and four medians plus a
#: count do that. The full per-set table is in ``rows`` for anything more.
_SPLIT_KEYS = ("bound_hard", "virial", "centre_offset_radii", "iso_ratio")


def _split_rows(rows: List[Dict[str, float]], key: str,
                tests: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
    """Median :data:`_SPLIT_KEYS` over the rows each test accepts, plus ``n``.

    Why this exists. ``MemberSets.parent_id`` was added so an eval could say
    whether the run built the TOP-LEVEL sets specifically, and nothing ever read
    it -- so ``all_blocks_smallhalo`` ran 8,000 steps and 6.9 GPU-hours whose
    entire hypothesis was about top-level halos, and reported one number pooled
    over both populations. The held-out Rockstar gate then said the top-level
    count had gone DOWN. A pooled median cannot distinguish "the objective is
    wrong" from "the objective never got any budget", and those need different
    next runs.

    ``nanmedian`` throughout: ``iso_ratio`` is legitimately NaN for a set whose
    HR background is too sparse to be a ratio (``iso_min_count``), and an empty
    split reports ``n = 0`` with NaN medians rather than vanishing from the
    artifact -- a missing key reads as a bug, a zero reads as a fact.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name, test in tests.items():
        sel = [r for r in rows if test(r.get(key, -1))]
        d: Dict[str, float] = {"n": float(len(sel))}
        for k in _SPLIT_KEYS:
            v = np.asarray([r.get(k, float("nan")) for r in sel],
                           dtype=np.float64)
            v = v[np.isfinite(v)]
            d[k] = float(np.median(v)) if v.size else float("nan")
        out[name] = d
    return out


def _log2(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return torch.log(x.clamp_min(1e-30) / ref.clamp_min(1e-30)) ** 2


def _hinge_below(x: torch.Tensor, ref: torch.Tensor,
                 kind: str = "hinge") -> torch.Tensor:
    """Cost of falling BELOW ``ref``; zero for meeting or beating it.

    ``hinge`` is ``[1 - x/ref]_+^2``. Both properties matter and they pull apart:
    it is zero above the reference, which is the point, and it is **capped at 1**
    for any ``x >= 0``, which is not. A term that starts at 1% of its reference
    can never contribute more than a term that starts 1% above its own.

    ``log`` is ``[log(ref/x)]_+^2`` -- identical where it matters (exactly zero
    at and above ``ref``, same sign, same monotonicity) and unbounded as
    ``x -> 0``, so a deficit is charged in proportion to how many factors of two
    short it is rather than saturating at one.
    """
    r = ref.clamp_min(1e-12)
    if kind == "log":
        return torch.clamp(torch.log(r / x.clamp_min(1e-12)), min=0.0) ** 2
    return torch.clamp(1.0 - x / r, min=0.0) ** 2


def _hinge_above(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """``[x/ref - 1]_+^2`` -- costs nothing for beating the reference."""
    return torch.clamp(x / ref.clamp_min(1e-12) - 1.0, min=0.0) ** 2


def shell_reduce(rc: torch.Tensor, rt: torch.Tensor,
                 cfg: MemberGatherConfig,
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sorted (and optionally quantile-binned) log radii, candidate and target.

    One sort per side, paired by RANK -- the 1-D optimal transport map, exact
    and O(N log N). No set is ever named against its OWN target, so what comes
    back describes the radius DISTRIBUTION and stays silent about the address
    ``w_radius`` could not predict (out-of-sample R^2 0.022).

    ``cfg.shell_quantiles`` then reduces both sorted arrays to Q equal-COUNT
    bins and returns the bins' MEAN LOG radius. Why: the full pairing hands the
    loss one degree of freedom per set (~134 for a big host), and MEASURED on
    2026-09-08 the ``w_shell=30`` arm drove the train residual 7.3x while
    holdout moved only 1.4x -- it was fitting per-host detail that did not
    transfer. The low-order shape is the part a shared operator can plausibly
    carry; the rank-by-rank wiggle is the part it can only memorise. Q is the
    dial between them. Q >= N is the full pairing again (one set per bin), so a
    small host is never charged on emptier bins than it has sets.

    Split in LOG space because that is the space the residual is charged in: a
    bin's target is then the same quantity as the residual it feeds.
    """
    lc = torch.log(torch.sort(rc).values)
    lt = torch.log(torch.sort(rt).values)
    n = int(lc.shape[0])
    q = int(cfg.shell_quantiles)
    if q <= 0 or q >= n:
        return lc, lt
    # Equal-count bins. The split points are a pure function of n and q, so
    # this is a segment mean with no data-dependent control flow, and it stays
    # differentiable through the candidate side.
    edges = torch.linspace(0.0, float(n), q + 1, device=lc.device,
                           dtype=lc.dtype)
    pos = torch.arange(n, device=lc.device, dtype=lc.dtype)
    idx = torch.clamp((pos.unsqueeze(0) >= edges[:-1].unsqueeze(1)).sum(dim=0)
                      - 1, 0, q - 1)
    cnt = torch.zeros(q, device=lc.device, dtype=lc.dtype).index_add_(
        0, idx, torch.ones_like(lc)).clamp_min(1.0)
    bc = torch.zeros(q, device=lc.device, dtype=lc.dtype).index_add_(0, idx, lc)
    bt = torch.zeros(q, device=lt.device, dtype=lt.dtype).index_add_(0, idx, lt)
    return bc / cnt, bt / cnt


def shell_charge(lc: torch.Tensor, lt: torch.Tensor,
                 cfg: MemberGatherConfig) -> torch.Tensor:
    """The soft-shell cost of one reduced curve against its target."""
    return _soft_quadratic(
        (lc - lt).abs(), float(np.log1p(float(cfg.shell_dead_zone))),
        0.0).mean()


def shell_radii(pos: torch.Tensor, sets: MemberSets, cfg: MemberGatherConfig,
                set_indices: Optional[Sequence[int]] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Candidate and target host-centric radii only -- no set statistics.

    The cheap path for ``--shell-noise-draws``. A noise-averaged shell needs the
    candidate radii under several realisations, and a full
    :func:`member_gather_loss` per draw would pay the O(N^2) pair sums for terms
    the average never uses. Everything here is a centroid and a norm, so an
    extra draw costs a forward pass and a gather.

    Returns ``(r_soft, r_tgt)`` in the SAME order and with the same softening as
    the in-loop shell, so the two are interchangeable: the caller sorts.
    """
    idx = (list(range(sets.n_sets)) if set_indices is None
           else [int(i) for i in set_indices])
    eps = float(cfg.shell_eps_mpc_h)
    now, tgt = [], []
    for s in idx:
        # `pos` stands in for the velocity argument: only the position half of
        # the return is read, and _gather_one applies the same cap subsample to
        # both. The draw is fresh per call exactly as it is in the main loop.
        p, _v, _f = _gather_one(pos, pos, sets, s,
                                cap=int(cfg.max_set_particles))
        cbar = p.mean(dim=0)
        now.append(torch.sqrt(((cbar - sets.centre_host[s]) ** 2).sum()
                              + eps ** 2))
        tgt.append(torch.linalg.vector_norm(
            sets.centre_target[s] - sets.centre_host[s]).clamp_min(eps))
    if not now:
        z = torch.zeros((0,), device=pos.device, dtype=pos.dtype)
        return z, z
    return torch.stack(now), torch.stack(tgt)


def _soft_quadratic(u: torch.Tensor, dead: float, huber: float) -> torch.Tensor:
    """``u^2``, with an optional free zone below ``dead`` and a linear arm above
    ``huber``. Continuous in value and slope at both joins.

    Shared by the centre and radius terms so "softened" means one thing in this
    module: two numbers, in gate search radii, with the same meaning in both.
    """
    x = torch.clamp(u - dead, min=0.0) if dead > 0.0 else u
    if huber <= 0.0:
        return x ** 2
    a = huber - dead                # where the linear arm starts, in x
    return torch.where(x <= a, x ** 2, a * (2.0 * x - a))


def _radius_cost(u: torch.Tensor, cfg: MemberGatherConfig) -> torch.Tensor:
    """Cost of a host-centric RADIUS error ``u``, in gate search radii.

    Softer than the centre term by default (``radius_dead_zone`` 1.0): the ask
    is a shell, and one search radius of slack on a median 2.80-radii error
    still charges 79% of sets while spending nothing on the last fraction the
    gate cannot resolve anyway.
    """
    return _soft_quadratic(u, float(cfg.radius_dead_zone),
                           float(cfg.radius_huber_radii))


def _centre_cost(u: torch.Tensor, cfg: MemberGatherConfig) -> torch.Tensor:
    """Cost of a centroid offset ``u``, in gate search radii.

    Defaults to ``u^2`` -- exactly the term the 72/154 run used -- and both knobs
    are off unless set, so this is the identity for every run recorded in
    ``docs/sr2_member_gather.md``.

    ``centre_dead_zone`` makes the term cost nothing inside ``d`` radii. The gate
    is a threshold at ONE search radius, so a run paying quadratic cost at 0.1
    radii is buying nothing there while competing with ``virial``, ``bound`` and
    ``d6`` for the same gradient.

    ``centre_huber_radii`` makes it linear beyond ``h`` radii, continuous in
    value and slope. Frozen sets sit a median 5.6 radii out (*measured*, the
    step-0 row of the 2026-08-23 pool), where a quadratic charges ~31 and the
    hopeless sets own the batch gradient. Linear there keeps the pull without
    letting the tail set the scale.
    """
    return _soft_quadratic(u, float(cfg.centre_dead_zone),
                           float(cfg.centre_huber_radii))


# --------------------------------------------------------------------------- #
# Building the sets
# --------------------------------------------------------------------------- #
def _apply_cap(keep_rows: np.ndarray, n_sites: np.ndarray, max_sets: int,
               how: str, seed: int, *,
               env: Optional[np.ndarray] = None) -> np.ndarray:
    """Thin ``keep_rows`` to ``max_sets`` under one of three policies.

    Returns rows in ascending order in every case, so the cap changes *which*
    sets are supervised and never their order -- a reordering alone would move
    every ``--sets-per-step`` sample and make two runs incomparable for a reason
    that has nothing to do with the selection.

    ``env`` adds a SECOND stratification axis to ``stratified``: bins become
    ``(log2(n_sites), environment)`` and drain round-robin over both. Mass
    alone is not enough once the pool spans environments, and for the same
    reason ``largest`` is not enough once it spans masses -- a cluster
    region's pool is dominated by field halos, so a mass-stratified cap
    still hands the budget to the field. *Measured*: the small-halo arm
    overshot HR in ``field`` (subhalos 3,316 against 2,466) while closing
    3.8% of the ``near`` deficit that holds 56% of the region's missing
    objects. ``None`` reproduces the mass-only bins exactly.

    ``largest`` is the historical behaviour and is kept as the default, but it is
    a trap for a mass-window run: the pool of a 4-tile cluster region is a mass
    function, so "keep the 256 biggest" keeps the top of it and drops exactly the
    small halos a small-halo arm exists to build. ``stratified`` bins by
    ``log2(n_sites)`` and drains the bins round-robin, which caps the total
    without truncating the range; ``random`` is a uniform draw.
    """
    if how == "largest":
        order = np.argsort(-np.asarray(n_sites))[:max_sets]
        return np.sort(np.asarray(keep_rows)[order])
    rng = np.random.default_rng(int(seed))
    if how == "random":
        pick = rng.choice(keep_rows.size, size=max_sets, replace=False)
        return np.sort(np.asarray(keep_rows)[pick])
    if how != "stratified":
        raise ValueError(f"set_select {how!r} is not 'largest', 'random' or "
                         "'stratified'")
    n = np.maximum(np.asarray(n_sites, dtype=np.float64), 1.0)
    b = np.floor(np.log2(n)).astype(np.int64)
    if env is not None:
        # A composite key, not a nested loop: the round-robin below already
        # drains whatever bins exist, and an empty (mass, env) cell simply
        # never appears. Offsetting by the mass-bin span keeps the two axes
        # from colliding without assuming either's range.
        e = np.asarray(env, dtype=np.int64)
        b = b * (int(e.max()) + 1 if e.size else 1) + np.maximum(e, 0)
    picked: List[int] = []
    # Shuffle inside each bin so the choice within a mass decade is not "whatever
    # order the catalog happened to be in", then take one from each non-empty bin
    # per round. Small bins empty out and the remaining budget flows to the bins
    # that still have members, so a range with one 100k-particle set does not
    # spend 1/nbins of the budget on it.
    queues = {}
    for k in np.unique(b):
        idx = np.flatnonzero(b == k)
        rng.shuffle(idx)
        queues[int(k)] = list(idx)
    keys = sorted(queues)
    while len(picked) < max_sets and any(queues[k] for k in keys):
        for k in keys:
            if not queues[k]:
                continue
            picked.append(queues[k].pop())
            if len(picked) >= max_sets:
                break
    return np.sort(np.asarray(keep_rows)[np.asarray(picked, dtype=np.int64)])


def build_member_sets(
    cat, owner_index, tiles: Sequence[int],
    hr_field: torch.Tensor, frozen_field: torch.Tensor,
    cfg: MemberGatherConfig, *,
    particle_mass_msun_h: float,
    ng_hr: int = 512, tile_hr: int = 64, boxsize_mpc_h: float = 100.0,
    dis_scale_mpc_h: float = 6.0, vel_scale_kms: float = 1.0,
    home: Optional[Dict[str, np.ndarray]] = None,
    frozen_box=None,
    host_pos: Optional[np.ndarray] = None,
    top_level: bool = False,
    env: Optional[np.ndarray] = None,
    report: Optional[Dict] = None,
) -> MemberSets:
    """Select the supervised sets and measure the reference this run can reach.

    Selection reuses :func:`subhalo_gather.subhalo_home_tiles` unchanged -- a
    second definition of "which tile is this subhalo's" would be a second
    definition of the experiment. ``top_level=True`` flips that one call to the
    **host** population (``parent_ids < 0``) so the identical machinery builds a
    *preservation* constraint on resolved hosts; nothing else in this function
    knows or cares which population it received. Three of that path's cuts survive
    (``min_num_p``, ``min_purity``, home tile is trained) and **one does not**:
    there is no window here, so the "window fits inside the scored cube" cut that
    took ``docs/sr2_gather_finetune.md`` section 4 from 58 candidates down to 43
    is simply gone. Coverage should rise for free.

    The reference is the hybrid field, and that matters
    ---------------------------------------------------
    A member whose Lagrangian site is outside the trained tiles cannot be moved
    by this run, and the splice will not move it either. So HR's *own* value on
    the full set is not a reachable target -- asking for it would charge the loss
    for material it does not control. The reference is therefore measured on
    **HR inside the trained tiles, frozen outside** -- which is exactly the
    section 8.1 ceiling construction (HR tiles spliced into the frozen box),
    evaluated per set. The pure-HR value is recorded in ``report`` alongside, so
    the cost of partial tile coverage is visible rather than assumed.

    ``env`` is the per-catalog-row environment label from
    :func:`subhalo_gather.halo_environment`, computed **once per box** by the
    caller because it costs a KD-tree over the whole catalog and this function
    is called once per host site. ``None`` disables ``cfg.env_select``, which
    then must be ``"any"`` -- a selection silently not applied is the failure
    mode this refuses, having already been paid for once by ``--set-select
    largest`` under ``--population all``.

    ``frozen_box`` is the cached full-box frozen SR2 field (a memmap is fine). It
    is what supplies those outside members. Passing ``None`` drops them, which
    computes the potential of a fraction of an object and **overstates how bound
    it is**; the call is allowed, because it is what the unit tests use, but it
    records ``outside_dropped`` in ``report`` and callers running the real
    experiment should pass the box.
    """
    from .subhalo_gather import env_mask, subhalo_home_tiles

    dev = hr_field.device
    rep: Dict = report if report is not None else {}
    tiles = [int(t) for t in tiles]

    # `top_level=True` is the host-preservation path's older spelling and still
    # decides on its own; otherwise the population is the config's. One
    # resolution, recorded, so no artifact is ambiguous about what it supervised.
    pop = "top" if top_level else str(cfg.population)
    if home is None:
        home = subhalo_home_tiles(cat, owner_index, ng_hr=int(ng_hr),
                                  tile_hr=int(tile_hr),
                                  min_num_p=int(cfg.min_num_p),
                                  max_num_p=int(cfg.max_num_p),
                                  population=pop)
    rep["top_level"] = bool(top_level)
    rep["population"] = pop
    rep["min_num_p"] = int(cfg.min_num_p)
    rep["max_num_p"] = int(cfg.max_num_p)

    trained = np.isin(home["tile"], np.asarray(tiles, dtype=np.int64))
    keep_rows = np.flatnonzero(trained & (home["purity"] >= float(cfg.min_purity)))
    rep["n_resolved"] = int(home["tile"].size)
    rep["n_home_tile_trained"] = int(trained.sum())
    rep["n_after_purity"] = int(keep_rows.size)

    # --- the environment cut, BEFORE the cap ----------------------------
    # Order matters and is not arbitrary: capping first and then filtering
    # would spend the whole --max-sets budget on the pool as it stands --
    # which a cluster region's field halos dominate -- and hand back a
    # handful of `near` sets, i.e. the exact miss this axis exists to fix.
    home_env = (np.full(home["tile"].size, -1, dtype=np.int64) if env is None
                else np.asarray(env, dtype=np.int64)[home["row"]])
    rep["env_select"] = str(cfg.env_select)
    rep["env_supplied"] = env is not None
    if str(cfg.env_select).strip() not in ("", "any", "all"):
        if env is None:
            raise ValueError(
                f"env_select={cfg.env_select!r} selects on environment but no "
                "`env` array was passed -- build_member_sets cannot label "
                "halos on its own (it costs a KD-tree over the whole "
                "catalog, so the caller computes it once per box). Pass "
                "subhalo_gather.halo_environment(cat), or the run would be "
                "an env_select='any' run under another name.")
        keep_rows = keep_rows[env_mask(home_env[keep_rows], str(cfg.env_select))]
    rep["n_after_env"] = int(keep_rows.size)

    # The cap keeps the LARGEST sets, and it is silent by default -- which was
    # fine while it never bound (154 sets at four tiles) and is not once the
    # tiling widens: at sixteen tiles the selection is 625 and the cap keeps 256,
    # so a run that does not report it looks like the live-fraction cut removed
    # 369 objects. It is recorded, and the caller is expected to say so.
    rep["max_sets"] = int(cfg.max_sets)
    rep["set_select"] = str(cfg.set_select)
    rep["n_dropped_by_cap"] = int(max(0, keep_rows.size - int(cfg.max_sets)))
    rep["cap_binds"] = bool(keep_rows.size > int(cfg.max_sets))
    if keep_rows.size > int(cfg.max_sets):
        keep_rows = _apply_cap(keep_rows, home["n_sites"][keep_rows],
                               int(cfg.max_sets), str(cfg.set_select),
                               int(cfg.bg_seed),
                               env=(None if env is None
                                    else home_env[keep_rows]))

    kw = dict(ng_hr=int(ng_hr), tile_hr=int(tile_hr),
              boxsize_mpc_h=float(boxsize_mpc_h),
              dis_scale_mpc_h=float(dis_scale_mpc_h),
              vel_scale_kms=float(vel_scale_kms))
    with torch.no_grad():
        hr_pos, hr_vel = tile_particles(hr_field, tiles, **kw)
        fz_pos, fz_vel = tile_particles(frozen_field, tiles, **kw)
    fz_pos_np = fz_pos.detach().cpu().numpy().astype(np.float64)

    rng = np.random.default_rng(int(cfg.bg_seed))
    halo_id, num_p, n_live_l = [], [], []
    live_rows, fixed_pos, fixed_vel, bg_rows, centres = [], [], [], [], []
    centre_tgt, centre_scl, centre_rh, centre_slf = [], [], [], []
    centre_hst = []
    # The host on the same periodic branch as the set it owns, or `rhat` is a
    # box-length artifact for any cluster near a face. Deferred into the loop:
    # the branch is per set, since `ctr` is that set's own unwrap centre.
    host_t = (None if host_pos is None else
              torch.as_tensor(np.asarray(host_pos, dtype=np.float64),
                              device=dev, dtype=hr_pos.dtype))
    parent_id: List[int] = []
    env_l: List[int] = []
    iso_r: List[float] = []
    ref_rows: List[SetTerms] = []
    pure_rows: List[Dict[str, float]] = []
    n_dropped_live = 0
    outside_dropped = 0

    for r in keep_rows:
        hid = int(home["halo_id"][r])
        crow = int(home["row"][r])
        ids = owner_index.members(hid).astype(np.int64)
        if ids.size < 2:
            continue
        rows_np, keep = _rows_for_ids(ids, tiles, ng_hr=int(ng_hr),
                                      tile_hr=int(tile_hr))
        frac = float(keep.size) / float(ids.size)
        if frac < float(cfg.min_live_frac) or keep.size < 2:
            n_dropped_live += 1
            continue

        # The unwrap branch is the HR catalog centre: a true constant, so the
        # candidate, the frozen and the HR fields all unwrap onto the SAME
        # branch. A per-field centroid would put the three on different branches
        # near a box face and make their statistics incomparable.
        ctr = torch.tensor(np.asarray(cat.pos[crow], dtype=np.float64),
                           device=dev, dtype=hr_pos.dtype)

        outside = np.setdiff1d(np.arange(ids.size), keep, assume_unique=False)
        fp = fv = None
        if outside.size and frozen_box is not None:
            from .bound_discriminator import particles_at
            op, ov = particles_at(frozen_box, ids[outside],
                                  boxsize_kpc_h=float(boxsize_mpc_h) * 1e3)
            fp = torch.as_tensor(op, device=dev, dtype=hr_pos.dtype)
            fv = torch.as_tensor(ov, device=dev, dtype=hr_pos.dtype)
        elif outside.size:
            outside_dropped += int(outside.size)

        rows_t = torch.as_tensor(rows_np, device=dev, dtype=torch.long)

        # --- the reachable reference: HR inside the tiles, frozen outside ----
        hp = unwrap_about(torch.cat([hr_pos[rows_t]] + ([fp] if fp is not None else []),
                                    dim=0), ctr, float(boxsize_mpc_h))
        hv = torch.cat([hr_vel[rows_t]] + ([fv] if fv is not None else []), dim=0)

        # --- local background: fixed ids, chosen on the frozen field ---------
        r_ref = float(torch.sqrt(
            ((hp - hp.mean(0)) ** 2).sum(1).mean()).detach())
        bg_t = None
        if int(cfg.bg_k) > 0:
            c = np.asarray(cat.pos[crow], dtype=np.float64)
            d = fz_pos_np - c
            d -= np.round(d / float(boxsize_mpc_h)) * float(boxsize_mpc_h)
            near = np.flatnonzero((d * d).sum(1)
                                  < (float(cfg.bg_radius_factor) * r_ref) ** 2)
            near = np.setdiff1d(near, rows_np, assume_unique=False)
            if near.size > int(cfg.bg_k):
                near = rng.choice(near, size=int(cfg.bg_k), replace=False)
            if near.size > 1:
                bg_t = torch.as_tensor(np.sort(near), device=dev, dtype=torch.long)
        # The isolation counting radius: a fixed fraction of the sphere the
        # background was sampled from, so the count is never counting the
        # sampling edge. Derived from `r_ref`, HR's own rms radius for this
        # set, which is a constant of the data -- deriving it from the
        # CANDIDATE's radius would let the run shrink the sphere instead of
        # clearing it, which is the whole failure mode this term is for.
        iso_radius = (float(cfg.iso_radius_frac) * float(cfg.bg_radius_factor)
                      * r_ref) if bg_t is not None else 0.0

        bp = bv = None
        if bg_t is not None:
            bp = unwrap_about(hr_pos[bg_t], ctr, float(boxsize_mpc_h))
            bv = hr_vel[bg_t]
        ref_rows.append(set_statistics_torch(
            hp, hv, particle_mass_msun_h=particle_mass_msun_h, cfg=cfg,
            bg_pos=bp, bg_vel=bv, iso_radius=iso_radius))

        # Pure HR, for the report only: how much does partial coverage cost?
        if fp is not None:
            php = unwrap_about(hr_pos[rows_t], ctr, float(boxsize_mpc_h))
            pure_rows.append(set_statistics_torch(
                php, hr_vel[rows_t], particle_mass_msun_h=particle_mass_msun_h,
                cfg=cfg).detached())
        else:
            pure_rows.append(ref_rows[-1].detached())

        # The reachable centroid, not the HR catalog position: the out-of-tile
        # members sit at frozen coordinates and pull the achievable centre of
        # mass away from HR's, so pinning to HR's would charge the loss for an
        # offset it cannot remove. Same logic as the hybrid reference above.
        centre_tgt.append(hp.mean(dim=0).detach())
        centre_scl.append(torch.tensor(
            max(float(cat.rvir[crow]) / 1000.0, 0.15),
            device=dev, dtype=hr_pos.dtype))

        # `centre_mode="self"`: the SAME collection `_gather_one` will build at
        # step 0 -- live rows from the frozen field plus the frozen stragglers,
        # unwrapped about the same `ctr`. Equal by construction, so the term is
        # exactly zero before the first step and the run is an anchor, not a
        # pull towards somewhere else.
        fzp = unwrap_about(
            torch.cat([fz_pos[rows_t]] + ([fp] if fp is not None else []),
                      dim=0), ctr, float(boxsize_mpc_h))
        centre_slf.append(fzp.mean(dim=0).detach())

        # `centre_mode="radial"`: the clustercentric direction at the TARGET,
        # not at the frozen centroid -- the direction is a property of where the
        # object belongs, and taking it at the moving centroid would make the
        # projection axis a function of the optimisation variable.
        # `w_radius` keeps the host POINT as well: the direction is enough to
        # split an offset, but a distance needs something to measure from.
        if host_t is None:
            centre_rh.append(torch.zeros(3, device=dev, dtype=hr_pos.dtype))
            centre_hst.append(torch.zeros(3, device=dev, dtype=hr_pos.dtype))
        else:
            hc = unwrap_about(host_t[None, :], ctr, float(boxsize_mpc_h))[0]
            centre_hst.append(hc.detach())
            radial = centre_tgt[-1] - hc
            dn = torch.linalg.vector_norm(radial)
            centre_rh.append(radial / dn if float(dn) > 1e-9
                             else torch.zeros(3, device=dev, dtype=hr_pos.dtype))

        halo_id.append(hid)
        num_p.append(int(cat.num_p[crow]))
        parent_id.append(int(cat.parent_ids[crow]))
        env_l.append(int(home_env[r]))
        iso_r.append(float(iso_radius))
        n_live_l.append(int(keep.size))
        live_rows.append(rows_t)
        fixed_pos.append(fp)
        fixed_vel.append(fv)
        bg_rows.append(bg_t)
        centres.append(ctr)

    rep["n_dropped_live_frac"] = int(n_dropped_live)
    rep["n_sets"] = len(halo_id)
    # The population mix and the mass range of what was ACTUALLY selected, after
    # every cut and the cap. Reported unconditionally rather than only for
    # `population="all"`: a `"sub"` run's row reading `n_top_level 0` is what
    # makes the `"all"` run's row readable as a change.
    if halo_id:
        _pid = np.asarray(parent_id, dtype=np.int64)
        _np_ = np.asarray(num_p, dtype=np.int64)
        rep["n_top_level"] = int(np.sum(_pid < 0))
        rep["n_sub"] = int(np.sum(_pid >= 0))
        # The environment mix of what was ACTUALLY selected. Reported
        # unconditionally: an `env_select="any"` run's row is what makes a
        # targeted run's row readable as a change, and a run that asked for
        # `near` and got 4 sets needs to say so in the shakeout, not in the
        # Rockstar gate ten hours later.
        _env = np.asarray(env_l, dtype=np.int64)
        rep["n_by_env"] = {lab: int(np.sum(_env == i))
                           for i, lab in enumerate(ENV_LABELS)}
        rep["n_env_unlabelled"] = int(np.sum(_env < 0))
        _iso = np.asarray(iso_r, dtype=np.float64)
        rep["median_iso_radius_mpc_h"] = float(np.median(_iso)) if _iso.size else 0.0
        rep["num_p_min"] = int(_np_.min())
        rep["num_p_median"] = float(np.median(_np_))
        rep["num_p_max"] = int(_np_.max())
        # sum N^2 over the selection: the pair bill this pool commits a step to,
        # available before the first step rather than as a wall-clock surprise.
        rep["sum_num_p_squared"] = float(np.sum(_np_.astype(np.float64) ** 2))
    rep["outside_dropped"] = int(outside_dropped)
    if outside_dropped:
        rep["outside_dropped_warning"] = (
            "frozen_box was None: members outside the trained tiles were "
            "dropped, so the potential is that of a fragment and bound_frac "
            "is overstated")

    if not halo_id:
        raise ValueError(
            f"no supervised sets survived: {rep['n_resolved']} resolved "
            f"{ {'sub': 'subhalos', 'top': 'hosts', 'all': 'halos'}[pop] } in "
            f"num_p [{cfg.min_num_p}, {cfg.max_num_p or 'inf'}), "
            f"{rep['n_home_tile_trained']} homed in tiles {tiles}, "
            f"{rep['n_after_purity']} past purity >= {cfg.min_purity}, "
            f"{n_dropped_live} past live fraction >= {cfg.min_live_frac}")

    def col(name: str) -> torch.Tensor:
        return torch.stack([getattr(t, name) if getattr(t, name) is not None
                            else torch.full((), float("nan"), device=dev,
                                            dtype=hr_pos.dtype)
                            for t in ref_rows]).detach()

    ref = {k: col(k) for k in ("r_rms", "sigma_v", "virial", "bound_soft",
                               "bound_hard", "d6", "bg_occ")}
    # The "hr" temperature option needs HR's own energy scale per set; derive it
    # from the monopole binding energy of the reference configuration, which is
    # a constant of the data and does not move with the optimisation.
    ref["bound_scale"] = (float(cfg.bound_tau) * G_MPC_KMS2_PER_MSUN
                          * float(particle_mass_msun_h)
                          * torch.tensor([float(n) for n in n_live_l], device=dev,
                                         dtype=hr_pos.dtype)
                          / ref["r_rms"].clamp_min(1e-12))
    def _med(a: np.ndarray) -> float:
        """``nanmedian`` without the all-NaN warning: ``d6`` is legitimately
        absent when ``bg_k`` is 0, and a warning there would train the reader to
        ignore warnings that do mean something."""
        a = np.asarray(a, dtype=np.float64)
        good = a[np.isfinite(a)]
        return float(np.median(good)) if good.size else float("nan")

    rep["reference_median"] = {k: _med(v.cpu().numpy()) for k, v in ref.items()}
    rep["pure_hr_median"] = {k: _med([p[k] for p in pure_rows])
                             for k in ("r_rms", "sigma_v", "virial", "bound_hard")}
    rep["median_reference_centre_offset_mpc_h"] = float(np.median(
        [float(torch.linalg.vector_norm(t - c))
         for t, c in zip(centre_tgt, centres)]))
    rep["host_pos_supplied"] = host_pos is not None
    rep["median_frozen_centre_offset_radii"] = float(np.median(
        [float(torch.linalg.vector_norm(t - f) / sc)
         for t, f, sc in zip(centre_tgt, centre_slf, centre_scl)]))
    rep["median_centre_scale_mpc_h"] = float(np.median(
        [float(x) for x in centre_scl]))
    # What `w_radius` would have to work with, measured at pool-build time on
    # the frozen field: the signed radius error of every set, and how many sit
    # too close. A run whose pool reports ~0 here has nothing for the term to do.
    if host_pos is not None:
        _dr = np.array([float(torch.linalg.vector_norm(f - h)
                              - torch.linalg.vector_norm(t - h))
                        for t, f, h in zip(centre_tgt, centre_slf, centre_hst)])
        _sc = np.array([float(x) for x in centre_scl])
        rep["median_frozen_radius_err_mpc_h"] = float(np.median(_dr))
        rep["median_frozen_radius_err_radii"] = float(
            np.median(np.abs(_dr) / _sc))
        rep["frac_frozen_radius_too_close"] = float(np.mean(_dr < 0.0))
    rep["median_live_frac"] = float(np.median(
        [nl / max(int(np_), 1) for nl, np_ in zip(n_live_l, num_p)]))

    return MemberSets(
        halo_id=np.asarray(halo_id, dtype=np.int64),
        num_p=np.asarray(num_p, dtype=np.int64),
        parent_id=np.asarray(parent_id, dtype=np.int64),
        env=np.asarray(env_l, dtype=np.int64),
        iso_radius=np.asarray(iso_r, dtype=np.float64),
        n_live=np.asarray(n_live_l, dtype=np.int64),
        live_rows=live_rows, fixed_pos=fixed_pos, fixed_vel=fixed_vel,
        bg_rows=bg_rows, centre_ref=torch.stack(centres),
        centre_target=torch.stack(centre_tgt),
        centre_scale=torch.stack(centre_scl),
        centre_rhat=torch.stack(centre_rh),
        centre_self=torch.stack(centre_slf),
        centre_host=torch.stack(centre_hst),
        ref=ref, particle_mass_msun_h=float(particle_mass_msun_h),
        boxsize_mpc_h=float(boxsize_mpc_h))
