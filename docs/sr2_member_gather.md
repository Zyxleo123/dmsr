# Gathering by particle id: the objective builds bound subhalos, and where it is told to

**Scope.** A results note. One differentiable loss, two free-field runs, two real
Rockstar gates, on set8 and on the same cluster every result in
`docs/sr2_gather_finetune.md` uses (host 271800, `log10 Mvir` 14.81).

It records the first thing in this line to move a halo finder. The window
objective of `sr2_gather_finetune.md` recovered **0 of 43** supervised subhalos
and, per that document's section 8.2, never moved the supervised material at
all. Rebuilt on id-gathered member sets, the same idea recovers **72 of 154**
with every guard holding. The two runs below are separated by a single missing
term, and section 5 is about how the gate found it.

Numbers are tagged *measured* (read from an artifact on disk), *derived*
(arithmetic on measured constants) or *design* (a proposed rule, not yet run).

Depends on `docs/sr2_gather_finetune.md` for the objective this replaces, its
section 8.1 for the gate's calibration and its section 8.2 for the diagnosis
that motivated the rebuild.

## 1. What changed, in one line

The window is gone. Membership comes from HR's `owner` array, which is indexed by
**flat Lagrangian id**, and SR2 and HR share that lattice -- so the same id set
addresses the same particles in every field, no halo matching enters anywhere,
and gathering a subhalo is one fancy-index into the generator's
`(B, 6, T, T, T)` output. It is cheaper than the CIC deposit
`sr2_gather_finetune.md` section 1.2 argued for and, unlike it, it is exactly
the object rather than a Gaussian window that mostly contains the host.

## 2. The loss

Six terms, each normalised to HR's own value on the same set, computed by the
same code that evaluates the candidate so the two can never disagree about the
estimator.

| term | shape | what it is |
| --- | --- | --- |
| `virial` `2T/\|W\|` | `log^2`, 2-sided | the driver: a **pair** sum over all `N(N-1)/2` pairs, not a moment of a smoothed field, and 64x from HR at the frozen start |
| `bound` | hinged below HR | Rockstar's unbinding rule itself, `0.5\|v-<v>\|^2 + phi < 0`, sigmoid-relaxed: N per-particle conditions coupled through an `N^2` kernel |
| `d6` | hinged above HR | 6-D compactness against a **local non-member background**, so "close and cold" cannot be bought by cooling the neighbourhood -- that cools the normaliser too |
| `centre` | quadratic | where the object is. Section 5. |
| `r_rms`, `sigma_v` | `log^2`, 2-sided | break the `sigma_v^2 * r_rms` degeneracy `virial` alone leaves open |

Hinged terms contribute exactly zero with zero gradient once HR is reached, so
the loss cannot ask for more than HR has -- `sr2_gather_finetune.md` section 1
property 2, kept. Two-sided terms are symmetric in over- and undershoot,
deliberately: an over-collapsed, over-cooled set is as wrong as a diffuse one.

### 2.1 The soft unbinding test is saturated, not merely low

*Derived.* A **fixed** sigmoid temperature makes `bound` numerically dead at the
start. Frozen SR2's supervised sets carry `ke ~ 5e5 (km/s)^2` against a monopole
binding scale `G N m / r_rms ~ 3e3`, so `energy/tau` is O(300) and
`sigmoid(-300)` is exactly 0 in float32 -- zero value **and zero gradient**, for
every particle in every set. That is why section 8.2's table reads
`bound_frac = 0.000` rather than something small.

So the temperature is scaled to the set's *current* energy spread and detached.
The term always has gradient, and as the set tightens the temperature shrinks
with it, sharpening the surrogate toward the true indicator -- a continuation,
not a fixed relaxation. The hard `bound_frac` is reported at every eval beside
it so the surrogate is never mistaken for it.
`tests/features/test_member_gather.py` pins the fixed-temperature gradient at
five orders of magnitude below the adaptive one.

### 2.2 The reference is the reachable field, not HR

*Measured.* A member whose Lagrangian site falls outside the trained tiles
cannot be moved by the run and will not be moved by the splice either. So HR's
own value on the full set is not a reachable target, and charging the loss for
it would be charging for material nobody controls.

The reference is therefore measured on **HR inside the trained tiles, frozen
outside** -- per set, exactly the section 8.1 ceiling construction. The
difference is not small, and it is not small because the missing fraction is
small:

| median over the 154 sets | pure HR | reachable |
| --- | ---: | ---: |
| `r_rms` (Mpc/h) | 0.149 | **0.388** |
| `sigma_v` (km/s) | 153 | **248** |
| `2T/\|W\|` | 2.68 | **10.0** |
| `bound_frac` | 0.687 | **0.534** |

*Measured*: median live fraction is **0.946**. *Derived*: `r_rms` is an RMS about
the centroid, so the 5.4% of members sitting ~1 Mpc/h out at frozen coordinates
dominate it quadratically -- `sqrt(0.95*0.15^2 + 0.05*1^2) = 0.27` reproduces
most of the gap by itself. **Even a perfect run lands well short of pure HR, and
that shortfall is geometry rather than objective.** Closing it is an
`--n-tiles` argument.

These are consistent with section 8.2's table rather than in tension with it:
that document's "HR tiles" column is also the hybrid construction, measured on
the old 43-set list (0.525 / 341 / 14.4 / 0.428). The 154-set population is
tighter and more bound because it includes small compact satellites the window
cut had been removing.

## 3. Why a free field, and not a fine-tune

*Design, and it paid for itself twice below.* `sr2_gather_finetune.md` spent
four GPU runs and four Rockstar gates before its section 8.2 revealed that the
objective had never moved the supervised material. Every one of those runs
confounded two questions: **is the loss right**, and **can the generator reach
it**.

`scripts/features/free_field_gather.py` removes the generator. The optimised
variable is `candidate = frozen + delta` with `delta` a free
`(4, 6, 64, 64, 64)` tensor at zero -- 6.3M parameters, no convolution, no
shared operator. Nothing constrains the answer except the loss and the guards.
It writes `tiles.npz` in the layout `finetune_host_gather.py` writes, so the
existing splice -> Rockstar -> compare chain gates it unchanged, against the
same calibrated ceiling.

Nothing here is a model. `delta` sees the true member sets and generalises to
nothing; its only job is to bracket what the objective *permits*.

## 4. Coverage: 43 -> 154 supervised sets, for free

*Measured*, jobs 35171 / 35189 / 35291.

| cut | count |
| --- | ---: |
| HR subhalos in set8 with >= 200p | 12666 |
| home Lagrangian tile is one of the 4 trained | 163 |
| purity >= 0.5 | 154 |
| live fraction >= 0.5 | **154** (0 dropped) |

Three of the window path's cuts survive and **one does not**: there is no
window, so the "window fits inside the scored cube" geometry cut that took
section 4's candidates from 58 down to 19 is simply gone. The trained tiles hold
42.4% of the host's Lagrangian sites, unchanged.

*Measured*: 474,478 of 1,048,576 particles are movable (45.25%) -- members plus
their 4096-particle local backgrounds, deduplicated.

## 5. Run 1: bound objects, wrong addresses

*Measured*, job 35190, 2000 steps, 1616 s on one A5000; gate jobs 35191-35193
and 35284.

**The loss worked on its own terms, and this time on the objects.** Every
statistic is over the HR member ids, so unlike section 3.2's window numbers
these are about the subhalos:

| median over 154 sets | frozen | run 1 | reference |
| --- | ---: | ---: | ---: |
| `bound_frac` | 0.002 | **0.623** | 0.534 |
| `2T/\|W\|` | 495 | **10.0** | 10.0 |
| `r_rms` / ref | 2.11 | **1.00** | 1 |
| `sigma_v` / ref | 2.42 | **1.10** | 1 |

Loss 13.44 at step 100 -> 0.046 at step 2000 (step 0 is not recorded: the eval row is written before the loop).

**And Rockstar moved.** Subhalos within `R_vir`, against section 8.1's
calibration:

| | HR | ceiling | run 1 | null | preserve | base | window run C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| total | 506 | 227 | **156** | 20 | 19 | 11 | 5 |

156 against a base of 11 and a `+-9` noise floor -- 69% of the ceiling, and the
first time anything in this line registered at the halo finder at all.

**Then the per-target row.** *Measured*: **8/154** recovered as bound halos
within `1 x r_vir` carrying `>= 25%` of HR's particles, against frozen's 3/154.
The threshold sweep and the miss profile were added to
`compare_gather_catalog.py` to read that:

| radius | mass frac | frozen | run 1 |
| --- | --- | ---: | ---: |
| 1 x r_vir | >= 25% | 3/154 | 8/154 |
| 2 x | >= 10% | 7/154 | 22/154 |
| 3 x | >= 5% | 19/154 | 55/154 |
| 4 x | >= 2% | 32/154 | 81/154 |

> the 146 misses: nearest halo of **any** mass is 0.414 Mpc/h away (median),
> search radius 0.150; **4%** have anything inside the radius at all.

That line is the diagnosis. With the mass requirement dropped entirely, 96% of
missed targets have **nothing** within the search radius. The objects are not
too light. They are not there.

### 5.1 The loss never said where

Every term in section 2 except `centre` is an internal moment about the set's
**own** centroid -- `r_rms`, `sigma_v`, `2T/|W|`, `bound`, `d6`. Nothing pinned
the centroid. The window objective had position for free, because the window sat
at the HR centre; the id-gather dropped it, and the gate priced the omission at
0.414 Mpc/h.

`centre` is `(|x_bar - x_bar_ref| / max(r_vir, 0.15))^2`, normalised by the same
radius `compare_gather_catalog.py` matches on, so one unit of the term is
exactly one search radius and driving it below 1 is driving the target into the
hit criterion. Its reference is the reachable centroid, per section 2.2.

### 5.2 The guard's gradient plus Adam rewrote the tile

*Measured.* 99.56% of particles in the trained tiles moved -- median 0.54 Mpc/h
and 132 km/s, max 30 Mpc/h. Only 0.44% were untouched. The run's own guard
caught it (`untouched drift 4.55`, verdict `FEASIBLE BUT UNGUARDED`).

*Derived*, and the arithmetic is unambiguous: the member loss alone has non-zero
gradient on exactly the member and background rows (pinned in the tests), but
the LR-scale guard block-averages over `scale_factor^3 = 512` cells, so one moved
particle puts a gradient on all 512 cells of its block. **Adam then rescales any
non-zero gradient, however tiny, to ~`lr` per step.** `lr * steps = 6.0`
normalised is the ceiling and the observed maximum was 5.0 -- the untouched
particles drifted at 83% of the fastest rate the optimiser permits.

Fix: mask `delta.grad` to the touched rows. Since `delta` starts at zero and the
masked entries never receive gradient, Adam's moments stay zero and those
particles remain bit-identical.

**Nothing in run 1 is attributable to the objective**, because the objective was
not the only thing moving the field.

## 6. Run 2: 3 -> 72 of 154

*Measured*, job 35292 (2000 steps, 1690 s), gate jobs 35293-35295. One change to
the loss (`w_centre = 1`) and one to the optimiser (`--mask-grad`, default on).

| | frozen | run 1 | **run 2** |
| --- | ---: | ---: | ---: |
| supervised, `1 x r_vir` `>= 25%` | 3/154 | 8/154 | **72/154** |
| `2 x` / `>= 10%` | 7/154 | 22/154 | **107/154** |
| `3 x` / `>= 5%` | 19/154 | 55/154 | **126/154** |
| `4 x` / `>= 2%` | 32/154 | 81/154 | **137/154** |
| subhalos in `R_vir` | 11 | 156 | 128 |
| centre offset | -- | 0.414 Mpc/h | **0.00 r** |
| untouched drift (normalised) | -- | 4.550 | **0.000** |
| hosts >= 200p, whole box | 23014 | 22980 (-34) | **23017 (+3)** |
| verdict | -- | FEASIBLE BUT UNGUARDED | **FEASIBLE** |

Both fixes did exactly and only what they were designed to do. `dx` fell
1.11 -> 0.16 -> 0.05 search radii by step 300 and reached 0.00 by 1300.
`untouched_max_abs_delta` is **0.000e+00 exactly**, not merely small.

The remaining 82 misses moved closer too: nearest halo of any mass 0.414 ->
**0.224** Mpc/h, and the share with anything inside the radius 4% -> **28%**.

Three details worth reading rather than skipping:

- **The `R_vir` count fell, 156 -> 128, and that is the right direction.** The
  supervised sets are selected by Lagrangian home tile, not by an Eulerian
  sphere. Once positions are pinned the objects go to their true addresses, many
  of which lie outside `R_vir`; the shell profile shows it, with the
  1.78-3.57 Mpc/h ring going +64 -> +145. Run 1's extra count was drift-generated
  objects that happened to land inside the sphere.
- **Collateral damage reversed.** Box-wide hosts >= 200p went from -34 to +3.
  Run 1 was fragmenting the box; run 2 is not.
- **The host survived, and slightly overshot.** *Measured*: `log10 Mvir`
  14.852 (base) -> 14.748, against HR's 14.814, centre offset unchanged at
  0.099 Mpc/h. Base sat 0.038 dex above HR and run 2 sits 0.066 below -- pulling
  154 satellites out of the smooth component costs the host mass, as it should,
  and this is a mild overcorrection rather than fragmentation.

### 6.1 The ceiling for these 154 targets, and how far run 2 is from it

*Measured*, jobs 35361-35363: `HG_WHICH=hr`, the true HR tiles from run 2's own
`tiles.npz` through the identical splice -> Rockstar -> compare chain.

**The ceiling is 151/154 (98.1%)**, and the `R_vir` count is 227 -- identical to
section 8.1's, which it must be, since it is the same four HR tiles. The host is
preserved exactly (offset 0.000 Mpc/h, `log10 Mvir` 14.831 against base's
14.852).

So the wider selection did **not** cost per-target sensitivity: the geometry
permits near-complete recovery of all 154, and 72/154 is not a saturated number.

| | frozen | run 2 | ceiling | *derived*: share of the achievable range |
| --- | ---: | ---: | ---: | ---: |
| supervised, `1 x r_vir` `>= 25%` | 3 | 72 | **151** | **0.47** |
| `2 x` / `>= 10%` | 7 | 107 | 151 | 0.69 |
| `3 x` / `>= 5%` | 19 | 126 | 152 | 0.81 |
| `4 x` / `>= 2%` | 32 | 137 | 152 | **0.88** |
| subhalos in `R_vir` | 11 | 128 | 227 | 0.54 |
| subhalos >= 50p, whole box | 30027 | 30256 | 30436 | 0.56 |

**Read the first and last rows of the sweep together, because they decompose the
residual.** Loosening the threshold from `1 x r_vir / >= 25%` to
`4 x / >= 2%` buys the ceiling **one** extra target (151 -> 152) and buys run 2
**sixty-five** (72 -> 137). A field holding genuine HR subhalos is
threshold-insensitive; run 2's objects are marginal.

*Derived.* The failure is therefore **not** missing objects. At the loosest
reading run 2 has recovered 88% of what the geometry allows -- the material is
in roughly the right place, in roughly the right amount, for 137 of 152
achievable targets. What it lacks is the last factor in concentration and
placement that turns "a clump near there" into "the halo Rockstar matches".
Halving the gap at `1 x / >= 25%` is a question about the objective's remaining
slack, not about whether it addresses the right objects.

### 6.2 Raising the ceiling: two knobs, and they move different bounds

*Measured*, 2026-08-22, jobs 35592 (curve) and 35606/35607/35609 (the 16-tile
ceiling). Section 6.1 measured two ceilings on the same four tiles and only one
of them can be raised:

- **151 of 154** supervised targets. Saturated; there is nothing here.
- **227 of HR's 506** subhalos in `R_vir`. Section 8.1 of
  `sr2_gather_finetune.md` reads that number straight off the coverage: the four
  trained tiles hold **42.4%** of the host's Lagrangian sites, and 227/506 =
  0.449. **This bound is not a property of the objective at all.** It is a
  property of which tiles the run is allowed to touch, and the splice will not
  move material the run never had.

So there are two knobs, they cost differently, and they raise different things.

**`--n-tiles` raises the ceiling itself.** More tiles, more of each subhalo's
Lagrangian material inside the trained field, more objects the splice could
possibly rebuild. Cost is linear and mild: `delta` is `n * 6 * 64^3` (1.6M
parameters a tile), the member sets grow roughly with covered material, and the
Rockstar gate costs exactly what it costs now.

**`min_num_p` raises how much of that ceiling anything is asking for.** It does
**not** move the ceiling. It matters because the two populations in section
6.1's table are badly mismatched: the gate counts subhalos at `>= 50p`, the loss
supervises at `>= 200p`, and section 8.1's bin table says only **151 of the 506**
are `>= 200p`. *Derived*: about seven in ten of the objects the headline row
scores are objects nothing in the loss ever mentions. Run 2's 128 in `R_vir` is
therefore a mix of ~58 supervised targets and ~70 objects that came along for
free, and the cheapest way to raise it may be to stop scoring on unsupervised
material. The pair sums are `O(N^2)` per set, so the added sets -- the small ones
-- are the cheap ones.

**The tile *ordering* is a third, free knob.** `host_tiles` ranks tiles by the
**host's** Lagrangian sites, which is the right supervision for the host and is
not obviously the ordering that maximises recovered *subhalos*: a satellite's
material sits where it sits. Ranking by `R_vir` subhalo material instead costs
nothing and might reach the same ceiling in fewer tiles.

`scripts/features/gather_coverage_curve.py` measures all of this **without a
halo finder**: one grouped pass over the box's subhalos gives each one its full
sparse occupancy over Lagrangian tiles (the same
`subhalo_gather.subhalo_home_tiles` the loss selects with, now with
`return_occupancy`), and from that, per rung of a tile ladder:

| column | what it is |
| --- | --- |
| `host_site_coverage` | the 42.4% number, generalised |
| `live_ge_{0.9,0.7,0.5}`, `sum_live` | of HR's `R_vir` subhalos, how many have that fraction of their **own** member particles inside the trained tiles -- **the predicted ceiling** |
| `supervised.<min_num_p>` | sets the loss would actually get, under all of the real cuts (home tile trained, purity, live fraction) |
| `delta_params`, `member_particles`, `sum_n_squared` | what a run at that rung costs |

**The predictors are calibrated, not asserted.** The 4-tile rung has a measured
answer -- 227 -- so the script prints every predictor against it and the curve
is only readable through whichever one reproduces it. That check is the reason
to trust a rung that has never been gated; it is not a substitute for gating the
rung that gets chosen.

Then the chosen rung gets a **measured** ceiling for the price of one splice and
one Rockstar, with no optimisation at all: `--steps 0` writes `tiles.npz` from
the frozen forward, and `FF_WHICH=hr` splices the true HR tiles through the
identical chain that produced 227. Section 9 has the commands.

#### The curve

*Measured*, job 35592, 48,522 resolved subhalos in set8, 506 of them in `R_vir`.
Coverage saturates far faster than the tile count suggests:

| `--n-tiles` | host coverage | predicted ceiling (`live>=0.5`) | sets @200p | sets @50p | free params |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.424 | 185 | 154 | 480 | 6.3M |
| 8 | 0.678 | 348 | 335 | 1132 | 12.6M |
| 12 | 0.855 | 437 | 482 | 1691 | 18.9M |
| **16** | **0.959** | **495** | 625 | 2198 | 25.2M |
| 24 | 0.995 | 505 | 839 | 3059 | 37.7M |
| 32 | 1.000 | 506 | 1087 | 3905 | 50.3M |

**The predictors are a conservative lower bound, and the calibration says by how
much.** At the one rung with a measured answer they all undershoot: `live>=0.5`
gives 185 and `sum_live` 187.5 against the measured **227**, i.e. -18%. Rockstar
recovers objects whose material is only *partly* replaced, which no threshold on
the live fraction can express. So the curve is read as a floor on the ceiling,
and its shape -- saturation by ~16 tiles -- is the part that decides anything.

#### The 16-tile ceiling, measured

*Measured*, jobs 35606 (`--steps 0`, 16 tiles, ~2 min GPU) / 35607 (splice) /
35609 (Rockstar + compare), `FF_WHICH=hr`, the identical chain that produced 227.

| | 4 tiles | **16 tiles** | HR |
| --- | ---: | ---: | ---: |
| subhalos in `R_vir` | 227 | **512** | 506 |
| ... as a fraction of HR | 0.449 | **1.012** | 1 |
| supervised targets | 42/43, later 151/154 | **256/256** | -- |
| host `log10 Mvir` (HR 14.814) | 14.831 | **14.814** | 14.814 |
| host centre offset | 0.000 | 0.001 Mpc/h | -- |
| box hosts `>=200p` vs base | +29 | +125 | -- |
| box spliced | 0.78% | 3.13% | -- |

**The `R_vir` ceiling is no longer a binding constraint.** 227 -> 512 is 2.26x,
it is 1.012 of HR's own count, and the host survives *better* than at four tiles
(14.814 against HR's 14.814, offset 0.001 Mpc/h). The bin table is flat across
every mass: 1.014 / 1.035 / 0.967 / 1.021 / 1.000 from 50p to 2000p+. The
frozen control on the same 256 targets scores 14/256, so the gate has not gone
blind at the wider tiling -- it has 18x dynamic range here.

*Derived*: the predicted 495 against a measured 512 is a -3.4% error, against
-18% at four tiles. The predictor tightens as coverage rises, exactly as it must
when the count saturates at HR's own.

#### The tile ordering: asked, and the answer is no

*Measured*, same job. Ranking tiles by `R_vir` subhalo material instead of by the
host's sites is **worse at almost every rung** -- -75 at four tiles, -87 at six,
-23 at eight -- and it saturates at 0.979 host coverage rather than 1.000. The
host-site ranking already concentrates the satellites, because the satellites
are the host's own Lagrangian material. `--tiles` stays as an escape hatch; the
default ordering stands, and this is closed.

#### What binds now: `max_sets`, and the `O(N^2)` pair sums

*Measured.* The 16-tile probe reports `647 homed in these tiles, 625 past purity`
and then **256** sets, at a median live fraction of 0.998 -- the live cut removed
nothing. `max_sets = 256` did. It never bound at four tiles (154 sets) and it
binds from about eight upward, so **a wider tiling left at the default supervises
41% of what it covers.** It is now reported as a cap rather than folded into the
live-fraction count (`cap_binds`, `n_dropped_by_cap`), and `FF_MAX_SETS` is
wired through the job scripts.

Raising it is not free, and the curve priced it. `sum_n_squared` -- what the pair
sums cost -- is 8.8e8 at four tiles and **2.39e10 at sixteen**, a factor of 27,
because a handful of massive satellites become fully live and one set of
`N = 10^5` is `10^10` on its own. The cap keeps the largest sets, so it is
simultaneously the coverage dial and the cost dial.

*Derived, and it is the useful half*: **`sum_n_squared` is almost flat in
`min_num_p`** -- 8.802e8 at `>=200p` against 8.841e8 at `>=50p` on the same four
tiles, +0.4%. Only the background term grows (630k -> 1.97M particles). So
widening supervision from 200p to 50p, which triples the set count and is the
knob that addresses the seven-in-ten unsupervised share, costs essentially
nothing in the driver. **Widening the mass cut is cheap; widening the tiling is
what is expensive.**

*Not established.* No *optimised* run has been done at any tiling above four --
512 is what the geometry permits at sixteen tiles, not what the objective
reaches. The wall clock of a real 16-tile run is untested and the `O(N^2)`
scaling above puts a 2000-step run near or past the 4 h limit unless `max_sets`
or a per-set particle subsample bounds it.

## 7. What is not established

0. ~~**No rung of section 6.2's curve above 4 tiles has been gated.**~~
   **Gated: 16 tiles reaches 512 of HR's 506** (section 6.2), so the `R_vir`
   ceiling is removed as a constraint. What is open is the *optimised* run at
   that tiling -- and its two new constraints, the `max_sets` cap and the 27x
   pair-sum cost.
1. ~~**The ceiling for this target list is unmeasured, and it gates the
   reading.**~~ **Measured, and it is high**: section 6.1. 151/154, so the wider
   selection cost no per-target sensitivity and 72/154 sits at 0.47 of the
   achievable range rather than near saturation. What this leaves open is the
   *reason* for the remaining factor of two, and section 6.1 narrows it: at the
   loosest threshold run 2 is already at 0.88, so the gap is concentration and
   placement at the margin, not absent objects.
2. **High-k power is unguarded and badly overshot.** *Measured*, displacement
   power above `k_split`, relative to HR:

   | | frozen | run 1 | run 2 |
   | --- | ---: | ---: | ---: |
   | high-k power / HR | 0.352 | 2.65 | **5.50** |

   Still climbing at step 2000. Nothing in the loss or the guards penalises it --
   `low_k` watches only the LR scale. The objective is a valid specification of
   *a bound halo at a location*; it is **not** a complete specification of a
   *field*. It needs the mirror of the low-k guard. *Design*: a hinge against
   exceeding HR's high-k power, not an L2 anchor -- section 3.3 of
   `sr2_gather_finetune.md` measured an L2 guard blurring for the same reason an
   L2 objective blurs.
3. **A free field is not a generator.** 6.3M unconstrained parameters that see
   the true member sets say what the objective *permits*, not what a
   convolutional operator applying one learned rule at every site can reach. The
   capacity question from `pilot_steps_2_4.md` section 2 is untouched here.
   *Being acted on*: `docs/sr2_member_gather_training.md` builds the generator
   fine-tune and quantifies the capacity gap at ~660x fewer parameters per set,
   shared rather than free. Not yet run.
4. **One host, one box, in-sample.** The supervision is the true subhalo
   positions and memberships. No generalisation claim is available from any run
   in this document. *Being acted on*: the owner arrays that gated this to one
   box now exist for set0-set12 (~18 min CPU each -- it was a missing job, not a
   data limit), and `docs/sr2_member_gather_training.md` selects 40 training and
   16 held-out hosts, 7,560 supervised sets. Setup only; nothing trained.
5. **`bound_frac` is this module's own statistic.** It is softening-dependent and
   it is a surrogate. Per `occupation-ratio-is-gameable` and
   `tile-overfit-proxy-exploitation`, the gate stays real Rockstar.

## 8. Module map

| file | role |
| --- | --- |
| `features/member_gather.py` | the sets, the statistics, the loss |
| `features/free_field_gather.py` | the free-field test; writes the splice source |
| `features/gather_coverage_curve.py` | section 6.2: the ceiling as a function of the tiling, without a halo finder |
| `reward/compare_gather_catalog.py` | the gate, plus the section 5 sweep and miss profile |
| `slurm/free_field_gather_gpu.sbatch` | one GPU job, shakeout or full |
| `slurm/submit_free_field_gather.sh` | shakeout -> optimise -> splice -> Rockstar -> compare |
| `slurm/gather_coverage_cpu.sbatch`, `slurm/submit_gather_coverage.sh` | the coverage curve, CPU, minutes |

Conventions match `features/bound_discriminator.py` exactly -- comoving Mpc/h,
peculiar km/s, Msun/h, `G = 4.30091e-9` -- so `tests/features/test_member_gather.py`
pins this module against that one's numpy on identical inputs. Absolute energies
are softening-dependent and are not physics; what is read is the comparison
between fields at one softening.

Tests: `tests/features/test_member_gather.py` (26),
`tests/reward/test_compare_gather_catalog.py` (8).

## 9. Reproduce

```bash
# selection only, minutes -- how many sets, and what each can reach
SHAKEOUT_ONLY=1 bash scripts/slurm/submit_free_field_gather.sh

# the full chain: optimise (GPU ~28 min) -> splice -> Rockstar -> compare (CPU)
FF_LABEL=_v2 bash scripts/slurm/submit_free_field_gather.sh

# section 6.1: the ceiling for THIS 154-target list
HG_WHICH=hr HG_SWEEP=1 \
HG_RUN_DIR=$DMSR_REWARD_ROOT/free_field_gather/set8_h271800_v2 \
  bash scripts/slurm/submit_gather_rockstar.sh

# run 1, for the record: no centre term, no gradient mask
FF_W_CENTRE=0 FF_MASK_GRAD=0 bash scripts/slurm/submit_free_field_gather.sh

# --- section 6.2: raising the ceiling ---------------------------------------
# 1. the curve (CPU, minutes): how high could the ceiling go, and at what cost
bash scripts/slurm/submit_gather_coverage.sh

# 2. MEASURE the ceiling at the rung the curve picks. --steps 0 writes tiles.npz
#    from the frozen forward without optimising anything, and FF_WHICH=hr
#    splices the true HR tiles -- the same construction that measured 227.
#    Done for 16 tiles: 512 of HR's 506, 256/256 supervised (jobs 35606-35609).
FF_WHICH=hr FF_STEPS=0 FF_N_TILES=16 FF_LABEL=_ceil16 SKIP_SHAKEOUT=1 \
  bash scripts/slurm/submit_free_field_gather.sh

# 3. then a REAL run at that rung. Two things the ceiling probe exposed:
#    FF_MAX_SETS must be raised or the tiling supervises 256 of 625, and the
#    pair sums are 27x four tiles -- so widen the mass cut (nearly free) before
#    widening the tiling further.
FF_N_TILES=16 FF_MIN_NUM_P=50 FF_MAX_SETS=1024 FF_LABEL=_t16p50 \
  bash scripts/slurm/submit_free_field_gather.sh

# an explicit tiling (e.g. the subhalo-ranked ordering the curve reports)
FF_TILES=12,13,20,21 bash scripts/slurm/submit_free_field_gather.sh
```

Artifacts under `$DMSR_REWARD_ROOT/free_field_gather/<box>_h<id><label>/`:
`targets.json` (the shakeout), `summary.json` (config, full history, verdict),
`metrics.jsonl` (one row per eval step), `subhalos.json` (per set, and what the
gate reads its target list from), `tiles.npz` (the splice source). The Rockstar
comparison lands in
`$DMSR_REWARD_ROOT/flow_rockstar/compare/<box>__freefield_<run>.json`.
