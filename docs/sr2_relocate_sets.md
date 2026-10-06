# Choosing the group's particles from the SR field: arms A, B and C

**Status.** Built and unit-tested 2026-09-15, **not run**. Nothing here is a
measurement. Numbers quoted from earlier work are tagged *measured* and carry
their source; everything about this experiment is *design* until the check
phase writes a `pool.json`.

## 1. The defect this is aimed at

`gather-holdout-rockstar-gate` is the one halo-finder win on this line:
*measured*, the tuned `self` arm reproduces HR's subhalo mass function on
held-out set9 (20 -> 366 against HR's 369). Its residual defects are placement
ones, and two of them are recorded:

* `host-damage-is-mostly-reparenting` -- *measured*, the -320 host count is ~88%
  bookkeeping, with 41 massive intact hosts re-parented by **central
  over-concentration**.
* `radial-profile-is-flat-in-host-mass` -- *measured*, HR's satellite radii
  scale 6.5x with host mass and the candidates sit flat at ~0.1 Mpc/h.

And `docs/sr2_member_gather_training.md` records the symptom that ties them
together: the **supervised** centres barely move while the **detected** catalog
concentrates centrally. The objective is building objects, and it is building
them in the wrong place.

The hypothesis this experiment tests is that the membership is why. Every loss
term is a statistic of a particle collection the loss is **not allowed to
choose** -- HR's owner array names it. If SR2's material for a given subhalo is
not the material HR used, the loss charges the generator for failing to bind
particles that were never going to be there, and rewards it for dragging in
whatever is nearest, which is the host's centre.

## 2. What the relaxation is, and what it is not

It is **not** `soft_membership`. That module made membership a latent variable
re-solved from the current field every step, and `soft-membership-relaxation`
records the result: *measured*, no variant beat its own control on held-out
`bound_hard`, and `coarse_pin` was half the control.

Here membership is re-chosen **once, before training, from the frozen field**,
and then frozen for the run. One selection, written to disk, fully inspectable
before a GPU-hour is spent. The existing loss then runs on it unchanged --
`tests/features/test_relocate_sets.py::test_the_existing_loss_runs_on_relocated_sets`
is the statement that nothing in `member_gather_loss` knows the difference.

## 3. The three arms

| arm | particle groups | initial group radii | flag |
| --- | --- | --- | --- |
| A | HR memberships | frozen-SR radii | `--relocate off` |
| B | new local SR groups | frozen-SR radii | `--relocate frozen` |
| C | new local SR groups | HR radii | `--relocate hr` |

Same HR targets, same loss, same split, same training setup, same grouping
procedure and the same candidate directions for B and C. **B measures the effect
of replacing the membership; C adds the requested placement change.** The two
differ in exactly one scalar per set: the requested host-centric radius.

## 4. The selection, step by step

`src/cosmo_sr/features/relocate_sets.py`, one pass over a finished
`MemberSets`. Sets are processed **largest first** and particles are claimed as
they are taken, so no particle serves two groups -- a big set has one or two
viable pieces of material at a given radius and a 200-particle one has many.

1. **Each HR subhalo is the template for one new group.** Its particle count
   `N_j`, its physical-loss targets and its normalised host distance
   `rho_j = |c_j - c_h| / R_h` are kept together, which is what preserves the
   relation between subhalo mass and radius.

   `N_j` is the size of the collection that produced `sets.ref` -- the live
   members **plus** the out-of-tile stragglers that entered the hybrid reference
   at frozen coordinates. Not `n_live` (which would ask the group to match a
   reference measured on more particles than it holds) and not `num_p` (fewer).
   The relocated group is wholly live, so it carries no stragglers at all.

2. **Several locations are proposed on the sphere of radius `rho_j R_h`.** The
   first direction is always the group's own frozen direction, so arm B's
   minimal answer -- the material already there -- is always in the running.
   The rest are a Fibonacci spiral under a per-set random rotation: at 8
   proposals, random draws clump, and a set whose proposals all landed on one
   side would report "no material here" when the failure was the sampler's.

3. **At each location the `N_j` nearest unclaimed particles form a candidate.**
   Scored by compactness and velocity coherence, both as **one-sided** log
   ratios to HR's own `r_rms` and `sigma_v`, plus the radius error in units of
   the tolerance. One-sided because the loss that follows is two-sided and
   hinged: a selector that minimised spread outright would prefer an
   over-collapsed core to the object HR actually had, and `rrms` would then have
   to undo it.

4. **Local swaps.** The proposal's warning is the design constraint here: *seeds
   at the correct radii are not enough, because their nearest-particle groups
   can have displaced centroids.* Each refinement pass recomputes the achieved
   centroid, projects it onto the requested sphere, and re-selects the nearest
   `N_j` about the corrected point. Selecting about the centroid is what
   improves compactness; the correction is what improves the radius error.

5. **The constraint is checked on the selected particles**, not on the seed:
   `| |xbar_j - c_h| / R_h - rho_j | <= eps`, with `eps = 0.05` by default.
   A group outside it is **kept and flagged**, never dropped -- dropping would
   make the three arms supervise different catalogs and their Rockstar gates
   incomparable. So is a **starved** set, which keeps its HR membership and is
   an arm-A row inside a B/C run; `MemberSets.relocated` is the flag every
   downstream split needs.

6. **The memberships are frozen and the existing loss runs.** `centre_self` is
   reset to the newly selected centroid, so `--centre-mode self` -- the mode
   every working arm uses -- costs exactly zero at step 0 and anchors the group
   where it was put while the internal terms gather it. `centre_target` is the
   requested point on the sphere, so the residual radius error is exactly
   `|centre_self - centre_target|`.

### 4.1 Where the host is

There is no SR halo catalog at pool-build time, so `c_h^SR` and `R_h^SR` are
taken as `c_h^HR` and `R_h^HR`. Every other radial quantity in this line
(`w_radius`, `shell_radii`) already measures from that same point, and a second
definition of "where the host is" would make these arms unreadable against the
results on disk.

### 4.2 What is deliberately not recomputed

`sets.ref` -- every HR target -- is carried through untouched. **The replacement
particles form no subhalo in HR**, so recomputing HR's statistics on their ids
would supervise the run to reproduce a random blob, and every metric would still
look fine. The one exception is `ref["bound_scale"]`, which is not a target but
a temperature (`tau G m N / r_rms`): `N` changes, so leaving it would scale the
unbinding surrogate by the wrong particle count.

The local non-member background (`d6`) **is** rebuilt, about the new centroid at
the same physical radius the original used. Keeping the old rows would normalise
the compactness of an object against a background a megaparsec away.

## 5. The gate has to look in the right place

`compare_gather_catalog.target_hit_rate` searches for each supervised target
around **the HR subhalo's catalog centre**. That is the right address for arm A
and the wrong one for B and C, which put the group somewhere else on purpose --
scoring them at HR's position would read ~0 whatever the run achieved.

So a relocated set now writes `target_pos` (and a `relocated` flag) into
`subhalos.json`, and the comparator searches there instead, reporting
`searched_at` per row and `n_targets_relocated` per catalog. It is emitted
**only** for sets that were actually relocated, so every arm-A number already on
disk stays bit-identical.

## 6. Running it

Two phases, and the first is not optional.

```bash
# phase 1: build every arm's pool, write the relocation report, train nothing
bash scripts/slurm/submit_relocate_arms.sh

# phase 2, only after reading phase 1
RA_PHASE=train bash scripts/slurm/submit_relocate_arms.sh
```

Phase 1 answers "does the material exist at the requested radii" for 2 shakeout
jobs instead of 3 x 10 GPU hours. Read `.relocation` in each arm's `pool.json`:

| field | what a bad value means |
| --- | --- |
| `frac_within_eps` | far below 1: the material is not at the requested radii. Widen `MG_RELOCATE_EPS` or look at *which* radii starved -- do not train anyway. |
| `n_starved` | above ~0: those sets kept their HR membership and are arm-A rows inside a B/C run. Every downstream split must exclude them. |
| `kept_hr_member_frac` | near 1: the selector re-found HR's own particles and the arm is arm A under another name. |
| `rrms_over_hr` | >> 1: material was recruited from far away and the "group" is a blob wearing an HR target. |

*Design.* The tolerance is achievable in the real run and is not in the unit
test's toy, and the arithmetic is worth writing down: centroid precision is
about `r_rms / sqrt(N)`. Production has 0.195 Mpc/h between HR particles,
`r_rms ~ 0.4` and `N ~ 200`, so ~0.03 Mpc/h against a cluster `R_vir` of ~2.1 --
comfortably inside `0.05 R_vir`. The toy puts 4096 particles on a 6.25 Mpc/h
lattice, where the same arithmetic gives ~0.3 `R_host`, which is why
`tests/features/test_relocate_sets.py` runs at `TOY_EPS = 0.3` and says so.

## 7. What to measure, and what not to believe

**The verdict the training jobs print is not the result.**
`tile-overfit-proxy-exploitation` is the reason: *measured*, a member-set
surrogate on this line reached +255 while real Rockstar showed no gain at all.
The result is the held-out Rockstar gate, and **arm A is its own baseline** --
run all three through `submit_gather_holdout_rockstar.sh` with
`HG_BASELINE_ARM=<the arm A run>`.

Measure **both halves**, because this experiment exists to move the second:

1. **Formation yield.** How many chosen groups came back as distinct Rockstar
   subhalos, at what masses, at what radii. `supervised_targets` in the compare
   JSON, now searched at the relocated address.
2. **The full catalog.** Whether the unwanted central objects are still there,
   and what happened to the hosts. The abundance and host-count rows.

A B/C arm that raises yield and leaves the central over-concentration untouched
has not tested the hypothesis in section 1; it has only moved the supervised
objects. And note `vel-rms-guard-is-a-constant` before reading any GUARD FAILED
line: *measured*, 1 of 56 arms has ever passed the `vel_rms` guard, so that line
alone says nothing -- read **which** guard fired.

## 8. Open, and deliberately not built

The proposal's **soft version** -- membership weights `w_ij >= 0` with
`sum_i w_ij = N_j`, `sum_j w_ij <= 1` and `sum_i w_ij x_i = N_j a_j`, minimising
a local phase-space assignment cost -- is not implemented. It is the right
follow-up *after* the hard-set arms, exactly as the proposal orders it, and its
exact-centroid constraint is feasible only when the available material makes it
so, which is precisely what phase 1 measures.
