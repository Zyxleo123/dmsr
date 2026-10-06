# Membership as a latent variable

Status: **RAN, and the gate is POSITIVE.** Built 2026-09-14; ladder run
2026-09-15 (jobs 54576-54580); held-out Rockstar gate 2026-09-16. `coarse_soft`
beats the hard-mask control on held-out set9 on hosts, subhalos, mass-function
shape and unbound crowding, while sharing only ~25% of its material with HR's
subhalos. Sections 1-5 are the pre-registration, written before the run and left
unedited; **section 7 is the result.**

## 1. The defect

Every term of the member-gather loss (`docs/sr2_member_gather.md`) is a
statistic of a particle collection the loss is **not allowed to choose**. HR's
owner array names it, once, before the run. The generator's only move is to
push those named particles around.

That is a strong assumption and it has never been tested. When SR2's material
for a given subhalo is not the material HR used, the loss charges the generator
for failing to bind particles that were never going to be there, and rewards it
for dragging in the wrong ones.

The arm works anyway — `gather-holdout-rockstar-gate`, tuned `self` on held-out
set9: 20 → 366 subhalos against HR's 369, the first halo-finder win on this
line. Its *remaining* defects are all placement defects:

| defect | measurement |
| --- | --- |
| radially collapsed satellites | median 0.35 host R_vir against HR's 1.13; 6,496 inside 0.25 R_vir where HR has 593 |
| host damage | −320 hosts, ~88% of it re-parenting rather than unbinding |
| misplaced small-scale power | +3.4x at k=4–8 in the *wrong spots* (0.915 → 0.318) |

Those are what a wrong-material objective looks like from the outside: the
objects are built, and they are built out of the wrong stuff in the wrong place.

## 2. The relaxation

Membership becomes a weight `P_ij` — how much of particle `i` belongs to
candidate `j` — solved from the **current SR field** between generator steps.

HR keeps supplying two things and loses only the third:

* **the region** the object's material comes from — the prior. `n_c` of an LR
  cell's `8^3 = 512` sites belong to this subhalo ⇒ *every* SR particle of that
  cell starts at membership `n_c/512`, the original members included and not
  privileged.
* **the physics** the object must satisfy — `sets.ref`, untouched. The target
  does not move.
* ~~the identities~~ — the current field decides those.

The objective is

```
L(theta) = min_P [ sum_j d(s_j^SR(theta, P), s_j^HR) + lambda * KL(P || Pi) ]
```

subject to a per-particle budget of one (shared with a background/host
category) and a per-candidate mass target. Fixing `P = Pi` at the HR mask
recovers the current method exactly; `lambda` is the only knob that
interpolates.

### The solve

A few EM sweeps of a phase-space mixture with a Sinkhorn dual for the mass
constraint:

```
P_ij  ∝  Pi_ij * exp(alpha_j) * exp( -(|x_i - c_j|^2/sigma_r^2
                                    + |v_i - vbar_j|^2/sigma_v^2) / (2 lambda) )
```

normalised per particle against a background term. Position **and** velocity,
which is the quantity Rockstar groups on. All under `no_grad`: the generator
step sees `P` as a constant, so the gradient it feels is the familiar gathering
and binding one — move the particles, not the labels.

### Three things that are load-bearing, not incidental

**The per-particle budget.** Without it, several candidates spend the same
convenient particles. Solved jointly over the batch, not per set — a per-set
solve cannot see the competition.

**The mass target.** Without it, a candidate discards everything and keeps a
tiny cold core that scores perfectly on every moment. The Sinkhorn dual also
rescues a window that is too narrow to fill: if the candidate is starved, the
dual widens it until the mass is there.

**The window's dispersions are `max(HR, candidate)`, not HR's.** Measured: the
frozen field is ~6x wider and ~20x hotter than HR per set, which puts every
candidate particle ~400 sigma² out. A window at HR's sigma alone selects
*nothing* — zero membership, zero gradient — the same saturation failure, and
the same continuation fix, as `_bound_temperature`. A window at the
*candidate's* own sigma is the unsafe direction: free to contract onto whatever
is already cold and call that the object. Bounded below by HR's, it cannot.

## 3. The risk, named before it is run

A prior uniform over a coarse cell **is** a Gaussian window, and Gaussian
windows already failed on this line: `gather-loss-never-touched-member-sets` —
their statistics described the surrounding host material rather than the
object. One LR cell is 1.5625 Mpc/h a side against a 0.15 Mpc/h search radius.

So the prior is the **support and the KL anchor, never the membership**. The
membership is what the phase-space solve returns. Whether that is enough is the
question `coarse_pin` below is built to answer, and it is a falsification arm:
it runs the coarse prior with the membership *frozen at it*.

The second risk is the one the advisor named and no unit test can reach:
weighted groups can look physically excellent without being distinct,
detectable subhalos. The mass budget and background competition push against
it; only Rockstar settles it.

## 4. The ladder

`bash scripts/slurm/submit_soft_membership_ladder.sh` (shakeout first;
`LADDER_GO=1` for the arms).

| arm | prior | lambda | what it answers |
| --- | --- | --- | --- |
| `control` | — | — | the hard arm, re-run under this ladder's shared config |
| `hard_pin` | hard | 1e6 | **correctness**: must reproduce `control` |
| `coarse_pin` | coarse | 1e6 | the coarse window ALONE, membership frozen |
| `hard_soft` | hard | 1 | HR's mask, opened — the conservative relaxation |
| `coarse_soft` | coarse | 1 | the proposal as written |

Shared and pinned, so the only variable is membership: rung `all_blocks`,
`centre_mode=self`, `w_d6=0` (the minimal set of `loss-term-ablation`), host
preservation guard on.

### Read in this order

1. **`hard_pin` vs `control`.** Same objective by construction. A gap is a bug,
   and everything downstream is void until it closes.
2. **`soft_mass_ratio`**, every arm, every eval. Must sit at ~1.0. Below it,
   lambda made the window unfillable and the candidates hold less mass than
   they claim — a misconfiguration, not a result.
3. **`soft_hr_fraction`.** 1.0 is the hard arm. A collapse toward 0 is the
   relaxation walking off the object entirely.
4. **`coarse_pin`.** If it is catastrophic, the coarse prior is a Gaussian
   window and only the solve makes it survivable — then read `coarse_soft`
   against `coarse_pin`, not against `control`.

### Cost

The pool is `(1 + ratio)` times the member count against an `O(pool^2)`
estimator: ratio 2 is ~9x the hard arm's pair bill, and the hard arm already
ran 10.6 h against a 12 h limit at 8000 steps. `MG_MAX_SET_PARTICLES` bounds it
(the ladder sets 768) and the stage-1 shakeout measures it —
`pool.json`'s `soft_pools.sum_pool_squared` against the hard arm's
`sum_num_p_squared` is the real ratio, before any of it is spent.

## 5. What this does not claim

The relaxation targets *which material*, not *where it goes*. Internal binding
statistics are unchanged when a whole clump is translated, so softer membership
supplies no outward-placement signal on its own — the radial defect of section
1 may survive it untouched. That is a hypothesis this ladder does not test.

> **Held.** Measured median r / host R_vir: HR 1.128, frozen 0.533, control
> 0.312, `coarse_soft` 0.347. A marginal improvement, still inverted. See
> `radial-placement-is-inverted`; isolation remains the only lever that moves
> this axis.

And the verdict in `summary.json` is not the result. Every number in it is a
member-set surrogate the objective computes about itself, and this line has
measured such a surrogate reaching +255 while real Rockstar showed no gain at
all (`tile-overfit-proxy-exploitation`). This arm is *more* exposed to that than
any before it — the loss now chooses its own particles, so "the statistics look
right" and "a halo finder sees a halo" have moved further apart, not closer.
The result is the held-out whole-box Rockstar gate, read in the edited region.

> **This was the operative warning.** The surrogate read was recorded as a
> failure (`no variant beats its own control on held-out bound_hard`) and the
> gate then reversed it. The reason is in section 7.2: under soft membership
> `bound_hard` is a weighted mean over *the particles the arm chose*, so
> `coarse_soft`'s 0.106 and control's 0.093 have different denominators and are
> not comparable in either direction. Do not rank soft arms against hard ones on
> any member-set statistic.

## 6. Where it lives

| file | what |
| --- | --- |
| `src/cosmo_sr/features/soft_membership.py` | the prior, the solve, the weighted statistics |
| `src/cosmo_sr/features/member_gather.py` | `soft_*` config, the `MemberSets.pools` slot, the branch in `member_gather_loss`, weighted sources in the pair potential |
| `scripts/features/finetune_member_gather.py` | `--soft-*` flags, pool attachment per host |
| `scripts/slurm/member_gather_train_gpu.sbatch` | `MG_SOFT_*` |
| `scripts/slurm/submit_soft_membership_ladder.sh` | the ladder |
| `tests/features/test_soft_membership.py` | 20 tests |

The host-preservation guard runs `soft_membership=False` and that is deliberate:
it says "the frozen field already resolves these hosts, do not break them",
which is an identity statement. A relaxed membership could satisfy it with a
*different* collection of particles — report a host intact while the host HR
named came apart.


## 7. RESULT

### 7.1 The correctness gate passed

`hard_pin` must reproduce `control` — same objective by construction. At matched
steps on held-out set9:

| step 500 | control | hard_pin |
| --- | --- | --- |
| train bound | 0.095 | 0.097 |
| 2T/\|W\| | 37.9 | 37.5 |
| hold bound | 0.075 | 0.077 |
| iso | 1.17x | 1.18x |
| low_k | 0.0137 | 0.0140 |
| high-k | 0.37x | 0.37x |

Agreement 1-3% throughout, and the residual is accounted for: at the shipped
`soft_prior_eps=1e-3` every pool non-member carries ~1e-3 of weight, and
`soft_hr_fraction` reads 0.9875 — exactly that contamination. The relaxation is
a faithful superset of the hard arm.

### 7.2 Which arms were actually gated — read this before any claim below

**Only `control` and `coarse_soft` have a Rockstar catalog.** `hard_pin`,
`hard_soft` and `coarse_pin` were never gated; their catalogs do not exist.
Every statement about those three rests on the training surrogate and the
field-level guards, and the two classes are not interchangeable:

| arm | gated? | hr_frac | is its surrogate comparable to control's? |
| --- | --- | --- | --- |
| `control` | **yes** | — | — |
| `coarse_soft` | **yes** | 0.2484 | **no** — different particles |
| `hard_pin` | no | 0.9875 | yes — membership never moved |
| `hard_soft` | no | 0.9933 | yes — membership never moved |
| `coarse_pin` | no | 0.1809 | **no** — different particles |

That table decides how much each claim is worth:

* **`hard_pin` and `hard_soft` are nulls — well supported without a gate.** At
  `hr_frac` ~0.99 they are measuring HR's own particles, so their member-set
  statistics *are* comparable to control's, and they match it to 1-3%. The
  membership-independent field guards are identical too (low_k 0.015, high-k
  0.363/0.355, vel_rms 0.719/0.720 against control's 0.015/0.363/0.721). A
  Rockstar gate would be measuring a model that did not move.
* **`coarse_pin` is NOT established as destructive.** Its `bound_hard` (+0.044
  vs control's +0.086) is precisely the statistic this document says cannot rank
  a soft arm against a hard one — at `hr_frac` 0.181 it has a different
  denominator. What survives is weaker: it is the only arm with
  `generalised_by_surrogate: False`, and it is worse on the membership-
  independent field guards (low_k 0.023 vs 0.015, high-k 0.732 vs 0.363). But
  `density-power-cannot-see-subhalos` says P(k) cannot see this line's target at
  all, so field guards are poor evidence about halo abundance specifically.
  **The honest status is: strongly suspected, not measured.** Gating it is the
  only way to close the falsification arm properly, and it is 19 minutes.

### 7.3 How far each arm actually went

`soft_hr_fraction` is the share of a solved candidate's mass that is HR's own
named members. `soft_mass_ratio` 1.000 means the candidate holds exactly HR's
particle count, so a fraction of 0.25 says: **same size as HR's object, three
quarters different material.**

| arm | prior / lambda | hr_frac (holdout) | mass | went past one-to-one? |
| --- | --- | --- | --- | --- |
| `control` | hard mask | — | — | no, by definition |
| `hard_pin` | hard / 1e6 | 0.9875 | 1.011 | no |
| `hard_soft` | hard / 1.0 | **0.9933** | 0.999 | **no — see below** |
| `coarse_pin` | coarse / 1e6 | 0.1809 | 1.000 | yes, but field-blind |
| `coarse_soft` | coarse / 1.0 | **0.2484** | 1.000 | **yes** |

`hard_soft` is the ladder's design error and it cost an arm. The hard prior's
gap is `log((1-eps)/eps) = log(999) ~ 6.9` nats against a phase-space cost of
`d^2/2*lambda ~ 1` at lambda=1, so the prior outvotes the field ~7:1 and the
membership never left HR's mask. **The hard prior needs lambda ~ 0.1-0.3**, and
at lambda=1 it is `hard_pin` under another name. Its null result is a statement
about lambda, not about the relaxation.

Two consistency checks on the machinery. `coarse_pin`'s 0.181 is just the mean
LR-cell occupancy, which is what a membership frozen at the coarse prior must
give. And the solve moves 0.181 -> 0.248 while concentrating the weights
(`n_eff`/target 2.42 -> 1.72), so the current SR field agrees with HR's
membership ~37% more than chance — partially, not fully.

### 7.4 What stays one-to-one

The correspondence moved from particles to **objects**. Each candidate is still
bound to one named HR subhalo and must match its `virial`, `r_rms`, `sigma_v`,
`bound_soft`, `d6` (`sets.ref`, measured on HR's actual members and never
recomputed), its particle count, its position (`centre_mode=self`), and its
**Lagrangian support** — at `soft_cell_dilate=0` only cells that contained
members of that subhalo are in the pool, so material is drawn from the ~1.5
Mpc/h neighbourhood the object came from and nowhere else. HR still says which
object, how big, where, and what it must look like. It no longer says which
particles.

### 7.5 The held-out Rockstar gate

set9, edited region, 32 tiles, dilate 2.0 Mpc/h. Both arms gated without
`HG_BASELINE_ARM` (neither had a catalog yet), so the A/B is read across two
region tables against a shared `base`/`hr` rather than from a `verdict_vs`
stage — `verdict` in each `region_counts.json` reads `REGION ONLY` for that
reason, and is not a negative finding.

| | hosts >=200p | subs >=50p | n_halos |
| --- | --- | --- | --- |
| base (frozen) | 3028 | 4541 | 26106 |
| control (hard mask) | 3118 | 9993 | 35458 |
| **`coarse_soft`** | **3148** | **10138** | **35608** |
| HR | 3775 | 12367 | 55846 |

Ahead on both axes, +30 hosts and +145 subs, with neither traded for the other.

Host-region subhalo mass function (host 168880):

| bin | HR | control | `coarse_soft` |
| --- | --- | --- | --- |
| 50-100p | 174 | 173 | 161 |
| 100-200p | 97 | 125 | 87 |
| 200-500p | 56 | 59 | 73 |
| 500-2000p | 34 | 47 | 41 |
| >2000p | 13 | **29** | **20** |
| **total** | **374** | **433** (1.16x HR) | **382** (1.02x HR) |

Control overproduces 16%, concentrated at the massive end. `coarse_soft` halves
that and lands at 1.02x. Summed relative per-bin deviation 1.23 against control's
1.96; the pipeline's own `shape_score` (sub) agrees, 0.1478 vs 0.1530 against a
frozen 0.8045.

**The predicted failure did not occur.** `coarse_soft` trained at `iso_ratio`
1.230 against control's 1.180, i.e. more non-member mass on each object, and the
pre-registered worry was that subhalos would rise while hosts fell — pedestal
re-parenting (`host-damage-is-mostly-reparenting`). Hosts rose. `knn_report`
corroborates: `unbound_x_frozen` 1.52 vs control's 1.87 and `sub50_x_frozen`
0.80 vs 0.67 — less unbound crowding, more small-scale structure retained. Both
arms carry the same lone `hosts_overconcentrated` flag, this line's constant.

### 7.6 Cost, measured

`pool_over_live` 2.374 median (2.650 max), `n_cells_median` 16. Naive pair bill
would be ~5.6x, but the measured per-step cost is **2.88 s/step against control's
2.54 — 1.13x**. `MG_MAX_SET_PARTICLES=768` absorbs nearly all the inflation,
because it lands on the few large sets where the cap binds anyway. Soft arms ran
~3.2 h for 4000 steps, faster than the historical uncapped arms at 6.6 s/step.

### 7.7 What to run next

1. **The hard prior at a lambda that bites** — `MG_SOFT_PRIOR=hard`,
   `MG_SOFT_LAMBDA=0.2`. This is the arm the ladder was supposed to contain and
   did not: HR's mask genuinely opened, rather than the coarse window partially
   closed. It starts from material that already works, so it is the more
   promising of the two directions.
2. **Confirm `coarse_soft` on set10.** One box is one box.
3. **Gate `coarse_pin`.** It is the falsification arm and it is currently the
   one unmeasured claim in this document. 19 minutes closes it.
4. Do **not** re-run `hard_soft` or `hard_pin` at lambda=1 — at `hr_frac` ~0.99
   they did not move, and gating them would measure the control twice.

### 7.8 What the solve picks, radially (diagnostic added 2026-09-18)

`viz_membership_solve.py --radial` asks the one question the evolution and
budget figures cannot: for each distinct pool particle, its largest membership
over all non-background candidates — `max_j P_ij`, i.e. *how subhalo-like is
this particle* — against its distance from the winning candidate's own parent
host, in `R_vir`. Two panels: the joint distribution with a median curve, and
the membership-weighted radial histogram split into HR's own named members and
the material the solve recruited. The gap between those two curves, and how far
out it reaches, is the substitution of section 7.3 made spatial.

`--panel` renders the mean-`<P>` x-z panel of the region heatmap on its own at
slide size (re-rendered from the npz, not cropped from the PNG).

Both are pure functions of the dump, so `MV_NPZ_IN=<npz>` in
`submit_membership_viz.sh` re-plots an existing solve with the CPU job alone —
no GPU, no re-solve. The radial axis needs the parent host's `R_vir`, which the
dump only carries when it ran with `--soft-host-profile`; by default the viz
looks it up in the HR Rockstar ascii by the id in `host_key` (column 5, kpc/h).

### 7.9 Where the solve AIMS, radially — targets are outer, subhalos are not (2026-09-18)

The hypothesis: the outer candidate centres exist — the EM solve places its
targets `c_j` at HR-like radii — but the generator does not condense detectable
bound subhalos there, so the placement defect (`radial-placement-is-inverted`)
is downstream of the membership target, not in it.

**Measured, and it holds.** For host `set9:h165513`, every candidate matched to
its OWN HR parent (join `halo_id` → HR Rockstar ascii → `parent_ids` → host,
normalise by that host's R_vir — the same rule `subhalo_radial.collect` uses, so
it is comparable to the gate):

| | median r / R_vir | > 1 R_vir |
| --- | --- | --- |
| HR subhalos (`centre_ref`) | 1.19 | 58% |
| **solve targets (`c_j` final)** | **1.42** | 58% |
| SR subhalos Rockstar finds (gate) | ~0.35 | — |

The optimized `c_j` sit at HR-like or slightly larger radii — for HR subhalos
beyond 1 R_vir the solve aims at median 1.95 — while the halos Rockstar actually
recovers in the SR field sit at 0.35. The targets are not collapsed; the objects
are.

**Caveat on what `c_j` is:** it is the membership-weighted mean of the material
the solve SELECTED on the current SR field, so "outer `c_j`" means "the solve
targets outer material", not "a bound clump already exists there" — exactly the
hypothesis (right target, no object forms). The pooled `centre_host` in the dump
is a single reference for a ~20 Mpc/h tile of many hosts and gives a nonphysical
4.6 R_vir median; the per-subhalo HR parent is the faithful reference and is what
the numbers above use.

Figure: `viz_membership_solve.py --centre-radial` (needs `--rockstar-ascii`;
`--sr-subhalo-median` draws the gate's SR line). Re-plots from the npz on CPU:
`MV_NPZ_IN=<npz> MV_CENTRE_RADIAL_ONLY=1 MV_SR_SUBHALO_MEDIAN=0.35
bash scripts/slurm/submit_membership_viz.sh`.
