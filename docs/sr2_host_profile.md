# The host as a background component

*Status: built 2026-09-17. **The gate PASSED on both questions** (jobs 56364 /
56365, set9 + set10). The two training arms are cleared to launch and have not
been run.*

> **Update 2026-10-15** -- see `docs/sr2_radial_prior.md`. The `lr`/`self`
> profiles below measure radius from the TASK host for every set, which owns
> only ~1/3 of a pool's sets; `soft_host_profile="parent"` (each set's own HR
> parent) fixes that. Pre-tested, `parent` drains the core but overshoots past
> 2 R_vir; trained together with a per-set radial prior it FAILED the held-out
> gate (radial 0.33 R_vir, abundance 0.52x HR).

## Gate result (2026-09-17)

**1. Supply -- PASS.** Frozen/HR median particle count in the decisive
1.0-1.5 R_vir shell: **0.889** (set9), **0.875** (set10), against a 0.8 floor.
The material is there; the selection is what is wrong. The full shell table
also shows the defect at FIELD level, not just in the catalog: frozen runs
*over* HR at 0.25-0.75 R_vir (1.05-1.15) and *under* it at 1.0-2.0 (0.87-0.92).
A ~10% redistribution inward, not a shortage.

**2. Drift -- PASS (INWARD).** The M step does pull candidates toward the host.
Sign is consistent -- 60-61% of sets move inward on both recentring sweeps --
and the magnitude scales with host mass, as `sr2-deficit-scales-with-host-mass`
would predict:

| grouping | median r, init -> final | net |
|---|---|---|
| h165513's own 51 subhalos (R_vir 1.33) | 0.834 -> 0.624 R_vir | **-0.21 R_vir** |
| all 159 sets, each signed to its own parent (46 parents) | 1.428 -> 1.419 R_vir | -0.0096 R_vir |

**The attribution matters and nearly inverted the verdict.** A task is named for
one host, but its sets are selected by HOME TILE, and a trained tile is 12.5
Mpc/h a side: of the 159 supervised subhalos in the set9 dump, only **51**
belong to the task's host and 108 belong to 45 other parents sharing those
tiles. Signing the drift against `centre_host` for all of them read **+0.08
R_vir (STABLE)** -- the opposite conclusion. `parent_geometry` in the probe now
resolves each set to its own parent, and `signed_against` records which reading
a result is. Same class of error as `host-row-overlay-collision`.

## The defect

`radial-placement-is-inverted`, measured: HR's median subhalo sits at
**1.13 R_vir** from its host. Every tuned arm this line has produced sits at
**0.30-0.35**. The objects exist, they are bound, they have HR's mass function
(`gather-holdout-rockstar-gate`, `knn6d-arms-gate-pass`) -- and they are in the
wrong place, piled toward the centre.

Everything tried against this so far has been a form of *matching HR's radial
distribution*: `w_radius` (host-centric radius, direction free), `w_shell`
(the soft shell distribution term), rank and quantile matching. All of them
failed, and `shell-levers-built` says how: the per-host radial distribution is
**real** (rho 0.9 against HR) but **not reproducible by a shared local
operator**, so every lever either overfits the training box or degrades
abundance monotonically. `radial-profile-is-flat-in-host-mass` explains why
that had to happen -- HR's profile scales 6.5x with host mass, the candidates
are flat at ~0.1 Mpc/h, and a translation-equivariant operator fed a 14^3 LR
crop emits the conditional mean, which IS a flat profile.

## Why the loss never fixes it

Look at what the member-gather loss actually charges for
(`member_gather.set_statistics_torch`):

| term | what it measures | depends on WHERE the object is? |
|---|---|---|
| `virial` | `2T/|W|` over members | no -- within-set differences |
| `bound` | fraction with `E < 0` | no |
| `rrms` | rms radius about the centroid | no |
| `sigmav` | velocity dispersion | no |
| `d6` | size against the local background | no |
| `isolation` | non-member mass on top of the object | **local density only** |
| `centre` | offset from `centre_target` | yes |

Seven of the eight are **exactly translation-invariant**: they cannot express
where a subhalo sits, only what it looks like once you are standing on it. The
eighth, `centre`, is the one positional term, and `centre-offset-is-an-address`
measured it explaining **13.4%** of the radial offset -- it is an address the
operator cannot predict, not a force.

So the radial distribution is not being optimised badly. It is **unconstrained**,
and it relaxes to whatever configuration is cheapest.

## What makes the core cheap

The membership solve (`soft_membership.solve_memberships`), not the loss.

```
logit_ij = log Pi_ij + alpha_j - d2_ij / (2 lambda)
P_ij     = softmax_j(logit) against a background term logbg_i
```

Two things in there both point inward:

1. **The prior is proportional to local density.** `pi_m = pi_n = max(n_c/f^3, eps)`
   (`soft_membership.py:236-240`) -- a candidate overlapping a dense coarse cell
   gets a higher log-prior on *every* particle in it. The host core is the
   densest cell in the region.
2. **The background is flat.** `logbg = log(1 - claimed)`
   (`soft_membership.py:398`) -- a residual with no position in it. Membership
   is decided by *absolute* phase-space proximity to `c_j`, with no reference to
   how much material would have been there anyway.

Together: a candidate sitting in the core meets its Sinkhorn mass target at tiny
`d^2` and near-zero dual. The same candidate at 1.1 R_vir must gather from a
thin field, so `alpha` is driven up and the window widens. The solve has a
systematic radial cost gradient pointing **inward**, and the gather loss then
trains the generator to build coherent bound objects wherever the solve found it
cheapest to name them. Nothing ever contradicts it.

## The change

Give the background the host's own density profile, so the comparison becomes a
log **overdensity**:

```
logit_ij - logbg_i = log Pi_ij + alpha_j - d2_ij/(2 lambda) - beta * log rho_host(r_i)
```

A lump in the core must now be denser than the core to win its particles; a lump
at 1.1 R_vir wins them cheaply. The radial preference changes sign **through
selection**, and no per-object radial target is stated anywhere -- which is the
point, since that family has already failed.

The solve stays `no_grad`, so the gradient path is unchanged ("move the
particles, not the labels"). What changes is *which particles carry the
supervision*: contrast selection denies the core lump the right to BE subhalo
`j`, and the pool's remaining mass at larger radius is what gets built into a
bound object instead.

### Why this class of term can work when matching could not

The one lever that has ever moved radial placement is `isolation`
(`radial-placement-is-inverted`: 0.718, beating frozen on both axes). And
`isolation` is `bg_occ` -- the soft count of non-member mass sitting on top of
the object (`member_gather.py:1025-1033`). That is a crude, one-radius density
contrast applied in the loss. This moves the same idea into the selection, where
it acts per particle and with a profile instead of a single sphere.

## Two sources, run as separate arms

| | `lr` | `self` |
|---|---|---|
| profile | analytic NFW from `(M_vir, R_vir)` | binned off the candidate field |
| needs | the LR Rockstar catalog | nothing |
| deployable | yes -- LR-only, exists at inference | yes |
| coverage | **57%** of region subhalos (`lr-host-channel-fidelity`) | all hosts |
| honest? | fully | **partly circular** |

`self`'s circularity is real: a core-heavy field reports a core-heavy profile and
so partly forgives itself. It cannot forgive itself completely -- the profile is
a smooth function of radius only, while a subhalo is a *localised* excess, so a
lump still has to beat the azimuthal average at its radius -- but the
circularity is why these are two arms and not one choice made in the code.

**`beta` is not comparable between them.** Measured on the toy: `self`'s
empirical profile spans ~34 nats peak-to-trough against NFW's ~10, so `self`
saturates at a `beta` where `lr` is still ramping. Read any dose ladder off
`soft_host_logrho_p10/p90`, which every solve reports, in nats.

## The gate, which comes first

Two questions can kill this before a GPU is spent
(`scripts/features/host_profile_probe.py`):

1. **Supply.** If the candidate field has no material at 0.8-1.5 R_vir, refusing
   the core does not relocate the object, it **starves** it: the mass constraint
   goes unmet, the dual runs away, and the loss is applied to a near-empty pool.
   That is worse than the defect. The probe measures the radial particle supply
   around each gated host, frozen against HR, in R_vir shells. The verdict is the
   frozen/HR median in the 1.0-1.5 shell: `SUPPLY OK` means the material is there
   and only the selection is wrong; `STARVED` means the arms should not launch.
2. **Drift.** Does the M step pull candidates inward? Read off the
   `--dump-membership-npz` snapshots, projected on the clustercentric direction.
   `INWARD` confirms the solve's fixed point is the core. `STABLE/OUTWARD` means
   the misplacement predates the loss and this is not the lever.

## Running it

```bash
# the gate (cpu, ~20 min; stage 2 is afterok on stage 1). Its --boxes must
# cover the TRAIN boxes, not just the holdout: `lr` needs a table per box it
# trains on, and job 56447 died on exactly that.
bash scripts/slurm/submit_host_profile.sh

# then, ONLY on SUPPLY OK -- siblings, they answer different questions.
# CARRY THE WINNING BASELINE (see below); three submitter defaults are wrong.
BASE="MG_SOFT_MEMBERSHIP=1 MG_W_D6=0.0 MG_CENTRE_MODE=self MG_W_HOST_SETS=1.0"
env $BASE MG_SOFT_HOST_PROFILE=lr   MG_SOFT_HOST_WEIGHT=1.0 \
  MG_LABEL=_hostlr   bash scripts/slurm/submit_member_gather_train.sh
env $BASE MG_SOFT_HOST_PROFILE=self MG_SOFT_HOST_WEIGHT=1.0 \
  MG_LABEL=_hostself bash scripts/slurm/submit_member_gather_train.sh
```

### The baseline this arm must be added to

`submit_member_gather_train.sh`'s defaults are NOT the configuration of
`all_blocks_sm_coarse_soft`, the soft arm that wins the held-out gate. Three
differ, and launching without them tests the host profile on top of a baseline
this line already rejected:

| variable | submitter default | coarse_soft | why it matters |
|---|---|---|---|
| `MG_W_D6` | 1.0 | **0.0** | `loss-term-ablation`: d6 is HARMFUL and not in the minimal set. At the default it took 37.1% of the step-0 budget in job 56450. |
| `MG_CENTRE_MODE` | full | **self** | `full` makes the raw centre term 127.7 (57.3% of the budget) against an anchored ~0 at step 0. |
| `MG_W_HOST_SETS` | 0.0 | **1.0** | the host-preservation guard that passed its own set9/set10 gate. |

Job 56450 (`_hostself`) was launched without these and is therefore a
feasibility check, not a comparable result: its step-0 `gather` is 216.2
against coarse_soft's 17.98, essentially all of it d6 + the unanchored centre
term.

`MG_SOFT_HOST_PROFILE=off` (the default) is byte-identical to every soft arm
already on disk -- pinned by
`tests/features/test_host_profile.py::test_beta_zero_reproduces_the_flat_background_exactly`,
which asserts `torch.equal` against the unmodified solve.

## What to read in the run

* `soft_host_valid_frac` -- how many hosts got a profile at all. For `lr` this is
  the LR completeness gate showing through; a run near 0 is the arm doing
  **nothing**, not the arm failing.
* `soft_host_logrho_p10` / `p90` -- the dose, in nats. The ladder is read here.
* `soft_mass_ratio` -- should stay ~1.0. Falling means the contrast is starving
  the candidates, which is the failure mode question 1 is about.
* Then the held-out Rockstar gate as usual, plus the radial placement number,
  which is the thing this arm exists to move.

## Files

| file | what |
|---|---|
| `src/cosmo_sr/features/host_profile.py` | the profile: NFW, empirical, interpolation, LR lookup |
| `src/cosmo_sr/features/soft_membership.py` | the background term in `_solve` |
| `scripts/features/build_lr_host_table.py` | gated LR host table per box |
| `scripts/features/host_profile_probe.py` | the supply + drift gate |
| `scripts/slurm/lr_host_table_cpu.sbatch` | stage 1 (cpu) |
| `scripts/slurm/host_profile_probe_cpu.sbatch` | stage 2 (cpu) |
| `scripts/slurm/submit_host_profile.sh` | the submitter |
| `tests/features/test_host_profile.py` | reproduction pin, profile shape, the outward claim |
| `tests/features/test_host_profile_probe.py` | shell counts, drift sign, the LR gate |
