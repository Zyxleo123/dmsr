# Aiming the membership solve at HR's radii: the radial prior

*Status: built, pre-tested and GATED 2026-10-12 to 10-15. **The trained arm
FAILED the held-out set9 Rockstar gate**: the solve aims at HR's radii, but the
generator does not build there on a box it never saw, and abundance fell from
0.82x to 0.52x HR. Treat this lever as closed, like the shell family
(`sr2_host_profile.md`, `shell-levers-built`). Five bugs found on the way are
fixed and stay useful; see section 5.*

## The question

Frozen SR2's host mass profile is nearly right -- about 10% too concentrated,
with 89% of HR's particles at 1.0-1.5 R_vir (`sr2_host_profile.md`, supply
gate). Yet every tuned member-gather arm piles its subhalos into the core:
HR's median subhalo sits at 1.13 R_vir, `coarse_soft`'s at 0.35, with 40% inside
0.25 R_vir against HR's 2.5%.

The soft-membership solve decides WHICH particles each supervised object is
made of; the loss then makes that material compact, cold and bound. So the
question was: if the solve is pointed at material at HR's radius, does the
generator learn to build there?

## 1. Where the solve aims today

Measured on a membership dump of `coarse_soft`'s own tuned held-out set9 field
(host h165513 region: 159 supervised sets belonging to **46 different
parents**), every radius against the set's OWN HR parent
(`scripts/features/solve_centre_condensation.py`, job 80377):

* **The solved centres `c_j` sit at HR's radius in the median**: r_fin/r_HR
  0.96-1.25 for 0.25-2 R_vir; median 1.42 vs HR 1.19.
* **EM did not put them there.** Median radius 1.43 at the first sweep and 1.42
  at the last: the starting centroid of HR's members in the tuned field is
  already there. EM changes which particles, not where.
* **But the per-band counts are wrong.** Only 39% of sets land within 1.5x of
  their HR radius, and the scatter smears the profile:

| band (R_vir) | 0-.25 | .25-.5 | .5-.75 | .75-1 | **1-1.5** | 1.5-2 | 2-3 | >3 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| HR subhalos | 7 | 14 | 20 | 25 | **37** | 32 | 16 | 8 |
| solved centres | 13 | 24 | 13 | 16 | **17** | 19 | 27 | 30 |
| centres with an arm halo nearby | 13 | 16 | 7 | 6 | 9 | 11 | 13 | 9 |

"Halo nearby" = an arm Rockstar halo of at least 50 particles within
max(R_vir of the HR subhalo, 0.15 Mpc/h) of `c_j`. It does not check identity
or uniqueness, and no chance-coincidence control was run.

## 2. The two levers and their pre-test

The pre-test is a membership dump on the SAME saved field and host plus the
per-band count above: one short GPU job and one CPU job, no training
(`MV_*` options in `scripts/slurm/submit_membership_viz.sh`, `MV_CONDENSE=1`).

**Lever 1 -- host profile in the background, centred on each set's own parent
(`MG_SOFT_HOST_PROFILE=parent`).** The existing `lr`/`self` profiles were
centred on the TASK host for every set, which owns only 51 of these 159 sets
(section 5). The `parent` profile drains the core but only repels; it has no
idea where HR's peak is, so material overshoots past 2 R_vir:

| beta | 0-.25 | .25-.5 | .5-.75 | .75-1 | 1-1.5 | 1.5-2 | 2-3 | >3 | within 1.5x |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 (control) | 13 | 24 | 13 | 16 | 17 | 19 | 27 | 30 | 39% |
| 0.5 | 7 | 19 | 16 | 9 | 20 | 15 | 32 | 41 | 36% |
| 1 | 4 | 13 | 16 | 8 | 19 | 19 | 29 | 51 | 30% |
| 2 | 3 | 4 | 14 | 9 | 14 | 15 | 31 | 69 | 27% |

**Lever 2 -- a per-set radial prior (`MG_SOFT_RADIAL_WEIGHT` = gamma,
`MG_SOFT_RADIAL_WIDTH` = w).** Each pool particle pays
`0.5 * gamma * ((r - r_HR) / (w * R_vir,parent))^2`, r measured from the set's
own HR parent, direction free. Only gamma/w^2 matters ((1, 0.2) and (4, 0.4)
gave bit-identical dumps). Combined with lever 1 at beta 0.5:

| setting | 0-.25 | .25-.5 | .5-.75 | .75-1 | **1-1.5** | 1.5-2 | 2-3 | >3 | within 1.5x | median mass |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| HR | 7 | 14 | 20 | 25 | **37** | 32 | 16 | 8 | -- | -- |
| control | 13 | 24 | 13 | 16 | 17 | 19 | 27 | 30 | 39% | 398 |
| radial gamma/w^2 = 25 | 15 | 26 | 19 | 17 | 34 | 28 | 10 | 10 | 58% | 361 |
| **radial 25 + parent 0.5** | **10** | 25 | 22 | 18 | **35** | 28 | 11 | 10 | **61%** | 361 |
| radial 100 + parent 0.5 | 10 | 27 | 26 | 16 | 39 | 20 | 11 | 10 | 64% | 351 |

The combination fills HR's peak, removes the outer overshoot and halves the
core excess. It does NOT touch the 0.25-0.5 excess, and it costs ~9% of solved
mass. **It also starves a tail: 5% of sets get under half their target mass,
the worst 0.02x** (the control has none below 0.998x). That tail is what later
produced the NaN gradients.

**What the pre-test cannot say:** it measures the solve's targets on one field
the generator was trained on without the levers. Whether the generator learns
to build at those radii on unseen boxes is a training run plus the gate.

## 3. The trained arm

`all_blocks_radparent`: `coarse_soft`'s recorded config (diffed field by field
against its `summary.json`; the only real difference was
`max_set_particles=768`, which the env default would have dropped) plus both
levers and the centre anchor of section 5.4:

```
env MG_SOFT_MEMBERSHIP=1 MG_SOFT_PRIOR=coarse MG_SOFT_LAMBDA=1.0 \
    MG_W_D6=0.0 MG_CENTRE_MODE=self_tan_parent MG_W_HOST_SETS=1.0 MG_STEPS=4000 \
    MG_MAX_SET_PARTICLES=768 \
    MG_SOFT_HOST_PROFILE=parent MG_SOFT_HOST_WEIGHT=0.5 \
    MG_SOFT_RADIAL_WEIGHT=1.0 MG_SOFT_RADIAL_WIDTH=0.2 \
    MG_EXCLUDE=gpu28 SKIP_SHAKEOUT=1 \
    MG_LABEL=_radparent bash scripts/slurm/submit_member_gather_train.sh
```

Job 84387 on an A6000, 4000 steps in 3.2 h. Gate chained with
`HG_AFTER_JOB`, `NO_CONTROL=1`, baseline `all_blocks_sm_coarse_soft`.

**Held-out learned in step with train** (step 0 -> 250: virial ratio 194 -> 49,
bound fraction 0.004 -> 0.039, size/HR 1.89 -> 1.52), then plateaued: held-out
bound ~0.055 from step 1000 on, against ~0.09 for `coarse_soft`. The
train/held-out gap on bound was wider (0.080 vs 0.056 at step 4000). 20 of 4000
updates were skipped as non-finite and one step's loss spiked to 1096 (step
3250); this run used the first virial fix, not the mass-free one.

## 4. Gate result (held-out set9, edited region)

`member_gather/all_blocks_radparent/holdout_set9/verdict_vs_coarse_soft.json`
and `flow_rockstar/compare/set9__mgho_all_blocks_radparent_set9.json`.
Verdict printed by the gate: `PARTIAL: abundance LOST, hosts NOT recovered`.

**Placement did not move:**

| | median r/R_vir | inside 0.25 R_vir | count at 1-1.5 R_vir |
|---|--:|--:|--:|
| HR | 1.13 | 2.5% | 6,545 |
| frozen SR2 | 0.53 | 29% | 526 |
| `coarse_soft` | 0.35 | 40% | 1,129 |
| **`radparent`** | **0.33** | **43%** | **710** |

**Abundance fell, and hosts fell below frozen:**

| | subhalos >=50p | hosts >=200p | subhalos in host 168880 |
|---|--:|--:|--:|
| HR | 12,367 | 3,775 | 374 |
| frozen SR2 | 4,541 | 3,028 | 20 |
| `coarse_soft` | 10,138 | 3,148 | 382 |
| **`radparent`** | **6,440** | **2,759** | **258** |

It built less in every band, including the 1-1.5 R_vir band it was meant to
fill. SUBFIND was not run: Rockstar already shows a clear loss.

## 5. Bugs found on the way (all fixed, all tested)

1. **Host profiles were centred on the task host.** `lr`/`self` measured radius
   from `centre_host` for every set, but sets are selected by home tile and a
   12.5 Mpc/h tile holds many hosts (51 of 159 sets here belong to the task
   host). Any trained `lr`/`self` arm would have shaped the background around
   the wrong host for ~2/3 of its sets. New: `MemberSets.centre_parent`,
   `parent_rvir`, `parent_mvir`, filled by `build_member_sets` from the HR
   catalog, and `soft_host_profile="parent"`.
2. **The host-guard config refused the arm at the first eval** (job 84183).
   `--w-host-sets` derives a `soft_membership=False` config; it reset the host
   profile but not the new radial weight, so `__post_init__` refused it -- the
   same failure that killed 56560/56562. `member_gather.host_guard_config()` is
   now the ONE place that resets solve-only options, and the test calls it
   instead of copying it. **A new solve-only option belongs in that list.**
3. **NaN gradients from starved sets** (job 84238 NaN by step 250; found with
   the new `MG_DETECT_ANOMALY=1`, job 84372, in `_SpecificPotentialBackward`).
   A starved set's |W| is tiny (quadratic in the weights); autograd's quotient
   rule for `2T/|W|` formed `T/W^2`, which overflows float32, and the
   potential's backward multiplied inf by its zero self-pair. The weighted
   virial is now `exp(log(4 sum w ke) - log|sum w phi|)`, capped at e^60: log
   space avoids the W^2, and leaving the particle mass out avoids a 5.8e8
   factor after the division. Tested finite down to weights of 1e-18.
4. **The centre anchor fought the radial prior** (job 84296). With
   `centre_mode=self` the centre term was 30.9 at step 0 against
   `coarse_soft`'s 4.96 (~80% of the loss): `self` anchors each object to where
   HR's members sit in the frozen field, which is the radius the prior just
   moved it from. The solve's move is 90% radial about each set's own parent
   (median 0.41 Mpc/h), so the fix is `centre_mode=self_tan_parent`: `self_tan`
   with the radial direction from the set's own parent (the old `self_tan` used
   the task host -- bug 1 again). The term fell to 15; the rest is the solve's
   sideways move (median 0.19 vs 0.09 Mpc/h), which does not change radius.
5. **The radial cost promoted the solve to float64** (the unwrapped positions
   inherit the float64 set centres), crashing the dual's scatter on a float32
   field. Cast to the solve's dtype.

Also added: the trainer skips and logs any update with a non-finite loss or
gradient (host, per-term losses, solve diagnostics, bad parameters) and stops
after `--max-nonfinite-steps` (25) in a row; eval rows carry
`n_nonfinite_steps`. `MG_EXCLUDE` pins the GPU class in
`submit_member_gather_train.sh`.

**Still broken: early gates.** `MG_GATE_AT_STEPS` writes `tuned_step<N>.pt`,
but the gate's export reads `summary.json`, which the trainer writes only at
the end (job 84389 failed on it). No early gate can run until the trainer
writes a provisional `summary.json` at the snapshot, or the export stops
needing it.

## 6. Reading

Same failure shape as the shell family: an objective that is right on the
training boxes (the solve does aim at HR's radii) produces a generator that
does not build there out of sample. The levers only change which material the
loss targets; the generator's input (`model(lr, noise)`) still carries no host
scale or host position, and HR's radii scale 6.5x with host mass while the
arms' do not (`radial-profile-is-flat-in-host-mass`). Retargeting the material
cannot supply information the operator does not see.

Open, cheap, not run: the same band-count pre-test on THIS arm's own held-out
field and catalog (the field is spliced and on disk:
`flow_rockstar/fields/set9__mgho_all_blocks_radparent_set9__seed0.npy`). It
would separate "aimed right, did not build" from "the aim drifted on unseen
data".

The untried route past the blocker is giving the generator host information:
the LR-host conditioning channel (`lr-host-conditioning-channel-built`) has
only ever run on the `fine` rung, never `all_blocks`.

## Files

| file | what |
|---|---|
| `src/cosmo_sr/features/soft_membership.py` | radial prior in `solve_memberships`; `parent` centring; mass-free log-space weighted virial |
| `src/cosmo_sr/features/host_profile.py` | `parent` source (NFW about each set's own HR parent) |
| `src/cosmo_sr/features/member_gather.py` | `centre_parent`/`parent_rvir`/`parent_mvir`; `host_guard_config()`; `centre_mode="self_tan_parent"`; config fields + refusals |
| `scripts/features/finetune_member_gather.py` | CLI flags; non-finite guard; `--detect-anomaly` |
| `scripts/features/solve_centre_condensation.py` | the per-band pre-test (solved centres vs HR vs arm halos) |
| `scripts/slurm/solve_centre_condensation_cpu.sbatch` | its CPU job |
| `scripts/slurm/submit_membership_viz.sh` | `MV_HOST_PROFILE`, `MV_RADIAL_*`, `MV_CONDENSE`, `MV_AFTER` |
| `scripts/slurm/member_gather_train_gpu.sbatch` | `MG_SOFT_RADIAL_*`, `MG_DETECT_ANOMALY` |
| `scripts/slurm/submit_member_gather_train.sh` | env pass-through; `MG_EXCLUDE` |
| `scripts/slurm/pytest_cpu.sbatch` | run a pytest selection on a CPU node |
| `tests/features/test_host_profile.py` | parent centring, radial prior (zero, pull, starvation, refusals), finite gradients for near-massless sets, `self_tan_parent`, the guard config |

Results: pre-test dumps and band counts in
`member_gather/membership_viz/` and `.../condensation/`; the arm in
`member_gather/all_blocks_radparent/`.
