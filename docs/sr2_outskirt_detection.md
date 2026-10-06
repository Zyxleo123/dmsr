# The outskirt candidates: why Rockstar reports nothing there

*Status: MEASURED 2026-09-19 on held-out set9, from artefacts already on disk.
No new training. Reproduce with `bash scripts/slurm/submit_outskirt_probe.sh`.*

## The question

`membership_viz` measured the optimised candidate centres at **median 1.419
R_vir** against HR's own **1.191** (job 57188, signed against each set's own
parent). The `c_j` are at the right radii. And `radial-placement-is-inverted`
measured every tuned arm's **detected** catalog at 0.30-0.35 R_vir against HR's
1.13. So the obvious reading is that outskirt clumps are built and the halo
finder rejects them as unbound.

**That reading is wrong, and the correction changes what to fix.**

## 1. The premise holds, and it is not a mass effect

Supervised targets recovered as bound halos, held-out set9, signed against each
set's own parent (`all_blocks_sm_coarse_soft`, the arm that wins the held-out
gate):

| r/R_vir | n | recovered | median num_p | 2T/\|W\| | bound_hard | live_frac |
|---|--:|--:|--:|--:|--:|--:|
| 0.00-0.25 | 44 | **40.9%** | 384 | 33.3 | 0.124 | 0.928 |
| 0.25-0.50 | 127 | 18.9% | 387 | 44.7 | 0.081 | 0.958 |
| 0.50-0.75 | 168 | 13.1% | 434 | 40.0 | 0.095 | 0.981 |
| 0.75-1.00 | 170 | 4.7% | 391 | 42.5 | 0.072 | 0.988 |
| 1.00-1.50 | 310 | **2.6%** | 372 | 52.8 | 0.060 | 0.992 |
| >1.50 | 308 | 4.9% | 433 | 25.8 | 0.164 | 0.964 |

A 16x gap from the innermost bin to 1.0-1.5 R_vir. The two obvious confounds
are both ruled out:

* **not mass** -- median `num_p` is flat across the bins (372-434), and the gap
  survives inside each mass bin: 18.0% inner vs 3.4% outer at 200-300p, 17.9 vs
  3.3 at 300-500p, 12.8 vs 1.8 at 500-1000p, 28.8 vs 8.1 above 1000p.
* **not out-of-tile stragglers** -- the live fraction is *higher* in the
  outskirts (0.99) than in the core (0.93).

## 2. It is not an unbinding failure. Nothing is built there.

The misses have no halo near them at all:

| r/R_vir | median distance to the nearest halo of ANY mass | % with anything inside the 0.15 Mpc/h search sphere |
|---|--:|--:|
| 0.00-0.25 | 0.205 Mpc/h | 38.5% |
| 0.75-1.00 | 0.409 Mpc/h | 3.1% |
| 1.00-1.50 | 0.487 Mpc/h | 3.0% |

Rockstar is not declining a marginal object. At 1 R_vir there is nothing within
half a Mpc/h of the target to decline.

## 3. The decomposition: placement, not construction

Recovery splits as `P(addressed) x P(hit | addressed)`, where *addressed* means
the gathered members' centroid landed within a tolerance of the HR subhalo.
At 0.30 Mpc/h (the gate's own radius is 0.15, which leaves 1-6 sets per bin):

| r/R_vir | P(addressed) | hit % given addressed | hit % given NOT |
|---|--:|--:|--:|
| 0.00-0.25 | **36.4%** | 68.8% | 25.0% |
| 0.25-0.50 | 16.5% | 28.6% | 17.0% |
| 0.50-0.75 | 7.7% | 30.8% | 11.6% |
| 0.75-1.00 | 2.9% | 20.0% | 4.2% |
| 1.00-1.50 | **3.9%** | 25.0% | 1.7% |
| ALL | 8.5% | 34.4% (n=96) | 6.0% |

**`P(addressed)` falls ~10x; `P(hit | addressed)` falls ~2.5x.** The median
offset between the gathered centroid and the HR subhalo goes **0.46 Mpc/h in
the core to 1.17 Mpc/h at 1.0-1.5 R_vir** -- against a search radius of 0.15.

And the fine-tune never closes it *by design*: `centre_mode=self` charges the
offset from the set's own frozen centroid, not from HR's address. Held out,
`centre_offset_radii` is **5.90 at step 0 and 5.88 at step 4000** -- unmoved.
`centre-offset-is-an-address` already measured why that choice was made: the
offset is only 13.4% predictable out of sample.

So for every `self` / `soft` / `knn` arm, the outskirt sets are **scored at an
address where the run never claimed to put anything**.

## 4. What the objects actually look like

Pooled hit vs miss, same arm:

| | n | r/R_vir | \|offset\| | r_rms / R_vir,HR | bound_hard | 2T/\|W\| | sigma_v |
|---|--:|--:|--:|--:|--:|--:|--:|
| recovered | 95 | 0.58 | **0.42** | **3.90** | 0.139 | 32.3 | 348 |
| lost | 1032 | 1.12 | **0.97** | **6.46** | 0.083 | 37.3 | 272 |

The discriminators are **placement and compactness**. `bound_hard` moves 0.139
vs 0.083 and `2T/|W|` 32 vs 37 -- close to nothing, and `sigma_v` moves the
*wrong* way. The term the loss charges for, and the one the gate is named
after, is not what separates a found object from a lost one at this operating
point.

## 5. The control that proves it: relocation

`all_blocks_reloc_B` re-chooses each group's members from material that is
actually there and is scored where it asked for the object. Its addresses are
therefore right by construction (`P(addressed) = 99.3%` at 0.15 Mpc/h), and the
radial gradient nearly disappears:

| r/R_vir | reloc_B recovered | soft-arm recovered |
|---|--:|--:|
| 0.00-0.25 | 35.7% | 40.9% |
| 0.25-0.50 | 35.7% | 18.9% |
| 0.50-0.75 | 26.8% | 13.1% |
| 0.75-1.00 | 23.8% | 4.7% |
| 1.00-1.50 | **17.7%** | **2.6%** |
| ALL | **23.2%** (261/1127) | 8.4% (95/1127) |

**2x across the radial range instead of 16x**, and 2.7x more supervised targets
recovered overall. Its frozen control at the same addresses is 148/1127
(13.1%), so the fine-tune contributes a 1.76x gain on top -- honest, and
smaller than the 3.3x the `soft` arm shows from a 2.6% base.

Read against the abundance number, though: reloc_B holds **0.46** of HR's
subhalo count inside the host's R_vir against the soft arm's **1.02**. The
count-matching arms are not recovering HR's subhalos; they are making a
comparable number of *different* objects, piled in the core. That dissociation
is the same fact as `radial-placement-is-inverted`, seen per target.

Once placement is removed as a variable, boundness becomes the discriminator it
was always assumed to be -- reloc_B, hit vs miss: `r_rms/R_vir,HR` **0.96 vs
2.46**, `2T/|W|` **13.0 vs 43.7**, `bound_hard` **0.282 vs 0.055**.

## 6. Why nothing is bound anywhere: the reference is inflated

`build_member_sets` measures its reference on **HR inside the trained tiles,
frozen outside**, because members whose Lagrangian site is outside those tiles
cannot be moved by the run or by the splice. It records HR's own value beside
it, and the two disagree (16 holdout hosts, median live fraction **0.979**):

| median over hosts | pure HR | reachable reference | ref / HR |
|---|--:|--:|--:|
| `r_rms` [Mpc/h] | 0.127 | 0.277 | **2.18** |
| `sigma_v` [km/s] | 130.6 | 198.3 | 1.52 |
| `2T/\|W\|` | 2.14 | 4.26 | **1.99** |
| `bound_hard` | 0.765 | 0.673 | 0.88 |

`corr(1 - live_frac, r_rms_ref / r_rms_HR) = 0.76` held out, **0.84** on the 40
training hosts. The inflation *is* the straggler tail: `r_rms` is an **rms**, so
~2% of members sitting ~2 Mpc/h away contribute ~78% of `r_rms^2` while 98% of
the object is where HR put it.

Two consequences, and the second is the worse one:

1. the loss's target is **2.2x too extended and 2x too hot** before any
   optimisation happens;
2. the *gradient* on the live, movable members is diluted by the same factor --
   compressing them changes `r_rms` by a quarter of what it should.

Against that already-loose target, the arms reach `r_rms` ~2x the reference and
`2T/|W| ~ 37`. Measured against HR's actual object that is **4.7x too extended
and 17x too hot**, at `bound_hard` 0.087 against HR's 0.765.

Note also that the docstring's justification for including the stragglers --
"that is what the halo finder will actually see" -- does not hold. Rockstar
does not see the HR owner array; it builds its own groups from the field, and a
member 2 Mpc/h away joins whatever is near *it*. Including it charges the
generator for failing to bind a particle that could never have been in the
object.

## 7. What follows

Ranked by measured effect size, not by novelty:

1. **Stop scoring self-anchored arms at HR's address** (or stop supervising it).
   Worth 8.4% -> 23.2% on supervised recovery and 16x -> 2x on the radial
   gradient, already measured by reloc_B. The abundance cost (1.02 -> 0.46 of
   HR inside R_vir) is the open question this raises, not a reason to dismiss
   it.
2. **De-bias the size and energy reference.** BUILT, see section 8.
3. **Then, and only then, spend budget on `bound`.** It holds 0.24% of the loss
   budget because `[1 - x/ref]_+^2` is capped at 1 (`loss-budget-is-hinge-
   geometry`); `--bound-penalty log` uncaps it. Doing this *before* (2) would
   be spending the budget on chasing a reference that is itself 2x wrong.

### A gate bug fixed here

`compare_gather_catalog`'s threshold sweep called `target_hit_rate` without
`target_pos`, so for a relocated arm the four-rung ladder searched HR's centre
while the headline row searched the relocated one. reloc_B read 261/1127 in the
headline and 66/1127 in the ladder's first rung, which are supposed to be the
same measurement. Fixed; any relocation-arm sweep already on disk is stale.

## Reproducing

```
bash scripts/slurm/submit_outskirt_probe.sh          # all gated arms, set9
OD_BOX=set10 bash scripts/slurm/submit_outskirt_probe.sh
ARMS="all_blocks_sm_coarse_soft all_blocks_reloc_B" \
    bash scripts/slurm/submit_outskirt_probe.sh
PLOT_ONLY=1 bash scripts/slurm/submit_outskirt_probe.sh   # re-render only
```

Two CPU stages. The probe reads the HR catalog, each arm's `subhalos.json`,
its `compare/*.json` and its `pool.json`, and writes
`$REWARD_ROOT/outskirt_probe/<box>/<arm>.json`; the plot stage redraws from
that JSON alone, so the figures never need the numbers recomputed.


## 8. The fix built for (7.2): `--stats-members live`

*Built 2026-09-19, default off, NOT yet run. Every arm on disk is
`stats_members=all` and stays bit-identical.*

### What it does

One flag, applied at a single seam in `build_member_sets` **before anything
reads the stragglers**, so the reference, the candidate, `centre_target`,
`centre_self` and the background radius all change together:

```
keep_stragglers = cfg.stats_members == "all"
fps, fvs = (fp, fv) if keep_stragglers else (None, None)
```

With `live`, the out-of-tile members are out of every set statistic. The
reference becomes HR's own object -- to within the Lagrangian purity of the
tiling, and `subhalos-are-lagrangian-pure` measured a median of 1 tile of
origin -- which is exactly the quantity `pool.json` has been reporting as
`pure_hr_median` all along. So the new target is already measured: **`r_rms`
0.127 instead of 0.277, `2T/|W|` 2.14 instead of 4.26** on the 16 holdout
hosts.

### Why an identity cut and not a core radius

An adaptive "rms over the inner 68%" would be gameable: the generator could
compact 68% of a set and scatter the rest, and the statistic would improve
while the object did not. The particles dropped here are fixed at pool-build
time by the tiling, so there is no subset for the run to choose.

### Why this is not simply "charging for less"

The original justification for including them (`_gather_one`) was that they are
"what the halo finder will actually see". They are not. Rockstar builds its own
groups from the field; a member 2 Mpc/h away joins whatever is near *it*, and
never this object. Including it charges the generator for failing to bind a
particle it cannot move and that no catalog would have put here.

### The diagnostic survives the fix

`hybrid_reference_median` is now written under **both** policies, so
`pure_hr_median` against it is the inflation whichever way the run is
configured. A run that could not see its own reference move is how this went
unnoticed for 56 arms; `reference_inflation` in the probe reads the hybrid
explicitly and refuses to report ratio 1.00 for an arm that simply avoided the
problem. `outside_excluded_by_policy` counts what the policy dropped, kept
distinct from `outside_dropped` (no frozen box was passed, which silently
*overstates* boundness and is a bug, not a choice).

### The invariant the tests pin

`--centre-mode self` must still cost exactly zero at step 0 -- it anchors each
set where the frozen field already put it, which is only true if `centre_self`
and what `_gather_one` builds see the same collection. A policy applied to one
and not the other would break that silently, so it is pinned under both
policies, along with: sets with no stragglers are untouched; `live` makes
`reference_median == pure_hr_median` exactly; and the `rrms` term at step 0 is
**larger** under `live`, which is the whole point -- the candidate is now being
compared to HR's real compactness instead of to a target the stragglers had
already loosened.

### Running it

The A/B control already exists: `all_blocks_sm_coarse_soft` is this same
configuration at `stats_members=all`.

```
env MG_SOFT_MEMBERSHIP=1 MG_W_D6=0.0 MG_CENTRE_MODE=self MG_W_HOST_SETS=1.0 \
    MG_STATS_MEMBERS=live MG_LABEL=_livestats \
    bash scripts/slurm/submit_member_gather_train.sh
```

Those four baseline settings are not optional: `mgtrain-defaults-are-not-the-
winning-arm` records job 56450 running d6 at 37% of its loss budget because
they were left at their defaults, which made the result incomparable with the
line it was meant to extend.

**Read the shakeout before letting it run 12 hours.** The job prints the policy
and, in `pool.json`, `reference_median` must now equal `pure_hr_median` and
differ from `hybrid_reference_median` by ~2x in `r_rms`. If they are all equal,
the pool has no stragglers and the arm is a no-op.

**Note on the 12 h limit.** `all_blocks_hostlr` and `all_blocks_hostself` were
killed at the wall clock at steps 3250 and 2500 of 8000, and the trainer has no
`--resume`, so their checkpoints cannot be continued. Expect the same here at
`MG_STEPS=8000` with soft membership on, and gate early with
`MG_GATE_AT_STEPS`.
