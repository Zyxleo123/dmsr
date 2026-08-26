# Gathering SR2's particles into true HR subhalos: the loss moved its windows, not the objects

**Scope.** A results note. Four fine-tuning runs, four real Rockstar gates and one
halo-finder-free discriminator, all on set8 and all on one cluster (host 271800,
`log10 Mvir` 14.81).

It was written to record an objective that hit every target it was given and a
halo finder that disagreed completely. The controls of §7 have since run and the
conclusion is sharper and less flattering: the loss's statistics are evaluated in
a spatial **window**, the window is not the object, and on the particles HR
actually binds the fine-tuned field is **statistically indistinguishable from the
frozen generator** (§8). It did not build a clump that failed to bind; it never
moved the material at all. §5's gate was also uncalibrated when it was written --
§8.1 supplies the ceiling and the noise floor that make it readable.

Numbers are tagged *measured* (read from an artifact on disk), *derived*
(arithmetic on measured constants) or *design* (a proposed rule, not yet run).

Depends on `docs/pilot_steps_2_4.md` §2 for why MSE was abandoned,
`docs/sr2_subhalo_deficit.md` for the deficit, and
`docs/sr2_substructure_module.md` §2 item 3 for why velocity is not optional.

## 1. What this line is

`docs/pilot_steps_2_4.md` §2 closed the capacity question: at 4.4 trainable
parameters per target value the generator could have memorised one cluster and
did not -- MSE plateaued at 0.39x frozen while high-k power fell. **The generator
can represent this cluster's substructure; the squared loss declines to ask for
it.** A squared error over a realisation the network cannot predict is minimised
by averaging, and averaging is the deficit.

This line keeps the supervision -- the true HR subhalo positions, which exist on
disk for set8/set9 -- and stops asking *which particle goes where*:

```
for each true HR subhalo s
    C(s) = sum_cells  K(|x_cell - x_s|; sigma_s) * w_compact(cell) * mass(cell)

    L(s) = [1 - C_theta(s)/C_HR(s)]_+^2                    # is a clump there
         + w_contrast * [1 - P_theta(s)/P_HR(s)]_+^2       # does it beat its surroundings
         + w_vdisp * log(sigma_v(s)/sigma_v,HR(s))^2       # is it as hot as HR's
         + w_vbulk * |v(s) - v_HR(s)|^2 / sigma_v,HR(s)^2  # does it move with HR's
```

`w_compact` is exactly `reward/soft_structure.py`'s compact-mass coordinate -- a
sigmoid on `u = log(1 + delta)`, the form that module records as the one where
"the gradient is well conditioned across the whole range". Nothing new is
invented; it is evaluated in a Gaussian window at a known location instead of
over a whole tile.

Four properties were the argument for it, and the first three held:

1. **No per-particle identity.** `C` sums over *cells*, so it is invariant to
   which particle landed where and to sub-kernel translation of the whole clump.
   Blurring strictly lowers it -- averaging is the worst available move rather
   than the minimiser. *Measured*: spreading the same mass over 27 cells drops
   the statistic by more than 20x (`tests/features/test_subhalo_gather.py`).
2. **Hinged on the density side.** A candidate matching HR contributes exactly
   zero with zero gradient, so the loss cannot ask for more contrast than HR has
   -- the step-4 over-sharpening failure is unreachable.
3. **Per-subhalo normalisation.** Dividing by `C_HR(s)` gives a 200-particle and
   a 3000-particle satellite the same weight, which is
   `sr2_substructure_module.md` §4.2's contrast equalisation moved into the loss.
4. **~~Matching these statistics means building the object.~~** False, and worse
   than §5 first recorded: matching them did not even *move* the object. §8.2.

The velocity terms are **two-sided, not hinged**, and deliberately: too hot is
unbound and a phase-space finder discards it, too cold is the measured
sub-virial defect. There is no direction in which being wrong about velocity is
safe.

### 1.1 Where HR enters, and where it does not

The catalog, the owner array and the reference statistics `C_HR`, `sigma_v,HR`
are built **once, before the loop**, into a `GatherTargets` tensor. They are data
in the loss exactly as HR displacements are data in an MSE. The generator's
inputs are unchanged, so a checkpoint from this line runs at inference from
`(Y, z)` alone. The supervision is nonetheless in-sample by construction, and
nothing here tests generalisation.

### 1.2 No KDTree, and one would be slower

*Derived.* A neighbour query per subhalo per step is `O(S N log N)` over 262,144
particles a tile. The CIC deposit is one `O(N)` pass every particle contributes
to exactly once, after which each subhalo costs a `(2H+1)^3` window read. It is
also the estimator the rest of the repository scores with, so the loss and the
eval cannot disagree about where the mass went. `dC/d(disp_i)` is non-zero for
exactly those particles whose CIC stencil touches a kernel-weighted cell -- i.e.
the particles within the threshold radius of the true centre -- so the selection
a KDTree would perform happens implicitly and for free.

## 2. Module map

| file | role |
| --- | --- |
| `features/subhalo_gather.py` | targets, the windowed statistics, the loss |
| `features/finetune_host_gather.py` | the trainer; reuses `overfit_host_mse.py`'s host selection and field report unchanged |
| `features/render_gather_slices.py` | eval panels, redrawn from `eval/step*.npz` on CPU |
| `reward/splice_gather_field.py` | tuned tiles into the frozen box, for the halo finder |
| `reward/compare_gather_catalog.py` | the Rockstar comparison and its four readings |
| `features/bound_discriminator.py` | §8.2: per-member-set statistics, no halo finder |
| `features/measure_bound_discriminator.py` | its I/O, discrimination table and verdict |
| `slurm/submit_host_gather.sh` | shakeout -> train -> redraw |
| `slurm/submit_gather_rockstar.sh` | splice -> Rockstar -> compare |
| `slurm/submit_bound_discriminator.sh` | the §8.2 measurement, one CPU job |

One CIC pass (`deposit_phase_space`) yields mass, mean velocity and dispersion on
the same cells, so "HR's mass but not HR's kinematics" can never be a statement
about two different regions. All three fields -- candidate, frozen, HR -- are
deposited on the **frozen generator's** rounded bulk offset, or the window that
is a subhalo in one grid would be its neighbour in another.

## 3. What ran

*Measured.* All on set8, host 271800, rung `fine` (335,954 trainable parameters
of 6,975,826), batch 2, `lr_scale` 10, seed 0, on one A5000.

| run | targets | velocity terms | `w_low` / `w_anchor` | steps | wall |
| --- | ---: | --- | --- | ---: | ---: |
| A | 146 (`min_num_p` 50) | none | 1 / 0.1 | 3000 | ~7 min |
| B | 43 (`min_num_p` 200) | on | 50 / 0.1 | 3000 | 482 s |
| C | 43 | on | 100 / **10** | 5000 | 826 s |

### 3.1 Run A: the statistic was measuring the host, not the subhalo

*Measured.* Compact-mass ratio (mean) went 1.144 -> 3.088, median 0.64 -> 1.48,
p90 5.7, with 7% of targets above 2x HR. The mean above 1 for the *frozen*
generator was the first clue: HR's own compact mass per subhalo particle is
**2.75** at 50-100p, falling to 0.52 at 500-5000p. At small masses the window
holds mostly the *host's* material, so "satisfied" partly meant "there is enough
cluster here", which SR2 already had.

The clean part of that run is the frozen deficit where the statistic *is* about
the object -- and it is strongly mass-dependent:

| M_sub | 50-100p | 100-200p | 200-500p | 500-5000p |
| --- | ---: | ---: | ---: | ---: |
| frozen `C/C_HR` (median) | 0.79 | 0.60 | 0.44 | **0.13** |

Costs: `low_k` 0.0308 against a 0.02 gate, high-k displacement power 0.352 ->
0.304 (away from HR), whole-tile `peak_contrast` 0.851 -> 0.606. Weighted
gradient norms were gather 15.6, low 0.13, anchor 0.03 -- the guards were never
in the fight.

### 3.2 Run B: both halves matched

*Measured*, 43 targets, HR compact per particle now 0.62 (the object dominates
its window).

| median over 43 targets | frozen | tuned |
| --- | ---: | ---: |
| `C/C_HR` | 0.330 | **1.058** (q25 0.99, q75 1.23) |
| `sigma_v / sigma_v,HR` | 1.40 (q75 2.47) | **1.02** (q25 0.99, q75 1.05) |
| bulk offset | 1.03 sigma | **0.05 sigma** |

Overshoot collapsed (targets above 2x HR: 7% -> 2%), `low_k` fell to 0.0225 and
high-k power recovered to 0.359 against frozen's 0.352 -- no collateral damage at
the LR scale at all.

**Read the two velocity rows as window statistics, not as the objects'**
(§8.2). `sigma_v / sigma_v,HR` = 1.02 here is the dispersion of whatever material
sits inside a Gaussian window at the target's location. Measured instead over the
**particles HR binds to that subhalo**, the same field is at **2.9x** HR -- and so
is the frozen generator, to within 1.6%. Raising `min_num_p` to 200 made the
window's *density* about the object (§3.1) and did not do the same for its
kinematics. The row is not wrong; it is not about the subhalo.

**The velocity finding is worth keeping.** At the sites where HR has a subhalo,
frozen SR2's material is 40-150% too *hot* and drifting a full dispersion in
bulk: under-dense and over-heated, which is one coherent failure -- unbound
streaming material rather than a settled object -- not two. This is not a
contradiction of the `sr2-halos-are-sub-virial` note, which measured halos SR2
*does* find; this measures windows where it found nothing.

What did not recover: whole-tile `peak_contrast` 0.583. Split by region
(*measured*, from the eval npz):

| `peak_contrast` / HR | frozen | tuned |
| --- | ---: | ---: |
| inside the windows (hard 2.5 sigma sphere) | 0.738 | 0.652 |
| **outside** (74% of the tiles' mass) | 0.834 | **0.570** |

Measured with the loss's *own* kernel weighting -- right at the centres --
contrast went 0.646 -> 1.110, i.e. matched. **The objective did exactly what it
was told precisely where it was told, and the field degraded everywhere else,
including the immediate surroundings of the clumps it built.**

### 3.3 Run C: the L2 anchor is the wrong instrument

*Measured.* Raising `w_anchor` 0.1 -> 10 (and `w_low` 50 -> 100, 3000 -> 5000
steps) moved the outside `peak_contrast` ratio **0.570 -> 0.517**: the wrong
direction. `low_k` did come under gate (0.0187), high-k held (0.361), density and
kinematics held (medians 1.039 and 1.082, bulk 0.13 sigma).

The reason is this project's recurring one: **an L2 guard blurs for the same
reason an L2 objective blurs.** Minimising `||Psi - Psi_0||^2` is cheapest to
satisfy with broad, low-amplitude change spread over the field, which is exactly
what erases local peaks. *Caveat*: the step count changed with the weight, so
this is not a clean A/B -- but it certainly did not help.

The underlying cause is structural, not a tuning miss. A convolutional generator
applies **one learned operator at every site**; supervising 43 windows changes
what it does everywhere, and nothing in the objective told it what to do at the
other 99%.

### 3.4 A verdict that reported a run it should have failed

Run C printed *ALL THREE HELD*. Its field check read `low_k` alone, which
constrains only the block-averaged LR scale, and never saw structure outside the
windows sitting at 0.52 of frozen. Fixed: `preserve_ratio` is computed every
eval and gates the verdict at `--contrast-drop-max` (default 0.10), with a test
pinning the exact case that slipped through.

## 4. Why only 43 targets

*Measured.* Of HR's **151** subhalos >= 200p inside the host's R_vir (1.785
Mpc/h):

| | count | keeps |
| --- | ---: | ---: |
| in R_vir, >= 200p | 151 | |
| home Lagrangian tile is one of the 4 trained | 62 | 41% |
| purity >= 0.5 | 58 | 94% |
| window fits inside the scored cube | **19** | 33% |

Three cuts. The four trained tiles hold **42.4%** of the host's 1,678,142
Lagrangian sites (`--n-tiles 4`, inherited from the MSE experiment). Purity
barely bites, which is the Lagrangian-purity result holding up. The third is pure
geometry: the differentiable density is the *valid-centre* deposit, only the
central half of each tile's grid is scored (`region_fraction` 0.5, a 6.25 Mpc/h
cube), the window needs 6 cells of margin leaving ~4.9 Mpc/h, and **each tile's
cube is centred on that tile's own bulk-displaced centre, not on the host**.

Note the two populations differ. Training selects by Lagrangian tile of origin
(43 targets); the R_vir count is an Eulerian sphere (151). They overlap in only
**19** objects -- the other 24 supervised targets live outside R_vir. The R_vir
metric therefore under-credits this run by construction, which is why
`compare_gather_catalog.py` also scores the supervised targets one at a time.

## 5. The Rockstar gate: nothing bound

*Measured.* Run C's four tiles spliced into the cached frozen set8 box (0.78% of
the box, max |change| 4.7 in normalised displacement), full-box Rockstar with the
frozen config, 212,940 halos, 12 min CPU.

| subhalos within R_vir | HR | frozen | tuned |
| --- | ---: | ---: | ---: |
| 100-200p | 144 | 3 | 0 |
| 200-500p | 91 | 1 | 0 |
| 500-2000p | 48 | 1 | 2 |
| 2000p+ | 12 | 6 | 3 |
| **total** | **506** | **11** | **5** |

It went **down**, 0.022 -> 0.010 of HR -- *and that direction is not readable*:
§8.1's null control moves the same number 11 -> 20 with no trained change at all,
so +-9 is the harness noise and 5 is inside it. What survives calibration is the
distance to the **ceiling of 227** (§8.1), which is 20x away and far outside any
noise. The host itself is untouched (identical
`log10 Mvir` 14.852, centre 0.099 -> 0.050 Mpc/h from HR's), so this is not
fragmentation. Halo counts fell in every radial shell (-7, -9, -32, -42, -11 out
to 28 Mpc/h) and box-wide: hosts >= 200p 23014 -> 23000, subhalos >= 50p 30027 ->
29894.

Per supervised target -- did a bound halo appear at each of the 43 places we
asked for one, carrying at least a given share of HR's particles?

| search radius | mass fraction | frozen | tuned |
| --- | ---: | ---: | ---: |
| 1 x r_vir | >= 25% | 0/43 | 0/43 |
| 2 x | >= 10% | 1/43 | 0/43 |
| 3 x | >= 5% | 1/43 | 1/43 |
| 4 x | >= 2% | 10/43 | 7/43 |

**Not one supervised target became a bound halo**, and there is no threshold at
which the tuned field overtakes frozen -- at the loosest it is behind. The strict
row alone has no discriminating power *between these two* (0 for both); the sweep
is what closes the "the test was too strict" loophole. Against the §8.1 ceiling
the same row reads **0/43 against 42/43**, which makes it the sharpest statistic
in this document -- a field that genuinely holds bound subhalos scores 42/43, so
0/43 is a real negative rather than a threshold artifact.

## 6. What this rules out

Every field statistic hit its target. Compact mass at the targets matched HR
(median 1.04), velocity dispersion matched (1.08), bulk velocity matched (0.13
sigma), the LR scale held, the slices look right to the eye. The 6-D finder
disagreed completely.

**Matching a few aggregate moments per window is not a specification of a bound
halo.** The loss constrains three numbers per subhalo -- soft compact mass,
window mean velocity, window dispersion. Enormously many field configurations
carry those moments, and gradient descent found ones that are not bound: mass
piled into the right place with the position-velocity correlation that makes a
halo *findable* scrambled rather than built. The corroborating detail was visible
before Rockstar ran and was under-read: inside the windows, kernel-weighted
contrast rose to 1.11 while hard-sphere contrast *fell* 0.738 -> 0.692. A raised
pedestal, not a sharper peak.

The "good gradient" property bought optimisability, not physical validity. This
is the proxy line's candidate collapse recurring one level down, and it is the
strongest argument so far for gating on the halo finder rather than on anything
differentiable.

**What it does not establish.** That the fine-tune *destroyed* the six missing
subhalos: only counts were compared, never identities, and the two controls in
§7 are unrun. Nor does it say anything about the collateral damage -- the spliced
box keeps the frozen field outside the four tiles by construction.

## 7. Open risks -- items 1-4 are now closed

1. **The ceiling.** ~~Unmeasured, and it gates the reading.~~ **Measured, and it
   passes**: §8.1. True HR tiles through the identical harness give 227 subhalos
   in R_vir and 42/43 supervised targets, with the host preserved to 0.000 Mpc/h.
   The splice geometry does not destroy substructure and §5 is measuring the
   model, not the harness.
2. **The null control.** ~~Unmeasured.~~ **Measured**: §8.1. A frozen re-splice
   with no trained change moves the count 11 -> 20, so the gate's noise is +-9.
   That retires §5's "it went down" and leaves the distance to the ceiling as the
   readable quantity.
3. **The preservation term.** ~~Built but never run.~~ **Run and gated**, and it
   works as designed on its own terms and changes nothing at the finder:
   `preserve_ratio` 1.069 against run C's 0.517, whole-tile `peak_contrast` 0.913
   of HR against frozen's 0.838, density 1.075 and kinematics 1.040 -- then 19
   subhalos in R_vir (null 20) and **0/43**. Collateral damage was real and
   fixing it was not the blocker.
4. **Nothing in the objective asks for binding.** Still true, and §8.2 measured
   what it is worth: `2T/|W|` is the single most separating statistic between the
   HR field and every SR2-derived one (|AUC-0.5| 0.468). **But do not read that as
   "add the virial term and the loss is fixed"** -- the controls separate almost
   as strongly, because §8.2's finding is that the loss never constrained the
   objects at all. A window-based virial term inherits the same defect. The term
   has to be gathered by particle id first; the choice of statistic is second.
5. **Coverage.** `--n-tiles 8` and a wider `region_fraction` would raise 43
   toward 151, at the cost of deposit accuracy (rel-RMS 0.030 -> 0.213 at 0.75).
   ~~Not worth spending until §7.1 says the metric is sound.~~ The metric *is*
   sound (§8.1), so coverage is now a live knob -- but it buys more supervised
   windows, and §8.2 says windows are the problem, so it is not the next spend.
6. **One host, one box, in-sample.** Still open. No generalisation claim is
   available from any run here, and §8.2's finding is about this host's 43 sets.

## 8. The controls, and what the member sets say

### 8.1 The gate, calibrated

*Measured*, 2026-08-20, `HG_WHICH=hr` and `HG_WHICH=frozen` through the identical
splice -> Rockstar -> compare chain (jobs 34861-34863 and 34836-34838).

| subhalos within R_vir | HR | base | ceiling (HR tiles) | null (frozen re-splice) | run C tuned | preserve |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50-100p | 211 | 0 | 91 | 0 | 0 | 1 |
| 100-200p | 144 | 3 | 68 | 4 | 0 | 1 |
| 200-500p | 91 | 1 | 45 | 4 | 0 | 3 |
| 500-2000p | 48 | 1 | 17 | 7 | 2 | 11 |
| 2000p+ | 12 | 6 | 6 | 5 | 3 | 3 |
| **total** | **506** | **11** | **227** | **20** | **5** | **19** |
| supervised bound | 43/43 | 0/43 | **42/43** | 0/43 | 0/43 | 0/43 |

Two numbers make every earlier reading interpretable. The **ceiling is 227** --
0.449 of HR, and 0.449 rather than 1.0 only because the four trained tiles hold
42.4% of the host's Lagrangian sites (§4), so the rest of its subhalos are built
from material that was never replaced. The **noise is +-9**, from a re-splice
whose only difference from the cached box is the run's own regeneration
(rms 1.1e-3, max \|change\| 0.059).

So the gate has ~20x dynamic range and near-perfect per-target sensitivity, and
the tuned and preserve runs sit indistinguishably on the floor.

### 8.2 The loss never moved the objects

*Measured*, 2026-08-21, `submit_bound_discriminator.sh` (job 35106, ~2 min CPU,
no halo finder). Every statistic is evaluated over the **HR member particle ids**
of the 43 supervised subhalos -- identical id sets across fields, because SR2 and
HR share the Lagrangian lattice, so no matching enters anywhere.

| over the HR member sets | HR tiles | run C tuned | frozen | preserve |
| --- | ---: | ---: | ---: | ---: |
| `bound_frac` | **0.428** | 0 | 0 | 0 |
| `2T/\|W\|` | 14.4 | 929 | 917 | 629 |
| `d6` (Rockstar's linking metric) | 0.51 | 1.19 | 1.20 | 1.15 |
| `sigma_v` km/s | 341 | 1005 | 1004 | 851 |
| `r_rms` Mpc/h | 0.525 | 1.00 | 1.10 | 1.01 |

**The fine-tune is indistinguishable from the frozen generator on the sets it was
supervising.** Tuned against frozen, Mann-Whitney AUC over the 43: `bound_frac`
0.483, `2T/|W|` 0.514, `d6` 0.483, `coldness` 0.479, `r_rms` 0.469, `sigma_v`
0.487 -- every one at chance. Median per-set change: **1.6%** in `sigma_v`, 9% in
`r_rms`.

**The cause is the window.** The loss's statistics are Gaussian-window statistics
at the target's location, and a window at a subhalo inside a cluster holds mostly
*host* material. So the loss could report `sigma_v` matched at 1.02 of HR (§3.2)
while the particles HR binds sat at 2.9x, unchanged. This is §3.1's "the statistic
was measuring the host, not the subhalo" recurring in the velocity channel --
§3.1 diagnosed it for run A's density term only, and raising `min_num_p` to 200
was believed to have retired it.

**What this does and does not license.** It does not say which statistic to add;
it says the loss has to be **gathered by particle id rather than deposited in a
window** before the choice of statistic matters. The ids are training-time data
exactly as the HR displacements are (§1.1), and a gather is cheaper than the CIC
deposit §1.2 argued for -- that section's efficiency case stands, but its
statistic does not track the object, which is now measured rather than assumed.
`bound_frac` is the cleanest available target (0.428 against exactly 0.000, AUC
0.93, saturated), with two caveats: it is a hard threshold and needs a soft form,
and per `occupation-ratio-is-gameable` and `tile-overfit-proxy-exploitation`
expect any differentiable form of it to be gamed unless the gate stays real
Rockstar.

The tool's own auto-verdict names `2T/|W|` as "the term the gather loss is
missing". **That wording is too generous and should not be quoted.** Its guard --
refuse to name a statistic that does not separate better than the controls --
fired only marginally (0.468 against 0.425), and the reason the controls separate
is the finding above.

## 9. Where this leaves the approach

Four gates now agree, and the last one closes the method as it stands: the
objective works on its own statistics (§3.2), the collateral damage is fixable
(§7.3), the gate is sound (§8.1), and the supervised material never moved (§8.2).
A differentiable surrogate built from a handful of **window** moments does not
reach the objects, and adding a fourth window moment is not the fix.

Two continuations, in order of cost. Within this method: rebuild the loss on
id-gathered member sets and re-gate -- §8.2 says that is the prerequisite, not an
optimisation. **This has now been done and it works**: `docs/sr2_member_gather.md`
records 72 of 154 supervised subhalos recovered as bound halos against this
document's 0 of 43, with the guards holding, after one further defect the gate
located -- the id-gathered loss constrains only internal moments, so it says what
the object must look like and nothing about where it must be. Outside it: the DMSR null-space flow + critic, already in the
repository, samples from a learned conditional distribution instead of matching
moments, so it cannot be gamed this way -- and it has never been Rockstar-tested
(`dmsr-flow-is-the-fresh-train-answer`).

## 10. Reproduce

```bash
# fine-tune (GPU ~8 min) -- shakeout, train, redraw
HG_MIN_NUM_P=200 HG_W_LOW=100 HG_W_ANCHOR=0.1 HG_W_PRESERVE=1 \
HG_STEPS=3000 HG_LABEL=_preserve bash scripts/slurm/submit_host_gather.sh

# the gate (CPU ~13 min) -- splice, Rockstar, compare
HG_RUN_DIR=$DMSR_REWARD_ROOT/host_gather/set8_h271800_fine_anchored \
bash scripts/slurm/submit_gather_rockstar.sh

# the two controls of section 8.1 (submit them SEPARATELY -- two submitters in
# the same second used to share one env file and both ran as `frozen`)
HG_WHICH=hr     HG_RUN_DIR=... bash scripts/slurm/submit_gather_rockstar.sh
HG_WHICH=frozen HG_RUN_DIR=... bash scripts/slurm/submit_gather_rockstar.sh

# section 8.2, ~2 min CPU, no halo finder -- and it predicted the preserve gate
bash scripts/slurm/submit_bound_discriminator.sh
```

Artifacts under `$DMSR_REWARD_ROOT/host_gather/<box>_h<id>_<rung><label>/`:
`summary.json` (config, targets, history, verdict), `metrics.jsonl` (one row per
eval step), `subhalos.json` (per target: asked, frozen, tuned, both kinematics),
`slices/*.png`, `eval/step*.npz` (the redraw source), `tiles.npz` (the splice
source). The Rockstar comparison lands in
`$DMSR_REWARD_ROOT/flow_rockstar/compare/<box>__<tag>.json`.

§8.2 writes `$DMSR_REWARD_ROOT/bound_discriminator/<box>__<run>.json` (every
set's statistics on every field, the discrimination table, the verdict).

Tests: `tests/features/test_bound_discriminator.py` (19),
`tests/features/test_subhalo_gather.py` (23),
`tests/features/test_finetune_host_gather.py` (12),
`tests/reward/test_compare_gather_catalog.py` (8).
