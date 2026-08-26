# Pilot steps 2-4: what ran, what it settled, what it retracted

**Scope.** A run log for `docs/sr2_substructure_module.md` section 9. It records
steps 2, 3 and 4 as executed on 2026-08-19: step 2 (conditional spread) done and
its consistency gate now passing, step 3's high-pass measurement done (section
3.5), step 4 (capacity) done, plus the two claims this note **retracts** and the
measurement bug that caused them. Nothing here changes the design; step 2
confirms it, step 3 tells it a fixed spectral cut will not work, and step 4
narrows why the module is needed. The one thing now on the critical path is a
design decision, not a measurement: how to replace the high-pass (section 6).

Numbers are tagged *measured* (read from an artifact on disk) or *retracted*
(published earlier in this file's history and now known to be wrong).

Depends on `docs/sr2_substructure_module.md` for the design and
`docs/host_crop_learnability.md` for the crop-scale measurements.

## Tooling

| | |
| --- | --- |
| spectra, bands, ridge, windows | `src/cosmo_sr/features/cond_spread.py` |
| steps 2 + 3 | `scripts/features/measure_conditional_spread.py` |
| step 4 | `scripts/features/overfit_host_mse.py` |
| jobs | `scripts/slurm/{cond_spread_cpu,overfit_host_mse_gpu}.sbatch`, `submit_pilot_steps.sh` |
| tests | `tests/features/test_{cond_spread,measure_conditional_spread,overfit_host_mse}.py` |

```bash
bash scripts/slurm/submit_pilot_steps.sh        # steps 2+3 (CPU) + the step-4 ladder (3 GPU)
ONLY=spread bash scripts/slurm/submit_pilot_steps.sh
```

Artifacts: `dmsr_reward/cond_spread/cond_spread_<fit>_<test>.json` and
`dmsr_reward/host_overfit/<box>_h<id>_<rung><label>/summary.json`.

## 1. Step 2 -- `p(HR | SR2)` is broad at subhalo scale

*Measured*, set8, fit and scored on disjoint slabs separated by 32 sites
(6.25 Mpc/h), targets divided by the local scale `s` of section 4.2, **480,000
sites** (bumped from 120k; see section 7). Held-out `R^2` of the best local map
from SR2's 11^3 displacement neighbourhood to HR's high-pass displacement at the
centre site:

| band | edge (h/Mpc) | linear | + random features | identity | low-pass control |
| --- | ---: | ---: | ---: | ---: | ---: |
| sigma 0.7 | 7.3 | **+0.0064** | +0.0043 | -0.153 | +0.976 |
| sigma 1 | 5.1 | +0.0145 | +0.0134 | -0.212 | +0.979 |
| sigma 2 | 2.6 | +0.0709 | +0.0766 | -0.237 | +0.986 |

Host-footprint sites (>= 1e13). Uniform sites give the same picture with a worse
identity (-0.39 at sigma 0.7).

**The identity gate now passes.** The run cross-checks the identity predictor in
site space against the exact spectrum and refuses a verdict when they disagree by
more than 0.25 (section 4). After the windowing fix a residual 0.42 gap remained;
section 4 records how it was closed, and this run reports `consistent: true`, gap
**0.231**, verdict **BROAD**. The `R^2` numbers above are now certified readable.

Three readings.

**1.1 The conditional mean is empty in the fine modes.** No local linear or
random-feature map finds anything at 7.3 h/Mpc, while the identical pipeline
recovers the smoothed field at 0.976. This is section 6.1's claim -- L2
converges to `E[HR | SR2]`, which is empty there -- as a measurement.

**1.2 Copying SR2's fine structure is worse than emitting nothing.** The
identity predictor scores **negative** at every band. SR2's high-k displacement
is not merely uninformative about HR's; it is wrong enough that a module
inheriting it starts behind zero.

**1.3 The asymmetry is real and the verdict states it.** Residual variance of
any predictor bounds `Var(HR | SR2)` from **above**, so a high `R^2` would have
killed "must sample" outright while a low one supports it without proving it.
The function class here is local, linear-or-random-feature, and 2.15 Mpc/h wide.

Corroborated independently: the windowed tile spectrum gives `r(k) ~ 0.00` above
7.3 h/Mpc, i.e. SR2's displacement and HR's are *uncorrelated* at subhalo scale.
Two routes, one answer.

## 2. Step 4 -- capacity is not the limit; the objective is

*Measured*, set8, host 271800 (`log10 Mvir` 14.81, 611,723 particles). Plain MSE
against HR, one rung unfrozen, 3000 steps. The ratio that licenses the reading
is trainable parameters over target values:

| rung / tiles | params/target | MSE vs frozen | peak_contrast / HR |
| --- | ---: | ---: | ---: |
| `all_blocks` / 1 | 4.43 | 0.392x | 0.840 -> 1.091 |
| `middle_fine` / 1 | 1.06 | 0.400x | 0.840 -> 1.218 |
| `fine` / 4 | 0.053 | 0.422x | 0.851 -> 0.607 |

At 4.43 parameters per target value the rung could have memorised the region
outright. It did not: MSE plateaued at 0.39x frozen, and the same answer appears
at 1.06. **The generator can represent this cluster; the squared loss declines to
make it.** Section 9 step 4 is closed.

Two details worth carrying forward:

* the over-parameterised runs drove `peak_contrast` **above** HR (1.09, 1.22)
  while broadband high-k power fell -- at high capacity L2 does not simply blur,
  it over-sharpens the predictable core and erases the rest. The verdict text
  reads the power ratio only and calls that "blurred"; it should consult the
  structure statistic. **Open.**
* frozen SR2 sits at 0.84 of HR's `peak_contrast` and *above* HR's fraction of
  cells over `delta = 100` (0.00855 against 0.00812) while holding ~7% of HR's
  clump count inside clusters (section 6.1's table). Amplitude and one-point
  contrast statistics are weak proxies for the deficit; it is a boundedness
  failure and only a halo finder settles it (section 7.1).

## 3. Retracted

Both retractions have the same cause, section 4.

**3.1 "12-13% of the residual power sits above 8 h/Mpc."** *Retracted* as
measured -- it came from unwindowed 64^3 tile spectra -- but the corrected value
is close: the exact whole-box spectrum puts **9.78%** of the residual power above
8 h/Mpc (section 3.5). The magnitude survived; the method did not, and the
number is now on the same footing as everything else.

**3.2 "`r(k)` never falls below 0.5 out to Nyquist."** *Retracted*, and it was
the opposite of the truth: windowed, `r` is ~0.00 above 7.3 h/Mpc.

A recommendation built on 3.2 -- that step 2 was measuring the wrong observable
and should be re-targeted from displacement onto a collapse statistic -- is
**withdrawn**. It rested entirely on displacement appearing predictable. It is
not, and step 2's target was correct as designed.

## 3.5 Step 3 -- no fixed `k` separates substructure from bulk

*Measured*, set8, the exact whole-box spectrum of `Psi_SR2` vs `Psi_HR`. This is
what skeleton item 4's high-pass cut has to be read off, and it **upgrades
`docs/host_crop_learnability.md` section 4 from arithmetic to measurement**.

SR2's displacement decorrelates from HR's early:

| statistic | k (h/Mpc) |
| --- | ---: |
| `r(k) = 0.5` (whole box) | **2.33** |
| `r(k) = 0.1` | **3.67** |
| cluster tiles, `r = 0.5` / `0.1` | 2.39 / 3.84 |
| HR power above 8 h/Mpc | 0.26% |
| residual `Psi_HR - Psi_SR2` power above 8 h/Mpc | 9.78% |

Against each object's Lagrangian scale `k = 2 pi / 2 R_L`:

| object | k (h/Mpc) |
| --- | ---: |
| 1e14 cluster (bulk) | 0.47 |
| 2000p subhalo | 2.06 |
| 1e12 host (bulk) | 2.16 |
| 366p subhalo (cluster median) | 3.62 |
| 50p subhalo | 7.04 |

Two conclusions, and they settle the *measurement* half of step 3.

**A fixed Lagrangian-`k` high-pass cannot separate the two.** A 2000-particle
subhalo (2.06) and a 1e12 host (2.16) sit at the same wavenumber -- they are the
same mass -- and subhalos span 2-7 h/Mpc while hosts span 0.47-2.16. The bands
overlap; the separation is clean only at the extremes (a cluster at 0.47 against
a 50p subhalo at 7.04). **The host/substructure distinction is relational -- a
mass ratio -- not spectral.**

**The design's 8 h/Mpc cut is in the wrong place.** SR2's error lives at 2-4
h/Mpc (where `r` falls through 0.5 to 0.1), entirely *below* 8, and only 0.26% of
HR's power and ~10% of the residual sit above it. Filtering the module's output
above 8 h/Mpc would remove ~90% of what it must add and forbid it from assembling
substructure rather than restricting it to substructure -- the 366-particle clump
it must build is a 3.6 h/Mpc mode. The guarantee item 4 wants ("do not move or
resize the host") is worth keeping; a fixed spectral cut is the wrong instrument.

**Still owed:** the *decision*. A host-relative cut (per-crop at a fraction of the
host's own `R_L`) or a constraint on the residual's low-order moments inside the
host footprint would express the guarantee without forbidding substructure.
Neither is built. This is a design choice, not a measurement (section 6 item 2).

## 4. The bug: a sub-cube's FFT is not the sub-cube's spectrum

The load-bearing lesson, and the reason this note exists.

`radial_cross_spectra` was called on 64^3 tiles cut out of the 512^3 box. An FFT
treats its input as periodic; a tile carved from a larger field is not, so it
carries an artificial step at every face. That step is the tile's **coherent
bulk flow**, which HR and SR2 share almost exactly -- so its leakage arrives in
both fields nearly identically and reports agreement at wavenumbers where there
is none.

*Measured*, set8, above 7.3 h/Mpc:

| tile | `r` unwindowed | `r` with Hann | `P_diff/P_HR` unwindowed | with Hann |
| ---: | ---: | ---: | ---: | ---: |
| 398 | 0.831 | **-0.017** | 0.314 | 1.118 |
| 462 | 0.835 | **-0.004** | 0.305 | 1.057 |
| 100 | 0.921 | 0.010 | 0.158 | 2.468 |
| 7 | 0.890 | 0.005 | 0.215 | 2.221 |

The windowed values agree with the site-space measurement (1.06-1.24) that the
unwindowed spectrum contradicted. Note the direction: leakage drags any
power **ratio** toward 1, so it hides deficits rather than inventing them.

**What caught it.** Not inspection -- a cross-check. The run scores the identity
predictor two ways, in site space and off the spectrum, and refuses to emit an
`R^2` verdict when they disagree by more than 0.25. They disagreed by 0.80. Two
plausible explanations for the gap (the `1/s` weighting, then the Gaussian band
definition) were both tested and both wrong before the real cause was found; the
gate is what kept a wrong number from being published in the interim.

**Fixed.** Tile spectra are Hann-windowed (`hann_window`); an exact whole-box
spectrum was added, which needs no window because the box genuinely is periodic
and is what the high-pass decision must be read off; the cross-check compares
only against that exact spectrum. `field_report` in the step-4 script was
windowed too -- the ladder's high-k ratios in section 2 are biased toward 1 and
need re-running. `test_an_unwindowed_subcube_fakes_correlation_at_high_k` pins
the reproduction.

**The residual 0.42 gap, and the second fix.** Windowing was necessary but not
sufficient: with the exact whole-box spectrum in place the gate still read
`inconsistent`, site space -0.15 against spectrum -0.57. Two mismatches, now
both closed (`test_spectrum_identity_matches_the_real_space_high_pass_by_parseval`
pins the first):

1. **The high-pass operator.** The site route's target is `x - gaussian_filter
   (x, sigma)`, a *soft* Gaussian whose Fourier transfer function
   `H(k) = 1 - exp(-(k dx sigma)^2/2)` is still 0.61 at the quoted 7.3 h/Mpc
   edge -- not a brick wall. The spectrum cross-check applied a *sharp*
   `k >= 7.3` mask, scoring the identical predictor on the fully-decorrelated
   tail only. `spectrum_identity_r2` now weights each mode by `H(k)^2`, so by
   Parseval it computes the site-space functional exactly. This is the bulk of
   the gap.
2. **The population.** The exact spectrum is the whole box, so the site score it
   is compared against must be the whole-box-sampling `uniform_sites`, not the
   host-enriched `host_sites` the physics reading uses. The gate now compares
   like with like and records `gate_subset`.

Post-fix the gap is 0.231 (`consistent: true`). This reconciles with the
"Gaussian band definition tested and wrong" attempt above: that was pre-windowing,
where leakage dominated and no band definition could close an 0.80 gap; once the
spectrum is leakage-free the Gaussian weighting is exactly the right correction.
The remaining 0.23 is not sampling noise -- bumping to 480k sites barely moved
the site identity (-0.398 -> -0.390) -- but the far-slab-vs-whole-box population
difference the held-out split imposes; only a cross-realisation split removes it.

## 5. State of the pilot

| step | status |
| --- | --- |
| 1 `mu` collapse | **unrun**, still gates the design (section 7.4) |
| 2 conditional spread | **done**, section 1 -- broad, design supported, gate consistent |
| 3 re-choose the high-pass | **decided** -- per-host affine-moment projection, `docs/sr2_moment_constraint.md` (measurement was section 3.5) |
| 4 capacity vs incentive | **done**, section 2 -- objective, not capacity |
| 5 train Option B | not started |
| 6 Rockstar gate | not started |

## 6. Next actions

1. ~~Re-run both jobs with the windowing fix.~~ **Done** (2026-08-19, 480k
   sites, 21 min CPU): step 2 returns `BROAD` and `consistent` (section 1); the
   exact whole-box spectrum is section 3.5.
2. **Decide the high-pass** (section 9 step 3, skeleton item 4). The measurement
   is now in (section 3.5) and confirms a fixed Lagrangian-k cut is the wrong
   instrument -- the bands overlap and 8 h/Mpc filters away ~90% of the residual.
   The candidate replacement is a host-relative cut or a constraint on the
   residual's low-order moments inside the host footprint. **Decided** in
   `docs/sr2_moment_constraint.md`: a per-host affine-moment projection (the
   moment route), which is relational rather than spectral and leaves the
   compaction mode of open risk 5 available. This was the last item feeding the
   module; step 5 is now unblocked.
3. **Run step 1.** Cheap, CPU, existing artifacts, and it still gates everything.
4. Fix the step-4 verdict to consult `peak_contrast` (section 2).
5. **Regenerate `set9_hr_owner.npy`** (Rockstar `--write-assignment`) to enable
   the cross-realisation split that would close the residual 0.23 identity gap
   (section 4) and lift limit 1.

## 7. Limits

1. **One box, one seed.** set8, seed 0. The held-out split is spatial, not a
   second realisation: set9 has no owner array and its raw particle dumps were
   deleted (`particles_deleted: true`), so it cannot contribute host-stratified
   sites without re-running Rockstar. Large-scale modes are therefore shared
   between fit and test; every target is high-pass and the slabs are 6.25 Mpc/h
   apart, but this is weaker than a cross-realisation split. It is also what
   leaves the identity gate at gap 0.23 rather than ~0 (section 4): the far-slab
   uniform sites are not the whole box the exact spectrum is computed on.
   Bumping the sample to 480k sites (from 120k) shrank the sampling component but
   not this population one -- the site identity moved only -0.398 -> -0.390.
2. **Step 2 bounds one function class**, local and 2.15 Mpc/h wide. It cannot
   exclude a predictor with a larger receptive field or a different form.
3. **Step 4 is one host in one box**, and says nothing about generalisation --
   overfitting was the point.
4. **No halo finder ran.** Every statement here is about fields. Whether the
   clumps are *bound* is section 9 step 6.
