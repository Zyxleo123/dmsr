# Reward-guided residual diffusion for catalog-faithful DMSR

Offline, reward-weighted residual diffusion on top of the frozen SR2 generator. No
null-space projection on the output, and no policy gradients *initially* — §5 says
when they would become justified and which objective to use. This document is the
implementation record: what exists, what was tested on CPU, what has to run on a
GPU, and what the decision rule is.

**Status as of 2026-08-01: Gate A failed, Gate B ran anyway and is void.**
See **[residual_prior_diagnosis.md](residual_prior_diagnosis.md)** for the results
and the diagnosis. In short: the residual prior is conditioned correctly and
correlates with the true residual at high `k` (`r = 0.33`), but only 11% of its
power is coherent signal, so composing it destroys the density field and Gate B's
candidates contained 1-6 halos per box. Gate B was sampled seven hours *before*
the Gate A verdict was written, so its numbers measure the bug, exactly as §3a
warns. **The support question remains unanswered** — no valid candidate has yet
reached the reward.

The reward model and the constraint calibration both succeeded and are sound; two
caveats (a diagonal `C` that puts 71% of `R_cat` on abundance, and a
`low_k_change_max` that rejects ground truth) are recorded in the diagnosis doc.

The decision table at the end is therefore still a *rule*, not a result.

**The scientific target is `<N_sub | M_host>`, the mean number of subhalos in
hosts of mass `M_host`.** SR2's occupation function is nearly *flat* where HR
rises by two decades. Low-mass subhalo abundance is a secondary deficit. A run
that fixes abundance and leaves occupation flat is an informative failure, not a
partial success, and §4 gives it its own verdict label.

Changes since the previous revision, all of which move a decision rather than
confirm one:

* **A new CPU audit (§2) found the chunk core mask deletes hosts in proportion to
  their mass**, leaving the `1e14 Msun/h` host bin completely empty at the
  then-configured `chunk_hr = 128` and both upper reliable bins below the usable
  threshold — so Gate B's occupation criterion was not evaluable at all.
  Measured over all 16 boxes, this moved the geometry to **`chunk_hr = 256`,
  `B = 8`**, where all four reliable bins clear the bar. The `1e14` bin is
  confirmed evaluation-only from two independent directions (host counts, and a
  covariance audit in which it absorbs 96% of the baseline `D²`).
* **Two statements in the previous revision were wrong** and are corrected in §1:
  a zero-initialised output head does **not** make a stochastic sample equal SR2
  (only `residual_scale = 0` does), and a short receptive field does **not**
  mathematically prevent low-`k` changes.
* **Gate B was reframed** (§4). It is a support check on the *current prior's
  ordinary samples*, not a final no-go test, and a negative result now routes
  into the escalation ladder in §5 instead of ending the project.
* **The reward gained `R_occ` and `R_abund`** (§1) and Gate B is decided on
  occupation, not on the joint `R_cat`.

---

## 1. Implementation

### Files added

Library (`src/cosmo_sr/reward/`):

| File | Role |
| --- | --- |
| `base.py` | frozen SR2 wrapper, `compose`, base-field cache |
| `targets.py` | paired residual targets `dPsi* = Psi_HR - Psi_base`, crop dataset |
| `model.py` | `ResidualDenoiser` over the existing `Map2MapUNet3D` |
| `diffusion.py` | VP-cosine schedule, eps-prediction, DDIM, `x0_clip` |
| `sampling.py` | receptive-field measurement, valid-core tiling, full-box sampling |
| `train.py` | one training loop for both the prior and the distillation |
| `geometry.py` | Lagrangian chunks, Eulerian purity grid, catalog core mask |
| `catalog.py` | chunk summaries, pooling, summary vector, JSONL I/O |
| `reward.py` | `mu_HR`, `C_reg`, `R_cat`, stratified ensembles |
| `constraints.py` | the five field-fidelity measurements and the feasibility filter |
| `replay.py` | marginal contribution, elite selection, bounded weights, manifest |
| `heldout.py` | catalog statistics that are *not* rewarded, box-level bootstrap |
| `fields.py` | memory-bounded power spectra, PDF, equilateral bispectrum |
| `pipeline.py` | field -> particles -> Rockstar -> attributed chunk summaries |
| `paths.py` | every large-output path, all under `$ZFS` |

Entry points (`scripts/reward/`): `cache_sr2_base.py`,
`audit_residual_targets.py`, `audit_host_counts.py`,
`audit_reward_covariance.py`, `measure_receptive_field.py`,
`train_residual_prior.py`, `catalog_summaries.py`, `fit_reward_model.py`,
`calibrate_constraints.py`, `sample_oracle_candidates.py`, `score_oracle.py`,
`oracle_report.py`, `build_replay.py`, `train_reward_distill.py`,
`generate_eval_boxes.py`, `eval_catalogs.py`, `eval_full_metrics.py`,
`sanity_real_catalogs.py`.

Batch scripts: 15 files under `scripts/slurm/`, sharing `_reward_common.sh`.

Configs: `configs/reward/reward.yaml` (reward, geometry, constraints, replay,
reliable host bins, split roles — single source of truth), `residual_prior.yaml`,
`distill_round0.yaml`.

Tests: `tests/reward/`, 122 tests.

### Model inputs and outputs

```
eps_hat = eps_phi( u_t , t , y_lr , Psi_base , z )
```

* `u_t` — noised **whitened** residual, `(6, N, N, N)`; whitening is a fixed
  per-channel `sigma_res` from the Stage-2 audit, so the network sees O(1) inputs
  without any learned normalisation that could drift between prior and student.
* `y_lr` — LR displacement+velocity, `(6, N/8, N/8, N/8)`, block-upsampled inside
  the model.
* `Psi_base` — the frozen SR2 output, concatenated as conditioning.
* `t` — continuous diffusion time, sinusoidal embedding, FiLM.
* `z` — redshift, same embedding path; constant at `z = 0` for now.
* Output — `eps_hat` in whitened units. The composed field is

```
Psi_hat = Psi_base + a * sigma_res * u_0        (a = residual_scale)
```

The output head is zero-initialised, so a freshly built model predicts `eps = 0`.
**That does not make a sampled field equal SR2**, and an earlier version of this
document said it did. With `eps_hat = 0` the DDIM update still returns
`u_0 = u_t / alpha(t)`, i.e. the *initial noise* rescaled — a stochastic,
generally large residual, not zero. The zero-init head only guarantees that the
network contributes nothing; it says nothing about the noise the sampler starts
from. This is not a hypothetical: it is exactly the failure §2 records as "DDIM
at `t_max` blows up for a zero-init model", and the mitigation there
(`t_max = 0.999`, `x0_clip = 4.0`) bounds the blow-up without removing it.

**The only thing that guarantees bit-exact SR2 is `residual_scale = 0`**, which
short-circuits composition and returns `Psi_base` unchanged. Every claim of the
form "the model starts at the frozen baseline" must cite `a = 0`, never the
zero-init head.

Because the clip is load-bearing rather than cosmetic, the **fraction of voxels
clipped at `x0_clip = 4.0`** is logged at every sampling step, in standardized
residual units (multiples of `sigma_res`, so the number is comparable across
channels and runs). A clip fraction that is not small means the sampler is
operating in the regime where `x0_clip` — not the learned model — is setting the
residual amplitude, and the sample is an artifact of the clip.

Composition is a plain sum in catnorm units — no null-space projection.

Normalisation is `ChannelGroupNorm3d` (pointwise), not `nn.GroupNorm`. GroupNorm
takes statistics over the whole volume, which couples every voxel to every other
and makes the output depend on the crop size; that silently breaks tiled
inference. This was found by a test, not by reasoning — see §2.

### Reward

**The scientific target is occupation.** `<N_sub | M_host>` — the average number
of subhalos inside hosts of mass `M_host` — is nearly *flat* in SR2 where HR
rises by two decades (§2). Low-mass subhalo abundance is a real but secondary
deficit. A run that improves abundance while leaving occupation flat has not
partially succeeded; it has produced a **scientifically informative failure**,
and it must be reported and labelled as one rather than counted toward a pass.

```
R_cat(E) = - (s(E) - mu_HR)^T C_reg^{-1} (s(E) - mu_HR),     C_reg = C + lambda I
```

`s(E)` is an 11-vector: 6 subhalo-abundance bins and 5 mean-occupation bins.
`mu_HR` and `C` come from HR chunk summaries, bootstrapped into ensembles of the
same size and stratification as the ones being scored, so `C` describes the
sampling noise of a *`B`-chunk ensemble* (`B = 8`, one box), not of a single chunk.
`lambda = 0.1 * mean(diag(C))`; the shrinkage and `cond(C_reg)` are written into
every manifest. Occupation is stored as separate numerator and denominator so
ensembles pool by summation and merging is exact.

#### `R_occ` and `R_abund` — the diagnostics Gate B is decided on

The 11-dimensional joint statistic is unchanged; two sub-scores are computed and
reported alongside it:

| Score | Definition |
| --- | --- |
| `R_cat` | joint, all 11 bins |
| `R_occ` | the 5 occupation bins against the corresponding sub-block of `C_reg` |
| `R_abund` | the 6 abundance bins against their sub-block |
| `R_occ_reliable` | `R_occ` restricted to the reliable host bins |

Each sub-score is a **marginal** Mahalanobis distance — the sub-vector against
the inverted sub-block of `C_reg` — not a slice of the joint precision. A
precision slice is the *conditional* form and mixes in the other block's
residual through the cross-covariance, which is exactly what these numbers exist
to separate. The three do not sum to `R_cat` and are not meant to.

**A joint Mahalanobis improvement alone does not pass Gate B; occupation
improvement is required explicitly.** Two facts make this non-negotiable rather
than stylistic, and both are pinned by tests in `tests/reward/test_sub_rewards.py`:

* an abundance-only fix leaves `R_occ` bit-for-bit unchanged, so `R_cat` moving
  is not evidence that occupation moved;
* with correlated bins, an abundance-only fix can make `R_cat` **worse** — being
  consistently wrong in both blocks costs less Mahalanobis distance than being
  wrong in one. The sign of `ΔR_cat` therefore carries no information about
  occupation in either direction.

#### Reliable host bins

Host bins, by index: `0: 1e12`, `1: 3.16e12`, `2: 1e13`, `3: 3.16e13`, `4: 1e14`
`Msun/h`.

* **reliable:** 0, 1, 2, 3
* **upper reliable:** 2, 3 (`1e13`, `3.16e13`) — the mass range where SR2's
  occupation deficit is largest and the hosts still exist in useful numbers
* **sparse, evaluation-only:** 4 (`1e14`)

Gate B requires improvement in **at least two reliable bins, including at least
one of the two upper reliable bins**, and may not rest on bin 4. An
abundance-only improvement is recorded under its own verdict label,
`abundance_only_improvement`, never folded into a pass.

§2 measures how well these bins are actually populated, and the answer changes
the geometry.

### Bins

Particle mass is `5.82e8 Msun/h` and Rockstar keeps halos with >= 20 particles, so
the resolution floor is `1.2e10 Msun/h`.

* subhalo mass: 6 bins, `log10 M` from 10.1 to 13.1
* host mass: 5 bins, `log10 M` from 12.0 to 14.5
* `min_sub_particles = min_host_particles = 20`, matching `rockstar.cfg`

### Boundary handling

Halo finding runs on the **complete 512³ periodic box**, exactly as the frozen
pipeline does. Displacements are large (median `|Psi| ~ 36` HR cells), so a
Lagrangian crop does not fill the Eulerian cube it would be compared against —
`cic_density_valid_center` already records `r = 0.08` for the naive wrapped crop.
Catalog objects are attributed back to Lagrangian chunks *afterwards*:

> A halo `h` belongs to chunk `c` iff, for every Eulerian cell `e` with
> `|centre(e) - pos(h)|_inf <= radius_mult * Rvir(h)` (periodic),
> `majority_id[e] == c` **and** `majority_frac[e] >= min_purity`.
> Otherwise `h` is boundary-contaminated and enters no summary.
> A subhalo additionally requires its host in the same chunk.

Chunk volume is the volume of the cells that pass the same test, not the nominal
chunk volume, so masking cannot bias number densities low. Testing the `Rvir`
sphere rather than Rockstar's bound-particle set rejects slightly more halos than
necessary — the safe direction.

`chunk_hr = 256` (50 Mpc/h, 8 chunks per box). The original value of 128 came
from the largest host whose Lagrangian patch must fit in one chunk (`1e14
Msun/h` collapses from 13.5 Mpc/h). That argument is necessary but not
sufficient — the surrounding *Eulerian* neighbourhood must also be chunk-pure,
and for a cluster it is not — so it was replaced by the measurement in §2.

### Field constraints

Feasibility filter, not a weighted reward. A candidate violating any hard
threshold cannot become an elite.

| Constraint | Definition | Placeholder |
| --- | --- | --- |
| `low_k_change_max` | `\|\|A(Psi_hat) - A(Psi_base)\|\| / \|\|A(Psi_base)\|\|` | 0.02 |
| `displacement_power_error_max` | mean `\|T(k) - 1\|` vs HR | 0.40 |
| `density_power_error_max` | mean `\|log(P_hat / P_HR)\|` | 0.05 |
| `lr_consistency_error_max` | `\|\|A(Psi_hat) - y_lr\|\| / \|\|y_lr\|\|` | 0.05 |
| `diversity_min` | residual sample spread / residual RMS | 0.05 |

These are marked `calibrated: false` in the config. `calibrate_constraints.py`
replaces them with values measured from the frozen SR2 baseline and the HR
box-to-box scatter — including a cosmic-variance floor for `low_k_change`, so the
threshold cannot be tighter than the natural HR-to-HR variation. A NaN is a
violation, never a pass.

### Replay selection

```
A_{k,i} = R(E_k) - R(E_k^{i -> base})
```

The counterfactual swaps chunk `i`'s cached summary for its frozen-SR2 summary, so
it is arithmetic on cached vectors and **the halo finder is never re-run** — which
is the only reason per-chunk credit is affordable. `A` is a leave-one-out
difference, not a Shapley value; with `B = 8` the exact version costs `2^8`
evaluations and the interactions are small.

A chunk is selected iff **both**: its ensemble is feasible and in the top 20% of
feasible rewards, **and** its own `A_{k,i} > 0`. Weights are
`w = exp(clip(A, 0, A_max) / tau)`, clipped at `w_max = 10` and mean-normalised,
and the run aborts if `max(w)/mean(w)` exceeds its bound.

### Tiling and the receptive field

Full-box sampling tiles, because a 512³ six-channel field at width 48 does not fit
on one GPU. Valid-core tiling only gives a tiling-independent answer if the tile
margin exceeds the receptive-field half-width, so that is *measured*, never
assumed (`scripts/reward/measure_receptive_field.py`). The measurement is
width-independent, so the production model is measured on a width-2 stand-in in
seconds instead of hours.

| levels | blocks | half-width | Mpc/h | margin | tile | overhead |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 16 | 3.1 | 24 | 176³ | 2.6× |
| 1 | 2 | 24 | 4.7 | 32 | 192³ | 3.4× |
| 2 | 1 | 26 | 5.1 | 32 | 192³ | 3.4× |
| **2** | **2** | **41** | **8.0** | **48** | **224³** | **5.4×** |
| 3 | 1 | 42 | 8.2 | 48 | 224³ | 5.4× |
| 3 | 2 | 65 | 12.7 | 72 | 272³ | 9.6× |

The original `levels=3, blocks=2` would need a 272³ tile and **9.6× the compute of
an untiled pass** — tiling that deep a model is close to self-defeating. The
configured model is now `levels=2, blocks=2`: the same reach as `levels=3,
blocks=1` at the same cost, ~8 Mpc/h of context (far beyond any subhalo's virial
radius, <= 1 Mpc/h), and 5.4× instead of 9.6×.

An earlier version of this document added that a model which "cannot see beyond
8 Mpc/h cannot manufacture a large-scale offset in the first place." **That is
false as stated and the low-k constraint must not lean on it.** A bounded
receptive field constrains how each output voxel depends on the input; it does
not bound the *aggregate* of those voxels. A residual of `+c` applied
independently at every voxel is a pure `k = 0` mode, and every voxel of it was
produced from a strictly local neighbourhood. More generally, a locally
generated field has power at all `k`; a short reach reduces the model's ability
to *coordinate* long-wavelength structure, and so makes large low-k excursions
less likely, but "less likely" is not "impossible" — and a reward optimiser is
precisely a machine for finding the unlikely.

So the receptive field is a compute-and-tiling argument only. The **explicit
low-k constraint stays in force and is load-bearing**: it is measured on the
composed field, per candidate, and it is a hard feasibility threshold, not a
prior belief about what the architecture can do.

The margin is `half-width + 2`, rounded up to the scale factor. The `+2` is not
padding for its own sake: the outermost shells of the support sit at the float32
noise floor, and the measurement wobbles by one cell between probe sizes (40, 41,
40 at probe 96, 144, 216). Both sampling jobs re-measure and **hard-exit** if the
margin is too small.

---

## 2. CPU tests — all run, all passing

`python -m pytest tests/reward -q` → **122 passed** (109 + 8 sub-reward + 5
clip-logging tests). Nothing outside `src/cosmo_sr/reward/` imports the package, and
`pytest tests --ignore=tests/reward` was run as a regression check →
**586 passed, 1 skipped**, so no existing behaviour changed.

Note `pyproject.toml` already sets `addopts = "-q"`. Passing `-q` again makes it
`-qq`, which suppresses the pass/fail summary line entirely — a run that looks
silent is not a run that failed.

(The `§n` references inside this table are to the sections of the original
*plan*, not to this document's numbering.)

| File | N | Covers |
| --- | --- | --- |
| `test_base_composition.py` | 16 | §1: `a = 0` is bit-exact, frozen params get no gradients, no null-space projection, zero-init head, seed reproducibility and seed diversity, shapes/dtypes, whiten roundtrip, CIC accepts the composed field, normalisation scale preserved |
| `test_residual_targets.py` | 11 | §2: `base + target` reconstructs HR, survives normalize/denormalize, cached == on-the-fly, periodic crop alignment and wrapping, split-leak detection, `crop_hr` divisibility, residual statistics |
| `test_sampling_and_train.py` | 14 | receptive field is measured not assumed, width-independence, zero-init head still measurable, too-small margin refused, margin rounding and slack, `TileSpec` geometry, tiling-independence of the core, full-box reproducibility, physical output scale, per-sample loss weights, CPU smoke of prior training and of distillation including its abort paths |
| `test_geometry.py` | 13 | `chunk_hr` justification and divisibility, tiling, purity under zero and large displacement, halo assignment, rejection of impure/straddling/oversized halos, periodic assignment, effective volume, NPZ roundtrip |
| `test_catalog_reward.py` | 19 | §4 all ten required properties: HR-identical is maximal, monotone degradation, correlated bins, shrinkage prevents singularity, empty host bins give no NaN, pooling is associative, chunk order irrelevant, boundary-excluded objects never enter, unoptimized fields don't move the reward, serialized == in-memory |
| `test_constraints.py` | 12 | §5: zero residual passes, large-scale offset fails low-k, small-scale noise does not, excess high-frequency noise fails power, LR-inconsistent field fails consistency, classification deterministic, NaN never passes, missing values are violations, disabled thresholds, diversity detects a collapsed sampler |
| `test_replay.py` | 14 | §7: counterfactual reward is exact, improving chunks get positive `A`, order-invariance, missing baseline errors, top-quantile + positive-marginal rule, infeasible exclusion, ensemble-level quantile, weight bounds and normalisation, manifest roundtrip and checksums, training refuses held-out boxes |
| `test_sub_rewards.py` | 8 | `R_occ`/`R_abund`: index blocks partition the vector, sub-scores are zero at `mu`, the blocks are **marginal** so one cannot move the other, an abundance-only fix leaves `R_occ` untouched (and can make `R_cat` *worse*), empty host bins give NaN gaps not fake matches, gaps shrink monotonically toward HR, the reliable-bin variant ignores excluded bins, block precision equals an explicit sub-matrix inverse and is *not* the joint-precision slice |
| `test_clip_logging.py` | 5 | a zero-init model does **not** sample a zero residual, one log entry per step in sigma units with descending `t`, the clip is load-bearing at step 0 (>50% of voxels), `x0_clip = 0` logs nothing, and the tiled full-box sampler logs identically to the untiled one |
| `test_heldout_metrics.py` | 10 | §9 held-out statistics: two-halo correlation on Poisson vs clustered fields, relative-velocity moments, isolated-vs-sub counts, resolution cuts, empty catalogs, box-level bootstrap brackets the mean and handles single boxes and non-finite entries |

Two bugs of consequence were found by these tests rather than by inspection, and
both would have produced quietly wrong science:

1. **`nn.GroupNorm` breaks tiled inference.** Its statistics are global, so the
   output of a tile depended on the tile size and valid-core stitching left seams.
   Fixed by switching the denoiser to pointwise `ChannelGroupNorm3d`.
2. **DDIM at `t_max` blows up for a zero-init model.** With `eps_hat = 0`,
   `u_0 = u_t / alpha(t_max)` and `alpha -> 0`, giving enormous first samples.
   Fixed with `t_max = 0.999` and `x0_clip = 4.0` on the predicted clean residual.

### Real-data sanity check — run, and it changes the picture

`scripts/reward/sanity_real_catalogs.py` was run on the existing `set12` catalogs
(1 HR run, 9 frozen-SR2 seeds, whole box, no chunk attribution) to confirm the
summary vector registers the target failure before paying for any HR catalog jobs.
It does — but the dominant failure is **not** the one the plan names.

| bin | HR | SR2 (mean of 9) | SR2/HR |
| --- | --- | --- | --- |
| `nsub` 1.26e10 | 33226 | 12499 | **0.376** |
| `nsub` 3.98e10 | 12802 | 9676 | 0.756 |
| `nsub` 1.26e11 | 4375 | 4284 | 0.979 |
| `nsub` 3.98e11 | 1615 | 1489 | 0.922 |
| `nsub` 1.26e12 | 555 | 436 | 0.786 |
| `nsub` 3.98e12 | 159 | 147 | 0.922 |
| `<N_sub>` @ 1e12 | 4.92 | 3.97 | 0.808 |
| `<N_sub>` @ 3.16e12 | 14.11 | 6.78 | 0.480 |
| `<N_sub>` @ 1e13 | 44.04 | 11.45 | 0.260 |
| `<N_sub>` @ 3.16e13 | 125.37 | 13.80 | **0.110** |
| `<N_sub>` @ 1e14 | 337.57 | 17.27 | **0.051** |

**Occupation is the primary failure; abundance is secondary.** The low-mass
abundance deficit is real (62% missing in the lowest bin) but it is the *smaller*
problem. **SR2's halo occupation is essentially flat in host mass:**
3.97, 6.78, 11.45, 13.80, 17.27 subhalos as host mass rises by two decades, where
HR goes 4.9 -> 337.6. SR2 does not merely under-resolve small subhalos, it fails
to populate rich hosts at all — it puts ~17 subhalos in a `1e14` host that should
contain ~338. That is a 20x deficit against 2.7x for the worst abundance bin, and
it means the one-halo term is being missed structurally rather than at the
resolution floor. Good news for this project: it is a large, unambiguous signal
for the reward to work with.

Two warnings follow from the same numbers, and both are actionable now:

* **The seed-scatter covariance is not usable as a reward.** It gives
  `Mahalanobis = 98007` over 11 bins, i.e. ~94 sigma per bin, because SR2's
  seed-to-seed scatter is the generator's own noise (`lambda = 6.7e-6`) rather than
  cosmic variance. Under that `C`, 88% of the distance comes from just three bins
  (`occ@3.16e13` 49%, `nsub@1.26e10` 22%, `occ@1e14` 19%) and several bins
  contribute *negatively* through off-diagonal terms. This is exactly the
  degenerate one-direction reward that §1's shrinkage discussion warns about. It is
  why `fit_reward_model` estimates `C` from **HR** ensembles; the number above is
  indicative of the deficit, not of the reward.
* **The top host-mass bins are sparsely populated.** `set12` has 86 hosts above
  `3.16e13` and 21 above `1e14` in the whole box, *before* purity masking — and
  masking is what turns this from thin into fatal (§2). The HR-ensemble covariance handles this correctly by
  giving those bins large variance and hence little precision — which is another
  reason Stage 4b is not optional. If `fit_reward_model` reports the top bin
  contributing most of the reward *and* a large `cond(C_reg)`, drop the `1e14` host
  bin rather than trying to optimise five hosts' worth of noise.

### Host-count audit — run on `set12`, and it moves the chunk geometry

`scripts/reward/audit_host_counts.py` asks the question the occupation reward
depends on: **how many independent hosts does each host-mass bin actually
have, after chunk attribution?** The denominator of `<N_sub | M_host,i>` is a
host count, so a bin with five hosts in an ensemble is not a measurement.

Run on `set12` (the only box with an HR catalog today), at the configured
`chunk_hr = 128`, `min_purity = 0.8`, `B = 16`:

| host bin | whole box | after attribution | retention | per `B=16` ens. | p10 | empty ens. | occ. CI95 rel. width | usable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1e12` | 2029 | 751 | 0.37 | 187.7 | 163 | 0% | 0.18 | yes |
| `3.16e12` | 690 | 204 | 0.30 | 50.7 | 41 | 0% | 0.34 | yes |
| `1e13` | 236 | 52 | 0.22 | 12.9 | 8 | 0% | 0.59 | **no** |
| `3.16e13` | 86 | 10 | 0.12 | 2.4 | 0 | 11.5% | 0.74 | **no** |
| `1e14` | 21 | **0** | **0.00** | 0.0 | 0 | 100% | — | **no** |

Three things follow, and the third is the one that matters.

1. **The `1e14` bin is not sparse — it is empty.** Not one of set12's 21
   cluster-mass hosts survives attribution. The plan's question "may the `1e14`
   bin be optimized if there are enough independent hosts?" is answered: no, and
   not for a counting reason that more boxes would fix. It stays evaluation-only.
2. **Both upper reliable bins fall short of the 20–30 effective-host bar** at
   this geometry: 12.9 and 2.4 hosts per ensemble, with `3.16e13` empty in 11.5%
   of ensembles and a ±37% occupation interval. Gate B's requirement that an
   improvement include `1e13` or `3.16e13` cannot be evaluated reliably as
   configured.
3. **Attribution retention falls monotonically with host mass** — 0.37, 0.30,
   0.22, 0.12, 0.00. This is not noise, it is a *mass-dependent selection*, and
   it is structural: a cluster accretes from a large Lagrangian volume, so the
   more massive the host the more certainly its `Rvir` neighbourhood spans a
   chunk boundary and fails the purity test. The core mask preferentially
   deletes exactly the hosts the occupation reward is about.

The geometry sweep separates the cause from the cure (retention per host bin,
`set12`):

| `chunk_hr` | Mpc/h | chunks/box | `min_purity` | `1e12` | `3.16e12` | `1e13` | `3.16e13` | `1e14` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 25 | 64 | 0.8 | 0.37 | 0.30 | 0.22 | 0.12 | 0.00 |
| 128 | 25 | 64 | 0.5 | 0.43 | 0.36 | 0.32 | 0.16 | 0.00 |
| **256** | **50** | **8** | **0.8** | **0.62** | **0.55** | **0.52** | **0.40** | **0.19** |
| 256 | 50 | 8 | 0.5 | 0.67 | 0.60 | 0.56 | 0.43 | 0.19 |
| 256 | 50 | 8 | 0.8 (`max_half_width=8`) | 0.62 | 0.55 | 0.52 | 0.40 | 0.19 |

* **Chunk size is the lever; purity is not.** Doubling `chunk_hr` roughly
  triples retention in the upper bins and takes `1e14` from 0.00 to 0.19.
  Loosening `min_purity` from 0.8 to 0.5 buys ~5 points and costs attribution
  cleanliness — the wrong trade.
* **`max_half_width` is irrelevant** (8 gives the identical column to 4), which
  confirms the rejection is the purity test rather than the width cap. The
  `chunk_hr = 128` justification in §1 — "wide enough to contain the Lagrangian
  patch of a `1e14` host (13.5 Mpc/h)" — was necessary but nowhere near
  sufficient: containing the collapse sphere does not make the surrounding
  Eulerian neighbourhood chunk-pure.
* **The cost is credit resolution.** At `chunk_hr = 256` a box holds 8 chunks, so
  each leave-one-out marginal contribution covers 1/8 of a box instead of 1/64.
  Replay credit gets much coarser. That is a real loss, and it is the right
  trade: a reward whose target bins are empty is not made better by attributing
  it precisely. It does mean `replay.py`'s weight-ratio abort matters more.

**No duplication.** The audit verifies rather than assumes it: chunks tile the
box disjointly, `assign_halos_to_chunks` returns at most one id per halo, and
the per-chunk host counts sum to exactly the number of assigned in-range hosts
(1017 = 1017, `duplicated: 0`). Overlapping-chunk double counting is not a risk
in this design.

#### Confirmed over all 16 boxes

`hr_catalog_summaries_cpu` has since run (array 22997, 16/16 tasks, ~31 min), so
the audit now uses a real **box** bootstrap rather than a chunk bootstrap inside
one box. At `chunk_hr = 128`, `B = 16`, 4000 draws:

| host bin | hosts /box | per `B=16` ens. | p10 | empty ens. | occ. | occ. CI95 rel. width | usable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `1e12` | 760.6 | 190.6 | 163 | 0% | 4.50 | 0.19 | yes |
| `3.16e12` | 227.8 | 56.9 | 45 | 0% | 12.69 | 0.31 | yes |
| `1e13` | 55.1 | 13.8 | 9 | 0% | 34.89 | 0.50 | **no** |
| `3.16e13` | 8.8 | 2.2 | 0 | 11.7% | 86.08 | **1.49** | **no** |
| `1e14` | 0.1 | 0.0 | 0 | 98.7% | — | — | **no** |

The single-box estimate was optimistic by exactly the factor predicted: the
`3.16e13` occupation interval widens from 0.74 to **1.49** once boxes rather
than chunks are resampled. `set12` was also a lucky box for the top bin — over
16 boxes only ~2 hosts above `1e14` survive attribution *in total*, i.e. 0.1 per
box, so that bin is empty in 98.7% of ensembles rather than merely thin.

**The conclusion is now measured rather than provisional: at `chunk_hr = 128`
neither upper reliable bin is usable, so Gate B's occupation criterion cannot be
evaluated at this geometry.**

#### Decision: `chunk_hr = 256`, `B = 8`

The same 16-box audit at the wider chunk, `B = 8` (8 chunks/box, so one ensemble
is exactly one box — the independent cosmological unit the bootstrap already
resamples; `B = 16` would silently span two):

| host bin | 128 / `B=16` | 256 / `B=8` | verdict |
| --- | --- | --- | --- |
| `1e12` | 190.6 hosts, CI 0.19 | 1285.9, CI 0.073 | usable |
| `3.16e12` | 56.9, CI 0.31 | 427.0, CI 0.123 | usable |
| `1e13` | 13.8, CI 0.50 | 130.1, CI 0.161 | **now usable** |
| `3.16e13` | 2.2, CI 1.49, 11.7% empty | 30.0, p10 = 22, **0% empty**, CI 0.331 | **now usable** |
| `1e14` | 0.0, 98.7% empty | 4.0, CI 1.02 | still evaluation-only |

All four reliable bins clear the 20–30 effective-host bar, including both upper
ones, which is what makes Gate B's occupation criterion evaluable at all. `1e14`
rises from empty to 4 hosts per ensemble — better, still under the bar, still
evaluation-only, and consistent with the covariance verdict below.

`configs/reward/reward.yaml` now carries `chunk_hr: 256` and
`ensemble_size_B: 8`, with this table inline as the justification.

Because `catalog_summaries.py` reuses an existing Rockstar ASCII unless
`--overwrite`, regenerating the summaries at the new geometry **does not re-run
the halo finder** — it is minutes per box, not half an hour.

### Covariance audit — run over 16 boxes, and it disqualifies the `1e14` bin

`scripts/reward/audit_reward_covariance.py` reports `cond(C_reg)` before and
after shrinkage, the eigenvalue spectrum with each mode's dominant bins,
leave-one-box-out stability of `mu` / the spectrum / the reward *ranking*
(Spearman over a fixed reference set), the per-bin contributions to the frozen
baseline's `D^2`, and all of it with and without the sparse `1e14` bin.

It **refuses to run on SR2 seed scatter**: seed scatter is the generator's own
noise, not cosmic variance, and §2 already measured what it produces — a
Mahalanobis distance of 98007 with 88% of it in three bins. The script therefore
requires `source == "hr"` summaries and exits if it finds anything else.

Result over all 16 boxes:

| variant | `cond(C_reg)` | raw `cond(C)` | max single-bin share of baseline `D²` | LOBO min Spearman | max `mu` shift | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| with `1e14` | 94.1 | 7001 | **0.963** | 0.984 | 0.23σ | **fail** |
| without `1e14` | 89.9 | 6976 | 0.470 | 0.983 | 0.12σ | **pass** |

**With the `1e14` bin included, 96.3% of the frozen baseline's Mahalanobis
distance comes from that single bin** — the bin that is empty in 98.7% of
ensembles. That is the "one-bin objective wearing an 11-bin costume" risk from
the §1 shrinkage discussion, confirmed as a measurement rather than a worry. It
also drives one abundance bin (`nsub@3.98e10`) to a *negative* contribution
through the off-diagonals.

Dropping it leaves a balanced objective: `R_cat = -4.8` decomposing into
`R_occ = -3.4` and `R_abund = -1.3`, top contributions `occ@1e13` 0.47,
`occ@3.16e12` 0.24, `nsub@1.26e10` 0.22, and no negative bins. The script's
recommendation is `drop_sparse_bin_to_evaluation_only`.

The estimator itself is healthy and that is worth stating separately: `cond ~90`
after shrinkage, and leave-one-box-out changes the reward *ranking* by a Spearman
of only 0.983 with `mu` moving at most 0.12σ. The covariance was never the
problem — a bin with no data was.

One caveat carried forward: even after dropping `1e14`, the top eigenmode is
dominated by `occ@3.16e13`, which the host-count audit rates unusable at
`chunk_hr = 128`. The two audits agree that the geometry, not the reward, is
what needs to change.

This audit is wired into `audit_reward_support_cpu.sbatch` immediately after the
array job and **before** `fit_reward_model`, so that an unusable bin is dropped
before the reward is fitted rather than apologised for afterwards.

### GPU smoke tests — for the user to run, not done here

```bash
sbatch scripts/slurm/smoke_residual_prior_gpu.sbatch
sbatch scripts/slurm/smoke_reward_distill_gpu.sbatch
```

---

## 3. Runs, in dependency order

All scripts take arguments as `VAR=value` after the script name. Large outputs go
to `$ZFS/DMSR/dmsr_reward/`; the repo keeps only configs, code and small JSON.

Stages 1–3 (GPU) and stage 4 (CPU) are **independent siblings** — submit the HR
catalog array immediately, do not wait for the prior to train. It is the long
pole and it gates the two audits that decide the reward's shape.

```bash
# --- Stage 1-2: frozen baseline and the paired residual targets ---------------
sbatch scripts/slurm/cache_sr2_base.sbatch
sbatch scripts/slurm/audit_residual_targets_cpu.sbatch     # writes sigma_res.json

# --- Stage 3: supervised residual prior ------------------- GATE A ------------
sbatch scripts/slurm/smoke_residual_prior_gpu.sbatch
sbatch scripts/slurm/train_residual_prior.sbatch

# --- Stage 4-5: reward target and calibrated constraints ----------------------
sbatch --array=0-15 scripts/slurm/hr_catalog_summaries_cpu.sbatch
sbatch scripts/slurm/audit_reward_support_cpu.sbatch    # host counts + covariance
sbatch scripts/slurm/fit_reward_model_cpu.sbatch        # only after reading both

# --- Stage 6: best-of-K support oracle -------------------- GATE B ------------
sbatch scripts/slurm/sample_reward_oracle.sbatch RUN_NAME=prior_k8
JID=$(sbatch --parsable scripts/slurm/catalog_reward_oracle_cpu.sbatch RUN_NAME=prior_k8)
sbatch --dependency=afterok:$JID --array=0-0 \
    scripts/slurm/catalog_reward_oracle_cpu.sbatch RUN_NAME=prior_k8 AGGREGATE=1

# --- Stage 7-8: elite replay and one distillation round ----------------------
sbatch scripts/slurm/build_replay_cpu.sbatch RUN_NAME=prior_k8 ROUND=round_000
sbatch scripts/slurm/smoke_reward_distill_gpu.sbatch
sbatch scripts/slurm/train_reward_distill_round0.sbatch

# --- Stage 9: evaluate average samples ---------------------------------------
sbatch scripts/slurm/generate_reward_eval_boxes.sbatch EVAL_RUN=final
JID=$(sbatch --parsable scripts/slurm/eval_catalogs_cpu.sbatch EVAL_RUN=final)
sbatch --dependency=afterok:$JID --array=0-0 \
    scripts/slurm/eval_catalogs_cpu.sbatch EVAL_RUN=final AGGREGATE=1
JID=$(sbatch --parsable scripts/slurm/eval_full_metrics_cpu.sbatch EVAL_RUN=final)
sbatch --dependency=afterok:$JID --array=0-0 \
    scripts/slurm/eval_full_metrics_cpu.sbatch EVAL_RUN=final AGGREGATE=1
```

Notes on the ordering:

* **Stop at Gate A.** `train_residual_prior` must show the prior reproduces the
  paired residual distribution with nonzero diversity and without wrecking low-k
  displacement or density power. There is no point sampling an oracle from a prior
  that cannot generate plausible residuals. Criteria in §3a.
* **Stop at Gate B.** `oracle_report.py` writes `gate_b.json`. If occupation does
  not improve, do not build a replay buffer — but do **not** conclude the residual
  action space is incapable either; go to the escalation ladder in §5.
* **`audit_reward_support_cpu` comes before `fit_reward_model`, not after.** It
  decides which host bins the reward may use and, on the `set12` evidence, whether
  `chunk_hr` should be 128 or 256. Fitting first and reinterpreting later is how a
  reward ends up with an empty target bin.
* Array-job aggregation is always a **second submission with a dependency**, never
  task 0 polling `squeue`. A partial Gate B verdict is worse than none.
* `catalog_reward_oracle_cpu` is the long pole: Rockstar on a 512³ box is ~10–20
  min plus a 3.7 GB GADGET2 dump that is deleted immediately after parsing. With
  `K=8` over 4 validation boxes that is 32 candidates plus baselines, sharded 8
  ways.

### Cost expectations, so a wrong number is recognisable

| Stage | Expected | Wrong if |
| --- | --- | --- |
| `cache_sr2_base` | ~20 min/box, 3.2 GB each | > 1 h/box means the frozen generator is re-tiling badly |
| `train_residual_prior` | 1–2 days, loss falling within the first 1k steps | flat loss at step 1k means the whitening or `sigma_res` is wrong |
| `sample_reward_oracle` | 5.4× an untiled pass per sample × 20 DDIM steps | a margin error would have hard-exited at startup |
| `catalog_reward_oracle_cpu` | ~20 min per candidate, 8 shards in parallel | > 1 h/candidate means Rockstar is thrashing on ZFS, not compute-bound |
| `audit_reward_support_cpu` | ~1 h, arithmetic on cached summaries | hours means it is re-attributing halos, not reading the cache |
| `build_replay_cpu` | minutes (arithmetic on cached summaries) | hours means the halo finder is being re-run — a bug |
| `hr_catalog_summaries_cpu` | ~30–60 min/box (Rockstar + attribution), 16 boxes in parallel | > 2 h/box means the GADGET2 dump is not being deleted |

Storage, all under `$ZFS/DMSR/dmsr_reward/` — nothing in home:

| Artifact | Size | Note |
| --- | --- | --- |
| SR2 base cache | 3.2 GB × 16 boxes ≈ **51 GB** | shared by every downstream stage |
| residual candidates | 3.2 GB each (1.6 GB at `float16`) | `K=8` × 4 boxes ≈ **102 GB** per oracle run; use `DTYPE=float16` if tight |
| HR chunk summaries | a few MB total | JSONL, per box |
| GADGET2 dumps | 3.7 GB transient | deleted immediately after parsing; if they accumulate, that is the bug |
| catalogs | ~100 MB/box | ASCII |
| checkpoints | ~1–2 GB per run | |

The residual candidates dominate, and they are the thing to delete first once
their summaries are cached — the reward only ever reads the JSONL afterwards.

---

## 3a. Gate A — is the residual prior any good?

Gate A is about the *prior*, not the reward. It asks only whether the model can
generate plausible residuals at all; nothing downstream is meaningful if it
cannot. All eight must hold:

| # | Criterion | Fails if |
| --- | --- | --- |
| 1 | Validation denoising loss falls | flat at step 1k → whitening or `sigma_res` is wrong |
| 2 | Residual diversity is not collapsed | spread/RMS below `diversity_min` → deterministic sampler, and best-of-K has nothing to select from |
| 3 | Residual RMS is plausible against the Stage-2 paired audit | orders of magnitude off → `residual_scale` or whitening |
| 4 | Residual power spectrum is plausible | power concentrated at the Nyquist scale → the model is emitting noise |
| 5 | `x0_clip` fraction is small, in standardized residual units | a large fraction means the clip, not the model, sets the amplitude (see §1) |
| 6 | No seam or tile-size dependence | re-run at two tile sizes; a difference means the margin or the normalisation is wrong |
| 7 | No systematic low-`k` or LR-consistency damage | measured explicitly, because the receptive field does **not** guarantee it (§1) |
| 8 | Density statistics physically plausible (PDF, `sigma`, power) | a good denoising loss with a broken PDF means the residual is not a field |

A Gate A failure is a **prior/sampling bug**, and the fix is in the prior, the
whitening, the sampler, or `residual_scale`. Do not evaluate catalog rewards on
a prior that fails Gate A — the reward numbers would be measuring the bug.

**This is enforced, not advisory.** `scripts/reward/gate_a_check.py` reads the
run's `metrics.csv` and writes `gate_a.json` with a per-criterion verdict;
`train_residual_prior.sbatch` runs it, and `sample_reward_oracle.sbatch` refuses
to sample (exit 0, with the failing criteria printed) unless `passed` is true.
Criteria 6 needs a second sampling job at two tile sizes and is reported
`not_evaluated` until one is supplied. `IGNORE_GATE_A=1` overrides, and makes
every downstream number conditional on an unchecked prior.

### Training and inference must see the same neighbourhood

The model pads circularly. On a bare `crop_hr` crop, every voxel whose receptive
field is wider than `crop_hr / 2` sees the crop wrapped onto itself — an
artificial neighbourhood that exists nowhere in the box. At the configured
`levels = 2 / blocks = 2` the measured half-width is **41 HR cells**, so a 64³
crop contaminates *every* voxel, while full-box sampling supplies 48 cells of
real context (`TILE_MARGIN`).

So `data.crop_hr` is the **scored core** and `data.context_margin` (48, the same
as `TILE_MARGIN`) is the real neighbourhood around it: the forward pass runs on
160³, the loss is taken on the central 64³, and validation carries the same
margin so the diagnostics cannot hide the discrepancy they exist to detect. The
cost is the reason `batch_size` is 1 with `grad_accum` 8. If that does not fit,
the lever is the **receptive field**, not the margin — a smaller margin does not
make the wraparound smaller, only invisible.

---

## 4. Gate B — can occupation be controlled at all?

For each validation conditioning input: sample `K` residual candidates, compose
each with the frozen SR2 field, assemble **complete periodic boxes**, and run
Rockstar on the complete boxes. Evaluate curves over `K = 1, 4, 8, 16` (extend to
64 only if the curve is still moving), over residual amplitude, over sampling
temperature/stochasticity (`churn`), and over multiple independent validation
boxes.

Report separately, never merged:

* the **average** candidate;
* a **randomly selected** candidate — this is the honest baseline, because it is
  the sample you would have taken anyway; the mean of `K` is a strictly easier
  target;
* the **best joint-reward** candidate (`R_cat`);
* the **best occupation-reward** candidate (`R_occ`);
* abundance **and every occupation bin** individually;
* all field constraints;
* catalog diversity.

### One group is one box

An **ensemble group is a complete box**: at `chunk_hr = 256` a box is exactly 8
chunks and the reward is fitted at `B = 8`, so a group is the independent
cosmological unit, the unit the bootstrap resamples, and the unit `C_reg`
describes. The baseline, the HR reference and every candidate of that box are
scored on **identical chunk ids**, which is what makes the comparison paired.
"Reproduced on ≥ 2 groups" therefore means "reproduced on ≥ 2 boxes".

Groups must not be drawn *across* boxes. A candidate field exists for one box
only, so a cross-box group of `B` chunks leaves each candidate scored on the
one or two chunks that happen to be its own — a short ensemble compared against
a full-length baseline, through a covariance fitted for `B`.

### Pass conditions

All six:

1. occupation improves in **at least two reliable bins** (0–3);
2. the improvement **includes `1e13` or `3.16e13`** (bins 2, 3);
3. it **beats a random draw by at least the gate target (20%)** — a margin, not
   a sign. Best-of-`K` beats the median of `K` essentially by construction, so
   a `> 0` test measures only that sampling noise exists;
4. it does **not rely on the sparse `1e14` bin** — which the §2 audit shows is
   empty at `chunk_hr = 128` and still under the bar (4 hosts/ensemble) at 256.
   `include_sparse_in_reward: false` now removes it from `R_cat` itself, not
   only from the criterion;
5. field constraints hold and object-level diversity does not collapse;
6. the constraints were **calibrated** (`constraints.calibrated: true`). With
   placeholder thresholds the feasibility filter — and therefore the elite set —
   is made of guesses, so `oracle_report.py` reports
   `blocked_uncalibrated_constraints` instead of a verdict.

Diversity is measured **twice**, and both matter. `diversity` (the feasibility
floor) is field-level: RMS spread across residual samples. `catalog_diversity`
is the spread of the occupation curve across candidates. SR2's documented
failure is precisely that the field varies while the subhalos do not, so the
field floor can be comfortably satisfied while best-of-`K` has nothing to select
from.

Field fidelity is a **feasibility filter**, not a term traded against catalog
reward. An infeasible candidate cannot be an elite at any reward.

`oracle_report.py` writes these as `gate_b.json` with three possible verdicts:
`support_present_occupation`, `abundance_only_improvement` (explicitly labelled,
never counted as a pass), and `support_absent`.

### What a Gate B failure does and does not mean

**Gate B is not a final no-go test.** A negative result shows exactly one thing:
that *ordinary samples from the current prior* do not contain accessible good
candidates. It does **not** show that the residual action space is incapable, and
it does not show that search or RL could not find such candidates. Best-of-`K`
over `K ≤ 16` draws from one prior is a very weak search. Four different
questions get conflated if this is not kept straight:

| Question | Answered by |
| --- | --- |
| Do good residuals appear by chance? | raw best-of-`K` |
| Do good residuals exist in the action space? | directed search (CEM) |
| Can the model represent them at all? | fixed-host reward overfitting |
| Is the failure the renderer or the conditioning? | true-catalog oracle |

---

## 5. If raw Gate B fails — the escalation ladder

Work down this ladder in order. Each rung answers a different question, and only
the last rung licenses the conclusion "the residual action space is
insufficient."

1. **Sweep `K`, temperature and residual scale** within the *calibrated* field
   constraints. Cheapest, and a flat reward-vs-`K` curve is itself diagnostic.
   Do not widen a constraint to make candidates pass.
2. **CEM / evolutionary search over the diffusion noise**, on a small fixed set
   of conditioning inputs. CEM = sample noise, keep the best few by reward,
   resample near them, repeat. This asks whether good residuals *exist*, not
   whether they are *likely*.
3. **Small reward-overfitting experiment on fixed massive hosts.** Deliberately
   overfit a handful of hosts. If reward cannot be raised even here, the problem
   is representational, not statistical.
4. **Distil or run RL — only if search has demonstrated reward variation.**
   Distil the discovered residuals first; policy gradients are the last resort,
   not the first.
5. **Larger-context or host-conditioned model**, if search succeeds locally but
   nothing generalises.
6. **True-HR-catalog oracle renderer** (the design in `Problem_writeup.txt`,
   which is not in this repository — it will need to be supplied before this
   rung can be built): feed the real HR catalog and ask whether the field can be
   made to realise *known* subhalos. This separates "the renderer / action
   representation cannot do it" from "the conditioning does not carry the
   information."

### Implementation status of the ladder

Only rung 1 exists today. The rest are described here so the decision logic is
fixed in advance, not so they can be submitted:

| Rung | Status |
| --- | --- |
| 1. sweep `K` / temperature / residual scale | **implemented** — `sample_reward_oracle.sbatch` already takes `K`, `RESIDUAL_SCALE`, `N_STEPS`, and `churn` via the diffusion config; each setting is a separate `RUN_NAME` |
| 2. CEM over diffusion noise | not implemented |
| 3. fixed-host reward overfitting | not implemented |
| 4. distillation / DDPO | distillation implemented; DDPO not |
| 5. larger-context / host-conditioned model | not implemented |
| 6. true-HR-catalog oracle renderer | not implemented, and its design document is not in this repository |

Rung 1 is a set of parallel submissions, one `RUN_NAME` per setting, each
followed by its own scoring and aggregation pair:

```bash
for s in 0.5 1.0 2.0; do
  sbatch scripts/slurm/sample_reward_oracle.sbatch \
      RUN_NAME=prior_k16_a$s K=16 RESIDUAL_SCALE=$s SPLIT=val
done
```

Score each `RUN_NAME` with the same `catalog_reward_oracle_cpu` +
`AGGREGATE=1` pair as in §3. A reward that is flat across all of these is itself
evidence, and it is the evidence rung 2 is designed to test.

### Interpretation

| Result | Next step |
| --- | --- |
| Raw samples contain rare successful candidates | Reward-weighted replay / distillation |
| CEM/search succeeds but raw sampling does not | Distil search elites, or try diffusion RL |
| Fixed-host reward overfitting succeeds | Model is controllable; study generalization |
| Only abundance improves | Add host-aware conditioning / context |
| True-catalog oracle succeeds | Pivot toward joint catalog–field generation |
| Even the catalog oracle fails | Renderer / action representation is insufficient |

### On policy-gradient RL

**Do not implement DDPG/DDPK-style continuous control.** This is a
high-dimensional *terminal-reward generation* problem, not a natural continuous
control setting: there is no per-step reward, the "action" is an entire field,
and the critic would have to model a Rockstar run. If policy gradients become
justified at rung 4, use a **stochastic diffusion-policy objective —
DDPO-style likelihood-ratio optimization** — begin with a small trainable
parameter subset (a LoRA or the output blocks, not the whole network), and
compare it head-to-head against CEM distillation. If CEM distillation matches it,
prefer CEM distillation: it is cheaper and it does not require the reward to be
queried inside the training loop.

---

## 6. Reward distillation

If and only if Gate B passes. Replay is built from **ensemble-level** reward and
**leave-one-out candidate contribution**.

### Gate B decides *whether*; a second oracle run decides *what from*

Gate B samples the **validation** split — that is what makes its verdict a
held-out statement. An offline replay buffer harvested from that same run would
turn validation data into training data, and `build_replay.py` refuses it.

So `submit_oracle.sh gate_c` runs its **own** oracle on the **training** boxes
(`REPLAY_RUN`, default `prior_train_k8`) and harvests the elites from there,
reading `gate_b.json` from the validation run (`GATE_B_RUN`) only for the
verdict. Chaining replay straight off Gate B's `run_name` could only ever end in
a hard refusal with the distillation job stranded behind it.

The reward is **not** another model input. Training remains

```
eps_phi( u_t , t , y_LR , Psi_base )
```

unchanged; what the reward does is **weight** denoising examples — a chunk whose
generated residual improved the ensemble reward contributes more to the loss.
The model never sees a reward value, so there is nothing to condition on and
nothing to game at inference.

Elite selection requires a **positive occupation contribution**, not merely a
positive joint reward — the same reason Gate B is decided on `R_occ`. Both the
elite cut and the leave-one-out credit are therefore taken in `R_occ`
(`oracle_report.py --credit-score`, default `R_occ`); the `R_cat` figures are
recorded alongside for diagnosis and never used as the training signal. The
elite's own `residual_scale` is applied to the residual the trainer sees, since
what earned the reward is `Psi_base + a·dPsi`, and the base field is resolved
from the entry's recorded `base_id`/`base_seed`, not from the config. Weights
stay bounded and mean-normalised (`w = exp(clip(A, 0, A_max)/tau)`, `w_max = 10`),
and the run **aborts** if a few chunks dominate: `max(w)/mean(w)` above its bound
means the leave-one-out credit assumption has failed and the buffer is not worth
training on.

---

## 7. Final comparisons

Seven arms, at minimum:

| # | Arm | Isolates |
| --- | --- | --- |
| 1 | frozen SR2 | the baseline |
| 2 | supervised residual prior, `K = 1` | does a residual prior alone help? |
| 3 | supervised prior, best-of-`K` | is there support to select from? |
| 4 | continued supervised training, **same compute** | is the gain reward alignment or just more compute? |
| 5 | abundance-only reward | does optimising the secondary target move the primary one? |
| 6 | occupation-primary reward | the intended model |
| 7 | reward-distilled model, `K = 1` | did the alignment transfer to the *average* sample? |

Arm 4 is the one that makes the claim falsifiable, and arm 5 is the one that
tests whether the occupation framing was necessary.

**Primary (catalog) evaluation:** occupation vs host mass; slope and ratios
relative to HR; subhalo abundance; radial occupation within hosts; one-halo
correlation; merged/missing/recovered failure classes; relative subhalo
velocities / phase-space coherence; catalog-level diversity.

**Field evaluation:** displacement and density power; cross-correlation and
transfer functions; density PDF; LR and low-`k` consistency; bispectrum if
already affordable.

**Statistics.** Confidence intervals bootstrapped over **independent boxes**,
with seeds nested inside boxes. Chunks are not independent cosmological
realisations, and multiple SR2 seeds of one box are not independent either — a
point the §2 audit had to make twice.

**Splits.** `set12` is **development/validation** for this project: the SR2
subhalo study and the reward sanity check have both already been run on it, and
a box that has been looked at repeatedly cannot carry a held-out claim.
`set13`, `set14`, `set15` are untouched and are reserved for the final numbers —
do not open them before the final comparison.

---

## 8. Intended paper claim

Claim success **only if the average reward-aligned model — not best-of-`K`** —
restores the increasing occupation–host-mass relationship while preserving
stochastic field quality. Best-of-`K` is a support check; it is not a model.

> Standard field losses can match marginal density statistics while collapsing
> conditional populations of coherent objects. We align a pretrained stochastic
> field generator using covariance-aware ensemble rewards from a
> non-differentiable scientific catalog pipeline.

The distinction from SR4: inference does not receive the missing high-resolution
initial modes, the output remains genuinely stochastic, and Rockstar outcomes are
used for **optimization** rather than evaluation alone.

---

## 9. Decision table

The rule, to be applied to `gate_b.json` and the Stage-9 tables.

### Gate A

| Verdict | Condition | Next |
| --- | --- | --- |
| **Pass** | all eight §3a criteria hold | proceed to the reward stages |
| **Fail** | any of the eight fails | fix the prior, whitening, sampler or `residual_scale`. Do **not** evaluate catalog rewards — the numbers would be measuring the bug |

### Gate B

| Verdict | Condition | Next |
| --- | --- | --- |
| **Occupation support present** | ≥ 2 reliable occupation bins improve, including `1e13` or `3.16e13`; beats a random draw by ≥ the gate target (20%) under box-level uncertainty; not carried by `1e14`; feasible; diversity intact; constraints calibrated | run the oracle again on the **train** boxes, build the replay buffer from *that* run, and distil |
| **Blocked: uncalibrated constraints** | `constraints.calibrated: false` | run `calibrate_constraints.py`, paste the block in, set the flag, re-aggregate. No verdict until then |
| **Abundance-only improvement** | `R_abund` improves, occupation criterion not met (even if `R_cat` improved) | a **scientifically informative failure**, reported as such. Ladder rung 5: host-aware conditioning / context. Not a pass |
| **Support absent in raw sampling** | best-of-`K` does not improve occupation | **not** a no-go. Work the §5 ladder from rung 1. Only rung 6 licenses "the action space is insufficient" |

Gate B's provisional quantitative bar: ≥ 20–25% reduction in the **occupation**
discrepancy `D²_occ` for best-of-`K` versus a random draw, no constraint
violation, reproduced on more than one ensemble group. This is a go/no-go
heuristic and **not** a statistical claim — `K = 8` gives 8 draws per group, so a
best-of-`K` maximum is a badly biased estimator of anything. It checks that
useful samples exist at all.

### After distillation

| Verdict | Condition |
| --- | --- |
| **Distillation failed** | Gate B passed, but the distilled model's *average* sample does not move occupation toward the best-of-`K` samples |
| **Reward success but suspicious** | occupation improves while held-out catalog metrics or field fidelity get worse. Treat as reward hacking, not a win |
| **Promising success** | average distilled samples improve occupation *and* abundance, at least one **unrewarded** statistic also improves (one-halo correlation is the informative one), field metrics stay inside the calibrated constraints, and diversity does not collapse |

**Current verdict: support unknown, and one prerequisite is now measured to be
unmet.** No GPU training stage has run, so Gate A and Gate B are both open.

What is no longer open: the HR catalogs exist for all 16 boxes and both §2 audits
have run against them. They agree, from two independent directions, that **the
chunk geometry — not the reward — is what blocks Gate B**:

* the `1e14` host bin is disqualified outright (0.1 hosts/box, empty in 98.7% of
  ensembles, 96.3% of the baseline `D²` if included) and is now evaluation-only
  in the config;
* at `chunk_hr = 128` the two *upper reliable* bins also failed the
  effective-host bar (13.8 and 2.2 hosts per ensemble), so the Gate B criterion
  "improvement must include `1e13` or `3.16e13`" could not be evaluated —
  **resolved** by moving to `chunk_hr = 256`, `B = 8`, where all four reliable
  bins clear it;
* the covariance estimator itself is sound (`cond ~90`, LOBO Spearman 0.983), so
  nothing here argues for changing the reward's form.

One prediction still to confirm: the covariance audit passed at `chunk_hr = 128`.
It should hold or improve at 256 — its top eigenmode was dominated by
`occ@3.16e13`, the bin that just went from 2.2 to 30 hosts — but that is
reasoning, not measurement, so `audit_reward_support_cpu` re-runs at the new
geometry **before** `fit_reward_model`.

### Three risks worth stating in advance

1. **Covariance calibration.** Demonstrated, not hypothetical: the seed-scatter
   covariance gives a 98007 Mahalanobis distance with 88% of it in three bins. If
   `fit_reward_model`'s HR-ensemble `C` also yields a huge distance with a poor
   `cond(C_reg)`, the reward is a one-bin objective wearing an 11-bin costume.
   Check `cond(C_reg)` and the per-bin contributions in the manifest before
   trusting any Gate B number.
2. **The target is harder than "low-mass deficit" implies.** The measured
   failure is a flat halo-occupation function, not a resolution-floor shortfall.
   A residual that adds small-scale power may raise the lowest abundance bin
   without fixing occupation in rich hosts, since that requires substructure to
   *survive inside* a deep potential. Abundance improving while
   `occ@3.16e13` does not is the **expected** informative failure mode, which is
   why it has its own verdict label and its own ladder rung — it points at
   residual scale and context, not at the reward.
3. **Leave-one-out credit.** `A_{k,i}` ignores chunk interactions. If elite
   selection is dominated by a handful of chunks with extreme weights, that is the
   assumption failing; the weight-ratio abort catches it during distillation rather
   than after. Note this risk *grows* if `chunk_hr` moves to 256: 8 chunks per box
   makes each marginal contribution coarser and each weight more influential.
4. **Chunk geometry is now a live variable, not a settled one.** §2 shows the
   core mask deletes hosts in a mass-dependent way — the very hosts the reward
   targets. `chunk_hr = 256` fixes the counts at the cost of credit resolution.
   Neither setting is obviously right, and the choice must be made from the
   16-box audit rather than from `set12` alone.
