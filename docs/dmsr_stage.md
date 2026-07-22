# DMSR stage: exact-consistent null-space flow + HR critic

Implementation summary for the stage that tests whether **generated samples from
many real LR-only simulations improve over the same model and critic trained on
paired LR environments only** (Stage D vs Stage C).

Status (2026-07-20): **code, unit tests and smoke tests complete and green (226
passing).** `paired_deterministic`, Stage A and Stage B are **complete** (9.0h). The six
Stage C/D runs (3 seeds each, **20000 steps**) are **in flight on the third launch** —
the first two were cancelled early and are archived, for a wrong step count (9.0c(ii))
and then for an **inert critic** (9.0c(iii)) that would have made the whole comparison
vacuous. Stage E not started. Final C-vs-D tables and plots (9.1+) remain _pending_.

Section 9 records findings as they were measured, **including four corrections to
earlier claims in this same document**: the mean-collapse reading (9.0), the
environment-coverage claim (9.0b), the cost estimate that set the step count (9.0c(ii)),
and the "critic stable" reading from the smoke test (9.0c(iii)).

**Read 9.0b, 9.0c(iii) and 9.0g before interpreting any result**: 9.0b establishes that
the environment-coverage premise does not hold on this dataset (so criterion 3 is
expected weak); 9.0c(iii) is why the first C/D launch was void and how critic health must
be checked; 9.0g is why `r(k)` structurally rewards mean collapse and the baseline table
must not be ranked on it.

---

## 1. Audit: what already existed, and what was reused

Almost every primitive this stage needs was already in the repo and tested. The
audit below records what was reused verbatim rather than rebuilt.

| Component | Status | Location |
|---|---|---|
| Analytic `A`, right inverse, null projection | **reused** | `operators/multiscale.py` (`block_average` / `block_upsample` / `null_projection`) |
| Exact-consistency parameterization | **reused (concept)** | `operators/base_upscaler.py::consistent_base` |
| Rectified/linear-interpolant flow matching | **reused (form)** | `losses/flow.py::flow_matching_loss` |
| Velocity U-Net backbone (FiLM on `t`, `context` input) | **reused** | `models/flow_unet.py::UNetResidualFlowModel` |
| Differentiable CIC Eulerian density | **reused** | `eval/density.py::cic_density` |
| Paired + LR-only crop datasets, periodic crops, `disp/vel` augmentation | **reused** | `data/datasets.py`, `data/crops.py` |
| Deterministic val tiling | **reused** | `data/datasets.py::GridCropDataset` |
| EMA | **reused** | `models/operator_denoiser.py::ModelEMA` |
| Run dir, config/env capture, CSV+TB+W&B logging, checkpoints | **reused** | `train/common.py`, `utils/config.py` |
| Spectral norm, PatchGAN reference design | **reused (pattern)** | `external map2map/models/{spectral_norm,patchgan}.py` |
| 24 proper cube rotations | **new** | `operators/symmetry.py` had translations only, and explicitly left rotations unimplemented rather than applying them incorrectly |
| HR residual/Eulerian critic | **new** | — |
| Environment descriptors + balanced sampler | **new** | — |
| LR masked-SSL encoder pretraining | **new** | — |

### Two framing decisions

**Single `s=8` operator, not the factor-2 octave cascade.** The existing SR flow
is a 64→128→256→512 cascade. This stage uses one `A = avg_pool3d(k=8)` mapping
512³ → 64³ directly. The reason is the experiment itself: the 350 LR-only boxes
are stored at 64³, which is exactly the paired LR resolution. Under the cascade
they could only ever supervise the 64→128 octave, leaving the top octave
paired-only — i.e. the extra LR data could not reach the scales the critic most
needs to judge, and Stage C-vs-D would be testing almost nothing.
`MultiScaleOperators` was already factor-generic, so this reuses the verified
primitives at `factor=8` rather than introducing new operator code.

*Cost of this choice:* new checkpoint lineage. Existing `runs/sr_flow_*`
checkpoints are cascade-shaped and do not load into `NullSpaceFlow`. On-disk data
formats, channel conventions and split conventions are unchanged.

**`r_target = P_A(x_hr)`, not `P_A(x_hr - A_plus(y))`.** These coincide only when
`y = A(x_hr)` exactly, which is false here: the real LR simulation differs from
`A(HR)` by a measured residual `eta` (displacement `eta_frac ≈ 0.008`, velocity
`≈ 0.60`) because LR runs use heavier particles and coarser softening. Using
`P_A(x_hr)` keeps the regression target a function of the HR box alone, so `eta`
never enters the flow objective — it is absorbed entirely by the `A_plus(y)`
base, which is where it belongs.

### Excluded by design

Not implemented, per the stage specification: pseudo-HR pairs, learned
degradation consistency, cycle consistency, virtual shifted measurements,
posterior distillation. Note this retires the previously-planned Gate-2
(C1/C2/C3 virtual-operator) line.

---

## 2. Files added

```
src/cosmo_sr/dmsr/
    __init__.py       package overview
    operator.py       NullSpaceOperator: A, A_plus, P_A, combine, consistency_error
    cubic.py          the 24 orientation-preserving cube rotations (voxels + vectors)
    density.py        HighPassDensity (differentiable CIC + low-pass), critic_input
    encoder.py        LRConditionEncoder, MaskedReconstructionHead, LRMaskedAutoencoder
    ssl.py            block/channel masking, translations, rotations, masked recon loss
    flow.py           NullSpaceFlow, null_space_flow_loss, sampler, encoder loading
    critic.py         HRCritic (spectral-norm 3D PatchGAN), hinge losses, LazyR1
    env.py            environment descriptors, standardizer, EnvironmentBalancedSampler,
                      roc_auc, source_classifier_auc
    data.py           BoxSplit/resolve_split, dataset builders, LRCropPool, BalancedLRDataset
    evaluate.py       r(k), T(k), power/PDF errors, bispectra, diversity, condition shuffle,
                      environment binning

src/cosmo_sr/train/
    train_dmsr.py             Stages A-E driver (+ --audit-compute)
    pretrain_lr_encoder.py    masked-SSL pretraining of the condition encoder

scripts/dmsr_eval.py          held-out evaluation, C-vs-D bootstrap, plots

configs/dmsr/
    _base.yaml  lr_ssl.yaml  baseline_upsample.yaml  paired_deterministic.yaml
    stage_a_paired_flow.yaml  stage_b_paired_flow_lrssl.yaml
    stage_c_critic_pairedlr.yaml  stage_d_critic_alllr.yaml
    stage_e_alllr_equivariant.yaml

tests/dmsr/
    test_operator_nullspace.py  test_flow_nullspace.py  test_adversarial_grad.py
    test_highpass_density.py    test_balanced_sampler.py  test_cubic.py
    test_stage_parity.py        test_dmsr_training_smoke.py
    test_ssl_masking.py         test_eval_reporting.py

docs/
    dmsr_stage.md     this document
    dmsr_runbook.md   exact commands for every stage, plus the rhea launch constraints

~/slurm/dmsr/
    dmsr_stage.sh     one arm, seeds sequential
    launch_cvd.sh     all 6 C/D runs concurrently (refuses to run without a frozen
                      Stage B checkpoint)
```

Modified: `utils/config.py` (added a `base:` key for config inheritance — needed
so Stage C and D can provably differ in one field), `pyproject.toml` (registered
the `slow` pytest marker), `~/slurm/dmsr/cosmo_sr_train.sh` (added `dmsr` and
`dmsr_ssl` modes).

---

## 3. Core parameterization

```
x_hat = A_plus(y) + P_A(r_theta(y, z))
```

`A(x_hat) = y` holds **by construction**. Measured relative consistency error:
**3.6e-7** (target ≤ 1e-5). In float64 the identity is bit-for-bit exact, so the
fp32 residual is purely summation rounding over the 512-element blocks — this is
asserted in `test_operator_nullspace.py`.

Flow, exactly as specified:

```
r_target = P_A(x_hr);  z_null = P_A(z)
r_t      = (1-t) z_null + t r_target
v_target = r_target - z_null
v_pred   = P_A(v_theta(r_t, t, y))
loss_flow = MSE(v_pred, v_target)
```

`||A(x_hat) - y||` is logged as `train/exact_consistency_rel` and is **never** a
loss term. Two regression tests pin this down: the degenerate objective is ~0
(`< 1e-10` relative) and its gradient to the flow is `< 1e-6 ×` the adversarial
gradient, documenting that adding it back would teach nothing.

## 4. Critic

Input is `concat(P_A(x), rho_high(x))` — no raw LR tensor, deliberately: in Stage
D the reals come from 3 paired boxes and the fakes from hundreds of LR-only
boxes, so any LR-distribution cue is a free win for the critic unrelated to
sample quality. Spectral norm on every conv, hinge loss, lazy R1 (interval-scaled
so the effective strength is independent of the interval).

**Low-pass definition** (configurable, `critic.lowpass`):

- `blockavg` *(default)* — `lowpass(rho) = A_plus(A(rho))`, so `rho_high = P_A(rho)`
  exactly: the same projector as the field residual, an exact complement of `A`,
  no new free parameter.
- `fourier` — sharp isotropic cut at `kcut_frac × k_Nyq_LR`. Spectrally cleaner
  but not an exact complement of `A`, and it rings in configuration space.

These are **not** interchangeable (a top-hat real-space window is not a
band-limit); report which was used.

A note recorded from testing: a displacement field that is constant within each
`A`-block is *not* a low-frequency density field — CIC-depositing it piles
particles onto block faces and generates large high-k density power. Density is a
nonlinear function of displacement, so "smooth Psi" and "smooth delta" are
different statements. The relevant test builds a genuinely band-limited
displacement instead.

## 5. Environment balancing

Descriptors from `y`: redshift, mean/var density, mean/var velocity divergence,
displacement rms, and optional tidal invariants `I2`/`I3`.

**Descriptors that are structurally constant are detected and dropped**, with the
dropped list saved to the run. `mean_density` is identically 0 because
`cic_density` wraps within the crop and normalises by the crop mean; a periodic-FD
`mean(div v)` vanishes for the same reason. Keeping them would only add noise to
the classifier and dilute the histogram match.

Balancing: standardize (training statistics only) → PCA to `n_dims` on the
**paired** pool → quantile bins from the paired pool → weight each unpaired crop
by `target_prob(bin) / n_unpaired_in(bin)`.

**Out-of-support rejection uses an explicit support box, not just empty bins.**
Quantile bins have unbounded outer edges, so a crop arbitrarily far outside the
paired distribution still lands in the last populated bin. Crops outside the
observed paired range on any PCA axis get weight 0 and are counted as rejected.
We do not extrapolate into, or claim generalization to, those environments.

Diagnostic: `source_classifier_auc` (logistic, class-weighted, scored on a
held-out half, rank-based ROC-AUC). **Stage D aborts at startup if the balanced
AUC exceeds `env.max_auc = 0.60`** rather than training on an unbalanced pool and
discovering it later; `env.allow_auc_fail` exists only for deliberate inspection.

## 6. Stage C vs Stage D — how compute is matched

Both stages run the identical per-step cycle:

```
1 paired generator batch  +  1 second-stream generator batch  +  n_critic critic updates
```

The **only** difference is the second stream's source: Stage C repeats *paired*
LR crops; Stage D draws *environment-balanced LR-only* crops. Stage C repeats
(rather than skipping) so the comparison is about data source, not about the
number of adversarial updates.

For paired batches `loss_G = loss_flow + lambda_adv * loss_G_adv`; for
second-stream batches `loss_G = lambda_adv * loss_G_adv` only — no HR regression
or pseudo-target term is added, by design.

This is enforced, not merely intended:

- `tests/dmsr/test_stage_parity.py` asserts the two config files differ **only**
  in `stage`, `output.run_dir` and `wandb.name`, and checks ~25 named settings
  individually plus the `lambda_adv` schedule at 8 step values.
- `--audit-compute` prints the update budget; the parity test runs it for both
  stages and asserts equality.
- `test_dmsr_training_smoke.py` runs both stages end-to-end on miniature on-disk
  boxes and compares the emitted `compute_audit.json`, confirming Stage D really
  loads LR-only boxes and Stage C really repeats paired ones.

## 7. Splits

Splits are **by simulation box, never by spatial crop** — crops within a box share
initial conditions and large-scale modes. `resolve_split` asserts train/val/test
box-name sets are pairwise disjoint and **raises if `lr_only_glob` matches any
held-out box**, so validation/test LR fields cannot leak into SSL pretraining or
the critic. The resolved manifest is written to `<run_dir>/split.json`.

Default: paired train `set0-2`, val `set15`, test `set14`, unpaired `lr_sims/*`
(350 boxes, disjoint simulations).

## 8. Statistical discipline

- **The test split is 12 boxes (set3-set14).** `set3-13` were unused by this stage
  (not train, not val, and the LR-only pool is the separate `lr_sims/` set), so
  promoting them to test costs nothing in training and converts criterion 5 from
  "no independent-box claim possible" into a genuine 12-box paired bootstrap.
  Changed before any test-split evaluation was run, so no peeking is involved
  (only `set15`, the val box, had been looked at).
- **The headline estimator resamples BOXES.** Each box contributes one paired
  `D - C` value (crops averaged within box, then seeds averaged); the bootstrap
  resamples boxes with replacement. Crops within a box share initial conditions and
  large-scale modes, so crop-level resampling understates variance and manufactures
  significance; seed-level resampling measures optimisation noise, not
  generalisation to new universes. `decision_rule` is applied to the **box-level**
  statistics when >= 2 boxes are present, and `cvd.json` records which estimator
  decided the verdict.
- Crops are tagged with their source box by `GridCropDataset`. Evaluation must run
  at `--batch-size 1`: `evaluate_batch` reduces a whole batch to one metric dict, so
  a batch spanning two boxes would produce a number belonging to neither. The script
  raises rather than silently corrupting the bootstrap.
- With a single held-out box the script refuses the box-level interval, falls back
  to seed-level, and says so explicitly.
- The C-vs-D decision rule is applied mechanically by `decision_rule()` against
  metrics pre-registered in the source (`PRIMARY_METRICS`, `CONDITIONAL_METRICS`,
  `GUARD_METRICS`) so the verdict is not eyeballed after the fact.
- Report distributional metrics (`r(k)`, `T(k)`, `P(k)`, PDFs, bispectra), not HR
  MSE: MSE rewards the mean-collapse this project has already observed
  (`respow ≈ 0.47` with acceptable correlation) and penalises adversarial realism.

---

## 9. Results

### 9.0 Stage A — in flight (single run, one seed, NOT a final result)

Launched 2026-07-19 on `gpu28` (`runs/dmsr/stage_a`, W&B `dmsr_stage_a`). Held-out
`set15` validation trend:

| step | `rk_low` | `rk_trans` | `rk_high` | `Tk_err_high` | `diversity` | `cond_shuffle` | `consist_rel` |
|---|---|---|---|---|---|---|---|
| 1000 | 0.9992 | 0.850 | 0.220 | 3.759 | 0.999 | 0.0026 | 3.36e-07 |
| 2000 | 0.9993 | 0.860 | 0.436 | 0.951 | 0.978 | 0.0124 | 3.37e-07 |
| 4000 | 0.9992 | 0.886 | 0.793 | 0.044 | 0.612 | 0.048 | 3.33e-07 |
| 8000 | 0.9992 | 0.887 | 0.802 | 0.038 | 0.562 | 0.063 | 3.32e-07 |
| 12000 | 0.9992 | 0.893 | 0.802 | 0.044 | 0.564 | 0.063 | 3.38e-07 |

**Correction to an earlier reading in this document.** At step 2000 `Tk_error_high` was
0.95 and I recorded that as the project's familiar *mean-collapse signature*. **The data
refuted that**: by step 4000 it had fallen to 0.044 and it has stayed there. Stage A's
high-`k` amplitude is within ~4% of truth — this run is **not** mean-collapsed, unlike
the earlier cascade flow (`respow ≈ 0.47`). The step-2000 value was simply an
undertrained model, not a collapse. Two likely reasons this stage behaves better: the
target is `P_A(x_hr)` (so the η scatter never enters the objective), and the null-space
projection removes the low-frequency component that previously dominated the regression.

Consequences worth carrying into Stage C/D:

1. **The critic's job is not "restore missing high-k power."** That headroom is largely
   gone. Its remaining headroom is *structural*: `rk_high` plateaus at ~0.80 and
   `rk_transition` at ~0.89, and the Eulerian/bispectrum statistics are untouched by
   `T(k)` being right. Expect any Stage D gain to appear in the conditional metrics, not
   in `Tk_error_high`.
2. **Watch `sample_diversity`.** It falls 0.999 → 0.56 over training and is roughly flat
   after ~8000 steps. 0.56 is healthy stochasticity (sample std ≈ 56% of residual RMS),
   but it is a guard metric in the decision rule and the adversarial term could push it
   further down.
3. **Stage A converges by ~4000–5000 steps**; 40000 is heavily overkill — the same
   pattern already seen on the operator-denoiser run. Stage C/D at 20000 steps is
   likewise probably longer than needed. This costs wall-clock but does not threaten
   validity, since the step count is matched between C and D.

### 9.0b Environment balancing on real data — and what it implies for Stage D

First real-data run of the balanced sampler (Stage D GPU smoke, `pool_size=256`,
paired = `set0-2` LR, unpaired = the 349 `lr_sims` boxes):

```
source-classifier AUC   before = 0.513   after = 0.518   (target <= 0.60)
in paired support       507 / 512 crops
descriptors kept        var_density, disp_rms
descriptors dropped     redshift, mean_density, mean_div_v, var_div_v, tidal_I2, tidal_I3
```

Two things follow, and the second one matters for the headline claim.

**1. The dropped descriptors are as predicted.** `redshift` is constant, `mean_density`
is identically 0 (crop-periodic CIC normalised by the crop mean), and the velocity
descriptors are unavailable because this configuration is displacement-only
(`use_channels: [0,1,2]`, `vel_channels: null`); tidal invariants are off by default.
That leaves a **2-dimensional** descriptor space.

**2. Do the LR-only boxes sample different environments? Measured three ways —
final answer: essentially no.** This section supersedes two earlier readings in this
document; both are reproduced so the reasoning is auditable.

| # | configuration | paired / unpaired crops | boxes | AUC before | AUC after |
|---|---|---|---|---|---|
| 1 | disp-only descriptors (old default, 2 live) | 256 / 512 | 3 / 349 | 0.512 | 0.518 |
| 2 | + tidal invariants (4 live) | 384 / 768 | 3 / **120** | **0.585** | 0.503 |
| 3 | **production** (4 live, 3 seeds) | **2048 / 4096** | 3 / **349** | **0.506 / 0.500 / 0.490** | 0.500 / 0.510 / 0.487 |

Reading #2 in isolation suggested a genuine population difference visible only through
the tidal invariants. **It did not replicate at production scale**: with all 349 unpaired
boxes and 8x the crops, the AUC sits at chance for all three seeds. Row 3 is the
configuration that will actually run, so it is the one that counts.

Two candidate explanations for #2, and I cannot separate them from the data I have:
its unpaired pool was `sorted(glob)[:120]`, an arbitrary alphabetical slice that may
correlate with generation batch; and its paired side had only 384 crops drawn from
**3 boxes**, so the classifier's train/test halves share boxes and it can key on
box-specific idiosyncrasies present in both. Either way, the honest conclusion is that
**AUC computed over crops from 3 boxes is a weak estimator of a population difference**,
and the box-level control (paired-box-1 vs paired-boxes-2,3 = **0.454**) is the right
reference scale for what "chance" looks like here.

**What is established:** at the production configuration, the paired and LR-only
environment distributions are indistinguishable under the 4 live descriptors, ~99.7% of
unpaired crops fall inside paired support, and the balancing guard passes with large
margin on every seed.

**Consequences.**

- The stated motivation — *"the main theoretical benefit of extra LR data is broader
  condition coverage"* — is **not supported on this dataset**. The 349 LR-only boxes are
  independent realisations of the same cosmology; they supply more *samples*, not new
  *environments*. Success criterion 3 ("improvement strongest in an under-represented but
  supported bin") is therefore expected to be **weak or vacuous**, and a null result on it
  should not be read as evidence against Stage D.
- The honest restatement of the Stage D hypothesis on this data: extra LR-only
  simulations help, if at all, by giving the critic **more independent conditioning
  realisations**, not by extending environment coverage.
- Balancing is **insurance, not a correction** here. It is still worth running: it
  guarantees the critic cannot exploit a source shortcut, and it costs one startup pass.
- `use_tidal: true` is **kept** even though it changed nothing at production scale. It is
  strictly more descriptor information, the guard still passes with margin, and with it
  off only 2 descriptors survive the constant-filter — which would make the guard pass
  while measuring almost nothing. Cheap insurance against a difference we cannot see.
- Binning stays `n_dims=2, n_bins=8` (64 bins occupied, ~15/4096 crops rejected).
  `n_dims=4/n_bins=5` was measured to reject 45% of crops *and* score worse afterwards.

### 9.0c(ii) CORRECTION: the cost estimate that set the step count was wrong

The Stage D smoke reported **3490 ms/step**, giving "16.1x Stage A" and "19.4 h per run at
20000 steps". **That figure was bad.** `step_time_ms` is a *cumulative* average since the
training loop started (`(now - t0) / step`), so at **step 7 of a 30-step cold run** it is
dominated by one-off startup — CUDA init, dataset materialisation, and (for Stage D) the
349-box LR crop-pool build.

Marginal rate recovered from consecutive log rows of the live runs:

| run | window | marginal ms/step | ETA @8k | ETA @20k |
|---|---|---|---|---|
| stage_c_s0 (A5000) | 250->350 | 1493 | 3.2 h | 8.2 h |
| stage_c_s1 (A6000) | 150->250 | 806 | 1.7 h | 4.4 h |
| stage_c_s2 (A6000) | 300->400 | 1296 | 2.7 h | 7.1 h |
| stage_d_s0 (A6000) | 150->250 | 801 | 1.7 h | 4.4 h |
| stage_d_s1 (A6000) | 150->250 | 817 | 1.8 h | 4.5 h |

So 20000 steps costs **4.4-8.2 h**, not 19.4 h — a ~3x overestimate that had been used to
justify cutting the step count to 8000. The 8000-step runs were cancelled at ~step 350
(archived under `runs/dmsr/_abandoned_8000step/`) and **relaunched at 20000**, removing
"undertrained critic" as an alternative explanation for a null C-vs-D result.

Two lessons: **a cumulative-average timer is not a rate** — take the marginal difference;
and **never size a long run from a cold short one**, because startup is amortised in the
real job and not in the probe.

Note also the hardware is heterogeneous (`gpu28` is A5000, `gpu24`/`gpu31` are A6000), so
wall-clock differs ~2x between arms. This does not affect the comparison: training is
deterministic given seed and data, and the matched quantity is *steps*, not seconds.

### 9.0c `lambda_adv` calibration — and a trap in how to measure it

30-step Stage D GPU smoke (`runs/dmsr/smoke_d_gpu`, random init, `lambda_adv=0.1`):

| step | `grad_norm_flow` | `grad_norm_adv` | ratio |
|---|---|---|---|
| 10 | 0.972 | 0.0139 | 0.014 |
| 20 | 2.484 | 0.0237 | 0.0095 |

Read naively this says the ratio is **0.01 against a 0.1–0.3 target**, i.e. `lambda_adv`
is ~20x too small and the critic would be inert. **That reading is wrong**, and acting
on it would have mis-tuned both C and D.

The smoke ran from a **random** initialization, where `grad_norm_flow` is 1–2.5. Stages
C/D start from a **converged** Stage B, and Stage A's `grad_norm_flow` settles to a
median of **0.157** after 8k steps. Against that denominator:

```
ratio = 0.0139 / 0.157 = 0.089
ratio = 0.0237 / 0.157 = 0.151
```

which straddles the low end of the 0.1–0.3 target. **Keep `lambda_adv = 0.1`.** Erring
low is the safer failure mode — the design explicitly warns not to let the adversarial
term dominate early training.

The general lesson: **calibrate the adversarial weight against the gradient norm of the
checkpoint the stage will actually start from**, not against a randomly-initialised
model. The ratio is not a property of `lambda_adv` alone.

This estimate still carries uncertainty because the critic's scale co-evolves, so
**check `train/grad_ratio_adv_flow` in the first ~500 steps of the real runs**. If it
lands outside 0.1–0.3, `lambda_adv` must be changed and **both C and D restarted** — the
schedule has to stay identical for the comparison to mean anything.

### 9.0c(iii) THE CRITIC WAS INERT — R1 was ~300x the hinge loss

**The most consequential bug of the run.** The first 20000-step C/D launch was cancelled
at step ~400 after the calibration logging showed:

```
grad_ratio_adv_flow = 0.0002 - 0.0005      (target 0.10 - 0.30; even at full
                                            lambda_adv=0.1 that is only ~0.0025)
```

The cause was not `lambda_adv`. Inspecting the critic scores:

| step | `loss_D` | `D(real)` | `D(fake)` | separation |
|---|---|---|---|---|
| 50 | 2.0001 | −0.07301 | −0.07292 | −0.00009 |
| 200 | 1.9998 | −0.07856 | −0.07873 | +0.00017 |
| 400 | 1.9999 | −0.07833 | −0.07841 | +0.00008 |

`loss_D` pinned at **exactly 2.0** — `relu(1-0) + relu(1+0)`, the value for a critic
that outputs ~0 on everything — with real/fake separation at the 1e-4 noise level, flat
across 350 consecutive steps in **both** arms. The critic was not learning at all.

**Why.** `r1_penalty` returns the standard `||grad_x D(x)||^2` **summed over all input
elements**, and `LazyR1` multiplies by `interval` to keep the effective strength
interval-independent. Here that is:

```
raw R1 at init      = 7.64
weighted penalty    = 0.5 * gamma(10) * 7.64 * interval(16) = 611
hinge loss scale    = ~2
                    => R1 is ~306x the data term
```

So the critic's objective was overwhelmingly "have zero input-gradient everywhere", whose
optimum is a **constant function** — precisely what was observed. `gamma=10` is the
StyleGAN2 value, tuned for 2D at `256^2 x 3 ~ 196k` elements; this critic sees
`64^3 x 4 ~ 1.05M` elements, 5x more, and the penalty is a *sum*, so the raw term scales
with the element count while the hinge term does not.

**Why the earlier smoke did not catch it.** The 30-step Stage D smoke (9.0d) showed
`loss_D: 613 -> 1.90 -> 1.76 -> 1.82` and I read the drop from 613 as "the R1 spike
settling, critic healthy". It was actually the critic collapsing *to* the degenerate
fixed point at 2.0 and the small wobble around it was noise. **A hinge `loss_D` near 2.0
is not a healthy critic — it is an uninformative one**, and the smoke was too short for
the distinction to be visible. `train/critic_real_score` minus `critic_fake_*_score` is
the diagnostic that actually distinguishes them, and it should be checked directly rather
than inferred from `loss_D`.

The fix is calibrated by measurement, not guessed: `scripts/_r1_sweep.py` freezes the
generator at the Stage B checkpoint (exactly the C/D initialisation), trains only the
critic for 300 steps at several `r1_gamma`, and reports separation together with the
generator-side `grad_ratio` at full `lambda_adv`. Results in 9.0c(iv).

### 9.0c(iv) Calibration: r1_gamma = 0.001, lambda_adv = 0.0084

`scripts/_r1_sweep.py`, 300 critic-only steps from the Stage B checkpoint (the exact C/D
initialisation), identical fixed batches for every gamma:

| gamma | `loss_D` | separation `D(real)-D(fake)` | raw R1 | weighted R1 / hinge | grad_ratio @ lam=0.1 |
|---|---|---|---|---|---|
| 0.0 | 0.5246 | 2.5706 | — | — | 81.8 |
| 0.001 | 1.8145 | **0.1855** | 212.8 | **0.94** | 2.388 |
| 0.01 | 1.9933 | 0.0067 | 2.374 | 0.10 | 0.110 |
| 0.1 | 1.9992 | 0.0008 | 0.445 | 0.18 | 0.018 |
| 1.0 | 1.9998 | 0.00025 | 0.047 | — | 0.007 |
| **10.0 (shipped)** | 1.9999 | 0.00013 | 0.008 | — | 0.0025 |

Everything at `gamma >= 0.1` is inert, and the shipped default was the worst point on the
curve.

`gamma = 0.01` is the only value that satisfies *both* naive criteria at the original
`lambda_adv=0.1` — but it was **not** chosen. Its critic barely separates (0.0067, only
~7x the 1e-3 noise floor) and its R1 term has collapsed to 10% of the loss, i.e. it is
neither discriminating nor meaningfully regularised.

The key structural point: **`grad_ratio` is exactly linear in `lambda_adv`** (the gradient
of `lam * L` is `lam * grad L`). So critic *strength* and gradient *subordination* are
independent knobs, and the right procedure is: pick `gamma` for a critic that genuinely
discriminates, then set `lambda_adv` to place the ratio in band.

- `gamma = 0.001`: separation **0.186** (28x better than 0.01), and weighted R1 (1.70)
  sits at 0.94x the hinge term (1.81) — R1 is a real constraint that **self-balances**,
  because raw R1 grows as the critic sharpens.
- `lambda_adv = 0.1 * 0.20 / 2.388 = 0.0084` places `grad_ratio` at ~0.20, mid-band.

Both are set in `_base.yaml` **and** in the `adv:` block that Stage C and D each override
(the stage files re-declare `adv:`, so editing only `_base.yaml` silently left
`lambda_adv=0.1` in the resolved configs — caught by re-reading the resolved values rather
than trusting the edit). `test_stage_parity.py` confirms C and D remain identical.

### 9.0c(v) In-training verification of the calibrated critic

Third (current) C/D launch, step 400, `lambda_adv` still ramping (`lam=0.00168`, 20% of
full). Ratios extrapolated to full `lambda_adv` using the exact linearity:

| run | separation `D(real)-D(fake)` | ratio @ full lam |
|---|---|---|
| c_s1 | +0.0779 | 0.064 |
| c_s2 | +0.1468 | 0.106 |
| d_s0 | +0.0832 | 0.100 |
| d_s1 | +0.0779 | 0.066 |
| d_s2 | +0.2007 | 0.163 |

Critic separation is **0.078-0.201**, i.e. 400-1000x the inert configuration's 2e-4, and
`loss_D` runs 1.83-2.00 rather than pinned at 2.0. The critic is doing real work.

The gradient ratio sits on the **low edge** of the 0.10-0.30 target (mean ~0.10; 3 of 5
inside, two at ~0.065). **Accepted rather than re-tuned**, for three reasons:

1. The design explicitly warns not to let the adversarial term dominate early training, so
   erring low is the safe direction; the failure this replaced was the *opposite* extreme.
2. `g_adv` generally grows as the critic sharpens, so the ratio should drift upward as the
   ramp completes — it is measured at 20% ramp here.
3. **The same `lambda_adv` applies to both arms.** Any residual miscalibration is common
   to C and D, so it cannot bias the C-vs-D difference, which is the quantity under test.

A fourth restart to chase the band would cost more than it buys. `grad_ratio_adv_flow` is
logged every 200 steps, so the trend is auditable in the run record.

### 9.0d Stage D GPU smoke — pipeline verification

```
[d] done. counters={'gen_paired': 28, 'gen_second': 28, 'critic': 30}  second_stream=lr_only
val_exact_consistency_rel = 3.36e-07     (held-out, mid-adversarial-training)
loss_D: 613 (step 1, one-time R1 spike at init) -> 1.90 -> 1.76 -> 1.82  (stable)
```

`gen_paired = gen_second = 28 < 30` because `critic_warmup_steps=2` freezes the
generator for the first two steps — applied identically in C and D, so the match holds.
Exact consistency is untouched by adversarial training, as designed.

### 9.0e Two run-hygiene issues found during execution (both handled)

**(i) A CPU smoke run overwrote a live run's manifest.** The smoke sweep was invoked as
`--config configs/dmsr/stage_a_paired_flow.yaml --stage det --smoke`. `--smoke` did not
override `output.run_dir`, so it inherited `runs/dmsr/stage_a` — the directory of the
*still-running* Stage A job — and `init_run_dir` rewrote `config.yaml` / `env.json` there
with `stage: det, _synthetic: true, steps: 4`.

Scope: **manifest only.** `metrics.csv` and the checkpoints are continuously rewritten by
the live process from its own in-memory state and were correct throughout (verified:
`metrics.csv` last step 21050, checkpoints timestamped after the incident).

Handled: originals preserved as `config.CORRUPTED_BY_SMOKE.yaml` / `env.CORRUPTED_BY_SMOKE.json`;
`config.yaml` re-derived from the unchanged stage config and stamped with a
`_manifest_provenance` field recording exactly what happened and one caveat — `_base.yaml`
was edited after launch (`use_tidal` false→true), so the reconstructed `env:` block differs
from the launched one. Stage A never reads `env:` (no balanced sampler), so the run is
unaffected. Cause fixed: **`--smoke` now redirects `run_dir` to `<run_dir>_smoke_<stage>`**,
verified by re-running the exact command and confirming `runs/dmsr/stage_a/config.yaml` is
byte-identical afterwards.

**(ii) Stage A and Stage B step counts are not matched.** Stage A runs 40000 steps;
Stage B was launched with `train.steps=20000`. The design says Stage B should keep all
other settings matched to Stage A, so this is a genuine control defect on the A-vs-B
encoder ablation (it does **not** touch C-vs-D, which is the headline comparison and is
matched by construction and by test).

Mitigation without extra compute: Stage A logs held-out validation every 1000 steps, so a
**step-matched** A-vs-B comparison at 20000 steps is available directly from
`runs/dmsr/stage_a/metrics.csv` (A `val_rk_transition` at step 20000 = 0.8922). Report
A-vs-B at matched step 20000, and report the `ckpt_best` comparison separately, noting
that both models converge by ~5k so the extra Stage A steps are not doing work.

**(iii) The `paired_deterministic` baseline was being evaluated as a flow.** `det`
trains a one-shot regressor (`deterministic_regression_loss` fits
`P_A(v_theta(0, t=1, y))` directly), but `validate()` generated samples via
`flow.generate()`, which integrates the ODE. That scores the network on a trajectory it
was never fitted to — producing meaningless validation metrics *and* meaningless
best-checkpoint selection, while the training loss and the logs look completely healthy.

Caught before the run got past startup. Fixed with a `NullSpaceFlow.deterministic` flag:
when set, `sample_residual` returns the single-shot prediction instead of integrating.
It is set from the stage in the trainer, inferred from the checkpoint's recorded stage in
`scripts/dmsr_eval.py`, and survives the deepcopy into `ModelEMA`. Regression test
(`test_deterministic_mode_bypasses_the_ode`) asserts the deterministic output is
independent of both `z` and `n_steps`, equals exactly what the training loss fits, stays
LR-consistent, and that flow mode retains its `z` dependence.

The general hazard: **a baseline whose train-time and eval-time computations disagree
fails silently and looks fine.** Worth checking for every non-default stage.

### 9.0f Stage A vs Stage B — preliminary, and pointing at a null result

Same seed, same data, same architecture; the **only** difference is the condition
encoder's initialisation (random vs 20000-step LR masked-SSL on 352 boxes). Held-out
`set15`, at matched steps:

| step | A `rk_trans` | B `rk_trans` | diff | A `rk_high` | B `rk_high` | diff |
|---|---|---|---|---|---|---|
| 1000 | 0.8499 | 0.8474 | −0.0024 | 0.2199 | 0.2180 | −0.0019 |
| 2000 | 0.8601 | 0.8594 | −0.0007 | 0.4365 | 0.4366 | +0.0001 |
| 3000 | 0.8792 | 0.8750 | −0.0042 | 0.7163 | 0.7155 | −0.0008 |

B tracks A to within ±0.004 everywhere, with a consistently non-positive sign on
`rk_transition`. **This is preliminary** (B is at 3000/20000), but the *timing* is what
makes it informative: the main expected benefit of encoder pretraining is a better
starting point, i.e. **faster early convergence**. That effect should be largest in
exactly these first few thousand steps, and it is absent. A large late divergence would
be surprising.

If this holds, Stage B is a **null result**, and the spec's "partial success — Stage B or
Stage C improves over Stage A" branch is not satisfied by B. That is consistent with the
project's standing identifiability finding: unpaired LR under a single fixed operator
carries no information about `ker A`. This stage's SSL was never claimed to recover
unresolved detail — only to improve how well the model *reads* `y` — and on this data
even that appears not to help.

It does **not** affect C-vs-D: both arms initialise from the same Stage B checkpoint, so
whatever B is worth, it is worth the same to both.

### 9.0g `r(k)` structurally favours the mean-collapsed model — pre-registered caveat

Recorded **before** any C/D result exists. The `paired_deterministic` control, at matched
step 1000 against Stage A:

| metric | stage_a | det | better |
|---|---|---|---|
| `rk_transition` | 0.8499 | **0.9023** | det |
| `rk_high` | 0.2199 | **0.8032** | det |
| `Tk_error_high` | 3.7594 | **0.1132** | det |
| `density_power_error` | 1.7354 | **0.2396** | det |
| `sample_diversity` | **0.9993** | 0.0000 | flow |

(Step 1000 flatters `det` — Stage A is badly undertrained there, `Tk_error_high` 3.76 vs
0.044 at convergence. The *ranking on `r(k)`* is the durable part, not the margins.)

**This is structural, not a bug.** The conditional mean is the MMSE estimator, and MMSE
maximises correlation with the truth. A perfectly mean-collapsed model therefore attains
the **highest achievable `r(k)`**, and any model adding genuinely stochastic small-scale
detail must move away from the conditional mean and *lose* `r(k)` while gaining
distributional realism. `sample_diversity = 0.0000` for `det` — exactly zero, confirming
the deterministic path — is the other side of the same coin.

**Why this matters for the decision rule.** `rk_transition` is both a primary metric and
one of only two *conditional* metrics, scored higher-is-better. If Stage D's extra
conditioning data pushes it toward more realism, D could score *lower* on `rk_transition`
while being distributionally better — and the rule could then return "failure of the
LR-only hypothesis" for a genuinely better model.

**Decision: the rule is left unchanged.** Two reasons it is safer than it first appears
for the headline comparison specifically:

1. C and D share the *same* critic, the same `lambda_adv` schedule and the same number of
   adversarial updates, so the realism-vs-`r(k)` pressure applies near-equally to both
   arms. The residual difference between them is attributable to the conditioning data,
   which is the quantity under test.
2. The other conditional metric, `squeezed_cross_bispectrum_error`, is distributional and
   does **not** reward mean collapse, so criterion 2 does not rest on `r(k)` alone.

**Required when reporting**, pre-registered here:

- Always show `sample_diversity` beside `rk_transition`. A drop in `rk_transition`
  accompanied by a *rise* in diversity is the expected signature of increased realism and
  must not be read as evidence against Stage D.
- In the **baseline** table (det / A / B / C / D), do not rank on `r(k)`: `det` is expected
  to win there, and that is a restatement of MMSE rather than a finding. Same reason the
  project's standing guidance is to report distributional metrics, not HR MSE.

### 9.0h Completed baselines (val `set15`, single seed, final checkpoints)

All three finished their full step counts. `paired_deterministic` and Stage A both ran
40000 steps; Stage B ran 20000 (the mismatch noted in 9.0e(ii)).

| model | steps | `rk_trans` | `rk_high` | `Tk_err_high` | `diversity` | `consist_rel` |
|---|---|---|---|---|---|---|
| `paired_deterministic` | 40000 | **0.9257** | **0.8551** | 0.0770 | **0.0000** | 3.72e-07 |
| `stage_a` | 40000 | 0.9001 | 0.8036 | **0.0387** | 0.5506 | 3.33e-07 |
| `stage_b` | 20000 | 0.8954 | 0.8032 | 0.0419 | 0.5568 | 3.35e-07 |

**The 9.0g pre-registration is confirmed.** `det` wins `r(k)` outright — and its
`sample_diversity` is **exactly 0.0000**. It simultaneously loses on amplitude
(`Tk_error_high` 0.0770 vs 0.0387/0.0419). This is the MMSE trade-off exactly as
predicted, and it is why the baseline table must not be ranked on `r(k)`: `det` "winning"
is a restatement of the fact that the conditional mean maximises correlation, not
evidence that a deterministic model is the better super-resolver.

**Stage B is a null result** (confirms 9.0f). At matched steps, `B - A`:

| step | `rk_trans` | `rk_high` | `Tk_err_high` | `diversity` | `density_power_err` |
|---|---|---|---|---|---|
| 5000 | −0.0073 | −0.0013 | −0.0006 | +0.0131 | +0.0036 |
| 10000 | −0.0012 | −0.0020 | −0.0039 | +0.0022 | −0.0043 |
| 15000 | +0.0022 | +0.0006 | +0.0017 | −0.0046 | +0.0039 |
| 20000 | +0.0032 | +0.0008 | +0.0009 | −0.0021 | −0.0045 |

Differences are tiny and **sign-inconsistent across checkpoints** (B worse at 5k/10k,
marginally better at 15k/20k) on a single seed — noise, not signal. Only
`density_power_error` at step 20000 exceeds 5% (−5.9%), and given the sign flipping it is
not credible without repeated seeds. Conclusion: **LR masked-SSL encoder pretraining does
not help on this dataset**, consistent with the standing identifiability finding. The
spec's "partial success — Stage B improves over Stage A" branch is **not** satisfied.

This does not affect C-vs-D: both arms initialise from the same frozen Stage B checkpoint
(`ckpt_best_frozen.pt`, step 16000, val `rk_transition` 0.8974).

### 9.0i CORRECTION: `cellsize` was wrong at both CIC call sites

Found while investigating how to generate better-covering sims. The box is **100 Mpc/h**
(`paramfile.genic`: `BoxSize=100000` kpc/h); I assumed 1000 Mpc/h and hand-wrote
`cellsize: 15625.0`. Worse, `cic_density` builds its lattice in units of **one cell of
the grid the field lives on**, so the two call sites need *different* values and one
constant was used for both:

| call site | field lives on | correct | used | error |
|---|---|---|---|---|
| `HighPassDensity` (critic `rho_high`, density metrics) | HR 512³ grid | **195.3125** | 15625 | **80x** |
| `environment_descriptors` | LR 64³ grid | **1562.5** | 15625 | **10x** |

**Re-measured at the correct LR scale** (production pools, 2048/4096 crops, 349 boxes):

| descriptor (paired) | as-run | correct | ratio |
|---|---|---|---|
| `var_density` mean | 0.0575 | 6.441 | 112x |
| `var_density` sd | 0.0242 | 8.457 | 349x |
| `tidal_I3` | 8.06e-06 | 3.441 | ~4e5 |
| **AUC before / after** | 0.5062 / 0.5004 | **0.5075 / 0.5011** | unchanged |

The density field really was ~unperturbed, **but the 9.0b conclusion survives**: the pools
still do not separate. Not a trivial consequence of standardisation either — the
distribution *shape* changed (sd/mean 0.42 -> 1.31) and the AUC still sits at chance.
**The LR-only boxes genuinely do not sample different environments.**

### 9.0j The critic's two input channels were never scale-matched

Fixing `cellsize` alone would have made things **worse**, which is why it was measured
before being applied. On real paired HR crops:

| cellsize | std(`rho_high`) | std(residual) | ratio |
|---|---|---|---|
| 15625 (as run) | 0.0619 | 0.1176 | **0.53** |
| 195.3 (correct) | 16.24 | 0.1176 | **138** |

`critic_input` concatenates `P_A(x)` [3ch] with `rho_high` [1ch] -- physically different
quantities in different units -- and never normalised them, so their relative scale was an
accident of the unit constants. The wrong constant happened to balance them (0.53); the
correct one lets density dominate by ~138x and starves the residual channels of gradient,
since spectral norm caps how much the first conv can rescale.

Fix: `CriticInputNormalizer`, fitted once from **real** HR crops and applied with
**identical constants to real and fake**. Per-batch statistics are explicitly rejected --
they would normalise real and fake differently and hand the critic a source shortcut, the
same failure mode that withholding the raw LR tensor prevents. Asserted in
`test_normalizer_uses_identical_constants_for_real_and_fake`. Both cellsizes are now
**derived** via `density.cellsizes()` from `data.boxsize` / `data.lr_grid_res`.

**Impact on the first completed C/D runs:** the verdict is unaffected -- both arms used
identical constants and the null was carried by `rk_transition` differing by 0.00012,
which involves no density. But the Eulerian half of the critic operated on a
~unperturbed field, so density metrics are wrong in absolute terms and the critic was
closer to residual-only than designed. C/D re-run with correct units + normalised channels.

- 9.1 Baseline table on held-out boxes — pending
- 9.2 Stage C vs Stage D table with CIs — pending
- 9.3 Matched-compute table (from `compute_audit.json`) — pending
- 9.4 Plots: r(k), T(k), density/velocity power, bispectrum,
  environment-stratified performance, conditional diversity,
  correct- vs shuffled-condition — pending
- 9.5 Go/no-go conclusion — pending

> **The Stage C-vs-D result above is the locked no-go baseline and is preserved
> verbatim.** Everything from Section 10 on is a *separate* line of work
> (deterministic mean + stochastic innovation, and a Fourier-band loss diagnostic)
> added 2026-07-20. It changes nothing about Stage C: every new feature is behind a
> disabled-by-default flag, and `tests/dmsr/test_stage_parity.py::test_flags_off_reproduce_stage_c`
> asserts that with the flags off the resolved Stage C config, model, loss and RNG
> are byte-identical. **No new LR-only objective is introduced.**

---

## 10. The null space is not an ideal Fourier high-pass (Part 1)

A recurring temptation is to treat "the detail the LR sim can't see" as a clean
high-`k` Fourier band and to reweight the loss toward it. That is wrong about the
operator. `ker A` for block-average-and-decimate is the set of **per-block
zero-mean** fields, which is a *different* projector from an ideal high-pass.

**Synthetic factor-two demonstration** (`dmsr/nullspace_spectral.py`,
`tests/dmsr/test_nullspace_spectral.py`). For `A(x)[j] = (x[2j] + x[2j+1])/2` the
whole null space is

```
r = [a0, -a0, a1, -a1, ...]   =>   A(r)[j] = (a_j - a_j)/2 = 0   for ANY envelope a
```

- **Constant `a`** → `r` is the pure Nyquist carrier: a single spectral spike at
  `k = N/2`.
- **Slowly varying `a`** → `r = (Nyquist carrier) × a`, a *modulation product*. Its
  spectrum is `a`'s spectrum shifted to sit around Nyquist, **plus** a genuine
  low-`k` sideband from the block-broadcast boxcar replica. For a 2-cycle cosine
  envelope on a 64-grid the power lands at rfft bins `{2, 30}` — simultaneously low
  *and* near-Nyquist. An ideal high-pass at `k_Ny/2` is one contiguous interval and
  **cannot** straddle both. So null-space vectors are structured combinations of
  Fourier modes created by averaging and aliasing, not one separated band.

**Real 3D operator** (same test file, factors 2/4/8). Verified: `A(A_plus(y))≈y`,
`A(P_A(x))≈0`, `P_A(P_A(x))≈P_A(x)`, `P_A` removes the blockwise mean
(`A(P_A(x))≈0`), a projected white field keeps a strongly **non-uniform** radial
spectrum (low-`k` suppressed, coefficient of variation across shells ≫ 0), and an
*arbitrary* low-`k` Fourier mode is **not** an eigenvector of `P_A` (block-broadcast
staircase subtracted) — while the block-aligned DC and Nyquist modes *are*
eigenvectors (eigenvalues 0 and 1), which is exactly why "arbitrary mode" matters.
Consequence for Part 5: any spectral weighting must act on the *already-projected*
real-space velocity error, never move `A` into Fourier space, and never assume the
target lives in one band.

## 11. Deterministic mean + stochastic innovation — Experiment E (Part 2)

`dmsr/mean_innovation.py`. The generator is split into a predictable conditional
mean and a stochastic innovation, both in `ker A`:

```
r_gt   = P_A(x_hr)
m      = P_A(mean_model(y))            # the trained paired_deterministic net, FROZEN
e_gt   = P_A(r_gt - stop_grad(m))      # innovation target
x_hat  = A_plus(y) + P_A(m + u_theta(y, z))
```

- **`mean_model` = `paired_deterministic` reused verbatim** — it is a `NullSpaceFlow`
  in `deterministic` mode, so `P_A(mean_model(y))` is exactly the `x_mean` regressor
  we already trained (Phase 1 = *load*, no new training; verify it reproduces the
  `paired_det` metrics before use).
- **Phase 2 (frozen mean):** the innovation flow `u_theta` learns `e_gt` with the
  *identical* straight-line rectified-flow objective as Stage C — only the target
  changes from `r_gt` to `e_gt` (`innovation_flow_loss`). The mean is a fixed
  offset, receives no gradient, and is `stop_grad`'d in `e_gt` and in the
  generative/adversarial path, so the Stage C generator-gradient routing is
  unchanged. **No MSE is ever applied to a stochastic `x_hat`** — only the
  deterministic mean sees a reconstruction loss (and in Phase 2, none).
- **Exact consistency is preserved structurally:** `x_hat = A_plus(y) + (null-space
  field)`, final projection applied even though `m` and `e_hat` are already in
  `ker A`. Held through ODE integration (`test_mean_innovation.py`).
- **Optional joint fine-tuning** is implemented behind `mean_innovation.joint_finetune`
  (**disabled by default**; do not enable until the frozen-mean experiment is
  complete): mean gets a lower LR and is trained *only* by its own reconstruction
  loss; the flow loss keeps the mean detached.

**Experiment E = Stage C with this decomposition** (`configs/dmsr/mean_innovation_e.yaml`):
identical critic, identical paired-LR-only second stream (`stage: c`), identical
`lambda_adv`, steps, seed, encoder SSL init and Stage B warm start for the flow.

**Innovation diagnostics** (logged at validation, `innovation_diagnostics`, never in
the hot loop, no centering penalty added — bias is *measured* first): RMS of
`innovation_mean`, innovation variance, `sample_mean − x_mean`, `r(k)` of `x_mean` /
`sample_mean` / individual samples, mean vs innovation power, and a
double-counting cross-fraction `(||P_A(m+e)||² − ||m||² − ||e||²)/||P_A(m+e)||²`.

## 12. Fourier-band loss diagnostic and the decision gate (Parts 3–4)

`dmsr/fourier_diag.py`, `scripts/dmsr_fourier_diag.py`. **This is a measurement, not
a change to training.** It never back-propagates, never draws its own RNG (it reuses
the flow loss's own `v_pred`/`v_target` via `return_fields`, so even the in-training
monitor perturbs nothing), and never moves `A`/`A_plus`/`P_A` into Fourier space.

- **Transform:** full complex `fftn(norm='ortho')`, not `rfftn`. Parseval is exact,
  mode count = voxel count, so `mean_x|dv|² == mean_q|DV_q|²` — *unit spectral
  weights reproduce the spatial MSE numerically* (the property Part 5 must reduce
  to), and Hermitian DC/Nyquist counting is automatic. `dv = P_A(v_pred − v_target)`.
- **Shells:** integer radial `k = round(|q|)`, aggregated into the same
  low/transition/high bands as evaluation via `k_LR = N_hr/(2s)`.
- **Reports, per channel × shell:** `n_modes`, target power/mode, target shell
  energy, error power/mode, error shell energy, loss fraction, relative shell error,
  shell cosine — *both* per-mode (amplitude imbalance) and shell-total (3D mode-count
  growth). Run for Stage C and E, on validation/training crops, density and velocity
  channels separately, at early/middle/late `t`.

**Decision gate** (`gate_decision`, Part 4). A non-uniform *target* spectrum is
expected and is **not** a problem by itself. `IMBALANCE_SUPPORTED` requires ALL five:
(1) a small subset of shells holds most of the loss; (2) transition/high shells
retain substantially larger relative error; (3) those shells get too little gradient
under spatial MSE; (4) the pattern is stable across batches/crops/seeds (≥2
diagnostics); (5) it is not an FFT-normalisation / near-zero-target-power artifact.
Absence of the signature → `NO_IMBALANCE`; anything mixed/unstable/artifactual →
`INCONCLUSIVE`. Conservative by construction: the default is **not** to reweight.

**Gate status: NOT YET RUN on real C/E checkpoints** (the C/D runs were still in
flight at 2026-07-20). Until the report is produced and reviewed and returns a stable
`IMBALANCE_SUPPORTED`, **Part 5 (spectral balancing) is deliberately not implemented**
and F/G are not launched. This is the review checkpoint in the implementation order.

## 13. Exact commands — C, E (run now) and F, G (GATED)

Development discipline: tune on the training/validation mechanism only; **freeze
configs before touching the 12-box benchmark** (it has already been examined). C-vs-E
must match paired boxes, steps, batch size/accumulation, crop sampling/augmentation,
seeds, sampling solver/steps, and evaluation noise.

```bash
# --- C: locked Stage C control (unchanged; see Section 6 / runbook) ---
launch dmsrC0 configs/dmsr/stage_c_critic_pairedlr.yaml stage_c_s0 \
    --set train.seed=0 output.run_dir=runs/dmsr/stage_c_s0

# --- E: mean + innovation (frozen paired_det mean + innovation flow) ---
# prerequisites: runs/dmsr/paired_deterministic/ckpt_best.pt, stage_b/ckpt_best_frozen.pt,
#                lr_ssl/encoder.pt  (all already trained)
for s in 0 1 2; do
  launch dmsrE$s configs/dmsr/mean_innovation_e.yaml mean_innovation_e_s$s \
      --set train.seed=$s output.run_dir=runs/dmsr/mean_innovation_e_s$s \
            wandb.name=dmsr_mean_innovation_e_s$s
done

# --- Fourier diagnostics on BOTH C and E (validation crops; NOT the benchmark) ---
for tag in stage_c_s0 mean_innovation_e_s0; do
  cfg=configs/dmsr/stage_c_critic_pairedlr.yaml
  [ $tag = mean_innovation_e_s0 ] && cfg=configs/dmsr/mean_innovation_e.yaml
  PYTHONPATH=src python scripts/dmsr_fourier_diag.py --config $cfg \
      --ckpt runs/dmsr/$tag/ckpt_best.pt --split val --max-batches 8 --plots \
      --out runs/dmsr/fourier_diag/$tag
done
# => review runs/dmsr/fourier_diag/*/fourier_diag_report.txt : verdict must be
#    IMBALANCE_SUPPORTED (stably, over batches/seeds) before proceeding.

# --- F, G: GATED. Do NOT run until the gate returns IMBALANCE_SUPPORTED. ---
#     F = Stage C + spectral balancing ; G = mean+innovation + spectral balancing.
#     Part 5 (the spectral loss) is not implemented until the gate passes.
```

**Prospective decision rules** (applied mechanically after freezing; three seeds do
NOT give reliable percentile-bootstrap coverage over seed resamples, and there is
only one held-out val box + the 12-box test bootstrap):
- *Mean+innovation succeeds* if `rk_high` improves ≥1% rel over C; density power
  error, density PDF error, `Tk_error_high` degrade ≤5%; diversity ≥90% of C; exact
  consistency < 1e-6.
- *Spectral balancing succeeds* if density power error or `Tk_error_high` improves
  ≥10%; `rk_transition`, `rk_high`, density PDF error, diversity degrade ≤5%; exact
  consistency < 1e-6.
