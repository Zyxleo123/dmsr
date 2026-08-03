# Why the residual prior destroys the field, and what it is not

Diagnosis of the first Gate A / Gate B runs. Status as of 2026-08-01.

Headline numbers, `residual_prior` (v1, converged at 60k steps), 8 validation
crops, core 64³ + 48 cells of context, 20 DDIM steps, EMA weights:

| quantity | value | reference |
|---|---|---|
| coherence with the TRUE residual, `r(k)` high band | **0.330** | 1.0 = perfect |
| coherent fraction of the residual's power (`r²`) | **11%** | 89% is noise |
| amplitude ratio, displacement / velocity | 1.90× / 2.27× | 1.0 = correct |
| density σ ratio at `alpha = 1` | 0.165 | SR2 alone: 1.018 |
| density σ ratio at *any* `alpha > 0` | worse than SR2 | — |
| conditioning shuffle, `r(k)` high band | 0.330 → **0.013** | conditioning IS used |
| `x0` clip fraction, step 0 / steps 1-19 | 0.90 / 0.01-0.02 | one step, not a regime |

## Conclusion: the model underfits the correction. It does not mis-scale it.

The residual prior has found a **weak but real** version of the right mapping.
It is conditioned correctly, it correlates with the true residual at high `k`
(`r = 0.33`), and that correlation vanishes when the conditioning is shuffled.
What it cannot do is make the correlation strong.

At `r = 0.33`, only `r² = 11%` of the residual's power is signal. With the
amplitude 1.9× too large (power 3.6×), the model delivers **0.39× the true
residual's power as coherent correction and 3.2× as incoherent noise**. That
ratio is the whole failure:

* The **displacement power spectrum improves** — `r(k)` high rises 0.787 → 0.808
  at `alpha = 0.2`, the transfer-function error nearly halves at `alpha = 0.5`.
  Injecting noise with roughly the right variance fixes SR2's high-`k` power
  deficit, and every amplitude statistic rewards it.
* The **density field collapses anyway**, monotonically, at every `alpha > 0`.
  Density depends on *phase* coherence: particles must converge on the same
  place to form a halo. Noise at 3× the signal scatters them out.

That is the classic "improved the power spectrum by adding noise" trap, and it
is why Gate B found 1-6 halos in a 100 Mpc/h box.

**`residual_scale` is therefore not a fix.** There is no `alpha` at which density
recovers; `alpha = 0` (exactly SR2) is the best member of the sweep. The
displacement improvements at `alpha ≈ 0.2-0.5` are real but are measured by
statistics that cannot see the failure.

Three things follow, in priority order:

1. The next test is a **one-crop overfit**: can the model reproduce a *single*
   known residual with `r → 1`? If not, the architecture or objective is the
   limit and no full training run is justified. This is the cheapest experiment
   that can distinguish "underfitting" from "cannot fit".
2. **Switch to v-prediction.** The `eps`-prediction endpoint is a confirmed
   defect (below) — bounded, but free to remove.
3. **High-pass the model's output.** The model has *no* low-`k` information
   (`r = -0.06`) yet writes `low_k_change = 1.17` there. §"Low-k" explains why
   this is justified even though the true residual does carry low-`k` content.

---

## What was actually run

| job | stage | result |
|---|---|---|
| 24261 `rw_prior` | v2 prior training | still running, ~4k/60k steps, will not finish |
| 24288 `rw_fit` | reward model + constraint calibration | **succeeded** |
| 24267_0-7 `rw_oracle_cat` | Gate B candidate scoring | ran; candidates are empty |
| 24275_0 `rw_oracle_cat` | Gate B verdict | `blocked_uncalibrated_constraints` |
| 24677 / 24679 `rw_res_diag` | residual diagnosis, v1 / v2 | **succeeded** |

Gate A had already failed 4 of 8 criteria when Gate B was sampled. See
§"Process failures".

---

## Gate B: what the candidates actually were

Every candidate produced **1-6 halos and 0 subhaloes in a whole 100 Mpc/h box**;
all four reliable host bins empty in all 32 candidates; `feasible_fraction = 0.0`
in all four groups.

Reported `R_occ = 0.0` for the sampled mean is **not** a good score — an empty
bin is filled with `mu`, so deleting every host scores a perfect zero. The
honest number is `R_abund = 1.57e6` against a baseline of 3170. The gate
documents this trap and refuses such candidates as infeasible; the CEM elite
selector does not (§"Bugs found").

Field metrics of a candidate: `density_sigma_ratio = 0.12`, `low_k_change =
0.96`, `density_power_error = 6.1`. One to two orders of magnitude outside any
threshold.

The verdict was additionally blocked because `constraints.calibrated: false` was
still committed when the aggregate ran — four minutes after the calibration job
that produced the replacement values finished. Cosmetic here: the observed
values miss by 10-100×, calibrated or not.

## The reward model and the constraints: both sound, with two caveats

`mu` and `C` from 8 training boxes, full-box HR catalogs, 10 active dimensions.
`cond(C) = 242 → cond(C_reg) = 41` at `lambda = 2.98e-5`, per-bin scatter
0.24%-9%. **The reward is well-posed.** Two things to carry forward:

**`C` is diagonal, and `R_cat` weights the wrong bins.** With 8 boxes and 10
dimensions the off-diagonals are not identifiable, so `covariance: auto`
correctly returns a diagonal `C`. Consequence: `R_cat = R_occ + R_abund` exactly,
and inverse-variance weighting puts

| bins | share of `R_cat` |
|---|---|
| 6 subhalo-abundance bins | **71%** |
| `occ_Mhost1e13` + `occ_Mhost3.16e13` (the Gate B target) | **7.4%** |

Weighting by reproducibility rewards low-mass counts. **Do not optimise `R_cat`**
— use `R_occ_reliable`. (`lambda` is load-bearing: it is 5× the smallest bin
variance, and without it `nsub_M3.98e+10` alone would carry >70% of the weight.)

**The `lr_consistency` constraint is vacuous.** Calibration measured the true HR
field at 0.464-0.495 on this statistic and the frozen base at 0.433-0.460 —
ground truth scores *worse* than the model. It is dominated by the deterministic
mismatch between `block_average(Psi)` and `y_lr`, not by model error. The
calibrated ceiling of 0.690 will reject nothing.

Calibration otherwise replaced four guesses with measurements, two of which were
far too generous and two absurdly tight:

| constraint | placeholder | calibrated | true residual scores |
|---|---|---|---|
| `low_k_change_max` | 0.02 | 0.1396 | **0.287-0.42** |
| `displacement_power_error_max` | 0.40 | 0.315 | — |
| `density_power_error_max` | 0.05 | 0.0375 | — |
| `lr_consistency_error_max` | 0.05 | 0.690 | — |

**`low_k_change_max` rejects ground truth** and must be re-derived. It is the one
threshold not anchored to a measured baseline — it was set to
`0.1 × HR box-to-box scatter (1.396)` by assumption. The correction required to
turn SR2 into HR scores 0.287 at crop level and 0.296/0.423/0.320 on 256³
sub-cubes of set0/set3/set8. The calibration script's own docstring warns
against exactly this: *"a threshold below this would reject a perfect model."*

---

## The diagnosis

Three hypotheses, three measurements. `scripts/reward/residual_diagnose.py`,
crop-level, same geometry and sampler as the oracle.

### Amplitude sweep — compose `Psi_base + alpha * dPsi_hat`

Medians over 8 crops (**not means**: the crop-level density σ ratio is
heavy-tailed — for SR2 itself the 8 crops read
`[3.73, 1.38, 1.00, 1.04, 0.59, 0.92, 1.21, 0.60]`, and the mean claims SR2 sits
outside the Gate A band while the median 1.018 agrees with the full-box
calibration at 0.99):

| alpha | density σ | low_k | disp `r(k)` high | disp `Tk` err high | disp pow err |
|---|---|---|---|---|---|
| 0.00 (= SR2) | **1.018** | 0 | 0.7867 | 0.0536 | 0.0956 |
| 0.10 | 0.655 | 0.117 | 0.8007 | 0.0468 | 0.0895 |
| 0.20 | 0.411 | 0.234 | **0.8077** | 0.0466 | 0.0899 |
| 0.33 | 0.335 | 0.386 | 0.8053 | 0.0436 | 0.0814 |
| 0.50 | 0.289 | 0.585 | 0.7832 | **0.0284** | **0.0601** |
| 1.00 | 0.165 | 1.171 | 0.6477 | 0.1857 | 0.3443 |
| TRUE | 1.000 | 0.287 | 1.0000 | 0 | 0 |

Displacement improves, density never does. See the conclusion above.

> Crop-level `density_power_error` is **unusable**: it reads 0.525 for SR2 itself
> against 0.007-0.025 full-box. The σ ratio survives cropping; the power error
> does not.

### Coherence — `r(k)` between `dPsi_hat` and the TRUE `dPsi`

This is the measurement nothing in the pipeline previously made. Every other
diagnostic compares the *composed field* to HR, where `Psi_base` dominates and a
useless residual still scores well.

| band | v1 | v2 |
|---|---|---|
| low | **-0.060** | -0.075 |
| transition | 0.206 | 0.061 |
| high | **0.330** | 0.138 |

**v2 is worse than v1 on every axis** and has no improvement window at all. More
training on the context geometry has not helped; at 18 s/step it reaches ~9.5k of
60k steps before its 2-day limit. Cancel it.

### Conditioning — same starting noise, `Psi_base`/`y_lr` from another crop

| | matched | shuffled |
|---|---|---|
| `r(k)` high | 0.330 | **0.013** |
| cosine | +0.216 | -0.003 |

**The conditioning is used.** Every bit of correlation with the true residual is
conditioning-driven. "The model learned generic unconditional denoising" is
falsified. (The sample itself is still largely noise-determined — shuffled vs
matched cosine 0.813 — but the *informative* part is not.)

### Low-k: why a null-space projection is now justified

The true residual carries real low-`k` content (0.287), so a high-pass is not
lossless in principle. But the model's low-`k` content is **uncorrelated with it**
(`r = -0.06`) while carrying large amplitude (`low_k_change = 1.17` at
`alpha = 1`). Projecting it out discards nothing the model was getting right and
removes pure damage. An earlier revision of this analysis argued against
projection on the grounds that the true residual has low-`k` power; that argument
is correct and irrelevant.

---

## Falsified: `sigma_res` was never the problem

Recorded because it cost a cycle and is an easy hypothesis to re-form. The
training log opens with `no sigma_res.json: sigma_res will be estimated from the
first batches` and `run audit_residual_targets.py first for a stable value`,
which makes mis-whitening look like the obvious suspect. It is not, on three
independent grounds:

**1. Measured directly.** Per-channel residual RMS over 3 boxes against the
checkpoint's `sigma_res`:

| channel | true RMS | `sigma_res` | whitened RMS |
|---|---|---|---|
| disp_x/y/z | 0.122 / 0.105 / 0.139 | 0.126 / 0.115 / 0.128 | 0.970 / 0.909 / 1.087 |
| vel_x/y/z | 1.008 / 0.953 / 1.061 | 0.951 / 0.911 / 0.963 | 1.060 / 1.047 / 1.103 |

Every channel whitens to within 10% of 1.0. The 8-crop fallback estimate was
good. **Running the target audit will not change anything.**

**2. The VP endpoint is scale-independent.** At `t = 0.999`,
`a_t = cos(pi t / 2) = 0.00157`, so `Var(u_t) = a_t² Var(u_0) + s_t² ≈ 0.999998`
for any plausible `Var(u_0)`. The starting distribution is standard Gaussian
whatever `sigma_res` is.

**3. The sampler recovers the data scale.** Feeding the analytic MMSE denoiser
for `u_0 ~ N(0, sigma²)` through the real `ddim_sample`:

| sigma | steps | out std | ratio |
|---|---|---|---|
| 1.00 | 20 / 100 | 0.940 / 0.988 | 0.940 / 0.988 |
| 0.32 | 20 / 100 | 0.288 / 0.313 | 0.899 / 0.979 |

`sigma = 0.32` comes back as 0.32; the residual deficit is DDIM discretisation and
closes at 100 steps.

**Where the misreading came from.** `diag_residual_rms_true` is *physical* (the
diagnostic un-whitens before measuring), *channel-mixed* (blending the 0.12
displacement channels with the 1.0 velocity channels), and computed on **one**
64³ core of **one** validation crop — `next(iter(val_loader))` returns the same
batch every time, so the value is byte-identical across steps. It cannot be
compared to per-channel `sigma_res`. The missing diagnostic was per-channel
`u_0 = dPsi / sigma_res` RMS, which is now the table above.

## Confirmed but bounded: the `eps`-prediction endpoint

DDIM reconstructs `x0_hat = (u_t - s_t eps_hat) / a_t`. At `t = 0.999` that
divides by 0.00157, so an `eps` error of 0.01 becomes an `x0` error of 6.4 — past
`x0_clip = 4`.

An **exact** denoiser clips 0.000 even at step 0, so a high clip fraction is a
genuine statement about model quality, not an inevitability of the endpoint.
Calibrating with a controlled `eps` error `delta`:

| delta | clip@0, `t_max`=0.999 | `t_max`=0.98 | `t_max`=0.95 |
|---|---|---|---|
| 0.01 | 0.531 | 0.000 | 0.000 |
| 0.05 | 0.898 | 0.011 | 0.000 |
| 0.10 | 0.949 | 0.207 | 0.001 |
| 0.30 | 0.982 | 0.677 | 0.291 |

The observed 0.90 implies `delta ~ 0.1-0.3` at `t ≈ 1` — where the task is
trivial (`u_t ≈ eps`, so the model need only echo its input). Consistent with the
zero-initialised head §1 of `reward_residual_diffusion.md` describes.

**But it is one step of twenty.** The measured profile is `0.90` at `t = 0.999`
then `0.01-0.02` for all remaining steps. And across the whole `(delta, t_max)`
sweep the sampler's output amplitude ran **0.82-0.95×** — never above 1. `eps`
error plus clipping *deflates*; it cannot produce the observed 1.9× overshoot.
v-prediction removes a real defect, not the failure.

---

## Process failures

**Gate B measured a model Gate A had already failed.** `candidates.json` was
written at 12:57; `gate_a.json` (`passed: false`, criteria 4/5/7/8) at 20:00. The
guard in `sample_reward_oracle.sbatch` had no file to read. The verdict file's own
note warns about precisely this. `submit_oracle.sh` also falls back from the
verdict job (`rw_gate_a`) to the *training* job (`rw_prior`) when it cannot find
the former in `squeue`, which permits the same race.

**Gate A criterion 5 was a false failure.** `diag_x0_clip_fraction_max` is simply
absent from v1's CSV header (older run), so it came out "never logged". v2 logs
it.

## Bugs found

| where | bug |
|---|---|
| `gate_a_check.py:126` | reads `audit["residual_rms"]`; the audit writes `per_box[*].stats.residual_rms`. Criterion 3 silently falls back to the last validation crop and **never uses the audit**, even after it is run. |
| `submit_oracle.sh` | Gate-A verdict → training-job fallback permits the race above. Should require the verdict job or an already-written passing `gate_a.json`. |
| `cem_select_elites.py:100` | sorts on `(feasible_field, R_occ)` with no empty-reliable-bin check, so a halo-*deleting* candidate scores `R_occ = -0.0` and ranks first. Reimplements Gate B's selection without Gate B's guard. Also pools purity-masked chunks rather than using the full-box summary `mu` was fitted on. |
| `train.py:628` | `ckpt_best` selected on `val_loss` alone. This run shows validation loss falling while every physical diagnostic worsens, so that criterion selects *for* the failure. |
| `train.py:499` | resume calls `model.load_state_dict`, and `sigma_res` is a registered buffer — a newly supplied value is silently overwritten by the checkpoint's, while metadata can record the new one. |
| Gate B output | `D2_hr` and `occupation_hr` for val boxes come from **chunk-pooled** summaries while `mu` is fitted on full-box ones. Pooling loses ~50% of subhaloes and 95% of `1e14` hosts (set0: `n_sub` 33066 → 16806, `n_host[4]` 21 → 1), which is why set10 reports `D2_hr = 26483` against `D2_base = 6916` — a true HR box scoring 4× worse than SR2. Fix: run `catalog_summaries.py --box set8..set11 --source hr`, as job 24279 already did for the train split. |

---

## Reproducing

```bash
# the diagnosis (GPU sample + CPU report, both arms, ~15 min each)
bash scripts/slurm/submit_residual_diagnose.sh all

# one arm only, or the sampler-endpoint arms
bash scripts/slurm/submit_residual_diagnose.sh v1
bash scripts/slurm/submit_residual_diagnose.sh tmax
```

Outputs, all redrawable without resampling:

```
$DMSR_REWARD_ROOT/audits/residual_diagnose_<tag>/
    residual_diagnose.jsonl           # per crop, per alpha, per arm
    residual_diagnose_summary.json    # aggregated + the verdict
    residual_diagnose.png
```

The verdict line in the report applies a fixed `r(k) > 0.10` floor to separate
"amplitude" from "direction". **That floor is too lenient** — v1 clears it at
0.330 and the report says "AMPLITUDE, set residual_scale ~ 0.53", which the sweep
in the same report overrules. Read the sweep, not the verdict line.
