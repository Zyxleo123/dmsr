# Why SR displacement looks fine but density is under-clustered

Investigation of the Stage 0–6 plan. Status as of 2026-07-25.

Headline numbers under investigation (full-box `compare_ceiling_unc_set14`, set14,
512³ periodic, from `t13_unconstrained_s0`):

| method | disp high-k P/P_HR | density high-k P/P_HR | density σ ratio |
|---|---|---|---|
| ours (unconstrained) | 0.487 | 0.282 | 0.766 |
| trilinear | 0.036 | 0.706 | 0.973 |
| SRS | 0.396 | 0.977 | 1.000 |

## Conclusion: Branch A, with a correction that must land first

Two defects were found and measured. Neither is "the model needs more capacity".

1. **The crop-level density evaluator and the density critic are computed on a
   scrambled field.** This is a bug, not a modelling problem, and it invalidates
   every crop-level density result including the signal the `pshuffle8` density
   critic is trained against.
2. **The LR conditioning the model is trained on is context-truncated**, and 93%
   of that truncation is a GroupNorm normalization artefact rather than genuinely
   missing information.

Branch A (context-in / center-out) is the right direction, but its ablation list
should be reordered: **normalization is the bigger term and is nearly free**,
larger context is the smaller and more expensive one. And the CIC fix gates
everything — until it lands, the density critic is optimising the wrong target.

## What was actually run

This session had no GPU (`gpu22` reports no devices), 2 CPU cores, and a hard
**~1.0 GiB per-process memory cap** (measured by bisection; unrelated to the
187 GiB of free RAM, and it silently SIGKILLs). So:

- **Ran to completion:** Stage 2 (full), Stage 3 restricted to the LR-encoder
  pathway, and the structural/architectural audit.
- **Built and validated, needs GPU:** Stages 0, 1, 3 (full ODE path), 4, 6.
  Launch commands at the bottom.

---

## Stage 2 — CIC buffer audit: **GATE FAILED**

`cosmo_sr.eval.density.cic_density` deposits a crop's particles with `% ng`, i.e.
it wraps them *inside the crop*. Its docstring calls that "an approximation".
It is not.

On set14 the **median** particle displacement is **36 HR cells** and the max is
**112**, against a **64³ HR training crop**. Only **9.75%** of a region's own
particles stay inside it.

Scored region 64³ at [224,224,224], reference = 64-cell buffer (verified
identical to the provably-exact 115-cell buffer):

| mode | rel RMS vs converged | σ/σ_ref | P_high-k ratio | **corr vs truth** | mass/uniform |
|---|---|---|---|---|---|
| **wrap in crop** (current) | **2.357** | **2.216** | 1.891 | **0.080** | 1.000 |
| buffer 0 | 0.849 | 0.277 | 0.072 | 0.652 | 0.097 |
| buffer 16 | 0.446 | 0.849 | 0.754 | 0.897 | 0.416 |
| buffer 32 | 0.129 | 0.986 | 0.980 | 0.992 | 0.636 |
| buffer 48 | 0.017 | 0.999 | 1.000 | 1.000 | 0.696 |
| buffer 64 | 0.000 | 1.000 | 1.000 | 1.000 | 0.704 |

The current construction produces a field whose error is **2.4× the signal** and
which is **essentially uncorrelated (r = 0.08)** with the true density of that
region, with σ inflated 2.2×. Convergence needs a **64 HR-cell buffer**.

Across 7 random crops the picture is more nuanced and worth stating precisely:

- voxelwise correlation with truth: mean **0.407**, range 0.062–0.814
- σ_wrapped/σ_true: mean **3.13**, range **0.60–12.19**
- but rank correlation of σ across crops: **0.964**

So the wrapped field retains a usable *relative ordering* of clumpiness between
crops, while being badly wrong voxelwise and in absolute scale. A convolutional
critic sees voxelwise structure, so the relevant number is 0.41 (worst 0.06).

**Consequences.** `HighPassDensity.density()` and `.density_pshuffle()`
(`src/cosmo_sr/dmsr/density.py:142,155`) call `cic_density` on 64³ training
crops. The `pshuffle8` density critic is therefore trained against this field.
All `val_density_*` metrics in `metrics.csv` are computed the same way.
The **full-box** `compare_ceiling_*` numbers are unaffected — there `% ng` is the
true box wrap, so those are valid.

Deliverables: `scripts/dmsr_cic_buffer_audit.py`,
`runs/dmsr/stage2_cic_buffer/{buffer_audit.json,buffer_convergence.png}`,
6 regression tests in `tests/test_cic_buffer.py` (all passing).

## Stage 2c — how to fix it, and what it costs

The 64-cell buffer above is the answer to "score a *fixed* region", which is the
wrong question for designing a training crop. Particles from a Lagrangian crop
``B`` land in the Eulerian cube centred at ``centre(B) + <Psi>_B``, shrunk by the
*internal* spread of ``Psi - <Psi>_B``. We are free to score that offset cube.

A rigid translation is an exact symmetry of the deposit, so subtracting the bulk
*and* moving the target changes nothing (measured: the two columns come out
bit-identical — an early version of this experiment was a tautology). The only
thing that buys back volume is that the internal spread is much smaller than the
displacement itself.

Within-crop spread of ``Psi`` (mean over random crops, HR cells):

| crop | crop_lr | max | p99.9 | rms |
|---|---|---|---|---|
| 64³ | 8 | 53.6 | 45.8 | 15.0 |
| 128³ | 16 | 74.9 | 53.6 | 18.1 |
| 192³ | 24 | 70.0 | 54.8 | 18.0 |

The worst-case bound is very conservative (rms is ~4× smaller than the max), so
what matters is the measured accuracy of the offset deposit versus the scored
fraction. **On the existing 64³ crop:**

| scored R | R/C | rel RMS | corr | σ ratio | mass kept |
|---|---|---|---|---|---|
| **current (wrapped, R=64)** | 1.00 | **2.357** | **0.080** | **2.216** | 1.000 |
| offset, R=64 | 1.00 | 0.297 | 0.955 | 0.925 | 0.817 |
| offset, R=48 | 0.75 | 0.055 | 0.9985 | 0.998 | 0.963 |
| **offset, R=32** | 0.50 | **0.005** | **1.0000** | 1.000 | 0.996 |
| offset, R=16 | 0.25 | 0.000 | 1.0000 | 1.000 | 1.000 |

So the fix is **much cheaper than a crop enlargement**. Keeping `crop_lr=8` and
scoring the central 32³ takes the density from *uncorrelated with reality*
(r = 0.08) to *exact* (r = 1.0000), at the cost of supervising 1/8 the volume per
crop. Even scoring the full 64³ with the offset — literally zero cost — reaches
r = 0.955. For reference, a 128³ crop scoring 64³ gives r = 0.9995.

## Stage 3 — context truncation: confirmed on the conditioning pathway

The LR encoder is the only non-local route by which LR information reaches the
flow. Feeding the *identical* central 8³ LR region inside progressively wider
windows and comparing the central conditioning features against their full-box
values (`t13_unconstrained_s0`, set14):

| LR context | rel RMS vs full box | corr | residual after per-channel affine | fraction of error that is affine |
|---|---|---|---|---|
| **8³ (training crop)** | **0.948** | **0.474** | 0.257 | **92.7%** |
| 12³ | 0.660 | 0.747 | 0.138 | 95.6% |
| 16³ | 0.493 | 0.878 | 0.068 | 98.1% |
| 24³ | 0.276 | 0.970 | 0.035 | 98.4% |
| 32³ | 0.149 | 0.991 | 0.016 | 98.9% |
| 64³ | 0 | 1.000 | — | — |

Two separable causes:

- **GroupNorm (dominant, ~93%).** `nn.GroupNorm` reduces over `(C/g, D, H, W)`,
  so every output voxel depends on the statistics of the whole window. Verified
  directly. This makes the dependency crop-global regardless of kernel reach, and
  it is why the discrepancy is almost exactly a per-channel gain/offset.
- **Genuinely missing spatial information (~26% rel RMS at 8³, 7% at 16³).** The
  encoder's dilations `(1,2,4)` give a **15-voxel receptive field on an 8³ crop**,
  with `padding_mode="circular"` — so at training time it wraps each crop onto
  itself roughly twice. It also refuses windows smaller than 5 outright.

Central predictions keep changing well past 8³, which is the plan's stated
evidence threshold for context truncation.

## Structural findings affecting the existing results

- **The velocity U-Net uses zero padding** (`padding="same"` →
  `padding_mode='zeros'`, `flow_unet.py:102`), while the LR encoder uses circular.
  Every training crop face is convolved against a fictitious vacuum.
- **Full-box tiled inference is not equivalent to a full-box forward.**
  `_dmsr_generate_tiled` claimed `halo*scale >= 64` makes each tile "identical to
  the full-box forward" because the model has only local ops. GroupNorm makes that
  false. Training used 8³ LR windows; that function is normally called with
  `tile=16, halo=8` → **32³ LR windows**, where the encoder features differ from
  their 8³ counterparts by ~15% rel RMS. **The headline 0.282 therefore carries a
  train/eval normalization shift on top of whatever the model gets wrong.**
  Docstring corrected; `tile`/`halo` now recorded in the results JSON.

## Corrections made to my own tooling

Both were caught by the regression tests, and both had been producing wrong
numbers before the fix:

- CIC deposit tested cell indices against the region *before* periodic wrapping,
  dropping the `floor(u)+1` corner for particles just below a face — a ~3/R
  surface mass deficit. Stage 2 was re-run after the fix (conclusions unchanged).
- Helmholtz split was non-Hermitian at Nyquist (`fftfreq` reports `−n/2`, whose
  mirror is itself), so `irfftn` silently discarded **7%** of the longitudinal
  energy and `|L|²+|T|² = |d|²` failed (0.954). Zeroing the direction-ambiguous
  Nyquist bin restores it to 1 − 2e−7, with transverse divergence ~1e−7.

---

## What to run next, in order

Prerequisite for anything density-related: **fix the CIC path**. `cic_density`
should stay as-is for full-box use, but every crop-level caller needs the
buffered construction (`cic_block_into_region`). That includes the density critic
— retraining `pshuffle8` against the current target is not worth the GPU time.

All commands run from `cosmo_sr_project/`, on a GPU node (per the cluster rules,
launch these yourself; `srun` only from `rhea`).

**Stage 0** — put `pshuffle8` (now finished, step 20000) on the same full-box
ruler, and cache the fields so Stages 1 and 6 do not regenerate them:

```bash
python scripts/compare_flow_baseline.py \
  --lr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr/set14.npy \
  --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set14.npy \
  --dmsr unconstrained:runs/dmsr/t13_unconstrained_s0/config.yaml:runs/dmsr/t13_unconstrained_s0/ckpt_best.pt \
  --dmsr pshuffle8:runs/dmsr/t13_unc_fulldisp_pshuffle8_l003_s0/config.yaml:runs/dmsr/t13_unc_fulldisp_pshuffle8_l003_s0/ckpt_best.pt \
  --lr-crop 64 --dmsr-tile 16 --dmsr-halo 8 --n-steps 20 --diversity 1 \
  --save-fields runs/dmsr/fields_set14 --out runs/dmsr/compare_ceiling_pshuffle8_set14
```

Worth adding: a second run with `--dmsr-tile 8 --dmsr-halo 8` (16³ LR windows).
Comparing the two isolates how much of the density deficit is the GroupNorm
train/eval shift rather than the model.

**Stage 3 (full ODE path)** — confirms the encoder result end to end:

```bash
python scripts/dmsr_context_oracle.py \
  --config runs/dmsr/t13_unconstrained_s0/config.yaml \
  --ckpt   runs/dmsr/t13_unconstrained_s0/ckpt_best.pt \
  --lr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr/set14.npy \
  --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set14.npy \
  --contexts 8 12 16 24 32 --mode generate --n-steps 20 --report-gn \
  --out runs/dmsr/stage3_context
```

Noise is position-hashed, so the central region gets byte-identical noise at every
context size (verified: overlapping windows agree exactly, periodic wrap agrees,
mean 0.00 / std 1.00).

**Stage 4** — receptive field, with and without the normalization ablation:

```bash
python scripts/dmsr_receptive_field.py \
  --config runs/dmsr/t13_unconstrained_s0/config.yaml \
  --ckpt   runs/dmsr/t13_unconstrained_s0/ckpt_best.pt \
  --lr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr/set14.npy \
  --window 16 --n-steps 2 4 8 --out runs/dmsr/stage4_rf
python scripts/dmsr_receptive_field.py ... --ablate-norm channel --out runs/dmsr/stage4_rf_chnorm
```

With GroupNorm active, r95 should come out at the window boundary; the ablation
gives the true convolutional radius. That difference is the quantitative case for
the normalization change.

**Stage 1** — residual destruction, from the cached fields (no regeneration):

```bash
python scripts/dmsr_residual_alpha.py --fields runs/dmsr/fields_set14 \
  --label unconstrained --alphas 0 0.1 0.25 0.5 0.75 1.0 1.25 \
  --components full long trans --out runs/dmsr/stage1_residual_alpha
```

**Stage 6** — best-of-K. Note the spec's K=32 full-box draws is ~170 GPU-hours per
box; scoring a 64³ region with the Stage-2-validated 64-cell buffer needs only a
24³ LR window per candidate:

```bash
python scripts/dmsr_best_of_k.py \
  --config runs/dmsr/t13_unconstrained_s0/config.yaml \
  --ckpt   runs/dmsr/t13_unconstrained_s0/ckpt_best.pt \
  --lr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr/set14.npy \
  --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set14.npy \
  --train-hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set0.npy \
  --K 32 --regions 4 --temperatures 0.8 1.0 1.2 --out runs/dmsr/stage6_bok
```

## Implemented (config-gated, defaults unchanged)

All three are off by default, so existing configs and checkpoints are unaffected.

| config key | default | effect |
|---|---|---|
| `critic.valid_center` | `0` | `>0` scores that central Eulerian cube with the bulk-offset deposit instead of the wrapped one. `32` is exact on a 64³ crop. |
| `model.norm` | `"group"` | `"channel"` uses `ChannelGroupNorm3d` — group statistics over channels only, never over space. |
| `model.padding_mode` | `"zeros"` | `"circular"` matches the periodic boxes and the LR encoder. |

New code: `cosmo_sr.eval.density.cic_density_valid_center`,
`cosmo_sr.models.flow_unet.ChannelGroupNorm3d`, `HighPassDensity(valid_center=...)`,
and `critic_input` now centre-crops the residual view to the density view (raising
rather than broadcasting on a mismatch).

Tests: `tests/test_cic_buffer.py` (6), `tests/test_density_valid_center.py` (10),
`tests/test_flow_unet_norm_padding.py` (13) — all passing. The padding test asserts
translation-equivariance on a periodic input, which circular satisfies and zero
provably does not.

## Recommended Branch A ablation order

Evidence-weighted, cheapest-decisive first:

1. **`critic.valid_center: 32`** — turns the density critic's target from
   r = 0.08 to r = 1.0000 against the truth. No crop change, no extra compute.
   Gates everything else, because until it lands the critic optimises a scrambled
   field. (`valid_center: 64` is the zero-cost half-measure at r = 0.955.)
2. **`model.norm: channel`** — the 93% term in the context error, free.
3. **`model.padding_mode: circular`** — matches the periodic data and the encoder.
4. **Larger context (16³–24³ LR in, 8³ out)** — the remaining ~26%→7% structural
   term, and the only expensive item. Note a 128³ crop scoring 64³ also gives an
   essentially exact density (r = 0.9995), so this and (1) compose well.
5. Deformation features (divergence, shear/tidal tensor) to the critic. These are
   local differential quantities and need **no** buffer at all, which makes them
   the cheap complement to any Eulerian density term.

Ablations 1–3 are each a one-line config change on the existing `t13` setup, and
the configs are written and verified to resolve:

| arm | config | change vs `t13_unc_fulldisp_pshuffle8_l003` |
|---|---|---|
| baseline | `t13_unc_fulldisp_pshuffle8_l003.yaml` | (already run, step 20000) |
| 1 | `t13_fix_vc32.yaml` | `critic.valid_center: 32` |
| 2 | `t13_fix_vc32_chnorm.yaml` | + `model.norm: channel` |
| 3 | `t13_fix_vc32_chnorm_circ.yaml` | + `model.padding_mode: circular` |

```bash
for arm in t13_fix_vc32 t13_fix_vc32_chnorm t13_fix_vc32_chnorm_circ; do
  python -m cosmo_sr.train.train_dmsr --config configs/dmsr/$arm.yaml
done
```

Do not add hard per-LR-cell density consistency: the earlier null-space gate
already refuted it, and Stage 2's mass bookkeeping reinforces why (a region holds
0.70 of a uniform share; the constraint is not even well posed per cell).

## Sweep result (2026-07-26, all four arms at step 20000)

**No arm has been put on the full-box ruler yet, so the question the sweep was
run to answer is still open.** What follows is what the training-time metrics do
and do not license.

### The density columns of `metrics.csv` are not comparable across the boundary

`HighPassDensity` is constructed once in `train_dmsr.py:456` and handed to *both*
`critic_input` and `validate` → `evaluate_batch`, which calls `highpass.density()`.
So `critic.valid_center: 32` changes the validation instrument as well as the
critic's target: arms 1–3 score a 32³ offset cube, the baseline scores a 64³
wrapped one. Different field, different grid, band edges at different physical
scales. Any base→arm difference in `val_density_*` measures the change of
instrument, not the change of model. They remain comparable *among* arms 1–3.

The displacement columns involve no CIC deposit and are comparable everywhere.

### Displacement — comparable across all four

| metric | base | vc32 | +chnorm | +circ |
|---|---|---|---|---|
| `val_mse` | 0.01609 | 0.01632 | 0.01598 | 0.01631 |
| `val_rk_high` | 0.8374 | 0.8362 | 0.8345 | 0.8270 |
| `val_Tk_error_high` | **0.0688** | **0.0702** | **0.0898** | **0.0962** |
| `val_Tk_error_transition` | 0.0230 | 0.0229 | 0.0384 | 0.0386 |
| `val_condition_shuffle_gap` | 0.0664 | 0.0686 | 0.0649 | 0.0570 |

`valid_center: 32` is neutral on displacement, which is what it should be — it
changes only what the density channel of the critic sees. `model.norm: channel`
and `model.padding_mode: circular` each cost real displacement power: the high-k
power error rises 30% then 40%, and the transition band 67%. That is a genuine
regression on a metric that means the same thing in all four runs.

### Density — among the valid-center arms only

| metric | vc32 | +chnorm | +circ |
|---|---|---|---|
| `val_density_power_error` | 0.425 | 1.047 | 1.019 |
| `val_density_Tk_error_high` | 0.206 | 0.402 | 0.393 |
| `val_density_pdf_error` | 0.0375 | 0.0572 | 0.0559 |

Same ordering: `vc32` alone is the best of the three, and the two architectural
changes hurt.

### The critic separates much better once its target is correct

Tail mean over the last 200 steps:

| | base | vc32 | +chnorm | +circ |
|---|---|---|---|---|
| real − fake score | 0.974 | **1.647** | 1.460 | 1.154 |
| `loss_D` | 1.190 | 0.844 | 0.978 | 1.154 |
| `grad_ratio_adv_flow` | 0.391 | 0.595 | 0.611 | 0.605 |

This is the clearest positive signal in the sweep and it is consistent with the
Stage 2 diagnosis: with a scrambled target (r = 0.08) the density channel carried
almost nothing the critic could exploit, so the real/fake gap stayed small. With
the correct target the gap grows 69% and the adversarial gradient roughly doubles
its share. Whether that better-informed critic actually moves the full-box density
is exactly what Stage 0 measures and this table cannot.

### Reading

- **Keep `valid_center: 32`.** Free on displacement, better density among
  comparable arms, and it repairs a target that was demonstrably wrong.
- **`model.norm: channel` and `padding_mode: circular` did not pay off** at this
  budget. Two readings are open, and Stage 4 distinguishes them: either the
  spatially-global GroupNorm statistics were doing useful work (an implicit
  large-scale conditioning path the local layer removes), or 20k steps is not
  enough to re-converge after changing every normalisation layer. Stage 0b is the
  other half of the test — if `chnorm` is the arm least sensitive to window size,
  the change did what it was designed to do even while scoring worse overall.
- **Do not conclude anything about the headline 0.282 from this table.** Run
  Stage 0.

## Running the evaluation: `submit_dfix.sh`

Per the project rule that all compute is submitted, the stage commands above are
now SLURM jobs. GPU work and artefact-only aggregation are separate jobs, so a
table never queues behind a generation.

```bash
cd /zfsauton2/home/yixiz/DMSR/cosmo_sr_project
DRY=1 bash scripts/slurm/submit_dfix.sh          # inspect the chain first
bash scripts/slurm/submit_dfix.sh                 # submit everything
STAGES="0 0b table" bash scripts/slurm/submit_dfix.sh   # the decisive part only
```

| job | partition | what |
|---|---|---|
| `dfix_stage0_ceiling.sbatch` | `general` (GPU) | full-box ruler, all four arms, caches fields |
| `dfix_stage0_tile8.sbatch` | `general` (GPU) | same ruler at the training window size |
| `dfix_stage3_context.sbatch` | `general` (GPU) | context oracle, full ODE |
| `dfix_stage4_rf.sbatch` | `general` (GPU) | receptive field, ± norm ablation |
| `dfix_stage6_bestofk.sbatch` | `general` (GPU) | best-of-K, auxiliary |
| `dfix_stage1_alpha.sbatch` | `cpu` | residual α, from Stage 0's cached fields |
| `dfix_table.sbatch` | `cpu` | `summary.md` / `.json` / `.png` |

Defaults live once in `scripts/slurm/_dfix_common.sh`; every knob is an
environment variable passed through `sbatch --export`. Only Stage 1 depends on
Stage 0 (it reads the cached `.npy`); the other GPU stages are siblings and queue
concurrently. Missing inputs exit 0 with a printed reason so dependents still
start and report their own skip. The table is redrawable on its own at any time:

```bash
sbatch scripts/slurm/dfix_table.sbatch
```

The headline to watch is `density_highk_pk_ratio` (baseline 0.282, SRS 0.977)
**with `highk_power_ratio` held at ~0.487** — a density gain that costs
displacement power is the mean-collapse failure returning, not progress.
`summary.png` plots exactly that plane, with trilinear and SRS marked.
