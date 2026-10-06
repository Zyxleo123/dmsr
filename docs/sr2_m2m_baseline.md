# Running map2map's own trainer

**Status 2026-10-01. Three claims below this line were FALSIFIED by longer
training; they are kept, struck through in prose, because the way they failed is
the lesson.** Everything in the 2026-09-29 section was measured at epoch <= 7 of
a schedule the colleague's own script runs to 1024. Reading a 7-epoch window as
a steady state was the single recurring error of this line of work.

## What 27+ epochs actually showed (jobs 69291 / 70225 / 70254)

Running THEIR tree with THEIR command (`srsgan.G/.D`, `--lag2eul --mesh-up-fac
2`, crop 9, batch 4, `M2M_VAL=0`) to epoch 27:

| | density set9 / set10 | \|1-d\| |
| --- | --- | --- |
| **epoch 27** | **1.003 / 0.954** | **0.003 / 0.046** |
| frozen SR2 | 0.993 / 0.941 | 0.007 / 0.059 |

Scored as `|1 - ratio|` -- 1.0 is the target, so "higher" is NOT "better" --
**epoch 27 is closer to HR than frozen SR2 on both held-out boxes.** First time
this project has got there from scratch.

### The spectral cutoff moves with training

`k_half` = largest k where the candidate still holds half of HR's power. LR is
64^3, so k = 32 is the LR Nyquist: below it the net can pass structure through,
above it everything must be GENERATED.

| epoch | disp | vel | | epoch | disp | vel |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (MSE) | 24 | 14 | | 16 | 42 | 73 |
| 4 | 32 | 33 | | 19 | 64 | 55 |
| 7 | 31 | 63 | | 22 | **95** | 108 |
| 10 | 41 | 91 | | 25 | 64 | 114 |
| 13 | 48 | 125 | | 27 | 64 | **115** |
| | | | | **frozen SR2** | **192** | **256** |

So the net DOES learn to synthesise above its input's Nyquist -- disp 24 -> 64
(2x), vel 14 -> 115 (3.6x) -- and it is still far from frozen SR2's 192 / 256.

**The scatter is large**: disp 48 -> 42 -> 64 -> 95 -> 64, vel 125 -> 73 -> 55
-> 108. Swings of +-30 between adjacent epochs, so NO single-epoch comparison
means anything, and "95 at epoch 22" is not a better model than "64 at epoch
27".

### What this falsifies in the section below

* **"The run PEAKS at adversarial epoch 1 and falls after"** -- FALSE over a
  longer window. 0.744 at epoch 2 vs 1.003 at epoch 27. The apparent peak was
  two points inside the noise band.
* **"Velocity is dead / the objective cannot see velocity"** -- overstated.
  Velocity's HIGH-K AVERAGE is still ~0.026 at epoch 27 (frozen 0.96), but
  `k_half` shows it tracking HR to 3.6x the input Nyquist and rising. The two
  metrics measure different things: `ratio_highk` averages k ~ 170-256, which the
  model has not reached yet; `k_half` says where it stops. Both are right.
* **"The nets only upsample, generating nothing above k=32"** -- true of the
  epoch-2 and epoch-5 checkpoints it was measured on, false as a statement about
  the method.

### Our loss term, and a limit of the diagnostic

The kNN-CDF term is ported to THEIR tree (`--extra-criterion`, 5 sites in
train.py + 3 flags in args.py, all guarded). Arms branched from `state_27`:

| weight | gradient vs baseline, n=20 | IQR |
| --- | --- | --- |
| 3000 | 0.99x / 0.93x | 0.85-1.06 |
| 50000 | 1.04x / 0.97x | 0.86-1.13 |

A **16.7x weight increase moved nothing measurable**. But the honest reading is
NOT "the term is inert": the IQR sets a detection threshold of roughly **+-15%**,
so a term contributing 1% at w=3000 and 17% at w=50000 would sit at the edge of
what this test can see. "Inert" and "contributes up to ~15%" are not
distinguishable here, and calling the first arm "inert" was overconfident.

The loss-share readout cannot settle it either: `extra / |adv/G|` read 11%, 123%,
55%, 31%, 8% across five consecutive readouts while the term itself moved 17%.
`adv/G` is an unbounded WGAN score that wanders -3e3 to -4e4, so **no fixed
weight holds a stable share**. The decisive measure is a field score on matched
epochs, which needs enough divergence to clear a ~0.3 density noise floor.

---

**Status 2026-09-29 (superseded above).** The `xw/styled_srsgan` branch is not the code that
produced the colleague's result — four independent facts below say so. But the
conclusion first drawn from them, that the branch was therefore a dead end, was
**WRONG**, and the correction is the most important thing on this page.

With `M2M_VAL=0` (Finding 3's workaround) the branch trains, and it beat every
previous from-scratch run — then degraded. Held-out set9, same estimator:

| checkpoint | `||grad D||` that epoch | density | velocity | disp |
| --- | --- | --- | --- | --- |
| `state_1`, MSE only | — | 0.047 / 0.048 | 0.000 | 0.001 |
| `state_2`, adv epoch 1, gp16 | 21–27 | **0.744 / 0.679** | 0.000 | 0.003 |
| `state_3`, adv epoch 2, gp16 | 32–35 | 0.523 / 0.501 | — | — |
| `state_2`, adv epoch 1, gp**1** | 35–63 | 0.284 / 0.274 | 0.000 | 0.000 |
| trilinear | | 0.706 | | |
| frozen SR2, same boxes | | 0.993 / 0.941 | 0.960 | 0.403 |
| best previous from-scratch (our trainer) | | 0.26 | | |

Read three things off it.

**1. The adversary does all the work.** MSE-only leaves 5% of HR's small-scale
density power and essentially zero on the other channels; one adversarial epoch
reaches 75% of frozen SR2. The frozen row at 0.993 against the published 0.977
on set14 is the scorer validating itself.

**2. The run PEAKS at adversarial epoch 1 and falls after**, and density tracks
`||grad D||` monotonically across three points. So Finding 4 is neither harmless
nor fatal: a Lipschitz estimate 25x its target still produced 0.744, but the
drift is what ends the gain. Checkpoint selection is not optional here, and
arresting the drift is the top lever for beating 0.744. `--adv-wgan-gp-interval`
is NOT that lever (see below); `--adv-lr`, currently 2x the generator's, is the
obvious untried one.

**2b. The kNN term does not destabilise the critic.** MEASURED over 26 matched
steps (jobs 68916 / 68949): both arms' `||grad D||` climbs ~1.9 -> ~22 within the
one epoch, with no systematic separation. The drift is intrinsic to this config,
not something an added loss term causes. Compare arms at MATCHED steps -- an
offset comparison manufactures a difference that is not there.

**3. Velocity is dead.** It stays at 0.000 against frozen's 0.960, and
displacement at 0.003 against 0.403, in the same checkpoint that scores 0.744 on
density. The adversary builds density structure and leaves velocity untouched —
which is what reading density P/P_HR alone would hide, and it is the defect the
phase-space kNN term is aimed at.

Note set10 gives 0.679 at the same checkpoint, below trilinear, so "clears
trilinear" is a set9-only statement; the stable figure is ~72-75% of frozen on
both boxes.

## Why we went here at all

Four from-scratch runs on our own trainer failed for four different reasons, and
every one traced to a component we had reimplemented rather than to the
experiment (`sr2-scratch-critic-runaway`, and the header of
`scripts/slurm/submit_m2m.sh`). The response was to stop reimplementing and run
their trainer: their G, their D, their WGAN-GP, their `lag2eul`, their
hyperparameters, on our data. Ours would be the data layout and an identity norm,
both through map2map's own interfaces.

That premise is sound. It just turns out there is no "their trainer" to run.

## Finding 1 — the colleague's flags exist in no commit

Swept all 26 branches across both remotes (`eelregit/map2map`,
`sagasv5-xw/map2map`):

| flag in their sbatch | branches defining it |
| --- | --- |
| `--lag2eul` | **0 / 26** |
| `--mesh-up-fac` | **0 / 26** |
| `--ps-interval` | **0 / 26** |
| `--tgt-ps-path` | **0 / 26** |
| `--Lbox` | **0 / 26** |

`map2map/args.py:28` is `parser.parse_args()`, not `parse_known_args`, so an
unknown flag is a hard error rather than an ignored one. A whole-tree grep (not
just `args.py`) over the six candidate branches finds **0 files** mentioning any
of them. Their script, run verbatim, dies at argparse on every commit in both
repositories.

## Finding 2 — their model choice cannot run on this branch

Their script says `--model srsgan.G --adv-model srsgan.D`. On this branch
`train.py:356` is

```python
output = model(input, style)
```

passed unconditionally, and `adv_model(..., style=style)` likewise at 412, 420
and 451. But `srsgan.G.forward(self, x)` and `srsgan.D.forward(self, x)` take no
style — `srsgan.py` has zero styled references. So the letter of their script is
unrunnable here, which is why `M2M_MODEL` defaults to `styled_srsgan.G`. That
substitution is forced, and it means we are not running their configuration.

Of the seven branches that have WGAN-GP, only `wgan` and `2DBranch` call
`model(input)` without style — i.e. only there would `srsgan.G` work — and
`2DBranch` is 2-D. Neither has any of the five missing flags.

## Finding 3 — the branch cannot finish one adversarial epoch

`train()` builds D's input as `cat([lag2eul(out), out])` and then the `--cgan`
cat: 8 + 6 + 6 = **20** channels (`train.py:378-389`). `validate()` does **only**
the cgan cat, 6 + 6 = **12**, and hands it to the same D
(`train.py:590-595`). So the first epoch with `epoch >= adv_start` runs to
completion and then dies in validation:

```
RuntimeError: shape '[1, 20, 64, 64, 64]' is invalid for input of size 3145728
```

`3145728 = 12 * 64^3` — exactly the eight missing `lag2eul` channels. Job 68906,
40 minutes into epoch 2.

Nobody trained 1024 epochs with `--adv-start 1` on this code.

**Workaround in use:** `M2M_VAL=0` omits the three `--val-*` patterns, which is
how map2map itself disables validation (`args.py:229` sets `args.val` from
`val_in_patterns is not None`). No code edit. The val MSE is no loss — MSE is the
wrong yardstick for this line anyway.

## Finding 4 — the critic never held its Lipschitz constraint

`wasserstein.py:48` is `penalty = 10 * ((||grad D|| - 1) ** 2).mean()`, and
`train.py:440` logs it **unscaled** — the `* args.adv_wgan_gp_interval` factor
goes only into the tensor that gets `.backward()`. So the logged number converts
directly to a Lipschitz estimate, `1 + sqrt(reg / 10)`:

| point in epoch 2 | logged `adv/reg` | implied `||grad D||` |
| --- | --- | --- |
| start | 150–250 | 5–6 |
| +1000 steps | 700–1340 | 9.4–12.6 |
| +2000 steps | 1672 | 13.9 |
| +3000 steps | 5997 | **25.5** |

Target is 1. It compounds rather than settling, and the generator's
`grad/adv/first` spikes to 3e4 alongside it.

Two things make this worse than it looks. The generator steps on the
**adversarial loss alone** after `adv_start` — `loss.backward()` is only in the
`else` branch (`train.py:448-457`), so there is no amplitude anchor at all, which
is their design, not our reimplementation. And the constraint is estimated from
**one interpolate every 16 steps**: `--adv-wgan-gp-interval 16` with effective
batch 1, under `betas=(0.0, 0.99)`, i.e. no gradient momentum.

### batch size cannot be raised on one GPU

`--batches` is per-GPU (`args.py:83`), so their 4 GPUs x `--batches 1` is an
effective batch of 4 and our single GPU is 1. Raising `--batches` to 4 does not
close that gap — it crashes:

```
RuntimeError: Given weight of size [2048, 6, 1, 1, 1], expected bias to be
1-dimensional with 2048 elements, but got bias of size [512] instead
```

`ConvStyled3d` (`style.py:176-181`) does per-sample styled convolution as a
grouped conv: it reshapes the **weight** to `[N*C_out, C_in, ...]` with
`groups=N` but leaves `bias=self.bias` at `[C_out]`. **`styled_srsgan` runs only
at batch size 1.** Their 4-GPU layout was not a choice about gradient noise; it
is the only way their model reaches effective batch 4. Job 68910.

## What is running now

| job | arm | `--adv-wgan-gp-interval` | note |
| --- | --- | --- | --- |
| 68916 | `m2m_baseline` | 16 (their value) | resumed from epoch-1 checkpoint |
| 68917 | `m2m_gp1` | 1 (map2map's default) | fresh |

Both with `M2M_VAL=0`, which is what lets them past epoch 2 at all — 68916 has
now written `state_2.pt`, the first completed adversarial epoch on this branch.

**RESULT: `--adv-wgan-gp-interval 1` is WORSE**, at matched adversarial step
counts into epoch 2:

| interval | ~470 adv steps | later |
| --- | --- | --- |
| 16 | reg 150–250, `||grad D||` 5–6 | 3995–6582 -> 21.0–26.7 |
| 1 | reg 4466, `||grad D||` **22.1** | 11870 -> **35.5** |

So the lazy schedule is not the cause and tuning it is a dead end. The reason is
**Adam**, not the raw force: `adv_loss_reg * interval` compensates exactly for
frequency in the gradient, but map2map accumulates the GP into the SAME buffers
as the adversarial loss and takes one `adv_optimizer.step()`
(`train.py:428-444`) under `betas=(0.0, 0.99)`. At interval 16 the burst is
`adv + 16*gp` while the second moment `v` still reflects 15 quiet steps, so
`sqrt(v)` lags and the burst lands as a real correction; at interval 1 the term
is 1/16 the size every step, `v` absorbs it, and it becomes a minor perturbation
of the adversarial direction. This is why StyleGAN2 applies lazy regularization
in a SEPARATE optimizer step rather than accumulating it.

That this cannot be fixed from the GP schedule sharpens question 2 below: with
no amplitude anchor on the generator, a power-spectrum loss behind
`--ps-interval` is the most likely thing keeping their run in bounds.

Epoch-2 averages from `state_2.pt`'s `min_loss` (map2map stores the whole
`epoch_loss` vector, not a scalar): MSE **0.611** against epoch 1's 0.219,
G adversarial -6770, D fake +6822, D real -13436.

Score checkpoints on **density P/P_HR**, never on the loss: frozen SR2 is 0.977,
trilinear is 0.706, and our best from-scratch run so far is 0.26.
`sr2-reproduction-baseline-yardstick` has the full argument.

## What to ask the colleague

1. **The commit.** Branch and SHA, or a tarball. Five flags in the script exist
   nowhere in either public repo, so this is a local fork.
2. **What `--ps-interval` / `--tgt-ps-path` / `--Lbox` do** — a power-spectrum
   loss term, or only logging? If it is a loss, it is a second objective we do
   not have and the adversarial-alone generator step would no longer be
   unanchored, which would explain Finding 4 going away for them.
3. **Whether their `validate()` computes `lag2eul`.** If it does, their fork has
   fixed Finding 3, which is further evidence it is a different tree.
4. **Whether `srsgan.G` took a style argument in their tree**, or their
   `train.py` called `model(input)` — Finding 2 either way.

## The loss terms

`src/cosmo_sr/features/knn_crop.py` (+ `tests/features/test_knn_crop.py`, 20
tests) computes the kNN-CDF term from `(output, target)` in the batch, with no
tile id and no catalog. This is trainer-agnostic and survives whatever happens
above.

Why it needs no index: map2map's `perm` permutes vector **components** together
with the spatial axes and `flip` negates the components it flips
(`data/fields.py:414-430`), so both are proper isometries — `q_c -> L - q_c` and
`d_c -> -d_c` give `p_c -> L - p_c`. Pairwise distances are preserved, the
velocity block gets the same treatment, so the full **6-D phase metric** is
invariant, not just the positional one. A kNN log-distance distribution is a
function of that metric alone. The tests drive map2map's own `flip`/`perm`
rather than a reimplementation, because a reimplementation would be testing the
paragraph rather than the code.

Two consequences:

* The reference is the **target of the same batch element**, which went through
  the same augmentation as the candidate. Nothing stored can fall out of step.
* **The crop-8 / no-`--aug-shift` geometry is retired.** It was chosen so these
  terms would stay indexable by `(box, tile)`; they do not need to be. The
  colleague's literal `--crop 9 --crop-step 9 --aug-shift 9` is available again.

Phase space is the default (`phase_lambda = 4000`), not `KnnConfig`'s 0, because
round 5b measured the pooled position-only statistic ranking arms at **+0.91**
against region subhalo counts — the wrong sign, it is dominated by host and
unbound particles — while `v / 4000` flips it to **-0.65** (p 0.002) with no
labels, which is the only kind of fix available on an unlabelled crop
(`knn-phase-space-round5`).

### Running it: the hook, and getting the weight right

`--extra-criterion srlosses.KnnCdf` (+ `--extra-criterion-args`,
`--extra-weight`) is the only change to their repo beyond a numpy-2 fix: 66
lines across `args.py` and `train.py`, all flag-guarded, so absent the flag the
loop is unchanged. It had to be an edit rather than a callback because the
generator steps on the ADVERSARIAL loss alone after `--adv-start`, so a term
plugged into `--criterion` computes, logs, and never reaches the optimizer.
The hook captures the RAW narrowed 6-channel tensors before `lag2eul`/`--cgan`
reassign them, and adds the term to `loss_adv` (adversarial path) and to `loss`
(pre-adversary path).

Cost: **171 ms/step baseline -> 281 ms/step**, a 1.64x slowdown at 256 queries
on an a6000. Critic stability is unaffected -- `||grad D||` tracks the baseline
at matched steps (14.0 vs 14.3 at step ~7400).

**`--extra-weight 1.0` is a NO-OP and cost a GPU-hour to discover.** The term
logs 0.02-0.54 against an `adv/G` of -1000 to -37000, i.e. **1e-4 to 1e-6** of
the objective -- the `loss-budget-is-hinge-geometry` failure again (d6 88%,
bound 0.24%). At `--extra-weight 3000` it sits near 45%.

Three ratios in this line turned out to need caveats rather than conclusions,
all failing the same way -- quoted before the denominator was understood:

* `|extra| / |adv/G|` is **not a budget**. `adv/G` is `-(critic score).mean()`
  and `styled_srsgan.D` has no normalisation anywhere, so the critic's output
  scale is arbitrary. It also CROSSES ZERO (-37620 to +239 inside one epoch),
  and the ratio then explodes on a vanishing denominator: a reading of "388% of
  adv" was `advG = +239.7`, not the term growing. Only read it when `|adv/G|`
  is large.
* The generator-gradient ratio between arms does not settle. Over n = 3 to 46
  matched steps the two estimators (median-of-ratios, ratio-of-medians) sat
  anywhere in 0.47-1.55 and repeatedly disagreed; the traces spike 10-20x within
  an epoch. Treat it as a stability check only.
* Even converged, that ratio is **confounded**: the arms share only their
  epoch-1 weights, so after a few hundred steps a matched STEP is not a matched
  STATE, and a ratio below 1 is equally "partial cancellation" or "the runs
  drifted apart".

So the only clean causal measurement is the epoch-2 score from a shared epoch-1
start. Three arms, identical `state_1`: baseline (0.744 / 0.679, velocity
0.000), kNN at weight 1 (inert -- therefore a RUN-TO-RUN NOISE datum, which is
what makes the third interpretable), kNN at weight 3000.

### RESOLVED — the member-gather term, and what it cost

**Built and run 2026-10-04/05.** The section below framed the choice as (a) thread
the tile id and augmentation axes out of their dataset or (b) derive host centres
on the fly. Neither was taken: the term **runs its own forward** on a fixed,
canonical, unaugmented batch of known tiles and ignores the `(output, target)` it
is handed. `train.py` gained one line
(`if hasattr(extra_criterion, 'set_model'): extra_criterion.set_model(model)`);
their dataset is untouched. See `scripts/m2m_callback/srlosses.py::MemberGather`.

Geometry: `srsgan.G` is 3 `SkipBlock`s, each `n -> 2n-6`, so `out(n) = 8n - 42`
(check: crop 9 + pad 3 = 15 in, and 8*15-42 = 78 is the shape their log prints).
A 64^3 tile inverts to a **14^3 LR window -> 70^3 -> `narrow_by(_, 3)` -> 64^3**,
the window starting at `8t - pad` so the tile's own 8 LR cells land at the central
`[3, 11)`. Asserted at runtime; the alignment is unit tested.

Two of the three minimal-set terms had to change reference, because
`centre_mode="self"` and the low-k guard both name the FROZEN field and a
from-scratch run never was there: `centre_mode="full"` (offset to the HR centroid)
and `sr2_scratch.lowk_anchor` (same operator against HR). `virial` is unchanged.

**The verdict is in `docs/sr2_m2m_halo_verdict.md` and it is negative**: the term
loses to the plain baseline on displacement `k_half`, on whole-box satellite
accuracy and on radial placement. Its own objective converged 28x while the field
got worse, so satisfying it is what costs the displacement spectrum.

### The original framing, kept for the record

`member_gather_loss` needs `MemberSets`, which is Rockstar-derived, and a random
64^3 crop has no catalog. It does not port the way the kNN term did:

* `FieldDataset.__getitem__` computes `ifile, icrop = divmod(idx, ncrop)` and
  returns `target_relpath` — the box — but never `icrop`.
* `train.py` constructs `FieldDataset` directly rather than through
  `import_attr`, so the class cannot be swapped via `--callback-at` either.
* `flip_axes` / `perm_axes` are local too, so even a table keyed by
  `(box, icrop)` would be in the wrong frame.

So the options are (a) precompute per-tile sets and thread the tile id **and**
the augmentation axes out of their dataset, or (b) derive host centres on the fly
from the HR target in the crop, which is index-free and augmentation-safe but is
not the Rockstar-supervised object the gate was built on. These are not
equivalent and the choice changes what gets built. Unresolved, pending a
decision.

### Integrating the term into their loop

No clean callback route exists. `--criterion` receives exactly the raw
`(output, target)` we want (`train.py:371`, before the reassignment) but is
**not** backpropagated after `adv_start`. `--adv-criterion` sees only the
critic's scalar score. A custom `--adv-model` would see the field, but it is
called at three sites — the detached D step, the GP interpolates, and the live G
step — and cannot reliably tell which. So the hook is a small additive edit to
`train.py`, guarded by a flag that defaults to off. **Not written yet**, because
which trainer it targets is exactly what Findings 1-3 put in doubt.
