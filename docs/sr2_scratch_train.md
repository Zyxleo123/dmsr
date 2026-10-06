# Training SR2 from scratch, with the structure terms in from step 1

Status: **RUN 1 FAILED on the critic; fixed and awaiting run 2 (2026-09-23).**
Built 2026-09-22. Shakeout job 62066 passed; arms 62067-62070 launched and the
adversary ran away — diagnosis, cause and fix in **section 8b**, which is the
only part of this document reporting a result of this line. Everything else is
either *measured* from an artifact on disk (including this repository's earlier
runs) or *design*: a proposed rule, not yet run. The radial verdict the whole
set exists for has not been measured.

## 1. The defect, and why a fine-tune cannot be the answer

Three measurements, all held-out, all already on disk:

| | median r / R_vir | frac inside 0.25 R_vir |
| --- | --: | --: |
| HR | **1.13** | 0.025 |
| frozen SR2 | 0.53 | 0.292 |
| every tuned arm (56 of them) | **0.30-0.35** | 0.41-0.45 |

*Measured*, `radial-placement-is-inverted`. HR barely has subhalos inside
0.25 R_vir; frozen SR2 has 29% of them there; **tuning raises that to 42%.** The
member-gather line wins abundance (`gather-holdout-rockstar-gate`: 20 → 366
subhalos against HR's 369) and pays for it in placement.

Three candidate explanations have been tested and closed:

* **"the membership target is wrong."** *Measured*, `docs/sr2_soft_membership.md`
  section 7.9: the solve's candidate centres sit at median **1.42** R_vir
  against HR's 1.19 — it aims correctly, or slightly outward, and the objects
  Rockstar recovers still sit at 0.35. Right target, no object.
* **"a radial term would fix it."** *Measured*, `soft-shell-distribution-term`:
  the shell family fails at w = 1, 10 and 30, dose-independently, and costs
  abundance monotonically. `shell-arm-radius-term` and
  `radial-profile-is-flat-in-host-mass` close the rest.
* **"the material isn't out there to gather."** *Measured*,
  `relocate-arms-hypothesis-falsified`: material exists at HR radii (0 starved)
  and relocating the supervised sets made the inward drift **worse**.

What is left is the initialisation. Every one of those experiments started from
`SRmodel/G_z0.pt`, whose own profile is already inverted (0.53 against 1.13)
before any loss term speaks. **The hypothesis: the mass is piled into the host
centre by the base training, and a fine-tune cannot undo it.**

## 2. The experiment

Train SR2's architecture from a random initialisation, and put the
placement-sensitive terms in the objective from the beginning rather than
bolting them onto a converged checkpoint.

| arm | objective | what it decides |
| --- | --- | --- |
| `base` | MSE + WGAN-GP on `SR2Critic` | **the control.** Does a from-scratch SR2 reproduce the inverted profile at all? If yes, the defect is the architecture or the base objective and no loss term will fix it. |
| `base_noms` | `--w-mse 0`, adversarial only | separates "MSE did this" from "the architecture did this" |
| `knn` | `base` + 6-D class-conditioned kNN-CDF | the kNN term, from scratch |
| `soft_radial` | `base` + soft-membership member-gather + `--w-radius` | the soft-membership term with radial control, from scratch |

`base` is not optional. Without it neither loss arm is interpretable: a good
radial profile could be the term or could be the fresh init, and a bad one could
be the term or could be a from-scratch generator that simply did not converge.

## 3. What it is built out of

Nothing in the objective is new. The point of the exercise is that the *same*
terms are charged against a *different* starting point, so anything reimplemented
would confound the comparison.

| piece | where it already lived |
| --- | --- |
| the generator | `cosmo_sr.tts.srs_noise.ControlledG` — SR2's architecture, state-dict-compatible with `G_z0.pt`, here built with random weights |
| the adversary | `cosmo_sr.reward.sr2_adversarial.SR2Critic` + `gradient_penalty` — SR2's own 20-channel input contract |
| the member-gather / soft-membership loss | `finetune_member_gather.host_loss`, imported as a module |
| the kNN-CDF term | `finetune_member_gather._unsup_knn` + `cosmo_sr.features.knn_cdf` |
| the guards | `free_field_gather.guards`, re-referenced (§5) |

New files:

| file | role |
| --- | --- |
| `src/cosmo_sr/train/sr2_scratch.py` | random init, the streaming paired-tile sampler, the base MSE, the WGAN-GP critic step, the HR-referenced low-k anchor, the term ramp |
| `scripts/features/train_sr2_scratch.py` | the trainer: composes the base objective with the imported structure terms; resumable |
| `scripts/slurm/_sr2_scratch_common.sh` | one place for every default; assembles the flags |
| `scripts/slurm/sr2_scratch_train_gpu.sbatch` | the GPU job |
| `scripts/slurm/submit_sr2_scratch.sh` | the submitter; calls `sbatch` and nothing else |
| `tests/train/test_sr2_scratch.py` | 54 tests |

## 4. The critic discriminates DENSITY, and that needs one setting

`SR2Critic`'s 20 channels are 6 trilinearly upsampled LR + 6 candidate
displacement/velocity + **8 inverse-pixel-shuffled CIC density** channels
(`grid_mult = 2`). The density half is why this critic and not the
member-gather high-pass one: it judges the mass field directly rather than the
displacement that produced it.

**That only works at `valid_center > 0`, and it is not a preference.** A training
example is a 64³ *tile*, not a periodic box. `cic_density`'s `% ng` deposit wraps
escaping particles back inside it, and *measured*
(`docs/density_collapse_investigation.md`) only **9.75%** of a tile's particles
stay put — median |Psi| ≈ 36 HR cells against a 64-cell tile. The wrapped field
correlates **r = 0.080** with that region's true density, with sigma inflated
2.2x. Eight of the critic's twenty channels would be noise, and every line the
run logged would still look healthy.

*Measured*, same document, for a 128³ crop against an exactly-buffered reference:

| region/crop | rel RMS | corr | mass kept |
| --: | --: | --: | --: |
| 0.25 | 0.000 | 1.0000 | 1.000 |
| **0.50** | **0.030** | **0.9995** | 0.972 |
| 1.00 | 0.396 | 0.9210 | 0.676 |
| wrapped | 2.36 | **0.080** | 1.000 |

So `valid_center = tile/2 = 32`, which is also what `critic.valid_center: 32`
fixed for the DMSR line. `check_density_channels` **raises** on 0 unless
`--allow-wrapped-density` is passed, and two tests pin that.

## 5. Four things that are deliberately not the fine-tune's settings

1. **The guard reference.** `free_field_gather.guards` anchors the
   block-averaged candidate to the *frozen field*, because in a fine-tune the
   candidate starts as that field and the guard is a drift detector. A random
   init has not drifted — it was never there — and anchoring to the frozen field
   would charge the run for failing to reproduce the concentration under test.
   `--guard-ref hr` (new, default `frozen`, every arm on disk unaffected) points
   the same operator at HR. `test_lowk_anchor_matches_the_finetune_guard_operator`
   pins the two to bit equality on identical inputs.
2. **The optimiser.** One uniform Adam rate over every parameter, not the
   depth-scaled micro-rates of `sr2_unfreeze`. Those exist to protect a
   pretrained function; `--rung` is forced to `full` and the job says so.
3. **The data.** A streaming sampler over every tile of the training boxes
   (5,632 `(box, tile)` pairs over 11 boxes), not a resident pool of ~56 hosts.
   *Derived*: 8,192 pairs as resident tensors would be 48 GB. HR stays memmapped
   and one 6 MB tile is sliced per example.
4. **A warm-up on the structure terms.** The soft solve chooses particles *from
   the current SR field* and the kNN term measures it; at step 0 that field is
   noise. `--term-warmup-steps 2000`, then a linear `--term-ramp-steps 2000`.

## 6. The settings that are not defaults, and why

**kNN.** Round 5b (`knn-phase-space-round5`, `docs/sr2_knn_distance.md` §12-13)
*measured* the only configuration that ranks the right way:
`--knn-phase-lambda 4000 --knn-ks 8 16 32 64 --knn-classes sub_50_200`.
Spearman of the term against held-out region subhalos over the working family:

| term | rho | p |
| --- | --: | --: |
| `class_cond[sub_50_200]`, lambda 4000 | **-0.65** | .002 |
| same, lambda 0 (control) | -0.23 | .32 |
| **pooled (hosts included)**, lambda 4000 | **+0.878** | 4e-07 |
| `class_cond_equal` (all 5 classes) | +0.76 | — |

**Hosts in the term flips its sign.** The trainer warns loudly if `--w-knn-unsup`
is on with no `--knn-classes` or with `--knn-phase-lambda 0`.

**Soft membership.** `--soft-prior coarse --soft-lambda 1`, the `coarse_soft`
arm that *measured* a held-out Rockstar win over the hard mask
(`soft-membership-relaxation`: hosts 3148 vs 3118, subs 10138 vs 9993).
`--max-set-particles 768` bounds the O(pool²) estimator.

**Radius.** `--w-radius 1.0 --radius-dead-zone 1.0`. This is the term *measured
dead* as a fine-tune lever, and that is the point: retrying it from a random
init is the direct test of §1's hypothesis.

**`--w-d6 1.0 --centre-mode full`** are set explicitly because the CLI defaults
are not the winning arm's (`mgtrain-defaults-are-not-the-winning-arm`).

## 7. The caveat that is not buried

The member-set **reference** is still built against the frozen SR2 field:
`build_member_sets` takes it as the reachable target, and the cached frozen box
supplies members outside the trained tiles. It is a *constant of the objective*,
fixed before step 1 and identical to the one every fine-tune arm used — a
supervision target, not an initialisation — but the structure arms are
therefore not frozen-SR2-free. The `base` arms are.

## 8. How to run it

```bash
# stage 1 -- the shakeout. Nothing else is submitted.
bash scripts/slurm/submit_sr2_scratch.sh

# read the three numbers it measures, then
SC_GO=1 bash scripts/slurm/submit_sr2_scratch.sh
```

Both from the **login node**: `sbatch` is denied from inside an allocation, and
the submitter refuses rather than failing obscurely.

*Measured* on a login-node CPU dry run (1 step, batch 1, `set0`): the loop runs
end to end, peak host RSS **2.0 GB**, and `noise_std_mean` moved 0 → 2.0e-4 in
one step, which is what confirms `AddNoise.std` is trainable from its zero init.
Step time there was 51 s on 4 CPU threads and says nothing about a GPU; that is
the number the shakeout exists to produce.

## 8b. RUN 1 (2026-09-23): the critic ran away, and why

*Measured.* Shakeout job 62066 passed — 0.98 s/step, peak VRAM 4.2 GB, peak host
RSS 27.4 GB. All four arms launched 14:41. Two findings, one of them a bug in
this code.

**The base objective works.** Held-out `set9`/`set10`, `base` at step 30k:
`mse` 0.290 → 0.196, `low_k` 0.0985 → 0.053, and `noise_std_mean` 2e-4 → 1.0e-2
— read-first items 1 and 2 both pass, and the generator is stochastic rather
than a deterministic upsampler.

**The critic ran away.** `d_real - d_fake` went **93** at step 2k → **298** at 5k
→ **2548** at 30k, monotone, with the gradient penalty at 1414, i.e.
`||grad_x D|| ~ 4` against the Lipschitz target of 1.

*Measured*, the cause, on real HR tiles of set0:

| channel group | RMS |
| --- | --: |
| upsampled LR (0-5) | 0.778 |
| field disp+vel (6-11) | 0.841 |
| **density (12-19)** | **4.911** |

with density `max 2786.1`, `p99.9 21.3`, `p50 -1.000`. A second draw gave `max
5634` and density RMS 21.6 — the tail varies by 2x between tiles, so the critic
saw wildly varying input magnitude batch to batch.

**The SR2 critic path had no input normaliser.** DMSR's
`CriticInputNormalizer` and the gather critic's `GatherCriticNorm` both exist
for exactly this, and the latter's docstring names the failure: *"their raw RMS
differ by orders of magnitude, so without normalisation the critic's first conv
cannot rescale them and one group starves the other of gradient."* DMSR records
the same near-miss at 138x. This path did not have one. That was a bug in
§3's wiring, not a tuning subtlety.

**The fix, in two parts.** `fit_critic_norm` reuses `GatherCriticNorm` (it is
channel-count agnostic) on the 20-channel input, fitted from **real HR tiles
only**, held fixed, applied identically to real and fake. And
`apply_density_transform` puts the density channels through `asinh` first:
a single per-channel scale divides a 570x spread down to 570x and no further,
whereas `asinh` maps 2786 → 8.6. `asinh` and not `log1p` because overdensity is
bounded below by exactly -1, where `log1p` is `-inf` and **the median voxel
sits** (p50 = -1.000; most of a tile is empty).

*Measured* after the fix, same tiles, batch 2:

| | head RMS | density RMS | input max |
| --- | --: | --: | --: |
| raw (as run) | 0.748 | 21.605 | 5634.6 |
| + scales only | 1.363 | 0.710 | 229.9 |
| **+ asinh + scales** | **1.363** | **0.958** | **10.2** |

**What this does NOT establish.** At random init `||grad_x D||` is ~1.95 in all
three rows — the normalisation does not change it, because a fresh critic's
gradient norm is set by its weight scale, not by input conditioning. The runaway
is a dynamics phenomenon over tens of thousands of steps and no single-point
measurement settles whether the input fix alone stops it. `SC_GP_EVERY=1` (the
penalty every step at lambda 10, instead of 1 step in 16 with the coefficient
scaled by 16) is an independent second lever and run 2 should carry both.

**One correction to an earlier reading in this line.** The observation that the
generator's adversarial loss (`-E[D(fake)] ~ 1223`) "dwarfs MSE (~0.2) by four
orders of magnitude" is not a statement about gradients: the generator feels
`grad_x D`, not `D`'s absolute offset, so a large `|D|` does not by itself imply
a large generator gradient. The defect is the input conditioning above.

**Also measured, and unrelated: the kNN term is 45x too expensive as
configured.** Job 62069, the term engaging at step 2000 took the step time from
0.30 s to >13.6 s — 15 days for 100k steps. The per-step bill is
`tiles x queries x neighbour-table rows` in float64, with tables at 1.3-3.25x a
tile. Both factors are linear, so `SC_UNSUP_TILES_PER_STEP=1 SC_KNN_QUERIES=256`
is ~32x cheaper and still a minibatch of the same estimator. `SC_KNN_QUERIES`
did not exist and now does; the cost knobs are now overridable from the
submitter, the round-5b settings deliberately are not.

Run 2 launch:

```bash
scancel <the run-1 job ids>
SC_GO=1 SC_GP_EVERY=1 SC_UNSUP_TILES_PER_STEP=1 SC_KNN_QUERIES=256 \
  bash scripts/slurm/submit_sr2_scratch.sh
```

## 8c. The reproduction baseline reframes the whole verdict (2026-09-23)

Section 9 originally said to watch `holdout_base.mse` fall. **That instruction
was wrong**, and the reproduction baseline is what shows it.

*Measured*, `runs/sr2_baseline/field_table` — the reproduction of real SR2
(set14, seed 0):

| field | P/P_HR (hi-k) | T(k) | r(k) |
|---|--:|--:|--:|
| displacement | 0.396 | 0.624 | 0.000 |
| velocity | 0.989 | 0.994 | −0.000 |
| density | 0.977 | 0.988 | 0.092 |

**SR2's high-k `r(k)` is ~0 on every channel.** It does not reproduce HR's
*realisation* at small scales — it generates a different one with the right
statistics, which is what the adversary is for. A pointwise MSE against HR's
realisation is therefore dominated by a component that is unpredictable *by
design*, and a model scoring **better** than SR2 on MSE is most likely
suppressing small-scale power. MSE cannot separate fidelity from blur.

*Measured* here, frozen SR2 scored on the trainer's own holdout metric, over 32
tiles of each held-out box (`set9`, `set10`) — the comparison the trainer never
had:

| | holdout mse | low_k | vel_rms |
|---|--:|--:|--:|
| **frozen SR2** | **0.3760** | **0.1093** | **1.029** |
| run1 `base` @68500 | 0.3658 | 0.1252 | 0.950 |
| run1 `base_noms` @71000 | 0.3474 | 0.1061 | 0.910 |

Run 1 ended essentially *at* SR2's numbers. Two consequences:

* **§8b's "held-out mse plateaued from step 24k" is withdrawn.** It read 0.137
  at 24k against 0.174 at 54k, but the end-of-run value is 0.366 and run 2 swung
  0.227 → 0.420 between steps 2500 and 6000. The trainer's holdout block averages
  4 batches of 4 — ~16 tiles — which cannot be read to three decimals. The
  plateau was sampling noise.
* **The base objective was never the bottleneck.** It tracks SR2.

The one gap that survives the correction is **velocity dispersion**: SR2 sits at
1.029 and every arm here at 0.83–0.95. That is consistent with this line's
recorded history — velocity small-scale power is the defect with no loss term
behind it.

`scripts/features/sr2_scratch_field_report.py` (+ its sbatch and submitter) is
the instrument this implies: it recomputes the table above for a checkpoint
**and** for the frozen SR2 field cached on the same boxes at the same seed, with
`reproduce_field_table.py`'s own estimator imported rather than reimplemented,
plus a whole-box holdout block over all 512 tiles per box. It reads checkpoints
and cached fields only, so it runs safely alongside training.

It is still a *field*-level check and cannot see subhalos — *measured*,
`density-power-cannot-see-subhalos`: frozen SR2 matches HR's P(k) to 1–5% while
Rockstar finds most of them missing. The radial verdict still comes from the
Rockstar gate.

## 8d. The MSE handoff diverges because the critic cannot see amplitude (2026-09-24)

Bracket probes A–D asked whether the generator survives `--mse-mode upstream`,
where MSE carries it until `--sr2-adv-warmup-steps` and the adversary carries it
alone after. A (MSE only) and B (`sum`) held. **C and D diverged**, each within
one eval interval of `w_mse_now` hitting 0:

| probe D | step 2000 (`w_mse` 1) | step 2250 (`w_mse` 0) |
| --- | --- | --- |
| `base_low_dis` | 0.0043 | 7.1e10 |
| `base_low_vel` | 0.059 | 5.1e11 |
| `base_vel_rms_ratio` | 0.82 | 5.8e05 |

Three properties of the failure rule out the obvious readings. It is
**immediate**, not a drift — D is healthy at 2000 and destroyed at 2250, and C
does the same 1000 steps earlier at its own handoff, so the trigger is
`w_mse → 0` and nothing else. **Velocity blows up 7–14× harder than
displacement**, consistently. And it **thrashes rather than climbs**: C runs
5.9e10 → 1.5e6 → 4.2e8 → 1.7e10 → 4.5e9 → 2.3e10, five decades, with `d_loss`
pinned near 0.001 throughout.

The natural objection is the right one: a generator putting out 1e5× the data
scale should be *trivial* for the discriminator, which should hand back a large
gradient. That is not a failure mode. And indeed the loss side is innocent —
the non-saturating BCE generator loss is `softplus(-D)`, whose derivative
`-sigmoid(-D)` tends to **-1**, its maximum, as `D → -inf`
(`test_bce_generator_loss_is_non_saturating`). "D saturated so G got no
gradient", the reading recorded after run 3, is wrong.

**MEASURED cause.** `SR2Critic` puts a `GroupNorm` after every block but the
first. GroupNorm divides out the per-group standard deviation, so
`GN(a·z) == GN(z)`; `Conv3d` (once its bias is negligible) and `LeakyReLU` are
positively homogeneous. The composition is therefore **exactly scale-invariant**.
On `width=16, depth=4`:

| input scale `a` | `D(a·x)` | `‖∂D/∂x‖` | `a·‖∂D/∂x‖` |
| --- | --- | --- | --- |
| 1e0 | 0.024976 | 6.94e-01 | 0.6941 |
| 1e2 | 0.009686 | 6.85e-03 | 0.6848 |
| 1e3 | 0.009522 | 6.85e-04 | 0.6847 |
| 1e5 | 0.009504 | 6.85e-06 | 0.6846 |
| 1e6 | 0.009504 | 6.85e-07 | 0.6846 |

`D` is pinned to four digits across three decades, and the gradient falls as
**exactly** `1/a`. `--critic-spectral-norm` does not change this — it was
measured with spectral norm on and behaves identically, which is why probes C
and D, both of which ran it, diverged anyway.

**`map2map` has the same property, and that is the point.** Its `PatchGAN42`
is `Conv -> LeakyReLU -> Conv -> BatchNorm3d -> LeakyReLU -> Conv -> BatchNorm3d
-> LeakyReLU -> Conv`, fed raw `cat([input, output])` (`train.py:364`). BatchNorm
divides out the per-channel std exactly as GroupNorm does, and there is no norm
after the first conv, so upstream's discriminator is scale-blind in precisely the
same way. Upstream survives `--adv-start` anyway because its `--adv-start` is in
EPOCHS: the generator arrives at the handoff already at the right amplitude
after a converged MSE phase, with Adam's state settled, so a free direction it
has no reason to leave is harmless. Ours arrives 1-2k steps old. So "be faithful
to map2map" is not a stability argument for this critic -- the faithful choice
is the one that diverges here, and `--critic-inner-norm none` is a deliberate,
named deviation.

Two consequences, and together they are the whole failure:

1. **Nothing in the objective constrains the output amplitude.** With `--w-mse`
   handed to 0 and `--w-base-low 0` (correctly, after run 2), the only remaining
   term is blind to scale by construction. Log-amplitude is a free direction, so
   the field random-walks along it. That is why it thrashes instead of climbing:
   this is an *unconstrained* direction, not a diverging one.
2. **The runaway is self-reinforcing.** The one signal that could pull the field
   back is damped by `1/a` as it walks out, so it weakens by the same factor the
   error grows.

`d_real` +9 against `d_fake` -8 with the field 1e5 too large is therefore not a
saturated discriminator. It is a discriminator that was **never shown the
scale** — it is still discriminating the field's *pattern*, competently, on a
field whose amplitude it cannot perceive. (A second effect compounds it: the 6
upsampled-LR conditioning channels stay O(1) while the 6 field and 8 density
channels go to 1e5, so the first conv's output is entirely field and the critic
stops being *conditional* at all.)

This also explains the velocity asymmetry. Displacement reaches the critic twice
— through channels 6–11 *and* through the 8 CIC density channels, where the
deposit is a non-homogeneous function of it (at 1e5 every particle escapes the
valid centre) — so it retains some non-scale-invariant handle. Velocity enters
**only** linearly, on the path GroupNorm annihilates exactly. It is the purest
free direction in the objective and it moves fastest.

### The two fixes, both default-off

| knob | what it does | probe |
| --- | --- | --- |
| `SC_W_SCALE` | `log(rms_cand/rms_hr)²` on the displacement and velocity groups | E |
| `SC_CRITIC_INNER_NORM=none` | drops the GroupNorms so the critic is positively homogeneous and *can* see amplitude | F |

`SC_W_SCALE` is the leash that is safe to keep **precisely because MSE is not**.
The reason for dropping MSE is the suspicion that a pointwise loss to one
realisation flattens the radial distribution of subhalo mass. The amplitude
anchor is *two scalars per batch* and carries no spatial information at all —
`test_scale_anchor_carries_no_spatial_information` shuffles the field in space
and the term does not move. It can say "the field is a thousand times too loud"
and it can say nothing else, so it cannot express a preference over radial
profiles. It is the `vel_rms_ratio` the guard already reported, promoted from a
diagnostic to an objective. Log-squared so over- and under-shoot cost the same.

`SC_CRITIC_INNER_NORM=none` is the root fix and needs no extra term: without the
GroupNorms the critic is positively homogeneous, `D` grows with the input
amplitude, and "far too big" becomes something it can learn to say. Pair it with
`SC_CRITIC_SPECTRAL_NORM=1` — the SNGAN combination — so the critic stays
bounded without being blinded. Note the two settings are **different
architectures** (dropping the norms shifts every later index in the
`Sequential`), so `--resume` across a change of this flag raises rather than
silently loading the wrong tensors.

`SC_MSE_HANDOFF` (linear decay of `w_mse` to 0 over N steps instead of
map2map's single-step cliff) is in both probes. It is an aggravator, not the
cause — a cliff also leaves Adam's second-moment estimate calibrated to a loss
that no longer exists — but it is free to remove.

The trainer now **prints a warning at startup** when `--mse-mode upstream` is
combined with `--w-scale 0`, `--w-base-low 0` and `--critic-inner-norm group`:
that is exactly C and D's configuration and there is no term left holding the
amplitude.

**Read `rms_ratio_dis` and `rms_ratio_vel` in `metrics.jsonl`.** Both are 1.0
when the field's scale matches HR. They are the quantity that actually ran away,
and they say so three to four decades before `base_mse` does.

## 9. What to read, in order

1. **The field report's `P/P_HR` against the frozen column**
   (`submit_sr2_scratch_field_report.sh`). Density ~0.98 and velocity ~0.99 are
   the targets; `r(k)` **higher** than frozen's ~0 is a regression-to-the-mean
   warning, not a win. Do **not** use `holdout_base.mse` for this — see §8c.
2. **`noise_std_mean`** — `AddNoise.std` starts at **zero** and SR2's fine-scale
   structure is written entirely by those six tensors. A run that leaves them at
   ~0 has trained a *deterministic upsampler*, whatever its spectrum says, and
   cannot produce substructure. If flat, rerun with `SC_INIT_NOISE_STD=1e-2`.
   (*Measured*: `G_z0` ended at ~1e-3 at z0-z4 and ~5e-2 at z5.)
3. **`adv.d_real - adv.d_fake`** — a WGAN critic whose scores have separated
   without bound has stopped constraining the generator.
4. Then gate the `tuned_step*.pt` snapshots on a held-out box with
   `submit_gather_holdout_rockstar.sh` and **read the radial row**. That is the
   number this whole set was built for: HR 1.13, frozen 0.53, every fine-tuned
   arm 0.30-0.35.

The snapshots cost the run nothing — training continues to `--steps` — and are
what lets the gate read an arm at hour ~3 rather than at the end of a multi-day
job.

## 10. What would falsify what

| outcome | reading |
| --- | --- |
| `base` lands at ~0.35 like the tuned arms | the init was **not** the obstacle; the architecture or the base objective is. Stop this line and go after the base objective. |
| `base` lands near HR's 1.13 but the loss arms do not | the fine-tune's inverted profile was an artifact of starting from `G_z0`, and the structure terms are actively harmful to placement. |
| `base` inverted, `base_noms` not | **MSE did it.** The cheapest and most actionable outcome. |
| `soft_radial` beats `base` on radial without losing abundance | the radius term works from a fresh init and failed as a fine-tune lever for the reason §1 proposes. |
| any arm's `holdout_base.mse` never falls | that arm is uninterpretable; it did not learn the map. Read nothing else from it. |
