# Direct reward-guided fine-tuning of SR2

`feat/direct-sr2-reward-ft`. This line fine-tunes the **original SR2 generator**
`G_theta(Y, z)` against the catalog reward. It does not train a residual
generator, does not implement ControlNet, and does not modify any existing
residual/reward code path — `configs/reward/reward.yaml` and everything it
drives are untouched and are *read* from here so the two lines score the same
reward.

## The shape of the argument

The catalog reward is a function of a Rockstar catalog, which is not
differentiable and takes hours per box. Three things bridge that gap, in order,
and each has a gate in front of the next:

1. **A fast differentiable surrogate** — a small feature vector
   (`reward/soft_structure.py`) feeding an ensemble of proxies
   (`reward/catalog_proxy.py`) that predict a *tile's* catalog sufficient
   statistics `(N, H, S)`, not a scalar reward.
2. **An exact differentiable copy of the reward**
   (`reward/torch_reward.py`) that turns those predicted statistics into
   `dR = R(S_0 - s_{0,j} + s_hat_{theta,j}) - R(S_0)`. Parity with the NumPy
   reward is tested as an equality, not a correlation.
3. **Real full-box Rockstar as the oracle**, periodically: to *label* proxy
   training data, to decide whether the small overfit actually worked, and to
   revalidate the proxy in the DAgger loop.

The proxy never becomes the criterion. Every advancement decision — enter the
next unfreezing rung, accept a checkpoint — is taken on a real catalog and a
real field measurement.

## Module map

| file | role |
| --- | --- |
| `train/sr2_finetune_data.py` | one generator tile per example; **bit-identical** to the corresponding slice of a full-box seeded inference |
| `train/sr2_unfreeze.py` | the five rungs, by exact parameter name, with depth-based learning-rate groups |
| `reward/soft_structure.py` | differentiable structural features from valid-centre CIC |
| `reward/torch_reward.py` | exact torch copy of `RewardModel`, plus the tile-swap form |
| `reward/catalog_proxy.py` | the proxy ensemble, its two losses, and `Q_safe` |
| `reward/direct_gates.py` | the paired, relative density gate and its calibration |
| `reward/sr2_adversarial.py` | the optional WGAN-GP arm (off; see below) |
| `train/train_sr2_direct.py` | the actor objective and step |
| `configs/reward/sr2_direct_finetune.yaml` | the single source of truth for this line |

## The objective

```
L = -Q_safe + λ_P·L_Pδ + λ_low·L_low + λ_prox·L_prox + λ_div·[d_min − d_struct]²₊
```

`Q_safe = mean_m Q_m − β·std_m Q_m` over the proxy ensemble; the proxies are
**frozen but differentiable** (parameters fixed, gradient passes through into
SR2). There is deliberately **no full-field L2** to HR or to frozen SR2: an L2
to HR is minimised by blurring, which is the exact failure that flattens SR2's
occupation curve.

Two implementation notes that are not cosmetic:

* `L_low` is the **squared** relative block-average difference. At step zero the
  actor *is* the frozen generator, so the un-squared form is exactly zero and
  `d(sqrt)/dx` there is infinite — measured, and it produced NaN weights within
  three steps. The gate still reports the un-squared RMS ratio.
* The soft thresholds are applied to `log1p(delta)`, not to `delta`. With a
  sigmoid of width proportional to the threshold, an *empty* cell scored 0.054
  on `compact_mass` against a real signal of a few percent — the pedestal was
  larger than the measurement.

## Unfreezing rungs

`proj_noise` → `fine` → `middle_fine` → `all_blocks` → `full`. Each is a
separate checkpointed job. Learning rates are grouped by depth (1e-5 / 3e-6 /
1e-6 / 3e-7) so a group's rate does not change when a later rung is entered.

**Unfreezing more is never the response to a proxy that fails its gate.** A bad
gradient with more parameters to move is strictly worse.

## Density preservation is a hard gate

`reward.yaml`'s `density_power_error_max = 0.03751` is 1.5× the worst frozen
SR2-vs-HR error. That is the right bound for "materially worse than the
baseline" and much too permissive for "no degradation" — it admits a checkpoint
that makes density 50% worse than SR2's own error. So this line gates on a
**paired, relative** quantity, `err_θ(box, seed) − err_0(box, seed)`, with the
tolerance calibrated from the frozen generator's own seed-to-seed spread. Until
that calibration is pasted into the config and `calibrated: true` is set,
`check_direct_gates` rejects everything — deliberately.

The residual line's severities are unchanged. The stricter rules live only in
`configs/reward/sr2_direct_finetune.yaml`.

## Adversarial continuation: not available here

Checked at implementation time: `external/SRS-map2map/SRmodel/` holds `G_z0.pt`
and `G_z2.pt` and nothing else, and no `D_*` checkpoint exists anywhere under
`$ZFS`. Therefore:

* the adversarial weight is **zero in the mainline**;
* a freshly initialised critic is **not** "ordinary SR2 continuation" — it is a
  random objective, and generator steps taken against it are noise applied to
  the checkpoint this line is trying not to damage. It may be run later as a
  clearly-labelled ablation after a frozen-generator warm-up
  (`allow_fresh_critic: true`), and `build_critic` stamps every such run with
  `source: fresh_random_init_ABLATION_ONLY`;
* `dmsr.HRCritic` is deliberately not reused: it was built for high-pass
  residual fields and sees different inputs. `SR2Critic` reproduces SR2's own
  20-channel contract (6 upsampled LR + 6 field + 8 inverse-pixel-shuffled fine
  CIC density), WGAN-GP with λ=10 and the penalty every 16 critic batches.

## Spatial scale of the label — tile vs region (SUPERSEDED conclusion below)

### ~~The per-tile target is not rankable — switch to pooled (2026-08-09)~~ — superseded

> **Superseded 2026-08-10.** The measurement below is preserved as historical
> evidence; its *conclusion* — that the local-credit route is dead and the only
> option is a whole-box scalar — was an over-reach on two development boxes and
> two methodological bugs, and is no longer the plan. See "Region-scale
> re-analysis" below.

The proxy in step 1 predicts a *tile's* statistics, so the label has to be
attributable to a tile. `reward/attribution_diagnostic.py` (run
`sd_attrib`, code `1f8bf99`) reads the saved `tile_weights.npz` and catalogs
and re-derives the per-tile reward change under two attribution schemes —
**fractional** (mask-weighted) and **majority** (winner-take-all) — then asks
whether the per-tile label is repeatable enough to rank on. It measures
repeatability against the frozen-seed churn: how much a tile's label moves
between four re-simulations of the *same* input, versus how much it moves under
intervention.

Verdict on the two completed boxes (`set0`, `set1`, 4608 rows each):

| scheme | "repeatability ceiling" (touched tiles) | SNR (touched) | pooled-cancel | rankable (old verdict) |
| --- | --- | --- | --- | --- |
| fractional | 0.58 | 0.70 | 93% | **no** |
| majority | 0.58 | 0.72 | 93% | **no** |

Both schemes fell below the 0.5 gate on `set1` and only scraped it on `set0`,
and ~93% of the per-tile signal cancelled when pooled to the whole box.

### Why those numbers were **not** a learnability ceiling

Two things were wrong with reading them as one:

1. **It was called a "repeatability ceiling", as if the four frozen seeds were
   re-measurements of one field.** They are not. Different SR2 seeds produce
   different fields and different proxy inputs, so the frozen-seed comparison
   measures **baseline-context stability** — how much a candidate's ranking
   depends on *which frozen box it is scored against* — not the irreducible
   noise of one measurement. Under a *fixed* baseline (which is what the actor
   and a seed-0-trained proxy actually use) the label is well defined.
2. **It measured one spatial scale — a single tile — when the real question is
   which scale is the smallest reliable one.** The 92–93% pooled cancellation is
   precisely the evidence that a *coarser* unit is more stable: a cluster's
   bound particles originate in many Lagrangian tiles, so its label churns
   between tiles while barely moving for the box. The right response is to find
   the region width at which the churn has cancelled *within* the region but the
   region is still a local unit the actor can move — not to jump straight to the
   whole box.

Two implementation bugs also inflated the pessimism, now fixed
(`reward/catalog_proxy.py`): `spearman()` used a double-`argsort` that invents an
order for tied or constant inputs (so an unrankable prediction scored as chance
rather than `NaN`), and the pairwise agreement scored a prediction tie on a
descending true pair as *correct*. Both made a noisy label look slightly more
rankable than it is, but the tie handling in the pair builder
(`make_within_tile_pairs`) was the opposite — it let exact-tie pairs through at
`min_margin=0` and trained on label noise.

### Region-scale re-analysis (2026-08-10)

`reward/regions.py` pools the exact tile statistics into cubic **regions** of
width 1, 2, 4, 8 tiles over the 8³ grid, for every valid periodic partition
offset, with three exact identities (width-1 = tiles, width-8 = one box, and
every partition sums to the whole-box catalog to 1e-6).
`reward/region_attribution_diagnostic.py` (job `sd_region`) uses it to measure,
per (box, scheme, **width**, offset): the number of informative regions and
non-tied candidate pairs; the tie-aware within-region Spearman and pairwise
agreement of the candidate ordering **across frozen baseline contexts**
(`baseline_context_stability`, computed within (box, region, offset) and
averaged per box, bootstrapped by box only — never one global correlation); the
regional signal-to-noise; and how much per-tile churn cancels once pooled to the
region.

The counterfactual is the exact swap form, lifted from a tile to a region and
computed **consistently**: for frozen seed `r`,
`ΔR_g = R(C_r − c_{r,g} + c_{c,g}) − R(C_r)`, with `C_r` and `c_{r,g}` always
from the *same* baseline `r`.

With only `set0` and `set1` this is **exploratory**: the report stamps
`evidence_status = exploratory_insufficient_boxes` and proposes which width(s)
to carry into the held-out pilot, and it deliberately emits **no**
"unlearnable" verdict — two development boxes cannot support one, and the
trained held-out proxy gate (phase 5) remains decisive. A width is *eligible*
for the pilot screen only with ordering stability ≳ 0.65 (Spearman and pairwise)
and neither screen box collapsing.

Result (job `sd_region`, `set0`+`set1`, mean over offsets and boxes, both
schemes ≈ identical so `fractional` shown):

| width | within-region stability ρ | pairwise | min-box ρ | regional SNR | pooled-cancel | informative units |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (tile) | 0.995 | 0.993 | 0.994 | 0.65 | 0.00 | 153–177 tiles |
| 2 | 0.994 | 0.992 | 0.994 | 0.63 | 0.55 | ~47 regions |
| 4 | 0.995 | 0.992 | 0.994 | 0.63 | 0.82 | 8 regions |
| 8 (box) | 1.000 | 1.000 | 1.000 | ~0.66 | 0.93 | 1 region |

The old 0.58 "ceiling" **does not reproduce** under the correct statistic. The
metric the gate actually uses — the tie-aware ordering of the candidate versions
*within* a (box, tile/region), across frozen baseline contexts — is ≈ 0.99 at
**every** width, including width 1. The 0.58 was the correlation of per-tile `dR`
*magnitudes* pooled across all tiles between two baseline halves; that global
concatenation mixes cross-tile magnitude scatter into the number and is exactly
the pooling the plan forbids. The 92–93% pooled cancellation reproduces (0.93 at
width 8) but is not evidence against the tile path: it grows smoothly with width
while the within-tile *ordering* is already stable at width 1.

Two honest caveats before this becomes a decision: (1) it is **baseline-context
stability**, not learnability — whether a proxy can predict the ordering from
features is decided only by the trained held-out gate (phase 5); (2) the
candidates here are the monotone intervention α-ladder, whose within-unit order
is baseline-robust by construction, so 0.99 is an upper-ish bound that real actor
candidates need not match. Regional SNR ≈ 0.65 (label signal comparable to
seed-churn) says the signal is real but modest.

Provisional read (exploratory, two dev boxes): the whole-box-scalar pivot was
**not** warranted; the simpler **width-1 tile path is not ruled out** and is the
one to carry — pending the set8/set9 pilot screen and the trained gate.

Reports: old `runs/direct_a/attribution_diagnostic.json` (unchanged, historical);
new `runs/direct_a/region_attribution_diagnostic.json`.

## Runbook

```bash
DRY=1 bash scripts/slurm/submit_sr2_direct.sh all      # print, submit nothing

bash scripts/slurm/submit_sr2_direct.sh data           # candidates + real labels
bash scripts/slurm/submit_sr2_direct.sh proxy          # fit + section-7 gate
bash scripts/slurm/submit_sr2_direct.sh baseline       # frozen field metrics
bash scripts/slurm/submit_sr2_direct.sh calibrate      # -> HUMAN STEP
# paste gate_calibration.json's proposal into the gates: block, calibrated: true
bash scripts/slurm/submit_sr2_direct.sh overfit        # section 10
bash scripts/slurm/submit_sr2_direct.sh rung RUNG=proj_noise
bash scripts/slurm/submit_sr2_direct.sh dagger RUNG=proj_noise
```

Two places the chain stops on purpose: after `calibrate` (a threshold has to be
pasted in by a human, or the gate scores against placeholders) and between rungs
(advancing is a decision taken on the real catalog result).

**Stages run in separate invocations carry no dependency on each other.** Only
`all` wires `baseline -> calibrate` with `afterok`. If you run `baseline` and
then `calibrate` back-to-back as two commands, the calibrate job starts before
the baseline fields exist, prints `MISSING INPUT`, and exits 0 (a no-op, not a
strand). Re-run it once baseline lands, or chain it explicitly with
`AFTER=<baseline jobid>`. The same applies to `proxy` after `data`, and to
`rung`/`overfit` after `proxy`.

### Order of evidence, section 10

1. Small overfit: one box, fixed host-rich tiles, `proj_noise`, short.
2. Generate a full periodic box, run **real** Rockstar.
3. Proceed only if the *real* result improves occupation in ≥ 2 reliable host
   bins including one of bins 2–3, with density preserved. The proxy improving
   is not the criterion.
4. Then DAgger: train, generate, label asynchronously, append, refit the proxy,
   revalidate, continue from the last **field-valid** checkpoint. Proxy and
   actor are never updated from the same minibatches.

## Splits

`set0–7` fit the proxy and train the actor; `set8–11` gate the proxy and
evaluate the actor. `set12` is already development-contaminated (the SR2 subhalo
study and the reward sanity check both ran on it) — usable as a late preflight,
never as final evidence; the scripts print a notice when it is touched.
`set13–15` are **sealed**: `assert_not_sealed` refuses to open them.

## Artifacts

Everything bulky goes under `$ZFS/DMSR/dmsr_reward/sr2_direct/`
(`DMSR_SR2_DIRECT_ROOT`): fields, catalogs, checkpoints, particle tables. The
`.particles` table (~7 GB per box) is streamed into a compressed summary and
**deleted in the job that produced it**.
