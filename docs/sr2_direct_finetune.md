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

## The per-tile target is not rankable — switch to pooled (2026-08-09)

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

| scheme | repeatability ceiling (touched tiles) | SNR (touched) | pooled-cancel | rankable |
| --- | --- | --- | --- | --- |
| fractional | 0.58 | 0.70 | 93% | **no** |
| majority | 0.58 | 0.72 | 93% | **no** |

Both schemes fall below the 0.5 gate on `set1` and only scrape it on `set0`,
and ~93% of the per-tile signal cancels when pooled — a tile's label churns
between frozen seeds nearly as much as it moves under intervention. This is an
intrinsic per-tile noise floor: it is set by re-simulation variance of a single
tile, not by sample count, so finishing the remaining boxes does **not** rescue
it. `recommendation.action = switch_to_pooled_target`,
`per_tile_ranking_viable = false`.

The same diagnostic reports `whole_box_repeatability_ceiling = 1.0` for both
schemes: the **pooled / whole-box** reward change is perfectly stable. So the
fix is to change the target, not the features — predict pooled reward change,
not per-tile `(N, H, S)`. The remaining 92/120 candidate labels need not be
generated for the per-tile proxy; the `index` job (`sd_index`) correctly leaves
no `labels_complete.json` and blocks the per-tile trainer.

Report: `runs/direct_a/attribution_diagnostic.json`.

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
