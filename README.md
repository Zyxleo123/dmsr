# cosmo_sr

Scarce-HR / many-LR super-resolution for cosmological displacement+velocity
fields. This parent project uses [`map2map`](https://github.com/eelregit/map2map)
and [`SRS-map2map`](https://github.com/yueyingn/SRS-map2map) as **read-only**
external dependencies while keeping our own method (fixed degrader `A`,
deterministic generator `G`, ambient LR-consistency loss, scarce paired-HR loss,
evaluation) under `src/cosmo_sr`.

## Method summary

* Canonical field format: `(C, N, N, N)` channel-first cubic, `float32`. For
  SRS-style data `C = 6` where channels `0:3` are displacement and `3:6` are
  velocity.
* Fixed degrader `A`: `x_hr (B,C,N_hr,...) -> y_lr (B,C,N_lr,...)` by average
  pooling over non-overlapping `scale_factor^3` blocks (`mode="average"`).
* Generator `G(y_lr) -> x_hr`: trilinear upsample by `scale_factor` + Conv3d
  residual blocks + output conv (`SimpleSRGenerator`).
* Losses:
  * supervised: `mse(G(y_lr), x_hr)` on scarce paired data,
  * ambient LR-consistency: `mse(A(G(y_lr)), y_lr)` on many LR-only samples,
  * optional deterministic regularizers (TV, finite-value penalty, mean/std).
* No stochastic flow/diffusion or entropy regularization yet (deterministic
  first; those are planned as separate branches).

## External dependencies (pinned)

The two external repos are referenced under `external/` (symlinks to the local
clones in this workspace). Pinned commits (see `external/COMMITS.txt`):

| repo | url | commit |
| --- | --- | --- |
| map2map | https://github.com/eelregit/map2map | `4ebc835f019b16e68060ea05acf0e9fe0a62e6bd` |
| SRS-map2map | https://github.com/yueyingn/SRS-map2map | `77e68576b538f710ade66443b7badc5afe8a6193` |

To reproduce the layout from scratch (submodule style):

```bash
mkdir -p external
git submodule add https://github.com/eelregit/map2map external/map2map
git submodule add https://github.com/yueyingn/SRS-map2map external/SRS-map2map
git -C external/map2map      checkout 4ebc835f019b16e68060ea05acf0e9fe0a62e6bd
git -C external/SRS-map2map  checkout 77e68576b538f710ade66443b7badc5afe8a6193
```

In this workspace the clones already exist one level up, so `external/map2map`
and `external/SRS-map2map` are symlinks to them.

## Setup

Uses the existing conda env `pjm` (Python 3.12; torch, numpy, scipy, matplotlib,
pyyaml, bigfile already installed).

```bash
conda activate pjm
pip install -e external/map2map    # external dep, editable
pip install -e .                   # our package
pip install pytest
```

Or simply:

```bash
bash scripts/setup_env.sh
```

Sanity imports:

```bash
python -c "import map2map; import cosmo_sr"
python -c "from map2map import models; print(models.UNet)"
```

## Tests

```bash
pytest -q
```

## Training

```bash
# supervised baseline (paired only)
python -m cosmo_sr.train.train_supervised --config configs/supervised_baseline.yaml
python -m cosmo_sr.train.train_supervised --config configs/supervised_baseline.yaml --smoke

# our method: ambient LR-consistency + scarce paired HR
python -m cosmo_sr.train.train_ambient --config configs/ambient_smoke.yaml --smoke
python -m cosmo_sr.train.train_ambient --config configs/ambient_full.yaml

# scale-shared stochastic null-space residual flow (64 -> 512 cascade)
python -m cosmo_sr.train.train_flow --config configs/flow_cascade.yaml --smoke
python -m cosmo_sr.train.train_flow --config configs/flow_cascade.yaml
```

Every run directory contains `config.yaml`, `env.json` (project + external
commit hashes, environment), `metrics.csv` (+ TensorBoard logs) and checkpoints.

### Logging & eval-while-training

* **Always logged (all trainers)**: training losses, `lr`, `grad_norm` (global
  gradient L2 norm), `step_time_ms` / `steps_per_sec` (throughput), and
  `gpu_mem_mb` (peak CUDA memory, when on GPU).
* **Validation during training**: all three trainers log validation metrics
  periodically. `train_supervised` logs `val_loss`; `train_ambient` logs
  `val_ambient` and `val_hr_mse`. `train_flow` runs a **per-octave** validation
  every `train.eval_every` steps, logging for each octave `R`:
  * `val_fm_R{R}`   — flow-matching loss,
  * `val_cons_R{R}` — sampled `A`-consistency (should be ~0 by construction),
  * `val_highk_R{R}` / `val_allk_R{R}` — power-spectrum recovery ratios,
  * `val_respow_R{R}` — residual power (gen/true amplitude),
  * `val_zdiv_R{R}` — `z`-diversity (variability across noise; the key
    stochastic-SR health metric),

  plus aggregates `val_fm`, `val_consistency_rel`, `val_highk`, `val_respow`,
  `val_zdiv`, and per-octave training series `loss_fm_R{R}`. Metrics land in
  `metrics.csv`, TensorBoard (`<run>/tb`), and Weights & Biases.
* **Weights & Biases**: enabled by default (uses `WANDB_API_KEY` from `~/.bashrc`).
  Configure per run via a `wandb:` block:

```yaml
wandb:
  mode: online        # online | offline | disabled
  project: cosmo_sr
  # entity: your_team
  name: flow_cascade
  group: null
```

  Set `mode: disabled` to turn it off, or `mode: offline` to log locally and
  `wandb sync <run>/wandb/offline-run-*` later. Smoke runs never touch wandb.

### Scale-shared stochastic null-space residual flow

A single flow-matching velocity network `v_phi`, **shared across octaves**
`R in {64,128,256}` (via an `R` embedding), models a *stochastic null-space
residual* at each factor-2 step. The forward map is always LR-consistent by
construction:

```
x_2R = B_cons_R(y_R) + P_null_R(r),   r ~ flow(v_phi | y_R, R)
```

* **Operators** (`operators/multiscale.py`): `A_R` (block-average `2R->R`),
  `U_R` (nearest broadcast `R->2R`), `P_null_R(h)=h-U_R(A_R(h))`. Identities:
  `A_R(U_R(y))=y` and `A_R(P_null_R(h))=0`.
* **Base upscaler** (`operators/base_upscaler.py`): interface `B_R(y_R)`;
  `IdentityUpscaler` (=`U_R`) or `BackboneUpscaler` (a learned SR2 backbone).
  `consistent_base` gives `B_cons = B_R(y) + U_R(y - A_R(B_R(y)))` with
  `A_R(B_cons)=y` exactly, for *any* `B_R`.
* **Paired training** (flow matching): for each adjacent pair `(x_R, x_2R)` from
  the HR pyramid, `r_star = P_null_R(x_2R - B_cons_R(x_R))`,
  `r_t = (1-t)z + t r_star`, `L_FM = mse(v_phi(r_t,t,x_R,R), r_star - z)`.
* **LR-only training** (multi-octave): sample `r` with the flow, `x_hat =
  B_cons_R(y_R) + P_null_R(r)`; `L_deg = mse(A_R(x_hat), y_R)` (≈0, a consistency
  safety) plus `L_band` matching generated residual band power to real
  paired-residual bands. `L_band` is the distributional term that lets LR-only
  sims actually inform the generator (a deterministic consistency term alone
  would not). LR-only sources are configured as a list of streams, each with its
  own octave `res`, native `crop`, and optional `pool` (average-pool factor), so
  the `Ng=64` sims feed `R=64` and the `Ng=256` sims feed `R=128` (pooled) and
  `R=256` — the model is trained LR-only across all octaves, not just `R=64`.
* **Sampling** (`inference/flow_sample.py`): `sample_step` integrates the ODE per
  octave; `super_resolve_cascade` chains `64->128->256->512`.

Config: `configs/flow_cascade.yaml`. SLURM: `bash ~/slurm/dmsr/cosmo_sr_train.sh flow`.

**Memory**: this model is much heavier than the deterministic generators — one
step does a paired forward/backward at up to `128³`, *plus* the LR-only branch
unrolls `loss.lr_sample_steps` full forward passes through the network for the
ODE integration (backprop-through-time). At the default `crop_hr=128,
width=64, depth=4` this needs ~46 GB in plain fp32, which OOMs even a 48 GB
GPU. Two config knobs address this (both on by default in
`flow_cascade.yaml`):

* `train.amp: true` — fp16 autocast + `GradScaler`. Note `GroupNorm` is one of
  the ops autocast forces back to fp32 for stability, so **AMP alone only
  gives a modest reduction** here (most activations get promoted back to fp32
  right after each norm).
* `model.grad_checkpoint: true` — wraps each residual block in
  `torch.utils.checkpoint`, discarding block activations and recomputing them
  on backward. Peak activation memory then scales with ~1 block instead of
  `depth` blocks (and multiplies out over the unrolled LR-only steps), at the
  cost of ~30% more compute. **This is the fix that actually matters** given
  the GroupNorm/autocast interaction above; combine both for the best result.

If still tight, reduce `data.crop_hr` and/or `loss.lr_sample_steps` as a
further, more drastic memory/quality trade-off.

## Evaluation

```bash
python -m cosmo_sr.eval.run_eval --config configs/ambient_smoke.yaml \
    --checkpoint runs/ambient_smoke/ckpt_last.pt --lr LR.npy [--hr HR.npy] --out eval_out
```

Writes `metrics.json`, `spectra.npz` and slice PNG(s).

### Flow cascade evaluation (`eval/flow_eval.py`, `eval/sr2_stats.py`)

For the residual flow model, `evaluate_cascade` scores each octave with:
`A`-consistency, high-k power recovery, residual power per octave, and `z`
diversity (variability across noise draws for the same `y`). SR2-style stage-2
statistics live in `eval/sr2_stats.py`: `equilateral_bispectrum` (shell
estimator), `two_point_correlation`, `velocity_statistics`, and a documented
`halo_abundance` hook (requires an external particle halo finder).

Runnable CLI:

```bash
python scripts/eval_flow.py \
    --config configs/flow_cascade.yaml \
    --checkpoint runs/flow_cascade/ckpt_last.pt \
    --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set15.npy \
    --crop 256 --n-steps 20 --diversity 3 --out runs/flow_cascade/eval_set15
```

Writes `flow_eval.json` (per-octave + final-level stats) and `power.png`.

## Preprocessing (real data)

Convert an MP-Gadget bigfile snapshot to a canonical `(6,Ng,Ng,Ng)` field
(mirrors `SRS-map2map/preproc.py`):

```bash
python scripts/preprocess_snapshot.py --inpath /path/to/snapshot --outpath catnorm.npy
```

Real data on this cluster:
* LR-only fields (Ng=64): `/zfsauton/scratch/yixiz/DMSR/lr_sims/set*/catnorm.npy`
* Larger fields (Ng=256): `/zfsauton/scratch/yixiz/DMSR/lr_sims_256/set*/catnorm.npy`

## Baselines & comparison

The natural baseline is the paper's own model: `SRS-map2map` is the code for
"AI-assisted super-resolution cosmological simulations" (Ni et al.), and it ships
the pretrained SR-GAN generator `external/SRS-map2map/SRmodel/G_z0.pt` (z=0). We
compare three methods on the *same* held-out paired set with the *same* metrics:

* `trilinear` — naive trilinear upsample of LR (reference floor),
* `srs` — pretrained SRS-map2map SR-GAN (`map2map.models.srsgan.G`), tiled with
  periodic pad + `narrow_like` trim as in `lr2sr.py`,
* `ours` — our trained generator.

Metrics per method: HR MSE, HR relative MSE, LR-reconstruction MSE
`mse(A(SR), LR)`, per-channel cross-correlation with HR, and isotropic power
spectra (overlaid). Outputs: `comparison.json`, `spectra.npz`, `spectra.png`,
`slices.png`.

```bash
# on a GPU node (full Ng=64 -> 512)
OUR_CKPT=runs/mixed_real/ckpt_last.pt SET=set15 bash ~/slurm/dmsr/cosmo_sr_compare.sh
# or directly:
python scripts/compare_baseline.py \
    --our-config configs/mixed_real.yaml --our-checkpoint runs/mixed_real/ckpt_last.pt \
    --lr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr/set15.npy \
    --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set15.npy \
    --nsplit 8 --out runs/compare_set15
```

Notes: the SRS generator is a *stochastic* GAN (noise injection) using valid
convolutions (a chunk of LR size `L` -> HR `8L-42`); `--seed` fixes its noise.
Because it is trained adversarially for realism, expect it to match HR power
spectra / higher-order statistics better, while an MSE-trained deterministic `G`
tends to win pointwise HR MSE (but blurs small scales) — quantify both with this
script. The SRS repo's own `map2map` fork is loaded by prepending it to
`sys.path`, so run the comparison in a fresh process (don't import the installed
`map2map` first).

## GPU runs

This login node has no GPUs. For GPU training/eval, `ssh rhea` and submit SLURM
jobs (see `~/slurm/*.sh` for templates). All training scripts have a CPU
`--smoke` mode for local validation.

## Repository layout

```
cosmo_sr_project/
  external/{map2map, SRS-map2map}   # read-only, pinned
  src/cosmo_sr/                     # our code
  scripts/                          # setup / preprocess / run helpers
  tests/                            # pytest suite
  configs/                          # YAML configs
```
