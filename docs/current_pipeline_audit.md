# Current pipeline audit

_Audit for the "few-HR, many-LR" symmetry-randomized ambient-prior plan. Records
what already exists so the plan is not re-implemented from scratch. Verified
2026-07-15 against git `b2bd5d3`, conda env `pjm`._

## TL;DR — do not rebuild these

| Plan asks for | Already exists | Where |
|---|---|---|
| Analytic `A`, `A⁺`, `P_null` | ✅ (factor-2 octave form) | `operators/multiscale.py` (`block_average`, `block_upsample`, `null_projection`), `operators/base_upscaler.py` (`consistent_base`) |
| Box-level split | ✅ | configs `paired_hr_glob=set[0-2]`, `val=set15`, test=set14 (memory) |
| Few-pair conditional flow (B2) | ✅ and **run** | `train/train_flow.py`, `models/flow_unet.py`, runs `sr_flow_*` |
| Distributional LR-only supervision | ✅ (band-power, not operator) | `losses/flow.py::band_statistics_loss` |
| Ambient LR path | ⚠️ exists but **vacuous** | `losses/ambient.py` — see "Known issues" |
| Fair distributional comparison tool | ✅ | `scripts/compare_flow_baseline.py` |
| Residual/η diagnostic (deliverable 4) | ✅ done | see `docs/gate1_operator_coverage.md` and memory |
| Operator-**conditioned** denoiser (C1/C2/C3) | ❌ new | this is the actual new work |
| Subcell-shift operators `H_g=A∘T_g` | ❌ new | Gate 1 done (below); code TBD |
| Exact-consistent posterior sampler | partial | `inference/flow_sample.py` (flow, not diffusion) |

## Data

- **HR**: `/zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set*.npy`, shape
  `(6, 512, 512, 512)` float32. **16 paired boxes** (set0–set15).
- **LR (paired)**: `.../paired_catnorm/lr/set*.npy`, `(6, 64, 64, 64)` float32.
  So the full-box degradation is **s = 8** (512→64).
- **LR-only (ambient)**: `.../lr_sims/set*/catnorm.npy` — **350 boxes** at res 64;
  `.../lr_sims_256/set*/catnorm.npy` — 13 boxes at res 256.
- **Channels (6)**: `disp[0:3] + vel[3:6]` (canonical layout, asserted in
  `data/datasets.py::_augment_field`). Augmentation flips/permutes in 3-vector
  triples and negates vector components (a flip/rotate without sign flip is wrong).
- **Normalization**: `catnorm` = concatenated + per-channel normalized on disk.
- **`use_channels`** already supports **displacement-only** (`[0,1,2]`) training;
  code notes velocity carries ~99% of the raw MSE. (Relevant: Gate-1 η-analysis
  says velocity is noise-limited — displacement-only is the right first target.)

## Split (box-level, verified against config)

- paired **train**: `set0, set1, set2` (3 boxes)
- paired **val**: `set15`; **test**: `set14` (memory)
- ambient LR-only: 350 boxes (res 64), 13 boxes (res 256)
- Splits are at the **box** level before crops (correct; crops within a box are
  correlated). Every run should still print/save its resolved manifest.

## Operators (`operators/`)

- `multiscale.py::MultiScaleOperators(factor=2)` — the pipeline is a **factor-2
  octave cascade** (64→128→256→512), *not* a single s=8 operator:
  - `A_R = block_average` (`avg_pool3d(k=2, s=2)`)
  - `U_R = block_upsample` (`interpolate mode="nearest"`)
  - `P_null_R(h) = h − U_R(A_R(h))`
  - Verified identities: `A_R(U_R y)=y`, `A_R(P_null_R h)=0` (`tests/test_multiscale.py`).
- `base_upscaler.py::consistent_base` — makes any upscaler exactly LR-consistent:
  `B_cons(y) = B(y) + U(y − A(B(y)))` ⇒ `A(B_cons(y)) = y`. This is the plan's
  `x = A⁺(y) + P_null(r)` parameterization, already implemented.
- **Missing (new work)**: subcell-shift operators `H_g = A∘T_g`, their
  `H_g⁺ = T_g⁻¹∘A⁺`, and operator-context embeddings. Gate-1 math is done (below);
  the torch modules + `tests/operators/` are not yet written.

## Models (`models/`)

- `flow_unet.py` — scale-shared 3D U-Net velocity field for the null-space
  residual flow (the SR generator B2). ~3.1M params (ckpt 12.6 MB).
- Also present: `unet_baseline.py`, `residual_flow.py`, `latent_flow.py`,
  `residual_autoencoder.py`, `learned_degrader.py`, `stochastic_degrader.py`,
  `hblock_flow.py`, `wrappers.py`. (Not all audited line-by-line.)

## Losses (`losses/`)

- `flow.py`:
  - `flow_matching_loss` — linear-interpolant CFM on the null-space residual
    `r = P_null(x_HR)`, per adjacent octave. **paired**.
  - `degrade_consistency_loss` — `mse(A_R(x̂), y_R)`; ≈0 by construction (monitor).
  - `band_statistics_loss` — 8 log-spaced radial power bands of `rfftn`, MSE to an
    EMA of paired-residual band power. **This is the working distributional
    LR-only channel** (see results).
- `ambient.py` — `loss_ambient = mse(A(G(y)), y)`. **Vacuous** under the
  null-space parameterization (`A(x̂)≡y` by construction) and, per the
  identifiability finding, teaches nothing about `null(A)` because `A` is fixed
  for every sample. Do **not** use as a main path. The plan's operator-conditioned
  ambient branch is what replaces it.
- `supervised.py`, `regularizers.py` — present.

## Training / config

- Configs: `configs/*.yaml`; loader `utils/config.py`; shared loop `train/common.py`.
- Each run saves `config.yaml` + `env.json` + `metrics.csv` + tb/ + wandb/ into
  `runs/<name>/`. GPU launch: `bash ~/slurm/dmsr/cosmo_sr_train.sh flow` (login
  node has no GPU).
- Metrics schema (`metrics.csv`) already logs per-octave
  `val_{fm,cons,highk,allk,respow,zdiv}_R{64,128,256}` — close to the plan's
  schema; missing keys should be logged as NaN, not omitted.

## Existing results to build on (not to redo)

- **η / residual diagnostic (deliverable 4)** — DONE. `η = LR − A(HR)` on held-out
  set15: displacement η-fraction ≈ **0.008** (avg-pool explains ~99.2% of disp
  variance), velocity η-fraction ≈ **0.60** rel. to signal (≈33% of real-LR
  variance). η is broadband/near-white. See `docs/gate1_operator_coverage.md`.
- **Flow collapse + fix (fresh, finished 2026-07-15)** — pure paired flow
  (`sr_flow_pairedonly`, λ_band=0) mean-collapses: top-octave residual power
  `respow≈0.47`, diversity `zdiv≈0.06`. Adding the band-power distributional loss
  (`sr_flow_strongband`, λ_band=8) **fixes it**: `respow 0.47→0.94`, `zdiv 3×` —
  but **overshoots** high-k at the lower octaves (`highk≈1.6`). Open thread:
  per-octave λ_band tuning.

## Known issues / inconsistencies

1. `losses/ambient.py` is vacuous (above). Superseded by operator-conditioned branch.
2. **Framing mismatch**: plan assumes a single s=8 operator; the codebase is a
   factor-2 octave cascade. Gate-1 coverage (below) evaluates both; **factor-2 is
   recommended** (reuses infra, ≥95% null recovery).
3. `scripts/compare_baseline.py` only supports the deterministic `SimpleSRGenerator`;
   use `scripts/compare_flow_baseline.py` for the flow. Report **distributional**
   metrics (P(k)/r(k)/σ), not HR MSE (MSE rewards mean-collapse).
4. Report distributional metrics for all generative comparisons; bootstrap CIs
   **by box**, not by crop.
