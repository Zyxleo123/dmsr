# Test-time scaling for the pretrained SR2/SRS generator

Spend more compute at inference on one LR box — draw several SR realisations,
score them, keep the good one — without retraining anything. SR2's weights are
frozen throughout; the only thing that varies is the injected noise.

Code lives in [`src/cosmo_sr/tts/`](../src/cosmo_sr/tts) and `scripts/`.
`external/SRS-map2map` is read-only and untouched.

---

## What the pretrained generator's noise actually does

Before any of the machinery, two measurements that should frame every result.

**1. Five of the six noise sites are nearly switched off.** The learned
per-channel scale `AddNoise.std` in `SRmodel/G_z0.pt`:

| site | stage | acts at | `rms(std)` |
|------|-------|---------|-----------|
| z0 | coarse | 1x LR | 0.0010 |
| z1 | coarse | 2x LR | 0.0024 |
| z2 | middle | 2x LR | 0.0027 |
| z3 | middle | 4x LR | 0.0023 |
| z4 | fine | 4x LR | 0.0018 |
| **z5** | **fine** | **8x LR (= HR)** | **0.069** |

SR2's stochasticity is overwhelmingly a single fine-scale injection, 30x
stronger than every other site.

**2. It moves velocity, not displacement.** One real LR tile of `set15`
(`14^3` input), three seeds, pretrained `G_z0`, relative rms spread across
seeds:

| channels | rms | across-seed pairwise rms | relative |
|----------|-----|--------------------------|----------|
| displacement (0:3) | 1.074 | 0.0045 | **0.42%** |
| velocity (3:6) | 1.114 | 0.583 | **52%** |

Density is reconstructed from displacement, so the *density* statistics of two
candidates differ by well under a percent, consistent with the SR2 paper's
"<1% power-spectrum variation across noise draws". Any selection leverage
should be expected to live in the velocity field and in whatever fine-scale
density structure a 0.4% displacement perturbation can move.

**3. Which site does what.** Zeroing one site while holding the other five
fixed (same tile, real weights) — relative rms change in the output:

| site | displacement | velocity |
|------|--------------|----------|
| z0 (coarse, 1x)  | **0.183%** | 1.77% |
| z1 (coarse, 2x)  | 0.140% | 1.11% |
| z2 (middle, 2x)  | 0.077% | 0.62% |
| z3 (middle, 4x)  | 0.023% | 0.29% |
| z4 (fine, 4x)    | 0.025% | 0.25% |
| z5 (fine, 8x)    | 0.175% | **39.1%** |

Velocity is essentially z5 alone. Displacement — and therefore *density* — is a
different story: the coarse sites contribute as much as the fine one, all at the
0.1–0.2% level. That is the concrete reason Stage 4 optimises coarse noise
first: the density-side leverage is not concentrated at the fine scale, even
though the total variance is.

This is a prediction, not a result — the Stage-1 audit is what measures it on
complete boxes. It does say that a flat oracle curve on density metrics would
be the *expected* outcome, and the pipeline is gated so that finding stops the
work rather than being papered over.

---

## Stage 0 — reproducible multi-sample inference

`cosmo_sr.tts.sampling`

```python
from cosmo_sr.tts import load_controlled_generator, generate_srs_candidates

G = load_controlled_generator("external/SRS-map2map/SRmodel/G_z0.pt", scale_factor=8)
cands = generate_srs_candidates(G, lr_field, seeds=[0, 1, 2, 3], nsplit=8, pad=3)
```

* **A root seed defines one complete full-box realisation.** Each tile's six
  noise tensors come from `blake2b(root_seed, tile_coordinate, site)`, so
  traversal order, batching and device cannot change the result. (Hashing, not
  affine mixing: tile coordinates are small consecutive integers and nearby
  seeds do not give independent torch streams.)
* Noise is drawn on the **CPU** and moved to the device — CUDA's generator gives
  a different stream for the same seed, and a realisation must not depend on
  which device produced it.
* Every candidate carries its seed and full `InferenceConfig`.
* Candidates are never averaged. `iter_srs_candidates` streams them: a 512³
  six-channel box is 3.2 GB, so K=32 in memory is 100 GB.
* The legacy `super_resolve_srs` path is untouched.

## Stage 1 — oracle best-of-K audit

`scripts/eval_srs_tts.py` — two phases (`generate` writes one JSONL row per
(box, candidate) and is resumable; `analyze` builds curves, CIs and plots).

Per candidate, on **complete periodic boxes**: displacement/velocity/density
`r(k)` and `T(k)` in three k-bands, density power and PDF error, equilateral and
squeezed bispectrum error, velocity power and divergence-PDF error,
`|A(x_SR) − y|`, tile-seam discontinuity, and across-candidate diversity from
running moments.

Density comes from **CIC deposition of `q + Ψ` using all three displacement
components** — never from channel 0 treated as a density.

Three selection results: `random` (= K=1), `phase` oracle (HR `r(k)`/`T(k)`),
`statistical` oracle (density power/PDF, bispectra, velocity — no
realisation-specific high-k phase). Composite scores z-normalise each component
using **validation-box** statistics, so a metric cannot dominate through its
units. Confidence intervals resample **boxes**, and best-of-K averages over many
random size-K subsets rather than one arbitrary ordering.

**Gate:** continue only if best-of-16 improves a primary density/higher-order
metric by ≥5% relative with a box-bootstrap CI excluding zero, and nothing
important degrades. A flat curve ends the project, and `run_srs_tts.sh` exits
cleanly at that point.

## Stage 2 — test-time-available selection

`scripts/train_srs_verifier.py`, `cosmo_sr.tts.features`, `cosmo_sr.tts.verifier`

Cheap features first: operator consistency, tile-seam ratio, density power/PDF
plausibility against a **training-box** HR reference, joint density–velocity
summaries (speed inside overdense cells, `div v`–`δ` correlation), rotation and
flip consistency, and noise diagnostics. Each is tried alone as a selector
before any model is fitted.

Rotation consistency rotates the **vector components** with the lattice and
transforms the **same noise realisation** along with the input — comparing
against independently drawn noise would only measure SR2's stochasticity.
Translation is deliberately *not* a feature: a fully convolutional generator fed
input and noise shifted together is exactly translation equivariant, so the
residual is float noise. It is checked instead as a correctness test of the
global-noise indexing (Stage 5).

The learned selector is a pairwise ranker (linear, or a small MLP) trained with
a RankNet loss on candidate pairs from the same LR input, targeting the
**statistical** oracle — not high-k cross-correlation, which is not a function
of the input. Splits are by simulation box, never by crop. A conditional 3-D
patch verifier (`PatchVerifier`) is available if summary features prove
insufficient.

Reported: within-input Spearman, pairwise accuracy, verifier best-of-K quality,
selection regret, and guard metrics (density power, bispectra, velocity) that
must not degrade.

**Gate:** at K=16 the selector must beat random with a box-bootstrap CI
excluding zero, recover ≥50% of the statistical-oracle gain, and damage nothing.

## Stage 3 — explicit multiscale noise control

`cosmo_sr.tts.srs_noise.ControlledG` — a local re-implementation of SR2's
generator with identical state-dict keys, so `G_z0.pt` loads directly and no
`sys.path` juggling with the SRS fork is needed.

```python
y, z = G(lr, record=True)                       # replay a realisation exactly
y2   = G(lr, noise={"coarse": [z0, z1],         # or inject explicitly
                    "middle": [z2, z3],
                    "fine":   [z4, z5]})
```

Verified in `tests/tts/test_srs_noise.py`: **bit-identical** to upstream under
the same global seed, exact replay from recorded noise, expected site shapes,
each site individually affects the output, coarse-site perturbations spread
further than fine-site ones, and gradients reach all six tensors.

Upstream's `AddNoise` adds its noise **twice** (`x = x + noise; return x + noise`).
That looks like a typo but the checkpoint was trained with it, so it is
reproduced verbatim.

`noise_site_layout` tracks the exact global coordinate of every site through the
valid-convolution and upsample bookkeeping (`upsample: c -> 2c + 1`,
`conv3d k3: c -> c + 1`). That is what makes Stage 5 possible.

## Stage 4 — best-of-K plus noise refinement

`scripts/tts_stage45.py --mode refine`, `cosmo_sr.tts.refine`

Sample K, keep the best few by verifier score, then optimise their noise with
SR2 and the verifier frozen. Coarse → middle → fine with a shrinking step, so
large-scale structure settles before the ~10⁸ fine-scale variables (where
verifier exploitation is easiest) are unlocked. Every step pays

```
L_noise = λ_μ μ(z)² + λ_σ (σ(z) − 1)² + λ₂ ‖z − z₀‖² / N
```

with the trust-region term normalised by element count so one `λ₂` means the
same thing at sites that differ 512x in size.

Refinement runs **per tile** — backprop through 512 tiles at once is not
memory-feasible — against a linear differentiable surrogate of the verifier
(`LinearFeatureObjective` over `DIFFERENTIABLE_FEATURE_KEYS`; histogram features
have zero gradient and are excluded from the surrogate but still used to score
the final candidate).

A cross-entropy-method arm is the gradient-free control. Score trajectories,
per-site noise statistics and distance from `z₀` are recorded, and runs whose
noise leaves the training distribution are **rejected**, not silently returned.

Caveat inherited from this repo's CIC measurements: density reconstructed on a
*crop* is badly biased (r ≈ 0.08 against truth, σ inflated ~2.2x without a
~64-cell buffer). The tile-level objective is therefore only valid as a
*relative* comparison between candidates sharing the same crop — which is what
refinement needs — and absolute tile densities must not be read off it. All
reported metrics are computed on complete boxes.

## Stage 5 — globally coherent tiled inference

`cosmo_sr.tts.tiling`

One coordinate-indexed global noise lattice per site for the whole box; each
tile reads the window its coordinates point at. Because the generator is fully
convolutional over valid convolutions, two overlapping tiles reading the same
global noise produce **identical** values in their shared region — verified, not
approximated (`tests/tts/test_tiling.py`), along with periodicity across
opposite faces and invariance to tile processing order. The control test
confirms per-tile noise genuinely disagrees there.

Overlapping tiles are then selected jointly:

```
S_joint = Σ_i S_verifier(x_i, y_i) + λ_overlap Σ_(i,j) ‖x_i − x_j‖²_overlap
```

minimised by coordinate descent over each tile's candidate list. Cropping or
cosine-blending happens **after** selection — blending first averages away the
small-scale variance candidates differ in and calls the smoother seam an
improvement.

Worth knowing before tuning `λ_overlap`: once the noise *is* coordinate-indexed,
the overlap term is essentially zero by construction (tiles agree exactly), so
the joint objective collapses to the sum of verifier scores and coordinate
descent barely moves — observed in the smoke run, where the joint score fell by
0.0002% over 512 tiles while the seam ratio sat at 1.13. The pairwise term earns
its keep in the *per-tile* noise regime, where it is the only thing penalising
visible seams. Global noise and joint selection are alternative fixes for the
same problem, not a stack.

## Final table

`scripts/tts_final_table.py` compares SR2 single sample, random-of-K,
hand-crafted best-of-K, verifier best-of-K, oracle best-of-K, verifier +
refinement, and verifier + global joint search, for K ∈ {1,2,4,8,16,32}: all
metrics with 95% box-bootstrap intervals, remaining oracle regret, diversity
before and after selection, wall clock, quality vs inference compute, and
**ensemble-level** power and PDF before vs after selection — the check that a
selector improved the field rather than biasing the output distribution.

---

## Running it

**Everything runs through SLURM, including the figures.** Nothing in this
pipeline is meant to be executed on a login node or inside an interactive
allocation.

```bash
bash scripts/slurm/submit_tts.sh                        # whole chain, K = 16
K=32 OUT=runs/tts_k32 bash scripts/slurm/submit_tts.sh  # bigger candidate pool
STAGES="1 viz" bash scripts/slurm/submit_tts.sh         # only those stages
DRY=1 bash scripts/slurm/submit_tts.sh                  # print sbatch lines, submit nothing
```

`submit_tts.sh` only calls `sbatch` — it runs no work itself. Jobs are chained
with `--dependency=afterok`; Stage 4's two arms and Stage 5 are siblings that run
concurrently, and Stage 6 plus the figures wait for all of them:

```
stage1 oracle ──► stage2 verifier ──┬──► stage4 refine (gradient) ──┐
   (GPU, 1d)        (CPU, 2h)       ├──► stage4 refine (CEM)      ──┼─► stage6 table ─┬─► viz figures  (CPU)
                                    └──► stage5 global tiling     ──┘   (CPU, 2h)     └─► viz slices   (GPU)
```

| job | script | partition |
|---|---|---|
| Stage 1 oracle audit | `slurm/tts_stage1_oracle.sbatch` | `general` + 1 GPU |
| Stage 2 verifier | `slurm/tts_stage2_verifier.sbatch` | `cpu` |
| Stage 4 refinement (`ARM=grad\|cem`) | `slurm/tts_stage4_refine.sbatch` | `general` + 1 GPU |
| Stage 5 global tiling | `slurm/tts_stage5_global.sbatch` | `general` + 1 GPU |
| Stage 6 final table | `slurm/tts_stage6_table.sbatch` | `cpu` |
| Figures from artefacts | `slurm/tts_viz.sbatch` | `cpu` |
| Density-slice figures | `slurm/tts_viz_slices.sbatch` | `general` + 1 GPU |

Configuration is entirely by environment variable — `K`, `OUT`, `DATA`, `MODEL`,
`TRAIN_BOXES`, `VAL_BOXES`, `TEST_BOXES`, `KEEP`, `CHUNK`, `STRIDE`,
`LAM_OVERLAP`, `SLICE_BOX` — with the defaults in one place,
[`scripts/slurm/_tts_common.sh`](../scripts/slurm/_tts_common.sh).

**Never use `sbatch --export` on this cluster.** Any explicit export list, with
`ALL` or without, makes sbatch set `SLURM_GET_USER_ENV=1`; slurmd then tries to
rebuild the login environment on the compute node, that lookup fails here, and
the job is requeued and **held** with reason
`(user env retrieval failed requeued held)`. It never starts, and every
dependent job sits on `Dependency` forever. This was measured twice on this
chain — job 22184 with `--export=ALL,…` and job 22243 with a plain
`--export=VAR=…` list — and in both cases it struck whichever job was next to be
allocated a node. The jobs that have always worked here pass no `--export` at
all.

So `submit_tts.sh` writes one timestamped env file per submission under
`runs/.tts_env/` and passes its path as a **positional argument** to the batch
script; arguments travel through the job record untouched. `_tts_common.sh`
walks `"$@"`, sourcing anything that is a path and exporting anything of the
form `VAR=value` (which is how Stage 4 gets `ARM=grad` vs `ARM=cem`). A file
also has no trouble with spaces, so box lists need no comma encoding. One file
per submission means re-running the submitter cannot retroactively change the
configuration of a job that is still queued. The preamble is self-sufficient
from an empty environment — verify with:

```bash
env -i /bin/bash -c 'source scripts/slurm/_tts_common.sh runs/.tts_env/<file>.sh'
```

`scontrol release <jobid>` retries an already-held job in place, but it will
usually just re-hold; cancel and resubmit with the current submitter instead.

Each job body sets `set -euo pipefail` itself and resolves the preamble through
`$SLURM_SUBMIT_DIR` with an absolute fallback. Both matter: Slurm copies the
batch script to `/var/spool/slurm/job<id>/slurm_script`, so
`$(dirname "${BASH_SOURCE[0]}")` points at the spool directory and sources
nothing, and without `set -e` in the body the job then runs the system python
and **exits 0** — releasing the whole chain behind a stage that did no work
(job 22242).

**Gates are enforced inside the jobs, not by the submitter.** A failed gate
prints why and exits *zero*, so the rest of the chain still starts and each
stage reports its own skip — a non-zero exit would strand every dependent job in
`DependencyNeverSatisfied` with no explanation.

Cost: one 512³ candidate is roughly 500 CPU-core-minutes, ~40 s on an A100.
Stage 1 at K=16 over 16 boxes is a few GPU-hours. `rows.jsonl` is appended
incrementally and the generate phase resumes, so a job that hits its time limit
can simply be resubmitted.

Monitor with:

```bash
squeue -u $USER -o '%.10i %.16j %.9P %.2t %.10M %R'
tail -f slurm-tts_oracle-<jobid>.out
```

`scripts/run_srs_tts.sh` is the same pipeline as a single inline shell script.
It exists for debugging on a machine you already own; the SLURM path above is
the one to use.

Splits (all by simulation box, non-overlapping): `set0–set7` fit the HR
plausibility reference and the verifier, `set8–set11` fit the score normaliser,
`set12–set15` are where every reported number is read. SR2 itself is pretrained
on other data, so no box in this repo is in its training set.

## Tests

```bash
pytest tests/tts -q          # run one file at a time on a memory-limited node
```

Covers upstream parity and noise replay (Stage 3), seed determinism and order
invariance (Stage 0), CIC/metric correctness and bootstrap behaviour (Stage 1),
features/rankers (Stage 2), refinement and its OOD rejection (Stage 4), and
overlap/periodicity/joint selection (Stage 5).
