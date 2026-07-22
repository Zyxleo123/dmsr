# DMSR stage runbook — exact commands

All GPU launches go through `ssh rhea`. **This is not optional**: `sbatch` is denied
for this account cluster-wide, `srun` only works from the login node, and
`cosmo_sr_train.sh` checks `$SLURM_JOB_ID` *first* — inside an existing allocation it
silently runs the trainer on the current node with **no GPU request and no error**.
A previous session lost four jobs that way (exit 137, OOM inside an 8 GB interactive
session).

Always check first:

```bash
echo $SLURM_JOB_ID; hostname     # if SLURM_JOB_ID is set, you are INSIDE an allocation
```

Log paths passed to `ssh rhea` must be **absolute** — the redirect is evaluated before
any `cd`, and `/tmp` is node-local and invisible from rhea.

---

## 0. Tests

```bash
cd /zfsauton2/home/yixiz/DMSR/cosmo_sr_project
conda activate pjm

# Unit tests (fast, ~40 s)
PYTHONPATH=src python -m pytest tests/dmsr -o addopts="" -q \
    --deselect tests/dmsr/test_dmsr_training_smoke.py::test_stage_c_and_d_short_equal_compute_run

# Full suite including the end-to-end Stage C/D run on miniature on-disk boxes
PYTHONPATH=src python -m pytest tests -o addopts="" -q
```

## 1. Smoke tests (CPU, before any GPU job)

```bash
# 8.1 overfit a tiny paired batch / 8.2 critic separation / 8.3 LR-only adversarial update
PYTHONPATH=src python -m pytest tests/dmsr/test_dmsr_training_smoke.py -o addopts="" -q -k "not equal_compute"

# 8.4 Stage C vs D short equal-compute run (real loaders, mini boxes)
PYTHONPATH=src python -m pytest tests/dmsr/test_dmsr_training_smoke.py -o addopts="" -q -k equal_compute

# Trainer's own synthetic smoke path, per stage
for s in det a b c d e; do
  PYTHONPATH=src python -m cosmo_sr.train.train_dmsr \
      --config configs/dmsr/stage_a_paired_flow.yaml --stage $s --smoke
done

# Compute-budget audit (must match between C and D)
PYTHONPATH=src python -m cosmo_sr.train.train_dmsr --config configs/dmsr/stage_c_critic_pairedlr.yaml --audit-compute
PYTHONPATH=src python -m cosmo_sr.train.train_dmsr --config configs/dmsr/stage_d_critic_alllr.yaml   --audit-compute
```

## 2. Stages

Helper — every launch uses this shape:

```bash
launch () {   # launch <jobname> <config> <logname> [extra --set args...]
  local NAME=$1 CFG=$2 LOG=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/runs/dmsr/logs/$3.log
  shift 3
  ssh rhea "setsid nohup env CONFIG=$CFG SLURM_JOB_NAME=$NAME SLURM_TIME=12:00:00 SLURM_MEM=96G \
      bash /zfsauton2/home/yixiz/slurm/dmsr/cosmo_sr_train.sh dmsr $* > $LOG 2>&1 < /dev/null &"
}
mkdir -p runs/dmsr/logs
```

### Encoder SSL pretraining (prerequisite for Stages B-E)

```bash
ssh rhea "setsid nohup env CONFIG=configs/dmsr/lr_ssl.yaml SLURM_JOB_NAME=dmsrSSL \
    SLURM_TIME=08:00:00 SLURM_MEM=96G \
    bash /zfsauton2/home/yixiz/slurm/dmsr/cosmo_sr_train.sh dmsr_ssl \
    > /zfsauton2/home/yixiz/DMSR/cosmo_sr_project/runs/dmsr/logs/lr_ssl.log 2>&1 < /dev/null &"
# writes runs/dmsr/lr_ssl/encoder.pt
```

### Baselines and Stages A-B

```bash
# baseline_upsample: nothing to train -- scored directly in step 3.

launch dmsrDET configs/dmsr/paired_deterministic.yaml      det
launch dmsrA   configs/dmsr/stage_a_paired_flow.yaml       stage_a        # RUNNING
launch dmsrB   configs/dmsr/stage_b_paired_flow_lrssl.yaml stage_b        # needs encoder.pt
```

### Stages C and D — three seeds each (the main experiment)

Both need `runs/dmsr/stage_b/ckpt_best.pt`. Run the pairs on matched hardware.

```bash
for s in 0 1 2; do
  launch dmsrC$s configs/dmsr/stage_c_critic_pairedlr.yaml stage_c_s$s \
      --set train.seed=$s output.run_dir=runs/dmsr/stage_c_s$s wandb.name=dmsr_stage_c_s$s
  launch dmsrD$s configs/dmsr/stage_d_critic_alllr.yaml   stage_d_s$s \
      --set train.seed=$s output.run_dir=runs/dmsr/stage_d_s$s wandb.name=dmsr_stage_d_s$s
done
```

Stage D **aborts at startup** if the balanced `source_classifier_auc > 0.60`. That is
intended — fix the matching (raise `env.n_bins`/`env.n_dims`, reduce descriptor
dimensionality, or restrict support) rather than setting `env.allow_auc_fail=true`,
which exists only for deliberate inspection.

### Stage E (optional; only after Stage D is stable and `tests/dmsr/test_cubic.py` passes)

```bash
launch dmsrE configs/dmsr/stage_e_alllr_equivariant.yaml stage_e
```

## 3. Evaluation

```bash
# Score every model on the held-out TEST box, with the A_plus(y) floor.
#
# --max-crops 256 is NOT arbitrary. Crops are stratified into 3 environment bins by
# Mahalanobis distance from the paired-training environment (edges = paired p50/p90),
# and success criterion 3 asks whether Stage D gains most in the OUTER bin. That bin
# holds ~17% of held-out crops, so:
#     --max-crops  12 -> bin2 n~2    (measured; useless)
#     --max-crops  32 -> bin2 n~5    (still far too few)
#     --max-crops 256 -> bin2 n~44   (workable)
# NOTE the test split is now 12 boxes (set3-set14) = 12 * 8^3 = 6144 available crops,
# and max_crops subsamples across ALL of them, so 256 gives only ~21 crops per box.
# For the box-level bootstrap use --max-crops 768 (~64/box); 256 is the floor.
# squeezed_cross_bispectrum is per-crop noisy (0.26 / 1.24 / 0.46 across bins at
# n=6/4/2), so it needs the averaging too.
# --batch-size 1 is REQUIRED: a batch spanning two boxes would produce a metric
# belonging to neither and corrupt the box-level bootstrap (the script raises).
PYTHONPATH=src python scripts/dmsr_eval.py --mode evaluate \
    --config configs/dmsr/stage_d_critic_alllr.yaml --split test --baseline \
    --ckpt paired_det:runs/dmsr/paired_deterministic/ckpt_best.pt \
    --ckpt stage_a:runs/dmsr/stage_a/ckpt_best.pt \
    --ckpt stage_b:runs/dmsr/stage_b/ckpt_best.pt \
    --ckpt stage_c:runs/dmsr/stage_c_s0/ckpt_best.pt \
    --ckpt stage_d:runs/dmsr/stage_d_s0/ckpt_best.pt \
    --max-crops 768 --batch-size 1 --out runs/dmsr/eval_test

# Per-seed metrics for the C-vs-D comparison
for s in 0 1 2; do
  for st in c d; do
    PYTHONPATH=src python scripts/dmsr_eval.py --mode evaluate \
        --config configs/dmsr/stage_${st}_*.yaml --split test \
        --ckpt stage_${st}:runs/dmsr/stage_${st}_s$s/ckpt_best.pt \
        --max-crops 768 --batch-size 1 --out runs/dmsr/stage_${st}_s$s/eval_test
  done
done

# The C-vs-D table. Reports BOTH estimators: a seed-level table and the headline
# BOX-LEVEL paired bootstrap over the 12 held-out boxes (set3-set14). The verdict is
# decided on the box-level statistics whenever >= 2 boxes are present.
PYTHONPATH=src python scripts/dmsr_eval.py --mode compare \
    --c runs/dmsr/stage_c_s0/eval_test/metrics.csv \
        runs/dmsr/stage_c_s1/eval_test/metrics.csv \
        runs/dmsr/stage_c_s2/eval_test/metrics.csv \
    --d runs/dmsr/stage_d_s0/eval_test/metrics.csv \
        runs/dmsr/stage_d_s1/eval_test/metrics.csv \
        runs/dmsr/stage_d_s2/eval_test/metrics.csv \
    --out runs/dmsr/cvd
```

## 4. Monitoring

```bash
squeue -u yixiz
tail -f runs/dmsr/logs/stage_a.log
```

Key training keys to watch:

| key | what it tells you |
|---|---|
| `exact_consistency_rel` | must stay ~1e-7. Any drift is a structural bug, not a tuning issue. |
| `grad_ratio_adv_flow` | calibration target **0.1-0.3**. Tune `adv.lambda_adv` if outside. |
| `sample_diversity` | mode collapse if it decays toward 0. |
| `critic_fake_paired_score` vs `critic_fake_unpaired_score` | a persistent gap in Stage D means the critic is still separating pools by source. |
| `Tk_error_high` | the mean-collapse signature this stage's critic is meant to fix. |
| `condition_shuffle_gap` | ~0 means the generator is ignoring `y`. |
