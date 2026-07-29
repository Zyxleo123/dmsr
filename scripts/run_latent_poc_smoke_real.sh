#!/bin/bash
# Fast debug of the latent POC pipeline on REAL data (NOT SyntheticPyramidDataset).
# Same modules/configs as run_latent_poc.sh, but tiny steps / small crops / no wandb.
# Intended to catch integration bugs quickly on a GPU (or CPU) node.
set -euo pipefail

PROJ="${PROJ:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
PYTHON="${PYTHON:-/zfsauton/scratch/yixiz/miniconda3/envs/pjm/bin/python}"
HR_VAL="${HR_VAL:-/zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set15.npy}"
cd "${PROJ}"

# Small, real-data overrides. crop_hr=64 keeps 3D volumes tiny while still real.
SMOKE_DATA="data.crop_hr=64"
SMOKE_TRAIN="train.steps=30 train.eval_every=10 train.save_every=0 train.amp=false"
SMOKE_WANDB="wandb.mode=disabled"

echo "=== [smoke-real] Experiment 0: operator sanity ==="
"${PYTHON}" -m cosmo_sr.eval.real_data_sanity --hr "${HR_VAL}" --crop 64 \
  --out runs/smoke_real/poc_real_data_sanity

echo "=== [smoke-real] residual AE ==="
"${PYTHON}" -m cosmo_sr.train.train_residual_ae \
  --config configs/poc_residual_ae_real3.yaml \
  --set ${SMOKE_DATA} ${SMOKE_TRAIN} ${SMOKE_WANDB} \
        output.run_dir=runs/smoke_real/residual_ae

"${PYTHON}" -m cosmo_sr.eval.eval_residual_ae \
  --config configs/poc_residual_ae_real3.yaml \
  --checkpoint runs/smoke_real/residual_ae/ckpt_last.pt \
  --hr "${HR_VAL}" --crop 64 --n-crops 1 \
  --out runs/smoke_real/residual_ae/eval_set15

echo "=== [smoke-real] learned degrader ==="
"${PYTHON}" -m cosmo_sr.train.train_degrader \
  --config configs/poc_learned_degrader_real3.yaml \
  --set ${SMOKE_DATA} ${SMOKE_TRAIN} ${SMOKE_WANDB} \
        output.run_dir=runs/smoke_real/learned_degrader

echo "=== [smoke-real] latent flow (paired only) ==="
"${PYTHON}" -m cosmo_sr.train.train_latent_flow \
  --config configs/poc_latent_flow_real3_paired_only.yaml \
  --set ${SMOKE_DATA} ${SMOKE_TRAIN} ${SMOKE_WANDB} \
        ae_checkpoint=runs/smoke_real/residual_ae/ckpt_last.pt \
        degrader_checkpoint=runs/smoke_real/learned_degrader/ckpt_last.pt \
        loss.eval_sample_steps=4 \
        output.run_dir=runs/smoke_real/latent_flow_paired_only

"${PYTHON}" -m cosmo_sr.eval.eval_latent_flow \
  --config configs/poc_latent_flow_real3_paired_only.yaml \
  --checkpoint runs/smoke_real/latent_flow_paired_only/ckpt_last.pt \
  --hr "${HR_VAL}" --crop 64 --n-steps 4 --diversity 2 --cfg-scales 0.0 1.0 \
  --set ae_checkpoint=runs/smoke_real/residual_ae/ckpt_last.pt \
        degrader_checkpoint=runs/smoke_real/learned_degrader/ckpt_last.pt \
  --out runs/smoke_real/latent_flow_paired_only/eval_set15

echo "Done: latent POC smoke-real"
