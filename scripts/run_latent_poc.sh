#!/bin/bash
# Latent residual flow proof-of-concept pipeline on REAL DMSR data.
#
# Runs, in order:
#   Exp 0  operator sanity          (CPU-friendly)
#   Exp 1  residual AE + eval       (GPU)
#   Exp 2  learned degrader         (GPU)
#   Exp 3  paired latent flow + eval(GPU)
#   Exp 4  latent flow + LR-only    (GPU)
#   Exp 5  final 64->512 cascade    (GPU)
#
# Run from the project root (or via ~/slurm/dmsr/latent_poc.sh on a GPU node).
set -euo pipefail

PROJ="${PROJ:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
PYTHON="${PYTHON:-/zfsauton/scratch/yixiz/miniconda3/envs/pjm/bin/python}"
HR_VAL="${HR_VAL:-/zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set15.npy}"
cd "${PROJ}"

echo "=== Experiment 0: real-data operator sanity ==="
"${PYTHON}" -m cosmo_sr.eval.real_data_sanity \
  --hr "${HR_VAL}" \
  --crop 128 \
  --out runs/poc_real_data_sanity

echo "=== Experiment 1: residual AE overfit on 3 real paired HR boxes ==="
"${PYTHON}" -m cosmo_sr.train.train_residual_ae \
  --config configs/poc_residual_ae_real3.yaml

"${PYTHON}" -m cosmo_sr.eval.eval_residual_ae \
  --config configs/poc_residual_ae_real3.yaml \
  --checkpoint runs/poc_residual_ae_real3/ckpt_last.pt \
  --hr "${HR_VAL}" \
  --crop 128 \
  --out runs/poc_residual_ae_real3/eval_set15

echo "=== Experiment 2: learned degrader real-data sanity ==="
"${PYTHON}" -m cosmo_sr.train.train_degrader \
  --config configs/poc_learned_degrader_real3.yaml

echo "=== Experiment 3: latent flow overfit (paired only) + eval ==="
"${PYTHON}" -m cosmo_sr.train.train_latent_flow \
  --config configs/poc_latent_flow_real3_paired_only.yaml

"${PYTHON}" -m cosmo_sr.eval.eval_latent_flow \
  --config configs/poc_latent_flow_real3_paired_only.yaml \
  --checkpoint runs/poc_latent_flow_real3_paired_only/ckpt_last.pt \
  --hr "${HR_VAL}" \
  --crop 128 \
  --n-steps 20 \
  --diversity 3 \
  --cfg-scales 0.0 1.0 2.0 \
  --out runs/poc_latent_flow_real3_paired_only/eval_set15

echo "=== Experiment 4: latent flow + LR-only band regularization ==="
"${PYTHON}" -m cosmo_sr.train.train_latent_flow \
  --config configs/poc_latent_flow_real3_plus_lronly.yaml

echo "=== Experiment 5: final 64->512 cascade demo on real validation field ==="
"${PYTHON}" -m cosmo_sr.eval.eval_latent_flow \
  --config configs/poc_latent_flow_real3_plus_lronly.yaml \
  --checkpoint runs/poc_latent_flow_real3_plus_lronly/ckpt_last.pt \
  --hr "${HR_VAL}" \
  --crop 256 \
  --n-steps 20 \
  --diversity 4 \
  --cfg-scales 0.0 1.0 2.0 \
  --out runs/poc_latent_flow_real3_plus_lronly/final_cascade_set15

echo "Done: latent POC pipeline"
