#!/usr/bin/env bash
# Stochastic (conditional flow-matching) degrader.
#
#   sbatch scripts/train_stoch_degrader.sbatch                          # disp-only (recommended)
#   sbatch scripts/train_stoch_degrader.sbatch configs/stoch_degrader_real.yaml   # all 6 channels
#
#SBATCH --job-name=stoch_degrader
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=gpu28
#SBATCH --output=slurm-stoch_degrader-%j.out

set -euo pipefail
source /zfsauton/scratch/yixiz/miniconda3/etc/profile.d/conda.sh
conda activate pjm

cd /zfsauton2/home/yixiz/DMSR/cosmo_sr_project
CONFIG="${1:-configs/stoch_degrader_disp.yaml}"
python -u -m cosmo_sr.train.train_stoch_degrader --config "$CONFIG"
