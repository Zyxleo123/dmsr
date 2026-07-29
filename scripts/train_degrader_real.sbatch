#!/usr/bin/env bash
#SBATCH --job-name=degrader_real
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=gpu28
#SBATCH --output=slurm-degrader_real-%j.out

set -euo pipefail
source /zfsauton/scratch/yixiz/miniconda3/etc/profile.d/conda.sh
conda activate pjm

cd /zfsauton2/home/yixiz/DMSR/cosmo_sr_project
python -u -m cosmo_sr.train.train_degrader --config configs/learned_degrader_real.yaml
