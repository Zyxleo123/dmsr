#!/usr/bin/env bash
#SBATCH --job-name=preprocess_paired
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-preprocess_paired-%j.out

set -euo pipefail
source /zfsauton/scratch/yixiz/miniconda3/etc/profile.d/conda.sh
conda activate pjm

cd /zfsauton2/home/yixiz/DMSR/cosmo_sr_project
python -u scripts/preprocess_paired.py \
    --sets set3 set4 set5 set6 set7 set8 set9 set10 set11 set12 set13 set14 \
    --out /zfsauton/scratch/yixiz/DMSR/paired_catnorm
