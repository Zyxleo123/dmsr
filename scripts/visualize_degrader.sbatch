#!/usr/bin/env bash
#SBATCH --job-name=viz_degrader
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-viz_degrader-%j.out

set -euo pipefail
source /zfsauton/scratch/yixiz/miniconda3/etc/profile.d/conda.sh
conda activate pjm

cd /zfsauton2/home/yixiz/DMSR/cosmo_sr_project
python -u scripts/visualize_degrader.py \
    --run-dir runs/degrader_real_512to64_w32_d2_resembed_lr1e-4_s20k_allboxes_aug \
    --lr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr/set14.npy \
    --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set14.npy \
    --out runs/degrader_real_512to64_w32_d2_resembed_lr1e-4_s20k_allboxes_aug/eval_viz/set14_halo.png
