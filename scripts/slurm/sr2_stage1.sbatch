#!/bin/bash
#SBATCH -J sr2_stage1
#SBATCH -p general
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH -o slurm-sr2_stage1-%j.out

set -euo pipefail
source /zfsauton/scratch/yixiz/miniconda3/etc/profile.d/conda.sh
conda activate pjm
cd /zfsauton2/home/yixiz/DMSR/cosmo_sr_project

OUT="$PWD/runs/sr2_baseline"
python scripts/sr2/freeze_baseline.py --out "$OUT"

# Seed check already passed on prior runs; re-run only if missing.
if [[ ! -f "$OUT/seed_verify.json" ]]; then
  python scripts/sr2/verify_seeds.py --box set14 --seeds 0,1,2,3 \
    --out "$OUT/seed_verify.json"
fi

# Nested seeds 0..31 for Stage-2 Gate A; field+halo on test boxes.
# Absolute --out so Rockstar never sees a cwd-relative snap path.
python scripts/sr2/stage1_subhalo_diagnose.py \
  --boxes set12,set13,set14,set15 \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  --out "$OUT/stage1"

python scripts/sr2/stage1_analyze.py --stage1 "$OUT/stage1" --seed0-only
python scripts/sr2/stage2_oracle_tts.py \
  --stage1 "$OUT/stage1" \
  --n-max 32 \
  --out "$OUT/stage2"

echo "=== Stage 1–2 finished ==="
cat "$OUT/stage2/gate_a.json"
