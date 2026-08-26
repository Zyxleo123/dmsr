#!/bin/bash
# Submit the flow -> Rockstar pipeline for the held-out box (default set15):
#
#   per checkpoint:  sample field (GPU, h200) --afterok--> full-box catalog (CPU)
#   once, at the end: compare all flow catalogs vs frozen HR + base (CPU)
#
# Only calls sbatch; every step's work lives in its own batch script. Config is
# passed as VAR=value positional args (never `--export`, which gets jobs held on
# this cluster). CANDIDATES is comma-joined so it survives as one argument.
#
#   scripts/slurm/submit_flow_rockstar.sh
#   BOX=set15 SEED=0 scripts/slurm/submit_flow_rockstar.sh
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SLURM_DIR/../.." && pwd)"
cd "$PROJECT"

BOX="${BOX:-set15}"
SEED="${SEED:-0}"
N_STEPS="${N_STEPS:-20}"

# Checkpoints to evaluate: "config|ckpt|tag".
CKPTS=(
    "configs/flow_cascade.yaml|runs/flow_cascade/ckpt_last.pt|flow_cascade"
    "configs/flow_unet_cascade.yaml|runs/flow_unet_cascade/ckpt_last.pt|flow_unet_cascade"
)

# Do not leak this shell's SLURM_* into sbatch (breaks env retrieval if we are
# inside an allocation).
for v in $(env | grep -oE '^SLURM_[A-Z_]+' || true); do unset "$v"; done

COMMON="BOX=$BOX SEED=$SEED"
cat_jobids=()
tags=()

for spec in "${CKPTS[@]}"; do
    IFS='|' read -r cfg ckpt tag <<< "$spec"
    tags+=("$tag")

    gen_id=$(sbatch --parsable scripts/slurm/flow_rockstar_sample_gpu.sbatch \
        $COMMON N_STEPS="$N_STEPS" CONFIG="$cfg" CKPT="$ckpt" TAG="$tag")
    echo "submitted sample  $tag -> job $gen_id (GPU, project/h200)"

    cat_id=$(sbatch --parsable --dependency=afterok:"$gen_id" \
        scripts/slurm/flow_rockstar_catalog_cpu.sbatch \
        $COMMON TAG="$tag")
    echo "submitted catalog $tag -> job $cat_id (CPU, afterok:$gen_id)"
    cat_jobids+=("$cat_id")
done

dep=$(IFS=:; echo "${cat_jobids[*]}")
cand=$(IFS=,; echo "${tags[*]}")
cmp_id=$(sbatch --parsable --dependency=afterok:"$dep" \
    scripts/slurm/flow_rockstar_compare_cpu.sbatch \
    BOX="$BOX" CANDIDATES="$cand")
echo "submitted compare -> job $cmp_id (CPU, afterok:$dep)"

LOGS="/zfsauton/scratch/yixiz/DMSR/dmsr_reward/flow_rockstar/logs"
ROOT="/zfsauton/scratch/yixiz/DMSR/dmsr_reward/flow_rockstar"
CACHE="/zfsauton/scratch/yixiz/DMSR/dmsr_reward/catalog_cache"
cat <<EOF

=== monitor ===
  sample/catalog/compare logs:
    tail -F $LOGS/slurm-flow_rs_gen-*.out
    tail -F $LOGS/slurm-flow_rs_cat-*.out
    tail -F $LOGS/slurm-flow_rs_cmp-*.out

=== results (when done) ===
  flow catalog metadata : $CACHE/${BOX}__candidate__<tag>.json
  comparison table+json : $ROOT/$BOX/flow_catalog_comparison.json
  mass-function figure   : $ROOT/$BOX/flow_mass_function.png
  reference catalogs     : $CACHE/${BOX}__hr__hr.json , ${BOX}__base__base.json
EOF
