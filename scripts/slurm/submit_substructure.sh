#!/bin/bash
# Submit pilot step 5 -> 6 for the substructure module (default set8, in-sample):
#
#   train (GPU) --afterok--> sample field (GPU) --afterok-->
#     full-box Rockstar catalog (CPU) --afterok--> compare vs frozen HR + base (CPU)
#
# Only calls sbatch; every step's work lives in its own batch script. Config
# travels as VAR=value positional args (never `--export`, which gets jobs held on
# this cluster -- the reason the preambles parse positional args). The catalog and
# compare stages are the existing flow_rockstar jobs, reused unchanged: they are
# generic on a field .npy + TAG.
#
#   scripts/slurm/submit_substructure.sh
#   BOX=set8 SEED=0 N_STEPS=20 scripts/slurm/submit_substructure.sh
#   DRY=1 scripts/slurm/submit_substructure.sh          # print, do not submit
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SLURM_DIR/../.." && pwd)"
cd "$PROJECT"

SUB_CONFIG="${SUB_CONFIG:-configs/substructure_set8.yaml}"
BOX="${BOX:-set8}"
SEED="${SEED:-0}"
N_STEPS="${N_STEPS:-20}"
TAG="${TAG:-substructure_set8}"
RUN_NAME="${RUN_NAME:-substructure_set8}"

ZFS=/zfsauton/scratch/yixiz
REWARD_ROOT="$ZFS/DMSR/dmsr_reward"
RUN_DIR="$REWARD_ROOT/sr2_direct/runs/$RUN_NAME"
CKPT="$RUN_DIR/ckpt_last.pt"
FLOW_RS_ROOT="$REWARD_ROOT/flow_rockstar"
FIELD_OUT="$FLOW_RS_ROOT/fields/${BOX}__${TAG}__seed${SEED}.npy"

sub() {  # echo + (unless DRY) run, returning the parsable job id
    if [[ "${DRY:-0}" == "1" ]]; then echo "DRY: sbatch $*" >&2; echo "DRYID"; else sbatch --parsable "$@"; fi
}

# Do not leak this shell's SLURM_* into sbatch (breaks env retrieval if we are
# inside an allocation).
for v in $(env | grep -oE '^SLURM_[A-Z_]+' || true); do unset "$v"; done

# 1. train (GPU) -- skipped with SKIP_TRAIN=1 to gate an existing checkpoint.
if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
    echo "SKIP_TRAIN=1: gating existing checkpoint $CKPT (no training job)"
    sample_dep=()
else
    train_id=$(sub scripts/slurm/train_substructure_gpu.sbatch \
        SUB_CONFIG="$SUB_CONFIG" SUB_BOX="$BOX" RUN_NAME="$RUN_NAME")
    echo "submitted train   -> job $train_id (GPU, general)"
    sample_dep=(--dependency=afterok:"$train_id")
fi

# 2. sample field (GPU), after training (or straight away if SKIP_TRAIN)
sample_id=$(sub "${sample_dep[@]}" \
    scripts/slurm/sample_substructure_gpu.sbatch \
    SUB_CONFIG="$SUB_CONFIG" CKPT="$CKPT" BOX="$BOX" TAG="$TAG" \
    SEED="$SEED" N_STEPS="$N_STEPS" RUN_NAME="$RUN_NAME")
echo "submitted sample  -> job $sample_id (GPU${sample_dep:+, afterok:$train_id})"

# 3. full-box Rockstar catalog (CPU), after sampling -- reused flow_rockstar job
cat_id=$(sub --dependency=afterok:"$sample_id" \
    scripts/slurm/flow_rockstar_catalog_cpu.sbatch \
    BOX="$BOX" TAG="$TAG" SEED="$SEED" FIELD_OUT="$FIELD_OUT")
echo "submitted catalog -> job $cat_id (CPU, afterok:$sample_id)"

# 4. compare vs frozen HR + base (CPU), after the catalog -- reused flow_rockstar job
cmp_id=$(sub --dependency=afterok:"$cat_id" \
    scripts/slurm/flow_rockstar_compare_cpu.sbatch \
    BOX="$BOX" CANDIDATES="$TAG")
echo "submitted compare -> job $cmp_id (CPU, afterok:$cat_id)"

LOGS_D="$REWARD_ROOT/sr2_direct/logs"
LOGS_R="$FLOW_RS_ROOT/logs"
cat <<EOF

=== monitor ===
  train  : tail -F $LOGS_D/slurm-sub_train-*.out    # READ THIS FIRST before trusting the chain
  sample : tail -F $LOGS_D/slurm-sub_sample-*.out
  catalog: tail -F $LOGS_R/slurm-flow_rs_cat-*.out
  compare: tail -F $LOGS_R/slurm-flow_rs_cmp-*.out

=== results (when done) ===
  checkpoint      : $CKPT
  sampled field   : $FIELD_OUT
  flow catalog    : $REWARD_ROOT/catalog_cache/${BOX}__candidate__${TAG}.json
  comparison JSON : $FLOW_RS_ROOT/$BOX/flow_catalog_comparison.json
  mass-function   : $FLOW_RS_ROOT/$BOX/flow_mass_function.png
  references       : $REWARD_ROOT/catalog_cache/${BOX}__hr__hr.json , ${BOX}__base__base.json

=== gate (docs/sr2_substructure_module.md section 9 step 6) ===
  subhalo count inside hosts > 1e14 should move 0.07 -> 0.4+, host mass function
  above 200 particles unchanged. Read the count table + subhalo mass function in
  the comparison outputs above.
EOF
