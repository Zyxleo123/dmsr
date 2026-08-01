#!/bin/bash
# Rung 2 (CPU): rank one scored CEM population and write the elites forward.
#
# Reads only saved artefacts (scored/*.json, summaries/*.jsonl), so it belongs on
# the CPU partition -- asking for a GPU to rank eight numbers just queues behind
# real work.
#
#   sbatch scripts/slurm/cem_select_elites_cpu.sbatch CEM_RUN=cem_a ITER=0
#
#SBATCH --job-name=cem_select
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-cem_select-%j.out

set -euo pipefail

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

set -euo pipefail

: "${CEM_CFG:=configs/reward/cem.yaml}"
: "${CEM_RUN:=cem_a}"
: "${ITER:=0}"
: "${REWARD_MODEL:=$DMSR_REWARD_ROOT/reward_model/reward_model.json}"
: "${KEEP_NOISE:=0}"

ORACLE="$DMSR_REWARD_ROOT/oracle/${CEM_RUN}_it${ITER}"
require_input "$ORACLE/candidates.json" "scripts/slurm/cem_generate_gpu.sbatch"
require_input "$REWARD_MODEL" "scripts/slurm/fit_reward_model_cpu.sbatch"

python -u scripts/reward/cem_select_elites.py \
    --config "$REWARD_CFG" --cem-config "$CEM_CFG" \
    --run-name "$CEM_RUN" --iteration "$ITER" \
    --reward-model "$REWARD_MODEL" \
    $([ "$KEEP_NOISE" = "1" ] && echo --keep-noise)

echo "=== elites -> $ORACLE/elites.json"
