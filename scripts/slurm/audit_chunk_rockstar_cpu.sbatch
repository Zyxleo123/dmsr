#!/bin/bash
#SBATCH --job-name=rw_chunk_rs
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_chunk_rs-%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_chunk_rs-%j.out
#
# Diagnostic: Rockstar on Lagrangian chunk crops (HR vs SR2 base).
# Pass overrides as positional args (not --export / env-before-sbatch):
#   sbatch scripts/slurm/audit_chunk_rockstar_cpu.sbatch BOX=set8 CHUNKS=0,1,2,3
#   sbatch scripts/slurm/audit_chunk_rockstar_cpu.sbatch CHUNK_HR=128 CHUNKS=0,1
set -euo pipefail
_D="$(cd "$(dirname "$0")" && pwd)"
[[ -r "$_D/_reward_common.sh" ]] \
    || { echo "missing $_D/_reward_common.sh" >&2; exit 1; }
# shellcheck source=/dev/null
source "$_D/_reward_common.sh" "$@"

BOX="${BOX:-set8}"
CHUNKS="${CHUNKS:-0,1}"
SOURCES="${SOURCES:-hr,base}"
BASE_SEED="${BASE_SEED:-0}"
PERIODIC="${PERIODIC:-1}"
EXTRA=()
[ -n "${CHUNK_HR:-}" ] && EXTRA+=(--set "geometry.chunk_hr=$CHUNK_HR")

python scripts/reward/audit_chunk_rockstar.py \
    --box "$BOX" --chunks "$CHUNKS" --sources "$SOURCES" \
    --base-seed "$BASE_SEED" --periodic "$PERIODIC" \
    ${EXTRA[@]+"${EXTRA[@]}"}
