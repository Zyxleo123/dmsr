#!/bin/bash
# Stage 1 (CPU): extract the held-out gather figure data. Reads two 3.2 GB
# fields ONCE and writes an npz+json the plot stage re-renders from cheaply.
#
#   sbatch scripts/slurm/gather_figures_data_cpu.sbatch <envfile>
#
# 64G, not the compare job 48G: field_to_particles materialises several
# (3,512,512,512) float64 arrays (3.2 GB each) alongside the loaded field.
# Fields are read one at a time and freed, so peak is ~one field + its particle
# arrays, not both fields at once.
#SBATCH --job-name=gfig_data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-gfig_data-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${GF_BOX:=set9}"
: "${GF_RUN_DIR:?GF_RUN_DIR must name the self-arm holdout export dir}"
: "${GF_HOST_ID:=168880}"
: "${GF_SELF_FIELD:?GF_SELF_FIELD must be the tuned field .npy}"
: "${GF_FROZEN_FIELD:?GF_FROZEN_FIELD must be the frozen-control field .npy}"
: "${GF_OUT:?GF_OUT must be the output stem}"

require_input "$GF_RUN_DIR/tiles.npz" "scripts/features/export_gather_tiles.py"
require_input "$GF_SELF_FIELD" "scripts/slurm/gather_splice_cpu.sbatch"
require_input "$GF_FROZEN_FIELD" "scripts/slurm/gather_splice_cpu.sbatch"

python -u scripts/features/gather_holdout_figures_data.py \
    --reward-root "$DMSR_REWARD_ROOT" \
    --box "$GF_BOX" \
    --run-dir "$GF_RUN_DIR" \
    --host-id "$GF_HOST_ID" \
    --self-field "$GF_SELF_FIELD" \
    --frozen-field "$GF_FROZEN_FIELD" \
    ${GF_BASE_FIELD:+--base-field "$GF_BASE_FIELD"} \
    ${GF_HR_FIELD:+--hr-field "$GF_HR_FIELD"} \
    --out "$GF_OUT"
