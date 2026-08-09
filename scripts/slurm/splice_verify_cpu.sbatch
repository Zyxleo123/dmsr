#!/bin/bash
# One real tile splice: assemble the field, run Rockstar on the complete
# periodic box, record measured dR next to the predicted one. One array task per
# splice; the array index IS the plan index.
#
# Same shape as a labelling job -- a full-box halo-finder run -- so the same
# resources. No member-particle table is needed here: only whole-box statistics
# are compared, and those come straight out of the ASCII catalog.
#
#   sbatch --array=0-11 scripts/slurm/splice_verify_cpu.sbatch <envfile> ARM=a
#SBATCH --job-name=sd_splice
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_splice-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${ARM:=a}"
IDX="${INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

require_input "$PROJECT/external/rockstar/rockstar" \
    "external/rockstar: make -j (see external/rockstar/BUILD_NOTE.txt)"

python -u scripts/reward/splice_verify.py \
    --config "$DIRECT_CFG" --run-name "$RUN_NAME" --arm "$ARM" \
    --stage run --index "$IDX"
