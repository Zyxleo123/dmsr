#!/bin/bash
# Proxy data, stage 2 (CPU): Rockstar with FULL_PARTICLE_CHUNKS=1 -> exact
# fractional tile summaries, and the ~7 GB .particles table DELETED in the same
# job. Nothing bulky survives the allocation and nothing goes near $HOME.
#
# Long and memory-hungry: the ASCII table is streamed in 8e6-row chunks (~1 GB
# resident) but the GADGET2 write and the halo finder itself want the rest.
#
# For SOURCE=hr this job ALSO extracts the Lagrangian member ids of the halos
# intervention_targets asked for, in a second pass over the same table before it
# is deleted. That is what lets every box have HR-directed interventions without
# a dedicated member-extraction run per box.
#
# It does NOT refresh the training index. Indexing is one job per submission,
# after all of these: see index_proxy_labels_cpu.sbatch for why.
#
#   sbatch scripts/slurm/label_proxy_candidates_cpu.sbatch <envfile> \
#       BOX=set0 SOURCE=frozen SEED=0
#   sbatch scripts/slurm/label_proxy_candidates_cpu.sbatch <envfile> \
#       BOX=set0 SOURCE=intervention ALPHA=0.5 MODE=vel
#SBATCH --job-name=sd_label
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_label-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set0}"
: "${SOURCE:=frozen}"
: "${SEED:=0}"
: "${ALPHA:=}"
: "${MODE:=both}"
: "${CHECKPOINT:=}"

read -r -a _BOXES <<< "${BOXES//,/ }"
BOX="${BOX:-${_BOXES[${SLURM_ARRAY_TASK_ID:-0}]}}"

require_input "$PROJECT/external/rockstar/rockstar" \
    "external/rockstar: make -j (see external/rockstar/BUILD_NOTE.txt)"

ARGS=(--config "$DIRECT_CFG" --run-name "$RUN_NAME" --stage label
      --box "$BOX" --source "$SOURCE" --seed "$SEED" --mode "$MODE")
[ -n "$ALPHA" ] && ARGS+=(--alpha "$ALPHA")
[ -n "$CHECKPOINT" ] && ARGS+=(--checkpoint "$CHECKPOINT")

python -u scripts/reward/collect_catalog_proxy_data.py "${ARGS[@]}"
