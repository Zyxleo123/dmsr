#!/bin/bash
# Redraw every eval figure of a gather fine-tune run from its eval/step*.npz.
# CPU only: the npz holds the three density grids and the true subhalo centres,
# so nothing here needs the generator, the box, or a GPU. Restyling a panel must
# never mean retraining.
#
#   sbatch scripts/slurm/render_gather_slices_cpu.sbatch <envfile>
#SBATCH --job-name=hgrender
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-hgrender-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${HG_BOX:=set8}"
: "${HG_RUNG:=fine}"
: "${HG_LABEL:=}"
: "${HG_SLAB:=4}"
: "${HG_RUN_DIR:=}"

if [[ -z "$HG_RUN_DIR" ]]; then
    # The host id is only known after the training job picked it, so glob for
    # the run this configuration produced rather than guessing the directory.
    HG_RUN_DIR=$(ls -d "$DMSR_REWARD_ROOT/host_gather/${HG_BOX}_h"*"_${HG_RUNG}${HG_LABEL}" \
                 2>/dev/null | head -1 || true)
fi
if [[ -z "$HG_RUN_DIR" || ! -d "$HG_RUN_DIR/eval" ]]; then
    gate_failed "no gather run with eval/ under $DMSR_REWARD_ROOT/host_gather for ${HG_BOX}_*_${HG_RUNG}${HG_LABEL}"
fi

echo "=== redraw: $HG_RUN_DIR"
python -u scripts/features/render_gather_slices.py \
    --run-dir "$HG_RUN_DIR" --slab "$HG_SLAB"

echo "=== figures: $HG_RUN_DIR/slices/"
