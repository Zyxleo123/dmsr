#!/bin/bash
# Stage 1 (CPU): splice a gather fine-tune's tuned tiles into the frozen SR2 box.
#
# No GPU: the run already saved its final tiles to tiles.npz, and the frozen box
# is cached, so this is a numpy copy. Two full (6, 512^3) float32 boxes are live
# at once inside splice_tiles (3.2 GiB each), hence 48 GB.
#
#   sbatch scripts/slurm/gather_splice_cpu.sbatch <envfile>
#SBATCH --job-name=hg_splice
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-hg_splice-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${HG_BOX:=set8}"
: "${HG_SEED:=0}"
: "${HG_RUN_DIR:?HG_RUN_DIR must name a host_gather run directory}"
: "${HG_WHICH:=out}"
: "${HG_RS_TAG:=}"
# A completed field is reused and only the sidecar rewritten, so a killed
# splice costs a minute on retry. Set to 1 to force a rebuild.
: "${HG_SPLICE_FORCE:=0}"

require_input "$HG_RUN_DIR/tiles.npz" "scripts/slurm/finetune_host_gather_gpu.sbatch"

echo "=== splice: run=$HG_RUN_DIR which=$HG_WHICH box=$HG_BOX"
python -u scripts/reward/splice_gather_field.py \
    --run-dir "$HG_RUN_DIR" \
    --box "$HG_BOX" \
    --seed "$HG_SEED" \
    --which "$HG_WHICH" \
    ${HG_RS_TAG:+--tag "$HG_RS_TAG"} \
    $([ "$HG_SPLICE_FORCE" = "1" ] && echo --force)
