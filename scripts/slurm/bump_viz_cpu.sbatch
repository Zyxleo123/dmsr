#!/bin/bash
# Draw the bump figures from bump_cutouts.py's saved files (CPU, no recomputation).
#
#   sbatch scripts/slurm/bump_viz_cpu.sbatch [envfile|BC_OUT_DIR=...]
#
#SBATCH --job-name=bumpviz
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-bumpviz-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 8G  partition cpu"
: "${BC_OUT_DIR:=$DMSR_REWARD_ROOT/placement_forensics/bumps}"
require_input "$BC_OUT_DIR/bump_null.json" "scripts/slurm/bump_cutouts_cpu.sbatch"
python -u scripts/features/bump_viz.py --dir "$BC_OUT_DIR"
