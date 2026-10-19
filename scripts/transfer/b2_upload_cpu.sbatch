#!/bin/bash
# Run b2_upload.sh tiers as a CPU batch job: rhea's login node killed the
# detached upload mid-tier-1 (no rclone error, the process just vanished).
#
#   sbatch scripts/transfer/b2_upload_cpu.sbatch 0 1
#   sbatch --dependency=afterany:<ids> scripts/transfer/b2_upload_cpu.sbatch verify
#
#SBATCH --job-name=b2_up
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/_b2_stage/slurm-b2_up-%j.out

set -euo pipefail
echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 8G  time 12:00:00  partition cpu  tiers: $*"
nvidia-smi -L 2>/dev/null || echo "(no GPU: CPU job)"
RCLONE="${RCLONE:-/zfsauton2/home/yixiz/bin/rclone}"
"$RCLONE" lsd b2:proteina --contimeout 30s --timeout 60s \
    || { echo "B2 unreachable from $(hostname): this node has no route out"; exit 1; }
PROJECT=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project
RCLONE="$RCLONE" bash "$PROJECT/scripts/transfer/b2_upload.sh" "$@"
