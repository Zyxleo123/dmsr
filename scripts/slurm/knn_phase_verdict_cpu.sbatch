#!/bin/bash
# kNN round 5 verdict: stop rules A (separation), B (ranking inside the working
# family), C (tile locality) for 3D and every phase-space lambda. JSON only.
# Runs afterany, so a lambda that failed or timed out is reported as MISSING or
# partial instead of stranding this job; always exits 0 once python has run.
#
#   sbatch scripts/slurm/knn_phase_verdict_cpu.sbatch <envfile>
#SBATCH --job-name=knnphasev
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-knnphasev-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${KP_GATE_JSON:?KP_GATE_JSON must be round 4 knn_gate_crosscheck_<box>.json}"
: "${KP_KNN:?KP_KNN must be the label=path list (3d=..., l500=..., ...)}"
: "${KP_OUT:?KP_OUT must be the verdict json}"
if [[ ! -s "$KP_GATE_JSON" ]]; then
    echo "no round-4 gate JSON at $KP_GATE_JSON -- run submit_knn_gate_crosscheck.sh first"
    exit 0
fi
python -u scripts/features/knn_phase_verdict.py \
    --gate-json "$KP_GATE_JSON" --knn $KP_KNN --out "$KP_OUT"
