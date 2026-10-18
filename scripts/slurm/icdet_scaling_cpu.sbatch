#!/bin/bash
# IC determinism, scaling readout (CPU, seconds): is recovery a function of how many
# of the INPUT run's own particles an object has? Reads the match .objects.npz of
# each run listed in ICD_SCALING_RUNS ("label:npz:cand:coarse_ng ..."), writes a JSON
# and the figure.
#
#   sbatch [--dependency=afterany:<match>] scripts/slurm/icdet_scaling_cpu.sbatch <envfile>
#
#SBATCH --job-name=icd_scale
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:15:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-icd_scale-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
source "$_D/_icdet_common.sh"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 4G  time 0:15:00  partition cpu"
nvidia-smi -L 2>/dev/null || echo "(no GPU: CPU job)"
: "${ICD_SCALING_RUNS:?ICD_SCALING_RUNS (label:npz:cand:coarse_ng ...) is required}"
ARGS=()
read -r -a _R <<< "$ICD_SCALING_RUNS"
for r in "${_R[@]}"; do
    IFS=: read -r LAB NPZ CAND NG <<< "$r"
    if [ ! -f "$NPZ" ]; then echo ">>> $LAB: no $NPZ (its match stage did not finish) -- left out"; continue; fi
    ARGS+=(--run "$LAB" "$NPZ" "$CAND" "$NG")
done
[ "${#ARGS[@]}" -gt 0 ] || { echo ">>> no match outputs at all -- nothing to read"; exit 0; }

python -u scripts/features/ic_determinism.py scaling "${ARGS[@]}" \
    --lag-tol "$ICD_LAG_TOL" --out "$ICD_SCALING_JSON"
python -u scripts/features/ic_determinism_plot.py \
    --scaling "$ICD_SCALING_JSON" --scaling-out "$ICD_SCALING_FIG"
