#!/bin/bash
# Knot oracle (CPU array, one variant per task): cool every dense outskirt
# peak's own particles to target x the surrounding dispersion (positions
# untouched), then run the frozen Rockstar config.
#
#   sbatch --array=0-<n-1> scripts/slurm/knot_oracle_cpu.sbatch <envfile>
#
# KO_VARIANTS: space-separated "name:field.npy:peaks.npz:k_core[:min_ratio[:save]]"
# (quote it). ":save" writes the modified field to KO_OUT_DIR for SUBFIND.
#
#SBATCH --job-name=knotorc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=8:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-knotorc-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 192G  partition cpu"
nvidia-smi -L 2>/dev/null || echo "(no GPU: CPU job)"
: "${BOX:=set9}"
: "${KO_VARIANTS:?KO_VARIANTS (name:field:peaks:k_core ...) is required}"
: "${KO_OUT_DIR:=$DMSR_REWARD_ROOT/knot_oracle}"
: "${KO_TARGET:=0.3}"
: "${KO_REGION_EXPORT:=$DMSR_REWARD_ROOT/member_gather/all_blocks_sm_coarse_soft/holdout_${BOX}/export.json}"
read -r -a _V <<< "$KO_VARIANTS"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#_V[@]}" ]; then echo ">>> index $IDX past KO_VARIANTS; nothing to do."; exit 0; fi
IFS=: read -r NAME FIELD PEAKS KCORE MINR SAVE <<< "${_V[$IDX]}"
EXTRA=(--min-ratio "${MINR:-0}")
if [ "${SAVE:-}" = "save" ]; then EXTRA+=(--save-field "$KO_OUT_DIR/${BOX}__${NAME}__field.npy"); fi
HR_CAT=$(ls "$DMSR_REWARD_ROOT"/halos/"${BOX}"__hr__hr/*_rockstar/halos_*.ascii 2>/dev/null | head -1 || true)
require_input "$FIELD" "a gate's splice stage"
require_input "$PEAKS" "host_phase_profile.py --peaks on the same field"
require_input "${HR_CAT:-/nonexistent}" "the frozen HR Rockstar catalog"
mkdir -p "$KO_OUT_DIR"
echo "=== task $IDX: $NAME  k_core $KCORE  target $KO_TARGET  <- $FIELD"
python -u scripts/features/knot_velocity_oracle.py --field "$FIELD" --name "$NAME" \
    --peaks "$PEAKS" --hr-catalog "$HR_CAT" --region-export "$KO_REGION_EXPORT" \
    --k-core "$KCORE" --target-coldbg "$KO_TARGET" --out-dir "$KO_OUT_DIR" "${EXTRA[@]}"
