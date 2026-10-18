#!/bin/bash
# Velocity-temperature oracle (CPU array, one variant per task): rescale a
# field's local velocity dispersion to HR's stacked host-centric profile
# (positions untouched), then run the frozen Rockstar config on it.
#
#   sbatch --array=0-<n-1> scripts/slurm/velocity_oracle_cpu.sbatch <envfile>
#
# VO_VARIANTS: space-separated "name:field_path:mode[:save]" (mode all|core|out;
# ":save" also writes the modified field for SUBFIND). Quote it in the env file.
#
#SBATCH --job-name=voracle
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=8:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-voracle-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 192G  partition cpu"
nvidia-smi -L 2>/dev/null || echo "(no GPU: CPU job)"

: "${BOX:=set9}"
: "${VO_VARIANTS:?VO_VARIANTS (name:field:mode[:save] ...) is required}"
: "${VO_OUT_DIR:=$DMSR_REWARD_ROOT/velocity_oracle}"
: "${VO_SPLIT:=0.5}"
: "${VO_REGION_EXPORT:=$DMSR_REWARD_ROOT/member_gather/all_blocks_sm_coarse_soft/holdout_${BOX}/export.json}"
: "${VO_HR_PROFILE:=$DMSR_REWARD_ROOT/placement_forensics/host_profiles/${BOX}__hr.json}"
: "${OVERWRITE:=0}"

read -r -a _V <<< "$VO_VARIANTS"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#_V[@]}" ]; then
    echo ">>> array index $IDX past the end of VO_VARIANTS (${#_V[@]}); nothing to do."
    exit 0
fi
IFS=: read -r NAME FIELD MODE SAVE <<< "${_V[$IDX]}"
HR_CAT=$(ls "$DMSR_REWARD_ROOT"/halos/"${BOX}"__hr__hr/*_rockstar/halos_*.ascii 2>/dev/null | head -1 || true)

require_input "$FIELD" "a gate's splice stage / paired_catnorm"
require_input "${HR_CAT:-/nonexistent}" "the frozen HR Rockstar catalog"
require_input "$VO_HR_PROFILE" "scripts/slurm/submit_placement_forensics.sh (profile array)"
require_input "$VO_REGION_EXPORT" "a held-out gate's export stage"
mkdir -p "$VO_OUT_DIR"

EXTRA=()
if [ "${SAVE:-}" = "save" ]; then EXTRA+=(--save-field "$VO_OUT_DIR/${BOX}__${NAME}__field.npy"); fi
if [ "$OVERWRITE" = "1" ]; then EXTRA+=(--overwrite); fi

echo "=== task $IDX: $NAME  mode $MODE  <- $FIELD"
python -u scripts/features/velocity_profile_oracle.py \
    --field "$FIELD" --name "$NAME" --mode "$MODE" --split "$VO_SPLIT" \
    --hr-profile "$VO_HR_PROFILE" --hr-catalog "$HR_CAT" \
    --region-export "$VO_REGION_EXPORT" --out-dir "$VO_OUT_DIR" "${EXTRA[@]}"

echo "=== catalog under $VO_OUT_DIR/$NAME/${NAME}_rockstar/"
