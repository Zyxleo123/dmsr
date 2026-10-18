#!/bin/bash
# Binding oracle (CPU, variants run IN SEQUENCE in one job -- the cpu partition
# caps queued jobs at 50): clip each outskirt bump's members to f x their own
# escape speed (positions untouched), then run the frozen Rockstar config.
#
#   sbatch scripts/slurm/binding_oracle_cpu.sbatch <envfile>
#
# BO_VARIANTS: space-separated "name:field.npy:peaks.npz:own_catalog:rate:mode[:select]"
# (quote it). rate = all | hr; mode = stats (criterion only, no finder) | run |
# save (run + write the field to BO_OUT_DIR for SUBFIND); select (with rate hr)
# = cold (default) | bound | easy | random.
#
#SBATCH --job-name=bindorc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=6:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-bindorc-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 192G  partition cpu"
nvidia-smi -L 2>/dev/null || echo "(no GPU: CPU job)"
: "${BOX:=set9}"
: "${BO_VARIANTS:?BO_VARIANTS (name:field:peaks:catalog:rate:mode ...) is required}"
: "${BO_OUT_DIR:=$DMSR_REWARD_ROOT/binding_oracle}"
: "${BO_F_ESC:=0.9}"
: "${BO_AP_CONTRAST:=3}"
: "${BO_X_MIN:=0.5}"
: "${BO_X_MAX:=3.0}"
: "${BO_HR_PEAKS:?BO_HR_PEAKS (HR census peaks, for --rate hr) is required}"
: "${BO_REGION_EXPORT:=$DMSR_REWARD_ROOT/member_gather/all_blocks_sm_coarse_soft/holdout_${BOX}/export.json}"
HR_CAT=$(ls "$DMSR_REWARD_ROOT"/halos/"${BOX}"__hr__hr/*_rockstar/halos_*.ascii 2>/dev/null | head -1 || true)
require_input "${HR_CAT:-/nonexistent}" "the frozen HR Rockstar catalog"
require_input "$BO_HR_PEAKS" "host_phase_profile.py --peaks on HR"
mkdir -p "$BO_OUT_DIR"
read -r -a _V <<< "$BO_VARIANTS"
for v in "${_V[@]}"; do
    IFS=: read -r NAME FIELD PEAKS CAT RATE MODE SEL <<< "$v"
    require_input "$FIELD" "a gate's splice stage"
    require_input "$PEAKS" "host_phase_profile.py --peaks on the same field"
    require_input "$CAT" "this field's own Rockstar catalog"
    EXTRA=(--select "${SEL:-cold}")
    case "$MODE" in
        stats) EXTRA+=(--stats-only) ;;
        save)  EXTRA+=(--save-field "$BO_OUT_DIR/${BOX}__${NAME}__field.npy") ;;
        run)   ;;
        *) echo "unknown mode '$MODE' for $NAME"; exit 2 ;;
    esac
    if [ -f "$BO_OUT_DIR/$NAME/${NAME}_rockstar/halos_0.0.ascii" ] && [ "$MODE" != "stats" ]; then
        echo ">>> $NAME already has a catalog; skipping (delete it to redo)"; continue
    fi
    echo "=== $(date '+%F %T')  $NAME  rate $RATE  select ${SEL:-cold}  mode $MODE  f_esc $BO_F_ESC  <- $FIELD"
    python -u scripts/features/binding_oracle.py --field "$FIELD" --name "$NAME" \
        --peaks "$PEAKS" --catalog "$CAT" --hr-catalog "$HR_CAT" --hr-peaks "$BO_HR_PEAKS" \
        --region-export "$BO_REGION_EXPORT" --rate "$RATE" --f-esc "$BO_F_ESC" \
        --ap-contrast "$BO_AP_CONTRAST" --x-min "$BO_X_MIN" --x-max "$BO_X_MAX" \
        --out-dir "$BO_OUT_DIR" "${EXTRA[@]}"
done
echo "=== done $(date '+%F %T')"
