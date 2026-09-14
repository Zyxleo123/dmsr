#!/bin/bash
# kNN damage reports (gate_knn_cpu.sbatch) for arms that were gated BEFORE the
# report became part of submit_gather_holdout_rockstar.sh, or on a box whose
# thresholds have not been checked. No splice, no Rockstar: it reads the spliced
# fields already on disk. Advisory -- every job exits 0.
#
#   BOX=set10 bash scripts/slurm/submit_knn_gate_backfill.sh      # out-of-sample check
#   BOX=set9 ARMS="all_blocks_abl_nod6 fine_condsmall" bash scripts/slurm/submit_knn_gate_backfill.sh
#   DRY=1 BOX=set10 bash scripts/slurm/submit_knn_gate_backfill.sh
#
# ARMS default: every mgho_<arm>_<box> field on disk except the frozen control.
# A "_stepNNNN" tag is an early gate: its run dir is <role>_<box>_stepNNNN.
# MAX_JOBS (default 12) caps the submission against the 50-job cpu limit.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
BOX="${BOX:?set BOX, e.g. BOX=set10}"
SEED="${SEED:-0}"
ROLE="${ROLE:-holdout}"
MAX_JOBS="${MAX_JOBS:-12}"
F="$REWARD_ROOT/flow_rockstar/fields"

if [ -z "${ARMS:-}" ]; then
    ARMS=""
    for p in "$F/${BOX}__mgho_"*"_${BOX}__seed${SEED}.npy"; do
        [[ -e "$p" ]] || continue
        t=$(basename "$p" .npy); t=${t#${BOX}__mgho_}; t=${t%_${BOX}__seed${SEED}}
        [[ "$t" == frozen ]] || ARMS="$ARMS $t"
    done
fi
[[ -s "$F/${BOX}__mgho_frozen_${BOX}__seed${SEED}.npy" ]] \
    || { echo "no frozen control field for $BOX -- gate one first"; exit 1; }

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
STAMP="$(date +%Y%m%d_%H%M%S)"
WATCH=(); RESULTS=(); n=0
for arm_tag in $ARMS; do
    arm="$arm_tag"; suffix=""
    if [[ "$arm_tag" =~ ^(.*)(_step[0-9]+)$ ]]; then
        arm="${BASH_REMATCH[1]}"; suffix="${BASH_REMATCH[2]}"
    fi
    rs_tag="mgho_${arm_tag}_${BOX}"
    run_dir="$REWARD_ROOT/member_gather/$arm/${ROLE}_${BOX}${suffix}"
    field="$F/${BOX}__${rs_tag}__seed${SEED}.npy"
    if [[ ! -s "$field" ]]; then echo "skip $arm_tag: no field $field"; continue; fi
    if [[ ! -d "$run_dir" ]]; then echo "skip $arm_tag: no run dir $run_dir"; continue; fi
    if [ "$n" -ge "$MAX_JOBS" ]; then
        echo "MAX_JOBS=$MAX_JOBS reached; rerun with ARMS= the remainder"; break
    fi
    env_f="$REWARD_ROOT/env/knn_backfill_${STAMP}_${arm_tag}_$$.env"
    cat > "$env_f" <<EOT
# Written by scripts/slurm/submit_knn_gate_backfill.sh at $(date '+%F %T').
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
HG_BOX=$BOX
HG_SEED=$SEED
HG_RS_TAG=$rs_tag
HG_RUN_DIR=$run_dir
EOT
    [[ -s "$env_f" ]] || { echo "FAILED to write $env_f (quota?)"; exit 1; }
    if [ "$DRY" = "1" ]; then
        echo "DRY: sbatch scripts/slurm/gate_knn_cpu.sbatch $env_f"; j=DRYID
    else
        j=$(sbatch --parsable scripts/slurm/gate_knn_cpu.sbatch "$env_f")
        echo "submitted $arm_tag -> job $j"
    fi
    WATCH+=("  $arm_tag: tail -F $REWARD_ROOT/logs/slurm-hg_knn-$j.out")
    RESULTS+=("  $arm_tag: $run_dir/knn_report.json")
    n=$((n + 1))
done
echo
echo "watch:"; printf '%s\n' "${WATCH[@]}"
echo
echo "results:"; printf '%s\n' "${RESULTS[@]}"
