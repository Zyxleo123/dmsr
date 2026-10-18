#!/bin/bash
# Stacked host-centric phase-space profiles (CPU array, one field per task):
# density, infall, velocity dispersion, k=32 local temperature, and the
# finder-free clumped / cold-clumped mass fraction around every massive HR host
# in the edited region. Answers "is the core cold, dense, or clumpy?" without
# a halo finder.
#
#   sbatch --array=0-<n-1> scripts/slurm/host_phase_profile_cpu.sbatch <envfile>
#
# HP_FIELDS is a space-separated list of name=path or name=path@box (the box
# whose HR catalog defines the hosts; default $BOX). Quote it in the env file.
# HP_REGION_EXPORT=" " (a space) selects hosts from the whole box.
#
#SBATCH --job-name=hostprof
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-hostprof-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 64G  partition cpu"
nvidia-smi -L 2>/dev/null || echo "(no GPU: CPU job)"

: "${BOX:=set9}"
: "${HP_FIELDS:?HP_FIELDS (name=path ...) is required}"
: "${HP_REGION_EXPORT:=$DMSR_REWARD_ROOT/member_gather/all_blocks_sm_coarse_soft/holdout_${BOX}/export.json}"
: "${HP_MIN_HOST_NUM_P:=10000}"
: "${HP_MAX_HOSTS:=60}"
: "${HP_PEAKS:=0}"
: "${HP_OUT_DIR:=$DMSR_REWARD_ROOT/placement_forensics/host_profiles}"

read -r -a _F <<< "$HP_FIELDS"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#_F[@]}" ]; then
    echo ">>> array index $IDX past the end of HP_FIELDS (${#_F[@]}); nothing to do."
    exit 0
fi
NAME="${_F[$IDX]%%=*}"
FIELD="${_F[$IDX]#*=}"
TBOX="$BOX"
if [[ "$FIELD" == *@* ]]; then TBOX="${FIELD##*@}"; FIELD="${FIELD%@*}"; fi
HR_CAT=$(ls "$DMSR_REWARD_ROOT"/halos/"${TBOX}"__hr__hr/*_rockstar/halos_*.ascii 2>/dev/null | head -1 || true)

require_input "$FIELD" "a gate's splice stage / the m2m box dump / paired_catnorm"
require_input "${HR_CAT:-/nonexistent}" "the frozen HR Rockstar catalog"
if [[ -n "${HP_REGION_EXPORT// /}" ]]; then
    require_input "$HP_REGION_EXPORT" "a held-out gate's export stage"
fi
mkdir -p "$HP_OUT_DIR"
PEAKS=()
if [ "$HP_PEAKS" = "1" ]; then PEAKS=(--peaks); fi
if [ "${HP_BIND:-0}" = "1" ]; then PEAKS+=(--bind); fi
if [ -n "${HP_PEAK_MIN_RATIO:-}" ]; then PEAKS+=(--peak-min-ratio "$HP_PEAK_MIN_RATIO"); fi

echo "=== task $IDX: $NAME <- $FIELD  (hosts from $TBOX HR)"
python -u scripts/features/host_phase_profile.py \
    --field "$FIELD" --name "$NAME" --hr-catalog "$HR_CAT" \
    --region-export "$HP_REGION_EXPORT" \
    --min-host-num-p "$HP_MIN_HOST_NUM_P" --max-hosts "$HP_MAX_HOSTS" "${PEAKS[@]}" \
    --out "$HP_OUT_DIR/${TBOX}__${NAME}"

echo "=== wrote $HP_OUT_DIR/${TBOX}__${NAME}.{npz,json}"
