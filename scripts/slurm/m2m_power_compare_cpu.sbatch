#!/bin/bash
# Overlay several arms' P(k) ratio curves on one axis (m2m_power_compare.py).
#
#   sbatch scripts/slurm/m2m_power_compare_cpu.sbatch <envfile>
#
# CPU, because the curves are already on disk: m2m_power_curves.py wrote the
# npz precisely so a figure never costs a 512^3 regeneration or a GPU. Asking
# for a GPU to draw a line plot only queues behind real work.
#
# The arms to overlay come from M2M_COMPARE_ARMS, one "LABEL=PATH" per line, so
# a list with spaces in the labels survives the env file (a quoted list value --
# an unquoted one would run its tail as a command).
#SBATCH --job-name=m2mcmp
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-m2mcmp-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_m2m_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_m2m_common.sh" "$@"
set -euo pipefail

: "${M2M_COMPARE_OUT:=$DMSR_REWARD_ROOT/m2m/power_compare.png}"
: "${M2M_COMPARE_FIELDS:=displacement,density}"
: "${M2M_COMPARE_TITLE:=}"

if [[ -z "${M2M_COMPARE_ARMS:-}" ]]; then
    echo ">>> M2M_COMPARE_ARMS is empty -- nothing to overlay."
    echo ">>> set it to newline-separated LABEL=PATH entries."
    exit 0
fi

# Gate: a missing curve file is an upstream job that has not run. Skip with
# exit 0 rather than stranding anything chained behind this.
args=()
missing=0
while IFS= read -r line; do
    [[ -z "${line// }" ]] && continue
    p="${line#*=}"
    if [[ -r "$p" ]]; then args+=(--npz "$line"); else
        echo ">>> missing curve: $p"; missing=$((missing + 1)); fi
done <<< "$M2M_COMPARE_ARMS"

if [[ ${#args[@]} -eq 0 ]]; then
    echo ">>> no curve files exist yet ($missing missing)."
    echo ">>> produced by: sbatch scripts/slurm/m2m_power_curves_gpu.sbatch"
    exit 0
fi

python scripts/features/m2m_power_compare.py "${args[@]}" \
    --fields "$M2M_COMPARE_FIELDS" \
    --out "$M2M_COMPARE_OUT" \
    ${M2M_COMPARE_TITLE:+--title "$M2M_COMPARE_TITLE"}

echo
echo "=== $(date '+%F %T')  done -> $M2M_COMPARE_OUT"
echo "    Redraw from the npz alone; no GPU and no checkpoint needed."
