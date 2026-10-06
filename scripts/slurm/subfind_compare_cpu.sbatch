#!/bin/bash
# Stage 3 (CPU): SUBFIND vs Rockstar, inside the trained host and box-wide.
#
# Reads only catalogues -- two npz tables and the Rockstar ASCII files -- so it
# is cheap and rerunnable: retuning the matching tolerance costs seconds and
# never touches a field or a finder.
#
#   sbatch scripts/slurm/subfind_compare_cpu.sbatch <envfile>
#SBATCH --job-name=sf_cmp
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/subfind/logs/slurm-sf_cmp-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_subfind_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_subfind_common.sh" "$@"
set -euo pipefail

: "${SF_BOX:=set9}"
: "${SF_NAME:?SF_NAME must name the candidate run dir}"
: "${SF_TAG:?SF_TAG must be the Rockstar candidate tag}"
: "${SF_HOST_ID:=168880}"
: "${SF_MIN_P:=50}"
: "${SF_HR_NPZ:=$SUBFIND_ROOT/${SF_BOX}__hr/subfind_catalog.npz}"
: "${SF_BASE_NPZ:=$SUBFIND_ROOT/${SF_BOX}__frozen/subfind_catalog.npz}"
: "${SF_CMP_OUT:=$SUBFIND_ROOT/compare/${SF_BOX}__${SF_NAME}}"
: "${SF_MATCH_TOL_MPC:=0.05}"
: "${SF_MATCH_RVIR_MULT:=1.0}"

use_env pjm

CAND="$SUBFIND_ROOT/$SF_NAME/subfind_catalog.npz"
if [ ! -f "$CAND" ]; then
    gate_stop "no SUBFIND catalogue for $SF_NAME ($CAND); its run stage did not finish"
fi
if [ ! -f "$SF_HR_NPZ" ]; then
    echo "!!! no HR SUBFIND catalogue at $SF_HR_NPZ."
    echo "!!! The comparison still runs, but WITHOUT the control there is no"
    echo "!!! calibration for what a good confirmation rate is -- read the"
    echo "!!! numbers as descriptive only."
fi

python -u scripts/features/subfind_compare.py \
    --box "$SF_BOX" --tag "$SF_TAG" \
    --subfind "$CAND" \
    ${SF_HR_NPZ:+--hr-subfind "$SF_HR_NPZ"} \
    ${SF_BASE_NPZ:+--base-subfind "$SF_BASE_NPZ"} \
    --host-id "$SF_HOST_ID" \
    --min-p "$SF_MIN_P" \
    --match-tol-mpc "$SF_MATCH_TOL_MPC" \
    --match-rvir-mult "$SF_MATCH_RVIR_MULT" \
    --out "$SF_CMP_OUT"
