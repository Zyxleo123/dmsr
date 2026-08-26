#!/bin/bash
# Final gate stage (CPU, ~30 s): A/B the new arm against a BASELINE arm on the
# real Rockstar catalogs, in the EDITED REGION.
#
#   sbatch scripts/slurm/gate_verdict_cpu.sbatch <envfile>
#
# 32G and two minutes: this reads Rockstar ASCII catalogs and counts, nothing
# more. The 3.2 GB spliced fields are NOT loaded -- that is the figures job.
#SBATCH --job-name=hg_verdict
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-hg_verdict-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${HG_BOX:=set9}"
: "${HG_RUN_DIR:?set HG_RUN_DIR to the holdout export dir of the new arm}"
: "${HG_RS_TAG:?set HG_RS_TAG to the Rockstar tag of the new arm}"
# Empty is legal: with no gated baseline for this box (a TRAIN box, or the
# first arm gated on a new one) the edited-region counts are still what
# is being asked for, and the python side degrades to a REGION ONLY report.
: "${HG_BASELINE_TAG:=}"
: "${HG_BASELINE_NAME:=self}"
: "${HG_HOST_ID:=168880}"
if [[ -z "$HG_BASELINE_TAG" ]]; then
    : "${HG_VERDICT_OUT:=$HG_RUN_DIR/region_counts}"
else
    : "${HG_VERDICT_OUT:=$HG_RUN_DIR/verdict_vs_${HG_BASELINE_NAME}}"
fi

# A gate whose inputs are missing must say so and exit 0. A non-zero exit here
# strands nothing today, but this stage is the tail of an afterok chain and the
# convention is what keeps it safe to extend.
if [[ ! -f "$HG_RUN_DIR/export.json" ]]; then
    echo "GATE: no export.json under $HG_RUN_DIR -- the export stage did not run."
    exit 0
fi

python -u scripts/features/gate_verdict_vs_baseline.py \
    --reward-root "$DMSR_REWARD_ROOT" \
    --box "$HG_BOX" \
    --run-dir "$HG_RUN_DIR" \
    --arm-tag "$HG_RS_TAG" \
    --baseline-tag "$HG_BASELINE_TAG" \
    --baseline-name "$HG_BASELINE_NAME" \
    --host-id "$HG_HOST_ID" \
    --out "$HG_VERDICT_OUT"

echo "=== result: ${HG_VERDICT_OUT}.json"
