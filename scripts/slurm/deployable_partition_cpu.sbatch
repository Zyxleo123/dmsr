#!/bin/bash
# Does the multiscale score survive removing the oracle partition? (note step 3->4)
#
# CPU only. Reuses the same on-disk fields/catalogs/member-ids as the progress
# signal test; adds frozen-SR2 candidate discovery (subhalos + density peaks +
# NMS) and radius-collected bags, and scores three conditions (oracle / contam
# bag / deployable) along the SR2->HR interpolation. No Rockstar, no GPU, no
# training. Four Ng^3 phase-space arrays in RAM (~6.5 GB).
#
#   sbatch scripts/slurm/deployable_partition_cpu.sbatch <envfile>
#SBATCH --job-name=deploy_part
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-deploy_part-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8,set9}"
: "${SCALE_MULTS:=1,2,4}"
: "${N_T:=11}"
: "${R_MULT:=2.0}"
: "${N_CONTROLS:=24}"
: "${OUT_NAME:=deployable_partition}"

python -u scripts/reward/diagnose_deployable_partition.py \
    --boxes "$BOXES" --scale-mults "$SCALE_MULTS" --n-t "$N_T" \
    --r-mult "$R_MULT" --n-controls "$N_CONTROLS" --out-name "$OUT_NAME"

python -u scripts/reward/plot_deployable_partition.py \
    --boxes "$BOXES" --out-name "$OUT_NAME"

echo "=== summaries + figures: $DMSR_REWARD_ROOT/audits/$OUT_NAME/<box>/"
echo "=== compare conditions A->B->C in summary.json aggregate:"
echo "===   A_oracle is the validated baseline; C_deployable is what training sees."
echo "===   Read frac_hr_gt_sr2_bind, frac_bound_t1, scram_ratio_median down the"
echo "===   three conditions -- how much does contamination + centre offset cost?"
