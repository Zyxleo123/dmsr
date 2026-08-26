#!/bin/bash
# Does the multiscale candidate score measure formation progress? (note step 3)
#
# CPU only. Reads the frozen-SR2 and HR fields and the extracted HR member ids
# already on disk, interpolates each oracle subhalo's members from SR2 to HR, and
# scores the path -- no Rockstar, no GPU, no training. Four Ng^3 phase-space
# arrays (pos+vel for both boxes) live in RAM at once (~6.5 GB).
#
#   sbatch scripts/slurm/progress_signal_cpu.sbatch <envfile>
#SBATCH --job-name=prog_sig
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-prog_sig-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8,set9}"
: "${SCALE_MULTS:=1,2,4}"
: "${N_T:=11}"
: "${N_CONTROLS:=24}"
: "${OUT_NAME:=progress_signal}"

python -u scripts/reward/diagnose_progress_signal.py \
    --boxes "$BOXES" --scale-mults "$SCALE_MULTS" --n-t "$N_T" \
    --n-controls "$N_CONTROLS" --out-name "$OUT_NAME"

python -u scripts/reward/plot_progress_signal.py \
    --boxes "$BOXES" --out-name "$OUT_NAME"

echo "=== summaries + figures: $DMSR_REWARD_ROOT/audits/$OUT_NAME/<box>/"
echo "=== read aggregate.missing_target in summary.json:"
echo "===   frac_monotone_bind ~1, early_slope_bind >> early_slope_narrow (gradient"
echo "===   is available early, unlike a single narrow scale), frac_bound_t1 high,"
echo "===   and scram_ratio_median < 1 (hot clump discounted)."
