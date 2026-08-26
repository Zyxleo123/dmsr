#!/bin/bash
# Can a frozen-SR2 DoG density peak mark a *missing* subhalo? (unblock deployable)
#
# CPU only. Reuses the same on-disk fields/catalogs/member-ids; adds a
# difference-of-Gaussians peak finder on a 512^3 density grid (two FFT smooths +
# a maximum_filter, run once and thresholded several times) and re-scores with
# peak-only centres. No Rockstar, no GPU, no training. Two Ng^3 phase-space
# arrays plus a 512^3 float grid in RAM (~9 GB).
#
#   sbatch scripts/slurm/peak_targeting_cpu.sbatch <envfile>
#SBATCH --job-name=peak_target
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-peak_target-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8,set9}"
: "${NG_PEAK:=512}"
: "${DOG_FINE:=1.5}"
: "${DOG_COARSE:=8.0}"
: "${DOG_THRESHOLDS:=0.5,1,2,4}"
: "${SCORE_THRESHOLD:=1.0}"
: "${R_MULT:=2.0}"
: "${OUT_NAME:=peak_targeting}"

python -u scripts/reward/diagnose_peak_targeting.py \
    --boxes "$BOXES" --ng-peak "$NG_PEAK" --dog-fine "$DOG_FINE" \
    --dog-coarse "$DOG_COARSE" --dog-thresholds "$DOG_THRESHOLDS" \
    --score-threshold "$SCORE_THRESHOLD" --r-mult "$R_MULT" --out-name "$OUT_NAME"

python -u scripts/reward/plot_peak_targeting.py \
    --boxes "$BOXES" --out-name "$OUT_NAME"

echo "=== summaries + figures: $DMSR_REWARD_ROOT/audits/$OUT_NAME/<box>/"
echo "=== Part 1: proximity.frac_within_rref -- do DoG peaks land on missing subhalos?"
echo "=== Part 2: scoring.C_peak.frac_hr_gt_sr2_bind vs the 0.17 subhalo centres gave."
