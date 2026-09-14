#!/bin/bash
# Loss-ablation table (CPU, seconds of work): reads the gate JSONs, writes
# member_gather/loss_ablation_<box>.{json,md}. Safe to resubmit any time.
#
#   sbatch scripts/slurm/loss_ablation_table_cpu.sbatch <envfile>
#
# Submitted afterany on the TRAINING jobs by submit_loss_ablation.sh, so it
# first polls (ABL_TABLE_WAIT_MINUTES) for each arm's FINAL verdict, which lands
# ~20-40 min after training. Arms with no tuned.pt (training died) are not
# waited for. On timeout it tabulates whatever exists and exits 0.
#SBATCH --job-name=abl_table
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-abl_table-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${ABL_ARMS:?ABL_ARMS must list the ablation labels}"
: "${ABL_BOX:=set9}"
: "${ABL_GATE_AT:=2000}"
: "${ABL_TABLE_OUT:=$DMSR_REWARD_ROOT/member_gather/loss_ablation_$ABL_BOX}"
: "${ABL_TABLE_WAIT_MINUTES:=180}"

MG="$DMSR_REWARD_ROOT/member_gather"
deadline=$(( $(date +%s) + ABL_TABLE_WAIT_MINUTES * 60 ))
while :; do
    pending=""
    for label in $ABL_ARMS; do
        d="$MG/all_blocks_abl_$label"
        [[ -f "$d/tuned.pt" ]] || continue   # training died: nothing to wait for
        [[ -f "$d/holdout_$ABL_BOX/verdict_vs_self.json" \
           || -f "$d/holdout_$ABL_BOX/region_counts.json" ]] || pending="$pending $label"
    done
    [[ -z "$pending" ]] && break
    if (( $(date +%s) > deadline )); then
        echo "TIMED OUT waiting for final gates of:$pending -- tabulating what exists."
        break
    fi
    echo "$(date '+%T') waiting for final gates:$pending"
    sleep 120
done

python -u scripts/features/loss_ablation_table.py \
    --reward-root "$DMSR_REWARD_ROOT" --box "$ABL_BOX" \
    --arms "$ABL_ARMS" --gate-at "$ABL_GATE_AT" --out "$ABL_TABLE_OUT"
