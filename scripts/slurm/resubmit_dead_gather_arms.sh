#!/usr/bin/env bash
# Resubmit the three member-gather arms that hit the walltime/OOM at step
# 5750-6750 of 8000 with no checkpoint on disk (unsup, selfbound, self_critic).
#
# Each arm is reproduced BIT-FAITHFULLY from the exact env file its original
# submission wrote -- every MG_* it lacks defaults OFF in the batch script, so
# sourcing the saved file gives back the same config. The trainer now writes a
# partial tuned.pt/critic.pt every eval (finetune_member_gather.py), so another
# kill near the end keeps a gate-loadable model instead of throwing the run away.
#
# The three died on a6000 nodes gpu26 (unsup+self_critic, shared) and gpu24
# (selfbound) at step 5750-6750 of 8000 -- no traceback, 7.5-8.5h of a 12h limit,
# and gpu24 is now IDLE+MAINTENANCE+RESERVED. That is node maintenance/preemption,
# not OOM or walltime, so the fix is a clean re-run + the new periodic checkpoint,
# not a bigger box. selfhostguard COMPLETED on gpu28 (a5000), so a5000 is fine and
# is NOT excluded. Resource override vs the header: keep general, bump host RAM
# 96G -> 120G as harmless headroom, and let SLURM take any free general GPU.
#
# This script ONLY calls sbatch. Run it from the repo root on the login node.
set -euo pipefail

REPO="/zfsauton2/home/yixiz/DMSR/cosmo_sr_project"
ENVDIR="/zfsauton/scratch/yixiz/DMSR/dmsr_reward/env"
BATCH="scripts/slurm/member_gather_train_gpu.sbatch"

# arm label -> the saved env file that produced it
declare -a ARMS=(
  "unsup:$ENVDIR/member_gather_20260825_154613_2956710.env"
  "selfbound:$ENVDIR/member_gather_20260825_141928_2938941.env"
  "self_critic:$ENVDIR/member_gather_20260825_154953_2957959.env"
)

RES=(--partition=general --gres=gpu:1 --mem=120G)

cd "$REPO"
declare -a IDS=()
for entry in "${ARMS[@]}"; do
    name="${entry%%:*}"; env="${entry#*:}"
    if [ ! -f "$env" ]; then echo "MISSING env for $name: $env" >&2; exit 1; fi
    id=$(sbatch --parsable "${RES[@]}" "$BATCH" "$env")
    echo "submitted $name -> job $id  (env $(basename "$env"))"
    IDS+=("$id")
done

echo
echo "monitor:"
for i in "${!ARMS[@]}"; do
    name="${ARMS[$i]%%:*}"; id="${IDS[$i]}"
    echo "  # $name"
    echo "  tail -F /zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-mgtrain-${id}.out"
done
echo
echo "results land in (tuned.pt now refreshed every eval, not only at the end):"
for entry in "${ARMS[@]}"; do
    name="${entry%%:*}"
    echo "  /zfsauton/scratch/yixiz/DMSR/dmsr_reward/member_gather/all_blocks_${name}/{tuned.pt,summary.json,metrics.jsonl}"
done
echo
echo "when a tuned.pt exists, gate it with:  scripts/slurm/submit_gather_holdout_rockstar.sh"
