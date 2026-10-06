#!/bin/bash
# The two questions owed BEFORE a host-profile arm trains:
#   1. SUPPLY  -- is there material at 1.0-1.5 R_vir for a contrast rule to
#                 select? If not, refusing the core STARVES the candidate
#                 instead of relocating it, which is worse than the defect.
#   2. DRIFT   -- does the solve's M step pull candidates INWARD? If it does
#                 not, the misplacement predates the loss and the background is
#                 not the lever.
# Reads cached fields and dumped membership npz files only. No model, no GPU.
#
# Sizing (MEASURED shape, not guessed): field_to_particles on a 512^3 box is
# 134M particles -- positions 3.2 GB float64, velocities 3.2 GB, and the
# per-host difference is another 3.2 GB temporary. Two fields are loaded in
# sequence and each is freed before the next. ~10 GB peak, so 32G is ~3x.
# Do not raise it without measuring: mgtrain-mem-was-9x-overasked is what an
# over-ask costs in queue time.
#
#   sbatch scripts/slurm/host_profile_probe_cpu.sbatch <envfile>
#SBATCH --job-name=hpprobe
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-hpprobe-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${HP_PROBE_BOXES:=}"     # supply stage; empty skips it
: "${HP_PROBE_DUMPS:=}"     # drift stage; empty skips it
: "${HP_MAX_HOSTS:=64}"
: "${HP_SEED:=0}"
: "${HP_SUPPLY_FLOOR:=0.8}"

# The supply stage needs the LR host table the previous stage writes. A missing
# table here means that stage was skipped or failed, and this is a GATE, not an
# error: print why and exit 0 so dependents report a gate rather than a
# DependencyNeverSatisfied with no explanation.
if [ -n "$HP_PROBE_BOXES" ]; then
  for _b in $HP_PROBE_BOXES; do
    if [ ! -r "$DMSR_REWARD_ROOT/lr_host_table/${_b}.npz" ]; then
      echo ">>> GATE: no LR host table for $_b."
      echo ">>> Run scripts/slurm/lr_host_table_cpu.sbatch first (seconds)."
      exit 0
    fi
  done
fi

python -u scripts/features/host_profile_probe.py \
    ${HP_PROBE_BOXES:+--boxes $HP_PROBE_BOXES} \
    ${HP_PROBE_DUMPS:+--dumps $HP_PROBE_DUMPS} \
    --max-hosts "$HP_MAX_HOSTS" \
    --seed "$HP_SEED" \
    --supply-floor "$HP_SUPPLY_FLOOR"
echo "=== host-profile probe -> $DMSR_REWARD_ROOT/host_profile_probe/"
echo "=== READ verdict first: supply_<box>.json .verdict"
echo "===   SUPPLY OK -> the material is there; run both arms."
echo "===   STARVED   -> the frozen field has no outskirts to select; a"
echo "===               contrast rule cannot relocate what is not there, and"
echo "===               the arms should NOT be launched on this evidence."
