#!/bin/bash
# Train SR2's architecture FROM SCRATCH (scripts/features/train_sr2_scratch.py).
#
# One arm per job. Configuration arrives as a POSITIONAL env file written by
# submit_sr2_scratch.sh -- never `sbatch --export`, which on this cluster sets
# SLURM_GET_USER_ENV=1 and gets the job requeued and HELD.
#
#   sbatch scripts/slurm/sr2_scratch_train_gpu.sbatch <envfile>
#
# NODE CHOICE, from `myfree` at write time (2026-09-22):
#   general  gpu24-27,30,31 a6000 48 GB  -- no free GPU, but this is a multi-day
#            gpu28 a5000 24 GB, gpu29 a6000 48 GB -- free, backfill-reserved
#   legacy   gpu20,21,23 v100 (16/32 GB) 4/4 free; gpu11-19 rtx_2080_ti 11 GB free
#   project  gpu2 h200 141 GB, 1/8 free
# The directive below is `general`, because this job runs for days and memory
# headroom matters more than start latency; the submitter overrides it with a
# plain CLI --partition (which is NOT --export and does not trip the requeue)
# for the shakeout, which wants to start now.
#
# GPU MEMORY IS NOT A DETAIL HERE. The generator is the production
# chan_base=512 ControlledG and a step holds a base minibatch (SC_BASE_BATCH
# tiles, 14^3 -> 70^3) and, on the structure arms, one host's four tiles. The
# 11 GB rtx_2080_ti nodes are excluded by default for that reason: a CUDA OOM
# on this path can kill the process with NO Python traceback, which reads as a
# mysterious silent FAILED. Re-measure with SHAKEOUT and widen if it fits.
#
# HOST RAM: the base arms hold only memmaps and a minibatch. The structure arms
# build the host pool, which `mgtrain` MEASURED at 10.1 GB peak RSS for 56 hosts
# at 2 draws -- this runs 44 hosts at 1 draw, so 32G has headroom. Do not raise
# it without a shakeout number: 96G did not fit any available node on `general`
# and stalled a whole sweep behind a figure nobody had checked.
#SBATCH --job-name=sr2scratch
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu11,gpu12,gpu13,gpu14,gpu15,gpu16,gpu17,gpu18,gpu19
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-sr2scratch-%j.out

# `set -euo pipefail` lives HERE as well as in the preamble. Without it a failed
# source leaves the job running under the system python and EXITING 0, which
# releases every afterok dependent behind a stage that did no work.
set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_scratch_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_scratch_common.sh" "$@"
set -euo pipefail

# ---- gates, as exit-0 skips -------------------------------------------------
# A non-zero exit here would strand every dependent job in
# DependencyNeverSatisfied with no explanation, so a missing input says why and
# exits 0. "Gate failed" and "upstream never ran" are distinguished by which
# message is printed.
for _b in $SC_TRAIN_BOXES $SC_HOLDOUT_BOXES; do
    for _side in lr hr; do
        _f="${DATA:-/zfsauton/scratch/yixiz/DMSR/paired_catnorm}/$_side/$_b.npy"
        if [[ ! -e "$_f" ]]; then
            echo ">>> MISSING INPUT: $_f"
            echo ">>> produced by: scripts/preprocess_paired.py"
            echo ">>> exiting 0 so dependents report the same rather than stranding."
            exit 0
        fi
    done
done
unset _b _side _f

# A structure arm needs the HR owner arrays; a base arm does not touch them.
if [[ "$SC_W_GATHER" != "0.0" && "$SC_W_GATHER" != "0" ]]; then
    echo "    structure arm: the host pool will be built (owner arrays required)."
    echo "    NOTE the recorded caveat -- the member-set REFERENCE is still"
    echo "    measured against the frozen SR2 field. It is a constant of the"
    echo "    objective, fixed before step 1 and identical to the fine-tune"
    echo "    arms', not an initialisation."
fi

mapfile -t _FLAGS < <(sc_flags)
echo
echo "=== python scripts/features/train_sr2_scratch.py \\"
printf '        %s\n' "${_FLAGS[@]}"
echo

srun --unbuffered python scripts/features/train_sr2_scratch.py "${_FLAGS[@]}"
_rc=$?

OUT_DIR="$DMSR_REWARD_ROOT/sr2_scratch/$SC_LABEL"
echo
echo "=== $(date '+%F %T')  arm $SC_LABEL finished rc=$_rc"
echo "    metrics:  $OUT_DIR/metrics.jsonl"
echo "    resume:   $OUT_DIR/latest.pt"
if (( SC_SHAKEOUT_STEPS > 0 )); then
    echo "    shakeout: $OUT_DIR/shakeout.json   <-- the s/step and peak memory"
    echo "              that size --time and --mem for the real arms"
    [[ -r "$OUT_DIR/shakeout.json" ]] && cat "$OUT_DIR/shakeout.json"
else
    echo "    summary:  $OUT_DIR/summary.json"
    echo "    snapshots: $OUT_DIR/tuned_step*.pt (gate these with"
    echo "               scripts/slurm/submit_gather_holdout_rockstar.sh)"
fi
exit "$_rc"
