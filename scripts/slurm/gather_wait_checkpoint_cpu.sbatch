#!/bin/bash
# Block until a training run writes an early-gate snapshot (CPU, 1 core).
#
#   sbatch scripts/slurm/gather_wait_checkpoint_cpu.sbatch <envfile>
#
# Why a job that mostly sleeps. The point of MG_GATE_AT_STEPS is a Rockstar
# answer at hour ~2.7 instead of hour ~10.3, with training untouched and still
# running to --steps. That needs the gate chain queued the moment training is
# submitted -- but its first stage cannot start until the snapshot exists, and
# Slurm has no "depend on a file". So: one core, no memory to speak of, polling
# a path on the shared filesystem. Against a GPU training job and a 16-core
# Rockstar it is free, and it is the only alternative to either babysitting the
# run by hand or guessing a --begin time that is wrong in one of two directions.
#
# It does NOT depend on the training job. A snapshot at step 2000 appears while
# that job is still RUNNING, so `afterok` would defeat the purpose and `after`
# only says "has started".
#
# On timeout this exits NON-ZERO, which strands the four dependent jobs on
# DependencyNeverSatisfied. That is the correct outcome -- the snapshot does not
# exist, so there is nothing to gate -- and it is not the unexplained stranding
# the project rule is about: the reason is printed here, and the submitter
# prints this log's path next to the warning before anything is queued.
#SBATCH --job-name=hg_wait
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-hg_wait-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${HG_TRAIN_DIR:?HG_TRAIN_DIR must name the member_gather run directory}"
: "${HG_CHECKPOINT:?HG_CHECKPOINT must name the snapshot file to wait for}"
# Minutes. The default covers step 2000 at the measured ~4.5 s/step (2.5 h) plus
# the ~15 min pool build and a wide margin for queueing.
: "${HG_WAIT_MINUTES:=300}"
: "${HG_WAIT_POLL_S:=60}"

TARGET="$HG_TRAIN_DIR/$HG_CHECKPOINT"
DEADLINE=$(( $(date +%s) + HG_WAIT_MINUTES * 60 ))

echo "=== wait for $TARGET"
echo "    deadline $(date -d "@$DEADLINE" '+%F %T') (${HG_WAIT_MINUTES} min), " \
     "poll every ${HG_WAIT_POLL_S}s"

# The trainer writes through a temp file and one os.replace, so the moment this
# path exists it is a COMPLETE checkpoint -- there is no partial-file window to
# guard against here, and no need to watch the size settle.
while [ ! -f "$TARGET" ]; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        echo
        echo ">>> TIMED OUT after ${HG_WAIT_MINUTES} min: $TARGET never appeared." >&2
        echo ">>> The four dependent jobs (export/splice/rockstar/compare) will" >&2
        echo ">>>   show DependencyNeverSatisfied. That is this timeout, not a" >&2
        echo ">>>   failure in any of them." >&2
        echo ">>> Check, in order:" >&2
        echo ">>>   1. was the training run launched with MG_GATE_AT_STEPS set?" >&2
        echo ">>>      Its log says 'early-gate snapshot -> ...' when it writes one." >&2
        echo ">>>   2. is the training job still queued rather than running?" >&2
        echo ">>>      squeue -u \$USER; the wait starts when THIS job starts," >&2
        echo ">>>      not when training does." >&2
        echo ">>>   3. is the step slower than the measured 4.5 s? Raise" >&2
        echo ">>>      HG_WAIT_MINUTES and resubmit the chain." >&2
        echo ">>> Nothing is lost: training is untouched and the normal" >&2
        echo ">>>   end-of-run gate still applies." >&2
        exit 1
    fi
    sleep "$HG_WAIT_POLL_S"
done

echo
echo "=== found after $(( ($(date +%s) - (DEADLINE - HG_WAIT_MINUTES * 60)) / 60 )) min"
ls -la "$TARGET"
python -u - "$TARGET" <<'PY'
import sys
import torch

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(f"    step {ck.get('step')}  rung {ck.get('rung')}  "
      f"partial={ck.get('partial')}  snapshot={ck.get('snapshot')}  "
      f"{len(ck.get('model', {}))} tensors")
if not ck.get("model"):
    raise SystemExit("checkpoint has no 'model' state dict -- refusing to gate it")
PY
echo "=== the gate chain is released; training is still running"
