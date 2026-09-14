#!/bin/bash
# Build the LR-host conditioning fields that a conditioned member-gather arm
# needs. One CPU job; the trainer refuses to start without these.
#
#   bash scripts/slurm/submit_lr_host_channel.sh
#   CH_BOXES="set9" bash scripts/slurm/submit_lr_host_channel.sh
#   CH_CHANNELS="r_over_rvir,log_mhost,valid,rhat_x,rhat_y,rhat_z" \
#       bash scripts/slurm/submit_lr_host_channel.sh
#   DRY=1 bash scripts/slurm/submit_lr_host_channel.sh
#
# The default box list is the member-gather trainer's own --train-boxes plus its
# --holdout-boxes, because a run needs the field for EVERY box it touches and a
# missing holdout box would only surface at the first eval.
#
# WHAT THE SETTINGS MEAN is in src/cosmo_sr/features/lr_host_channel.py. The two
# that decide whether this is worth a GPU at all:
#   CH_MIN_HR_NUM_P  20000 -- where lr_host_fidelity measured the LR catalog
#                             channel stops working. Lower it and the channel
#                             degrades off a cliff, not gracefully.
#   CH_MODE          derived -- assign every site to the host minimising
#                             r/R_vir. `membership` uses the Rockstar
#                             .particles table instead; MEASURED as covered out
#                             to 2 R_vir and partial beyond, so the two are
#                             close and this is a real A/B, not a settled call.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

CH_BOXES="${CH_BOXES:-set3 set4 set5 set6 set7 set9 set10}"
CH_CHANNELS="${CH_CHANNELS:-r_over_rvir,log_mhost,valid}"
CH_MODE="${CH_MODE:-derived}"
CH_MIN_HR_NUM_P="${CH_MIN_HR_NUM_P:-20000}"
CH_R_MAX_RVIR="${CH_R_MAX_RVIR:-3.0}"
CH_LOG_MHOST_REF="${CH_LOG_MHOST_REF:-13.0}"
CH_LOG_MHOST_SCALE="${CH_LOG_MHOST_SCALE:-1.0}"
CH_RERUN_ROCKSTAR="${CH_RERUN_ROCKSTAR:-0}"
CH_OVERWRITE="${CH_OVERWRITE:-0}"
# Where an LR catalog may ALREADY exist (build_lagrangian_host.py, and
# lr_host_fidelity's own runs). Reused rather than re-found: two Rockstar runs
# of the same box can differ, and the channel must be one definite thing.
CH_WORK_ROOT="${CH_WORK_ROOT:-$REWARD_ROOT/lagrangian_host}"
CH_OUT_DIR="${CH_OUT_DIR:-$REWARD_ROOT/lr_host_channel}"

# An unset variable in the CALLER's shell is the failure mode here: writing
# CH_OUT_DIR=$DMSR_REWARD_ROOT/lr_host_channel_rhat in a shell that never
# exported it expands to /lr_host_channel_rhat, and the mkdir dies on
# permissions at the root. Catch it with a message that says what happened.
for _d in CH_OUT_DIR CH_WORK_ROOT; do
    _v="${!_d}"
    case "$_v" in
        /*/*) : ;;   # a real nested absolute path
        *) echo "ERROR: $_d='$_v' is not a nested absolute path. This almost" >&2
           echo "  always means a variable was unset in your shell -- e.g." >&2
           echo "  CH_OUT_DIR=\$DMSR_REWARD_ROOT/... with DMSR_REWARD_ROOT unset" >&2
           echo "  expands to '/...'. Use a literal path, or export" >&2
           echo "  DMSR_REWARD_ROOT=$REWARD_ROOT first." >&2
           exit 2 ;;
    esac
done

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$CH_OUT_DIR" "$CH_WORK_ROOT"

ENVFILE="$REWARD_ROOT/env/lr_host_channel_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_lr_host_channel.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument. List values are quoted:
# an unquoted "set3 set4" here would RUN set4 as a command.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
CH_BOXES="$CH_BOXES"
CH_CHANNELS="$CH_CHANNELS"
CH_MODE=$CH_MODE
CH_MIN_HR_NUM_P=$CH_MIN_HR_NUM_P
CH_R_MAX_RVIR=$CH_R_MAX_RVIR
CH_LOG_MHOST_REF=$CH_LOG_MHOST_REF
CH_LOG_MHOST_SCALE=$CH_LOG_MHOST_SCALE
CH_RERUN_ROCKSTAR=$CH_RERUN_ROCKSTAR
CH_OVERWRITE=$CH_OVERWRITE
CH_WORK_ROOT=$CH_WORK_ROOT
CH_OUT_DIR=$CH_OUT_DIR
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

# Submitting from inside an allocation would otherwise leak this shell's job
# context into the child job.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch scripts/slurm/lr_host_channel_cpu.sbatch $ENVFILE"
    exit 0
fi

JID=$(sbatch --parsable scripts/slurm/lr_host_channel_cpu.sbatch "$ENVFILE")
echo "submitted lr_host_channel -> job $JID (CPU)"
echo
echo "watch:   tail -F $REWARD_ROOT/logs/slurm-lr_host_chan-$JID.out"
echo "results: $CH_OUT_DIR/summary.json"
echo "         $CH_OUT_DIR/<box>.npz   (the (C, 64, 64, 64) stack + provenance)"
echo
echo "read first: the per-box 'valid' fraction. It is the fraction of Lagrangian"
echo "sites within $CH_R_MAX_RVIR R_vir of a gated LR host, i.e. the fraction of the box the"
echo "channel says anything about. Near 0 means the gate left almost no hosts and"
echo "the arm would train on a near-constant; near 1 means the r_max is so wide"
echo "that 'belongs to this host' has stopped meaning anything."
echo
echo "then launch the conditioned arm:"
echo "    MG_COND_CHANNELS=$CH_CHANNELS bash scripts/slurm/submit_member_gather_train.sh"
