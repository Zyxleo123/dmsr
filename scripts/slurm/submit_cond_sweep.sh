#!/bin/bash
# Overnight sweep: every lever worth pulling on top of the LR-host conditioning
# channel, each with its Rockstar gate chained behind it.
#
#   bash scripts/slurm/submit_cond_sweep.sh
#   SWEEP_ARMS="condsmall smallctl" bash scripts/slurm/submit_cond_sweep.sh
#   DRY=1 bash scripts/slurm/submit_cond_sweep.sh
#
# WHY A SWEEP RATHER THAN ONE ARM. Sections 14-16b closed the shell family and
# 16c added the conditioning channel; the intermediate read of `fine_cond`
# (step 1750) says the channel IS used -- cond0 off zero, bound_hard +17% train
# / +11% held out -- but the radial diagnostics did not move, because this
# objective charges NOTHING for radius. So the open question is not "does the
# channel work" but "which lever does it unlock", and that is a sweep.
#
# SIZING. 6000 steps at the measured 4.6 s/step is ~7.7 h, which fits an
# overnight window with the ~19 min gate behind it. MG_GATE_AT_STEPS=3000 writes
# an immutable mid-run snapshot at ~4 h so a SECOND, earlier gate can be
# submitted alongside -- insurance against the queue slipping. Neither arm now
# running (47468/47463, 8000 steps) set that, which is why their only gateable
# checkpoint is the live `tuned.pt`.
#
# EVERY ARM IS COMPARABLE TO EVERY OTHER: same steps, same pool settings except
# the one lever named, same seed. `smallctl` is not a spare -- it is the control
# that makes `condsmall` readable, because the 50-200p population changes the
# supervised set on its own.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

SWEEP_STEPS="${SWEEP_STEPS:-6000}"
SWEEP_GATE_AT="${SWEEP_GATE_AT:-3000}"
SWEEP_RUNG="${SWEEP_RUNG:-fine}"
COND="r_over_rvir,log_mhost,valid"
RHAT_ROOT="${RHAT_ROOT:-$REWARD_ROOT/lr_host_channel_rhat}"

# The arms. One line each: LABEL then the env that defines it.
#
#  condsmall  the MEASURED-motivated one. The channel carries a per-HOST scale;
#             on the >=200p sets a per-host rescale removes only 5.2% of the
#             radius MSE, but 86% of HR's subhalos are 50-200p and have NEVER
#             been supervised. Population that owns the profile + the feature
#             that can express host scale.
#  smallctl   condsmall with NO channel. The control that makes it readable.
#  condshell  the shell term, done correctly: self_tan, not full, or the centre
#             term competes for the same displacement and settles at exactly
#             half the correction (submit_member_gather_train.sh:137).
#  condboth   population AND shell term together.
#  condrhat   6 channels including the unit vector to the host. A radius says
#             how far to move material, never which way.
#  condbound  the abundance lever: r(bound_hard, subhalo count) = +0.95, and
#             bound currently gets 0.24% of the loss budget because [1-x/ref]_+^2
#             is capped at 1. `log` + --term-norm is the built, never-run fix.
declare -a ARMS=(
"condsmall|MG_COND_CHANNELS=$COND MG_MIN_NUM_P=50 MG_MAX_NUM_P=2000 MG_SET_SELECT=stratified MG_MEM=48G"
"smallctl|MG_COND_CHANNELS= MG_MIN_NUM_P=50 MG_MAX_NUM_P=2000 MG_SET_SELECT=stratified MG_MEM=48G"
"condshell|MG_COND_CHANNELS=$COND MG_W_RADIUS=1.0 MG_CENTRE_MODE=self_tan"
"condboth|MG_COND_CHANNELS=$COND MG_MIN_NUM_P=50 MG_MAX_NUM_P=2000 MG_SET_SELECT=stratified MG_W_RADIUS=1.0 MG_CENTRE_MODE=self_tan MG_MEM=48G"
"condrhat|MG_COND_CHANNELS=r_over_rvir,log_mhost,valid,rhat_x,rhat_y,rhat_z MG_COND_ROOT=$RHAT_ROOT"
"condbound|MG_COND_CHANNELS=$COND MG_BOUND_PENALTY=log MG_TERM_NORM=1"
)

SWEEP_ARMS="${SWEEP_ARMS:-condsmall smallctl condshell condboth condrhat condbound}"

# condrhat needs the 6-channel fields. Fail HERE, not 8 minutes into a shakeout.
if echo "$SWEEP_ARMS" | grep -qw condrhat; then
    if [ ! -f "$RHAT_ROOT/set3.npz" ]; then
        echo "ERROR: condrhat needs the 6-channel conditioning fields and" >&2
        echo "  $RHAT_ROOT/set3.npz does not exist. Build them first (~65 s):" >&2
        echo >&2
        echo "    CH_CHANNELS=$COND,rhat_x,rhat_y,rhat_z \\" >&2
        echo "      CH_OUT_DIR=$RHAT_ROOT \\" >&2
        echo "      bash scripts/slurm/submit_lr_host_channel.sh" >&2
        echo >&2
        echo "  Or drop it: SWEEP_ARMS=\"condsmall smallctl condshell condboth condbound\"" >&2
        exit 1
    fi
fi

echo "=== conditioning sweep: $SWEEP_STEPS steps, rung $SWEEP_RUNG"
echo "    arms: $SWEEP_ARMS"
echo "    early snapshot at step $SWEEP_GATE_AT, final gate chained afterok"
echo

# Submitting from inside an allocation would leak this shell's job context.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

declare -a LAUNCHED=()
for spec in "${ARMS[@]}"; do
    label="${spec%%|*}"
    envs="${spec#*|}"
    echo "$SWEEP_ARMS" | grep -qw "$label" || continue

    echo "──────── $label"
    if [ "$DRY" = "1" ]; then
        echo "DRY: env $envs MG_STEPS=$SWEEP_STEPS MG_GATE_AT_STEPS=$SWEEP_GATE_AT \\"
        echo "       MG_RUNG=$SWEEP_RUNG MG_LABEL=_$label \\"
        echo "       bash scripts/slurm/submit_member_gather_train.sh"
        continue
    fi

    out=$(env $envs MG_STEPS="$SWEEP_STEPS" MG_GATE_AT_STEPS="$SWEEP_GATE_AT" \
              MG_RUNG="$SWEEP_RUNG" MG_LABEL="_$label" \
              bash scripts/slurm/submit_member_gather_train.sh 2>&1) || {
        echo "$out" | tail -20
        echo ">>> $label FAILED TO SUBMIT; continuing with the rest" >&2
        continue
    }
    tid=$(echo "$out" | sed -n 's/.*submitted fine-tune  *-> job \([0-9]*\).*/\1/p' | head -1)
    if [ -z "$tid" ]; then
        echo "$out" | tail -20
        echo ">>> could not parse a job id for $label; gate NOT chained" >&2
        continue
    fi
    arm="${SWEEP_RUNG}_${label}"
    echo "  train job $tid  (arm $arm)"

    # THE RUN DIR MUST EXIST BEFORE THE GATE IS SUBMITTED. This is what actually
    # killed the 2026-09-10 overnight sweep: submit_gather_holdout_rockstar.sh
    # tolerates a missing CHECKPOINT when HG_AFTER_JOB is set ("job N is expected
    # to write it"), but it exits 1 on a missing RUN DIR, to catch a typo'd arm
    # name early. At submit time the trainer has not created it yet, so every
    # gate refused and six arms trained ungated. The trainer creates this same
    # directory itself (paths.subdir(..., create=True)), so making it here is
    # not a race -- it is the same mkdir, earlier.
    mkdir -p "$REWARD_ROOT/member_gather/$arm"

    # The gate the moment training ends. This is the only number that counts:
    # everything the trainer prints is a member-set surrogate scoring itself.
    # NEVER swallow this. The first run of this sweep hid the gate submitter's
    # refusal here -- at submit time no arm has a tuned.pt yet, the submitter
    # said so on stderr, and the unconditional "chained" echo below reported
    # success anyway. Result: six arms trained overnight and NOTHING was gated,
    # which is the only number that counts.
    if MG_ARMS="$arm" HG_AFTER_JOB="$tid" \
       bash scripts/slurm/submit_gather_holdout_rockstar.sh; then
        echo "  final gate chained behind $tid"
    else
        echo ">>> $label: GATE NOT SUBMITTED (see the error above). Submit it" >&2
        echo ">>>   by hand once the arm has a checkpoint:" >&2
        echo ">>>   MG_ARMS=$arm HG_AFTER_JOB=$tid \\" >&2
        echo ">>>     bash scripts/slurm/submit_gather_holdout_rockstar.sh" >&2
    fi

    LAUNCHED+=("$label:$tid:$arm")
done

echo
echo "════════ launched ${#LAUNCHED[@]} arm(s)"
for l in "${LAUNCHED[@]}"; do echo "    $l"; done
echo
echo "watch:  tail -F $REWARD_ROOT/logs/slurm-mgtrain-<jid>.out"
echo "compare across arms once rows exist:"
echo "    for d in $REWARD_ROOT/member_gather/${SWEEP_RUNG}_*; do"
echo "      echo \"\$(basename \$d) \$(tail -1 \$d/metrics.jsonl | python -c 'import json,sys; r=json.load(sys.stdin); print(r[\"step\"], round(r[\"train\"][\"bound_hard\"],4), round(r[\"holdout\"][\"bound_hard\"],4))')\""
echo "    done"
echo
echo "INSURANCE -- submit the EARLY gate now, alongside training. A 1-core CPU"
echo "job polls for the step-$SWEEP_GATE_AT snapshot (~4 h in) and gates it while"
echo "training continues, so a slipped queue still leaves a gated result:"
for l in "${LAUNCHED[@]}"; do
    arm="${l##*:}"
    echo "    MG_ARMS=$arm HG_CHECKPOINT=tuned_step${SWEEP_GATE_AT}.pt NO_CONTROL=1 \\"
    echo "      bash scripts/slurm/submit_gather_holdout_rockstar.sh"
done
echo
echo "READ THE GATE, NOT THE LOSS. tile-overfit-proxy-exploitation measured a"
echo "surrogate reaching +255 while the halo finder showed no gain at all."
echo "Calibration: ceiling 151/154, noise +-9, frozen 3/154, free field 72/154,"
echo "in the EDITED REGION -- never box-wide, where 93.75% is untouched frozen"
echo "SR2 and swamps the signal."
