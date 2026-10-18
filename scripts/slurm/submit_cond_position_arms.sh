#!/bin/bash
# Give the generator the host's SIZE and POSITION, on the two objectives that
# work, with the held-out Rockstar gate chained behind each arm.
#
#   bash scripts/slurm/submit_cond_position_arms.sh
#   POS_ARMS="radparent_pos" bash scripts/slurm/submit_cond_position_arms.sh
#   DRY=1 bash scripts/slurm/submit_cond_position_arms.sh
#
# WHY. all_blocks_radparent (2026-10-15, held-out set9) failed the gate: the
# soft solve aimed each set at HR's radius, and the generator did not build
# there on a box it never saw -- radial median 0.33 R_vir vs HR 1.13, the same
# failure shape as the shell family. The standing diagnosis is that
# G(lr, noise) is local and translation-equivariant and has NO input saying
# where the host is or how big it is. The LR-host channel supplies exactly that
# (src/cosmo_sr/features/lr_host_channel.py), from the LR box alone:
#   size      log_mhost
#   position  r_over_rvir + rhat_x/y/z (unit vector to the host centre)
#   mask      valid (a gated LR host owns this site)
#
# WHAT IS NEW HERE. The channel has run before, but every conditioned arm
# (fine_cond*, all_blocks_condsmall/condboth/condshell, fine_condrhat) ran on
# the REJECTED base config -- w_d6 1, centre full/self_tan, no host guard, no
# soft membership -- and the gated ones landed in the "damaging" family
# (docs/sr2_knn_distance.md section 7). The direction channels never ran on
# all_blocks and were never gated. None ran on the soft-membership objectives.
#
# THE ARMS. Each is an existing gated arm plus the channel and nothing else;
# cond0 is zero-initialised, so step 0 is bit-identical to the parent arm.
#   radparent_pos  all_blocks_radparent + channel. THE TEST: the one objective
#                  that asks for HR's radii, now with the input it lacked.
#                  Control: all_blocks_radparent (gated, failed).
#   soft_pos       all_blocks_sm_coarse_soft + channel. Its objective anchors
#                  centres where the frozen field put them (centre_mode self),
#                  so it asks nothing of placement; this reads whether the
#                  channel helps or harms abundance/hosts on the winning arm.
#                  Control: all_blocks_sm_coarse_soft.
# One caveat on radparent_pos vs radparent: the weighted virial is now computed
# in log space (mathematically the same ratio; fewer non-finite steps).
#
# RESOURCES. MG_EXCLUDE=gpu28 pins an A6000 on `general` (gpu28 is the only
# A5000 there): no soft arm has recorded its GPU peak, and radparent ran on an
# A6000 (3.6 h / 4000 steps). The trainer now prints the peak at the end.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

POS_ARMS="${POS_ARMS:-radparent_pos soft_pos}"
POS_GATE_AT="${POS_GATE_AT:-2000}"
POS_EXCLUDE="${POS_EXCLUDE:-gpu28}"
COND_ROOT="$REWARD_ROOT/lr_host_channel_rhat"
COND="r_over_rvir,log_mhost,valid,rhat_x,rhat_y,rhat_z"
BASELINE_ARM="all_blocks_sm_coarse_soft"

# The winning soft arm, as the submitter-default diff of its env file
# (member_gather_20261015_010308_1237941.env minus radparent's own levers).
# Every one of these differs from a submit_member_gather_train.sh default.
SOFT="MG_SOFT_MEMBERSHIP=1 MG_W_D6=0.0 MG_CENTRE_MODE=self MG_W_HOST_SETS=1.0 MG_MAX_SET_PARTICLES=768 MG_STEPS=4000"
RADPARENT="MG_SOFT_MEMBERSHIP=1 MG_W_D6=0.0 MG_CENTRE_MODE=self_tan_parent MG_W_HOST_SETS=1.0 MG_MAX_SET_PARTICLES=768 MG_STEPS=4000 MG_SOFT_HOST_PROFILE=parent MG_SOFT_HOST_WEIGHT=0.5 MG_SOFT_RADIAL_WEIGHT=1.0 MG_SOFT_RADIAL_WIDTH=0.2"
CHANNEL="MG_COND_CHANNELS=$COND MG_COND_ROOT=$COND_ROOT"

declare -a ARMS=(
"radparent_pos|$RADPARENT $CHANNEL"
"soft_pos|$SOFT $CHANNEL"
)

# The 6-channel fields. Fail HERE, not in a shakeout.
for b in set3 set4 set5 set6 set7 set9 set10; do
    if [ ! -f "$COND_ROOT/$b.npz" ]; then
        echo "ERROR: $COND_ROOT/$b.npz is missing. Build the 6-channel fields:" >&2
        echo "    CH_CHANNELS=$COND CH_OUT_DIR=$COND_ROOT \\" >&2
        echo "      bash scripts/slurm/submit_lr_host_channel.sh" >&2
        exit 1
    fi
done
if [ ! -f "$REWARD_ROOT/member_gather/$BASELINE_ARM/holdout_set9/subhalos.json" ]; then
    echo "ERROR: the baseline arm $BASELINE_ARM has no set9 gate on disk." >&2
    exit 1
fi

# Submitting from inside an allocation would leak this shell's job context.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

declare -a WATCH=()
for spec in "${ARMS[@]}"; do
    label="${spec%%|*}"
    envs="${spec#*|}"
    echo "$POS_ARMS" | grep -qw "$label" || continue
    arm="all_blocks_${label}"
    echo "──────── $arm"

    if [ "$DRY" = "1" ]; then
        echo "DRY: env $envs MG_RUNG=all_blocks MG_LABEL=_$label \\"
        echo "       MG_GATE_AT_STEPS=$POS_GATE_AT MG_EXCLUDE=$POS_EXCLUDE \\"
        echo "       bash scripts/slurm/submit_member_gather_train.sh"
        continue
    fi

    out=$(env $envs MG_RUNG=all_blocks MG_LABEL="_$label" \
              MG_GATE_AT_STEPS="$POS_GATE_AT" MG_EXCLUDE="$POS_EXCLUDE" \
              bash scripts/slurm/submit_member_gather_train.sh 2>&1) || {
        echo "$out" | tail -20
        echo ">>> $label FAILED TO SUBMIT; continuing with the rest" >&2
        continue
    }
    echo "$out" | grep -E "^envfile|submitted" || true
    tid=$(echo "$out" | sed -n 's/.*submitted fine-tune  *-> job \([0-9]*\).*/\1/p' | head -1)
    if [ -z "$tid" ]; then
        echo "$out" | tail -20
        echo ">>> could not parse a job id for $label; gates NOT chained" >&2
        continue
    fi
    WATCH+=("$arm train:  tail -F $REWARD_ROOT/logs/slurm-mgtrain-$tid.out")

    # The gate submitter refuses a missing RUN DIR; the trainer makes the same
    # directory, so making it first is not a race (see submit_cond_sweep.sh).
    mkdir -p "$REWARD_ROOT/member_gather/$arm"

    # Final gate, the moment training ends. Never swallowed.
    if MG_ARMS="$arm" HG_AFTER_JOB="$tid" NO_CONTROL=1 \
       HG_BASELINE_ARM="$BASELINE_ARM" HG_BASELINE_NAME=coarse_soft \
       bash scripts/slurm/submit_gather_holdout_rockstar.sh; then
        echo "  final gate chained behind $tid"
    else
        echo ">>> $label: FINAL GATE NOT SUBMITTED (see above)" >&2
    fi
    # Early gate on the step-$POS_GATE_AT snapshot. Works since the trainer
    # writes run_config.json at the start of the run (job 84389 died without it).
    # The waiter's clock starts NOW, before the train job has a GPU; 660 min
    # (under its 12 h limit) leaves ~9 h of queue on top of ~2 h to the snapshot.
    if MG_ARMS="$arm" HG_CHECKPOINT="tuned_step${POS_GATE_AT}.pt" NO_CONTROL=1 \
       HG_WAIT_MINUTES=660 \
       HG_BASELINE_ARM="$BASELINE_ARM" HG_BASELINE_NAME=coarse_soft \
       bash scripts/slurm/submit_gather_holdout_rockstar.sh; then
        echo "  early gate (step $POS_GATE_AT) submitted; polls for the snapshot"
    else
        echo ">>> $label: EARLY GATE NOT SUBMITTED (see above)" >&2
    fi
    WATCH+=("$arm verdict:   $REWARD_ROOT/member_gather/$arm/holdout_set9/verdict_vs_coarse_soft.json")
    WATCH+=("$arm early:     $REWARD_ROOT/member_gather/$arm/holdout_set9_step${POS_GATE_AT}/verdict_vs_coarse_soft.json")
done

echo
echo "════════ monitor"
for w in "${WATCH[@]}"; do echo "    $w"; done
echo "    gate job logs: printed per stage above (hg_wait / hg_export / rockstar / hg_verdict)"
echo
echo "READ (in the verdict json, edited region of held-out set9):"
echo "  radial_distribution.*.median_r_over_rvir  HR 1.128, frozen 0.533,"
echo "      coarse_soft 0.347, radparent 0.327 -- the number this exists to move"
echo "  radial counts at 1-1.5 R_vir               HR 6545, coarse_soft 1129, radparent 710"
echo "  mass_function.*.total (main host)           HR 374, coarse_soft 382, radparent 258"
echo "  then subhalos.json: subs >=50p (HR 12367, coarse_soft 10138) and hosts"
echo "  >=200p (HR 3775, frozen 3028, coarse_soft 3148, radparent 2759)"
echo "Compare radparent_pos against all_blocks_radparent/holdout_set9/verdict_vs_coarse_soft.json."
