#!/bin/bash
# Launch the host-centric TEMPERATURE arms (cosmo_sr.features.host_temperature)
# on the coarse_soft recipe, each with an early (step-2000) and a final held-out
# set9 gate A/B'd against all_blocks_sm_coarse_soft, the final one with the
# SUBFIND placement stage (HG_SUBFIND=1).
#
# Why (placement forensics, 2026-10-15, docs/sr2_placement_forensics.md): the
# tuned arms' core excess is Rockstar fragmenting host cores the tune made too
# cold; imposing HR's local-temperature profile on coarse_soft's velocities took
# its SUBFIND core satellites 672 -> 292 (HR 252) with hosts unchanged. These
# arms ask whether a TRAINED generator gets there.
#
#   temp_full  w_temp on every bin, 0-3 R_vir
#   temp_core  w_temp on r < 0.5 R_vir only (the half the oracle validated)
#
# Only calls the existing submitters (which only call sbatch).
#
#   bash scripts/slurm/submit_temp_arms.sh
#   TEMP_ARMS="temp_core" TEMP_W=100 bash scripts/slurm/submit_temp_arms.sh
#   DRY=1 bash scripts/slurm/submit_temp_arms.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
TEMP_ARMS="${TEMP_ARMS:-temp_full temp_core}"
TEMP_W="${TEMP_W:-30}"
GATE_AT="${GATE_AT:-2000}"
EXCLUDE="${EXCLUDE:-gpu28}"
BASELINE_ARM="all_blocks_sm_coarse_soft"
# coarse_soft's recorded config, spelled out: the trainer's defaults are NOT the
# winning arm (mgtrain-defaults-are-not-the-winning-arm), and max_set_particles
# 768 is the one the env default would silently drop.
SOFT="MG_SOFT_MEMBERSHIP=1 MG_SOFT_PRIOR=coarse MG_SOFT_LAMBDA=1.0 MG_W_D6=0.0 MG_CENTRE_MODE=self MG_W_HOST_SETS=1.0 MG_MAX_SET_PARTICLES=768 MG_STEPS=4000"
declare -a ARMS=(
"temp_full|$SOFT MG_W_TEMP=$TEMP_W MG_TEMP_MAX_X=3.0"
"temp_core|$SOFT MG_W_TEMP=$TEMP_W MG_TEMP_MAX_X=0.5"
)

if [ ! -f "$REWARD_ROOT/member_gather/$BASELINE_ARM/holdout_set9/subhalos.json" ]; then
    echo "ERROR: the baseline arm $BASELINE_ARM has no set9 gate on disk." >&2
    exit 1
fi
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

declare -a WATCH=()
for spec in "${ARMS[@]}"; do
    label="${spec%%|*}"
    envs="${spec#*|}"
    echo "$TEMP_ARMS" | grep -qw "$label" || continue
    arm="all_blocks_${label}"
    echo "──────── $arm  (w_temp $TEMP_W)"
    if [ "$DRY" = "1" ]; then
        echo "DRY: env $envs MG_RUNG=all_blocks MG_LABEL=_$label \\"
        echo "       MG_GATE_AT_STEPS=$GATE_AT MG_EXCLUDE=$EXCLUDE \\"
        echo "       bash scripts/slurm/submit_member_gather_train.sh"
        continue
    fi
    out=$(env $envs MG_RUNG=all_blocks MG_LABEL="_$label" \
              MG_GATE_AT_STEPS="$GATE_AT" MG_EXCLUDE="$EXCLUDE" \
              bash scripts/slurm/submit_member_gather_train.sh 2>&1) || {
        echo "$out" | tail -20
        echo ">>> $label FAILED TO SUBMIT; continuing with the rest" >&2
        continue
    }
    echo "$out" | grep -E "^envfile|submitted|temp" || true
    tid=$(echo "$out" | sed -n 's/.*submitted fine-tune  *-> job \([0-9]*\).*/\1/p' | head -1)
    if [ -z "$tid" ]; then
        echo "$out" | tail -20
        echo ">>> could not parse a job id for $label; gates NOT chained" >&2
        continue
    fi
    WATCH+=("$arm train:  tail -F $REWARD_ROOT/logs/slurm-mgtrain-$tid.out")
    mkdir -p "$REWARD_ROOT/member_gather/$arm"
    if MG_ARMS="$arm" HG_AFTER_JOB="$tid" NO_CONTROL=1 HG_SUBFIND=1 \
       HG_BASELINE_ARM="$BASELINE_ARM" HG_BASELINE_NAME=coarse_soft \
       bash scripts/slurm/submit_gather_holdout_rockstar.sh; then
        echo "  final gate (+ SUBFIND) chained behind $tid"
    else
        echo ">>> $label: FINAL GATE NOT SUBMITTED (see above)" >&2
    fi
    if MG_ARMS="$arm" HG_CHECKPOINT="tuned_step${GATE_AT}.pt" NO_CONTROL=1 \
       HG_WAIT_MINUTES=660 \
       HG_BASELINE_ARM="$BASELINE_ARM" HG_BASELINE_NAME=coarse_soft \
       bash scripts/slurm/submit_gather_holdout_rockstar.sh; then
        echo "  early gate (step $GATE_AT) submitted; polls for the snapshot"
    else
        echo ">>> $label: EARLY GATE NOT SUBMITTED (see above)" >&2
    fi
    WATCH+=("$arm verdict:  $REWARD_ROOT/member_gather/$arm/holdout_set9/verdict_vs_coarse_soft.json")
    WATCH+=("$arm SUBFIND:  $REWARD_ROOT/member_gather/$arm/holdout_set9/placement_subfind.json")
    WATCH+=("$arm early:    $REWARD_ROOT/member_gather/$arm/holdout_set9_step${GATE_AT}/verdict_vs_coarse_soft.json")
done

echo
echo "════════ monitor"
for w in "${WATCH[@]}"; do echo "    $w"; done
echo
echo "READ, in this order (held-out set9 edited region; coarse_soft / HR):"
echo "  metrics.jsonl  holdout temp_core_ratio -> 1 (coarse_soft ~0.7)  temp_out_ratio"
echo "  verdict  structure_distribution.new: core structures (539 / 292),"
echo "           frag_per_structure_core (12.6 / 2.0), outskirts (3716 / 17013)"
echo "  SUBFIND  placement_subfind.json: sats < 0.25 R_vir (672 / 252), 1-1.5 R_vir (387 / 2246)"
echo "  hosts    region_counts hosts >= 200p (3148 / 3775) must not fall"
