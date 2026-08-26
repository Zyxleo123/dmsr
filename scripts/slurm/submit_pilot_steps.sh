#!/bin/bash
# The pilot of docs/sr2_substructure_module.md section 9, steps 2-4.
#
#   step 2 + 3  is p(HR|SR2) broad at subhalo scale, and where does the residual
#               displacement actually sit in k?          CPU, cond_spread_cpu
#   step 4      capacity or incentive: overfit ONE cluster under plain MSE and
#               watch the high-k power.                  GPU, overfit_host_mse_gpu
#
# The two are SIBLINGS, not a chain: neither reads the other's output, so they
# queue concurrently and a failure in one leaves the other's result standing.
#
# This submitter ONLY calls sbatch. Configuration goes into ONE timestamped env
# file passed as a POSITIONAL argument -- never `sbatch --export`, which on this
# cluster sets SLURM_GET_USER_ENV=1 and gets the job requeued and HELD.
#
# Step 4 submits as a LADDER by default: the same experiment at three ratios of
# trainable parameters to target values. That ratio is what makes the result
# readable -- below ~1 the rung cannot memorise the region, so it has to fit a
# shared local function across every subhalo-scale patch in it, and a squared
# loss over patches whose fine realisation it cannot predict is minimised by
# blurring. A single under-parameterised run therefore cannot tell "cannot
# express substructure" from "declined to commit to a realisation". Running the
# rungs together is what separates them, and they are siblings so it costs
# wall-clock, not sequencing.
#
#   bash scripts/slurm/submit_pilot_steps.sh
#   ONLY=spread bash scripts/slurm/submit_pilot_steps.sh
#   ONLY=overfit LADDER=0 OH_RUNG=middle_fine OH_STEPS=6000 bash scripts/slurm/submit_pilot_steps.sh
#   DRY=1 bash scripts/slurm/submit_pilot_steps.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
ONLY="${ONLY:-both}"
LADDER="${LADDER:-1}"

# --- step 2 + 3: the conditional spread ------------------------------------
# One box -> fit and score on disjoint slabs of it, with a 6.25 Mpc/h buffer.
# Two boxes -> the cleaner cross-realisation held-out set, but the second box
# needs an owner array to contribute host-stratified sites, and set9's raw
# particle dumps were deleted so it currently cannot.
CS_BOXES="${CS_BOXES:-set8}"
CS_SPLIT_BUFFER="${CS_SPLIT_BUFFER:-32}"
CS_SIGMAS="${CS_SIGMAS:-0.7,1,2}"     # high-pass scales in sites -> 7.3/5.1/2.6 h/Mpc
CS_RADIUS="${CS_RADIUS:-5}"           # 11^3 neighbourhood, 2.15 Mpc/h
CS_N_SITES="${CS_N_SITES:-480000}"
CS_N_TILES="${CS_N_TILES:-512}"
CS_K_SPLIT="${CS_K_SPLIT:-8.0}"       # the design's high-pass cut, for the band split
CS_SITE_LOG_MVIR="${CS_SITE_LOG_MVIR:-13.0}"
CS_RFF_DIM="${CS_RFF_DIM:-1024}"
CS_NL_DIM="${CS_NL_DIM:-64}"
CS_SEED="${CS_SEED:-0}"

# --- step 4: capacity vs incentive -----------------------------------------
OH_BOX="${OH_BOX:-set8}"
OH_HOST_ID="${OH_HOST_ID:--1}"        # -1 = the most massive HR host in the box
OH_RUNG="${OH_RUNG:-fine}"
OH_STEPS="${OH_STEPS:-3000}"
OH_BATCH="${OH_BATCH:-2}"
OH_N_TILES="${OH_N_TILES:-4}"
OH_LR_SCALE="${OH_LR_SCALE:-10.0}"
OH_EVAL_EVERY="${OH_EVAL_EVERY:-100}"
OH_K_SPLIT="${OH_K_SPLIT:-4.0}"
OH_LABEL="${OH_LABEL:-}"
OH_SEED="${OH_SEED:-0}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"
ENVFILE="$REWARD_ROOT/env/pilot_steps_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_pilot_steps.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
CS_BOXES=$CS_BOXES
CS_SIGMAS=$CS_SIGMAS
CS_RADIUS=$CS_RADIUS
CS_N_SITES=$CS_N_SITES
CS_N_TILES=$CS_N_TILES
CS_K_SPLIT=$CS_K_SPLIT
CS_SITE_LOG_MVIR=$CS_SITE_LOG_MVIR
CS_RFF_DIM=$CS_RFF_DIM
CS_NL_DIM=$CS_NL_DIM
CS_SEED=$CS_SEED
CS_SPLIT_BUFFER=$CS_SPLIT_BUFFER
OH_BOX=$OH_BOX
OH_HOST_ID=$OH_HOST_ID
OH_RUNG=$OH_RUNG
OH_STEPS=$OH_STEPS
OH_BATCH=$OH_BATCH
OH_N_TILES=$OH_N_TILES
OH_LR_SCALE=$OH_LR_SCALE
OH_EVAL_EVERY=$OH_EVAL_EVERY
OH_K_SPLIT=$OH_K_SPLIT
OH_LABEL=$OH_LABEL
OH_SEED=$OH_SEED
EOT

echo "envfile: $ENVFILE"
cat "$ENVFILE"
echo

# Scrub SLURM_* so a submission from inside an allocation cannot leak an
# --export-like environment onto the batch job.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

# The step-4 ladder: (rung, tiles, params/target). all_blocks on one tile is the
# only rung that could memorise the region outright, so it is the one whose
# plateau -- or whose recovery of high-k power -- is a statement about capacity.
# `fine` on four tiles is the deployed regime and is here for contrast, not for
# a verdict of its own.
LADDER_ROWS=("all_blocks 1 4.43" "middle_fine 1 1.06" "fine 4 0.05")

if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch scripts/slurm/cond_spread_cpu.sbatch $ENVFILE"
    if [ "$LADDER" = "1" ]; then
        for row in "${LADDER_ROWS[@]}"; do
            set -- $row
            echo "DRY: sbatch scripts/slurm/overfit_host_mse_gpu.sbatch $ENVFILE" \
                 "OH_RUNG=$1 OH_N_TILES=$2 OH_LABEL=_pt$3"
        done
    else
        echo "DRY: sbatch scripts/slurm/overfit_host_mse_gpu.sbatch $ENVFILE"
    fi
    exit 0
fi

SID=""
OIDS=()
if [ "$ONLY" != "overfit" ]; then
    SID=$(sbatch scripts/slurm/cond_spread_cpu.sbatch "$ENVFILE" | awk '{print $NF}')
    echo "submitted step 2+3 (conditional spread, CPU) job $SID"
fi
if [ "$ONLY" != "spread" ]; then
    if [ "$LADDER" = "1" ]; then
        for row in "${LADDER_ROWS[@]}"; do
            set -- $row
            jid=$(sbatch scripts/slurm/overfit_host_mse_gpu.sbatch "$ENVFILE" \
                  "OH_RUNG=$1" "OH_N_TILES=$2" "OH_LABEL=_pt$3" | awk '{print $NF}')
            OIDS+=("$jid")
            echo "submitted step 4 rung=$1 tiles=$2 (params/target ~$3) job $jid"
        done
    else
        jid=$(sbatch scripts/slurm/overfit_host_mse_gpu.sbatch "$ENVFILE" \
              | awk '{print $NF}')
        OIDS+=("$jid")
        echo "submitted step 4 (capacity vs incentive, GPU) job $jid"
    fi
fi

echo
echo "--- watch ---"
[ -n "$SID" ] && echo "tail -F $REWARD_ROOT/logs/slurm-condspr-$SID.out"
for j in "${OIDS[@]:-}"; do
    [ -n "$j" ] && echo "tail -F $REWARD_ROOT/logs/slurm-ohmse-$j.out"
done
echo
echo "--- results ---"
[ -n "$SID" ] && echo "$REWARD_ROOT/cond_spread/cond_spread_${CS_BOXES%%,*}_${CS_BOXES##*,}.json"
[ "${#OIDS[@]}" -gt 0 ] && echo "$REWARD_ROOT/host_overfit/${OH_BOX}_h<halo_id>_<rung><label>/summary.json  (+ metrics.jsonl, tiles.npz)"
echo
echo "the verdict of each is the last line of its .out, and the 'verdict' key of its json."
echo "step 4: read the ladder together -- only the over-parameterised rung's"
echo "result is a statement about capacity. Whether a REGRESSOR could work at all"
echo "is step 2's question, not step 4's."
