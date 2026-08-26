#!/bin/bash
# Can a GENERATOR reach the member-gather objective, and does it generalise?
#
#   shakeout (GPU, pool only) --afterok--> fine-tune (GPU)
#
# docs/sr2_member_gather.md settled that the objective is satisfiable: a free
# field recovered 72 of 154 supervised subhalos as bound halos real Rockstar
# matches, against a frozen 3 and a measured ceiling of 151. Its section 7 then
# names what that leaves open, and this chain is items 3 and 4:
#
#   item 3  a free field is not a generator -- 6.3M unconstrained parameters
#           that SEE the true member sets say what the objective PERMITS, not
#           what one learned operator applied at every site can reach.
#   item 4  one host, one box, in-sample -- no generalisation claim is available
#           from any run in that document.
#
# So the training pool is many hosts across many boxes and the score that
# matters is on boxes never trained on and never used to develop this line.
#
# WHY THESE BOXES (configs/reward/sr2_direct_finetune.yaml `split`):
#   train   set3-set7  SR2's paired training was set0-2 ONLY, and these five
#                      were never used to develop the gather line.
#   holdout set9,set10 never seen by SR2, never seen by us.
#   anchor  set8       DELIBERATELY EXCLUDED from both. It is where the ceiling
#                      (151/154) and the oracle (72/154) were measured, so it is
#                      development-contaminated -- but it is also the only box
#                      with a calibrated ceiling, which makes it the right place
#                      to ask "how far short of the free field did the generator
#                      fall on the IDENTICAL problem". Run it as a reported
#                      anchor, never as evidence of generalisation.
#   sealed  set13-15   never touched.
#
# PREREQUISITE: every box needs its HR owner array -- membership IS the
# supervision. Build them first, once, with:
#     bash scripts/slurm/submit_owner_arrays.sh
# A box without one is skipped with a message rather than failing the job.
#
# This submitter ONLY calls sbatch. Configuration goes into ONE timestamped env
# file passed as a POSITIONAL argument -- never `sbatch --export`, which on this
# cluster sets SLURM_GET_USER_ENV=1 and gets the job requeued and HELD.
#
#   bash scripts/slurm/submit_member_gather_train.sh
#   MG_POOL_ONLY=1 bash scripts/slurm/submit_member_gather_train.sh   # shakeout
#   MG_STEPS=20000 MG_LABEL=_long bash scripts/slurm/submit_member_gather_train.sh
#   DRY=1 bash scripts/slurm/submit_member_gather_train.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
SHAKEOUT_ONLY="${SHAKEOUT_ONLY:-0}"
SKIP_SHAKEOUT="${SKIP_SHAKEOUT:-0}"

# --- the split ---------------------------------------------------------------
MG_TRAIN_BOXES="${MG_TRAIN_BOXES:-set3 set4 set5 set6 set7}"
MG_HOLDOUT_BOXES="${MG_HOLDOUT_BOXES:-set9 set10}"
MG_HOLDOUT_HOSTS="${MG_HOLDOUT_HOSTS:-}"
# --- the pool ----------------------------------------------------------------
# 4 tiles is the oracle's geometry and should stay there: the measured ceiling at
# 4 tiles is 151/154, so the ~5.4% of members starting outside the trained tiles
# cost THREE targets. Widening buys almost nothing and halves the pool.
MG_N_TILES="${MG_N_TILES:-4}"
MG_MAX_HOSTS_PER_BOX="${MG_MAX_HOSTS_PER_BOX:-8}"
MG_MIN_LOG_MVIR="${MG_MIN_LOG_MVIR:-13.5}"
# --- optimisation ------------------------------------------------------------
MG_RUNG="${MG_RUNG:-fine}"
MG_LR_SCALE="${MG_LR_SCALE:-10.0}"
MG_STEPS="${MG_STEPS:-8000}"
MG_HOSTS_PER_STEP="${MG_HOSTS_PER_STEP:-2}"
MG_CLIP="${MG_CLIP:-1.0}"
MG_EVAL_EVERY="${MG_EVAL_EVERY:-250}"
# --- the objective -----------------------------------------------------------
MG_W_VIRIAL="${MG_W_VIRIAL:-1.0}"
MG_W_BOUND="${MG_W_BOUND:-1.0}"
MG_W_D6="${MG_W_D6:-1.0}"
MG_W_RRMS="${MG_W_RRMS:-0.3}"
MG_W_SIGMAV="${MG_W_SIGMAV:-0.3}"
MG_W_CENTRE="${MG_W_CENTRE:-1.0}"
# The three arms of docs/sr2_member_gather_training.md section 9.4, all off by
# default -- `full` + 0 + 0 IS the objective that scored 72/154. The two shaping
# knobs were read by the batch script from 2026-08-23 but were NEVER written
# into the env file, so section 12's softened-centre recipe silently ran the
# default; they are here now because a knob the submitter drops is worse than
# one that does not exist.
# The loss BUDGET, section 10. Off by default, because every number recorded so
# far was measured with the unnormalised terms and the capped hinge.
#   MG_BOUND_PENALTY=log  charges a boundness deficit in log space instead of
#                         [1-x/ref]_+^2, which is capped at 1 by construction.
#   MG_TERM_NORM=1        divides each term by its frozen-field value so the
#                         declared weights are the actual budget (and removes
#                         d6's 88% head start, which is the same fact).
MG_BOUND_PENALTY="${MG_BOUND_PENALTY:-hinge}"
MG_TERM_NORM="${MG_TERM_NORM:-0}"
MG_CENTRE_MODE="${MG_CENTRE_MODE:-full}"
MG_CENTRE_DEAD_ZONE="${MG_CENTRE_DEAD_ZONE:-0.0}"
MG_CENTRE_HUBER="${MG_CENTRE_HUBER:-0.0}"
# --- the host preservation guard, docs/sr2_member_gather_training.md ----------
# The advisor's fix for the collateral host damage (resolved hosts 3028 base ->
# 2708 tuned, HR wants 3775; gather-holdout-rockstar-gate). The SAME member
# loss on the resolved HOST halos homing in a cluster's tiles, referenced on the
# FROZEN field -- so it is a preservation guard that starts at ~0 and fires only
# as a host comes apart, not a second objective. All OFF by default:
# MG_W_HOST_SETS=0 is byte-identical to the finished arms.
MG_W_HOST_SETS="${MG_W_HOST_SETS:-0.0}"
MG_HOST_MIN_NUM_P="${MG_HOST_MIN_NUM_P:-200}"
MG_HOST_MAX_SETS="${MG_HOST_MAX_SETS:-256}"
MG_HOST_SETS_PER_STEP="${MG_HOST_SETS_PER_STEP:-64}"
MG_MIN_NUM_P="${MG_MIN_NUM_P:-200}"
MG_MIN_PURITY="${MG_MIN_PURITY:-0.5}"
MG_MIN_LIVE_FRAC="${MG_MIN_LIVE_FRAC:-0.5}"
MG_SOFTENING="${MG_SOFTENING:-0.01}"
MG_BOUND_TAU="${MG_BOUND_TAU:-0.5}"
MG_BG_K="${MG_BG_K:-4096}"
MG_BG_RADIUS="${MG_BG_RADIUS:-4.0}"
MG_POT_CHUNK="${MG_POT_CHUNK:-2048}"
# --- the guards --------------------------------------------------------------
MG_W_LOW="${MG_W_LOW:-100.0}"
# Absent from every free-field run, which ended at 5.50x HR and still climbing.
MG_W_HIGHK="${MG_W_HIGHK:-10.0}"
MG_K_SPLIT="${MG_K_SPLIT:-4.0}"
MG_LOW_K_MAX="${MG_LOW_K_MAX:-0.02}"
MG_HIGHK_MAX="${MG_HIGHK_MAX:-1.5}"
MG_VEL_RMS_TOL="${MG_VEL_RMS_TOL:-0.10}"
# --- section 11.6: band-resolved high-k, and the velocity power guard ---------
# Measured on held-out set9 (job 36394): `self` is +3.4x HR at k=4-8 and 9x SHORT
# at k>10, which the single scalar averages to 1.59; and EVERY arm took velocity
# power above k_split from frozen's 1.02x HR down to 0.034-0.053x, with no term
# in the loss and no criterion in the verdict. All default to the old behaviour.
MG_HIGHK_BINS="${MG_HIGHK_BINS:-0}"
MG_HIGHK_K_MAX="${MG_HIGHK_K_MAX:-0.0}"
MG_HIGHK_TOL="${MG_HIGHK_TOL:-0.0}"
MG_HIGHK_TWO_SIDED="${MG_HIGHK_TWO_SIDED:-0}"
MG_HIGHK_REDUCE="${MG_HIGHK_REDUCE:-mean}"
MG_HIGHK_MAX_HOLDOUT="${MG_HIGHK_MAX_HOLDOUT:-0.0}"
MG_W_VEL_HIGHK="${MG_W_VEL_HIGHK:-0.0}"
MG_VEL_HIGHK_TOL="${MG_VEL_HIGHK_TOL:-0.25}"
MG_VEL_HIGHK_MIN="${MG_VEL_HIGHK_MIN:-0.0}"
# --- the box-wide (unsupervised-tile) high-k support, selfvel-arm-failed-the-gate
# The supervised hinge held 0.56x on the training tiles yet 3.9x held out because
# its support is the 40 hosts' 160 tiles -- 6.25% of the box. These charge the
# SAME banded hinge on random tiles drawn box-wide. All default OFF.
MG_UNSUP_TILES_PER_BOX="${MG_UNSUP_TILES_PER_BOX:-0}"
MG_UNSUP_TILES_PER_STEP="${MG_UNSUP_TILES_PER_STEP:-8}"
MG_W_HIGHK_UNSUP="${MG_W_HIGHK_UNSUP:-0.0}"
MG_UNSUP_HIGHK_MAX="${MG_UNSUP_HIGHK_MAX:-0.0}"
# --- the HR critic, docs/sr2_gather_critic.md. A REGULARISER on the gather loss,
# the general form of the two hand-crafted guards above: it sees whole HR tiles
# vs tuned tiles (6-channel high-pass, velocity included) and charges for the
# field-realism defects the gate found. All default OFF -- MG_W_ADV=0 is the
# four-arm baseline, byte-for-byte.
MG_W_ADV="${MG_W_ADV:-0.0}"
MG_ADV_WARMUP="${MG_ADV_WARMUP:-500}"
MG_ADV_RAMP="${MG_ADV_RAMP:-2000}"
MG_N_CRITIC="${MG_N_CRITIC:-1}"
MG_CRITIC_LR="${MG_CRITIC_LR:-2e-4}"
MG_CRITIC_WIDTH="${MG_CRITIC_WIDTH:-64}"
MG_CRITIC_LAYERS="${MG_CRITIC_LAYERS:-3}"
MG_CRITIC_GLOBAL_POOL="${MG_CRITIC_GLOBAL_POOL:-0}"
MG_CRITIC_R1_GAMMA="${MG_CRITIC_R1_GAMMA:-10.0}"
MG_CRITIC_R1_INTERVAL="${MG_CRITIC_R1_INTERVAL:-16}"
MG_CRITIC_NORM_FIT_TILES="${MG_CRITIC_NORM_FIT_TILES:-32}"
MG_LABEL="${MG_LABEL:-}"
MG_SEED="${MG_SEED:-0}"
# online / offline / disabled. Empty uses configs/reward/sr2_direct_finetune.yaml's
# wandb block (mode: online, project: cosmo_sr_direct). The API KEY is never put
# in the env file -- that lives on world-readable shared scratch; the job reads it
# out of $HOME instead.
MG_WANDB_MODE="${MG_WANDB_MODE:-}"

RUN_DIR="$REWARD_ROOT/member_gather/${MG_RUNG}${MG_LABEL}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"
ENVFILE="$REWARD_ROOT/env/member_gather_$(date +%Y%m%d_%H%M%S)_$$.env"
# EVERY value is quoted -- see submit_owner_arrays.sh. A sourced
# `MG_TRAIN_BOXES=set3 set4` is bash for "run the command `set4`", not an
# assignment, and it takes the job down before anything starts.
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_member_gather_train.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT="$PROJECT"
ZFS="$ZFS"
DMSR_REWARD_ROOT="$REWARD_ROOT"
MG_TRAIN_BOXES="$MG_TRAIN_BOXES"
MG_HOLDOUT_BOXES="$MG_HOLDOUT_BOXES"
MG_HOLDOUT_HOSTS="$MG_HOLDOUT_HOSTS"
MG_N_TILES="$MG_N_TILES"
MG_MAX_HOSTS_PER_BOX="$MG_MAX_HOSTS_PER_BOX"
MG_MIN_LOG_MVIR="$MG_MIN_LOG_MVIR"
MG_RUNG="$MG_RUNG"
MG_LR_SCALE="$MG_LR_SCALE"
MG_STEPS="$MG_STEPS"
MG_HOSTS_PER_STEP="$MG_HOSTS_PER_STEP"
MG_CLIP="$MG_CLIP"
MG_EVAL_EVERY="$MG_EVAL_EVERY"
MG_W_VIRIAL="$MG_W_VIRIAL"
MG_W_BOUND="$MG_W_BOUND"
MG_W_D6="$MG_W_D6"
MG_W_RRMS="$MG_W_RRMS"
MG_W_SIGMAV="$MG_W_SIGMAV"
MG_W_CENTRE="$MG_W_CENTRE"
MG_BOUND_PENALTY="$MG_BOUND_PENALTY"
MG_TERM_NORM="$MG_TERM_NORM"
MG_CENTRE_MODE="$MG_CENTRE_MODE"
MG_CENTRE_DEAD_ZONE="$MG_CENTRE_DEAD_ZONE"
MG_CENTRE_HUBER="$MG_CENTRE_HUBER"
MG_W_HOST_SETS="$MG_W_HOST_SETS"
MG_HOST_MIN_NUM_P="$MG_HOST_MIN_NUM_P"
MG_HOST_MAX_SETS="$MG_HOST_MAX_SETS"
MG_HOST_SETS_PER_STEP="$MG_HOST_SETS_PER_STEP"
MG_MIN_NUM_P="$MG_MIN_NUM_P"
MG_MIN_PURITY="$MG_MIN_PURITY"
MG_MIN_LIVE_FRAC="$MG_MIN_LIVE_FRAC"
MG_SOFTENING="$MG_SOFTENING"
MG_BOUND_TAU="$MG_BOUND_TAU"
MG_BG_K="$MG_BG_K"
MG_BG_RADIUS="$MG_BG_RADIUS"
MG_POT_CHUNK="$MG_POT_CHUNK"
MG_W_LOW="$MG_W_LOW"
MG_W_HIGHK="$MG_W_HIGHK"
MG_K_SPLIT="$MG_K_SPLIT"
MG_LOW_K_MAX="$MG_LOW_K_MAX"
MG_HIGHK_MAX="$MG_HIGHK_MAX"
MG_VEL_RMS_TOL="$MG_VEL_RMS_TOL"
MG_HIGHK_BINS="$MG_HIGHK_BINS"
MG_HIGHK_K_MAX="$MG_HIGHK_K_MAX"
MG_HIGHK_TOL="$MG_HIGHK_TOL"
MG_HIGHK_TWO_SIDED="$MG_HIGHK_TWO_SIDED"
MG_HIGHK_REDUCE="$MG_HIGHK_REDUCE"
MG_HIGHK_MAX_HOLDOUT="$MG_HIGHK_MAX_HOLDOUT"
MG_W_VEL_HIGHK="$MG_W_VEL_HIGHK"
MG_VEL_HIGHK_TOL="$MG_VEL_HIGHK_TOL"
MG_VEL_HIGHK_MIN="$MG_VEL_HIGHK_MIN"
MG_UNSUP_TILES_PER_BOX="$MG_UNSUP_TILES_PER_BOX"
MG_UNSUP_TILES_PER_STEP="$MG_UNSUP_TILES_PER_STEP"
MG_W_HIGHK_UNSUP="$MG_W_HIGHK_UNSUP"
MG_UNSUP_HIGHK_MAX="$MG_UNSUP_HIGHK_MAX"
MG_W_ADV="$MG_W_ADV"
MG_ADV_WARMUP="$MG_ADV_WARMUP"
MG_ADV_RAMP="$MG_ADV_RAMP"
MG_N_CRITIC="$MG_N_CRITIC"
MG_CRITIC_LR="$MG_CRITIC_LR"
MG_CRITIC_WIDTH="$MG_CRITIC_WIDTH"
MG_CRITIC_LAYERS="$MG_CRITIC_LAYERS"
MG_CRITIC_GLOBAL_POOL="$MG_CRITIC_GLOBAL_POOL"
MG_CRITIC_R1_GAMMA="$MG_CRITIC_R1_GAMMA"
MG_CRITIC_R1_INTERVAL="$MG_CRITIC_R1_INTERVAL"
MG_CRITIC_NORM_FIT_TILES="$MG_CRITIC_NORM_FIT_TILES"
MG_LABEL="$MG_LABEL"
MG_SEED="$MG_SEED"
MG_WANDB_MODE="$MG_WANDB_MODE"
EOT

echo "envfile: $ENVFILE"
cat "$ENVFILE"
echo

# Source it in a CLEAN shell and check the list-valued variables round-trip,
# before anything is queued. An env file that does not source is a whole-chain
# failure found one job id later; this costs milliseconds.
_check=$(env -i bash -c "set -eu; source '$ENVFILE'; printf '%s|%s' \"\$MG_TRAIN_BOXES\" \"\$MG_HOLDOUT_BOXES\"" 2>&1) || {
    echo ">>> the env file does not source cleanly:" >&2
    echo ">>> $_check" >&2
    exit 1
}
if [ "$_check" != "$MG_TRAIN_BOXES|$MG_HOLDOUT_BOXES" ]; then
    echo ">>> env file round-trip MISMATCH -- values were mangled by sourcing." >&2
    echo ">>>   wrote: $MG_TRAIN_BOXES|$MG_HOLDOUT_BOXES" >&2
    echo ">>>   read:  $_check" >&2
    exit 1
fi
echo "env file sources cleanly and round-trips."
echo
echo "run dir: $RUN_DIR"
echo

# Owner arrays, checked here so a missing prerequisite is visible before the
# queue rather than as a quiet skip inside the job.
echo "owner arrays (membership IS the supervision):"
_MISSING=0
for b in $MG_TRAIN_BOXES $MG_HOLDOUT_BOXES; do
    p="$REWARD_ROOT/halos_particles/${b}__hr__hr/${b}_hr_owner.npy"
    if [ -r "$p" ]; then printf "  %-7s ok\n" "$b"
    else printf "  %-7s MISSING -- that box will be SKIPPED\n" "$b"; _MISSING=$((_MISSING + 1)); fi
done
if [ "$_MISSING" -gt 0 ]; then
    echo
    echo ">>> $_MISSING box(es) have no owner array. Build them with:"
    echo ">>>   bash scripts/slurm/submit_owner_arrays.sh"
    echo ">>> Continuing: the run will use whatever boxes are ready."
fi
echo

# Scrub SLURM_* so a submission from inside an allocation cannot leak an
# --export-like environment onto the batch jobs.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

sub() {
    if [ "$DRY" = "1" ]; then echo "DRY: sbatch $*" >&2; echo "DRYID"
    else sbatch --parsable "$@"; fi
}

SID=""; TID=""

if [ "$SKIP_SHAKEOUT" != "1" ]; then
    # Selection and the reachable reference are everything that can be wrong
    # before a step is taken. Minutes, and it stands between a misconfigured
    # split and twelve GPU hours.
    SID=$(sub scripts/slurm/member_gather_train_gpu.sbatch "$ENVFILE" \
          MG_POOL_ONLY=1 MG_STEPS=0)
    echo "submitted shakeout (pool only) -> job $SID (GPU)"
fi

if [ "$SHAKEOUT_ONLY" = "1" ]; then
    echo
    [ -n "$SID" ] && echo "watch: tail -F $REWARD_ROOT/logs/slurm-mgtrain-$SID.out"
    echo "then read: $RUN_DIR/pool.json"
    echo
    echo "READ THE SHAKEOUT BEFORE TRAINING. Check that:"
    echo "  - every intended box appears (a missing owner array is a silent skip)"
    echo "  - 'rejected' is empty, or you understand each entry"
    echo "  - the held-out pool is not tiny; it is the only thing being claimed"
    exit 0
fi

DEP=(); [ -n "$SID" ] && DEP=(--dependency=afterok:"$SID")
TID=$(sub "${DEP[@]}" scripts/slurm/member_gather_train_gpu.sbatch "$ENVFILE")
echo "submitted fine-tune            -> job $TID (GPU${SID:+, afterok:$SID})"

echo
[ -n "$SID" ] && echo "watch shakeout:  tail -F $REWARD_ROOT/logs/slurm-mgtrain-$SID.out"
[ -n "$TID" ] && echo "watch fine-tune: tail -F $REWARD_ROOT/logs/slurm-mgtrain-$TID.out"
echo
echo "results:"
echo "  $RUN_DIR/pool.json      hosts, sets, and each host's reachable reference"
echo "  $RUN_DIR/metrics.jsonl  one row per eval -- train AND holdout, per host"
echo "  $RUN_DIR/summary.json   config, history, VERDICT"
echo "  $RUN_DIR/tuned.pt       {'model': full state dict}, loadable by the gate"
echo
echo "wandb: group 'member_gather', run 'member_gather-${MG_RUNG}${MG_LABEL}'."
echo "  Curves mirror metrics.jsonl (which stays the source of truth, so an"
echo "  outage never loses a number). The RESULT -- verdict text, its three"
echo "  booleans, frozen->final deltas for every pooled metric, and a per-host"
echo "  table -- lands in the run SUMMARY, so the runs table is readable without"
echo "  opening any chart. Running the rung ladder puts fine / middle_fine /"
echo "  all_blocks in one group for direct comparison."
echo
echo "THE VERDICT THIS CHAIN PRINTS IS NOT THE RESULT. Every number in it is a"
echo "member-set surrogate the objective computes about itself, and this line has"
echo "measured such a surrogate reaching +255 while real Rockstar showed no gain"
echo "at all (tile-overfit-proxy-exploitation). The result is the whole-box"
echo "Rockstar gate on the HELD-OUT pool -- not yet wired; see the note at the"
echo "end of scripts/slurm/submit_member_gather_train.sh."
echo
echo "# GATE, once tuned.pt exists (the pieces all exist; the chaining does not):"
echo "#   1. full-box generation from the tuned checkpoint, held-out boxes:"
echo "#        scripts/reward/evaluate_sr2_direct.py --checkpoint $RUN_DIR/tuned.pt \\"
echo "#          --boxes $(echo "$MG_HOLDOUT_BOXES" | tr ' ' ',')"
echo "#   2. Rockstar on each generated box:  scripts/slurm/flow_rockstar_catalog_cpu.sbatch"
echo "#   3. per-target recovery against the held-out hosts' member sets, the way"
echo "#      scripts/reward/compare_gather_catalog.py scores the oracle's 154."
echo "#   Read it against the oracle's calibration: ceiling 151/154, noise +-9,"
echo "#   frozen 3/154, free field 72/154."
