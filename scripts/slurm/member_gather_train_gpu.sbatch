#!/bin/bash
# Fine-tune SR2 on the id-gathered member-set objective over many hosts and boxes
# (scripts/features/finetune_member_gather.py).
#
# This is the run docs/sr2_member_gather.md section 7 items 3 and 4 ask for. The
# free field established that the objective is satisfiable and Rockstar-verifiable
# (72/154 against a ceiling of 151); it said nothing about whether one learned
# operator can reach it, because a free field sees the answer and generalises to
# nothing. Here the variable is the generator's weights and the score that matters
# is on hosts and boxes never trained on.
#
# Node choice, checked with `myfree` at write time: `general` has gpu28 (a5000,
# 24 GB, 3/4 free, 468 GB host free), gpu24 and gpu29 (a6000, 48 GB) AVAILABLE.
# The job is not pinned to a GPU class because the footprint does not need it:
# gradient accumulation runs ONE host at a time (four 64^3 tiles, batch 4) and
# frees between, so peak GPU memory is a single host's forward/backward plus the
# chunked N x N potential -- the free-field run measured everything fitting in
# 11 GB. If `general` is full, `legacy` rtx_2080_ti (11 GB) is marginal but
# workable at --pot-chunk 1024; switch the partition AND the --qos together,
# since rhea enforces one QOS per partition.
#
# HOST RAM is the real ask and it is why this is 96G. build_pool loads one box's
# 537 MB owner array at a time and inverts it into a CSR index (2-3 GB peak),
# and the pool itself holds ~60 MB of tile data per host on the CPU -- the GPU
# never holds more than one host, deliberately, so the pool size is bounded by
# host RAM rather than by GPU memory.
#
#   sbatch scripts/slurm/member_gather_train_gpu.sbatch <envfile>
#SBATCH --job-name=mgtrain
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-mgtrain-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${DATA:=/zfsauton/scratch/yixiz/DMSR/paired_catnorm}"
: "${MG_TRAIN_BOXES:=set3 set4 set5 set6 set7}"
: "${MG_HOLDOUT_BOXES:=set9 set10}"
: "${MG_HOLDOUT_HOSTS:=}"
: "${MG_N_TILES:=4}"
: "${MG_MAX_HOSTS_PER_BOX:=8}"
: "${MG_MIN_LOG_MVIR:=13.5}"
: "${MG_POOL_ONLY:=0}"
: "${MG_RUNG:=fine}"
: "${MG_LR_SCALE:=10.0}"
: "${MG_STEPS:=8000}"
: "${MG_HOSTS_PER_STEP:=2}"
: "${MG_SETS_PER_STEP:=0}"      # 0 = all of a host's sets, the full batch
: "${MG_CLIP:=1.0}"
: "${MG_EVAL_EVERY:=250}"
# --- the objective, identical to the free field's defaults -------------------
: "${MG_W_VIRIAL:=1.0}"
: "${MG_W_BOUND:=1.0}"
: "${MG_W_D6:=1.0}"
: "${MG_W_RRMS:=0.3}"
: "${MG_W_SIGMAV:=0.3}"
: "${MG_W_CENTRE:=1.0}"
: "${MG_W_HOST_SETS:=0.0}"           # host preservation guard; 0 = off (baseline)
: "${MG_HOST_MIN_NUM_P:=200}"
: "${MG_HOST_MAX_SETS:=256}"
: "${MG_HOST_SETS_PER_STEP:=64}"
: "${MG_MIN_NUM_P:=200}"
: "${MG_MIN_PURITY:=0.5}"
: "${MG_MIN_LIVE_FRAC:=0.5}"
: "${MG_SOFTENING:=0.01}"
: "${MG_BOUND_TAU:=0.5}"
: "${MG_BG_K:=4096}"
: "${MG_BG_RADIUS:=4.0}"
: "${MG_POT_CHUNK:=2048}"
# Rows per pair block are min(MG_POT_CHUNK, MG_POT_MAX_ELEMS / N). The chunk
# ALONE is not a memory bound and that is what killed jobs 35746-35748: a
# 118,000-particle satellite at chunk 2048 is a 2.7 GiB allocation, and the
# autograd tape used to keep one per chunk. Both are fixed; this is the belt.
: "${MG_POT_MAX_ELEMS:=16777216}"
: "${MG_MAX_SET_PARTICLES:=0}"  # 0 = the exact estimator; a TIME knob
: "${MG_BOUND_PENALTY:=hinge}"  # hinge = the form every run so far used
: "${MG_TERM_NORM:=0}"         # 0 = the incidental budget of every run so far
: "${MG_CENTRE_MODE:=full}"    # full = the term that scored 72/154
: "${MG_CENTRE_DEAD_ZONE:=0.0}" # 0 = the term that scored 72/154
: "${MG_CENTRE_HUBER:=0.0}"
# --- the guards --------------------------------------------------------------
: "${MG_W_LOW:=100.0}"
: "${MG_W_HIGHK:=10.0}"
: "${MG_K_SPLIT:=4.0}"
: "${MG_LOW_K_MAX:=0.02}"
: "${MG_HIGHK_MAX:=1.5}"
: "${MG_VEL_RMS_TOL:=0.10}"
# --- band-resolved high-k + the velocity power guard, sr2_member_gather_training
# section 11.6. ALL default to the behaviour the four finished arms had, so an
# unset knob reproduces them rather than quietly redefining the objective.
: "${MG_HIGHK_BINS:=0}"        # 0 = the single mode-mean the four arms used
: "${MG_HIGHK_K_MAX:=0.0}"     # 0 = to the cube corner, as the scalar's mask
: "${MG_HIGHK_TOL:=0.0}"       # 0 = no dead zone, as the four arms had
: "${MG_HIGHK_TWO_SIDED:=0}"   # displacement lower arm; see 11.6 for why NOT
: "${MG_HIGHK_REDUCE:=mean}"
: "${MG_HIGHK_MAX_HOLDOUT:=0.0}"  # 0 = the historical train-only verdict
: "${MG_W_VEL_HIGHK:=0.0}"     # 0 = measure velocity power, charge nothing
: "${MG_VEL_HIGHK_TOL:=0.25}"
: "${MG_VEL_HIGHK_MIN:=0.0}"   # 0 = no velocity-power criterion, as before
# --- the box-wide (unsupervised-tile) high-k support -------------------------
# selfvel-arm-failed-the-gate: the supervised hinge held 0.56x on train tiles yet
# 3.9x held out because its support is 6.25% of the box. These charge the SAME
# banded hinge on random tiles drawn box-wide. ALL default to OFF, so an unset
# knob reproduces the supervised-only guard.
: "${MG_UNSUP_TILES_PER_BOX:=0}"   # 0 = off; >0 builds the pool (~15 MB/tile)
: "${MG_UNSUP_TILES_PER_STEP:=8}"  # tiles carrying the hinge each step
: "${MG_W_HIGHK_UNSUP:=0.0}"       # 0 = only measure the pool; >0 adds the loss
: "${MG_UNSUP_HIGHK_MAX:=0.0}"     # 0 = no box-wide verdict gate
# --- the HR critic, docs/sr2_gather_critic.md. Default OFF: MG_W_ADV=0 makes the
# run byte-identical to the four finished arms. The hand-crafted high-k/velocity
# hinges above each charge for ONE named statistic; the critic is their general
# form -- it sees whole HR tiles vs tuned tiles (6-channel high-pass, velocity
# included) and learns the field-realism defects the gate found (section 11.6).
: "${MG_W_ADV:=0.0}"           # 0 = critic off entirely, the four-arm baseline
: "${MG_ADV_WARMUP:=500}"      # critic trains here; generator adv held at 0
: "${MG_ADV_RAMP:=2000}"       # linear ramp of w_adv after warmup
: "${MG_N_CRITIC:=1}"
: "${MG_CRITIC_LR:=2e-4}"
: "${MG_CRITIC_WIDTH:=64}"
: "${MG_CRITIC_LAYERS:=3}"
: "${MG_CRITIC_GLOBAL_POOL:=0}" # 1 = single-score D (tile-scale stats); 0 = PatchGAN
: "${MG_CRITIC_R1_GAMMA:=10.0}"
: "${MG_CRITIC_R1_INTERVAL:=16}"
: "${MG_CRITIC_NORM_FIT_TILES:=32}"
: "${MG_LABEL:=}"
: "${MG_SEED:=0}"
: "${MG_WANDB_MODE:=}"          # online / offline / disabled; empty = config

# The preamble builds a MINIMAL environment on purpose and never sources
# ~/.bashrc, so WANDB_API_KEY -- which lives there -- would be absent and every
# run would silently downgrade to offline. Read just that one variable out of
# $HOME rather than sourcing the whole file (which would fight the preamble's
# conda setup) and rather than putting it in the env file (which lives on
# world-readable shared scratch; a secret does not belong there).
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$HOME/.bashrc" ]; then
    # \042 = double quote, \047 = single quote: octal keeps this line free of
    # nested-quote gymnastics, which is how it got mangled the first time.
    _k=$(grep -m1 '^[[:space:]]*export[[:space:]]\+WANDB_API_KEY=' "$HOME/.bashrc" \
         | cut -d= -f2- | tr -d '\042\047 ')
    [ -n "$_k" ] && export WANDB_API_KEY="$_k"
    unset _k
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "=== WANDB_API_KEY present (online logging available)"
else
    echo "=== no WANDB_API_KEY: wandb falls back to OFFLINE (sync later with"
    echo "    'wandb sync \$DMSR_REWARD_ROOT/member_gather/<run>/wandb/offline-*')"
fi

require_input "$PROJECT/external/SRS-map2map/SRmodel/G_z0.pt" \
    "the frozen SR2 checkpoint (configs/sr2_baseline/freeze.yaml)"

# Membership IS the supervision: a box with no owner array contributes nothing
# and is skipped inside the trainer with a message. Gate only on there being at
# least one, so a partially-built pool still runs rather than stranding the job.
_HAVE=0
for b in $MG_TRAIN_BOXES; do
    [ -r "$DMSR_REWARD_ROOT/halos_particles/${b}__hr__hr/${b}_hr_owner.npy" ] \
        && _HAVE=$((_HAVE + 1))
done
if [ "$_HAVE" = "0" ]; then
    echo ">>> none of the training boxes ($MG_TRAIN_BOXES) has an HR owner array."
    echo ">>> produced by: bash scripts/slurm/submit_owner_arrays.sh"
    echo ">>> exiting 0 so dependent jobs report the same rather than sitting in"
    echo ">>> DependencyNeverSatisfied with no explanation."
    exit 0
fi
echo "=== $_HAVE training boxes have owner arrays"

EXTRA=()
[[ "$MG_POOL_ONLY" == "1" ]] && EXTRA+=(--pool-only)
[[ -n "$MG_WANDB_MODE" ]] && EXTRA+=(--wandb-mode "$MG_WANDB_MODE")

echo "=== member-gather fine-tune"
echo "    train boxes:   $MG_TRAIN_BOXES"
echo "    holdout boxes: $MG_HOLDOUT_BOXES"
echo "    holdout hosts: ${MG_HOLDOUT_HOSTS:-<none>}"
echo "    rung $MG_RUNG, lr_scale $MG_LR_SCALE, $MG_STEPS steps,"
echo "    $MG_HOSTS_PER_STEP hosts/step, sets/step ${MG_SETS_PER_STEP} (0=all),"
echo "    <= $MG_MAX_HOSTS_PER_BOX hosts/box"
echo "    pair sums: chunk $MG_POT_CHUNK capped at $MG_POT_MAX_ELEMS elements,"
echo "    per-set particle cap ${MG_MAX_SET_PARTICLES} (0 = exact estimator)"
if awk "BEGIN{exit !($MG_W_HOST_SETS > 0)}"; then
  echo "    HOST PRESERVATION GUARD ON: w $MG_W_HOST_SETS, >= $MG_HOST_MIN_NUM_P p,"
  echo "      <= $MG_HOST_MAX_SETS host sets/cluster, ${MG_HOST_SETS_PER_STEP} host sets/step"
else
  echo "    host preservation guard OFF (--w-host-sets 0)"
fi
# store_true: the flag is PRESENT or ABSENT. Deliberately unquoted at the call
# site so an empty value disappears under word splitting rather than becoming an
# empty argument python would reject.
MG_TERM_NORM_FLAG=""
[ "$MG_TERM_NORM" = "1" ] && MG_TERM_NORM_FLAG="--term-norm"
MG_HIGHK_TWO_SIDED_FLAG=""
[ "$MG_HIGHK_TWO_SIDED" = "1" ] && MG_HIGHK_TWO_SIDED_FLAG="--highk-two-sided"
MG_CRITIC_GLOBAL_POOL_FLAG=""
[ "$MG_CRITIC_GLOBAL_POOL" = "1" ] && MG_CRITIC_GLOBAL_POOL_FLAG="--critic-global-pool"

echo "    budget: bound penalty ${MG_BOUND_PENALTY}, term-norm ${MG_TERM_NORM}"
if [ "$MG_BOUND_PENALTY" = "hinge" ] && [ "$MG_TERM_NORM" != "1" ]; then
echo "            (both defaults -- MEASURED, that budget is d6 88%, bound"
echo "             0.24%, and bound is the gate's own criterion. Section 10.)"
fi
echo "    centre: mode ${MG_CENTRE_MODE}, dead zone ${MG_CENTRE_DEAD_ZONE}r,"
echo "            Huber knee ${MG_CENTRE_HUBER}r"
if [ "$MG_CENTRE_MODE" != "full" ]; then
echo "    NOTE: section 9.4. This arm charges for LESS than the address, which"
echo "          is the point -- the address is only 13.4% predictable out of"
echo "          sample. Read centre_penalised_radii against centre_offset_radii"
echo "          and score the HELD-OUT hosts, because in-sample this arm can"
echo "          only look worse than the full term."
fi
echo "            (0/0 is the pure quadratic that scored 72/154 -- the control)"
echo "    guards: low_k w=$MG_W_LOW, HIGH-K HINGE w=$MG_W_HIGHK above k=$MG_K_SPLIT"
if [ "$MG_HIGHK_BINS" -gt 0 ] 2>/dev/null; then
    echo "            high-k BANDED: $MG_HIGHK_BINS log bins, k_max ${MG_HIGHK_K_MAX}" \
         "(0=cube corner), tol $MG_HIGHK_TOL, two-sided $MG_HIGHK_TWO_SIDED," \
         "reduce $MG_HIGHK_REDUCE"
    echo "            section 11.6: the excess is at k=4-8, not Nyquist -- the"
    echo "            scalar averages self's +3.4x there with a 9x deficit above"
    echo "            k=10 and calls it 1.59."
fi
echo "            VELOCITY high-k w=$MG_W_VEL_HIGHK (0 = measured, not charged)," \
     "tol $MG_VEL_HIGHK_TOL, verdict floor $MG_VEL_HIGHK_MIN"
if [ "${MG_W_VEL_HIGHK%%.*}" != "0" ] 2>/dev/null; then
    echo "    NOTE: the velocity term is ON. Measured on held-out set9, FROZEN SR2"
    echo "          carries 1.02x HR of velocity power above k_split -- it has this"
    echo "          right -- and all four finished arms ended at 0.034-0.053x, a"
    echo "          19-30x collapse with no term in the loss. vel_rms_ratio is one"
    echo "          global std and read 0.71 throughout, so it could not see it."
fi
echo "    verdict: high-k max $MG_HIGHK_MAX train / $MG_HIGHK_MAX_HOLDOUT held out" \
     "(0 = train only, as all four finished arms were judged)"
if [ "$MG_UNSUP_TILES_PER_BOX" -gt 0 ] 2>/dev/null; then
    echo "    BOX-WIDE high-k support: $MG_UNSUP_TILES_PER_BOX unsup tiles/box," \
         "$MG_UNSUP_TILES_PER_STEP/step, w=$MG_W_HIGHK_UNSUP, verdict max" \
         "$MG_UNSUP_HIGHK_MAX held out"
    echo "            selfvel-arm-failed-the-gate: the supervised hinge held 0.56x"
    echo "            on train yet 3.9x held out -- its support is 6.25% of the box."
    echo "            This charges the banded hinge on random box-wide tiles."
fi
if awk "BEGIN{exit !($MG_W_ADV > 0)}"; then
    echo "    HR CRITIC ON: w_adv=$MG_W_ADV (warmup $MG_ADV_WARMUP, ramp" \
         "$MG_ADV_RAMP), n_critic $MG_N_CRITIC, R1 gamma $MG_CRITIC_R1_GAMMA," \
         "global_pool $MG_CRITIC_GLOBAL_POOL"
    echo "            A REGULARISER on the gather loss, not a replacement. It sees"
    echo "            the 6-channel HIGH-PASS (disp AND vel) of cand vs HR, so it"
    echo "            charges for the field-realism defects the gate found (11.6):"
    echo "            velocity power collapsed 19-30x and small-scale placement went"
    echo "            to the wrong spots -- both invisible to every moment term."
fi
echo "    NOTE: the high-k hinge is ON here and was ABSENT from the free field,"
echo "          which ran to 5.50x HR. A generator rewrites every site of every"
echo "          box, so that is corruption rather than a caveat."
echo "    NOTE: the verdict this job prints is FEASIBILITY ONLY -- every number"
echo "          in it is a member-set surrogate this module computes about"
echo "          itself. The gate is whole-box Rockstar on the HELD-OUT pool."

python -u scripts/features/finetune_member_gather.py \
    --config "$DIRECT_CFG" \
    --train-boxes "$MG_TRAIN_BOXES" \
    --holdout-boxes "$MG_HOLDOUT_BOXES" \
    --holdout-hosts "$MG_HOLDOUT_HOSTS" \
    --n-tiles "$MG_N_TILES" \
    --max-hosts-per-box "$MG_MAX_HOSTS_PER_BOX" \
    --min-log-mvir "$MG_MIN_LOG_MVIR" \
    --rung "$MG_RUNG" \
    --lr-scale "$MG_LR_SCALE" \
    --steps "$MG_STEPS" \
    --hosts-per-step "$MG_HOSTS_PER_STEP" \
    --sets-per-step "$MG_SETS_PER_STEP" \
    --clip "$MG_CLIP" \
    --eval-every "$MG_EVAL_EVERY" \
    --w-virial "$MG_W_VIRIAL" \
    --w-bound "$MG_W_BOUND" \
    --w-d6 "$MG_W_D6" \
    --w-rrms "$MG_W_RRMS" \
    --w-sigmav "$MG_W_SIGMAV" \
    --w-centre "$MG_W_CENTRE" \
    --w-host-sets "$MG_W_HOST_SETS" \
    --host-min-num-p "$MG_HOST_MIN_NUM_P" \
    --host-max-sets "$MG_HOST_MAX_SETS" \
    --host-sets-per-step "$MG_HOST_SETS_PER_STEP" \
    --min-num-p "$MG_MIN_NUM_P" \
    --min-purity "$MG_MIN_PURITY" \
    --min-live-frac "$MG_MIN_LIVE_FRAC" \
    --softening "$MG_SOFTENING" \
    --bound-tau "$MG_BOUND_TAU" \
    --bg-k "$MG_BG_K" \
    --bg-radius "$MG_BG_RADIUS" \
    --pot-chunk "$MG_POT_CHUNK" \
    --pot-max-elems "$MG_POT_MAX_ELEMS" \
    --max-set-particles "$MG_MAX_SET_PARTICLES" \
    --bound-penalty "$MG_BOUND_PENALTY" \
    $MG_TERM_NORM_FLAG \
    --centre-mode "$MG_CENTRE_MODE" \
    --centre-dead-zone "$MG_CENTRE_DEAD_ZONE" \
    --centre-huber-radii "$MG_CENTRE_HUBER" \
    --w-low "$MG_W_LOW" \
    --w-highk "$MG_W_HIGHK" \
    --k-split "$MG_K_SPLIT" \
    --low-k-max "$MG_LOW_K_MAX" \
    --highk-max "$MG_HIGHK_MAX" \
    --vel-rms-tol "$MG_VEL_RMS_TOL" \
    --highk-bins "$MG_HIGHK_BINS" \
    --highk-k-max "$MG_HIGHK_K_MAX" \
    --highk-tol "$MG_HIGHK_TOL" \
    --highk-reduce "$MG_HIGHK_REDUCE" \
    $MG_HIGHK_TWO_SIDED_FLAG \
    --highk-max-holdout "$MG_HIGHK_MAX_HOLDOUT" \
    --w-vel-highk "$MG_W_VEL_HIGHK" \
    --vel-highk-tol "$MG_VEL_HIGHK_TOL" \
    --vel-highk-min "$MG_VEL_HIGHK_MIN" \
    --unsup-tiles-per-box "$MG_UNSUP_TILES_PER_BOX" \
    --unsup-tiles-per-step "$MG_UNSUP_TILES_PER_STEP" \
    --w-highk-unsup "$MG_W_HIGHK_UNSUP" \
    --unsup-highk-max "$MG_UNSUP_HIGHK_MAX" \
    --w-adv "$MG_W_ADV" \
    --adv-warmup-steps "$MG_ADV_WARMUP" \
    --adv-ramp-steps "$MG_ADV_RAMP" \
    --n-critic "$MG_N_CRITIC" \
    --critic-lr "$MG_CRITIC_LR" \
    --critic-width "$MG_CRITIC_WIDTH" \
    --critic-layers "$MG_CRITIC_LAYERS" \
    $MG_CRITIC_GLOBAL_POOL_FLAG \
    --critic-r1-gamma "$MG_CRITIC_R1_GAMMA" \
    --critic-r1-interval "$MG_CRITIC_R1_INTERVAL" \
    --critic-norm-fit-tiles "$MG_CRITIC_NORM_FIT_TILES" \
    --label "$MG_LABEL" \
    --seed "$MG_SEED" \
    "${EXTRA[@]}" \
    ${DEVICE:+--device "$DEVICE"}

echo "=== result: $DMSR_REWARD_ROOT/member_gather/${MG_RUNG}${MG_LABEL}/"
echo "    pool.json      the shakeout: hosts, sets, and the reachable reference"
echo "    metrics.jsonl  one row per eval, train AND holdout, per host"
echo "    summary.json   config, history, VERDICT (feasibility only)"
echo "    tuned.pt       {'model': full state dict} -- loadable by the gate"
echo "    wandb: group 'member_gather', run 'member_gather-${MG_RUNG}${MG_LABEL}'."
echo "           metrics.jsonl stays the source of truth and is mirrored, so a"
echo "           wandb outage never loses a number. The VERDICT and the final"
echo "           per-host table land in the run SUMMARY, not just the charts."
