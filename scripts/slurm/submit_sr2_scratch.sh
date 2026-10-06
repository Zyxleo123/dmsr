#!/bin/bash
# Launch the FROM-SCRATCH SR2 arms. This script only calls sbatch.
#
#   bash scripts/slurm/submit_sr2_scratch.sh            # stage 1: the shakeout
#   SC_GO=1 bash scripts/slurm/submit_sr2_scratch.sh    # stage 2: every arm
#
# WHY A SHAKEOUT FIRST. Nothing has ever trained this architecture from a random
# initialisation in this repository, so the step time, the peak GPU memory and
# the peak host RSS are all unmeasured. The shakeout runs SC_SHAKEOUT_STEPS
# steps of the `base` arm on one GPU and writes shakeout.json with exactly those
# three numbers. Read it, put them in the sbatch, then SC_GO=1.
#
# THE ARMS
#   base         map2map's own recipe from noise: MSE + WGAN-GP on SR2Critic
#                (6 upsampled LR + 6 field + 8 CIC DENSITY channels). The
#                CONTROL, and the one that answers the actual question: does a
#                from-scratch SR2 reproduce the inverted radial profile at all?
#                If it does, no loss term will fix it.
#   base_noms    --w-mse 0. Pointwise MSE to one realisation is this repo's
#                recorded cause of mean-collapse, and mean-collapse is what a
#                centrally concentrated halo looks like. One flag separates
#                "MSE did this" from "the architecture did this".
#   knn          base + the 6-D class-conditioned kNN-CDF term at the settings
#                round 5b MEASURED: phase lambda 4000, ks 8-64, classes
#                sub_50_200. HOSTS STAY OUT: the pooled term ranks the working
#                arms at rho +0.878 (p 4e-07), the wrong sign.
#   soft_radial  base + member-gather under SOFT membership (coarse prior,
#                lambda 1 -- the arm that beat the hard mask on the held-out
#                Rockstar gate) + the host-centric RADIUS term. That term is
#                measured dead as a fine-tune lever; retrying it from a random
#                init is the direct test of the hypothesis this whole set exists
#                for.
#
# CONFIGURATION TRAVELS AS A POSITIONAL ENV FILE, never `sbatch --export`:
# any explicit export list sets SLURM_GET_USER_ENV=1, slurmd fails to rebuild
# the login environment, and the job is requeued and HELD. One timestamped file
# per arm per submission, so a resubmission never races a running job's config.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/../.." && pwd)"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
LOGS="$ROOT/logs"
ENVD="$ROOT/sr2_scratch/envs"
mkdir -p "$LOGS" "$ENVD"

STAMP="$(date '+%Y%m%d-%H%M%S')"
JOB="$HERE/sr2_scratch_train_gpu.sbatch"

# Submitting from inside an allocation inherits SLURM_* and confuses sbatch, and
# `sbatch` itself is denied from a compute node ("Access/permission denied").
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: you are inside allocation $SLURM_JOB_ID. sbatch only works" >&2
    echo "       from the login node. Exit the allocation and rerun." >&2
    exit 1
fi

# --- knobs --------------------------------------------------------------------
SC_GO="${SC_GO:-0}"
# base_noms (--w-mse 0) is DROPPED from the default set: the bracket measured
# w_mse=0 as exactly the setting that diverges (probes C/D), so the slot would
# buy a known failure. `SC_ARMS='repro base_noms knn soft_radial'` still runs it.
SC_ARMS="${SC_ARMS:-baseline knn soft_radial}"
SC_SHAKEOUT_STEPS="${SC_SHAKEOUT_STEPS:-300}"
SC_STEPS="${SC_STEPS:-100000}"
SC_SEED="${SC_SEED:-0}"
# Gate snapshots. These cost the run nothing -- training continues to SC_STEPS --
# and they are what lets a Rockstar gate read the arm at hour ~3 instead of at
# the end of a multi-day job.
# Strictly BELOW SC_STEPS: gate_steps() refuses a snapshot at or past the last
# step, because the final checkpoint is already gated by the normal chain and a
# gate queued against a snapshot that will never be written strands on Dependency.
SC_GATE_AT_STEPS="${SC_GATE_AT_STEPS:-20000,50000,80000}"
# Stage 1 wants a GPU now; stage 2 wants headroom for days. `myfree` at write
# time: legacy v100 gpu20/21/23 were 4/4 free, general a6000 were not.
SHAKE_PARTITION="${SHAKE_PARTITION:-legacy}"
ARM_PARTITION="${ARM_PARTITION:-general}"
SHAKE_TIME="${SHAKE_TIME:-01:00:00}"
ARM_TIME="${ARM_TIME:-12:00:00}"
ARM_MEM="${ARM_MEM:-32G}"
# The 11 GB rtx_2080_ti nodes. --exclude, not --nodelist: a node LIST forces the
# node COUNT, and a 3-node list on a 1-GPU job pends forever.
EXCLUDE_2080="${EXCLUDE_2080:-gpu11,gpu12,gpu13,gpu14,gpu15,gpu16,gpu17,gpu18,gpu19}"

# write_env <arm> -> prints the path of the env file it wrote.
# Reads ARM_SHAKEOUT from the caller: 0 for a real arm, >0 for the shakeout.
# SC_RUN_TAG appends a suffix to the OUTPUT DIRECTORY (not to the arm's
# settings), so a re-run of the same arm under a changed objective writes
# somewhere fresh instead of resuming the previous run's latest.pt. The resume
# guard cannot catch that case -- it compares the critic-input contract, which
# is unchanged when only the optimiser, learning rate or adversarial form
# differ -- and relying on remembering to `mv` the old directory has already
# gone wrong twice.
write_env () {
    local arm="${1:?write_env needs an arm name}"
    local label="${arm}${SC_RUN_TAG:-}"
    local f="$ENVD/${label}-${STAMP}.env"
    {
        echo "# $arm, written $(date '+%F %T') by submit_sr2_scratch.sh"
        echo "# Every list value is QUOTED: an unquoted 'SC_TRAIN_BOXES=set3 set4'"
        echo "# runs set4 as a command and kills the job. A later line WINS:"
        echo "# the per-arm block below deliberately overrides the common ones."
        echo "SC_LABEL='$label'"
        echo "SC_SEED='$SC_SEED'"
        echo "SC_STEPS='$SC_STEPS'"
        echo "SC_GATE_AT_STEPS='$SC_GATE_AT_STEPS'"
        echo "SC_SHAKEOUT_STEPS='${ARM_SHAKEOUT:-0}'"
        # Overridable for every arm. SC_GP_EVERY is the one to reach for if the
        # critic runs away: MEASURED (jobs 62067/62068 at step 30k) d_real-d_fake
        # reached +2548 with the gradient penalty at 1414, i.e. an interpolate
        # gradient norm of ~4 against the Lipschitz target of 1. The lazy
        # schedule penalises 1 step in 16 while the critic steps every step;
        # SC_GP_EVERY=1 charges it every step at lambda=10.
        echo "SC_GP_EVERY='${SC_GP_EVERY:-16}'"
        echo "SC_W_SR2_ADV='${SC_W_SR2_ADV:-1.0}'"
        echo "SC_DENSITY_TRANSFORM='${SC_DENSITY_TRANSFORM:-asinh}'"
        echo "SC_CRITIC_NORM_TILES='${SC_CRITIC_NORM_TILES:-32}'"
        # The objective's shape. Defaults are map2map's own; see the preamble.
        echo "SC_ADV_LOSS='${SC_ADV_LOSS:-wgan_gp}'"
        # `sum` = probe B, the bracket's only survivor. NOT map2map's form;
        # see the preamble for the measurement that forced the deviation.
        echo "SC_MSE_MODE='${SC_MSE_MODE:-upstream}'"
        echo "SC_W_BASE_LOW='${SC_W_BASE_LOW:-0.0}'"
        echo "SC_LR='${SC_LR:-1e-5}'"
        echo "SC_CRITIC_LR='${SC_CRITIC_LR:-2e-5}'"
        echo "SC_ADV_START_EPOCHS='${SC_ADV_START_EPOCHS:-1}'"
        echo "SC_BETAS='${SC_BETAS:-0 0.99}'"
        echo "SC_R1_INTERVAL='${SC_R1_INTERVAL:-16}'"
        # 1 is right for a job that runs in 12-hour chunks over days. Set 0 for
        # the FIRST chunk of a run whose objective changed -- otherwise the job
        # resumes a checkpoint trained under the old objective. The trainer now
        # refuses a critic-contract mismatch outright, but an arm with no critic
        # has no contract to check, so this stays a knob.
        echo "SC_RESUME='${SC_RESUME:-1}'"
        case "$arm" in
        base)
            : ;;                              # every default is this arm
        shakeout_base)
            # The base arm, short, with everything that only matters over a long
            # run turned down: the point is s/step and peak memory, not a model.
            echo "SC_EVAL_EVERY='50'"
            echo "SC_SAVE_EVERY='100'"
            echo "SC_ADV_WARMUP='50'"
            echo "SC_GATE_AT_STEPS=' '"       # a SPACE, not "": "" re-fires the default
            ;;
        baseline|repro)
            # PROBE B's configuration, exactly as measured: MSE kept alongside a
            # BCE + lazy-R1 adversary, no LR anchor, Adam 1e-4 at default betas.
            # Everything here is a preamble default, so this arm is the bracket's
            # survivor and nothing else. No spectral norm and no instance noise:
            # probe B passed WITHOUT them, and adding untested pieces to a
            # measured configuration is how the last three runs went wrong.
            : ;;
        base_noms)
            echo "SC_W_MSE='0.0'"
            ;;
        knn)
            # Round 5b's measured settings. --knn-classes is what keeps HOSTS
            # out of the term, which is the single fact that decides its sign;
            # those three are NOT overridable, because changing them changes
            # which experiment this is.
            echo "SC_W_KNN='1.0'"
            echo "SC_KNN_PHASE_LAMBDA='4000'"
            echo "SC_KNN_KS='8 16 32 64'"
            echo "SC_KNN_CLASSES='sub_50_200'"
            echo "SC_KNN_MEASURE='1'"
            # These two are the COST knobs and they ARE overridable from the
            # caller's environment. MEASURED (job 62069): 8 tiles/step at 1024
            # queries costs >13.6 s/step against 0.30 s/step with the term off
            # -- 15 days for 100k steps. Both factors are linear in the per-step
            # bill, so `SC_UNSUP_TILES_PER_STEP=1 SC_KNN_QUERIES=256` is ~32x
            # cheaper and still a minibatch of the same estimator.
            echo "SC_UNSUP_TILES_PER_BOX='${SC_UNSUP_TILES_PER_BOX:-8}'"
            echo "SC_UNSUP_TILES_PER_STEP='${SC_UNSUP_TILES_PER_STEP:-8}'"
            echo "SC_KNN_QUERIES='${SC_KNN_QUERIES:-1024}'"
            ;;
        soft_radial)
            echo "SC_W_GATHER='1.0'"
            echo "SC_SOFT_MEMBERSHIP='1'"
            echo "SC_SOFT_PRIOR='coarse'"
            echo "SC_SOFT_LAMBDA='1.0'"
            echo "SC_SOFT_POOL_RATIO='2.0'"
            echo "SC_MAX_SET_PARTICLES='768'"
            echo "SC_W_RADIUS='${SC_W_RADIUS:-1.0}'"
            echo "SC_RADIUS_DEAD_ZONE='${SC_RADIUS_DEAD_ZONE:-1.0}'"
            # Not the CLI defaults -- see mgtrain-defaults-are-not-the-winning-arm
            echo "SC_W_D6='1.0'"
            echo "SC_CENTRE_MODE='full'"
            echo "SC_HOSTS_PER_STEP='1'"
            echo "SC_MAX_HOSTS_PER_BOX='4'"
            ;;
        *)
            echo "unknown arm: $arm" >&2; return 1 ;;
        esac
    } > "$f"
    echo "$f"
}

declare -a SUBMITTED=() NAMES=()

# --------------------------------------------------------------------------- #
# Stage 1: the shakeout
# --------------------------------------------------------------------------- #
if [[ "$SC_GO" != "1" ]]; then
    echo "=== stage 1: shakeout ($SC_SHAKEOUT_STEPS steps of the base arm)"
    echo "    nothing else is submitted. Read shakeout.json, then SC_GO=1."
    # Plain assignments, not a `VAR=x write_env ...` prefix: a prefix on a
    # FUNCTION call is the one form where bash's argument handling bit here.
    ARM_SHAKEOUT="$SC_SHAKEOUT_STEPS"
    SC_STEPS="$SC_SHAKEOUT_STEPS"
    ENVF="$(write_env shakeout_base)"
    ID=$(sbatch --parsable \
                --job-name=sr2scr_shake \
                --partition="$SHAKE_PARTITION" \
                --time="$SHAKE_TIME" \
                --mem="$ARM_MEM" \
                --exclude="$EXCLUDE_2080" \
                "$JOB" "$ENVF")
    SUBMITTED+=("$ID"); NAMES+=("shakeout_base")
    echo "    submitted $ID  (env $ENVF)"
else
    # ----------------------------------------------------------------------- #
    # Stage 2: the arms, concurrently (siblings: no afterok between them)
    # ----------------------------------------------------------------------- #
    echo "=== stage 2: $SC_ARMS"
    for arm in $SC_ARMS; do
        ARM_SHAKEOUT=0
        ENVF="$(write_env "$arm")"
        armlabel="${arm}${SC_RUN_TAG:-}"
        ID=$(sbatch --parsable \
                    --job-name="sr2scr_${armlabel}" \
                    --partition="$ARM_PARTITION" \
                    --time="$ARM_TIME" \
                    --mem="$ARM_MEM" \
                    --exclude="$EXCLUDE_2080" \
                    "$JOB" "$ENVF")
        SUBMITTED+=("$ID"); NAMES+=("$armlabel")
        echo "    $armlabel -> $ID  (env $ENVF)"
    done
fi

# --------------------------------------------------------------------------- #
# How to watch it, and where the answers land
# --------------------------------------------------------------------------- #
cat <<EOF

=== watch
EOF
for i in "${!SUBMITTED[@]}"; do
    echo "  ${NAMES[$i]}:"
    echo "    tail -F $LOGS/slurm-sr2scratch-${SUBMITTED[$i]}.out"
done
cat <<EOF

=== results
EOF
for n in "${NAMES[@]}"; do
    echo "  $n:  $ROOT/sr2_scratch/$n/"
    echo "        metrics.jsonl   per-eval rows (base_mse, base_low_k, ramp,"
    echo "                        noise_std_mean, adv.*, holdout_base.*)"
    echo "        summary.json    s/step, peak GPU GB, peak host GB"
    echo "        tuned_step*.pt  gate snapshots, readable while training runs"
done
cat <<'EOF'

=== the three numbers to read FIRST, in order
  1. holdout_base.mse and holdout_base.low_k -- did it learn the
     super-resolution map at all, on a box it never saw? If these do not fall,
     nothing downstream means anything and the arm should be stopped.
  2. noise_std_mean -- AddNoise.std starts at ZERO and SR2's fine-scale
     structure is written entirely by those six tensors. A run that leaves them
     at ~0 has trained a DETERMINISTIC upsampler, whatever its spectrum says,
     and it cannot produce substructure. If it stays flat, rerun with
     SC_INIT_NOISE_STD=1e-2.
  3. adv.d_real - adv.d_fake -- a WGAN critic whose scores have separated
     without bound has stopped constraining the generator.

  If the shakeout dies with NO Python traceback, that is a CUDA OOM, not a
  crash: it landed on a 16 GB v100 and the production chan_base=512 generator
  at SC_BASE_BATCH=4 did not fit. Rerun it with either
     SC_BASE_BATCH=2 bash scripts/slurm/submit_sr2_scratch.sh
  or SHAKE_PARTITION=general (a6000, 48 GB -- it will queue). Whichever you
  pick, use the SAME SC_BASE_BATCH for the arms, or their step time and their
  gradient scale are not the shakeout's.

  Only then gate the snapshots on a held-out box:
     HG_ARMS='base knn soft_radial' bash scripts/slurm/submit_gather_holdout_rockstar.sh
  and read the RADIAL row of the verdict. That is the number this whole set was
  built for: HR 1.13 R_vir, frozen SR2 0.53, every fine-tuned arm 0.30-0.35.
EOF
