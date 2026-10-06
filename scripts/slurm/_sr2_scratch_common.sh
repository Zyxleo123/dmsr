# Shared preamble for the FROM-SCRATCH SR2 training jobs. Sourced, never run.
#
# CONFIGURATION COMES IN AS POSITIONAL ARGUMENTS, NOT `sbatch --export`.
# On this cluster any `--export=<list>` (including `--export=ALL,K=v` and
# `--export=NONE`) makes sbatch set SLURM_GET_USER_ENV=1; slurmd then fails to
# rebuild the login environment on the compute node and the job is requeued and
# HELD with "(user env retrieval failed requeued held)", stranding every
# dependent job on Dependency. Arguments travel through the job record
# untouched, so they are the safe channel. Each argument is either `VAR=value`
# or a file to source.
#
#   sbatch scripts/slurm/sr2_scratch_train_gpu.sbatch /path/to/arm.env
#
# List-valued variables in that file MUST be quoted -- an unquoted
# `SC_TRAIN_BOXES=set3 set4` runs `set4` as a command and kills the job. And an
# empty value re-fires the `: "${VAR:=default}"` below, so a deliberately empty
# list is a single space, not "".
#
# Self-sufficiency checked with `env -i`: nothing below needs a login shell.
set -euo pipefail

for _arg in "$@"; do
    case "$_arg" in
        *=*)  export "$_arg" ;;
        *)    [[ -r "$_arg" ]] || { echo "config file not readable: $_arg" >&2; exit 1; }
              # shellcheck disable=SC1090
              source "$_arg" ;;
    esac
done
unset _arg

export PATH="${PATH:-/usr/local/bin:/usr/bin:/bin}"
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-$(getent passwd "$USER" | cut -d: -f6)}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-mpl}"

source /zfsauton/scratch/yixiz/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pjm}"

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts/reward:$PROJECT/scripts/features:${PYTHONPATH:-}"

# The cluster exports SCRATCH itself, so every variable here is prefixed rather
# than relying on ${SCRATCH:-default}.
export ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
export DMSR_REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
LOGS="${LOGS:-$DMSR_REWARD_ROOT/logs}"
mkdir -p "$DMSR_REWARD_ROOT" "$LOGS"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --------------------------------------------------------------------------- #
# The arm
# --------------------------------------------------------------------------- #
: "${SC_LABEL:=base}"                 # names the output dir and the wandb run
: "${SC_CFG:=configs/reward/sr2_direct_finetune.yaml}"
: "${SC_SEED:=0}"
: "${SC_DEVICE:=}"

# --- the data ---------------------------------------------------------------
# set13-15 stay sealed. set9/set10 are the held-out gate boxes and must not
# appear in SC_TRAIN_BOXES: the whole verdict is a box never trained on.
: "${SC_TRAIN_BOXES:=set0 set1 set2 set3 set4 set5 set6 set7 set8 set11 set12}"
: "${SC_HOLDOUT_BOXES:=set9 set10}"
: "${SC_BASE_BATCH:=4}"
: "${SC_BASE_EXCLUDE_SUPERVISED:=0}"

# --- the base objective (map2map's own recipe) ------------------------------
# MSE + a conditional adversary is what produced G_z0.pt. `SC_W_MSE=0` is the
# `base_noms` arm: a pointwise loss to one realisation is this repo's recorded
# cause of mean-collapse, which is what a centrally concentrated halo looks
# like, and it is a prime suspect for the radial defect under test.
: "${SC_W_MSE:=1.0}"
# NO LR-SCALE ANCHOR. There is none in map2map, and 100 -- copied from the
# fine-tune's --w-low, where the term sits near zero because the candidate
# STARTS as the frozen field -- was MEASURED at 96-100% of the non-adversarial
# budget from a random init. Runs 1-2 ended with ~4x LESS small-scale density
# power than trilinear upsampling (0.153 vs 0.706; SR2 is 0.977).
: "${SC_W_BASE_LOW:=0.0}"
# AMPLITUDE-ONLY leash, log(rms_cand/rms_hr)^2 on the displacement and velocity
# channel groups. This is NOT SC_W_BASE_LOW: that one matches an LR-scale FIELD
# and was measured at 96-100% of the budget, smoothing the result. This is two
# scalars per batch with no spatial content at all, so it can say "1e5x too
# loud" and nothing else. It is the term that has to hold the scale once
# SC_W_MSE is 0, because the critic provably cannot.
: "${SC_W_SCALE:=0.0}"
# Linear decay of SC_W_MSE to 0 over this many steps after the adversary starts,
# instead of map2map's single-step cliff. 0 is the cliff. An aggravator, not the
# cause: a cliff also leaves Adam's second moment calibrated to a dead loss.
: "${SC_MSE_HANDOFF:=0}"
# map2map's example-train.slurm: --lr 1e-4, --optimizer Adam, --optimizer-args
# '{}' i.e. PyTorch's default betas.
# The WORKING SR2 recipe (a colleague's z2_train script, 2026-09-25):
#   --lr 1e-5 --adv-lr 2e-5 --optimizer-args '{"betas": [0, 0.99]}'
# The generator rate is 10x LOWER than what runs 1-4 used, and that is the
# largest single difference from the configuration that diverged.
: "${SC_LR:=1e-5}"
: "${SC_BETAS:=0 0.99}"
# `bce` is map2map's own adversary: BCEWithLogitsLoss on labels (its
# --adv-criterion default) + lazy R1 on the REAL batch. There is NO WGAN and no
# interpolate penalty anywhere in map2map/train.py -- the "every 16" that run 1
# ran as a WGAN-GP schedule is R1's idiom, mislabelled.
# WGAN-GP, NOT bce. The working recipe passes `--adv-wgan-gp-interval 16`,
# a flag that does NOT exist in the pinned external/map2map (eelregit) -- SR2
# was trained with a DIFFERENT FORK. Reading the pinned repo and concluding
# "upstream is BCE + R1" was wrong; sr2_adversarial.py's original docstring
# ("WGAN-GP, lambda 10, every 16 -- the upstream schedule") was right.
: "${SC_ADV_LOSS:=wgan_gp}"
# `sum` keeps MSE alongside the adversary. This is a DELIBERATE, MEASURED
# deviation from map2map, which steps the generator on the adversarial loss
# ALONE after --adv-start. The bracket (probes A-D, 2026-09-24) measured why:
#   probe B  sum                                   OK  -- 1750 steps with the
#            adversary live, base_mse 0.09-0.91, d_loss 0.59-1.42 (~ln2)
#   probe C  upstream + spectral norm + inst noise DIVERGED 5.9e10
#   probe D  probe C + a 2x longer handoff         DIVERGED 4.2e11
# C and D blew up within ONE eval interval of w_mse_now hitting 0, with d_loss
# at ~0.001 and the score gap at 16-20. Upstream's own stabilisers did not save
# it.
#
# THE CAUSE IS NOW MEASURED, and it is not "a saturated discriminator" and not
# "map2map's --adv-start is in epochs so its generator converged first" -- both
# were guesses and both are wrong. SR2Critic puts GroupNorm after every block
# but the first, and GroupNorm divides out the per-group std: GN(a*z) == GN(z).
# Conv and LeakyReLU are positively homogeneous, so the WHOLE critic is exactly
# scale-invariant. MEASURED on width=16 depth=4: D pinned at 0.009504 for input
# scalings a from 1e3 to 1e6, while ||grad D|| falls as exactly 1/a (the product
# a*||grad D|| is 0.6846 at every decade), with AND without spectral norm.
#
# So with w_mse 0 and SC_W_BASE_LOW 0, NOTHING in the objective constrains the
# output amplitude, and the one term left both cannot see it and damps its own
# corrective gradient by 1/a as the field walks out. That is why the runs thrash
# across five decades instead of climbing: an unconstrained direction, not a
# diverging one. d_real +9 / d_fake -8 against a field 1e5 too large is a critic
# that was never shown the scale.
#
# map2map's own PatchGAN42 is scale-blind the same way (BatchNorm3d after convs
# 2 and 3, none after conv 1). It survives because its --adv-start is in EPOCHS,
# so its generator reaches the handoff at the right amplitude already; ours is
# 1-2k steps old. "Faithful to map2map" is therefore not a stability argument.
#
# TWO FIXES, both default-off and independently testable (probes E and F):
#   SC_W_SCALE           an amplitude-only leash, log(rms_cand/rms_hr)^2 on the
#                        displacement and velocity groups. Two scalars per batch
#                        and NO spatial information, so unlike MSE it cannot
#                        touch the radial distribution of subhalo mass -- which
#                        is the whole reason for wanting MSE gone.
#   SC_CRITIC_INNER_NORM=none  drops the GroupNorms so the critic is positively
#                        homogeneous and amplitude becomes learnable. The root
#                        fix; pair it with SC_CRITIC_SPECTRAL_NORM=1 (SNGAN).
# `sum` stays the default until one of those is gated.
# `upstream`: MSE carries the generator for --adv-start epochs, then the
# adversary carries it alone. That is the working recipe's form (--adv-start 1).
# Runs 3 and probes C/D diverged under it -- but at lr 1e-4 with betas
# (0.9, 0.999) and BCE. `sum` is the fallback the bracket measured as stable
# (probe B) if this diverges again at the corrected rates.
: "${SC_MSE_MODE:=upstream}"
: "${SC_ADV_LABEL_SMOOTHING:=1.0}"
: "${SC_R1_GAMMA:=10.0}"
# map2map's --adv-r1-reg-interval default is 0 (off); kept ON here because a GAN
# from a random init is unstable without it. A named deviation, not a silent one.
: "${SC_R1_INTERVAL:=16}"
# map2map's two stabilisers, both default-off upstream and both aimed at exactly
# the failure that ended run 3 (d_real +8.5 vs d_fake -8.2, d_loss 0.0005, the
# generator free to walk to 1e11 with no gradient back).
: "${SC_CRITIC_SPECTRAL_NORM:=0}"
# `group` (the default) makes SR2Critic EXACTLY scale-invariant -- see the long
# note above the SC_MSE_MODE block. `none` is the root fix for the handoff
# divergence and wants SC_CRITIC_SPECTRAL_NORM=1 alongside it.
: "${SC_CRITIC_INNER_NORM:=group}"
: "${SC_INSTANCE_NOISE:=0.0}"
: "${SC_INSTANCE_NOISE_BATCHES:=10000}"
: "${SC_CLIP:=1.0}"

# --- the DENSITY critic -----------------------------------------------------
# SR2Critic's 20 channels are 6 upsampled LR + 6 field + 8 inverse-pixel-
# shuffled CIC DENSITY. The density half is the point: it discriminates the
# mass field rather than the displacement that produced it.
#
# SC_DENSITY_VALID_CENTER=0 means "tile/2 = 32", the EXACT deposit (r = 1.0000).
# It does NOT mean the wrapped `% ng` deposit -- that one correlates r = 0.08
# with a tile's own density (docs/density_collapse_investigation.md), so the 8
# density channels would be noise and the run would look healthy anyway. The
# trainer refuses it unless SC_ALLOW_WRAPPED_DENSITY=1.
: "${SC_W_SR2_ADV:=1.0}"
: "${SC_CRITIC_LR:=2e-5}"
: "${SC_CRITIC_WIDTH:=64}"
: "${SC_CRITIC_DEPTH:=4}"
: "${SC_CRITIC_STEPS:=1}"
: "${SC_GP_LAMBDA:=10.0}"
: "${SC_GP_EVERY:=16}"
# Their --adv-start is 1 EPOCH, not a step count. SC_ADV_START_EPOCHS>0
# overrides this, computing it from the sampler: 5632 pairs / batch 4 = 1408
# steps per epoch on our 11 training boxes.
: "${SC_ADV_WARMUP:=1000}"
: "${SC_ADV_START_EPOCHS:=1}"
: "${SC_DENSITY_GRID_MULT:=2}"
: "${SC_DENSITY_VALID_CENTER:=0}"
: "${SC_ALLOW_WRAPPED_DENSITY:=0}"
# The two fixes for the critic runaway MEASURED in jobs 62067/62068 (score gap
# +2548 at step 30k, ||grad D|| ~ 4 against the Lipschitz target of 1):
#   * per-channel scales fitted from real HR tiles. Raw, density enters at 5.8x
#     the field channels, so the field half starves. DMSR's
#     CriticInputNormalizer and the gather critic's GatherCriticNorm both exist
#     for this; the SR2 critic path did not have one, which was the bug.
#   * asinh on the density channels. Raw max 2786 against RMS 4.9 -- a 570x
#     spread that a single scale cannot fix, and it is what makes the gradient
#     penalty unenforceable off the density directions. asinh maps 2786 -> 8.6
#     and is finite at delta = -1, the MEDIAN voxel (log1p is -inf there).
# SC_DENSITY_TRANSFORM=none is the faithful-to-upstream arm, and is the one
# measured to diverge.
: "${SC_DENSITY_TRANSFORM:=asinh}"
: "${SC_CRITIC_NORM_TILES:=32}"

# --- the initialisation -----------------------------------------------------
# AddNoise.std starts at 0 (upstream's own init), so the generator is
# deterministic until the gradient grows it. SR2's fine-scale structure is
# written entirely by those six tensors, so `noise_std_mean` in metrics.jsonl is
# a first-class number, not a curiosity: a run that leaves them at zero has
# produced a deterministic upsampler however good its spectrum looks.
: "${SC_INIT_WEIGHT_STD:=0.0}"
: "${SC_INIT_NOISE_STD:=0.0}"

# --- the structure terms ----------------------------------------------------
# Held at zero for SC_TERM_WARMUP steps and ramped over SC_TERM_RAMP. This is
# not a nicety: the soft-membership solve picks particles FROM the current SR
# field and the kNN term measures it, and at step 0 that field is noise.
: "${SC_TERM_WARMUP:=2000}"
: "${SC_TERM_RAMP:=2000}"

# member-gather (0 = the `base` arms; no host pool is built at all)
: "${SC_W_GATHER:=0.0}"
: "${SC_SOFT_MEMBERSHIP:=0}"
: "${SC_SOFT_PRIOR:=coarse}"
: "${SC_SOFT_LAMBDA:=1.0}"
: "${SC_SOFT_POOL_RATIO:=2.0}"
: "${SC_MAX_SET_PARTICLES:=768}"
: "${SC_W_RADIUS:=0.0}"
: "${SC_RADIUS_DEAD_ZONE:=1.0}"
: "${SC_HOSTS_PER_STEP:=1}"
: "${SC_N_TILES:=4}"
: "${SC_MAX_HOSTS_PER_BOX:=4}"
: "${SC_MIN_LOG_MVIR:=13.5}"
# MG_W_D6 / MG_CENTRE_MODE / MG_W_HOST_SETS defaults are NOT the winning arm's
# (`mgtrain-defaults-are-not-the-winning-arm`), so they are set here explicitly
# rather than inherited.
: "${SC_W_D6:=1.0}"
: "${SC_CENTRE_MODE:=full}"
: "${SC_W_HOST_SETS:=0.0}"

# kNN-CDF (0 = off). Round 5b MEASURED these and they are not the CLI defaults:
#   phase lambda 4000, ks "8 16 32 64", classes sub_50_200.
# HOSTS MUST STAY OUT OF THE TERM. The pooled loss ranks the working arms at
# rho +0.878 (p 4e-07) -- the wrong sign -- and 6-D makes it worse.
: "${SC_W_KNN:=0.0}"
: "${SC_KNN_PHASE_LAMBDA:=4000}"
: "${SC_KNN_KS:=8 16 32 64}"
: "${SC_KNN_CLASSES:=sub_50_200}"
: "${SC_KNN_MEASURE:=0}"
# MEASURED, job 62069: with SC_UNSUP_TILES_PER_STEP=8 and 1024 queries the term
# costs >13.6 s/step against 0.30 s/step before it engages -- a 45x slowdown, or
# 15 days for 100k steps. The per-step cost is (tiles x queries x neighbour-table
# rows) in float64, and the tables are 1.3-3.25x a tile each. Both factors are
# linear, so tiles-per-step 1 and 256 queries is ~32x off that bill.
: "${SC_KNN_QUERIES:=1024}"
: "${SC_UNSUP_TILES_PER_BOX:=0}"
: "${SC_UNSUP_TILES_PER_STEP:=8}"

# --- the schedule -----------------------------------------------------------
: "${SC_STEPS:=100000}"
: "${SC_EVAL_EVERY:=500}"
: "${SC_SAVE_EVERY:=500}"
: "${SC_GATE_AT_STEPS:=}"
: "${SC_RESUME:=1}"
: "${SC_SHAKEOUT_STEPS:=0}"
: "${SC_NO_WANDB:=0}"

# numpy's default is one thread per core; several concurrent jobs on an 88-core
# node then fight each other. Take it from the allocation.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

echo "=== $(date '+%F %T')  job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}"
echo "    arm=$SC_LABEL  steps=$SC_STEPS  seed=$SC_SEED"
echo "    train boxes: $SC_TRAIN_BOXES"
echo "    holdout:     $SC_HOLDOUT_BOXES"
echo "    base: w_mse=$SC_W_MSE w_adv(SR2Critic,density)=$SC_W_SR2_ADV lr=$SC_LR"
echo "    critic input: density transform=$SC_DENSITY_TRANSFORM, scales from $SC_CRITIC_NORM_TILES real HR tiles"
echo "    objective: adv=$SC_ADV_LOSS mse-mode=$SC_MSE_MODE w_base_low=$SC_W_BASE_LOW lr=$SC_LR betas=($SC_BETAS) R1 $SC_R1_GAMMA/$SC_R1_INTERVAL"
echo "    structure: w_gather=$SC_W_GATHER w_radius=$SC_W_RADIUS w_knn=$SC_W_KNN"
echo "               warmup=$SC_TERM_WARMUP ramp=$SC_TERM_RAMP"
echo "    reward_root=$DMSR_REWARD_ROOT"
echo "    threads=$OMP_NUM_THREADS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# ---- flag assembly ---------------------------------------------------------
# Built here so the job body is one command and so an arm's flags are visible in
# the log next to the numbers they produced.
sc_flags() {
    local -a f=(
        --config "$SC_CFG"
        --label "$SC_LABEL"
        --seed "$SC_SEED"
        --train-boxes "$SC_TRAIN_BOXES"
        --holdout-boxes "$SC_HOLDOUT_BOXES"
        --steps "$SC_STEPS"
        --eval-every "$SC_EVAL_EVERY"
        --save-every "$SC_SAVE_EVERY"
        --clip "$SC_CLIP"
        --scratch-lr "$SC_LR"
        --adv-loss "$SC_ADV_LOSS"
        --mse-mode "$SC_MSE_MODE"
        --adv-label-smoothing "$SC_ADV_LABEL_SMOOTHING"
        --r1-gamma "$SC_R1_GAMMA"
        --r1-interval "$SC_R1_INTERVAL"
        --adv-start-epochs "$SC_ADV_START_EPOCHS"
        --instance-noise "$SC_INSTANCE_NOISE"
        --instance-noise-batches "$SC_INSTANCE_NOISE_BATCHES"
        --init-weight-std "$SC_INIT_WEIGHT_STD"
        --init-noise-std "$SC_INIT_NOISE_STD"
        --base-batch "$SC_BASE_BATCH"
        --w-mse "$SC_W_MSE"
        --w-base-low "$SC_W_BASE_LOW"
        --w-scale "$SC_W_SCALE"
        --mse-handoff-steps "$SC_MSE_HANDOFF"
        --critic-inner-norm "$SC_CRITIC_INNER_NORM"
        --w-sr2-adv "$SC_W_SR2_ADV"
        --sr2-critic-lr "$SC_CRITIC_LR"
        --sr2-critic-width "$SC_CRITIC_WIDTH"
        --sr2-critic-depth "$SC_CRITIC_DEPTH"
        --sr2-critic-steps "$SC_CRITIC_STEPS"
        --sr2-gp-lambda "$SC_GP_LAMBDA"
        --sr2-gp-every "$SC_GP_EVERY"
        --sr2-adv-warmup-steps "$SC_ADV_WARMUP"
        --density-grid-mult "$SC_DENSITY_GRID_MULT"
        --density-valid-center "$SC_DENSITY_VALID_CENTER"
        --density-transform "$SC_DENSITY_TRANSFORM"
        --sr2-critic-norm-tiles "$SC_CRITIC_NORM_TILES"
        --term-warmup-steps "$SC_TERM_WARMUP"
        --term-ramp-steps "$SC_TERM_RAMP"
        --w-gather "$SC_W_GATHER"
        --hosts-per-step "$SC_HOSTS_PER_STEP"
        --n-tiles "$SC_N_TILES"
        --max-hosts-per-box "$SC_MAX_HOSTS_PER_BOX"
        --min-log-mvir "$SC_MIN_LOG_MVIR"
        --w-d6 "$SC_W_D6"
        --centre-mode "$SC_CENTRE_MODE"
        --w-host-sets "$SC_W_HOST_SETS"
        --w-radius "$SC_W_RADIUS"
        --radius-dead-zone "$SC_RADIUS_DEAD_ZONE"
        --w-knn-unsup "$SC_W_KNN"
        --knn-phase-lambda "$SC_KNN_PHASE_LAMBDA"
        --knn-queries "$SC_KNN_QUERIES"
        --unsup-tiles-per-box "$SC_UNSUP_TILES_PER_BOX"
        --unsup-tiles-per-step "$SC_UNSUP_TILES_PER_STEP"
        --max-set-particles "$SC_MAX_SET_PARTICLES"
    )
    # shellcheck disable=SC2206
    [[ -n "${SC_BETAS// }" ]] && f+=(--scratch-betas ${SC_BETAS})
    # shellcheck disable=SC2206
    [[ -n "${SC_KNN_KS// }" ]] && f+=(--knn-ks ${SC_KNN_KS})
    # shellcheck disable=SC2206
    [[ -n "${SC_KNN_CLASSES// }" ]] && f+=(--knn-classes ${SC_KNN_CLASSES})
    # ONE argument, quoted: --gate-at-steps takes a single "1000,2000" string,
    # not a list. Unquoted it splits and argparse rejects the tail.
    [[ -n "${SC_GATE_AT_STEPS// }" ]] && f+=(--gate-at-steps "$SC_GATE_AT_STEPS")
    [[ "$SC_SOFT_MEMBERSHIP" == "1" ]] && f+=(
        --soft-membership --soft-prior "$SC_SOFT_PRIOR"
        --soft-lambda "$SC_SOFT_LAMBDA" --soft-pool-ratio "$SC_SOFT_POOL_RATIO")
    [[ "$SC_KNN_MEASURE" == "1" ]] && f+=(--knn-measure)
    [[ "$SC_BASE_EXCLUDE_SUPERVISED" == "1" ]] && f+=(--base-exclude-supervised)
    [[ "$SC_ALLOW_WRAPPED_DENSITY" == "1" ]] && f+=(--allow-wrapped-density)
    [[ "$SC_CRITIC_SPECTRAL_NORM" == "1" ]] && f+=(--critic-spectral-norm)
    [[ "$SC_RESUME" == "1" ]] && f+=(--resume)
    [[ "$SC_NO_WANDB" == "1" ]] && f+=(--no-wandb)
    [[ -n "$SC_DEVICE" ]] && f+=(--device "$SC_DEVICE")
    (( SC_SHAKEOUT_STEPS > 0 )) && f+=(--shakeout-steps "$SC_SHAKEOUT_STEPS")
    printf '%s\n' "${f[@]}"
}
