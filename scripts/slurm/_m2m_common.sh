# Shared preamble for the runs that use map2map's OWN trainer. Sourced, never run.
#
# CONFIGURATION COMES IN AS POSITIONAL ARGUMENTS, NOT `sbatch --export`.
# Any explicit export list makes sbatch set SLURM_GET_USER_ENV=1; slurmd then
# fails to rebuild the login environment and the job is requeued and HELD with
# "(user env retrieval failed requeued held)". Arguments travel through the job
# record untouched. Each argument is either `VAR=value` or a file to source.
#
# List values in that file MUST be quoted -- an unquoted `M2M_TRAIN_BOXES=set3
# set4` runs `set4` as a command. An empty value re-fires the `:=` default
# below, so a deliberately empty list is a single space, not "".
#
# WHY THIS EXISTS AT ALL. Four from-scratch runs on our own trainer failed for
# four different reasons, and every one traced to a component we had
# reimplemented rather than to the experiment: a critic with no input
# normaliser, an LR-scale anchor at 96-100% of the budget, a BCE adversary we
# adopted after misreading which map2map fork trained SR2, and a GroupNorm
# critic that is exactly scale-invariant. This runs map2map's `styled_srsgan`
# branch as-is instead. The ONLY things of ours that may enter are the two new
# loss terms this whole line exists to test.
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

# map2map's OWN checkout, on the branch whose adversary we are adopting. It is
# a git worktree of DMSR/map2map at xw/styled_srsgan, so `git -C` there shows
# exactly which commit ran.
export M2M_REPO="${M2M_REPO:-/zfsauton2/home/yixiz/DMSR/map2map_styled}"
export PYTHONPATH="$M2M_REPO:$PROJECT/src:$PROJECT/scripts/reward:${PYTHONPATH:-}"

# The cluster exports SCRATCH itself, so every variable here is prefixed.
export ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
export DMSR_REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
LOGS="${LOGS:-$DMSR_REWARD_ROOT/logs}"
mkdir -p "$DMSR_REWARD_ROOT" "$LOGS"

# --- data -------------------------------------------------------------------
: "${M2M_DATA_ROOT:=$ZFS/DMSR/paired_catnorm}"
: "${M2M_LAYOUT:=$ZFS/DMSR/m2m_layout}"
# set13-15 stay sealed. set9/set10 are the held-out boxes and must never appear
# in the training list: the verdict of this line is a box never trained on.
: "${M2M_TRAIN_BOXES:=set0 set1 set2 set3 set4 set5 set6 set7 set8 set11 set12}"
: "${M2M_VAL_BOXES:=set9 set10}"
# 0 disables validation entirely -- see the note at the val patterns below.
: "${M2M_VAL:=1}"

# --- which map2map tree ------------------------------------------------------
# The colleague's own tree (map2map_colleague, from the zip they sent) differs
# from xw/styled_srsgan in ways that change the ARGUMENT LIST, not just values:
#
#   M2M_STYLE=0     their args.py has NO style at all and their FieldDataset
#                   takes no style_pattern, so the three --*-style-pattern args
#                   are a hard argparse error there. Their srsgan.G/.D take no
#                   style either -- train.py:397 is `output = model(input)`.
#   M2M_LAG2EUL=1   there it is a FLAG ('concatenate eul to Discriminator');
#                   on styled_srsgan the same thing happens unconditionally.
#   M2M_MESH_UP_FAC the up-factor lag2eul paints to. Their D sizing is
#                   `inchannel += 1; if fac > 1: inchannel += fac**3 - 1`
#                   (train.py:164-170), so fac=2 gives 8 -- exactly the `+8`
#                   that styled_srsgan.D hard-codes. Same arithmetic.
#   M2M_NORMS=0     omit --in-norms/--tgt-norms entirely. See the note at the
#                   norm lines in m2m_write_args: on their tree that is exactly
#                   identity AND it is the only way the dataloader survives.
: "${M2M_STYLE:=1}"
: "${M2M_NORMS:=1}"
: "${M2M_LAG2EUL:=0}"
: "${M2M_MESH_UP_FAC:=1}"

# --- scoring (m2m_field_report_gpu.sbatch) ----------------------------------
# Empty means "this run's checkpoint.pt". The report boxes are HELD OUT and
# must stay that way: M2M_TRAIN_BOXES above deliberately excludes set9/set10,
# and the verdict of this line is a box never trained on.
: "${M2M_CKPT:=}"
: "${M2M_REPORT_BOXES:=set9 set10}"
: "${M2M_REPORT_SEED:=0}"
# The style is the scale factor a. Our boxes are z = 0 -> a = 1.0. The
# styled_srsgan branch calls model(input, style) and lag2eul(..., a=style)
# UNCONDITIONALLY, so a style file is mandatory, not optional.
: "${M2M_STYLE_A:=1.0}"
: "${M2M_SPLIT_OVERWRITE:=}"

# --- the objective, straight off the working sbatch -------------------------
# srsgan.G/.D would be the letter of that script, but this branch passes `style`
# to both the model and the adversary unconditionally, and only the styled
# classes take it. Using styled_srsgan.G/.D keeps THEIR code unedited, which is
# the point of running their trainer at all.
: "${M2M_MODEL:=styled_srsgan.G}"
: "${M2M_ADV_MODEL:=styled_srsgan.D}"
: "${M2M_CRITERION:=MSELoss}"
: "${M2M_ADV_CRITERION:=WDistLoss}"
: "${M2M_ADV_WGAN_GP_INTERVAL:=16}"
: "${M2M_ADV_START:=1}"
: "${M2M_LR:=1e-5}"
: "${M2M_ADV_LR:=2e-5}"
# NOT `: "${VAR:=default}"` -- bash ends the expansion at the first unescaped
# `}`, so a JSON default is truncated to `{"betas": [0, 0.99]` and argparse
# rejects it. That killed job 65050 before a single step.
if [[ -z "${M2M_OPTIMIZER_ARGS:-}" ]]; then
    # 0.0 not 0: their script writes [0, 0.99], which json parses as (int, float),
    # and torch 2.11's Adam raises "betas must be either both floats or both
    # Tensors". Numerically identical, and it is a torch-version fix, not a
    # change to their recipe.
    M2M_OPTIMIZER_ARGS='{"betas": [0.0, 0.99]}'
fi
: "${M2M_EPOCHS:=1024}"
: "${M2M_BATCHES:=1}"
# map2map reads each file WHOLE on first touch (fields.py: "use memmap after
# reading a file once"), and an HR box is 6 GB across disp+vel. Peak scales with
# the worker count, so this stays low until m2m_preflight_cpu.sbatch MEASURES it.
: "${M2M_LOADER_WORKERS:=2}"
# OUR ONLY DEVIATION IN GEOMETRY, and it is deliberate. Their script uses
# `--crop 9 --crop-step 9 --aug-shift 9`. Our two new loss terms are indexed by
# (box, tile) on the fixed 8-cell LR tile grid -- the member-gather loss needs
# the HR owner array and Rockstar catalog for THAT tile, and the kNN term needs
# its precomputed HR neighbour table. Arbitrary 9-cell crops at random shifts
# correspond to no tile we have catalogs for. Crop 8 / step 8 with no shift
# keeps the losses valid without touching map2map. Cubic flips and permutations
# (--augment) still apply; only the sub-cell shift is dropped.
: "${M2M_CROP:=8}"
: "${M2M_CROP_STEP:=8}"
: "${M2M_PAD:=3}"
: "${M2M_SCALE_FACTOR:=8}"
: "${M2M_AUGMENT:=1}"
: "${M2M_AUG_SHIFT:=}"
# Identity: our boxes are ALREADY in cosmology.dis space (that norm divides by
# dis_std * D(z) = 6000 kpc/h at z = 0, which is exactly what paired_catnorm was
# written with). Applying it again would shrink every displacement by 6000x.
# --callback-at is map2map's own mechanism for this, not an edit to their code.
: "${M2M_CALLBACK_AT:=$PROJECT/scripts/m2m_callback}"
: "${M2M_IN_NORMS:=srnorms.identity,srnorms.identity}"
: "${M2M_TGT_NORMS:=srnorms.identity,srnorms.identity}"
# map2map's --scheduler-args DEFAULT is '{"verbose": true}', and torch removed
# ReduceLROnPlateau's `verbose` argument. Empty JSON keeps their scheduler with
# its own defaults and needs no edit to their code.
# Same trap as M2M_OPTIMIZER_ARGS, and I walked into it twice: bash ends a
# ${VAR:=default} expansion at the first unescaped `}`, so ANY brace-bearing
# default must be a plain conditional assignment, never `:=`.
if [[ -z "${M2M_SCHEDULER_ARGS:-}" ]]; then
    M2M_SCHEDULER_ARGS='{}'
fi
: "${M2M_RUN_NAME:=m2m_baseline}"
: "${M2M_EXTRA_ARGS:=}"

export M2M_OUT="${M2M_OUT:-$DMSR_REWARD_ROOT/m2m/$M2M_RUN_NAME}"

# The dataloader workers must be able to `import srnorms` themselves.
# MEASURED, job 69288: the colleague's FieldDataset resolves --in-norms to
# FUNCTION OBJECTS at __init__ (data/fields.py:71-73) and stores them, so the
# spawn-based workers have to pickle them and die with
#   PicklingError: Can't pickle <function identity ...>: import of module
#   'srnorms' failed
# because --callback-at only put the directory on the PARENT's sys.path.
# xw/styled_srsgan stores the norm NAMES and resolves per call inside
# __getitem__ (fields.py:235-241), so it never hits this. Exporting the
# callback dir fixes it for both trees and edits neither.
export PYTHONPATH="$M2M_CALLBACK_AT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$M2M_OUT"

# numpy's default is one thread per core; several concurrent jobs on an 88-core
# node then fight each other. Take it from the allocation.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

# ---- the argument list, in ONE place ---------------------------------------
# Both the preflight and the training job build their arguments here, so the
# thing that is validated is character-for-character the thing that runs. The
# two earlier failures were both argument-level, and a preflight that rebuilt
# the list itself could have passed while the job still died.
#
# Written one-per-line, which also sidesteps the quoting that broke the first
# hand-run preflight: `--optimizer-args {"betas": [0.0, 0.99]}` word-splits into
# `{betas:` the moment it passes through an unquoted shell expansion.
m2m_write_args() {
    local out="${1:?m2m_write_args needs an output path}"
    mkdir -p "$(dirname "$out")"
    {
        printf '%s\n' --train-in-patterns "$M2M_LAYOUT/train/lr/set*/disp.npy,$M2M_LAYOUT/train/lr/set*/vel.npy"
        printf '%s\n' --train-tgt-patterns "$M2M_LAYOUT/train/hr/set*/disp.npy,$M2M_LAYOUT/train/hr/set*/vel.npy"
        [[ "$M2M_STYLE" != 0 ]] && \
            printf '%s\n' --train-style-pattern "$M2M_LAYOUT/train/style/set*.npy"
        # M2M_VAL=0 omits the three val patterns, which is how map2map itself
        # turns validation off: args.py:229 sets `args.val` from
        # `val_in_patterns is not None`. No code edit, their own interface.
        #
        # MEASURED, job 68906: this branch's validate() CANNOT run an
        # adversarial epoch. train() builds D's input as
        # cat([lag2eul(out), out]) and then the cgan cat, 8 + 6 + 6 = 20
        # channels (train.py:378-389). validate() does ONLY the cgan cat, 6 + 6
        # = 12, and feeds that to the same D (train.py:590-595). So the first
        # epoch with `epoch >= adv_start` dies at the END of it, in validation:
        #   RuntimeError: shape '[1, 20, 64, 64, 64]' is invalid for input of
        #   size 3145728            (3145728 = 12 * 64^3, i.e. the missing 8)
        # Losing the val MSE costs nothing here -- MSE is the wrong yardstick
        # for this line anyway, and checkpoints are scored on density P/P_HR.
        if [[ "$M2M_VAL" != 0 ]]; then
            printf '%s\n' --val-in-patterns "$M2M_LAYOUT/val/lr/set*/disp.npy,$M2M_LAYOUT/val/lr/set*/vel.npy"
            printf '%s\n' --val-tgt-patterns "$M2M_LAYOUT/val/hr/set*/disp.npy,$M2M_LAYOUT/val/hr/set*/vel.npy"
            [[ "$M2M_STYLE" != 0 ]] && \
                printf '%s\n' --val-style-pattern "$M2M_LAYOUT/val/style/set*.npy"
        fi
        # M2M_NORMS=0 omits --in-norms/--tgt-norms ENTIRELY, which on their tree
        # is exactly identity: fields.py resolves norms only `if in_norms is not
        # None` (:68) and applies them only `if self.in_norms is not None`
        # (:152). Passing nothing == passing a no-op norm, with none of the cost.
        #
        # And the cost is real. `srnorms.identity` reached through --callback-at
        # cannot survive their dataloader: their FieldDataset resolves norms to
        # FUNCTION OBJECTS at __init__ (:71-73), which the spawn workers must
        # then pickle. MEASURED:
        #   job 69288  PicklingError: ... import of module 'srnorms' failed
        #   job 69290  PicklingError: ... it's not the same object as
        #              srnorms.identity  -- PYTHONPATH fixed the import, but
        #              import_attr loads the file as its OWN module instance, so
        #              pickle's identity check still fails.
        # xw/styled_srsgan stores the norm NAMES and resolves per call
        # (:235-241), never pickling a function, which is why it worked there.
        if [[ "$M2M_NORMS" != 0 ]]; then
            printf '%s\n' --in-norms "$M2M_IN_NORMS" --tgt-norms "$M2M_TGT_NORMS"
        fi
        printf '%s\n' --callback-at "$M2M_CALLBACK_AT"
        printf '%s\n' --model "$M2M_MODEL" --adv-model "$M2M_ADV_MODEL" --cgan
        printf '%s\n' --criterion "$M2M_CRITERION" --adv-criterion "$M2M_ADV_CRITERION"
        [[ "$M2M_LAG2EUL" == "1" ]] && printf '%s\n' --lag2eul
        (( M2M_MESH_UP_FAC > 1 )) && printf '%s\n' --mesh-up-fac "$M2M_MESH_UP_FAC"
        printf '%s\n' --adv-start "$M2M_ADV_START"
        printf '%s\n' --adv-wgan-gp-interval "$M2M_ADV_WGAN_GP_INTERVAL"
        printf '%s\n' --lr "$M2M_LR" --adv-lr "$M2M_ADV_LR"
        printf '%s\n' --optimizer-args "$M2M_OPTIMIZER_ARGS"
        printf '%s\n' --scheduler-args "$M2M_SCHEDULER_ARGS"
        printf '%s\n' --crop "$M2M_CROP" --crop-step "$M2M_CROP_STEP"
        printf '%s\n' --pad "$M2M_PAD" --scale-factor "$M2M_SCALE_FACTOR"
        printf '%s\n' --batches "$M2M_BATCHES" --loader-workers "$M2M_LOADER_WORKERS"
        printf '%s\n' --epochs "$M2M_EPOCHS"
        [[ "$M2M_AUGMENT" == "1" ]] && printf '%s\n' --augment
        [[ -n "${M2M_AUG_SHIFT// }" ]] && printf '%s\n' --aug-shift "$M2M_AUG_SHIFT"
        [[ -r "$M2M_OUT/checkpoint.pt" ]] && printf '%s\n' --load-state "$M2M_OUT/checkpoint.pt"
        # shellcheck disable=SC2086
        [[ -n "${M2M_EXTRA_ARGS// }" ]] && printf '%s\n' ${M2M_EXTRA_ARGS}
        :
    } > "$out"
}

echo "=== $(date '+%F %T')  job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}"
echo "    run=$M2M_RUN_NAME  out=$M2M_OUT"
echo "    map2map=$M2M_REPO @ $(git -C "$M2M_REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "    layout=$M2M_LAYOUT"
echo "    train: $M2M_TRAIN_BOXES"
echo "    val:   $M2M_VAL_BOXES"
echo "    model=$M2M_MODEL adv=$M2M_ADV_MODEL crit=$M2M_ADV_CRITERION gp_every=$M2M_ADV_WGAN_GP_INTERVAL"
echo "    lr=$M2M_LR adv_lr=$M2M_ADV_LR opt='$M2M_OPTIMIZER_ARGS' adv_start=$M2M_ADV_START"
echo "    crop=$M2M_CROP step=$M2M_CROP_STEP pad=$M2M_PAD scale=$M2M_SCALE_FACTOR augment=$M2M_AUGMENT"
echo "    norms=$M2M_IN_NORMS (identity: our data is already in cosmology.dis space)"
echo "    threads=$OMP_NUM_THREADS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
