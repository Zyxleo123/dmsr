# Shared preamble for every density-fix SLURM job. Sourced, never executed.
#
# Every knob is an environment variable with a default, so a job is configured
# by its arguments alone, and the defaults live here once instead of drifting
# across eight .sbatch files.
#
#   sbatch scripts/slurm/dfix_stage0_ceiling.sbatch OUT=runs/dfix_v2
#   sbatch scripts/slurm/dfix_stage0_ceiling.sbatch /path/to/dfix_env.sh
#
# CONFIGURATION COMES IN AS POSITIONAL ARGUMENTS, NOT `sbatch --export`.
# On this cluster any `--export=<list>` (with or without ALL) makes sbatch set
# SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login environment on
# the compute node, and the job is requeued and held with
# `(user env retrieval failed requeued held)` -- it never starts. Arguments go
# through the job record untouched, so they are the safe channel. Each argument
# is either a `VAR=value` override or a file to source.
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

: "${DATA:=/zfsauton/scratch/yixiz/DMSR/paired_catnorm}"
: "${BOX:=set14}"
: "${OUT:=runs/dmsr/dfix}"
: "${DEVICE:=cuda}"
: "${N_STEPS:=20}"

LR_NPY="$DATA/lr/$BOX.npy"
HR_NPY="$DATA/hr/$BOX.npy"

# The four arms of the density-critic fix sweep, as `label:run_dir`. All four
# share the same data split, seed and step budget; they differ only in the
# config knobs named in the label.
#
#   base   t13_unc_fulldisp_pshuffle8_l003 -- wrapped CIC target, GroupNorm, zero pad
#   vc32   + critic.valid_center: 32       -- correct density target
#   chnorm + model.norm: channel           -- spatially local normalisation
#   circ   + model.padding_mode: circular  -- periodic padding
#
# `sbatch --export` splits its argument on commas, so a value containing spaces
# survives only by accident of quoting; ARMS is therefore passed comma-separated
# and translated back to a space list here, so both forms work.
: "${ARMS:=base:runs/dmsr/t13_unc_fulldisp_pshuffle8_l003_s0 vc32:runs/dmsr/t13_fix_vc32_s0 chnorm:runs/dmsr/t13_fix_vc32_chnorm_s0 circ:runs/dmsr/t13_fix_vc32_chnorm_circ_s0}"
ARMS="${ARMS//,/ }"

# The arm the single-model diagnostics (Stages 3, 4, 6) probe. Those stages ask
# "what does this network see", which is a property of one checkpoint, so
# running them on all four would quadruple GPU time for no extra decision.
: "${PROBE_ARM:=vc32}"

# `--dmsr label:config:ckpt` triples for compare_flow_baseline.py.
DMSR_ARGS=()
for spec in $ARMS; do
  label="${spec%%:*}"; dir="${spec#*:}"
  DMSR_ARGS+=(--dmsr "$label:$dir/config.yaml:$dir/ckpt_best.pt")
done

probe_dir() {  # echo the run dir of $1, empty if that arm is not in ARMS
  for spec in $ARMS; do
    [ "${spec%%:*}" = "$1" ] && { echo "${spec#*:}"; return 0; }
  done
  return 1
}

mkdir -p "$OUT"

echo "=== $(date '+%F %T')  job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}"
echo "    project=$PROJECT  out=$OUT  box=$BOX  device=$DEVICE"
echo "    arms: $ARMS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Gate helpers -------------------------------------------------------------- #
# A stage whose input is missing must exit 0. A non-zero exit puts every
# dependent job into DependencyNeverSatisfied, where it sits with no explanation
# of what actually went wrong. Exiting 0 with a printed reason lets each later
# stage reach the same conclusion and say so in its own log.
#
# "gate failed" and "upstream never ran" stay distinct so that resubmitting one
# stage on its own is not mistaken for a real failure.
require_input() {  # $1 = path, $2 = which stage produces it
  if [ ! -e "$1" ]; then
    echo ">>> MISSING INPUT: $1"
    echo ">>> produced by: $2 -- it did not run, or it ran and produced nothing."
    echo ">>> skipping this stage (exit 0 so dependents can report the same)."
    exit 0
  fi
}

require_ckpt() {   # $1 = run dir, $2 = arm label
  if [ ! -f "$1/ckpt_best.pt" ]; then
    echo ">>> NO CHECKPOINT for arm '$2' at $1/ckpt_best.pt -- training did not finish."
    echo ">>> skipping this stage."
    exit 0
  fi
}
