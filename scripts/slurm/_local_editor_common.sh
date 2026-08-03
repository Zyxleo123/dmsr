# Shared preamble for the host-conditioned local-editor jobs. Sourced, never run.
#
# CONFIGURATION COMES IN AS POSITIONAL ARGUMENTS, NOT `sbatch --export`.
# On this cluster any `--export=<list>` makes sbatch set SLURM_GET_USER_ENV=1,
# slurmd then fails to rebuild the login environment on the compute node, and
# the job is requeued and HELD with `(user env retrieval failed requeued held)`,
# stranding every dependent job on `Dependency`. Arguments travel through the
# job record untouched, so they are the safe channel. Each argument is either
# `VAR=value` or a file to source.
#
#   sbatch scripts/slurm/local_editor_candidates_cpu.sbatch RUN_NAME=le_a BOX=set8
#
# Self-sufficiency was checked with `env -i`: nothing below depends on a login
# shell having run.
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
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts/reward:${PYTHONPATH:-}"

export ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
# Every bulky artifact of THIS line lands here -- a sibling of the reward root,
# never inside it, never under home or the repository.
export DMSR_LOCAL_EDITOR_ROOT="${DMSR_LOCAL_EDITOR_ROOT:-$ZFS/DMSR/dmsr_local_editor}"
# The reward root is still read: the frozen SR2 field cache, the frozen base
# catalogs and the Experiment-1 oracle rows all live there and are INPUTS here.
export DMSR_REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
LOGS="${LOGS:-$DMSR_LOCAL_EDITOR_ROOT/logs}"
mkdir -p "$DMSR_LOCAL_EDITOR_ROOT" "$LOGS"

: "${DATA:=/zfsauton/scratch/yixiz/DMSR/paired_catnorm}"
: "${LE_CFG:=configs/reward/local_editor.yaml}"
: "${REWARD_CFG:=configs/reward/reward.yaml}"
: "${RUN_NAME:=le_a}"
: "${BASE_SEED:=0}"
: "${SEED:=0}"
: "${DEVICE:=cuda}"

# numpy's default is one thread per core; several concurrent FFT jobs on an
# 88-core node then fight each other. Set it from the allocation.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

echo "=== $(date '+%F %T')  job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}"
echo "    project=$PROJECT  run=$RUN_NAME"
echo "    local_editor_root=$DMSR_LOCAL_EDITOR_ROOT"
echo "    reward_root(read-only inputs)=$DMSR_REWARD_ROOT"
echo "    threads=$OMP_NUM_THREADS"

require_input() {   # $1 = path, $2 = which stage produces it
  if [ ! -e "$1" ]; then
    echo ">>> MISSING INPUT: $1"
    echo ">>> produced by: $2"
    echo ">>> skipping (exit 0 so dependent jobs report the same instead of"
    echo ">>> sitting in DependencyNeverSatisfied with no explanation)."
    exit 0
  fi
}

gate_failed() {     # $1 = why
  echo ">>> GATE FAILED: $1"
  echo ">>> exiting 0 so dependents report the same rather than stranding."
  exit 0
}
