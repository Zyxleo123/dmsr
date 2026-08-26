# Shared preamble for the Lagrangian-only-critic sweep. Sourced, never run.
#
# CONFIGURATION COMES IN AS POSITIONAL ARGUMENTS, NOT `sbatch --export`.
# On this cluster any `--export=<list>` (`--export=ALL,K=v`, `--export=K=v` and
# `--export=NONE` alike) makes sbatch set SLURM_GET_USER_ENV=1; slurmd then fails
# to rebuild the login environment on the compute node and the job is requeued and
# HELD with "(user env retrieval failed requeued held)". Arguments travel through
# the job record untouched, so they are the safe channel. Each argument is either
# `VAR=value` or a file to source.
#
# Self-sufficiency checked with `env -i`: nothing below needs a login shell.
set -euo pipefail

for _arg in "$@"; do
    case "$_arg" in
        *=*)  export "$_arg" ;;
        *)    [[ -r "$_arg" ]] || { echo "env file not readable: $_arg" >&2; exit 1; }
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
export PYTHONPATH="$PROJECT/src:${PYTHONPATH:-}"

export ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
LOGS="${LOGS:-$ZFS/DMSR/cosmo_sr_runs/dmsr/lagonly/logs}"
mkdir -p "$LOGS"

: "${ARM:=t13_lagonly_l003}"
: "${WANDB_MODE:=online}"
export WANDB_MODE

# numpy's default is one thread per core; concurrent arms on one node otherwise
# fight each other. Take it from the allocation.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

echo "=== $(date '+%F %T')  job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}"
echo "    project=$PROJECT  arm=$ARM  threads=$OMP_NUM_THREADS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

require_input() {   # $1 = path, $2 = which stage produces it
  if [ ! -e "$1" ]; then
    echo ">>> MISSING INPUT: $1"
    echo ">>> produced by: $2"
    echo ">>> gate failed -- exiting 0 so dependents start and report their own skip."
    exit 0
  fi
}
