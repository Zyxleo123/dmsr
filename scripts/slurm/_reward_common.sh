# Shared preamble for the reward-guided residual diffusion jobs. Sourced, never run.
#
# CONFIGURATION COMES IN AS POSITIONAL ARGUMENTS, NOT `sbatch --export`.
# On this cluster any `--export=<list>` makes sbatch set SLURM_GET_USER_ENV=1,
# slurmd then fails to rebuild the login environment on the compute node, and
# the job is requeued and held with `(user env retrieval failed requeued held)`.
# Arguments pass through the job record untouched, so they are the safe channel.
# Each argument is either `VAR=value` or a file to source.
#
#   sbatch scripts/slurm/train_residual_prior.sbatch STEPS=80000 SEED=1
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

# Every bulky artifact lands here. Home has a tight quota and Rockstar's
# per-catalog dumps filled it once already.
export ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
export DMSR_REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
LOGS="${LOGS:-$DMSR_REWARD_ROOT/logs}"
mkdir -p "$DMSR_REWARD_ROOT" "$LOGS"

: "${DATA:=/zfsauton/scratch/yixiz/DMSR/paired_catnorm}"
: "${REWARD_CFG:=configs/reward/reward.yaml}"
: "${MODEL_CFG:=configs/reward/residual_prior.yaml}"
: "${DISTILL_CFG:=configs/reward/distill_round0.yaml}"
: "${SR2_CONFIG:=}"          # frozen SR2 run config.yaml (from freeze.yaml if empty)
: "${SR2_CKPT:=}"            # frozen SR2 checkpoint    (from freeze.yaml if empty)
: "${DEVICE:=cuda}"
: "${SEED:=0}"
: "${BASE_SEED:=0}"

# BLAS threads: numpy's default is one thread per core, which on an 88-core CPU
# node makes several concurrent FFT jobs fight each other and run slower than
# one. Set it explicitly from the allocation.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

echo "=== $(date '+%F %T')  job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}"
echo "    project=$PROJECT"
echo "    reward_root=$DMSR_REWARD_ROOT  data=$DATA"
echo "    threads=$OMP_NUM_THREADS  device=$DEVICE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

require_input() {   # $1 = path, $2 = which stage produces it
  if [ ! -e "$1" ]; then
    echo ">>> MISSING INPUT: $1"
    echo ">>> produced by: $2"
    echo ">>> skipping (exit 0 so dependent jobs report the same instead of"
    echo ">>> sitting in DependencyNeverSatisfied with no explanation)."
    exit 0
  fi
}
