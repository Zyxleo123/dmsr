# Shared preamble for the DIRECT SR2 fine-tuning jobs. Sourced, never run.
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
#   sbatch scripts/slurm/train_sr2_direct_gpu.sbatch RUN_NAME=direct_a RUNG=fine
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
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts/reward:${PYTHONPATH:-}"

export ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
# The residual line's root is READ (fitted reward model, frozen SR2 cache,
# oracle interventions are all inputs here) and never written.
export DMSR_REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
# Everything THIS line produces -- fields, catalogs, checkpoints, particle
# tables -- lands here. Never the repository, never $HOME.
export DMSR_SR2_DIRECT_ROOT="${DMSR_SR2_DIRECT_ROOT:-$DMSR_REWARD_ROOT/sr2_direct}"
LOGS="${LOGS:-$DMSR_SR2_DIRECT_ROOT/logs}"
mkdir -p "$DMSR_SR2_DIRECT_ROOT" "$LOGS"

: "${DIRECT_CFG:=configs/reward/sr2_direct_finetune.yaml}"
: "${REWARD_CFG:=configs/reward/reward.yaml}"
: "${RUN_NAME:=direct_a}"
: "${DEVICE:=}"
: "${BASE_SEED:=0}"

# numpy's default is one thread per core; several concurrent FFT jobs on an
# 88-core node then fight each other. Take it from the allocation.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

echo "=== $(date '+%F %T')  job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}"
echo "    project=$PROJECT  run=$RUN_NAME  cfg=$DIRECT_CFG"
echo "    sr2_direct_root=$DMSR_SR2_DIRECT_ROOT"
echo "    reward_root(read-only inputs)=$DMSR_REWARD_ROOT"
echo "    threads=$OMP_NUM_THREADS"
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

gate_failed() {     # $1 = why
  echo ">>> GATE FAILED: $1"
  echo ">>> exiting 0 so dependents report the same rather than stranding."
  exit 0
}

# The fitted reward model is an input to almost everything here.
REWARD_MODEL_JSON="$DMSR_REWARD_ROOT/reward_model/reward_model.json"
