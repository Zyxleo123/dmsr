# Shared preamble for the SUBFIND jobs. Sourced, never run.
#
# CONFIGURATION COMES IN AS POSITIONAL ARGUMENTS, NOT `sbatch --export`.
# On this cluster any `--export=<list>` makes sbatch set SLURM_GET_USER_ENV=1;
# slurmd then fails to rebuild the login environment on the compute node and the
# job is requeued and HELD, stranding every dependent job on Dependency.
# Arguments travel through the job record untouched. Each argument is either
# `VAR=value` or a file to source.
#
#   sbatch scripts/slurm/subfind_run_cpu.sbatch <envfile>
#
# TWO ENVIRONMENTS, deliberately.
#   `pjm`  the project's own env: torch, the cosmo_sr package, everything that
#          reads fields and Rockstar catalogs.
#   `g4`   the GADGET-4 toolchain: a compiler (there is no system gcc on these
#          nodes at all), OpenMPI, GSL, FFTW, HDF5, and h5py to read what
#          SUBFIND writes. It deliberately does NOT have torch or cosmo_sr.
# A job picks one with `use_env g4` or `use_env pjm`; nothing sources both,
# because the point of the second finder is that it shares no code with the
# first.
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

_CONDA_SH=/zfsauton/scratch/yixiz/miniconda3/etc/profile.d/conda.sh
# shellcheck disable=SC1090
source "$_CONDA_SH"

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"

export ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
export DMSR_REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
# Everything the SUBFIND cross-check produces -- snapshots, GADGET-4 output
# directories, parsed catalogues, comparisons. Never the repository, never $HOME:
# a 512^3 snapshot is 4.3 GB and there are several of them.
export SUBFIND_ROOT="${SUBFIND_ROOT:-$DMSR_REWARD_ROOT/subfind}"
export SUBFIND_LOGS="${SUBFIND_LOGS:-$SUBFIND_ROOT/logs}"
mkdir -p "$SUBFIND_ROOT" "$SUBFIND_LOGS"

export G4_DIR="${G4_DIR:-$PROJECT/external/gadget4}"
export G4_BIN="${G4_BIN:-$G4_DIR/Gadget4}"

use_env() {
    conda activate "$1"
    case "$1" in
        pjm) export PYTHONPATH="$PROJECT/src:$PROJECT/scripts/reward:${PYTHONPATH:-}" ;;
        g4)
            # The OpenMPI wrappers have no compiler of their own to fall back on
            # here: without these the build dies with "the wrapper compiler was
            # unable to find the specified compiler".
            export OMPI_CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
            export OMPI_CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
            # Single-node runs: no fabric to probe, and the probe is noisy.
            export OMPI_MCA_btl="^openib"
            export OMPI_MCA_plm_rsh_agent=""
            unset PYTHONPATH || true
            ;;
        *) echo "use_env: unknown environment '$1'" >&2; return 1 ;;
    esac
}

# Launch GADGET-4 under MPI.
#
# THE CPU NODES ARE HYPERTHREADED: 2 sockets x 22 cores x 2 threads = 88 CPUs
# but only 44 physical cores. Slurm allocates CPUs, i.e. hwthreads, so a job
# asking --ntasks=N --cpus-per-task=1 owns N hwthreads = N/2 cores, and
# `mpirun --bind-to core` then refuses to start:
#   "A request was made to bind that would require binding processes to more
#    cpus than are available in your allocation"  (jobs 57232, 57247)
# The fix is --cpus-per-task=2 in the job script, so each rank owns a whole
# physical core; this function assumes that and falls back to unbound ranks
# rather than dying, because binding is a performance choice and the catalogue
# it produces is identical either way.
run_gadget4() {   # run_gadget4 <nranks> <param> <restartflag> <snapnum>
    local n="$1" param="$2" flag="$3" snap="$4"
    echo "=== mpirun -np $n --bind-to core  (allocation: ${SLURM_CPUS_PER_TASK:-?} cpus/task)"
    if mpirun -np "$n" --bind-to core "$G4_BIN" "$param" "$flag" "$snap"; then
        return 0
    fi
    local rc=$?
    # Only a binding refusal is worth retrying: GADGET-4's own failures (a bad
    # parameter tag, a snapshot it cannot read) will fail again identically and
    # the retry would just double the log.
    echo "=== mpirun exited $rc; retrying UNBOUND (--bind-to none)." >&2
    echo "=== If this run is slow, the job needs --cpus-per-task=2." >&2
    mpirun -np "$n" --bind-to none "$G4_BIN" "$param" "$flag" "$snap"
}

require_input() {
    local f="$1" how="${2:-}"
    [[ -e "$f" ]] && return 0
    echo "missing input: $f" >&2
    [[ -n "$how" ]] && echo "  produce it with: $how" >&2
    exit 1
}

# A gate that failed is not a job that failed: exiting non-zero here would strand
# every dependent job in DependencyNeverSatisfied with no explanation. Print why
# and leave with 0 so the chain reports the reason instead of vanishing.
gate_stop() {
    echo "=== GATE STOP: $*"
    echo "=== downstream jobs will run and find no input; that is the signal."
    exit 0
}
