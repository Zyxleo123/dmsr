#!/bin/bash
# Build GADGET-4 with FOF + SUBFIND (CPU). Minutes.
#
# The executable this produces never integrates anything: it is started in
# post-processing mode (`Gadget4 param.txt 3 <snapnum>`), reads a snapshot, runs
# FOF and SUBFIND, writes the group catalogue and exits. See
# external/gadget4/Config.sh for why each compile option is set.
#
# It is a batch job rather than a login-node `make` for the usual reason: real
# compute on the login node gets SIGKILLed, and a 300-file C++ build is real
# compute. It also guarantees the binary is built on the SAME hardware it runs
# on, which matters because Makefile.comp.gcc compiles with -march=native.
#
#   sbatch scripts/slurm/subfind_build_cpu.sbatch
#   sbatch scripts/slurm/subfind_build_cpu.sbatch G4_CLEAN=0     # incremental
#SBATCH --job-name=sf_build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/subfind/logs/slurm-sf_build-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_subfind_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_subfind_common.sh" "$@"
set -euo pipefail

: "${G4_CLEAN:=1}"

use_env g4

require_input "$G4_DIR/Config.sh" "it is committed at external/gadget4/Config.sh"
require_input "$G4_DIR/Makefile.systype" "printf 'SYSTYPE=\"autonlab\"\\n' > $G4_DIR/Makefile.systype"

# The systype decides where the dependencies are looked for, and getting it
# wrong fails only at the LINK, after a full compile. "Generic-gcc" pulls in
# Makefile.gen.libs, which points at hand-built deps under an unset $(LIB_DIR)
# and emits -I/hdf5/build/include; "autonlab" pulls in Makefile.path.default,
# which points at $CONDA_PREFIX. Checked here so the failure is one line at the
# top of the log rather than 300 compiles later (job 57211).
if ! grep -q 'SYSTYPE="autonlab"' "$G4_DIR/Makefile.systype"; then
    echo "Makefile.systype is not autonlab:" >&2
    cat "$G4_DIR/Makefile.systype" >&2
    echo "  fix: printf 'SYSTYPE=\"autonlab\"\\n' > $G4_DIR/Makefile.systype" >&2
    exit 1
fi
for _lib in libz.so libhdf5.so libgsl.so libfftw3.so libfftw3f.so; do
    [ -e "$CONDA_PREFIX/lib/$_lib" ] || {
        echo "missing $CONDA_PREFIX/lib/$_lib -- the LINK needs the .so symlink," >&2
        echo "  not just the versioned runtime library." >&2
        echo "  fix: conda install -n g4 -c conda-forge zlib hdf5 gsl fftw" >&2
        exit 1
    }
done

echo "=== toolchain"
echo "  CONDA_PREFIX = $CONDA_PREFIX"
echo "  OMPI_CXX     = $OMPI_CXX"
mpicxx --version | head -1
echo "  FOF/SUBFIND options:"
grep -E "^(FOF|SUBFIND)" "$G4_DIR/Config.sh" | sed 's/^/    /'

cd "$G4_DIR"
if [ "$G4_CLEAN" = "1" ]; then
    echo "=== make clean"
    make clean >/dev/null 2>&1 || true
fi

echo "=== make -j ${SLURM_CPUS_PER_TASK:-8}"
make -j "${SLURM_CPUS_PER_TASK:-8}"

[ -x "$G4_BIN" ] || { echo "build finished but $G4_BIN is missing" >&2; exit 1; }

echo
echo "=== built $G4_BIN"
ls -l "$G4_BIN"
# The compile-time options are baked into the binary; print them so a catalogue
# can always be traced to the exact finder that produced it.
echo "=== compiled-in options"
strings "$G4_BIN" | grep -E "^(SUBFIND|FOF|PERIODIC|PMGRID|NTYPES|GADGET2_HEADER)" \
    | sort -u | sed 's/^/    /' || true

echo
echo "next: smoke test the whole chain before spending a node on 512^3 --"
echo "  bash scripts/slurm/submit_subfind.sh smoke"
