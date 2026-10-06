#!/bin/bash
# Stage 2 (CPU, MPI): run GADGET-4's FOF + SUBFIND on the snapshot, then parse
# the group catalogue into a plain npz.
#
# Post-processing mode: `Gadget4 param.txt 3 <snapnum>` reads the snapshot,
# builds the tree, runs FOF (b = 0.2), decomposes each group with SUBFIND and
# writes fof_subhalo_tab_<snapnum>.hdf5. Nothing is integrated and the snapshot
# is not modified.
#
# Sizing: 512^3 = 134M particles on one 88-core / 768 GB cpu node. MaxMemSize in
# param.txt is PER RANK in MByte and is written by the snapshot stage from
# SF_MAX_MEM_MB, so the two must be kept consistent -- the submitter does that.
#
#   sbatch scripts/slurm/subfind_run_cpu.sbatch <envfile>
#SBATCH --job-name=sf_run
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=44
# 2 because the node is hyperthreaded: 88 CPUs are 44 physical cores, and a rank
# per core (not per hwthread) is both what --bind-to core needs and what a
# memory-latency-bound tree walk actually wants. 44 x 2 = the whole node.
#SBATCH --cpus-per-task=2
#SBATCH --mem=600G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/subfind/logs/slurm-sf_run-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_subfind_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_subfind_common.sh" "$@"
set -euo pipefail

: "${SF_NAME:?SF_NAME must name this run}"
: "${SF_OUT_DIR:=$SUBFIND_ROOT/$SF_NAME}"
: "${SF_SNAPNUM:=0}"
: "${SF_NRANKS:=${SLURM_NTASKS:-44}}"
: "${SF_MIN_P:=50}"
: "${SF_FORCE:=0}"

use_env g4

SNAPNUM3=$(printf "%03d" "$SF_SNAPNUM")
PARAM="$SF_OUT_DIR/param.txt"
SNAP="$SF_OUT_DIR/snapshot_${SNAPNUM3}"
CATALOG="$SF_OUT_DIR/fof_subhalo_tab_${SNAPNUM3}.hdf5"

require_input "$G4_BIN" "sbatch scripts/slurm/subfind_build_cpu.sbatch"
require_input "$PARAM" "sbatch scripts/slurm/subfind_snapshot_cpu.sbatch"
require_input "$SNAP"  "sbatch scripts/slurm/subfind_snapshot_cpu.sbatch"

if [ -f "$CATALOG" ] && [ "$SF_FORCE" != "1" ]; then
    echo "=== catalogue already present, skipping the finder: $CATALOG"
else
    echo "=== SUBFIND: $SF_NAME, $SF_NRANKS ranks, snapshot $SNAP"
    echo "    $(du -h "$SNAP" | cut -f1) snapshot, MaxMemSize $(grep -E '^MaxMemSize' "$PARAM")"
    cd "$G4_DIR"
    run_gadget4 "$SF_NRANKS" "$PARAM" 3 "$SF_SNAPNUM"
    cd "$PROJECT"
fi

if [ ! -f "$CATALOG" ]; then
    # Split catalogue, or the run wrote nothing at all.
    _alt=$(ls "$SF_OUT_DIR"/fof_subhalo_tab_${SNAPNUM3}* 2>/dev/null | head -1 || true)
    [ -n "$_alt" ] || gate_stop "GADGET-4 wrote no group catalogue for $SF_NAME; read the log above"
    CATALOG="$SF_OUT_DIR/fof_subhalo_tab_${SNAPNUM3}"
fi

echo
echo "=== parse the catalogue (h5py lives in the g4 env, so this stays here)"
python -u "$PROJECT/scripts/features/subfind_catalog.py" \
    --group-file "$CATALOG" \
    --min-particles "$SF_MIN_P" \
    --out "$SF_OUT_DIR/subfind_catalog"

echo
echo "=== done: $SF_OUT_DIR/subfind_catalog.npz"
