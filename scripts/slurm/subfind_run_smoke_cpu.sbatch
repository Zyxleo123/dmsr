#!/bin/bash
# The smoke run: SUBFIND on a small synthetic snapshot with planted clumps, then
# a verdict on whether it found them.
#
# Same chain as subfind_run_cpu.sbatch -- same binary, same parameter file
# layout, same parser -- at 1/500th the size, so a broken header or a wrong tag
# surfaces in minutes instead of after a node-hour. The verdict job is what
# makes it a test rather than a rehearsal: the clumps' centres and particle
# counts are known, so "ran fine, found nothing" cannot pass.
#
#   sbatch scripts/slurm/subfind_run_smoke_cpu.sbatch <envfile>
#SBATCH --job-name=sf_smoke
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
# See run_gadget4 in _subfind_common.sh: the node is hyperthreaded, so a rank
# needs 2 CPUs to own one physical core and satisfy --bind-to core.
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/subfind/logs/slurm-sf_smoke-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_subfind_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_subfind_common.sh" "$@"
set -euo pipefail

: "${SF_NAME:=smoke}"
: "${SF_OUT_DIR:=$SUBFIND_ROOT/$SF_NAME}"
: "${SF_SNAPNUM:=0}"
: "${SF_NRANKS:=4}"
: "${SF_MIN_P:=20}"

use_env g4

SNAPNUM3=$(printf "%03d" "$SF_SNAPNUM")
PARAM="$SF_OUT_DIR/param.txt"
SNAP="$SF_OUT_DIR/snapshot_${SNAPNUM3}"

require_input "$G4_BIN" "sbatch scripts/slurm/subfind_build_cpu.sbatch"
require_input "$SNAP" "sbatch scripts/slurm/subfind_snapshot_cpu.sbatch SF_SMOKE=64"

echo "=== SUBFIND smoke: $SF_NRANKS ranks on $SNAP"
cd "$G4_DIR"
run_gadget4 "$SF_NRANKS" "$PARAM" 3 "$SF_SNAPNUM"
cd "$PROJECT"

CATALOG="$SF_OUT_DIR/fof_subhalo_tab_${SNAPNUM3}.hdf5"
[ -f "$CATALOG" ] || CATALOG="$SF_OUT_DIR/fof_subhalo_tab_${SNAPNUM3}"

echo
echo "=== parse"
python -u "$PROJECT/scripts/features/subfind_catalog.py" \
    --group-file "$CATALOG" --min-particles "$SF_MIN_P" \
    --out "$SF_OUT_DIR/subfind_catalog"

echo
echo "=== verdict: were the planted clumps found?"
python -u "$PROJECT/scripts/features/subfind_smoke_verdict.py" \
    --run-dir "$SF_OUT_DIR"
