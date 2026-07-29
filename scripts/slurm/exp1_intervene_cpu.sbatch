#!/bin/bash
# Experiment 1 (CPU array): one intervention box per task.
#
# The grid is (box) x (kind) x (mode) x (alpha), flattened to the array index,
# with the degenerate cells removed:
#   * alpha = 0 is the frozen SR2 box, identical for every kind and mode, so it
#     is scored ONCE per box rather than 9 times;
#   * `control` and `exact_particles` only run at alpha = 1 -- the control is a
#     yes/no test at full strength, and "replace exactly the HR particles" has
#     no partial version.
# That takes the grid from 4x3x3 = 36 boxes to 16 per box pair. The enumeration
# lives in one python block so the submitter and the job cannot disagree about
# what index N means.
#
#   sbatch --array=0-15 scripts/slurm/exp1_intervene_cpu.sbatch
#   sbatch scripts/slurm/exp1_intervene_cpu.sbatch PRINT_GRID=1     # dry run
#
#SBATCH --job-name=exp1_interv
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-exp1_interv-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8 set9}"
: "${ALPHAS:=0 0.25 0.5 1.0}"
: "${MODES:=disp vel both}"
: "${KINDS:=targeted control exact_particles}"
: "${PRINT_GRID:=0}"
: "${SKIP_FIELDS:=0}"

read -r -d '' GRID_PY <<'PY' || true
import os, sys
boxes = os.environ["BOXES"].replace(",", " ").split()
alphas = [float(a) for a in os.environ["ALPHAS"].replace(",", " ").split()]
modes = os.environ["MODES"].replace(",", " ").split()
kinds = os.environ["KINDS"].replace(",", " ").split()
cells = []
for b in boxes:
    # alpha = 0 is the frozen box: one anchor per box, not one per (kind, mode).
    if 0.0 in alphas:
        cells.append((b, "targeted", modes[0], 0.0))
    for k in kinds:
        for m in modes:
            for a in alphas:
                if a == 0.0:
                    continue
                # A control and an exact-particle replacement are full-strength
                # yes/no tests; partial versions answer nothing extra.
                if k != "targeted" and a != 1.0:
                    continue
                cells.append((b, k, m, a))
if sys.argv[1] == "count":
    print(len(cells))
else:
    i = int(sys.argv[1])
    if i >= len(cells):
        print("OUT_OF_RANGE")
    else:
        print(" ".join(str(x) for x in cells[i]))
PY

export BOXES ALPHAS MODES KINDS

if [ "$PRINT_GRID" = "1" ]; then
    N=$(python -c "$GRID_PY" count)
    echo "=== $N cells; submit with --array=0-$((N - 1))"
    for i in $(seq 0 $((N - 1))); do
        printf '  %3d  %s\n' "$i" "$(python -c "$GRID_PY" "$i")"
    done
    exit 0
fi

require_input "$DMSR_REWARD_ROOT/oracle_hr" "scripts/slurm/exp1_select_targets_cpu.sbatch"
require_input "external/rockstar/rockstar" "make -C external/rockstar"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
CELL=$(python -c "$GRID_PY" "$IDX")
if [ "$CELL" = "OUT_OF_RANGE" ]; then
    echo ">>> array index $IDX is beyond the grid; nothing to do."
    exit 0
fi
read -r BOX KIND MODE ALPHA <<< "$CELL"
echo "=== cell $IDX: box=$BOX kind=$KIND mode=$MODE alpha=$ALPHA"

# alpha != 0 needs the Lagrangian member sites; a missing extraction is a gate
# failure, not a crash, so dependents still get a readable log.
if [ "$ALPHA" != "0" ] && [ "$ALPHA" != "0.0" ]; then
    MEM="$DMSR_REWARD_ROOT/halos_particles/${BOX}__hr__hr/${BOX}_hr_members.npz"
    if [ ! -r "$MEM" ]; then
        echo ">>> GATE FAILED: no member ids at $MEM"
        echo ">>> produced by: exp0_rockstar_particles_cpu.sbatch with EXTRACT=1,"
        echo ">>> which requires exp1_select_targets_cpu.sbatch to have run first."
        echo ">>> exiting 0 so dependents report the same rather than stranding."
        exit 0
    fi
fi

ARGS=(--config "$REWARD_CFG" --box "$BOX" --kind "$KIND" --mode "$MODE"
      --alpha "$ALPHA" --base-seed "$BASE_SEED")
[ "$SKIP_FIELDS" = "1" ] && ARGS+=(--skip-fields)

python -u scripts/reward/oracle_intervene.py "${ARGS[@]}"
echo "=== cell $IDX done"
