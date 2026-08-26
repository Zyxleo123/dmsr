#!/bin/bash
# Which statistic separates a bound halo from a field that only matches its
# moments? docs/sr2_gather_finetune.md sections 5-7.
#
# CPU only, and it runs no halo finder. It reads the spliced fields by memmap
# (only the ~28k supervised particles are touched, not the 3.2 GB box), one
# chunked pass over the HR owner array, and writes a small JSON. The whole point
# is that the answer is available without the 12-minute Rockstar the gate needs.
#
#   sbatch scripts/slurm/bound_discriminator_cpu.sbatch <envfile>
#SBATCH --job-name=bound_disc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-bound_disc-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BD_BOX:=set8}"
: "${BD_RUN_DIR:=$DMSR_REWARD_ROOT/host_gather/set8_h271800_fine_anchored}"
: "${BD_HOST_ID:=271800}"
: "${BD_SEED:=0}"
: "${BD_TAGS:=}"
: "${BD_REF_TAG:=}"
: "${BD_SOFTENING:=0.01}"

# Gates. A missing input is "nothing to do here", not a failure: exiting
# non-zero would strand any dependent job on DependencyNeverSatisfied with no
# explanation. The script gates on the fields itself and says which are absent.
if [ ! -f "$BD_RUN_DIR/subhalos.json" ]; then
    echo "GATE: no subhalos.json under $BD_RUN_DIR"
    echo "GATE: point BD_RUN_DIR at a finished host_gather run; skipping."
    exit 0
fi
OWN="$DMSR_REWARD_ROOT/halos_particles/${BD_BOX}__hr__hr/${BD_BOX}_hr_owner.npy"
if [ ! -f "$OWN" ]; then
    echo "GATE: no HR owner array at $OWN"
    echo "GATE: it comes from scripts/reward/rockstar_particles.py "\
         "--write-assignment; skipping."
    exit 0
fi

ARGS=(--box "$BD_BOX" --run-dir "$BD_RUN_DIR" --host-id "$BD_HOST_ID"
      --seed "$BD_SEED" --softening-mpc-h "$BD_SOFTENING")
[ -n "$BD_TAGS" ]    && ARGS+=(--tags "$BD_TAGS")
[ -n "$BD_REF_TAG" ] && ARGS+=(--reference-tag "$BD_REF_TAG")

python -u scripts/features/measure_bound_discriminator.py "${ARGS[@]}"

echo "=== result: $DMSR_REWARD_ROOT/bound_discriminator/${BD_BOX}__$(basename "$BD_RUN_DIR").json"
echo "=== the verdict is the VERDICT line above and the 'verdict' key of that json"
