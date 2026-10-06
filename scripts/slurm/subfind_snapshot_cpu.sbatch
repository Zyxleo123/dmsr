#!/bin/bash
# Stage 1 (CPU): field -> GADGET-2 snapshot + GADGET-4 parameter file.
#
# The snapshot is built by the same writer that feeds Rockstar
# (cosmo_sr.eval.particles.write_gadget2_snapshot), so both finders see exactly
# the same particles and any difference in their catalogues is a difference
# between the algorithms, not between two nearly-identical fields.
#
# 64G: field_to_particles materialises the displacement, velocity and position
# lattices in float64 at 512^3 (3.2 GB each) before casting down.
#
#   sbatch scripts/slurm/subfind_snapshot_cpu.sbatch <envfile>
#SBATCH --job-name=sf_snap
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/subfind/logs/slurm-sf_snap-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_subfind_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_subfind_common.sh" "$@"
set -euo pipefail

: "${SF_BOX:=set9}"
: "${SF_SOURCE:=hr}"
: "${SF_TAG:=}"
: "${SF_SEED:=0}"
: "${SF_NAME:?SF_NAME must name this run, e.g. set9__hr or set9__mgho_all_blocks_self_set9}"
: "${SF_OUT_DIR:=$SUBFIND_ROOT/$SF_NAME}"
: "${SF_MAX_MEM_MB:=13000}"
: "${SF_SMOKE:=0}"
: "${SF_FIELD:=}"

use_env pjm

echo "=== snapshot: name=$SF_NAME source=$SF_SOURCE tag=${SF_TAG:-none} -> $SF_OUT_DIR"
python -u scripts/features/subfind_snapshot.py \
    --out-dir "$SF_OUT_DIR" \
    --box "$SF_BOX" \
    --source "$SF_SOURCE" \
    --seed "$SF_SEED" \
    --max-mem-mb "$SF_MAX_MEM_MB" \
    ${SF_TAG:+--tag "$SF_TAG"} \
    ${SF_FIELD:+--field "$SF_FIELD"} \
    $([ "$SF_SMOKE" != "0" ] && echo --smoke "$SF_SMOKE")

echo "=== done: $SF_OUT_DIR"
ls -l "$SF_OUT_DIR"
