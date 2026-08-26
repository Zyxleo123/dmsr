#!/bin/bash
# Per-host, per-tile HR/SR2 subhalo counts for the viewer's panel 7.
#
# CPU only, and it re-runs no halo finder: it reads the cached LR features, the
# two `<box>_<tag>_owner.npy` arrays (512 MB each, one source at a time) and the
# HR/SR2 catalogs, then writes a small JSON. Memory is sized for one owner array
# plus the per-host HR-child blocks, not for both at once.
#
#   sbatch scripts/slurm/host_subhalo_tiles_cpu.sbatch <envfile>
#SBATCH --job-name=lag_subs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-lag_subs-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8}"
: "${SUBTILES_N_HOSTS:=0}"    # 0 = every LR host

# Gates, per box. A missing input is a "nothing to do here" for THIS box, not a
# failure: exiting non-zero would strand the render job on DependencyNeverSat.
for b in ${BOXES//,/ }; do
    F="$DMSR_REWARD_ROOT/lagrangian_host/$b/${b}_lagrangian_host.npz"
    OH="$DMSR_REWARD_ROOT/halos_particles/${b}__hr__hr/${b}_hr_owner.npy"
    OB="$DMSR_REWARD_ROOT/halos_particles/${b}__base__base/${b}_base_owner.npy"
    if [ ! -f "$F" ]; then
        echo "GATE: no cached LR features at $F"
        echo "GATE: run scripts/slurm/lagrangian_host_build_cpu.sbatch first; "\
             "skipping $b."
        continue
    fi
    if [ ! -f "$OH" ] && [ ! -f "$OB" ]; then
        echo "GATE: no owner arrays for $b (looked for $OH and $OB)."
        echo "GATE: they come from scripts/reward/rockstar_particles.py "\
             "--write-assignment; panel 7 stays disabled for $b."
        continue
    fi
    [ -f "$OH" ] || echo "NOTE: no HR owner array for $b; only the SR2 side."
    [ -f "$OB" ] || echo "NOTE: no SR2 owner array for $b; only the HR side."

    python -u scripts/features/collect_host_subhalo_tiles.py \
        --boxes "$b" --n-hosts "$SUBTILES_N_HOSTS"
done

echo "=== per-host subhalo tiles: $DMSR_REWARD_ROOT/lagrangian_host/<box>/<box>_host_subhalo_tiles.json"
