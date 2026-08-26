#!/bin/bash
# The SR2-vs-HR subhalo deficit diagnosis behind docs/sr2_subhalo_deficit.md.
#
# CPU only and cheap: it parses the two existing Rockstar catalogs and the
# cached tile weights, runs no halo finder, touches no owner array and no GPU.
# Everything it prints also lands in <box>_subhalo_deficit.json, so the doc's
# tables are redrawable instead of transcribed.
#
#   sbatch scripts/slurm/subhalo_deficit_cpu.sbatch <envfile>
#SBATCH --job-name=sub_defic
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-sub_defic-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8}"

for b in ${BOXES//,/ }; do
    F="$DMSR_REWARD_ROOT/lagrangian_host/$b/${b}_lagrangian_host.npz"
    H="$DMSR_REWARD_ROOT/halos_particles/${b}__hr__hr/${b}_hr_tilew.npz"
    S="$DMSR_REWARD_ROOT/halos_particles/${b}__base__base/${b}_base_tilew.npz"
    if [ ! -f "$F" ] || [ ! -f "$H" ] || [ ! -f "$S" ]; then
        echo "GATE: $b is missing one of:"
        echo "GATE:   $F (build_lagrangian_host.py)"
        echo "GATE:   $H / $S (rockstar_particles.py)"
        echo "GATE: skipping $b."
        continue
    fi
    python -u scripts/features/analyze_sr2_subhalo_deficit.py --boxes "$b"
done

echo "=== deficit tables: $DMSR_REWARD_ROOT/lagrangian_host/<box>/<box>_subhalo_deficit.json"
