#!/bin/bash
# Lay our paired boxes out for map2map's loader (scripts/features/m2m_split_data.py).
#
#   sbatch scripts/slurm/m2m_split_data_cpu.sbatch <envfile>
#
# CPU, not GPU: this is a pure copy -- 13 boxes x 3 GB of HR read and rewritten
# as two 3-channel files each. Asking for a GPU to split .npy files would just
# queue behind real work.
#
# DISK. ~3 GB per HR box x 13 = ~39 GB, plus a negligible amount for LR and the
# style files. /zfsauton/scratch was at 86% (5.4 TB free) when this was written.
# The per-user quota is the recurring failure mode on this cluster, so check
# `df` before a rerun with more boxes.
#
# MEM. The split is slab-wise through open_memmap, so resident memory is one
# 64-plane slab (~50 MB), not a 3 GB field. 16G is generous.
#SBATCH --job-name=m2msplit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-m2msplit-%j.out

# `set -euo pipefail` HERE as well as in the preamble: a failed source otherwise
# leaves the job on the system python and EXITING 0, releasing every afterok
# dependent behind a stage that did no work.
set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_m2m_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_m2m_common.sh" "$@"
set -euo pipefail

for _b in $M2M_TRAIN_BOXES $M2M_VAL_BOXES; do
    for _s in lr hr; do
        _f="$M2M_DATA_ROOT/$_s/$_b.npy"
        if [[ ! -e "$_f" ]]; then
            echo ">>> MISSING INPUT: $_f"
            echo ">>> produced by: scripts/preprocess_paired.py"
            echo ">>> exiting 0 so dependents report the same rather than stranding."
            exit 0
        fi
    done
done
unset _b _s _f

echo
echo "=== splitting into $M2M_LAYOUT"
srun --unbuffered python scripts/features/m2m_split_data.py \
    --data-root "$M2M_DATA_ROOT" \
    --out "$M2M_LAYOUT" \
    --train-boxes "$M2M_TRAIN_BOXES" \
    --val-boxes "$M2M_VAL_BOXES" \
    --scale-factor-a "$M2M_STYLE_A" \
    ${M2M_SPLIT_OVERWRITE:+--overwrite}
_rc=$?

echo
echo "=== $(date '+%F %T')  split finished rc=$_rc"
echo "    manifest: $M2M_LAYOUT/manifest.json  (carries the exact globs the"
echo "    training job will use, so the two cannot drift apart)"
du -sh "$M2M_LAYOUT" 2>/dev/null || true
exit "$_rc"
