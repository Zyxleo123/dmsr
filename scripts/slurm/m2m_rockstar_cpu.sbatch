#!/bin/bash
# Rockstar on an m2m candidate box (scripts/reward/catalog_summaries.py).
#
#   sbatch scripts/slurm/m2m_rockstar_cpu.sbatch <envfile>
#
# Unlike the member-gather arms, an m2m candidate is a WHOLE BOX, so there is no
# splice step and no seam: the catalog is of the model's own field everywhere.
# That also means the comparison is against this box's own HR and base catalogs
# (halos/<box>__hr__hr and <box>__base__base), which already exist for set9.
#
# CPU, 192G, 12h: the same ask flow_rockstar_catalog_cpu.sbatch uses for the
# identical call -- GADGET2 particle conversion plus Rockstar on 512^3 = 134M
# particles.
#SBATCH --job-name=m2mrs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-m2mrs-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_m2m_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_m2m_common.sh" "$@"
set -euo pipefail

: "${M2M_BOX_BOX:=set9}"
: "${M2M_RS_TAG:=$(basename "$M2M_OUT")_$(basename "${M2M_CKPT:-checkpoint}" .pt)}"
: "${M2M_RS_ROOT:=$DMSR_REWARD_ROOT/m2m/rockstar}"
: "${REWARD_CFG:=configs/reward/reward.yaml}"
# reward.yaml, NOT sr2_direct_finetune.yaml: catalog_summaries.py reads a `data`
# section (line 67) that only the reward configs carry, and the direct-finetune
# config has none -- it dies with KeyError: 'data' after the box is already made.
# This is the same REWARD_CFG default every other Rockstar job here uses.

if [[ -z "${M2M_BOX_OUT:-}" || ! -r "$M2M_BOX_OUT" ]]; then
    echo ">>> MISSING INPUT: ${M2M_BOX_OUT:-<unset>}"
    echo ">>> produced by: sbatch scripts/slurm/m2m_box_dump_gpu.sbatch"
    echo ">>> exiting 0 so nothing chained behind this strands."
    exit 0
fi

echo "=== rockstar: box=$M2M_BOX_BOX tag=$M2M_RS_TAG"
echo "    field=$M2M_BOX_OUT"
python -u scripts/reward/catalog_summaries.py \
    --config "$REWARD_CFG" \
    --box "$M2M_BOX_BOX" --source candidate --tag "$M2M_RS_TAG" \
    --field "$M2M_BOX_OUT" \
    --work "$M2M_RS_ROOT/halos/${M2M_BOX_BOX}__candidate__${M2M_RS_TAG}"

echo
echo "=== $(date '+%F %T')  done"
echo "    catalog under $M2M_RS_ROOT/halos/${M2M_BOX_BOX}__candidate__${M2M_RS_TAG}"
echo "    compare against halos/${M2M_BOX_BOX}__hr__hr and __base__base"
