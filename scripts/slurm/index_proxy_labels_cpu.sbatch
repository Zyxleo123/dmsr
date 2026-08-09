#!/bin/bash
# THE indexing job: one per submission, depending on every labelling job.
#
# It is deliberately not part of the labelling jobs. 120 of those finishing in
# overlapping windows would each rewrite one rows.jsonl in place, and a trainer
# ordered after the last-SUBMITTED label job is not ordered after the SLOWEST
# one -- so it could fit on a table missing whole boxes and report a validation
# number for a dataset that no longer exists.
#
# Depend on this with afterany, not afterok: this job's whole purpose is to
# report what is missing, and it can only do that if it runs when something
# upstream failed.
#
#   sbatch scripts/slurm/index_proxy_labels_cpu.sbatch <envfile>
#SBATCH --job-name=sd_index
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_index-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

python -u scripts/reward/collect_catalog_proxy_data.py \
    --config "$DIRECT_CFG" --run-name "$RUN_NAME" --stage index

MARK="$DMSR_SR2_DIRECT_ROOT/proxy_data/labels_complete.json"
if [ -r "$MARK" ]; then
    echo "=== labels complete: $MARK"
    echo "=== the proxy benchmark may now be submitted:"
    echo "===   bash scripts/slurm/submit_proxy_benchmark.sh all"
else
    echo "=== labels INCOMPLETE. See the report above and"
    echo "===   $DMSR_SR2_DIRECT_ROOT/proxy_data/index_report.json"
    echo "=== Re-running the label submitter resumes: every candidate that is"
    echo "=== already generated and input-identical is skipped."
fi
