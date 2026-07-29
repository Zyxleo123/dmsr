#!/bin/bash
# Stage 4c (CPU, ~1 h): the two audits that decide which host-mass bins the
# occupation reward is allowed to use.
#
# Both read only cached HR chunk summaries, so they are pure arithmetic and no
# halo finder runs here. Submit after the HR catalog array job, and before
# fit_reward_model: if the sparse 1e14 host bin turns out to be unusable, the
# reward should be fitted without it rather than fitted and then apologised for.
#
#   sbatch scripts/slurm/audit_reward_support_cpu.sbatch
#   sbatch scripts/slurm/audit_reward_support_cpu.sbatch TARGET_HOSTS=30
#
#SBATCH --job-name=rw_audit_support
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_audit_support-%j.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${TARGET_HOSTS:=20}"        # effective hosts per reward estimate
: "${MAX_EMPTY_FRAC:=0.05}"
: "${N_DRAWS:=4000}"
: "${DROP_HOST_BINS:=4}"       # the sparse 1e14 Msun/h bin
: "${OUT_DIR:=runs/reward}"

require_input "$DMSR_REWARD_ROOT/catalog_cache" \
    "sbatch --array=0-15 scripts/slurm/hr_catalog_summaries_cpu.sbatch"

mkdir -p "$OUT_DIR"

python -u scripts/reward/audit_host_counts.py \
    --config "$REWARD_CFG" \
    --n-draws "$N_DRAWS" \
    --target-hosts "$TARGET_HOSTS" \
    --max-empty-frac "$MAX_EMPTY_FRAC" \
    --out "$OUT_DIR/host_count_audit.json"

python -u scripts/reward/audit_reward_covariance.py \
    --config "$REWARD_CFG" \
    --drop-host-bins "$DROP_HOST_BINS" \
    --base-summaries "$DMSR_REWARD_ROOT/catalog_cache" \
    --out "$OUT_DIR/covariance_audit.json"

echo "=== decide from these two files BEFORE fitting the reward:"
echo "===   $OUT_DIR/host_count_audit.json     -> usable_bins / evaluation_only_bins"
echo "===   $OUT_DIR/covariance_audit.json     -> recommendation"
echo "=== if the recommendation is drop_sparse_bin_to_evaluation_only, set"
echo "===   reward.occupation.include_sparse_in_reward: false  (already the default)"
echo "===   and refit with  --set catalog.host_mass_bins.n_bins=4"
