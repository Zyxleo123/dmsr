#!/bin/bash
# Decompose the member-gather high-k guard on the ALREADY-EXPORTED held-out
# tiles. CPU only and reads nothing but saved artefacts -- no model, no GPU.
#
#   sbatch scripts/slurm/highk_spectrum_cpu.sbatch <envfile>
#
# 32G: three (32, 6, 64, 64, 64) float32 arrays (201 MB each) per arm, loaded
# once and freed before the next arm, plus complex128 FFT scratch of one host's
# four tiles. The npz members re-inflate on every access, which is why the
# script reads them once rather than inside its loop.
#SBATCH --job-name=hk_spec
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-hk_spec-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${HK_BOX:=set9}"
: "${HK_ARMS:=all_blocks_self all_blocks_nocentre all_blocks_full all_blocks_radial}"
: "${HK_BINS:=20}"
: "${HK_OUT:?HK_OUT must be the output stem}"

# The gate's export is the only input. Without it there is nothing to decompose,
# and a job that silently analysed zero arms would read as a clean pass.
_found=0
for _a in $HK_ARMS; do
    [[ -f "$DMSR_REWARD_ROOT/member_gather/$_a/holdout_$HK_BOX/tiles.npz" ]] \
        && _found=$((_found + 1))
done
if [[ "$_found" -eq 0 ]]; then
    echo "GATE: no arm in '$HK_ARMS' has holdout_$HK_BOX/tiles.npz."
    echo "      Run scripts/slurm/submit_gather_holdout_rockstar.sh first."
    exit 0
fi
echo "=== $_found of $(echo $HK_ARMS | wc -w) arms have an export"

python -u scripts/features/highk_spectrum_diagnostic.py \
    --reward-root "$DMSR_REWARD_ROOT" \
    --box "$HK_BOX" \
    --arms $HK_ARMS \
    --bins "$HK_BINS" \
    --out "$HK_OUT"

echo "=== result: ${HK_OUT}.json"
