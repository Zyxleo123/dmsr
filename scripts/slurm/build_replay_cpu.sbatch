#!/bin/bash
# Stage 7 (CPU, minutes): harvest the scored oracle into an elite replay buffer.
#
# Pure arithmetic on cached catalog summaries -- no halo finding, no GPU. The
# only slow part is sha256 over the referenced residual fields, which is what
# makes the manifest verifiable; pass NO_CHECKSUM=1 to skip it.
#
#   sbatch scripts/slurm/build_replay_cpu.sbatch RUN_NAME=prior_k8 ROUND=round_000
#
#SBATCH --job-name=rw_replay
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_replay-%j.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${RUN_NAME:=prior_k8}"
: "${ROUND:=round_000}"
: "${ELITE_QUANTILE:=0.2}"
: "${TAU:=1.0}"
: "${A_MAX:=5.0}"
: "${W_MAX:=10.0}"
: "${NO_CHECKSUM:=0}"

require_input "$DMSR_REWARD_ROOT/oracle/$RUN_NAME/oracle_manifest.jsonl" \
    "scripts/slurm/catalog_reward_oracle_cpu.sbatch AGGREGATE=1"

EXTRA=()
[ "$NO_CHECKSUM" = "1" ] && EXTRA+=(--no-checksum)

python -u scripts/reward/build_replay.py \
    --config "$REWARD_CFG" \
    --run-name "$RUN_NAME" --round "$ROUND" \
    --elite-quantile "$ELITE_QUANTILE" \
    --tau "$TAU" --a-max "$A_MAX" --w-max "$W_MAX" \
    "${EXTRA[@]}"

echo "=== replay -> $DMSR_REWARD_ROOT/replay/$ROUND/replay.jsonl"
