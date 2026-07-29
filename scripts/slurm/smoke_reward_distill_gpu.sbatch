#!/bin/bash
# Stage 8 smoke test (GPU, minutes): a few steps of the three-term distillation.
#
# Proves the composite step runs -- paired loss, weighted elite loss, teacher
# reference loss, and the abort checks -- before committing a day of GPU. With no
# REPLAY given it uses synthetic elite batches, so it can run before the oracle
# has finished; pass REPLAY=... to exercise the real manifest and its checksums.
#
#   sbatch scripts/slurm/smoke_reward_distill_gpu.sbatch
#   sbatch scripts/slurm/smoke_reward_distill_gpu.sbatch REPLAY=$ZFS/DMSR/dmsr_reward/replay/round_000/replay.jsonl
#
#SBATCH --job-name=rw_smoke_distill
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_smoke_distill-%j.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${PRIOR_CKPT:=$DMSR_REWARD_ROOT/checkpoints/residual_prior/ckpt_best.pt}"
: "${REPLAY:=}"
: "${RUN_DIR:=$DMSR_REWARD_ROOT/checkpoints/smoke_distill}"

INIT=()
[[ -r "$PRIOR_CKPT" ]] && INIT=(--init-from "$PRIOR_CKPT" --teacher "$PRIOR_CKPT")

python -u scripts/reward/train_reward_distill.py \
    --config "$DISTILL_CFG" \
    --out "$RUN_DIR" \
    --smoke \
    "${INIT[@]}" \
    ${REPLAY:+--replay "$REPLAY"} \
    --set train.device="$DEVICE"

echo "=== smoke distill -> $RUN_DIR"
