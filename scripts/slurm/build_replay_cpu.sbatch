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

# RUN_NAME is the oracle run the ELITES come from, and it must be on TRAIN
# boxes -- build_replay.py refuses held-out ones, and rightly so.
# GATE_B_RUN is the run whose gate_b.json carries the VERDICT, which is on the
# validation split. They are two different runs; defaulting them to the same
# name guaranteed either a leak or a hard failure with the distill job stranded
# behind it. See the gate_c stage of scripts/slurm/submit_oracle.sh.
: "${RUN_NAME:=prior_train_k8}"
: "${GATE_B_RUN:=prior_k8}"
: "${ROUND:=round_000}"
: "${ELITE_QUANTILE:=0.2}"
: "${TAU:=1.0}"
: "${A_MAX:=5.0}"
: "${W_MAX:=10.0}"
: "${NO_CHECKSUM:=0}"

require_input "$DMSR_REWARD_ROOT/oracle/$RUN_NAME/oracle_manifest.jsonl" \
    "scripts/slurm/catalog_reward_oracle_cpu.sbatch AGGREGATE=1"

# Gate C runs only if Gate B passed: docs/reward_residual_diffusion.md sec.6
# is explicit that distillation is gated on Gate B's OCCUPATION verdict, not
# merely on the oracle manifest existing. Skip (exit 0) on any other decision
# so a dependent train_reward_distill_round0.sbatch reports the same reason
# via its own require_input on the (unwritten) replay file, instead of
# training on a buffer that was never supposed to exist.
GATE_B_JSON="$DMSR_REWARD_ROOT/oracle/$GATE_B_RUN/gate_b.json"
require_input "$GATE_B_JSON" "scripts/slurm/catalog_reward_oracle_cpu.sbatch AGGREGATE=1"
GATE_B_DECISION=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['gate']['decision'])" "$GATE_B_JSON")
if [ "$GATE_B_DECISION" != "support_present_occupation" ]; then
    echo ">>> Gate B decision for GATE_B_RUN=$GATE_B_RUN is '$GATE_B_DECISION', not"
    echo ">>> 'support_present_occupation'. Gate C (replay + distillation) does"
    echo ">>> not run on this verdict -- see sec.6 of"
    echo ">>> docs/reward_residual_diffusion.md and $GATE_B_JSON for diagnosis."
    echo ">>> skipping (exit 0) rather than blocking downstream with no reason."
    exit 0
fi
echo "=== Gate B decision: $GATE_B_DECISION -- proceeding with replay"
echo "=== verdict from '$GATE_B_RUN' (val); elites from '$RUN_NAME' (train)"

EXTRA=()
[ "$NO_CHECKSUM" = "1" ] && EXTRA+=(--no-checksum)

python -u scripts/reward/build_replay.py \
    --config "$REWARD_CFG" \
    --run-name "$RUN_NAME" --round "$ROUND" \
    --elite-quantile "$ELITE_QUANTILE" \
    --tau "$TAU" --a-max "$A_MAX" --w-max "$W_MAX" \
    "${EXTRA[@]}"

echo "=== replay -> $DMSR_REWARD_ROOT/replay/$ROUND/replay.jsonl"
