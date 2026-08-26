#!/bin/bash
# Measure whether the member-gather centre term is learnable, before spending a
# rung ladder on finding out.
#
#   centre-offset decomposition (GPU, ~25 min: it is one frozen forward per host
#   plus the pool build, which is what takes the time)
#
# Why this and why now. docs/sr2_member_gather.md section 6 measured that adding
# `centre` moved the gate 8/154 -> 72/154, so the term carries the result. It is
# also the only term in the loss that is not an invariant statistic: the other
# five are moments about the set's OWN centroid, and a shared convolutional
# operator can learn a moment because a moment is a rule. `centre` is a
# per-object address, and the pool asks for 5,396 of them from 335,954 shared
# parameters. Frozen SR2 starts a median 5.59 search radii out (measured, the
# step-0 row of the 2026-08-23 pool) -- about 1 Mpc/h per object.
#
# The question this answers is whether that 5.59 is a systematic infall deficit
# (radial, environment-dependent, LEARNABLE) or isotropic scatter (realisation
# noise LR does not contain, NOT learnable by any architecture). The rules are
# fitted on the training hosts and scored on the held-out ones -- the same split
# the fine-tune uses -- and the number to read is the share of held-out sets
# that would land inside ONE search radius, which is compare_gather_catalog's
# own hit criterion. Against 72/154 = 46.8% (the free field, which saw every
# address) and 151/154 = 98.1% (the geometric ceiling).
#
# This submitter ONLY calls sbatch. Configuration goes into ONE timestamped env
# file passed as a POSITIONAL argument -- never `sbatch --export`, which on this
# cluster sets SLURM_GET_USER_ENV=1 and gets the job requeued and HELD. Every
# value is QUOTED in the file: an unquoted `BOXES=set0 set1` in a SOURCED file
# runs `set1` as a command, which killed all 13 owner-array tasks on 2026-08-23.
#
#   bash scripts/slurm/submit_centre_offset.sh
#   CO_MAX_HOSTS_PER_BOX=2 CO_LABEL=_quick bash scripts/slurm/submit_centre_offset.sh
#   DRY=1 bash scripts/slurm/submit_centre_offset.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

CO_TRAIN_BOXES="${CO_TRAIN_BOXES:-set3 set4 set5 set6 set7}"
CO_HOLDOUT_BOXES="${CO_HOLDOUT_BOXES:-set9 set10}"
CO_N_TILES="${CO_N_TILES:-4}"
CO_MAX_HOSTS_PER_BOX="${CO_MAX_HOSTS_PER_BOX:-8}"
CO_MIN_LOG_MVIR="${CO_MIN_LOG_MVIR:-13.5}"
CO_MIN_NUM_P="${CO_MIN_NUM_P:-200}"
CO_MIN_PURITY="${CO_MIN_PURITY:-0.5}"
CO_MIN_LIVE_FRAC="${CO_MIN_LIVE_FRAC:-0.5}"
CO_MAX_SETS="${CO_MAX_SETS:-256}"
CO_BG_K="${CO_BG_K:-0}"
CO_LABEL="${CO_LABEL:-}"
CO_SEED="${CO_SEED:-0}"

RUN_DIR="$REWARD_ROOT/centre_offset/pool${CO_LABEL}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"
ENVFILE="$REWARD_ROOT/env/centre_offset_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_centre_offset.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument. Every value is QUOTED:
# this file is SOURCED, so a bare \`X=a b\` runs \`b\` as a command.
PROJECT="$PROJECT"
ZFS="$ZFS"
DMSR_REWARD_ROOT="$REWARD_ROOT"
CO_TRAIN_BOXES="$CO_TRAIN_BOXES"
CO_HOLDOUT_BOXES="$CO_HOLDOUT_BOXES"
CO_N_TILES="$CO_N_TILES"
CO_MAX_HOSTS_PER_BOX="$CO_MAX_HOSTS_PER_BOX"
CO_MIN_LOG_MVIR="$CO_MIN_LOG_MVIR"
CO_MIN_NUM_P="$CO_MIN_NUM_P"
CO_MIN_PURITY="$CO_MIN_PURITY"
CO_MIN_LIVE_FRAC="$CO_MIN_LIVE_FRAC"
CO_MAX_SETS="$CO_MAX_SETS"
CO_BG_K="$CO_BG_K"
CO_LABEL="$CO_LABEL"
CO_SEED="$CO_SEED"
EOT

# Prove the file round-trips through an EMPTY environment before spending a
# queue slot on it. This exact check reproduces the 2026-08-23 unquoted-list
# failure, which cost 13 jobs and was invisible until they died in seconds.
if ! env -i bash -c "set -euo pipefail; source '$ENVFILE'; \
        [ -n \"\$CO_TRAIN_BOXES\" ] && [ -n \"\$CO_HOLDOUT_BOXES\" ]" 2>/dev/null; then
    echo "ERROR: $ENVFILE does not source cleanly in an empty environment." >&2
    echo "       A list value is almost certainly unquoted; fix it before" >&2
    echo "       submitting, or the job dies on the compute node instead." >&2
    exit 1
fi

echo "envfile: $ENVFILE"
cat "$ENVFILE"
echo
echo "run dir: $RUN_DIR"
echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch scripts/slurm/centre_offset_gpu.sbatch $ENVFILE" >&2
    JID="DRYID"
else
    JID=$(sbatch --parsable scripts/slurm/centre_offset_gpu.sbatch "$ENVFILE")
fi
echo "submitted centre-offset decomposition -> job $JID (GPU, general)"

echo
echo "watch: tail -F $REWARD_ROOT/logs/slurm-coffset-$JID.out"
echo
echo "results:"
echo "  $RUN_DIR/offsets.json   per-set offsets, the radial split, the rules"
echo "  re-print the tables any time, no recompute:"
echo "    python scripts/features/centre_offset_decompose.py \\"
echo "        --from-json $RUN_DIR/offsets.json"
echo
echo "how to read it -- the held-out 'frac within 1 search radius' column:"
echo "  < 25%   the offset is an ADDRESS. A shared operator is being charged"
echo "          for information its input does not carry. Soften the term"
echo "          (MG_CENTRE_DEAD_ZONE / MG_CENTRE_HUBER) and put position on a"
echo "          self-consistency condition instead."
echo "  25-60%  partly a rule. Keep the term, soften its tail so the hopeless"
echo "          sets stop owning the gradient."
echo "  > 60%   a learnable rule. The objective is fine as it is and the"
echo "          fine-tune's problem is capacity and receptive field."
echo "  Reference points: free field 72/154 = 46.8%, ceiling 151/154 = 98.1%,"
echo "  frozen 3/154 = 1.9%  (docs/sr2_member_gather.md section 6.1)."
