#!/bin/bash
# The host-as-a-background-component line (docs/sr2_host_profile.md).
#
# The defect: HR's median subhalo sits at 1.13 R_vir; every tuned arm this line
# has produced sits at 0.30-0.35 (radial-placement-is-inverted). Matching HR's
# radial distribution directly -- one-to-one, rank, quantile -- has already
# failed on held-out data (shell-levers-built), because the per-host radial
# distribution is real but not reproducible by a shared local operator.
#
# The hypothesis: nothing in the gather loss ever ASKS for a radius. 7 of its 8
# terms are exactly translation-invariant, and the 8th explains 13.4% of the
# offset. What does have a radial preference is the membership SOLVE: its flat
# background plus its n_c/f^3 prior make the host core the cheapest place for a
# candidate to meet its Sinkhorn mass target. Giving the background the host's
# own density profile turns membership into a log OVERDENSITY test, which
# removes that preference without stating any per-object radial target.
#
# This submits the GATE, which can kill the idea before a GPU is spent:
#   stage 1 (cpu, seconds)  the gated LR host table per box
#   stage 2 (cpu, ~20 min)  supply + drift probe, afterok on stage 1
#
#   bash scripts/slurm/submit_host_profile.sh                 # gate only
#   BOXES="set9 set10" bash scripts/slurm/submit_host_profile.sh
#   DUMPS="/path/a.npz /path/b.npz" bash scripts/slurm/submit_host_profile.sh
#   DRY=1 bash scripts/slurm/submit_host_profile.sh
#
# The two training arms are NOT submitted here. They are printed at the end,
# to be run only once the supply verdict reads SUPPLY OK -- that is the whole
# reason this gate exists.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

# set9 and set10 are the HOLDOUT boxes the Rockstar gate scores on, so the
# supply question is asked where the verdict will be read. Adding the training
# boxes costs one field load each and answers whether the arms can even train.
# TRAIN boxes AND holdout boxes. The gate's supply question is about the
# holdout, where the verdict is read -- but `--soft-host-profile lr` needs a
# table for every box it TRAINS on, and job 56447 died on exactly that: the
# tables existed for set9/set10 while the arm trained on set3-set7. Cheap to
# over-cover (seconds per box), expensive to under-cover (a dead GPU job).
BOXES="${BOXES:-set3 set4 set5 set6 set7 set9 set10}"
DUMPS="${DUMPS:-}"
MIN_HR_NUM_P="${MIN_HR_NUM_P:-20000}"
MAX_HOSTS="${MAX_HOSTS:-64}"
SEED="${SEED:-0}"
SUPPLY_FLOOR="${SUPPLY_FLOOR:-0.8}"

# Pre-flight here, not in the job: a queued job that waits and then dies on an
# unreadable path spends queue time to report a typo.
_MISS=0
for _b in $BOXES; do
    _d="$REWARD_ROOT/lagrangian_host/$_b/${_b}_lr_rockstar"
    if ! ls "$_d"/halos_*.ascii >/dev/null 2>&1; then
        echo "no LR Rockstar catalog for $_b under $_d"
        echo "  (build_lagrangian_host.py leaves one per box; without it stage 1"
        echo "   would have to RUN Rockstar, which is not what this job is sized for)"
        _MISS=1
    fi
    if ! ls "$REWARD_ROOT/cache/sr2_base/${_b}_seed${SEED}_"*.npy >/dev/null 2>&1; then
        echo "no cached frozen SR2 field for $_b seed $SEED"
        echo "  build it with scripts/reward/cache_sr2_base.py"
        _MISS=1
    fi
done
for _f in $DUMPS; do
    [[ -r "$_f" ]] || { echo "dump not readable: $_f"; _MISS=1; }
done
[ "$_MISS" = "0" ] || { echo; echo "fix the above and resubmit"; exit 1; }

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$REWARD_ROOT/host_profile_probe"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
ENVFILE="$REWARD_ROOT/env/hostprofile_$STAMP.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_host_profile.sh at $(date '+%F %T').
# List values QUOTED (env-file-list-values-must-be-quoted).
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
HP_BOXES="$BOXES"
HP_MIN_HR_NUM_P=$MIN_HR_NUM_P
HP_NG_LR=64
HP_NG_HR=512
HP_PROBE_BOXES="$BOXES"
HP_PROBE_DUMPS="$DUMPS"
HP_MAX_HOSTS=$MAX_HOSTS
HP_SEED=$SEED
HP_SUPPLY_FLOOR=$SUPPLY_FLOOR
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

# Scrub SLURM_* so this works from inside an allocation as well as the login
# node (and note: sbatch itself only works from the login node here).
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch scripts/slurm/lr_host_table_cpu.sbatch $ENVFILE"
    echo "DRY: sbatch --dependency=afterok:<J1> scripts/slurm/host_profile_probe_cpu.sbatch $ENVFILE"
    exit 0
fi

J1=$(sbatch --parsable scripts/slurm/lr_host_table_cpu.sbatch "$ENVFILE")
echo "submitted lrhosttbl -> job $J1 (cpu)"
J2=$(sbatch --parsable --dependency=afterok:"$J1" \
     scripts/slurm/host_profile_probe_cpu.sbatch "$ENVFILE")
echo "submitted hpprobe   -> job $J2 (cpu, afterok:$J1)"
echo
echo "watch:"
echo "  tail -F $REWARD_ROOT/logs/slurm-lrhosttbl-$J1.out"
echo "  tail -F $REWARD_ROOT/logs/slurm-hpprobe-$J2.out"
echo
echo "results:"
echo "  $REWARD_ROOT/lr_host_table/summary.json          (gated hosts per box)"
echo "  $REWARD_ROOT/host_profile_probe/summary.json     (supply + drift)"
for _b in $BOXES; do
echo "  $REWARD_ROOT/host_profile_probe/supply_${_b}.json"
done
echo
echo "verdict:"
echo "  python -c \"import json,glob;[print(f, json.load(open(f))['verdict']) for f in sorted(glob.glob('$REWARD_ROOT/host_profile_probe/supply_*.json'))]\""
echo
echo "THEN, only if the supply verdict reads SUPPLY OK, launch the two arms."
echo
echo "CARRY THE WINNING BASELINE. Three submitter defaults differ from"
echo "all_blocks_sm_coarse_soft, the soft arm that WINS the held-out gate, and"
echo "job 56450 was launched without them -- it ran d6 at 37% of its loss"
echo "budget, which loss-term-ablation measured as HARMFUL, so its result is"
echo "not comparable to the coarse_soft line at all:"
echo "    MG_W_D6=0.0          (default 1.0; d6 is harmful and NOT in the minimal set)"
echo "    MG_CENTRE_MODE=self  (default full)"
echo "    MG_W_HOST_SETS=1.0   (default 0.0; the host guard that PASSED its own gate)"
echo
echo "  BASE=\"MG_SOFT_MEMBERSHIP=1 MG_W_D6=0.0 MG_CENTRE_MODE=self MG_W_HOST_SETS=1.0\""
echo
echo "  # arm 1: the inference-honest NFW from the LR catalog"
echo "  env \$BASE MG_SOFT_HOST_PROFILE=lr   MG_SOFT_HOST_WEIGHT=1.0 \\"
echo "    MG_LABEL=_hostlr   bash scripts/slurm/submit_member_gather_train.sh"
echo
echo "  # arm 2: the complete-but-circular profile off the candidate field"
echo "  env \$BASE MG_SOFT_HOST_PROFILE=self MG_SOFT_HOST_WEIGHT=1.0 \\"
echo "    MG_LABEL=_hostself bash scripts/slurm/submit_member_gather_train.sh"
echo
echo "beta is NOT comparable between the two -- read the ladder off"
echo "soft_host_logrho_p10/p90 in metrics.jsonl, in nats."
