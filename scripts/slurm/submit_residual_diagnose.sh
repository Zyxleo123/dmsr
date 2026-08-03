#!/bin/bash
# Submitter for the residual-prior failure decomposition. ONLY calls sbatch.
#
#   bash scripts/slurm/submit_residual_diagnose.sh            # both checkpoints
#   bash scripts/slurm/submit_residual_diagnose.sh v1         # residual_prior only
#   bash scripts/slurm/submit_residual_diagnose.sh tmax       # the t_max arm too
#   DRY=1 bash scripts/slurm/submit_residual_diagnose.sh      # print, submit nothing
#
# Each arm is a GPU job (sample + measure) with a CPU report job chained behind
# it on afterok. Arms are SIBLINGS -- they share no inputs, so they queue
# concurrently rather than serialising behind one another.
#
# Configuration goes in ONE timestamped env file passed as a POSITIONAL
# argument; never `sbatch --export`, which on this cluster sets
# SLURM_GET_USER_ENV=1 and gets the job requeued and held.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
STAGE="${1:-all}"
DRY="${DRY:-0}"

: "${N_CROPS:=8}"
: "${ALPHAS:=0,0.1,0.2,0.33,0.5,1.0}"

mkdir -p "$ROOT/logs" "$ROOT/env"
ENVFILE="$ROOT/env/resdiag_$(date +%Y%m%d_%H%M%S).env"
cat > "$ENVFILE" <<EOF
# Written by scripts/slurm/submit_residual_diagnose.sh at $(date '+%F %T').
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$ROOT
REWARD_CFG=configs/reward/reward.yaml
MODEL_CFG=configs/reward/residual_prior.yaml
BASE_SEED=0
SEED=0
N_CROPS=$N_CROPS
ALPHAS=$ALPHAS
SHUFFLE_COND=1
USE_EMA=1
EOF
echo "env file: $ENVFILE"

# Submitting from inside an allocation would leak SLURM_* into the child jobs.
for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v"; done

SUB_OVERRIDES=()
sub() {   # sub <label> <sbatch args...>
    local label="$1"; shift
    if [ "$DRY" = "1" ]; then
        echo "DRY  $label:" >&2
        echo "       sbatch $* $ENVFILE ${SUB_OVERRIDES[*]}" >&2
        echo "000000"; return 0
    fi
    local jid rc
    jid=$(sbatch --parsable "$@" "$ENVFILE" "${SUB_OVERRIDES[@]}") || rc=$?
    if [ -n "${rc:-}" ] || [ -z "$jid" ]; then
        echo "!!! sbatch FAILED for: $label -- nothing further submitted." >&2
        exit 1
    fi
    echo "  $label -> job $jid" >&2
    echo "$jid"
}

# One arm = one (checkpoint, sampler setting) pair, with its own output
# directory so two arms never write the same JSONL.
arm() {   # arm <tag> <ckpt> [EXTRA=VAL ...]
    local tag="$1" ckpt="$2"; shift 2
    if [ ! -r "$ckpt" ]; then
        echo "  !! no checkpoint $ckpt -- skipping arm '$tag'" >&2
        return 0
    fi
    SUB_OVERRIDES=("TAG=$tag" "PRIOR_CKPT=$ckpt" "$@")
    local jg
    jg=$(sub "diagnose [$tag] (GPU)" scripts/slurm/residual_diagnose_gpu.sbatch)
    SUB_OVERRIDES=("TAG=$tag")
    sub "report   [$tag] (CPU)" --dependency=afterok:"$jg" \
        scripts/slurm/residual_diagnose_report_cpu.sbatch >/dev/null
    SUB_OVERRIDES=()
}

V1="$ROOT/checkpoints/residual_prior/ckpt_best.pt"
V2="$ROOT/checkpoints/residual_prior_v2/ckpt_best.pt"

# v1 is the converged run (60k steps) and the one Gate B actually sampled, so it
# is the checkpoint whose failure needs explaining. v2 is the partly-trained
# context run -- included because its clip and RMS are still moving, and a
# diagnosis that differs between them says the failure is training-stage
# dependent rather than structural.
case "$STAGE" in
    all)  arm v1 "$V1"; arm v2 "$V2" ;;
    v1)   arm v1 "$V1" ;;
    v2)   arm v2 "$V2" ;;
    # The endpoint arm: same checkpoint, t_max below the region where x0 = (u -
    # s*eps)/alpha divides by 0.0016. If the clip is doing the damage this is
    # where it shows.
    tmax) arm v1 "$V1"
          arm v1_tmax095 "$V1" "SCAN_T_MAX=0.95"
          arm v1_steps100 "$V1" "SCAN_STEPS=100" ;;
    *)    echo "unknown stage: $STAGE (all|v1|v2|tmax)" >&2; exit 1 ;;
esac

echo
echo "queue:   squeue -u \$USER -o '%.9i %.16j %.2t %.10M %R'"
echo "logs:    $ROOT/logs"
echo "results: $ROOT/audits/residual_diagnose*/residual_diagnose_summary.json"
