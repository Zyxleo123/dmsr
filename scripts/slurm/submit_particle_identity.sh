#!/bin/bash
# Submitter for the SR2-vs-HR particle identity study. It ONLY calls sbatch.
#
#   bash scripts/slurm/submit_particle_identity.sh              # main + control
#   bash scripts/slurm/submit_particle_identity.sh main         # SR2 vs HR only
#   bash scripts/slurm/submit_particle_identity.sh control      # seed vs seed
#   PID_BOXES="set8 set9" bash scripts/slurm/submit_particle_identity.sh
#   DRY=1 bash scripts/slurm/submit_particle_identity.sh        # print only
#
# Structure. Per box, the three Rockstar passes (HR, SR2 seed 0, SR2 seed 1) are
# SIBLINGS -- none of them needs another's output, so chaining them would only
# add queue latency. The two analyses hang off them with --dependency=afterok:
#
#   HR ------------\
#   SR2 seed 0 -----+--> analyse  A=hr      B=base:0   (does the residual have
#   SR2 seed 1 --\                                      to reshuffle particles?)
#                 \---> analyse  A=base:0  B=base:1   (does SR2's own sampling
#                                                      noise reshuffle them
#                                                      anyway? -- the control
#                                                      that decides whether the
#                                                      first number is a target
#                                                      or an irreducible floor)
#
# The control is not optional garnish: if two SR2 seeds disagree about a
# subhalo's particles as much as SR2 and HR do, then no residual trained
# pointwise on (SR2 -> HR) can fix it, because the input does not determine the
# answer. Submitting both at once costs one extra Rockstar run and one CPU job.
#
# Configuration goes into ONE timestamped env file passed as a POSITIONAL
# argument. Never `sbatch --export`: on this cluster any explicit export list
# makes sbatch set SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login
# environment on the compute node, and the job is requeued and HELD with
# "(user env retrieval failed requeued held)", stranding every dependent job.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
STAGE="${1:-all}"
DRY="${DRY:-0}"

PID_BOXES="${PID_BOXES:-set8}"
PID_CONTROL_SEED="${PID_CONTROL_SEED:-1}"
PID_CLASSES="${PID_CLASSES:-hosts,subhalos}"
PID_MIN_PARTICLES="${PID_MIN_PARTICLES:-50}"
PID_MAX_PAIRS="${PID_MAX_PAIRS:-20000}"
PID_CHUNKS="${PID_CHUNKS:-8}"

mkdir -p "$ROOT/logs" "$ROOT/env"
ENVFILE="$ROOT/env/particle_identity_$(date +%Y%m%d_%H%M%S).env"
cat > "$ENVFILE" <<EOF
# Written by scripts/slurm/submit_particle_identity.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$ROOT
REWARD_CFG=configs/reward/reward.yaml
PID_CLASSES=$PID_CLASSES
PID_MIN_PARTICLES=$PID_MIN_PARTICLES
PID_MAX_PAIRS=$PID_MAX_PAIRS
PID_CHUNKS=$PID_CHUNKS
EOF
echo "env file: $ENVFILE"

# Submitting from inside an allocation would otherwise leak SLURM_* into the
# child job's environment and confuse its own array/step bookkeeping.
for v in $(env | sed -n 's/^\(SLURM_[A-Z_]*\)=.*/\1/p'); do unset "$v" || true; done

submit() {   # submit <label> [sbatch args...] -- echoes the job id
    local label="$1"; shift
    if [ "$DRY" = "1" ]; then echo "DRY [$label]: sbatch $*" >&2; echo "000000"; return; fi
    sbatch --parsable "$@"
}

for BOX in $PID_BOXES; do
    echo "--- $BOX"
    HR=$(submit prep --job-name="pid_hr_$BOX" \
        scripts/slurm/particle_identity_prep_cpu.sbatch \
        "$ENVFILE" "BOXES=$BOX" "SOURCES=hr" "BASE_SEED=0")
    echo "  prep HR            $HR"

    B0=$(submit prep --job-name="pid_b0_$BOX" \
        scripts/slurm/particle_identity_prep_cpu.sbatch \
        "$ENVFILE" "BOXES=$BOX" "SOURCES=base" "BASE_SEED=0")
    echo "  prep SR2 seed 0    $B0"

    if [ "$STAGE" = "all" ] || [ "$STAGE" = "main" ]; then
        A1=$(submit analyse --job-name="pid_main_$BOX" \
            --dependency="afterok:$HR:$B0" \
            scripts/slurm/particle_identity_cpu.sbatch \
            "$ENVFILE" "PID_BOX=$BOX" "PID_A=hr" "PID_B=base:0")
        echo "  analyse SR2 vs HR  $A1  (after $HR,$B0)"
    fi

    if [ "$STAGE" = "all" ] || [ "$STAGE" = "control" ]; then
        B1=$(submit prep --job-name="pid_b${PID_CONTROL_SEED}_$BOX" \
            scripts/slurm/particle_identity_prep_cpu.sbatch \
            "$ENVFILE" "BOXES=$BOX" "SOURCES=base" "BASE_SEED=$PID_CONTROL_SEED")
        echo "  prep SR2 seed $PID_CONTROL_SEED    $B1"
        A2=$(submit analyse --job-name="pid_ctrl_$BOX" \
            --dependency="afterok:$B0:$B1" \
            scripts/slurm/particle_identity_cpu.sbatch \
            "$ENVFILE" "PID_BOX=$BOX" "PID_A=base:0" "PID_B=base:$PID_CONTROL_SEED")
        echo "  analyse seed ctrl  $A2  (after $B0,$B1)"
    fi
done

cat <<EOF

Results land in $ROOT/particle_identity/<box>__<A>__<B>/
  summary.json   headline numbers, both matched-pair and matching-free
  pairs.jsonl    one row per analysed object
  metrics.npz    flat arrays; figures redraw from this alone
  figures/       fig1 identity, fig2 translation-vs-reshuffle, fig3 radius,
                 fig4 chunk, fig5 fate-vs-mass

Read the two summary.json side by side. The seed-vs-seed control is the
denominator for the SR2-vs-HR number: what it does NOT explain is what a
residual could in principle fix.
EOF
