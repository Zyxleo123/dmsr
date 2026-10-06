#!/bin/bash
# Score from-scratch SR2 checkpoints against the frozen SR2 reference.
# This script only calls sbatch.
#
#   bash scripts/slurm/submit_sr2_scratch_field_report.sh              # every snapshot found
#   FR_ARMS='base knn' bash scripts/slurm/submit_sr2_scratch_field_report.sh
#   FR_CKPT=/path/to/tuned_step20000.pt bash scripts/slurm/submit_sr2_scratch_field_report.sh
#
# WHAT IT ANSWERS. `train_sr2_scratch.py` logs `holdout_base.mse` and an earlier
# reading of this line treated a falling mse as progress. That was wrong:
# MEASURED (runs/sr2_baseline/field_table), real SR2's high-k r(k) is ~0 on
# every channel -- it generates a DIFFERENT realisation with the right
# statistics, so MSE against HR's realisation is dominated by a component that
# is unpredictable by design. A model that beats SR2 on mse is most likely
# suppressing small-scale power. The power ratios are the yardstick:
#
#     field          P/P_HR (hi-k)   T(k)    r(k)
#     displacement       0.396       0.624   0.000
#     velocity           0.989       0.994  -0.000
#     density            0.977       0.988   0.092
#
# This job recomputes that table for a checkpoint AND for the frozen SR2 field
# cached on the same boxes at the same seed, with the same estimator, so the
# comparison is like for like rather than against another box's numbers.
#
# It reads checkpoints and cached fields and writes JSON + markdown. It does not
# touch anything a training job owns, so it is safe to run while the arms train.
# Prefer a `tuned_step*.pt` snapshot over `latest.pt`: the snapshots are
# immutable, `latest.pt` is rewritten every SC_SAVE_EVERY steps and a report
# reading it is racing the trainer.
#
# CONFIGURATION TRAVELS AS A POSITIONAL ENV FILE, never `sbatch --export`, which
# on this cluster sets SLURM_GET_USER_ENV=1 and gets the job requeued and HELD.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
LOGS="$ROOT/logs"
ENVD="$ROOT/sr2_scratch/envs"
SCRATCH_DIR="$ROOT/sr2_scratch"
mkdir -p "$LOGS" "$ENVD"

STAMP="$(date '+%Y%m%d-%H%M%S')"
JOB="$HERE/sr2_scratch_field_report_gpu.sbatch"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: you are inside allocation $SLURM_JOB_ID. sbatch only works" >&2
    echo "       from the login node. Exit the allocation and rerun." >&2
    exit 1
fi

FR_ARMS="${FR_ARMS:-baseline knn_r5 soft_radial_r5 repro knn soft_radial}"
FR_BOXES="${FR_BOXES:-set9 set10}"
FR_SEED="${FR_SEED:-0}"
FR_PARTITION="${FR_PARTITION:-general}"
FR_TIME="${FR_TIME:-02:00:00}"
FR_MEM="${FR_MEM:-48G}"
# LATEST=1 scores latest.pt when an arm has no snapshot yet. It RACES the
# trainer's next save, so the step it reports may not be the step it scored.
# Off by default for that reason.
FR_LATEST="${FR_LATEST:-0}"

declare -a CKPTS=() TAGS=()

if [[ -n "${FR_CKPT:-}" ]]; then
    CKPTS+=("$FR_CKPT"); TAGS+=("$(basename "$(dirname "$FR_CKPT")")")
else
    for arm in $FR_ARMS; do
        d="$SCRATCH_DIR/$arm"
        [[ -d "$d" ]] || { echo "  $arm: no output directory yet, skipping"; continue; }
        # Newest immutable snapshot first; latest.pt only if asked.
        mapfile -t snaps < <(ls -1 "$d"/tuned_step*.pt 2>/dev/null | sort -V)
        if (( ${#snaps[@]} )); then
            CKPTS+=("${snaps[-1]}"); TAGS+=("$arm")
        elif [[ "$FR_LATEST" == "1" && -f "$d/latest.pt" ]]; then
            echo "  $arm: no snapshot yet, using latest.pt (RACES the trainer)"
            CKPTS+=("$d/latest.pt"); TAGS+=("$arm")
        else
            echo "  $arm: no tuned_step*.pt yet (first gate is at 20000);" \
                 "FR_LATEST=1 to score latest.pt anyway"
        fi
    done
fi

if (( ! ${#CKPTS[@]} )); then
    echo
    echo "Nothing to score. The arms write their first snapshot at step 20000"
    echo "(SC_GATE_AT_STEPS), which at ~0.31 s/step is about 1.7 h in."
    exit 0
fi

declare -a SUBMITTED=()
for i in "${!CKPTS[@]}"; do
    arm="${TAGS[$i]}"
    # ONE FILE PER JOB, never shared. The job sources this file when it STARTS,
    # not when it is submitted, and STAMP only resolves seconds: two FR_CKPT
    # submissions for the same arm in one second (repro 50k + 72k, pasted
    # together, 2026-09-25) wrote the same name, and the first job would have
    # scored the second one's checkpoint. mktemp makes the name unique.
    stem="$(basename "${CKPTS[$i]}" .pt)"
    f="$(mktemp "$ENVD/fieldreport_${arm}_${stem}-${STAMP}-XXXX.env")"
    {
        echo "# field report for $arm, written $(date '+%F %T')"
        echo "# List values are QUOTED: an unquoted 'FR_BOXES=set9 set10' would"
        echo "# run set10 as a command and kill the job."
        echo "FR_CKPT='${CKPTS[$i]}'"
        echo "FR_BOXES='$FR_BOXES'"
        echo "FR_SEED='$FR_SEED'"
    } > "$f"
    ID=$(sbatch --parsable \
                --job-name="sr2fld_$arm" \
                --partition="$FR_PARTITION" \
                --time="$FR_TIME" \
                --mem="$FR_MEM" \
                "$JOB" "$f")
    SUBMITTED+=("$ID:$arm")
    echo "  $arm -> $ID   $(basename "${CKPTS[$i]}")"
done

cat <<EOF

=== watch
EOF
for s in "${SUBMITTED[@]}"; do
    id="${s%%:*}"; arm="${s##*:}"
    echo "  $arm:"
    echo "    tail -F $LOGS/slurm-sr2scrfield-${id}.out"
done
cat <<EOF

=== results
EOF
for s in "${SUBMITTED[@]}"; do
    arm="${s##*:}"
    echo "  $arm:  $SCRATCH_DIR/$arm/field_report/field_report_*.{json,md}"
done
cat <<'EOF'

=== how to read it
  Compare the candidate's P/P_HR column against the FROZEN column beside it,
  per field. Those two were computed on the same boxes, the same seed and the
  same estimator, so the difference is the model and nothing else.

    density  P/P_HR -> frozen is ~0.98. Below it = missing small-scale mass.
    velocity P/P_HR -> frozen is ~0.99. This is the defect with no loss term
                       behind it on this line; every arm so far sits low.
    r(k)     -> frozen is ~0. HIGHER than frozen is NOT better: it means the
                model is reproducing HR's realisation instead of generating
                its own, i.e. regressing to the mean and killing substructure.

  mse and low_k are reported over EVERY tile (512/box) rather than the ~16 the
  training loop samples. Do not compare them against the trainer's own
  metrics.jsonl values, which are too noisy to read at three decimals.

  This report is a FIELD-level check. It cannot see subhalos -- MEASURED,
  frozen SR2 matches HR's P(k) to 1-5% while Rockstar finds most of them
  missing. The radial verdict still comes from the Rockstar gate.
EOF
