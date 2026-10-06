#!/bin/bash
# Run map2map's OWN trainer (styled_srsgan branch) on our data. Only calls sbatch.
#
#   bash scripts/slurm/submit_m2m.sh              # split the data, then train (chained)
#   M2M_STAGE=split bash scripts/slurm/submit_m2m.sh
#   M2M_STAGE=train bash scripts/slurm/submit_m2m.sh
#
# WHY. Four from-scratch runs on our own trainer failed for four different
# reasons, and every one traced to a component we had reimplemented rather than
# to the experiment:
#   run 1  SR2Critic had NO input normaliser        -> score gap 93 -> 5718
#   run 2  --w-base-low 100 was 96-100% of the budget -> density 0.153, BELOW
#                                                        trilinear's 0.706
#   run 3  BCE adversary, adopted after misreading which map2map fork trained
#          SR2                                      -> generator diverged to 1e11
#   run 4  GroupNorm critic is exactly scale-invariant
# This runs their code instead. The generator, discriminator, WGAN-GP, lag2eul
# and hyperparameters are all theirs, unedited.
#
# WHAT IS STILL OURS, and why:
#   * the data layout (their loader wants one .npy per 3-channel vector field,
#     or --augment silently skips the component negation)
#   * an identity norm via --callback-at, because our boxes are ALREADY in
#     cosmology.dis space (that norm divides by 6000 kpc/h at z=0, exactly what
#     paired_catnorm was written with)
#   * crop 8 / step 8 and no --aug-shift, so that the two new loss terms stay
#     indexable by (box, tile). That is the ONE geometry deviation and it is
#     what option A bought.
#
# THIS FIRST RUN CARRIES NONE OF OUR LOSSES. It is the control that has never
# been run: does THEIR trainer on OUR data reach density P/P_HR ~ 0.98? If it
# does not, our loss terms were never the problem and no arm built on them is
# interpretable.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
LOGS="$ROOT/logs"; ENVD="$ROOT/m2m/envs"
mkdir -p "$LOGS" "$ENVD"
STAMP="$(date '+%Y%m%d-%H%M%S')"

# Submitting from inside an allocation inherits SLURM_* and confuses sbatch.
# MEASURED 2026-09-28: on lov4 the submit host IS the controller's client, so
# sbatch works there even with SLURM_JOB_ID set -- the test is the HOSTNAME, not
# the presence of an allocation. Elsewhere sbatch is denied outright
# ("Access/permission denied") and refusing early is clearer than that error.
#
# Either way the inherited SLURM_* are scrubbed before calling sbatch: they
# would otherwise leak this shell's job into the new submission.
if [[ "$(hostname)" != lov4* && -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: you are inside allocation $SLURM_JOB_ID on $(hostname)." >&2
    echo "       sbatch only works from the login node or lov4." >&2
    exit 1
fi
for _v in $(compgen -v | grep '^SLURM_'); do unset "$_v"; done
unset _v

# all = split -> preflight -> train, chained afterok.
# The preflight stage is cheap (CPU, minutes) and is what stops a misconfigured
# argument list from reaching a GPU: two earlier submissions died before a
# single training step and each cost a queue round-trip.
M2M_STAGE="${M2M_STAGE:-all}"
M2M_RUN_NAME="${M2M_RUN_NAME:-m2m_baseline}"
M2M_LAYOUT="${M2M_LAYOUT:-$ZFS/DMSR/m2m_layout}"
M2M_REPO="${M2M_REPO:-/zfsauton2/home/yixiz/DMSR/map2map_styled}"
SPLIT_PARTITION="${SPLIT_PARTITION:-cpu}"
TRAIN_PARTITION="${TRAIN_PARTITION:-general}"
TRAIN_TIME="${TRAIN_TIME:-12:00:00}"
TRAIN_MEM="${TRAIN_MEM:-32G}"
EXCLUDE_2080="${EXCLUDE_2080:-gpu11,gpu12,gpu13,gpu14,gpu15,gpu16,gpu17,gpu18,gpu19}"

if [[ ! -r "$M2M_REPO/m2m.py" ]]; then
    echo "ERROR: no map2map checkout at $M2M_REPO" >&2
    echo "  git -C /zfsauton2/home/yixiz/DMSR/map2map worktree add \\" >&2
    echo "      $M2M_REPO xw/styled_srsgan" >&2
    exit 1
fi

ENVF="$ENVD/${M2M_RUN_NAME}-${STAMP}.env"
{
    echo "# $M2M_RUN_NAME, written $(date '+%F %T') by submit_m2m.sh"
    echo "# List values are QUOTED: an unquoted list runs its tail as a command."
    echo "M2M_RUN_NAME='$M2M_RUN_NAME'"
    echo "M2M_LAYOUT='$M2M_LAYOUT'"
    echo "M2M_REPO='$M2M_REPO'"
    [[ -n "${M2M_TRAIN_BOXES:-}" ]] && echo "M2M_TRAIN_BOXES='$M2M_TRAIN_BOXES'"
    [[ -n "${M2M_VAL_BOXES:-}"   ]] && echo "M2M_VAL_BOXES='$M2M_VAL_BOXES'"
    [[ -n "${M2M_EPOCHS:-}"      ]] && echo "M2M_EPOCHS='$M2M_EPOCHS'"
    [[ -n "${M2M_EXTRA_ARGS:-}"  ]] && echo "M2M_EXTRA_ARGS='$M2M_EXTRA_ARGS'"
    # Every OTHER M2M_* knob that is set, so the env file is the whole
    # configuration rather than the handful someone remembered to add here.
    # MEASURED: M2M_BATCHES reached job 68906 only through sbatch's DEFAULT
    # --export=ALL -- it worked, but the env file, which is the reproducible
    # record of a submission, did not mention the knob being varied. An arm
    # whose defining flag is absent from its own record is not a result.
    # printf %q, not bare quoting: M2M_EXTRA_ARGS holds spaces.
    for _n in $(compgen -v M2M_ | sort); do
        case "$_n" in
            M2M_STAGE|M2M_RUN_NAME|M2M_LAYOUT|M2M_REPO|M2M_TRAIN_BOXES) continue ;;
            M2M_VAL_BOXES|M2M_EPOCHS|M2M_EXTRA_ARGS|M2M_OUT) continue ;;
        esac
        printf '%s=%q\n' "$_n" "${!_n}"
    done
    unset _n
} > "$ENVF"
echo "env: $ENVF"

declare -a IDS=() NAMES=()
SPLIT_ID=""

if [[ "$M2M_STAGE" == "all" || "$M2M_STAGE" == "split" ]]; then
    SPLIT_ID=$(sbatch --parsable \
                      --partition="$SPLIT_PARTITION" \
                      "$HERE/m2m_split_data_cpu.sbatch" "$ENVF")
    IDS+=("$SPLIT_ID"); NAMES+=("split")
    echo "  split -> $SPLIT_ID"
fi

PRE_ID=""
if [[ "$M2M_STAGE" == "all" || "$M2M_STAGE" == "preflight" ]]; then
    dep=()
    [[ -n "$SPLIT_ID" ]] && dep=(--dependency="afterok:$SPLIT_ID")
    PRE_ID=$(sbatch --parsable \
                    --partition="$SPLIT_PARTITION" \
                    "${dep[@]}" \
                    "$HERE/m2m_preflight_cpu.sbatch" "$ENVF")
    IDS+=("$PRE_ID"); NAMES+=("preflight")
    echo "  preflight -> $PRE_ID${SPLIT_ID:+  (afterok:$SPLIT_ID)}"
fi

if [[ "$M2M_STAGE" == "all" || "$M2M_STAGE" == "train" ]]; then
    dep=()
    # afterok, not afterany: training on a half-written layout would pair the
    # wrong boxes, and the split writes atomically precisely so a partial file
    # is never globbed.
    # afterok on the PREFLIGHT when there is one: it is the gate, and a GPU
    # should not be allocated behind a configuration that failed on CPU.
    if [[ -n "$PRE_ID" ]]; then
        dep=(--dependency="afterok:$PRE_ID")
    elif [[ -n "$SPLIT_ID" ]]; then
        dep=(--dependency="afterok:$SPLIT_ID")
    fi
    TRAIN_ID=$(sbatch --parsable \
                      --job-name="m2m_$M2M_RUN_NAME" \
                      --partition="$TRAIN_PARTITION" \
                      --time="$TRAIN_TIME" \
                      --mem="$TRAIN_MEM" \
                      --exclude="$EXCLUDE_2080" \
                      "${dep[@]}" \
                      "$HERE/m2m_train_gpu.sbatch" "$ENVF")
    IDS+=("$TRAIN_ID"); NAMES+=("train")
    echo "  train -> $TRAIN_ID${PRE_ID:+  (afterok:$PRE_ID)}"
fi

# The verdict stage. Chained afterok on training when both run, but usually
# submitted on its own against a checkpoint a run has already written --
# training is 12h-capped and resumable, so scoring waits for nobody.
if [[ "$M2M_STAGE" == "all" || "$M2M_STAGE" == "report" ]]; then
    dep=()
    [[ -n "${TRAIN_ID:-}" ]] && dep=(--dependency="afterok:$TRAIN_ID")
    REPORT_ID=$(sbatch --parsable \
                       --partition="$TRAIN_PARTITION" \
                       "${dep[@]}" \
                       "$HERE/m2m_field_report_gpu.sbatch" "$ENVF")
    IDS+=("$REPORT_ID"); NAMES+=("report")
    echo "  report -> $REPORT_ID${TRAIN_ID:+  (afterok:$TRAIN_ID)}"
fi

# Full P(k) curves, the un-reduced form of the report's single number.
if [[ "$M2M_STAGE" == "curves" ]]; then
    CURVE_ID=$(sbatch --parsable \
                      --partition="$TRAIN_PARTITION" \
                      "$HERE/m2m_power_curves_gpu.sbatch" "$ENVF")
    IDS+=("$CURVE_ID"); NAMES+=("curves")
    echo "  curves -> $CURVE_ID"
fi

# Redraw the P(k) figure from saved curves. CPU, reads npz only.
if [[ "$M2M_STAGE" == "plot" ]]; then
    PLOT_ID=$(sbatch --parsable --partition="$SPLIT_PARTITION" \
                     "$HERE/m2m_power_plot_cpu.sbatch" "$ENVF")
    IDS+=("$PLOT_ID"); NAMES+=("plot")
    echo "  plot -> $PLOT_ID"
fi

cat <<EOF

=== watch
EOF
for i in "${!IDS[@]}"; do
    case "${NAMES[$i]}" in
      split) echo "  split:  tail -F $LOGS/slurm-m2msplit-${IDS[$i]}.out" ;;
      preflight) echo "  preflight: tail -F $LOGS/slurm-m2mpre-${IDS[$i]}.out" ;;
      train) echo "  train:  tail -F $LOGS/slurm-m2mtrain-${IDS[$i]}.out" ;;
      report) echo "  report: tail -F $LOGS/slurm-m2mfield-${IDS[$i]}.out" ;;
      curves) echo "  curves: tail -F $LOGS/slurm-m2mpow-${IDS[$i]}.out" ;;
      plot) echo "  plot:   tail -F $LOGS/slurm-m2mplot-${IDS[$i]}.out" ;;
    esac
done
cat <<EOF

=== results
  layout:     $M2M_LAYOUT/manifest.json  (the exact globs the trainer uses)
  checkpoint: $ROOT/m2m/$M2M_RUN_NAME/checkpoint.pt
  score:      $ROOT/m2m/$M2M_RUN_NAME/field_report_*.{json,md}   (M2M_STAGE=report)
  resume:     resubmit with M2M_STAGE=train -- the job picks up checkpoint.pt
EOF
cat <<'EOF'

=== what to read, and what NOT to
  NOT the loss. map2map logs MSE, and MSE is the wrong yardstick here: frozen
  SR2's own high-k r(k) is ~0 on every channel, so it does not reproduce HR's
  realisation at small scales by design. A model that beats SR2 on MSE is most
  likely suppressing small-scale power. MEASURED on set14: trilinear
  interpolation scores hr_mse 0.241 against SR2's 0.435, and loses on density.

  Score a checkpoint on P/P_HR instead. Reference, same estimator:
      density   frozen SR2 0.977   trilinear 0.706
      velocity  frozen SR2 0.989
      r(k)      frozen SR2 ~0  -- HIGHER is a regression-to-the-mean warning

  Our best from-scratch run so far reached density 0.26. Below 0.706 is worse
  than interpolation, and that is the bar this control has to clear before any
  arm carrying our loss terms means anything.
EOF
