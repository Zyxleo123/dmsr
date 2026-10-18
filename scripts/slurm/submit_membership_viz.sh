#!/bin/bash
# Visualise the soft-membership SOLVE for the coarse_soft arm.
#
#   dump (GPU) --afterok--> viz (CPU)
#
# The dump job reuses the coarse_soft arm's OWN pool machinery
# (member_gather_train_gpu.sbatch, which translates the MG_* env below into the
# finetune flags), runs ZERO training steps, solves one host's membership on a
# saved SR field with the per-sweep recorder on, and writes a single npz. The
# viz job reads that npz and renders the two figures. So the expensive pool
# build happens once on the GPU and the figures redraw cheaply on the CPU --
# the same split every other diagnostic in this repo uses.
#
# WHY THIS IS FAITHFUL: everything that defines the pool and the solve (prior,
# lambda, factor, pool ratio, tiling, min mass, centre mode) is the arm's own
# config, pinned below to the soft-membership ladder's coarse_soft values. The
# field it solves on defaults to that arm's saved tuned set9 box, so the P you
# see is the one the arm actually converged to (soft_hr_fraction ~0.25).
#
#   bash scripts/slurm/submit_membership_viz.sh
#   MV_FIELD=<...>.npy MV_HOST=set9:h12345 bash scripts/slurm/submit_membership_viz.sh
#   DRY=1 bash scripts/slurm/submit_membership_viz.sh          # print, submit nothing
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

# --- what to solve, and on which field --------------------------------------
MV_BOX="${MV_BOX:-set9}"
MV_FIELD="${MV_FIELD:-$REWARD_ROOT/flow_rockstar/fields/${MV_BOX}__mgho_all_blocks_sm_coarse_soft_${MV_BOX}__seed0.npy}"
MV_HOST="${MV_HOST:-}"                 # empty = the richest host in the box
MV_PRIOR="${MV_PRIOR:-coarse}"         # coarse (the arm) | hard (the control)
MV_LAMBDA="${MV_LAMBDA:-1.0}"
MV_N_EVOLUTION="${MV_N_EVOLUTION:-4}"
MV_N_MATRIX="${MV_N_MATRIX:-6}"
MV_SELECT="${MV_SELECT:-spread}"       # spread | mass | substitution
# The host-profile background in the solve (MG_SOFT_HOST_PROFILE). `parent`
# centres it on each set's OWN host; lr/self use the task host for every set.
MV_HOST_PROFILE="${MV_HOST_PROFILE:-off}"   # off | lr | self | parent
# A job id the dump waits on (afterok) -- e.g. a splice job that is about to
# write MV_FIELD. Skips the "field readable" check at submit time.
MV_AFTER="${MV_AFTER:-}"
MV_HOST_WEIGHT="${MV_HOST_WEIGHT:-0.0}"
# The per-set radial prior (MG_SOFT_RADIAL_*): gamma and width in R_vir.
MV_RADIAL_WEIGHT="${MV_RADIAL_WEIGHT:-0.0}"
MV_RADIAL_WIDTH="${MV_RADIAL_WIDTH:-0.2}"
# Chain the per-band centre-count job (solve_centre_condensation_cpu.sbatch)
# behind the dump. MV_ARM is the arm whose Rockstar catalog is searched.
MV_CONDENSE="${MV_CONDENSE:-0}"
MV_ARM="${MV_ARM:-mgho_all_blocks_sm_coarse_soft_${MV_BOX:-set9}}"

# --- re-plot an EXISTING dump: no GPU, no re-solve --------------------------
# The npz is the artefact; every figure is a pure function of it. Point
# MV_NPZ_IN at one and only the CPU job is submitted, which is how a new figure
# gets added to an old solve without paying for the pool build again.
MV_NPZ_IN="${MV_NPZ_IN:-}"

# --- which figures ----------------------------------------------------------
MV_RADIAL="${MV_RADIAL:-0}"            # max_j P per particle vs host radius
MV_RADIAL_ONLY="${MV_RADIAL_ONLY:-0}"
MV_PANEL="${MV_PANEL:-0}"              # mean-<P> x-z panel alone, slide size
MV_PANEL_ONLY="${MV_PANEL_ONLY:-0}"
MV_HEATMAP="${MV_HEATMAP:-0}"
MV_HEATMAP_ONLY="${MV_HEATMAP_ONLY:-0}"
MV_HEATMAP_HALFWIDTH="${MV_HEATMAP_HALFWIDTH:-10.0}"
MV_R_MAX_RVIR="${MV_R_MAX_RVIR:-4.0}"
MV_CENTRE_RADIAL="${MV_CENTRE_RADIAL:-0}"        # radial dist of optimized c_j vs HR
MV_CENTRE_RADIAL_ONLY="${MV_CENTRE_RADIAL_ONLY:-0}"
MV_CENTRE_RADIAL_BARS="${MV_CENTRE_RADIAL_BARS:-0}"
MV_CENTRE_RADIAL_BARS_ONLY="${MV_CENTRE_RADIAL_BARS_ONLY:-0}"
# The gate's measured SR subhalo median r/Rvir, drawn as a reference line so the
# "targets are outer, subhalos land inner" gap is visible on one axis.
MV_SR_SUBHALO_MEDIAN="${MV_SR_SUBHALO_MEDIAN:-nan}"
# The radial axis needs the parent host's R_vir. The dump only carries it when
# it ran with --soft-host-profile, so by default the viz looks it up in the HR
# Rockstar ascii by the id in the npz's host_key (column 5, kpc/h).
MV_HOST_RVIR="${MV_HOST_RVIR:-0}"
MV_ROCKSTAR_ASCII="${MV_ROCKSTAR_ASCII:-$REWARD_ROOT/halos/${MV_BOX}__hr__hr/hr_rockstar/halos_0.0.ascii}"

TS="$(date +%Y%m%d_%H%M%S)_$$"
OUT_DIR="$REWARD_ROOT/member_gather/membership_viz"
_HP_TAG=""
[[ "$MV_HOST_PROFILE" != "off" ]] && _HP_TAG="_hp${MV_HOST_PROFILE}${MV_HOST_WEIGHT}"
[[ "$MV_RADIAL_WEIGHT" != "0.0" && "$MV_RADIAL_WEIGHT" != "0" ]] \
    && _HP_TAG="${_HP_TAG}_rad${MV_RADIAL_WEIGHT}w${MV_RADIAL_WIDTH}"
NPZ="$OUT_DIR/membership_${MV_PRIOR}_lam${MV_LAMBDA}${_HP_TAG}_${MV_BOX}_${TS}.npz"
mkdir -p "$OUT_DIR" "$REWARD_ROOT/logs" "$REWARD_ROOT/env"

if [[ -n "$MV_NPZ_IN" ]]; then
    [[ -r "$MV_NPZ_IN" ]] || { echo ">>> MV_NPZ_IN not readable: $MV_NPZ_IN" >&2; exit 1; }
    NPZ="$MV_NPZ_IN"
elif [[ -z "${MV_AFTER:-}" && ! -r "$MV_FIELD" ]]; then
    echo ">>> field not readable: $MV_FIELD" >&2
    echo ">>> pass MV_FIELD=<a full-box (6,ng,ng,ng) .npy>, MV_BOX=<box>, or" >&2
    echo ">>> MV_NPZ_IN=<an existing dump> to re-plot without the GPU." >&2
    exit 1
fi

# --- the DUMP env file (positional arg, never --export) ----------------------
# Only the pool/solve-relevant knobs are pinned; the batch script's own defaults
# supply the rest and already match the arm (min mass 13.5, 4 tiles, 8 hosts,
# population sub, purity/live 0.5). The split is the target box ALONE, so
# build_pool does one box's frozen forwards, not five. Every value is quoted --
# an unquoted `set9 set10` would source as a command.
DUMP_ENV="$REWARD_ROOT/env/membership_dump_${TS}.env"
cat > "$DUMP_ENV" <<EOT
# Written by submit_membership_viz.sh at $(date '+%F %T'); sourced positionally.
PROJECT="$PROJECT"
ZFS="$ZFS"
DMSR_REWARD_ROOT="$REWARD_ROOT"
MG_TRAIN_BOXES="$MV_BOX"
# A single space, NOT empty. The batch script reads holdout as
# \${MG_HOLDOUT_BOXES:=set9 set10}, and := treats an EMPTY value as unset and
# restores that default -- which put the target box in BOTH lists and split_pool
# rejected it. A space is non-null (so := keeps it) and word-splits to nothing,
# so the pool is the target box alone with an empty holdout.
MG_HOLDOUT_BOXES=" "
MG_RUNG="all_blocks"
MG_CENTRE_MODE="self"
MG_SOFT_MEMBERSHIP="1"
MG_SOFT_PRIOR="$MV_PRIOR"
MG_SOFT_LAMBDA="$MV_LAMBDA"
MG_SOFT_FACTOR="8"
MG_SOFT_POOL_RATIO="2.0"
MG_SOFT_SIGMA="adaptive"
MG_MAX_SET_PARTICLES="768"
MG_STEPS="0"
MG_WANDB_MODE="disabled"
MG_LABEL="_membership_viz"
MG_DUMP_MEMBERSHIP_NPZ="$NPZ"
MG_DUMP_MEMBERSHIP_FIELD="$MV_FIELD"
MG_DUMP_MEMBERSHIP_HOST="$MV_HOST"
MG_SOFT_HOST_PROFILE="$MV_HOST_PROFILE"
MG_SOFT_HOST_WEIGHT="$MV_HOST_WEIGHT"
MG_SOFT_RADIAL_WEIGHT="$MV_RADIAL_WEIGHT"
MG_SOFT_RADIAL_WIDTH="$MV_RADIAL_WIDTH"
EOT

# Source it in a clean shell before queuing: an env file that does not source is
# a whole-chain failure found one job id later.
_check=$(env -i bash -c "set -eu; source '$DUMP_ENV'; printf '%s' \"\$MG_TRAIN_BOXES\"" 2>&1) || {
    echo ">>> dump env file does not source cleanly:" >&2; echo ">>> $_check" >&2; exit 1; }
[[ "$_check" == "$MV_BOX" ]] || { echo ">>> dump env round-trip mismatch: $_check" >&2; exit 1; }

# --- the VIZ env file --------------------------------------------------------
VIZ_ENV="$REWARD_ROOT/env/membership_viz_${TS}.env"
cat > "$VIZ_ENV" <<EOT
# Written by submit_membership_viz.sh at $(date '+%F %T'); sourced positionally.
PROJECT="$PROJECT"
ZFS="$ZFS"
DMSR_REWARD_ROOT="$REWARD_ROOT"
MV_NPZ="$NPZ"
MV_OUT_DIR="$OUT_DIR"
MV_N_EVOLUTION="$MV_N_EVOLUTION"
MV_N_MATRIX="$MV_N_MATRIX"
MV_SELECT="$MV_SELECT"
MV_RADIAL="$MV_RADIAL"
MV_RADIAL_ONLY="$MV_RADIAL_ONLY"
MV_PANEL="$MV_PANEL"
MV_PANEL_ONLY="$MV_PANEL_ONLY"
MV_HEATMAP="$MV_HEATMAP"
MV_HEATMAP_ONLY="$MV_HEATMAP_ONLY"
MV_HEATMAP_HALFWIDTH="$MV_HEATMAP_HALFWIDTH"
MV_HOST_RVIR="$MV_HOST_RVIR"
MV_ROCKSTAR_ASCII="$MV_ROCKSTAR_ASCII"
MV_R_MAX_RVIR="$MV_R_MAX_RVIR"
MV_CENTRE_RADIAL="$MV_CENTRE_RADIAL"
MV_CENTRE_RADIAL_ONLY="$MV_CENTRE_RADIAL_ONLY"
MV_CENTRE_RADIAL_BARS="$MV_CENTRE_RADIAL_BARS"
MV_CENTRE_RADIAL_BARS_ONLY="$MV_CENTRE_RADIAL_BARS_ONLY"
MV_SR_SUBHALO_MEDIAN="$MV_SR_SUBHALO_MEDIAN"
EOT

echo "=== membership viz"
echo "  box:    $MV_BOX"
echo "  field:  $MV_FIELD"
echo "  host:   ${MV_HOST:-<richest in box>}"
echo "  prior:  $MV_PRIOR   lambda: $MV_LAMBDA"
echo "  host profile: $MV_HOST_PROFILE  beta=$MV_HOST_WEIGHT"
echo "  radial prior: gamma=$MV_RADIAL_WEIGHT  width=$MV_RADIAL_WIDTH R_vir"
echo "  npz:    $NPZ"
echo "  dump env: $DUMP_ENV"
echo "  viz  env: $VIZ_ENV"
echo

# Scrub SLURM_* so a submission from inside an allocation does not trip
# SLURM_GET_USER_ENV and get the job requeued-and-held.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

if [[ "$DRY" == "1" ]]; then
    echo "DRY: would submit"
    if [[ -z "$MV_NPZ_IN" ]]; then
        echo "  sbatch --parsable --job-name=mgmemdump --time=00:40:00 \\"
        echo "      scripts/slurm/member_gather_train_gpu.sbatch $DUMP_ENV"
        echo "  sbatch --dependency=afterok:<dump> membership_viz_cpu.sbatch $VIZ_ENV"
    else
        echo "  (re-plot only, MV_NPZ_IN set -- no GPU dump)"
        echo "  sbatch scripts/slurm/membership_viz_cpu.sbatch $VIZ_ENV"
    fi
    exit 0
fi

# Re-plotting an existing dump: the CPU job alone, with no dependency.
if [[ -n "$MV_NPZ_IN" ]]; then
    VIZ_ID=$(sbatch --parsable scripts/slurm/membership_viz_cpu.sbatch "$VIZ_ENV")
    echo "viz job (CPU): $VIZ_ID   (re-plot of $MV_NPZ_IN, no dump)"
    cat <<EOS

=== WATCH
  tail -F $REWARD_ROOT/logs/slurm-mgmemviz-$VIZ_ID.out

=== RESULTS in $OUT_DIR/
  $(basename "${NPZ%.npz}")_radial.png     max_j P per particle vs r/Rvir
  $(basename "${NPZ%.npz}")_meanP_xz.png   mean <P> per bin, x-z panel alone
EOS
    exit 0
fi

# Optional resource overrides for the dump. The batch script defaults to
# partition=general (a6000), which is often congested; a light INFERENCE dump
# (build the pool, one no_grad solve, no training tape) runs immediately on an
# idle legacy node. legacy enforces --qos=qos_legacy (scontrol show partition).
#   MV_DUMP_PARTITION=legacy MV_DUMP_QOS=qos_legacy MV_DUMP_GRES=gpu:v100:1 \
#     bash scripts/slurm/submit_membership_viz.sh
MV_DUMP_PARTITION="${MV_DUMP_PARTITION:-}"
MV_DUMP_QOS="${MV_DUMP_QOS:-}"
MV_DUMP_GRES="${MV_DUMP_GRES:-}"
MV_DUMP_EXCLUDE="${MV_DUMP_EXCLUDE:-}"

DUMP_ID=$(sbatch --parsable --job-name=mgmemdump --time=00:40:00 \
    ${MV_AFTER:+--dependency="afterok:$MV_AFTER"} \
    ${MV_DUMP_PARTITION:+--partition="$MV_DUMP_PARTITION"} \
    ${MV_DUMP_QOS:+--qos="$MV_DUMP_QOS"} \
    ${MV_DUMP_GRES:+--gres="$MV_DUMP_GRES"} \
    ${MV_DUMP_EXCLUDE:+--exclude="$MV_DUMP_EXCLUDE"} \
    scripts/slurm/member_gather_train_gpu.sbatch "$DUMP_ENV")
echo "dump job (GPU): $DUMP_ID  (partition ${MV_DUMP_PARTITION:-general})"

VIZ_ID=$(sbatch --parsable --dependency="afterok:$DUMP_ID" \
    scripts/slurm/membership_viz_cpu.sbatch "$VIZ_ENV")
echo "viz  job (CPU): $VIZ_ID   (afterok:$DUMP_ID)"
COND_ID=""
if [[ "$MV_CONDENSE" == "1" ]]; then
    COND_ID=$(sbatch --parsable --dependency="afterok:$DUMP_ID" \
        scripts/slurm/solve_centre_condensation_cpu.sbatch \
        "SC_BOX=$MV_BOX" "SC_ARM=$MV_ARM" "SC_NPZ=$NPZ")
    echo "band job (CPU): $COND_ID   (afterok:$DUMP_ID)"
fi
echo

cat <<EOS
=== WATCH
  tail -F $REWARD_ROOT/logs/slurm-mgtrain-$DUMP_ID.out     # the solve
  tail -F $REWARD_ROOT/logs/slurm-mgmemviz-$VIZ_ID.out     # the plotting
  squeue -u \$USER -n mgmemdump,mgmemviz

=== RESULTS in $OUT_DIR/
  $(basename "$NPZ")   the per-sweep snapshots
  <stem>_evolution.png   membership P per pool particle, one column per EM sweep
  <stem>_matrix.png      particle x candidate budget heatmap (final sweep)
EOS
if [[ -n "$COND_ID" ]]; then
    echo "  tail -F $REWARD_ROOT/logs/slurm-solve_condense-$COND_ID.out   # per-band table"
    echo "  $OUT_DIR/condensation/$(basename "${NPZ%.npz}")_condensation.json"
fi
