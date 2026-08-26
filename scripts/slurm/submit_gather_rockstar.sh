#!/bin/bash
# Is the gathered substructure BOUND? Splice -> real Rockstar -> compare.
#
#   splice tuned tiles into the frozen box (CPU, minutes)
#     --afterok--> full-box Rockstar, frozen config (CPU, hours)
#     --afterok--> compare against base + HR catalogs (CPU, minutes)
#
# The fine-tune supervised four tiles of one cluster, and Rockstar needs a whole
# periodic box, so the box under test is: frozen SR2 everywhere, fine-tuned
# output in those four tiles. Every catalog difference against set8__base__base
# is then attributable to the trained tiles -- and no GPU is needed, because the
# run already saved its final tiles.
#
# This deliberately does NOT measure the collateral damage: the field outside the
# four tiles is frozen here by construction. The host-mass-function gate needs a
# whole-box regeneration, which is a different job.
#
# Only calls sbatch. Config travels in ONE timestamped env file passed as a
# POSITIONAL argument -- never `sbatch --export`.
#
#   HG_RUN_DIR=<...> bash scripts/slurm/submit_gather_rockstar.sh
#   HG_WHICH=hr  HG_RUN_DIR=<...> bash scripts/slurm/submit_gather_rockstar.sh
#   COMPARE_ONLY=1 HG_RS_TAG=<tag> bash scripts/slurm/submit_gather_rockstar.sh
#   DRY=1 HG_RUN_DIR=<...> bash scripts/slurm/submit_gather_rockstar.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
COMPARE_ONLY="${COMPARE_ONLY:-0}"

HG_BOX="${HG_BOX:-set8}"
HG_SEED="${HG_SEED:-0}"
HG_HOST_ID="${HG_HOST_ID:-271800}"
HG_MIN_P="${HG_MIN_P:-50}"
# `out` is the fine-tuned field. `frozen` rebuilds the control on the SAME splice
# edges, and `hr` the ceiling -- both useful if an edge artifact is suspected.
HG_WHICH="${HG_WHICH:-out}"
HG_RUN_DIR="${HG_RUN_DIR:-$REWARD_ROOT/host_gather/set8_h271800_fine_anchored}"
_RUN_NAME="$(basename "$HG_RUN_DIR")"
_SUFFIX=""; [ "$HG_WHICH" != "out" ] && _SUFFIX="_$HG_WHICH"
HG_RS_TAG="${HG_RS_TAG:-gather_${_RUN_NAME}${_SUFFIX}}"
FIELD="$REWARD_ROOT/flow_rockstar/fields/${HG_BOX}__${HG_RS_TAG}__seed${HG_SEED}.npy"

if [ "$COMPARE_ONLY" != "1" ] && [ ! -f "$HG_RUN_DIR/tiles.npz" ]; then
    echo "no tiles.npz under $HG_RUN_DIR -- point HG_RUN_DIR at a finished run" >&2
    exit 1
fi

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" \
         "$REWARD_ROOT/flow_rockstar/fields" "$REWARD_ROOT/flow_rockstar/logs"
ENVFILE="$REWARD_ROOT/env/gather_rockstar_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_gather_rockstar.sh at $(date '+%F %T');
# sourced by the job preambles as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
HG_BOX=$HG_BOX
HG_SEED=$HG_SEED
HG_RUN_DIR=$HG_RUN_DIR
HG_WHICH=$HG_WHICH
HG_RS_TAG=$HG_RS_TAG
HG_HOST_ID=$HG_HOST_ID
HG_MIN_P=$HG_MIN_P
EOT

echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo
echo "field:   $FIELD"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

sub() {
    if [ "$DRY" = "1" ]; then echo "DRY: sbatch $*" >&2; echo "DRYID"
    else sbatch --parsable "$@"; fi
}

SPLICE=""; RS=""
if [ "$COMPARE_ONLY" != "1" ]; then
    SPLICE=$(sub scripts/slurm/gather_splice_cpu.sbatch "$ENVFILE")
    echo "submitted splice   -> job $SPLICE (CPU)"

    # The Rockstar stage is the existing generic job: it reads a field .npy and a
    # tag, and writes the catalog in the same frozen format as HR and base. It
    # takes its own variable names, so they are passed explicitly.
    RS=$(sub --dependency=afterok:"$SPLICE" \
         scripts/slurm/flow_rockstar_catalog_cpu.sbatch \
         "BOX=$HG_BOX" "TAG=$HG_RS_TAG" "SEED=$HG_SEED" "FIELD_OUT=$FIELD")
    echo "submitted rockstar -> job $RS (CPU, afterok:$SPLICE)"
fi

DEP=(); [ -n "$RS" ] && DEP=(--dependency=afterok:"$RS")
CMP=$(sub "${DEP[@]}" scripts/slurm/gather_compare_cpu.sbatch "$ENVFILE")
echo "submitted compare  -> job $CMP (CPU${RS:+, afterok:$RS})"

echo
[ -n "$SPLICE" ] && echo "watch splice:   tail -F $REWARD_ROOT/logs/slurm-hg_splice-$SPLICE.out"
[ -n "$RS" ] && echo "watch rockstar: tail -F $REWARD_ROOT/flow_rockstar/logs/slurm-flow_rs_cat-$RS.out"
echo "watch compare:  tail -F $REWARD_ROOT/logs/slurm-hg_cmp-$CMP.out"
echo
echo "result: $REWARD_ROOT/flow_rockstar/compare/${HG_BOX}__${HG_RS_TAG}.json"
echo "  subhalos_within_rvir   HR vs frozen vs tuned, binned by particle count"
echo "  host                   its mass and centre must not have moved"
echo "  shell_profile          the splice-edge artifact check -- read it FIRST"
echo "  whole_box              hosts >= 200p, which the gate wants unchanged"
