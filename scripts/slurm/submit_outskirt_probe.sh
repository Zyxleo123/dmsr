#!/bin/bash
# Why does Rockstar report the inner supervised sets and not the outskirt ones?
# (docs/sr2_outskirt_detection.md)
#
# The premise: `radial-placement-is-inverted` measured HR's median subhalo at
# 1.13 R_vir and every tuned arm's DETECTED catalog at 0.30-0.35, while the
# membership solve's optimised c_j sit at 1.42 R_vir against HR's 1.19. The
# candidates are at the right radii and the halo finder reports nothing there.
#
# This asks whether that is a BOUNDNESS failure or a PLACEMENT failure, by
# splitting the recovery rate into P(addressed) x P(hit | addressed) per radial
# bin, at fixed num_p. The two need different experiments, and a pooled rate
# cannot tell them apart.
#
# Two CPU stages, chained afterok. Nothing here recomputes a field.
#
#   bash scripts/slurm/submit_outskirt_probe.sh
#   OD_BOX=set10 bash scripts/slurm/submit_outskirt_probe.sh
#   ARMS="all_blocks_self all_blocks_reloc_B" bash scripts/slurm/submit_outskirt_probe.sh
#   PLOT_ONLY=1 bash scripts/slurm/submit_outskirt_probe.sh
#   DRY=1 bash scripts/slurm/submit_outskirt_probe.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
MG="$REWARD_ROOT/member_gather"
DRY="${DRY:-0}"
PLOT_ONLY="${PLOT_ONLY:-0}"

OD_BOX="${OD_BOX:-set9}"
OD_ADDRESS_TOL="${OD_ADDRESS_TOL:-0.30}"
# NOT `${TAG_FORMAT:-mgho_{arm}_{box}}`: bash closes the expansion
# at the first `}` inside the default, which silently yields the
# nonsense tag `mgho_{arm_{box}}` and finds no arm at all.
[ -n "${TAG_FORMAT:-}" ] || TAG_FORMAT='mgho_{arm}_{box}'

# Default to every arm that has BOTH an export for this box and a compare
# catalog: the probe needs the two joined, and an arm with only one of them
# would be skipped inside the job after spending queue time. Resolved here so
# the submitter's own output says exactly which arms will be read.
if [ -n "${ARMS:-}" ]; then
    SEL="$ARMS"
else
    SEL=""
    for _d in "$MG"/*/; do
        _a=$(basename "$_d")
        _tag=${TAG_FORMAT//\{arm\}/$_a}; _tag=${_tag//\{box\}/$OD_BOX}
        [[ -r "$_d/holdout_$OD_BOX/subhalos.json" ]] || continue
        [[ -r "$REWARD_ROOT/flow_rockstar/compare/${OD_BOX}__${_tag}.json" ]] || continue
        SEL="$SEL $_a"
    done
    SEL="${SEL# }"
fi

if [ -z "${SEL// }" ]; then
    echo "no arm under $MG has both holdout_$OD_BOX/subhalos.json and a"
    echo "compare/${OD_BOX}__${TAG_FORMAT}.json. Nothing to probe."
    echo "  (run the gate chain for an arm first: submit_gate_all_arms.sh)"
    exit 1
fi
echo "=== outskirt-detection probe -- box $OD_BOX"
echo "    arms:"; for _a in $SEL; do echo "      $_a"; done; echo

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" \
         "$REWARD_ROOT/outskirt_probe/$OD_BOX"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
ENVFILE="$REWARD_ROOT/env/outskirt_$STAMP.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_outskirt_probe.sh at $(date '+%F %T').
# List values QUOTED (env-file-list-values-must-be-quoted).
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
OD_BOX=$OD_BOX
OD_ARMS="$SEL"
OD_TAG_FORMAT=$TAG_FORMAT
OD_ADDRESS_TOL=$OD_ADDRESS_TOL
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

# Scrub SLURM_* so this works from inside an allocation as well as the login
# node (sbatch itself only works from the login node here).
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

if [ "$DRY" = "1" ]; then
    [ "$PLOT_ONLY" = "1" ] \
        || echo "DRY: sbatch scripts/slurm/outskirt_probe_cpu.sbatch $ENVFILE"
    echo "DRY: sbatch scripts/slurm/outskirt_plot_cpu.sbatch $ENVFILE"
    exit 0
fi

DEP=""
if [ "$PLOT_ONLY" != "1" ]; then
    J1=$(sbatch --parsable scripts/slurm/outskirt_probe_cpu.sbatch "$ENVFILE")
    echo "submitted outprobe -> job $J1 (cpu)"
    DEP="--dependency=afterok:$J1"
fi
J2=$(sbatch --parsable $DEP scripts/slurm/outskirt_plot_cpu.sbatch "$ENVFILE")
echo "submitted outplot  -> job $J2 (cpu${DEP:+, afterok:$J1})"
echo
echo "watch:"
[ "$PLOT_ONLY" = "1" ] \
    || echo "  tail -F $REWARD_ROOT/logs/slurm-outprobe-$J1.out"
echo "  tail -F $REWARD_ROOT/logs/slurm-outplot-$J2.out"
echo
echo "results:"
echo "  $REWARD_ROOT/outskirt_probe/$OD_BOX/summary.json"
for _a in $SEL; do
echo "  $REWARD_ROOT/outskirt_probe/$OD_BOX/${_a}.json"
done
echo "  $REWARD_ROOT/outskirt_probe/$OD_BOX/fig1_radial.png"
echo "  $REWARD_ROOT/outskirt_probe/$OD_BOX/fig2_address.png"
echo "  $REWARD_ROOT/outskirt_probe/$OD_BOX/fig3_contrast.png"
echo
echo "READ the address decomposition first. If P(addressed) carries the radial"
echo "trend, the outskirt failure is PLACEMENT and no boundness term will fix"
echo "it; if P(hit | addressed) carries it, it is CONSTRUCTION and the virial /"
echo "bound / rrms references are the lever."
