#!/bin/bash
# Render the talk figures from the SAVED all-hosts report + host forensics.
# Cheap CPU render; re-run any time the diagnostics are re-run.
#
#   bash scripts/slurm/submit_gather_all_hosts_plot.sh
#   DRY=1 bash scripts/slurm/submit_gather_all_hosts_plot.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

AH="$REWARD_ROOT/member_gather/all_hosts"
AHP_REPORT="${AHP_REPORT:-$AH/report.json}"
AHP_OUT="${AHP_OUT:-$AH/figures}"
# The forensic jsons that exist; each is arm=path. Only self/selfhostguard were
# run -- the plotter skips any that are missing.
FOR_SELF="$REWARD_ROOT/member_gather/all_blocks_self/holdout_set9/forensics/host_forensics_set9.json"
FOR_GUARD="$REWARD_ROOT/member_gather/all_blocks_selfhostguard/holdout_set9/forensics/host_forensics_set9.json"
AHP_FORENSICS="${AHP_FORENSICS:-self=$FOR_SELF selfhostguard=$FOR_GUARD}"
AHP_RADIAL_DIR="${AHP_RADIAL_DIR:-$REWARD_ROOT/subhalo_radial/set9}"
# Only self/selfhostguard/smallhalo were run on both set9 and set10.
AHP_TUNED_ARM="${AHP_TUNED_ARM:-all_blocks_self}"

[ -f "$AHP_REPORT" ] || { echo "missing input: $AHP_REPORT" >&2; exit 1; }
mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$AHP_OUT"

ENVFILE="$REWARD_ROOT/env/ah_plot_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_gather_all_hosts_plot.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument. AHP_FORENSICS is a
# space-separated list of arm=path pairs (a file, so spaces travel fine).
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
AHP_REPORT=$AHP_REPORT
AHP_OUT=$AHP_OUT
AHP_FORENSICS="$AHP_FORENSICS"
AHP_RADIAL_DIR=$AHP_RADIAL_DIR
AHP_TUNED_ARM=$AHP_TUNED_ARM
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch scripts/slurm/gather_all_hosts_plot_cpu.sbatch $ENVFILE"; exit 0
fi
JID=$(sbatch --parsable scripts/slurm/gather_all_hosts_plot_cpu.sbatch "$ENVFILE")
echo "submitted ah_plot -> job $JID (CPU)"
echo
echo "watch:   tail -F $REWARD_ROOT/logs/slurm-ah_plot-$JID.out"
echo "figures: $AHP_OUT/"
echo "  sub_mass_function.png     subhalo count per num_p bin, HR/base/tuned (+ _log)"
echo "  fig1_abundance.png    subs/host as % of HR per arm + host-survival cost"
echo "  fig2_occupation.png   mean subs/host vs host mass (cluster-scale + overshoot)"
echo "  fig3_tradeoff.png     abundance vs host survival, per arm (the Pareto view)"
echo "  fig4_forensic.png     lost-host fates, genuine vs bookkeeping, self vs guard"
echo "  fig5_fate_by_mass.png self's lost hosts by mass bin x fate"
