#!/bin/bash
# Submit the IC-determinism TRACK stage (CPU, existing data; needs the dmo-256
# Rockstar catalog from submit_ic_determinism.sh):
#
#   T  icd_track  (A) where HR's object particles end up in dmo-256 / frozen SR2
#                 (B) the same-IC-patch object: same radius vs same direction
#   P  icd_plot   both figures, redrawn from the JSONs              (afterany T)
#
# Only calls sbatch. Configuration goes in ONE env file passed positionally; the
# defaults live in scripts/slurm/_icdet_common.sh, and any ICD_* variable set when
# calling this script is written into the env file.
#
#   bash scripts/slurm/submit_ic_track.sh
#   DRY=1 bash scripts/slurm/submit_ic_track.sh
#   ICD_RADIUS_TOL=1.5 bash scripts/slurm/submit_ic_track.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
R="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DATA="${DATA:-$ZFS/DMSR/paired_catnorm}"
DRY="${DRY:-0}"
BOX="${BOX:-set9}"
one() { ls $1 2>/dev/null | head -1 || true; }

ICD="${ICD_ROOT:-$R/ic_determinism}"
TAG="${ICD_TAG:-dmo${ICD_DMO_RES:-256}}"
if [ "$TAG" = "dmo256" ]; then SFX=""; else SFX="_$TAG"; fi
DMO="$ZFS/DMSR/int_redshift_same_cosmology/dmo-${ICD_DMO_RES:-256}/$BOX/PART_009"
DMC="$ICD/${BOX}__$TAG/${TAG}_rockstar/halos_0.0.ascii"
HRF="$DATA/hr/$BOX.npy"
SRF=$(one "$R/cache/sr2_base/${BOX}_seed0_*.npy")
HRC=$(one "$R/halos/${BOX}__hr__hr/*_rockstar/halos_*.ascii")
SRC=$(one "$R/halos/${BOX}__base__base/*_rockstar/halos_*.ascii")
for f in "$DMO/disp.npy" "$DMC" "$HRF" "$SRF" "$HRC" "$SRC"; do
    [ -n "$f" ] && [ -e "$f" ] || {
        echo "missing input: '${f}' (the dmo-256 catalog comes from submit_ic_determinism.sh)"
        exit 1; }
done

mkdir -p "$R/logs" "$R/env" "$ICD"
STAMP=$(date +%Y%m%d_%H%M%S)_$$
ENVFILE="$R/env/icdtrack_${STAMP}.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_ic_track.sh at $(date '+%F %T')
PROJECT="$PROJECT"
ZFS="$ZFS"
DMSR_REWARD_ROOT="$R"
DATA="$DATA"
BOX="$BOX"
ICD_ROOT="$ICD"
ICD_DMO_DIR="$DMO"
ICD_SR2_FIELD="$SRF"
ICD_HR_CATALOG="$HRC"
ICD_SR2_CATALOG="$SRC"
EOT
for v in $(compgen -v | grep '^ICD_' || true); do
    case "$v" in
        ICD_ROOT|ICD_DMO_DIR|ICD_SR2_FIELD|ICD_HR_CATALOG|ICD_SR2_CATALOG) continue ;;
    esac
    printf '%s="%s"\n' "$v" "${!v}" >> "$ENVFILE"
done
echo "envfile: $ENVFILE"
cat "$ENVFILE"
echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
run() {
    echo "+ $*" >&2
    if [ "$DRY" = "1" ]; then echo "DRY_$RANDOM"; else "$@" | awk '{print $NF}'; fi
}

TR=$(run sbatch scripts/slurm/icdet_track_cpu.sbatch "$ENVFILE")
echo "T  icd_track  job $TR"
PL=$(run sbatch --dependency=afterany:"$TR" scripts/slurm/icdet_plot_cpu.sbatch "$ENVFILE")
echo "P  icd_plot   job $PL  (afterany:$TR)"

L="$R/logs"
echo
echo "monitor:"
echo "  squeue -u \$USER -o '%.10i %.12j %.8T %.10M %R'"
echo "  tail -F $L/slurm-icd_track-$TR.out"
echo "  tail -F $L/slurm-icd_plot-$PL.out"
echo "  tail -F $L/slurm-icd_{track-$TR,plot-$PL}.out"
echo
echo "results:"
echo "  $ICD/${BOX}${SFX}_track.json   per r/R_vir: (A) same-matter offset/compactness, (B) same-patch radius vs direction"
echo "  $ICD/${BOX}${SFX}_track.objects.npz   per-object rows"
echo "  $ICD/${BOX}${SFX}_track.png   (redraw: sbatch scripts/slurm/icdet_plot_cpu.sbatch $ENVFILE)"
