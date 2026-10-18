#!/bin/bash
# Submit the IC-determinism test (all CPU, existing data, nothing trained): do the
# IC ripples finer than the LR grid fix where outskirt subhalos sit today?
#
#   R  icd_rip    r(k) HR vs dmo-256 vs frozen SR2: do the 256^3 and 512^3 runs
#                 share the fine ripples? (the gate the verdict reads)
#   S  icd_rs     Rockstar on the dmo-256 box                     (sibling of R)
#   M  icd_match  host-relative + same-IC-patch match of HR's objects, frozen SR2
#                 as the LR-only control                          (afterany R, afterok S)
#   T  icd_track  same matter + same-IC-patch object, radius vs direction (afterok S)
#   P  icd_plot   figures from the JSONs                          (afterany M, T)
#
# Only calls sbatch. Configuration goes in ONE env file passed positionally; the
# defaults live in scripts/slurm/_icdet_common.sh, and any ICD_* variable set when
# calling this script is written into the env file.
#
#   bash scripts/slurm/submit_ic_determinism.sh
#   DRY=1 bash scripts/slurm/submit_ic_determinism.sh
#   BOX=set10 bash scripts/slurm/submit_ic_determinism.sh
#   ICD_OBJ_MIN_NP=800 bash scripts/slurm/submit_ic_determinism.sh
#   ICD_DMO_RES=128 ICD_GATE_LO=2.2 ICD_GATE_HI=3.6 bash scripts/slurm/submit_ic_determinism.sh
#     (a coarser run: outputs get _dmo128 in their names; the gate band must sit
#      below that run's own grid limit, 4.0 h/Mpc for 128^3)
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
R="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DATA="${DATA:-$ZFS/DMSR/paired_catnorm}"
DRY="${DRY:-0}"
BOX="${BOX:-set9}"
one() { ls $1 2>/dev/null | head -1 || true; }

DMO="$ZFS/DMSR/int_redshift_same_cosmology/dmo-${ICD_DMO_RES:-256}/$BOX/PART_009"
HRF="$DATA/hr/$BOX.npy"
SRF=$(one "$R/cache/sr2_base/${BOX}_seed0_*.npy")
HRC=$(one "$R/halos/${BOX}__hr__hr/*_rockstar/halos_*.ascii")
SRC=$(one "$R/halos/${BOX}__base__base/*_rockstar/halos_*.ascii")
for f in "$DMO/disp.npy" "$DMO/vel.npy" "$HRF" "$SRF" "$HRC" "$SRC"; do
    [ -n "$f" ] && [ -e "$f" ] || { echo "missing input: '${f}'"; exit 1; }
done

ICD="${ICD_ROOT:-$R/ic_determinism}"
TAG="${ICD_TAG:-dmo${ICD_DMO_RES:-256}}"
if [ "$TAG" = "dmo256" ]; then SFX=""; else SFX="_$TAG"; fi
mkdir -p "$R/logs" "$R/env" "$ICD"
STAMP=$(date +%Y%m%d_%H%M%S)_$$
ENVFILE="$R/env/icdet_${STAMP}.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_ic_determinism.sh at $(date '+%F %T')
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

RIP=$(run sbatch scripts/slurm/icdet_ripples_cpu.sbatch "$ENVFILE")
echo "R  icd_rip    job $RIP"
RS=$(run sbatch scripts/slurm/icdet_rockstar_cpu.sbatch "$ENVFILE")
echo "S  icd_rs     job $RS"
MT=$(run sbatch --dependency=afterany:"$RIP",afterok:"$RS" \
     scripts/slurm/icdet_match_cpu.sbatch "$ENVFILE")
echo "M  icd_match  job $MT  (afterany:$RIP, afterok:$RS)"
TR=$(run sbatch --dependency=afterok:"$RS" scripts/slurm/icdet_track_cpu.sbatch "$ENVFILE")
echo "T  icd_track  job $TR  (afterok:$RS)"
PL=$(run sbatch --dependency=afterany:"$MT":"$TR" scripts/slurm/icdet_plot_cpu.sbatch "$ENVFILE")
echo "P  icd_plot   job $PL  (afterany:$MT:$TR)"

L="$R/logs"
echo
echo "monitor:"
echo "  squeue -u \$USER -o '%.10i %.12j %.8T %.10M %R'"
echo "  tail -F $L/slurm-icd_rip-$RIP.out"
echo "  tail -F $L/slurm-icd_rs-$RS.out"
echo "  tail -F $L/slurm-icd_match-$MT.out"
echo "  tail -F $L/slurm-icd_track-$TR.out"
echo "  tail -F $L/slurm-icd_plot-$PL.out"
echo "  tail -F $L/slurm-icd_{rip-$RIP,rs-$RS,match-$MT,track-$TR,plot-$PL}.out"
echo
echo "results:"
echo "  $ICD/${BOX}${SFX}_ripples.json   r(k) + gate: does the $TAG run share HR's fine ripples?"
echo "  $ICD/${BOX}__$TAG/${TAG}_rockstar/halos_0.0.ascii   (+ rockstar_summary.json)"
echo "  $ICD/${BOX}${SFX}_match.json   per r/R_vir: match, null, same IC patch; VERDICT"
echo "  $ICD/${BOX}${SFX}_match.objects.npz   per-object rows (re-threshold without re-matching)"
echo "  $ICD/${BOX}${SFX}_track.json   same matter + same-patch radius vs direction"
echo "  $ICD/${BOX}${SFX}_{ic_determinism,track}.png   (redraw: sbatch scripts/slurm/icdet_plot_cpu.sbatch $ENVFILE)"
