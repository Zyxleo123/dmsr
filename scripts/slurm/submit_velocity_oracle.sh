#!/bin/bash
# Submit the velocity-temperature oracle (all CPU, nothing trained):
#
#   O  voracle   array, one Rockstar run per variant: a field whose local
#                velocity dispersion is rescaled to HR's stacked host-centric
#                profile (positions untouched)
#                  cs_vfix_all   coarse_soft, every radial bin        (+ field saved)
#                  cs_vfix_core  coarse_soft, r < 0.5 R_vir only (heat the core)
#                  cs_vfix_out   coarse_soft, r >= 0.5 R_vir only (cool the outskirts)
#                  frz_vfix_all  frozen SR2, every bin                (+ field saved)
#                  hr_vfix_all   HR itself, every bin -- the intervention's null
#   F  framechk  which transform maps the m2m e77 boxes onto HR's frame
#   C  placefor  placement forensics on the oracle catalogs (afterany O)
#
# Needs the HR host profile from submit_placement_forensics.sh (the target).
# Only calls sbatch; configuration in ONE env file passed positionally.
#
#   bash scripts/slurm/submit_velocity_oracle.sh
#   DRY=1 bash scripts/slurm/submit_velocity_oracle.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
R="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DATA="${DATA:-$ZFS/DMSR/paired_catnorm}"
DRY="${DRY:-0}"
BOX="${BOX:-set9}"
VO="$R/velocity_oracle"
OUT_DIR="$R/placement_forensics"

F="$R/flow_rockstar/fields"
CS="$F/${BOX}__mgho_all_blocks_sm_coarse_soft_${BOX}__seed0.npy"
FRZ="$F/${BOX}__mgho_frozen_${BOX}__seed0.npy"
HRF="$DATA/hr/${BOX}.npy"
VARIANTS="cs_vfix_all:$CS:all:save cs_vfix_core:$CS:core cs_vfix_out:$CS:out"
VARIANTS+=" frz_vfix_all:$FRZ:all:save hr_vfix_all:$HRF:all"
read -r -a _V <<< "$VARIANTS"
NV=${#_V[@]}

H="$R/flow_rockstar/halos"
one() { ls $1 2>/dev/null | head -1; }      # resolve a glob to one path (may be empty)
FC="frozen=$(one "$H/${BOX}__candidate__mgho_frozen_${BOX}/*/halos_*.ascii")"
FC+=",m2m_base_e77=$(one "$H/${BOX}__candidate__m2m_theirs_e77/*/halos_*.ascii")"
FC+=",m2m_knn_e77=$(one "$H/${BOX}__candidate__m2m_theirs_knn50_e77/*/halos_*.ascii")"
FC+=",m2m_mg_e77=$(one "$H/${BOX}__candidate__m2m_theirs_mg_e77/*/halos_*.ascii")"

EXTRA="frozen_gate=$H/${BOX}__candidate__mgho_frozen_${BOX}/*/halos_*.ascii"
for v in "${_V[@]}"; do
    name="${v%%:*}"
    EXTRA+=",$name=$VO/$name/${name}_rockstar/halos_*.ascii"
done

mkdir -p "$R/logs" "$R/env" "$VO"
ENVFILE="$R/env/voracle_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_velocity_oracle.sh at $(date '+%F %T')
PROJECT="$PROJECT"
ZFS="$ZFS"
DMSR_REWARD_ROOT="$R"
DATA="$DATA"
BOX="$BOX"
VO_VARIANTS="$VARIANTS"
VO_OUT_DIR="$VO"
VO_SPLIT="0.5"
FC_CANDS="$FC"
PF_ARMS="coarse_soft=mgho_all_blocks_sm_coarse_soft_${BOX}"
PF_EXTRA="$EXTRA"
PF_SUBFIND=" "
PF_OUT="$OUT_DIR/${BOX}_forensics_oracle.json"
EOT
echo "envfile: $ENVFILE"
cat "$ENVFILE"
echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
run() {
    echo "+ $*" >&2
    if [ "$DRY" = "1" ]; then echo "DRY_$RANDOM"; else "$@" | awk '{print $NF}'; fi
}

OID=$(run sbatch --array=0-$((NV - 1)) scripts/slurm/velocity_oracle_cpu.sbatch "$ENVFILE")
echo "O  voracle array ($NV variants) job $OID"
FID=$(run sbatch scripts/slurm/frame_offset_check_cpu.sbatch "$ENVFILE")
echo "F  framechk job $FID"
# afterany: a variant that dies leaves its catalog absent, and the forensics
# skip a missing extra by name -- the others are still worth reading.
CID=$(run sbatch --dependency=afterany:"$OID" \
      scripts/slurm/placement_forensics_cpu.sbatch "$ENVFILE")
echo "C  placefor (oracle catalogs) job $CID  (afterany:$OID)"

echo
echo "monitor:"
echo "  squeue -u \$USER -o '%.10i %.12j %.8T %.10M %R'"
echo "  tail -F \$(ls -t $R/logs/slurm-voracle-*.out | head -$NV)   # array tasks log by their own id"
echo "  tail -F $R/logs/slurm-framechk-$FID.out"
echo "  tail -F $R/logs/slurm-placefor-$CID.out"
echo
echo "results:"
for v in "${_V[@]}"; do
    name="${v%%:*}"
    echo "  $VO/$name/${name}_oracle.json   scale + achieved sigma_loc; catalog in ${name}_rockstar/"
done
echo "  $VO/${BOX}__cs_vfix_all__field.npy, ${BOX}__frz_vfix_all__field.npy   (for SUBFIND)"
echo "  $OUT_DIR/${BOX}_frame_check.json"
echo "  $OUT_DIR/${BOX}_forensics_oracle.json"
