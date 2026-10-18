#!/bin/bash
# Submit the binding oracle (all CPU, nothing trained): does making each outskirt
# bump self-bound by its OWN gravity turn SR bumps into HR's satellites?
#
#   B  bindorc   one job, variants in sequence (cpu 50-job cap):
#                  hr_bindstats  HR, criterion only: does "needs KE removed to be
#                                bound" separate HR's converting bumps? (no finder)
#                  cs_bind_hr    coarse_soft, per band only HR's own conversion
#                                rate, easiest bumps first      (+ field saved)
#                  cs_bind_all   coarse_soft, every bump: the ceiling (+ field saved)
#                  frz_bind_hr   frozen SR2 at HR's rate (Rockstar only)
#   R  bindread  gate verdicts + peak_supply 0.5-1 / 1-3 R_vir     (afterok B)
#   S  SUBFIND   snapshot + run on the two saved fields            (afterok B)
#   C  placefor  SUBFIND + Rockstar placement on everything        (afterany S)
#
# Bumps are the low-contrast census (>3x, host_profiles_peaks_lo). Only calls
# sbatch; configuration in ONE env file passed positionally.
#
#   bash scripts/slurm/submit_binding_oracle.sh
#   DRY=1 bash scripts/slurm/submit_binding_oracle.sh
#   BO_F_ESC=0.7 TAG=_f07 bash scripts/slurm/submit_binding_oracle.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
R="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DATA="${DATA:-$ZFS/DMSR/paired_catnorm}"
DRY="${DRY:-0}"
BOX="${BOX:-set9}"
TAG="${TAG:-}"
BO="$R/binding_oracle"
SF="$R/subfind"
P="$R/placement_forensics/host_profiles_peaks_lo"
F="$R/flow_rockstar/fields"
H="$R/flow_rockstar/halos"
KO="$R/knot_oracle"
one() { ls $1 2>/dev/null | head -1; }

HRF="$DATA/hr/${BOX}.npy"
CSF="$F/${BOX}__mgho_all_blocks_sm_coarse_soft_${BOX}__seed0.npy"
FRF="$F/${BOX}__mgho_frozen_${BOX}__seed0.npy"
HRC=$(one "$R/halos/${BOX}__hr__hr/*_rockstar/halos_*.ascii")
CSC=$(one "$H/${BOX}__candidate__mgho_all_blocks_sm_coarse_soft_${BOX}/*/halos_*.ascii")
FRC=$(one "$H/${BOX}__candidate__mgho_frozen_${BOX}/*/halos_*.ascii")
for f in "$HRF" "$CSF" "$FRF" "$HRC" "$CSC" "$FRC" "$P/${BOX}__hr_peaks.npz" \
         "$P/${BOX}__coarse_soft_peaks.npz" "$P/${BOX}__frozen_peaks.npz"; do
    [ -n "$f" ] && [ -e "$f" ] || { echo "missing input: '${f}'"; exit 1; }
done

V="hr_bindstats$TAG:$HRF:$P/${BOX}__hr_peaks.npz:$HRC:all:stats"
V+=" cs_bind_hr$TAG:$CSF:$P/${BOX}__coarse_soft_peaks.npz:$CSC:hr:save"
V+=" cs_bind_all$TAG:$CSF:$P/${BOX}__coarse_soft_peaks.npz:$CSC:all:save"
V+=" frz_bind_hr$TAG:$FRF:$P/${BOX}__frozen_peaks.npz:$FRC:hr:run"
CST="mgho_all_blocks_sm_coarse_soft_${BOX}"
VERD="cs_bind_hr$TAG:$CST:coarse_soft cs_bind_all$TAG:$CST:coarse_soft"
VERD+=" frz_bind_hr$TAG:mgho_frozen_${BOX}:frozen"
SUP="hr=$P/${BOX}__hr_peaks.npz:$HRC,frozen=$P/${BOX}__frozen_peaks.npz:$FRC"
SUP+=",coarse_soft=$P/${BOX}__coarse_soft_peaks.npz:$CSC"
SUP+=",cs_k256_lo=$P/${BOX}__coarse_soft_peaks.npz:$KO/cs_k256_lo/cs_k256_lo_rockstar/halos_0.0.ascii"
for n in cs_bind_hr cs_bind_all; do
    SUP+=",$n$TAG=$P/${BOX}__coarse_soft_peaks.npz:$BO/$n$TAG/$n${TAG}_rockstar/halos_0.0.ascii"
done
SUP+=",frz_bind_hr$TAG=$P/${BOX}__frozen_peaks.npz:$BO/frz_bind_hr$TAG/frz_bind_hr${TAG}_rockstar/halos_0.0.ascii"

mkdir -p "$R/logs" "$R/env" "$BO" "$SF/env"
STAMP=$(date +%Y%m%d_%H%M%S)_$$
ENVFILE="$R/env/bindorc_${STAMP}.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_binding_oracle.sh at $(date '+%F %T')
PROJECT="$PROJECT"
ZFS="$ZFS"
DMSR_REWARD_ROOT="$R"
DATA="$DATA"
BOX="$BOX"
BO_VARIANTS="$V"
BO_OUT_DIR="$BO"
BO_F_ESC="${BO_F_ESC:-0.9}"
BO_AP_CONTRAST="${BO_AP_CONTRAST:-3}"
BO_HR_PEAKS="$P/${BOX}__hr_peaks.npz"
BO_VERDICTS="$VERD"
BO_SUPPLY_FIELDS="$SUP"
BO_HR_SUBFIND="$SF/${BOX}__hr/subfind_catalog.npz"
EOT
echo "envfile: $ENVFILE"
cat "$ENVFILE"
echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
run() {
    echo "+ $*" >&2
    if [ "$DRY" = "1" ]; then echo "DRY_$RANDOM"; else "$@" | awk '{print $NF}'; fi
}

BID=$(run sbatch scripts/slurm/binding_oracle_cpu.sbatch "$ENVFILE")
echo "B  bindorc job $BID"
RID=$(run sbatch --dependency=afterok:"$BID" scripts/slurm/binding_oracle_read_cpu.sbatch "$ENVFILE")
echo "R  bindread job $RID  (afterok:$BID)"

RUNS=()
for n in cs_bind_hr cs_bind_all; do
    ENVF="$SF/env/${n}${TAG}_${STAMP}.env"
    cat > "$ENVF" <<EOT
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$R
SUBFIND_ROOT=$SF
SF_BOX=$BOX
SF_SEED=0
SF_MIN_P=50
SF_NRANKS=44
SF_MAX_MEM_MB=13000
SF_NAME=${BOX}__$n$TAG
SF_SOURCE=candidate
SF_TAG=$n$TAG
SF_HOST_ID=168880
SF_FIELD=$BO/${BOX}__$n${TAG}__field.npy
EOT
    a=$(run sbatch --dependency=afterok:"$BID" scripts/slurm/subfind_snapshot_cpu.sbatch "$ENVF")
    b=$(run sbatch --dependency=afterok:"$a" scripts/slurm/subfind_run_cpu.sbatch "$ENVF")
    echo "S  $n$TAG: sf_snap $a -> sf_run $b"
    RUNS+=("$b")
done
DEP=$(IFS=:; echo "${RUNS[*]}")
SFL="hr=$SF/${BOX}__hr/subfind_catalog.npz,frozen=$SF/${BOX}__frozen/subfind_catalog.npz"
SFL+=",coarse_soft=$SF/${BOX}__${CST}/subfind_catalog.npz"
SFL+=",combo_cs=$SF/${BOX}__combo_cs/subfind_catalog.npz,cs_k256_lo=$SF/${BOX}__cs_k256_lo/subfind_catalog.npz"
SFL+=",cs_bind_hr$TAG=$SF/${BOX}__cs_bind_hr$TAG/subfind_catalog.npz"
SFL+=",cs_bind_all$TAG=$SF/${BOX}__cs_bind_all$TAG/subfind_catalog.npz"
EXTRA="frozen_gate=$H/${BOX}__candidate__mgho_frozen_${BOX}/*/halos_*.ascii"
for n in cs_bind_hr cs_bind_all frz_bind_hr; do
    EXTRA+=",$n$TAG=$BO/$n$TAG/$n${TAG}_rockstar/halos_*.ascii"
done
# afterany: a SUBFIND that dies leaves its catalog absent and the forensics skip
# it by name -- the rest is still worth reading.
CID=$(run sbatch --dependency=afterany:"$DEP" scripts/slurm/placement_forensics_cpu.sbatch \
      "$ENVFILE" "PF_ARMS=coarse_soft=$CST" "PF_EXTRA=$EXTRA" "PF_SUBFIND=$SFL" \
      "PF_OUT=$R/placement_forensics/${BOX}_forensics_binding$TAG.json")
echo "C  placefor job $CID  (afterany:$DEP)"

echo
echo "monitor:"
echo "  squeue -u \$USER -o '%.10i %.12j %.8T %.10M %R'"
echo "  tail -F $R/logs/slurm-bindorc-$BID.out"
echo "  tail -F $R/logs/slurm-bindread-$RID.out"
echo "  tail -F $SF/logs/slurm-sf_run-{${RUNS[0]},${RUNS[1]}}.out"
echo "  tail -F $R/logs/slurm-placefor-$CID.out"
echo
echo "results:"
echo "  $BO/hr_bindstats$TAG/hr_bindstats${TAG}_binding_oracle.json   per-band criterion on HR (AUC, converted vs not)"
for n in cs_bind_hr cs_bind_all frz_bind_hr; do
    echo "  $BO/$n$TAG/${n}${TAG}_binding_oracle.json   + verdict_vs_*/ ; catalog in ${n}${TAG}_rockstar/"
done
echo "  $BO/${BOX}_peak_supply_0.5_1.0.json, ${BOX}_peak_supply_1.0_3.0.json"
echo "  $R/placement_forensics/${BOX}_forensics_binding$TAG.json   (SUBFIND + Rockstar core/outskirt)"
