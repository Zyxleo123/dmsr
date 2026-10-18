#!/bin/bash
# Submit the placement-forensics measurements (all CPU, nothing trained):
#
#   A  placefor     every gated set9 Rockstar catalog + SUBFIND npz + each arm's
#                   summary.json -> radial-by-mass, fragment structures,
#                   kinematics, outskirt mass shift, cross-arm correlations
#   S  chanswap     2 Rockstar runs on velocity-swapped fields:
#                     cs_disp_frzvel  coarse_soft displacement + frozen velocity
#                     frz_disp_csvel  frozen displacement + coarse_soft velocity
#   B  placefor     the same forensics on the two swap catalogs (afterok S)
#   P  hostprof     stacked host phase-space profiles, one field per task
#
# A, S and P are siblings (queue concurrently); only B waits.
# This submitter ONLY calls sbatch. Configuration goes into ONE timestamped env
# file passed as a POSITIONAL argument (never `sbatch --export`, which gets jobs
# requeued+held on this cluster); B's overrides travel as VAR=value arguments.
#
#   bash scripts/slurm/submit_placement_forensics.sh
#   DRY=1 bash scripts/slurm/submit_placement_forensics.sh
#   SKIP_SWAP=1 SKIP_PROFILES=1 bash scripts/slurm/submit_placement_forensics.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
R="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DATA="${DATA:-$ZFS/DMSR/paired_catnorm}"
DRY="${DRY:-0}"
SKIP_SWAP="${SKIP_SWAP:-0}"
SKIP_PROFILES="${SKIP_PROFILES:-0}"
BOX="${BOX:-set9}"
OUT_DIR="$R/placement_forensics"

F="$R/flow_rockstar/fields"
CS_FIELD="$F/${BOX}__mgho_all_blocks_sm_coarse_soft_${BOX}__seed0.npy"
FRZ_FIELD="$F/${BOX}__mgho_frozen_${BOX}__seed0.npy"
H="$R/flow_rockstar/halos"
FROZEN_CAT="$H/${BOX}__candidate__mgho_frozen_${BOX}/*/halos_*.ascii"
M2M_EXTRA="m2m_base_e77=$H/${BOX}__candidate__m2m_theirs_e77/*/halos_*.ascii"
M2M_EXTRA+=",m2m_knn_e77=$H/${BOX}__candidate__m2m_theirs_knn50_e77/*/halos_*.ascii"
M2M_EXTRA+=",m2m_mg_e77=$H/${BOX}__candidate__m2m_theirs_mg_e77/*/halos_*.ascii"
SW="$R/channel_swap"
SWAP_EXTRA="cs_disp_frzvel=$SW/${BOX}__cs_disp_frzvel/cs_disp_frzvel_rockstar/halos_*.ascii"
SWAP_EXTRA+=",frz_disp_csvel=$SW/${BOX}__frz_disp_csvel/frz_disp_csvel_rockstar/halos_*.ascii"

HP_FIELDS="hr=$DATA/hr/${BOX}.npy frozen=$FRZ_FIELD coarse_soft=$CS_FIELD"
HP_FIELDS+=" radparent=$F/${BOX}__mgho_all_blocks_radparent_${BOX}__seed0.npy"
HP_FIELDS+=" iso2k=$F/${BOX}__mgho_all_blocks_abl_shp_iso1_step2000_${BOX}__seed0.npy"
HP_FIELDS+=" m2m_base_e77=$R/m2m/boxes/m2m_theirs__state_77__${BOX}.npy"
HP_FIELDS+=" m2m_knn_e77=$R/m2m/boxes/m2m_theirs_knn50__state_77__${BOX}.npy"
HP_FIELDS+=" m2m_mg_e77=$R/m2m/boxes/m2m_theirs_mg__state_77__${BOX}.npy"
read -r -a _HPF <<< "$HP_FIELDS"
NPROF=${#_HPF[@]}

mkdir -p "$R/logs" "$R/env" "$OUT_DIR"
ENVFILE="$R/env/placefor_$(date +%Y%m%d_%H%M%S)_$$.env"
# Every value QUOTED: the preamble sources this file, and an unquoted list with
# spaces runs its second word as a command (env-file-list-values-must-be-quoted).
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_placement_forensics.sh at $(date '+%F %T')
PROJECT="$PROJECT"
ZFS="$ZFS"
DMSR_REWARD_ROOT="$R"
DATA="$DATA"
BOX="$BOX"
PF_ARMS="auto"
PF_EXTRA="frozen_gate=$FROZEN_CAT,$M2M_EXTRA"
PF_SUBFIND="auto"
PF_OUT="$OUT_DIR/${BOX}_forensics.json"
ARMS="cs_disp_frzvel:$CS_FIELD:$FRZ_FIELD frz_disp_csvel:$FRZ_FIELD:$CS_FIELD"
OVERWRITE="0"
BASE_SEED="0"
HP_FIELDS="$HP_FIELDS"
HP_OUT_DIR="$OUT_DIR/host_profiles"
EOT
echo "envfile: $ENVFILE"
cat "$ENVFILE"
echo

# B's overrides, as positional VAR=value arguments after the env file.
B_ARGS=("PF_ARMS=coarse_soft=mgho_all_blocks_sm_coarse_soft_${BOX}"
        "PF_EXTRA=frozen_gate=$FROZEN_CAT,$SWAP_EXTRA"
        "PF_SUBFIND= "
        "PF_OUT=$OUT_DIR/${BOX}_forensics_swap.json")

# Submitting from inside an allocation: scrub SLURM_* so nothing leaks.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

run() {  # echo the exact sbatch line, then run it (or not, under DRY)
    echo "+ $*" >&2
    if [ "$DRY" = "1" ]; then echo "DRY_$RANDOM"; else "$@" | awk '{print $NF}'; fi
}

AID=$(run sbatch scripts/slurm/placement_forensics_cpu.sbatch "$ENVFILE")
echo "A  placefor (all catalogs)  job $AID"
SID=""; BID=""
if [ "$SKIP_SWAP" != "1" ]; then
    SID=$(run sbatch --array=0-1 scripts/slurm/channel_swap_rockstar_cpu.sbatch "$ENVFILE")
    echo "S  chanswap array (2 swaps) job $SID"
    BID=$(run sbatch --dependency=afterok:"$SID" \
          scripts/slurm/placement_forensics_cpu.sbatch "$ENVFILE" "${B_ARGS[@]}")
    echo "B  placefor (swap catalogs) job $BID  (afterok:$SID)"
fi
PID=""
if [ "$SKIP_PROFILES" != "1" ]; then
    PID=$(run sbatch --array=0-$((NPROF - 1)) \
          scripts/slurm/host_phase_profile_cpu.sbatch "$ENVFILE")
    echo "P  hostprof array ($NPROF fields) job $PID"
fi

echo
echo "monitor:"
echo "  squeue -u \$USER -o '%.10i %.12j %.8T %.10M %R'"
echo "  tail -F $R/logs/slurm-placefor-$AID.out"
if [ -n "$SID" ]; then
    echo "  tail -F \$(ls -t $R/logs/slurm-chanswap-*.out | head -2)   # array tasks log by their own id"
    echo "  tail -F $R/logs/slurm-placefor-$BID.out"
fi
if [ -n "$PID" ]; then
    echo "  tail -F \$(ls -t $R/logs/slurm-hostprof-*.out | head -$NPROF)"
fi
echo
echo "results:"
echo "  $OUT_DIR/${BOX}_forensics.json        A: every catalog, SUBFIND, cross-arm table"
echo "  $OUT_DIR/${BOX}_forensics_swap.json   B: the two velocity swaps (+ coarse_soft, frozen)"
echo "  $SW/${BOX}__cs_disp_frzvel/ , $SW/${BOX}__frz_disp_csvel/   swap catalogs"
echo "  $OUT_DIR/host_profiles/${BOX}__<field>.{npz,json}   P: per-field profiles"
