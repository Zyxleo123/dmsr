#!/bin/bash
# Why does SUBFIND refuse the arms' subhalos? Run the forensic on every field.
#
# HR is not optional here. It is the control that separates "these objects are
# bad" from "SUBFIND is strict": in the 200-2000p band HR confirms 86-100%, so
# whatever distinguishes confirmed from unconfirmed THERE is the finder's own
# selection, and whatever is different about the arms on top of that is theirs.
#
#   bash scripts/slurm/submit_subfind_forensics.sh
#   DRY=1 bash scripts/slurm/submit_subfind_forensics.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
SUBFIND_ROOT="${SUBFIND_ROOT:-$REWARD_ROOT/subfind}"
FIELDS="$REWARD_ROOT/flow_rockstar/fields"
DATA_ROOT="${DATA_ROOT:-$ZFS/DMSR/paired_catnorm}"
DRY="${DRY:-0}"
SF_BOX="${SF_BOX:-set9}"
SF_HOST_ID="${SF_HOST_ID:-168880}"
SF_BAND_MIN="${SF_BAND_MIN:-200}"
SF_BAND_MAX="${SF_BAND_MAX:-2000}"
SF_TARGETS="${SF_TARGETS:-hr frozen self soft knn}"

declare -A ARM_OF=([self]=all_blocks_self [soft]=all_blocks_sm_coarse_soft \
                   [knn]=all_blocks_knn6d_w5)

mkdir -p "$SUBFIND_ROOT/logs" "$SUBFIND_ROOT/env" "$SUBFIND_ROOT/forensics"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

declare -a WATCH=()
for t in $SF_TARGETS; do
    case "$t" in
        hr)     NAME="${SF_BOX}__hr";     SRC=hr;   TAG="mgho_all_blocks_self_${SF_BOX}"
                FIELD="$DATA_ROOT/hr/${SF_BOX}.npy" ;;
        frozen) NAME="${SF_BOX}__frozen"; SRC=base; TAG="mgho_all_blocks_self_${SF_BOX}"
                FIELD="$FIELDS/${SF_BOX}__mgho_frozen_${SF_BOX}__seed0.npy" ;;
        *)      ARM="${ARM_OF[$t]:-}"; [ -n "$ARM" ] || { echo "unknown '$t'" >&2; exit 1; }
                TAG="mgho_${ARM}_${SF_BOX}"; NAME="${SF_BOX}__${TAG}"; SRC=cand
                FIELD="$FIELDS/${SF_BOX}__${TAG}__seed0.npy" ;;
    esac
    for f in "$SUBFIND_ROOT/$NAME/subfind_catalog.npz" "$FIELD"; do
        [ -f "$f" ] || { echo "──────── $t: missing $f -- SKIPPED"; continue 2; }
    done
    ENVF="$SUBFIND_ROOT/env/forensic_${t}_${STAMP}.env"
    cat > "$ENVF" <<EOT
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
SUBFIND_ROOT=$SUBFIND_ROOT
SF_BOX=$SF_BOX
SF_NAME=$NAME
SF_TAG=$TAG
SF_SOURCE=$SRC
SF_FIELD=$FIELD
SF_HOST_ID=$SF_HOST_ID
SF_BAND_MIN=$SF_BAND_MIN
SF_BAND_MAX=$SF_BAND_MAX
SF_FOR_OUT=$SUBFIND_ROOT/forensics/${SF_BOX}__${t}
EOT
    echo "──────── $t  (source $SRC)"
    if [ "$DRY" = "1" ]; then echo "  DRY: sbatch subfind_forensics_cpu.sbatch $ENVF"; continue; fi
    J=$(sbatch --parsable scripts/slurm/subfind_forensics_cpu.sbatch "$ENVF"); J="${J%%;*}"
    echo "  forensic -> job $J"
    WATCH+=("sf_forensic:$J")
done

echo
echo "watch:"
for w in "${WATCH[@]}"; do echo "  tail -F $SUBFIND_ROOT/logs/slurm-${w%%:*}-${w##*:}.out"; done
echo
echo "results: $SUBFIND_ROOT/forensics/${SF_BOX}__<t>.json"
echo "  read 'summary': medians of virial_ratio and r_half_over_r, confirmed vs not."
echo "  HR is the control -- compare each arm against it, not against 1."
