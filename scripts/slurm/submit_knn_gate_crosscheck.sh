#!/bin/bash
# Is the kNN statistic or the Rockstar count right where they disagree?
#   knnsplit (CPU): kNN terms in the spliced region for EVERY set9 arm field on disk
#   knngate  (CPU, afterok): catalog forensics for every gated arm + Spearman + figure
#
#   bash scripts/slurm/submit_knn_gate_crosscheck.sh
#   SKIP_KNN=1 bash scripts/slurm/submit_knn_gate_crosscheck.sh   # reuse an existing kNN JSON
#   DRY=1 bash scripts/slurm/submit_knn_gate_crosscheck.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
SKIP_KNN="${SKIP_KNN:-0}"
BOX="${BOX:-set9}"

F="$REWARD_ROOT/flow_rockstar/fields"
OUTDIR="$REWARD_ROOT/member_gather/knn_cdf"
REGION_JSON="${REGION_JSON:-$F/${BOX}__mgho_all_blocks_selfhostguard_${BOX}__seed0.json}"
KNN_JSON="${KNN_JSON:-$OUTDIR/knn_cdf_region_allarms_${BOX}.json}"
KG_OUT="${KG_OUT:-$OUTDIR/knn_gate_crosscheck_${BOX}.json}"
[[ -r "$REGION_JSON" ]] || { echo "no region json: $REGION_JSON"; exit 1; }

# Every spliced field on disk, labelled by its tag with mgho_ and _<box> stripped,
# which is the label knn_gate_crosscheck.py looks for. frozen -> "frozen".
FIELDS="hr=$ZFS/DMSR/paired_catnorm/hr/${BOX}.npy"
for p in "$F/${BOX}__mgho_"*"_${BOX}__seed0.npy"; do
    [[ -e "$p" ]] || continue
    tag=$(basename "$p" .npy); tag=${tag#${BOX}__}; tag=${tag%__seed0}
    label=${tag#mgho_}; label=${label%_${BOX}}
    FIELDS="$FIELDS $label=$p"
done
N_FIELDS=$(($(wc -w <<<"$FIELDS") - 1))
echo "$N_FIELDS arm fields on disk (incl. frozen)"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$OUTDIR"
ENVFILE="$REWARD_ROOT/env/knngate_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_knn_gate_crosscheck.sh at $(date '+%F %T').
# List values QUOTED (env-file-list-values-must-be-quoted).
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
KS_FIELDS="$FIELDS"
KS_CATALOG=$(ls $REWARD_ROOT/halos/${BOX}__hr__hr/*_rockstar/halos_*.ascii | head -1)
KS_OWNER=$REWARD_ROOT/halos_particles/${BOX}__hr__hr/${BOX}_hr_owner.npy
KS_OUT=$KNN_JSON
KS_REGION_JSON=$REGION_JSON
KS_BOUNDARY_TILES=0
KS_TUNED_LABEL=all_blocks_selfhostguard
KG_REGION_JSON=$REGION_JSON
KG_KNN_JSON=$KNN_JSON
KG_OUT=$KG_OUT
KG_BOX=$BOX
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
if [ "$DRY" = "1" ]; then
    [ "$SKIP_KNN" = "1" ] || echo "DRY: sbatch --time=06:00:00 scripts/slurm/knn_cdf_split_cpu.sbatch $ENVFILE"
    echo "DRY: sbatch [--dependency=afterok:<id>] scripts/slurm/knn_gate_crosscheck_cpu.sbatch $ENVFILE"
    exit 0
fi
DEP=()
if [ "$SKIP_KNN" != "1" ]; then
    # ~16 fields x ~1.5 min plus HR; 6 h is generous and a time limit costs only
    # the unfinished fields (the JSON is rewritten after every field).
    J1=$(sbatch --parsable --time=06:00:00 scripts/slurm/knn_cdf_split_cpu.sbatch "$ENVFILE")
    echo "submitted knnsplit (all arms) -> job $J1 (CPU)"
    echo "watch:   tail -F $REWARD_ROOT/logs/slurm-knnsplit-$J1.out"
    DEP=(--dependency=afterok:$J1)
fi
J2=$(sbatch --parsable "${DEP[@]}" scripts/slurm/knn_gate_crosscheck_cpu.sbatch "$ENVFILE")
echo "submitted knngate -> job $J2 (CPU${DEP:+, ${DEP[*]}})"
echo "watch:   tail -F $REWARD_ROOT/logs/slurm-knngate-$J2.out"
echo "result:  $KG_OUT"
echo "figure:  ${KG_OUT%.json}.png"
echo "kNN:     $KNN_JSON"
