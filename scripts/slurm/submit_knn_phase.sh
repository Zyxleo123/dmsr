#!/bin/bash
# kNN round 5: is PHASE-SPACE kNN the statistic position kNN was not?
#   knnsplit x N (CPU): 6D kNN terms in the spliced set9 region for EVERY arm field,
#                       one job per lambda (km/s per Mpc/h), boundary per class
#   knnsplit 3D  (CPU): hr/frozen/selfhostguard only, per-class boundary = C's 3D reference
#   knnphasev    (CPU, afterany all): stop rules A/B/C -> knn_phase_verdict_<box>.json
# Gate counts for B come from round 4's knn_gate_crosscheck_<box>.json (must exist).
#
#   bash scripts/slurm/submit_knn_phase.sh
#   LAMBDAS="1000" bash scripts/slurm/submit_knn_phase.sh
#   RESUME=1 bash scripts/slurm/submit_knn_phase.sh   # after a time limit: keep measured fields
#   Round 5b (pre-registered after round 5: k=128 had the wrong sign, B rose with lambda):
#   LAMBDAS="0 2000 4000 8000" KS="8 16 32 64" KNN_3D_REF=0 bash scripts/slurm/submit_knn_phase.sh
#   lambda 0 = position-only over ALL arms at the same ranks (the fair 3D baseline).
#   DRY=1 bash scripts/slurm/submit_knn_phase.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
RESUME="${RESUME:-0}"
BOX="${BOX:-set9}"
# V_vir/r_vir ~ 10 H0 = 1000 km/s per Mpc/h (Delta=200c); bracket by 2x each way.
LAMBDAS="${LAMBDAS:-500 1000 2000}"
KNN_3D_REF="${KNN_3D_REF:-1}"
KS="${KS:-8 32 64 128}"
# Output tag: empty for round 5's default ranks, so its file names are unchanged.
KTAG=""
[[ "$KS" == "8 32 64 128" ]] || KTAG="_k$(tr ' ' '-' <<<"$KS")"
BOUNDARY_TILES="${BOUNDARY_TILES:-8}"
TIME_6D="${TIME_6D:-12:00:00}"
MEM_6D="${MEM_6D:-64G}"

F="$REWARD_ROOT/flow_rockstar/fields"
OUTDIR="$REWARD_ROOT/member_gather/knn_cdf"
REGION_JSON="${REGION_JSON:-$F/${BOX}__mgho_all_blocks_selfhostguard_${BOX}__seed0.json}"
GATE_JSON="${GATE_JSON:-$OUTDIR/knn_gate_crosscheck_${BOX}.json}"
KNN_3D_ALL="${KNN_3D_ALL:-$OUTDIR/knn_cdf_region_allarms_${BOX}.json}"
VERDICT="${VERDICT:-$OUTDIR/knn_phase_verdict${KTAG}_${BOX}.json}"
[[ -r "$REGION_JSON" ]] || { echo "no region json: $REGION_JSON"; exit 1; }
[[ -r "$GATE_JSON" ]] || { echo "no round-4 gate json: $GATE_JSON (run submit_knn_gate_crosscheck.sh)"; exit 1; }

# Every spliced field on disk, labelled as round 4 labelled them.
HR_FIELD="$ZFS/DMSR/paired_catnorm/hr/${BOX}.npy"
FIELDS="hr=$HR_FIELD"
REF_FIELDS="hr=$HR_FIELD"
for p in "$F/${BOX}__mgho_"*"_${BOX}__seed0.npy"; do
    [[ -e "$p" ]] || continue
    tag=$(basename "$p" .npy); tag=${tag#${BOX}__}; tag=${tag%__seed0}
    label=${tag#mgho_}; label=${label%_${BOX}}
    FIELDS="$FIELDS $label=$p"
    case "$label" in frozen|all_blocks_selfhostguard) REF_FIELDS="$REF_FIELDS $label=$p";; esac
done
echo "$(($(wc -w <<<"$FIELDS") - 1)) arm fields on disk (incl. frozen)"
[[ $(wc -w <<<"$REF_FIELDS") -eq 3 ]] || { echo "frozen or all_blocks_selfhostguard field missing"; exit 1; }

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$OUTDIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
CATALOG=$(ls $REWARD_ROOT/halos/${BOX}__hr__hr/*_rockstar/halos_*.ascii | head -1)
OWNER=$REWARD_ROOT/halos_particles/${BOX}__hr__hr/${BOX}_hr_owner.npy

write_env() {   # <envfile> <lambda> <fields> <out>
    cat > "$1" <<EOT
# Written by scripts/slurm/submit_knn_phase.sh at $(date '+%F %T').
# List values QUOTED (env-file-list-values-must-be-quoted).
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
KS_FIELDS="$3"
KS_CATALOG=$CATALOG
KS_OWNER=$OWNER
KS_OUT=$4
KS_REGION_JSON=$REGION_JSON
KS_TUNED_LABEL=all_blocks_selfhostguard
KS_BOUNDARY_TILES=$BOUNDARY_TILES
KS_BOUNDARY_QUERIES=1000
KS_SIZES="0 4 8"
KS_BOUNDARY_BY_CLASS=1
KS_PHASE_LAMBDA=$2
KS_KS="$KS"
KS_RESUME=$RESUME
EOT
}

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
JOBS=()
KP_KNN="3d=$KNN_3D_ALL"
LOGS=()
submit() {   # <name> <envfile> <sbatch args...>
    local name=$1 env=$2; shift 2
    if [ "$DRY" = "1" ]; then
        echo "DRY: sbatch $* scripts/slurm/knn_cdf_split_cpu.sbatch $env"; return
    fi
    local j
    j=$(sbatch --parsable "$@" scripts/slurm/knn_cdf_split_cpu.sbatch "$env")
    echo "submitted $name -> job $j (CPU)"
    JOBS+=("$j")
    LOGS+=("tail -F $REWARD_ROOT/logs/slurm-knnsplit-$j.out   # $name")
}

if [ "$KNN_3D_REF" = "1" ]; then
    OUT3="$OUTDIR/knn_phase_3dref${KTAG}_${BOX}.json"
    E="$REWARD_ROOT/env/knnphase_3dref_${STAMP}.env"
    write_env "$E" 0 "$REF_FIELDS" "$OUT3"
    submit 3dref "$E" --time=04:00:00 --mem=48G
    KP_KNN="$KP_KNN 3dref=$OUT3"
fi
for L in $LAMBDAS; do
    OUTL="$OUTDIR/knn_phase_l${L}${KTAG}_region_allarms_${BOX}.json"
    E="$REWARD_ROOT/env/knnphase_l${L}${KTAG}_${STAMP}.env"
    write_env "$E" "$L" "$FIELDS" "$OUTL"
    if [ "$L" = "0" ]; then
        submit "l0${KTAG} (3D, all arms)" "$E" --time=04:00:00 --mem=48G
    else
        submit "l$L${KTAG}" "$E" --time="$TIME_6D" --mem="$MEM_6D"
    fi
    KP_KNN="$KP_KNN l$L=$OUTL"
done

EV="$REWARD_ROOT/env/knnphasev_${STAMP}.env"
cat > "$EV" <<EOT
# Written by scripts/slurm/submit_knn_phase.sh at $(date '+%F %T').
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
KP_GATE_JSON=$GATE_JSON
KP_KNN="$KP_KNN"
KP_OUT=$VERDICT
EOT
echo "envfiles: $REWARD_ROOT/env/knnphase*_${STAMP}.env"; cat "$EV"; echo
if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch --dependency=afterany:<all> scripts/slurm/knn_phase_verdict_cpu.sbatch $EV"
    exit 0
fi
DEP=$(IFS=:; echo "${JOBS[*]}")
JV=$(sbatch --parsable --dependency=afterany:$DEP scripts/slurm/knn_phase_verdict_cpu.sbatch "$EV")
echo "submitted knnphasev -> job $JV (CPU, afterany:$DEP)"
echo
echo "monitor:"
for l in "${LOGS[@]}"; do echo "  $l"; done
echo "  tail -F $REWARD_ROOT/logs/slurm-knnphasev-$JV.out   # verdict"
echo "results:"
echo "  verdict: $VERDICT"
for kv in $KP_KNN; do echo "  ${kv%%=*}: ${kv#*=}"; done
echo "after a time limit: RESUME=1 LAMBDAS=\"<unfinished>\" KNN_3D_REF=0 bash scripts/slurm/submit_knn_phase.sh"
