#!/bin/bash
# Halo-find the m2m from-scratch arms: dump a full box (GPU) -> Rockstar (CPU).
#
#   bash scripts/slurm/submit_m2m_halos.sh
#   M2M_HALO_EPOCH=77 M2M_HALO_BOX=set9 bash scripts/slurm/submit_m2m_halos.sh
#   M2M_HALO_ARMS="m2m_theirs m2m_theirs_mg" bash scripts/slurm/submit_m2m_halos.sh
#   DRY=1 bash scripts/slurm/submit_m2m_halos.sh
#
# WHY, when the field report already ranked these arms. It ranked them on POWER,
# and the objects at issue are sub-grid: frozen SR2 already matches HR's density
# spectrum to 1-5% at every resolvable k, and subhalos live above k = 32 where
# the LR input carries nothing. The spectrum cannot see them. Rockstar reads
# particles, and a 512^3 candidate box is exactly what it wants -- with no splice
# and no seam, because an m2m candidate IS the whole box.
#
# Only calls sbatch. Each arm is box-dump -> rockstar chained afterok; the arms
# are siblings so they queue concurrently.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

EPOCH="${M2M_HALO_EPOCH:-77}"
BOX="${M2M_HALO_BOX:-set9}"
ARMS="${M2M_HALO_ARMS:-m2m_theirs m2m_theirs_knn50 m2m_theirs_mg}"
ENVD="$ROOT/m2m/envs"; mkdir -p "$ENVD" "$ROOT/m2m/boxes"

if [[ "$(hostname)" != lov4* && -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: inside allocation $SLURM_JOB_ID on $(hostname); sbatch needs lov4." >&2
    exit 1
fi
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

STAMP="$(date +%Y%m%d-%H%M%S)"
WATCH=()

sub() {  # sub <dep-or-empty> <script> <envfile> [extra sbatch args...]
    local dep="$1" script="$2" envf="$3"; shift 3
    if [ "$DRY" = "1" ]; then
        echo "DRY: sbatch ${dep:+--dependency=afterok:$dep} $* $script $envf" >&2
        echo "0"; return
    fi
    sbatch --parsable ${dep:+--dependency=afterok:$dep} "$@" "$script" "$envf"
}

echo "=== m2m halo finding: epoch $EPOCH on $BOX"
echo

for arm in $ARMS; do
    CKPT="$ROOT/m2m/$arm/state_${EPOCH}.pt"
    if [[ ! -r "$CKPT" ]]; then
        echo "──────── $arm: no state_${EPOCH}.pt -- SKIPPED"
        continue
    fi
    TAG="${arm}_e${EPOCH}"
    FIELD="$ROOT/m2m/boxes/${arm}__state_${EPOCH}__${BOX}.npy"

    # Reuse the arm's own training env file so the model/repo/scale match the
    # run that produced the checkpoint; a mismatched --model would build a
    # different generator and file its field under this arm's name.
    SRC=$(ls -t "$ENVD/${arm}-2"*.env 2>/dev/null | head -1)
    if [[ -z "$SRC" ]]; then
        echo "──────── $arm: no env file in $ENVD -- SKIPPED"
        continue
    fi
    ENVF="$ENVD/${arm}-halo${EPOCH}-${STAMP}.env"
    grep -vE '^M2M_EXTRA_ARGS=|^M2M_CKPT|^M2M_BOX_|^M2M_RS_' "$SRC" > "$ENVF"
    {
        echo "M2M_CKPT='$CKPT'"
        echo "M2M_BOX_BOX='$BOX'"
        echo "M2M_BOX_OUT='$FIELD'"
        echo "M2M_RS_TAG='$TAG'"
    } >> "$ENVF"

    echo "──────── $arm  (tag $TAG)"
    JB=$(sub "" scripts/slurm/m2m_box_dump_gpu.sbatch "$ENVF" --job-name="box_$arm")
    echo "    box dump -> $JB"
    JR=$(sub "$JB" scripts/slurm/m2m_rockstar_cpu.sbatch "$ENVF" --job-name="rs_$arm")
    echo "    rockstar -> $JR  (afterok:$JB)"
    WATCH+=("m2mbox:$JB" "m2mrs:$JR")
    echo
done

echo "=== watch"
for w in "${WATCH[@]}"; do
    echo "  tail -F $ROOT/logs/slurm-${w%%:*}-${w##*:}.out"
done
echo
echo "=== results"
echo "  catalogs:  $ROOT/m2m/rockstar/halos/${BOX}__candidate__<arm>_e${EPOCH}/"
echo "  reference: $ROOT/halos/${BOX}__hr__hr  and  ${BOX}__base__base"
echo "  boxes:     $ROOT/m2m/boxes/  -- 3.2 GB each, DELETE once catalogs exist"
echo
echo "=== the comparison to make"
echo "  subhalo counts and the mass function against HR, NOT against each other."
echo "  Rockstar links in 6D and reports objects with no spatial overdensity"
echo "  behind them, so a count that beats HR is a fragmentation warning, not a"
echo "  win -- cross-check with SUBFIND before believing any gain."
