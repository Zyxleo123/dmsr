#!/bin/bash
# Does SUBFIND's own resolution decide the SR/HR outskirt ratio? Rerun SUBFIND
# with finer settings (default DesNumNgb 32 = the bump census's density scale,
# DesLinkNgb 10) on HR and three SR fields, then compare each to HR at the SAME
# setting against the published-default catalogs that already exist.
#
#   S  sf_snap + sf_run per field (afterok), into subfind/<name>__<TAG>
#        hr            the yardstick moves too -- only same-setting ratios count
#        coarse_soft   the trained arm
#        cs_bind_all   binding oracle, every bump self-bound (snapshot reused)
#        combo_cs      knot + temperature oracle (snapshot reused)
#   C  sfsetcmp  subfind_setting_compare.py over default + new   (afterany S)
#
# Existing snapshots are LINKED into the new run dirs (the snapshot stage keeps
# a present snapshot and only rewrites param.txt), saving ~7.5 GB on a 98% full
# scratch. Only calls sbatch otherwise; config in env files passed positionally.
#
#   bash scripts/slurm/submit_subfind_settings.sh
#   DRY=1 bash scripts/slurm/submit_subfind_settings.sh
#   NUM_NGB=48 LINK_NGB=20 bash scripts/slurm/submit_subfind_settings.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
R="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
SF="$R/subfind"
DRY="${DRY:-0}"
BOX="${BOX:-set9}"
NUM_NGB="${NUM_NGB:-32}"
LINK_NGB="${LINK_NGB:-10}"
TAG="${TAG:-n${NUM_NGB}l${LINK_NGB}}"
CS="mgho_all_blocks_sm_coarse_soft_${BOX}"

# name | SF_SOURCE | SF_TAG | SF_FIELD | default-setting dir (catalog, maybe a snapshot)
FIELDS=(
  "hr|hr|||${BOX}__hr"
  "coarse_soft|candidate|$CS|$R/flow_rockstar/fields/${BOX}__${CS}__seed0.npy|${BOX}__$CS"
  "cs_bind_all_v2|candidate|cs_bind_all_v2|$R/binding_oracle/${BOX}__cs_bind_all_v2__field.npy|${BOX}__cs_bind_all_v2"
  "combo_cs|candidate|combo_cs|$R/knot_oracle/${BOX}__combo_cs__field.npy|${BOX}__combo_cs"
)
HRC=$(ls "$R"/halos/"${BOX}"__hr__hr/*_rockstar/halos_*.ascii 2>/dev/null | head -1 || true)
EXP="$R/member_gather/all_blocks_sm_coarse_soft/holdout_${BOX}/export.json"
for f in "$HRC" "$EXP"; do [ -n "$f" ] && [ -e "$f" ] || { echo "missing input: '$f'"; exit 1; }; done
for spec in "${FIELDS[@]}"; do
    IFS='|' read -r name src stag field base <<< "$spec"
    [ -z "$field" ] || [ -e "$field" ] || { echo "missing field for $name: $field"; exit 1; }
    [ -e "$SF/$base/subfind_catalog.npz" ] || { echo "missing default catalog: $SF/$base"; exit 1; }
done
NQ=$(squeue -u "$USER" -h -p cpu 2>/dev/null | wc -l)
NEED=$(( ${#FIELDS[@]} * 2 + 1 ))
echo "cpu jobs queued: $NQ (+$NEED this submit; QOS cap 50)"
[ $((NQ + NEED)) -le 50 ] || { echo "would exceed the 50-job cpu cap; wait and rerun"; exit 1; }

mkdir -p "$R/logs" "$SF/env"
STAMP=$(date +%Y%m%d_%H%M%S)_$$
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
run() {
    echo "+ $*" >&2
    if [ "$DRY" = "1" ]; then echo "DRY_$RANDOM"; else "$@" | awk '{print $NF}'; fi
}

RUNS=(); DEF=""; NEW=""
for spec in "${FIELDS[@]}"; do
    IFS='|' read -r name src stag field base <<< "$spec"
    out="${base}__${TAG}"
    if [ "$DRY" != "1" ]; then
        mkdir -p "$SF/$out"
        if [ -e "$SF/$base/snapshot_000" ] && [ ! -e "$SF/$out/snapshot_000" ]; then
            ln -s "$SF/$base/snapshot_000" "$SF/$out/snapshot_000"
            echo "linked existing snapshot $SF/$base/snapshot_000"
        fi
    fi
    ENVF="$SF/env/${out}_${STAMP}.env"
    cat > "$ENVF" <<EOT
# Written by scripts/slurm/submit_subfind_settings.sh at $(date '+%F %T')
PROJECT="$PROJECT"
ZFS="$ZFS"
DMSR_REWARD_ROOT="$R"
SUBFIND_ROOT="$SF"
SF_BOX="$BOX"
SF_SEED="0"
SF_MIN_P="50"
SF_NRANKS="44"
SF_MAX_MEM_MB="13000"
SF_NAME="$out"
SF_SOURCE="$src"
SF_TAG="$stag"
SF_FIELD="$field"
SF_HOST_ID="168880"
SF_DES_NUM_NGB="$NUM_NGB"
SF_DES_LINK_NGB="$LINK_NGB"
EOT
    a=$(run sbatch scripts/slurm/subfind_snapshot_cpu.sbatch "$ENVF")
    b=$(run sbatch --dependency=afterok:"$a" scripts/slurm/subfind_run_cpu.sbatch "$ENVF")
    echo "S  $name: sf_snap $a -> sf_run $b   ($SF/$out)"
    RUNS+=("$b")
    DEF+="${DEF:+,}$name=$SF/$base/subfind_catalog.npz"
    NEW+="${NEW:+,}$name=$SF/$out/subfind_catalog.npz"
done
DEP=$(IFS=:; echo "${RUNS[*]}")
OUT="$R/placement_forensics/${BOX}_subfind_settings_${TAG}.json"
W="set -euo pipefail; echo host \$(hostname) cpus 2 mem 16G partition cpu; "
W+="source $ZFS/miniconda3/etc/profile.d/conda.sh; conda activate pjm; cd $PROJECT; "
W+="python -u scripts/features/subfind_setting_compare.py --runs 'default:$DEF;$TAG:$NEW' "
W+="--hr-catalog $HRC --region-export $EXP --out $OUT"
CID=$(run sbatch --dependency=afterany:"$DEP" --partition=cpu --cpus-per-task=2 --mem=16G \
      --time=30:00 --job-name=sfsetcmp --output="$R/logs/slurm-sfsetcmp-%j.out" --wrap "$W")
echo "C  sfsetcmp job $CID  (afterany:$DEP)"

echo
echo "monitor:"
echo "  squeue -u \$USER -o '%.10i %.12j %.8T %.10M %R'"
echo "  tail -F $SF/logs/slurm-sf_run-{$(IFS=,; echo "${RUNS[*]}")}.out"
echo "  tail -F $R/logs/slurm-sfsetcmp-$CID.out"
echo
echo "results:"
echo "  $OUT   (same-setting ratios vs HR, default vs $TAG)"
for spec in "${FIELDS[@]}"; do IFS='|' read -r name src stag field base <<< "$spec"; echo "  $SF/${base}__${TAG}/subfind_catalog.npz"; done
