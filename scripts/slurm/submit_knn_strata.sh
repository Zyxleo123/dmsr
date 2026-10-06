#!/bin/bash
# GATE for the catalog-free kNN strata arm (docs/sr2_knn_strata.md).
#
# The question: --knn-classes sub_50_200 is the one Rockstar label left in the
# kNN term's gradient. Can a stratum of each field's OWN r_k replace it?
# MEASURED on synthetic already (tests/features/test_knn_cdf.py): the densest
# stratum is 65-83% HOST particles, and host over-concentration is this line's
# characteristic damage. This job asks whether that holds on the real box.
#
#   bash scripts/slurm/submit_knn_strata.sh            # 3D, set9, box-wide
#   PHASE=1 bash scripts/slurm/submit_knn_strata.sh    # 6D at lambda=4000
#   REGION=1 bash scripts/slurm/submit_knn_strata.sh   # only the spliced tiles
#   DRY=1 bash scripts/slurm/submit_knn_strata.sh
#
# PHASE=1 is the one the arm would actually train in: round 5's pass (rho -0.65)
# was measured in 6D, and position-only ranked the arms at -0.23. Run BOTH if
# there is queue for it -- if the verdict flips between them, that is itself the
# finding.
#
# Reading the result: verdict.pass. FAIL means a density cut does NOT stand in
# for the class mask and the GPU arm is not launched -- submit_knn_strata_arm.sh
# refuses to run without a passing verdict json.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
PHASE="${PHASE:-0}"
REGION="${REGION:-0}"
BOX="${BOX:-set9}"

F="$REWARD_ROOT/flow_rockstar/fields"
HR="$ZFS/DMSR/paired_catnorm/hr/$BOX.npy"
SEED="${SEED:-0}"

# The frozen SR2 baseline is the sr2_base CACHE, not flow_rockstar/fields --
# that directory holds ARM outputs and its .npy files have been purged (60
# .json sidecars, 1 .npy, jobs 54403/54404 died on exactly this). Resolved by
# the same rule as cosmo_sr.reward.base.find_base_field: exactly one hit, or
# refuse. Two cached fields differ in weights, LR field or code version, and
# picking one is how a run mixes baselines.
if [ -z "${KC_FIELDS:-}" ]; then
    mapfile -t _BASE < <(ls "$REWARD_ROOT/cache/sr2_base/${BOX}_seed${SEED}_"*.npy 2>/dev/null | grep -v '\.tmp\.npy$' || true)
    if [ "${#_BASE[@]}" -eq 0 ]; then
        echo "no cached SR2 base field for $BOX seed $SEED in $REWARD_ROOT/cache/sr2_base"
        echo "build it with: scripts/reward/cache_sr2_base.py"
        exit 1
    elif [ "${#_BASE[@]}" -gt 1 ]; then
        echo "${#_BASE[@]} cached base fields for $BOX seed $SEED -- they differ in"
        echo "weights, LR field or code version. Name one with KC_FIELDS rather"
        echo "than letting this pick, which would mix baselines:"
        printf '  %s\n' "${_BASE[@]}"
        exit 1
    fi
    KC_FIELDS="hr=$HR frozen=${_BASE[0]}"
    # Arm fields are OPTIONAL extras: include any whose .npy survived the purge,
    # and say plainly which ones did not rather than failing the job on them.
    for _arm in selfhostguard smallhalo condsmall nearclean2; do
        _p="$F/${BOX}__mgho_all_blocks_${_arm}_${BOX}__seed0.npy"
        if [ -r "$_p" ]; then
            KC_FIELDS="$KC_FIELDS ${_arm}=$_p"
        elif [ -r "${_p%.npy}.json" ]; then
            echo "note: arm field '$_arm' was purged (.json kept, .npy gone) -- skipping"
        fi
    done
fi
KC_CATALOG="${KC_CATALOG:-$(ls $REWARD_ROOT/halos/${BOX}__hr__hr/*_rockstar/halos_*.ascii 2>/dev/null | head -1)}"
KC_OWNER="${KC_OWNER:-$REWARD_ROOT/halos_particles/${BOX}__hr__hr/${BOX}_hr_owner.npy}"
TAG="$BOX"
MEM_NOTE="3D"
if [ "$PHASE" = "1" ]; then
    KC_PHASE_LAMBDA="${KC_PHASE_LAMBDA:-4000}"
    TAG="${TAG}_phase"
    MEM_NOTE="6D"
else
    KC_PHASE_LAMBDA="${KC_PHASE_LAMBDA:-0}"
fi
if [ "$REGION" = "1" ]; then
    KC_REGION_JSON="${KC_REGION_JSON:-$F/${BOX}__mgho_all_blocks_selfhostguard_${BOX}__seed0.json}"
    [[ -r "$KC_REGION_JSON" ]] || { echo "no region json: $KC_REGION_JSON"; exit 1; }
    TAG="${TAG}_region"
else
    KC_REGION_JSON="${KC_REGION_JSON:-}"
fi
KC_OUT="${KC_OUT:-$REWARD_ROOT/member_gather/knn_cdf/knn_strata_${TAG}.json}"

[[ -n "$KC_CATALOG" && -r "$KC_CATALOG" ]] || { echo "no HR catalog for $BOX"; exit 1; }
[[ -r "$KC_OWNER" ]] || { echo "no owner array: $KC_OWNER"; exit 1; }
# Pre-flight EVERY field here, not in the job. Jobs 54403/54404 queued, waited,
# started and died on an unreadable path -- a check the submitter could have
# made for free. A submitter that only validates its own two paths is one that
# spends queue time to report a typo.
_MISS=0
for _f in $KC_FIELDS; do
    _lab="${_f%%=*}"; _path="${_f#*=}"
    if [ ! -r "$_path" ]; then
        echo "field '$_lab' is not readable: $_path"
        _MISS=1
    fi
done
[ "$_MISS" = "0" ] || { echo "fix KC_FIELDS (or unset it for the resolved defaults)"; exit 1; }
echo "fields:"
for _f in $KC_FIELDS; do printf '  %-14s %s\n' "${_f%%=*}" "${_f#*=}"; done
echo

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$(dirname "$KC_OUT")"
ENVFILE="$REWARD_ROOT/env/knnstrata_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_knn_strata.sh at $(date '+%F %T').
# List values QUOTED (env-file-list-values-must-be-quoted).
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
KC_FIELDS="$KC_FIELDS"
KC_CATALOG=$KC_CATALOG
KC_OWNER=$KC_OWNER
KC_OUT=$KC_OUT
KC_GATE_LABEL="${KC_GATE_LABEL:-frozen}"
KC_KS="${KC_KS:-8 32 64 128}"
KC_STRATA_RANK=${KC_STRATA_RANK:-0}
KC_STRATA="${KC_STRATA:-4 8 16}"
KC_CHARGE="${KC_CHARGE:-0 0,1}"
KC_TILES=${KC_TILES:-24}
KC_QUERIES=${KC_QUERIES:-4096}
KC_BUFFER_CELLS=${KC_BUFFER_CELLS:-8.0}
KC_N_QUANTILES=${KC_N_QUANTILES:-16}
KC_DEAD_ZONE=${KC_DEAD_ZONE:-0.02}
KC_PHASE_LAMBDA=$KC_PHASE_LAMBDA
KC_REGION_JSON="$KC_REGION_JSON"
KC_MIN_RECALL_ENRICHMENT=${KC_MIN_RECALL_ENRICHMENT:-2.0}
KC_MIN_RHO=${KC_MIN_RHO:-0.5}
KC_SEED=${KC_SEED:-0}
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
MEM=48G
[ "$PHASE" = "1" ] && MEM=64G
if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch --mem=$MEM scripts/slurm/knn_strata_composition_cpu.sbatch $ENVFILE"
    exit 0
fi
J1=$(sbatch --parsable --mem=$MEM scripts/slurm/knn_strata_composition_cpu.sbatch "$ENVFILE")
echo "submitted knnstrata ($MEM_NOTE, $MEM) -> job $J1 (CPU)"
echo
echo "watch:   tail -F $REWARD_ROOT/logs/slurm-knnstrata-$J1.out"
echo "result:  $KC_OUT"
echo "verdict: python -c \"import json;d=json.load(open('$KC_OUT'));print(d['verdict'])\""
echo
echo "then, ONLY if verdict.pass is true:"
echo "  KC_VERDICT=$KC_OUT bash scripts/slurm/submit_knn_strata_arm.sh"
