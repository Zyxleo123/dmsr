#!/bin/bash
# Phases 1-3: build the REAL labelled dataset. This submitter ONLY calls sbatch,
# and it has NO path into proxy training or SR2 fine-tuning -- by design.
#
#   DRY=1 bash scripts/slurm/submit_proxy_labels.sh all    # print, submit nothing
#   bash scripts/slurm/submit_proxy_labels.sh smoke        # 4 candidates, integrity only
#   bash scripts/slurm/submit_proxy_labels.sh all          # the predeclared 120
#   bash scripts/slurm/submit_proxy_labels.sh all BOXES="set0 set1"
#   bash scripts/slurm/submit_proxy_labels.sh index        # re-index what has landed
#
# WHAT THIS FIXES
# ---------------
# The previous data flow labelled only set0-7, so the held-out boxes had no rows
# at all and the "gate" had nothing to gate on; every labelling job rewrote the
# shared index, so a trainer could read a half-written table; and proxy training
# was chained behind the LAST SUBMITTED label job rather than the SLOWEST, so it
# could fit on a dataset missing whole boxes. Here: all twelve boxes, one
# indexing job depending on all labelling jobs, and a labels_complete.json the
# trainer refuses to start without.
#
# THE DEPENDENCY THAT IS NOT OPTIONAL
# -----------------------------------
#   targets -> generate HR -> label HR -> masks -> generate intervention -> label
# The HR labelling job is what extracts the Lagrangian member ids the masks are
# built from, out of a .particles table it then deletes. Everything else is a
# sibling and queues concurrently.
#
# Configuration goes into ONE timestamped env file passed as a POSITIONAL
# argument. Never `sbatch --export`: on this cluster any explicit export list
# makes sbatch set SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login
# environment on the compute node, and the job is requeued and HELD with
# "(user env retrieval failed requeued held)", stranding every dependent job.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
ROOT="${DMSR_SR2_DIRECT_ROOT:-$REWARD_ROOT/sr2_direct}"
STAGE="${1:-all}"; shift || true
DRY="${DRY:-0}"

RUN_NAME="${RUN_NAME:-direct_a}"
DIRECT_CFG="${DIRECT_CFG:-configs/reward/sr2_direct_finetune.yaml}"
BOXES="${BOXES:-}"          # empty = the predeclared list in the config
AFTER="${AFTER:-}"

for kv in "$@"; do
    case "$kv" in *=*) export "$kv" ;; esac
done

mkdir -p "$ROOT/logs" "$ROOT/env"
ENVFILE="$ROOT/env/labels_$(date +%Y%m%d_%H%M%S).env"
cat > "$ENVFILE" <<EOF
# Written by scripts/slurm/submit_proxy_labels.sh at $(date '+%F %T'); sourced by
# the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
DMSR_SR2_DIRECT_ROOT=$ROOT
DIRECT_CFG=$DIRECT_CFG
REWARD_CFG=configs/reward/reward.yaml
RUN_NAME=$RUN_NAME
BASE_SEED=0
EOF
echo "env file: $ENVFILE"

# Submitting from inside an allocation would leak SLURM_* into the child jobs.
for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v"; done

# The candidate matrix lives in the config and is enumerated by ONE function --
# _proxy_matrix.candidate_matrix, which the indexer and the completeness check
# also call -- so the submitter cannot disagree with them about what the dataset
# is. That module imports nothing heavy, but the *interpreter* still has to have
# PyYAML, and the login node's default python does not; use the project env.
PYBIN="${PYBIN:-$ZFS/miniconda3/envs/${CONDA_ENV:-pjm}/bin/python}"
[ -x "$PYBIN" ] || { echo "!!! no python at $PYBIN (set PYBIN=)" >&2; exit 1; }
MATRIX=$(PYTHONPATH="$PROJECT/scripts/reward" "$PYBIN" - "$DIRECT_CFG" "$BOXES" <<'PY'
import sys, yaml
from _proxy_matrix import candidate_matrix
cfg = yaml.safe_load(open(sys.argv[1]))
boxes = sys.argv[2].split() if len(sys.argv) > 2 and sys.argv[2].strip() else None
for c in candidate_matrix(cfg, boxes=boxes):
    print(c["box"], c["source"], c["seed"],
          "-" if c["alpha"] is None else f"{c['alpha']:.4f}", c["mode"], c["tag"])
PY
) || { echo "!!! could not enumerate the candidate matrix; nothing submitted" >&2; exit 1; }
N_CAND=$(printf '%s\n' "$MATRIX" | grep -c . || true)
echo "candidate matrix: $N_CAND candidates"

SUB_OVERRIDES=()
ABORT_FLAG="$ROOT/.submit_proxy_labels_aborted.$$"
trap 'rm -f "$ABORT_FLAG"' EXIT

sub() {   # sub <human label> <sbatch args...>
    local label="$1"; shift
    if [ "$DRY" = "1" ]; then
        # The command goes to stderr: stdout is the job id, which the caller
        # captures with $(...) and feeds to the next --dependency.
        echo "DRY  $label:" >&2
        echo "       sbatch $* $ENVFILE ${SUB_OVERRIDES[*]}" >&2
        echo "000000"
        return 0
    fi
    local jid rc
    jid=$(sbatch --parsable "$@" "$ENVFILE" "${SUB_OVERRIDES[@]}") || rc=$?
    if [ -n "${rc:-}" ] || [ -z "$jid" ]; then
        echo "" >&2
        echo "!!! sbatch FAILED for: $label" >&2
        echo "!!! nothing further has been submitted. Every stage skips work it" >&2
        echo "!!! already did and whose inputs are unchanged, so re-running this" >&2
        echo "!!! submitter resumes rather than restarts." >&2
        touch "$ABORT_FLAG"
        echo ""
        return 1
    fi
    echo "  $label -> job $jid" >&2
    echo "$jid"
}

die_if_aborted() {
    if [ -e "$ABORT_FLAG" ]; then
        echo "aborting the submitter; see the message above." >&2
        exit 1
    fi
}

dep_of() { [ -n "$1" ] && echo "--dependency=afterok:$1" || echo ""; }

LABEL_JIDS=()

# --- smoke: four candidates, integrity only --------------------------------
# The plan is explicit that no MLP -- not even a preliminary one -- may be
# fitted on these. They exist to prove the pipeline writes what it claims:
# member consistency ok, tile sums equal to the direct whole-box counts, the
# .particles table gone, both arms' features present.
if [ "$STAGE" = "smoke" ]; then
    echo "=== smoke test: 4 candidates on one box, DATA INTEGRITY ONLY"
    SBOX="${SMOKE_BOX:-set0}"
    JT=$(sub "smoke: intervention targets (CPU)" $(dep_of "$AFTER") \
        scripts/slurm/intervention_targets_cpu.sbatch); die_if_aborted

    SUB_OVERRIDES=("BOX=$SBOX" "SOURCE=hr" "SEED=0")
    JG=$(sub "smoke: generate HR $SBOX (GPU)" $(dep_of "$AFTER") \
        scripts/slurm/generate_proxy_candidates_gpu.sbatch); die_if_aborted
    JH=$(sub "smoke: label HR $SBOX (CPU)" --dependency=afterok:"$JG":"$JT" \
        scripts/slurm/label_proxy_candidates_cpu.sbatch); die_if_aborted
    SUB_OVERRIDES=()

    for SEED in 0 1; do
        SRC=$([ "$SEED" = "0" ] && echo frozen || echo frozen_seed)
        SUB_OVERRIDES=("BOX=$SBOX" "SOURCE=$SRC" "SEED=$SEED")
        J=$(sub "smoke: generate $SRC seed$SEED $SBOX (GPU)" $(dep_of "$AFTER") \
            scripts/slurm/generate_proxy_candidates_gpu.sbatch); die_if_aborted
        LABEL_JIDS+=("$(sub "smoke: label $SRC seed$SEED $SBOX (CPU)" \
            --dependency=afterok:"$J" \
            scripts/slurm/label_proxy_candidates_cpu.sbatch)"); die_if_aborted
        SUB_OVERRIDES=()
    done

    SUB_OVERRIDES=("BOX=$SBOX")
    JM=$(sub "smoke: intervention mask $SBOX (CPU)" --dependency=afterok:"$JH" \
        scripts/slurm/intervention_masks_cpu.sbatch); die_if_aborted
    SUB_OVERRIDES=("BOX=$SBOX" "SOURCE=intervention" "SEED=0" "ALPHA=1.0" "MODE=both")
    J=$(sub "smoke: generate intervention a=1.0 $SBOX (GPU)" \
        --dependency=afterok:"$JM" \
        scripts/slurm/generate_proxy_candidates_gpu.sbatch); die_if_aborted
    LABEL_JIDS+=("$(sub "smoke: label intervention a=1.0 $SBOX (CPU)" \
        --dependency=afterok:"$J" \
        scripts/slurm/label_proxy_candidates_cpu.sbatch)"); die_if_aborted
    LABEL_JIDS+=("$JH")
    SUB_OVERRIDES=()
    echo "  !! INSPECT ONLY. For each candidate under $ROOT/candidates/, check"
    echo "  !! label_report.json: label_ok true, member_consistency.ok true,"
    echo "  !! tile_sum_vs_direct_max_abs_residual ~1e-10, and no .particles left."
    echo "  !! Do NOT train anything on these four."
fi

# --- all: the predeclared matrix on every proxy box ------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "data" ]; then
    echo "=== proxy dataset ($RUN_NAME): $N_CAND candidates"
    JT=$(sub "targets: HR subhalos for every box (CPU)" $(dep_of "$AFTER") \
        scripts/slurm/intervention_targets_cpu.sbatch); die_if_aborted

    # Pass 1: HR and frozen. Siblings of each other, so they queue concurrently
    # rather than serialising behind one 12-hour Rockstar run.
    declare -A HR_LABEL=()
    while read -r BOX SOURCE SEED ALPHA MODE TAG; do
        [ -z "${BOX:-}" ] && continue
        [ "$SOURCE" = "intervention" ] && continue
        SUB_OVERRIDES=("BOX=$BOX" "SOURCE=$SOURCE" "SEED=$SEED")
        JG=$(sub "generate $BOX/$TAG (GPU)" $(dep_of "$AFTER") \
            scripts/slurm/generate_proxy_candidates_gpu.sbatch); die_if_aborted
        if [ "$SOURCE" = "hr" ]; then
            # Needs the target list too: it extracts those halos' member ids.
            JL=$(sub "label $BOX/$TAG (CPU)" --dependency=afterok:"$JG":"$JT" \
                scripts/slurm/label_proxy_candidates_cpu.sbatch); die_if_aborted
            HR_LABEL[$BOX]="$JL"
        else
            JL=$(sub "label $BOX/$TAG (CPU)" --dependency=afterok:"$JG" \
                scripts/slurm/label_proxy_candidates_cpu.sbatch); die_if_aborted
        fi
        LABEL_JIDS+=("$JL")
        SUB_OVERRIDES=()
    done <<< "$MATRIX"

    # Pass 2: masks, one per box, behind that box's HR label.
    declare -A MASK_JID=()
    for BOX in "${!HR_LABEL[@]}"; do
        SUB_OVERRIDES=("BOX=$BOX")
        MASK_JID[$BOX]=$(sub "mask $BOX (CPU)" \
            --dependency=afterok:"${HR_LABEL[$BOX]}" \
            scripts/slurm/intervention_masks_cpu.sbatch); die_if_aborted
        SUB_OVERRIDES=()
    done

    # Pass 3: the interventions, behind their own box's mask only.
    while read -r BOX SOURCE SEED ALPHA MODE TAG; do
        [ -z "${BOX:-}" ] && continue
        [ "$SOURCE" != "intervention" ] && continue
        DEP="${MASK_JID[$BOX]:-}"
        SUB_OVERRIDES=("BOX=$BOX" "SOURCE=intervention" "SEED=$SEED"
                       "ALPHA=$ALPHA" "MODE=$MODE")
        JG=$(sub "generate $BOX/$TAG (GPU)" \
            $([ -n "$DEP" ] && echo "--dependency=afterok:$DEP" || dep_of "$AFTER") \
            scripts/slurm/generate_proxy_candidates_gpu.sbatch); die_if_aborted
        LABEL_JIDS+=("$(sub "label $BOX/$TAG (CPU)" --dependency=afterok:"$JG" \
            scripts/slurm/label_proxy_candidates_cpu.sbatch)"); die_if_aborted
        SUB_OVERRIDES=()
    done <<< "$MATRIX"
fi

# --- index: ONE job, after every labelling job -----------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "data" ] || [ "$STAGE" = "smoke" ] \
        || [ "$STAGE" = "index" ]; then
    DEP=""
    if [ "${#LABEL_JIDS[@]}" -gt 0 ]; then
        # afterany, not afterok: this job's purpose is to report what is missing,
        # which it can only do if it runs when something upstream failed. afterok
        # would leave it in DependencyNeverSatisfied and say nothing at all.
        DEP="--dependency=afterany:$(IFS=:; echo "${LABEL_JIDS[*]}")"
    elif [ -n "$AFTER" ]; then
        DEP="--dependency=afterany:$AFTER"
    fi
    echo "=== index (${#LABEL_JIDS[@]} labelling jobs upstream)"
    sub "index: join features and labels, write labels_complete.json (CPU)" \
        $DEP scripts/slurm/index_proxy_labels_cpu.sbatch >/dev/null
    die_if_aborted
    echo "  table   -> $ROOT/proxy_data/rows.jsonl"
    echo "  marker  -> $ROOT/proxy_data/labels_complete.json"
    echo "  report  -> $ROOT/proxy_data/index_report.json"
fi

echo "=== submitted. watch with: squeue -u \$USER -o '%.10i %.20j %.10T %R'"
echo "=== NOTHING here trains a proxy or moves an SR2 parameter. When the"
echo "=== marker exists: bash scripts/slurm/submit_proxy_benchmark.sh all"
