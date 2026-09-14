#!/bin/bash
# Re-run ONLY the verdict stage on arms whose Rockstar catalogs already exist.
#
#   bash scripts/slurm/submit_reverdict.sh                       # the ablation arms
#   RV_ARMS="all_blocks_abl_nod6" bash scripts/slurm/submit_reverdict.sh
#   RV_STAGES="holdout_set9 holdout_set9_step2000" bash scripts/slurm/submit_reverdict.sh
#
# WHY. gate_verdict_vs_baseline gained a REGION-WIDE binned mass function and a
# shape score (2026-09-14). The per-host mass function it used to report is ~370
# objects; the region holds ~12k, and the open defect -- right total, 1.5-1.8x too
# many subhalos above 200p -- is a shape claim that needs those counts. The
# catalogs are already on disk, so this is one ~30 s CPU job per arm, no Rockstar.
#
# It REFUSES an arm whose catalog is missing rather than queueing a job that
# would write "GATE INCOMPLETE" over a good verdict file.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
RV_BOX="${RV_BOX:-set9}"
RV_HOST_ID="${RV_HOST_ID:-168880}"
RV_BASELINE_ARM="${RV_BASELINE_ARM:-all_blocks_self}"
RV_BASELINE_NAME="${RV_BASELINE_NAME:-self}"
RV_STAGES="${RV_STAGES:-holdout_${RV_BOX} holdout_${RV_BOX}_step2000}"
RV_ARMS="${RV_ARMS:-$(cd "$REWARD_ROOT/member_gather" && ls -d all_blocks_abl_* 2>/dev/null | tr '\n' ' ')}"

STAMP="$(date +%Y%m%d_%H%M%S)"
# An EMPTY RV_BASELINE_ARM is legal and means "no A/B": gate_verdict then writes
# region_counts.json with the same region/shape/radial fields and no Q1/Q2. That
# is the right mode for the baseline arm itself, which would otherwise be A/B'd
# against its own catalog.
BASELINE_TAG=""
[ -n "$RV_BASELINE_ARM" ] && BASELINE_TAG="mgho_${RV_BASELINE_ARM}_${RV_BOX}"
mkdir -p "$REWARD_ROOT/env"
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

n=0; skipped=0
for arm in $RV_ARMS; do
    for stage in $RV_STAGES; do
        run_dir="$REWARD_ROOT/member_gather/$arm/$stage"
        [ -d "$run_dir" ] || continue
        # The Rockstar tag mirrors the run dir: holdout_<box>_step<N> -> _step<N>.
        suffix=""
        case "$stage" in *_step*) suffix="_step${stage##*_step}" ;; esac
        rs_tag="mgho_${arm}${suffix}_${RV_BOX}"
        cat_json="$REWARD_ROOT/flow_rockstar/compare/${RV_BOX}__${rs_tag}.json"
        if [ ! -f "$cat_json" ]; then
            echo "  SKIP $arm $stage -- no catalog ($cat_json)"
            skipped=$((skipped + 1)); continue
        fi
        f="$REWARD_ROOT/env/reverdict_${STAMP}_${arm}${suffix}_$$.env"
        cat > "$f" <<EOT
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
HG_RUN_DIR=$run_dir
HG_BOX=$RV_BOX
HG_RS_TAG=$rs_tag
HG_BASELINE_TAG=$BASELINE_TAG
HG_BASELINE_NAME=$RV_BASELINE_NAME
HG_HOST_ID=$RV_HOST_ID
EOT
        [ -s "$f" ] || { echo "FAILED to write $f (quota?)" >&2; exit 1; }
        if [ "$DRY" = "1" ]; then
            echo "  DRY $arm $stage -> $rs_tag"
        else
            jid=$(sbatch --parsable scripts/slurm/gate_verdict_cpu.sbatch "$f")
            echo "  $arm $stage -> job $jid"
        fi
        n=$((n + 1))
    done
done

echo
echo "submitted $n verdict job(s), skipped $skipped"
echo "watch:   tail -F $REWARD_ROOT/logs/slurm-hg_verdict-<jid>.out"
echo "results: $REWARD_ROOT/member_gather/<arm>/holdout_${RV_BOX}*/verdict_vs_${RV_BASELINE_NAME}.json"
echo "  each gains region_mass_function (sub/host x num_p bins) and shape_score."
echo "then:    the ablation tables pick the new fields up on their next run."
