#!/bin/bash
# Raise the ceiling: measure it as a function of the tile count, then spend one
# Rockstar run on the rung that is worth it.
#
#   coverage curve (CPU, minutes)      <- this submitter
#     then, at the chosen rung:
#   FF_WHICH=hr FF_STEPS=0 FF_N_TILES=<n> bash scripts/slurm/submit_free_field_gather.sh
#     -> shakeout (GPU) -> tiles.npz, no optimisation (GPU, minutes)
#        -> splice -> Rockstar -> compare (CPU): the MEASURED ceiling at <n>
#
# Why two steps. docs/sr2_member_gather.md section 6.1 measured two ceilings and
# only one of them can be raised. The per-target ceiling is 151/154 -- saturated.
# The R_vir ceiling is 227 of HR's 506, and section 8.1 reads that straight off
# the coverage: the four trained tiles hold 42.4% of the host's Lagrangian sites.
# So the R_vir ceiling is a property of WHICH TILES ARE TRAINED, and the curve
# below is that function, computed without a halo finder.
#
# The second knob the curve costs out is min_num_p, and it raises a different
# bound: the gate counts subhalos at >= 50p while the loss supervises at
# >= 200p, and only 151 of the 506 are >= 200p. Lowering it does not move the
# ceiling; it moves how much of the ceiling anything is even asking for.
#
# This submitter ONLY calls sbatch. Configuration goes into ONE timestamped env
# file passed as a POSITIONAL argument -- never `sbatch --export`, which on this
# cluster sets SLURM_GET_USER_ENV=1 and gets the job requeued and HELD.
#
#   bash scripts/slurm/submit_gather_coverage.sh
#   GC_TILE_LADDER=1,4,8,16,32,64,128 bash scripts/slurm/submit_gather_coverage.sh
#   GC_BOX=set9 GC_HOST_ID=<id> bash scripts/slurm/submit_gather_coverage.sh
#   DRY=1 bash scripts/slurm/submit_gather_coverage.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

GC_BOX="${GC_BOX:-set8}"
GC_HOST_ID="${GC_HOST_ID:-271800}"
# The ladder walks the SAME host-site ranking `--n-tiles n` trains on, so rung
# n is exactly the run you would launch. 4 is on it on purpose: it is the only
# rung with a measured Rockstar answer, and it is what calibrates the rest.
GC_TILE_LADDER="${GC_TILE_LADDER:-1,2,4,6,8,12,16,24,32,48,64}"
GC_MIN_NUM_P_LADDER="${GC_MIN_NUM_P_LADDER:-200,100,50}"
GC_MIN_P="${GC_MIN_P:-50}"          # the cut the gate counts R_vir subhalos at
GC_MIN_PURITY="${GC_MIN_PURITY:-0.5}"
GC_MIN_LIVE_FRAC="${GC_MIN_LIVE_FRAC:-0.5}"
GC_RADIUS_FACTOR="${GC_RADIUS_FACTOR:-1.0}"
GC_BG_K="${GC_BG_K:-4096}"
GC_LABEL="${GC_LABEL:-}"

RUN_DIR="$REWARD_ROOT/free_field_gather/${GC_BOX}_h${GC_HOST_ID}${GC_LABEL}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"
ENVFILE="$REWARD_ROOT/env/gather_coverage_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_gather_coverage.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
GC_BOX=$GC_BOX
GC_HOST_ID=$GC_HOST_ID
GC_TILE_LADDER=$GC_TILE_LADDER
GC_MIN_NUM_P_LADDER=$GC_MIN_NUM_P_LADDER
GC_MIN_P=$GC_MIN_P
GC_MIN_PURITY=$GC_MIN_PURITY
GC_MIN_LIVE_FRAC=$GC_MIN_LIVE_FRAC
GC_RADIUS_FACTOR=$GC_RADIUS_FACTOR
GC_BG_K=$GC_BG_K
GC_LABEL=$GC_LABEL
EOT

echo "envfile: $ENVFILE"
cat "$ENVFILE"
echo
echo "run dir: $RUN_DIR"
echo

# Scrub SLURM_* so a submission from inside an allocation cannot leak an
# --export-like environment onto the batch job.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch scripts/slurm/gather_coverage_cpu.sbatch $ENVFILE" >&2
    JID="DRYID"
else
    JID=$(sbatch --parsable scripts/slurm/gather_coverage_cpu.sbatch "$ENVFILE")
fi
echo "submitted coverage curve -> job $JID (CPU)"

echo
echo "watch: tail -F $REWARD_ROOT/logs/slurm-gcover-$JID.out"
echo
echo "results:"
echo "  $RUN_DIR/coverage_curve.json   the curve, and the cost of each rung"
echo "  re-print it any time, no recompute:"
echo "    python scripts/features/gather_coverage_curve.py --from-json \\"
echo "        $RUN_DIR/coverage_curve.json"
echo
echo "then, at the rung the curve picks, MEASURE the ceiling there:"
echo "  FF_WHICH=hr FF_STEPS=0 FF_N_TILES=<n> FF_LABEL=_ceil<n> \\"
echo "    bash scripts/slurm/submit_free_field_gather.sh"
echo "  (steps 0 writes tiles.npz without optimising, so the chain splices the"
echo "   TRUE HR tiles -- the same construction that measured 227 at n_tiles=4.)"
