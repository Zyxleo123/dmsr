#!/bin/bash
# Tile-overfit CONTROL LADDER: submit several increasing-control rungs at once to
# bracket where the proxy reward stops running away (dR_occ goes non-positive)
# and whether real occupancy ever moves. Each rung is an INDEPENDENT sibling
# chain (its own optimiser -> Rockstar array -> overlay/GIF), written to its own
# LABEL so nothing overwrites anything. This wrapper only calls the tile
# submitter, which only calls sbatch.
#
#   DRY=1 bash scripts/slurm/submit_tile_ladder.sh          # print all rungs
#   bash scripts/slurm/submit_tile_ladder.sh                # submit the ladder
#   ITERS=1200 TILE=373 bash scripts/slurm/submit_tile_ladder.sh
#
# After every rung's Rockstar monitor has finished, draw the comparison:
#   sbatch scripts/slurm/plot_tile_ladder_cpu.sbatch <envfile> \
#       LABELS=ctlA,ctlB,ctlC,ctlD BOX=set0 TILE=486 ARM=c
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"

ARM="${ARM:-c}"; BOX="${BOX:-set0}"; TILE="${TILE:-486}"
ITERS="${ITERS:-800}"; CKPT_EVERY="${CKPT_EVERY:-100}"; ROCKSTAR="${ROCKSTAR:-1}"
# Each Rockstar snapshot writes a ~3.76 GB gadget2; the rungs run concurrently,
# so cap EACH rung's array at 1 running task -> ~= (#rungs) gadget2 at peak, well
# under the per-user disk quota that truncated the first ladder attempt.
ROCK_CONC="${ROCK_CONC:-1}"
# ROCK_ONLY=1 resumes: skip the optimisers, reuse existing snapshots, just run
# the (throttled) Rockstar monitor + plots. Use after a quota-truncated attempt.
ROCK_ONLY="${ROCK_ONLY:-0}"
DRY="${DRY:-0}"

# Rungs, weakest control (still games) -> strongest (should hold dR_occ <= 0).
# Fields: LABEL  FIELD_LR  W_HOST  W_LOWK  W_PROX  W_REWARD   (-1 = config default)
RUNGS=(
  "ctlA 3e-4 0     -1    -1    -1"
  "ctlB 1e-4 200   400   20    1"
  "ctlC 3e-5 600   1200  100   0.4"
  "ctlD 1e-5 1500  3000  400   0.15"
  "ctlE 3e-6 3000  6000  800   0.05"
)

echo "=== tile control ladder: arm $ARM  $BOX/t$TILE  ${#RUNGS[@]} rungs "
echo "===   ITERS=$ITERS CKPT_EVERY=$CKPT_EVERY ROCKSTAR=$ROCKSTAR"
LABELS=()
for r in "${RUNGS[@]}"; do
    read -r LABEL FLR WH WL WP WR <<< "$r"
    LABELS+=("$LABEL")
    echo
    echo "########## rung $LABEL: FIELD_LR=$FLR W_HOST=$WH W_LOWK=$WL W_PROX=$WP W_REWARD=$WR"
    DRY="$DRY" bash scripts/slurm/submit_proxy_gradient.sh tile \
        ARM="$ARM" BOX="$BOX" TILE="$TILE" LABEL="$LABEL" \
        FIELD_LR="$FLR" W_HOST="$WH" W_LOWK="$WL" W_PROX="$WP" W_REWARD="$WR" \
        ITERS="$ITERS" TOL=0 ROCKSTAR="$ROCKSTAR" CKPT_EVERY="$CKPT_EVERY" \
        ROCK_CONC="$ROCK_CONC" ROCK_ONLY="$ROCK_ONLY"
    sleep 1   # distinct env-file timestamp per rung
done

IFS=,; LABELS_CSV="${LABELS[*]}"; unset IFS
echo
echo "=== all ${#RUNGS[@]} rungs submitted (independent siblings)."
echo "=== when every rung's Rockstar monitor has finished, draw the comparison:"
echo "===   sbatch scripts/slurm/plot_tile_ladder_cpu.sbatch <envfile> \\"
echo "===       LABELS=$LABELS_CSV BOX=$BOX TILE=$TILE ARM=$ARM"
