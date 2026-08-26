#!/bin/bash
# The feasibility test for the id-gathered member-set objective: optimise the
# tile FIELD directly, with no generator (scripts/features/free_field_gather.py).
#
# Run this before any network fine-tune on the loss. docs/sr2_gather_finetune.md
# spent four GPU runs and four Rockstar gates before its section 8.2 revealed
# that the objective had never moved the supervised material; every one of those
# runs confounded "is the loss right" with "can the generator reach it". Removing
# the generator separates them, and costs minutes.
#
# Node choice, checked with `myfree` at write time: `general` has gpu28 (a5000,
# 24 GB, 2/4 free, 436 GB host free) and gpu31 (a6000, 1/8 free) AVAILABLE. The
# GPU footprint is NOT load-bearing and the job is deliberately not pinned: the
# free field is 4 x 6 x 64^3 float32 (25 MB) plus two Adam moments, the particle
# table is 1.05M x 3 x 2 floats (25 MB), and the only large intermediate is the
# chunked N x N potential block at FF_POT_CHUNK x max_set_size -- ~70 MB at the
# defaults. Everything fits in 11 GB, so an rtx_2080_ti in `legacy` would work
# too if `general` is full; switch the partition AND the --qos together, since
# rhea enforces one QOS per partition.
#
# Host RAM is the real ask, exactly as in finetune_host_gather_gpu.sbatch: the
# 512^3 owner array is loaded whole and inverted into a CSR index (~2-3 GB peak).
# The cached frozen box is mmap'd and read by fancy index for the out-of-tile
# members only, which touches pages rather than loading 3.2 GB.
#
#   sbatch scripts/slurm/free_field_gather_gpu.sbatch <envfile>
#SBATCH --job-name=ffgather
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-ffgather-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${DATA:=/zfsauton/scratch/yixiz/DMSR/paired_catnorm}"
: "${FF_BOX:=set8}"
: "${FF_HOST_ID:=-1}"
: "${FF_N_TILES:=4}"
: "${FF_STEPS:=2000}"
: "${FF_LR:=3e-3}"
: "${FF_EVAL_EVERY:=100}"
: "${FF_W_VIRIAL:=1.0}"
: "${FF_W_BOUND:=1.0}"
: "${FF_W_D6:=1.0}"
: "${FF_W_RRMS:=0.3}"
: "${FF_W_SIGMAV:=0.3}"
: "${FF_W_CENTRE:=1.0}"
# What the centre term charges for, and how hard. All three default to the
# 72/154 objective, so an unset run is bit-for-bit the run docs/sr2_member_gather.md
# section 6 records. See section 9 for why the alternatives exist.
: "${FF_BOUND_PENALTY:=hinge}"
: "${FF_CENTRE_MODE:=full}"
: "${FF_CENTRE_DEAD_ZONE:=0.0}"
: "${FF_CENTRE_HUBER:=0.0}"
: "${FF_MASK_GRAD:=1}"
: "${FF_W_LOW:=100.0}"
: "${FF_MIN_NUM_P:=200}"
# The cap on supervised sets. It did not bind at 4 tiles (154 sets) and it BINDS
# at 16 (647 homed, 625 past purity, truncated to 256), so a wider tiling that
# leaves this at the default supervises 41% of what it covers. It keeps the
# largest sets, and the pair sums are O(N^2), so it is the cost dial too.
: "${FF_MAX_SETS:=256}"
: "${FF_MIN_PURITY:=0.5}"
: "${FF_MIN_LIVE_FRAC:=0.5}"
: "${FF_SOFTENING:=0.01}"
: "${FF_BOUND_TAU:=0.5}"
: "${FF_BG_K:=4096}"
: "${FF_BG_RADIUS:=4.0}"
: "${FF_POT_CHUNK:=2048}"
: "${FF_GAIN_FRAC:=0.5}"
: "${FF_LOW_K_MAX:=0.02}"
: "${FF_VEL_RMS_TOL:=0.10}"
: "${FF_TARGETS_ONLY:=0}"
# Tiles per generator forward pass. The whole tiling in one batch is fine at 4
# and is an allocation failure on a 2080 Ti at 24, which on this cluster kills
# the process with no traceback. Only the peak allocation depends on it.
: "${FF_FORWARD_CHUNK:=4}"
# An explicit tiling, overriding the host-site ranking FF_N_TILES takes. Empty
# means "use the ranking"; a list is how a tiling gather_coverage_curve.py
# recommends gets trained.
: "${FF_TILES:=}"
: "${FF_LABEL:=}"
: "${FF_SEED:=0}"

require_input "$DATA/hr/$FF_BOX.npy"   "scripts/reward/preprocess (paired_catnorm)"
require_input "$DATA/lr/$FF_BOX.npy"   "scripts/reward/preprocess (paired_catnorm)"
require_input "$DMSR_REWARD_ROOT/halos_particles/${FF_BOX}__hr__hr/${FF_BOX}_hr_owner.npy" \
    "scripts/reward/rockstar_particles.py -- membership IS the supervision here"
require_input "$PROJECT/external/SRS-map2map/SRmodel/G_z0.pt" \
    "the frozen SR2 checkpoint (configs/sr2_baseline/freeze.yaml)"

EXTRA=()
[[ "$FF_TARGETS_ONLY" == "1" ]] && EXTRA+=(--targets-only)
[[ -n "$FF_TILES" ]] && EXTRA+=(--tiles "$FF_TILES")
[[ "$FF_MASK_GRAD" == "0" ]] && EXTRA+=(--no-mask-grad)

echo "=== free-field member gather: box=$FF_BOX steps=$FF_STEPS lr=$FF_LR"
echo "    objective: virial=$FF_W_VIRIAL bound=$FF_W_BOUND d6=$FF_W_D6"
echo "               r_rms=$FF_W_RRMS sigma_v=$FF_W_SIGMAV centre=$FF_W_CENTRE"
echo "               centre mode=$FF_CENTRE_MODE dead=$FF_CENTRE_DEAD_ZONE"
echo "                      huber=$FF_CENTRE_HUBER"
echo "               guard low=$FF_W_LOW  grad mask=$FF_MASK_GRAD"
if [ "$FF_CENTRE_MODE" != "full" ]; then
echo "    NOTE: this arm charges for LESS than the full offset, so the free"
echo "          field CANNOT beat the 72/154 that saw the whole address. Read"
echo "          centre_penalised_radii against centre_offset_radii: the first"
echo "          is what the loss drove to zero, the second is what the gate"
echo "          scores, and the gap is the part of the target this arm dropped."
fi
echo "    tiles: $FF_N_TILES (the R_vir CEILING knob -- 4 tiles hold 42.4% of"
echo "           the host's Lagrangian sites and cap the gate at 227 of 506;"
echo "           scripts/features/gather_coverage_curve.py is that function)"
echo "    sets: >= ${FF_MIN_NUM_P}p, purity >= $FF_MIN_PURITY, cap $FF_MAX_SETS,"
echo "          live fraction >= $FF_MIN_LIVE_FRAC, background $FF_BG_K particles"
echo "    NOTE: virial carries the descent; bound_frac is saturated at the frozen"
echo "          start (section 8.2 reads 0.000 because sigmoid(-300) is 0, not"
echo "          because it is merely low) and is the target, not the driver."

python -u scripts/features/free_field_gather.py \
    --config "$DIRECT_CFG" \
    --box "$FF_BOX" \
    --host-id "$FF_HOST_ID" \
    --n-tiles "$FF_N_TILES" \
    --forward-chunk "$FF_FORWARD_CHUNK" \
    --steps "$FF_STEPS" \
    --lr "$FF_LR" \
    --eval-every "$FF_EVAL_EVERY" \
    --w-virial "$FF_W_VIRIAL" \
    --w-bound "$FF_W_BOUND" \
    --w-d6 "$FF_W_D6" \
    --w-rrms "$FF_W_RRMS" \
    --w-sigmav "$FF_W_SIGMAV" \
    --w-centre "$FF_W_CENTRE" \
    --bound-penalty "$FF_BOUND_PENALTY" \
    --centre-mode "$FF_CENTRE_MODE" \
    --centre-dead-zone "$FF_CENTRE_DEAD_ZONE" \
    --centre-huber-radii "$FF_CENTRE_HUBER" \
    --w-low "$FF_W_LOW" \
    --min-num-p "$FF_MIN_NUM_P" \
    --max-sets "$FF_MAX_SETS" \
    --min-purity "$FF_MIN_PURITY" \
    --min-live-frac "$FF_MIN_LIVE_FRAC" \
    --softening "$FF_SOFTENING" \
    --bound-tau "$FF_BOUND_TAU" \
    --bg-k "$FF_BG_K" \
    --bg-radius "$FF_BG_RADIUS" \
    --pot-chunk "$FF_POT_CHUNK" \
    --gain-frac "$FF_GAIN_FRAC" \
    --low-k-max "$FF_LOW_K_MAX" \
    --vel-rms-tol "$FF_VEL_RMS_TOL" \
    --label "$FF_LABEL" \
    --seed "$FF_SEED" \
    "${EXTRA[@]}" \
    ${DEVICE:+--device "$DEVICE"}

echo "=== result: $DMSR_REWARD_ROOT/free_field_gather/${FF_BOX}_h<id>${FF_LABEL}/"
echo "    summary.json  metrics.jsonl  subhalos.json  tiles.npz"
echo "    tiles.npz is in the SAME layout finetune_host_gather.py writes, so the"
echo "    splice -> Rockstar -> compare gate runs on it unchanged."
