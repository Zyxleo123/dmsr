#!/bin/bash
# Why does the soft shell fit on train and not transfer? Three levers + the 2x2.
#
# THE FINDING THIS ANSWERS (2026-09-08). Raw `term_raw_shell`, step 0 -> 8000:
#     w=1   train 0.0507 -> 0.0589 (worse)   holdout 0.0539 -> 0.0844 (worse)
#     w=10  train 0.0507 -> 0.0205 (2.5x)    holdout 0.0539 -> 0.0599 (worse)
#     w=30  train 0.0507 -> 0.0069 (7.3x)    holdout 0.0539 -> 0.0390 (1.4x)
# Only w=30 moves out of sample, at a 5.6x train/holdout gap. Two things it is
# NOT: a pooling artifact (MG_SETS_PER_STEP=0 charges every set on both sides),
# and a target made of noise -- split-half reliability of a host's own radial
# quantile curve is 0.84-0.94 on HR set9 (101k subhalos), so each host really
# does have its own radial distribution. But host MASS explains only R^2
# 0.01-0.02 of it. The signal is real; it is just not in the obvious feature.
#
# The prime suspect is the FIXED NOISE DRAW: `build_tiles` takes one seed and a
# run holds (LR tile, noise tile) constant for 8000 steps, so the noise is an
# input channel with real per-tile detail and zero generalisable information.
# Anything the LR field does not carry can still be fitted through it, and all
# of it evaporates on a held-out box's fresh draw. Receptive field is NOT the
# cause: num_levels=3 over a 64-cell (12.5 Mpc/h) tile spans the host.
#
# THE ARMS. All sit on `all_blocks` + w_shell=30 + centre_mode=self_tan, which
# is `all_blocks_softshell30` exactly, so each is a one-variable change from a
# run already on disk.
#
#   ss30_fresh    the CONTROL and the measurement. Unchanged objective; only
#                 --fresh-noise-eval added. This is the arm that ARBITRATES:
#                 it re-scores the existing setup on a draw no step ever saw.
#   ss30_rot      (a) --noise-mode rotate over 2 draws. Removes the channel.
#   ss30_navg     (b) + shell charged on the MEAN reduced curve over 2 draws.
#   ss30_q4       (c) --shell-quantiles 4. Drops the rank-by-rank wiggle.
#   ss30_rot_q4   (a) + (c), the combination.
#
# READ ss30_fresh FIRST. If train_fresh falls back to holdout, the gap was
# memorised noise and (a)/(b) are the fixes; if train_fresh tracks train, the
# model learned from the LR field, the gap is about boxes and hosts, and (c) is
# the lever that matters. The other four are already queued either way, which is
# the point of running them as siblings -- but the interpretation is fixed by
# this one and should be written down before the others land.
#
# COST. Every arm carrying more than one draw pays a pool build of roughly
# N_DRAWS x the single-draw run, because centre_self and the hybrid reference
# are functions of the frozen field and each draw needs its own table.
#
# This submitter ONLY calls sbatch (through the single-arm submitter, which does
# nothing else either). Run it FROM THE LOGIN NODE.
#
#   bash scripts/slurm/submit_shell_levers.sh
#   ARMS="ss30_fresh" bash scripts/slurm/submit_shell_levers.sh   # just the 2x2
#   DRY=1 bash scripts/slurm/submit_shell_levers.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
DRY="${DRY:-0}"
ARMS="${ARMS:-ss30_fresh ss30_rot ss30_navg ss30_q4 ss30_rot_q4}"

# The frozen configuration shared by every arm: `all_blocks_softshell30`.
export MG_RUNG="${MG_RUNG:-all_blocks}"
export MG_W_SHELL="${MG_W_SHELL:-30.0}"
export MG_CENTRE_MODE="${MG_CENTRE_MODE:-self_tan}"
export MG_STEPS="${MG_STEPS:-8000}"
export MG_GATE_AT_STEPS="${MG_GATE_AT_STEPS:-2000}"
# The shakeout is the same pool for every arm; run it once, on the first arm.
export SKIP_SHAKEOUT="${SKIP_SHAKEOUT:-0}"
# --- the resource request ----------------------------------------------------
# MEASURED (job 46659, ss30_fresh pool-only shakeout): peak host RSS after the
# pool build is 10.1 GB at 56 hosts and 2 noise draws. The batch script's own
# default is 32G, which covers the 3-draw arms and the mmap'd frozen box too.
# This is not a detail: the first launch of this set went out at the old,
# never-measured 96G and every arm sat PENDING because that fit no available
# node on `general` -- which reads as a fairshare problem and is not one.
export MG_MEM="${MG_MEM:-32G}"
# 8000 steps ran 10.6 h against a 12 h limit, so the default margin is ~1.4 h.
# Arms that add work per step need more; `ss30_navg` adds a forward pass per
# step for the noise-averaged shell and gets its own limit below.
export MG_TIME="${MG_TIME:-14:00:00}"
MG_TIME_NAVG="${MG_TIME_NAVG:-20:00:00}"

submit_arm() {
    local label="$1"; shift
    echo "=================================================================="
    echo "arm $label:  $*"
    echo "=================================================================="
    # `env` rather than exports so one arm's settings cannot leak into the next
    # -- the failure mode that would silently make two arms the same run.
    if [ "$DRY" = "1" ]; then
        echo "DRY: env $* MG_LABEL=_$label bash scripts/slurm/submit_member_gather_train.sh"
        return 0
    fi
    env "$@" MG_LABEL="_$label" \
        bash scripts/slurm/submit_member_gather_train.sh
    # Only the first arm needs the pool shakeout; the rest share its verdict.
    export SKIP_SHAKEOUT=1
}

for arm in $ARMS; do
    case "$arm" in
    # The control: objective untouched, measurement added.
    ss30_fresh)
        submit_arm ss30_fresh MG_FRESH_NOISE_EVAL=1
        ;;
    # (a) the memorisation channel removed.
    ss30_rot)
        submit_arm ss30_rot MG_NOISE_DRAWS=2 MG_NOISE_MODE=rotate \
                   MG_FRESH_NOISE_EVAL=1
        ;;
    # (b) mean-then-charge: a distribution no single realisation can fake.
    ss30_navg)
        submit_arm ss30_navg MG_NOISE_DRAWS=2 MG_NOISE_MODE=rotate \
                   MG_SHELL_NOISE_DRAWS=1 MG_FRESH_NOISE_EVAL=1 \
                   MG_TIME="$MG_TIME_NAVG"
        ;;
    # (c) ask only for the low-order shape.
    ss30_q4)
        submit_arm ss30_q4 MG_SHELL_QUANTILES=4 MG_FRESH_NOISE_EVAL=1
        ;;
    ss30_rot_q4)
        submit_arm ss30_rot_q4 MG_SHELL_QUANTILES=4 MG_NOISE_DRAWS=2 \
                   MG_NOISE_MODE=rotate MG_FRESH_NOISE_EVAL=1
        ;;
    *)
        echo ">>> unknown arm '$arm'. Known: ss30_fresh ss30_rot ss30_navg" \
             "ss30_q4 ss30_rot_q4" >&2
        exit 1
        ;;
    esac
done

R="${DMSR_REWARD_ROOT:-/zfsauton/scratch/yixiz/DMSR/dmsr_reward}"
cat <<MON

=================================================================
MONITOR
=================================================================
  squeue -u "\$USER" -o '%.9i %.30j %.8T %.10M %R'

  tail -F $R/logs/slurm-mgtrain-<jobid>.out

RESULTS, per arm (label = the arm name with a leading underscore):
$(for a in $ARMS; do echo "  $R/member_gather/all_blocks_$a/"; done)

  metrics.jsonl   one row per eval. With MG_FRESH_NOISE_EVAL=1 each row also
                  carries train_fresh / holdout_fresh -- the reserved draw.
  summary.json    config, history, verdict (feasibility only, NOT the gate).

READ IT LIKE THIS -- the numbers that decide the question:
  python - <<'PY'
  import json
  rows=[json.loads(l) for l in open("$R/member_gather/all_blocks_ss30_fresh/metrics.jsonl")]
  for r in (rows[0], rows[-1]):
      print(r["step"], {k: round(r[k]["term_raw_shell"], 5)
                        for k in ("train","holdout","train_fresh","holdout_fresh")
                        if k in r})
  PY
  train_fresh ~ holdout  -> the gap was memorised noise; (a)/(b) are the fix.
  train_fresh ~ train    -> the LR field carried it; the gap is boxes/hosts,
                            and (c) is the lever to read.

The Rockstar A/B gate is a SEPARATE job and is what any of these has to pass:
  HG_AFTER_JOB=<trainjob> bash scripts/slurm/submit_gate_ab.sh
MON
