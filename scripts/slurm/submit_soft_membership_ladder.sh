#!/bin/bash
# Does relaxing MEMBERSHIP help, and is the coarse prior safe on its own?
#
# Two stages, and the first is not optional:
#
#   stage 1 (default)  one GPU shakeout: build the pools, print their size and
#                      their O(N^2) pair bill, stop. READ pool.json.
#   stage 2 (LADDER_GO=1)  the four-arm ladder plus its control, as training runs.
#
# WHY THE GATE. The relaxation multiplies the pair bill by (1 + ratio)^2 -- at
# MG_SOFT_POOL_RATIO=2 that is ~9x -- and the hard arm already ran 10.6 h
# against a 12 h limit at 8000 steps. Committing five arms to that on an
# estimate is how a ladder turns into five walltime kills. pool.json's
# `soft_pools.sum_pool_squared` against the hard arm's `sum_num_p_squared` is
# the actual ratio, per host, before any of it is spent.
#
# THE LADDER. `lambda` is the KL weight against the prior and the only knob that
# interpolates, so the controls are corners of the same objective rather than
# different code paths:
#
#   control      hard mask, no relaxation        the arm on disk, re-run
#   hard_pin     prior=hard,   lambda=1e6        must reproduce `control`
#   coarse_pin   prior=coarse, lambda=1e6        the coarse window ALONE
#   hard_soft    prior=hard,   lambda=1          HR's mask, opened
#   coarse_soft  prior=coarse, lambda=1          the proposal as written
#
# `coarse_pin` is the one to read first and it is a FALSIFICATION arm. An LR
# cell is 1.5625 Mpc/h a side against a 0.15 Mpc/h search radius, so at large
# lambda every candidate is a cell-wide blob -- which is the Gaussian window
# that already failed on this line (docs/sr2_gather_finetune.md section 6: its
# statistics described the surrounding host material, not the object). If
# `coarse_soft` beats `coarse_pin` by a wide margin, the phase-space solve is
# what is doing the work. If `coarse_pin` is merely bad and `coarse_soft` is
# only slightly better, the prior is carrying the damage and the answer is
# `hard_soft`, not more lambda.
#
# `hard_pin` must land on `control`. It is the same objective by construction
# (lambda -> inf pins P at the prior, and the hard prior IS the HR mask), so a
# gap between them is a bug in the relaxation, not a result -- and it is much
# cheaper to find here than in a Rockstar gate.
#
# Node choice, checked with `myfree` at write time: `general` has gpu28 (a5000,
# 24 GB, 420 GB host free) and gpu30/gpu31 (a6000, 48 GB) AVAILABLE -- three
# free GPUs against five arms, so the tail of the ladder WILL queue. That is
# expected; the arms are independent and are submitted as siblings rather than
# chained, so they fill the partition as it drains.
#
# This submitter ONLY calls sbatch, through submit_member_gather_train.sh, which
# writes one timestamped env file per arm and passes it as a POSITIONAL
# argument. Never `sbatch --export`: on this cluster that sets
# SLURM_GET_USER_ENV=1 and the job is requeued and HELD.
#
#   bash scripts/slurm/submit_soft_membership_ladder.sh              # shakeout
#   LADDER_GO=1 bash scripts/slurm/submit_soft_membership_ladder.sh  # the ladder
#   LADDER_ARMS="coarse_soft hard_soft" LADDER_GO=1 bash ...         # a subset
#   DRY=1 LADDER_GO=1 bash ...                                       # print only
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
LADDER_GO="${LADDER_GO:-0}"
DRY="${DRY:-0}"

# --- what every arm shares ---------------------------------------------------
# The objective is the MINIMAL SET (`loss-term-ablation`): virial + centre +
# low-k guard, with d6 off -- d6 was measured HARMFUL and it held 88% of the
# budget. centre_mode=self is the mode of the one arm that has ever passed a
# held-out Rockstar gate. Changing any of this at the same time as the
# membership would make the ladder unreadable, so it is pinned here and the
# ONLY thing that varies between arms is the prior and lambda.
: "${MG_RUNG:=all_blocks}"
: "${MG_CENTRE_MODE:=self}"
: "${MG_W_D6:=0.0}"
: "${MG_W_HOST_SETS:=1.0}"
: "${MG_STEPS:=4000}"
: "${MG_SOFT_POOL_RATIO:=2.0}"
# The per-set cap is what bounds the O(pool^2) bill, and it is NOT optional on a
# soft arm: without it a large satellite's pool is (1+ratio) times an already
# large set. 768 keeps a 256-member set's pool exact (256*3 = 768) and thins
# only the sets above that, where the estimator is a noisy-but-unbiased
# subsample rather than a different objective.
: "${MG_MAX_SET_PARTICLES:=768}"
# 8000 steps ran 10.6 h against a 12 h limit on the HARD arm. These do more work
# per step even capped, so the ladder halves the steps and raises the limit.
# Re-cut both against the shakeout's measured pair bill.
: "${MG_TIME:=16:00:00}"
: "${MG_MEM:=48G}"

export MG_RUNG MG_CENTRE_MODE MG_W_D6 MG_W_HOST_SETS MG_STEPS \
       MG_SOFT_POOL_RATIO MG_MAX_SET_PARTICLES MG_TIME MG_MEM

LADDER_ARMS="${LADDER_ARMS:-control hard_pin coarse_pin hard_soft coarse_soft}"

arm_env() {
    # Echoes the two variables that define an arm. Everything else is shared.
    case "$1" in
        control)     echo "MG_SOFT_MEMBERSHIP=0 MG_SOFT_PRIOR=hard   MG_SOFT_LAMBDA=1.0" ;;
        hard_pin)    echo "MG_SOFT_MEMBERSHIP=1 MG_SOFT_PRIOR=hard   MG_SOFT_LAMBDA=1e6" ;;
        coarse_pin)  echo "MG_SOFT_MEMBERSHIP=1 MG_SOFT_PRIOR=coarse MG_SOFT_LAMBDA=1e6" ;;
        hard_soft)   echo "MG_SOFT_MEMBERSHIP=1 MG_SOFT_PRIOR=hard   MG_SOFT_LAMBDA=1.0" ;;
        coarse_soft) echo "MG_SOFT_MEMBERSHIP=1 MG_SOFT_PRIOR=coarse MG_SOFT_LAMBDA=1.0" ;;
        *) echo "unknown arm $1" >&2; return 1 ;;
    esac
}

launch() {
    local arm="$1"; shift
    local vars; vars="$(arm_env "$arm")"
    local cmd="env $vars MG_LABEL=_sm_$arm $* bash scripts/slurm/submit_member_gather_train.sh"
    echo "--- $arm: $vars"
    if [ "$DRY" = "1" ]; then
        echo "    DRY: $cmd"
    else
        eval "$cmd"
    fi
}

if [ "$LADDER_GO" != "1" ]; then
    echo "=== STAGE 1: pool shakeout only (coarse prior, ratio $MG_SOFT_POOL_RATIO)"
    echo "    No training is submitted. Set LADDER_GO=1 to submit the ladder."
    echo
    launch coarse_soft SHAKEOUT_ONLY=1
    echo
    echo "READ THIS BEFORE SUBMITTING THE LADDER:"
    echo "  $REWARD_ROOT/member_gather/${MG_RUNG}_sm_coarse_soft/pool.json"
    echo
    echo "  soft_pools.pool_over_live    how much wider the candidate pool is"
    echo "                               than the HR mask. ~$(awk "BEGIN{print 1+$MG_SOFT_POOL_RATIO}")x is expected."
    echo "  soft_pools.sum_pool_squared  the pair bill this commits a step to."
    echo "                               Divide by the hard arm's sum_num_p_squared:"
    echo "                               that ratio times 10.6 h is the 8000-step"
    echo "                               cost, and MG_STEPS is halved to $MG_STEPS"
    echo "                               against it. Re-cut MG_TIME if it is off."
    echo "  soft_pools.n_pool_max        the largest single set's pool. If this is"
    echo "                               far above MG_MAX_SET_PARTICLES ($MG_MAX_SET_PARTICLES) the cap"
    echo "                               is doing the bounding, which is intended."
    echo
    echo "Then: LADDER_GO=1 bash scripts/slurm/submit_soft_membership_ladder.sh"
    exit 0
fi

echo "=== STAGE 2: the ladder -- $LADDER_ARMS"
echo "    shared: rung $MG_RUNG, centre_mode $MG_CENTRE_MODE, w_d6 $MG_W_D6,"
echo "            host guard $MG_W_HOST_SETS, $MG_STEPS steps,"
echo "            pool ratio $MG_SOFT_POOL_RATIO, per-set cap $MG_MAX_SET_PARTICLES"
echo
for arm in $LADDER_ARMS; do
    # Siblings, not a chain: the arms are independent and chaining them would
    # serialise five GPU-days behind one queue slot.
    launch "$arm" SKIP_SHAKEOUT=1
    echo
done

cat <<EOS

=== WATCH
Each arm printed its own job id and log path above. All of them land in:
  tail -F $REWARD_ROOT/logs/slurm-mgtrain-<jobid>.out
  squeue -u \$USER -n mgtrain

=== RESULTS, per arm <a> in: $REWARD_ROOT/member_gather/${MG_RUNG}_sm_<a>/
  pool.json      hosts, sets, reachable reference, AND soft_pools (the cost)
  metrics.jsonl  one row per eval, train AND holdout, per host
  summary.json   config, history, VERDICT (feasibility only)
  tuned.pt       {'model': full state dict} -- what the gate loads

=== READ IN THIS ORDER
1. hard_pin vs control. They are the SAME objective by construction. A gap is a
   bug in the relaxation and everything downstream is void until it is closed.
2. soft_mass_ratio, every arm, every eval. It must sit at ~1.0. Below it, lambda
   made the phase-space window unfillable and the candidates are holding less
   mass than they claim -- a misconfiguration, not a result.
3. soft_hr_fraction. 1.0 is the hard arm. This is the number that says how much
   of each solved object is still HR's own material, and a collapse toward 0 is
   the relaxation walking off the object entirely.
4. coarse_pin. If it is catastrophic, the coarse prior is a Gaussian window and
   only the phase-space solve makes it survivable -- read coarse_soft against it,
   not against control.

=== THE VERDICT IN summary.json IS NOT THE RESULT
Every number in it is a member-set surrogate the objective computes about
itself, and this line has measured such a surrogate reaching +255 while real
Rockstar showed no gain at all (tile-overfit-proxy-exploitation). And this arm
is MORE exposed to that than any before it: the loss now chooses its own
particles, so "the statistics look right" and "a halo finder sees a halo" have
come further apart, not closer. The result is the held-out whole-box gate:

  MG_ARMS="${MG_RUNG}_sm_coarse_soft ${MG_RUNG}_sm_hard_soft" \\
    HG_BASELINE_ARM=${MG_RUNG}_sm_control HG_BASELINE_NAME=hard_mask \\
    bash scripts/slurm/submit_gather_holdout_rockstar.sh

Read it in the EDITED REGION, never box-wide: 93.75% of the box is untouched
frozen SR2 and swamps the signal.
EOS
