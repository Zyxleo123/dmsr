#!/bin/bash
# Does replacing the MATERIAL of each supervised group help, and does placing it
# at HR's radius help on top of that?
#
#   arm A   relocate=off      HR memberships          frozen-SR radii
#   arm B   relocate=frozen   new local SR groups     frozen-SR radii
#   arm C   relocate=hr       new local SR groups     HR radii
#
# Same HR targets, same loss, same split, same training setup in all three. B
# isolates replacing the membership; C adds the placement change on top of B.
# See src/cosmo_sr/features/relocate_sets.py and docs/sr2_relocate_sets.md.
#
# WHY. `gather-holdout-rockstar-gate` is a real win and its residual defects are
# placement ones. docs/sr2_member_gather_training.md records the symptom this
# targets: the SUPERVISED centres barely move while the detected catalog
# concentrates centrally. If SR2's material for a subhalo is not where HR's was,
# the loss spends itself dragging distant particles together instead of binding
# material that is already there.
#
# TWO PHASES, and the first is not optional.
#
#   RA_PHASE=check   (default) shakeout only. Builds every arm's pool, writes
#                    the relocation report into pool.json, trains nothing.
#                    Seeds at the right radii are NOT enough -- a correct seed
#                    can carry a displaced centroid, and a radius with no
#                    material produces a starved set that silently keeps its HR
#                    membership. Read `relocation` in each pool.json first.
#   RA_PHASE=train   the three full runs, each behind its own shakeout.
#
#   bash scripts/slurm/submit_relocate_arms.sh                    # phase 1
#   RA_PHASE=train bash scripts/slurm/submit_relocate_arms.sh     # phase 2
#   RA_ARMS="B C" RA_PHASE=train bash scripts/slurm/submit_relocate_arms.sh
#   DRY=1 bash scripts/slurm/submit_relocate_arms.sh
#
# This submitter only calls another submitter, which only calls sbatch. No
# --export anywhere: each inner call writes its own timestamped env file and
# passes the path as a positional argument.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"

RA_PHASE="${RA_PHASE:-check}"
RA_ARMS="${RA_ARMS:-A B C}"
RA_TAG="${RA_TAG:-reloc}"
export DRY="${DRY:-0}"

case "$RA_PHASE" in
  check|train) ;;
  *) echo "RA_PHASE=$RA_PHASE is not 'check' or 'train'" >&2; exit 2 ;;
esac

# --- the shared training setup ----------------------------------------------
# Every knob below is IDENTICAL across the three arms, deliberately: the only
# difference between them must be which particles each group is made of and at
# what radius, or the gates cannot be read against each other.
#
# The loss is the MINIMAL SET (`loss-term-ablation`): virial + centre + the
# low-k guard. d6 measured harmful, centre restrains overproduction, the low-k
# guard prevents drift. centre_mode=self is the mode this experiment needs --
# relocation sets each group's `centre_self` to its newly selected centroid, so
# the anchor holds the group where it was put and the internal terms gather it.
export MG_RUNG="${MG_RUNG:-all_blocks}"
export MG_CENTRE_MODE="${MG_CENTRE_MODE:-self}"
export MG_W_VIRIAL="${MG_W_VIRIAL:-1.0}"
export MG_W_BOUND="${MG_W_BOUND:-0.0}"
export MG_W_D6="${MG_W_D6:-0.0}"
export MG_W_RRMS="${MG_W_RRMS:-0.0}"
export MG_W_SIGMAV="${MG_W_SIGMAV:-0.0}"
export MG_W_CENTRE="${MG_W_CENTRE:-1.0}"
export MG_STEPS="${MG_STEPS:-8000}"
export MG_SEED="${MG_SEED:-0}"
# Relocation and soft membership both replace the member set -- one before
# training, one every step -- and no result from a run with both could be
# attributed to either. The config refuses the combination; refuse it here too,
# where it costs nothing instead of after the queue.
export MG_SOFT_MEMBERSHIP=0

# The selection knobs. eps is in the HOST's R_vir: a cluster R_vir is ~2.1 Mpc/h
# and the HR lattice is 0.195 Mpc/h between particles, so a 200-particle group's
# centroid is good to ~r_rms/sqrt(N) ~ 0.03 Mpc/h -- comfortably inside 0.05
# R_vir. If the check phase reports frac_within_eps well below 1, the answer is
# to widen this or to look at WHICH radii starved, not to train anyway.
export MG_RELOCATE_EPS="${MG_RELOCATE_EPS:-0.05}"
export MG_RELOCATE_DIRS="${MG_RELOCATE_DIRS:-8}"
export MG_RELOCATE_REFINE_ITERS="${MG_RELOCATE_REFINE_ITERS:-3}"
export MG_RELOCATE_OVERSAMPLE="${MG_RELOCATE_OVERSAMPLE:-4.0}"
export MG_RELOCATE_W_VEL="${MG_RELOCATE_W_VEL:-0.5}"
export MG_RELOCATE_W_RADIUS="${MG_RELOCATE_W_RADIUS:-1.0}"

arm_mode() {
    case "$1" in
        A) echo off ;;
        B) echo frozen ;;
        C) echo hr ;;
        *) echo "unknown arm '$1' -- expected A, B or C" >&2; exit 2 ;;
    esac
}

echo "=== relocation arms: $RA_ARMS, phase $RA_PHASE"
echo "    rung $MG_RUNG, centre_mode $MG_CENTRE_MODE, $MG_STEPS steps"
echo "    loss: virial $MG_W_VIRIAL, centre $MG_W_CENTRE, bound $MG_W_BOUND,"
echo "          d6 $MG_W_D6, rrms $MG_W_RRMS, sigmav $MG_W_SIGMAV"
echo "    eps $MG_RELOCATE_EPS R_vir_host, $MG_RELOCATE_DIRS directions"
echo

SUBMITTED=()
for arm in $RA_ARMS; do
    mode="$(arm_mode "$arm")"
    label="_${RA_TAG}_${arm}"
    # Arm A has nothing to check -- its membership is HR's owner array and there
    # is no selection to inspect -- so the check phase skips it rather than
    # queueing a shakeout whose report would be empty.
    if [ "$RA_PHASE" = "check" ] && [ "$mode" = "off" ]; then
        echo "--- arm A: no selection to check (membership is HR's owner "
        echo "    array). Skipped in the check phase; it runs in phase train."
        echo
        continue
    fi
    echo "--- arm $arm  (--relocate $mode)  label $label"
    if [ "$RA_PHASE" = "check" ]; then
        MG_RELOCATE="$mode" MG_LABEL="$label" SHAKEOUT_ONLY=1 \
            bash scripts/slurm/submit_member_gather_train.sh
    else
        MG_RELOCATE="$mode" MG_LABEL="$label" \
            bash scripts/slurm/submit_member_gather_train.sh
    fi
    SUBMITTED+=("${MG_RUNG}${label}")
    echo
done

echo "======================================================================"
if [ "$RA_PHASE" = "check" ]; then
    echo "CHECK PHASE. Nothing trains. For each arm read:"
    for a in "${SUBMITTED[@]}"; do
        echo "  $REWARD_ROOT/member_gather/$a/pool.json   -> .relocation"
    done
    echo
    echo "The three numbers that decide whether phase 2 is worth running:"
    echo "  frac_within_eps   near 1, or the material is not at the requested"
    echo "                    radii and training answers a different question"
    echo "  n_starved         near 0. A starved set keeps its HR membership --"
    echo "                    it is an arm-A row inside a B/C run, and every"
    echo "                    downstream split must exclude it"
    echo "  kept_hr_member_frac  near 1 means the selector re-found HR's own"
    echo "                    particles and the arm is arm A under another name"
    echo "  rrms_over_hr      >> 1 means material was recruited from far away"
    echo
    echo "Then: RA_PHASE=train bash scripts/slurm/submit_relocate_arms.sh"
else
    echo "TRAINING. Watch each arm's fine-tune job with the tail -F lines the"
    echo "inner submitter printed above; each arm writes to"
    echo "  $REWARD_ROOT/member_gather/<rung><label>/"
    echo
    echo "THE VERDICT THOSE JOBS PRINT IS NOT THE RESULT (see"
    echo "tile-overfit-proxy-exploitation: a member-set surrogate has read +255"
    echo "on this line while real Rockstar showed nothing). The result is the"
    echo "held-out Rockstar gate, and arm A is its own baseline here:"
    echo
    for a in "${SUBMITTED[@]}"; do
        echo "  MG_ARMS=$a HG_AFTER_JOB=<that arm's fine-tune job id> \\"
        echo "    HG_BASELINE_ARM=${MG_RUNG}_${RA_TAG}_A HG_BASELINE_NAME=armA \\"
        echo "    bash scripts/slurm/submit_gather_holdout_rockstar.sh"
    done
    echo
    echo "MEASURE BOTH YIELD AND THE FULL CATALOG. How many chosen groups became"
    echo "distinct Rockstar subhalos, at what masses and radii, AND whether the"
    echo "unwanted central objects are still there -- the gate reports both, and"
    echo "the second half is the one this experiment exists to move."
fi
