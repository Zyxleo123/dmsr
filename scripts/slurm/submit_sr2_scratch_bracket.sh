#!/bin/bash
# Find a from-scratch configuration that does not diverge, before committing a
# multi-day run. This script only calls sbatch.
#
#   bash scripts/slurm/submit_sr2_scratch_bracket.sh
#   BR_PROBES='A D' bash scripts/slurm/submit_sr2_scratch_bracket.sh
#
# WHY. Three long runs have now failed for three different reasons:
#   run 1  the critic ran away          (no input normaliser; score gap -> 5718)
#   run 2  the field came out smooth    (--w-base-low 100 was 96-100% of the
#                                        non-adversarial budget; density P/P_HR
#                                        0.153 against TRILINEAR's 0.706)
#   run 3  the generator diverged       (MSE handed off to a pure adversary at
#                                        step 1000; base_mse 0.20 -> 1.6e8 in
#                                        500 steps, D saturated at +8.5/-8.2)
# Each cost ~12 GPU-hours before the problem was visible. Run 3's divergence,
# though, shows up by step 1500 -- so the same question can be answered in
# minutes instead of hours. That is what this does.
#
# THE PROBES. Each runs BR_STEPS steps on one GPU and is judged by whether
# `base_mse` stays O(0.1) instead of exploding.
#
#   A  a REAL MSE phase before the handoff (SC_ADV_WARMUP=20000 > BR_STEPS, so
#      the adversary never takes over inside the probe). Isolates "is the
#      MSE-only phase itself healthy?" -- the control, and it must pass or
#      something more basic is wrong.
#   B  SC_MSE_MODE=sum: keep MSE as a leash alongside the adversary. NOT
#      upstream's form, but it is what runs 1-2 used and they did not diverge.
#   C  upstream's own two stabilisers: spectral norm on D + annealed instance
#      noise, with the short handoff run 3 used. If this holds, run 3 failed for
#      want of stabilisers rather than for want of a longer warmup.
#   D  C plus a longer handoff (SC_ADV_WARMUP=2000 within a 3000-step probe, so
#      the handoff DOES happen and is observed under the stabilisers).
#
# A-D RAN 2026-09-24. A and B held; C and D DIVERGED to 5.9e10 / 4.2e11 within
# one eval interval of w_mse_now hitting 0. The cause was then measured and it
# is NOT a saturated discriminator: SR2Critic's GroupNorms make it EXACTLY
# scale-invariant (D pinned at 0.009504 for input scalings 1e3 to 1e6, with and
# without spectral norm, while ||grad D|| falls as exactly 1/a). With --w-mse 0
# nothing in the objective constrains amplitude at all, and the only term left
# is blind to it and damps its own corrective gradient by 1/a. Hence the
# thrashing across five decades rather than a climb. Probes E and F test the two
# fixes, and both keep the pure-adversarial (`upstream`) form that C and D ran,
# so they are read against C/D directly:
#
#   E  the AMPLITUDE LEASH. SC_W_SCALE=1, a log(rms_cand/rms_hr)^2 term on the
#      displacement and velocity groups: two scalars per batch, no spatial
#      content, so it cannot do what MSE is suspected of doing to the radial
#      distribution of subhalo mass. Tests "is an amplitude constraint all that
#      was missing?"
#   F  the ROOT FIX. SC_CRITIC_INNER_NORM=none + spectral norm, so the critic is
#      positively homogeneous and can see amplitude itself. No leash at all.
#      Tests "can the adversary hold the scale once it is allowed to see it?"
#
# E and F are deliberately separate. If E holds and F does not, keep the leash;
# if F holds, the objective needs no extra term and the radial question is clean.
# Run BR_PROBES='E F' to test them without re-running the settled A-D.
#
# A passing probe is necessary, not sufficient: 3000 steps says the
# configuration is stable, not that it reproduces SR2. The field report on a
# long run is what says that.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
LOGS="$ROOT/logs"; ENVD="$ROOT/sr2_scratch/envs"
mkdir -p "$LOGS" "$ENVD"
STAMP="$(date '+%Y%m%d-%H%M%S')"
JOB="$HERE/sr2_scratch_train_gpu.sbatch"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: you are inside allocation $SLURM_JOB_ID. sbatch only works" >&2
    echo "       from the login node. Exit the allocation and rerun." >&2
    exit 1
fi

BR_PROBES="${BR_PROBES:-A B C D}"
BR_STEPS="${BR_STEPS:-3000}"
BR_PARTITION="${BR_PARTITION:-general}"
BR_TIME="${BR_TIME:-01:00:00}"
BR_MEM="${BR_MEM:-32G}"
BR_SEED="${BR_SEED:-0}"
# The 11 GB rtx_2080_ti. --exclude, not --nodelist: a node LIST forces the node
# COUNT and a multi-node list on a 1-GPU job pends forever.
EXCLUDE_2080="${EXCLUDE_2080:-gpu11,gpu12,gpu13,gpu14,gpu15,gpu16,gpu17,gpu18,gpu19}"

declare -a SUBMITTED=() NAMES=()

for probe in $BR_PROBES; do
    label="probe_${probe}"
    f="$ENVD/${label}-${STAMP}.env"
    {
        echo "# bracket probe $probe, written $(date '+%F %T')"
        echo "# List values are QUOTED: an unquoted list runs its tail as a command."
        echo "SC_LABEL='$label'"
        echo "SC_SEED='$BR_SEED'"
        echo "SC_STEPS='$BR_STEPS'"
        echo "SC_SHAKEOUT_STEPS='$BR_STEPS'"
        echo "SC_GATE_AT_STEPS=' '"        # a SPACE: "" re-fires the default
        echo "SC_EVAL_EVERY='250'"
        echo "SC_SAVE_EVERY='$BR_STEPS'"
        echo "SC_RESUME='0'"              # a probe must never resume anything
        echo "SC_W_GATHER='0.0'"
        echo "SC_W_KNN='0.0'"
        case "$probe" in
        A)  # MSE-only: the handoff is pushed beyond the probe's horizon.
            echo "SC_ADV_WARMUP='20000'" ;;
        B)  # MSE kept alongside the adversary (runs 1-2's form, minus the anchor).
            echo "SC_MSE_MODE='sum'"
            echo "SC_ADV_WARMUP='1000'" ;;
        C)  # upstream's stabilisers, run 3's short handoff.
            echo "SC_ADV_WARMUP='1000'"
            echo "SC_CRITIC_SPECTRAL_NORM='1'"
            echo "SC_INSTANCE_NOISE='0.1'"
            echo "SC_INSTANCE_NOISE_BATCHES='10000'" ;;
        D)  # stabilisers AND a longer handoff, still inside the probe.
            echo "SC_ADV_WARMUP='2000'"
            echo "SC_CRITIC_SPECTRAL_NORM='1'"
            echo "SC_INSTANCE_NOISE='0.1'"
            echo "SC_INSTANCE_NOISE_BATCHES='10000'" ;;
        E)  # D, plus the amplitude-only leash. The critic stays scale-blind.
            echo "SC_ADV_WARMUP='2000'"
            echo "SC_CRITIC_SPECTRAL_NORM='1'"
            echo "SC_INSTANCE_NOISE='0.1'"
            echo "SC_INSTANCE_NOISE_BATCHES='10000'"
            echo "SC_W_SCALE='1.0'"
            echo "SC_MSE_HANDOFF='250'" ;;
        F)  # D, plus a critic that can SEE amplitude. No leash.
            echo "SC_ADV_WARMUP='2000'"
            echo "SC_CRITIC_SPECTRAL_NORM='1'"
            echo "SC_CRITIC_INNER_NORM='none'"
            echo "SC_INSTANCE_NOISE='0.1'"
            echo "SC_INSTANCE_NOISE_BATCHES='10000'"
            echo "SC_MSE_HANDOFF='250'" ;;
        *)  echo "unknown probe: $probe" >&2; exit 1 ;;
        esac
    } > "$f"
    ID=$(sbatch --parsable \
                --job-name="sr2br_$probe" \
                --partition="$BR_PARTITION" \
                --time="$BR_TIME" \
                --mem="$BR_MEM" \
                --exclude="$EXCLUDE_2080" \
                "$JOB" "$f")
    SUBMITTED+=("$ID"); NAMES+=("$label")
    echo "  probe $probe -> $ID   ($BR_STEPS steps)"
done

cat <<EOF

=== watch
EOF
for i in "${!SUBMITTED[@]}"; do
    echo "  ${NAMES[$i]}:"
    echo "    tail -F $LOGS/slurm-sr2scratch-${SUBMITTED[$i]}.out"
done
cat <<EOF

=== verdict (CPU, reads metrics.jsonl only; safe to run while they go)
  PYTHONPATH=src:scripts/reward:scripts/features \\
    python scripts/features/sr2_scratch_bracket_verdict.py

=== results
EOF
for n in "${NAMES[@]}"; do echo "  $n:  $ROOT/sr2_scratch/$n/metrics.jsonl"; done
cat <<'EOF'

=== what to look for
  base_mse staying O(0.1). Run 3 went 0.20 -> 1.6e8 within 500 steps of the
  handoff, so a probe that is still O(0.1) at the end of its run has cleared
  the failure that killed it.

  Read `w_mse_now` alongside it: 1.0 means MSE is still carrying the generator
  and the adversary has not taken over yet. A probe that never reached
  w_mse_now = 0 has NOT tested the handoff -- that is true of probe A by
  design, which is why A is the control and D is the real test.

  And `adv.d_real - adv.d_fake`: run 3 saturated at ~+16.7 with d_loss 0.0005.
  A healthy discriminator stays O(1) and its loss stays near ln2 ~ 0.69.

  For E and F, read `rms_ratio_dis` and `rms_ratio_vel` -- the new amplitude
  columns, both 1.0 when the field's scale matches HR. They are the quantity
  that actually ran away (C/D hit vel_rms_ratio 5.8e5), and they say it 3-4
  decades before base_mse does. A probe whose rms ratios stay O(1) through
  w_mse_now = 0 has fixed the divergence; one whose ratios drift while base_mse
  still looks fine is on its way out.
EOF
