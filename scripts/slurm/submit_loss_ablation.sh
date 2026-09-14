#!/bin/bash
# Leave-one-out ablation of the member-gather loss, on the gate-winning `self` arm.
#
#   bash scripts/slurm/submit_loss_ablation.sh
#   ABL_ARMS="repro nobound nod6" bash scripts/slurm/submit_loss_ablation.sh
#   DRY=1 bash scripts/slurm/submit_loss_ablation.sh
#   ABL_SET=minimal bash scripts/slurm/submit_loss_ablation.sh   # minimal-set search
#
# WHAT IS BEING ABLATED. `all_blocks_self` (centre_mode self, every weight at its
# default) is the only arm on this line that moved a halo finder out of sample:
# set9 host 168880 subhalos 20 -> 366 vs HR 369, edited-region subs 4541 -> 9187,
# hosts 3028 -> 2708 (gather-holdout-rockstar-gate). Its loss is a plain weighted
# sum (finetune_member_gather.host_loss + member_gather.member_gather_loss):
#
#     virial 1.0   bound 1.0   d6 1.0   rrms 0.3   sigmav 0.3   centre 1.0
#     + guards:    low_k 100   highk 10
#
# so a weight of 0 removes exactly one term and nothing else. Each arm below is
# `self` with ONE weight zeroed; `repro` zeroes nothing.
#
# WHY `repro` IS NOT OPTIONAL. `self` was trained 2026-08-24 and the trainer has
# changed a lot since (term_norm, shell, isolation, cond channel, pool changes).
# Every other arm here is read as "repro minus one term", never as "the August
# checkpoint minus one term". If repro does not land near 366 / 9187 / 2708, the
# ablation is uninterpretable and that is the first finding.
#
# `nocentre` already exists as all_blocks_nocentre (w_centre 0, gated set9:
# region subs 7798, hosts 2621) -- but under the August code. It is rerun here so
# the whole table shares one code version; drop it from ABL_ARMS to save 10 GPU h.
#
# ONE SEED. Rockstar-count noise between seeds is unmeasured for tuned arms (the
# resplice noise on supervised targets is +-9 of 154). Treat a difference smaller
# than repro-vs-August-self as noise until a second seed says otherwise.
#
# COST. 8000 steps at ~4.5 s/step = ~10.5 h per arm on an A5000/A6000 (12 h limit;
# self measured 10.6 h), plus a ~19 min gate per checkpoint. 9 arms = ~95 GPU h.
# MG_GATE_AT_STEPS=2000 snapshots at ~2.5 h, and its gate is submitted now, so
# a first read of every arm arrives long before the runs finish (the objective
# is essentially done by step 2000: cheap-rockstar-ab-gate / early gate notes).
#
# RERUNNABLE, AND IT MUST BE. The `cpu` partition caps a user at 50 queued jobs
# (QOSMaxSubmitJobPerUserLimit, hit on the first launch 2026-09-12). Every gate
# is 4 CPU jobs (splice, rockstar, compare, verdict; +1 wait for an early gate),
# so 9 arms x 2 gates is ~80 CPU jobs and can NEVER be queued at once. So:
#   - each arm keeps a LEDGER (<run dir>/.abl_ledger/{train,final,step<N>,table})
#     of the job ids it submitted;
#   - a rerun skips anything done (verdict on disk) or in flight (a ledgered job
#     still in the queue), and submits the rest in priority order -- training,
#     then FINAL gates, then early gates, then the table -- while the live CPU
#     count stays under ABL_CPU_LIMIT;
#   - a gate whose jobs left the queue without a verdict is reported FAILED and
#     not resubmitted unless ABL_RETRY=1.
# Rerun the same command every few hours until it says nothing is left.
#
# An early gate is only submitted once its training job has FINISHED: the export
# replays summary.json, which the trainer writes only at the end.
#
# This submitter ONLY calls sbatch/squeue/scancel (scancel only on a chain it
# just half-submitted itself).
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

ABL_STEPS="${ABL_STEPS:-8000}"
ABL_GATE_AT="${ABL_GATE_AT:-2000}"
ABL_BOX="${ABL_BOX:-set9}"
ABL_BASELINE_ARM="${ABL_BASELINE_ARM:-all_blocks_self}"
# The wait job polls for the step-$ABL_GATE_AT snapshot from the moment it
# starts, so its budget must cover the TRAINING job's queue wait too. 700 min
# stays under the wait job's 12 h wall. If it still times out, only the early
# gate is lost; the final gate hangs off the training job, not the wait.
ABL_WAIT_MINUTES="${ABL_WAIT_MINUTES:-700}"

# The base: `self` exactly as all_blocks_self ran it. Weights are spelled out
# rather than inherited so a future default change cannot silently move the base.
BASE="MG_RUNG=all_blocks MG_CENTRE_MODE=self MG_SEED=0 MG_MAX_SET_PARTICLES=0"
BASE="$BASE MG_W_VIRIAL=1.0 MG_W_BOUND=1.0 MG_W_D6=1.0 MG_W_RRMS=0.3"
BASE="$BASE MG_W_SIGMAV=0.3 MG_W_CENTRE=1.0 MG_W_LOW=100.0 MG_W_HIGHK=10.0"

# LABEL | the ONE override on top of BASE.
# min_vc_l, the minimal set: virial + centre + the low-k guard, everything else off.
MINVC="MG_W_D6=0.0 MG_W_RRMS=0.0 MG_W_SIGMAV=0.0 MG_W_HIGHK=0.0 MG_W_BOUND=0.0"

declare -a ARMS=(
"repro|"
"novirial|MG_W_VIRIAL=0.0"
"nobound|MG_W_BOUND=0.0"
"nod6|MG_W_D6=0.0"
"norrms|MG_W_RRMS=0.0"
"nosigmav|MG_W_SIGMAV=0.0"
"nocentre|MG_W_CENTRE=0.0"
"nolow|MG_W_LOW=0.0"
"nohighk|MG_W_HIGHK=0.0"
# --- ABL_SET=minimal: backward elimination from nod6 (2026-09-13) ---------------
# The leave-one-out set above cannot name a minimal set: dropping d6 hands its
# ~88% of the loss budget to the others, so a term that "did nothing" with d6 on
# (bound, rrms, sigmav, the high-k guard) may matter once it is gone. These nest:
#   nod6 = virial bound rrms sigmav centre low highk   (11823 subs / 3201 hosts)
#   min_vcb_l  virial centre bound + low guard
#   min_vc_l   virial centre       + low guard
#   min_v_l    virial              + low guard
#   min_vc     virial centre, NO guards (is the low-k guard needed once d6 is gone?)
# Read each against nod6 (final 11823/3201, step 2000 11453/3231) and against its
# parent in the chain. Noise scale: repro vs Aug self, region subs ~1%, host ~50.
"min_vcb_l|MG_W_D6=0.0 MG_W_RRMS=0.0 MG_W_SIGMAV=0.0 MG_W_HIGHK=0.0"
"min_vc_l|MG_W_D6=0.0 MG_W_RRMS=0.0 MG_W_SIGMAV=0.0 MG_W_HIGHK=0.0 MG_W_BOUND=0.0"
"min_v_l|MG_W_D6=0.0 MG_W_RRMS=0.0 MG_W_SIGMAV=0.0 MG_W_HIGHK=0.0 MG_W_BOUND=0.0 MG_W_CENTRE=0.0"
"min_vc|MG_W_D6=0.0 MG_W_RRMS=0.0 MG_W_SIGMAV=0.0 MG_W_HIGHK=0.0 MG_W_BOUND=0.0 MG_W_LOW=0.0"
# --- ABL_SET=shape: fix the mass-function SHAPE, 2026-09-14 --------------------
# min_vc_l recovers the COUNT (11766 region subs vs HR 12367, 0.95) but not the
# shape: 1.5-1.7x HR in every host bin above 200p. Too many massive subhalos, not
# too few small ones. Each arm adds ONE restraint to min_vc_l and is scored by the
# region-wide shape score (gate_verdict_vs_baseline, ~12k objects), not by counts.
#   shp_iso1/3  the ISOLATION term, the only inter-set statement in the loss: it
#               charges non-member mass sitting ON the object against HR's, hinged
#               above. A supervised set that swallowed its neighbourhood is exactly
#               what an over-massive subhalo is, so this is the mechanism-matched
#               fix. Two doses, because the term has never run outside smallhalo.
#   shp_rrms1   r_rms at 1.0 (3x its old 0.3, which the ablation showed was dead
#               weight AT that weight): caps how concentrated a set may get.
#   shp_d6_01   d6 at a tenth. Full d6 suppressed everything (0.8x HR per bin) and
#               is the source of the high-k excess; a tenth asks whether it is a
#               usable brake on the massive end rather than an all-or-nothing term.
# MG_TIME 20 h, not the 12 h default: the isolation term adds a background-particle
# pass per set per step and 8000 steps already ran 10.6 h without it. A run that hits
# the wall writes no tuned.pt and no summary.json, so its gate has nothing to export.
"shp_iso1|$MINVC MG_W_ISOLATION=1.0 MG_TIME=20:00:00"
"shp_iso3|$MINVC MG_W_ISOLATION=3.0 MG_TIME=20:00:00"
"shp_rrms1|$MINVC MG_W_RRMS=1.0"
"shp_d6_01|$MINVC MG_W_D6=0.1"
)
# ABL_SET picks the default arm list and names the table, so the two sets keep
# separate tables and table ledgers. ABL_TABLE_EXTRA rows are already-gated arms
# shown for reference; they are read, never trained or gated.
ABL_SET="${ABL_SET:-loo}"
case "$ABL_SET" in
    loo)     _def="repro novirial nobound nod6 norrms nosigmav nocentre nolow nohighk"
             _tsuffix=""; _extra="" ;;
    minimal) _def="min_vcb_l min_vc_l min_v_l min_vc"
             _tsuffix="_minimal"; _extra="repro nod6" ;;
    shape)   _def="shp_iso1 shp_iso3 shp_rrms1 shp_d6_01"
             _tsuffix="_shape"; _extra="nod6 min_vc_l min_v_l" ;;
    *) echo "ABL_SET must be loo, minimal or shape, got '$ABL_SET'" >&2; exit 1 ;;
esac
ABL_ARMS="${ABL_ARMS:-$_def}"
ABL_TABLE_EXTRA="${ABL_TABLE_EXTRA:-$_extra}"

# The A/B baseline must itself be gated on this box, or every verdict refuses.
_bl="$REWARD_ROOT/flow_rockstar/compare/${ABL_BOX}__mgho_${ABL_BASELINE_ARM}_${ABL_BOX}.json"
if [ ! -f "$_bl" ]; then
    echo "ERROR: baseline $ABL_BASELINE_ARM has no gated catalog for $ABL_BOX:" >&2
    echo "  $_bl" >&2
    exit 1
fi

ABL_CPU_LIMIT="${ABL_CPU_LIMIT:-50}"
ABL_RETRY="${ABL_RETRY:-0}"
MG="$REWARD_ROOT/member_gather"
USER="${USER:-$(id -un)}"

echo "=== loss ablation: base all_blocks_self, $ABL_STEPS steps, gate box $ABL_BOX"
echo "    arms: $ABL_ARMS"
echo "    early gate at step $ABL_GATE_AT, final gate afterok, A/B vs $ABL_BASELINE_ARM"
echo "    cpu budget: $ABL_CPU_LIMIT queued jobs"
echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --- queue helpers -------------------------------------------------------------
refresh_queue() { squeue -u "$USER" -h -o "%i %T %P" > "$TMP/queue"; }
job_state() { awk -v j="$1" '$1==j {print $2}' "$TMP/queue"; }
any_in_queue() {  # any of the ids in ledger file $1 still queued?
    local id; for id in $(cat "$1"); do [ -n "$(job_state "$id")" ] && return 0; done; return 1
}
cpu_used() { awk '$3=="cpu"' "$TMP/queue" | wc -l; }
cpu_free() { echo $(( ABL_CPU_LIMIT - $(cpu_used) )); }
refresh_queue

declare -a PENDING_WORK=() FAILED=() WATCH=()

ledger_dir() { echo "$MG/all_blocks_abl_$1/.abl_ledger"; }

# Run a gate submitter; on success ledger every `-> job N` it printed. On failure
# cancel the half-chain it left behind (an orphan export would later collide with
# the resubmitted gate's tiles.npz) and ledger nothing.
run_gate() {
    local label="$1" stage="$2"; shift 2
    local out="$TMP/$label.$stage"
    if env "$@" NO_CONTROL=1 HG_BOX="$ABL_BOX" \
         HG_BASELINE_ARM="$ABL_BASELINE_ARM" HG_BASELINE_NAME=self \
         bash scripts/slurm/submit_gather_holdout_rockstar.sh > "$out" 2>&1; then
        [ "$DRY" = "1" ] && { echo "  DRY $stage gate: $*"; return 0; }
        sed -n 's/.*-> job \([0-9][0-9]*\).*/\1/p' "$out" > "$(ledger_dir "$label")/$stage"
        echo "  $stage gate -> jobs $(tr '\n' ' ' < "$(ledger_dir "$label")/$stage")"
        grep "hg_verdict" "$out" | sed 's/^/  /' || true
        refresh_queue
        return 0
    fi
    cat "$out"
    local partial; partial=$(sed -n 's/.*-> job \([0-9][0-9]*\).*/\1/p' "$out" | tr '\n' ' ')
    if [ -n "$partial" ] && [ "$DRY" != "1" ]; then
        echo ">>> $label $stage: gate submission failed part-way; cancelling its partial chain: $partial" >&2
        scancel $partial || true
    fi
    refresh_queue
    return 1
}

# --- phase 1: training (GPU; not limited by the cpu budget) ---------------------
declare -A TRAIN_ID=()
for spec in "${ARMS[@]}"; do
    label="${spec%%|*}"; override="${spec#*|}"
    echo "$ABL_ARMS" | grep -qw "$label" || continue
    L="$(ledger_dir "$label")"; mkdir -p "$L"
    if [ -s "$L/train" ]; then
        TRAIN_ID[$label]=$(cat "$L/train")
        continue
    fi
    echo "──────── train $label  (${override:-no override})"
    if [ "$DRY" = "1" ]; then
        echo "  DRY: env $BASE $override SKIP_SHAKEOUT=1 MG_STEPS=$ABL_STEPS MG_GATE_AT_STEPS=$ABL_GATE_AT MG_LABEL=_abl_$label"
        TRAIN_ID[$label]="DRY"; continue
    fi
    # SKIP_SHAKEOUT: the pool is `self`'s default, built fine by this week's
    # all_blocks arms; a bad pool still fails minutes into the training job.
    if ! env $BASE $override SKIP_SHAKEOUT=1 MG_STEPS="$ABL_STEPS" \
             MG_GATE_AT_STEPS="$ABL_GATE_AT" MG_LABEL="_abl_$label" \
             bash scripts/slurm/submit_member_gather_train.sh > "$TMP/$label.train" 2>&1; then
        tail -20 "$TMP/$label.train"
        echo ">>> $label FAILED TO SUBMIT training" >&2
        continue
    fi
    tid=$(sed -n 's/.*submitted fine-tune  *-> job \([0-9]*\).*/\1/p' "$TMP/$label.train" | head -1)
    if [ -z "$tid" ]; then
        tail -20 "$TMP/$label.train"; echo ">>> no job id parsed for $label" >&2; continue
    fi
    echo "$tid" > "$L/train"; TRAIN_ID[$label]=$tid
    echo "  train job $tid"
done
refresh_queue

# --- phase 2+3: gates, FINAL before early -------------------------------------
# state of one gate: done | inflight | failed | todo
gate_state() {
    local label="$1" stage="$2" sub
    [ "$stage" = "final" ] && sub="holdout_$ABL_BOX" || sub="holdout_${ABL_BOX}_$stage"
    if [ -f "$MG/all_blocks_abl_$label/$sub/verdict_vs_self.json" ]; then echo done; return; fi
    local f; f="$(ledger_dir "$label")/$stage"
    if [ -s "$f" ]; then
        if any_in_queue "$f"; then echo inflight; return; fi
        if [ "$ABL_RETRY" = "1" ]; then rm -f "$f"; echo todo; return; fi
        echo failed; return
    fi
    echo todo
}

EARLY="step$ABL_GATE_AT"
for stage in final "$EARLY"; do
    for spec in "${ARMS[@]}"; do
        label="${spec%%|*}"
        echo "$ABL_ARMS" | grep -qw "$label" || continue
        tid="${TRAIN_ID[$label]:-}"
        [ -n "$tid" ] && [ "$tid" != "DRY" ] || continue
        arm="all_blocks_abl_$label"; d="$MG/$arm"
        st=$(gate_state "$label" "$stage")
        case "$st" in
            done|inflight) printf "  %-9s %-9s %s\n" "$label" "$stage" "$st"; continue ;;
            failed) FAILED+=("$label $stage: jobs $(tr '\n' ' ' < "$(ledger_dir "$label")/$stage")"); continue ;;
        esac
        tstate=$(job_state "$tid")
        need=4; extra=()
        if [ "$stage" = "final" ]; then
            if [ -n "$tstate" ]; then extra=(HG_AFTER_JOB="$tid")
            elif [ ! -f "$d/tuned.pt" ]; then
                FAILED+=("$label train: job $tid left the queue with no tuned.pt"); continue
            fi
        else
            extra=(HG_CHECKPOINT="tuned_$stage.pt" HG_WAIT_MINUTES="$ABL_WAIT_MINUTES")
            # The export replays the run config from summary.json, which the
            # trainer writes only at the END (measured 2026-09-13: repro's early
            # gate died in export with summary.json missing). So an early gate
            # cannot run while its training does; queue it once training is done.
            if [ -f "$d/tuned_$stage.pt" ] && [ -f "$d/summary.json" ] && [ -z "$tstate" ]; then :
            elif [ -n "$tstate" ]; then
                PENDING_WORK+=("$label $stage (export needs summary.json; training $tid is $tstate)"); continue
            else
                FAILED+=("$label $stage: training $tid ended with no tuned_$stage.pt"); continue
            fi
        fi
        if [ "$DRY" != "1" ] && [ "$(cpu_free)" -lt "$need" ]; then
            PENDING_WORK+=("$label $stage (needs $need cpu slots, $(cpu_free) free)"); continue
        fi
        echo "──────── $label $stage gate (train $tid ${tstate:-finished})"
        mkdir -p "$d"
        run_gate "$label" "$stage" MG_ARMS="$arm" "${extra[@]}" \
            || PENDING_WORK+=("$label $stage (submission refused, see above)")
    done
done

# --- phase 4: the table, once every final gate is in flight or done -------------
TABLE_OUT="$MG/loss_ablation${_tsuffix}_${ABL_BOX}"
finals_all_in=1; train_deps=(); labels=""
for spec in "${ARMS[@]}"; do
    label="${spec%%|*}"; echo "$ABL_ARMS" | grep -qw "$label" || continue
    labels="$labels $label"
    [ -n "${TRAIN_ID[$label]:-}" ] || { finals_all_in=0; continue; }
    case "$(gate_state "$label" final)" in done|inflight|failed) ;; *) finals_all_in=0 ;; esac
    [ -n "$(job_state "${TRAIN_ID[$label]}")" ] && train_deps+=("${TRAIN_ID[$label]}")
done
TL="$MG/.abl_table${_tsuffix}_${ABL_BOX}"
if [ "$DRY" = "1" ]; then :
elif [ -s "$TL" ] && any_in_queue "$TL"; then
    echo "  table job $(cat "$TL") in flight"
elif [ "$finals_all_in" = "1" ] && { [ ! -s "$TL" ] || [ ! -f "$TABLE_OUT.md" ] \
        || [ -n "$(find "$MG"/all_blocks_abl_*/holdout_* -name verdict_vs_self.json -newer "$TABLE_OUT.md" 2>/dev/null | head -1)" ]; }; then
    # Not yet run, or a verdict landed after the last table: (re)build it.
    if [ "$(cpu_free)" -ge 1 ]; then
        TABLE_ENV="$REWARD_ROOT/env/loss_ablation_table_$(date +%Y%m%d_%H%M%S)_$$.env"
        cat > "$TABLE_ENV" <<EOT
DMSR_REWARD_ROOT="$REWARD_ROOT"
ABL_BOX="$ABL_BOX"
ABL_ARMS="$(echo $ABL_TABLE_EXTRA $labels)"
ABL_GATE_AT="$ABL_GATE_AT"
ABL_BASELINE_ARM="$ABL_BASELINE_ARM"
ABL_TABLE_OUT="$TABLE_OUT"
EOT
        # afterany on TRAINING jobs (which always terminate), never on gate jobs,
        # which can pend forever on DependencyNeverSatisfied. The job then polls
        # for the final verdicts itself.
        dep=(); [ "${#train_deps[@]}" -gt 0 ] && dep=(--dependency="afterany:$(IFS=:; echo "${train_deps[*]}")")
        TBL=$(sbatch --parsable "${dep[@]}" scripts/slurm/loss_ablation_table_cpu.sbatch "$TABLE_ENV")
        echo "$TBL" > "$TL"
        echo "  table job $TBL (env $TABLE_ENV)"
    else
        PENDING_WORK+=("table (no cpu slot free)")
    fi
else
    [ "$finals_all_in" = "1" ] || PENDING_WORK+=("table (waits until every final gate is submitted)")
fi

# --- report ---------------------------------------------------------------------
refresh_queue
echo
echo "════════ cpu partition: $(cpu_used)/$ABL_CPU_LIMIT queued"
echo "watch (training):"
for spec in "${ARMS[@]}"; do
    label="${spec%%|*}"; tid="${TRAIN_ID[$label]:-}"
    echo "$ABL_ARMS" | grep -qw "$label" && [ -n "$tid" ] || continue
    printf "    %-9s %-8s tail -F %s\n" "$label" "$(job_state "$tid" || true)" "$REWARD_ROOT/logs/slurm-mgtrain-$tid.out"
done
[ -s "$TL" ] && echo "    table     tail -F $REWARD_ROOT/logs/slurm-abl_table-$(cat "$TL").out"
echo
echo "results:"
echo "    $TABLE_OUT.md / .json   one row per arm x {step $ABL_GATE_AT, final}"
echo "    $MG/all_blocks_abl_<label>/holdout_${ABL_BOX}{,_step$ABL_GATE_AT}/verdict_vs_self.json"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo
    echo "FAILED (not resubmitted; read the logs, then ABL_RETRY=1 to resubmit):"
    printf '    %s\n' "${FAILED[@]}"
fi
if [ "${#PENDING_WORK[@]}" -gt 0 ]; then
    echo
    echo "NOT YET SUBMITTED -- rerun this same command later (it skips what is in):"
    printf '    %s\n' "${PENDING_WORK[@]}"
else
    echo
    echo "Everything is submitted or done."
fi
echo
if [ "$ABL_SET" = "shape" ]; then
    echo "READ the SHAPE score and the per-bin new/HR, not the totals: min_vc_l already"
    echo "has the count (0.95 of HR) and misses the shape (1.5-1.7x HR above 200p)."
elif [ "$ABL_SET" = "minimal" ]; then
    echo "READ against nod6 (11823 subs / 3201 hosts): the smallest arm that holds it is the minimal set."
else
    echo "READ repro FIRST. Every other row is 'repro minus one term'."
fi
