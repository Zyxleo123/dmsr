#!/bin/bash
# Gate EVERY trained member-gather arm on the held-out box, so a good arm cannot
# sit unmeasured. Rerunnable: it submits only what is missing, in waves under the
# cpu partition's 50-job cap.
#
#   bash scripts/slurm/submit_gate_all_arms.sh            # do what fits, report the rest
#   DRY=1 bash scripts/slurm/submit_gate_all_arms.sh      # classify only
#   GA_ARMS="all_blocks_self fine_nocentre" bash scripts/slurm/submit_gate_all_arms.sh
#
# THREE CLASSES, because the work differs by an order of magnitude:
#   A  catalog already on disk, no verdict  -> VERDICT ONLY, 1 cpu job (~30 s).
#      These are arms gated before the verdict stage existed, or used only as a
#      baseline (all_blocks_self is one: its catalog is what every A/B compares
#      against, yet it has no verdict of its own).
#   B  no catalog                           -> full chain, 1 gpu export + 4 cpu.
#   C  no summary.json                      -> CANNOT be exported. The export
#      replays the run config through summary.json, which the trainer writes only
#      at the end; a still-running or cancelled arm has none. Reported, not queued.
#
# Every gate carries the region mass function, the shape score and the radial
# distribution, so this backfills those onto the whole history in one pass.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
GA_BOX="${GA_BOX:-set9}"
GA_BASELINE_ARM="${GA_BASELINE_ARM:-all_blocks_self}"
GA_CPU_LIMIT="${GA_CPU_LIMIT:-50}"
MG="$REWARD_ROOT/member_gather"
USER="${USER:-$(id -un)}"
GA_ARMS="${GA_ARMS:-$(cd "$MG" && ls -d */ 2>/dev/null | tr -d / | tr '\n' ' ')}"

cpu_used() { squeue -u "$USER" -h -p cpu -o "%i" | wc -l; }
cpu_free() { echo $(( GA_CPU_LIMIT - $(cpu_used) )); }

QUEUE="$(mktemp)"; trap 'rm -f "$QUEUE"' EXIT
squeue -u "$USER" -h -o "%i %T" > "$QUEUE"
declare -a DONE_=() CLASS_A=() CLASS_A0=() CLASS_B=() CLASS_C=() DEFERRED=() INFLIGHT=()
for arm in $GA_ARMS; do
    d="$MG/$arm"
    [ -f "$d/tuned.pt" ] || continue
    # The baseline's own verdict is named for itself, not "vs self".
    if [ -f "$d/holdout_${GA_BOX}/verdict_vs_self.json" ] \
       || [ -f "$d/holdout_${GA_BOX}/region_counts.json" ]; then
        DONE_+=("$arm"); continue
    fi
    # IN FLIGHT: this submitter classifies by output files, and a chain takes
    # ~20 min to produce any. Without the ledger a rerun re-submits every chain
    # already queued -- two exports writing one tiles.npz, and 3 GB of spliced
    # field per duplicate on a quota that has already died once.
    _led="$d/.gate_ledger_${GA_BOX}"
    if [ -s "$_led" ]; then
        _live=""
        for _j in $(cat "$_led"); do
            grep -q "^$_j " "$QUEUE" && { _live=1; break; }
        done
        [ -n "$_live" ] && { INFLIGHT+=("$arm"); continue; }
    fi
    if [ ! -f "$d/summary.json" ]; then CLASS_C+=("$arm"); continue; fi
    if [ -f "$REWARD_ROOT/flow_rockstar/compare/${GA_BOX}__mgho_${arm}_${GA_BOX}.json" ]; then
        # The baseline arm is scored STANDALONE: A/Bing it against its own
        # catalog would report a 1.000 ratio and no numbers of its own.
        if [ "$arm" = "$GA_BASELINE_ARM" ]; then CLASS_A0+=("$arm")
        else CLASS_A+=("$arm"); fi
    else
        CLASS_B+=("$arm")
    fi
done

echo "=== gate-all on $GA_BOX (baseline $GA_BASELINE_ARM)"
echo "  already gated:            ${#DONE_[@]}"
echo "  in flight (ledgered):     ${#INFLIGHT[@]}  ${INFLIGHT[*]:-}"
echo "  A verdict only (1 cpu):   $(( ${#CLASS_A[@]} + ${#CLASS_A0[@]} ))  ${CLASS_A[*]:-} ${CLASS_A0[*]:-}"
echo "  B full chain (1gpu+4cpu): ${#CLASS_B[@]}  ${CLASS_B[*]:-}"
echo "  C no summary.json:        ${#CLASS_C[@]}  ${CLASS_C[*]:-}"
echo "  cpu queue: $(cpu_used)/$GA_CPU_LIMIT"
echo
if [ "${#CLASS_C[@]}" -gt 0 ]; then
    echo "  C arms cannot be exported: the export replays summary.json, which the"
    echo "  trainer writes only at the end. A still-RUNNING arm gets gated by its"
    echo "  own chained gate; a CANCELLED one needs summary.json reconstructed"
    echo "  from pool.json before it can be gated at all."
    echo
fi
[ "$DRY" = "1" ] && exit 0

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

# --- class A: the verdict stage alone -----------------------------------------
if [ "${#CLASS_A[@]}" -gt 0 ] || [ "${#CLASS_A0[@]}" -gt 0 ]; then
    _needA=$(( ${#CLASS_A[@]} + ${#CLASS_A0[@]} ))
    if [ "$(cpu_free)" -ge "$_needA" ]; then
        if [ "${#CLASS_A[@]}" -gt 0 ]; then
            echo "──────── A: verdict only, A/B vs $GA_BASELINE_ARM"
            RV_ARMS="${CLASS_A[*]}" RV_STAGES="holdout_${GA_BOX}" RV_BOX="$GA_BOX" \
                RV_BASELINE_ARM="$GA_BASELINE_ARM" \
                bash scripts/slurm/submit_reverdict.sh
        fi
        if [ "${#CLASS_A0[@]}" -gt 0 ]; then
            echo "──────── A: verdict only, STANDALONE (the baseline itself)"
            RV_ARMS="${CLASS_A0[*]}" RV_STAGES="holdout_${GA_BOX}" RV_BOX="$GA_BOX" \
                RV_BASELINE_ARM="" \
                bash scripts/slurm/submit_reverdict.sh
        fi
    else
        DEFERRED+=("class A ($_needA arms, needs $_needA cpu slots)")
    fi
    echo
fi

# --- class B: the full chain, while the budget lasts --------------------------
for arm in "${CLASS_B[@]}"; do
    if [ "$(cpu_free)" -lt 4 ]; then
        DEFERRED+=("$arm (full chain, $(cpu_free) cpu slots free, needs 4)"); continue
    fi
    echo "──────── B: $arm"
    # NO_CONTROL: the frozen control for this box is already gated and shared.
    _out="$(mktemp)"
    if MG_ARMS="$arm" HG_BOX="$GA_BOX" NO_CONTROL=1 \
       HG_BASELINE_ARM="$GA_BASELINE_ARM" HG_BASELINE_NAME=self \
       bash scripts/slurm/submit_gather_holdout_rockstar.sh > "$_out" 2>&1; then
        grep -E "^ *(export|splice|rockstar|compare|region) " "$_out" || true
        sed -n 's/.*-> job \([0-9][0-9]*\).*/\1/p' "$_out" > "$MG/$arm/.gate_ledger_${GA_BOX}"
        squeue -u "$USER" -h -o "%i %T" > "$QUEUE"
    else
        cat "$_out"
        echo ">>> $arm: gate submission FAILED (see above)" >&2
    fi
    rm -f "$_out"
done

echo
echo "cpu queue now: $(cpu_used)/$GA_CPU_LIMIT"
if [ "${#DEFERRED[@]}" -gt 0 ]; then
    echo
    echo "NOT SUBMITTED -- rerun this same command when the queue drains:"
    printf '    %s\n' "${DEFERRED[@]}"
else
    echo "everything gateable is submitted."
fi
echo
echo "watch:   tail -F $REWARD_ROOT/logs/slurm-hg_verdict-<jid>.out"
echo "results: $MG/<arm>/holdout_${GA_BOX}/verdict_vs_self.json"
echo "         each carries region_mass_function, shape_score, radial_distribution"
echo "table:   python scripts/features/loss_ablation_table.py --reward-root $REWARD_ROOT \\"
echo "           --box $GA_BOX --arms '<labels>' --out $MG/loss_ablation_<name>_${GA_BOX}"
