#!/usr/bin/env bash
# Test-time scaling for the pretrained SR2/SRS generator: the full staged run.
#
# Run this on a GPU node (it needs one; a single 512^3 candidate is ~500 CPU-core
# minutes). Every stage is gated: later stages are skipped automatically if the
# earlier gate failed, so launching the whole thing is safe and cheap when the
# answer is "SR2's noise carries no selection leverage".
#
#   bash scripts/run_srs_tts.sh                # full run, K = 16
#   K=32 STAGES="1" bash scripts/run_srs_tts.sh   # oracle audit only, K = 32
#
# Stage 1 cost: n_boxes * K full-box generations. On one A100, ~40 s per 512^3
# box, so 16 boxes x K=16 is roughly 3 h plus metric time.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-/zfsauton/scratch/yixiz/miniconda3/envs/pjm/bin/python}"
DATA="${DATA:-/zfsauton/scratch/yixiz/DMSR/paired_catnorm}"
MODEL="${MODEL:-$HERE/external/SRS-map2map/SRmodel/G_z0.pt}"
OUT="${OUT:-runs/tts}"
K="${K:-16}"
DEVICE="${DEVICE:-cuda}"
STAGES="${STAGES:-1 2 4 5 6}"

# Splits are by simulation box and never overlap. SR2 is pretrained elsewhere, so
# "train" here means: boxes used to fit the HR plausibility reference and the
# verifier; "val" fits the score normaliser; "test" is where every number is read.
TRAIN_BOXES="${TRAIN_BOXES:-set0 set1 set2 set3 set4 set5 set6 set7}"
VAL_BOXES="${VAL_BOXES:-set8 set9 set10 set11}"
TEST_BOXES="${TEST_BOXES:-set12 set13 set14 set15}"
ALL_BOXES="$TRAIN_BOXES $VAL_BOXES $TEST_BOXES"

SEEDS="$($PYTHON -c "print(' '.join(str(i) for i in range($K)))")"
KVALS="$($PYTHON -c "
k=1; out=[]
while k <= $K:
    out.append(str(k)); k *= 2
print(' '.join(out))")"

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

# 0 = gate passed, 1 = gate failed, 2 = no report (stage was skipped). The three
# are kept distinct so that running a single later stage on its own does not get
# mistaken for a failed gate.
gate_status() {  # $1 = report json, $2 = dotted path to a boolean
  $PYTHON - "$1" "$2" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except OSError:
    sys.exit(2)
for key in sys.argv[2].split('.'):
    d = d.get(key, {})
sys.exit(0 if d is True else 1)
EOF
}

# --------------------------------------------------------------------------- #
if has_stage 1; then
  echo "=== Stage 1: oracle best-of-K audit (K=$K) ==="
  $PYTHON scripts/eval_srs_tts.py --stage all \
    --lr "$DATA/lr" --hr "$DATA/hr" --model "$MODEL" \
    --boxes $ALL_BOXES --train-boxes $TRAIN_BOXES \
    --val-boxes $VAL_BOXES --test-boxes $TEST_BOXES \
    --seeds $SEEDS --k-values $KVALS \
    --device "$DEVICE" --equivariance --out "$OUT/oracle"
fi

gate_status "$OUT/oracle/oracle_report.json" "decision_gate.pass" && s1=0 || s1=$?
if [[ $s1 == 1 ]]; then
  echo
  echo ">>> Stage-1 gate FAILED: best-of-$K does not improve any primary density /"
  echo ">>> higher-order metric by >=5% with a CI excluding zero."
  echo ">>> Per the plan, stop here: SR2's noise carries no useful selection leverage."
  echo ">>> See $OUT/oracle/oracle_report.json and oracle_scaling.png."
  exit 0
elif [[ $s1 == 2 ]]; then
  echo ">>> no Stage-1 report at $OUT/oracle; continuing unchecked (stage was skipped)"
fi

if has_stage 2; then
  echo "=== Stage 2: test-time selector ==="
  $PYTHON scripts/train_srs_verifier.py \
    --rows "$OUT/oracle/rows.jsonl" \
    --train-boxes $TRAIN_BOXES --val-boxes $VAL_BOXES --test-boxes $TEST_BOXES \
    --k-values $KVALS --k-gate "$K" --hidden \
    --out "$OUT/verifier"
fi

gate_status "$OUT/verifier/verifier_report.json" "gate.pass" && s2=0 || s2=$?
if [[ $s2 == 1 ]]; then
  echo
  echo ">>> Stage-2 gate FAILED: no test-time-available selector beats random with"
  echo ">>> a CI excluding zero while recovering >=50% of the oracle gain."
  echo ">>> Stop before refinement. See $OUT/verifier/verifier_report.json."
  exit 0
elif [[ $s2 == 2 ]]; then
  echo ">>> no Stage-2 report at $OUT/verifier; continuing unchecked (stage was skipped)"
fi

if has_stage 4; then
  echo "=== Stage 4: best-of-K plus noise refinement ==="
  $PYTHON scripts/tts_stage45.py --mode refine \
    --lr "$DATA/lr" --hr "$DATA/hr" --model "$MODEL" \
    --verifier "$OUT/verifier" --hr-reference "$OUT/oracle/hr_reference.npz" \
    --boxes $TEST_BOXES --k "$K" --keep 4 --device "$DEVICE" --out "$OUT/stage4"
  echo "--- gradient-free control (CEM) ---"
  $PYTHON scripts/tts_stage45.py --mode refine --cem \
    --lr "$DATA/lr" --hr "$DATA/hr" --model "$MODEL" \
    --verifier "$OUT/verifier" --hr-reference "$OUT/oracle/hr_reference.npz" \
    --boxes $TEST_BOXES --k "$K" --keep 4 --device "$DEVICE" --out "$OUT/stage4_cem"
fi

if has_stage 5; then
  echo "=== Stage 5: globally coherent tiled inference ==="
  $PYTHON scripts/tts_stage45.py --mode global \
    --lr "$DATA/lr" --hr "$DATA/hr" --model "$MODEL" \
    --verifier "$OUT/verifier" --hr-reference "$OUT/oracle/hr_reference.npz" \
    --boxes $TEST_BOXES --k "$K" --chunk 16 --stride 8 \
    --device "$DEVICE" --out "$OUT/stage5"
fi

if has_stage 6; then
  echo "=== Final comparison table ==="
  $PYTHON scripts/tts_final_table.py \
    --rows "$OUT/oracle/rows.jsonl" --verifier "$OUT/verifier" \
    --refine-rows "$OUT/stage4/rows.jsonl" --global-rows "$OUT/stage5/rows.jsonl" \
    --profiles "$OUT/oracle/profiles.npz" --box-summary "$OUT/oracle/box_summary.json" \
    --val-boxes $VAL_BOXES --test-boxes $TEST_BOXES \
    --k-values $KVALS --k-main "$K" --out "$OUT/final"
fi

echo "done -> $OUT"
