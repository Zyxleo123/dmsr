#!/bin/bash
# The catalog-free kNN arm: density strata INSTEAD of the HR owner-class mask.
# A thin wrapper that sets the MG_KNN_* variables and hands off to
# submit_member_gather_train.sh -- it runs no work of its own.
#
# GATED, and the gate is not advisory. --knn-classes sub_50_200 is the only
# Rockstar label left in this term's gradient, and replacing it with a density
# cut is only sound if that cut BOTH enriches the deficit population AND ranks
# tiles the way the class loss does. Both are measured by
# submit_knn_strata.sh; this script refuses to launch without its passing
# verdict, because the alternative is spending ~10 GPU-hours to discover the
# synthetic result (the dense stratum is 65-83% HOST particles) on real data.
#
#   bash scripts/slurm/submit_knn_strata.sh          # 1. the gate (CPU)
#   KC_VERDICT=<json> bash scripts/slurm/submit_knn_strata_arm.sh   # 2. the arm
#
#   FORCE=1  launch anyway. Only with a reason to disbelieve the gate, and the
#            run is then a test OF the gate, not evidence for the arm.
#   DRY=1    print what would be submitted.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
FORCE="${FORCE:-0}"
KC_VERDICT="${KC_VERDICT:-$REWARD_ROOT/member_gather/knn_cdf/knn_strata_set9_phase.json}"
PY="${PY:-/zfsauton/scratch/yixiz/miniconda3/envs/pjm/bin/python}"

if [ ! -r "$KC_VERDICT" ]; then
    echo "no verdict json: $KC_VERDICT"
    echo "run the gate first:  bash scripts/slurm/submit_knn_strata.sh"
    [ "$FORCE" = "1" ] || exit 1
fi

# The gate chooses the arm's only real hyper-parameters, so they are READ from
# the verdict rather than typed here -- a hand-typed strata count that the gate
# never passed would be an unmeasured arm wearing a measured arm's name.
if [ -r "$KC_VERDICT" ]; then
    read -r PASS BEST_STRATA BEST_CHARGE BEST_RHO BEST_ENR <<< "$(
        "$PY" - "$KC_VERDICT" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
v = d["verdict"]
gate = d["per_field"][v["gate_label"]]
ok = v["passing_configs"]
if not ok:
    print("false 0 0 nan nan")
else:
    # Among passing configs, the one whose per-tile ranking agrees best with the
    # class loss: that agreement IS the claim the arm is making.
    best = max(ok, key=lambda k: gate[k]["rho_dense_vs_class"])
    g = gate[best]
    print("true", g["strata"], ",".join(str(j) for j in g["charge"]),
          round(g["rho_dense_vs_class"], 3),
          round(g["median_recall_enrichment"], 2))
PYEOF
)"
else
    PASS=false; BEST_STRATA=0; BEST_CHARGE=""; BEST_RHO=nan; BEST_ENR=nan
fi

echo "verdict:  $KC_VERDICT"
echo "  pass=$PASS  strata=$BEST_STRATA  charge=$BEST_CHARGE  rho=$BEST_RHO  recall_enrichment=$BEST_ENR"
if [ "$PASS" != "true" ]; then
    echo
    echo "GATE FAILED: no (strata, charge) both enriches sub_50_200 and ranks"
    echo "tiles like the class loss. A density stratum is not a catalog-free"
    echo "substitute for --knn-classes on this box. That is a RESULT -- write it"
    echo "down -- not a reason to retune the gate."
    [ "$FORCE" = "1" ] || exit 1
    echo "FORCE=1: launching anyway. This run tests the gate, not the arm."
    BEST_STRATA="${MG_KNN_STRATA:-8}"; BEST_CHARGE="${MG_KNN_STRATA_CHARGE:-0}"
fi

# Round 5's pass was measured in 6D; position-only ranked the arms at -0.23.
export MG_KNN_PHASE_LAMBDA="${MG_KNN_PHASE_LAMBDA:-4000}"
export MG_W_KNN_UNSUP="${MG_W_KNN_UNSUP:-1.0}"
export MG_KNN_KS="${MG_KNN_KS:-8 32 64 128}"
export MG_KNN_STRATA="${MG_KNN_STRATA:-$BEST_STRATA}"
export MG_KNN_STRATA_RANK="${MG_KNN_STRATA_RANK:-0}"
export MG_KNN_STRATA_CHARGE="${MG_KNN_STRATA_CHARGE:-$BEST_CHARGE}"
# The point of the arm: the label is MEASURED and REPORTED, never trained on.
export MG_KNN_CLASSES="${MG_KNN_CLASSES:-sub_50_200}"
export MG_KNN_CLASSES_DIAG_ONLY=1
# The kNN term needs the unsupervised pool, and w_centre stays on: this term is
# translation-invariant and cannot place an object (knn_cdf, "What this cannot
# do"). The 2026-08-21 free-field run without a centre term landed a median
# 0.414 Mpc/h off against a 0.150 search radius.
export MG_UNSUP_TILES_PER_BOX="${MG_UNSUP_TILES_PER_BOX:-8}"
export MG_LABEL="${MG_LABEL:-_knnstrata}"

echo
echo "arm: w_knn=$MG_W_KNN_UNSUP strata=$MG_KNN_STRATA charge=$MG_KNN_STRATA_CHARGE"
echo "     phase_lambda=$MG_KNN_PHASE_LAMBDA classes=$MG_KNN_CLASSES (DIAGNOSTIC ONLY)"
echo "     label=$MG_LABEL"
echo
exec bash scripts/slurm/submit_member_gather_train.sh
