#!/bin/bash
# WHERE is SR2's halo-count deficit -- in small halos, or in all of them?
#
# Reads the SAVED ASCII catalogs (no field read, no halo finder), writes every
# count plus a verdict, then renders the figures. CPU and submitted because
# parsing the catalogs is real work the login node kills; it is minutes.
#
# Node choice, checked with `myfree` at write time: this asks for no GPU, so it
# goes to the `cpu` partition where it will not queue behind training.
#
#   sbatch scripts/slurm/halo_gap_census_cpu.sbatch <envfile>
#SBATCH --job-name=halo_gap_census
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-halogap-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${HG_BOX:=set9}"
: "${HG_KINDS:=base,hr}"
: "${HG_ARM_TAGS:=}"
: "${HG_RUN_DIR:=}"          # an arm's holdout export dir -> the edited region
: "${HG_DILATE:=2.0}"
: "${HG_HOST_FLOOR:=200}"
: "${HG_OUT_DIR:?HG_OUT_DIR must be the output directory}"

RUN_DIR_ARG=()
[[ -n "$HG_RUN_DIR" ]] && RUN_DIR_ARG=(--run-dir "$HG_RUN_DIR")
ARM_TAGS_ARG=()
[[ -n "$HG_ARM_TAGS" ]] && ARM_TAGS_ARG=(--arm-tags "$HG_ARM_TAGS")

python -u scripts/features/halo_gap_census.py \
    --reward-root "$DMSR_REWARD_ROOT" \
    --box "$HG_BOX" \
    --kinds "$HG_KINDS" \
    "${ARM_TAGS_ARG[@]}" \
    "${RUN_DIR_ARG[@]}" \
    --dilate "$HG_DILATE" \
    --host-floor "$HG_HOST_FLOOR" \
    --out-dir "$HG_OUT_DIR"

python -u scripts/features/halo_gap_census_plot.py --out-dir "$HG_OUT_DIR"

echo
echo "=== VERDICT ==="
python -u - "$HG_OUT_DIR/census.json" <<'PY'
import json, sys
c = json.loads(open(sys.argv[1]).read())
for scope, s in c["scopes"].items():
    print(f"[{scope}] {s['verdict']['text']}")
PY

echo
echo "=== data + figures -> $HG_OUT_DIR"
ls -la "$HG_OUT_DIR"
