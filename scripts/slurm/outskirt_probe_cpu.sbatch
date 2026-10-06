#!/bin/bash
# Why does Rockstar report the inner supervised sets and not the outskirt ones?
#
# Pure function of artefacts already on disk: the HR Rockstar catalog, each
# arm's holdout_<box>/subhalos.json, its flow_rockstar/compare/*.json and its
# pool.json. No field, no model, no GPU -- so the figures redraw without
# recomputation and the plot stage below is separate and free.
#
# Sizing: the HR ASCII catalog is ~160 MB and np.loadtxt holds it as ~315k x 53
# float64 (~130 MB) plus the parse temporaries. The per-arm join is a few
# thousand rows. 8G is ~3x the measured peak; do not raise it without measuring
# (mgtrain-mem-was-9x-overasked). The parsed columns are cached as an npz keyed
# on the catalog's mtime, so every arm after the first is seconds.
#
#   sbatch scripts/slurm/outskirt_probe_cpu.sbatch <envfile>
#SBATCH --job-name=outprobe
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-outprobe-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${OD_BOX:=set9}"
: "${OD_ARMS:=}"
# NOT `${OD_TAG_FORMAT:=mgho_{arm}_{box}}`: bash closes the
# expansion at the first `}` inside the default and the tag
# becomes `mgho_{arm_{box}}`.
[ -n "${OD_TAG_FORMAT:-}" ] || OD_TAG_FORMAT='mgho_{arm}_{box}'

if [ -z "${OD_ARMS// }" ]; then
    echo ">>> GATE: OD_ARMS is empty -- nothing to probe."
    echo ">>> The submitter fills it from the member_gather run dirs that have"
    echo ">>> a holdout_\$OD_BOX/subhalos.json. Exiting 0 so dependents report"
    echo ">>> a gate rather than DependencyNeverSatisfied."
    exit 0
fi

python -u scripts/features/outskirt_detection_probe.py \
    --reward-root "$DMSR_REWARD_ROOT" \
    --box "$OD_BOX" \
    --tag-format "$OD_TAG_FORMAT" \
    --arms $OD_ARMS

echo "=== results: $DMSR_REWARD_ROOT/outskirt_probe/$OD_BOX/"
echo "=== READ IN ORDER:"
echo "===   1. radial profile        -- is the gap real at fixed num_p?"
echo "===   2. address decomposition -- P(addressed) x P(hit|addressed)."
echo "===      Whichever carries the radial trend is the defect to fix:"
echo "===      placement and construction need different experiments."
echo "===   3. hit/miss contrast     -- which axis actually separates them."
echo "===   4. reference inflation   -- how far the loss's reachable target"
echo "===      sits from HR's own object, and whether the straggler tail"
echo "===      explains it (corr near 1 = yes)."
