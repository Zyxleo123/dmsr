#!/bin/bash
# Validate a map2map configuration AND measure what one dataloader read costs.
#
#   sbatch scripts/slurm/m2m_preflight_cpu.sbatch <envfile>
#
# TWO JOBS IN ONE, both cheap:
#
# 1. Catch configuration errors on a CPU node instead of in the GPU queue. Two
#    submissions of the baseline died before a single training step -- a JSON
#    default truncated by bash brace parsing, then `betas must be either both
#    floats or both Tensors` -- and a third was waiting in `np.float(style)`,
#    removed in numpy 2.0, on a line that only runs once the adversary starts.
#    Each cost a queue round-trip to discover.
#
# 2. MEASURE the peak RSS of a single sample. map2map's loader reads each file
#    WHOLE on first touch (`fields.py`: "use memmap after reading a file once"),
#    and an HR box is 3 GB per field, 6 GB for disp+vel. With N loader workers
#    each doing its own first read, the peak is ~6N GB. The training job's --mem
#    has to cover that, and this is the number rather than a guess -- the
#    alternative is how `mgtrain` ended up asking 9x what it used and stalling
#    the queue behind a figure nobody had checked.
#
# CPU, not GPU: the model is constructed on CPU and nothing here trains.
#SBATCH --job-name=m2mpre
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-m2mpre-%j.out

# `set -euo pipefail` HERE as well as in the preamble: a failed source otherwise
# leaves the job on the system python and EXITING 0.
set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_m2m_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_m2m_common.sh" "$@"
set -euo pipefail

if [[ ! -r "$M2M_LAYOUT/manifest.json" ]]; then
    echo ">>> MISSING INPUT: $M2M_LAYOUT/manifest.json"
    echo ">>> produced by: sbatch scripts/slurm/m2m_split_data_cpu.sbatch"
    echo ">>> exiting 0 so dependents report the same rather than stranding."
    exit 0
fi

ARGS_FILE="$M2M_OUT/m2m_args.txt"
m2m_write_args "$ARGS_FILE"
echo "=== argument list under test ($(wc -l < "$ARGS_FILE") lines): $ARGS_FILE"

echo
echo "=== preflight, under /usr/bin/time -v for the peak RSS"
set +e
/usr/bin/time -v python scripts/features/m2m_preflight.py \
    --repo "$M2M_REPO" --args-file "$ARGS_FILE" 2>&1 | tee "$M2M_OUT/preflight.log"
_rc=${PIPESTATUS[0]}
set -e

echo
PEAK_KB=$(grep -oP 'Maximum resident set size \(kbytes\): \K[0-9]+' "$M2M_OUT/preflight.log" | tail -1)
if [[ -n "${PEAK_KB:-}" ]]; then
    PEAK_GB=$(awk -v k="$PEAK_KB" 'BEGIN{printf "%.1f", k/1048576}')
    echo "=== MEASURED peak RSS for ONE sample: ${PEAK_GB} GB"
    echo "    map2map reads each file WHOLE on first touch, so a training job"
    echo "    with W loader workers can peak near W x that. Size --mem from this:"
    for w in 2 4 8; do
        awk -v g="$PEAK_GB" -v w="$w" \
            'BEGIN{printf "      --loader-workers %d -> ~%.0f GB\n", w, g*w}'
    done
    echo "    (an upper bound: workers only collide when they first-touch"
    echo "     different boxes at the same time)"
fi

echo
if (( _rc != 0 )); then
    echo ">>> PREFLIGHT FAILED. This is a CONFIGURATION error, not a missing"
    echo ">>> input, so this exits non-zero: a 0 here would read as success."
    exit 1
fi
echo "=== preflight passed; $M2M_OUT/preflight.log has the detail"
exit 0
