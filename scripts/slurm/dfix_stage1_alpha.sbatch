#!/bin/bash
# Stage 1 -- residual-destruction alpha sweep (CPU: reads cached fields only).
#
# Dials the model's correction in as `trilinear + alpha * (model - trilinear)`
# and asks where density and displacement each peak. If density is best at
# alpha < 1 while displacement power is best at alpha ~ 1, the correction is
# carrying real high-k power that is nonetheless destroying collapse -- Branch B
# rather than Branch A.
#
# `--components long trans` splits the correction with a periodic Helmholtz
# decomposition (`dPsi_long = k_hat (k_hat . dPsi)`), which separates the part
# that moves mass along the flow from the part that shears it across. Note the
# split zeroes the direction-ambiguous Nyquist wavevector component: `fftfreq`
# reports Nyquist as -n/2, whose mirror is itself, so without that fix `irfftn`
# silently discards ~7% of the longitudinal energy. After the fix, energy
# closure is 1 - 2e-7 and orthogonality -6e-10.
#
# This stage needs no GPU and no model: it is arithmetic on the .npy fields
# Stage 0 cached, which is also what makes every figure below redrawable
# without regenerating a 512^3 box.
#
#   sbatch scripts/slurm/dfix_stage1_alpha.sbatch
#   sbatch --export=ALPHA_LABEL=vc32 scripts/slurm/dfix_stage1_alpha.sbatch
#
#SBATCH --job-name=dfix_s1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-dfix_s1-%j.out

set -euo pipefail

# Slurm copies this script to /var/spool/slurm/job<id>/slurm_script before
# running it, so ${BASH_SOURCE[0]} does NOT point into the repo -- resolving
# the preamble that way silently sources nothing (job 22242). Use the submit
# directory, falling back to the checkout's own absolute path.
_SLURMDIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_SLURMDIR/_dfix_common.sh" ]] \
    || _SLURMDIR=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_SLURMDIR/_dfix_common.sh" "$@"

: "${ALPHAS:=0 0.1 0.25 0.5 0.75 1.0 1.25}"
: "${COMPONENTS:=full long trans}"
: "${ALPHA_LABELS:=base vc32}"
ALPHAS="${ALPHAS//,/ }"
COMPONENTS="${COMPONENTS//,/ }"
ALPHA_LABELS="${ALPHA_LABELS//,/ }"

FIELDS="$OUT/fields_$BOX"
require_input "$FIELDS" "dfix_stage0_ceiling.sbatch (--save-fields)"

for label in $ALPHA_LABELS; do
  if [ ! -f "$FIELDS/$label.npy" ]; then
    echo ">>> no cached field for arm '$label' at $FIELDS/$label.npy -- skipping this arm."
    continue
  fi
  echo "--- alpha sweep: $label"
  python -u scripts/dmsr_residual_alpha.py \
      --fields "$FIELDS" --label "$label" \
      --alphas $ALPHAS --components $COMPONENTS \
      --out "$OUT/stage1_alpha_$label"
done

echo "=== stage 1 done: $OUT/stage1_alpha_*"
