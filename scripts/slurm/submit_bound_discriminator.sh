#!/bin/bash
# Name the term the gather objective is missing -- one CPU job, no halo finder.
#
# docs/sr2_gather_finetune.md section 5 measured 0/43 supervised subhalos bound
# for the tuned field; the ceiling run measured 42/43 for the true HR tiles in
# the identical splice geometry. Both fields are on disk. Whatever statistic
# separates them is the term the loss never constrained -- section 7.4 guesses
# the virial ratio, section 6 guesses the position-velocity correlation, and
# this settles it instead of guessing.
#
# Only calls sbatch. Config travels in ONE env file passed as a POSITIONAL
# argument -- never `sbatch --export`.
#
#   bash scripts/slurm/submit_bound_discriminator.sh
#   BD_TAGS=<a,b,c> bash scripts/slurm/submit_bound_discriminator.sh
#   DRY=1 bash scripts/slurm/submit_bound_discriminator.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

BD_BOX="${BD_BOX:-set8}"
BD_HOST_ID="${BD_HOST_ID:-271800}"
BD_SEED="${BD_SEED:-0}"
BD_SOFTENING="${BD_SOFTENING:-0.01}"
BD_RUN_DIR="${BD_RUN_DIR:-$REWARD_ROOT/host_gather/set8_h271800_fine_anchored}"
_RUN="$(basename "$BD_RUN_DIR")"
# Default set: the verified positive (HR tiles), the tuned field it is being
# read against, the frozen re-splice that fixes the harness noise floor, and the
# preserve run -- which shares this run's 43 supervised targets exactly, so its
# field is scored on the same sets. Its Rockstar gate is still running, so this
# is also a prediction of that result.
BD_TAGS="${BD_TAGS:-gather_${_RUN}_hr,gather_${_RUN},gather_${_RUN}_frozen,gather_set8_h271800_fine_preserve}"
BD_REF_TAG="${BD_REF_TAG:-gather_${_RUN}_hr}"

if [ ! -f "$BD_RUN_DIR/subhalos.json" ]; then
    echo "no subhalos.json under $BD_RUN_DIR -- point BD_RUN_DIR at a finished run" >&2
    exit 1
fi

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$REWARD_ROOT/bound_discriminator"
ENVFILE="$REWARD_ROOT/env/bound_disc_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_bound_discriminator.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
BD_BOX=$BD_BOX
BD_RUN_DIR=$BD_RUN_DIR
BD_HOST_ID=$BD_HOST_ID
BD_SEED=$BD_SEED
BD_TAGS=$BD_TAGS
BD_REF_TAG=$BD_REF_TAG
BD_SOFTENING=$BD_SOFTENING
EOT

echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch scripts/slurm/bound_discriminator_cpu.sbatch $ENVFILE"
    exit 0
fi

JID=$(sbatch --parsable scripts/slurm/bound_discriminator_cpu.sbatch "$ENVFILE")
echo "submitted bound discriminator -> job $JID (CPU)"
echo
echo "--- watch ---"
echo "tail -F $REWARD_ROOT/logs/slurm-bound_disc-$JID.out"
echo
echo "--- results ---"
echo "$REWARD_ROOT/bound_discriminator/${BD_BOX}__${_RUN}.json"
echo "the discrimination table and the VERDICT line are printed in the .out;"
echo "controls (r_rms, sigma_v) separating as strongly as the candidates means"
echo "the fields differ generically and nothing is attributable."
