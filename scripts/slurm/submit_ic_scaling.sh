#!/bin/bash
# Submit the IC-determinism scaling readout (CPU, seconds): recovery vs object size
# in particles of the INPUT run, for the 256^3 run, the 128^3 run and frozen SR2
# (whose input is the 64^3 LR box). Reads the match .objects.npz files; set AFTER to
# a match job id to wait for it.
#
# Only calls sbatch; configuration in ONE env file passed positionally.
#
#   AFTER=<match job id> bash scripts/slurm/submit_ic_scaling.sh
#   DRY=1 bash scripts/slurm/submit_ic_scaling.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
R="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
BOX="${BOX:-set9}"
AFTER="${AFTER:-}"
ICD="${ICD_ROOT:-$R/ic_determinism}"
RUNS="dmo256:$ICD/${BOX}_match.objects.npz:dmo256:256"
RUNS+=" dmo128:$ICD/${BOX}_dmo128_match.objects.npz:dmo128:128"
RUNS+=" sr2:$ICD/${BOX}_match.objects.npz:sr2:64"

mkdir -p "$R/logs" "$R/env"
STAMP=$(date +%Y%m%d_%H%M%S)_$$
ENVFILE="$R/env/icdscale_${STAMP}.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_ic_scaling.sh at $(date '+%F %T')
PROJECT="$PROJECT"
ZFS="$ZFS"
DMSR_REWARD_ROOT="$R"
BOX="$BOX"
ICD_ROOT="$ICD"
ICD_SCALING_RUNS="$RUNS"
EOT
echo "envfile: $ENVFILE"
cat "$ENVFILE"
echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
run() {
    echo "+ $*" >&2
    if [ "$DRY" = "1" ]; then echo "DRY_$RANDOM"; else "$@" | awk '{print $NF}'; fi
}
DEP=()
[ -n "$AFTER" ] && DEP=(--dependency=afterany:"$AFTER")
SC=$(run sbatch "${DEP[@]}" scripts/slurm/icdet_scaling_cpu.sbatch "$ENVFILE")
echo "C  icd_scale  job $SC  ${AFTER:+(afterany:$AFTER)}"
echo
echo "monitor:"
echo "  tail -F $R/logs/slurm-icd_scale-$SC.out"
echo
echo "results:"
echo "  $ICD/${BOX}_scaling.json   recovery per object size in the input's own particles"
echo "  $ICD/${BOX}_scaling.png"
