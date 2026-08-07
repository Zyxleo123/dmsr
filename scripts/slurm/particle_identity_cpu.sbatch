#!/bin/bash
# Stage B of the SR2-vs-HR particle identity study (CPU only, no model, no GPU).
#
# Reads the two owner arrays and the two displacement fields and answers, at
# three granularities, whether an SR2 halo is made of the same particles as its
# HR counterpart. Analysis and plotting are separate stages so the figures can
# be redrawn from metrics.npz without re-reading the catalogs.
#
#   sbatch scripts/slurm/particle_identity_cpu.sbatch
#   sbatch scripts/slurm/particle_identity_cpu.sbatch PID_A=base:0 PID_B=base:1
#   sbatch scripts/slurm/particle_identity_cpu.sbatch PID_STAGE=plot   # redraw
#
#SBATCH --job-name=pid_analyse
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-pid_analyse-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

# PID_-prefixed on purpose: the login environment already exports generic names
# (SCRATCH, OUT, STAGE), and a ${STAGE:-analyse} would silently inherit one.
: "${PID_BOX:=set8}"
: "${PID_A:=hr}"
: "${PID_B:=base:0}"
: "${PID_CLASSES:=hosts,subhalos}"
: "${PID_MIN_PARTICLES:=50}"
: "${PID_MAX_PAIRS:=20000}"
: "${PID_CHUNKS:=8}"
: "${PID_STAGE:=both}"
: "${PID_OUT:=$DMSR_REWARD_ROOT/particle_identity/${PID_BOX}__${PID_A//:/}__${PID_B//:/}}"

owner_of() {   # $1 = side spec (hr | base[:seed])
    local src="${1%%:*}" seed="${1#*:}" tag
    [ "$seed" = "$1" ] && seed=0
    if [ "$src" = "hr" ] || [ "$seed" = "0" ]; then tag="$src"; else tag="${src}_seed${seed}"; fi
    echo "$DMSR_REWARD_ROOT/halos_particles/${PID_BOX}__${src}__${tag}/${PID_BOX}_${tag}_owner.npy"
}

# Gate inside the job, not in the submitter: a missing owner array means Stage A
# has not run, and require_input exits 0 so a dependent plotting job reports the
# same gate rather than sitting in DependencyNeverSatisfied.
if [ "$PID_STAGE" != "plot" ]; then
    require_input "$(owner_of "$PID_A")" \
        "scripts/slurm/particle_identity_prep_cpu.sbatch (side A)"
    require_input "$(owner_of "$PID_B")" \
        "scripts/slurm/particle_identity_prep_cpu.sbatch (side B)"
else
    require_input "$PID_OUT/metrics.npz" \
        "the analyse stage of this same job script"
fi

echo "=== particle identity: $PID_BOX  A=$PID_A  B=$PID_B  stage=$PID_STAGE"
echo "    out=$PID_OUT"
mkdir -p "$PID_OUT"

python -u scripts/sr2/particle_identity.py \
    --box "$PID_BOX" --a "$PID_A" --b "$PID_B" \
    --reward-root "$DMSR_REWARD_ROOT" \
    --out "$PID_OUT" \
    --classes "$PID_CLASSES" \
    --min-particles "$PID_MIN_PARTICLES" \
    --max-pairs "$PID_MAX_PAIRS" \
    --chunks "$PID_CHUNKS" \
    --stage "$PID_STAGE"

echo "=== done"
ls -la "$PID_OUT" "$PID_OUT/figures" 2>/dev/null || true
[ -r "$PID_OUT/summary.json" ] && python -c "
import json,sys
s=json.load(open('$PID_OUT/summary.json'))
for k in ('hosts','subhalos'):
    if k in s: print(k, json.dumps(s[k], indent=2))
"
