#!/bin/bash
# Density power spectrum P(k)/P_HR(k) for the member-gather arms whose fields
# are on disk (frozen, smallhalo, selfhostguard) vs HR. Displaces the particles,
# CIC-deposits to a grid_mult-finer overdensity mesh, and takes P(k) -- the
# physical density power, NOT the trainer's displacement-component highk_ratio.
#
# CPU and submitted: each field is a 3.2 GB load + a 1024^3 CIC scatter + FFT;
# real work the login node kills. cic_density is device-agnostic torch and
# power_spectrum is numpy, so no GPU is needed -- host RAM is the ask (~40 GB
# peak at grid_mult 2).
#
#   sbatch scripts/slurm/density_power_cpu.sbatch <envfile>
#SBATCH --job-name=dpower
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-dpower-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${DP_FIELDS:?DP_FIELDS must be the space-separated label=path list (incl HR)}"
: "${DP_GRID_MULT:=2}"
: "${DP_OUT:?DP_OUT must be the output json}"

python -u scripts/features/density_power_spectrum.py \
    --fields $DP_FIELDS \
    --grid-mult "$DP_GRID_MULT" \
    --out "$DP_OUT"

echo "=== density P(k) -> $DP_OUT"
