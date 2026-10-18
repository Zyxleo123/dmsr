#!/bin/bash
# Run a pytest selection on a CPU node (the login node kills real compute).
#
#   sbatch scripts/slurm/pytest_cpu.sbatch [envfile|VAR=value ...]
#   e.g. PT_TARGETS="tests/features/test_host_profile.py" via a VAR=value arg
#SBATCH --job-name=pytest
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-pytest-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 16G"
: "${PT_TARGETS:=tests}"
# shellcheck disable=SC2086
python -m pytest -q -x $PT_TARGETS
