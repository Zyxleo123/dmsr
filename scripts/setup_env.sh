#!/usr/bin/env bash
# Set up the cosmo_sr environment.
#
# Uses the existing conda env `pjm` which already has torch, numpy, scipy,
# matplotlib, pyyaml and bigfile. The two external repos (map2map, SRS-map2map)
# are referenced under external/ (symlinks to the local clones). We install
# map2map and our own package in editable mode.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

ENV_NAME="${CONDA_ENV:-pjm}"
echo "Activating conda env: ${ENV_NAME}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "Pinned external commits:"
cat external/COMMITS.txt || true

echo "Installing map2map (editable) ..."
pip install -e external/map2map

echo "Installing cosmo_sr (editable) ..."
pip install -e .

echo "Installing test deps ..."
pip install pytest

echo "Sanity import check ..."
python -c "import map2map; import cosmo_sr; from map2map import models; print('map2map.models.UNet =', models.UNet); print('cosmo_sr', cosmo_sr.__version__)"

echo "Done."
