#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
CONFIG="${1:-configs/ambient_smoke.yaml}"
shift || true
python -m cosmo_sr.train.train_ambient --config "$CONFIG" --smoke "$@"
