#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
# Usage: run_eval.sh <config> <checkpoint> [--lr LR.npy] [--hr HR.npy] [--out DIR] ...
CONFIG="${1:-configs/ambient_smoke.yaml}"
CKPT="${2:-}"
shift || true
shift || true
if [[ -n "$CKPT" ]]; then
  python -m cosmo_sr.eval.run_eval --config "$CONFIG" --checkpoint "$CKPT" "$@"
else
  python -m cosmo_sr.eval.run_eval --config "$CONFIG" "$@"
fi
