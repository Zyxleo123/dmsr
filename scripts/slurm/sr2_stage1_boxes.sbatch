#!/bin/bash
#SBATCH -J sr2_stage1_boxes
#SBATCH -p legacy
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH -o /zfsauton/scratch/yixiz/DMSR/sr2_baseline/logs/slurm-sr2_stage1_boxes-%j.out

set -euo pipefail
export PATH=/zfsauton/scratch/yixiz/miniconda3/envs/pjm/bin:$PATH
cd /zfsauton2/home/yixiz/DMSR/cosmo_sr_project
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Large artifacts on scratch only (home quota).
SCRATCH=/zfsauton/scratch/yixiz/DMSR/sr2_baseline
OUT="$SCRATCH/stage1"
mkdir -p "$SCRATCH/logs" "$OUT"

# Box-to-box variance: 4 seeds × set13–15. Do NOT chase set12 seeds 9–31.
python scripts/sr2/stage1_subhalo_diagnose.py \
  --boxes set13,set14,set15 \
  --seeds 0,1,2,3 \
  --density-probe \
  --out "$OUT"

# Rematch existing set12 catalogs with repaired matcher (idempotent).
python scripts/sr2/rematch_halos.py \
  --halos-root "$OUT/halos" \
  --boxes set12 \
  --seeds 0,1,2,3,4,5,6,7 \
  --out "$OUT/rematch_set12"

# Merge rematch set12 + new box match rows for Gate A
python - <<'PY'
import json
from pathlib import Path
out = Path("/zfsauton/scratch/yixiz/DMSR/sr2_baseline/stage1")
gate = out / "gate_a_input"
gate.mkdir(exist_ok=True)
rows = []
for p in [out / "rematch_set12" / "match_rows.jsonl", out / "match_rows.jsonl"]:
    if not p.exists():
        continue
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # Prefer rematch for set12; skip old set12 rows from stage1 appends
        if "rematch" not in str(p) and r.get("box") == "set12":
            continue
        rows.append(r)
seen = {}
for r in rows:
    seen[(r["box"], int(r["seed"]))] = r
with open(gate / "match_rows.jsonl", "w") as fh:
    for r in sorted(seen.values(), key=lambda x: (x["box"], int(x["seed"]))):
        fh.write(json.dumps(r) + "\n")
for name in ("halo_rows.jsonl", "field_rows.jsonl"):
    src = out / name
    dst = gate / name
    if src.exists():
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
print(f"gate input: {len(seen)} match rows")
PY

python scripts/sr2/stage1_analyze.py --stage1 "$OUT" --seed0-only \
  --out "$OUT/analyze"

python scripts/sr2/stage2_oracle_tts.py \
  --stage1 "$OUT/gate_a_input" \
  --n-max 8 \
  --out "$SCRATCH/stage2"

echo "=== Stage 1 boxes + Gate A finished ==="
cat "$SCRATCH/stage2/gate_a.json"
