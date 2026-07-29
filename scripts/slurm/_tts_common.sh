# Shared preamble for every TTS SLURM job. Sourced, never executed directly.
#
# Every knob is an environment variable with a default, so a job is configured
# by its arguments alone. That keeps one copy of the defaults (here) instead of
# the same box lists drifting across six .sbatch files.
#
#   sbatch scripts/slurm/tts_stage1_oracle.sbatch K=32 OUT=runs/tts_k32
#   sbatch scripts/slurm/tts_stage1_oracle.sbatch /path/to/tts_env.sh
#
# CONFIGURATION COMES IN AS POSITIONAL ARGUMENTS, NOT `sbatch --export`.
# On this cluster any `--export=<list>` (with or without ALL) makes sbatch set
# SLURM_GET_USER_ENV=1, slurmd then tries to rebuild the login environment on
# the compute node, that lookup fails, and the job is requeued and held with
# `(user env retrieval failed requeued held)` -- it never starts. Arguments go
# through the job record untouched, so they are the safe channel. Each argument
# is either a `VAR=value` override or a file to source; submit_tts.sh writes one
# env file per submission and passes its path.
set -euo pipefail

for _arg in "$@"; do
    case "$_arg" in
        *=*)  export "$_arg" ;;
        *)    [[ -r "$_arg" ]] || { echo "config file not readable: $_arg" >&2; exit 1; }
              # shellcheck disable=SC1090
              source "$_arg" ;;
    esac
done
unset _arg

# Belt and braces: the job inherits a full environment, but these three are all
# conda and matplotlib really need, so the preamble also runs under `env -i`.
export PATH="${PATH:-/usr/local/bin:/usr/bin:/bin}"
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-$(getent passwd "$USER" | cut -d: -f6)}"

source /zfsauton/scratch/yixiz/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pjm}"

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"

: "${DATA:=/zfsauton/scratch/yixiz/DMSR/paired_catnorm}"
: "${MODEL:=$PROJECT/external/SRS-map2map/SRmodel/G_z0.pt}"
: "${OUT:=runs/tts}"
: "${K:=16}"
: "${DEVICE:=cuda}"

# Splits are by simulation box and must not overlap. SR2 is pretrained
# elsewhere, so "train" here means: boxes used for the HR plausibility reference
# and the verifier; "val" fits the score normaliser; "test" is where every
# reported number comes from.
: "${TRAIN_BOXES:=set0 set1 set2 set3 set4 set5 set6 set7}"
: "${VAL_BOXES:=set8 set9 set10 set11}"
: "${TEST_BOXES:=set12 set13 set14 set15}"
# `sbatch --export` splits its argument on commas, so a value containing spaces
# survives only by accident of quoting. submit_tts.sh therefore exports box
# lists comma-separated; translating back here means both forms work and no
# caller has to know which one is in play.
TRAIN_BOXES="${TRAIN_BOXES//,/ }"
VAL_BOXES="${VAL_BOXES//,/ }"
TEST_BOXES="${TEST_BOXES//,/ }"
ALL_BOXES="$TRAIN_BOXES $VAL_BOXES $TEST_BOXES"

SEEDS="$(python -c "print(' '.join(str(i) for i in range($K)))")"
KVALS="$(python -c "
k, out = 1, []
while k <= $K:
    out.append(str(k)); k *= 2
print(' '.join(out))")"

mkdir -p "$OUT"

echo "=== $(date '+%F %T')  job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}"
echo "    project=$PROJECT  out=$OUT  K=$K  device=$DEVICE"
echo "    test boxes: $TEST_BOXES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Gate helpers -------------------------------------------------------------- #
# 0 = gate passed, 1 = gate failed, 2 = report missing (upstream never ran).
# The three stay distinct so that resubmitting one stage on its own is not
# mistaken for a failed gate.
gate_status() {   # $1 = report json, $2 = dotted path to a boolean
  python - "$1" "$2" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except OSError:
    sys.exit(2)
for key in sys.argv[2].split('.'):
    d = d.get(key, {})
sys.exit(0 if d is True else 1)
EOF
}

# Exit the job *successfully* when an upstream gate failed: in a dependency
# chain a non-zero exit would leave every later job stuck in DependencyNeverSatisfied
# instead of letting them make the same skip decision and report it.
require_gate() {  # $1 = report json, $2 = dotted path, $3 = message
  local status=0
  gate_status "$1" "$2" || status=$?
  case "$status" in
    0) ;;
    1) echo ">>> GATE FAILED: $3"; echo ">>> see $1"; echo ">>> skipping this stage."; exit 0 ;;
    2) echo ">>> no report at $1 -- upstream stage did not run; continuing unchecked" ;;
  esac
}
