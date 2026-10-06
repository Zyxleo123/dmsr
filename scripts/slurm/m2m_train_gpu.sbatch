#!/bin/bash
# Train with map2map's OWN trainer, styled_srsgan branch, unedited.
#
#   sbatch scripts/slurm/m2m_train_gpu.sbatch <envfile>
#
# This is the control the whole from-scratch line has been missing: THEIR
# trainer, THEIR generator and discriminator, THEIR WGAN-GP and lag2eul, THEIR
# hyperparameters, on OUR data. Four earlier runs used our reimplementations and
# failed for four different reasons, each traced to a component we had
# substituted rather than to the experiment. Nothing of ours enters here except
# the data layout and an identity norm, both through map2map's own interfaces.
#
# NODE. Their script asks for 4x A100-40 over 2 days. We run 1 GPU: their
# --batches 1 with 4 ranks is an effective batch of 4, and a single rank at
# --batches 1 is 4x fewer samples per step, so the same --epochs covers the same
# data in 4x the steps. Expect to resubmit; --load-state makes that a resume.
#
# GPU MEMORY is not the binding constraint (our own trainer MEASURED 4.2 GB for
# the same architecture at batch 4), but the 11 GB rtx_2080_ti are excluded
# because a CUDA OOM on this path can kill the process with NO Python traceback,
# which reads as a mysterious silent FAILED.
#SBATCH --job-name=m2mtrain
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu11,gpu12,gpu13,gpu14,gpu15,gpu16,gpu17,gpu18,gpu19
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-m2mtrain-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_m2m_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_m2m_common.sh" "$@"
set -euo pipefail

# ---- gates, as exit-0 skips ------------------------------------------------
# A non-zero exit would strand dependents in DependencyNeverSatisfied with no
# explanation, so a missing input says why and exits 0.
if [[ ! -r "$M2M_LAYOUT/manifest.json" ]]; then
    echo ">>> MISSING INPUT: $M2M_LAYOUT/manifest.json"
    echo ">>> produced by: sbatch scripts/slurm/m2m_split_data_cpu.sbatch"
    echo ">>> exiting 0 so dependents report the same rather than stranding."
    exit 0
fi
if [[ ! -r "$M2M_REPO/m2m.py" ]]; then
    echo ">>> MISSING INPUT: $M2M_REPO/m2m.py"
    echo ">>> produced by: git -C /zfsauton2/home/yixiz/DMSR/map2map worktree add \\"
    echo ">>>                  $M2M_REPO xw/styled_srsgan"
    exit 0
fi

# The argument list comes from m2m_write_args in the preamble, so the list the
# preflight validates is character-for-character the list that runs. Building it
# twice is how a preflight passes while the job still dies.
ARGS_FILE="$M2M_OUT/m2m_args.txt"
m2m_write_args "$ARGS_FILE"

# ---- preflight, as a GATE --------------------------------------------------
# Two submissions of this job died before a single training step, on a
# truncated JSON default and then on Adam's betas, and a third failure was
# waiting in numpy-2's removal of np.float on a line that only runs once the
# adversary starts. Each cost a queue round-trip. This runs their argparse,
# their FieldDataset and their model/optimizer construction first, in seconds.
#
# A failure here exits NON-ZERO, unlike the missing-input gates above: a missing
# upstream artifact is someone else's job not having run, and exits 0 so
# dependents report the same rather than stranding. A bad argument list is THIS
# job being misconfigured, and a 0 would read as success.
echo
echo "=== preflight ($(wc -l < "$ARGS_FILE") argument lines)"
# M2M_PREFLIGHT=0 runs THEIR argparse only. Our preflight's later stages are
# written against xw/styled_srsgan -- they pass style_pattern to FieldDataset
# and style_size to the model, neither of which exists in the colleague's tree.
# Rather than grow a two-tree abstraction, check the thing that actually keeps
# biting (the argument list) with their own parser, and let the job surface the
# rest in its first seconds.
if [[ "${M2M_PREFLIGHT:-1}" == "0" ]]; then
    python - "$M2M_REPO" "$ARGS_FILE" <<'PYEOF' || { echo ">>> their argparse REJECTED the argument list."; exit 1; }
import sys
repo, argfile = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
argv = [l.strip() for l in open(argfile) if l.strip() and not l.startswith('#')]
sys.argv = ["m2m.py", "train"] + argv
from map2map.args import get_args
a = get_args()
print(f"    ok: their argparse accepted {len(argv)} args")
print(f"    model={a.model} adv={a.adv_model} crop={a.crop} pad={a.pad} "
      f"batches={a.batches} lag2eul={getattr(a,'lag2eul',None)} "
      f"mesh_up_fac={getattr(a,'mesh_up_fac',None)}")
PYEOF
elif ! python scripts/features/m2m_preflight.py \
        --repo "$M2M_REPO" --args-file "$ARGS_FILE"; then
    echo
    echo ">>> PREFLIGHT FAILED -- not starting training."
    echo ">>> The GPU is released now instead of after a wasted allocation."
    exit 1
fi

mapfile -t _ARGS < "$ARGS_FILE"

cd "$M2M_OUT"     # m2m.py writes checkpoint.pt and its logs into the CWD

echo
echo "=== python $M2M_REPO/m2m.py train \\"
printf '        %s\n' "${_ARGS[@]}"
echo

srun --unbuffered python "$M2M_REPO/m2m.py" train "${_ARGS[@]}"
_rc=$?

echo
echo "=== $(date '+%F %T')  finished rc=$_rc"
echo "    checkpoint: $M2M_OUT/checkpoint.pt   (resubmit this same job to resume)"
echo "    tensorboard: $M2M_OUT"
echo
echo "    The number that decides this run is NOT the loss. Score a checkpoint"
echo "    against frozen SR2 on the same held-out boxes:"
echo "      density P/P_HR (hi-k) -- frozen SR2 0.977, TRILINEAR 0.706."
echo "    Anything below 0.706 is worse than interpolation."
exit "$_rc"
