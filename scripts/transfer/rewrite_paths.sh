#!/bin/bash
# Rewrite rhea's two absolute roots in every tracked file to the new cluster's.
# ~300 sbatch scripts hard-code them in #SBATCH --output lines and preamble
# fallbacks, which no environment variable can override. Run on a fresh branch:
#
#   git switch -c babel
#   NEW_HOME=$HOME NEW_SCRATCH=/data/user_data/$USER bash scripts/transfer/rewrite_paths.sh
#   git diff --stat && git commit -am "Point paths at babel"
#
# The conda root moves with the scratch root ($NEW_SCRATCH/miniconda3); install
# conda there or edit _reward_common.sh by hand. Partition/QOS/GPU names in the
# #SBATCH lines are NOT rewritten: they are cluster policy, check them with sinfo.
set -euo pipefail
: "${NEW_HOME:?set NEW_HOME (replaces /zfsauton2/home/yixiz)}"
: "${NEW_SCRATCH:?set NEW_SCRATCH (replaces /zfsauton/scratch/yixiz)}"
cd "$(git rev-parse --show-toplevel)"
git grep -lz -e /zfsauton2/home/yixiz -e /zfsauton/scratch/yixiz -- . ':!scripts/transfer' \
    | xargs -0 sed -i -e "s|/zfsauton2/home/yixiz|$NEW_HOME|g" \
                      -e "s|/zfsauton/scratch/yixiz|$NEW_SCRATCH|g"
echo "remaining rhea paths: $(git grep -c -e /zfsauton -- . ':!scripts/transfer' | wc -l) files"
