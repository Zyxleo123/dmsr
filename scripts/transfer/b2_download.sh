#!/bin/bash
# Pull the rhea snapshot back from Backblaze B2 onto a new cluster (babel). The
# inverse of b2_upload.sh; see docs/babel_migration.md for the full procedure.
#
#   NEW_SCRATCH=/data/user_data/$USER NEW_HOME=$HOME \
#       bash scripts/transfer/b2_download.sh            # tiers 0 1 2 3
#   NEW_SCRATCH=... NEW_HOME=... bash scripts/transfer/b2_download.sh 0 1   # small parts only
#
# NEW_SCRATCH plays the role of rhea's ZFS=/zfsauton/scratch/yixiz (bulk data lands
# in $NEW_SCRATCH/DMSR/...); NEW_HOME plays /zfsauton2/home/yixiz (the code tree is
# $NEW_HOME/DMSR/...). Tier sizes: 0 ~1 GB, 1 ~4 GB, 2 ~110 GB, 3 ~100 GB.
# Download is `rclone copy`, so rerunning resumes; run it as a CPU batch job if the
# login node kills long processes, but check the compute nodes can reach B2 first.
set -euo pipefail

: "${NEW_SCRATCH:?set NEW_SCRATCH (the bulk-data root, rhea's /zfsauton/scratch/yixiz)}"
: "${NEW_HOME:?set NEW_HOME (the code root, rhea's /zfsauton2/home/yixiz)}"
RCLONE="${RCLONE:-rclone}"
SRC="${B2_DEST:-b2:proteina/dmsr}"
S="$NEW_SCRATCH/DMSR"
FLAGS=(--transfers 8 --checkers 16 --multi-thread-streams 8 --stats 120s
       --stats-one-line -v)
[ -n "${DRY:-}" ] && FLAGS+=(--dry-run)

say() { echo "[$(date '+%F %T')] $*"; }
down() { say "copy $SRC/$1 -> $2"; "$RCLONE" copy "$SRC/$1" "$2" "${FLAGS[@]}"; }

tier0() {
    down bundle "$S/_b2_bundle"
    down home/DMSR/cosmo_sr_project/runs "$NEW_HOME/DMSR/cosmo_sr_project/runs"
    down home/DMSR/cosmo_sr_project/docs/figures "$NEW_HOME/DMSR/cosmo_sr_project/docs/figures"
}
tier1() {
    for d in ic_determinism halos logs env; do
        down "scratch/DMSR/dmsr_reward/$d" "$S/dmsr_reward/$d"
    done
    down scratch/DMSR/lr_sims "$S/lr_sims"         # tier 3's catnorm*.npy land here too
    down scratch/DMSR/lr_sims_256 "$S/lr_sims_256"
}
tier2() { down scratch/DMSR/int_redshift_same_cosmology "$S/int_redshift_same_cosmology"; }
tier3() {
    down scratch/DMSR/paired_catnorm "$S/paired_catnorm"
    down scratch/DMSR/dmsr_reward/cache/sr2_base "$S/dmsr_reward/cache/sr2_base"
}

"$RCLONE" cat "$SRC/MANIFEST.txt" 2>/dev/null \
    || echo "(no MANIFEST.txt: the rhea upload never reached its verify step)"
[ $# -gt 0 ] || set -- 0 1 2 3
for t in "$@"; do
    case "$t" in
        0|1|2|3) "tier$t" ;;
        *) echo "unknown tier: $t" >&2; exit 2 ;;
    esac
done
say "done: $*"
