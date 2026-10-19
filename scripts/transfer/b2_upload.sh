#!/bin/bash
# Upload what the IC experiments need from the autonlab cluster (rhea) to Backblaze
# B2 before the shutdown, laid out so babel can pull it back path-for-path.
#
# Layout under $DEST (default b2:proteina/dmsr):
#   scratch/DMSR/...   mirrors $ZFS/DMSR/...           (ZFS=/zfsauton/scratch/yixiz)
#   home/DMSR/...      mirrors /zfsauton2/home/yixiz/DMSR/... (ignored files only;
#                      the tracked code is on GitHub)
#   bundle/...         tarballs + exports that have no path of their own
#   MANIFEST.txt       rclone size of every prefix, written by the `verify` tier
#
# Tiers, in priority order (small and irreplaceable first):
#   0  bundle: git bundle of this repo, local patches + untracked files of the sibling
#      clones (SRS-map2map, map2map, map2map_styled, map2map_colleague, MP-Gadget
#      Options.mk), external/rockstar + external/gadget4 sources, conda env exports
#      (pjm, g4), Claude memory; plus runs/ and docs/figures/ (git-ignored).
#   1  results: dmsr_reward/{ic_determinism,halos,logs,env}, lr_sims{,_256}
#      paramfiles + seed manifests + logs.
#   2  raw sims: int_redshift_same_cosmology dmo-64/128/256, then dmo-512 set9+set10,
#      then the rest of dmo-512 (~110 GB total).
#   3  derived, regenerable but slow: paired_catnorm (49 GB), cache/sr2_base (48 GB),
#      lr_sims catnorm*.npy.
#   verify  rclone check --one-way of every prefix + MANIFEST.txt
#
#   bash scripts/transfer/b2_upload.sh              # tiers 0 1 2 3 verify
#   bash scripts/transfer/b2_upload.sh 2            # one tier
#   DRY=1 bash scripts/transfer/b2_upload.sh 0 1    # rclone --dry-run
#
# Every step is `rclone copy`, which skips files already uploaded, so a killed run
# is resumed by running it again. Network I/O only: it runs on the login node.
set -euo pipefail

RCLONE="${RCLONE:-$HOME/bin/rclone}"
DEST="${B2_DEST:-b2:proteina/dmsr}"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
S="$ZFS/DMSR"
H="${HOME_DMSR:-/zfsauton2/home/yixiz/DMSR}"
PROJECT="$H/cosmo_sr_project"
STAGE="${STAGE:-$S/_b2_stage}"
CONDA="${CONDA:-/zfsauton/scratch/yixiz/miniconda3}"
MEMORY="$HOME/.claude/projects/-zfsauton2-home-yixiz-DMSR-cosmo-sr-project/memory"

FLAGS=(--transfers 8 --checkers 16 --b2-chunk-size 96M --b2-upload-concurrency 8
       --stats 120s --stats-one-line -v)
[ -n "${DRY:-}" ] && FLAGS+=(--dry-run)

say() { echo "[$(date '+%F %T')] $*"; }
up() {  # up <src> <dest-relative> [rclone filter args...]
    local src="$1" dst="$2"; shift 2
    say "copy $src -> $DEST/$dst $*"
    "$RCLONE" copy "$src" "$DEST/$dst" "${FLAGS[@]}" "$@"
}

tier0() {
    local b="$STAGE/bundle"
    mkdir -p "$b/repos"
    say "tier 0: staging bundle in $b"
    git -C "$PROJECT" bundle create "$b/cosmo_sr_project.bundle" --all
    for r in SRS-map2map map2map map2map_styled MP-Gadget; do
        local d="$H/$r"
        {
            echo "remote: $(git -C "$d" remote get-url origin)"
            echo "branch: $(git -C "$d" rev-parse --abbrev-ref HEAD)"
            echo "commit: $(git -C "$d" rev-parse HEAD)"
            git -C "$d" remote -v
        } > "$b/repos/$r.txt"
        git -C "$d" diff HEAD > "$b/repos/$r.patch"
        # untracked files, minus MP-Gadget's build output (depends/lib64)
        git -C "$d" ls-files --others --exclude-standard | grep -v '^depends/' \
            > "$b/repos/$r.untracked" || true
        if [ -s "$b/repos/$r.untracked" ]; then
            tar -C "$d" -czf "$b/repos/$r.untracked.tar.gz" -T "$b/repos/$r.untracked"
        fi
    done
    cp "$H/MP-Gadget/Options.mk" "$b/repos/MP-Gadget.Options.mk"
    tar -C "$H" -czf "$b/repos/map2map_colleague.tar.gz" map2map_colleague
    tar -C "$PROJECT/external" -czf "$b/repos/external_rockstar_gadget4.tar.gz" \
        --exclude='*.o' rockstar gadget4 COMMITS.txt
    for e in pjm g4; do
        "$CONDA/bin/conda" env export -n "$e" > "$b/conda_$e.yml"
        "$CONDA/envs/$e/bin/python" -m pip freeze > "$b/pip_$e.txt" \
            || say "WARN pip freeze failed for $e"
    done
    tar -C "$(dirname "$MEMORY")" -czf "$b/claude_memory.tar.gz" memory
    cp "$HOME/.claude/CLAUDE.md" "$b/claude_global_CLAUDE.md"
    up "$b" bundle
    up "$PROJECT/runs" home/DMSR/cosmo_sr_project/runs
    up "$PROJECT/docs/figures" home/DMSR/cosmo_sr_project/docs/figures
}

tier1() {
    say "tier 1: results, catalogs, sim metadata"
    for d in ic_determinism halos logs env; do
        up "$S/dmsr_reward/$d" "scratch/DMSR/dmsr_reward/$d"
    done
    for d in lr_sims lr_sims_256; do
        up "$S/$d" "scratch/DMSR/$d" \
            --include '/manifest*.tsv' --include '/multiz_manifest.tsv' \
            --include '/*/paramfile*' --include '/logs/**'
    done
}

tier2() {
    say "tier 2: raw simulations"
    local I="$S/int_redshift_same_cosmology" O=scratch/DMSR/int_redshift_same_cosmology
    for r in 64 128 256; do up "$I/dmo-$r" "$O/dmo-$r"; done
    for s in set9 set10; do up "$I/dmo-512/$s" "$O/dmo-512/$s"; done
    up "$I/dmo-512" "$O/dmo-512"
}

tier3() {
    say "tier 3: derived data"
    up "$S/paired_catnorm" scratch/DMSR/paired_catnorm
    up "$S/dmsr_reward/cache/sr2_base" scratch/DMSR/dmsr_reward/cache/sr2_base
    for d in lr_sims lr_sims_256; do
        up "$S/$d" "scratch/DMSR/$d" --include '/*/catnorm*.npy'
    done
}

verify() {
    say "verify: one-way size check of every prefix"
    local m="$STAGE/MANIFEST.txt" bad=0
    : > "$m"
    for p in dmsr_reward/ic_determinism dmsr_reward/halos dmsr_reward/logs dmsr_reward/env \
             int_redshift_same_cosmology paired_catnorm dmsr_reward/cache/sr2_base; do
        if "$RCLONE" check "$S/$p" "$DEST/scratch/DMSR/$p" --one-way --size-only -q; then
            echo "OK   scratch/DMSR/$p  $("$RCLONE" size "$DEST/scratch/DMSR/$p" | tr '\n' ' ')" >> "$m"
        else
            echo "FAIL scratch/DMSR/$p" >> "$m"; bad=1
        fi
    done
    echo "bundle  $("$RCLONE" size "$DEST/bundle" | tr '\n' ' ')" >> "$m"
    echo "written $(date -Iseconds) from $(hostname)" >> "$m"
    cat "$m"
    "$RCLONE" copyto "$m" "$DEST/MANIFEST.txt" -q
    return $bad
}

[ $# -gt 0 ] || set -- 0 1 2 3 verify
for t in "$@"; do
    case "$t" in
        0|1|2|3) "tier$t" ;;
        verify)  verify ;;
        *) echo "unknown tier: $t" >&2; exit 2 ;;
    esac
done
say "done: $*"
