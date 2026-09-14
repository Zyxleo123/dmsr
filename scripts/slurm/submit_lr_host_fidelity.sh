#!/bin/bash
# Is a pre-run LR halo catalog (or the LR potential) a usable conditioning
# channel for the host centre and scale the SR operator is missing?
#
# Two chained CPU jobs: stage 1 finds/reuses the LR catalog and measures it
# against HR, stage 2 draws the figures from the saved npz.
#
#   bash scripts/slurm/submit_lr_host_fidelity.sh
#   LHF_BOXES=set8,set9,set10 bash scripts/slurm/submit_lr_host_fidelity.sh
#   DRY=1 bash scripts/slurm/submit_lr_host_fidelity.sh
#
# set8 already has an LR catalog on disk (build_lagrangian_host.py); set9/set10
# are found here for the first time -- a 262k-particle box, minutes.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

# set9 is where every radial number this job exists to interpret was measured;
# set8 is the box whose LR catalog already exists, so it costs nothing and says
# whether the answer is box-specific.
LHF_BOXES="${LHF_BOXES:-set8,set9}"
LHF_HR_MIN_NUM_P="${LHF_HR_MIN_NUM_P:-200}"
LHF_MATCH_RVIR="${LHF_MATCH_RVIR:-1.0}"
LHF_POT_NG="${LHF_POT_NG:-64}"
LHF_POT_SCALES="${LHF_POT_SCALES:-0,1.5625,3.125,6.25}"
LHF_RERUN_ROCKSTAR="${LHF_RERUN_ROCKSTAR:-0}"
# Stage 3: 20 reaches below both the 50p Rockstar floor and the 200p supervision
# floor, which is where 83% of a cluster's subhalos live.
LMC_HR_MIN_HOST_NUM_P="${LMC_HR_MIN_HOST_NUM_P:-200}"
LMC_SUB_MIN_NUM_P="${LMC_SUB_MIN_NUM_P:-20}"
LMC_CHUNK="${LMC_CHUNK:-8000000}"
LHF_LR_WORK_ROOT="${LHF_LR_WORK_ROOT:-$REWARD_ROOT/lagrangian_host}"
LHF_OUT_DIR="${LHF_OUT_DIR:-$REWARD_ROOT/lr_host_fidelity}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$LHF_OUT_DIR" "$LHF_LR_WORK_ROOT"

ENVFILE="$REWARD_ROOT/env/lr_host_fidelity_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_lr_host_fidelity.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument. List values are quoted:
# an unquoted "set8 set9" here would RUN set9 as a command.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
LHF_BOXES="$LHF_BOXES"
LHF_HR_MIN_NUM_P=$LHF_HR_MIN_NUM_P
LHF_MATCH_RVIR=$LHF_MATCH_RVIR
LHF_POT_NG=$LHF_POT_NG
LHF_POT_SCALES="$LHF_POT_SCALES"
LHF_RERUN_ROCKSTAR=$LHF_RERUN_ROCKSTAR
LMC_HR_MIN_HOST_NUM_P=$LMC_HR_MIN_HOST_NUM_P
LMC_SUB_MIN_NUM_P=$LMC_SUB_MIN_NUM_P
LMC_CHUNK=$LMC_CHUNK
LHF_LR_WORK_ROOT=$LHF_LR_WORK_ROOT
LHF_OUT_DIR=$LHF_OUT_DIR
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

# Submitting from inside an allocation would otherwise leak this shell's job
# context into the child jobs.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch scripts/slurm/lr_host_fidelity_cpu.sbatch $ENVFILE"
    echo "DRY: sbatch --dependency=afterok:<jid> scripts/slurm/lr_host_fidelity_plot_cpu.sbatch $ENVFILE"
    echo "DRY: sbatch --dependency=afterok:<jid> scripts/slurm/lr_membership_coverage_cpu.sbatch $ENVFILE"
    exit 0
fi

JID=$(sbatch --parsable scripts/slurm/lr_host_fidelity_cpu.sbatch "$ENVFILE")
echo "submitted lr_host_fidelity      -> job $JID (CPU)"
PID=$(sbatch --parsable --dependency=afterok:"$JID" \
        scripts/slurm/lr_host_fidelity_plot_cpu.sbatch "$ENVFILE")
echo "submitted lr_host_fidelity_plot -> job $PID (CPU, afterok:$JID)"
# A SIBLING of the plot, not a child: it needs only stage 1's LR catalog, so the
# two queue concurrently rather than serialising behind a figure.
MID=$(sbatch --parsable --dependency=afterok:"$JID" \
        scripts/slurm/lr_membership_coverage_cpu.sbatch "$ENVFILE")
echo "submitted lr_membership_coverage -> job $MID (CPU, afterok:$JID)"
echo
echo "watch:   tail -F $REWARD_ROOT/logs/slurm-lr_host_fid-$JID.out"
echo "         tail -F $REWARD_ROOT/logs/slurm-lr_host_fid_plot-$PID.out"
echo "         tail -F $REWARD_ROOT/logs/slurm-lr_memb_cov-$MID.out"
echo "results: $LHF_OUT_DIR/summary.json"
echo "         $LHF_OUT_DIR/<box>.npz  +  <box>_hosts.jsonl  (redrawable)"
echo "figures: $LHF_OUT_DIR/lr_host_fidelity_<box>.png"
echo
echo "         $LHF_OUT_DIR/membership_summary.json  (stage 3)"
echo
echo "read (a) first: centre error vs HR's own median subhalo radius, per host"
echo "mass bin. USABLE (<0.3) means the channel can place a subhalo; BLUR (>=1)"
echo "means it cannot, and that bin needs the potential channel or nothing."
echo
echo "then stage 3, coverage vs r/R_vir. HR subhalos PEAK at 1.0-1.5 R_vir. If"
echo "that row reads COVERED, strict membership is enough. If NULL or PARTIAL,"
echo "build the channel as a DERIVED r/R_vir field out to 2-3 R_vir instead --"
echo "membership is defined where the halo already is, not where you need it."
