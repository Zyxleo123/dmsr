#!/bin/bash
# Fine-tune SR2 on the id-gathered member-set objective over many hosts and boxes
# (scripts/features/finetune_member_gather.py).
#
# This is the run docs/sr2_member_gather.md section 7 items 3 and 4 ask for. The
# free field established that the objective is satisfiable and Rockstar-verifiable
# (72/154 against a ceiling of 151); it said nothing about whether one learned
# operator can reach it, because a free field sees the answer and generalises to
# nothing. Here the variable is the generator's weights and the score that matters
# is on hosts and boxes never trained on.
#
# Node choice, checked with `myfree` at write time: `general` has gpu28 (a5000,
# 24 GB, 3/4 free, 468 GB host free), gpu24 and gpu29 (a6000, 48 GB) AVAILABLE.
# The job is not pinned to a GPU class because the footprint does not need it:
# gradient accumulation runs ONE host at a time (four 64^3 tiles, batch 4) and
# frees between, so peak GPU memory is a single host's forward/backward plus the
# chunked N x N potential -- the free-field run measured everything fitting in
# 11 GB. If `general` is full, `legacy` rtx_2080_ti (11 GB) is marginal but
# workable at --pot-chunk 1024; switch the partition AND the --qos together,
# since rhea enforces one QOS per partition.
#
# HOST RAM is the real ask and it is why this is 96G. build_pool loads one box's
# 537 MB owner array at a time and inverts it into a CSR index (2-3 GB peak),
# and the pool itself holds ~60 MB of tile data per host on the CPU -- the GPU
# never holds more than one host, deliberately, so the pool size is bounded by
# host RAM rather than by GPU memory.
#
#   sbatch scripts/slurm/member_gather_train_gpu.sbatch <envfile>
#SBATCH --job-name=mgtrain
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# --mem was 96G from this job's first day and nothing had ever measured it. On
# 2026-09-09 that cost real time: 96G did not fit ANY available node on
# `general` (the only two with a free, unreserved GPU had 20G and 84G free), so
# every arm of the shell-lever set sat PENDING behind a number nobody had
# checked -- and it read as a fairshare problem, which it was not.
#
# MEASURED by the ss30_fresh pool-only shakeout (job 46659), which is what
# `report_rss` in finetune_member_gather.py exists to print:
#     peak host RSS after the pool build (56 hosts, 2 draws) = 10.1 GB
# 32G covers that, the 3-draw arms, and the mmap'd frozen box (3.2 GB per box,
# which ru_maxrss does NOT count but the cgroup may charge), with headroom.
#
# Both of these are plain sbatch CLI flags, so the submitter overrides them with
# MG_MEM / MG_TIME on the command line. That is NOT `--export` and does not trip
# SLURM_GET_USER_ENV -- a CLI --mem simply wins over the directive above.
# Re-measure with SHAKEOUT_ONLY=1 before raising either.
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-mgtrain-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${DATA:=/zfsauton/scratch/yixiz/DMSR/paired_catnorm}"
: "${MG_TRAIN_BOXES:=set3 set4 set5 set6 set7}"
: "${MG_HOLDOUT_BOXES:=set9 set10}"
: "${MG_HOLDOUT_HOSTS:=}"
: "${MG_N_TILES:=4}"
: "${MG_MAX_HOSTS_PER_BOX:=8}"
: "${MG_MIN_LOG_MVIR:=13.5}"
: "${MG_POOL_ONLY:=0}"
: "${MG_RUNG:=fine}"
: "${MG_LR_SCALE:=10.0}"
: "${MG_STEPS:=8000}"
: "${MG_HOSTS_PER_STEP:=2}"
: "${MG_SETS_PER_STEP:=0}"      # 0 = all of a host's sets, the full batch
: "${MG_CLIP:=1.0}"
: "${MG_DETECT_ANOMALY:=0}"     # debug: autograd anomaly detection (slow)
: "${MG_EVAL_EVERY:=250}"
# --- the objective, identical to the free field's defaults -------------------
: "${MG_W_VIRIAL:=1.0}"
: "${MG_W_BOUND:=1.0}"
: "${MG_W_D6:=1.0}"
: "${MG_W_RRMS:=0.3}"
: "${MG_W_SIGMAV:=0.3}"
: "${MG_W_CENTRE:=1.0}"
: "${MG_W_HOST_SETS:=0.0}"           # host preservation guard; 0 = off (baseline)
: "${MG_HOST_MIN_NUM_P:=200}"
: "${MG_HOST_MAX_SETS:=256}"
: "${MG_HOST_SETS_PER_STEP:=64}"
: "${MG_MIN_NUM_P:=200}"
# --- the LR-host conditioning channel ---------------------------------------
# EMPTY = the unconditioned control, byte-identical to every arm before this
# existed. Set it and the generator gains a zero-initialised input embedding for
# the channels named, which is the only thing in this pipeline that gives a
# local operator the host CENTRE and SCALE it needs to stop emitting a flat
# radial profile (`radial-profile-is-flat-in-host-mass`).
#
# The fields must already be built -- the trainer refuses to start otherwise:
#     bash scripts/slurm/submit_lr_host_channel.sh
: "${MG_COND_CHANNELS:=}"
: "${MG_COND_MODE:=derived}"
: "${MG_COND_MIN_HR_NUM_P:=20000}"
: "${MG_COND_R_MAX_RVIR:=3.0}"
: "${MG_COND_LOG_MHOST_REF:=13.0}"
: "${MG_COND_LOG_MHOST_SCALE:=1.0}"
: "${MG_COND_ROOT:=}"
# cond0 starts at exactly ZERO and has no pretrained function to protect, so it
# is trained at the LARGEST rate of any group. A coarse-sized 3e-7 would leave
# the channel wired in and inert for ten hours.
: "${MG_COND_LR:=1e-5}"
# --- WHICH halos the member loss is responsible for --------------------------
# `sub` is every arm recorded so far. The held-out gate says the assumption
# behind it is wrong: in the edited region HR resolves 3,775 top-level halos
# >=200p against base's 3,028 -- a 747-halo deficit OUTSIDE hosts that no
# supervised set has ever pointed at. `all` selects on SIZE alone. It REQUIRES
# MG_MAX_NUM_P (the job refuses otherwise: without a ceiling the cluster host
# enters the supervised pool at ~100k particles and 1e10 pairs), and it wants
# MG_SET_SELECT=stratified, or the `largest` cap keeps the top of the mass
# function and drops the small halos the arm exists to build.
: "${MG_POPULATION:=sub}"
: "${MG_MAX_NUM_P:=0}"
: "${MG_SET_SELECT:=largest}"
# --- the THIRD selection axis: WHICH ENVIRONMENT ----------------------------
# `any` = every arm so far. The set9 census (region scope, 32 tiles) says this
# is the axis that was missing: of the region's 11,698 missing objects 6,557
# sit in `near` (1-3 R_vir of a bigger resolved host, the infall region) and
# MG_POPULATION=all closed 249 of them -- 3.8% -- while OVERSHOOTING HR in
# `field` (subhalos 3,316 against HR's 2,466, and 2-6x too many above 200p).
# Selecting on size cannot reach `near`: a cluster region's pool is dominated
# by field halos and `stratified` stratifies on MASS. `near` sits inside a
# cluster's own Lagrangian tiles, so unlike `field` it is reachable by
# selection rather than by a wider MG_N_TILES.
: "${MG_ENV_SELECT:=any}"
: "${MG_ENV_HOST_FLOOR:=200}"
# --- isolation: how much NON-MEMBER mass sits on the object ------------------
# 0 = off = every arm so far. This is the only INTER-set term in the loss, and
# it exists because "top-level" is an inter-set property: Rockstar files a halo
# as a subhalo when a bigger halo's R_vir contains its centre, which is not a
# property of its members at all. Every other term is a moment about the set's
# own centroid, so an intra-set loss pointed at top-level HR halos scores full
# marks on an object that a pedestal then swallows -- MEASURED, set9:
# MG_POPULATION=all built 3,374 net objects and Rockstar filed every one as a
# subhalo (subs +4,260, top-level -886). `d6` does not close this: it
# normalises by the background's DISPERSION, not its density.
# Host-centric local temperature (host_temperature.py): two-sided, velocity
# only. MG_TEMP_MAX_X=0.5 is the core-only arm; MG_TEMP_MEASURE=1 logs it at 0.
: "${MG_W_TEMP:=0.0}"
: "${MG_TEMP_MAX_X:=3.0}"
: "${MG_TEMP_QUERIES:=256}"
: "${MG_TEMP_MEASURE:=0}"
: "${MG_W_ISOLATION:=0.0}"
: "${MG_ISO_RADIUS_FRAC:=0.5}"
: "${MG_ISO_SOFTNESS:=0.15}"
: "${MG_ISO_MIN_COUNT:=8.0}"
# --- the EARLY gate ----------------------------------------------------------
# Steps at which to freeze an IMMUTABLE tuned_step<N>.pt, so a Rockstar gate can
# run CONCURRENTLY with the rest of training. Empty = none = every arm so far.
# It costs the run nothing: training still goes to MG_STEPS.
: "${MG_GATE_AT_STEPS:=}"
: "${MG_MIN_PURITY:=0.5}"
: "${MG_MIN_LIVE_FRAC:=0.5}"
: "${MG_STATS_MEMBERS:=all}"   # all = the hybrid every arm on disk trained on
: "${MG_SOFTENING:=0.01}"
: "${MG_BOUND_TAU:=0.5}"
: "${MG_BG_K:=4096}"
: "${MG_BG_RADIUS:=4.0}"
: "${MG_POT_CHUNK:=2048}"
# Rows per pair block are min(MG_POT_CHUNK, MG_POT_MAX_ELEMS / N). The chunk
# ALONE is not a memory bound and that is what killed jobs 35746-35748: a
# 118,000-particle satellite at chunk 2048 is a 2.7 GiB allocation, and the
# autograd tape used to keep one per chunk. Both are fixed; this is the belt.
: "${MG_POT_MAX_ELEMS:=16777216}"
: "${MG_MAX_SET_PARTICLES:=0}"  # 0 = the exact estimator; a TIME knob
: "${MG_BOUND_PENALTY:=hinge}"  # hinge = the form every run so far used
: "${MG_TERM_NORM:=0}"         # 0 = the incidental budget of every run so far
: "${MG_CENTRE_MODE:=full}"    # full = the term that scored 72/154
: "${MG_CENTRE_DEAD_ZONE:=0.0}" # 0 = the term that scored 72/154
: "${MG_CENTRE_HUBER:=0.0}"
# --- the shell: right distance from the host, any direction ------------------
# 0 = off = every arm so far. See submit_member_gather_train.sh for the arm.
: "${MG_W_RADIUS:=0.0}"
: "${MG_RADIUS_DEAD_ZONE:=1.0}"
# --- SOFT MEMBERSHIP: which particles, as a latent variable ------------------
# 0 = off = every arm recorded so far, and the hard path stays byte-identical.
#
# What it changes. Every term of this loss is a statistic of a particle
# collection the loss is NOT allowed to choose -- HR's owner array names it. So
# when SR2's material for a subhalo is not the material HR used, the loss
# charges the generator for failing to bind particles that were never going to
# be there. Here HR still supplies the object's REGION (a coarse Lagrangian
# prior: every particle of an LR cell gets that cell's member fraction) and its
# PHYSICS (the reference, untouched), and the CURRENT SR field decides which
# particles participate -- under a per-particle budget of one shared with a
# background/host category, and a per-candidate mass target. Both constraints
# are load-bearing: without the first, several candidates spend the same
# convenient particles; without the second, a candidate throws everything away
# and keeps a tiny cold core that scores perfectly on every moment.
#
# MG_SOFT_LAMBDA is the only knob that interpolates: lambda -> inf pins the
# membership at the prior, and at MG_SOFT_PRIOR=hard that limit IS the arm on
# disk. So the sweep's two controls are `hard` at large lambda (reproduces the
# gate-passing arm) and `coarse` at large lambda (says whether the coarse
# window ALONE is already destructive -- the Gaussian-window failure of
# docs/sr2_gather_finetune.md section 6, which is the risk this construction
# is designed around).
#
# COST: the pool is (1 + ratio) times the member count and the estimator is
# O(pool^2), so ratio 2 is ~9x the hard arm's pair bill. MG_MAX_SET_PARTICLES
# is what bounds it -- leave it set for any soft arm.
#
# WATCH in metrics.jsonl: soft_mass_ratio must sit at ~1.0 (below it means
# lambda made the window unfillable) and soft_hr_fraction says how much of each
# solved object is still HR's own material -- 1.0 is the hard arm, and a
# collapse toward 0 is the relaxation walking off the object.
: "${MG_SOFT_MEMBERSHIP:=0}"
: "${MG_SOFT_LAMBDA:=1.0}"
: "${MG_SOFT_PRIOR:=coarse}"     # coarse | hard
: "${MG_SOFT_FACTOR:=8}"         # HR sites per LR cell side; the run's SR factor
: "${MG_SOFT_POOL_RATIO:=2.0}"
: "${MG_SOFT_CELL_DILATE:=0}"
: "${MG_SOFT_PRIOR_EPS:=1e-3}"
: "${MG_SOFT_BG_FLOOR:=1e-3}"
: "${MG_SOFT_STEPS:=3}"
: "${MG_SOFT_MASS_STEPS:=10}"
: "${MG_SOFT_MASS_ETA:=1.0}"
: "${MG_SOFT_MASS_STEP_CAP:=20.0}"
: "${MG_SOFT_SIGMA:=adaptive}"   # adaptive | hr | cand
: "${MG_SOFT_SIGMA_SCALE:=1.0}"

# --- THE HOST AS A BACKGROUND COMPONENT (the radial-placement arm) -----------
# MEASURED (radial-placement-is-inverted): HR's median subhalo sits at 1.13
# R_vir and every tuned arm at 0.30-0.35. The cause is selection, not the loss:
# 7 of the 8 gather terms are translation-invariant and cannot express WHERE an
# object sits, while the membership solve's flat background plus its n_c/f^3
# prior make the host core the cheapest place for a candidate to meet its mass
# target. MG_SOFT_HOST_PROFILE subtracts the host's log density from the
# background so membership becomes a log OVERDENSITY test.
#
#   lr    analytic NFW from the LR catalog -- inference-honest, and BLANK for
#         the ~43% of hosts under the LR completeness gate, which fall back to
#         the flat background. Needs the table from build_lr_host_table.py.
#   self  binned off the candidate field itself -- complete, partly circular.
#   parent NFW about each set's OWN HR parent (lr and self are centred on the
#         TASK host, which owns only ~1/3 of a pool's sets). Training-side only.
#
# MG_SOFT_HOST_WEIGHT=0 is byte-identical to every soft arm already on disk,
# which is the zero of the ladder. beta is NOT comparable between lr and self
# (their log-density ranges differ by ~3x) -- read the ladder off
# soft_host_logrho_p10/p90 in the run's diagnostics, in nats.
: "${MG_SOFT_HOST_PROFILE:=off}"   # off | lr | self | parent
: "${MG_SOFT_HOST_WEIGHT:=0.0}"
: "${MG_SOFT_HOST_CONC:=5.0}"
: "${MG_SOFT_HOST_BINS:=24}"
: "${MG_SOFT_HOST_R_MAX_RVIR:=4.0}"
: "${MG_LR_HOST_TABLE:=}"
: "${MG_LR_HOST_R_MAX_RVIR:=2.0}"
# Per-set radial prior in the solve (pulls each centre to its own HR radius,
# measured from its own HR parent). 0 = off.
: "${MG_SOFT_RADIAL_WEIGHT:=0.0}"
: "${MG_SOFT_RADIAL_WIDTH:=0.2}"

# --- membership-solve dump (visualisation only) ------------------------------
# Set MG_DUMP_MEMBERSHIP_NPZ to build this arm's pool, solve one host's
# membership on MG_DUMP_MEMBERSHIP_FIELD with the per-sweep recorder on, write
# the snapshots to the npz, and EXIT before training. Everything else (prior,
# lambda, rung, guards) is the arm's own config, so the dumped solve is the one
# the arm actually ran. MG_DUMP_MEMBERSHIP_HOST defaults to the first held-out
# host.
: "${MG_DUMP_MEMBERSHIP_NPZ:=}"
: "${MG_DUMP_MEMBERSHIP_FIELD:=}"
: "${MG_DUMP_MEMBERSHIP_HOST:=}"

# --- relocation: which PARTICLES each supervised group is made of ------------
# The arm A / B / C experiment (src/cosmo_sr/features/relocate_sets.py).
#   off     arm A -- HR's owner array. Every result recorded in this line.
#   frozen  arm B -- groups re-chosen from compact frozen-SR material at the
#                    radius the group's frozen centroid already sits at.
#   hr      arm C -- the same selection at HR's radius.
# The HR targets are IDENTICAL in all three; only the material changes, so the
# three Rockstar gates are directly comparable.
#
# MG_RELOCATE_CHECK_ONLY=1 builds the pool, writes the relocation report into
# pool.json and exits WITHOUT training -- the proposal's "check this
# distribution before training". Read `frac_within_eps` and `n_starved` there.
: "${MG_RELOCATE:=off}"            # off | frozen | hr
: "${MG_RELOCATE_DIRS:=8}"
: "${MG_RELOCATE_EPS:=0.05}"       # radius tolerance, in the HOST's R_vir
: "${MG_RELOCATE_OVERSAMPLE:=4.0}"
: "${MG_RELOCATE_REFINE_ITERS:=3}"
: "${MG_RELOCATE_W_VEL:=0.5}"
: "${MG_RELOCATE_W_RADIUS:=1.0}"
: "${MG_RELOCATE_CHECK_ONLY:=0}"
: "${MG_W_SHELL:=0.0}"
: "${MG_SHELL_DEAD_ZONE:=0.02}"
# Reduce the sorted pairing to this many equal-COUNT quantile bins. 0 = the full
# rank-by-rank pairing every arm through 2026-09-08 ran. MEASURED: at Q=0 the
# w_shell=30 arm drove its train residual 7.3x while holdout moved 1.4x -- the
# full pairing hands the loss one degree of freedom per set and most of them are
# per-host detail that does not transfer. Try 3-5.
: "${MG_SHELL_QUANTILES:=0}"
: "${MG_SHELL_EPS:=0.01}"
: "${MG_RADIUS_HUBER:=0.0}"
# --- the guards --------------------------------------------------------------
: "${MG_W_LOW:=100.0}"
: "${MG_W_HIGHK:=10.0}"
: "${MG_K_SPLIT:=4.0}"
: "${MG_LOW_K_MAX:=0.02}"
: "${MG_HIGHK_MAX:=1.5}"
: "${MG_VEL_RMS_TOL:=0.10}"
# --- band-resolved high-k + the velocity power guard, sr2_member_gather_training
# section 11.6. ALL default to the behaviour the four finished arms had, so an
# unset knob reproduces them rather than quietly redefining the objective.
: "${MG_HIGHK_BINS:=0}"        # 0 = the single mode-mean the four arms used
: "${MG_HIGHK_K_MAX:=0.0}"     # 0 = to the cube corner, as the scalar's mask
: "${MG_HIGHK_TOL:=0.0}"       # 0 = no dead zone, as the four arms had
: "${MG_HIGHK_TWO_SIDED:=0}"   # displacement lower arm; see 11.6 for why NOT
: "${MG_HIGHK_REDUCE:=mean}"
: "${MG_HIGHK_MAX_HOLDOUT:=0.0}"  # 0 = the historical train-only verdict
: "${MG_W_VEL_HIGHK:=0.0}"     # 0 = measure velocity power, charge nothing
: "${MG_VEL_HIGHK_TOL:=0.25}"
: "${MG_VEL_HIGHK_MIN:=0.0}"   # 0 = no velocity-power criterion, as before
# --- the box-wide (unsupervised-tile) high-k support -------------------------
# selfvel-arm-failed-the-gate: the supervised hinge held 0.56x on train tiles yet
# 3.9x held out because its support is 6.25% of the box. These charge the SAME
# banded hinge on random tiles drawn box-wide. ALL default to OFF, so an unset
# knob reproduces the supervised-only guard.
: "${MG_UNSUP_TILES_PER_BOX:=0}"   # 0 = off; >0 builds the pool (~15 MB/tile)
: "${MG_UNSUP_TILES_PER_STEP:=8}"  # tiles carrying the hinge each step
: "${MG_W_HIGHK_UNSUP:=0.0}"       # 0 = only measure the pool; >0 adds the loss
: "${MG_UNSUP_HIGHK_MAX:=0.0}"     # 0 = no box-wide verdict gate
# --- the box-wide kNN-distance support, src/cosmo_sr/features/knn_cdf.py. Same
# pool as the hinge; grid-free and catalog-free. Default OFF. MG_KNN_KS max sets
# halo vs fragments (measured) -- keep it >= 64. Set MG_KNN_BUFFER_CELLS from
# knn_cdf_diagnostic.py's boundary experiment, not by taste.
: "${MG_W_KNN_UNSUP:=0.0}"
: "${MG_KNN_KS:=8 32 64 128}"
: "${MG_KNN_QUERIES:=1024}"
: "${MG_KNN_QUANTILES:=64}"
: "${MG_KNN_DEAD_ZONE:=0.02}"
: "${MG_KNN_BUFFER_CELLS:=0}"
: "${MG_KNN_NBR_MARGIN_CELLS:=8}"
: "${MG_KNN_CENTRE:=centroid}"
: "${MG_KNN_ONE_SIDED:=0}"
# MEASURED (round 5b, docs/sr2_knn_distance.md section 12): the term is worth
# training on ONLY in phase space and ONLY per subhalo class. Position-only
# ranks the working arms at rho -0.23 (p 0.32); (x, v/4000) at -0.65 (p 0.002).
# Hosts belong in the monitor, never the loss: the working arms gain subhalos by
# over-concentrating hosts in phase space (class term vs count, rho +0.91).
: "${MG_KNN_PHASE_LAMBDA:=0}"      # km/s per Mpc/h; 4000 is the measured pass
: "${MG_KNN_CLASSES:=}"            # e.g. "sub_50_200"; empty = pooled (do not)
# --- the catalog-free strata arm (docs/sr2_knn_strata.md). Density strata each
# field reads off its OWN r_k, in place of the HR owner-class mask. Gated by
# submit_knn_strata.sh: do NOT set these without a passing verdict json.
# MG_KNN_STRATA alone is the pooled loss in finer bins -- MEASURED. The arm is
# STRATA + STRATA_CHARGE, and CLASSES_DIAG_ONLY is what takes the label out of
# the gradient while keeping it in the report.
: "${MG_KNN_STRATA:=0}"
: "${MG_KNN_STRATA_RANK:=0}"       # 0 = min(ks)
: "${MG_KNN_STRATA_CHARGE:=}"      # e.g. "0" or "0,1"; empty = all (= pooled)
: "${MG_KNN_CLASSES_DIAG_ONLY:=0}"
# --- the HR critic, docs/sr2_gather_critic.md. Default OFF: MG_W_ADV=0 makes the
# run byte-identical to the four finished arms. The hand-crafted high-k/velocity
# hinges above each charge for ONE named statistic; the critic is their general
# form -- it sees whole HR tiles vs tuned tiles (6-channel high-pass, velocity
# included) and learns the field-realism defects the gate found (section 11.6).
: "${MG_W_ADV:=0.0}"           # 0 = critic off entirely, the four-arm baseline
: "${MG_ADV_WARMUP:=500}"      # critic trains here; generator adv held at 0
: "${MG_ADV_RAMP:=2000}"       # linear ramp of w_adv after warmup
: "${MG_N_CRITIC:=1}"
: "${MG_CRITIC_LR:=2e-4}"
: "${MG_CRITIC_WIDTH:=64}"
: "${MG_CRITIC_LAYERS:=3}"
: "${MG_CRITIC_GLOBAL_POOL:=0}" # 1 = single-score D (tile-scale stats); 0 = PatchGAN
: "${MG_CRITIC_R1_GAMMA:=10.0}"
: "${MG_CRITIC_R1_INTERVAL:=16}"
: "${MG_CRITIC_NORM_FIT_TILES:=32}"
: "${MG_LABEL:=}"
: "${MG_SEED:=0}"
# --- the NOISE realisation (the memorisation channel) ------------------------
# A run holds (LR tile, noise tile) constant for every step, so the fixed noise
# is an input carrying real per-tile detail and zero generalisable information.
# Anything the LR field does not carry can still be fitted through it, and all
# of that evaporates on a held-out box's fresh draw -- which is the shape of the
# 2026-09-08 shell result. These three knobs remove it, and default to OFF so an
# unset run reproduces every arm on disk exactly.
: "${MG_NOISE_DRAWS:=1}"        # realisations per tile available to training
: "${MG_NOISE_MODE:=fixed}"     # fixed | rotate
: "${MG_FRESH_NOISE_EVAL:=0}"   # reserve one never-trained draw for the 2x2
: "${MG_SHELL_NOISE_DRAWS:=0}"  # average the shell curve over this many extras
: "${MG_WANDB_MODE:=}"          # online / offline / disabled; empty = config

# The preamble builds a MINIMAL environment on purpose and never sources
# ~/.bashrc, so WANDB_API_KEY -- which lives there -- would be absent and every
# run would silently downgrade to offline. Read just that one variable out of
# $HOME rather than sourcing the whole file (which would fight the preamble's
# conda setup) and rather than putting it in the env file (which lives on
# world-readable shared scratch; a secret does not belong there).
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$HOME/.bashrc" ]; then
    # \042 = double quote, \047 = single quote: octal keeps this line free of
    # nested-quote gymnastics, which is how it got mangled the first time.
    _k=$(grep -m1 '^[[:space:]]*export[[:space:]]\+WANDB_API_KEY=' "$HOME/.bashrc" \
         | cut -d= -f2- | tr -d '\042\047 ')
    [ -n "$_k" ] && export WANDB_API_KEY="$_k"
    unset _k
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "=== WANDB_API_KEY present (online logging available)"
else
    echo "=== no WANDB_API_KEY: wandb falls back to OFFLINE (sync later with"
    echo "    'wandb sync \$DMSR_REWARD_ROOT/member_gather/<run>/wandb/offline-*')"
fi

require_input "$PROJECT/external/SRS-map2map/SRmodel/G_z0.pt" \
    "the frozen SR2 checkpoint (configs/sr2_baseline/freeze.yaml)"

# Membership IS the supervision: a box with no owner array contributes nothing
# and is skipped inside the trainer with a message. Gate only on there being at
# least one, so a partially-built pool still runs rather than stranding the job.
_HAVE=0
for b in $MG_TRAIN_BOXES; do
    [ -r "$DMSR_REWARD_ROOT/halos_particles/${b}__hr__hr/${b}_hr_owner.npy" ] \
        && _HAVE=$((_HAVE + 1))
done
if [ "$_HAVE" = "0" ]; then
    echo ">>> none of the training boxes ($MG_TRAIN_BOXES) has an HR owner array."
    echo ">>> produced by: bash scripts/slurm/submit_owner_arrays.sh"
    echo ">>> exiting 0 so dependent jobs report the same rather than sitting in"
    echo ">>> DependencyNeverSatisfied with no explanation."
    exit 0
fi
echo "=== $_HAVE training boxes have owner arrays"

EXTRA=()
[[ "$MG_POOL_ONLY" == "1" ]] && EXTRA+=(--pool-only)
[[ "$MG_SOFT_MEMBERSHIP" == "1" ]] && EXTRA+=(--soft-membership)
[[ "$MG_RELOCATE_CHECK_ONLY" == "1" ]] && EXTRA+=(--relocate-check-only)
if [[ "$MG_SOFT_HOST_PROFILE" != "off" ]]; then
  if [[ "$MG_SOFT_MEMBERSHIP" != "1" ]]; then
    echo ">>> GATE FAILED: MG_SOFT_HOST_PROFILE=$MG_SOFT_HOST_PROFILE needs"
    echo ">>> MG_SOFT_MEMBERSHIP=1 -- the hard arm's member sets are HR's"
    echo ">>> labels and no background competes for them. Exiting 0 so"
    echo ">>> dependents report a gate, not a DependencyNeverSatisfied."
    exit 0
  fi
  EXTRA+=(--soft-host-profile "$MG_SOFT_HOST_PROFILE"
          --soft-host-weight "$MG_SOFT_HOST_WEIGHT"
          --soft-host-conc "$MG_SOFT_HOST_CONC"
          --soft-host-bins "$MG_SOFT_HOST_BINS"
          --soft-host-r-max-rvir "$MG_SOFT_HOST_R_MAX_RVIR"
          --lr-host-r-max-rvir "$MG_LR_HOST_R_MAX_RVIR")
  [[ -n "$MG_LR_HOST_TABLE" ]] && EXTRA+=(--lr-host-table "$MG_LR_HOST_TABLE")
  echo "=== HOST PROFILE: $MG_SOFT_HOST_PROFILE at beta=$MG_SOFT_HOST_WEIGHT"
  echo "=== watch soft_host_valid_frac (how many hosts got a profile) and"
  echo "=== soft_host_logrho_p10/p90 (the ladder, in nats)."
fi
if [[ "$MG_SOFT_RADIAL_WEIGHT" != "0" && "$MG_SOFT_RADIAL_WEIGHT" != "0.0" ]]; then
  EXTRA+=(--soft-radial-weight "$MG_SOFT_RADIAL_WEIGHT"
          --soft-radial-width "$MG_SOFT_RADIAL_WIDTH")
  echo "=== RADIAL PRIOR: gamma=$MG_SOFT_RADIAL_WEIGHT width=$MG_SOFT_RADIAL_WIDTH R_vir"
fi
[[ "$MG_DETECT_ANOMALY" == "1" ]] && EXTRA+=(--detect-anomaly)
[[ -n "$MG_WANDB_MODE" ]] && EXTRA+=(--wandb-mode "$MG_WANDB_MODE")
if [[ -n "$MG_DUMP_MEMBERSHIP_NPZ" ]]; then
  EXTRA+=(--dump-membership-npz "$MG_DUMP_MEMBERSHIP_NPZ" \
          --dump-membership-field "$MG_DUMP_MEMBERSHIP_FIELD")
  [[ -n "$MG_DUMP_MEMBERSHIP_HOST" ]] && \
    EXTRA+=(--dump-membership-host "$MG_DUMP_MEMBERSHIP_HOST")
fi

echo "=== member-gather fine-tune"
echo "    train boxes:   $MG_TRAIN_BOXES"
echo "    holdout boxes: $MG_HOLDOUT_BOXES"
echo "    holdout hosts: ${MG_HOLDOUT_HOSTS:-<none>}"
echo "    rung $MG_RUNG, lr_scale $MG_LR_SCALE, $MG_STEPS steps,"
echo "    $MG_HOSTS_PER_STEP hosts/step, sets/step ${MG_SETS_PER_STEP} (0=all),"
echo "    <= $MG_MAX_HOSTS_PER_BOX hosts/box"
echo "    pair sums: chunk $MG_POT_CHUNK capped at $MG_POT_MAX_ELEMS elements,"
echo "    per-set particle cap ${MG_MAX_SET_PARTICLES} (0 = exact estimator)"
if [ "$MG_SOFT_MEMBERSHIP" = "1" ]; then
  echo "    SOFT MEMBERSHIP ON: prior $MG_SOFT_PRIOR, lambda $MG_SOFT_LAMBDA,"
  echo "      factor $MG_SOFT_FACTOR, pool ratio $MG_SOFT_POOL_RATIO,"
  echo "      window $MG_SOFT_SIGMA, ${MG_SOFT_STEPS}x${MG_SOFT_MASS_STEPS} solve."
  echo "      READ soft_mass_ratio (~1.0) and soft_hr_fraction in metrics.jsonl;"
  echo "      pool_over_live and sum_pool_squared in pool.json are the cost."
  if [ "$MG_MAX_SET_PARTICLES" = "0" ]; then
    echo "      WARNING: MG_MAX_SET_PARTICLES=0 with a ${MG_SOFT_POOL_RATIO}x pool"
    echo "      leaves the O(N^2) pair sum uncapped -- expect ~9x the hard arm's"
    echo "      step time. Set a cap unless the bill has been checked."
  fi
else
  echo "    soft membership OFF (hard HR member mask, every arm to date)"
fi

if [ "$MG_RELOCATE" != "off" ]; then
  _arm=$([ "$MG_RELOCATE" = "frozen" ] && echo B || echo C)
  echo "    RELOCATION ON: arm $_arm (--relocate $MG_RELOCATE)."
  echo "      Each group's particles are re-chosen ONCE, before training, from"
  echo "      compact frozen-SR material at the requested host-centric radius."
  echo "      HR's targets are unchanged. eps $MG_RELOCATE_EPS R_vir_host,"
  echo "      $MG_RELOCATE_DIRS directions, $MG_RELOCATE_REFINE_ITERS swap passes."
  echo "      READ pool.json 'relocation': frac_within_eps near 1 and n_starved"
  echo "      near 0, or the material is not there and training answers nothing."
  if [ "$MG_RELOCATE_CHECK_ONLY" = "1" ]; then
    echo "      CHECK ONLY: this job writes pool.json and does NOT train."
  fi
else
  echo "    relocation OFF (arm A: HR's owner array, every arm to date)"
fi
if awk "BEGIN{exit !($MG_W_HOST_SETS > 0)}"; then
  echo "    HOST PRESERVATION GUARD ON: w $MG_W_HOST_SETS, >= $MG_HOST_MIN_NUM_P p,"
  echo "      <= $MG_HOST_MAX_SETS host sets/cluster, ${MG_HOST_SETS_PER_STEP} host sets/step"
else
  echo "    host preservation guard OFF (--w-host-sets 0)"
fi
# store_true: the flag is PRESENT or ABSENT. Deliberately unquoted at the call
# site so an empty value disappears under word splitting rather than becoming an
# empty argument python would reject.
MG_TERM_NORM_FLAG=""
[ "$MG_TERM_NORM" = "1" ] && MG_TERM_NORM_FLAG="--term-norm"
MG_HIGHK_TWO_SIDED_FLAG=""
[ "$MG_HIGHK_TWO_SIDED" = "1" ] && MG_HIGHK_TWO_SIDED_FLAG="--highk-two-sided"
MG_CRITIC_GLOBAL_POOL_FLAG=""
[ "$MG_CRITIC_GLOBAL_POOL" = "1" ] && MG_CRITIC_GLOBAL_POOL_FLAG="--critic-global-pool"
MG_FRESH_NOISE_FLAG=""
[ "$MG_FRESH_NOISE_EVAL" = "1" ] && MG_FRESH_NOISE_FLAG="--fresh-noise-eval"

echo "    noise:  ${MG_NOISE_DRAWS} draw(s), mode ${MG_NOISE_MODE}," \
     "fresh-noise eval ${MG_FRESH_NOISE_EVAL}," \
     "shell averaged over +${MG_SHELL_NOISE_DRAWS}"
if [ "$MG_NOISE_MODE" = "rotate" ] || [ "$MG_FRESH_NOISE_EVAL" = "1" ]; then
  echo "      NOTE each extra draw costs one frozen forward AND one full"
  echo "           reference build per host at pool time: centre_self and the"
  echo "           hybrid reference are functions of the frozen field, so a"
  echo "           draw carries its own set table or it is not an honest draw."
  echo "           Budget pool build time as ~N_DRAWS x the single-draw run."
fi

echo "    budget: bound penalty ${MG_BOUND_PENALTY}, term-norm ${MG_TERM_NORM}"
if [ "$MG_BOUND_PENALTY" = "hinge" ] && [ "$MG_TERM_NORM" != "1" ]; then
echo "            (both defaults -- MEASURED, that budget is d6 88%, bound"
echo "             0.24%, and bound is the gate's own criterion. Section 10.)"
fi
if [ "$MG_STATS_MEMBERS" = "live" ]; then
echo "    REFERENCE: live members only -- the out-of-tile stragglers are OUT of"
echo "      every statistic (reference, candidate, centre targets, bg radius)."
echo "      MEASURED: the hybrid reference sits at 2.18x HR's r_rms and 1.99x"
echo "      its 2T/|W| at live_frac 0.979, because r_rms is an RMS."
echo "      READ pure_hr_median against hybrid_reference_median in pool.json --"
echo "      reference_median must now equal pure_hr_median."
else
echo "    reference: all members (hybrid: live + frozen stragglers), the"
echo "      historical default. MG_STATS_MEMBERS=live de-biases it."
fi
echo "    centre: mode ${MG_CENTRE_MODE}, dead zone ${MG_CENTRE_DEAD_ZONE}r,"
echo "            Huber knee ${MG_CENTRE_HUBER}r"
if [ "$MG_CENTRE_MODE" != "full" ]; then
echo "    NOTE: section 9.4. This arm charges for LESS than the address, which"
echo "          is the point -- the address is only 13.4% predictable out of"
echo "          sample. Read centre_penalised_radii against centre_offset_radii"
echo "          and score the HELD-OUT hosts, because in-sample this arm can"
echo "          only look worse than the full term."
fi
echo "            (0/0 is the pure quadratic that scored 72/154 -- the control)"
echo "    env:    env_select ${MG_ENV_SELECT} (host floor ${MG_ENV_HOST_FLOOR}p)"
echo "    temp:   w_temp ${MG_W_TEMP}, max_x ${MG_TEMP_MAX_X}, queries ${MG_TEMP_QUERIES}, measure ${MG_TEMP_MEASURE}"
echo "    iso:    w_isolation ${MG_W_ISOLATION}, radius ${MG_ISO_RADIUS_FRAC}"\
     " of the bg sphere, min count ${MG_ISO_MIN_COUNT}"
echo "    shell:  w_radius ${MG_W_RADIUS}, dead zone ${MG_RADIUS_DEAD_ZONE}r," \
     "Huber knee ${MG_RADIUS_HUBER}r"
if [ "${MG_W_RADIUS%%.*}" != "0" ] || [ "${MG_W_RADIUS#*.}" != "0" ]; then
echo "    NOTE: the shell arm. Charges the host-centric RADIUS only, direction"
echo "          free -- the 60% of the centre offset that is a rule (58% of"
echo "          frozen sets sit too CLOSE, median error 2.80 radii) and none of"
echo "          the transverse part no input predicts. It targets the \`self\`"
echo "          arm's radial collapse: held-out set9 median 0.35 host R_vir vs"
echo "          HR's 1.13, 6496 subhalos inside 0.25 R_vir where HR has 593."
if [ "$MG_CENTRE_MODE" = "self" ]; then
echo "    WARNING: --centre-mode self PINS the radius this term is trying to"
echo "             move. Both quadratic, they settle at w_radius/(w_centre +"
echo "             w_radius) of the correction -- HALF at equal weights. Use"
echo "             MG_CENTRE_MODE=self_tan to make them orthogonal instead."
fi
echo "          Read train/radius_ratio -> 1.0 and holdout/radius_ratio; the"
echo "          host preservation sets never carry this term (a host has no"
echo "          host) and are forced to w_radius=0."
fi
echo "    soft:   w_shell ${MG_W_SHELL}, dead zone ${MG_SHELL_DEAD_ZONE}," \
     "quantile bins ${MG_SHELL_QUANTILES} (0=full rank pairing)" \
     "fractional, eps ${MG_SHELL_EPS} Mpc/h"
if [ "${MG_W_SHELL%%.*}" != "0" ] || [ "${MG_W_SHELL#*.}" != "0" ]; then
echo "    NOTE: the SOFT shell. Sorts this host's candidate and target radii and"
echo "          pairs them by RANK -- the radius DISTRIBUTION, no addresses. It"
echo "          is w_radius's post-mortem: that address has an out-of-sample R2"
echo "          of 0.022, so all_blocks_selfshell memorised it (train err 2.86"
echo "          -> 1.32, held out 2.68 -> 2.81) and paid 55% of the abundance"
echo "          (host subhalos 367 -> 167 vs HR 374). Only the multiplicative"
echo "          part transferred, and this asks for that alone."
if [ "$MG_CENTRE_MODE" = "self" ]; then
echo "    WARNING: --centre-mode self PINS the radii this term redistributes,"
echo "             exactly as it does w_radius. Use MG_CENTRE_MODE=self_tan."
fi
echo "          Read shell_resid (-> 0), NOT radius_err_radii: a perfect"
echo "          distribution match still leaves 79%/85% of the per-set address"
echo "          error standing, so the address metric barely moves on a win."
fi
echo "    guards: low_k w=$MG_W_LOW, HIGH-K HINGE w=$MG_W_HIGHK above k=$MG_K_SPLIT"
if [ "$MG_HIGHK_BINS" -gt 0 ] 2>/dev/null; then
    echo "            high-k BANDED: $MG_HIGHK_BINS log bins, k_max ${MG_HIGHK_K_MAX}" \
         "(0=cube corner), tol $MG_HIGHK_TOL, two-sided $MG_HIGHK_TWO_SIDED," \
         "reduce $MG_HIGHK_REDUCE"
    echo "            section 11.6: the excess is at k=4-8, not Nyquist -- the"
    echo "            scalar averages self's +3.4x there with a 9x deficit above"
    echo "            k=10 and calls it 1.59."
fi
echo "            VELOCITY high-k w=$MG_W_VEL_HIGHK (0 = measured, not charged)," \
     "tol $MG_VEL_HIGHK_TOL, verdict floor $MG_VEL_HIGHK_MIN"
if [ "${MG_W_VEL_HIGHK%%.*}" != "0" ] 2>/dev/null; then
    echo "    NOTE: the velocity term is ON. Measured on held-out set9, FROZEN SR2"
    echo "          carries 1.02x HR of velocity power above k_split -- it has this"
    echo "          right -- and all four finished arms ended at 0.034-0.053x, a"
    echo "          19-30x collapse with no term in the loss. vel_rms_ratio is one"
    echo "          global std and read 0.71 throughout, so it could not see it."
fi
echo "    verdict: high-k max $MG_HIGHK_MAX train / $MG_HIGHK_MAX_HOLDOUT held out" \
     "(0 = train only, as all four finished arms were judged)"
if [ "$MG_UNSUP_TILES_PER_BOX" -gt 0 ] 2>/dev/null; then
    echo "    BOX-WIDE high-k support: $MG_UNSUP_TILES_PER_BOX unsup tiles/box," \
         "$MG_UNSUP_TILES_PER_STEP/step, w=$MG_W_HIGHK_UNSUP, verdict max" \
         "$MG_UNSUP_HIGHK_MAX held out"
    echo "            selfvel-arm-failed-the-gate: the supervised hinge held 0.56x"
    echo "            on train yet 3.9x held out -- its support is 6.25% of the box."
    echo "            This charges the banded hinge on random box-wide tiles."
fi
if awk "BEGIN{exit !($MG_W_ADV > 0)}"; then
    echo "    HR CRITIC ON: w_adv=$MG_W_ADV (warmup $MG_ADV_WARMUP, ramp" \
         "$MG_ADV_RAMP), n_critic $MG_N_CRITIC, R1 gamma $MG_CRITIC_R1_GAMMA," \
         "global_pool $MG_CRITIC_GLOBAL_POOL"
    echo "            A REGULARISER on the gather loss, not a replacement. It sees"
    echo "            the 6-channel HIGH-PASS (disp AND vel) of cand vs HR, so it"
    echo "            charges for the field-realism defects the gate found (11.6):"
    echo "            velocity power collapsed 19-30x and small-scale placement went"
    echo "            to the wrong spots -- both invisible to every moment term."
fi
echo "    NOTE: the high-k hinge is ON here and was ABSENT from the free field,"
echo "          which ran to 5.50x HR. A generator rewrites every site of every"
echo "          box, so that is corruption rather than a caveat."
echo "    NOTE: the verdict this job prints is FEASIBILITY ONLY -- every number"
echo "          in it is a member-set surrogate this module computes about"
echo "          itself. The gate is whole-box Rockstar on the HELD-OUT pool."

python -u scripts/features/finetune_member_gather.py \
    --config "$DIRECT_CFG" \
    --train-boxes "$MG_TRAIN_BOXES" \
    --holdout-boxes "$MG_HOLDOUT_BOXES" \
    --holdout-hosts "$MG_HOLDOUT_HOSTS" \
    --n-tiles "$MG_N_TILES" \
    --max-hosts-per-box "$MG_MAX_HOSTS_PER_BOX" \
    --min-log-mvir "$MG_MIN_LOG_MVIR" \
    --rung "$MG_RUNG" \
    --lr-scale "$MG_LR_SCALE" \
    --steps "$MG_STEPS" \
    --hosts-per-step "$MG_HOSTS_PER_STEP" \
    --sets-per-step "$MG_SETS_PER_STEP" \
    --clip "$MG_CLIP" \
    --eval-every "$MG_EVAL_EVERY" \
    --gate-at-steps "$MG_GATE_AT_STEPS" \
    --w-virial "$MG_W_VIRIAL" \
    --w-bound "$MG_W_BOUND" \
    --w-d6 "$MG_W_D6" \
    --w-rrms "$MG_W_RRMS" \
    --w-sigmav "$MG_W_SIGMAV" \
    --w-centre "$MG_W_CENTRE" \
    --w-host-sets "$MG_W_HOST_SETS" \
    --host-min-num-p "$MG_HOST_MIN_NUM_P" \
    --host-max-sets "$MG_HOST_MAX_SETS" \
    --host-sets-per-step "$MG_HOST_SETS_PER_STEP" \
    --min-num-p "$MG_MIN_NUM_P" \
    --cond-channels "$MG_COND_CHANNELS" \
    --cond-mode "$MG_COND_MODE" \
    --cond-min-hr-num-p "$MG_COND_MIN_HR_NUM_P" \
    --cond-r-max-rvir "$MG_COND_R_MAX_RVIR" \
    --cond-log-mhost-ref "$MG_COND_LOG_MHOST_REF" \
    --cond-log-mhost-scale "$MG_COND_LOG_MHOST_SCALE" \
    --cond-root "$MG_COND_ROOT" \
    --cond-lr "$MG_COND_LR" \
    --population "$MG_POPULATION" \
    --max-num-p "$MG_MAX_NUM_P" \
    --set-select "$MG_SET_SELECT" \
    --env-select "$MG_ENV_SELECT" \
    --env-host-floor "$MG_ENV_HOST_FLOOR" \
    --min-purity "$MG_MIN_PURITY" \
    --min-live-frac "$MG_MIN_LIVE_FRAC" \
    --stats-members "$MG_STATS_MEMBERS" \
    --softening "$MG_SOFTENING" \
    --bound-tau "$MG_BOUND_TAU" \
    --bg-k "$MG_BG_K" \
    --bg-radius "$MG_BG_RADIUS" \
    --pot-chunk "$MG_POT_CHUNK" \
    --pot-max-elems "$MG_POT_MAX_ELEMS" \
    --max-set-particles "$MG_MAX_SET_PARTICLES" \
    --bound-penalty "$MG_BOUND_PENALTY" \
    $MG_TERM_NORM_FLAG \
    --centre-mode "$MG_CENTRE_MODE" \
    --centre-dead-zone "$MG_CENTRE_DEAD_ZONE" \
    --centre-huber-radii "$MG_CENTRE_HUBER" \
    --w-radius "$MG_W_RADIUS" \
    --w-temp "$MG_W_TEMP" --temp-max-x "$MG_TEMP_MAX_X" \
    --temp-queries "$MG_TEMP_QUERIES" \
    $([ "$MG_TEMP_MEASURE" = "1" ] && echo --temp-measure) \
    --w-isolation "$MG_W_ISOLATION" \
    --iso-radius-frac "$MG_ISO_RADIUS_FRAC" \
    --iso-softness "$MG_ISO_SOFTNESS" \
    --iso-min-count "$MG_ISO_MIN_COUNT" \
    --radius-dead-zone "$MG_RADIUS_DEAD_ZONE" \
    --w-shell "$MG_W_SHELL" \
    --shell-dead-zone "$MG_SHELL_DEAD_ZONE" \
    --shell-quantiles "$MG_SHELL_QUANTILES" \
    --noise-draws "$MG_NOISE_DRAWS" \
    --noise-mode "$MG_NOISE_MODE" \
    --shell-noise-draws "$MG_SHELL_NOISE_DRAWS" \
    $MG_FRESH_NOISE_FLAG \
    --shell-eps-mpc-h "$MG_SHELL_EPS" \
    --radius-huber-radii "$MG_RADIUS_HUBER" \
    --w-low "$MG_W_LOW" \
    --w-highk "$MG_W_HIGHK" \
    --k-split "$MG_K_SPLIT" \
    --low-k-max "$MG_LOW_K_MAX" \
    --highk-max "$MG_HIGHK_MAX" \
    --vel-rms-tol "$MG_VEL_RMS_TOL" \
    --highk-bins "$MG_HIGHK_BINS" \
    --highk-k-max "$MG_HIGHK_K_MAX" \
    --highk-tol "$MG_HIGHK_TOL" \
    --highk-reduce "$MG_HIGHK_REDUCE" \
    $MG_HIGHK_TWO_SIDED_FLAG \
    --highk-max-holdout "$MG_HIGHK_MAX_HOLDOUT" \
    --w-vel-highk "$MG_W_VEL_HIGHK" \
    --vel-highk-tol "$MG_VEL_HIGHK_TOL" \
    --vel-highk-min "$MG_VEL_HIGHK_MIN" \
    --unsup-tiles-per-box "$MG_UNSUP_TILES_PER_BOX" \
    --unsup-tiles-per-step "$MG_UNSUP_TILES_PER_STEP" \
    --w-highk-unsup "$MG_W_HIGHK_UNSUP" \
    --unsup-highk-max "$MG_UNSUP_HIGHK_MAX" \
    --w-knn-unsup "$MG_W_KNN_UNSUP" \
    --knn-ks $MG_KNN_KS \
    --knn-queries "$MG_KNN_QUERIES" \
    --knn-quantiles "$MG_KNN_QUANTILES" \
    --knn-dead-zone "$MG_KNN_DEAD_ZONE" \
    --knn-buffer-cells "$MG_KNN_BUFFER_CELLS" \
    --knn-nbr-margin-cells "$MG_KNN_NBR_MARGIN_CELLS" \
    --knn-centre "$MG_KNN_CENTRE" \
    --knn-phase-lambda "$MG_KNN_PHASE_LAMBDA" \
    ${MG_KNN_CLASSES:+--knn-classes $MG_KNN_CLASSES} \
    --knn-strata "$MG_KNN_STRATA" \
    --knn-strata-rank "$MG_KNN_STRATA_RANK" \
    ${MG_KNN_STRATA_CHARGE:+--knn-strata-charge $MG_KNN_STRATA_CHARGE} \
    $( [ "$MG_KNN_CLASSES_DIAG_ONLY" = "1" ] && echo --knn-classes-diagnostic-only ) \
    $( [ "$MG_KNN_ONE_SIDED" = "1" ] && echo --knn-one-sided ) \
    --soft-lambda "$MG_SOFT_LAMBDA" \
    --soft-prior "$MG_SOFT_PRIOR" \
    --soft-factor "$MG_SOFT_FACTOR" \
    --soft-pool-ratio "$MG_SOFT_POOL_RATIO" \
    --soft-cell-dilate "$MG_SOFT_CELL_DILATE" \
    --soft-prior-eps "$MG_SOFT_PRIOR_EPS" \
    --soft-bg-floor "$MG_SOFT_BG_FLOOR" \
    --soft-steps "$MG_SOFT_STEPS" \
    --soft-mass-steps "$MG_SOFT_MASS_STEPS" \
    --soft-mass-eta "$MG_SOFT_MASS_ETA" \
    --soft-mass-step-cap "$MG_SOFT_MASS_STEP_CAP" \
    --soft-sigma "$MG_SOFT_SIGMA" \
    --soft-sigma-scale "$MG_SOFT_SIGMA_SCALE" \
    --relocate "$MG_RELOCATE" \
    --relocate-dirs "$MG_RELOCATE_DIRS" \
    --relocate-eps "$MG_RELOCATE_EPS" \
    --relocate-oversample "$MG_RELOCATE_OVERSAMPLE" \
    --relocate-refine-iters "$MG_RELOCATE_REFINE_ITERS" \
    --relocate-w-vel "$MG_RELOCATE_W_VEL" \
    --relocate-w-radius "$MG_RELOCATE_W_RADIUS" \
    --w-adv "$MG_W_ADV" \
    --adv-warmup-steps "$MG_ADV_WARMUP" \
    --adv-ramp-steps "$MG_ADV_RAMP" \
    --n-critic "$MG_N_CRITIC" \
    --critic-lr "$MG_CRITIC_LR" \
    --critic-width "$MG_CRITIC_WIDTH" \
    --critic-layers "$MG_CRITIC_LAYERS" \
    $MG_CRITIC_GLOBAL_POOL_FLAG \
    --critic-r1-gamma "$MG_CRITIC_R1_GAMMA" \
    --critic-r1-interval "$MG_CRITIC_R1_INTERVAL" \
    --critic-norm-fit-tiles "$MG_CRITIC_NORM_FIT_TILES" \
    --label "$MG_LABEL" \
    --seed "$MG_SEED" \
    "${EXTRA[@]}" \
    ${DEVICE:+--device "$DEVICE"}

echo "=== result: $DMSR_REWARD_ROOT/member_gather/${MG_RUNG}${MG_LABEL}/"
echo "    pool.json      the shakeout: hosts, sets, and the reachable reference"
echo "    metrics.jsonl  one row per eval, train AND holdout, per host"
echo "    summary.json   config, history, VERDICT (feasibility only)"
echo "    tuned.pt       {'model': full state dict} -- loadable by the gate"
echo "    wandb: group 'member_gather', run 'member_gather-${MG_RUNG}${MG_LABEL}'."
echo "           metrics.jsonl stays the source of truth and is mirrored, so a"
echo "           wandb outage never loses a number. The VERDICT and the final"
echo "           per-host table land in the run SUMMARY, not just the charts."
