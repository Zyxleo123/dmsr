#!/bin/bash
# Can a GENERATOR reach the member-gather objective, and does it generalise?
#
#   shakeout (GPU, pool only) --afterok--> fine-tune (GPU)
#
# docs/sr2_member_gather.md settled that the objective is satisfiable: a free
# field recovered 72 of 154 supervised subhalos as bound halos real Rockstar
# matches, against a frozen 3 and a measured ceiling of 151. Its section 7 then
# names what that leaves open, and this chain is items 3 and 4:
#
#   item 3  a free field is not a generator -- 6.3M unconstrained parameters
#           that SEE the true member sets say what the objective PERMITS, not
#           what one learned operator applied at every site can reach.
#   item 4  one host, one box, in-sample -- no generalisation claim is available
#           from any run in that document.
#
# So the training pool is many hosts across many boxes and the score that
# matters is on boxes never trained on and never used to develop this line.
#
# WHY THESE BOXES (configs/reward/sr2_direct_finetune.yaml `split`):
#   train   set3-set7  SR2's paired training was set0-2 ONLY, and these five
#                      were never used to develop the gather line.
#   holdout set9,set10 never seen by SR2, never seen by us.
#   anchor  set8       DELIBERATELY EXCLUDED from both. It is where the ceiling
#                      (151/154) and the oracle (72/154) were measured, so it is
#                      development-contaminated -- but it is also the only box
#                      with a calibrated ceiling, which makes it the right place
#                      to ask "how far short of the free field did the generator
#                      fall on the IDENTICAL problem". Run it as a reported
#                      anchor, never as evidence of generalisation.
#   sealed  set13-15   never touched.
#
# PREREQUISITE: every box needs its HR owner array -- membership IS the
# supervision. Build them first, once, with:
#     bash scripts/slurm/submit_owner_arrays.sh
# A box without one is skipped with a message rather than failing the job.
#
# This submitter ONLY calls sbatch. Configuration goes into ONE timestamped env
# file passed as a POSITIONAL argument -- never `sbatch --export`, which on this
# cluster sets SLURM_GET_USER_ENV=1 and gets the job requeued and HELD.
#
#   bash scripts/slurm/submit_member_gather_train.sh
#   MG_POOL_ONLY=1 bash scripts/slurm/submit_member_gather_train.sh   # shakeout
#   MG_STEPS=20000 MG_LABEL=_long bash scripts/slurm/submit_member_gather_train.sh
#   DRY=1 bash scripts/slurm/submit_member_gather_train.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
SHAKEOUT_ONLY="${SHAKEOUT_ONLY:-0}"
SKIP_SHAKEOUT="${SKIP_SHAKEOUT:-0}"
# --- the SLURM resource request ----------------------------------------------
# Passed as sbatch CLI flags, which override the .sbatch's own #SBATCH lines.
# These are NOT `--export` and do not trip SLURM_GET_USER_ENV, so the job is not
# requeued-and-held the way an export list would leave it.
#
# MG_MEM. The default in the batch script is 32G, sized from a MEASURED 10.1 GB
# peak host RSS (56 hosts, 2 noise draws; job 46659). It was 96G for months on
# no evidence, and on 2026-09-09 that request fit no available node on
# `general` and stalled the whole shell-lever set. Raise it only against a fresh
# SHAKEOUT_ONLY=1 measurement, not on a hunch.
#
# MG_TIME. 8000 steps ran 10.6 h against the 12 h limit, so the margin is thin
# and any arm that adds work per step can hit the wall. --shell-noise-draws adds
# a forward pass per step; give those arms more.
MG_MEM="${MG_MEM:-}"
MG_TIME="${MG_TIME:-}"
# MG_EXCLUDE. Pins the GPU class by excluding nodes (a multi-node --nodelist
# forces the node count). The soft arms' GPU peak was never recorded and they
# ran on 48 GB A6000s, so exclude the 24 GB A5000 node (gpu28) unless measured.
MG_EXCLUDE="${MG_EXCLUDE:-}"

# --- the split ---------------------------------------------------------------
MG_TRAIN_BOXES="${MG_TRAIN_BOXES:-set3 set4 set5 set6 set7}"
MG_HOLDOUT_BOXES="${MG_HOLDOUT_BOXES:-set9 set10}"
MG_HOLDOUT_HOSTS="${MG_HOLDOUT_HOSTS:-}"
# --- the pool ----------------------------------------------------------------
# 4 tiles is the oracle's geometry and should stay there: the measured ceiling at
# 4 tiles is 151/154, so the ~5.4% of members starting outside the trained tiles
# cost THREE targets. Widening buys almost nothing and halves the pool.
MG_N_TILES="${MG_N_TILES:-4}"
MG_MAX_HOSTS_PER_BOX="${MG_MAX_HOSTS_PER_BOX:-8}"
MG_MIN_LOG_MVIR="${MG_MIN_LOG_MVIR:-13.5}"
# --- optimisation ------------------------------------------------------------
# DEFAULT CHANGED 2026-09-11, fine -> all_blocks, on gate evidence. The
# held-out set9 gate of the `fine` sweep put BOTH >=200p arms BELOW the frozen
# control (supervised-target rate 1.86% / 1.51% against frozen's 2.66%, and 4-5
# subhalos within R_vir against frozen's 18). The one prior halo-finder win on
# this line (20 -> 366 subhalos) was rung `all_blocks`. `fine` is too shallow to
# clear the frozen baseline, so it is no longer the default.
# Runs recorded before this date used `fine` unless their config says otherwise.
MG_RUNG="${MG_RUNG:-all_blocks}"
MG_LR_SCALE="${MG_LR_SCALE:-10.0}"
MG_STEPS="${MG_STEPS:-8000}"
MG_HOSTS_PER_STEP="${MG_HOSTS_PER_STEP:-2}"
MG_CLIP="${MG_CLIP:-1.0}"
MG_EVAL_EVERY="${MG_EVAL_EVERY:-250}"
# --- the objective -----------------------------------------------------------
MG_W_VIRIAL="${MG_W_VIRIAL:-1.0}"
MG_W_BOUND="${MG_W_BOUND:-1.0}"
MG_W_D6="${MG_W_D6:-1.0}"
MG_W_RRMS="${MG_W_RRMS:-0.3}"
MG_W_SIGMAV="${MG_W_SIGMAV:-0.3}"
MG_W_CENTRE="${MG_W_CENTRE:-1.0}"
# The three arms of docs/sr2_member_gather_training.md section 9.4, all off by
# default -- `full` + 0 + 0 IS the objective that scored 72/154. The two shaping
# knobs were read by the batch script from 2026-08-23 but were NEVER written
# into the env file, so section 12's softened-centre recipe silently ran the
# default; they are here now because a knob the submitter drops is worse than
# one that does not exist.
# The loss BUDGET, section 10. Off by default, because every number recorded so
# far was measured with the unnormalised terms and the capped hinge.
#   MG_BOUND_PENALTY=log  charges a boundness deficit in log space instead of
#                         [1-x/ref]_+^2, which is capped at 1 by construction.
#   MG_TERM_NORM=1        divides each term by its frozen-field value so the
#                         declared weights are the actual budget (and removes
#                         d6's 88% head start, which is the same fact).
MG_BOUND_PENALTY="${MG_BOUND_PENALTY:-hinge}"
MG_TERM_NORM="${MG_TERM_NORM:-0}"
MG_CENTRE_MODE="${MG_CENTRE_MODE:-full}"
MG_CENTRE_DEAD_ZONE="${MG_CENTRE_DEAD_ZONE:-0.0}"
MG_CENTRE_HUBER="${MG_CENTRE_HUBER:-0.0}"
# --- the SHELL arm: the right distance from the host, any direction ----------
# The `self` arm's subhalos are radially collapsed. MEASURED on held-out set9
# (subhalo_radial/set9/summary.json), restricted to the edited region:
#
#            r / host R_vir      <0.25 R_vir     0.75-1.0 R_vir
#   HR             1.13               593             3846
#   self           0.35              6496              932
#
# Too many in the middle, not enough anywhere else -- an 11x excess in the
# innermost bin against a 4x deficit at the virial radius.
#
# MG_W_RADIUS charges | |x_bar - x_host| - |x_target - x_host| |: the set's
# host-centric RADIUS, with the direction left free. Why only the radius, and
# why it is not a weaker version of the centre term: over the 7,560-set pool in
# centre_offset/pool/offsets.json the radius error is 60% of the whole centre
# offset by magnitude and it is the half that is a RULE -- 58.3% of frozen sets
# sit too close, median error 2.80 search radii (p90 8.18) -- while the
# transverse half has no direction to predict from and a fit on (d_host, num_p,
# M_host) explains 11.5% of it out of sample. The shell is the part a conv can
# be asked for. MG_RADIUS_DEAD_ZONE is the "not too strict" knob: one search
# radius of slack, which still charges 79% of frozen sets.
#
# IT CONFLICTS WITH MG_CENTRE_MODE=self, which is the arm it is meant to extend.
# `self` anchors the whole position, radius included, so the radial step this
# term asks for is a step `self` charges for; both quadratic, they settle at
# delta* = |dr| w_radius/(w_centre + w_radius) -- exactly HALF the correction at
# equal weights. `self_tan` is `self` with the radial direction projected out:
# the anchor keeps the two transverse coordinates (the ones the shell term says
# nothing about) and the shell term owns the radial one. Orthogonal by
# construction, and the whole reason the mode exists.
#
#   the arm:  MG_CENTRE_MODE=self_tan MG_W_RADIUS=1.0 MG_LABEL=_selfshell \
#             bash scripts/slurm/submit_member_gather_train.sh
#   the control it must beat is all_blocks_self (same objective, w_radius 0).
# All OFF by default: MG_W_RADIUS=0 is byte-identical to the finished arms.
MG_W_RADIUS="${MG_W_RADIUS:-0.0}"
MG_RADIUS_DEAD_ZONE="${MG_RADIUS_DEAD_ZONE:-1.0}"
MG_RADIUS_HUBER="${MG_RADIUS_HUBER:-0.0}"
# --- the SOFT shell, docs/sr2_member_gather_training.md section 14.6 ----------
# MG_W_RADIUS ran, as all_blocks_selfshell, and the result is why this exists.
# It charges each set against ITS OWN target radius -- a per-object ADDRESS --
# and on the 7,560-set pool a linear probe on (log d_host, log num_p, log Mvir,
# log n_live) predicts that address with an out-of-sample R2 of 0.022 in the
# term's own units. Section 14.1's "the radius is the half that is a rule" is
# true of the AGGREGATE (58.3% of sets sit too close, median ratio 0.974) and
# false per set; those are different claims and w_radius was built on the wrong
# one. The trajectory is that stated in metrics: train radius_err 2.860 ->
# 1.318 (memorised), held out 2.678 -> 2.808 (the address is not there),
# held-out radius_ratio 0.9744 -> 0.9872 (the multiplicative part, and ONLY it,
# transferred). It cost 55% of the abundance: host subhalos 367 -> 167 vs 374.
#
# MG_W_SHELL asks for that transferring component alone. Sort this host's
# candidate radii, sort its target radii, pair by RANK -- the 1-D optimal
# transport map, exact and O(N log N) -- and charge the log ratio. No set is
# ever named against its own target. Measured against the identity pairing
# (train / holdout): median |log ratio| 0.164/0.143 -> 0.079/0.064, while the
# SIGNED median, the inward bias that is the whole signal, survives intact at
# -0.026/-0.022 -> -0.041/-0.026. The monotone map removes 79% of the mean
# square where a per-host constant rescale would remove 1.9%, so what it fixes
# is the SHAPE of the radial distribution, not its mean.
#
# Read shell_resid, NOT radius_err_radii. A perfect distribution match still
# leaves 79% (train) / 85% (holdout) of the per-set address error standing, so
# the address metric barely moves even on a total win -- gate on it and you
# will call a success a failure.
#
# Same MG_CENTRE_MODE=self conflict as w_radius, for the same reason: `self`
# pins the radii this term redistributes. Pair it with self_tan.
#
#   the arm:  MG_CENTRE_MODE=self_tan MG_W_SHELL=1.0 MG_LABEL=_softshell \
#             bash scripts/slurm/submit_member_gather_train.sh
#   the controls are all_blocks_self (w_shell 0) and all_blocks_selfshell.
# All OFF by default: MG_W_SHELL=0 leaves the loss and its gradient unchanged.
# --- SOFT MEMBERSHIP: which particles, as a latent variable ------------------
# OFF by default; MG_SOFT_MEMBERSHIP=0 leaves the loss and its gradient exactly
# as every arm on disk ran them.
#
# The standing defect this targets. Every term here is a statistic of a particle
# collection the loss cannot choose: HR's owner array names it, once, before the
# run. When SR2's material for a subhalo is not the material HR used, the loss
# charges the generator for failing to bind particles that were never going to
# be there and rewards it for dragging in the wrong ones. The gate-passing arm
# works anyway (20 -> 366 held-out subhalos against HR's 369) and its remaining
# defects are placement ones -- radially collapsed satellites, host damage that
# is ~88% re-parenting -- which is what a wrong-material objective would look
# like from the outside.
#
# The relaxation. HR keeps supplying the object's REGION (a coarse Lagrangian
# prior: n_c of an LR cell's 512 sites belong to this subhalo, so EVERY SR
# particle of that cell starts at membership n_c/512, the original members
# included and not privileged) and its PHYSICS (the reference, untouched). The
# CURRENT SR field decides which particles participate, by a 6-D phase-space
# fit -- position AND velocity, the same quantity Rockstar groups on. Two
# constraints make the solve honest and both are load-bearing:
#   * a per-particle budget of one, shared with a background/host category, so
#     two candidates cannot spend the same convenient particles; and
#   * a per-candidate mass target, so a candidate cannot discard everything and
#     keep a tiny cold core that scores perfectly on every moment.
#
# THE RISK, named before it is run: a coarse window is exactly the Gaussian
# window that already failed on this line (docs/sr2_gather_finetune.md section
# 6 -- its statistics described the surrounding host material, not the object).
# That is why the prior is the SUPPORT and never the membership: one LR cell is
# 1.5625 Mpc/h a side against a 0.15 Mpc/h search radius. And it is why the
# ladder below runs the coarse-at-large-lambda control, which is the arm that
# says whether the window alone is already destructive.
#
# THE LADDER (run the controls, or the arm proves nothing):
#   hard, lambda 1e6   reproduces the arm on disk -- a correctness check
#   coarse, lambda 1e6 the window ALONE, membership frozen at the prior
#   hard, lambda 1     the conservative relaxation: HR's mask, opened
#   coarse, lambda 1   the proposal as written
# Use submit_soft_membership_ladder.sh, which submits all four plus the control.
#
# WATCH soft_mass_ratio (must sit at ~1.0) and soft_hr_fraction (1.0 is the hard
# arm; a collapse toward 0 is the relaxation walking off the object).
#
# COST: the pool is (1 + ratio) times the member count against an O(pool^2)
# estimator, so ratio 2 is ~9x the hard arm's pair bill. MG_MAX_SET_PARTICLES
# bounds it and the ladder sets one.
MG_SOFT_MEMBERSHIP="${MG_SOFT_MEMBERSHIP:-0}"
MG_SOFT_LAMBDA="${MG_SOFT_LAMBDA:-1.0}"
MG_SOFT_PRIOR="${MG_SOFT_PRIOR:-coarse}"
MG_SOFT_FACTOR="${MG_SOFT_FACTOR:-8}"
MG_SOFT_POOL_RATIO="${MG_SOFT_POOL_RATIO:-2.0}"
MG_SOFT_CELL_DILATE="${MG_SOFT_CELL_DILATE:-0}"
MG_SOFT_PRIOR_EPS="${MG_SOFT_PRIOR_EPS:-1e-3}"
MG_SOFT_BG_FLOOR="${MG_SOFT_BG_FLOOR:-1e-3}"
MG_SOFT_STEPS="${MG_SOFT_STEPS:-3}"
MG_SOFT_MASS_STEPS="${MG_SOFT_MASS_STEPS:-10}"
MG_SOFT_MASS_ETA="${MG_SOFT_MASS_ETA:-1.0}"
MG_SOFT_MASS_STEP_CAP="${MG_SOFT_MASS_STEP_CAP:-20.0}"
MG_SOFT_SIGMA="${MG_SOFT_SIGMA:-adaptive}"
MG_SOFT_SIGMA_SCALE="${MG_SOFT_SIGMA_SCALE:-1.0}"

# --- the host as a background component (docs/sr2_host_profile.md) ----------
# The radial-placement arm. OFF by default: MG_SOFT_HOST_PROFILE=off is
# byte-identical to every soft arm on disk. See the batch script for what the
# two sources are and why beta is not comparable between them.
MG_SOFT_HOST_PROFILE="${MG_SOFT_HOST_PROFILE:-off}"
MG_SOFT_HOST_WEIGHT="${MG_SOFT_HOST_WEIGHT:-0.0}"
MG_SOFT_HOST_CONC="${MG_SOFT_HOST_CONC:-5.0}"
MG_SOFT_HOST_BINS="${MG_SOFT_HOST_BINS:-24}"
MG_SOFT_HOST_R_MAX_RVIR="${MG_SOFT_HOST_R_MAX_RVIR:-4.0}"
MG_SOFT_RADIAL_WEIGHT="${MG_SOFT_RADIAL_WEIGHT:-0.0}"
MG_SOFT_RADIAL_WIDTH="${MG_SOFT_RADIAL_WIDTH:-0.2}"
MG_LR_HOST_TABLE="${MG_LR_HOST_TABLE:-}"
MG_LR_HOST_R_MAX_RVIR="${MG_LR_HOST_R_MAX_RVIR:-2.0}"
MG_W_SHELL="${MG_W_SHELL:-0.0}"
# 0.02, not radius_dead_zone's 1.0 rescaled: as a FRACTIONAL error 0.02 charges
# 81%/78% of sets against a sorted median residual of 0.079/0.064, while 0.05
# charges only 61%/55% and eats most of a signal half the size of the one the
# radius term's dead zone was set against.
MG_SHELL_DEAD_ZONE="${MG_SHELL_DEAD_ZONE:-0.02}"
# --- the three 2026-09-09 levers, all OFF by default -------------------------
# WHY they exist. MEASURED on the w_shell sweep (2026-09-08), raw term_raw_shell
# step 0 -> 8000:
#     w=1   train 0.0507 -> 0.0589 (worse)   holdout 0.0539 -> 0.0844 (worse)
#     w=10  train 0.0507 -> 0.0205 (2.5x)    holdout 0.0539 -> 0.0599 (worse)
#     w=30  train 0.0507 -> 0.0069 (7.3x)    holdout 0.0539 -> 0.0390 (1.4x)
# Only w=30 improves out of sample, at a 5.6x train/holdout gap. It is NOT a
# sampling artifact (MG_SETS_PER_STEP=0 pools every set on both sides) and it is
# NOT that the target is noise: split-half reliability of a host's own radial
# quantile curve is 0.84-0.94 on HR (set9, 101k subhalos). But host MASS explains
# only R^2 0.01-0.02 of it, so the signal is real and not in the obvious feature.
#
#   (a) MG_NOISE_MODE=rotate   the fixed (LR, noise) pair is constant for 8000
#                              steps, so the noise is an input channel with real
#                              per-tile detail and zero generalisable content --
#                              a memorisation channel. Rotating removes it.
#   (b) MG_SHELL_NOISE_DRAWS   charge the MEAN reduced curve over draws, not the
#                              mean of the per-draw charges: a distribution no
#                              single realisation can fake.
#   (c) MG_SHELL_QUANTILES     the full pairing is one degree of freedom per set
#                              (~134 for a big host). Q keeps the low-order shape
#                              and drops the rank-by-rank wiggle.
# MG_FRESH_NOISE_EVAL is the MEASUREMENT that arbitrates: it reserves one draw no
# step ever sees and reports train_fresh / holdout_fresh, giving the 2x2 over
# (seen box, seen noise). Cheap and worth setting on every arm below.
MG_SHELL_QUANTILES="${MG_SHELL_QUANTILES:-0}"
MG_NOISE_DRAWS="${MG_NOISE_DRAWS:-1}"
MG_NOISE_MODE="${MG_NOISE_MODE:-fixed}"
MG_FRESH_NOISE_EVAL="${MG_FRESH_NOISE_EVAL:-0}"
MG_SHELL_NOISE_DRAWS="${MG_SHELL_NOISE_DRAWS:-0}"
MG_SHELL_EPS="${MG_SHELL_EPS:-0.01}"
# --- the host preservation guard, docs/sr2_member_gather_training.md ----------
# The advisor's fix for the collateral host damage (resolved hosts 3028 base ->
# 2708 tuned, HR wants 3775; gather-holdout-rockstar-gate). The SAME member
# loss on the resolved HOST halos homing in a cluster's tiles, referenced on the
# FROZEN field -- so it is a preservation guard that starts at ~0 and fires only
# as a host comes apart, not a second objective. All OFF by default:
# MG_W_HOST_SETS=0 is byte-identical to the finished arms.
MG_W_HOST_SETS="${MG_W_HOST_SETS:-0.0}"
MG_HOST_MIN_NUM_P="${MG_HOST_MIN_NUM_P:-200}"
MG_HOST_MAX_SETS="${MG_HOST_MAX_SETS:-256}"
MG_HOST_SETS_PER_STEP="${MG_HOST_SETS_PER_STEP:-64}"
MG_MIN_NUM_P="${MG_MIN_NUM_P:-200}"
# --- the LR-host conditioning channel ----------------------------------------
# EMPTY is the unconditioned control. This is the ONE lever in this file that
# changes what the generator can see rather than what it is charged for: the
# radial profile is flat because a local operator cannot know the host centre or
# scale, and no reweighting of the existing terms can tell it.
#
# Requires the fields, built once per box by a CPU job:
#     bash scripts/slurm/submit_lr_host_channel.sh
# The trainer refuses to start if a box in --train-boxes or --holdout-boxes has
# no field, or if the cached field was built under different settings.
#
#   the arm: MG_COND_CHANNELS=r_over_rvir,log_mhost,valid \
#            MG_LABEL=_cond bash scripts/slurm/submit_member_gather_train.sh
MG_COND_CHANNELS="${MG_COND_CHANNELS:-}"
MG_COND_MODE="${MG_COND_MODE:-derived}"
MG_COND_MIN_HR_NUM_P="${MG_COND_MIN_HR_NUM_P:-20000}"
MG_COND_R_MAX_RVIR="${MG_COND_R_MAX_RVIR:-3.0}"
MG_COND_LOG_MHOST_REF="${MG_COND_LOG_MHOST_REF:-13.0}"
MG_COND_LOG_MHOST_SCALE="${MG_COND_LOG_MHOST_SCALE:-1.0}"
MG_COND_ROOT="${MG_COND_ROOT:-}"
MG_COND_LR="${MG_COND_LR:-1e-5}"
# --- the SMALL-HALO arm: supervise on size, not on the parent split ----------
# Every arm so far supervised `parent_ids >= 0`. That encodes an assumption the
# held-out Rockstar gate contradicts. In the edited region (32 spliced tiles,
# dilated 2 Mpc/h):
#
#                        base SR2    tuned `self`    HR truth
#   subhalos >=50p          4,541          9,187      12,367
#   hosts    >=200p         3,028          2,708       3,775
#
# The 747-halo top-level deficit (3,775 - 3,028) is OUTSIDE hosts and no
# supervised set in any arm has ever pointed at it. MG_POPULATION=all selects on
# num_p alone and lets the parent split fall where it falls.
#
# WHETHER that deficit is a small-halo deficit or a flat one across every mass is
# measured, not assumed:  bash scripts/slurm/submit_halo_gap_census.sh
# Read its verdict before choosing MG_MAX_NUM_P -- the ceiling IS the hypothesis.
#
#   the arm:  MG_POPULATION=all MG_MAX_NUM_P=2000 MG_SET_SELECT=stratified #             MG_MIN_NUM_P=50 MG_CENTRE_MODE=self MG_W_HOST_SETS=1.0 #             MG_LABEL=_smallhalo bash scripts/slurm/submit_member_gather_train.sh
#   the control it must beat is all_blocks_selfhostguard (same objective,
#   population sub), on the SAME held-out gate.
# All OFF by default: MG_POPULATION=sub is byte-identical to the finished arms.
MG_POPULATION="${MG_POPULATION:-sub}"
MG_MAX_NUM_P="${MG_MAX_NUM_P:-0}"
MG_SET_SELECT="${MG_SET_SELECT:-largest}"

# --- the environment axis, and the isolation term ----------------------------
# The small-halo arm RAN (job 39083, 8,000 steps) and was gated on both held-out
# boxes. What it measured, set9 region scope (32 tiles, dilate 2), summed over
# every mass bin:
#
#   population  env     base    smallhalo   HR      deficit before -> after
#   top-level   near      670         575   2,819    2,149 -> 2,244
#   top-level   field   9,871       9,066  11,580    1,709 -> 2,514
#   sub         inside  2,064       3,912   4,264    2,200 ->   352
#   sub         near    1,229       1,573   5,637    4,408 -> 4,064
#   sub         field   1,248       3,316   2,466    1,218 ->  -850   (OVERSHOT)
#
# Three facts, and they point at two different fixes:
#  1. it BUILT objects -- 3,374 net -- and Rockstar filed every one of them as a
#     SUBHALO (subs +4,260, top-level -886), so the top-level deficit the arm
#     exists to close went the wrong way, 3,872 -> 4,758.
#  2. it overshot in the field: subhalos 3,316 against HR's 2,466, and 2-6x too
#     many above 200p, i.e. spurious substructure inside small field hosts.
#  3. 56% of the region's 11,698 missing objects are in `near` -- the infall
#     region -- and it closed 249 of them, 3.8%.
#
# MG_W_ISOLATION answers (1) and (2): every other term in the loss is a moment
# of the MEMBERS about their own centroid, and "top-level" is not a property of
# the members at all -- it is whether a bigger halo's R_vir contains the centre.
# So the loss scores full marks on an object a pedestal then swallows. This term
# charges the non-member mass on the object against HR's on the SAME particle
# ids, hinged above, and it is the only inter-set statement in the loss.
#
# MG_ENV_SELECT answers (3): `near` sits inside a cluster's own Lagrangian tiles,
# so it is reachable by changing the SELECTION rather than MG_N_TILES (which is
# what `field` would need). The cut runs BEFORE the cap, so the budget follows.
#
#   the arm:  MG_POPULATION=all MG_MAX_NUM_P=2000 MG_SET_SELECT=stratified \
#             MG_MIN_NUM_P=50 MG_ENV_SELECT=inside,near MG_W_ISOLATION=1.0 \
#             MG_CENTRE_MODE=self MG_W_HOST_SETS=1.0 MG_LABEL=_nearclean \
#             bash scripts/slurm/submit_member_gather_train.sh
#   the control it must beat is all_blocks_smallhalo (same objective, env any,
#   no isolation term), on the SAME held-out gate.
# All OFF by default: MG_ENV_SELECT=any and MG_W_ISOLATION=0 are byte-identical
# to the finished arms.
MG_ENV_SELECT="${MG_ENV_SELECT:-any}"
MG_ENV_HOST_FLOOR="${MG_ENV_HOST_FLOOR:-200}"
MG_W_ISOLATION="${MG_W_ISOLATION:-0.0}"
MG_ISO_RADIUS_FRAC="${MG_ISO_RADIUS_FRAC:-0.5}"
# Host-centric local temperature (host_temperature.py; placement forensics
# 2026-10-15): two-sided, velocity-only. MG_TEMP_MAX_X=0.5 charges the core only.
MG_W_TEMP="${MG_W_TEMP:-0.0}"
MG_TEMP_MAX_X="${MG_TEMP_MAX_X:-3.0}"
MG_TEMP_QUERIES="${MG_TEMP_QUERIES:-256}"
MG_TEMP_MEASURE="${MG_TEMP_MEASURE:-0}"
MG_ISO_SOFTNESS="${MG_ISO_SOFTNESS:-0.15}"
MG_ISO_MIN_COUNT="${MG_ISO_MIN_COUNT:-8.0}"

# --- the EARLY Rockstar gate -------------------------------------------------
# Steps at which the trainer freezes an IMMUTABLE tuned_step<N>.pt so a Rockstar
# gate can run CONCURRENTLY with the rest of training. Empty = none.
#
# Why step 2000 and not "wait for the run". MEASURED on the three finished arms,
# held out, the objective is essentially done by then and what continues is
# damage:
#
#   arm                       bound_frac @2000    @8000   high-k max @2000 -> @8000
#   all_blocks_smallhalo            0.0874       0.0877     1.52x -> 3.15x
#   all_blocks_selfhostguard        0.0950       0.0976     2.02x -> 4.79x
#   all_blocks_self                 0.1202       0.1395     1.92x -> 3.87x
#
# So a step-2000 gate is a real read, not a peek -- and it takes NOTHING from
# the arm: training runs to MG_STEPS regardless, the snapshot is a file copy,
# and the gate's 19 min are on other nodes. It turns a 10.3 h wait for the first
# Rockstar number into ~2.7 h.
#
# READ IT ONE-SIDED. A clear negative early (top-level still falling, subs still
# exploding) says the mechanism did not change. A null is NOT evidence of
# failure -- the remaining 6000 steps are 3-14% more objective.
MG_GATE_AT_STEPS="${MG_GATE_AT_STEPS:-}"
MG_MIN_PURITY="${MG_MIN_PURITY:-0.5}"
MG_MIN_LIVE_FRAC="${MG_MIN_LIVE_FRAC:-0.5}"
MG_STATS_MEMBERS="${MG_STATS_MEMBERS:-all}"
MG_SOFTENING="${MG_SOFTENING:-0.01}"
MG_BOUND_TAU="${MG_BOUND_TAU:-0.5}"
MG_BG_K="${MG_BG_K:-4096}"
MG_BG_RADIUS="${MG_BG_RADIUS:-4.0}"
MG_POT_CHUNK="${MG_POT_CHUNK:-2048}"
# The per-set particle cap. The sbatch has read MG_MAX_SET_PARTICLES since it was
# added, and this submitter never WROTE it into the env file -- so every run
# launched through here silently got the exact estimator, and the 2026-08-26
# `selfhostguard` arm only got its 2048 because it was hand-passed as a
# positional argument. Same class of bug as the centre knobs above: a knob the
# submitter drops is worse than one that does not exist.
#
# It is a TIME knob, not a memory one (the pair blocks are recomputed in
# backward), and it matters most for the HOST preservation guard: measured on
# job 37229, the host sets bill 1.770e+11 with a largest set of 247,745
# particles, against 4.178e+09 at a 2048 cap -- 42x. The supervised sets of a
# `--max-num-p 2000` run top out at 1,999 particles, so a 2048 cap never binds
# on them and their estimator stays EXACT while the guard gets cheap.
MG_MAX_SET_PARTICLES="${MG_MAX_SET_PARTICLES:-0}"
# --- the guards --------------------------------------------------------------
MG_W_LOW="${MG_W_LOW:-100.0}"
# Absent from every free-field run, which ended at 5.50x HR and still climbing.
MG_W_HIGHK="${MG_W_HIGHK:-10.0}"
MG_K_SPLIT="${MG_K_SPLIT:-4.0}"
MG_LOW_K_MAX="${MG_LOW_K_MAX:-0.02}"
MG_HIGHK_MAX="${MG_HIGHK_MAX:-1.5}"
MG_VEL_RMS_TOL="${MG_VEL_RMS_TOL:-0.10}"
# --- section 11.6: band-resolved high-k, and the velocity power guard ---------
# Measured on held-out set9 (job 36394): `self` is +3.4x HR at k=4-8 and 9x SHORT
# at k>10, which the single scalar averages to 1.59; and EVERY arm took velocity
# power above k_split from frozen's 1.02x HR down to 0.034-0.053x, with no term
# in the loss and no criterion in the verdict. All default to the old behaviour.
MG_HIGHK_BINS="${MG_HIGHK_BINS:-0}"
MG_HIGHK_K_MAX="${MG_HIGHK_K_MAX:-0.0}"
MG_HIGHK_TOL="${MG_HIGHK_TOL:-0.0}"
MG_HIGHK_TWO_SIDED="${MG_HIGHK_TWO_SIDED:-0}"
MG_HIGHK_REDUCE="${MG_HIGHK_REDUCE:-mean}"
MG_HIGHK_MAX_HOLDOUT="${MG_HIGHK_MAX_HOLDOUT:-0.0}"
MG_W_VEL_HIGHK="${MG_W_VEL_HIGHK:-0.0}"
MG_VEL_HIGHK_TOL="${MG_VEL_HIGHK_TOL:-0.25}"
MG_VEL_HIGHK_MIN="${MG_VEL_HIGHK_MIN:-0.0}"
# --- the box-wide (unsupervised-tile) high-k support, selfvel-arm-failed-the-gate
# The supervised hinge held 0.56x on the training tiles yet 3.9x held out because
# its support is the 40 hosts' 160 tiles -- 6.25% of the box. These charge the
# SAME banded hinge on random tiles drawn box-wide. All default OFF.
MG_UNSUP_TILES_PER_BOX="${MG_UNSUP_TILES_PER_BOX:-0}"
MG_UNSUP_TILES_PER_STEP="${MG_UNSUP_TILES_PER_STEP:-8}"
MG_W_HIGHK_UNSUP="${MG_W_HIGHK_UNSUP:-0.0}"
MG_UNSUP_HIGHK_MAX="${MG_UNSUP_HIGHK_MAX:-0.0}"
# --- the box-wide kNN-distance support (src/cosmo_sr/features/knn_cdf.py). All
# default OFF; needs MG_UNSUP_TILES_PER_BOX > 0. Run submit_knn_cdf_diagnostic.sh
# FIRST: it decides whether the statistic has signal and sets the buffer.
MG_W_KNN_UNSUP="${MG_W_KNN_UNSUP:-0.0}"
MG_KNN_KS="${MG_KNN_KS:-8 32 64 128}"
MG_KNN_QUERIES="${MG_KNN_QUERIES:-1024}"
MG_KNN_QUANTILES="${MG_KNN_QUANTILES:-64}"
MG_KNN_DEAD_ZONE="${MG_KNN_DEAD_ZONE:-0.02}"
MG_KNN_BUFFER_CELLS="${MG_KNN_BUFFER_CELLS:-0}"
MG_KNN_NBR_MARGIN_CELLS="${MG_KNN_NBR_MARGIN_CELLS:-8}"
MG_KNN_CENTRE="${MG_KNN_CENTRE:-centroid}"
MG_KNN_ONE_SIDED="${MG_KNN_ONE_SIDED:-0}"
# Round 5b: 4000 and a subhalo class list are the measured settings; 0/empty
# keeps the position-only pooled form of rounds 1-4 (which cannot rank arms).
MG_KNN_PHASE_LAMBDA="${MG_KNN_PHASE_LAMBDA:-0}"
MG_KNN_CLASSES="${MG_KNN_CLASSES:-}"
# The catalog-free strata arm (docs/sr2_knn_strata.md). Gated: see
# scripts/slurm/submit_knn_strata_arm.sh, which sets these for you and refuses
# to run without a passing verdict json.
MG_KNN_STRATA="${MG_KNN_STRATA:-0}"
MG_KNN_STRATA_RANK="${MG_KNN_STRATA_RANK:-0}"
MG_KNN_STRATA_CHARGE="${MG_KNN_STRATA_CHARGE:-}"
MG_KNN_CLASSES_DIAG_ONLY="${MG_KNN_CLASSES_DIAG_ONLY:-0}"
# --- the HR critic, docs/sr2_gather_critic.md. A REGULARISER on the gather loss,
# the general form of the two hand-crafted guards above: it sees whole HR tiles
# vs tuned tiles (6-channel high-pass, velocity included) and charges for the
# field-realism defects the gate found. All default OFF -- MG_W_ADV=0 is the
# four-arm baseline, byte-for-byte.
MG_W_ADV="${MG_W_ADV:-0.0}"
MG_ADV_WARMUP="${MG_ADV_WARMUP:-500}"
MG_ADV_RAMP="${MG_ADV_RAMP:-2000}"
MG_N_CRITIC="${MG_N_CRITIC:-1}"
MG_CRITIC_LR="${MG_CRITIC_LR:-2e-4}"
MG_CRITIC_WIDTH="${MG_CRITIC_WIDTH:-64}"
MG_CRITIC_LAYERS="${MG_CRITIC_LAYERS:-3}"
MG_CRITIC_GLOBAL_POOL="${MG_CRITIC_GLOBAL_POOL:-0}"
MG_CRITIC_R1_GAMMA="${MG_CRITIC_R1_GAMMA:-10.0}"
MG_CRITIC_R1_INTERVAL="${MG_CRITIC_R1_INTERVAL:-16}"
MG_CRITIC_NORM_FIT_TILES="${MG_CRITIC_NORM_FIT_TILES:-32}"
# --- relocation: the arm A / B / C experiment --------------------------------
# off = arm A (HR memberships, every result on this line so far); frozen = arm B
# (groups re-chosen from frozen-SR material at the frozen radius); hr = arm C
# (the same selection at HR's radius). Same targets, same setup, one scalar per
# set between B and C. scripts/slurm/submit_relocate_arms.sh runs all three.
MG_RELOCATE="${MG_RELOCATE:-off}"
MG_RELOCATE_DIRS="${MG_RELOCATE_DIRS:-8}"
MG_RELOCATE_EPS="${MG_RELOCATE_EPS:-0.05}"
MG_RELOCATE_OVERSAMPLE="${MG_RELOCATE_OVERSAMPLE:-4.0}"
MG_RELOCATE_REFINE_ITERS="${MG_RELOCATE_REFINE_ITERS:-3}"
MG_RELOCATE_W_VEL="${MG_RELOCATE_W_VEL:-0.5}"
MG_RELOCATE_W_RADIUS="${MG_RELOCATE_W_RADIUS:-1.0}"
MG_RELOCATE_CHECK_ONLY="${MG_RELOCATE_CHECK_ONLY:-0}"
MG_LABEL="${MG_LABEL:-}"
MG_SEED="${MG_SEED:-0}"
# online / offline / disabled. Empty uses configs/reward/sr2_direct_finetune.yaml's
# wandb block (mode: online, project: cosmo_sr_direct). The API KEY is never put
# in the env file -- that lives on world-readable shared scratch; the job reads it
# out of $HOME instead.
MG_WANDB_MODE="${MG_WANDB_MODE:-}"

RUN_DIR="$REWARD_ROOT/member_gather/${MG_RUNG}${MG_LABEL}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"
ENVFILE="$REWARD_ROOT/env/member_gather_$(date +%Y%m%d_%H%M%S)_$$.env"
# EVERY value is quoted -- see submit_owner_arrays.sh. A sourced
# `MG_TRAIN_BOXES=set3 set4` is bash for "run the command `set4`", not an
# assignment, and it takes the job down before anything starts.
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_member_gather_train.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT="$PROJECT"
ZFS="$ZFS"
DMSR_REWARD_ROOT="$REWARD_ROOT"
MG_TRAIN_BOXES="$MG_TRAIN_BOXES"
MG_HOLDOUT_BOXES="$MG_HOLDOUT_BOXES"
MG_HOLDOUT_HOSTS="$MG_HOLDOUT_HOSTS"
MG_N_TILES="$MG_N_TILES"
MG_MAX_HOSTS_PER_BOX="$MG_MAX_HOSTS_PER_BOX"
MG_MIN_LOG_MVIR="$MG_MIN_LOG_MVIR"
MG_RUNG="$MG_RUNG"
MG_LR_SCALE="$MG_LR_SCALE"
MG_STEPS="$MG_STEPS"
MG_HOSTS_PER_STEP="$MG_HOSTS_PER_STEP"
MG_CLIP="$MG_CLIP"
MG_EVAL_EVERY="$MG_EVAL_EVERY"
MG_W_VIRIAL="$MG_W_VIRIAL"
MG_W_BOUND="$MG_W_BOUND"
MG_W_D6="$MG_W_D6"
MG_W_RRMS="$MG_W_RRMS"
MG_W_SIGMAV="$MG_W_SIGMAV"
MG_W_CENTRE="$MG_W_CENTRE"
MG_BOUND_PENALTY="$MG_BOUND_PENALTY"
MG_TERM_NORM="$MG_TERM_NORM"
MG_CENTRE_MODE="$MG_CENTRE_MODE"
MG_CENTRE_DEAD_ZONE="$MG_CENTRE_DEAD_ZONE"
MG_CENTRE_HUBER="$MG_CENTRE_HUBER"
MG_W_RADIUS="$MG_W_RADIUS"
MG_RADIUS_DEAD_ZONE="$MG_RADIUS_DEAD_ZONE"
MG_W_SHELL="$MG_W_SHELL"
MG_RELOCATE="$MG_RELOCATE"
MG_RELOCATE_DIRS="$MG_RELOCATE_DIRS"
MG_RELOCATE_EPS="$MG_RELOCATE_EPS"
MG_RELOCATE_OVERSAMPLE="$MG_RELOCATE_OVERSAMPLE"
MG_RELOCATE_REFINE_ITERS="$MG_RELOCATE_REFINE_ITERS"
MG_RELOCATE_W_VEL="$MG_RELOCATE_W_VEL"
MG_RELOCATE_W_RADIUS="$MG_RELOCATE_W_RADIUS"
MG_RELOCATE_CHECK_ONLY="$MG_RELOCATE_CHECK_ONLY"
MG_SOFT_MEMBERSHIP="$MG_SOFT_MEMBERSHIP"
MG_SOFT_LAMBDA="$MG_SOFT_LAMBDA"
MG_SOFT_PRIOR="$MG_SOFT_PRIOR"
MG_SOFT_FACTOR="$MG_SOFT_FACTOR"
MG_SOFT_POOL_RATIO="$MG_SOFT_POOL_RATIO"
MG_SOFT_CELL_DILATE="$MG_SOFT_CELL_DILATE"
MG_SOFT_PRIOR_EPS="$MG_SOFT_PRIOR_EPS"
MG_SOFT_BG_FLOOR="$MG_SOFT_BG_FLOOR"
MG_SOFT_STEPS="$MG_SOFT_STEPS"
MG_SOFT_MASS_STEPS="$MG_SOFT_MASS_STEPS"
MG_SOFT_MASS_ETA="$MG_SOFT_MASS_ETA"
MG_SOFT_MASS_STEP_CAP="$MG_SOFT_MASS_STEP_CAP"
MG_SOFT_SIGMA="$MG_SOFT_SIGMA"
MG_SOFT_HOST_PROFILE="$MG_SOFT_HOST_PROFILE"
MG_SOFT_HOST_WEIGHT="$MG_SOFT_HOST_WEIGHT"
MG_SOFT_HOST_CONC="$MG_SOFT_HOST_CONC"
MG_SOFT_HOST_BINS="$MG_SOFT_HOST_BINS"
MG_SOFT_HOST_R_MAX_RVIR="$MG_SOFT_HOST_R_MAX_RVIR"
MG_SOFT_RADIAL_WEIGHT="$MG_SOFT_RADIAL_WEIGHT"
MG_DETECT_ANOMALY="${MG_DETECT_ANOMALY:-0}"
MG_SOFT_RADIAL_WIDTH="$MG_SOFT_RADIAL_WIDTH"
MG_LR_HOST_TABLE="$MG_LR_HOST_TABLE"
MG_LR_HOST_R_MAX_RVIR="$MG_LR_HOST_R_MAX_RVIR"
MG_SOFT_SIGMA_SCALE="$MG_SOFT_SIGMA_SCALE"
MG_SHELL_DEAD_ZONE="$MG_SHELL_DEAD_ZONE"
MG_SHELL_QUANTILES="$MG_SHELL_QUANTILES"
MG_NOISE_DRAWS="$MG_NOISE_DRAWS"
MG_NOISE_MODE="$MG_NOISE_MODE"
MG_FRESH_NOISE_EVAL="$MG_FRESH_NOISE_EVAL"
MG_SHELL_NOISE_DRAWS="$MG_SHELL_NOISE_DRAWS"
MG_SHELL_EPS="$MG_SHELL_EPS"
MG_RADIUS_HUBER="$MG_RADIUS_HUBER"
MG_W_HOST_SETS="$MG_W_HOST_SETS"
MG_HOST_MIN_NUM_P="$MG_HOST_MIN_NUM_P"
MG_HOST_MAX_SETS="$MG_HOST_MAX_SETS"
MG_HOST_SETS_PER_STEP="$MG_HOST_SETS_PER_STEP"
MG_MIN_NUM_P="$MG_MIN_NUM_P"
MG_COND_CHANNELS="$MG_COND_CHANNELS"
MG_COND_MODE="$MG_COND_MODE"
MG_COND_MIN_HR_NUM_P=$MG_COND_MIN_HR_NUM_P
MG_COND_R_MAX_RVIR=$MG_COND_R_MAX_RVIR
MG_COND_LOG_MHOST_REF=$MG_COND_LOG_MHOST_REF
MG_COND_LOG_MHOST_SCALE=$MG_COND_LOG_MHOST_SCALE
MG_COND_ROOT="$MG_COND_ROOT"
MG_COND_LR=$MG_COND_LR
MG_POPULATION="$MG_POPULATION"
MG_MAX_NUM_P="$MG_MAX_NUM_P"
MG_SET_SELECT="$MG_SET_SELECT"
MG_ENV_SELECT="$MG_ENV_SELECT"
MG_ENV_HOST_FLOOR="$MG_ENV_HOST_FLOOR"
MG_W_ISOLATION="$MG_W_ISOLATION"
MG_ISO_RADIUS_FRAC="$MG_ISO_RADIUS_FRAC"
MG_W_TEMP="$MG_W_TEMP"
MG_TEMP_MAX_X="$MG_TEMP_MAX_X"
MG_TEMP_QUERIES="$MG_TEMP_QUERIES"
MG_TEMP_MEASURE="$MG_TEMP_MEASURE"
MG_ISO_SOFTNESS="$MG_ISO_SOFTNESS"
MG_ISO_MIN_COUNT="$MG_ISO_MIN_COUNT"
MG_GATE_AT_STEPS="$MG_GATE_AT_STEPS"
MG_MIN_PURITY="$MG_MIN_PURITY"
MG_MIN_LIVE_FRAC="$MG_MIN_LIVE_FRAC"
MG_STATS_MEMBERS=$MG_STATS_MEMBERS
MG_SOFTENING="$MG_SOFTENING"
MG_BOUND_TAU="$MG_BOUND_TAU"
MG_BG_K="$MG_BG_K"
MG_BG_RADIUS="$MG_BG_RADIUS"
MG_POT_CHUNK="$MG_POT_CHUNK"
MG_MAX_SET_PARTICLES="$MG_MAX_SET_PARTICLES"
MG_W_LOW="$MG_W_LOW"
MG_W_HIGHK="$MG_W_HIGHK"
MG_K_SPLIT="$MG_K_SPLIT"
MG_LOW_K_MAX="$MG_LOW_K_MAX"
MG_HIGHK_MAX="$MG_HIGHK_MAX"
MG_VEL_RMS_TOL="$MG_VEL_RMS_TOL"
MG_HIGHK_BINS="$MG_HIGHK_BINS"
MG_HIGHK_K_MAX="$MG_HIGHK_K_MAX"
MG_HIGHK_TOL="$MG_HIGHK_TOL"
MG_HIGHK_TWO_SIDED="$MG_HIGHK_TWO_SIDED"
MG_HIGHK_REDUCE="$MG_HIGHK_REDUCE"
MG_HIGHK_MAX_HOLDOUT="$MG_HIGHK_MAX_HOLDOUT"
MG_W_VEL_HIGHK="$MG_W_VEL_HIGHK"
MG_VEL_HIGHK_TOL="$MG_VEL_HIGHK_TOL"
MG_VEL_HIGHK_MIN="$MG_VEL_HIGHK_MIN"
MG_UNSUP_TILES_PER_BOX="$MG_UNSUP_TILES_PER_BOX"
MG_UNSUP_TILES_PER_STEP="$MG_UNSUP_TILES_PER_STEP"
MG_W_HIGHK_UNSUP="$MG_W_HIGHK_UNSUP"
MG_UNSUP_HIGHK_MAX="$MG_UNSUP_HIGHK_MAX"
MG_W_KNN_UNSUP="$MG_W_KNN_UNSUP"
MG_KNN_KS="$MG_KNN_KS"
MG_KNN_QUERIES="$MG_KNN_QUERIES"
MG_KNN_QUANTILES="$MG_KNN_QUANTILES"
MG_KNN_DEAD_ZONE="$MG_KNN_DEAD_ZONE"
MG_KNN_BUFFER_CELLS="$MG_KNN_BUFFER_CELLS"
MG_KNN_NBR_MARGIN_CELLS="$MG_KNN_NBR_MARGIN_CELLS"
MG_KNN_CENTRE="$MG_KNN_CENTRE"
MG_KNN_ONE_SIDED="$MG_KNN_ONE_SIDED"
MG_KNN_PHASE_LAMBDA="$MG_KNN_PHASE_LAMBDA"
MG_KNN_CLASSES="$MG_KNN_CLASSES"
MG_KNN_STRATA=$MG_KNN_STRATA
MG_KNN_STRATA_RANK=$MG_KNN_STRATA_RANK
MG_KNN_STRATA_CHARGE="$MG_KNN_STRATA_CHARGE"
MG_KNN_CLASSES_DIAG_ONLY=$MG_KNN_CLASSES_DIAG_ONLY
MG_W_ADV="$MG_W_ADV"
MG_ADV_WARMUP="$MG_ADV_WARMUP"
MG_ADV_RAMP="$MG_ADV_RAMP"
MG_N_CRITIC="$MG_N_CRITIC"
MG_CRITIC_LR="$MG_CRITIC_LR"
MG_CRITIC_WIDTH="$MG_CRITIC_WIDTH"
MG_CRITIC_LAYERS="$MG_CRITIC_LAYERS"
MG_CRITIC_GLOBAL_POOL="$MG_CRITIC_GLOBAL_POOL"
MG_CRITIC_R1_GAMMA="$MG_CRITIC_R1_GAMMA"
MG_CRITIC_R1_INTERVAL="$MG_CRITIC_R1_INTERVAL"
MG_CRITIC_NORM_FIT_TILES="$MG_CRITIC_NORM_FIT_TILES"
MG_LABEL="$MG_LABEL"
MG_SEED="$MG_SEED"
MG_WANDB_MODE="$MG_WANDB_MODE"
EOT

echo "envfile: $ENVFILE"
cat "$ENVFILE"
echo

# Source it in a CLEAN shell and check the list-valued variables round-trip,
# before anything is queued. An env file that does not source is a whole-chain
# failure found one job id later; this costs milliseconds.
_check=$(env -i bash -c "set -eu; source '$ENVFILE'; printf '%s|%s' \"\$MG_TRAIN_BOXES\" \"\$MG_HOLDOUT_BOXES\"" 2>&1) || {
    echo ">>> the env file does not source cleanly:" >&2
    echo ">>> $_check" >&2
    exit 1
}
if [ "$_check" != "$MG_TRAIN_BOXES|$MG_HOLDOUT_BOXES" ]; then
    echo ">>> env file round-trip MISMATCH -- values were mangled by sourcing." >&2
    echo ">>>   wrote: $MG_TRAIN_BOXES|$MG_HOLDOUT_BOXES" >&2
    echo ">>>   read:  $_check" >&2
    exit 1
fi
echo "env file sources cleanly and round-trips."
echo
echo "run dir: $RUN_DIR"
echo

# Owner arrays, checked here so a missing prerequisite is visible before the
# queue rather than as a quiet skip inside the job.
echo "owner arrays (membership IS the supervision):"
_MISSING=0
for b in $MG_TRAIN_BOXES $MG_HOLDOUT_BOXES; do
    p="$REWARD_ROOT/halos_particles/${b}__hr__hr/${b}_hr_owner.npy"
    if [ -r "$p" ]; then printf "  %-7s ok\n" "$b"
    else printf "  %-7s MISSING -- that box will be SKIPPED\n" "$b"; _MISSING=$((_MISSING + 1)); fi
done
if [ "$_MISSING" -gt 0 ]; then
    echo
    echo ">>> $_MISSING box(es) have no owner array. Build them with:"
    echo ">>>   bash scripts/slurm/submit_owner_arrays.sh"
    echo ">>> Continuing: the run will use whatever boxes are ready."
fi
echo

# The shell arm's one configuration trap, checked BEFORE the queue rather than
# discovered in a metrics file eight GPU hours later. `self` pins the radius
# `w_radius` is trying to move; the two settle at w_radius/(w_centre+w_radius)
# of the correction, which is half at the default weights.
if [ "$MG_W_RADIUS" != "0.0" ] && [ "$MG_W_RADIUS" != "0" ]; then
    echo "shell arm: w_radius=$MG_W_RADIUS, dead zone ${MG_RADIUS_DEAD_ZONE}r," \
         "centre mode $MG_CENTRE_MODE"
    if [ "$MG_CENTRE_MODE" = "self" ]; then
        echo ">>> WARNING: --centre-mode self ANCHORS the very radius w_radius"
        echo ">>>   is charging to move. At w_centre=$MG_W_CENTRE and"
        echo ">>>   w_radius=$MG_W_RADIUS the two settle at"
        echo ">>>   w_radius/(w_centre + w_radius) of the correction."
        echo ">>>   Use MG_CENTRE_MODE=self_tan unless the tug-of-war IS the"
        echo ">>>   experiment. Continuing."
    elif [ "$MG_CENTRE_MODE" = "self_tan" ]; then
        echo "  centre mode self_tan: the anchor keeps the two transverse"
        echo "  coordinates and this term owns the radial one. Orthogonal."
    fi
    echo
fi

# The SOFT shell's traps, same place and for the same reason: `self` pins the
# radii it redistributes, and running it TOGETHER with w_radius reinstates the
# per-set address this term exists to stop asking for.
if [ "$MG_W_SHELL" != "0.0" ] && [ "$MG_W_SHELL" != "0" ]; then
    echo "soft shell: w_shell=$MG_W_SHELL, dead zone $MG_SHELL_DEAD_ZONE" \
         "fractional, centre mode $MG_CENTRE_MODE"
    if [ "$MG_CENTRE_MODE" = "self" ]; then
        echo ">>> WARNING: --centre-mode self ANCHORS the radii w_shell is"
        echo ">>>   redistributing, exactly as it does w_radius. Use"
        echo ">>>   MG_CENTRE_MODE=self_tan. Continuing."
    elif [ "$MG_CENTRE_MODE" = "self_tan" ]; then
        echo "  centre mode self_tan: the anchor keeps the two transverse"
        echo "  coordinates and this term owns the radial one. Orthogonal."
    fi
    if [ "$MG_W_RADIUS" != "0.0" ] && [ "$MG_W_RADIUS" != "0" ]; then
        echo ">>> WARNING: MG_W_RADIUS=$MG_W_RADIUS is ALSO on. w_radius"
        echo ">>>   charges the per-set address (out-of-sample R2 0.022) that"
        echo ">>>   w_shell exists to stop asking for; running both puts the"
        echo ">>>   memorisation back in. Set MG_W_RADIUS=0 unless the"
        echo ">>>   composition IS the experiment. Continuing."
    fi
    echo "  Read shell_resid -> 0. radius_err_radii will NOT move much even on"
    echo "  a total win: a perfect distribution match leaves 79%/85% of the"
    echo "  per-set address error standing, by construction."
    if [ "$MG_SHELL_QUANTILES" != "0" ]; then
        echo "  quantile bins: $MG_SHELL_QUANTILES. The term now asks for the"
        echo "  LOW-ORDER shape only; a rank-by-rank wiggle inside a bin is"
        echo "  free. term_raw_shell is NOT comparable with a Q=0 arm's --"
        echo "  compare shell_resid, which is measured the same way at every Q."
    fi
    echo
fi

# --- the noise realisation ---------------------------------------------------
# Each extra draw costs a frozen forward AND a full reference build per host at
# pool time, because centre_self and the hybrid reference are functions of the
# frozen field. That is a real bill on the pool stage, not the step loop.
if [ "$MG_NOISE_MODE" != "fixed" ] || [ "$MG_FRESH_NOISE_EVAL" = "1" ] \
   || [ "$MG_SHELL_NOISE_DRAWS" != "0" ]; then
    echo "noise: $MG_NOISE_DRAWS training draw(s), mode $MG_NOISE_MODE," \
         "fresh-noise eval $MG_FRESH_NOISE_EVAL," \
         "shell averaged over +$MG_SHELL_NOISE_DRAWS"
    echo "  Pool build costs roughly N_DRAWS x the single-draw run. Budget it."
    if [ "$MG_NOISE_MODE" = "rotate" ] && [ "$MG_NOISE_DRAWS" = "1" ]; then
        echo ">>> FATAL: rotate over ONE draw is 'fixed' under another name."
        echo ">>>   Set MG_NOISE_DRAWS=2 or more."
        exit 1
    fi
    if [ "$MG_SHELL_NOISE_DRAWS" != "0" ]; then
        if [ "$MG_W_SHELL" = "0.0" ] || [ "$MG_W_SHELL" = "0" ]; then
            echo ">>> FATAL: MG_SHELL_NOISE_DRAWS averages the SHELL curve, but"
            echo ">>>   MG_W_SHELL is 0 -- there is no term to average."
            exit 1
        fi
        if [ "$MG_NOISE_DRAWS" = "1" ]; then
            echo ">>> FATAL: nothing to average over at MG_NOISE_DRAWS=1."
            exit 1
        fi
    fi
    if [ "$MG_FRESH_NOISE_EVAL" = "1" ]; then
        echo "  metrics.jsonl gains train_fresh / holdout_fresh. THE 2x2:"
        echo "    train_fresh falls back to holdout -> the train/holdout gap was"
        echo "      the fixed noise being memorised. The objective carries no"
        echo "      more transferable signal than holdout already shows."
        echo "    train_fresh tracks train -> the model learned from the LR"
        echo "      field and the gap is about BOXES and HOSTS instead: a"
        echo "      different problem with a different fix."
    fi
    echo
fi

# The host guard's pair bill. MEASURED on job 37229: 1.770e+11 uncapped over
# 5,198 host sets, largest 247,745 particles. A cluster is not a subhalo, and
# an uncapped host guard is ~40x the supervised bill.
if [ "$MG_W_HOST_SETS" != "0.0" ] && [ "$MG_W_HOST_SETS" != "0" ] \
   && [ "$MG_MAX_SET_PARTICLES" = "0" ]; then
    echo ">>> WARNING: MG_W_HOST_SETS=$MG_W_HOST_SETS with no"
    echo ">>>   MG_MAX_SET_PARTICLES. Host sets reach ~2.5e5 particles, so the"
    echo ">>>   guard's pair sum is ~1.8e11 against the supervised sets' ~4e9."
    echo ">>>   MG_MAX_SET_PARTICLES=2048 is what the selfhostguard arm ran."
    echo ">>>   Continuing."
    echo
fi

# The small-halo arm's two configuration traps, checked BEFORE the queue rather
# than after the pool build has already spent minutes of a GPU allocation.
if [ "$MG_POPULATION" != "sub" ]; then
    echo "population: $MG_POPULATION, num_p window [$MG_MIN_NUM_P, ${MG_MAX_NUM_P:-inf}), cap $MG_SET_SELECT"
    if [ -n "$MG_COND_CHANNELS" ]; then
        echo "cond:       $MG_COND_CHANNELS (mode $MG_COND_MODE, gate >=$MG_COND_MIN_HR_NUM_P HR p, lr $MG_COND_LR)"
        echo "            the generator SEES the host centre and scale in this arm."
        echo "            read the radial diagnostic, not the loss: a flat profile with"
        echo "            a falling loss means the channel is being ignored."
    else
        echo "cond:       none (unconditioned control)"
    fi
    if [ "$MG_MAX_NUM_P" = "0" ]; then
        echo ">>> REFUSED: MG_POPULATION=$MG_POPULATION with no MG_MAX_NUM_P" >&2
        echo ">>>   selects the CLUSTER HOST itself as a supervised set: ~100k" >&2
        echo ">>>   particles, 1e10 pairs against a 200p set's 2e4, and an HR" >&2
        echo ">>>   reference that asks the generator to concentrate a 1e14" >&2
        echo ">>>   halo -- a different objective, and one MG_W_HOST_SETS" >&2
        echo ">>>   already handles against the FROZEN reference." >&2
        echo ">>>   Set MG_MAX_NUM_P (e.g. 2000)." >&2
        exit 1
    fi
    if [ "$MG_SET_SELECT" = "largest" ]; then
        echo ">>> WARNING: MG_SET_SELECT=largest keeps the TOP of the mass"
        echo ">>>   function when MG_MAX_SETS binds, which drops exactly the"
        echo ">>>   small halos this population was widened to reach. Use"
        echo ">>>   MG_SET_SELECT=stratified unless truncation IS the"
        echo ">>>   experiment. Continuing."
    fi
    echo
fi

# The environment axis and the isolation term, same principle: checked BEFORE
# the queue rather than after a pool build has spent minutes of an allocation.
if [ "$MG_ENV_SELECT" != "any" ]; then
    echo "env_select: $MG_ENV_SELECT (resolved-host floor ${MG_ENV_HOST_FLOOR}p)"
    for _e in $(echo "$MG_ENV_SELECT" | tr ',' ' '); do
        case "$_e" in
            inside|near|field) ;;
            *)
                echo ">>> REFUSED: MG_ENV_SELECT=$MG_ENV_SELECT names '$_e'," >&2
                echo ">>>   which is not one of inside/near/field. A typo that" >&2
                echo ">>>   quietly selected nothing would surface as the" >&2
                echo ">>>   unrelated 'no supervised sets survived' in the pool" >&2
                echo ">>>   build, minutes into a GPU allocation." >&2
                exit 1 ;;
        esac
    done
    case ",$MG_ENV_SELECT," in
        *,field,*)
            if [ "$MG_POPULATION" = "sub" ]; then
                echo ">>> WARNING: env_select field with MG_POPULATION=sub"
                echo ">>>   selects satellites of hosts BELOW the ${MG_ENV_HOST_FLOOR}p"
                echo ">>>   floor, not the isolated field halos the census's"
                echo ">>>   'field' deficit is about -- that deficit is"
                echo ">>>   top-level. Add MG_POPULATION=all if the field halos"
                echo ">>>   themselves are the target. Continuing."
            fi ;;
    esac
    echo
fi

if [ -n "$MG_GATE_AT_STEPS" ]; then
    echo "early gate: snapshots at step(s) $MG_GATE_AT_STEPS of $MG_STEPS"
    for _st in $(echo "$MG_GATE_AT_STEPS" | tr ',' ' '); do
        case "$_st" in
            ''|*[!0-9]*)
                echo ">>> REFUSED: MG_GATE_AT_STEPS='$MG_GATE_AT_STEPS' contains" >&2
                echo ">>>   '$_st', which is not a positive integer." >&2
                exit 1 ;;
        esac
        if [ "$_st" -le 0 ] || [ "$_st" -ge "$MG_STEPS" ]; then
            echo ">>> REFUSED: MG_GATE_AT_STEPS step $_st is not inside" >&2
            echo ">>>   (0, MG_STEPS=$MG_STEPS). The final checkpoint is already" >&2
            echo ">>>   gated by the normal chain; a snapshot at or past the last" >&2
            echo ">>>   step would duplicate it, and one past it would NEVER be" >&2
            echo ">>>   written -- stranding the gate queued against it on" >&2
            echo ">>>   DependencyNeverSatisfied with nothing to read." >&2
            exit 1
        fi
    done
    echo ">>>   Training is UNAFFECTED: it still runs to $MG_STEPS. The snapshot"
    echo ">>>   is a file copy and the gate's 19 min are on other nodes."
    echo
fi

if [ "$MG_W_ISOLATION" != "0.0" ] && [ "$MG_W_ISOLATION" != "0" ]; then
    echo "isolation: w=$MG_W_ISOLATION, radius ${MG_ISO_RADIUS_FRAC} of the bg sphere" \
         "(${MG_BG_RADIUS} set radii), min HR count ${MG_ISO_MIN_COUNT}"
    if [ "$MG_BG_K" = "0" ]; then
        echo ">>> REFUSED: MG_W_ISOLATION>0 with MG_BG_K=0. The term counts" >&2
        echo ">>>   LOCAL NON-MEMBER particles and MG_BG_K=0 gathers none, so" >&2
        echo ">>>   the term is identically zero and the run is a" >&2
        echo ">>>   MG_W_ISOLATION=0 run under another name." >&2
        exit 1
    fi
    echo ">>>   NOTE: the background ids are chosen ONCE, on the frozen field,"
    echo ">>>   inside ${MG_BG_RADIUS} set radii. Mass arriving from further out is"
    echo ">>>   invisible to this term. Widen MG_BG_RADIUS if the pedestal the"
    echo ">>>   gate reports is larger than that sphere."
    echo
fi

# Scrub SLURM_* so a submission from inside an allocation cannot leak an
# --export-like environment onto the batch jobs.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

# Built once and applied to BOTH submissions: a shakeout that fits while the
# fine-tune behind it does not is the failure this centralisation prevents.
RES=()
[ -n "$MG_MEM" ]  && RES+=(--mem="$MG_MEM")
[ -n "$MG_TIME" ] && RES+=(--time="$MG_TIME")
[ -n "$MG_EXCLUDE" ] && RES+=(--exclude="$MG_EXCLUDE")

sub() {
    if [ "$DRY" = "1" ]; then echo "DRY: sbatch ${RES[*]-} $*" >&2; echo "DRYID"
    else sbatch --parsable "${RES[@]}" "$@"; fi
}

SID=""; TID=""

if [ "$SKIP_SHAKEOUT" != "1" ]; then
    # Selection and the reachable reference are everything that can be wrong
    # before a step is taken. Minutes, and it stands between a misconfigured
    # split and twelve GPU hours.
    SID=$(sub scripts/slurm/member_gather_train_gpu.sbatch "$ENVFILE" \
          MG_POOL_ONLY=1 MG_STEPS=0)
    echo "submitted shakeout (pool only) -> job $SID (GPU)"
fi

if [ "$SHAKEOUT_ONLY" = "1" ]; then
    echo
    [ -n "$SID" ] && echo "watch: tail -F $REWARD_ROOT/logs/slurm-mgtrain-$SID.out"
    echo "then read: $RUN_DIR/pool.json"
    echo
    echo "READ THE SHAKEOUT BEFORE TRAINING. Check that:"
    echo "  - every intended box appears (a missing owner array is a silent skip)"
    echo "  - 'rejected' is empty, or you understand each entry"
    echo "  - the held-out pool is not tiny; it is the only thing being claimed"
    exit 0
fi

DEP=(); [ -n "$SID" ] && DEP=(--dependency=afterok:"$SID")
TID=$(sub "${DEP[@]}" scripts/slurm/member_gather_train_gpu.sbatch "$ENVFILE")
echo "submitted fine-tune            -> job $TID (GPU${SID:+, afterok:$SID})"

echo
[ -n "$SID" ] && echo "watch shakeout:  tail -F $REWARD_ROOT/logs/slurm-mgtrain-$SID.out"
[ -n "$TID" ] && echo "watch fine-tune: tail -F $REWARD_ROOT/logs/slurm-mgtrain-$TID.out"
echo
echo "results:"
echo "  $RUN_DIR/pool.json      hosts, sets, and each host's reachable reference"
echo "  $RUN_DIR/metrics.jsonl  one row per eval -- train AND holdout, per host"
echo "  $RUN_DIR/summary.json   config, history, VERDICT"
echo "  $RUN_DIR/tuned.pt       {'model': full state dict}, loadable by the gate"
echo
echo "wandb: group 'member_gather', run 'member_gather-${MG_RUNG}${MG_LABEL}'."
echo "  Curves mirror metrics.jsonl (which stays the source of truth, so an"
echo "  outage never loses a number). The RESULT -- verdict text, its three"
echo "  booleans, frozen->final deltas for every pooled metric, and a per-host"
echo "  table -- lands in the run SUMMARY, so the runs table is readable without"
echo "  opening any chart. Running the rung ladder puts fine / middle_fine /"
echo "  all_blocks in one group for direct comparison."
echo
echo "THE VERDICT THIS CHAIN PRINTS IS NOT THE RESULT. Every number in it is a"
echo "member-set surrogate the objective computes about itself, and this line has"
echo "measured such a surrogate reaching +255 while real Rockstar showed no gain"
echo "at all (tile-overfit-proxy-exploitation). The result is the whole-box"
echo "Rockstar gate on the HELD-OUT pool:"
echo
echo "  # chained behind THIS job, fires the moment training finishes (~19 min):"
echo "  MG_ARMS=${MG_RUNG}${MG_LABEL} HG_AFTER_JOB=$TID \\"
echo "    HG_BASELINE_ARM=<a previously gated arm> HG_BASELINE_NAME=<its name> \\"
echo "    bash scripts/slurm/submit_gather_holdout_rockstar.sh"
if [ -n "$MG_GATE_AT_STEPS" ]; then
    _first_gate="$(echo "$MG_GATE_AT_STEPS" | tr ',' ' ' | awk '{print $1}')"
    echo
    echo "  # EARLY gate -- submit NOW, alongside training. A 1-core CPU job polls"
    echo "  # for the snapshot, then the chain runs while training continues:"
    echo "  MG_ARMS=${MG_RUNG}${MG_LABEL} HG_CHECKPOINT=tuned_step${_first_gate}.pt \\"
    echo "    NO_CONTROL=1 HG_BASELINE_ARM=<a previously gated arm> \\"
    echo "    HG_BASELINE_NAME=<its name> \\"
    echo "    bash scripts/slurm/submit_gather_holdout_rockstar.sh"
    echo
    echo "  Its artifacts land in ${RUN_DIR}/holdout_<box>_step${_first_gate}/ and under"
    echo "  Rockstar tag mgho_${MG_RUNG}${MG_LABEL}_step${_first_gate}_<box>, so nothing"
    echo "  collides with the end-of-run gate above."
    echo "  READ IT ONE-SIDED: a clear negative says the mechanism did not change"
    echo "  and is worth acting on; a null is NOT evidence of failure."
fi
echo
echo "  Read it against the oracle's calibration: ceiling 151/154, noise +-9,"
echo "  frozen 3/154, free field 72/154 -- and in the EDITED REGION, never"
echo "  box-wide, where 93.75% is untouched frozen SR2 and swamps the signal."
