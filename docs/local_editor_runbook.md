# Host-conditioned generative local editor — runbook

Implementation record and operating instructions for

```
Psi_out = Psi_SR2 + E(Psi_SR2, C, a)
```

where `Psi_SR2` is frozen, `C` is a set of proposed subhalo tokens, `a` is a
low-dimensional stochastic editing action, `E` moves only particles it has
explicitly claimed, and Rockstar supplies the training reward. **No HR residual
field is used as a training target, and no stage of this pipeline resumes or
trains the six-channel residual diffusion model.**

Branch `fix/reward-pipeline-review`, starting commit `047d609`.

---

## 0. Why this exists, and what it inherits

The full-field residual prior failed: a diffusion model asked to correct a
512³ displacement field never learned to place a subhalo. The targeted HR
oracle ([`docs/catalog_reward_oracle.md`](catalog_reward_oracle.md)) established
the complementary fact — **localised editing of the frozen field does have
causal leverage on the halo catalog**: displacement-only intervention took
subhalo recovery from 37.5% to 72.9%, displacement plus velocity to 95.8%.

That oracle reads the answer, so it is not deployable. This pipeline is the
deployable analogue of the same intervention, and it is deliberately built as a
**separate line**: separate config (`configs/reward/local_editor.yaml`),
separate artifact root (`$ZFS/DMSR/dmsr_local_editor/`, a sibling of the reward
root, never inside it), separate submitter. It *reads* four things from the
reward line and writes none of them:

| input | produced by |
| --- | --- |
| frozen SR2 field cache `cache/sr2_base/<box>_seed0_*.npy` | `scripts/slurm/cache_sr2_base.sbatch` |
| frozen SR2 catalogs `halos/<box>__base__base/` | `scripts/slurm/hr_catalog_summaries_cpu.sbatch SOURCES=base` |
| catalog bins, for the *reported* full-box statistics | `configs/reward/reward.yaml` |
| Experiment-1 oracle rows (calibration + upper bound only) | `scripts/slurm/exp1_intervene_cpu.sbatch` |

The Experiment-1 rows are used in exactly two places, both non-training:
threshold calibration (stage 0, §2) and the upper-bound row of the final
comparison (§5).
`tests/reward/test_no_hr_leak.py` enforces the boundary by source scan, import
graph, and a runtime `np.load` poison over a full compose-and-score cycle.

---

## 1. What was built

### Modules (`src/cosmo_sr/reward/`)

| file | contents |
| --- | --- |
| `local_editor.py` | `SubhaloToken`, `EditorAction`, `ActionCodec`, host particle pools, the periodic contraction/cooling transformation |
| `local_reward.py` | object-level Rockstar reward, host-damage and artifact penalties, the quarantined compactness proxy, `gate1_verdict` |
| `cem.py` | bounded, resumable, per-mode cross-entropy search |
| `action_flow.py` | conditional flow matching + the mandatory Gaussian-mixture baseline + bounded reward weights |
| `token_bootstrap.py` | variable-cardinality `C_h` by bootstrapping normalised training-host catalogs |

### The action space

A proposal splits into *what* and *how*. The searched vector is 12-dimensional:
a 4-d **token** block and an 8-d **action** block, in that order (the flow
slices the action block off by index, so the ordering is load-bearing and is
pinned by a test).

```
token   log_mass_ratio  radius_rvir  dir_cos_theta  dir_phi
action  center_offset_{x,y,z}  source_radius_rvir  contraction
        velocity_cooling  bulk_velocity_mix  edge_softness
```

`ActionCodec` maps `R^12` onto the boxed parameters with a scaled sigmoid, so
CEM samples ordinary Gaussians and the flow transports ordinary Gaussians —
neither ever rejects or clips a proposal. The three edit modes (`disp`, `both`,
`vel`) are implemented by **pinning** bounds, not by branching in the editor, so
"displacement-only returns velocities bit-for-bit" is a property of one number
being zero. `both` is displacement-*dominant* by construction: cooling is capped
below the contraction ceiling so the search cannot quietly become velocity-only
and report it as the winning mode.

### The transformation

For each claimed particle with smooth window weight `w_i`:

```
dx_i = -kappa_x w_i MI(x_i - c)
v_i' = v_ref + (1 - kappa_v w_i)(v_i - v_ref)
```

`MI` is the periodic minimum image; `w` is 1 in the core and **exactly** zero at
`r = source_radius`. Reaching exactly zero is what makes "only claimed particles
changed" checkable rather than approximate.

Selection and support are separate decisions on purpose. *Which* particles: the
`n` nearest the proposed centre, `n` from the token's mass ratio (particles are
equal-mass, so the mass ratio **is** the count ratio). *How strongly*: the
window from the action's radius. Coupling them ("take everything inside R")
would make count a dependent variable of radius, and the search could not move
mass and concentration independently — which are the two things that decide
whether Rockstar sees a bound clump.

### The reward

`R_cat`/`R_occ` are hopeless as a *search* signal here: one extra subhalo moves
a full-box occupation curve by well under its sampling noise, so ranking on it
would rank Rockstar jitter. The search reward is per proposal:

```
r_j = r_detected + w_m r_mass + w_p r_position
      - lambda_h r_host_damage - lambda_a r_artifacts
```

`r_detected` is the only term positive on its own; the shaping terms are gated
on detection. New-ness is decided by a **one-to-one periodic match of the whole
candidate subhalo population against the whole frozen one** (hosts included), so
an object that matched anything is permanently ineligible. Full-box `R_occ`,
`R_abund` and `R_cat` are measured and reported on every candidate as evidence,
never as the objective.

The compactness proxy exists only for a round where every candidate scores
identically. It never enters `r_j`, never marks a success, and
`is_scientific_success()` — the gate for the flow's training set — ignores it.

---

## 2. Commands, in order

Everything is submitted; nothing runs on a login node. `DRY=1` prints the
`sbatch` lines and submits nothing.

```bash
cd /zfsauton2/home/yixiz/DMSR/cosmo_sr_project
DRY=1 bash scripts/slurm/submit_local_editor.sh setup      # inspect first
```

### Stage 1–2 — hosts and member ids

```bash
bash scripts/slurm/submit_local_editor.sh setup
```

Selects 8 well-separated hosts per box in host-mass bins 2 and 3, then runs the
Rockstar member-id pass on the frozen SR2 box and reduces it to a pool cache.

### Stage 0 — calibrate the feasibility thresholds

```bash
bash scripts/slurm/submit_local_editor.sh audit
```

**This stage ends in a human step and the submitter stops there.** When it
finishes:

1. read `$ZFS/DMSR/dmsr_local_editor/audit/constraints_proposal.json`;
2. check `frozen_arm_exactly_zero` is `true` — if not, the measurement path is
   broken and nothing after it means anything;
3. check the accept population covers the oracle's **successful** interventions;
4. paste `proposed_constraints_block` into the `constraints:` block of
   `configs/reward/local_editor.yaml`, set `calibrated: true`, commit.

Until then every scoring script exits with an explanation. Do not copy the
thresholds from `configs/reward/reward.yaml`: `low_k_change_max = 0.02` was
derived for a whole-field residual and would pass everything a local edit can
do, destructive ones included.

### Stage 3 — Gate 1, the deployment-legal action-space oracle

```bash
bash scripts/slurm/submit_local_editor.sh gate1
```

12 random-action candidates per box in each of four arms — targeted plus the
three controls — then the verdict.

### Stage 4 — CEM

```bash
ROUNDS=3 CANDIDATES=28 bash scripts/slurm/submit_local_editor.sh cem
```

### Stage 5 — distillation, and its arms

```bash
bash scripts/slurm/submit_local_editor.sh distill
```

### Final comparison

```bash
bash scripts/slurm/submit_local_editor.sh eval
# and, ONCE, at the very end:
FINAL=1 BOXES=set13,set14,set15 bash scripts/slurm/submit_local_editor.sh eval
```

`set13`/`set14`/`set15` are refused by every other stage.

---

## 3. Expected artifacts

Everything under `$ZFS/DMSR/dmsr_local_editor/` (`RUN=runs/le_a`):

| path | what | size |
| --- | --- | --- |
| `$RUN/hosts/hosts_<box>.json` | selected hosts + their subhalos | KB |
| `$RUN/hosts/halo_ids_<box>.json` | ids for the member-id pass | KB |
| `$RUN/pools/pools_<box>.npz` | member particle ids per object | ~10 MB |
| `$RUN/pools/pool_summary_<box>.json` | pool sizes, smooth fractions | KB |
| `$RUN/cem/<mode>/round_NNN.json` | CEM state, sampled `z`, rewards | KB |
| `$RUN/cem/round_NNN_actions.json` | the manifest candidates read | ~1 MB |
| `$RUN/candidates/rows_<box>.jsonl` | one row per candidate: actions, plans, outcomes, constraints, full-box catalog | MB |
| `$RUN/round_NNN_summary.json` | per-mode CEM update + Gate 1 | KB |
| `$RUN/gate1.json` | the Gate 1 verdict | KB |
| `$RUN/flow/action_policy.pt` | flow + reference + GMM | ~1 MB |
| `$RUN/final_comparison.json` | the six-arm table | KB |
| `audit/constraints_proposal.json` | the threshold proposal | KB |
| `candidates/<run>/<box>/<cid>/` | Rockstar ASCII catalog per candidate | ~10 MB each |

No 512³ candidate field is written; candidates are composed in memory and handed
straight to the halo finder. `--save-field` exists for debugging one.

Figures and tables are redrawable from the JSONL rows alone — the evaluation and
aggregation jobs are CPU and read only saved artifacts.

---

## 4. Cost

The unit is one full-box Rockstar run: ~15 min plus a 3.7 GB GADGET2 dump
deleted immediately, on one CPU node at 16 cores / 192 GB.

| stage | Rockstar runs | wall (with 4 shards) | partition |
| --- | --- | --- | --- |
| 1 select hosts | 0 | seconds | cpu |
| 2 member ids | 1 per box (with member output, ~2×) | ~1–2 h/box | cpu |
| 0 audit | 9 per box | ~1 h | cpu |
| 3 Gate 1 | 4 arms × 12 × 2 boxes = 96 | ~6 h | cpu |
| 4 CEM | 3 × 28 × 2 = 168 | ~11 h | cpu |
| 5 train flow + GMM | 0 | ~10 min | general (1 GPU) |
| 5b flow/gmm arms | 2 × 28 × 2 = 112 | ~7 h | cpu |
| eval | 0 | seconds | cpu |

**≈ 380 Rockstar runs, ~26 h of wall time at 4-way sharding.** The 8-hosts-per-box
batching is what makes this affordable: each run returns 8 independent
proposal-level rewards, so stage 4 is 1344 CEM samples from 168 halo runs.

Every stage skips work it has already done (`row.json` present ⇒ candidate
skipped), so a time limit means resubmitting, not restarting.

---

## 5. Pass/fail interpretation, stage by stage

**Stage 0 — calibration. DONE, 2026-08-02.** Thresholds measured over 18 editor
candidates, 18 successful HR-oracle interventions, 6 controls and 2 frozen
anchors on set8/set9, and committed to `configs/reward/local_editor.yaml`.

The no-op arm measured `low_k_change` **exactly 0.0** on the real 512³ box, so
the bit-for-bit invariant holds in production. But **neither threshold binds on
the editor**: `low_k_change` has 302× headroom (it is set by the HR oracle's much
larger interventions, which the plan requires it to accept), and
`lr_consistency_error` is 0.46432 for the frozen box and moves in the 6th decimal
under an edit. So `feasible_field: true` will be true for essentially anything
this editor can do — read `host_preserved_rate` and `host_damage_mean` as the
safety signal instead, not `field_feasible_rate`.

The generic notes below still apply if the audit is ever re-run:

**Stage 0 — calibration (how to re-run).** `frozen_arm_exactly_zero: false` is a bug in the
measurement path; stop. `separates: false` against the random-particle control
is *expected* — an equal-count control has the same field-level footprint by
design — and means the threshold bounds edit **size**, not edit quality. Say
that wherever the thresholds are used.

**Stage 3 — Gate 1.** Pass requires all of: ≥5 legitimate new subhalos, across
≥3 hosts, across ≥2 non-final boxes, with parents preserved, fields feasible,
and the targeted arm ahead of the equal-count random-particle control. The
`near_subhalo` control must score **zero** detections; a nonzero score there is a
bug report against the reward, not a result.

*If Gate 1 fails, do not implement the flow.* The plan's instruction is to
expand the editor representation first — a compensating shell, or a small
learned local residual basis. `train_action_flow.py` enforces this: it reads
`gate1.json` and exits 0 with an explanation.

**Stage 4 — CEM.** Read `update.reason` per mode. `all_candidates_equivalent`
every round means the reward is flat and the action space, not the search, is
the problem. `no_feasible_candidate` means the thresholds are too tight for the
actions being drawn — recheck the calibration rather than widening them.
`proxy_tiebreak: true` means that round produced no detections at all and the
distribution moved on a proxy; treat that round as exploration, not progress.
A high `inert_fraction` (the job warns above 25%) means the budget went on
edits that moved nothing — raise the lower bound of `source_radius_rvir` before
concluding anything about the action space.

**Stage 5 — distillation.** The flow is justified **only** if it beats the
Gaussian mixture on realised reward or on catalog diversity at equal Rockstar
budget (`flow_vs_gaussian_mixture.flow_justified`). If it does not, report the
mixture; an 8-dimensional action space did not need a flow.

**Final.** Primary result: improvement in occupation in ≥2 reliable host bins
including ≥1 upper bin (host bins 2 or 3), where "improvement" is a strict
reduction of `|occ - occ_HR|` — overshooting is not progress. Reported
alongside: host preservation, field feasibility, object-realisation rate,
artifacts per proposal, and catalog diversity across seeds. The masked-HR oracle
row is an **unattainable upper bound**, not a competitor.

---

## 6. Stated limitations

* The feasibility filter is legally restricted to quantities a deployed model
  could measure — the frozen base field and the LR input — so the HR-referenced
  constraints (displacement/density power against HR) are `null`. The filter
  therefore bounds how far a candidate moves the LR-visible scales, and says
  nothing about small-scale fidelity against HR.
* The reference-velocity penalty in `action_flow.py` follows the *motivation* of
  ORW-CFM-W2 ([arXiv:2502.06061](https://arxiv.org/abs/2502.06061)) — a W2 trust
  region on the transported distribution — but is a first-order surrogate, not
  that paper's objective or training loop. Do not describe it as a reproduction.
* Stage 6's catalog generator is the **empirical bootstrap**, not a learned
  point process. It copies whole donor hosts' satellite lists, which preserves
  the joint structure of count, mass function and radial distribution for free;
  a learned set-flow replaces it only after this baseline works.
* Artifacts that cannot be attributed to any proposal are charged evenly across
  the proposals in that candidate box. A box-wide change nobody can be blamed
  for individually still has to cost something, or the search learns to make it
  — but it does correlate the rewards of proposals sharing a box.
* **The velocity/displacement balance was re-derived from measurement, and it
  disagrees with the plan's prior.** The plan treats velocity as the junior
  partner, from the HR oracle's 72.9% displacement-only recovery. That does not
  transfer, because the oracle masks a real subhalo's *Lagrangian* member sites —
  particles that are already kinematically coherent, so moving them together
  suffices. This editor selects the *Eulerian*-nearest particles, and measured on
  the selected set8 hosts those carry the host's full dispersion: σ₃D ≈ 613 km/s
  and r₉₀ ≈ 100 kpc, against targets of ≈ 89 km/s and ≈ 128 kpc for the
  ~2.3e11 M⊙/h object being asked for. The set is already the right *size* and
  seven times too *hot*, so cooling (κ_v ≈ 0.82–0.91) is the dominant lever and
  contraction (κ_x ≈ 0) is nearly irrelevant. The original caps — κ_v pinned to 0
  in `disp`, capped at 0.60 in `both` — put the entire viable region outside the
  search space, with 90% of the Rockstar budget on modes that could not bind an
  object. Now `both_mode_cooling_cap: 0.92` and `mode_weights: {disp: 0.15,
  both: 0.60, vel: 0.25}`. `disp` is retained as the control for exactly this
  argument: if it succeeds, the reasoning above is wrong and that matters more
  than the rest of the round.
* **Inert edits.** Selection ("the `n` nearest") and support (`source_radius`)
  are independent by design, so an action with a small enough radius claims
  particles that all fall outside its own window: `w = 0` everywhere, the field
  is untouched, and the candidate still costs a full Rockstar run and still
  scores zero — indistinguishable, from the reward alone, from an edit that was
  applied and failed. A counting argument on the host density (see the comment
  on `editor.bounds.source_radius_rvir`) puts the boundary near `0.08 Rvir`,
  which is where the lower bound is set. Because the argument uses the *mean*
  host density it is approximate at both ends of the radial range, so
  `n_active` is recorded per proposal, `aggregate_cem_round.py` prints the inert
  fraction per mode and warns above 25%, and `evaluate_local_editor.py` carries
  an `inert` column. If that column is large, every other rate in the row is a
  rate over a smaller effective sample than it appears.
* `extract_editor_members.py` reuses `rockstar_particles.py`, which writes a
  small tile-summary JSONL into the reward line's `tile_cache/`. That is a
  by-product of the shared code path (the same frozen base box either line would
  summarise) and is kilobytes.

---

## 7. Tests

```bash
/zfsauton/scratch/yixiz/miniconda3/envs/pjm/bin/python -m pytest tests/reward -q
```

| file | covers |
| --- | --- |
| `test_local_editor.py` | catnorm/physical round trips against `field_to_particles`, periodic minimum-image contraction, exact no-op (bit-for-bit), exact particle-count selection, order-independent claims, exclusion of existing subhalo members, disjoint multi-proposal assignment, periodic-translation invariance, codec bounds and inverse |
| `test_local_reward.py` | proposal↔subhalo one-to-one matching, no reward for existing base subhalos, wrong-host rejection, host-destruction and artifact penalties, periodic reward invariance, the proxy's quarantine, Gate 1 |
| `test_local_cem.py` | determinism, improvement on a synthetic objective, variance floor, refusal to update on ties or infeasibility, resume equivalence |
| `test_action_flow.py` | sample shape, conditioning actually used, diversity, bounded reward weights, reference penalty |
| `test_token_bootstrap.py` | host-normalised libraries, donor selection, subtraction of what SR2 already has |
| `test_local_scripts.py` | the calibrated-constraints guard, the final-box guard, config↔code agreement |
| `test_no_hr_leak.py` | source scan, import graph, and runtime `np.load` poison over a full cycle |
