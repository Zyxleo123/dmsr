# kNN-distance statistics for subhalos (2026-09-12 → 2026-09-13)

**Status: finished as a research line. Kept as an advisory stage of the Rockstar gate; not a training objective.**

Four CPU measurement rounds, each run before any GPU was spent, took a differentiable
k-nearest-neighbour distance loss from idea to verdict:

- It **sees** the subhalo deficit that the power spectrum cannot. On HR's 50–200p subhalo
  particles, frozen SR2's k-th neighbours are 1.35–1.88× farther away than in HR (box-wide).
- As a loss it **cannot rank the arms that work.** Across 20 working set9 arms (4,768–7,936
  region subhalos) no kNN term correlates with any gate metric (|ρ| ≤ 0.4, not significant).
- It is a **very strong damage detector.** Across 12 damaging arms, every term tracks every
  gate metric at |ρ| ≈ 0.8–0.9. A two-term threshold flags all 12 and none of the 20.
- The cheap tile-local version is not viable where it matters. Neighbours from the frozen box
  give a p95 error of 41% on r₁₂₈ inside edited tiles.

So it runs next to Rockstar in `submit_gather_holdout_rockstar.sh` and writes
`knn_report.json` per arm (§8). The trainer term exists, but it is off by default and should
stay off (§9).

---

## 1. Origin: the synthetic-particle idea

The starting point was a standalone synthetic experiment. Particles on a Lagrangian lattice are
moved by a displacement field and a kNN search finds each particle's neighbours. The loss is a
sorted (W1) match of `log r_k` for k = 8, 16, 32 between candidate and target. It optimised
displacements directly, with no network and no physics. Two properties made it worth adapting:

- **The gradient survives the search.** Neighbour *identities* come from a detached search.
  The distances are then recomputed on the live tensor, and that is exactly the gradient of an
  all-pairs differentiable `topk`.
- **It needs no catalog and no grid.** That is precisely where the member-gather line was
  blind:
  - `density-power-cannot-see-subhalos`: P(k) matches HR to 1–5% while Rockstar is missing
    most subhalos, because subhalos are sub-grid.
  - `build_tile_pool`'s own comment: the box-wide high-k guard "only constrains the AMOUNT of
    small-scale power, never its placement".

The idea was explicitly **not** a physics-informed loss: no virial, boundness or potential
terms.

## 2. What was built

| file | role |
| --- | --- |
| `src/cosmo_sr/features/knn_cdf.py` | the estimator and loss: `knn_neighbour_ids` (detached, chunked, periodic, self excluded by row identity), `knn_log_distances` (differentiable), `knn_cdf_loss` (quantile-binned sorted match, two-sided squared dead-zone hinge), `interior_queries`, `knn_tile_loss` (with `cand_extra`/`hr_extra` neighbour tiles and `centre="centroid"`), `periodic_centroid`, `box_positions`, `box_neighbours`, `tile_rows_flat` |
| `scripts/features/finetune_member_gather.py` | `--w-knn-unsup` term on the unsupervised tile pool (default **0**), `attach_knn_neighbours`, `eval_unsup_knn`, flags `--knn-{ks,queries,quantiles,dead-zone,buffer-cells,nbr-margin-cells,centre,one-sided,measure}`; `MG_*KNN*` in the train submitter |
| `scripts/features/knn_cdf_diagnostic.py` (+`_plot`) | round 1: box-wide separation + single-tile boundary experiment |
| `scripts/features/knn_cdf_split_diagnostic.py` (+`knn_cdf_split_plot.py`) | rounds 2–3: split by HR owner class, `--region-json`, `loss_forms`, neighbour-tile boundary modes |
| `scripts/features/knn_gate_crosscheck.py` (+`_plot`) | round 4: catalog forensics for every gated arm + Spearman against the kNN terms |
| `scripts/features/knn_gate_report.py` | the gate stage's flags (§8) |
| `slurm/submit_knn_cdf_diagnostic.sh`, `knn_cdf_{diagnostic,plot}_cpu.sbatch` | round 1 |
| `slurm/submit_knn_cdf_split.sh` (`REGION=1` for round 3), `knn_cdf_split{,_plot}_cpu.sbatch` | rounds 2–3 |
| `slurm/submit_knn_gate_crosscheck.sh`, `knn_gate_crosscheck_cpu.sbatch` | round 4 |
| `slurm/gate_knn_cpu.sbatch` + `submit_gather_holdout_rockstar.sh` | the gate stage |
| `slurm/submit_knn_gate_backfill.sh` | the same report for arms gated before the stage existed, or on another box; reads fields on disk, no splice |

Tests (48):

| test file | count | what it pins |
| --- | --- | --- |
| `test_knn_cdf.py` | 22 | gradient vs all-pairs `topk` to 1e-10 and vs finite differences; chunking exactness; self-match by identity; periodicity; translation invariance; the fragmentation regression (§3); neighbour tiles and centroid |
| `test_knn_cdf_diagnostic.py` | 3 | positions and search agree with `tile_particles` and the torch search |
| `test_knn_cdf_split_diagnostic.py` | 5 | classes; region restriction; loss forms, including a cancelling class swap that the unpaired form reads as exactly 0 |
| `test_finetune_knn.py` | 7 | per-box tables never shared; gradient reaches displacements only; neighbours come from the frozen box for the candidate and the HR box for the reference |
| `test_knn_gate_crosscheck.py` | 7 | faithful, fragmenting and misplaced arms on hand-built catalogs; end-to-end run |
| `test_knn_gate_report.py` | 4 | flag logic, plus a replay of all 32 set9 arms through the thresholds |

## 3. Loss design, and what the toy measurements forced

**Sorted, quantile-binned, two-sided.**
- Sorted by rank, not paired by particle id. Per-particle addresses are what the shell family
  overfit (`shell-target-is-real-not-noise`).
- Binned (`n_quantiles`, default 64) so the loss asks only for the distribution's low-order
  shape.
- Two-sided, because central over-concentration is this line's characteristic damage and it
  reads as r_k below HR's.

**The largest k decides whether the minimum is a halo or a field of knots.** In
`test_small_ranks_fragment_large_ranks_build`, one HR clump of 150 particles sits in 300
background particles. The candidate clump starts diffuse, and Adam runs on this loss alone for
400 steps. Every rank set reaches zero loss:

| ks | final loss | largest FOF group of the 150 (HR 112) | clump r50 (HR 0.44) |
| --- | --- | --- | --- |
| (2, 4, 8) | 0 | 13 — fragments | 2.57 |
| (8, 16, 32) — the original script's | 0 | 40 | 1.40 |
| (8, 32, 64) | 0 | 90 | 0.48 |
| (16, 64, 128) | 0 | 107 | 0.49 |

The rule: `max(k)` must be comparable to the particle count of the smallest object to build.
The deficit is at 50–200p, so the default became **(8, 32, 64, 128)**.

**The median is blind.** A 120-particle clump in 520 particles moved the loss about 1500× and
the median ratio by 0.2%. Diagnostics therefore report `ratio_q10`, the bottom decile of
queries, and the whole per-bin ratio curve.

**It is translation-invariant** (tested). It can never place an object, so it is a support
term and not a substitute for `w_centre`.

## 4. Round 1 — box-wide separation and the single-tile boundary (job 49259)

`knn_cdf_set9.{json,png}`: 200k box-wide Lagrangian ids, full-box periodic cKDTree.

| field | `ratio_q10` at k = 8 / 32 / 64 / 128 | loss |
| --- | --- | --- |
| frozen | 1.054 / 1.045 / 1.037 / 1.028 | 3.4e-4 |
| selfhostguard | 1.020 / 1.009 / 1.001 / **0.990** | 2.1e-4 |
| smallhalo | 1.022 / 1.012 / 1.003 / 0.993 | 2.8e-4 |
| condsmall (both rungs) | ~1.10 / 1.10 / 1.09 / 1.08 | 2.0–2.7e-3 |

Only a 3–5% signal. The single-tile boundary result was decisive, though:

| buffer | 0 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- |
| queries missing a neighbour | 56% | 54% | 48% | 43% |
| p95 overestimate of r₁₂₈ | +70% | +66% | +71% | +62% |

**A buffer cannot fix it.** On the one HR tile checked (the box-centre tile), the median
displacement is **11.2 Mpc/h**, against a 12.5 Mpc/h tile side. At z=0 a tile's particles are
not in its cube, so much of the material around a query comes from other tiles. Round 2's
same-field neighbour test (§5) confirmed this is the whole cause: adding other tiles'
particles makes the error exactly zero.

## 5. Round 2 — whose clumping? (job 51232)

`knn_cdf_split_set9.{json,png}`. Particles are classed by the HR owner array (innermost binding
halo), 100k per class, and the same ids are measured in every field. Paired median r_k ratio to
HR, box-wide:

| HR owner class | frozen, k = 8 → 128 |
| --- | --- |
| subhalo 50–200p | **1.88 → 1.35** |
| subhalo < 50p | 1.85 → 1.19 |
| host | 1.43 → 1.29 |
| unbound | **0.73 → 0.85** |

Frozen SR2 moves clustering out of bound objects and into their surroundings. A pooled
distribution match cancels most of it, which is why round 1 saw only 3–5%.

Boundary result: same-field neighbour tiles make tile-local r_k exact at every margin. Missing
neighbours fall to 3.6% at margin 0, 0.1% at margin 4 and 0 at margin 8, at a cost of 2.3× /
2.9× / 3.6× a tile's particles. Centring the cube on the centroid **alone** does not help.

**Confound.** Every arm field is the frozen box with only 32 tiles (6.25%) spliced in; the tile
list is in each field's `.json`, and all four arms used the same 32. Box-wide samples therefore
diluted arm-vs-frozen deltas about 16×. None of the 16 random boundary tiles was edited, so the
frozen-neighbour check came out trivially exact.

## 6. Round 3 — the spliced region only (job 51399)

`knn_cdf_split_region_set9.{json,png}` (`REGION=1`).

| loss form | frozen | selfhostguard | smallhalo | condsmall_ab | condsmall_fine |
| --- | --- | --- | --- | --- | --- |
| unpaired (the form built) | 0.0004 | 0.014 | 0.015 | 0.38 | 0.24 |
| class-conditioned, mass-weighted | 0.111 | 0.130 | 0.130 | 0.48 | 0.33 |
| class-conditioned, equal | 0.129 | 0.132 | 0.140 | 0.50 | 0.34 |
| paired | 0.839 | 0.892 | 0.890 | 1.14 | 0.99 |

Every form scored `selfhostguard`, the arm that passed Rockstar, as worse than frozen. By class:
- **50–200p subhalos improve a little:** 0.151 → 0.137. The paired ratio goes 1.54 → 1.41 at
  k=8 and 1.11 → 1.00 at k=128.
- **Hosts get much worse:** 0.057 → 0.134. Host particles end up over-concentrated, at a paired
  ratio of 0.84–0.87.
- **Paired is mostly realisation noise:** frozen already scores 0.84, and the arms move it
  about ±6%.
- **Rockstar is a 50-particle threshold; kNN is continuous.** The gate passes long before the
  distances converge.

**Frozen neighbours fail inside edited tiles.** Same-field neighbours are exact (cluster tiles
cost 5.7–8.0× a tile). Frozen neighbours give a median error of 0 but **p95 41% (r₁₂₈),
74% (r₈), p99 77%**, flat in margin. A trainer term would need live neighbours: a no-grad
forward of the 26 surrounding tiles, or block training.

## 7. Round 4 — 32 arms against their catalogs (jobs 51576, 51577)

`knn_cdf_region_allarms_set9.json` holds the kNN terms for all 33 set9 fields on disk.
`knn_gate_crosscheck_set9.{json,png}` holds forensics for 38 gated catalogs, in the gate's
region (spliced tiles + 2 Mpc/h) with the gate's match rule (max(r_vir, 0.15 Mpc/h),
num_p within 2×).

Pooled across all 32 arms, the kNN subhalo-class term correlates at ρ = −0.79 with region
subhalos and +0.61 with lost ≥200p halos. That is the family split, not ranking power:

| group | arms | region 50–200p subs (HR 9,024; frozen 2,762) | frozen's ≥200p lost (of 4,809) | kNN vs gate, within the group |
| --- | --- | --- | --- | --- |
| damaging | fine_cond, fine_condctl, fine_condsmall, fine_smallctl, condsmall, smallctl, hostguard, condboth, condshell, selfshell, abl_novirial(±step2000) — 12 | 2,327–4,019 | 513–742 | **\|ρ\| ≈ 0.8–0.9, every term vs every metric** |
| working | selfhostguard, smallhalo, softshell, softshell10, softshell30, and the 15 loss ablations other than novirial (nobound, nocentre, nod6, nohighk, nolow, norrms, nosigmav, repro, most with a step-2000 twin) — 20 | 4,768–7,936 | 152–743 | **\|ρ\| ≤ 0.4, not significant** |

`abl_nod6` has the most subhalos (7,936), the fewest lost halos (152–234) and the best subhalo
class term (0.088 vs frozen 0.151). Its unbound class term, however, is 2.6× frozen's.

**Corrections to earlier readings:**
- `condsmall` does **not** gain 50–200p subhalos over frozen in the region (2,587 vs 2,762).
  Its "+43%" (`cond-channel-gate-result`) was against its own `smallctl` control. The kNN
  "worse than frozen" therefore agrees with the counts.
- The fragmentation hypothesis is **refuted** (40 fragment candidates). The cond family
  dissolves ≥200p halos without making small ones.

**The matching is address-limited.** Frozen precision is 6%, working arms reach 14–18%, and HR
recall is at most 5%. SR and HR positions differ by more than the 0.15 Mpc/h radius, the same
fact as `centre-offset-is-an-address`. The fragment-candidate count just scales with the number
of unmatched objects, so it cannot tell real subhalos from misplaced ones.

## 8. The gate stage: `knn_report.json`

`submit_gather_holdout_rockstar.sh` submits one `gate_knn_cpu.sbatch` per arm, after
`afterok:<arm splice>:<frozen splice>`, as a sibling of Rockstar; `KNN_REPORT=0` skips it.

It measures HR, frozen and the arm in the spliced tiles with the calibration settings: ks 8 32
64 128, 100k per class, 200k uniform, 32 quantiles, dead zone 0.02, seed 0. It writes
`$RUN_DIR/knn_report.json`, `knn_region.json` and `knn_region.png`.

**Every path exits 0.** Missing inputs, a frozen control spliced on different tiles, or a
failed diagnostic all print the reason. Rockstar stays the verdict.

| report field | meaning | set9 working | set9 damaging | flag |
| --- | --- | --- | --- | --- |
| `sub50_x_frozen` | class-conditioned loss on HR's 50–200p subhalo particles ÷ frozen's | ≤ 1.13 | ≥ 1.45 | `DAMAGE` when > 1.30 **and** the next row > 1.75 |
| `ge200_x_frozen` | the same on ≥200p subhalo particles | ≤ 1.34 | ≥ 2.16 | (the other half of `DAMAGE`) |
| `host_r128` | paired median r₁₂₈ on HR host particles vs HR (frozen 1.048) | 0.63–1.07 | > 1.14 | `hosts_diffused` > 1.10; `hosts_overconcentrated` < 0.85 |
| `unbound_x_frozen` | unbound-class loss ÷ frozen's | ≤ 1.47, except `abl_nod6` 2.55–2.67 | ≤ 0.46 | `unbound_crowded` > 2 |

`test_thresholds_separate_the_set9_families` replays all 32 arms: `DAMAGE` matches the damaging
family exactly.

**Limits.**
- Calibrated on **set9 only**; reports for any other box say UNCALIBRATED.
- It flags damage but does not rank working arms.
- It costs one CPU queue slot per arm (`cpu-partition-50-job-cap`).
- Arms gated before 2026-09-13 have no `knn_report.json`; their set9 numbers are in
  `knn_cdf_region_allarms_set9.json`.

## 9. The trainer term: built, off, and why it stays off

`--w-knn-unsup` defaults to 0.

`--knn-measure` defaults to **on**. Any run with `--unsup-tiles-per-box > 0` therefore loads
the frozen and HR boxes once per box at startup to attach neighbour tiles (about 1.6 GB each
per load) and logs `unsup/*/knn_q10_k*`. Pass `--no-knn-measure` to skip that.

Defaults: `--knn-centre centroid --knn-nbr-margin-cells 8 --knn-buffer-cells 0`, with
neighbours from the cached frozen box (candidate) and the HR box (reference).

Do not turn it on without new evidence, for three reasons from §6 and §7:
1. It has no ranking power inside the working family, the regime a fine-tune operates in.
2. Every form penalised the Rockstar-passing arm, because of its host over-concentration.
3. Frozen-box neighbours are wrong in edited tiles (p95 41%), and live neighbours cost about
   27× the forward pass.

If the line is ever revived, the order is:
- live neighbours first (a no-grad forward of adjacent tiles, or 2×2×2 blocks with interior
  queries);
- then a one-sided guard referenced to **frozen** rather than HR, using the §8 thresholds;
- never a two-sided HR match.

## 10. Reproduce

```bash
bash scripts/slurm/submit_knn_cdf_diagnostic.sh          # round 1  (~7 min CPU)
bash scripts/slurm/submit_knn_cdf_split.sh               # round 2  (~26 min CPU)
REGION=1 bash scripts/slurm/submit_knn_cdf_split.sh      # round 3  (~31 min CPU)
bash scripts/slurm/submit_knn_gate_crosscheck.sh         # round 4  (~35 min CPU)
SKIP_KNN=1 bash scripts/slurm/submit_knn_gate_crosscheck.sh   # redo forensics only
# gate stage: automatic in every submit_gather_holdout_rockstar.sh chain (KNN_REPORT=0 to skip)
BOX=set10 bash scripts/slurm/submit_knn_gate_backfill.sh    # reports for already-gated arms (MAX_JOBS=12)
```

Artefacts live under `$DMSR_REWARD_ROOT/member_gather/knn_cdf/` and, per arm,
`$DMSR_REWARD_ROOT/member_gather/<arm>/<role>_<box>/knn_{report,region}.json`.

## 11. Open

- **Out-of-sample check of the thresholds (submitter built, not yet run).** set10 has three
  gated working arms (self, selfhostguard, smallhalo) plus a frozen control, all with fields on
  disk: `BOX=set10 bash scripts/slurm/submit_knn_gate_backfill.sh`. It tests specificity (none
  should be flagged `DAMAGE`) but not sensitivity, because set10 has no damaging arm.
- **Why `abl_nod6` crowds unbound particles** (2.6× frozen's unbound term) while producing the
  best counts. Not investigated.
- A real-vs-misplaced test for subhalos needs particle-level membership of the arm catalogs.
  Rockstar particle files exist only for HR and base, so position matching is the ceiling.

## 12. Round 5 — phase space (built 2026-09-13, not run)

**Hypothesis.** Rounds 1–4 measured *position* kNN. A 50–200p subhalo inside a
host is barely denser than the host around it, so most of its 128 nearest
neighbours are host particles and its `r_k` mostly measures the host. That would
explain all three reasons in §9: the subhalo-class term moved only 9% while
counts doubled, the working family showed no ranking, and hosts dominated every
loss form. Rockstar separates subhalos in **phase space**. In 6D,
`(x, v/λ)` with one fixed λ, a subhalo's particles are each other's neighbours.
Virial scaling (`V_vir/r_vir ≈ 10 H₀ = 1000 km/s per Mpc/h`) lets one λ serve
every mass. `test_phase_space_sees_a_subhalo_that_position_cannot` pins the
mechanism on a toy.

**Built.**
- `knn_cdf_split_diagnostic.py --phase-lambda L`: 6D, periodic in x, open in v.
- `--boundary-by-class`: class-stratified tile-locality error per class.
- `--resume`: fields already measured are skipped.
- `knn_phase_verdict.py`: stop rules from saved JSON, using round 4's gate counts.
- `submit_knn_phase.sh`: λ ∈ {500, 1000, 2000} over all 33 set9 fields, a 3D
  per-class boundary reference, and a verdict job that runs `afterany`.

**Stop rules** (`knn_phase_verdict_set9.json`):

| rule | question | pass |
| --- | --- | --- |
| A | does frozen's sub_50_200 look more wrong in 6D than 3D? | max \|log paired ratio\| > 3D's (0.428) |
| B | does the class_cond[sub_50_200] term rank the 20 working arms? | ρ ≤ −0.6, p < 0.05 vs region 50–200p count (3D: −0.18, p 0.46) |
| C | are subhalo particles' 6D neighbours tile-local? | `tuned_centroid_frozen` p95 r_kmax on sub_50_200 < 0.10 (3D pooled: 0.41) |

B is the decisive rule. If B fails at every λ, both spaces have failed and §9
stands. If B passes, build the class-conditioned 6D term on the member-gather
sets. C then decides whether frozen neighbours are good enough.

```bash
bash scripts/slurm/submit_knn_phase.sh                 # 3 x 6D + 3D ref + verdict
RESUME=1 LAMBDAS="1000" KNN_3D_REF=0 bash scripts/slurm/submit_knn_phase.sh   # after a time limit
```

### Round 5 result (jobs 51941–51945, 2026-09-13)

**Pre-registered verdict: B fails at every λ.** A passes, C narrowly fails.

| space | A: frozen sub_50_200 paired r_k/HR (k=8 → 128) | B: working ρ vs subs (k-mean) | B: working ρ vs HR-matched subs | C: frozen-neighbour p95 (sub_50_200) |
| --- | --- | --- | --- | --- |
| 3D | 1.53 → 1.11 | −0.18 (p 0.46) | −0.16 | 0.35 |
| λ=500 | 2.73 → 1.53 | −0.31 (p 0.18) | −0.49 | 0.13 |
| λ=1000 | 2.89 → 1.53 | −0.45 (p 0.047) | −0.62 | 0.12 |
| λ=2000 | 2.83 → 1.42 | −0.53 (p 0.017) | −0.66 (p 0.002) | 0.14 |

- **The signal is real and grows with λ, but 3D (λ → ∞) is the worst.** The best λ is
  somewhere at 2000 or above; it was not bracketed.
- **k=128 has the wrong sign** in every space: ρ is +0.48 in 3D and +0.65 at λ=2000.
  k ∈ {8, 32, 64} at λ=2000 gives −0.57/−0.55/−0.51 vs subs and −0.68/−0.67/−0.65 vs
  matched. For a 50–200p object, r₁₂₈ reaches past the object into the host. *This is a
  post-hoc observation, not the registered rule.*
- **In 6D the working arms trade hosts for subhalos.** Within the working family, the
  host, unbound and ≥200p class terms rise with subhalo count at ρ +0.85 to +0.93. The
  3D ρ values were 0.26–0.40. Host paired r_k/HR is 0.51–0.65 in 6D vs 0.83 in 3D:
  hosts are over-concentrated in phase space, which fits the measured collapse of
  small-scale velocity power. Every kNN form that includes hosts scores the arms that
  make more subhalos as worse.
- **The POOLED (`unpaired`) form — the original loss, the catalog-free one — has the
  wrong sign everywhere, and 6D makes it worse.** Recomputed 2026-09-14 from the saved
  round-5b JSONs with the same `ranking()` and the same 20 working arms, because
  "can we just train the original loss and drop the label" is the obvious question and
  deserved a number rather than an argument:

  | space | `unpaired` (og, catalog-free) | `class_cond[sub_50_200]` | `class_cond_equal` |
  | --- | --- | --- | --- |
  | 3D | **+0.451** (p 0.046) | −0.233 (p 0.32) | +0.322 (p 0.17) |
  | λ=2000 | **+0.905** (p 4e-08) | −0.549 (p 0.012) | +0.767 (p 8e-05) |
  | λ=4000 | **+0.878** (p 4e-07) | **−0.647 (p 0.002) PASS** | +0.759 (p 1e-04) |
  | λ=8000 | **+0.847** (p 3e-06) | −0.595 (p 0.006) | +0.759 (p 1e-04) |

  Positive means *lower pooled loss ↔ fewer subhalos*: descending the og loss descends
  away from the goal, at p ~ 1e-7 — not a noise result that a longer run might flip.
  Only the `sub_50_200`-only charge has the right sign; `class_cond_equal`, which
  averages all five classes, is +0.759. The discriminating variable is not pooled vs
  conditioned, it is **whether host particles are in the term at all**.
- **C:** frozen neighbours cut the error 3× vs 3D (0.35 → 0.12–0.14), and it is flat in
  margin. The residual comes from stale neighbour positions, not from missing
  neighbours. The tile alone gives 0.22–0.30 with 77% of queries incomplete.
- **Caveat:** 7 of the 20 working arms are step-2000 twins of other arms, so n = 20
  overstates independence.

### Round 5b — pre-registered 2026-09-13, before any run

**Motivation:** round 5 showed B rising with λ, with the optimum unbracketed, and
k=128 carrying the wrong sign.

**Settings:** ks = (8, 16, 32, 64); λ ∈ {0 (3D, all arms), 2000, 4000, 8000}; same
fields, region, seeds, and verdict script.

```bash
LAMBDAS="0 2000 4000 8000" KS="8 16 32 64" KNN_3D_REF=0 bash scripts/slurm/submit_knn_phase.sh
```

**Rules, fixed now:**
- **B passes** at a λ when the working-family ρ of class_cond[sub_50_200] (k-mean) vs
  region 50–200p subhalos is ≤ −0.6 with p < 0.05. The count is the gate metric; the
  matched count is reported but does not decide.
- If B passes at more than one λ, the smallest |ρ| gap decides.
- **The λ=0 row must fail B.** If it passes, dropping k=128 alone explains the gain and
  6D adds nothing.
- **If no λ passes,** kNN stays advisory (§9) and the line closes.
- **Also read**, not a gate: whether the host class-term correlation (+0.85 to +0.93
  in round 5) persists at large λ. This is the host phase-space trade-off.

Output: `knn_phase_verdict_k8-16-32-64_set9.json`.

### Round 5b result (jobs 52090–52094, 2026-09-13) — **B PASSES at λ=4000**

ks (8, 16, 32, 64), working-family ρ of `class_cond[sub_50_200]` vs region 50–200p
subhalos (pass: ≤ −0.6, p < 0.05):

| λ | ρ (n=20) | p | ρ without step-2000 twins (n=13) | per k = 8/16/32/64 | host-term ρ | C: frozen-nbr p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (3D control) | −0.23 | 0.32 | −0.31 | −0.33/−0.31/−0.22/−0.14 | +0.29 | 0.35 |
| 2000 | −0.55 | 0.012 | −0.61 | −0.57/−0.57/−0.55/−0.51 | +0.91 | 0.14 |
| **4000** | **−0.65** | **0.002** | −0.58 | −0.65/−0.67/−0.65/−0.57 | +0.91 | 0.16 |
| 8000 | −0.60 | 0.006 | −0.58 | −0.63/−0.63/−0.60/−0.25 | +0.93 | 0.21 |

- **The registered rule passes at λ=4000 and only there.** The λ=0 control fails, so the
  gain is phase space, not the dropped k=128. ρ peaks at λ=4000 and falls off on both
  sides, and every rank agrees.
- **Robustness:** dropping the 7 step-2000 twins leaves ρ = −0.58 (p 0.039), just under
  the threshold on n=13. The ranking is not an artefact of duplicated arms, but n is small.
- **The host trade-off is confirmed and large** (ρ +0.91). Host particles in the working
  arms sit at 0.51–0.66× HR's phase-space neighbour distance; frozen sits at 1.05–1.10.
  Arms gain subhalos while over-concentrating hosts in phase space. **So a first term
  must be conditioned on the subhalo classes only**, with hosts monitored, not penalised:
  a host term would pull against every gain (reversing §9's suggestion).
- **C still fails** (0.16 at λ=4000) and worsens with λ. Same-field neighbours are exact
  at 5.4× a tile's particles; frozen-box neighbours are the cheap approximation.

## 13. The training term, round-5b form (built 2026-09-13, not yet run)

§9's term (position-only, pooled) stays off and should never be turned on. What
round 5b licensed is a different statistic, and it is now what `--w-knn-unsup`
charges when the two new flags are set:

| flag | measured setting | why |
| --- | --- | --- |
| `--knn-phase-lambda` | **4000** (km/s per Mpc/h) | ρ −0.65 vs −0.23 for position only; peak of the λ scan |
| `--knn-classes` | **sub_50_200** | the class the rule passed on. Hosts must NOT be listed: their term rises with subhalo count (ρ +0.91) |
| `--knn-ks` | **8 16 32 64** | k=128 has the wrong sign in every space |

Both default to off (λ=0, no classes), so every existing arm is unchanged, and
the trainer prints a warning if the term is switched on in the pooled or
position-only form.

**What changed in the code.** `knn_cdf.py`: the search and the differentiable
distance take 6-column points, wrapping only the position axes (velocity is not
periodic), so **gradient reaches the velocity field as well as the positions** —
the first loss in this line that does. `knn_tile_loss` takes `classes` and
`class_ids` and charges each class separately on its own queries, averaging the
classes equally; the query cube is still placed by the whole tile. Added
`box_phase` (6-column neighbour material), `classify_owner` and `OWNER_CLASSES`.
`finetune_member_gather.py`: `attach_knn_classes` caches each box's class vector
(`classes_<box>.npy`), neighbour tables carry velocity, and the per-tile term
builds `(x, v/λ)`. Env: `MG_KNN_PHASE_LAMBDA`, `MG_KNN_CLASSES`.

Tests (+16, 65 kNN tests total): 6D search vs brute force with periodicity in x
only; 6D gradient vs finite differences, including the velocity columns;
translation AND Galilean invariance; class conditioning charges the named class,
averages classes equally, and skips a class too small to fill its bins; the
neighbour tables carry the scaled velocity; `classify_owner` matches the
diagnostic's cuts.

**Next, in order:** a `SHAKEOUT_ONLY=1` measurement run to read the term's scale
before choosing a weight; then arms at that weight, each behind the cheap
Rockstar A/B gate. Its controls are already gated on set9 (`all_blocks_self`,
`abl_repro`). Open: C failed, so the candidate's neighbours are the frozen box's
(p95 16% on subhalo particles); if an arm's gain stalls, live neighbours are the
next thing to buy.
