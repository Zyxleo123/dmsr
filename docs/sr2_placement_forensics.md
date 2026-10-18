# Subhalo placement: what the inverted radial distribution is made of

*Status 2026-10-15. Held-out set9, the 32 spliced tiles (dilated 2 Mpc/h), the 60
most massive HR hosts in that region. "Frozen" is frozen SR2; "fine-tuned" is
`all_blocks_sm_coarse_soft` unless another arm is named. Everything here is CPU
analysis of saved catalogs and fields plus Rockstar/SUBFIND on modified fields; no
generator was trained for it.*

## 1. The headline number was mostly a fragment count

Every gate since 2026-09-14 reports the region's Rockstar subhalos by r/R_vir.
Tuned arms read median 0.35 R_vir with 40% inside 0.25 (HR 1.13, 2.5%). Linking
subhalos whose centre lies inside another's r_vir and counting each group once:

| | core structures (<0.25 R_vir) | fragments / core structure | outskirt structures (0.5-3) |
|---|--:|--:|--:|
| HR | 292 | 2.0 | 17,013 |
| frozen | 546 | 3.6 | 2,381 |
| fine-tuned | 539 | 12.6 | 3,716 |
| every tuned arm | 464-566 | 12-19 | |

Across 33 arms, core fragments per structure tracks host phase-space collapse
(6D kNN host r_k/HR) at Spearman -0.97. Counted as structures the median radius
is HR-like (1.21-1.46 vs 1.26); the shape is still wrong. SUBFIND satellites
agree: arms' median 0.69-0.98 R_vir (HR 0.88), core 1.8-2.7x HR, outskirts
15-23% of HR.

Now in every gate: `structure_distribution` in the verdict json
(`gate_verdict_vs_baseline.py`), and opt-in SUBFIND placement
(`HG_SUBFIND=1 submit_gather_holdout_rockstar.sh` -> `placement_subfind.json`).

## 2. The core: a velocity-temperature defect, fixed by an oracle

`host_phase_profile.py` (k=32 local dispersion, density, clumped fractions):

* density profile within ~10% of HR at every radius, frozen and fine-tuned;
* local sigma 0.68-0.75x HR inside 0.1 R_vir (fine-tuned massive hosts 0.34x,
  with core density 2x), 1.26-1.75x HR beyond 0.75 R_vir;
* HR's profile is universal: 0.3-2% box-to-box scatter over set3-7, 9, 10.

Velocity swap (Rockstar): the tune's gains, core and outskirt, ride ~85-90% on
its cooled velocity field. Velocity oracle (`velocity_profile_oracle.py`:
deviations from the k=32 mean rescaled to HR's stacked profile, positions fixed):

| SUBFIND sats | core | 1-1.5 R_vir | median r/R_vir |
|---|--:|--:|--:|
| HR | 252 | 2,246 | 0.88 |
| fine-tuned | 672 | 387 | 0.69 |
| fine-tuned + oracle | **292** | 451 | **1.02** |
| frozen / + oracle | 148 / 163 | 217 / 298 | 1.01 / 1.04 |

Hosts unchanged; the same rescaling of HR is a no-op. **Lever built:**
`cosmo_sr.features.host_temperature` + `--w-temp` (two-sided, velocity-only,
target = HR on the task's own tiles); arms `all_blocks_temp_full` and
`all_blocks_temp_core` (`submit_temp_arms.sh`).

## 3. The outskirts: the bigger defect, not temperature

Cooling the outskirts to HR's temperature adds 18-37% at 1-1.5 R_vir (HR 5-12x).
A finder-free census (`--peaks`): frozen and fine-tuned outskirts hold HR's
number of dense peaks (>10x shell mean: 0.86-1.43x), with HR's shapes, but
per-peak Rockstar conversion at 0.75-3 R_vir is HR 0.76, frozen 0.32,
fine-tuned 0.40. An HR-fitted logistic on nine peak properties (contrast,
T/|U|, Lagrangian spread, velocity offset, coldness, shape, enclosed mass,
isolated binding) predicts 0.70-0.77 for them: the cause is not among those.
Failures are small isolated knots (HR converts them 86-91%, SR 16-27%).

Knot oracle (`knot_velocity_oracle.py`: each peak's k nearest cooled to 0.3x the
surrounding dispersion, mean kept): conversion at 1-1.25 R_vir 0.35 -> 0.63
(fine-tuned, k=256), 0.22 -> 0.49 (frozen); outskirt structures 0.22 -> 0.29
of HR and 0.14 -> 0.19.

**Low-contrast census (>3x shell mean), 0.75-3 R_vir, corrects an earlier
inference:** the clumps EXIST at every contrast -- frozen / fine-tuned hold
72-82% of HR's 3-10x peaks and 90-117% of the denser ones -- but they fail to
become Rockstar objects at every contrast:

| peaks: n / conversion | 3-10x | 10-30x | 30-100x | >100x |
|---|--:|--:|--:|--:|
| HR | 8449 / 0.15 | 3267 / 0.43 | 2143 / 0.76 | 5072 / 0.98 |
| frozen | 6949 / 0.01 | 2864 / 0.09 | 2220 / 0.15 | 5454 / 0.50 |
| fine-tuned | 6085 / 0.01 | 2949 / 0.09 | 2564 / 0.24 | 5924 / 0.62 |
| fine-tuned + temperature oracle | 6085 / 0.02 | 2949 / 0.12 | 2564 / 0.30 | 5924 / 0.72 |

So the outskirt defect is kinematic at the clump scale across the board (the
density structure is there, the binding velocity structure is not), and a
knot-coherence lever is not capped by missing clumps.

## 4. Measurement problems found on the way

* SR2's output is shifted -0.174 Mpc/h in x vs HR (frozen and every spliced
  arm; the LR host table is aligned). Address-based metrics with 0.15 Mpc/h radii
  are biased by more than their radius. (`frame_offset_check.py`)
* The m2m e77 boxes are translated 0.4-1.5 Mpc/h (differently per arm: a learned
  shift); in-host counts at HR host 168880 are invalid for them.
* Isolation's "SUBFIND 9 in host" was one host; region-wide it is the best-placed
  fine-tune by SUBFIND (core fraction 9%), at the cost of hollowed cores.

## Files

`scripts/features/{placement_forensics, host_phase_profile, velocity_profile_oracle,
knot_velocity_oracle, peak_halo_match, peak_property_analysis, frame_offset_check,
placement_report}.py`, their `scripts/slurm/*_cpu.sbatch`, submitters
`submit_placement_forensics.sh`, `submit_velocity_oracle.sh`, `submit_temp_arms.sh`.
Results under `$DMSR_REWARD_ROOT/{placement_forensics, velocity_oracle, knot_oracle}/`.
