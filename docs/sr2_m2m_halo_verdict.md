# Halo-finder verdict on the map2map from-scratch arms

Three arms, all forked from the same `state_27` of the colleague's own tree, all
scored on **held-out set9 at matched epoch 77**:

| arm | run dir | what it adds |
| --- | --- | --- |
| baseline | `m2m/m2m_theirs` | nothing — their command, unedited |
| knn50 | `m2m/m2m_theirs_knn50` | kNN-CDF term, `--extra-weight 50000` |
| member-gather | `m2m/m2m_theirs_mg` | member-gather minimal set, `--extra-weight 30` |

Final epochs were 147 / 77 / 124; 77 is the deepest all three reach.

## Why halo finders at all

`docs/sr2_m2m_baseline.md` ranks these arms on P(k), and that ranking cannot see
the objects this project exists to produce. Frozen SR2 already matches HR's
density spectrum to 1-5% at every resolvable k, and subhalos are sub-grid — they
live above k = 32, where the 64^3 LR input carries nothing. A field can match HR's
spectrum everywhere it is measurable and still have the wrong halos in it.

Rockstar and SUBFIND read particles. An m2m candidate is a **whole box**, so
unlike every member-gather arm there is no splice and no seam: the catalog is of
the model's own field everywhere.

## The headline, stated carefully

**From-scratch training produces roughly the right NUMBER of self-bound
satellites box-wide — and that number is the wrong shape in two different ways.**

SUBFIND, whole box, satellites >= 50 particles:

| arm | FOF groups | centrals | sats >=50p | grp/HR | cen/HR | sats/HR |
| --- | --: | --: | --: | --: | --: | --: |
| **HR (truth)** | 144,808 | 137,908 | 17,068 | 1.000 | 1.000 | 1.000 |
| frozen SR2 | 142,312 | 102,332 | 4,420 | 0.983 | 0.742 | **0.259** |
| baseline e77 | 51,408 | 37,521 | 18,310 | 0.355 | 0.272 | **1.073** |
| knn50 e77 | 44,468 | 26,511 | 14,575 | 0.307 | 0.192 | **0.854** |
| member-gather e77 | 51,674 | 37,879 | 20,985 | 0.357 | 0.275 | **1.229** |

Frozen SR2 is at 0.26x — that deficit is what this project has been chasing. The
from-scratch arms reach 0.85-1.23x with no member loss, no catalog and no splice.
**But read on before quoting it.**

### Wrong shape 1 — by location, which matters most

`subfind_compare` inside host 168880 (aperture = its R_vir, >= 50p):

| catalog | Rockstar | SUBFIND | vs HR |
| --- | --: | --: | --: |
| **HR** | 369 | **146** | 1.00 |
| baseline e77 | 172 | **14** | **0.096** |
| frozen SR2 | 20 | **3** | 0.021 |

Inside a massive host the arm has a **10x shortfall**, not parity. It is a real
4.7x improvement on frozen SR2 and nowhere near closing the gap. The SR2 deficit
is known to **scale with host mass**, so this is the number that counts and the
box total is the one that misleads.

### Wrong shape 2 — by mass

SUBFIND satellites, whole box, baseline arm against HR:

| bin | HR | baseline | ratio |
| --- | --: | --: | --: |
| 50-100p | 8,129 | 10,631 | **1.31** |
| 100-200p | 4,381 | 4,350 | 0.99 |
| 200-500p | 2,769 | 1,774 | **0.64** |
| 500-2000p | 1,416 | 915 | **0.65** |
| 2000+p | 373 | 640 | 1.72 |
| total | 17,068 | 18,310 | 1.07 |

Over-produces the smallest bin by 31%, under-produces 200-2000p by ~35%. The arm
manufactures cheap small clumps in low-density regions while failing inside
clusters.

## Rockstar overstates, and by how much

Rockstar's own counts look like a win and are not:

| arm | halos | hosts | subs | host % | subs/HR | hosts/HR |
| --- | --: | --: | --: | --: | --: | --: |
| **HR** | 314,238 | 212,771 | 101,467 | **67.7%** | 1.000 | 1.000 |
| frozen SR2 | 212,017 | 166,106 | 45,911 | **78.3%** | 0.452 | 0.781 |
| baseline e77 | 195,568 | 29,980 | 165,588 | **15.3%** | 1.632 | 0.141 |
| knn50 e77 | 186,847 | 23,079 | 163,768 | **12.4%** | 1.614 | 0.108 |
| member-gather e77 | 204,359 | 29,182 | 175,177 | **14.3%** | 1.726 | 0.137 |

The 1.6-1.7x "subhalo surplus" is **re-parenting**. Total halos are LOWER than
HR; the arms file 85% of them as substructure because the field is concentrated
enough that most halos land inside another's virial radius. Note the baseline is
in the same band as both term arms, so this is a property of the from-scratch
adversarial objective, not of anything we added.

SUBFIND confirmation rate in-host, by Rockstar mass bin (confirmed / Rockstar):

| bin | baseline | knn50 | member-gather |
| --- | --- | --- | --- |
| 50-100p | **0/85** | **0/56** | **0/31** |
| 100-200p | **0/37** | **0/23** | 2/18 |
| 200-500p | 5/28 | 0/21 | 0/8 |
| 2000+p | 6/6 | 6/7 | 2/2 |

Overall rate: HR **0.393**, baseline 0.081 (0.21 of HR's), knn50 0.051 (0.13),
member-gather 0.134 (0.34). HR itself only reaches 0.39, so Rockstar over-reports
even on truth — the *relative* number is the fair one, and every arm is 3-8x
worse than HR. **Not one 50-100p Rockstar object is confirmed, for any arm.**

## Radial placement — the defect nothing has fixed

`subhalo_radial.py`, whole box, Rockstar catalogs:

| catalog | subhalos | median r/R_vir |
| --- | --: | --: |
| **HR** | 101,467 | **1.20** |
| frozen SR2 | 45,911 | 0.62 |
| baseline e77 | 165,588 | 0.56 |
| knn50 e77 | 163,768 | 0.60 |
| member-gather e77 | 175,177 | 0.54 |

Fraction per r/R_vir bin:

| catalog | 0-.25 | .25-.5 | .5-.75 | .75-1 | **1-1.5** | 1.5-2 | 2-3 | 3-5 |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| **HR** | **0.027** | 0.087 | 0.134 | 0.147 | **0.251** | 0.153 | 0.143 | 0.057 |
| frozen SR2 | **0.290** | 0.178 | 0.087 | 0.054 | 0.096 | 0.087 | 0.119 | 0.089 |
| baseline | **0.231** | 0.258 | 0.153 | 0.088 | 0.095 | 0.055 | 0.062 | 0.057 |
| knn50 | **0.213** | 0.250 | 0.157 | 0.095 | 0.103 | 0.058 | 0.065 | 0.060 |
| member-gather | **0.257** | 0.246 | 0.144 | 0.084 | 0.095 | 0.055 | 0.062 | 0.058 |

HR peaks at 1-1.5 R_vir and puts 2.7% inside 0.25; every candidate peaks inside
0.5 and puts 21-29% there. Beyond 1.5 R_vir HR holds 35%, frozen SR2 29%, the
from-scratch arms ~17-18%. **From-scratch does not fix placement and is slightly
worse than frozen SR2.** Three unrelated approaches — frozen SR2, the fine-tuned
gather arms (0.30-0.35) and from-scratch — all fail this axis.

## Neither loss term helps

On displacement P(k) at matched epoch 52, `k_half` (largest k holding half of
HR's power) ranks **baseline 129 > member-gather 98 > knn50 64**, and each arm's
own trajectory agrees: baseline climbs 129 -> 145 while member-gather falls
98 -> 69 over 63 further epochs. The halo finders say the same thing from a
different direction. Both terms lose to doing nothing.

**The arms do not order the same way on every axis**, which is why no single
number is safe: member-gather is worst on whole-box abundance accuracy and on
radius, but has the best in-host confirmation rate (0.34 of HR's vs baseline's
0.21).

## Two metrics that ranked these arms BACKWARDS

Both cost a wrong conclusion in this line; neither is safe alone.

* **`ratio_highk`** averages P/P_HR over the top third of k (~170-256), where
  every from-scratch arm holds 3-8% of HR's power. Comparing 0.077 against 0.063
  there ranked the three arms in exactly the reverse of `k_half`. Use `k_half`,
  or the curve.
* **The box-total satellite count** hides a 10x in-host shortfall and an inverted
  mass function. Report the in-host count and the mass bins.

## Running it

```bash
# box dump (GPU) -> Rockstar (CPU), chained afterok, arms as siblings
bash scripts/slurm/submit_m2m_halos.sh                 # M2M_HALO_EPOCH, _BOX, _ARMS
# SUBFIND; m2m_* targets are whole-box candidates (no splice, no tiles.npz)
SF_TARGETS="m2m_base m2m_knn m2m_mg" bash scripts/slurm/submit_subfind.sh full
# radial distribution; needs the symlink note below
sbatch scripts/slurm/subhalo_radial_cpu.sbatch <envfile>
# P(k) overlay across arms, CPU, from the saved npz
sbatch scripts/slurm/m2m_power_compare_cpu.sbatch <envfile>
```

Gotchas, all of them measured here:

* `catalog_summaries.py` needs `configs/reward/reward.yaml` — the direct-finetune
  config has no `data` section and dies with `KeyError: 'data'` *after* the box
  is built.
* `compare_gather_catalog.find_catalogs` and `subhalo_radial.py` look for
  candidates **only** under `flow_rockstar/halos/`, so m2m catalogs need a
  symlink there. Create it before the compare stage runs, not after.
* `np.save` appends `.npy` when the name does not end in it, so writing to
  `foo.npy.tmp` produces `foo.npy.tmp.npy` and the rename fails *after* 3.2 GB
  is written. `m2m_box_dump.py` hands it an open file object instead.
* A box is 3.2 GB and a GADGET snapshot 3.5 GB. Delete them once catalogs exist
  — the catalogs are the measurement and the box regenerates in ~12 min from a
  fixed seed. 46 stray fields caused a quota death on 2026-09-14.
* `m2m_box_dump_gpu.sbatch` asks `--mem=8G`: MEASURED 3.9 GiB. The first version
  asked 24G by analogy with the field report, which was 6x over.

## Artifacts

| what | where |
| --- | --- |
| Rockstar catalogs | `m2m/rockstar/halos/set9__candidate__<arm>_e77/` |
| Rockstar summaries | `catalog_cache/set9__candidate__<arm>_e77.json` |
| SUBFIND catalogs | `subfind/set9__m2m_*/subfind_catalog.{json,npz}` |
| SUBFIND cross-check | `subfind/compare/set9__m2m_*.json` |
| radial profile | `m2m/radial/{summary.json,*.npz,*.png}` |
| P(k) overlay | `m2m/power_compare_epoch52_set9.png` |
| slices | `m2m/<arm>/slices_state_77_set9_{displacement,velocity}.png` |

## What is actually open

The trade is now measured on both sides: **frozen SR2 has the halo population
(0.98x FOF groups, correct-ish radii) and no substructure (0.26x); from-scratch
has box-total substructure (1.07x) and neither the population (0.36x groups) nor
the placement (0.54 vs 1.20 R_vir) — and still a 10x in-host shortfall.**

So the question is no longer which loss term to add. None of the three tried
helps, and the plain baseline beats both. It is whether the over-concentration
can be fixed while keeping the satellites, and nothing in the current objective
can see either side of that trade.
