# Are the gained subhalos real? Two independent tests

## The question

Rockstar says the tuned arms add hundreds of subhalos inside the trained host
(`ratio_total_tuned` 0.99 against HR's, up from the frozen box's 0.054). The
projected density figures show nothing where those subhalos are. Both cannot be
straightforwardly true, and there are exactly two ways out:

1. **the picture is uninformative.** The host panels project the host's whole
   ~25 Mpc/h Lagrangian column. A 200-particle subhalo is ~0.1 Mpc/h across, so
   that column stacks ~250 subhalo-diameters of unrelated material over it. The
   contrast offered to the eye is `(subhalo + column) / column`, a percent-level
   tint **even when the subhalo is perfectly real**.
2. **the catalogue is wrong.** Rockstar links particles in six dimensions. A
   cold, coherent set of particles with no spatial overdensity behind it is a
   halo to Rockstar and nothing to anyone else. A fine-tune that sharpened
   velocities without moving mass would manufacture those in bulk.

These are different failures with different fixes, and neither the existing
figures nor the existing gate can tell them apart. So each gets its own test,
and each test carries its own control.

---

## Test 1 — project the subhalo, not the host

`scripts/features/subhalo_zoom_data.py` + `subhalo_zoom_plot.py`,
launched by `scripts/slurm/submit_subhalo_zoom.sh`.

For each reported subhalo inside the host's `R_vir` it takes a window a few
subhalo radii wide and a slab **only as deep as the subhalo**, and bins it in
all three fields — tuned, frozen, HR — in the same window. Then it measures the
same excess at five depths:

| measurement | what it is |
|---|---|
| `contrast_3d` | `rho(<r) / rho(3r..5r)`, no projection at all |
| ladder rung 1 | the same after projecting a slab of ±1 `R_vir,sub` |
| rungs 2–4 | ±2, ±5, ±10 `R_vir,sub` |
| last rung | the full column — what the host figures show |

**The ladder is the experiment.** A real object buried by projection starts high
and decays towards 1 as the slab opens. A phantom sits at 1 at *every* depth,
including the thinnest, where there is no projection left to blame.

Each rung reports two numbers, because "is it there" and "would I see it" are
different questions:

* `ratio` — the overdensity. What the colour map shows.
* `snr` — the excess over the local background in units of its own Poisson
  noise, `(n_in - rho_bg·A) / sqrt(rho_bg·A)`. A 5% excess is invisible at
  snr 1 and a solid detection at snr 40.

### The control (run 2, 2026-09-23) — and a misreading worth recording

The `hrtruth` target runs the measurement on **HR's own subhalos**. Those are
real by definition, so what they score is what "real" looks like here. Read it
first.

**It passes.** HR subhalo centres measured in the HR field, `host_shell_core`:

| HR subhalo | R/R_vir,host | ratio | snr |
|---|---|---|---|
| 13200p | 0.81 | 49.5× | 491 |
| 3950p | 0.95 | 82.1× | 253 |
| 7884p | 0.33 | 14.5× | 232 |
| 321p | 0.44 | 11.3× | 32 |
| 780p | 0.39 | 9.7× | 51 |
| 126p | 0.28 | 2.2× | 2.3 (marginal) |
| 50p | 0.77 | — | unmeasurable, see below |

Seven of nine are clean detections. For comparison the naive
`contrast_3d` (full R_vir, local annulus) gives the same objects 1.2–173×
with no ordering — so the aperture/background change was worth making, but **not
for the reason first claimed here.**

**The misreading.** Run 1 reported the control as failing, with HR subhalos at
0.4–1.2× and snr −105. Those numbers were real, but they were
`stats["tuned"]` — HR's centres measured in the **self arm's field**, printed
regardless of `--select` by a logging bug. That is a different and much more
interesting quantity, and it is now printed for every field with the owning one
starred. Do not quote run 1's control numbers.

**What that quantity actually says.** At HR's own subhalo positions, the tuned
and frozen fields score 0.15–4.3×, mostly below 1, against HR's 2.2–82× at the
same coordinates. The arms do not put mass where HR's subhalos are. Meanwhile
each arm's *own* catalogue centres score 2.5–106× in its own field: they are
genuine compact overdensities. Every arm subhalo sampled sits at
R/R_vir,host ≈ 0.19–0.37 against HR's 0.28–0.95.

So for the question this document opens with — is Rockstar reporting phantoms? —
test 1's answer is **no**: the objects are real density concentrations. They are
in the wrong place, which is the result [[radial-placement-is-inverted]] already
measured by a different route.

### Two things the measurement refuses to report

Both are flagged as `usable: false` with a `why` string rather than returned as
a number:

* **Aperture below the resolution.** A 50-particle subhalo has `r_vir = 0.022`
  Mpc/h, so its 0.3 R_vir core is 0.0066 — an eighth of a 512³ cell. It holds no
  particles. That reported `0.00×`, which reads as "measured, and empty".
* **Exclusion sphere swallowing the background.** A 0.7 Mpc/h subhalo at
  R = 0.33 Mpc/h — which the tuned arms produce routinely — has
  `3 × r_vir > R`, so the background shell vanished and the ratio came back
  `inf`, for the largest objects in the sample. The exclusion radius is now
  capped at `R/2`.

### Where the particles come from

By default the complete spliced box, when the field is still on disk. When it
has been purged the chain falls back to the export's `tiles.npz` — the host's
Lagrangian tiles, which is what the existing host figures bin. That set omits
material that drifted in from other tiles, biasing the local background low and
every contrast high, equally in all three panels. Which was used is recorded as
`particle_source` in the data file and printed in the figure title; quote it
with any number taken from it.

### Output

```
$MG/<arm>/holdout_set9/figures/subzoom_<t>.json          the numbers
$MG/<arm>/holdout_set9/figures/subzoom_<t>_h168880/
    fig1_subhalo_grid.png    rows = subhalos, cols = tuned | frozen | HR, thin slab
    fig2_depth_ladder.png    one subhalo across five slab depths -- the burial itself
    fig3_contrast.png        ratio and snr against depth, all subhalos
    fig4_aperture.png        the four aperture/background choices -- QUOTE THIS
```

All panels in a row share one colour scale built from the three arms together.
Without that the eye compares three independently normalised images and always
sees a knot in each.

---

## Test 2 — ask a second halo finder

`scripts/slurm/submit_subfind.sh`, wrapping GADGET-4's SUBFIND.

SUBFIND disagrees with Rockstar in exactly the right way. It defines
substructure in **configuration space** — a locally overdense region delimited
by a saddle point of the 3-D density field, kept only if it is self-bound.
Velocities enter once, at the end, as an unbinding criterion, never as a linking
dimension. So:

* an object **both** finders report is real by two independent definitions;
* an object **only Rockstar** reports is a phase-space feature with no density
  peak behind it — the false positive the picture would be right about;
* an object **only SUBFIND** reports is a density peak Rockstar split or dropped.

GADGET-4 is run in post-processing mode (`Gadget4 param.txt 3 <snapnum>`): it
reads a snapshot, runs FOF and SUBFIND, writes the group catalogue and exits.
Nothing is re-simulated. The snapshot is written by the **same writer that feeds
Rockstar** (`cosmo_sr.eval.particles.write_gadget2_snapshot`), so both finders
see the same particles and any difference is a difference between algorithms.

### RESULT (2026-09-24, set9 host 168880, >=50p inside R_vir)

| field | Rockstar | SUBFIND | confirmed_frac |
|---|---:|---:|---:|
| HR | 369 | **146** | 0.393 |
| soft | 377 | **40** | 0.101 |
| knn | 419 | **32** | 0.076 |
| self | 366 | **22** | 0.060 |
| frozen | 20 | 3 | 0.100 |

**Rockstar says the arms reach parity with HR. SUBFIND says they reach 15-27% of
it.** HR's own 0.393 is the calibration — two independent finders agreeing on
real objects — and the arms confirm 4–6× below that.

This does not contradict test 1. It sharpens it: **an overdensity is not a
self-bound subhalo.** SUBFIND requires a density peak delimited by a saddle
*and* survival of iterative unbinding; the zoom tested neither, and said so. The
two results together read: the arms deposit real mass concentrations that are
too diffuse, or too unbound, to be substructure. The figures agree — HR panels
show many small sharp knots, the tuned panels broad smooth bright regions.

Arm ordering soft > knn > self matches the Rockstar gate's.

**Box-wide numbers are not a second data point.** A spliced candidate box is
~94% frozen field (`spliced_volume_fraction` 0.0625), so the arm's box-wide
SF/RS ratio (soft 0.152) is just frozen's (0.147). The frozen figure is measured
directly and does mean something: SR2 box-wide sits at 0.147 against HR's 0.347,
so SR2 halos are systematically less SUBFIND-detectable than HR's, with no
fine-tune involved.

**A reading trap.** `rockstar_bins` and `confirmed_by_bin` bin by the Rockstar
object's particle count; `subfind_bins` bins by the SUBFIND object's own. They
are not row-comparable — `confirmed` can exceed the SUBFIND count in a row,
because a matched pair lands in different bins. The JSON now carries a
`bins_note` saying so.

### Why does SUBFIND refuse them? What is measured so far

**1. It is not a resolution effect.** Resolved by Rockstar particle count, the
confirmation rate splits into three regimes:

| bin | HR | frozen | self | soft | knn |
|---|---|---|---|---|---|
| 50-100p | 11/169 **7%** | 0/3 | 1/168 1% | 2/159 1% | 0/184 0% |
| 100-200p | 39/97 **40%** | 0/0 | 2/95 2% | 3/85 4% | 1/93 1% |
| 200-500p | 48/56 **86%** | 1/6 17% | 3/56 5% | 1/72 1% | 2/79 3% |
| 500-2000p | 34/34 **100%** | 1/8 12% | 4/33 12% | 12/41 29% | 12/45 27% |
| 2000+p | 13/13 **100%** | 0/3 | 12/14 86% | 20/20 100% | 17/18 94% |

*(Both columns of each cell are binned by the Rockstar object's `num_p`, so they
are comparable. Do not read them against `subfind_bins`.)*

* **≤200p** — HR confirms only 7–40%, so this is SUBFIND's own resolution limit
  (`DesNumNgb=64`, `DesLinkNgb=20` cannot isolate a saddle-bounded peak from a
  hundred particles). It says nothing about the arms.
* **200–2000p** — HR confirms **86–100%**, the arms **1–29%**. SUBFIND
  demonstrably finds objects of this size when they are real. **This band is
  the defect**, and it is where the forensic looks.
* **>2000p** — every field confirms ~100%. These are the shared central
  structures, present in frozen too; not gains.

**2. It is not the counting aperture.** All rows are counted inside HR host
168880's `R_vir`, and the tuned host sits 0.40–0.47 Mpc/h off HR's centre
(`R_vir` = 1.553). Recounting each field about **its own** most-massive host
changes nothing: self 22→22, soft 40→41, knn 32→33, HR 146→146.

**3. Each row uses its own field's catalogues, not HR's positions.** For an arm
row, both the Rockstar and the SUBFIND table are that arm's own, so the arm's
subhalos are measured wherever they actually are. The Rockstar counts differing
per row (366/377/419/369/20) is the check: sampling HR's positions would give
369 everywhere.

### The mechanism — MEASURED (2026-09-24): Rockstar fragments, SUBFIND counts once

The proposal that settled it: *Rockstar is counting many subsets of one big
blob inside a single `r_vir`, while SUBFIND counts it once.* That is exactly
what the catalogues show.

**Step 1 — the spheres interpenetrate.** For Rockstar subhalos in the 200-2000p
band inside the host:

| field | N | median NN separation | **in units of r_vir** | % centres inside another's sphere | mean overlapping neighbours |
|---|---:|---:|---:|---:|---:|
| HR | 90 | 0.2155 Mpc/h | **2.02** | **11%** | 0.9 |
| frozen | 14 | 0.0207 | 0.18 | 93% | 8.0 |
| self | 89 | 0.0357 | **0.25** | **92%** | **39.8** |
| soft | 113 | 0.0428 | 0.37 | 85% | 31.9 |
| knn | 124 | 0.0321 | 0.29 | 85% | 35.5 |

HR's subhalos sit two of their own radii apart — separate objects. The arms' sit
a *quarter* of their own radius apart, 85-92% with a centre inside another's
sphere, each overlapping 32-40 others.

**Step 2 — link the overlapping ones and the count falls out.** Treating
subhalos whose virial spheres intersect as one structure (connected components):

| field | Rockstar subhalos | distinct structures | fragments/structure | largest group | **SUBFIND finds** |
|---|---:|---:|---:|---:|---:|
| HR | 90 | **55** | 1.6 | 7 | **55** |
| frozen | 14 | 3 | 4.7 | 11 | 0 |
| self | 89 | **6** | 14.8 | **70** | **5** |
| soft | 113 | 11 | 10.3 | 91 | **15** |
| knn | 124 | 15 | 8.3 | 100 | **17** |

**The number of distinct structures predicts SUBFIND's count in every field:**
55→55, 6→5, 11→15, 15→17. SUBFIND is not missing objects — it is refusing to
report one structure many times. In the `self` arm, **70 of 89 "subhalos" are
fragments of a single blob.**

This also explains `n_in / num_p` ≈ 90 (section below): the sphere spans the
whole pile, ~43,000 particles, and ~70 fragments share it, each assigned ~380.

**It reframes HR's own 46% confirmation too.** HR fragments mildly as well (1.6
per structure), and the matching is one-to-one, so only one fragment per
structure can claim its SUBFIND counterpart. HR's "unconfirmed" 54% is mostly
that bookkeeping, not failed objects — which is why the *structure* count, not
`confirmed_frac`, is the number to quote.

**What is genuinely wrong with the arms**, stated at the right altitude: not that
their subhalos are phantoms, but that where HR has ~55 separate bound objects
inside the host, the arms have **5-17 over-dense piles** that Rockstar shreds
into 89-124 catalogue entries. The abundance match Rockstar reports
(`ratio_total_tuned` 0.99) is a fragmentation count, not an object count.

### The mechanism — MEASURED (2026-09-24)

`scripts/features/subfind_miss_forensics.py`, 200–2000p band, properties taken
from the **particles** inside Rockstar's own `r_vir`, never from its columns.

The apertures are like-for-like: median `num_p` 378–390 and `r_vir`
0.104–0.117 Mpc/h in every field.

**The discriminator is `n_in / num_p` — how much of the claimed object's own
volume the claimed object actually accounts for.**

| field | num_p | n_in | **n_in/num_p** | ρ(<r)/mean | σ_v km/s | R/R_vir,host |
|---|---:|---:|---:|---:|---:|---:|
| HR | 390 | 520 | **1.15** | 821 | 972 | 0.61 |
| frozen | 548 | 33521 | 52.7 | 36078 | 737 | 0.25 |
| self | 384 | 43100 | **89.8** | 47782 | 452 | 0.30 |
| soft | 378 | 16486 | **47.3** | 20481 | 481 | 0.27 |
| knn | 386 | 24604 | **53.6** | 32836 | 510 | 0.28 |

For HR, a subhalo *is* the mass inside its own radius (520 particles present,
390 claimed — ratio 1.15, local density 821× mean, a sensible subhalo core). For
the arms, Rockstar claims ~380 particles in a volume holding **16,000–43,000**:
the claimed object is **1–2% of the mass in its own radius**. It is a ripple on
a smooth dense core, not a distinct clump — and a 1% perturbation has no saddle
to delimit it, which is exactly what SUBFIND requires.

#### Control: matched host radius

The arms' objects sit deeper in the host (0.25–0.30 R_vir) than HR's (0.61),
where surroundings are denser anyway. Restricting every field to the arms' own
shell, **0.20–0.40 R_vir,host**:

| field | N | num_p | n_in | **n_in/num_p** | σ_v | r_half/r | confirmed |
|---|---:|---:|---:|---:|---:|---:|---:|
| HR | 17 | 485 | 2267 | **3.76** | 1338 | 0.744 | **0.59** |
| frozen | 11 | 563 | 43516 | 64.0 | 749 | 0.679 | 0.00 |
| self | 75 | 384 | 51767 | **111** | 439 | 0.745 | 0.03 |
| soft | 77 | 392 | 34712 | **85.6** | 460 | 0.733 | 0.13 |
| knn | 79 | 383 | 60377 | **137** | 488 | 0.729 | 0.10 |

At the same host radius and the same particle count, the arms' apertures hold
**15–26× more mass** than HR's, and the claimed object is **23–36× smaller a
fraction** of it. So it is not a placement artefact. HR's own objects in that
shell still confirm at 59% against the arms' 3–13%.

#### And why *Rockstar* finds them: the velocity field is ~3× too cold

In the same shell, σ_v is **439–488 km/s for the arms against HR's 1338**.
Rockstar links in six dimensions, so a kinematically coherent cold group is a
halo to it regardless of whether a density peak is there. The arms suppress
small-scale velocity dispersion — the same defect [[highk-guard-is-blind]]
measured as "killed 95% of velocity small-scale power" and the `vel_rms` guard
has always failed — and Rockstar reads that coldness as substructure.

**So the two finders are detecting different things.** Rockstar is detecting the
arms' velocity coldness inside an over-concentrated core; SUBFIND asks for a
spatial density peak and finds none. The zoom figures show the same thing
visually: broad smooth bright regions where HR has sharp knots.

#### Two honest caveats

* **`r_half_over_r` does not discriminate here.** At matched radius it reads
  0.73–0.75 for HR and the arms alike, against the uniform-sphere value 0.794.
  Deep in the core the aperture is filled by smooth host material in every
  field, so the shape statistic measures the host, not the object. The
  occupancy ratio is the number that carries the result.
* **`virial_ratio` is not usable as a boundness test in this regime**, for the
  same reason: `v_circ` is computed from `n_in`, which is host mass, so the arms
  score 0.4–0.8 ("cold") and HR 7.5 ("hot") through aperture contamination
  rather than through binding. The raw `sigma_v` comparison above is the
  defensible velocity statement; a real boundness test needs SUBFIND's own
  member lists, which is what its refusal already tells us.
* **The over-concentration is inherited, not created.** Frozen SR2 shows the
  same 36,078× core density and 64 occupancy ratio, so the fine-tune did not
  build the pile — it added cold kinematic structure inside a pile SR2 already
  had. See [[host-damage-is-mostly-reparenting]].

### The headline number

`confirmed_frac` — the fraction of Rockstar's subhalos inside the host's `R_vir`
that have a SUBFIND counterpart within a tolerance scaled to their own `R_vir`.
**HR's own rate is the calibration**: two finders never agree perfectly, and
whatever HR scores is what "as good as real" means here. An arm near HR's rate
has made real objects; an arm far below it has made catalogue entries.

The censuses are reported alongside, per particle-count bin, because the
confirmation rate alone is gameable in both directions — an arm that produces
almost nothing confirms 100% of its handful, and an arm that shatters one real
clump into five confirms one in five.

Counts are on **bound particle number**, which both finders mean the same thing
by. No attempt is made to reconcile Rockstar's `mvir` with SUBFIND's bound mass.

### Build

There is no SUBFIND binary on this cluster and no system compiler, MPI, GSL,
FFTW or HDF5 at all, so the toolchain is a dedicated conda environment:

```bash
conda create -y -n g4 -c conda-forge \
    gcc_linux-64 gxx_linux-64 openmpi openmpi-mpicc openmpi-mpicxx \
    fftw gsl hdf5 h5py numpy
```

`external/gadget4/Config.sh` records why each compile option is set. Three are
load-bearing and easy to get wrong:

* **`GADGET2_HEADER`** — our snapshots are legacy format-1 with the 256-byte
  GADGET-2/3 header. Without this GADGET-4 expects its own header and the read
  aborts on a block-size mismatch. It also pins `NTYPES_HEADER` to 6
  independently of `NTYPES`, which is what lets `NTYPES=2` work.
* **`ICFormat 1`, `SnapFormat 3`** — in post-processing mode 3 the snapshot is
  read with `ICFormat` (only mode 2 copies `SnapFormat` over it), while
  `SnapFormat` governs what the *group catalogue* is written as. So: legacy
  binary in, HDF5 out.
* **`OutputDir` needs its trailing slash** — GADGET-4 builds the snapshot path
  as `OutputDir + SnapshotFileBase + "_%03d"` with no separator of its own.

### Run the smoke test first

```bash
bash scripts/slurm/submit_subfind.sh smoke
```

It writes a small synthetic snapshot with planted, self-bound clumps at known
centres and runs the whole chain on it in minutes. Because the answer is known,
it checks that the pipeline **finds what is there** rather than merely exiting 0
— a wrong header, a mis-scaled velocity block or a parameter file that silently
disables SUBFIND all produce a clean exit and an empty catalogue. The verdict is
`$SUBFIND_ROOT/smoke/smoke_verdict.json`; anything short of all clumps recovered
means the chain is wrong, not the data.

### Then the real run

```bash
bash scripts/slurm/submit_subfind.sh full
```

Five fields: HR, frozen, and the three arms. Each is `snapshot (CPU, 64G)` →
`SUBFIND (CPU, 44 ranks, 600G)` → and then one comparison per arm once every
finder has run. An arm whose spliced field was purged is re-spliced from its
`tiles.npz` first — no GPU is needed anywhere in this chain.

---

## The arms under test

| label | run | what it is |
|---|---|---|
| `self` | `all_blocks_self` | the baseline every A/B is measured against |
| `soft` | `all_blocks_sm_coarse_soft` | soft-membership; wins the held-out Rockstar gate |
| `knn` | `all_blocks_knn6d_w5` | the kNN-CDF term **on top of** self (`w_knn_unsup=5` with `centre_mode=self`, `w_virial=1`, `w_centre=1`, `w_low=100`) |

`all_blocks_knn6d_only` is deliberately *not* the kNN arm here: it drops the
self terms (`centre_mode=full`, every other weight 0) and answers a different
question.

---

## How the two tests combine

|  | SUBFIND confirms | SUBFIND refuses |
|---|---|---|
| **thin slab shows a knot** | the objects are real; the host figures were burying them. Fix the figures. | contradiction — check the matching tolerance and the particle source before believing either |
| **thin slab shows nothing** | real but diffuse: bound, no compact core. The gain is not the gain we wanted. | Rockstar is reporting phase-space artefacts. The gate has been measuring a number that does not exist. |

Neither test alone closes the question; the second row is why the second test
had to be an independent finder rather than a better plot.
