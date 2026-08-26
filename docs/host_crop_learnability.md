# Host-frame crops: what SR2 already knows, and where the note's high-pass sits

**Scope.** A measurement note for `docs/sr2_substructure_module.md`. It builds
Option A's host-frame crop on real data, asks whether the position of an HR
subhalo is recoverable from SR2's own field, and reports what the residual the
module must emit actually looks like. No model is trained. One result is
arithmetic on the design note's own constants and contradicts one of its
commitments; that is section 4 and it is the reason this note exists.

Numbers are tagged *measured* (read from an artifact on disk), *derived*
(arithmetic from measured constants), or *provisional* (measured, but on a
reduced volume whose magnitude is biased -- sign trustworthy, value not).

Depends on `docs/sr2_substructure_module.md` for the design,
`docs/sr2_subhalo_deficit.md` for the deficit, and
`docs/lagrangian_host_features.md` for the lattice conventions.

## Tooling

| | |
| --- | --- |
| geometry, reduction, ranking | `src/cosmo_sr/features/host_crops.py` |
| collection | `scripts/features/collect_host_crops.py` |
| page | `scripts/features/render_host_crops_app.py` |
| jobs | `scripts/slurm/host_crops{,_render}_cpu.sbatch`, `submit_host_crops.sh` |
| tests | `tests/features/test_host_crops.py`, `test_host_crops_page.py` |

Artifacts land in `dmsr_reward/lagrangian_host/<box>/<box>_host_crops.{json,npz}`
and the page is a pure redraw of them.

## 0. The sample

*Measured*, set8, seed 0. 14 HR hosts: all six above `1e14`, then two per
half-decade down to `1e12`, spanning `log10 Mvir` 12.02 to 14.81 and 2 to 2296
subhalos each. The crop is the design note's section 3 cube -- side `2 R_L`
about the host's Lagrangian centroid -- taken on the **native** 512^3 lattice and
never resampled, so what is measured is the crop's content rather than an
interpolation of it.

*Measured* consequence, and the first thing Option A costs:

| log10 Mvir | crop side (sites) | side (Mpc/h) | ×96^3 resample |
| ---: | ---: | ---: | ---: |
| 14.81 | 148 | 28.9 | ×0.65 (decimated) |
| 14.39 | 100 | 19.5 | ×0.96 |
| 13.84 | 72 | 14.1 | ×1.33 |
| 13.03 | 38 | 7.4 | ×2.53 |
| 12.02 | 24 | 4.7 | ×4.00 (interpolated) |

A fixed 96^3 grid throws away three quarters of a cluster's sites per axis and
invents four times as many as a `1e12` host has. This is section 3's "resampling
waste at the bottom", now with numbers on both ends.

## 1. Does SR2 know where the subhalos go?

The probe. Over the Lagrangian sites of one host's own footprint, label a site
positive if HR binds its particle to one of that host's **subhalos** and negative
if it binds it to the host itself. Score each site by a scalar and report the
Mann-Whitney AUC -- `P(a random subhalo site scores above a random smooth site)`.
Three scores, the same population and label:

* **SR2 density** -- k-nearest-neighbour local *Eulerian* log density (k=32)
  computed on SR2's particles and gathered back onto the Lagrangian site, then
  Gaussian-smoothed over sigma sites. The probe.
* **HR density** -- the identical estimator on the true field. The ceiling.
* **distance from the host's Lagrangian centre** -- no field at all. The
  baseline that says how much of any apparent signal is just geometry.

*Measured*, sigma = 2 sites:

| log10 Mvir | n_sub | base rate | SR2 | HR (ceiling) | radius only |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 14.81 | 2296 | 59.5% | 0.373 | 0.417 | **0.211** |
| 14.61 | 989 | 22.7% | 0.279 | 0.302 | **0.186** |
| 14.44 | 680 | 41.6% | 0.337 | 0.433 | **0.258** |
| 14.40 | 668 | 35.4% | 0.315 | 0.392 | **0.257** |
| 14.39 | 643 | 56.5% | 0.449 | 0.524 | **0.399** |
| 14.34 | 740 | 54.7% | 0.381 | 0.398 | 0.399 |
| 13.84 | 236 | 41.4% | 0.369 | 0.416 | **0.270** |
| 13.64 | 148 | 29.0% | 0.344 | 0.417 | **0.290** |
| 13.41 | 67 | 12.2% | 0.384 | 0.395 | **0.309** |
| 13.03 | 43 | 14.4% | 0.312 | 0.282 | **0.234** |
| 12.92 | 32 | 27.9% | 0.408 | 0.505 | **0.276** |
| 12.71 | 12 | 8.2% | 0.264 | 0.291 | **0.201** |

Three readings, in order of how much weight they carry.

**1.1 The ceiling is below 0.5 too, so the probe is the wrong sign, not the
field.** HR's own density predicts HR's own subhalo membership at ~0.40. Local
density is not a subhalo detector inside a host: the densest material is the
smooth central core, and the subhalos that survive live in the outskirts. The
radius baseline says the same thing more directly and, in eleven of twelve rows,
more strongly than either density. **Most of what this probe measures is the
radial profile.** Distance from 0.5 is the signal; below 0.5 the ranking is
simply reversed and a model exploits it identically.

**1.2 SR2 sits on the HR ceiling.** Row by row the two densities agree to within
0.10, and in one row SR2 is the better predictor of HR's own subhalos than HR is.
At this level of description **SR2's field is as informative about subhalo
position as the true field is**, which is evidence that the conditioning is not
the bottleneck and the failure is generative. This is the note's most useful
result and also its weakest: one scalar per site cannot express "there is a
coherent four-site clump here", so matching HR on this statistic does not
establish matching it on the statistic that matters. It rules out one failure
mode -- *SR2 has thrown the information away* -- and nothing more.

**1.3 The two rows above 0.5 are noise.** `logM 12.25` (AUC 0.702) and `12.02`
(0.515) hold 11 and 2 subhalos. They are in the artifact for completeness.

## 2. What the residual actually is

*Provisional* -- computed on the block-max reduced volumes the page draws from,
so the gap is inflated; the sign and ordering are robust, the magnitude is not.
Over each host's footprint, in dex of local log density:

| log10 Mvir | pedestal (mean HR−SR2) | at subhalo cells | at smooth cells | gap | corr |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 14.81 | +0.67 | +0.78 | +0.46 | +0.32 | 0.160 |
| 14.61 | +0.74 | +1.06 | +0.67 | +0.39 | 0.148 |
| 14.44 | +0.59 | +0.90 | +0.41 | +0.49 | 0.275 |
| 14.40 | +0.49 | +0.93 | +0.28 | +0.64 | 0.298 |
| 14.39 | +0.55 | +0.62 | +0.32 | +0.30 | 0.090 |
| 14.34 | +0.42 | +0.50 | +0.39 | +0.11 | 0.085 |
| 13.84 | +0.28 | +0.53 | +0.15 | +0.37 | 0.183 |
| 13.64 | +0.57 | +0.95 | +0.51 | +0.44 | 0.178 |

The residual has **two components, not one**:

1. a **broad compaction of the entire host patch**, +0.28 to +0.74 dex, present
   at smooth sites as much as at subhalo sites -- SR2's whole cluster is puffier
   than HR's. This is the sub-virial defect of the channel-swap result, seen in
   density rather than in velocity;
2. a **localised excess at subhalo sites**, positive in all eight hosts, on top
   of that pedestal.

Component 1 is comparable to or larger than component 2. Both are things the
module would have to emit, and section 4 is about whether it is allowed to.

*Measured* alongside: SR2 binds **79-88%** of each host's Lagrangian sites into
some object. The material is present and bound; it is not fragmented. That
supports section 6.1's "the work is fragmenting smooth material, not deleting and
rebuilding".

## 3. Why none of this is visible in the crop panel

Recorded so the picture is not over-read. The localised excess is 0.3-0.6 dex on
a pedestal of 0.5 dex, rendered against a ±3.78 dex colour range: **4-8% of the
colour bar**. Two rendering choices compound it -- the difference view projects
HR and SR2 separately and subtracts the projections rather than projecting the
difference, and a cluster pixel is a max over ~4^3 sites times ~4 slab cells,
about 256 native sites, while a 500-particle subhalo spans ~500. The panel is a
legibility aid; every number in this note is computed on the native cube.

(A registration bug in that panel -- the reduced cube was padded to a multiple of
a fixed output side with edge replication, 44 of 192 sites for the largest crop,
while the overlay scaled by the unpadded side -- was found and fixed. Output side
is now derived from the block factor, padding is 0-2 sites and `nan` rather than
replicated, and `vol_extent_sites` carries the scale the overlay must use.
`test_overlay_and_image_agree_after_the_block_reduction` pins it. The AUC tables
were computed on native cubes throughout and are byte-identical across the fix.)

## 4. The high-pass cut is set an order of magnitude too high

The load-bearing finding, and *derived* entirely from the design note's own
constants -- `m_p = 5.81881e8`, `rho_bar = 7.809e10`, `R_L = (3M/4 pi rho_bar)^(1/3)`.

Skeleton item 4 filters the module's output above **8 h/Mpc** in Lagrangian k, so
that "cannot move or resize a host" is a property of the parameterization. The
stated justification is that this is where SR2's displacement `r(k)` falls off.
That is a statement about SR2's fidelity. The commitment it is being used for --
separating *host* scales from *substructure* scales -- is a different criterion,
and the two do not land in the same place.

Taking each object's Lagrangian diameter `2 R_L` as its scale, `k = 2 pi / 2 R_L`:

| object | num_p | R_L (Mpc/h) | k (h/Mpc) |
| --- | ---: | ---: | ---: |
| 50p subhalo (the deficit's floor) | 50 | 0.446 | **7.04** |
| 366p subhalo (HR median inside clusters) | 366 | 0.867 | **3.62** |
| 2000p subhalo (the survivors SR2 gets right) | 2000 | 1.527 | **2.06** |
| 1e12 host | 1719 | 1.451 | 2.16 |
| 1e13 host | 17186 | 3.127 | 1.00 |
| 1e14 host (cluster) | 171856 | 6.737 | 0.47 |

Two consequences.

**4.1 Every subhalo the module exists to build is below the cut.** A cut at
8 h/Mpc passes wavelengths under 0.785 Mpc/h, which is 4.0 HR sites, the
Lagrangian diameter of a **34-particle** object -- below Rockstar's own
50-particle output floor. The pass-band therefore contains no resolvable subhalo
at all: it is the band of structure *finer* than anything the catalog can
report. Filtering the residual there does not restrict the module to
substructure; it forbids the module from assembling substructure, because the
coherent bulk displacement that gathers 366 particles into a clump is a
3.6 h/Mpc mode. It would also remove all of section 2's component 1, the
+0.5 dex compaction, which is a 0.47 h/Mpc mode for a cluster.

**4.2 No single k separates hosts from subhalos anyway.** A `1e12` host sits at
2.16 h/Mpc and a 2000-particle subhalo at 2.06 -- the same scale, because they
are the same mass. The host/subhalo distinction is *relational*, not spectral.
The separation is clean only at the extremes: a cluster at 0.47 against a 50p
subhalo at 7.04, 1.2 decades apart.

So the guarantee item 4 wants is real and worth having, but a fixed Lagrangian-k
high-pass is the wrong instrument for it. Something host-relative -- a cut placed
per-crop at a fraction of the host's own `R_L`, which Option A's normalisation
already makes natural, or a constraint on the residual's mean displacement and
its low-order moments inside the footprint rather than on its spectrum -- would
express "do not move or resize this host" without also forbidding its
substructure. This wants a decision before any training.

## 5. What this settles, and what it does not

| design-note claim | status after this note |
| --- | --- |
| §2.1 freeze SR2, hosts >200p correct | untouched; but §2 shows cluster *profiles* are puffy at the right total mass |
| §2.4 high-pass at 8 h/Mpc | **contradicted**, section 4 |
| §6.1 the work is fragmenting, not rebuilding | supported: SR2 binds 79-88% of host sites |
| §7.3 is `p(HR|SR2)` broad at subhalo scale? | **measured BROAD** on set8 (`docs/pilot_steps_2_4.md` §1); section 1.2 here ruled out "SR2 threw the information away", §1 there closes the other half |
| §9.2 measure the conditional spread | **done** (`docs/pilot_steps_2_4.md`); cross-realisation split still owed |

Nothing here argues against sampling rather than regressing. A crisp,
deterministic HR−SR2 signature at every subhalo would have argued for a
regressor, and there is none -- but this note's probes are too blunt to
distinguish "the fine modes are genuinely broad" from "the probe cannot see
them", so that is not evidence and is not claimed as any.

## 6. Two corrections to the design note

**6.1 §6.1's reason is not the true one.** The text rules out residual
regression, correctly, then says flow matching "never computes a difference". If
the generated variable is `d_disp` -- and it must be, since §4 applies the output
as `disp_SR2 + d_disp` and §2.4 high-passes it -- then the flow's training data
*is* the per-site difference field. That is fine, because conditioned on SR2 a
shift by SR2 is a bijection and `p(Delta | SR2) = p(HR | SR2)`. But the argument
that carries the design is the other one: **L2 converges to `E[Delta | SR2]`,
which is empty in the fine modes, while flow matching samples the conditional,
which is not.** Same target array, different functional of it. The note should
also state explicitly that the flow's `x_1` is the delta and not the full field;
only that reading is consistent with §4's inference rule and §2.4's filter.

**6.2 §8 counts two different things as one.** "The right one is ~201k cluster
subhalos" is not right as stated, because a subhalo is never a training example
-- under Option A the unit is one host, under Option B one tile, and §6.2 forbids
subhalo membership from entering the loss at all. The two counts bound different
quantities and the note should say which:

| | Option A | Option B |
| --- | ---: | ---: |
| data unit | one host crop | one 64^3 tile |
| units per box / ×16 | 832 hosts, 24 clusters / 384 | 512 / 8,192 |
| independent draws of *host-scale* conditioning | 384 clusters | 8,192 tiles |
| local supervision events | ~201k cluster subhalos | ~1.61M subhalos |

The subhalo count bounds the gradient a fully-convolutional module receives *if*
the mechanism it must learn is local and translation-invariant. It does not bound
anything that varies at host scale -- how abundance depends on `M_host`, how a
cluster's environment differs -- for which the honest number is 384. The two are
not iid either: subhalos in one cluster share a host, an environment and a
realization, so the effective sample size lies between them, nearer 384 for
anything host-scale. §4.3's importance sampling is implicitly a bet on the local
reading.

## 7. Limits of these measurements

1. **One box, one seed.** set8, seed 0. Nothing here is a variance estimate.
2. **Section 2's magnitudes are provisional** -- block-max reduced volumes
   inflate the subhalo-vs-smooth gap. Recomputing the pedestal and the gap on the
   native cube inside the collector would put them on the same footing as
   section 1, and is a small change to `collect_host_crops.py`.
3. **Section 4 is arithmetic, not a spectrum.** It uses each object's Lagrangian
   diameter as its scale. **The direct measurement has since been run**
   (`docs/pilot_steps_2_4.md` §3.5): the exact whole-box residual spectrum puts
   `r(k)` through 0.5 at k=2.3 and 0.1 at 3.7 h/Mpc, with ~10% of the residual
   power above 8 h/Mpc -- confirming this section's conclusion that the 8 h/Mpc
   cut is an order of magnitude too high and that no single k separates
   substructure from bulk. Section 4 is no longer the only evidence.
4. **The AUC probe is one scalar per site** and is dominated by the radial
   profile (section 1.1). It bounds learnability from below and loosely.
5. **Host selection is by HR catalog**, so nothing here exercises Option A's
   worst dependency, the SR2-to-HR host matching that a real Option A training
   set would need below `1e14`.

## 8. Reproduce

```bash
bash scripts/slurm/submit_host_crops.sh          # collect (CPU, ~1 min) then render
BOXES=set8,set9 CROP_N_HOSTS=24 bash scripts/slurm/submit_host_crops.sh
RENDER_ONLY=1 bash scripts/slurm/submit_host_crops.sh
```

Sections 0-2 are `dmsr_reward/lagrangian_host/set8/set8_host_crops.json`; the
page is `host_crops_set8.html` beside it. Section 4 is arithmetic on the header
constants of `docs/sr2_substructure_module.md` and needs no artifact:

```python
import numpy as np
mp, rho = 5.81881e8, 7.809e10
RL = lambda n: (3 * n * mp / (4 * np.pi * rho)) ** (1 / 3)
for n in (50, 366, 2000, 1e12 / mp, 1e14 / mp):
    print(f"{n:9.0f}p  R_L={RL(n):6.3f} Mpc/h  k={2 * np.pi / (2 * RL(n)):6.2f} h/Mpc")
```

**TODO.** Limits 2 and 3 are both small additions to
`scripts/features/collect_host_crops.py` and would move section 2's magnitudes
and section 4's whole argument from *provisional*/*derived* to *measured*.
