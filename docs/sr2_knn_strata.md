# Ditching the membership label from the kNN term

**Status, 2026-09-14: GATE FAILED in both 3D and 6D (jobs 54405/54406). The arm
is dead; `--knn-classes` stays.** See "The verdict" at the bottom — it is the
only section with numbers from a real box. Everything above it that says
MEASURED was measured on synthetic fixtures in `tests/features/test_knn_cdf.py`.

## The question

`src/cosmo_sr/features/knn_cdf.py` is already the catalog-free statistic: it is
charged on the unsupervised box-wide tile pool, it does no halo matching, no set
pairing, no per-particle address. One flag spoils that claim. `--knn-classes`
restricts the query set to particles whose HR owner class is `sub_50_200`, using
`classify_owner` on HR's Rockstar catalog — a label, in the gradient.

It is not there by preference. Round 5b measured the pooled form ranking the
working arms **backwards** (rho +0.91 against region subhalo count) because host
and unbound particles are ~95% of any tile interior and the working arms gain
subhalos by over-concentrating hosts. Masking by class fixed the ranking
(rho −0.65, in 6D at `phase_lambda=4000`; the position-only statistic ranks at
−0.23 and is not usable).

So: the label is doing one job, and it is not membership. It picks out the dense
tail of the query population. That is a property of the particles, and each
field can read it off its own positions. Hence `KnnConfig.strata`.

## What was built

`KnnConfig.strata` / `strata_rank` / `strata_charge`, `strata_partition`,
`knn_stratified_loss` — each field sorts its own queries by its own
`r_{strata_rank}`, splits them into equal-count groups (dense first), and the
loss is charged per group. The partition is detached, exactly as the neighbour
search is and for the same reason.

Trainer: `--knn-strata`, `--knn-strata-rank`, `--knn-strata-charge`, and
`--knn-classes-diagnostic-only`, which keeps the class labels attached, measured
and reported at every eval while passing `class_ids=()` to the training call.
All default off; an unset run is byte-identical to the existing arms.

## Two measured findings, and the second is the problem

**1. Partitioning is not reweighting.** Equal-count strata of `Q` bins each
partition the queries into exactly the `strata * Q` equal-count bins the pooled
form already uses. For the stratifying column this is an *identity*, pinned to
1e-12 in `test_strata_on_the_stratifying_rank_is_finer_binning`. Measured on the
dilution fixture: pooled 0.0881, eight equal-weight strata 0.0912 — while the
densest stratum alone carries 0.726 and the other seven are ≤ 0.004. The dense
tail's share of the gradient is unchanged. Only `strata_charge` changes it.

What partitioning *does* buy is the readout: on a field whose dense tail is
1.4–2.0x HR's `r_k`, the pooled `ratio_med` reads 1.00 and the dense stratum's
reads 1.27–1.76. That is now reported as `ratio_med_dense` / `knn_dense_k*`.

**2. A density cut is not a subhalo cut.** MEASURED on the same fixture: the
densest stratum is **65–83% host particles**, and its subhalo fraction gets
*worse* as the candidate's subhalo is made more diffuse — the damaged field has
fewer of its own subhalo particles in its own dense tail. Host
over-concentration is this line's characteristic damage
(`host-damage-is-mostly-reparenting`, `radial-profile-is-flat-in-host-mass`), so
charging a host-dominated population risks deepening the defect the Rockstar
gate keeps catching.

Finding 2 is why this is gated rather than launched. It is a synthetic number,
and the real box is a harder case in the direction that matters: a 50–200p
subhalo orbiting inside a host is barely denser than the host in 3D, which is
why round 5 went to 6D at all.

## The gate

`scripts/features/knn_strata_composition.py`, submitted by
`scripts/slurm/submit_knn_strata.sh` (CPU). Per field and per tile it reports
the owner-class composition of every stratum, the dense stratum's `sub_50_200`
purity and recall as **enrichments over the base rate**, and the two competing
losses — dense-only stratified vs class-conditioned — computed in the same bins
with the same hinge.

**The stop rule, fixed before the run.** The arm gets GPU only if, on the frozen
field, some `(strata, charge)` has BOTH

1. `sub_50_200` recall enrichment in the charged strata ≥ 2.0, and
2. Spearman rho between the per-tile dense-only and class-conditioned losses
   ≥ 0.5.

(1) alone is a cut that finds subhalos; (2) alone could be two statistics
agreeing because both are dominated by the same hosts. A pooled-vs-class control
rho is reported alongside, so "the dense cut agrees" can be read against "any
form agrees". `submit_knn_strata_arm.sh` refuses to launch without a passing
verdict and reads `strata` / `charge` out of it rather than taking them typed.

If the gate fails, that is the result: a density stratum is not a catalog-free
substitute for the class mask on this box. Write it down; do not retune the gate.

## What this does NOT remove

`knn_cdf` is translation-invariant — provable, and proven in
`test_loss_is_invariant_under_rigid_translation`. It cannot place an object. The
supervised `w_centre` stays on, and so does virial: `loss-term-ablation` measured
the minimal working set as virial + centre + low-k guard, and the 2026-08-21
free-field run without a centre term landed a median 0.414 Mpc/h from target
against a 0.150 search radius. This arm removes the label from the *kNN support
term*, not the membership supervision from the run.

The gate stays real Rockstar on held-out set9/set10. This surrogate is gameable
in principle and the expectation is that it will be gamed if the gate is not real.

## Run order

```bash
bash scripts/slurm/submit_knn_strata.sh              # 3D, set9, box-wide
PHASE=1 bash scripts/slurm/submit_knn_strata.sh      # 6D at lambda=4000 -- the
                                                     # space the arm trains in
# read verdict.pass, then ONLY if true:
KC_VERDICT=<json> bash scripts/slurm/submit_knn_strata_arm.sh
```

Run both 3D and 6D if there is queue for it. If the verdict flips between them,
that is itself the finding.


## The verdict (MEASURED, set9, jobs 54405/54406)

Both configurations FAIL, and not narrowly. 24 tiles, gate field `frozen`.

**The density cut selects *against* the deficit population.** `sub_50_200` is
2.76% of a tile interior; the densest eighth of frozen's queries is 1.7% of it —
a recall enrichment of **0.76**, i.e. below chance. The dense tail is host cores
and massive subhalos, not the 50-200p objects the deficit is made of:

| dense stratum (frozen, strata8_charge0) | 3D | 6D λ=4000 |
|---|---|---|
| host | 67.8% | 56.6% |
| unbound | 14.3% | 29.0% |
| sub_ge200 | 11.6% | 9.0% |
| **sub_50_200** | **1.7%** | **2.1%** |
| recall enrichment | **0.76** | **0.82** |

The synthetic fixture got the host fraction right (65-83% predicted, 67.8%
measured) and the subhalo side badly wrong: it predicted an impure-but-enriched
cut, and the box gives an anti-selective one.

**The pooled control kills the remaining hope.** The best rho on frozen is
+0.549 (3D, strata4_charge0-1) against a **pooled control of +0.594** — the
plain pooled loss tracks the class loss *better* than the density cut does. In
3D no config beats its control; in 6D two do and neither reaches 0.5 while
recall stays ≤ 1.02.

**Why it cannot be tuned out.** The deciding contrast is HR against frozen:

| recall enrichment, strata16_charge0 | HR | frozen |
|---|---|---|
| 3D | 0.66 | 0.57 |
| 6D λ=4000 | **5.03** | **0.73** |

In 6D the density cut works beautifully on HR — 5.03x, 12.3% of the dense
stratum is `sub_50_200` — and not at all on the candidate. A selection defined
by the candidate's own density cannot find objects the candidate does not yet
have. That is structural, not a hyper-parameter: the field the loss must act on
is exactly the field where the cut fails. No `(strata, charge)` fixes it, and
the 6x HR/frozen gap says a better rank or a finer partition will not either.

**Incidental confirmation of "partitioning is not reweighting."** The sweep
returns identical numbers for `strata16_charge0-1` and `strata8_charge0`, and
for `strata4_charge0` and `strata8_charge0-1` — those pairs charge the same 1/8
and 1/4 of the queries. The charged *fraction* determines the result; the number
of strata does not. The synthetic identity holds on real data.

**What stays.** `--knn-classes sub_50_200` remains the way this term is charged,
Rockstar label and all. The strata code stays in place, default off, because
`ratio_med_dense` / `knn_dense_k*` is a genuinely better *diagnostic* readout
than the pooled median (which reads 1.00 through a real defect) — but it is a
monitor, not an objective. The catalog-free objective is not available by this
route.

## Postscript: what about the original pooled loss?

Asked and answered from the saved round-5b JSONs (no new job needed) — see the
`unpaired` row added to `docs/sr2_knn_distance.md` §12. The pooled form is
catalog-free and it ranks the 20 working arms at **+0.878 (p 4e-07)** at λ=4000
against subhalo count: lower loss, fewer subhalos. It is not a weaker version of
the class-conditioned term, it points the other way.

This also explains the strata failure one level down. The strata arm was an
attempt to approximate "charge only subhalo particles" without a catalog; the
density cut lands on hosts (67.8% of the dense stratum), so it inherits the
wrong-sign behaviour of every host-including form. Two measurements, one cause.

Note the `+0.594` pooled control in the gate table above is not in tension with
the `+0.878` here: the control measures per-TILE agreement between two losses on
one field, and this measures per-ARM ranking against real Rockstar counts. Only
the second bears on whether a form is usable as an objective.

**The corner this leaves.** A catalog-free objective for this term needs a query
selector that (a) excludes host particles, since every host-including form has
the wrong sign, and (b) does not key on the candidate's own density, since the
candidate lacks the objects being selected for. Nothing in this line currently
satisfies both.