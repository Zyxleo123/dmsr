# Reward-only residual refinement of frozen SR2

A second line alongside the supervised residual prior, not a replacement for it.
The supervised pipeline (`residual_prior.yaml`, `sample_oracle_candidates.py`,
`train_reward_distill.py`) is untouched and still the baseline every arm is
compared against.

The question this line asks: **can a correction to the frozen SR2 field be
learned from the catalog reward alone, with no paired HR residual as a training
target?**

Paired HR enters at exactly two points, both calibration, both a single number
chosen before training:

| what | where from | used for |
|---|---|---|
| amplitude bound `s` | `calibrate_correction_scales.py` | the size of the largest edit any policy may propose |
| coarse allowance `alpha` | the projection oracle | how much of the LR-visible field a policy may touch |

No per-example supervision, no HR reconstruction loss, no paired residual crop.
`tests/reward/test_gaussian_scripts.py` pins that the committed config cannot
quietly acquire one.

---

## The one correction transform

`cosmo_sr.reward.correction` is the single place an action becomes an edit.

With `A` the block-average degrader to the LR grid and `A^†` the block-constant
broadcast back,

```
P_R = A^† A          the blockwise coarse component  (what the LR field sees)
P_N = I - P_R        the within-block, zero-mean part (what it cannot)
```

| mode | `delta` | meaning |
|---|---|---|
| `none` | `u` | unconstrained |
| `block_null` | `P_N u` | LR-invisible edits only |
| `block_leaky` | `P_N u + alpha P_R u` | bounded coarse allowance (**default**) |
| `split` | `P_N u + A^† c` | coarse part from its own LR-grid head; `A delta = c` exactly |

`block_null` is **not** the default, deliberately. Forcing `A delta = 0` is the
exact null-space projection `cosmo_sr.reward.base` already refuses to apply to
`Psi_hat`, because enforcing LR consistency cell by cell fights the
phase-coherent collapse that makes halos. Whether it is affordable is an
empirical question — the projection oracle's question.

Actions are bounded before composition, `u = s ⊙ tanh(h)`, with **separate `s`
and `alpha` for displacement and velocity**. The saturated fraction of `tanh` is
measured and returned on every call: a policy pinned against its own bound is one
whose gradient has vanished and whose samples are set by `s`, not by the network.

`amplitude = 0` short-circuits to an exactly zero edit, and
`compose(..., residual_scale=0)` returns `psi_base` itself. Those are the only
bit-exact SR2 fallbacks — see the correction below.

### Correction to a false claim

`cosmo_sr/reward/model.py` used to say that because the diffusion output head is
zero-initialised, "an untrained model predicts `eps = 0` **and the composed field
starts exactly at the frozen SR2 output**". The first half is true; the second
does not follow. DDIM starts from `u_T ~ N(0, I)`, and with `eps_hat = 0` the
step reduces to `u_next = (alpha_next / alpha_cur) * u` — the initial noise
*rescaled*, and amplified near `t_max` by `1/alpha(t_max)`. That is the only
reason `x0_clip` exists. A zero head therefore emits a residual of the clip's
amplitude, not of zero amplitude.

Only `residual_scale = 0` recovers frozen SR2 bit for bit, because
`compose` short-circuits rather than adding `0 * dPsi` (which would be NaN for a
diverged model). `tests/reward/test_zero_init_claim.py` measures all three facts
and fails if the sentence reappears in the docstring.

---

## Phase 1.2 — the projection oracle

For paired diagnostic boxes and several fixed SR2 seeds:

```
B = Psi_SR2 ,  r = Psi_HR - B ,  T_alpha(r) = P_N r + alpha P_R r
X = B + T_{alpha_dis}(r_dis) + T_{alpha_vel}(r_vel)
```

`alpha = 1` gives `Psi_HR` exactly (`T_1 = I`), so the sweep interpolates between
the frozen baseline's coarse scales and HR's while holding the within-block part
at HR's. Three sweeps — `joint`, `disp_only` (`alpha_vel = 1`), `vel_only`
(`alpha_dis = 1`) — plus `sr2` and `hr` anchors. Complete periodic box assembly
and the real Rockstar pipeline; no surrogate.

**This stage writes no training data.** Fields are scored and discarded.

### Decision rule

1. Reject the hard null projection if `alpha = 0` is *damaged* on any primary
   metric relative to `alpha = 1`.
2. Otherwise take the **smallest** allowance whose paired difference from
   `alpha = 1` is indistinguishable (or better) on **every** primary metric.
3. If none qualifies, recommend `alpha = 1` — no allowance was shown affordable,
   so none is claimed.
4. Displacement and velocity are decided from their own sweeps, separately.

`undetermined` (fewer than two paired boxes) never counts as passing: too little
data to resolve a difference is not evidence that there is none.

Uncertainty is a **box bootstrap** — seeds averaged within a box first, boxes
resampled with replacement — and every comparison against `alpha = 1` is
**paired on the box**. `tests/reward/test_projection_oracle.py` includes the case
that shows why: a constant +1 effect under box-to-box scatter of ±100 is resolved
by the paired bootstrap and invisible to the unpaired one.

### Report schema

`$DMSR_REWARD_ROOT/audits/projection_oracle/<run>/projection_oracle_report.json`

```jsonc
{
  "run_name": "proj0",
  "n_rows": 195,
  "arms_present": ["sr2", "hr", "joint_a0", ...],
  "arms_missing": [],
  "reference_arm": "joint_a1",
  "bootstrap": {"n_boot": 2000, "seed": 0, "ci": 0.95,
                "unit": "box (seeds averaged within a box first)",
                "paired": "comparisons against the reference are paired on the box"},
  "primary_metrics": ["R_occ_reliable", "host_recovery_fraction", "density_power_error"],

  "per_arm": {
    "<arm>": {
      "arm": "joint_a0p25", "alpha_disp": 0.25, "alpha_vel": 0.25, "sweep": "joint",
      "n_rows": 12, "boxes": ["set8", ...], "seeds": [0, 1, 2],
      "metrics": {                       // every REPORTED metric
        "<metric>": {"n_boxes": 4, "mean": .., "lo": .., "hi": .., "se": ..}
      },
      "vs_reference": {                  // absent for sr2/hr/the reference itself
        "<metric>": {"arm": .., "reference": .., "metric": ..,
                     "higher_is_better": true,
                     "arm_mean": .., "reference_mean": ..,
                     "diff": {"n_boxes": .., "n_paired_boxes": .., "mean": ..,
                              "lo": .., "hi": .., "se": ..},
                     "verdict": "improved|indistinguishable|damaged|undetermined"}
      },
      "occupation_per_host_bin": {"host_bin_0": {mean, lo, hi, se, n_boxes}, ...},
      "abundance_per_sub_bin":   {"sub_bin_0":  {mean, lo, hi, se, n_boxes}, ...},
      "reconstruction_rel_rms_vs_hr_max": 3e-7   // alpha=1 arm only
    }
  },

  "decisions": {
    "<sweep>": {
      "sweep": "disp_only", "reference_arm": "joint_a1", "varying": "alpha_disp",
      "hard_null_rejected": true,
      "hard_null_damaged_metrics": ["R_occ_reliable"],
      "recommended": {"alpha": 0.25, "arm": "disp_a0p25"},
      "blocked_by": {"disp_a0": {"R_occ_reliable": "damaged"}},
      "arms": {"<arm>": {"alpha": .., "metrics": {"<metric>": <comparison>}}},
      "metrics": [...], "ci": 0.95, "n_boot": 2000
    }
  },

  "recommendation": {
    "alpha_disp": 0.25, "alpha_vel": 0.0,
    "hard_null_rejected": true,
    "correction_mode": "block_leaky",
    "source_sweeps": {"alpha_disp": "disp_only", "alpha_vel": "vel_only"}
  },
  "scope": "This report chooses a constraint only. ... no training example was produced."
}
```

Alongside it: `projection_oracle_report.md` (the same content as tables),
`projection_oracle_arms.csv` (one row per arm, with verdicts), and
`projection_oracle_rows.csv` (every scored row, flat — so figures can be redrawn
without recomputation).

**The recommendation is not applied automatically.** Paste `alpha_disp` /
`alpha_vel` into `configs/reward/gaussian_policy.yaml` yourself.

### Measured result — run `proj0`, 2026-08-04

172 rows: 4 val boxes (set8–set11) × 3 fixed SR2 seeds × 15 arms, complete
periodic boxes through the real Rockstar pipeline. Self-consistency holds — the
`alpha = 1` arm reproduces `Psi_HR` to 3.7e-8 relative RMS, and its catalog
metrics match the independently-run `hr` arm (`R_occ_reliable` −3.53 vs −3.49,
host recovery 0.998 vs 1.0).

**The hard null projection is rejected, decisively.** Every `alpha < 1` is
damaged on all three primary metrics, monotonically, in all three sweeps.

| arm | gap closed | `low_k_change` | `density_power_error` | host recovery | subhalos |
|---|---|---|---|---|---|
| SR2 | 0% | 0 | 0.023 | – | 36 775 |
| `alpha = 0` | 69.0% | ~0 | **0.954** | 0.898 | 60 833 |
| `alpha = 0.1` | 70.9% | 0.034 | 0.853 | 0.892 | 63 396 |
| `alpha = 0.25` | 75.1% | 0.086 | 0.699 | 0.880 | 67 538 |
| `alpha = 0.5` | 85.3% | 0.172 | 0.439 | 0.846 | 78 204 |
| `alpha = 1` (=HR) | 100% | 0.344 | ~0 | 0.998 | 81 254 |

"gap closed" is the fraction of the SR2→HR distance in `R_occ_reliable`.

The decisive number is **not** the occupation gap — `alpha = 0` still recovers
69% of it. It is the **density**. Applying HR's within-block component without
its coarse component makes the density field *40× worse than not editing at all*
(0.954 vs SR2's 0.023). The two parts of the residual are not separable: moving
particles inside a block without the coherent bulk motion of the block destroys
the phase coherence that produces the density field. That is a physical result,
and it is the strongest argument against a hard null projection — stronger than
the occupation number alone.

Split by field, the damage is almost entirely **displacement**:

| | gap closed | `low_k_change` | `density_power_error` |
|---|---|---|---|
| `disp_a0` (kill coarse displacement) | 67.1% | 0.339 | 0.954 |
| `vel_a0` (kill coarse velocity) | **97.5%** | 0.058 | ~0 |

Constraining velocity's coarse component costs 2.5% of the gap and cuts
`low_k_change` by 6× (0.344 → 0.058) with no density damage. The decision rule
still calls it *damaged* — the CI on `R_occ_reliable` excludes zero — so it is
not free, just nearly so. If a feasibility budget forces a choice, that is where
to spend it.

### The blocker this run exposed, and what was done about it

The committed feasibility thresholds in `reward.yaml` were marked
`calibrated: false`, and this run showed what that cost: **no arm of the sweep
passed both of them, including HR itself.**

| threshold | placeholder | HR (`alpha = 1`) | `alpha = 0` |
|---|---|---|---|
| `low_k_change_max` | 0.02 | **0.344 — 17× over** | ~0 ✓ |
| `density_power_error_max` | 0.05 | ~0 ✓ | **0.954 — 19× over** |

A feasibility filter that declares the target field infeasible cannot be used to
filter approximations to it. With those thresholds in force, the Phase 2 support
gate would have reported zero feasible candidates *regardless of how good the
policy is* — a "no support" verdict that says nothing about the reward.

Two changes, both now in `reward.yaml`.

**Calibrated thresholds** (`calibrate_constraints.py`, 8 train boxes, margin
1.5, `low_k_fraction` 0.1). The three HR-referenced bounds are 1.5× the worst
frozen-SR2-vs-HR value; `low_k_change` has no baseline, so it is 0.1× the median
LR-visible difference between two independent HR boxes.

| threshold | placeholder | calibrated | from |
|---|---|---|---|
| `low_k_change_max` | 0.02 | 0.139595 | 0.1 × HR box-to-box scatter 1.39595 |
| `displacement_power_error_max` | 0.40 | 0.31504 | 1.5 × worst baseline 0.21003 |
| `density_power_error_max` | 0.05 | 0.03751 | 1.5 × worst baseline 0.02501 |
| `lr_consistency_error_max` | 0.05 | 0.689801 | 1.5 × worst baseline 0.45987 |

The LR-consistency placeholder was unsatisfiable by anything: frozen SR2's own
value is 0.44 and HR's is 0.47, so it is a property of the degrader, not of the
edit. `diversity_min` stays 0.05 and stays uncalibrated on purpose — it is a
collapse detector, not a fidelity measure.

**Severity.** The catalog is the objective, so a field statistic being over its
calibrated bound is now a reason to look rather than an automatic rejection.
Each constraint carries a level, and `block` is the default for anything the
config does not name:

| constraint | severity | why |
|---|---|---|
| `low_k_change` | `block` | a candidate that rewrote the LR-visible scales is not a *correction* to SR2 |
| `diversity` | `block` | a collapsed sampler scores well and is useless |
| `density_power_error` | `critical` | non-blocking, but counted and printed everywhere |
| `displacement_power_error` | `warn` | dominated by the frozen baseline (already 21% off) |
| `lr_consistency_error` | `warn` | dominated by the degrader (baseline 0.44) |

A downgraded constraint stops *rejecting*; it never stops being *measured*. Every
breach lands on the row (`constraint_warnings`, `constraint_critical`) and in
every report with its severity, and `check_feasible` returns the blocking
breaches only so a caller cannot accidentally read "feasible" as "clean".

`critical` exists for exactly the mode this run measured: `alpha = 0` closes 69%
of the occupation gap **and** makes the density 40× worse than not editing. That
candidate is now feasible, which is the point — but it is impossible to report as
a clean win. The support gate therefore adds a criterion,
`improvement_is_not_a_critical_breach` (`min_clean_positive`, default 1): if
every positive candidate carries a critical breach, the gate fails, because that
is the density-collapse mode rather than a result.

Note what this leaves: `low_k_change` and `diversity` are now the entire blocking
filter. `low_k_change = 0.139595` would still reject the HR field itself (0.344),
by design — the residual may nudge the scales SR2 already gets right, not rewrite
them — but it is the one bound that can still stop a candidate on field grounds,
so it is worth knowing that is where all the remaining veto power sits.

---

## Phase 2 — the Gaussian residual U-Net

Called a **Gaussian residual U-Net**, not an MLP. One step, no sampler loop.

```
input   [Psi_base, U(y_lr)]                     12 channels
trunk   two-level 3D U-Net, width 48, 2 residual blocks/level,
        pointwise ChannelGroupNorm3d, SiLU, circular padding, no attention,
        gradient checkpointing
heads   1x1x1 convs at coarse / middle / fine -> (mu_s, log sigma_s)
action  a_s = mu_s + sigma_s * eps_s
field   h = sum_s upsample_s(a_s)               fixed interpolation
edit    delta = CorrectionTransform(h)
```

The sampled **coefficient fields `a_s`** are the action, not `h` and not `delta`.
That is what gives an exact log probability

```
log pi_theta(a | B, Y) = sum_s log N(a_s; mu_s, sigma_s^2)
```

with no ELBO and no score estimate. A policy defined on `delta` would have no
density at all — `P_N` is a projection, so `delta` lives on a measure-zero
subspace.

Means are zero-initialised; sigmas start small, **nonzero**, configurable, with
coarse below middle/fine and separate displacement/velocity scales. `sigma_min`
and `sigma_max` are enforced by clamping `log sigma` itself, so the floor still
holds once training has moved the head weights.

Noise is a **global coordinate-aligned field**: `eps_s` at global index `i`
depends on `(seed, scale, i)` alone, generated per fixed-size block of the global
lattice. Overlapping tiles therefore read identical numbers, and a tiled full-box
sample equals an untiled one — the property `sample_policy_box` relies on and
`test_gaussian_policy.py` measures.

### Reward-only training

```
L_G = - sum_i w_i log pi_theta(a_i | B_i, Y_i) / sum_i w_i
      + beta KL(pi_theta || pi_ref)
      + lambda_edit C_edit
```

* `w_i = 1[feasible] 1[A_i > 0] exp(clip(A_i, 0, A_max)/tau)` — the existing
  bounded weights, unchanged.
* Replay selection is the existing rule, unchanged: the ensemble must be feasible
  and top-quantile **and** the chunk's own leave-one-out contribution positive.
* `a_i` is **detached**. Reward-weighted maximum likelihood on samples the
  behaviour policy already produced — not a policy gradient through a
  differentiable surrogate of Rockstar.
* `pi_ref` is a frozen copy of the initial policy; both are diagonal Gaussians so
  the KL is closed form.
* **No HR reconstruction term.** One would make "the reward improved the catalog"
  unfalsifiable.

The behaviour action is reconstructed exactly rather than stored: `a_s = mu_s^b +
sigma_s^b eps_s` and `eps` is global, so one `no_grad` forward pass of the
behaviour checkpoint recovers it. The behaviour checkpoint's **parameter hash** is
verified against the action records first — reconstructing `a_i` under the wrong
parameters would reinforce a different action than the one that earned `w_i`, and
that has to stop the job.

The candidate manifest is written in `score_oracle.py`'s format on purpose, so
constraints, Rockstar, chunk attribution, credit assignment and `build_replay.py`
are reused **unchanged** and the two arms cannot drift into measuring different
things.

### Support gate (Phase 2.3) — report schema

`$DMSR_REWARD_ROOT/oracle/<run>/support_gate.json`

```jsonc
{
  "run_name": "gauss_ref_k16",
  "untrained_reference_policy": true,
  "policy_hash": "…",
  "correction": { …the CorrectionConfig that produced the candidates… },
  "reliable_host_bins": [0,1,2,3], "upper_reliable_host_bins": [2,3],

  "thresholds": {"min_feasible_positive": 5, "min_ess": 8.0,
                 "min_improved_reliable_bins": 2, "min_improved_upper_bins": 1,
                 "max_tanh_saturation": 0.05,
                 "min_occupation_spread": 0.02, "min_subhalo_count_spread": 0.02},
  "thresholds_note": "Report thresholds, not scientific truths …",

  "n_candidates": 192, "n_boxes": 4, "amplitude_arms": [0.25, 0.5, 1.0],

  "pooled":  { <arm summary> },
  "per_arm": {"amplitude_0.25": { <arm summary> }, …},

  // <arm summary>:
  //   n, n_feasible, feasible_fraction,
  //   n_feasible_positive, positive_fraction_of_feasible,
  //   ess,
  //   dR_occ_reliable: {min, median, max},
  //   dR_abund_median,
  //   occupation_spread_per_bin[], occupation_spread_mean,
  //   subhalo_count_spread, host_count_spread,
  //   tanh_saturation_max,
  //   improved_bin_counts {bin: n},
  //   best_candidate {box, seed, amplitude, dR_occ_reliable,
  //                   improved_reliable_bins, improved_upper_bins}

  "candidates": [ { "box", "seed", "amplitude", "projection_mode",
                    "alpha_disp", "alpha_vel", "feasible", "violations",
                    "R_occ_reliable", "R_occ", "R_cat", "R_abund",
                    "dR_occ_reliable", "dR_cat", "dR_abund",
                    "n_subs_full_box", "n_hosts_full_box",
                    "occupation_per_bin", "d_occupation_per_bin",
                    "improved_reliable_bins", "improved_upper_bins",
                    "tanh_saturated_fraction", "coarse_fraction",
                    "low_k_change" } ],

  "criteria": {
    "feasible_positive_examples": {"value", "threshold", "passed", "detail"},
    "effective_sample_size":      { … },
    "improved_reliable_bins":     { … },
    "catalog_responds":           { … },
    "bound_is_not_the_model":     { … }
  },
  "failed": [], "passed": true, "verdict": "support_usable",
  "next_steps": [ …only when failed… ]
}
```

`dR_occ = R_occ_reliable(B + delta) - R_occ_reliable(B)` is a **paired** difference
against the candidate's own frozen SR2 realisation. SR2's seed-to-seed scatter is
larger than the effect, so an unpaired version measures the seed.

`R_cat` and `R_abund` are logged, and abundance-only improvement never counts as
success.

If the gate fails, the report prints, in order: sweep amplitude and the
multiscale variance allocation; try latent optimisation/search over `a_s`; add
shaped phase-space or density-collapse diagnostics; and **do not assume diffusion
will fix zero reward variation** — a richer sampler over the same action space
with the same reward has the same support.

---

## Not built yet, and why

**Phase 3 (reward-only diffusion)** and **Phase 4 (SR2-feature-conditioned
variant)** are deliberately not in this change. Phase 3 is gated on the Gaussian
policy finding usable support, and Phase 4 on the independent policy producing
off-manifold edits. Both gates are measurements that have not been taken —
building either now would mean choosing an architecture before the evidence that
selects it exists.

When Phase 3 does happen: an EDM-preconditioned or stable v-prediction sampler,
bootstrapped from the structured Gaussian reference policy rather than from HR
residuals, modelling the raw action *before* `tanh` and projection so replay
actions reproduce exactly. Not the current zero-head DDIM behaviour as a
reference distribution — see the correction above. No DDPO until reward-weighted
diffusion repeatedly produces feasible improvements.
