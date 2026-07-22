# Gate 1 — operator coverage of subcell-shift virtual operators

_Go/no-go for the plan's central premise: do subcell-shift operators
`H_g = A∘T_g` expand the identifiable subspace vs a fixed `A`? Computed
2026-07-15, env `pjm`. Scripts in scratchpad; arrays saved alongside._

## Verdict

**PASS (nominal), with a sharp channel split under real noise:**

- Subcell-shift diversity massively expands nominal coverage — it collapses the
  unidentified null space of `A` (factor-2: **95%** recovered; s=8: **71%**).
- Folding in the **measured degradation noise η**, the realizable gain is
  **concentrated in displacement** (η≈0.008) and **largely lost for velocity**
  (η≈0.60) *unless* measurements are genuinely independent across operators.
- **Recommended framing: factor-2 octave cascade** (reuses `MultiScaleOperators`,
  ≥95% null recovery). The s=8 single-jump adds only weakly-covered modes that
  don't survive η.
- **Recommended first target: displacement channels only.**

This confirms and sharpens the prior "operator diversity is the lever" finding
(`cosmo-sr-identifiability`): the lever is real, but the *virtual* (single real
measurement, distributionally relabeled) form realizes mainly the displacement part.

## 1D sanity check (matches plan's analytic prediction)

Periodic length-512 signal, width-`w` box averaging, stacked over all subcell
shifts `g∈{0..w-1}` (`rank(stack)`; full dim 512):

| operator set | rank | nullity |
|---|---|---|
| **s=8 fixed A** | 64 | 448 |
| s=8, all 8 shifts | **505** | **7** |
| factor-2 fixed A | 256 | 256 |
| factor-2, shifts {0,1} | 511 | 1 |

The **505** reproduces the plan's predicted stacked rank exactly (7 common-null
modes at frequencies `k ∈ {64,128,…,448}`, the boxcar-DFT zeros).

## 2. Real 3D operator coverage

`G = Σ_g H_g^T H_g` on a 64³ HR crop. Because box-average and axis-aligned shift
are separable, the 3D eigenvalues are outer products of the 1D eigenvalues — the
full spectrum is exact without forming any large matrix. Comparison sets: fixed A /
x-axis shifts / xy shifts / xyz shifts. Full dim = 64³ = 262144.

**Nominal (per-case, normalized by max eigenvalue):**

| framing | case | rank | nullity | eff-rank | %modes >1e-1 / 1e-2 / 1e-3 / 1e-4 |
|---|---|---|---|---|---|
| s=8 | fixed A | 512 | 261632 | 512 | 512 / 512 / 512 / 512 |
| s=8 | xyz shifts | 185193 | 76951 | 4472 | 991 / 5771 / 20009 / 55901 |
| factor-2 | fixed A | 32768 | 229376 | 32768 | 32768 (all) |
| factor-2 | xyz shifts | 250047 | **12097** | 104408 | 86571 / 168771 / 218907 / 241087 |

Note the **soft tail**: nominal rank jumps hugely, but eigenvalues decay to
~6e-3·max, so the extra modes are *weakly* covered — which is exactly why noise
matters.

## 3. η-weighted identifiability (the honest picture)

Measured on held-out `set15`: `η = LR − A(HR)`, per-channel variance fraction
relative to signal: **displacement ≈ 0.008**, **velocity ≈ 0.60** (broadband,
near-white; 24–36% of η power above 0.5·k_Ny).

Posterior identifiability criterion (Gaussian): HR mode `k` is recoverable when its
accumulated coverage exceeds the noise floor, `λ_k / λ_single > η_frac`
(single-operator normalization ⇒ monotonic; this is the **C2 upper bound**, only
reachable with genuinely independent per-operator noise).

| framing | η-identifiable **disp** (fixed→xyz) | η-identifiable **vel** (fixed→xyz) |
|---|---|---|
| s=8 | 512 → 93525 (**183×**) | 512 → 18521 (36×) |
| factor-2 | 32768 → 218475 (**6.7×**) | 32768 → 98525 (3.0×) |

**Interpretation — C2 vs C3 (the crux the plan flags):**

- **C2 (true multi-operator):** each `H_g` is an independent noisy measurement, so
  noise averages down — even velocity benefits. This is only available with
  *physically distinct* measurements (e.g. true multi-resolution LR at 64/256,
  which we **have** — 350 boxes at 64, 13 at 256).
- **C3 (virtual operators):** we relabel **one** real fixed-A measurement
  `y = A x + η` as various `H_g` — no independent noise, so the velocity η=0.60
  floor is **not** averaged down. The realizable virtual-operator gain lives in
  **displacement**, whose η=0.008 floor sits well below the coverage tail.

![operator coverage](figures/gate1_operator_coverage.png)

Left cliff = fixed A (few modes fully covered, rest zero); shift-diversity extends
the coverage curve rightward. The disp η-floor (0.008) sits low enough to capture
most of the extension; the vel floor (0.60) captures only the leftmost modes.

## 4. Decisions for the next phase

1. **Gate 1: PASS** — proceed to the controlled synthetic study (Phase C).
2. **Framing: factor-2 octave cascade** (reuse `MultiScaleOperators`; ≥95% null
   recovery; s=8's extra modes don't survive η).
3. **First target: displacement-only** (`use_channels=[0,1,2]`, already supported).
   Treat velocity as noise-limited on the measured octave.
4. **C2 vs C3 is the real experiment**, not fixed-vs-shift. Prediction from this
   analysis: **C3 (virtual) should improve displacement** over fixed-A (C1); the
   **velocity/large-noise gains need C2 (true multi-resolution LR)**. Gate 2 (C3 >
   C1 across seeds) remains the binding test.
5. Independently, the fresh band-power result (`respow 0.47→0.94`) is a *different*
   working distributional channel — the operator-conditioned study should be
   measured against it, not in a vacuum.

## Reproduce

```
conda activate pjm
python scripts/gate1/gate1_1d.py        # 1D rank (505) check
python scripts/gate1/eta_spectrum.py    # η fractions + spectrum on set15
python scripts/gate1/coverage_3d.py     # 3D coverage + η-weighting tables
python scripts/gate1/plot_cov.py        # figures/gate1_operator_coverage.png
```
_(Scripts to be promoted into `operators/spectral_analysis.py` + `tests/operators/`
when the operator modules are written.)_
