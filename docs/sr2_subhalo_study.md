# SR2 subhalo failure study (Stages 0–2)

Research question: why does SR2 keep accurate global density statistics while
losing low-mass subhalos toward \(z=0\), and can nested stochastic sampling plus
oracle selection recover the missing conditional substructure?

**Scope of this document:** Stages 0–2 only, **\(z=0\)**. The released `G_z2.pt`
is recorded in the freeze but unused until matched \(z=2\) catnorm pairs exist.
No new model training until Gate A passes.

## Frozen baseline

```bash
conda activate pjm
python scripts/sr2/freeze_baseline.py
# One command that reproduces the field-level table:
bash runs/sr2_baseline/reproduce_field_table.sh
# Or:
python scripts/sr2/reproduce_field_table.py
```

Artifacts land in [`runs/sr2_baseline/`](../runs/sr2_baseline): `manifest.json`,
copied `freeze.yaml` / `rockstar.cfg`, and `field_table/`.

Config source of truth: [`configs/sr2_baseline/freeze.yaml`](../configs/sr2_baseline/freeze.yaml).

| Knob | Value |
|------|-------|
| Model | `external/SRS-map2map/SRmodel/G_z0.pt` |
| Data | `/zfsauton/scratch/yixiz/DMSR/paired_catnorm` (16 boxes) |
| Inference | `nsplit=8`, `pad=3`, `noise_mode=per_tile` |
| Test boxes | `set12–set15` |
| Nested seeds | `0 .. 31` |
| Halo finder | Rockstar (`external/rockstar/rockstar`) + `configs/sr2_baseline/rockstar.cfg` |

## Stage 0 checks

```bash
# Same LR, different noise; nested prefix stability (CPU smoke):
python scripts/sr2/verify_seeds.py --smoke-ng 8 --seeds 0,1,2,3
# Full LR tile (GPU recommended):
python scripts/sr2/verify_seeds.py --box set14 --seeds 0,1,2,3
```

Rockstar was built with TIPSY I/O stubbed (missing `libtirpc` on this cluster).
GADGET2 input is the supported path; see `external/rockstar/BUILD_NOTE.txt`.

## Stage 1 — localise the failure

**Large outputs go on scratch** (`/zfsauton/scratch/yixiz/DMSR/sr2_baseline/`).
Do not write GADGET / Rockstar particle dumps to home.

**Box-first (preferred):** finish 2–4 seeds on each of `set13–15` for box-to-box
variance. Do **not** chase seeds 9–31 on set12 — eight seeds already show
sampling variance is negligible.

```bash
# Preferred SLURM entry (scratch OUT, 4 seeds × set13–15):
sbatch scripts/slurm/sr2_stage1_boxes.sbatch

# Or manually:
python scripts/sr2/stage1_subhalo_diagnose.py \
  --boxes set13,set14,set15 \
  --seeds 0,1,2,3 \
  --density-probe \
  --out /zfsauton/scratch/yixiz/DMSR/sr2_baseline/stage1
```

Validate host matching before trusting class fractions:

```bash
python scripts/sr2/validate_host_match.py \
  --catalog .../halos/set12/hr/hr_rockstar/halos_0.0.ascii
python scripts/sr2/rematch_halos.py \
  --halos-root .../stage1/halos --boxes set12 --seeds 0,1,2,3,4,5,6,7 \
  --out .../stage1/rematch_set12
python scripts/sr2/audit_rockstar.py --stage1-halos .../stage1/halos/set12
```

Rockstar note: upstream `setup_config` forced `PERIODIC=0` whenever
`PARALLEL_IO=0`. We patch `external/rockstar/config.c` so frozen `PERIODIC=1`
is honored. Existing set12 catalogs were found with `PERIODIC=0` (HR≡SR still);
new boxes use the patched binary. `FULL_PARTICLE_CHUNKS=0` by default.

Reports field controls (P/T/r, density PDF, equilateral bispectrum) and halo
metrics (HMF, SHMF, \(\langle N_{\rm sub}|M_{\rm host}\rangle\), radial profile,
one-halo proxy, \(V_{\max}\)), plus host matching and subhalo classes
(recovered / shifted / biased / velocity_incoherent / merged / missing /
absent_peak / diffuse_peak).

Bootstrap: treat **boxes** as the independent unit (never crops or seeds).

## Stage 2 — nested oracle TTS / Gate A

```bash
# After Stage 1 with seeds 0..31:
python scripts/sr2/stage2_oracle_tts.py \
  --stage1 runs/sr2_baseline/stage1 \
  --n-max 32 \
  --out runs/sr2_baseline/stage2
```

Oracle objective is **not** the reported coverage metric (avoids score hacking).
`gate_a.json` decides:

* `pass_tts` → build practical scorers (Stage 6; later)
* `fail_unhelpful_diversity` → improve training / flow refinement before a selector
* `fail_noise_ignore` → stochastic refinement first

Do **not** claim calibrated conditional uncertainty from one HR per LR.

## Stages 3–6 (deferred)

Flow residual refinement, flow-time diagnostics, soft substructure losses, and
practical best-of-N selection start only after Gate A. Ablation arms A–E + Oracle
are defined in the project brief; do not train until Stage-1 redshift failure
decomposition (here: \(z=0\) localisation) and Gate A are done.
