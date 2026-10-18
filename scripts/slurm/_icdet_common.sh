# Shared defaults for the IC-determinism test (icdet_*_cpu.sbatch). Sourced AFTER
# _reward_common.sh, never run. Anything already set (from the submitter's env
# file or a VAR=value argument) wins over these defaults.
#
# The test: do the IC ripples finer than the LR grid fix where outskirt subhalos
# sit today? The 256^3 run of the box holds them down to 0.78 Mpc/h (LR: 3.1);
# frozen SR2 sees only LR. See scripts/features/ic_determinism.py.

: "${BOX:=set9}"
: "${ICD_SIMS:=$ZFS/DMSR/int_redshift_same_cosmology}"
: "${ICD_DMO_RES:=256}"
: "${ICD_TAG:=dmo$ICD_DMO_RES}"
: "${ICD_DMO_DIR:=$ICD_SIMS/dmo-$ICD_DMO_RES/$BOX/PART_009}"
: "${ICD_HR_FIELD:=$DATA/hr/$BOX.npy}"
: "${ICD_ROOT:=$DMSR_REWARD_ROOT/ic_determinism}"
: "${ICD_RS_DIR:=$ICD_ROOT/${BOX}__$ICD_TAG}"
: "${ICD_DMO_CATALOG:=$ICD_RS_DIR/${ICD_TAG}_rockstar/halos_0.0.ascii}"
# the 256^3 run keeps the original names; any other resolution gets its tag in them
if [ "$ICD_TAG" = "dmo256" ]; then _icd_sfx=""; else _icd_sfx="_$ICD_TAG"; fi
: "${ICD_RIPPLES_JSON:=$ICD_ROOT/${BOX}${_icd_sfx}_ripples.json}"
: "${ICD_MATCH_JSON:=$ICD_ROOT/${BOX}${_icd_sfx}_match.json}"
: "${ICD_FIG:=$ICD_ROOT/${BOX}${_icd_sfx}_ic_determinism.png}"
: "${ICD_TRACK_JSON:=$ICD_ROOT/${BOX}${_icd_sfx}_track.json}"
: "${ICD_TRACK_FIG:=$ICD_ROOT/${BOX}${_icd_sfx}_track.png}"
# scaling readout: recovery vs object size in the coarse input's own particles
: "${ICD_SCALING_JSON:=$ICD_ROOT/${BOX}_scaling.json}"
: "${ICD_SCALING_FIG:=$ICD_ROOT/${BOX}_scaling.png}"

# track: same radius = r ratio within this factor; same direction = cosine >= this;
# same-patch objects are searched within ICD_POOL_RMAX R_vir of a matched host
: "${ICD_RADIUS_TOL:=1.25}"
: "${ICD_DIR_COS:=0.9}"
: "${ICD_POOL_RMAX:=3.5}"

# ripples gate: r(k) band between the LR (2.0) and 256^3 (8.0) grid limits, h/Mpc
: "${ICD_GATE_LO:=2.5}"
: "${ICD_GATE_HI:=6.0}"
: "${ICD_GATE_MIN:=0.3}"
: "${ICD_GATE_MARGIN:=0.2}"

# matching: hosts >= 20k HR p (~1.2e13 Msun/h); objects >= 400 HR p (= 50 dmo-256 p)
: "${ICD_HOST_MIN_NP:=20000}"
: "${ICD_OBJ_MIN_NP:=400}"
: "${ICD_MASS_TOL:=3}"
: "${ICD_MATCH_RVIR:=1.0}"
: "${ICD_MATCH_MIN_MPC:=0.1}"
: "${ICD_N_NULL:=10}"
: "${ICD_LAG_K:=512}"
: "${ICD_LAG_TOL:=1.0}"

_icd_one() { ls $1 2>/dev/null | head -1 || true; }
[ -n "${ICD_SR2_FIELD:-}" ] || ICD_SR2_FIELD=$(_icd_one "$DMSR_REWARD_ROOT/cache/sr2_base/${BOX}_seed0_*.npy")
[ -n "${ICD_HR_CATALOG:-}" ] || ICD_HR_CATALOG=$(_icd_one "$DMSR_REWARD_ROOT/halos/${BOX}__hr__hr/*_rockstar/halos_*.ascii")
[ -n "${ICD_SR2_CATALOG:-}" ] || ICD_SR2_CATALOG=$(_icd_one "$DMSR_REWARD_ROOT/halos/${BOX}__base__base/*_rockstar/halos_*.ascii")
mkdir -p "$ICD_ROOT"
echo "    icdet: box=$BOX dmo=$ICD_DMO_DIR out=$ICD_ROOT"
