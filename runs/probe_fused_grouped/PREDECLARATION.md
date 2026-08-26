# PREDECLARATION — E2 fused-stack parcel-grouped probe

Written before launch, per docs/PLAN_2026-08-26_POSTREVIEW_EXECUTION.md E2.

**Script:** `probe_fused_grouped.py` (new; copies `probe_dry_season_grouped.py`'s
split/sampling/pipeline verbatim).

**Design:** two arms, identical rows, identical parcel-wise split (seed 42,
crop-wise parcel halving, N_PER_SIDE=4,000 pixels/crop/side, uniform prior).
- Arm A (control): 40 S2-index columns.
- Arm B (treatment): all 153 columns (+5 DEM, +108 S1). Own gamma = 1/153,
  never Arm A's gamma (1/40) reused.

**Read rule:** primary = per-class F1 (parcel split), B vs A on the same test
rows. This is a probe, not a gate — no cascade config changes on this result
alone.

**Falsifier threshold (predeclared):** any of Coconut / Mangosteen / Rambutan
/ Longan gains >= +0.10 F1 in B vs A -> record the **optical-ceiling branch**
(fused pipeline becomes the rare-class direction; V2(b) in the ceiling verdict
weakens). Langsat is reported but excluded from any claim (8 test parcels).

**Scope note:** this probe's numbers are never comparable to cascade numbers
(different split unit granularity per crop, different sampling) — they compare
only Arm B to Arm A, nothing else.

**Fold 2:** untouched. This script never loads `splits/split_assign.npy`.
