# Report: E1-E7 execution and the final fold-2 read

**Date:** 2026-08-28. **Executes:** `docs/PLAN_2026-08-26_POSTREVIEW_EXECUTION.md`, itself
§4 of `docs/HANDOFF_CODEX_2026-08-26_CEILING_VERDICT.md` after Codex review. Ran end to end
over 2026-08-27 to 2026-08-28 in one session (E1 committed `f012cbe` through E7 committed
`4e0f61a`).

---

## Headline result

**Final strict test macro F1: 0.2429** (5,500,269 rows) vs **M5's 0.2344** — **delta +0.0085**.
Lands at the very low end of the predeclared base scenario (0.24–0.26); well under the 0.28
upside. Fold 2 was read **exactly once**, per the plan's iron rule, at a predeclared operating
point selected only on fold 1's calibration half (alpha2=0.4, alpha3=0.5).

## What shipped in the final config

| component | source | gate |
|---|---|---|
| Stage 1 | M5, frozen | never touched |
| Stage 2 | E4 retune (n_components=1200, gamma=0.5x rule, C=30) on s2mass's exact pool/weights | G3 PASS (weights) + G4 PASS (hyperparams) |
| Orchards expert | E4 retune, same hyperparams, unweighted | G4 PASS; G5's weighted version excluded |
| Plantation expert | M5, frozen, unweighted | G5 FAILED, no change |
| Field expert | M5, frozen | never touched |

## Per-crop F1, M5 vs final

| crop | M5 | final | delta |
|---|---|---|---|
| Rice | 0.4181 | 0.4457 | +0.0276 |
| Cassava | 0.3451 | 0.3379 | -0.0072 |
| Pineapple | 0.4208 | 0.4205 | -0.0003 |
| Rubber | 0.8721 | 0.8658 | -0.0063 |
| Oil palm | 0.4158 | 0.5587 | **+0.1429** |
| Durian | 0.3827 | 0.3642 | -0.0185 |
| Rambutan | 0.0184 | 0.0406 | +0.0222 |
| Coconut | 0.0085 | 0.0095 | +0.0010 |
| Mango | 0.0659 | 0.0812 | +0.0153 |
| Longan | 0.0054 | 0.0024 | -0.0030 |
| Jackfruit | 0.0810 | 0.0198 | **-0.0612** |
| Mangosteen | 0.0132 | 0.0112 | -0.0020 |
| Langsat | 0.0000 | 0.0000 | 0.0000 |

Alive-crops count (F1 >= 0.01) held at **10/13** in both M5 and the final config.

## Reading the result

- **V3's mid-frequency prediction is confirmed.** The gain concentrates almost entirely in oil
  palm (+0.1429, driving most of the total +0.0085*13=+0.1105 summed delta) and rice (+0.0276).
  Oil palm's jump is the single largest per-crop movement measured in this whole project year.
- **The rare five stay near zero**, exactly as the ceiling verdict's V2 said to expect: coconut
  moved +0.0010 (0.0085 -> 0.0095), still far below any usable threshold; longan and langsat did
  not move meaningfully. Rambutan and mangosteen are technically "rare" but showed small
  positive/negative movement respectively (rambutan +0.0222, mangosteen -0.0020) — consistent
  with V3's note that gains are driven by parcels-to-learn-from, and rambutan has somewhat more
  support than coconut/longan/langsat.
- **Jackfruit regressed notably** (-0.0612, 0.0810 -> 0.0198) — this was not predicted by any of
  E1-E5's tune-half readings and is the one result in this report that deserves a follow-up
  question rather than a shrug. E4's orchards retune (n_components 600->1200, gamma halved, C
  10->30) changed the orchards expert's decision surface; jackfruit sits between durian's
  dominant mass and lower-support crops in that expert, and the retune may have pushed its
  boundary the wrong way even though it won the honest GroupKFold search and Gate G4 overall.
  Not chased further here — fold 2 is read, the plan's scope ends at reporting.
- **Durian, rubber, cassava, pineapple, longan, mangosteen** moved by small amounts in both
  directions (-0.02 to +0.00). These are small relative to the scale of change this cascade
  produces between retrains, but that is an impression, not a measurement: **no noise floor for
  per-crop E7 deltas has been estimated.** An earlier version of this bullet cited G3's
  three-draw result (mean +0.0078, range 0.0072-0.0087) as a "noise floor" — that was wrong and
  is corrected here. G3's nonzero mean is its measured *treatment effect*, and its between-draw
  spread describes sensitivity to the pool draw for *that* experiment; neither quantity bounds
  the retrain variability of an unrelated per-crop change in E7.

## Per-experiment summary (E1-E7)

| step | result | gate |
|---|---|---|
| E1 | instrumented `train_parcel_cascade.py` to save Stage-2/3 fit/cal indices + SHA-256 | n/a (hygiene) |
| E2 | fused-stack (DEM+S1+S2) parcel-grouped probe | falsifier NOT triggered (all watch crops < +0.10 F1); rare-crop ceiling is not optical |
| E3 | s2mass's +0.0081 replicated over 3 fresh pool draws | **G3 PASS** (mean +0.0078, range +0.0072 to +0.0087, all three favour treatment) |
| E4 | honest GroupKFold(3) hyperparameter retune, Stage 2 + orchards expert | **G4 PASS** (delta +0.0041); both components converge on n_components=1200, gamma=0.5x rule, C=30 |
| E5 | Stage-3 subtype mass inside plantation + orchards | **G5 FAIL** (delta -0.0012, alive-crops guard failed on jackfruit) |
| E6 | date-difference features | **skipped** (demoted/optional per plan; not run, no time-to-spare trigger met) |
| E7 | final consolidated cascade, single fold-2 read | **0.2429, +0.0085 vs M5** |

## What this means for the paper (per the plan's reporting tasks)

- The RF-comparison rewrite (`docs/S2_SVM_ANALYSIS.md`) and the "raise with the professor" note
  about more parcels (V6.1) are unchanged by this result and still stand as written in the
  ceiling-verdict handoff.
- The improvement narrative should now cite the ACTUAL number: +0.0085 total from this whole
  execution cycle, driven overwhelmingly by oil palm and rice, not a broad lift across crops.
  This is smaller than the "whole prior year bought +0.0096" comparator cited in the plan's
  context capsule — worth stating plainly rather than rounding up.
- The rare-five ceiling verdict (V2, "no credible reason to expect strict F1 0.33") is now backed
  by a completed, gated, fold-2-verified result, not just tune-half projections.

## Re-derivation recipe

- Final report: `runs/s2_2018_3date_parcel_e7_final/report_hard.csv`, `manifest.json`.
- M5 baseline: `runs/s2_2018_3date_parcel_m5/report_hard.csv` (macro F1 0.2344).
- Gate artifacts: `runs/s2mass_pool_sensitivity/gate_g3.json`,
  `runs/retune_stage2_orchards/gate_g4.json`, `runs/s3mass_experts/gate_g5.json`.
- Predeclarations: `runs/s2_2018_3date_parcel_e7_final/PREDECLARATION.md` (written and committed
  `a4c46f6` before the fold-2 read) and one per gated experiment (E2, E3, E4, E5).
