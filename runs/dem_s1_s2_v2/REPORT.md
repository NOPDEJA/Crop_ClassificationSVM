# Corrected-Gamma v2 Chain — Interim Results (DEM+S1+S2)

**Run date:** 2026-07-04 to 2026-07-05
**Status:** Both v2 chains (DEM+S1+S2 and S1+DEM-only) complete, all 3 stages each, plus end-to-end evaluation. Final answer below.
**Feature matrix:** `svm_dem_s1_s2_features_labels.npz` — same as the v1 combined run (134 used features after dropping 19 all-NaN columns).
**Config change from v1:** Nystroem `gamma` grid corrected from the off-scale `{0.5, 1.0, 2.0}` to `{None, 0.005, 0.02}` (≈1/n_features), `n_components` ceiling raised to `{150, 250}`. See `[[combined-run-kernel-dilution]]` memory for the full diagnosis. Everything else (seeds, caps, splits) is unchanged from v1.

## Headline: the gamma fix reversed the regression

The v1 combined run (`runs/dem_s1_s2`) had concluded that adding S2 indices made the cascade *worse* at every stage, and diagnosed RBF kernel dilution + an off-scale gamma grid as the cause (see `runs/dem_s1_s2/REPORT.md`). This v2 run tests that diagnosis directly — same data, same routing logic, only the gamma/capacity grid corrected.

### Stage 1 (4-class superclass: econ / water / others / forest)

| Metric | s1_dem v1 (pre-S2 baseline) | dem_s1_s2 v1 (broken gamma) | dem_s1_s2 v2 (corrected gamma) |
|---|---|---|---|
| Accuracy | 0.667 | 0.648 | **0.766** |
| econ F1 | 0.677 | 0.658 | **0.755** |
| water F1 | 0.750 | 0.738 | **0.875** |
| others F1 | 0.506 | 0.532 | **0.606** |
| forest F1 | 0.816 | 0.751 | **0.880** |

Every class improved past both baselines. Winning params: `C=1.0, n_components=250, gamma=0.02` — gamma landed mid-grid rather than pinned at an edge (`n_components` did hit its ceiling, worth widening to `{150,250,350}` in a future run).

### Stage 2 (econ subclass routing: orchards / plantation / field)

| Metric | s1_dem v1 | dem_s1_s2 v1 (broken gamma) | dem_s1_s2 v2 (corrected gamma) |
|---|---|---|---|
| Accuracy | 0.509 | 0.466 | **0.858** |
| orchards F1 | 0.405 | (recall 0.154) | **0.845** |
| plantation F1 | 0.448 | (recall 0.257) | **0.848** |
| field F1 | 0.600 | — | **0.880** |

`other_econ` (subclass 4) has an empty LU-code list in this mapping and correctly receives ~0 pixels in every run — not a bug. Winning params: `C=10.0, n_components=150, gamma=None` (≈1/n_features), consistent with the standalone gamma-scale experiment that first confirmed the fix (`runs/exp_gamma_scale`, accuracy 0.466→0.868 on a quick rerun of Stage 2 alone).

### Stage 3 (fine-grained LU_CODE within each subclass)

**Field crops** — clean, balanced 3-class comparison (10,500 test pixels/class in all three runs):

| Metric | s1_dem v1 | dem_s1_s2 v1 (broken gamma) | dem_s1_s2 v2 (corrected gamma) |
|---|---|---|---|
| Accuracy | 0.657 | 0.434 | **0.829** |
| Rice (2101) F1 | 0.791 | 0.379 | **0.981** |
| Cassava (2204) F1 | 0.599 | 0.191 | **0.741** |
| Pineapple (2205) F1 | 0.608 | 0.533 | **0.762** |

Unambiguous win — v2 beats both the broken-gamma combined run *and* the pre-S2 baseline on every field crop.

**Plantation crops:**

| Metric | s1_dem v1 | dem_s1_s2 v1 (broken gamma) | dem_s1_s2 v2 (corrected gamma) |
|---|---|---|---|
| Accuracy | 0.747 | 0.771 | **0.898** |
| Rubber (2302) F1 | 0.784 | 0.869 | **0.906** |
| Oil palm (2303) F1 | 0.698 | 0.102 | **0.897** |
| Coconut (2405) F1 | 0.311 | n/a (0 test support) | **0.668** |

v2 wins across the board; Oil palm in particular recovers from a near-total collapse (0.102) in the broken-gamma run.

**Orchards — a genuine nuance, not a clean win on raw accuracy:**

| Metric | s1_dem v1 | dem_s1_s2 v1 (broken gamma) | dem_s1_s2 v2 (corrected gamma) |
|---|---|---|---|
| Accuracy | 0.734 | 0.829 | 0.635 |
| Macro F1 | 0.227 | 0.288 | **0.519** |
| Durian (2403) F1 (majority class) | 0.848 | 0.905 | 0.712 |
| Rambutan (2404) F1 | 0.144 | 0.400 | **0.332** |
| Mango (2407) F1 | 0.086 | 0.037 | **0.902** |
| Longan (2413) F1 | 0.167 | 0.122 | **0.497** |
| Jackfruit (2416) F1 | 0.049 | 0.118 | **0.386** |
| Mangosteen (2419) F1 | 0.066 | 0.144 | **0.397** |

Both v1 baselines achieved their higher *accuracy* by nearly always predicting Durian (the majority class, ~82% of the v1 test set) — minority-class recall was 2–8%. v2's lower raw accuracy reflects a much more balanced classifier: minority-species recall jumps into the 27–55% range and Mango F1 goes from 0.037/0.086 to 0.902. **Macro-F1 is the fairer metric here, and v2 nearly doubles it.** This isn't a regression — it's the model actually learning the rare classes instead of defaulting to the majority one. Durian itself did lose some F1 (0.905→0.712), worth a closer look later (possibly needs its own threshold tuning or more per-LU samples), but the net effect across all seven orchard species is a clear improvement.

## Interpretation

The v1 combined run's conclusion — "S2 indices make things worse" — does not hold once the gamma/capacity grid is corrected. Every stage and nearly every class improves over both the broken-gamma combined run and the original pre-S2 (DEM+S1-only) baseline. This is strong evidence for the kernel-dilution diagnosis: the problem was never the S2 features themselves, it was RBF gamma being ~100x off-scale for the data's dimensionality.

## Final answer: yes, S2 indices genuinely help — decisively

The `s1_dem_v2` chain (DEM+S1 only, `aligned_features/svm_add_data_features_labels.npz`, 109 features, same corrected gamma grids as above) is now complete, giving a genuine apples-to-apples v2-vs-v2 comparison. **S2 wins every single metric, at every stage, for every class, with no exceptions.**

### Stage 1

| Metric | s1_dem_v2 (no S2) | dem_s1_s2_v2 (+S2) |
|---|---|---|
| Accuracy | 0.733 | **0.766** |
| econ F1 | 0.725 | **0.755** |
| water F1 | 0.830 | **0.875** |
| others F1 | 0.549 | **0.606** |
| forest F1 | 0.875 | **0.880** |

### Stage 2

| Metric | s1_dem_v2 | dem_s1_s2_v2 |
|---|---|---|
| Accuracy | 0.773 | **0.858** |
| orchards F1 | 0.746 | **0.845** |
| plantation F1 | 0.730 | **0.848** |
| field F1 | 0.843 | **0.880** |

### Stage 3

**Field:**

| Metric | s1_dem_v2 | dem_s1_s2_v2 |
|---|---|---|
| Accuracy | 0.812 | **0.829** |
| Rice F1 | 0.974 | **0.981** |
| Cassava F1 | 0.710 | **0.741** |
| Pineapple F1 | 0.746 | **0.762** |

**Plantation:**

| Metric | s1_dem_v2 | dem_s1_s2_v2 |
|---|---|---|
| Accuracy | 0.836 | **0.898** |
| Rubber F1 | 0.846 | **0.906** |
| Oil palm F1 | 0.857 | **0.897** |
| Coconut F1 | 0.537 | **0.668** |

**Orchards** (macro-F1 is the fair comparison here — see the raw-accuracy caveat above):

| Metric | s1_dem_v2 | dem_s1_s2_v2 |
|---|---|---|
| Accuracy | 0.537 | **0.635** |
| Macro F1 | 0.399 | **0.519** |
| Durian F1 | 0.651 | **0.712** |
| Rambutan F1 | 0.246 | **0.332** |
| Mango F1 | 0.825 | **0.902** |
| Longan F1 | 0.373 | **0.497** |
| Jackfruit F1 | 0.207 | **0.386** |
| Mangosteen F1 | 0.291 | **0.397** |
| Langsat F1 | 0.201 | **0.409** |

Unlike the v1-vs-v1 orchards comparison, here S2 wins on *both* raw accuracy and macro-F1 — no majority-collapse artifact muddying the picture.

### End-to-end econ recall (the number LDD actually cares about)

Computed via `evaluate_end_to_end.py` — composes the full Stage 1→2→3 cascade over all 24.3M pixels and scores against ground truth, so routing errors at any stage count against the final number.

| | s1_dem_v2 (no S2) | dem_s1_s2_v2 (+S2) | Δ |
|---|---|---|---|
| **Overall econ recall** | 47.3% | **61.7%** | **+14.4 pp** |
| Orchards recall | 44.2% | **51.7%** | +7.5 pp |
| Plantation recall | 47.4% | **63.4%** | +16.0 pp |
| Field recall | 47.7% | **53.0%** | +5.3 pp |

For reference, the *first-session* end-to-end baseline (pre-gamma-fix, DEM+S1 only) was 23.1%, and the broken-gamma combined run was 19.0%. The corrected-gamma combined chain's 61.7% is roughly **2.7x** the original pre-S2 baseline.

### Conclusion

1. The original "S2 makes things worse" finding was entirely an artifact of RBF gamma being ~100x off-scale — confirmed twice over (the quick gamma-scale experiment, and now this full apples-to-apples v2 rebuild).
2. Once gamma is corrected, **S2 indices provide a large, consistent, unambiguous improvement** at every stage of the cascade and in the end-to-end metric that matters most to LDD.
3. Recommended next step: promote `dem_s1_s2_v2` artifacts to be the project's production models (currently `runs/s1_dem` and `runs/dem_s1_s2` are the stale pre-fix baselines referenced in `CLAUDE.md`/`skill.md` — those docs should eventually point at `dem_s1_s2_v2` once this is adopted as the default pipeline).
4. Secondary follow-ups noted but not yet pursued: widen the Stage 1 `n_components` grid past its 250 ceiling; investigate Durian's F1 drop in the corrected-gamma orchards model (0.905→0.712) — likely needs its own threshold tuning given how the balanced classifier now spreads probability mass across 7 classes instead of collapsing to 1.

## Artifacts

- Stage 1: `stage1_dem_s1_s2.joblib`, `_report.csv`, `_report_confusion_matrix.csv`, `_thresholds.json`, `_pred.npy`, `_prob.npy`
- Stage 2: `stage2_dem_s1_s2_model.joblib`, `_report.csv`, `_confusion_matrix.csv`, `_meta.json`, `_pred.npy`, `_stats_per_lu.csv`, `_stats_per_group.csv`
- Stage 3: `stage3_dem_s1_s2_{orchards,plantation,field}_{model.joblib,report.csv,confusion_matrix.csv,meta.json}`
