# Stage 1 — DEM + Sentinel-1 vs S2-Only Baseline

**Run date:** 2026-06-12  
**Model:** `stage1_s1_dem.joblib`  
**Best params:** n_components=150, gamma=0.5, C=10.0  
**Training time:** ~6.5 h (12 CV fits + CalibratedClassifierCV refit on 2.03 M rows)

---

## Results at a Glance

| Class | S2-Only P | S2-Only R | S2-Only F1 | DEM+S1 P | DEM+S1 R | DEM+S1 F1 | Δ F1 |
|-------|-----------|-----------|------------|----------|----------|-----------|------|
| econ | 0.705 | 0.682 | **0.693** | 0.619 | 0.747 | **0.677** | −0.016 |
| water | 0.799 | 0.775 | **0.787** | 0.922 | 0.632 | **0.750** | −0.037 |
| others | 0.690 | 0.717 | **0.703** | 0.501 | 0.510 | **0.506** | −0.197 |
| forest | — | — | — | 0.864 | 0.773 | **0.816** | (new) |
| **Accuracy** | | | **0.712** | | | **0.667** | −0.045 |
| **Macro F1** | | | **0.728** | | | **0.687** | −0.041 |

---

## DEM+S1 Confusion Matrix (test set, 435 K pixels)

|          | pred econ | pred water | pred others | pred forest |
|----------|-----------|------------|-------------|-------------|
| **econ** (150K) | 112,079 | 1,485 | 29,532 | 6,904 |
| **water** (75K) | 7,358 | 47,381 | 19,920 | 341 |
| **others** (120K) | 52,644 | 2,498 | 61,164 | 3,694 |
| **forest** (90K) | 9,004 | 33 | 11,357 | 69,606 |

---

## Key Observations

### 1. Not a fair apples-to-apples comparison

The two models differ in multiple ways:

| Aspect | S2-Only | DEM+S1 |
|--------|---------|--------|
| Features | 15 optical spectral indices | 5 DEM terrain + 109 SAR features |
| Classes | 3 (econ / water / others) | 4 (+forest) |
| Test set size | 19,753 pixels | 435,000 pixels |
| "Forest" treatment | lumped into others | separate class |

The S2-only model was evaluated on ~22× fewer pixels, which makes its metrics less robust. The DEM+S1 evaluation is more reliable statistically.

> **Baseline caveat (found 2026-07-02):** the Phase 1 S2-only features BSI, NDBI, SWIR_NIR and SWIR_RATIO were computed with swapped bands — the composite's band order (sorted string order: …B08, B11, B12, B8A) did not match the index scripts' assumed order (…B08, B8A, B11, B12), so "SWIR1" actually read B12 and "SWIR2" read B8A. NDVI/EVI/NDWI/MSAVI were unaffected. The S2-only baseline is therefore weaker than a correctly-computed S2 model would be. Fixed in `compute_indices.py` / `compute_extra_indices.py` for future runs.

### 2. Forest is the standout win (F1 = 0.816)

The DEM+S1 model cleanly separates forest from non-forest — terrain and SAR backscatter strongly distinguish closed-canopy forest from agricultural land. The S2-only model could not do this (forest was lumped into "others").

### 3. "Others" performance collapsed (F1 0.703 → 0.506)

The large drop is structural, not a model failure:
- In the S2-only model, forest pixels were inside "others" — making that class easier to predict (it was the largest class and included easily-discriminated forest signatures).
- In the DEM+S1 model, forest is removed from "others", leaving it as a harder catch-all. The confusion matrix shows **52,644 econ pixels misclassified as others** — the main source of errors.
- Without spectral indices, distinguishing many "others" subtypes from economic crops is genuinely harder for SAR/terrain features alone.

### 4. Econ recall improved (0.682 → 0.747) at the cost of precision (0.705 → 0.619)

Balanced class weights and the low probability threshold (0.4) recover more economic crop pixels. Good for recall-heavy applications (finding where crops are), but generates more false positives.

> **Correction (2026-07-02):** an earlier version of this report claimed an ECON class-weight boost of ×2.0 was applied. Code review found that block was dead code — the trained model used plain `class_weight='balanced'`. The boost code has been removed from `stage1_weight_scale.py`.

### 5. Water precision jumped to 0.922 (SAR is highly discriminative for water)

Open water bodies have distinctly low SAR backscatter (specular reflection away from sensor), making them very easy to identify with VV/VH. Recall dropped (0.775 → 0.632) possibly because mixed/transitional water pixels are harder without spectral confirmation.

### 6. All probability thresholds hit the search floor (t = 0.4)

The threshold grid searched 0.4–0.95. Every class landed at 0.4 (the minimum), meaning no higher threshold improved F1. This suggests either:
- The model is conservative (probabilities spread across classes, so the boost from lower thresholds always helps recall more than it hurts precision).
- The search floor should be extended to 0.2–0.35 in the next run.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Train set | 2,030,000 pixels (from 2.9M rebalanced) |
| Val / Test | 435,000 / 435,000 |
| Class caps | econ=1M, others=800K, forest=600K, water=500K |
| Class weights | `balanced` (no ECON boost — see correction in Observation 4) |
| Best Nystroem n_components | 150 |
| Best gamma | 0.5 |
| Best C | 10.0 |
| PCA | 10 components |
| max_iter (LinearSVC) | 5000 |
| All-NaN columns dropped | 22 of 131 |
| Features used | 109 |

---

## Suggested Next Steps

1. **Add Sentinel-2 spectral indices** to the feature set and retrain. The DEM+S1 features provide terrain and structural context; S2 optical indices provide crop-type discrimination that SAR alone struggles with. The combined feature set should outperform both baselines.

2. **Lower the threshold search floor** to 0.2 — all thresholds hit the 0.4 minimum, so the optimal threshold may be below it.

3. **Investigate "others" confusion** — 52K econ pixels predicted as others. Check which LU_CODEs drive this (likely mixed-use or transitional parcels) and consider upweighting those specific codes.

4. **Run Stage 2 and Stage 3** on the DEM+S1 Stage 1 predictions to evaluate full pipeline crop-type accuracy.

5. **Increase Nystroem components** — the search selected n_components=150 (the maximum candidate). Expanding to [150, 200, 300] in a follow-up run may improve accuracy at the cost of training time.
