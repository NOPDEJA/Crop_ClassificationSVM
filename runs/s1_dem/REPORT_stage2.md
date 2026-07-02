# Stage 2 — DEM + Sentinel-1 vs S2-Only Baseline (Econ Subclass)

**Run date:** 2026-07-01  
**Model:** `stage2_s1_dem_model.joblib`  
**Best params:** n_components=150, gamma=0.5, C=10.0  
**Input:** Stage 1 econ predictions → 12,689,478 econ pixels flagged; 10,479,011 also true-econ in labels

---

## Results at a Glance

| Subclass | S2-Only P | S2-Only R | S2-Only F1 | DEM+S1 P | DEM+S1 R | DEM+S1 F1 | Δ F1 |
|----------|-----------|-----------|------------|----------|----------|-----------|------|
| orchards (1) | 0.739 | 0.691 | **0.714** | 0.639 | 0.296 | **0.405** | −0.310 |
| plantation (2) | 0.851 | 0.750 | **0.797** | 0.680 | 0.334 | **0.448** | −0.349 |
| field (3) | 0.690 | 0.861 | **0.766** | 0.441 | 0.937 | **0.600** | −0.166 |
| other_econ (4) | 0.846 | 0.022 | **0.044** | 0.625 | 0.004 | **0.008** | −0.036 |
| **Accuracy** | | | **0.751** | | | **0.509** | −0.242 |
| **Macro F1** | | | **0.580** | | | **0.365** | −0.215 |

> S2-only baseline: `2018/stage2_weighted_increased_report.csv` (2025-12-19), 4 subclasses, 91,959 test pixels  
> DEM+S1: `runs/s1_dem/stage2_s1_dem_report.csv` (2026-07-01), 4 subclasses, 92,396 test pixels

> **Caveats (found 2026-07-02):**
> 1. This DEM+S1 run was trained with Longan (2413) missing from `orchards_codes` — Longan pixels were labeled *other_econ* during Stage 2 training, inconsistent with Stage 3 and with the project LU mapping. Fixed in `stage2_weighted.py`; orchards recall and other_econ precision here are slightly depressed by it.
> 2. The S2-only baseline's SWIR-based features (BSI, NDBI, SWIR_NIR, SWIR_RATIO) were computed with swapped bands (see `runs/s1_dem/REPORT.md`), so the baseline itself understates what a correct S2 model can do.

---

## DEM+S1 Confusion Matrix (test set, 92,396 pixels)

|                   | pred orchards | pred plantation | pred field | pred other_econ |
|-------------------|---------------|-----------------|------------|-----------------|
| **orchards** (30K) | 8,876 | 3,644 | 17,475 | 5 |
| **plantation** (30K) | 2,900 | 10,034 | 17,066 | 0 |
| **field** (30K) | 1,112 | 775 | 28,112 | 1 |
| **other_econ** (2.4K) | 995 | 309 | 1,082 | 10 |

## S2-Only Confusion Matrix (test set, 91,959 pixels)

|                   | pred orchards | pred plantation | pred field | pred other_econ |
|-------------------|---------------|-----------------|------------|-----------------|
| **orchards** (30K) | 20,730 | 2,592 | 6,674 | 4 |
| **plantation** (30K) | 3,016 | 22,492 | 4,489 | 3 |
| **field** (30K) | 3,105 | 1,067 | 25,827 | 1 |
| **other_econ** (1.96K) | 1,207 | 270 | 438 | 44 |

---

## Key Observations

### 1. DEM+S1 fails badly at crop subclass separation (expected)

The model is overwhelmingly biased toward predicting "field": 17,475 true orchards and 17,066 true plantation pixels are predicted as field. This collapses orchards recall to 29.6% and plantation recall to 33.4%.

SAR backscatter and terrain features are **not designed to separate crop subtypes**:
- Orchards, plantations, and field crops all grow on similar flat/gentle terrain → DEM provides no discriminating signal
- SAR volume scattering from tree canopies vs. row crops does carry some structural signal, but it is overwhelmed by the sheer dominance of plantation (class 2) in the dataset (8.9M true pixels vs 411K orchards, 1.1M field)

### 2. Field recall is unusually high (93.7%) but precision is low (44.1%)

The model learned that "when uncertain, predict field." This is not because field is genuinely easier to classify with SAR — it reflects that both orchards and plantation look similar to field in DEM+SAR feature space, so they all collapse into the field prediction.

### 3. S2-only Stage 2 is dramatically better for tree-crop vs field separation

S2 spectral indices (NDVI, EVI, NDWI, SWIR ratios) encode crop phenology and canopy chemistry:
- Rubber/oil palm/coconut (plantation) have distinct multi-season NDVI signatures vs rice/cassava
- Durian/rambutan/mango (orchards) show persistent high NDVI with seasonal fruiting signals
- These differences are invisible to SAR and DEM

### 4. other_econ (class 4) is near zero in both models

Both runs achieve nearly zero recall for other_econ (15,973 training pixels after capping). This is a tiny class that cannot be learned at this sample size. It is noise in Stage 2 — Stage 3 will not meaningfully process it.

### 5. 14 additional all-NaN columns within econ subset

The imputer emitted warnings about 14 columns with no observed values specifically within the econ-pixel subset (indices 4, 11, 20, 27, 36, 43, 52, 59, 67, 74, 82, 89, 97, 104 after valid_cols filtering). These are S1 features from acquisitions that have no spatial overlap with the econ parcel areas. They were skipped by the imputer. A future fix: extend the NaN-column drop to be computed on the econ subset specifically, saving those 14 features from occupying any model capacity.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Econ pixels (Stage 1 pred) | 12,689,478 |
| True econ within those | 10,479,011 |
| Subclass distribution (raw) | plantation=8,913,568, field=1,138,428, orchards=411,042, other_econ=15,973 |
| Cap per group | 200,000 |
| Training size (after cap+split) | 431,181 train / 92,396 val / 92,396 test |
| Train distribution | orchards=140K, plantation=140K, field=140K, other_econ=11,181 |
| Best Nystroem n_components | 150 |
| Best gamma | 0.5 |
| Best C | 10.0 |
| max_iter (LinearSVC) | 5,000 |
| Features used | 109 (valid_cols from Stage 1) |
| NaN imputation | SimpleImputer(strategy='median') |

---

## Suggested Next Steps

1. **Add Sentinel-2 spectral indices — critical for Stage 2.** The subclass separation problem is fundamentally optical. A combined DEM+S1+S2 feature set is expected to dramatically improve orchards and plantation F1, since NDVI seasonality and SWIR ratios are the primary discriminators for tree crops vs field crops.

2. **Extend valid_cols to econ subset.** Drop the 14 columns that are all-NaN within econ pixels before fitting Stage 2, to avoid wasting model capacity on zero-information features.

3. **Consider oversampling other_econ or merging it.** 15,973 pixels is too few to train a reliable subclass. Either upsample aggressively or collapse it into the nearest group.

4. **Stage 3 is running** — but results will be degraded because Stage 2's low recall means ~70% of true orchards and ~67% of true plantation pixels are never passed to the correct Stage 3 sub-model.
