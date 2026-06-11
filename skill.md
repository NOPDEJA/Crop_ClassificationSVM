# Crop Pipeline Skill

Quick-reference for working with the LDD crop classification pipeline. Use as a prompt cheat sheet or task template.

---

## Stage Execution Prompts

### Step 3a — Compute DEM features
```
Run compute_dem_feature.py.
Input: dem/dem_47PQQ.tif (merged Copernicus DSM).
Output: indices/ELEVATION_dem.tif, SLOPE_dem.tif, ASPECT_dem.tif, TPI_dem.tif, ROUGHNESS_dem.tif.
verify: all 5 TIFs exist, shapes match label/label_47PQQ_buffered.tif, elevation range sane for Rayong (~0–500 m).
```

### Step 3b — Preprocess Sentinel-1 (requires SNAP on PATH)
```
Run S1_preprocess_snap.py.
Input: S1_data/*.zip SAFE files.
Output: S1_processed/*.tif  (2-band: VV dB, VH dB, terrain-corrected).
verify: output GeoTIFFs exist, band count == 2, pixel values roughly in range [-30, 5] dB.
```

### Step 3c — Compute SAR features
```
Run s1_compute_features.py.
Input: S1_processed/*.tif.
Output: indices/VV_<stem>.tif, VH_<stem>.tif, VV_VH_RATIO_<stem>.tif, RVI_<stem>.tif, RFDI_<stem>.tif, DPR_<stem>.tif, VVVH_DIFF_<stem>.tif.
verify: 7 TIFs per input file exist in indices/, shapes consistent.
```

### Step 3d — Compute S2 spectral indices
```
Run compute_indices.py (NDVI, EVI, NDWI, BSI, NDBI) and compute_extra_indices.py (MSAVI, SWIR_NIR, SWIR_RATIO).
Input: S2_data/*.tif multi-band composites.
Output: indices/<INDEX>_<date>.tif.
verify: per-date TIFs exist, NDVI values in [-1, 1].
```

### Step 5 — Align and stack all features
```
Run align_indices_labels.py.
Input: all TIFs in indices/, label/label_47PQQ_buffered.tif.
Output: aligned_features/svm_add_data_features_labels.npz  (X shape: [n_pixels, n_features], y shape: [n_pixels]).
verify: .npz loads cleanly, X.shape[1] equals number of index TIFs, y contains expected LU codes including economic crops (2101, 2302, 2403 etc.).
```

### Step 6a — Train Stage 1
```
Run stage1_weight_scale.py.
Input: aligned_features/svm_add_data_features_labels.npz.
Output: stage1_svm_weight_scale_increased.joblib, stage1_thresholds_svm_weight_scale_increased.json, stage1_svm_weight_scale_increased.npy (full predictions), stage1_report_svm_weight_scale_increased.csv.
verify: CSV report shows F1 > 0 for all 4 classes (econ, water, forest, others); .npy has same length as X rows in npz.
```

### Step 6b — Train Stage 2
```
Run stage2_weighted.py.
Depends on: stage1_svm_weight_scale_increased.npy (or .joblib to regenerate it).
Output: stage2_weighted_increased_model.joblib, stage2_weighted_increased.npy, stage2_weighted_increased_report.csv.
verify: report shows 4 subclasses (orchards=1, plantation=2, field=3, other_econ=4); stage2 .npy is non-zero only at indices where stage1 predicted econ.
```

### Step 6c — Train Stage 3
```
Run stage3_new_weight.py.
Depends on: stage1 + stage2 .npy predictions.
Output: stage3_{group}_weightscale_model.joblib per subclass group (orchards, plantation, field, other_econ).
verify: one .joblib per group, each report CSV contains the expected LU_CODEs.
```

---

## Common Debugging Patterns

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Shape mismatch between index TIF and label | Index not reprojected to 10 m / label CRS | Re-run compute_dem_feature.py / s1_compute_features.py using label raster as reference |
| Feature count in .npz is wrong | Extra/missing TIFs in indices/ | List `glob("./indices/*.tif")` — count must match expected; remove stale files |
| OOM during Stage 1 training | Too many samples after rebalance | Lower `CAP_ECON / CAP_OTHERS` in stage1_weight_scale.py |
| OOM during chunked prediction | PRED_CHUNK too large | Reduce `PRED_CHUNK` (default 2 000 000) in the relevant script |
| Stage 2 receives no pixels | Stage 1 predicted no class=1 pixels | Check stage1 .npy: `np.unique(pred, return_counts=True)` — if class 1 is missing, retrain Stage 1 |
| Stage 1 all predicted `others` | Feature ordering changed after retrain | Rebuild .npz with align_indices_labels.py before retraining |
| `gpt` not found | SNAP not on PATH | Add SNAP bin dir to PATH: `C:\Program Files\snap\bin` |
| Thai text garbled in CSV | Wrong encoding | Use `encoding="utf-8-sig"` (BOM) on all CSV writes |
| CRS is None on S2 band | Raw SAFE zip missing CRS | raster.py assigns EPSG:32647 as default — verify province is in UTM zone 47N |
| Forest class missing from Stage 1 report | forest_code pixels absent in label raster | Check LDD shapefile: codes 3100–3501 must appear in `LU_ID_L3` column |

---

## Feature Index Cheat Sheet

Feature order in the `.npz` is alphabetical from `glob("./indices/*.tif")`. Typical ordering with a full feature set:

```
ASPECT_dem.tif
BSI_<date1>.tif, BSI_<date2>.tif, ...
DPR_<s1_stem>.tif, ...
ELEVATION_dem.tif
EVI_<date>.tif, ...
NDBI_<date>.tif, ...
NDVI_<date>.tif, ...
NDWI_<date>.tif, ...
RFDI_<s1_stem>.tif, ...
ROUGHNESS_dem.tif
RVI_<s1_stem>.tif, ...
SLOPE_dem.tif
SWIR_NIR_<date>.tif, ...
SWIR_RATIO_<date>.tif, ...
TPI_dem.tif
VH_<s1_stem>.tif, ...
VV_<s1_stem>.tif, ...
VV_VH_RATIO_<s1_stem>.tif, ...
VVVH_DIFF_<s1_stem>.tif, ...
MSAVI_<date>.tif, ...
```

> Any change to this list (adding/removing files) requires re-running `align_indices_labels.py` **and** retraining all three stages. Saved `.joblib` models baked the old feature order in.

---

## Model File Reference

| File | Stage | Description |
|------|-------|-------------|
| `stage1_svm_weight_scale_increased.joblib` | 1 | 4-class SVM (econ/water/forest/others), Nystroem+calibrated |
| `stage1_thresholds_svm_weight_scale_increased.json` | 1 | Per-class probability thresholds from val-set search |
| `stage1_svm_weight_scale_increased.npy` | 1 | Full-dataset class predictions (uint8) |
| `stage1_prob_svm_weight_scale_increased.npy` | 1 | Full-dataset class probabilities (float32) |
| `stage2_weighted_increased_model.joblib` | 2 | Subclass SVM (orchards/plantation/field/other_econ) |
| `stage2_weighted_increased.npy` | 2 | Full-dataset subclass predictions (non-zero at econ pixels only) |
| `stage3_{group}_weightscale_model.joblib` | 3 | Per-subclass fine-grained LU_CODE classifier |

**Always pair a `.joblib` model with its matching `.json` thresholds and the `.npy` prediction array it was trained alongside.**

---

## Sampling / Hyperparameter Reference

| Parameter | Default | Where | Effect |
|-----------|---------|-------|--------|
| `SAMPLES_PER_LU` | 400 000 | stage1_weight_scale.py | Max samples per LU_CODE before superclass mapping |
| `CAP_ECON` | 1 000 000 | stage1_weight_scale.py | Max econ samples in training set |
| `CAP_FOREST` | 600 000 | stage1_weight_scale.py | Max forest samples |
| `CAP_WATER` | 500 000 | stage1_weight_scale.py | Max water samples |
| `CAP_OTHERS` | 800 000 | stage1_weight_scale.py | Max others samples |
| `ECON_BOOST` | 2.0 | stage1_weight_scale.py | Multiplicative boost to econ inverse-frequency weight |
| `PER_GROUP_CAP` | 200 000 | stage2_weighted.py | Max samples per Stage 2 subclass group |
| `PRED_CHUNK` | 2 000 000 | all stage scripts | Rows per prediction chunk (reduce if OOM) |
| `BLOCK_ROWS` | 512 | s1_compute_features.py | SAR processing rows per block (reduce if OOM) |

---

## Behavioral Principles (Karpathy Guidelines)

### Think Before Coding
Confirm which stage, which model variant, and whether the task targets root-level (current) or `2018/` (archived) before any change.

### Simplicity First
Research scripts — no abstractions for single-use operations, no config files unless a script is being generalized to multiple tiles/years.

### Surgical Changes
Match flat-script style: top-level `Config` block, `print()` for progress, relative paths. Don't restructure working scripts.

### Goal-Driven Execution
Every task needs a measurable outcome (metric improved, file produced, crash resolved). Don't mark done until output is verified.
