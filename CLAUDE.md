# LDD Crop Classification — SVM Pipeline

Research project with LDD (Land Development Department, Thailand) to classify agricultural land use in Rayong province. The model produces pixel-level crop type maps from a combination of **Copernicus DEM terrain features**, **Sentinel-1 SAR backscatter**, and **Sentinel-2 spectral indices**.

---

## Project History

| Phase | Data | Status | Location |
|-------|------|--------|----------|
| Phase 1 | Sentinel-2 spectral indices only | Archived | `2018/` folder |
| Phase 2 (current) | DEM terrain + Sentinel-1 SAR + Sentinel-2 indices | Active | Root-level scripts |

> **Rule:** Everything inside `2018/` is the old S2-only experiment. Do not modify those files unless specifically asked. All current work targets root-level scripts.

---

## Repository Structure

```
SVM_attempts/
├── CLAUDE.md
├── skill.md
│
│  ── Utilities ──────────────────────────────────────────────────────────────
├── raster.py                   Sentinel-2 band reprojection + alignment
├── parsing.py                  S2 XML metadata parser (BOA offset / quantification)
├── zip_manager.py              Zip archive reader for raw .SAFE zips
├── credential.py               Copernicus credentials (username/password)
├── tile_selector.py            MGRS tile extraction from province shapefile
│
│  ── Download ──────────────────────────────────────────────────────────────
├── tile_download.py            Download Sentinel-2 from Copernicus DataSpace
├── S1_download.py              Download Sentinel-1 GRD from Copernicus DataSpace
│
│  ── DEM pipeline ──────────────────────────────────────────────────────────
├── merge_dem.py                Merge Copernicus DSM tiles → dem/dem_47PQQ.tif
├── compute_dem_feature.py      DEM → Elevation, Slope, Aspect, TPI, Roughness TIFs
│
│  ── S1 SAR pipeline ───────────────────────────────────────────────────────
├── S1_preprocess_snap.py       SNAP GPT graph: orbit → denoise → calibrate → speckle
│                               filter → terrain correction → dB → GeoTIFF
├── s1_compute_features.py      VV, VH, VV_VH_RATIO, RVI, RFDI, DPR, VVVH_DIFF
│
│  ── S2 spectral indices ─────────────────────────────────────────────────
├── compute_indices.py          NDVI, EVI, NDWI, BSI, NDBI per S2 date
├── compute_extra_indices.py    MSAVI, SWIR_NIR, SWIR_RATIO per S2 date
│
│  ── Label preparation ─────────────────────────────────────────────────────
├── rasterize_parcel.py         LDD shapefile (LU_ID_L3) → 10 m label raster
├── buffer_labels.py            3-pixel erosion to reduce mixed-pixel edge noise
│
│  ── Feature alignment ──────────────────────────────────────────────────────
├── align_indices_labels.py     Stack all index TIFs → aligned_features/svm_add_data_features_labels.npz
│
│  ── SVM training (current 3-stage) ─────────────────────────────────────────
├── stage1_weight_scale.py      Stage 1: 4-class SVM (econ/water/forest/others)
├── stage2_weighted.py          Stage 2: subclass within economic crops
├── stage3_new_weight.py        Stage 3: per-subclass fine-grained LU_CODE
├── stage3_weighted.py          Stage 3 alternate variant
│
│  ── Analysis / evaluation ──────────────────────────────────────────────────
├── features_important_check.py Permutation feature importance
├── check.py                    Sanity checks
├── evaluate_against_buffered_label.py  Compare prediction vs buffered label
├── class_count_stage1.py       Count pixels per predicted class
├── plot_performance.py         Plot P/R/F1 over time from CSV reports
│
│  ── Labels & data ──────────────────────────────────────────────────────────
├── label/
│   ├── label_47PQQ.tif                Raw rasterized LDD labels
│   └── label_47PQQ_buffered.tif       3-px eroded labels (used for training)
├── dem/
│   └── dem_47PQQ.tif                  Merged Copernicus DSM (after merge_dem.py)
├── S1_data/                           Raw Sentinel-1 .zip SAFE files
├── S1_processed/                      Calibrated/TC GeoTIFFs (after SNAP)
├── S2_data/                           Sentinel-2 multi-band composites (current)
├── indices/                           All index TIFs: DEM + SAR + S2 indices
├── aligned_features/
│   └── svm_add_data_features_labels.npz  Master feature matrix for training
│
└── 2018/                              ← ARCHIVED Phase 1 (S2-only). Do not modify.
    ├── S2_data/, indices/, aligned_features/
    └── stage1/, stage2/               Old models and reports
```

---

## Data

| Item | Detail |
|------|--------|
| Satellite | Sentinel-1 GRD IW + Sentinel-2 L2A + Copernicus DSM COG 10m |
| Tile | 47PQQ (Rayong province, Eastern Thailand) |
| CRS | EPSG:32647 (WGS 84 / UTM zone 47N) |
| Resolution | 10 m (all features resampled to match label raster) |
| S1 AOI | `POLYGON((101.0 12.5, 102.0 12.5, 102.0 13.5, 101.0 13.5, 101.0 12.5))` |
| Label source | LDD shapefile `LU_RYG_2561.shp`, column `LU_ID_L3` |
| Nodata values | `0` (background), `32767` (no-data sentinel) |
| Training NPZ | `aligned_features/svm_add_data_features_labels.npz` |

---

## Feature Set

All features are stored as individual GeoTIFFs in `./indices/`, then stacked by `align_indices_labels.py` into the `.npz`.

### DEM terrain features (5)
| Feature | Description |
|---------|-------------|
| ELEVATION_dem.tif | Raw DEM elevation (m), reprojected to 10 m |
| SLOPE_dem.tif | Slope in degrees (central-difference gradient) |
| ASPECT_dem.tif | Aspect in degrees, compass bearing (0=North) |
| TPI_dem.tif | Topographic Position Index (ridge = +, valley = −) |
| ROUGHNESS_dem.tif | Max−min elevation in 3-px window |

### Sentinel-1 SAR features (6 per acquisition date)
| Feature | Description |
|---------|-------------|
| VV | VV backscatter (dB, clipped to −30…+5) |
| VH | VH backscatter (dB) |
| VV_VH_RATIO | VV − VH (dB difference) |
| RVI | Radar Vegetation Index = 4VH/(VV+VH), linear domain [0–1] |
| RFDI | Radar Forest Degradation Index = (VV−VH)/(VV+VH) [−1–1] |
| DPR | Dual Polarization Ratio = VH/VV |

> VVVH_DIFF (an exact duplicate of VV_VH_RATIO) was removed 2026-07-02, and its 18 TIFs were deleted from `indices/`. The current `.npz` and `runs/s1_dem` models were trained *with* those columns — rebuild the `.npz` and retrain from Stage 1 before any new inference from TIFs.

### Sentinel-2 spectral indices (8 per acquisition date)
| Feature | Description |
|---------|-------------|
| NDVI | Normalized Difference Vegetation Index |
| EVI | Enhanced Vegetation Index |
| NDWI | Normalized Difference Water Index |
| BSI | Bare Soil Index |
| NDBI | Normalized Difference Built-up Index |
| MSAVI | Modified Soil-Adjusted Vegetation Index |
| SWIR_NIR | SWIR1/NIR ratio |
| SWIR_RATIO | (SWIR1−SWIR2)/(SWIR1+SWIR2) |

> **Feature order matters.** `align_indices_labels.py` uses `glob("./indices/*.tif")` (alphabetical). The exact order must be identical between training and inference. If you add or remove features, re-run `align_indices_labels.py` and retrain from Stage 1.

---

## LU Code Mappings

### Economic Crops (Stage 3 targets)

| LU_CODE | Thai | English | Stage 2 Subclass |
|---------|------|---------|-----------------|
| 2101 | ข้าว | Rice | field |
| 2204 | มันสำปะหลัง | Cassava | field |
| 2205 | สับปะรด | Pineapple | field |
| 2302 | ยางพารา | Rubber | plantation |
| 2303 | ปาล์มน้ำมัน | Oil palm | plantation |
| 2403 | ทุเรียน | Durian | orchards |
| 2404 | เงาะ | Rambutan | orchards |
| 2405 | มะพร้าว | Coconut | plantation |
| 2407 | มะม่วง | Mango | orchards |
| 2413 | ลำไย | Longan | orchards |
| 2416 | ขนุน | Jackfruit | orchards |
| 2419 | มังคุด | Mangosteen | orchards |
| 2420 | ลางสาด/ลองกอง | Langsat | orchards |

### Stage 1 Super-class Mapping (4 classes)

| Code | Super-class | LU codes included |
|------|-------------|-------------------|
| 1 | economic_crops | all above |
| 2 | water | 4101–4103, 4201–4203 |
| 3 | others | all remaining valid codes |
| 4 | forest | 3100, 3101, 3200, 3201, 3300, 3301, 3401, 3501 |
| 0 | background/nodata | excluded from training |

### Stage 2 Subclass Mapping

| Subclass label | Name | LU codes |
|---------------|------|---------|
| 1 | orchards | 2403, 2404, 2407, 2413, 2416, 2419, 2420 |
| 2 | plantation | 2302, 2303, 2405 |
| 3 | field | 2101, 2204, 2205 |
| 4 | other_econ | remaining economic codes |

---

## Pipeline Stages (Current)

```
1. Download
   tile_download.py           → S2_data/*.tif
   S1_download.py             → S1_data/*.zip

2. Preprocess SAR
   S1_preprocess_snap.py      → S1_processed/*.tif  (requires SNAP/gpt on PATH)

3. Compute features
   merge_dem.py               → dem/dem_47PQQ.tif
   compute_dem_feature.py     → indices/ELEVATION_dem.tif, SLOPE_dem.tif, ...
   s1_compute_features.py     → indices/VV_<stem>.tif, VH_<stem>.tif, ...
   compute_indices.py         → indices/NDVI_<date>.tif, EVI_<date>.tif, ...
   compute_extra_indices.py   → indices/MSAVI_<date>.tif, SWIR_NIR_<date>.tif, ...

4. Prepare labels
   rasterize_parcel.py        → label/label_47PQQ.tif
   buffer_labels.py           → label/label_47PQQ_buffered.tif

5. Align and stack
   align_indices_labels.py    → aligned_features/svm_add_data_features_labels.npz

6. Train
   stage1_weight_scale.py     → stage1_*.joblib + stage1_*.npy (full predictions)
   stage2_weighted.py         → stage2_*.joblib + stage2_*.npy
   stage3_new_weight.py       → stage3_{group}_*.joblib per subclass

7. Analyze
   features_important_check.py → stage1_features_importance_balanced.json
   evaluate_against_buffered_label.py → stage1_evaluation.csv
```

---

## Key Conventions

- **Working directory**: all paths in scripts are relative to `SVM_attempts/` root.
- **Training NPZ**: current pipeline uses `svm_add_data_features_labels.npz` (not the old `svm_features_labels.npz` in `2018/`).
- **Feature index order**: determined alphabetically by `glob("./indices/*.tif")` in `align_indices_labels.py`. Never reorder files after training; retrain if you do.
- **S2 composite band order**: `tile_download.py` stacks bands in `sorted()` string order, which puts B11/B12 **before** B8A (`"B11" < "B8A"`). On-disk order: B02,B03,B04,B05,B06,B07,B08,B11,B12,B8A. The `BAND_MAPPING` in `compute_indices.py`/`compute_extra_indices.py` matches this — do not "correct" it to the natural band order.
- **Chunked processing**: all stages use `PRED_CHUNK = 2_000_000` rows per chunk to avoid OOM.
- **Calibration**: Stage 1 and 2 models use `CalibratedClassifierCV` (sigmoid) to produce probabilities for threshold tuning.
- **Threshold tuning**: Stage 1 searches a threshold grid on a held-out validation set per class (econ=1, water=2, forest=4). Thresholds saved as `.json` next to the model.
- **SNAP requirement**: `S1_preprocess_snap.py` requires ESA SNAP 9.x with `gpt` on PATH.
- **Thai text**: LU code names include Thai characters. All CSV outputs use `encoding="utf-8-sig"` (BOM, for Excel compatibility).
- **`2018/` is archived**: do not edit or delete files there. It is the Phase 1 S2-only reference.

---

## Coding Guidelines

Behavioral principles to reduce common LLM coding mistakes. These bias toward caution over speed — use judgment for trivial tasks.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

*In this project*: always confirm whether the task targets the current root-level pipeline or the archived `2018/` scripts. Confirm which stage (1/2/3) and which model variant before touching training scripts.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use scripts.
- No error handling for impossible scenarios (e.g., don't add CRS guards when all data is EPSG:32647).
- If you write 200 lines and it could be 50, rewrite it.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing flat-script style: top-level `Config` block, `log()` or `print()`, relative paths.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add a new SAR feature" → "Feature TIF produced, shape matches existing, values in expected range, `.npz` rebuilt, Stage 1 retrained, F1 compared vs baseline"
- "Fix OOM" → "Full tile prediction completes without crash; output `.npy` has expected length"
- "Improve Stage 1 forest recall" → "Forest F1 improves vs baseline in `stage1_report_svm_weight_scale_increased.csv`"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

### 5. Session & Token Hygiene

**Cost comes from context size per turn, not from idle time.** Leaving a session open costs nothing; every message re-sends the whole conversation, so long sessions get expensive per turn.

- **Keep sessions task-scoped.** Finished a task or switching to something unrelated → `/clear` and start fresh. This project has a populated auto-memory (`MEMORY.md` + memory files), so a fresh session rehydrates the important state (run status, past diagnoses, feedback) automatically.
- **Use `/compact` only to continue the *same* long task** (e.g. babysitting a multi-hour training run) with less overhead. Don't `/compact` *and* start a new session — the new session discards the compacted history anyway.
- **Lean on memory, not long chats.** Persist durable project state (experiment results, diagnoses, queued next steps) to the memory files so `/clear` is cheap and safe between training runs.
