# LDD Crop Classification — hierarchical SVM

Pixel-level agricultural land-use classification for Rayong province, Thailand, in
collaboration with the Land Development Department (LDD). A three-stage cascade of
calibrated SVM classifiers maps 13 economic crop types over Sentinel-2 tile 47PQQ.

Terms used throughout the repository are defined in [`CONTEXT.md`](CONTEXT.md) — read that
first. It is a glossary, not a spec.

---

## Current status (2026-08-24)

The active work is the **Sentinel-2-only arms** (`s2_2018_3date_v2`, `s2_2018_5date`). The
earlier DEM + Sentinel-1 + Sentinel-2 phase is complete and is kept for reference.

The most recent result, on a **parcel-disjoint** split — no parcel contributes pixels to
both training and test:

| | rows scored | macro F1 | weighted F1 |
|---|---|---|---|
| hard routing | 5,500,269 | **0.2248** (95 % CI 0.2071–0.2389) | 0.8018 |
| joint routing | 5,500,269 | 0.1949 | 0.7725 |

> **Read older numbers in this repository with care.** Every figure produced before
> 2026-08-23 was measured on a pixel-level split, and 86.1 % of parcels contributed at
> least one fitted pixel to it. Those numbers are not wrong arithmetic, but they describe
> performance on ground the model had partly seen. The measurement, and what it does and
> does not license, is in [`docs/REPORT_2026-08-25.md`](docs/REPORT_2026-08-25.md) §2.

---

## Repository map

All scripts are flat at the repository root and run from there (`python <script>.py`), so
relative paths resolve. The arm is chosen by the `ARM` environment variable, which
[`config.py`](config.py) resolves to a tag, a feature matrix and every output path.

### Shared modules
| file | role |
|---|---|
| `config.py` | the arm switch; sampling constants; every artifact path |
| `raster.py` | band reprojection and grid alignment |
| `parsing.py`, `zip_manager.py`, `credential.py` | Sentinel-2 metadata, archive reading, Copernicus credentials |
| `tile_selector.py` | MGRS tile extraction from a province shapefile |

### 1 — Acquire
`tile_download.py` (Sentinel-2) · `S1_download.py` (Sentinel-1 GRD) ·
`merge_dem.py` (Copernicus DSM tiles)

### 2 — Build features
`compute_indices.py` (NDVI, EVI, NDWI, BSI, NDBI) ·
`compute_extra_indices.py` (MSAVI, SWIR_NIR, SWIR_RATIO) ·
`compute_dem_feature.py` (elevation, slope, aspect, TPI, roughness) ·
`s1_compute_features.py` (VV, VH, ratios, RVI, RFDI, DPR) ·
`S1_preprocess_snap.py` (SNAP GPT graph; needs `gpt` on PATH)

### 3 — Labels and alignment
`rasterize_parcel.py` → `buffer_labels.py` (3-px erosion) →
`align_indices_labels.py` (stack to `.npz`) · `make_subset_npz.py` (cut an arm's columns) ·
`label_extractor.py`

### 4 — Splits
`build_parcel_split.py` — parcel-atomic train/val/test split and a contamination audit of
the pixel-level splits. Writes `splits/`.

### 5 — Train
| file | what it trains |
|---|---|
| `train_parcel_cascade.py` | **current** — the whole cascade under the parcel split, one balancing mechanism, frozen hyperparameters |
| `stage1_weight_scale.py`, `stage2_weighted.py`, `stage3_new_weight.py` | the published per-stage pipeline (pixel splits, hyperparameter searches) |
| `run_chain.sh` | runs a whole arm end to end behind a per-arm lock |
| `queue_*.sh` | serialise arms so two cascades never contend for memory |

### 6 — Evaluate
`evaluate_end_to_end.py` (compose the cascade, score per crop) ·
`evaluate_flat_15class.py` (flat taxonomy for non-cascade comparison) ·
`reconstruct_sampled_rows.py` (recover which rows were fitted) ·
`audit_parcel_disjoint.py` (score on parcel-disjoint rows) ·
`rescore_collaborator_protocol.py` (score under the XGBoost study's protocol) ·
`confusion_report.py` · `compare_stage1_arms.py` · `evaluate_against_buffered_label.py`

### 7 — Diagnose
`diagnose_error_budget.py` (where end-to-end loss goes) ·
`probe_stage1_prior.py`, `probe_dry_season.py`, `probe_pca_bottleneck.py`,
`probe_kernel_capacity.py` (each isolates one configuration choice) ·
`features_important_check.py`

### 8 — Class priors
`save_stage23_probs.py` → `save_stage3_probs_all_econ.py` → `sweep_prior_alpha.py` →
`apply_operating_point.py`. `apply_prior_correction.py` and `validate_alpha_split.py`
support the same question. Nothing here refits a model; all of it reweights saved
probabilities.

### 9 — Cross-year transfer
`prepare_epoch.py <year>` → `predict_new_epoch.py <year> <arm>` →
`score_by_churn.py <year> <arm>` → `compare_epochs.py`. The last is the one to cite: the
others score each epoch on its own survey's pixels, and that comparison is not valid.

### Outputs
`runs/<arm>/` — one directory per arm holding every artifact of every stage.
`splits/` — the shared parcel split. `aligned_features/` — feature matrices.
`indices/`, `label/`, `dem/`, `S1_processed/` — inputs.

---

## Data

| Source | Detail |
|---|---|
| Sentinel-2 L2A | 8 spectral indices per acquisition date; Oct/Nov/Dec 2018 (3-date arm), plus Mar/Apr (5-date) |
| Sentinel-1 GRD (IW) | VV/VH backscatter, terrain-corrected via SNAP — earlier phase |
| Copernicus DSM (10 m) | elevation, slope, aspect, TPI, roughness — earlier phase |
| Labels | LDD parcel survey `LU_RYG_2561` (2018), field `LU_ID_L3`, rasterised to 10 m and edge-eroded by 3 px |
| Tile | 47PQQ, Rayong — EPSG:32647, 10,980 × 10,980 px, 24,323,769 labelled pixels |

Class imbalance is severe and drives much of the behaviour: rubber is 81 % of economic
pixels, Langsat 0.012 % — about 6,685:1.

---

## Reproducing

```bash
python make_subset_npz.py s2_3date          # 24-column feature matrix
python build_parcel_split.py                # parcel-atomic split + contamination audit
python train_parcel_cascade.py              # cascade under that split, both routing rules
```

The published per-stage pipeline, for the pixel-split arms:

```bash
ARM=s2_2018_3date_v2 USE_PCA=0 RUN_STAGE1=1 bash run_chain.sh
```

`SMOKE=1` runs `train_parcel_cascade.py` on a subsample for a fast end-to-end check.

---

## Documentation

| file | contents |
|---|---|
| [`CONTEXT.md`](CONTEXT.md) | glossary — what each term means |
| [`docs/REPORT_2026-08-25.md`](docs/REPORT_2026-08-25.md) | most recent progress report |
| [`docs/S2_SVM_ANALYSIS.md`](docs/S2_SVM_ANALYSIS.md) | full study write-up; §6 is the evidence base, §7 the comparison notes |
| [`docs/DECISIONS_AND_ANALYSIS.md`](docs/DECISIONS_AND_ANALYSIS.md) | decision record and post-mortems |
| [`docs/MEETING_BRIEF_2026-07-10.md`](docs/MEETING_BRIEF_2026-07-10.md) | earlier meeting brief |
| [`CLAUDE.md`](CLAUDE.md) | conventions and working agreements |

`2018/` is the archived Phase 1 (Sentinel-2 only, superseded). Do not modify it.
