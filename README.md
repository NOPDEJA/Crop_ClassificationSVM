# LDD Crop Classification — SVM Pipeline

Pixel-level agricultural land-use classification for Rayong province, Thailand, built in
collaboration with the Land Development Department (LDD). The pipeline fuses **Copernicus
DEM terrain features**, **Sentinel-1 SAR backscatter**, and **Sentinel-2 spectral indices**
through a 3-stage cascade of calibrated SVM classifiers to map 13 economic crop types plus
water, forest, and background classes.

## Headline result

Adding Sentinel-2 spectral indices to the DEM+S1 feature set improves the classifier at
**every stage, for every class, with no exceptions** — once the RBF kernel's `gamma`
hyperparameter is scaled correctly for the feature dimensionality:

| Metric | DEM+S1 only | DEM+S1+S2 | Δ |
|---|---|---|---|
| Stage 1 accuracy (superclass) | 0.733 | **0.766** | +0.033 |
| Stage 2 accuracy (subclass routing) | 0.773 | **0.858** | +0.085 |
| End-to-end economic-crop recall | 47.3% | **61.7%** | **+14.4 pp** |

Full comparison tables and methodology: [`runs/dem_s1_s2_v2/REPORT.md`](runs/dem_s1_s2_v2/REPORT.md).

## Data

| Source | Detail |
|---|---|
| Sentinel-1 GRD (IW) | VV/VH backscatter, terrain-corrected via SNAP |
| Sentinel-2 L2A | 10 spectral indices per acquisition date |
| Copernicus DSM (COG, 10 m) | Elevation, slope, aspect, TPI, roughness |
| Labels | LDD parcel shapefile (`LU_ID_L3` field), rasterized to 10 m and edge-eroded |
| Tile | 47PQQ, Rayong province — EPSG:32647 (UTM zone 47N) |

## Pipeline

```
Download (S1_download.py, tile_download.py)
        │
        ▼
Preprocess (S1_preprocess_snap.py — SNAP GPT graph)
        │
        ▼
Compute features (compute_dem_feature.py, s1_compute_features.py,
                   compute_indices.py, compute_extra_indices.py)
        │
        ▼
Prepare labels (rasterize_parcel.py, buffer_labels.py)
        │
        ▼
Align & stack (align_indices_labels.py) → aligned_features/*.npz
        │
        ▼
Train cascade:
  Stage 1 — 4-class superclass SVM   (stage1_weight_scale.py)
  Stage 2 — economic subclass router (stage2_weighted.py)
  Stage 3 — per-subclass fine-grained LU_CODE classifiers (stage3_new_weight.py)
        │
        ▼
Evaluate (evaluate_end_to_end.py, evaluate_against_buffered_label.py)
```

See [`CLAUDE.md`](CLAUDE.md) for the full stage-by-stage breakdown, feature tables, LU code
mappings, and coding conventions, and [`skill.md`](skill.md) for copy-pasteable run/verify
prompts per stage.

## Setup

```bash
conda create -n svm_env python=3.10
conda activate svm_env
conda install --file requirements_conda.txt -c conda-forge
pip install -r requirements_pip.txt
```

Sentinel-1 preprocessing additionally requires ESA SNAP 9.x with `gpt` on `PATH`.

## Repository layout

```
SVM_attempts/
├── CLAUDE.md, skill.md      Project conventions and run playbooks
├── *.py                     Download / preprocess / feature / training scripts (see CLAUDE.md)
├── aligned_features/        Master feature matrices (.npz)
├── label/, dem/, indices/   Intermediate raster products
├── runs/                    Per-experiment outputs, reports, and models
│   └── dem_s1_s2_v2/        Current best model — see REPORT.md
└── 2018/                    Archived Phase 1 (Sentinel-2-only), do not modify
```

## Status

The corrected-gamma DEM+S1+S2 chain (`runs/dem_s1_s2_v2/`) is the current best pipeline.
Two prior findings are documented for context in `runs/`:

- `runs/dem_s1_s2/REPORT.md` — the original run, which appeared to show S2 hurting accuracy.
  Diagnosed as an RBF kernel-dilution artifact from an off-scale `gamma` grid.
- `runs/exp_gamma_scale/` — the quick experiment that confirmed the diagnosis (Stage 2
  accuracy 0.466 → 0.868 from the gamma fix alone).

Open follow-ups: widen the Stage 1 `n_components` search past its current ceiling, and
retune per-class thresholds for Durian (F1 regressed slightly under the corrected model).
