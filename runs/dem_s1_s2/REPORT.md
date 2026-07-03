# DEM + S1 + S2 Combined Run — Full Chain Results

**Run date:** 2026-07-03
**Feature matrix:** `svm_dem_s1_s2_features_labels.npz` — 24,323,769 pixels × 153 features
(5 DEM + 108 S1 + 40 corrected S2 indices), 19 all-NaN columns dropped → 134 used
**Config:** identical to the DEM+S1 baseline (`runs/s1_dem`) — same seeds, caps, splits,
search grids, PCA(10) in Stage 1 only. Only the feature set differs.

## Headline Result

**Adding 40 correctly-computed S2 spectral indices made the cascade WORSE at every level.**

| Metric | DEM+S1 | DEM+S1+S2 | Δ |
|---|---|---|---|
| Stage 1 accuracy | 0.667 | 0.648 | −0.019 |
| Stage 1 forest F1 | 0.816 | 0.751 | −0.065 |
| Stage 2 accuracy | 0.509 | 0.466 | −0.043 |
| Stage 2 orchards recall | 0.296 | 0.154 | −0.142 |
| Stage 2 plantation recall | 0.334 | 0.257 | −0.077 |
| Stage 3 Rice F1 | 0.791 | 0.379 | −0.412 |
| Stage 3 Oil palm F1 | 0.698 | 0.102 | −0.596 |
| Stage 3 Durian F1 | 0.848 | 0.905 | +0.057 |
| Stage 3 Rubber F1 | 0.784 | 0.868 | +0.084 |
| **End-to-end econ recall** | **23.1%** | **19.0%** | **−4.1 pp** |

End-to-end loss accounting (15.16M true econ pixels):

| Where lost | DEM+S1 | DEM+S1+S2 |
|---|---|---|
| Dropped at Stage 1 (not predicted econ) | 4.68M (30.9%) | 4.44M (29.3%) |
| Misrouted at Stage 2 (wrong subclass) | 6.32M (41.7%) | 7.27M (47.9%) |

## Why the data is NOT the problem

- Band order of the S2 composites was **verified empirically**: band 10 correlates 0.90
  with B08 at 1.08× brightness (= B8A), bands 8/9 show SWIR signatures. The corrected
  `BAND_MAPPING` is right; the 40 indices are validly computed.
- All indices are finite (EVI clipped ±3, SWIR_NIR clipped [0,20], NaN only at cloud
  gaps: 0.2–19.5% per date).
- The same optical source data, alone, achieved orchards F1 = 0.714 at Stage 2 in the
  Phase 1 S2-only run — the discriminative signal exists.

## Diagnosis: RBF kernel distance dilution + off-scale gamma

An RBF kernel weighs all features equally in its distance metric. This cascade feeds it
94 highly redundant SAR/DEM columns (18 near-duplicate acquisition dates × 6 features)
plus 40 informative optical columns — squared distances are dominated by the SAR block,
so adding optical columns *diluted* the class structure instead of enriching it.

Compounding this, the gamma search grid {0.5, 1.0} is roughly two orders of magnitude
above the conventional scale for standardized data (~1/n_features ≈ 0.007 at 134 dims),
and the mismatch worsens as dimensionality grows.

Consistent evidence:
1. Stage 1 (PCA→10 dims) degraded least; Stages 2–3 (no PCA, full 134 dims) degraded most.
2. Stage 3 crops that *improved* (Durian, Rubber) did so on smaller, more survivor-biased
   routed subsets — pure selection effect. Field crops kept near-identical routing
   (Stage 2 field recall 0.94→0.99) yet Rice F1 halved: same pixels, worse geometry.
3. Nystroem n_components hit its 150 search ceiling in every stage of both runs.

## Next experiments (cheapest, most decisive first)

1. **Gamma-scale test** — Stage 2 with gamma ∈ {1/n_features, 0.01, 0.05}. ~1 h.
2. **S2-only Stage 2 ablation** — same routing, 40 optical columns only. If orchards
   recovers toward 0.7, dilution is confirmed as the mechanism.
3. **S1 temporal aggregation** — 18 dates → per-orbit/monthly medians (108 → ~12–24
   columns) before recombining with S2. Also eliminates the all-NaN column problem.
4. Consider tree ensembles (RF/GBM) as a feature-robust reference — they perform
   per-feature selection natively and are insensitive to this failure mode.

## Artifacts

- `end_to_end_lu_pred.tif` — final crop map (LU codes; 0 = not classified)
- `stage1_dem_s1_s2_pred.tif` — Stage 1 superclass map
- Per-stage reports, confusion matrices, meta JSONs alongside this file.
