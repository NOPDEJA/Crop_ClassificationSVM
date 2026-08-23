# LDD Crop Classification — Decisions, Technical Terms, Code & Analysis

This document explains **why** the project is built the way it is: the reasoning behind every
major design decision, the technical vocabulary used in the reports, the key code patterns,
and the full analysis behind the project's most important finding (the gamma-scale /
kernel-dilution story). It complements:

- `README.md` — quick overview and headline result
- `CLAUDE.md` — repository structure, conventions, LU-code mappings
- `runs/*/REPORT.md` — raw per-run results and observations

Written 2026-07-09, after completion of the corrected-gamma v2 chains.

---

## 1. The Problem and the Approach

**Goal:** produce a pixel-level (10 m) map of 13 economic crop types (rice, cassava,
rubber, durian, …) plus water and forest for Rayong province, Thailand, from free satellite
data, validated against LDD's parcel survey (`LU_RYG_2561.shp`).

**Approach:** a 3-stage cascade of SVM classifiers over a fused feature stack of
Copernicus DEM terrain features, Sentinel-1 SAR backscatter, and Sentinel-2 spectral
indices.

```
24.3M pixels × 134 features
        │
Stage 1: econ / water / others / forest        (4 superclasses)
        │  keep only pixels predicted "econ"
Stage 2: orchards / plantation / field         (subclass router)
        │  route each pixel to its subclass model
Stage 3: fine-grained LU_CODE per subclass     (e.g. durian vs mango vs rambutan)
```

### Why a cascade instead of one flat 16-class classifier?

1. **Class imbalance is extreme.** Plantation crops alone are 12.8M pixels while some
   orchard species are a few thousand. A flat classifier collapses onto the majority
   classes. The cascade lets each stage rebalance (cap + class weights) at its own level.
2. **Different stages need different signals.** Water/forest vs cropland is easy with SAR
   and terrain; separating durian from mangosteen needs optical phenology. Splitting the
   problem lets each model specialize.
3. **Errors become interpretable.** The end-to-end evaluation can attribute every lost
   pixel to a specific stage ("dropped at Stage 1" vs "misrouted at Stage 2"), which
   directly told us where to focus (see §6.4).

The cost of the cascade: Stage 2/3 train only on pixels the previous stage passed through,
which biases their training data (see Known Limitations, §7).

### Why fuse three data sources?

| Source | What it contributes | What it cannot do |
|---|---|---|
| **DEM terrain** (5 features) | Landform context — paddy is flat, durian orchards sit on gentle slopes, forest on hills | No crop-type information on similar terrain |
| **Sentinel-1 SAR** (6 features × 18 dates) | All-weather structure: canopy volume scattering, rice-flooding signature, smooth water. Immune to the cloud cover that plagues optical data in monsoon Thailand | Weak at separating crop *species* — orchard canopies look alike to C-band radar |
| **Sentinel-2 indices** (8 features × 5 dates) | Phenology and canopy chemistry (NDVI seasonality, SWIR moisture) — the primary crop-species discriminator | Blocked by clouds (0.2–19.5 % gaps per date here) |

The experiments confirmed this division of labor empirically: DEM+S1 alone got forest
F1 = 0.88 but only 0.365 macro-F1 on crop subclasses; adding S2 lifted subclass routing
accuracy from 0.773 to 0.858 and end-to-end econ recall from 47.3 % to 61.7 %.

---

## 2. Experiment Timeline (Progress Narrative)

| # | Run | Date | Feature set | Outcome |
|---|-----|------|-------------|---------|
| 0 | `2018/` (Phase 1) | 2025 | S2 indices only, 3 superclasses | Stage 1 acc 0.712; Stage 2 acc 0.751. Later found to have a SWIR **band-order bug** (§6.2) — baseline understates a correct S2 model |
| 1 | `runs/s1_dem` | 2026-06-12 → 07-01 | DEM + S1 (no S2) | Forest excellent (F1 0.816), water precision 0.922, but crop subclassing collapsed (Stage 2 acc 0.509). End-to-end econ recall **23.1 %** |
| 2 | `runs/dem_s1_s2` | 2026-07-03 | DEM + S1 + S2 (same config as #1) | **Worse at every stage** (end-to-end 19.0 %). Diagnosed as RBF kernel dilution + gamma ~100× off-scale — *not* a property of the S2 data |
| 3 | `runs/exp_gamma_scale` | 2026-07-03 | Stage 2 only, corrected gamma | Stage 2 accuracy **0.466 → 0.868** from the gamma fix alone. Diagnosis confirmed |
| 4 | `runs/dem_s1_s2_v2` | 2026-07-04/05 | DEM + S1 + S2, corrected gamma | Best model. Stage 1 acc 0.766, Stage 2 acc 0.858, end-to-end **61.7 %** |
| 5 | `runs/s1_dem_v2` | 2026-07-05 | DEM + S1, corrected gamma | Fair "no-S2" comparison arm: end-to-end 47.3 %. S2 wins **every metric at every stage** |

Bottom line so far: the corrected-gamma DEM+S1+S2 cascade (`runs/dem_s1_s2_v2`) is the
production candidate, at ~2.7× the end-to-end recall of the first working baseline.

---

## 3. Decisions and Their Reasons

### 3.1 Modeling decisions

**LinearSVC + Nystroem instead of a true kernel SVC.**
A full RBF `SVC` is O(n²)–O(n³) in training samples; at 2M training rows it is simply
infeasible. The Nystroem transform approximates the RBF kernel's feature map with
`n_components` landmark points, after which a *linear* SVM (which trains in near-linear
time) operates in that approximated space. This is the standard large-scale kernel-SVM
recipe. Trade-off: `n_components` becomes a capacity hyperparameter that must be searched
(and it kept hitting our search ceiling — see follow-ups).

**PCA(10) in Stage 1 only.**
Stage 1 trains on the most rows (2.03M), so we compress 134 features → 10 principal
components first for speed. Stages 2/3 train on fewer rows and skip PCA to keep full
feature detail. This asymmetry mattered later: in the broken-gamma run, PCA-protected
Stage 1 degraded least, which became a diagnostic clue (§6.1).

**`class_weight='balanced'` + per-class caps.**
Raw class frequencies span 3 orders of magnitude. Caps (e.g. econ ≤ 1M, water ≤ 500K in
Stage 1; 200K per subclass in Stage 2) bound training time and stop majority flooding;
`balanced` weights compensate for the remaining imbalance inside the loss function. An
earlier idea of an extra ×2.0 "econ boost" turned out to be dead code that never reached
the model — the report was corrected accordingly (honesty over narrative).

**CalibratedClassifierCV (sigmoid) + threshold tuning.**
`LinearSVC` outputs raw decision margins, not probabilities. Sigmoid (Platt) calibration
maps margins → probabilities on held-out folds, which enables *per-class decision
thresholds* tuned on a validation set instead of a fixed argmax. This lets us trade
precision against recall per class (LDD cares more about finding all crop area than about
a few false positives). In the v1 run every threshold hit the search floor (0.4), telling
us the floor was set too high — a small but real finding.

**RandomizedSearchCV with deliberately small budgets.**
`N_ITER_SEARCH=4–6`, 3-fold CV, `NYST_COMPONENTS ≤ 150–250`, `max_iter=5000`. The original
"thorough" grid was estimated at ~96 hours per stage on this machine; the reduced budget
finishes a stage in ~6.5 h with little measured accuracy loss. Rationale: on 32 GB RAM
with 24M-pixel inference afterwards, wall-clock time per experiment *is* the research
bottleneck — a 10× slower search that prevents running the v2 comparison at all is a net
scientific loss.

### 3.2 Data decisions

**3-pixel label erosion (`buffer_labels.py`).**
Parcel boundaries in the LDD shapefile are approximate, and boundary pixels are mixtures
of two land uses at 10 m resolution. Eroding each parcel 3 px inward removes the noisiest
labels from training. Trade-off: fewer training pixels, and evaluation on eroded labels is
slightly optimistic about boundary behavior.

**Alphabetical feature order as a hard convention.**
`align_indices_labels.py` stacks whatever `glob("./indices/*.tif")` returns, in
alphabetical order. The trained model knows features only by column index, so *any*
change to the files in `indices/` silently changes what each column means. Rule: adding
or removing any TIF requires rebuilding the `.npz` and retraining from Stage 1. This is
why the removal of the duplicate feature VVVH_DIFF (2026-07-02) came with an explicit
warning not to run new TIF-based inference against the old models.

**The "wrong-looking" S2 band mapping is correct.**
`tile_download.py` stacks bands in Python string-sort order, and `"B11" < "B8A"` as
strings — so on disk the order is B02,B03,B04,B05,B06,B07,B08,**B11,B12,B8A**, not the
natural wavelength order. The `BAND_MAPPING` in the index scripts matches the *disk*
order. It looks like a bug; "fixing" it to natural order would reintroduce the Phase 1
band-swap bug (§6.2).

**Nodata conventions: 0 = background, 32767 = sentinel, NaN in index TIFs.**
Index scripts write NaN at cloud gaps; the training scripts must therefore handle NaN
explicitly (imputer + valid-column mask, §4.2) rather than assuming clean data.

### 3.3 Process decisions

**Retrain *both* arms before drawing feature-set conclusions.**
The single most expensive lesson of the project (~14 h of extra training): after fixing
gamma, the corrected combined run could not fairly be compared to the *old* DEM+S1
baseline, because that baseline also used the broken grid. Both `dem_s1_s2_v2` and
`s1_dem_v2` were retrained with the identical corrected config. Only that apples-to-apples
pair supports the claim "S2 helps." Any future ablation must follow the same rule.

**Serial training runs only.**
32 GB RAM. One stage's search + 24M-row chunked inference approaches the ceiling on its
own; two concurrent runs risk OOM and invalidate wall-clock comparisons.

**Detached, unbuffered launches for multi-hour jobs.**
Long runs are launched as native detached processes with `python -u`
(see §5.5) because (a) harness-tracked background shells were observed being killed by
unrelated events on this machine, and (b) without `-u`, Python buffers stdout — a crash or
reboot leaves an *empty* log with no clue how far the run got. A machine reboot on
2026-07-03 destroyed 6.5 h of un-flushed progress and motivated this rule.

**Reports written per-run, with corrections kept visible.**
Every run directory gets a `REPORT.md` with results *and* dated caveat blocks when later
audits found problems (band swap, Longan bug, dead-code weight boost). Errors are
annotated, not silently rewritten — the correction history is part of the evidence trail.

---

## 4. Technical Glossary

### 4.1 Remote sensing terms

| Term | Meaning |
|---|---|
| **Sentinel-1 GRD (IW)** | ESA C-band radar satellite; Ground Range Detected product from Interferometric Wide swath mode. Radar sees through clouds, day or night |
| **SAR backscatter** | Strength of the radar echo returned to the sensor. Depends on surface roughness, geometry, and moisture. Smooth water reflects the pulse *away* (very low backscatter → water is easy to spot); dense canopies scatter within the volume (high VH) |
| **VV / VH polarization** | Transmit Vertical, receive Vertical / receive Horizontal. VH (cross-pol) responds strongly to volume scattering in vegetation canopies; VV more to surface roughness |
| **dB (decibel)** | Backscatter is log-scaled: `10·log10(linear)`. We clip VV to −30…+5 dB. Ratio-type indices (RVI, DPR) are computed in the *linear* domain first |
| **Speckle filtering** | SAR images have salt-and-pepper interference noise; a Lee/Refined-Lee filter smooths it during preprocessing |
| **Terrain correction** | Radar images are geometrically distorted by topography (foreshortening, layover); SNAP's Range-Doppler terrain correction projects pixels onto the DEM into map coordinates |
| **RVI** | Radar Vegetation Index, `4·VH/(VV+VH)` in linear units, ∈[0,1]. Higher = more vegetation volume |
| **RFDI** | Radar Forest Degradation Index, `(VV−VH)/(VV+VH)`. Separates forest (low) from degraded/open land (high) |
| **DPR** | Dual Polarization Ratio, `VH/VV` |
| **Copernicus DSM** | 10 m Digital Surface Model (elevation including canopy) used both for SAR terrain correction and as direct features |
| **TPI** | Topographic Position Index: pixel elevation minus mean of its neighborhood. Positive = ridge, negative = valley |
| **Roughness** | max−min elevation within a 3×3 window — local terrain ruggedness |
| **Sentinel-2 L2A / BOA** | ESA optical satellite, Level 2A = atmospherically corrected Bottom-Of-Atmosphere reflectance |
| **NDVI / EVI / MSAVI** | Vegetation indices from red/NIR (and blue for EVI). Track greenness/biomass; their *seasonal trajectory* (phenology) distinguishes crop species |
| **NDWI** | Water index (green vs NIR) — open water and canopy moisture |
| **BSI / NDBI** | Bare-soil and built-up indices, using SWIR bands |
| **SWIR_NIR / SWIR_RATIO** | Shortwave-infrared ratios sensitive to leaf water content and soil — key discriminators for tree crops |
| **MGRS tile 47PQQ** | The 110×110 km Sentinel-2 grid cell covering Rayong. All rasters are aligned to it in EPSG:32647 (UTM 47N) at 10 m |
| **LU_CODE / LU_ID_L3** | LDD's hierarchical land-use codes, e.g. 2403 = durian; level-3 field in the parcel shapefile |

### 4.2 Machine-learning terms

| Term | Meaning |
|---|---|
| **SVM** | Support Vector Machine — finds the maximum-margin boundary between classes |
| **RBF kernel** | `K(x,y) = exp(−γ·‖x−y‖²)`: similarity decays with squared Euclidean distance, letting the SVM learn non-linear boundaries |
| **gamma (γ)** | *The* critical scale parameter of this project. Sets how fast similarity decays with distance. For standardized d-dimensional data, `‖x−y‖² ≈ 2d` for typical pairs, so the conventional scale is `γ ≈ 1/d`. With 134 features that is ≈ 0.007. Our v1 grid {0.5, 1.0, 2.0} was **~100× too large**: at γ=0.5, typical pairwise similarities are `exp(−134)` ≈ 0 — every point looks maximally dissimilar to every other, the kernel matrix approaches the identity, and the model can only memorize, not generalize. See §6.1 |
| **C** | SVM regularization: higher C = fit training data harder, less regularization |
| **Nystroem approximation** | Approximates the kernel feature map using `n_components` sampled landmark points so a linear model can be used at scale (§3.1) |
| **`gamma=None`** | In scikit-learn's Nystroem, `None` means `1/n_features` — exactly the conventional scale. This was the winning value in Stage 2 |
| **PCA** | Principal Component Analysis — orthogonal projection onto directions of largest variance; used to compress 134→10 dims in Stage 1 |
| **StandardScaler** | Per-feature standardization to mean 0, variance 1 — required so Euclidean distances (and hence the RBF kernel) are not dominated by large-unit features |
| **SimpleImputer** | Fills NaN with the column mean/median. Needed because cloud gaps and partial SAR coverage leave NaN, which Nystroem rejects |
| **valid_cols mask** | Column indices that are not entirely NaN. Saved by Stage 1 (`stage1_*_valid_cols.npy`) and re-applied everywhere downstream so feature dimensions always match the trained model |
| **OneVsRest** | Trains one binary classifier per class; needed because LinearSVC is binary at heart and we want per-class probabilities |
| **CalibratedClassifierCV (sigmoid)** | Platt scaling: fits a logistic function mapping SVM margins to probabilities on held-out folds |
| **Decision threshold** | Instead of argmax, class c is asserted when `P(c) ≥ t_c`; the `t_c` are tuned per class on the validation set to maximize F1 |
| **RandomizedSearchCV** | Samples `n_iter` random hyperparameter combinations, evaluates each with k-fold cross-validation, refits the best on all training data. The final refit produces a long silent phase in the logs (>100 min observed) — expected, not a hang |
| **Precision / Recall / F1** | Of predicted-c pixels, how many are truly c / of truly-c pixels, how many were found / their harmonic mean |
| **Macro vs weighted F1** | Macro averages F1 equally across classes (fair when classes are imbalanced); weighted averages by class size (can hide minority-class collapse). §6.3 shows why macro is the right lens for orchards |
| **Confusion matrix** | Rows = true class, columns = predicted; the off-diagonal cells tell you *which* classes are being confused |
| **Majority collapse** | Degenerate solution where a classifier predicts the dominant class for nearly everything — high accuracy, useless map |
| **Survivor bias (in the cascade)** | Stage 3 only sees pixels Stage 2 routed correctly; when Stage 2 recall is low, those survivors are the easiest pixels, inflating Stage 3's apparent scores |
| **End-to-end recall** | Fraction of *all* true econ pixels that receive the correct final LU code after the whole cascade — the metric LDD actually consumes, and the only one that penalizes routing errors between stages |

---

## 5. Code Walkthrough

The scripts are deliberately flat, single-purpose files (Config block at top, run to
completion). The load-bearing patterns:

### 5.1 The training pipeline (Stage 1, `stage1_weight_scale.py`)

```python
steps = [('imputer', SimpleImputer(strategy='mean')),
         ('scaler', StandardScaler())]
if USE_PCA:
    steps.append(('pca', PCA(n_components=PCA_NCOMP, random_state=RANDOM_STATE)))   # 10 comps
steps.append(('nyst', Nystroem(kernel='rbf', random_state=RANDOM_STATE)))
steps.append(('svc', LinearSVC(class_weight='balanced', max_iter=5000,
                               random_state=RANDOM_STATE)))
pipe = Pipeline(steps)
```

Each step exists for a specific failure it prevents: the imputer for NaN (cloud gaps /
partial SAR coverage), the scaler because RBF distance is meaningless on mixed units
(elevation in metres vs indices in [−1,1]), PCA for Stage 1 speed, Nystroem+LinearSVC as
the scalable kernel-SVM substitute. Wrapping in `OneVsRest` + `CalibratedClassifierCV`
happens outside this pipeline.

### 5.2 The corrected gamma grids (the project's key fix)

```python
# stage1_weight_scale.py — PCA(10) runs first, so None = 1/10 = 0.1
NYST_GAMMA_CANDIDATES = [None, 0.005, 0.02]

# stage2_weighted.py — no PCA, 134 raw dims, 1/d ≈ 0.007
NYST_GAMMA = [None, 0.01, 0.05]     # None = 1/n_features
```

The v1 grid was `[0.5, 1.0, 2.0]`. Note the grids differ per stage *because the
dimensionality entering Nystroem differs* (10 after PCA vs 134 raw) — gamma is not a
universal constant, it is a per-geometry scale.

### 5.3 The valid-columns contract

```python
# Stage 1: drop all-NaN columns once, persist the mask
nan_mask = np.all(np.isnan(Xs), axis=0)
valid_cols = np.where(~nan_mask)[0]
np.save(f"{OUT_DIR}/stage1_..._valid_cols.npy", valid_cols)

# Every later consumer (Stage 2/3 training, chunked inference, feature importance):
X = X[:, valid_cols]
```

Why: 19–22 of the raw columns are entirely NaN (SAR acquisitions with zero overlap of the
labeled area) — `SimpleImputer` cannot impute a column with no observed values, and a
model trained on 109/134 columns will crash or silently misread data fed with 131/153.
The saved `.npy` is the single source of truth for "which columns the model means."

### 5.4 Chunked inference

```python
PRED_CHUNK = 2_000_000          # rows per chunk
```

Full-tile prediction is 24.3M rows × 134 features of float32 — materializing intermediate
Nystroem features for all rows at once would exceed 32 GB. All stages predict in 2M-row
chunks and append to a preallocated output array.

### 5.5 The long-run launch pattern (operational, PowerShell)

```powershell
Start-Process -FilePath "C:\Conda_environment\envs\svm_env\python.exe" `
  -ArgumentList "-u","stage1_weight_scale.py" -WorkingDirectory "<repo>" `
  -RedirectStandardOutput "stage1_run.log" -RedirectStandardError "stage1_run.err" `
  -WindowStyle Hidden
```

`-u` = unbuffered stdout, so the log is written *as it happens* and survives crashes.
Detached `Start-Process` survives terminal/session death. Health monitoring checks that
the process's **CPU time is strictly increasing** between checks — "process exists" alone
cannot distinguish a working fit from a zombie.

### 5.6 End-to-end evaluation (`evaluate_end_to_end.py`)

Composes the actual production path: Stage 1 model → econ mask → Stage 2 model → subclass
routing → the routed subclass's Stage 3 model → final LU code, over all 24.3M pixels, then
scores against ground truth. Crucially it also produces the *loss accounting* (§6.4) that
attributes every missed pixel to the stage that lost it.

---

## 6. Deep-Dive Analyses

### 6.1 The kernel-dilution / gamma story (the project's central finding)

**Observation (2026-07-03):** adding 40 correctly-computed S2 index columns to the
DEM+S1 feature set made *every stage worse* — Stage 1 accuracy 0.667→0.648, Stage 2
0.509→0.466, Stage 3 Rice F1 0.791→0.379, end-to-end econ recall 23.1 %→19.0 %. This was
paradoxical: the same optical data alone had reached Stage 2 accuracy 0.751 in Phase 1.

**First, the data was proven innocent.** Band order was verified empirically (the band at
disk index 10 correlates 0.90 with B08 at 1.08× brightness — that is B8A's signature, so
the composites really are in sorted-string order and the corrected mapping is right); all
indices were finite and clipped; cloud-gap NaN rates were known (0.2–19.5 % per date).
Only after eliminating the data did the model become the suspect.

**Diagnosis — two compounding mechanisms:**

1. *Distance dilution.* An RBF kernel weighs all features equally inside `‖x−y‖²`. The
   stack contains 94 highly redundant SAR/DEM columns (18 near-duplicate acquisition
   dates × ~6 features that mostly repeat the same information) versus 40 informative
   optical columns. Squared distances are dominated by the redundant block, so appending
   the optical columns *diluted* class structure rather than enriching it. Kernel methods
   have no built-in feature selection — unlike tree ensembles, they cannot ignore useless
   columns on their own.
2. *Off-scale gamma.* The v1 grid {0.5, 1.0, 2.0} sits ~100× above the conventional
   `1/n_features ≈ 0.007` scale for standardized 134-dim data. At γ=0.5, typical
   inter-point similarities are `exp(−γ·2d) ≈ exp(−134)` — numerically zero. The kernel
   matrix approaches the identity, the Nystroem features carry almost no neighborhood
   information, and adding dimensions (more S2 columns) pushes distances even further out,
   making the problem *worse with more data* — exactly the observed signature.

**Evidence that fit the diagnosis before it was confirmed:**
- Stage 1 (protected by PCA→10 dims, where the same gamma is far less off-scale) degraded
  least; Stages 2–3 (raw 134 dims) degraded most.
- Stage 3 field crops kept nearly identical routing (Stage 2 field recall 0.94→0.99) yet
  Rice F1 halved — *same pixels, worse decision geometry*, ruling out routing effects.
- The apparent "improvements" (Durian 0.848→0.905, Rubber 0.784→0.868) occurred on
  smaller, survivor-biased routed subsets — a selection artifact, not real skill.
- Nystroem `n_components` pinned at its 150 search ceiling in every stage of both runs
  (a starved kernel approximation grabbing all capacity it is offered).

**Confirmation, twice:** (1) a one-hour Stage 2 rerun with γ ∈ {None, 0.01, 0.05} jumped
accuracy 0.466 → 0.868 (winner γ=None ≈ 1/d) — `runs/exp_gamma_scale`. (2) Full v2 chains
for *both* feature sets with identical corrected grids: the +S2 chain then won every
metric at every stage (end-to-end 61.7 % vs 47.3 %).

**Lesson:** a hyperparameter grid tuned (or copied) for one feature geometry silently
became ~100× off-scale in another. "Feature set X hurts" conclusions are only valid
under hyperparameters appropriate to X — and the comparison baseline must be retrained
under the same corrected configuration.

### 6.2 The Sentinel-2 band-order bug (Phase 1's hidden flaw)

`tile_download.py` builds composites by stacking band files in Python `sorted()` order.
As *strings*, `"B11" < "B8A"`, so the disk order ends …B08, B11, B12, B8A — while the
Phase 1 index scripts assumed natural order …B08, B8A, B11, B12. Result: every index
that reads SWIR bands (BSI, NDBI, SWIR_NIR, SWIR_RATIO) was computed with "SWIR1"
actually reading B12 and "SWIR2" reading B8A (a narrow-NIR band). NDVI/EVI/NDWI/MSAVI
(red/NIR/green/blue only) were unaffected.

Consequences: the Phase 1 "S2-only baseline" quoted in the early reports *understates*
what a correct S2 model can do; and it is a reminder that the most durable comparisons in
this project are the v2-vs-v2 pair, which both use the corrected mapping. The fix keeps
the mapping matched to the *actual disk order* — the counter-intuitive convention is
documented in `CLAUDE.md` precisely so nobody "fixes" it back.

### 6.3 The orchards accuracy paradox: why macro-F1 is the honest metric

In the v1 runs, the Stage 3 orchards model had *higher accuracy* than the corrected v2
model (0.829 vs 0.635) — yet the v2 model is clearly better. The v1 test set was ~82 %
Durian, and the v1 model predicted Durian almost regardless of input: minority-species
recall was 2–8 %. Accuracy rewards that collapse. The corrected model spreads predictions
across all 7 species (Mango F1 0.037→0.902, Longan 0.122→0.497), doubling macro-F1
(0.288→0.519) while losing raw accuracy. A map that calls everything Durian is useless to
LDD however "accurate" it is — so orchards comparisons in this project are made on
macro-F1, with the accuracy caveat stated. (In the final v2-vs-v2 comparison, S2 wins on
both metrics anyway, so the conclusion does not depend on the choice.)

### 6.4 End-to-end loss accounting: where the missing 38 % goes

Of 15.16M true economic-crop pixels (dem_s1_s2_v2):

| Fate | Pixels | Share |
|---|---|---|
| Correct final LU code | 9.36M | **61.7 %** |
| Dropped at Stage 1 (not predicted econ) | 2.86M | 18.9 % |
| Misrouted at Stage 2 (wrong subclass) | 1.97M | 13.0 % |
| Wrong LU within the right subclass (Stage 3) | ~0.97M | ~6.4 % |

Versus the no-S2 arm (47.3 % recall; 3.28M dropped, 3.09M misrouted): S2's gain comes
mostly from *routing* — Stage 2 misrouting drops by over a million pixels. This table is
the project's prioritization tool: the next percentage points live in Stage 1 recall and
Stage 2 routing, not in the Stage 3 models.

### 6.5 Smaller bugs the 2026-07-02 audit caught (and their morals)

| Bug | Impact | Moral |
|---|---|---|
| Longan (2413) missing from `orchards_codes` in Stage 2 | Longan trained as *other_econ*, so its Stage 3 F1 (0.167) was doubly biased | Keep class mappings in one place; cross-check scripts against the canonical table in CLAUDE.md |
| ECON_BOOST dead code in Stage 1 | Report claimed a ×2.0 weight boost the model never used | Verify a claim reaches the model object before reporting it |
| VVVH_DIFF ≡ VV_VH_RATIO duplicate | 18 columns of pure redundancy feeding kernel dilution | Redundant features are not harmless in kernel methods |
| Masked-array garbage at always-cloudy pixels in `tile_download.py` | Garbage reflectance values at composite gaps | Fill masks explicitly; declare nodata |
| DN scaling missing (÷10000) in index scripts | EVI/MSAVI (which have additive constants) were invalid on raw DN | Ratio indices hide scaling bugs; indices with constants expose them |
| Stale absolute paths / `2018/` paths in stage scripts | Scripts crashed or read archived data | "Interface drift between scripts written at different times" was the recurring failure class — hence the conventions section in CLAUDE.md |

---

## 7. Known Limitations (accepted, documented, not yet fixed)

1. **Cascade trains on upstream predictions.** Stage 2/3 training pixels are filtered by
   the previous stage's output, so their training distributions are biased toward "easy"
   pixels. Conditional (per-stage) metrics are therefore optimistic; the end-to-end number
   is the honest one.
2. **Pixel-level train/test splits inflate metrics.** Neighboring pixels from the same
   parcel are near-duplicates; some end up in train and some in test. Parcel-level
   (group) splits are the recommended fix and would lower all reported numbers somewhat —
   relative comparisons between runs remain valid since all runs share the same split
   scheme.
3. **`other_econ` is unlearnable** at its sample size (~16K pixels pre-cap); it currently
   contributes nothing and effectively disappears in the v2 mapping.
4. **Search ceilings were hit**: Stage 1 `n_components` chose 250 (its max). Widening to
   {150, 250, 350} may buy more accuracy.
5. **Durian F1 regressed** under the balanced v2 orchards model (0.905→0.712) — likely
   needs per-class threshold tuning now that probability mass spreads over 7 species.
6. **Labels are 2018 vintage (LU_RYG_2561)** while imagery spans later dates — land-use
   change between survey and acquisition is unmodeled label noise.

---

## 8. Current State & Next Steps

**State (2026-07-09):** `runs/dem_s1_s2_v2` is the best full chain (end-to-end econ recall
61.7 %) and the candidate production model. The S2-helps question is closed. Repo is on
`dev`, in sync with `main`, clean tree.

**Queued next steps, in rough priority order:**
1. Promote `dem_s1_s2_v2` to production status (update CLAUDE.md/skill.md references from
   the stale `runs/s1_dem` / `runs/dem_s1_s2` baselines).
2. Attack the loss table (§6.4): Stage 1 econ recall and Stage 2 routing are where the
   remaining 38 % lives. Candidate levers: lower Stage 1 threshold floor below 0.4, widen
   `n_components`, second-pass valid_cols on the econ subset.
3. Per-class threshold tuning for the Stage 3 orchards model (Durian recovery).
4. S1 temporal aggregation (18 dates → per-orbit/monthly medians, 108 → ~24 columns) —
   reduces the redundancy that drove kernel dilution and removes the all-NaN column
   problem at the source.
5. Tree-ensemble reference run (RF/GBM) as a feature-robust sanity baseline.
6. Parcel-level splits for honest absolute metrics before anything is delivered to LDD.
