# Draft report — hierarchical SVM crop classification, Rayong (47PQQ)

**Status: rough draft, 2026-08-25.** Numbers are measured unless marked otherwise.
One result is still running (M5, §7) and is marked *pending*. Terms follow `CONTEXT.md`.

---

## 1. Concept

The task is to produce a **pixel-level crop-type map** of Rayong province from Sentinel-2
imagery: for every 10 m ground cell, name the crop growing on it. The Land Development
Department maintains a parcel survey that says what *was* growing there in 2018; the model
learns the relationship between a pixel's spectral behaviour over the season and that
survey label, so the map can later be produced for years and places the survey does not
cover.

**Why a hierarchy rather than one flat classifier.** Thirteen crops is not a balanced
problem: rubber is 81% of all crop pixels and langsat is 0.01% — a ratio of roughly
6,700:1. A single 13-way model spends nearly all its capacity on the rubber/not-rubber
boundary. The hierarchy splits the decision into three easier ones:

```
every labelled pixel
      |
   Stage 1   is this an economic crop at all?  (economic / water / forest / other)
      |
   economic pixels
      |
   Stage 2   what KIND of crop?  (orchards / plantation / field / sink)
      |
   Stage 3   which crop exactly?  one specialist model per kind
      |
   one of 13 LU codes
```

Each stage sees a population and a label set narrow enough to be learnable, and each later
stage only has to separate crops that are already similar to one another.

The learner is a **kernel SVM** throughout. This is deliberate: the collaborating study
uses gradient-boosted trees, so this arm is the kernel-method half of a paired comparison
and has to stay one.

---

## 2. Dataset

### Imagery

| item | value |
|---|---|
| sensor | Sentinel-2 L2A, surface reflectance |
| tile | 47PQQ (Rayong), EPSG:32647, 10 m |
| tile size | 10,980 × 10,980 = 120,560,400 pixels |
| dates | 2018-10-31, 2018-11-30, 2018-12-31 |
| bands used | B02–B08, B8A, B11, B12 |

Three dates, chosen to bracket the end of the wet season into the dry season — the window
the joint paper controls, so both arms of the comparison see the same acquisitions.

### Features

Raw bands are not fed to the classifier. Each date is reduced to **8 spectral indices**
capturing greenness, water content, bare soil and built-up signal:

> NDVI, EVI, NDWI, BSI, NDBI, MSAVI, SWIR_NIR, SWIR_RATIO

**8 indices × 3 dates = 24 features per pixel.** A later experiment (§6, M3) adds MTCI
(red-edge chlorophyll) and raw B11 per date, giving 30.

So one training row is 24 numbers describing how one 10 m patch of ground looked in
October, November and December, and one crop code.

### Labels

LDD parcel survey `LU_RYG_2561` (2018), field `LU_ID_L3`, rasterised to the 10 m grid and
then **eroded by 3 pixels**, which discards parcel-boundary cells that mix two land covers.

**24,323,769 pixels carry a usable label — 20.2% of the tile.** The rest is unsurveyed,
background, or removed by the erosion.

### Class balance — the defining property of this dataset

| superclass | pixels | share |
|---|---|---|
| economic crops | 15,157,223 | 62.3% |
| others | 5,726,725 | 23.5% |
| forest | 2,536,964 | 10.4% |
| water | 902,857 | 3.7% |

Within the economic crops:

| crop | pixels | share of crops |
|---|---|---|
| Rubber | 12,320,950 | 81.29% |
| Cassava | 807,250 | 5.33% |
| Pineapple | 714,833 | 4.72% |
| Oil palm | 490,953 | 3.24% |
| Durian | 353,768 | 2.33% |
| Rice | 296,743 | 1.96% |
| Mango | 68,673 | 0.45% |
| Jackfruit | 42,873 | 0.28% |
| Longan | 19,692 | 0.13% |
| Mangosteen | 19,315 | 0.13% |
| Coconut | 13,945 | 0.09% |
| Rambutan | 6,385 | 0.04% |
| Langsat | 1,843 | 0.01% |

Almost every difficulty in this project traces back to this table.

### Splitting

Pixels from the same survey parcel are near-duplicates: adjacent, same crop, same
management, often the same day's reflectance. Splitting at pixel level therefore puts
near-copies of a training pixel into the test set and inflates every score.

The split here is **parcel-atomic** — a whole parcel goes entirely to one fold:

| fold | pixels | parcels |
|---|---|---|
| train | 14,842,294 | 17,475 |
| validation | 3,981,206 | 5,818 |
| test | 5,500,269 | 5,824 |

Validation is further halved **by parcel** into a calibration half (2,195,222 px) and a
tuning half (1,785,984 px), so that anything tuned later is not tuned on rows the
calibrator already used.

---

## 3. How the pipeline works

**Stage 1 — superclass.** A 4-way model over all 24.3 M rows: economic / water / forest /
other. Non-economic pixels leave the cascade here.

**Stage 2 — subclass.** Over pixels Stage 1 called economic. Four outcomes:

| subclass | crops |
|---|---|
| orchards | Durian, Rambutan, Mango, Longan, Jackfruit, Mangosteen, Langsat |
| plantation | Rubber, Oil palm, Coconut |
| field | Rice, Cassava, Pineapple |
| **sink** | everything Stage 1 wrongly called economic |

The sink matters: without it, every misrouted forest or water pixel is *forced* to become
some crop, which manufactures false positives. The subclass is **routing only** — it is
never reported.

**Stage 3 — crop.** One specialist per subclass, emitting the final LU code.

Routing between stages is **hard**: each stage commits to its highest-scoring branch. A
pixel routed wrongly cannot be recovered, because the model it reaches was never trained on
its true class. An alternative *joint* rule (multiply the probabilities along every path,
then argmax) was tested and is consistently worse — 0.2025 against 0.2283 — so hard routing
stands.

---

## 4. How the model is trained

**One balancing mechanism only: caps.** Earlier versions stacked three (per-class caps,
minority upsampling, and `class_weight='balanced'` inside the SVM), which produced an
effective training prior nobody could write down. Now there is one: each class is capped
before fitting, nothing is upsampled, no class weights.

| model | rows fitted | calibration rows |
|---|---|---|
| Stage 1 | 2,882,151 | 300,000 |
| Stage 2 | 800,000 | 300,000 |
| Stage 3 orchards | 172,371 | 55,708 |
| Stage 3 plantation | 148,344 | 300,000 |
| Stage 3 field | 210,000 | 217,338 |

Note the scale: Stage 1 fits on **2.88 M of the 14.84 M available training rows — 19.4%**.
Across all five models roughly 4.2 M rows are ever fitted, about **3.5% of the tile**.
Everything else is used for prediction and scoring.

**The estimator** at every stage is the same:

```
SimpleImputer(median) -> StandardScaler -> Nystroem(RBF) -> LinearSVC -> OneVsRest
                                                         -> per-class Platt calibration
```

`Nystroem` approximates an RBF kernel with a finite feature map, which is what makes a
kernel SVM tractable on millions of rows.

**Calibration** is a per-class Platt sigmoid fitted on the validation *calibration half*,
on top of a base model fitted once on the training fold and never refitted. The
calibration rows are a random subsample, not a per-class one, so the natural class
frequencies survive — the probabilities come out on the real prior rather than a
rebalanced one.

**Hyperparameters are frozen**, not searched: Stage 1 `C=1.0, n_components=250,
gamma=0.02`; Stages 2–3 `C=10.0, n_components=600, gamma=None`. Freezing removes the
inner cross-validation entirely, which is what stopped model selection from leaking (§5).

**Stage-3 experts are selected by true group membership**, not by whether Stage 2 routed
the pixel correctly.

**Stage-2 training candidates come from cross-fitted Stage-1 routes**: the training fold is
split by parcel into three parts, each routed by a model fitted on the other two, so no
training row is routed by a model that has seen it.

---

## 5. Problems

### 5.1 The published evaluation population was not held out

Of 29,117 parcels carrying labelled pixels, **25,084 (86.1%) donated at least one pixel to
training**. The population previously published as "unseen" — 21.2 M rows — contained only
**44,561 genuinely parcel-disjoint rows (0.18%)**. Every previously reported figure was
measured on ground the model had partly seen.

### 5.2 Code defects that invalidated published numbers

Found by review, each verified against source:

| defect | effect |
|---|---|
| Stage 3 trained only on correctly-routed pixels | experts never saw the contaminants they must reject |
| Stage 2 filtered to true economic crops before training | the sink was deleted; the "4-way" model was 3-way |
| `other_econ` was structurally unreachable | the 13 crops partition the 3 groups exactly, so label 4 could never be assigned |
| no `scoring=` in any search | all five models were selected by **accuracy** while the study argues macro F1 |
| calibration and search cross-validated at pixel level | leakage reached model selection, not just the final score |

### 5.3 A scoring convention that hid false positives

Three of six analysis scripts masked the population to crop-truth rows *before* computing
precision, so crop predictions landing on water, forest or "others" never counted against
anything. Measured: **278,123 such rows, 7.7% of all crop predictions**, and a **stable
+0.034 macro-F1 inflation** across a 256-cell sweep.

### 5.4 There is no stable class prior

Prior correction assumes the "true" prior is a property of the world. It is not here — it
is a property of which parcels were drawn. Across three disjoint parcel samples of the same
tile, within-group prevalence swings by **4.9× for oil palm** (42,868–109,693 px), 3.7× for
mango, 2.9× for rice. So a correction "toward the truth" has no well-defined target.

### 5.5 The uncertainty is larger than the effects being chased

A paired parcel bootstrap puts the 95% interval on the headline macro F1 at roughly
**±0.014**. Every improvement tested in §6 is smaller than that.

### 5.6 The real bottleneck: Stage 2 destroys the rare crops

This is the important one. Tracing every true crop pixel through the cascade:

| crop | survives Stage 1 | survives Stage 2 | final | **lost at Stage 2** |
|---|---|---|---|---|
| Jackfruit | 0.955 | 0.205 | 0.000 | **0.750** |
| Rambutan | 0.866 | 0.138 | 0.000 | **0.729** |
| Longan | 0.928 | 0.318 | 0.000 | **0.610** |
| Mango | 0.645 | 0.036 | 0.019 | **0.608** |
| Mangosteen | 0.870 | 0.276 | 0.003 | **0.594** |
| Durian | 0.909 | 0.353 | 0.352 | **0.556** |
| Rubber | 0.947 | 0.875 | 0.874 | 0.072 |

Stage 1 is mostly fine. **Stage 2 loses 27–75% of every crop except rubber.** And the
reason is specific — every orchard crop is routed predominantly into *plantation*:

| crop | → orchards | → plantation |
|---|---|---|
| Jackfruit | 0.215 | **0.653** |
| Rambutan | 0.159 | **0.641** |
| Langsat | 0.208 | **0.523** |
| Longan | 0.343 | **0.490** |
| Durian | 0.388 | **0.477** |

The plantation expert can only emit rubber, oil palm or coconut. So these crops score
**exactly 0.0000** end-to-end — while their own expert, measured in isolation, scores them
0.07–0.26. **The species are separable; the architecture throws the capability away.**

---

## 6. Solutions

### Fixed

**Parcel-atomic splitting** at every level, including inside model selection. Verified: no
parcel spans folds; no test row is ever fitted; calibration rows are disjoint from fitting
rows.

**One balancing mechanism** (caps only), so the training prior is stateable.

**A reachable sink**, which now receives 1.4–1.7 M training rows — the contaminants the
experts must learn to reject, which the old pipeline deleted.

**Cross-fitted routing.** Measured effect: in-sample routing was hiding **10.3% of the
contamination** Stage 2 exists to reject (sink share 15.66% → 17.03% on identical rows,
same model). Replicated on two independent runs.

**Strict scoring everywhere**, and the three defective scorers corrected.

**Honest uncertainty**: every headline now carries a parcel-level bootstrap interval.

### Tested, and small

| change | effect | verdict |
|---|---|---|
| protocol repairs (parcel-clean selection) | +0.0032, CI [+0.0001, +0.0070] | costs nothing — that is the result |
| operating point (α₂=0.2, α₃=0.7) | +0.0112, 7 → 10 crops above F1 0.01 | largest single gain; a decision rule, not new information |
| MTCI + raw B11 (24 → 30 columns) | +0.001 … +0.005 per expert | real but tiny; helps *most* where support is already largest |
| kernel capacity 600 → 1200 | +0.0092 (orchards expert) | interior optimum — 1800 is worse, so capacity saturates |

**All four are at or below the ±0.014 noise band on the headline.** Individually and
together they do not make this model competitive. That is itself a finding: features,
calibration, decision rules and capacity have each been tested and are each small.

### The change that targets the actual bottleneck

Given §5.6, the proposed fix is to **stop asking Stage 2 to make a distinction it
demonstrably cannot make**. Merge orchards and plantation into a single *tree crops*
subclass: Stage 2 becomes 3-way (tree / field / sink) and Stage 3 gets a 10-class tree
expert. The hierarchy stays three-stage and no reported taxonomy changes, because the
subclass was never reported.

Projected Stage-2 recall, computed from the measured routing shares above:

| crop | now | after merge |
|---|---|---|
| Rambutan | 0.138 | **0.693** |
| Jackfruit | 0.205 | **0.829** |
| Durian | 0.353 | **0.786** |
| Longan | 0.318 | **0.773** |
| Mangosteen | 0.276 | **0.636** |
| Langsat | 0.141 | **0.498** |
| Rubber | 0.875 | 0.879 |

Two-to-fivefold recovery for the orchard species at no cost to rubber.

**Stated as the bet it is:** this does not eliminate the orchards/plantation
discrimination, it relocates it into a 10-class expert trained on capped, near-balanced
data, instead of a router whose argmax is dominated by a 68.5% plantation prior. It is
untested as of this draft and will be judged on the validation tuning half — *not* on the
test fold, which has had its one read.

---

## 7. Where it stands

| | macro F1 | notes |
|---|---|---|
| published pixel-split figure | 0.2245 | measured on ground the model had partly seen |
| parcel-disjoint baseline | 0.2248 | honest protocol, first clean number |
| + protocol repairs (M0) | **0.2283** | 95% CI [0.2119, 0.2404] |
| + features, capacity, operating point (M5) | *pending* | predeclared before the test fold was read |
| + tree merge | *not yet run* | judged on validation only |

**The comparison the joint paper has to survive** is with a random-forest study reporting
0.714 — which is a *support-weighted* F1, not macro. Recomputed from its own per-class
table, its macro F1 is ≈0.602 over the 13 crops. It is also measured on a rebalanced
population (~390 k pixels per class) with a pixel-level split, so the two numbers are not
directly comparable. Establishing a shared evaluation population is a prerequisite to any
joint table, and is a conversation with the collaborators rather than a script.

**Honest summary of the gap.** This arm is behind, and the reasons are now measured rather
than guessed: not the features, not the kernel, not the calibration — the cascade loses
most of its rare crops at Stage 2 before any expert sees them. That is a fixable
architectural problem, and §6 proposes the fix.
