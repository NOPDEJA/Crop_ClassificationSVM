# Sentinel-2 + Hierarchical SVM — Crop Classification in Rayong (47PQQ)

Analysis of the S2-only SVM cascade, prepared for joint publication with the
XGBoost cascade developed in parallel.

Run `s2_2018_3date`, completed 2026-08-20. All figures below come from that run
unless labelled otherwise; the DEM+S1 and DEM+S1+S2 comparison figures come from
the existing `s1_dem_v2` and `dem_s1_s2_v2` runs.

**Summary of findings**

1. Sentinel-2 indices alone match a 134-feature DEM+S1+S2 stack at superclass
   separation (0.762 vs 0.766) and beat DEM+S1 (0.733) — §6.1.
2. The optical signal runs out at species level: S2-only orchard macro F1 (0.405)
   is indistinguishable from DEM+S1 (0.399) and far below the fused stack
   (0.519). Separating dense evergreen canopies needs structure, not spectra — §6.1.
3. The dominant end-to-end error is **not** spectral confusion but class-prior
   collapse: every stage trains on rebalanced data and is applied to a tile with
   real priors, over-predicting rare orchard species by 12–168× — §6.4.
4. That finding applies to the whole project, including the current best model.
   The headline "61.7 % end-to-end recall" is a recall-only figure that
   over-prediction inflates, and it is dominated by rubber — §6.4.
5. Traced pixel by pixel, the cascade is *redistributing rubber*: 83 % of
   predicted oil palm, 72 % of predicted coconut and 61 % of predicted durian is
   truly rubber, and the loss splits almost evenly across the three stages
   (14.7 / 18.0 / 7.5 %) — §6.5.
6. Post-hoc prior correction was tested. At Stage 2 it is free and strictly
   positive (accuracy 0.660 → 0.697, no class worse). At Stage 3 it raises
   accuracy further but drives the rarest species to exactly zero F1 — it buys
   precision by ceasing to predict them — §6.6.
7. Swept properly as a correction *strength*, the response surface is **flat**:
   keeping all 13 crops alive costs 0.0074 macro F1, and the theoretically
   correct full correction (α=1) is not optimal. The published operating point
   (α₂ 0.8, α₃ 0.3) lifts accuracy 0.660 → 0.742 and κ 0.497 → 0.577 with **no
   class regressing**. Split-half validation puts the selection bias at **0.0004
   macro F1**, so those figures stand as reported — §6.7.
8. Stage 1's tuned decision thresholds **are never applied to the tile**; the
   deployed rule is plain argmax, verified on all 24,323,769 rows. Correcting
   Stage 1's prior is monotone to α=1 and its per-crop gains are ordered by
   rarity (Langsat +0.244, Rambutan +0.206). The optimal correction strength
   tracks class separability and must be set per stage — §6.8.
9. The largest defects are not in the model family at all, but in three inherited
   configuration choices — the Oct–Dec window (two of five available 2018
   composites unused, and the one window in which rubber and the evergreen
   orchards are least separable), Stage 1's `PCA(10)`, and the `n_components=150`
   kernel ceiling in stages 2 and 3. They interact: at 10 PCA components the
   5-date arm scores *below* the 3-date one, and at 150 kernel components the
   extra dates look nearly worthless. Fixed together they are worth **+0.119
   macro F1 (+25 % relative)** on a balanced probe — more than every prior
   correction combined. Confirmed end to end on the tile: κ 0.489 → 0.535,
   monotone on every metric, no class regressing. Every one of the eight
   hyperparameter searches across the two fixed arms chose the **top** of its
   capacity grid, so this is a lower bound — §6.9.
10. The rare species are **not** spectrally indistinguishable. Under balanced
    priors, five dates and a non-throttled kernel, the same feature family
    separates them at F1 0.44–0.68 — Langsat, with 27 parcels province-wide,
    scores highest of the rare orchards. Their end-to-end collapse is a property
    of the pipeline, not of Sentinel-2. Balanced-prior figures are not achievable
    end-to-end and are not quoted as such — §6.10.
11. Arms of this pipeline **never share a held-out set**, even on the same matrix
    with the same seed: stages 2 and 3 sample from the pixels Stage 1 routed, so
    a change anywhere upstream moves the split. Cross-arm scores must be taken on
    the intersection of every arm's held-out set, and per-crop cross-arm claims
    for the rare species cannot be made at all — one population inflates them by
    conditional sampling, the other empties them by construction — §6.9.
12. The cascade **transfers across years and is not memorising parcels**. Scored
    on one population stable across all three surveys and never fitted on, the 2018
    models lose 0.067 accuracy and 0.108 κ at two years and 0.099 and 0.135 at six —
    monotone with elapsed time. On ground that *did* change class it reports the
    **new** class 3.4× more often than the old one, 10:1 for rubber, the one class it
    does not over-predict. The annual field crops break the monotone pattern by
    dipping in 2020 and recovering in 2024, which two tested explanations fail to
    account for — §6.11.

---

## 1. What this study is

A three-stage SVM cascade classifies 13 economic crop types at 10 m pixel
resolution from Sentinel-2 spectral indices alone, validated against the LDD
2561 (2018) parcel survey for tile 47PQQ, Rayong province.

It is the S2-only counterpart to two neighbouring pieces of work:

| Study | Sensor | Classifier | Structure | Epoch |
|---|---|---|---|---|
| Prior RF paper | Sentinel-2 | Random Forest | flat, 15 classes | 2024 (+2020) |
| **This study** | **Sentinel-2** | **SVM** | **3-stage cascade** | **2018** |
| Collaborator | Sentinel-2 | XGBoost | cascade (water → buildings → crops) | 2018, cross-tested 2020/2024 |

This study and the collaborator's share a training epoch (2018) and a date
window (October–December), which makes them directly comparable. The prior RF
paper differs in epoch and feature set and is treated as related work, not as a
controlled comparison arm.

## 2. Data

**Imagery.** Sentinel-2 Level-2A, tile 47PQQ, three monthly composites:
2018-10-31, 2018-11-30, 2018-12-31. Each composite is a per-band masked median
over the widest available acquisition window in that month, which suppresses
cloud and haze. Bottom-of-atmosphere offset and quantification are applied from
the scene metadata; invalid pixels are removed with the scene classification
layer before compositing.

The October–December window is chosen to match the collaborator's crop model and
the prior RF paper. It also covers the fruit-blooming period that separates the
orchard species.

**Labels.** LDD parcel survey `LU_RYG_2561`, attribute `LU_ID_L3`, rasterized to
10 m. Parcel polygons extend past the true crop boundary, so each polygon is
contracted inward by 3 pixels (30 m) before rasterization — the same erosion
distance the prior RF paper used. This removes mixed pixels where the surveyed
boundary and the actual field edge disagree.

**Extent.** 24,323,769 labeled pixels.

## 3. Features

Eight spectral indices per date × 3 dates = **24 features**.

> Five 2018 composites exist for this tile; the two dry-season dates (2018-03-31,
> 2018-04-30) are not used here. That is inherited from the comparison window,
> and §6.9 measures what it costs.

| Index | Purpose |
|---|---|
| NDVI | vegetation greenness; the baseline separator of vegetation from bare surfaces |
| EVI | greenness without NDVI's saturation in dense canopy |
| MSAVI | greenness with soil-brightness correction, for sparse or early-growth cover |
| NDWI | separates open water from land |
| BSI | bare-soil signal, for fallow and freshly prepared fields |
| NDBI | built-up signal, to keep settlement out of the crop classes |
| SWIR_NIR | canopy and soil water content |
| SWIR_RATIO | contrast between the two SWIR bands; moisture and residue |

Band-order note: the Sentinel-2 composites stack bands in `sorted()` order, which
places B11 and B12 *before* B8A. The index computation matches this on-disk
order. An earlier version of this pipeline did not, and its SWIR-derived indices
were computed from the wrong bands; all results here use the corrected mapping.

No terrain, radar, or texture features are used. In particular this study uses
no neighbourhood statistics, so it is a strictly pixel-wise model — a relevant
difference from the collaborator's feature set, which appends a 3×3 local mean
and variance for every feature.

Missing data: cloud gaps leave 5.4 %, 0.3 % and 1.0 % of pixels unobserved in the
October, November and December composites respectively. No feature column is
entirely empty. Gaps are mean-imputed inside the model pipeline.

## 4. Model

A cascade of three SVM stages. Each stage is a Nyström RBF kernel approximation
feeding a linear SVM, wrapped one-vs-rest and probability-calibrated.

```
24.3M pixels × 24 features
        │
Stage 1 ── econ / water / forest / others          (4 superclasses)
        │  keep pixels predicted "econ"
Stage 2 ── orchards / plantation / field           (routing)
        │  send each pixel to its subclass model
Stage 3 ── final LU_CODE within the subclass       (e.g. durian vs mangosteen)
```

**Why a cascade.** Class imbalance spans four orders of magnitude — plantation
crops have millions of pixels, the rarest orchard species a few thousand — and a
flat classifier collapses onto the majority classes. Each stage rebalances at its
own level. The stages also need different evidence: separating water and forest
from cropland is easy, separating durian from mangosteen is not.

**What a cascade costs.** A pixel routed to the wrong subclass at Stage 2 cannot
be recovered at Stage 3, because the model it reaches was never trained on its
true class. This error propagation is why per-stage accuracy overstates the
system, and why the end-to-end number below is the one that matters.

**Kernel scale.** The Nyström `gamma` is left at `1/n_features` or searched in a
grid bracketing it. This is not a detail: an earlier configuration used a gamma
grid roughly 100× above the correct scale, at which every pixel appears maximally
dissimilar to every other, and adding informative features *degraded* every
metric. Feature-set conclusions are only valid under hyperparameters scaled to
that feature set.

Stage 1 additionally reduces to 10 principal components before the kernel, so its
gamma scale is set by the PCA dimensionality rather than the raw feature count.

The Stage-1 hyperparameter search samples 3 candidates (reduced from 5 under
deadline pressure after a process loss cost one full search). The grid itself is
unchanged, and because the sampler is seeded the retained candidates are the
first three of the original five - including the configuration that won the
equivalent search in the DEM+S1+S2 run. Stages 2 and 3 search 6 candidates each,
unchanged.

## 5. Evaluation protocol

**Sampling.** Stage 1 samples up to 400,000 pixels per LU code, then caps each
superclass (econ 1.0 M, others 0.8 M, forest 0.6 M, water 0.5 M) to 2,900,000
pixels total, split 70/15/15 train/val/test. Stages 2 and 3 sample from the
pixels the previous stage routed to them, capped per group and per LU code.

**The population problem.** Per-stage reports score only pixels the previous
stage passed, so they are conditional and survivor-biased. The end-to-end score
covers every labeled pixel — but that includes the pixels the models were fitted
on, which makes it optimistic in the other direction.

This analysis therefore scores on three explicit populations:

| Population | Definition | Use |
|---|---|---|
| `all` | every labeled pixel | comparable to this project's earlier end-to-end figures; optimistic |
| `unseen` | pixels no stage ever sampled — 21,423,769 of 24,323,769 | **the number to publish** |
| `matched` | `unseen`, subsampled to the prior RF paper's per-class test supports | class priors matched, so weighted metrics are comparable |

The `unseen` set is recovered exactly rather than approximately: every sampling
step is driven by a seeded generator and depends only on the labels and the saved
stage predictions, never on the feature values, so the selection is replayed
deterministically after the fact (`reconstruct_sampled_rows.py`). The
reconstruction was verified against the training log — 2,900,000 rows with
superclass counts matching exactly.

**Flat re-scoring.** To compare against models that are not cascades, the
cascade's final answer per pixel is collapsed to one flat label set: the 13 crop
codes, plus reservoir and others. Forest folds into others to match the
comparison taxonomy but is also reported separately so it is not hidden.

**Split caveat.** Splits are pixel-level, so pixels from one parcel can fall on
both sides of the split and absolute numbers are optimistic relative to a
parcel-level (grouped) split. This is consistent with the prior RF paper and with
the collaborator's protocol, so relative comparisons hold; parcel-level
validation is future work and is required before operational delivery to LDD.

## 6. Results

Stage-level scores are reported first (§6.1), then the composed cascade (§6.2),
then the flat re-scoring that permits comparison with non-cascaded models (§6.3).
§6.4 explains why the stage-level and end-to-end numbers diverge as sharply as
they do, §6.5 measures that divergence pixel by pixel, and §6.6 tests the fix
§6.4 proposes; together they are the most important sections for interpreting
everything above them.

### 6.1 Per-stage

**Stage 1 — superclass separation** (held-out test split, 435,000 pixels).

Selected hyperparameters: `C=1.0`, `n_components=250`, `gamma=0.02`.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| econ | 0.703 | 0.855 | 0.772 | 150,000 |
| water | 0.895 | 0.852 | 0.873 | 75,000 |
| forest | 0.803 | 0.869 | 0.835 | 90,000 |
| others | 0.729 | 0.508 | 0.598 | 120,000 |
| **accuracy** | | | **0.762** | 435,000 |
| macro avg | 0.782 | 0.771 | 0.769 | |

**Against the fused-sensor arms.** Because every arm's sampling is driven by the
same seed and depends only on the labels, all three arms are trained and tested
on *identical pixels*. The comparison is therefore controlled:

| Stage 1 | S2-only (24 feat) | DEM+S1 (94 feat) | DEM+S1+S2 (134 feat) |
|---|---|---|---|
| accuracy | 0.762 | 0.733 | **0.766** |
| econ F1 | **0.772** | 0.725 | 0.755 |
| water F1 | 0.873 | 0.830 | **0.875** |
| forest F1 | 0.835 | 0.875 | **0.880** |
| others F1 | 0.598 | 0.549 | **0.606** |
| macro F1 | 0.769 | 0.745 | **0.779** |

Two things stand out. First, Sentinel-2 indices alone reach within 0.004 accuracy
of the full 134-feature fused stack while using 24 features, and clearly beat
DEM+S1 (0.733). Nearly all of the superclass signal is optical. Second, S2-only
has the **best economic-crop F1 of the three arms** — adding terrain and radar
actually costs econ F1 (0.772 → 0.755), consistent with those channels diluting
the kernel for the class that optical evidence already separates well.

Where the fused stack earns its place is forest (0.880 vs 0.835): terrain and
radar structure carry information about forest that spectral indices do not.

The weakest class in every arm is `others` (recall 0.51 here), which is expected —
it is a catch-all of everything that is not crop, water or forest, and has no
coherent spectral signature.

**Stage 2 — subclass routing** (held-out test split, 90,000 pixels, 30,000 per
group). Selected hyperparameters: `C=10.0`, `n_components=150`, `gamma=1/n_features`.

| Subclass | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| orchards | 0.810 | 0.841 | 0.825 | 30,000 |
| plantation | 0.893 | 0.781 | 0.833 | 30,000 |
| field | 0.805 | 0.875 | 0.839 | 30,000 |
| **accuracy** | | | **0.832** | 90,000 |

| Stage 2 | S2-only | DEM+S1 | DEM+S1+S2 |
|---|---|---|---|
| accuracy | 0.832 | 0.773 | **0.858** |
| macro F1 | 0.832 | 0.773 | **0.858** |

S2-only again sits far above DEM+S1 (+5.9 pp) and below the fused stack
(-2.6 pp). Routing between orchards, plantation and field is largely an optical
problem, but it is the one place where the fused stack's advantage is real.

*Caveat, unlike Stage 1:* each arm's Stage 2 is evaluated on the pixels its own
Stage 1 routed to it, so the three arms score on different populations. Group
sizes are equalised by capping, but the pixels are not the same. These numbers
are indicative; the controlled comparison is the end-to-end score below.

**Stage 3 — crop code within subclass** (held-out test splits per group).

| | S2-only | DEM+S1 | DEM+S1+S2 |
|---|---|---|---|
| **field** — accuracy | 0.786 | 0.812 | **0.829** |
| Rice F1 | 0.929 | 0.974 | **0.981** |
| Cassava F1 | 0.708 | 0.710 | **0.741** |
| Pineapple F1 | 0.718 | 0.746 | **0.762** |
| **plantation** — accuracy | 0.863 | 0.836 | **0.898** |
| Rubber F1 | 0.879 | 0.846 | **0.906** |
| Oil palm F1 | 0.867 | 0.857 | **0.897** |
| Coconut F1 | 0.237 | 0.537 | **0.668** |
| **orchards** — accuracy | 0.548 | 0.537 | **0.635** |
| macro F1 | 0.405 | 0.399 | **0.519** |
| Durian F1 | 0.670 | 0.651 | **0.712** |
| Mango F1 | 0.708 | 0.825 | **0.902** |
| Mangosteen F1 | 0.258 | 0.291 | **0.397** |
| Langsat F1 | 0.235 | 0.201 | **0.409** |

**The central finding of this study.** Read together with Stages 1 and 2, the
three stages tell a consistent and interpretable story:

| Task | S2-only vs DEM+S1 | S2-only vs fused |
|---|---|---|
| Stage 1 — is this cropland, water or forest? | **+2.9 pp** | −0.4 pp |
| Stage 2 — which crop group? | **+5.9 pp** | −2.6 pp |
| Stage 3 — which species? | ≈ equal (orchards +0.6 pp) | **−11.4 pp** |

Sentinel-2 indices alone are sufficient to *locate* and *route* crops: at
superclass separation they match a 134-feature fused stack, and at subclass
routing they beat terrain-plus-radar decisively. But at species level the optical
signal runs out. Orchard macro F1 for S2-only (0.405) is statistically
indistinguishable from DEM+S1 (0.399) and far below the fused stack (0.519).

The interpretation is physical rather than statistical. Durian, mangosteen,
langsat and rambutan are all dense evergreen tropical canopies with similar
chlorophyll and water content; their spectral indices overlap heavily whatever
the date. What separates them is canopy architecture and terrain preference —
exactly the information SAR backscatter and DEM slope carry and spectral indices
do not. Coconut is the clearest case: F1 0.237 from optical alone versus 0.668
with structure added.

**Consequence for an S2-only product.** An S2-only pipeline can deliver reliable
cropland masks, water and forest layers, and crop-group maps. It should not be
expected to deliver species-level orchard maps at useful accuracy. That is a
finding worth stating plainly rather than an embarrassment — it bounds the
method and motivates sensor fusion for the species task.

### 6.2 End-to-end

Composing all three stages over every labeled pixel and requiring an exact
LU-code match:

| End-to-end econ recall | S2-only | DEM+S1 | DEM+S1+S2 |
|---|---|---|---|
| all economic crops | **0.596** | 0.473 | 0.617 |

S2-only again lands close to the fused stack (−2.1 pp) and far above DEM+S1
(+12.3 pp), consistent with Stages 1 and 2.

Where the 40.4 % of missed pixels go, out of 15,157,223 true economic pixels:

| Outcome | Pixels | Share |
|---|---|---|
| correct LU code end-to-end | 9,040,000 | 59.6 % |
| dropped at Stage 1 (not predicted econ) | 2,166,284 | 14.3 % |
| misrouted at Stage 2 (wrong subclass) | 2,665,828 | 17.6 % |
| wrong code inside the right subclass | ~1,285,000 | 8.5 % |

By subclass: plantation 0.611, field 0.542, orchards 0.423.

**But this metric is recall, and recall alone is misleading here** — see §6.4.

### 6.3 Flat 15-class

Scored on the three populations of §5. `held_out` is the honest number.

| Metric | all | **held_out** | matched | *prior RF paper (test)* |
|---|---|---|---|---|
| pixels | 24,323,769 | 21,180,058 | 303,948 | *303,947* |
| accuracy | 0.663 | **0.660** | 0.514 | *0.716* |
| Cohen's κ | 0.525 | **0.497** | 0.453 | *0.678* |
| F1 weighted | 0.706 | **0.714** | 0.525 | *0.714* |
| F1 macro | 0.279 | **0.226** | 0.308 | |

Per-class F1 on `held_out`: Para rubber 0.760, Reservoir 0.750, Others 0.745,
Pineapple 0.347, Cassava 0.288, Rice 0.252, Oil palm 0.124, Durian 0.110, and
the six remaining orchard species all below 0.01. Forest, scored as its own
class rather than folded into others, reaches F1 0.706 (precision 0.577, recall
0.909, support 2,026,964).

**The comparison with the prior RF paper must be read carefully.** The `matched`
row is the closest analogue to its setup (same class priors, similar test size),
and there the S2-only SVM cascade scores 0.514 accuracy / 0.453 κ against the RF
paper's 0.716 / 0.678. On that reading the cascade is clearly behind. However the
two are *not* a controlled comparison — different epoch (2018 vs 2024+2020),
different label survey, different feature set (they use MTCI, we use MSAVI and
two SWIR ratios), and flat versus cascaded structure. The gap should be attributed
to §6.4 before it is attributed to the classifier.

**Re-measured against the prior RF paper, after correction.** The `matched` population is
`held_out` resampled to that paper's per-class test supports, so class priors agree and
weighted metrics are comparable. Drawn from reporting half B only, it yields 302,545
pixels against the RF paper's 303,947 (Longan reaches 2,077 of 2,790 and Jackfruit 3,829
of 4,519, since B is half of `held_out`; every other class is met in full).

| on `matched` | accuracy | κ | F1 weighted |
|---|---|---|---|
| SVM cascade, baseline | 0.5161 | 0.4552 | 0.5277 |
| SVM cascade, α = (0.7, 0.3) | **0.5277** | **0.4630** | 0.5216 |
| SVM cascade, α = (0.8, 0.3) | 0.5138 | 0.4474 | 0.5050 |
| **prior RF paper** | **0.7160** | **0.6780** | **0.7140** |

The cascade remains well behind — roughly 19 points of accuracy and 22 of κ — and prior
correction does not close that gap.

**The correction behaves differently here than on the tile, and the direction is
informative.** The milder correction (0.7, 0.3) *improves* `matched` accuracy and κ over
baseline, while the stronger one (0.8, 0.3) degrades both. That is what the construction
predicts: `matched` is artificially resampled toward the RF paper's balance, so correcting
toward the tile's real priors moves away from it, and the further the correction goes the
worse the mismatch. `matched` is the right population for comparing with that paper and
the wrong one for judging prior correction — the earlier reading in §6.6, that correction
simply hurts on `matched`, was drawn from the stronger point alone and is refined here.

The caveats of §6.3 are unchanged and still carry most of the weight: different epoch
(2018 against 2024+2020), different label survey, different feature set, and a flat model
against a cascade. To those, §6.9 adds three configuration defects that depress this side
of the comparison and not the RF paper's. The gap should be attributed there before it is
attributed to SVM-versus-RF, and the `s2_2018_3date_v2` arm — same window, defects removed
— is the run that will test how much of it survives.


### 6.4 Where the errors are: class-prior collapse

The single largest error source is not spectral confusion. It is that **every
stage is trained on rebalanced data and then applied to a tile with the real
class priors.**

Predicted versus true pixel counts on held-out pixels:

| Crop | true px | predicted px | over-prediction |
|---|---|---|---|
| Langsat | 464 | 77,840 | **168×** |
| Rambutan | 1,818 | 187,481 | **103×** |
| Longan | 4,136 | 148,028 | 36× |
| Coconut | 8,648 | 199,327 | 23× |
| Jackfruit | 7,555 | 127,290 | 17× |
| Mangosteen | 5,936 | 80,971 | 14× |
| Mango | 28,648 | 341,209 | 12× |
| Durian | 126,842 | 853,058 | 6.7× |
| Rice | 143,554 | 364,915 | 2.5× |
| Oil palm | 304,356 | 747,688 | 2.5× |
| Cassava | 560,245 | 1,346,075 | 2.4× |
| Pineapple | 475,589 | 1,004,299 | 2.1× |
| **Para rubber** | 11,960,721 | 7,625,531 | **0.64×** |

Total predicted economic pixels (13.10 M) closely matches total true economic
pixels (13.63 M). The *quantity* is right; the *distribution* is inverted. The
cascade takes rubber — 88 % of all economic pixels — and distributes it across
the rare orchard species.

The mechanism is structural. Stage 1 caps each superclass; Stage 2 caps each
subclass group to 200,000 pixels, making orchards, plantation and field equally
likely when in reality they differ by more than an order of magnitude; Stage 3
caps to 70,000 per LU code, upsamples minority codes to a target of 20,000, and
additionally applies `class_weight='balanced'`. Each model therefore learns
approximately uniform priors, and nothing restores the true base rates at
inference time.

This explains the otherwise puzzling gap between conditional and end-to-end
scores. Stage 3's orchard model scores Mango F1 0.708 on its own balanced test
split; the same model contributes Mango F1 0.005 end-to-end. Both are correct
measurements of different things — the first on a population where Mango is 13 %
of pixels, the second where it is 0.14 %.

**This affects every arm of the project, not just the S2-only one.** The current
best model, `dem_s1_s2_v2`, shows the same signature in its own end-to-end
report: precision 0.011 for Rambutan, 0.013 for Coconut, 0.013 for Langsat,
0.038 for Mangosteen, 0.055 for Mango, against recalls of 0.11–0.62. Only rubber
has meaningful precision (0.956).

**Consequence for the project's headline metric.** End-to-end *recall* is exactly
the quantity that over-prediction inflates: predicting a class far too often
raises its recall while destroying its precision. Rubber alone is 81 % of
economic pixels and contributes roughly 0.52 of the reported 0.617 recall for
`dem_s1_s2_v2`. That figure therefore mostly measures how well the pipeline finds
rubber, not how well it distinguishes 13 crops. Any future reporting should pair
it with precision, or use macro F1.

**This is a fixable problem, but only partly.** Options, cheapest first:
(1) post-hoc prior correction — divide predicted probabilities by the training
prior and multiply by the true prior, which needs no retraining and only the
saved probability arrays; (2) drop `class_weight='balanced'` at Stage 3 and keep
the natural distribution; (3) tune per-class decision thresholds against macro F1
on a held-out split rather than accepting argmax. **Option 1 has since been
tested — see §6.6.** It is a clear win at Stage 2 and a double-edged one at
Stage 3, where it raises accuracy by suppressing the rare species entirely.
Options 2 and 3 remain untested.

### 6.5 Anatomy of the errors

§6.4 identifies the mechanism. This section measures it, on the `held_out`
population (13,628,512 economic pixels), by tracing every pixel through the
cascade rather than scoring each stage separately.

**The loss is spread evenly across the three stages.**

| Fate | Pixels | Share |
|---|---|---|
| dropped at Stage 1 (not predicted econ) | 2,004,856 | 14.7 % |
| misrouted at Stage 2 (wrong subclass) | 2,447,507 | 18.0 % |
| wrong code inside the right subclass | 1,021,843 | 7.5 % |
| **correct end to end** | **8,154,306** | **59.8 %** |

No stage dominates, so no single intervention recovers the missing 40 %. In
particular, improving Stage 3 — the stage that gets the most attention because
it is the one that names crops — can recover at most 7.5 pp.

**Rubber is the raw material of every other class's false positives.** Reading
the confusion the other way round — for each *predicted* class, what was
actually there — makes the prior collapse concrete:

| Predicted | Pixels | True composition (top 3) |
|---|---|---|
| Para rubber | 7,625,531 | Rubber 98 % |
| Oil palm | 747,688 | **Rubber 83 %**, Oil palm 9 % |
| Coconut | 199,327 | **Rubber 72 %** |
| Jackfruit | 127,290 | **Rubber 65 %**, Oil palm 7 % |
| Durian | 853,058 | **Rubber 61 %**, Durian 6 % |
| Rambutan | 187,481 | **Rubber 55 %**, Oil palm 5 % |
| Langsat | 77,840 | **Rubber 53 %** |
| Mangosteen | 80,971 | **Rubber 51 %**, Oil palm 10 % |
| Pineapple | 1,004,299 | **Rubber 46 %**, Pineapple 26 % |
| Mango | 341,209 | **Rubber 45 %**, Oil palm 6 % |
| Cassava | 1,346,075 | **Rubber 42 %**, Cassava 20 % |
| Longan | 148,028 | **Rubber 41 %**, Oil palm 25 % |
| Rice | 364,915 | **Rubber 29 %**, Rice 18 % |

Every non-rubber class is majority-rubber by true label, and the two majority
classes (rubber, oil palm) account for most of what is left. Total predicted
economic pixels (13.10 M) match total true economic pixels (13.63 M) to within
4 %. The cascade is not inventing cropland; it is **redistributing rubber**.
This is why the flat-15 macro F1 (0.226) sits so far below the weighted F1
(0.714): the weighted figure is rubber's score, and the macro figure is the
score of twelve classes made mostly of rubber.

**Rubber's own 38 % loss is thin and spread, not concentrated.**

| Stage | Pixels | Share of true rubber | Goes to |
|---|---|---|---|
| dropped at Stage 1 | 1,617,543 | 13.5 % | forest 7.8 %, others 5.6 % |
| misrouted at Stage 2 | 2,143,608 | 17.9 % | field 9.6 %, orchards 8.4 % |
| wrong code at Stage 3 | 761,389 | 6.4 % | oil palm 5.2 %, coconut 1.2 % |
| **correct** | **7,438,181** | **62.2 %** | |

**The rare orchards fail in both directions at once.** Stage-1 economic-crop
recall, per true crop:

| Crop | S1 econ recall | Crop | S1 econ recall |
|---|---|---|---|
| Para rubber | 0.865 | Coconut | 0.560 |
| Pineapple | 0.858 | Longan | 0.404 |
| Cassava | 0.811 | Langsat | 0.349 |
| Oil palm | 0.735 | Rambutan | 0.334 |
| Durian | 0.701 | Mango | 0.255 |
| Jackfruit | 0.624 | Mangosteen | 0.211 |
| Rice | 0.590 | | |

Mangosteen loses 79 % of its pixels before Stage 2 ever sees them — 48 % to
*others*, 30 % to *forest* — while simultaneously being predicted 14× too
often elsewhere in the tile. The rare orchard species are therefore not merely
over-predicted: **the model has no usable signal for them in either direction**,
and balanced training converts that absence of signal into confident noise
scattered across rubber. Prior correction (§6.4, option 1) addresses the
over-prediction half of this; it cannot recover pixels already discarded at
Stage 1.

**One large error is physical rather than statistical.** 7.8 % of rubber and
20.1 % of oil palm are classified as *forest* at Stage 1. Rubber and oil palm
plantations are closed-canopy trees, and October–December spectral indices do
not distinguish a managed tree crop from natural forest. This is the same gap
the fused stack closes (forest F1 0.835 → 0.880, §6.1), and it is an argument
for adding structural sensors rather than for further tuning of the optical
model.

**Consequences for the fix list in §6.4.** The measurements above split the
error into three parts with different remedies:

| Error | Size | Remedy |
|---|---|---|
| rubber redistributed into 12 classes | ~18 pp of econ, and essentially all of the macro-F1 loss | prior correction (§6.4, option 1) — no retraining |
| rare orchards discarded at Stage 1 | 0.2–0.8 of each rare class | Stage-1 per-class thresholds; ultimately a labelling/support problem |
| tree crops read as forest | ~1.4 M pixels | structural features (SAR, terrain); not solvable from optical indices |

### 6.6 Prior correction, tested

§6.4 proposed post-hoc prior correction as the cheapest fix and predicted it
"would likely move the flat-15 numbers substantially". It does — but not in the
way that phrasing implies, and the qualification is the point of this section.

**Method.** Each stage's probabilities are reweighted at inference time by the
ratio of the target prior to the prior it was fitted on, then renormalised:

    p'(c|x) ∝ p(c|x) · π_target(c) / π_train(c)

No model is refitted. π_train is read off the counts each stage actually saw
(Stage 2: exactly 140,000 pixels per subclass, i.e. uniform; Stage 3: the
post-upsampling counts in its meta JSON). π_target is estimated two ways —
**em**, Saerens-Latinne-Decaestecker EM on the probabilities alone, which uses
no labels and is therefore deployable; and **oracle**, the true class
distribution of the routed pixels, which is not deployable and only bounds what
the technique can buy. Stage 1 is left untouched so routing into the cascade is
identical to the baseline and the comparison isolates the Stage-2/3 effect.

The priors involved are badly mis-specified, as expected:

| Stage 2 (orchards / plantation / field) | prior |
|---|---|
| fitted on | 0.333 / 0.333 / 0.333 |
| EM estimate | 0.044 / 0.791 / 0.165 |
| truth | 0.031 / 0.852 / 0.117 |

**Results** (flat 15-class, `held_out`):

| Variant | accuracy | κ | F1 weighted | F1 macro |
|---|---|---|---|---|
| baseline | 0.660 | 0.497 | 0.714 | 0.226 |
| **em, Stage 2 only** | **0.697** | **0.530** | **0.738** | **0.245** |
| oracle, Stage 2 only | 0.699 | 0.528 | 0.739 | 0.243 |
| em, Stages 2+3 | 0.725 | 0.555 | 0.752 | 0.219 |
| oracle, Stages 2+3 | **0.767** | **0.597** | **0.769** | 0.262 |

Per-class F1 on `held_out`:

| Class | support | baseline | em S2-only | em S2+3 | oracle S2+3 |
|---|---|---|---|---|---|
| Para rubber | 11,960,721 | 0.760 | 0.797 | 0.835 | **0.848** |
| Others | 7,073,689 | 0.745 | 0.745 | 0.745 | 0.745 |
| Cassava | 560,245 | 0.288 | **0.307** | 0.278 | 0.295 |
| Reservoir | 477,857 | 0.750 | 0.750 | 0.750 | 0.750 |
| Pineapple | 475,589 | 0.347 | **0.374** | **0.000** | 0.371 |
| Oil palm | 304,356 | 0.124 | 0.197 | 0.381 | **0.386** |
| Rice | 143,554 | 0.252 | 0.284 | 0.263 | **0.363** |
| Durian | 126,842 | 0.110 | **0.169** | **0.000** | 0.154 |
| Mango | 28,648 | 0.005 | 0.005 | 0.004 | 0.002 |
| Coconut | 8,648 | 0.001 | 0.004 | 0.004 | **0.012** |
| Jackfruit | 7,555 | 0.007 | **0.021** | 0.020 | 0.005 |
| Mangosteen | 5,936 | 0.003 | **0.008** | **0.000** | **0.000** |
| Longan | 4,136 | 0.004 | **0.012** | **0.000** | 0.001 |
| Rambutan | 1,818 | 0.001 | **0.002** | 0.001 | **0.000** |
| Langsat | 464 | 0.001 | **0.003** | **0.000** | **0.000** |

**Three findings, in order of importance.**

*1. Correcting Stage 2 alone is a free and unambiguous win.* `em, Stage 2 only`
improves accuracy by 3.7 pp, κ by 3.3 pp, weighted F1 by 2.4 pp and macro F1 by
1.9 pp, and **not one of the fifteen classes gets worse**. It requires no
labels, no retraining and no new data — only the saved probabilities. It also
lands within 0.002 accuracy of the oracle version, meaning EM recovers the
Stage-2 prior essentially perfectly. This should be adopted.

*2. Correcting Stage 3 as well raises accuracy but lowers macro F1, because it
buys precision by ceasing to predict the rare classes at all.* Under
`oracle, Stages 2+3` the tile-level metrics look best of any variant — accuracy
0.767, κ 0.597 — while Mangosteen, Rambutan and Langsat fall to **exactly
zero F1**, and Mango, Longan and Jackfruit to ≤0.005. The correction is working
as designed: told that Langsat is 0.34 % of orchards, the model stops emitting
it. That removes the false positives of §6.5 and removes the true positives
with them. Prior correction therefore **fixes the precision half of §6.5's
"fails in both directions" and makes the recall half worse.** It is a better map
and a worse species classifier, and which of those is wanted is a decision for
LDD, not a modelling detail to be settled silently by an argmax.

*3. EM is reliable at Stage 2 and unreliable at Stage 3.* Its Stage-3 orchards
estimate is badly wrong — it infers 70 % Rambutan where the truth is 73 % Durian
— which is why `em, Stages 2+3` zeroes Durian (0.110 → 0.000) and destroys
Pineapple (0.347 → 0.0001) while still gaining accuracy from the Stage-2 half of
the correction. SLD-EM assumes well-calibrated probabilities and an unchanged
class-conditional density; neither holds well for seven overlapping evergreen
canopies, and many pixels routed to Stage 3 are not economic crops at all. **EM
must not be applied at Stage 3 without validation against known priors.**

**Reading the `matched` column.** Every corrected variant scores *worse* than
baseline on the `matched` population (0.514 → 0.493 for `em, Stage 2 only`).
This is expected rather than contradictory: `matched` is deliberately
re-sampled to the prior RF paper's class balance, so a correction toward the
tile's real priors is a mismatch there by construction. `matched` remains the
right population for comparing against that paper and the wrong one for judging
prior correction.

**Revised recommendation, replacing §6.4's option list.** Adopt EM prior
correction at Stage 2 (finding 1). Do not adopt it at Stage 3 by default
(findings 2 and 3); if a species map is the deliverable, the rare classes need
recall they do not currently have, and suppressing them is not the same as
classifying them. The remaining options in §6.4 — dropping `class_weight`
at Stage 3, and per-class threshold tuning against macro F1 — are untested and
now look more promising than prior correction for the species problem, because
both can trade precision for recall continuously instead of collapsing a class
to zero.

### 6.7 The correction regime, swept

§6.6 tested two fixed correction targets. Neither is a principled choice, so the
correction was reparameterised as a *strength* and swept.

The obvious form, `π_target ∝ N_c^α`, assumes the model's probabilities already
carry a uniform prior, so that α=0 is a no-op. That holds at Stage 2, fitted on
exactly 140,000 pixels per subclass, but **not** at Stage 3, whose sampled prior
is capped-and-upsampled (≈ `N_c^0.27`) and whose sigmoid calibration is fitted on
that sampled data. Under that form α=0 would already be a correction and the grid
would not contain the baseline. The geometric form fixes it:

```
p'(c|x) ∝ p(c|x) · (π_true(c) / π_train(c))^α
```

α=0 is the model as trained, α=1 the fully corrected model, and the form reduces
to the naive one wherever π_train is uniform. `π_true` is counted on the
population actually routed to each stage, recomputed per α₂ because α₂ changes
what Stage 3 receives.

256 cells were swept over α₂ (Stage-2 routing) × α₃ (Stage-3 species) from the
saved full-tile probability arrays — nothing is refitted, so a cell costs seconds.
At (0, 0) the harness reproduces the published baseline exactly: accuracy 0.6599,
κ 0.4965, flat-15 macro F1 0.226, econ-13 macro F1 0.1678.

| cell | note | econ-13 macro F1 | crops alive | accuracy | κ |
|---|---|---|---|---|---|
| (0, 0) | baseline | 0.1678 | 12 | 0.6599 | 0.4965 |
| (0.552, 0.552) | parcel-proportional exponent | 0.2145 | 12 | 0.7371 | 0.5756 |
| (0.858, 0.858) | §6.6's EM correction | 0.2195 | 9 | 0.7609 | 0.5951 |
| (1.0, 1.0) | full true prior | 0.2132 | 8 | 0.7671 | 0.5976 |
| **(0.8, 0.8)** | best overall | **0.2203** | 10 | 0.7575 | 0.5929 |
| **(0.8, 0.3)** | best keeping all 13 | **0.2129** | **13** | 0.7418 | 0.5770 |

Three results.

*1. The response surface is flat.* The Pareto frontier — best macro F1 at each
number of surviving crops — runs 13 → 0.2129, 12 → 0.2186, 11 → 0.2199,
10 → 0.2203, 9 → 0.2197, 8 → 0.2132. **Keeping every crop alive costs 0.0074
macro F1.** There is no sharp optimum to tune toward, which makes the operating
point a reporting decision rather than a tuning one, and makes the exact value of
α far less consequential than §6.6 implied.

*2. α=1 is not optimal*, despite being the theoretically correct full correction.
This is consistent with Garg et al. (2020), where prior-correction error scales as
1/σ_f: these classes are barely separable, so the correction amplifies calibration
error faster than it repairs the prior.

*3. The two axes are not interchangeable.* α₂ carries almost all of the gain;
α₃ trades macro F1 against the number of surviving species. The best all-13-alive
cell is *off-diagonal* — a strong routing correction with a weak species one.

**Published operating point: α₂ = 0.8, α₃ = 0.3.** The best cell overall scores
0.2203 against 0.2129 but reports three species as absent from a province where
LDD maps them; 0.0074 macro F1 does not buy that. `apply_operating_point.py`
writes this point out as prediction arrays, flat-15 tables and the full confusion
set, under its own tag, leaving the uncorrected baseline intact.

| population | accuracy | κ | F1 weighted | F1 macro |
|---|---|---|---|---|
| baseline, held_out | 0.660 | 0.497 | 0.714 | 0.226 |
| **corrected, held_out** | **0.7418** | **0.5770** | **0.7616** | **0.2573** |

Per-class F1 on `held_out`, baseline → corrected. **No class regresses.**

| class | baseline | corrected | Δ |
|---|---|---|---|
| Para rubber | 0.7595 | 0.8356 | +0.0761 |
| Oil palm | 0.1244 | 0.3026 | +0.1782 |
| Durian | 0.1099 | 0.1846 | +0.0747 |
| Rice | 0.2518 | 0.3097 | +0.0579 |
| Pineapple | 0.3469 | 0.3739 | +0.0270 |
| Cassava | 0.2877 | 0.3058 | +0.0181 |
| Jackfruit | 0.0065 | 0.0171 | +0.0106 |
| Mangosteen | 0.0032 | 0.0082 | +0.0050 |
| Coconut | 0.0005 | 0.0078 | +0.0073 |
| Longan | 0.0038 | 0.0077 | +0.0039 |
| Langsat | 0.0012 | 0.0032 | +0.0020 |
| Mango | 0.0045 | 0.0070 | +0.0025 |
| Rambutan | 0.0008 | 0.0010 | +0.0002 |
| Reservoir | 0.7504 | 0.7504 | 0.0000 |
| Others | 0.7446 | 0.7446 | 0.0000 |

Read backwards, the correction does what §6.5 said was needed: the share of
predicted oil palm that is truly rubber falls from 82.6 % to 49.6 %, jackfruit
from 64.6 % to 40.8 %, mangosteen from 51.2 % to 24.6 %. **But the rare orchards
do not come back.** Rambutan, Mango, Longan, Mangosteen and Langsat remain below
F1 0.02 at *every* one of the 256 cells. Prior collapse and species
indistinguishability are separate failures and only the first is post-hoc
fixable — §6.9.

**Split-half validation of the operating point.** The α above was chosen by ranking cells
on `held_out` and is reported on `held_out` — selection on the evaluation set, which
biases the corrected figures optimistically. The bias ought to be negligible (two scalar
parameters against 21 M pixels, on a surface measured to be flat), but that is a
prediction, so it was measured. `held_out` was split at random into a selection half A
and a reporting half B, 10,590,029 pixels each; α was chosen on A by the identical rule,
and scored on B, which was never consulted during selection.

| scored on B (10,590,029 px) | accuracy | κ | F1 weighted | F1 macro | econ-13 macro F1 |
|---|---|---|---|---|---|
| baseline (0, 0) | 0.6596 | 0.4962 | 0.7139 | 0.2263 | 0.1678 |
| **selected on A** — (0.7, 0.3) | 0.7367 | 0.5733 | 0.7595 | 0.2564 | **0.2127** |
| oracle on B — (0.8, 0.3) | 0.7416 | 0.5767 | 0.7614 | 0.2574 | **0.2131** |

**Selection bias = 0.0004 macro F1.** The honest, unbiased figure (0.2127) and the figure
selection could have reached had it cheated (0.2131) are indistinguishable. The published
corrected numbers therefore stand as reported: 0.7418 / 0.5770 on the full `held_out`
against 0.7416 / 0.5767 on the untouched half.

Two incidental confirmations. Selection on A chose **(0.7, 0.3)**, a *different* cell from
the full-data winner (0.8, 0.3), and the two score within 0.0004 of each other on B —
independent evidence for the flat response surface, arriving from a different direction
than the Pareto frontier. And the baseline reproduces to four decimals on the half
(0.6596 / 0.4962 against 0.6599 / 0.4965), confirming the split is not itself skewed.

The operating point is kept at (0.8, 0.3): it is selected against the tile's real class
distribution, which is what the deliverable is scored on, and the measured bias is smaller
than the difference between the two candidate cells.

**A note on α = 0.552.** This exponent was derived from a Kish design-effect
argument: it is the power at which pixel-proportional allocation becomes
parcel-proportional, given that pixels within a parcel are near-perfectly
correlated. It constrains **sampling allocation**, not the decision-rule prior,
and placing it on this axis conflates the two. That it scores 0.2145 here is
coincidence and should not be reported as confirmation. Testing it properly
requires retraining under the allocation, not reweighting after the fact.

### 6.8 Stage 1's prior, and a rule that was never applied

The sweep held Stage 1 fixed because its documented decision rule is not argmax
but a sequential threshold overwrite — argmax, then econ where p > 0.40, water
where p > 0.48, forest where p > 0.50 — and prior correction rescales the
probabilities those thresholds were tuned against.

**That rule is not the one in use.** `argmax` of the saved probabilities, with
`classes_ = [1,2,3,4]`, reproduces the saved full-tile predictions on all
24,323,769 rows with **zero disagreements**. `stage1_weight_scale.py` tunes the
thresholds, writes them to JSON and scores its own test split through them, but
the full-tile predictions every number in this document rests on come from
`chunked_predict_and_save(...)` → `predict()`. The thresholds never reach the tile.

This is worth stating plainly because the method section would otherwise describe
a step that has no effect on any published result. Two further consequences:
applying the thresholds would have *hurt* (macro F1 0.7490 against 0.7529 for
argmax — it buys econ recall by giving up `others`), and the econ threshold had
pinned to the floor of its own search grid, so it was a constraint rather than an
optimum. Stage 1 being an argmax stage also means prior correction composes with
it cleanly, and it can be swept like the others.

Sweeping α₁ on `held_out`, every superclass improves at every step, monotonically
to α₁ = 1.0:

| α₁ | econ recall | forest F1 | water F1 | others F1 | macro F1 | κ |
|---|---|---|---|---|---|---|
| 0.0 (deployed) | 0.8529 | 0.7058 | 0.7504 | 0.6858 | 0.7529 | 0.6403 |
| 0.552 | 0.8799 | 0.7384 | 0.8096 | 0.6938 | 0.7803 | 0.6591 |
| 0.8 | 0.8901 | 0.7490 | 0.8244 | 0.6951 | 0.7877 | 0.6639 |
| **1.0** | **0.8978** | **0.7555** | **0.8314** | **0.6954** | **0.7916** | **0.6661** |

**Stages 2 and 3 have an interior optimum near α = 0.8; Stage 1 does not.** The
difference is separability: four physically distinct superclasses leave σ_f large,
so by the same 1/σ_f argument the correction costs almost no calibration error.
The practical statement for the paper is therefore not "use α = 0.8" but **the
optimal correction strength tracks class separability, and must be set per stage.**

The gain is ordered by rarity — per-crop Stage-1 econ recall, which is the ceiling
on everything stages 2 and 3 can still get right:

| crop | α₁ = 0 | α₁ = 1 | Δ |
|---|---|---|---|
| Langsat | 0.3491 | 0.5927 | **+0.244** |
| Rambutan | 0.3339 | 0.5396 | **+0.206** |
| Longan | 0.4038 | 0.5515 | +0.148 |
| Mangosteen | 0.2109 | 0.3396 | +0.129 |
| Oil palm | 0.7353 | 0.8606 | +0.125 |
| Mango | 0.2552 | 0.3708 | +0.116 |
| Jackfruit | 0.6241 | 0.7330 | +0.109 |
| Rice | 0.5900 | 0.6744 | +0.084 |
| Para rubber | 0.8648 | 0.9072 | +0.042 |

The three species §6.5 named as Stage-1 casualties gain most; rubber, the class
doing the swamping, gains least. The correction acts exactly where the diagnosis
located the damage.

α₁ is **not** included in the published operating point. Correcting Stage 1
enlarges the set of pixels routed to Stage 2, and those pixels have no saved
Stage-2/3 probabilities, so unlike α₂/α₃ this is not a free post-hoc composition.
It belongs in the next arm trained from scratch.

### 6.9 The acquisition window, not the classifier

Every result above treats the feature set as fixed. It should not have been.

Five Sentinel-2 composites exist for 2018 on tile 47PQQ: **2018-03-31, 2018-04-30**,
2018-10-31, 2018-11-30, 2018-12-31. This study uses the last three. The Oct–Dec
window was inherited because it is the window of the prior RF paper and of the
collaborator's XGBoost cascade, which makes it the right choice for §7's
controlled comparison — and, it turns out, a poor choice for the classification
itself. October to December is the tail of the monsoon, when rubber, oil palm and
every evergreen orchard canopy are simultaneously at maximum greenness. That is
the window in which the model is asked to tell them apart, and §6.5 measured what
happens: it cannot, so it predicts rubber.

Tested directly on a balanced 13-crop subsample (96,228 pixels, 50/50 split), same
Nystroem → LinearSVC → OvR family as the real stages, equal samples per crop so the
prior is uniform in every arm and only the features differ:

| arm | features | macro F1 | accuracy |
|---|---|---|---|
| wet3 — the window used here | 24 | 0.5025 | 0.5072 |
| dry2 — the two unused dates alone | 16 | 0.3981 | 0.4094 |
| **all5** | **40** | **0.5355** | **0.5371** |

The gain holds at seeds 42 / 7 / 2026 (+0.033, +0.031, +0.036) and *widens* to
+0.063 when the kernel is given more capacity (Nystroem 800 components), so it is
capacity-limited rather than noise. It concentrates in precisely the species the
cascade currently loses: Langsat +0.069, Mangosteen +0.062, Longan +0.059,
Jackfruit +0.047, Rambutan +0.046 — against +0.001 for rubber, which needs no help.

**A bottleneck that hides the effect.** Stage 1 — and only Stage 1; stages 2 and 3
go scaler → kernel directly — applies `PCA(n_components=10)` before the kernel
approximation. Repeating the experiment with PCA inserted where the real stage has
it:

| PCA components | wet3 (24 feat.) | all5 (40 feat.) |
|---|---|---|
| 10 (as deployed) | 0.4652 | **0.4413** |
| 14 | 0.4949 | 0.4718 |
| 20 | 0.5026 | 0.5112 |
| none | 0.5025 | **0.5355** |

**At 10 components the 5-date arm scores below the 3-date arm.** The two extra
dates are projected away before the classifier sees them, and the effect only
appears once at least 20 components survive. Retraining on the inherited
configuration would have produced evidence that the dry season does not help.

PCA is also costing the *current* arm 0.037 macro F1 for nothing — 24 features is
not a dimensionality problem — and it puts the kernel scale off: the components of
standardised data have variance equal to their eigenvalues, so `gamma = None`
(1/10) meets inputs whose spread is nothing like unit. That is the same failure
mode as the gamma miscalibration corrected in the DEM+S1+S2 arm.

Together these two are worth about **+0.070 macro F1 (+15 % relative)** on this
balanced probe, from data already on disk and one configuration constant — larger
than everything post-hoc prior correction can buy.

**A second bottleneck, larger than the first.** Stages 2 and 3 — the stages that
actually separate species — search Nystroem `n_components` over `[50, 100, 150]`.
That ceiling was never justified against a measurement. Repeating the balanced
13-crop experiment across the capacity axis:

| Nystroem components | wet3 (24 feat.) | all5 (40 feat.) |
|---|---|---|
| 50 | 0.3971 | 0.3827 |
| 100 | 0.4523 | 0.4583 |
| **150 (deployed ceiling)** | **0.4742** | **0.4876** |
| 300 | 0.5025 | 0.5355 |
| 600 | 0.5180 | 0.5708 |
| 1000 | 0.5259 | **0.5927** |

The curve is still climbing steeply at the ceiling, and the ceiling **throttles the
features as well as the classifier**: at 150 components the 5-date set beats the
3-date set by only 0.013, at 1000 by 0.067. Judged at the deployed capacity, the
extra dates would look nearly worthless. They are not; there is simply no capacity
to represent them.

Combining the three findings, on this probe: 0.4742 at the deployed configuration
(3 dates, 150 components) against 0.5927 with all five dates and 1000 components —
**+0.119 macro F1, +25 % relative**, with no change to the labels, the architecture
family, the splits, or the sampling scheme.

The `s2_2018_5date` arm therefore changes three things together, each measured
separately first: five dates instead of three, no PCA in Stage 1, and a raised
kernel ceiling in stages 2 and 3 (`[300, 600]`). Stage 1 keeps `[150, 250]` — it
fits 2.03 M rows, where 600 components would not fit in memory, and its four
superclasses are the part of the problem that already works.

**Both arms measured, and a correction to how they were first scored.** `s2_2018_5date`
finished 2026-08-21 16:23 and `s2_2018_3date_v2` 2026-08-22 10:42. The first version of this
section compared arms on the 3-date arm's `held_out` set, reasoning that `y` is
byte-identical between arms so the pixels correspond. The pixels correspond; the *splits do
not*, and they cannot.

**Arms never share a split, even on the same NPZ with the same seed.** Stages 2 and 3 train
on the pixels *Stage 1 routed to econ*, so the moment Stage 1's predictions differ, the
sampled rows differ too. `s2_2018_3date` and `s2_2018_3date_v2` differ only in the PCA
removal and still diverge by ~506,000 rows in each direction; the 5-date arm diverges by
514,174. Because those rows are selected *conditional on correct routing*, an arm scores
Stage-1 econ recall of exactly 1.000 on its own, by construction. Scoring one arm on
another's `held_out` therefore inflates it — and inflates it most for the rare crops, which
lose the largest share of themselves to sampling (50 % of Jackfruit, 42 % of Longan, 31 % of
Mangosteen).

Everything below is therefore scored on the **intersection**: the 20,244,314 pixels held out
by all three arms.

| Stage 1, common held-out (20,244,314 px) | econ F1 | forest F1 | water F1 | others F1 | macro F1 | accuracy | κ |
|---|---|---|---|---|---|---|---|
| 3-date, PCA(10) — published arm | 0.8625 | 0.7083 | 0.7508 | 0.6887 | 0.7526 | 0.8000 | 0.6386 |
| 3-date, no PCA (`v2`) | 0.8668 | 0.7175 | **0.7556** | 0.6989 | 0.7597 | 0.8064 | 0.6494 |
| 5-date, no PCA | **0.8739** | **0.7235** | 0.7537 | **0.7160** | **0.7668** | **0.8155** | **0.6654** |

The aggregate result is robust: it is unchanged to three decimals across all three choices
of population, so nothing about it depends on the correction above. PCA removal is worth
+0.0071 macro F1, the two extra dates a further +0.0071 — the two defects cost almost
exactly the same, and Stage 2 will reverse their order.

**The per-crop version of this table is withdrawn, and cannot be rebuilt.** Earlier drafts
reported large per-crop swings in Stage-1 econ recall in both directions; both were
artifacts, and the design admits no correct version. Each arm's own `held_out` is inflated
by conditional sampling. The intersection has the opposite defect, and it is one this
document already names in §5 for the "never sampled" population: the per-LU caps absorb
almost every correctly-routed pixel of a rare class, so what survives in the intersection is
the misrouted residue, which scores near zero *by construction*. It also leaves too little
to measure — Langsat retains 284 pixels, Rambutan 1,130, and these are drawn from 27 and
~90 parcels respectively, so the Kish design effect of §6.7 puts the effective sample size
in the single digits. A per-crop difference of ±0.05 there is noise. Cross-arm claims in
this section are aggregate claims only; per-crop evidence for the rare species comes from
the balanced probes of §6.10, which are not subject to either defect.

The search selected `n_components = 250`, the top of Stage 1's candidate range, in both
arms, so Stage 1 remains capacity-limited even after the PCA removal; its ceiling was left
at `[150, 250]` deliberately because it fits 2.03 M rows. This gain is also independent of,
and composable with, the α₁ correction of §6.8, which was measured on the 3-date arm.

**The whole cascade, end to end.** The 5-date arm finished 2026-08-21 15:45 and `v2`
2026-08-22 10:42. In **both** arms, Stage 2 and all three Stage-3 groups selected
`n_components = 600` at `C = 10` — the top of the widened range — as Stage 1 selected 250,
the top of its own. Eight independent searches, eight ceiling hits: capacity is still
binding after the raise from 150, so the +0.119 attributed to the three defects is a lower
bound rather than a converged gain, and a future grid should extend past 600.

On the balanced Stage-3 test splits the effect is large, and concentrated exactly where
§6.10 predicted:

| balanced test split, macro F1 | 3-date | `v2` (capacity only) | 5-date + capacity |
|---|---|---|---|
| Stage 2 (3 subclasses) | 0.8324 | 0.8517 | **0.8598** |
| Stage 3, orchards | 0.4050 | 0.5002 | **0.5981** |
| Stage 3, plantation | 0.6610 | 0.7459 | **0.8736** |
| Stage 3, field | 0.7850 | 0.8055 | **0.8099** |

The `v2` column isolates the kernel ceiling, since it shares the 3-date window. **The two
defects change rank down the cascade.** At Stage 1 they cost the same (+0.0071 each); at
Stage 2 the ceiling is much the larger term (+0.0193 against +0.0080 for the dates); by
Stage 3 they are comparable again on orchards (+0.095, +0.098) with the dates dominating on
plantation (+0.085, +0.128). A single headline number for "the window" or "the capacity"
would hide this — the dates buy separability the deeper stages need, the capacity buys the
ability to use it, and neither substitutes for the other.

Coconut moves 0.237 → 0.786, Langsat 0.235 → 0.619, Jackfruit 0.336 → 0.578, Mangosteen
0.258 → 0.449. Rubber-versus-oil-palm, the confusion §6.7 measured as the cascade's
dominant error mode, reaches 0.919 / 0.916. These are balanced-split numbers and do not
survive contact with the tile's real 3,199× imbalance; the end-to-end figures below are
what the deliverable is scored on, and they are much lower.

| uncorrected, `held_out` (21.2 M px) | 3-date | 5-date + capacity |
|---|---|---|
| accuracy | 0.6599 | **0.6979** |
| κ | 0.4965 | **0.5416** |
| F1 weighted | 0.7141 | **0.7451** |
| F1 macro, 15-class | 0.2264 | **0.2393** |
| F1 macro, econ-13 | 0.1462 | **0.1596** |

The same direction holds on all three populations — `all` econ-13 macro 0.2029 → 0.2335,
`matched` 0.2531 → 0.2771 — and **no class regresses on any population**, the single
exception being Reservoir on `matched` at −0.0015, which is noise. The routing losses of
§6.5 both shrink: pixels dropped at Stage 1 fall from 2,166,284 to 2,022,839 and pixels
misrouted at Stage 2 from 2,665,828 to 2,289,492.

**Scored on common ground, both arms hold — and `v2` is the one the paper can use.**
Re-scored on the 20,244,314 pixels held out by all three arms:

| uncorrected, common held-out | 3-date (published) | 3-date `v2` | 5-date |
|---|---|---|---|
| accuracy | 0.6601 | 0.6823 | **0.6986** |
| κ | 0.4885 | 0.5140 | **0.5349** |
| F1 weighted | 0.7207 | 0.7380 | **0.7521** |
| F1 macro, 15-class | 0.1990 | 0.2069 | **0.2111** |
| F1 macro, econ-13 | 0.1144 | 0.1226 | **0.1267** |

Monotone on every metric, and **`v2` captures 55 % of the total κ gain (+0.0255 of +0.0464)
without touching the acquisition window**. That matters for §7: `v2` uses the same three
dates as the collaborator's XGBoost cascade and the prior RF paper, so it stays inside the
controlled comparison, and it is the arm to put in the head-to-head table. The 5-date arm is
the ceiling estimate, reported alongside.

Per class, rubber runs 0.7569 → 0.7814 → 0.7973, `others` 0.7476 → 0.7540 → 0.7664,
Pineapple 0.2760 → 0.3028 → 0.3193, Cassava 0.2255 → 0.2404 → 0.2660. The rare orchards sit
at ~0.000 for **all three arms** on this population, for the construction reason given above,
not because the arms differ — which is precisely why the rare-species question is settled in
§6.10 and not here.

Both arms score lower here than on their own splits, because the intersection is the harder
residue with every arm's sampled rows removed. The per-arm tables above remain each arm's
own honest self-report; this table is the one to cite for any comparison *between* arms.

**Two results that should not be conflated.** The 5-date arm here is *uncorrected*, and on
`held_out` it does not reach the corrected 3-date operating point of §6.7 (0.7418 / 0.5770
/ 0.7616). Post-hoc prior correction is still worth more on these aggregate metrics than
the window and capacity fixes are. But the two act on different failures — correction
repairs a decision rule, capacity repairs a fit — so they should compose, and the 5-date
arm has not yet had its own sweep. Its separability changed, so the α optima measured in
§6.7 cannot be assumed to carry over and no operating point is quoted for this arm until
they are re-measured.

Against the prior RF paper on `matched`, the gap narrows but does not close: 0.5388 / 0.4798
/ 0.5472 against 0.7160 / 0.6780 / 0.7140, roughly 18 points of accuracy behind where it
was 20. The §6.3 caveats on that comparison are unchanged.

Per-stage and per-crop confusion matrices for this arm are in
`runs/s2_2018_5date/confusion/`.

### 6.10 Are the rare species separable at all?

§6.5 and §6.7 leave seven of the thirteen crops below F1 0.02 end-to-end, and five of
them stay there at **every one of the 256 swept cells**. The natural reading is that
Sentinel-2 simply cannot distinguish these evergreen canopies, and that reading was
supported by three independent arguments: parcel counts (Langsat has 27 parcels
province-wide against rubber's 5,571), theory (Garg et al.'s consistency conditions
fail for overlapping class-conditional score distributions), and measurement
(Mangosteen loses 79 % of its pixels at Stage 1, before any prior applies).

All three are arguments about **deployment conditions**, not about information content.
Parcel count constrains estimator variance; the 1/σ_f bound constrains the prior
correction; the Stage-1 loss is measured under a rebalanced prior at a throttled
capacity in the narrower date window. None isolates the features.

Isolating them — balanced priors, all five dates, 800 kernel components, no cascade —
gives a different answer:

| crop | end-to-end F1 (§6.7) | balanced probe, 3 dates | balanced probe, **5 dates** |
|---|---|---|---|
| Langsat | 0.0032 | 0.5612 | **0.6774** |
| Coconut | 0.0078 | 0.5379 | **0.6329** |
| Mango | 0.0070 | 0.5502 | **0.6150** |
| Rambutan | 0.0010 | 0.5204 | **0.6092** |
| Longan | 0.0077 | 0.4593 | **0.5543** |
| Mangosteen | 0.0082 | 0.4345 | **0.5385** |
| Jackfruit | 0.0171 | 0.3739 | **0.4540** |

Langsat — the species with the fewest parcels, and the strongest candidate for "the data
cannot support it" — is the **best performing** of the rare orchards under balanced
conditions, ahead of Durian, which has 353,768 pixels. Parcel count predicts estimator
variance, not class overlap, and this is the counterexample that separates the two.

**These numbers are not achievable end-to-end and must not be quoted as such.** The probe
scores under a uniform prior on a balanced test half; the tile carries a 3,199× pixel
imbalance, and at real priors precision on a rare class is bounded by that imbalance
however good the decision boundary is. Lifting Rambutan from 0.001 to 0.6 is not on offer.

The defensible claim is narrower and still decisive: **the ceiling on these species is
not spectral.** What limits them is a chain of pipeline choices — the acquisition window,
the PCA and kernel-capacity ceilings, and the uncorrected class prior — every one of
which is a property of this implementation rather than of Sentinel-2. The rare-species
problem is therefore an engineering constraint, not a capability limit, and it is
reported in §6.9 rather than in §8.

This does not disturb §6.1. S2-only orchard macro F1 remains indistinguishable from
DEM+S1 and well below the fused stack, so structure still helps; "not information-limited"
is not "as good as it can get". That comparison was run at a common configuration, so the
three defects of §6.9 apply equally to both sides of it and it stands as measured.

### 6.11 Does it transfer to another year?

The cascade is trained on 2018 imagery and the 2018 survey. Whether it is a *spectral*
classifier that would work on a new acquisition, or a model that has effectively memorised
where the parcels are, is not answerable from any 2018 score — the pixels are the same
ground either way. It is answerable by running it on 2024.

Tile 47PQQ was reacquired at 2024-10-31, -11-30 and -12-31 — the same three dates as the
published window — and LDD resurveyed Rayong as `LU_RYG_2567`. Nothing is refitted: the
`s2_2018_3date_v2` models are loaded and applied. Only the three-date arms can be tested
this way, since no dry-season imagery exists for 2024, which is a further argument for `v2`
as the deliverable.

| 2024, models from `s2_2018_3date_v2` | pixels | accuracy | κ | F1 weighted |
|---|---|---|---|---|
| all labelled 2024 pixels | 21,521,969 | 0.6163 | 0.4676 | 0.6481 |
| also held out in 2018 | 18,087,716 | 0.6171 | 0.4501 | 0.6552 |

**A raw transfer score cannot be read as model error**, because three things are mixed into
it: the model's ability to transfer, genuine land-use change over six years, and differences
between two LDD survey rounds. The second is measurable. Rasterising both surveys onto the
same grid, **9.55 % of pixels labelled in both changed class** — and unevenly, exactly as
agronomy predicts: the rotational field crops churn (Jackfruit 35.9 %, Cassava 34.9 %,
Longan 27.4 %, Pineapple 27.0 %, Mango 26.9 %) while the perennial stands do not (Durian
3.5 %, Mangosteen 4.3 %, Rambutan 6.5 %, rubber 10.0 %).

Scoring the two apart separates transfer from change:

| 2024 | pixels | accuracy | κ | F1 weighted |
|---|---|---|---|---|
| unchanged since 2018 | 18,977,865 | 0.6356 | 0.4783 | 0.6713 |
| changed since 2018 | 2,002,968 | 0.4536 | 0.3073 | 0.4872 |

**But those rows still cannot be compared with the 2018 figures**, for the reason finding 11
gives: each epoch is scored on its own survey's pixels, and the class mixes differ — rubber
is 49 % of the 2020 population against 45.5 % of the 2024 one. An accuracy computed over a
different class mix is a different quantity. Every epoch is therefore re-scored on **one
identical population**: the 16,102,201 pixels labelled in all three surveys, carrying the
same class in all three, and never fitted on by the `v2` arm. `compare_epochs.py` builds it.

| same 16,102,201 px, `v2` models | accuracy | κ | F1 weighted | F1 macro |
|---|---|---|---|---|
| **2018** (in-epoch) | **0.7354** | **0.5946** | **0.7753** | 0.2585 |
| **2020** (+2 years) | 0.6687 | 0.4871 | 0.7082 | 0.1885 |
| **2024** (+6 years) | 0.6367 | 0.4601 | 0.6808 | 0.2213 |

**Transfer costs 0.067 accuracy and 0.108 κ at two years, and 0.099 and 0.135 at six.**
Accuracy and κ decay monotonically with elapsed time, which is the expected shape and a
check that the two epochs were prepared consistently.

An earlier draft of this section reported the 2024 cost as 0.047 accuracy and 0.036 κ, by
comparing the 2024 unchanged-ground row against the 2018 arm's own held-out figures. Those
are different populations, so the comparison understated the cost; the controlled numbers
above supersede it. This is the fourth time in this study that comparing across
independently-defined populations produced a wrong answer, which is why finding 11 is stated
as a rule rather than an anecdote.

**Macro F1 does not decay monotonically, and the reason is class-specific.** Rubber does
decay in order (0.8231 → 0.7632 → 0.6966), as do Reservoir and `others`. The annual field
crops instead *dip in 2020 and recover in 2024*: Pineapple 0.4650 → 0.1032 → 0.3321, Cassava
0.3544 → 0.2039 → 0.3191, Oil palm 0.1660 → 0.0398 → 0.1344. Something specific to the 2020
acquisition hurts the annuals while leaving the perennials on trend. Two candidate
explanations were tested and **both failed**: the global covariate shift from 2018 is
essentially the same for both epochs (mean standardised shift 0.083 for 2020 against 0.060
for 2024, medians 0.049 and 0.056), and the per-crop NDVI drift on unchanged parcels is
*larger* in 2024 for exactly the crops that do better there (Cassava 0.044 against 0.031,
Pineapple 0.048 against 0.046). The effect is real and reproducible but unexplained; it is
recorded here rather than attributed to a mechanism the data does not support.

**The model reads the imagery; it has not memorised the parcels.** On the 1,715,723 changed
pixels the taxonomy actually separates, the cascade reports the **2024** class 39.9 % of the
time and the **2018** class only 11.7 % — a 3.4:1 ratio in favour of what the ground became.
Per crop, of parcels that *became* rubber by 2024 it calls 50.1 % rubber against 4.9 % their
old class; for durian, 51.8 % against 5.1 %. Rubber is the decisive case: it is the one
class the cascade does not over-predict (precision 0.952), so its 10:1 ratio cannot be
explained by promiscuity, which is the obvious alternative reading of durian's.

This is the answer to the reviewer's question that no in-epoch result can give. The
degradation on changed ground (0.4536) is therefore not a failure — a substantial part of it
is the model correctly reporting a class the 2024 survey has moved away from, or moving with
it and being scored against a survey round that disagrees.

Two caveats. `all` and `unseen_2018` differ by only 0.0008 accuracy but 0.0175 κ, so the
2018 fitted rows flatter the agreement statistic more than the raw score; the strict
population is quoted throughout. And the 2024 survey is not a re-survey of the same parcel
geometry — it has 65,136 polygons against 2018's 41,571 — so some "change" is resurvey
resolution rather than land-use change, which biases the changed population toward the model
and is a reason to lean on the unchanged row.

## 7. Comparison notes for the joint paper

**What is controlled between this study and the XGBoost cascade:** epoch (2018),
tile, label survey and erosion, date window (Oct–Dec), sensor.

> The date window is controlled *and* suboptimal (§6.9). The `s2_2018_5date` arm
> deliberately breaks this axis, so it is reported **alongside** the 3-date arm
> rather than replacing it: the 5-date arm quantifies what the shared window costs
> both studies, and presenting its figures in the head-to-head table would
> silently un-control the comparison.
>
> **Use `s2_2018_3date_v2` in the head-to-head table, not the original 3-date arm.**
> It holds the window, the tile, the survey and the split protocol fixed, and fixes
> only two things internal to this study's own configuration — a PCA(10) bottleneck
> and a kernel ceiling, neither of which the XGBoost cascade shares or is affected
> by. It carries 55 % of the total κ gain (κ 0.4885 → 0.5140 of 0.5349). Reporting
> the un-fixed arm would understate this side of the comparison for a reason that
> has nothing to do with SVM-versus-XGBoost, which is the axis the paper is about.

**What is not controlled, and must be stated rather than glossed.** The items below
were verified against the collaborator's repository (`github.com/Gunkartan/geospatial`)
on 2026-08-23; earlier revisions of this section described the XGBoost study from
second-hand notes and were wrong on points 1, 2 and 3.

1. *Feature sets differ.* The XGBoost study uses five quantities per date — NDVI, EVI,
   NDWI, MTCI and the raw long-SWIR band — over Oct/Nov/Dec, for 15 columns
   (`extract_crops.py`). This study uses eight indices over the same three dates, for
   24 columns. Shared by name: NDVI, EVI, NDWI. Only there: MTCI (a red-edge index) and
   raw long-SWIR. Only here: BSI, NDBI, MSAVI, SWIR_NIR, SWIR_RATIO.
   *NDWI is shared in name only*: theirs is (green − SWIR)/(green + SWIR), the MNDWI
   form; ours is (green − NIR)/(green + NIR).
2. *Texture.* Neither study uses texture. Both are strictly per-pixel. (A previous
   revision of this section stated that the XGBoost study appends 3×3 local mean and
   variance per feature; it does not.)
3. *Model shape differs.* This study is a three-stage cascade. The XGBoost study runs
   three independent detectors — water, buildings, crops — of which the crop model is a
   **flat 14-class classifier** (the 13 crops plus an `others` class), not a stage of a
   cascade. Only the final crop map is comparable.
4. *Splits.* Both studies split at pixel level, so **both carry parcel leakage**: pixels
   of one parcel fall on both sides of the split and are near-duplicates. Theirs is a
   stratified 60/20/20 at `random_state=42`, with metrics reported on the 20 % CV
   partition (the test partition is created and not evaluated). Ours is stated in §5.
   The leakage is quantified for this study in §5 and is a shared limitation of the
   joint paper rather than an asymmetry between the two arms.
5. *Class-prior handling, and the evaluation population.* Both studies rebalance by
   capping: the XGBoost study samples up to 200,000 pixels per class after a 3-pixel
   erosion, with no class weighting and no upsampling; this study caps per LU code and
   additionally upsamples and applies `class_weight='balanced'` at every stage (§6.4).
   The more consequential difference is **what each set of numbers is computed over**.
   The XGBoost metrics are computed on a partition of that capped sample, where the
   most and least common crops stand at roughly 200,000 against a few thousand pixels.
   This study's end-to-end metrics are computed over the natural tile population, where
   rubber is 81 % of economic pixels and Langsat is 0.012 %. Rare-class precision is
   largely a function of prevalence, so the two sets of figures are not comparable as
   they stand, and the gap between them is not evidence about XGBoost versus SVM.
   Agreeing one evaluation population is the prerequisite for any joint table.
   (Note also that the sampling in `extract_crops.py` uses an unseeded
   `np.random.choice`, so their sample is not exactly reproducible; the seed at
   `train_crops.py` governs only the later split.)

**Consequence.** The honest framing of the joint result is a comparison of two
complete pipelines, not of two algorithms. Presenting it as "SVM vs XGBoost"
without the qualifications above would be over-claiming.

## 8. Limitations

- **Class priors are not calibrated** in the baseline (§6.4, §6.5). §6.7 now
  supplies a corrected operating point that fixes the routing half of this at no
  cost to any class. The rare-species half is *not* fixed by prior correction at
  any strength — five species stay below F1 0.02 across all 256 swept cells — but
  §6.10 shows this is an engineering constraint rather than a capability limit,
  so it is not listed here as a limit of the sensor.
- **The acquisition window is a stronger constraint than the classifier** (§6.9)
  and two available dates are unused. The 5-date arm addresses this but breaks
  the §7 comparison axis, so both arms have to be carried.
- **A tuned component of Stage 1 has no effect on any published number** (§6.8):
  the per-class thresholds are computed and saved but never applied to the tile.
  Nothing here is invalid because of it — the deployed rule is argmax and is
  scored as such — but the method description had to be corrected to match.
- α₁ is measured but not composed end-to-end (§6.8), because correcting Stage 1
  enlarges the econ set beyond the pixels for which Stage-2/3 probabilities were
  saved. Its compounding benefit is therefore *not* included in any figure here.
- Pixel-level splits (see §5); parcel-level validation outstanding.
- Labels are the 2018 survey and imagery is 2018, so the epochs agree — but
  survey error and land-use change within the year are still baked in.
- Stage 2 and 3 metrics are conditional on upstream routing.
- **The label raster cannot be regenerated from the repository.** `label/label_47PQQ.tif`
  is uint16 and holds the 109 pure four-digit `LU_ID_L3` codes plus `32767`. The survey also
  contains 73 nine-digit compound codes of the form `<code4>1<code4>` marking mixed-crop
  parcels — `220412302` is cassava sharing a parcel with rubber — and every one of them was
  mapped to `32767`, the sentinel `align_indices_labels.py` drops. Mixed parcels are
  therefore excluded from training, which is defensible and was undocumented.
  `rasterize_parcel.py` does not implement that rule: it passes raw values at `int16` and
  raises on the cast, and writes int16 rather than uint16. The committed label raster came
  from an unrecorded path, so §9's command list does not fully reproduce the inputs. This
  was found on 2026-08-22 when the cross-year script hit the identical cast on the 2024
  survey; `prepare_epoch.py` reproduces the convention explicitly. `rasterize_parcel.py` has
  deliberately **not** been changed — re-running it would overwrite the raster every trained
  model depends on.
- Cross-year transfer is **done for both epochs** (§6.11). On one controlled population the
  2018 models lose 0.067 accuracy / 0.108 κ at two years and 0.099 / 0.135 at six, and the
  model demonstrably tracks change rather than parroting the 2018 labels. Only the three-date
  arms are testable this way — no dry-season imagery exists for 2020 or 2024 — which is a
  further argument for `s2_2018_3date_v2` as the deliverable. Two residual unknowns: the 2024
  survey has 65,136 polygons against 2018's 41,571, so part of the measured churn is resurvey
  resolution rather than land-use change; and the 2020 dip in the annual field crops is
  unexplained, with the two obvious explanations tested and rejected.

## 9. Reproducing

```
python make_subset_npz.py s2_3date      # 24-column feature matrix
python stage1_weight_scale.py           # Stage 1
python stage2_weighted.py               # Stage 2
python stage3_new_weight.py             # Stage 3
python evaluate_end_to_end.py           # compose cascade, end-to-end scores
python reconstruct_sampled_rows.py      # recover the unseen population
python evaluate_flat_15class.py         # flat 15-class tables
python diagnose_error_budget.py         # the error anatomy of §6.5

python save_stage23_probs.py            # full-tile Stage-2/3 probabilities
python save_stage3_probs_all_econ.py    #   ... every Stage-3 model on every econ pixel
python apply_prior_correction.py        # the prior-correction test of §6.6

python sweep_prior_alpha.py             # the 256-cell alpha sweep of §6.7
python apply_operating_point.py 0.8 0.3 # write + score the published operating point
python probe_stage1_prior.py            # Stage 1's prior and threshold rule, §6.8
python probe_dry_season.py              # the acquisition-window experiment, §6.9
python probe_pca_bottleneck.py          # the PCA(10) bottleneck, §6.9
```

The 5-date arm, end to end:

```
python make_subset_npz.py s2_only                             # 40-column matrix
ARM=s2_2018_5date USE_PCA=0 RUN_STAGE1=1 bash run_chain.sh    # all stages + scoring
```

The 3-date arm with the two defects removed, and the cross-arm comparison:

```
ARM=s2_2018_3date_v2 USE_PCA=0 RUN_STAGE1=1 bash run_chain.sh
python compare_stage1_arms.py s2_2018_3date s2_2018_3date_v2 s2_2018_5date
```

`compare_stage1_arms.py` scores every arm named on the intersection of their held-out
sets. Arms do not share a split, so passing one arm's `trainval_rows_mask.npy` as the
reference for another inflates it — see the correction in §6.9.

Cross-year transfer (§6.11), inference only:

```
python prepare_epoch.py 2024              # indices, labels on the 2018 grid, aligned NPZ
python predict_new_epoch.py 2024 s2_2018_3date_v2
python score_by_churn.py 2024 s2_2018_3date_v2   # split the score by whether ground moved
python compare_epochs.py                  # 2018/2020/2024 on one controlled population
```

`compare_epochs.py` is the one to cite: the other three score each epoch on its own survey's
pixels, whose class mixes differ, and that comparison is not valid — §6.11.

The arm is selected by the `ARM` environment variable, which `config.py` resolves
to its `TAG` and `NPZ`; every path derives from it, and the default is the
published 3-date arm. `USE_PCA` overrides the Stage-1 constant the same way.
Terms used above are defined in `CONTEXT.md`.
