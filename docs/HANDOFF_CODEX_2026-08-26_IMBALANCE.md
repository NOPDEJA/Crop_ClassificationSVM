# Handoff to Codex — how to attack the rare-crop problem next

**Written:** 2026-08-26, by Claude, after executing `PLAN_2026-08-26_WEIGHTED_RUN.md`.
**What is wanted:** an independent, adversarial second opinion. Not confirmation.
**Separate from:** the report review brief in `PLAN_2026-08-26_WEIGHTED_RUN.md` §6, which is
about whether `REPORT_2026-08-27.md` is honest. This document is about what to do *next*.

Two findings in this project's history were retracted only after someone re-derived them
from the artefacts. Assume at least one claim below is wrong and go looking for it.

---

## 0. RESOLVED, 2026-08-26 — Codex answered, and I verified every load-bearing number

| claim | verdict | what replaced it |
|---|---|---|
| C1 caps are the class-weighting scheme | **too categorical** | caps balance the *label*; the *subtype inside* each group is untouched. See below. |
| C2 rubber not over-represented in fitting | **FALSE, and the important one** | Stage 2's plantation pool is **96.82 % rubber, 0.08 % coconut**, so a uniform 200k draw yields ~193,645 rubber against ~159 coconut. Same defect in orchards: durian 66.7 %, Langsat 0.60 %. |
| C3 effective n = parcel count | **directionally right, literally wrong** | Langsat has **10** training parcels, not 27, and one holds 1,310 of its 1,639 pixels. Parcel count is a proxy, not an identity. |
| C4 duplication == sample_weight | **false for this pipeline** | weights reach only `LinearSVC`; duplication also moves the imputer median, scaler moments and the Nystroem landmark draw. |
| C5 SMOTE would densify parcels | **survives** | at k=5, 97.9 % of Langsat nearest-neighbour edges are within the same parcel. |

Two premises in this file were also contaminated and are corrected in
`docs/REPORT_2026-08-27.md` §5: the "+0.0426 optimism" was a parcel-half level difference,
not selection bias (true optimism is +0.0008), and "fold 2 unread" should read "no weighted
end-to-end fold-2 score was computed or used".

**The decisive new number:** freeze every model and replace only the learned Stage-2 route
with the true group, and tuning-half macro F1 goes **0.2294 → 0.3785, worth +0.149**. Soft
probability-product routing does not fix it (0.2279), so the defect is the learned boundary.
The next experiment is the paired Stage-2 subtype-mass probe, two Stage-2 fits rather than two
full cascades. All of this is verified locally, not taken on trust.

---

## 1. The state of the evidence

### What is measured and reproducible

| fact | artefact |
|---|---|
| M5 test macro F1 **0.2344**, weighted 0.7974, 5,500,269 rows, strict convention | `runs/s2_2018_3date_parcel_m5/report_hard.csv`, `manifest.json` |
| M5 tuning-half macro F1 **0.2294** at (0.3, 0.6) selected on the calibration half | `runs/s2_2018_3date_parcel_m5/opsweep_selection.json` |
| Tempered class weights: tuning-half **0.2272**, gate closed, no weighted end-to-end fold-2 score computed | `runs/s2_2018_3date_parcel_weighted/{opsweep_selection.json,PREDECLARATION.md}` |
| ~~Selecting and reporting on the same half is optimistic by +0.0426~~ **WRONG, see §0.** True selection optimism **+0.0008**; the two parcel halves differ by +0.0478 at plain argmax | `opsweep.csv` |
| Tree-merge probe **0.2158** vs M5's **0.2283** at the same fixed cell | `runs/s2_2018_3date_parcel_tree/val_tune_score_a02_07.csv` |
| MTCI ranks **last** in both permutation probes (0.023 / 0.039) | `runs/s2_2018_3date_parcel_m5/permutation_importance_by_index.csv` |

### The per-class fit populations, which are the subject of this document

Stage 1, after a 400,000 per-LU cap then per-superclass caps:
econ 1,000,000 · others 800,000 · forest 600,000 · water 482,151 (below its 500,000 cap).

Stage 2: all four groups capped to **200,000 each**, exactly equal, from candidate pools of
241,238 / 7,046,818 / 950,064 / 1,677,100.

Stage 3, `PER_LU_CAP = 70,000`:

| expert | fitted pixels per crop |
|---|---|
| plantation | rubber 70,000 · oil palm 70,000 · **coconut 8,344** |
| field | rice 70,000 · cassava 70,000 · pineapple 70,000 |
| orchards | durian 70,000 · mango 43,199 · jackfruit 27,008 · longan 14,483 · mangosteen 12,272 · **rambutan 3,770** · **Langsat 1,639** |

---

## 2. Claims I want attacked

### C1. The caps already are the class-weighting scheme

Because `w = sqrt(n_max / n)` is computed from post-cap counts, and because the cap binds for
every class at Stage 2 and in the field expert, those weights came out exactly 1.00. The
weighted run therefore differed from M5 only in the orchards expert and coconut.

*The claim:* anywhere a cap binds for all classes in a model, weights computed after capping
have nothing to correct, so future imbalance work must go at the cap schedule, not at
`class_weight`.

*Attack it:* is the right conclusion instead that D5 picked the wrong denominator? Weighting
by **pre-cap** frequency is the other obvious choice and would give Langsat roughly 50x
rather than 6.5x. I claim neither is obviously correct and that the choice of denominator
*is* the experiment. Say whether you agree, and if you think pre-cap weighting is defensible,
say what stops it double-correcting an imbalance the cap already removed.

### C2. Rubber is not over-represented anywhere in fitting

Nop asked whether we can cap rubber down to match the rare crops. My answer was that it is
already capped equal to oil palm, rice, cassava, pineapple and durian at 70,000, and that its
group is cut from 7.05 M candidates to 200,000 at Stage 2. So the surviving imbalance is not
"too much rubber", it is classes that cannot *reach* the cap.

*The claim:* rubber's prevalence now survives only in two places, the Platt calibration
population (a deliberate random 300,000-row subsample of fold 1 at natural priors, `CALIB_MAX`)
and the test metric itself (3,235,887 of 5,500,269 pixels). Both are handled by the operating
point alpha, not by caps.

*Attack it:* is there a third place I have missed? Specifically check whether rubber's
dominance re-enters through the Stage-1 per-LU cap of 400,000, or through the cross-fitted
Stage-1 routes that build Stage 2's candidate pool.

### C3. The binding constraint is parcels, not pixels

Langsat has 1,639 training pixels from roughly 27 parcels. Those are not 1,639 independent
observations. The literature on spatial autocorrelation reports performance overestimated by
up to 28 % under random versus spatial cross-validation, and object-based sampling being
markedly less affected than pixel-based.

*The claim:* the effective sample size for a rare crop is its parcel count, and our caps count
pixels, so a 70,000-pixel cap on durian may be drawing from a few hundred parcels while the
same cap on a rare crop draws from tens. We have never measured parcels-per-class in the fit
sets.

*Attack it:* this is the claim I am least sure of. It is plausible and it is unmeasured. The
data to settle it is in `splits/parcel_id_row.npy`. If it is true it reframes every cap
decision, and if the parcel counts turn out comparable it is a dead end.

**Counter-evidence to weigh:** the project has a probe (see `rare-species-not-spectrally-limited`
in the session memory) showing these same crops reach 0.44 to 0.68 F1 in isolation under
balanced priors, five dates and higher kernel capacity, with **Langsat scoring highest despite
having the fewest parcels**. That result argues against a pure sample-size story and for an
engineering story about cascade routing. I have not reconciled the two and I would like you
to.

### C4. Naive upsampling is already tested, indirectly

For hinge loss with L2 regularisation, duplicating a sample k times contributes exactly k
times the loss, which is what `sample_weight = k` does. So duplication to equal counts is
`sample_weight = n_max / n`, and last night ran `sqrt(n_max / n)`, half that exponent, and it
lost by 0.0022 on the tuning half.

*The claim:* full upsampling is the more aggressive end of the same one-dimensional axis, and
the operating-point sweep independently shows that pushing toward uniform costs more than it
buys, so it is very likely worse, though not proven.

*Attack it:* is the duplication-equals-weighting equivalence exact for our pipeline? Note the
pipeline is `SimpleImputer → StandardScaler → Nystroem → LinearSVC` inside a
`OneVsRestClassifier`. Duplicating rows changes what the imputer's median, the scaler's mean
and variance, and the **Nystroem landmark draw** see, and `sample_weight` does not. Whether
that difference is material at these ratios is a real question and I have not tested it. This
is the claim most likely to be wrong.

### C5. SMOTE would synthesise the wrong thing

With 1,639 Langsat pixels from ~27 parcels, almost every nearest neighbour of a Langsat pixel
is another pixel of the same parcel. So SMOTE would interpolate inside a parcel's own spectral
cluster, inflating within-parcel density (which we have) and adding no between-parcel variance
(which we lack). It should therefore improve cross-validation while leaving the parcel-disjoint
test fold flat.

*Supporting evidence:* RUESVMs reports beating SVM-SMOTE by about 4.95 percentage points of
overall accuracy on Sentinel-2 land cover.

*Attack it:* this is a mechanism argument, not a measurement. It is cheap to test directly by
computing, for each Langsat pixel, whether its k nearest neighbours in feature space belong to
the same parcel. If they mostly do not, the argument collapses.

---

## 3. Candidate directions, ranked by my confidence, for you to re-rank

1. **Cap-strategy experiment.** Hand-picked caps versus a uniform cap versus sqrt(n)-proportional,
   judged on the tuning half so fold 2 stays shut. Falls directly out of C1.
   *Known design hazard:* equalising the orchards expert down to Langsat's 1,639 gives 11,473
   training rows total, at which point 1,200 Nystroem components is no longer supportable and
   capacity would have to drop, confounding the experiment. Say how you would avoid that.
2. **Undersampling ensemble (RUESVMs).** Many balanced subsets, one SVM each, vote. Sidesteps
   both the data-discarding problem of hard undersampling and the synthesis problem of SMOTE.
   This is my favourite and I want it challenged, particularly on how it interacts with the
   Platt calibration, which currently assumes one base fitted once.
3. **Honest hyperparameter re-tune.** `GroupKFold` on parcel ID with `scoring='f1_macro'`.
   Current parameters came from a pixel-level leaky search scored on **accuracy**, which is the
   least defensible thing left in the pipeline. The literature warns that C, gamma and the class
   costs are not separable, so a one-at-a-time grid finds the wrong point.
4. **Parcel-aware or K-Means SMOTE**, if C5 survives your attack.
5. **Difference-of-date features**, unrelated to imbalance but the cheapest untried signal
   (rubber defoliates Oct→Dec).

---

## 4. Reading list

Verify these yourself rather than trusting my summaries. I read abstracts and search summaries,
not all full texts, and one of them (MDPI Geomatics) returned 403 to my fetch so I have it only
second-hand.

- [RUESVMs, Remote Sensing 12(21) 3484](https://doi.org/10.3390/rs12213484) — random undersampling
  ensemble of SVMs for land cover in GEE, Sentinel-2 time series. Reports about +4.95 pp overall
  accuracy over SVM-SMOTE. The direct template for direction 2.
- [Waske et al., Classifying Remote Sensing Data with SVM and Imbalanced Training Data](https://link.springer.com/chapter/10.1007/978-3-642-02326-2_38) — the older cost-sensitive SVM reference already cited in the report.
- [K-Means SMOTE for imbalanced land cover, Information 12(7) 266](https://doi.org/10.3390/info12070266) —
  oversampling that targets distinctive minority spectral signatures rather than blind neighbours.
- [Exploring balanced and imbalanced multi-class distributions on fruit-tree crop classification, Geomatics 3(1) 4](https://doi.org/10.3390/geomatics3010004) —
  closest analogue to our problem: pixel-based Sentinel-2, fruit-tree crops, imbalance, multiple
  classifiers. SVM reportedly best at 71 % against AdaBoost 67, RF 65, XGBoost 63, GB 62. **I could
  not fetch the full text**, so the sampling-technique conclusions need checking by someone who can.
- [Spatial dependence between training and test sets, Machine Learning 110](https://link.springer.com/article/10.1007/s10994-021-05972-1) and
  [spatially autocorrelated samples inflate CNN assessment](https://www.sciencedirect.com/science/article/pii/S2667393222000072) —
  the basis for C3 and C5. Up to 28 % overestimation, object-based less affected than pixel-based.
- [Multi-class imbalanced SVM via differential evolution, arXiv 2502.14597](https://arxiv.org/pdf/2502.14597) and
  [optimizing cost-sensitive SVM, arXiv 1702.01504](https://arxiv.org/pdf/1702.01504) — joint
  optimisation of C, gamma and class costs. Relevant to direction 3.
- [Fine-grained crop classification by boosting rare-class representations, Frontiers in Remote Sensing 2026](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2026.1822070/full) —
  101 crop classes, rare categories represented by few parcels and pixels, rare-class-aware
  augmentation. Recent and close to our framing.

---

## 5. What I want back

A ranked response, with a re-derivation command or a `file:line` for anything you dispute.
Specifically:

1. Which of C1 to C5 is wrong. At least one probably is, and C4 and C3 are my own top suspects.
2. Your reconciliation of C3 against the counter-evidence that Langsat scores *highest* in the
   isolated probe.
3. Your ranking of §3, with the cheapest decisive experiment named first. Cheap matters: one
   full cascade run is about 3 h 20 m with `SKIP_TEST=1` on a 32 GB machine, serial only.
4. Anything in §3 you would delete outright, and anything missing from it.
5. A design for whichever you rank first, in enough detail to write a predeclaration from,
   including what would count as failure.
