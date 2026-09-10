# Joint Paper: Proposed Angle and Shared Protocol

> **SUPERSEDED 2026-09-10 by `docs/JOINT_PROTOCOL_2026-09-10_CROSSOVER.md`.**
> The 2x2 factorial on a common 15-column feature set is withdrawn: it required four new
> runs before anything could be said, and rule (A) is satisfied by a single pair without
> it. The replacement is a crossover — each side runs the other's algorithm inside its own
> architecture on its own dataset. Control 5 ("the same ordered training pixel IDs") is
> also withdrawn as unsatisfiable inside a cascade and replaced by a framework rule.
> This document is kept unedited for the history of how the design got here; do not
> execute from it.


Rayong crop classification, 2018 LDD survey, tile 47PQQ.
Drafted 2026-09-03, after reading the collaborator's 2 September 2026 report and their
repository at commit `f5f1a6a`.
**Revised 2026-09-04** after an independent review
(`docs/CODEX_REVIEW_2026-09-04_JOINT_PROTOCOL_AND_REPORTS.md`). The revision is substantial:
the architecture factor has been redefined, no existing result is treated as a factorial cell any
more, and two controls have been added. The disposition of every review finding is in
`docs/CODEX_DISPOSITION_2026-09-04.md`.

This document is meant to be read by the collaborator and the professor and then either agreed or
amended. Its purpose is to fix the comparison rules **before** either side runs another
experiment, because the evaluation protocol has already been shown to move macro F1 more than
most modelling changes do.

---

## 1. The rule we are working under

The professor's condition for a joint paper is that one axis must be held fixed:

- **(A)** different algorithm, same architecture and same control variables, or
- **(B)** same algorithm, different method.

We have chosen **(A)**. The rest of this document is about what (A) actually requires, because
the two systems are not yet aligned enough for it.

## 2. The problem: two different hierarchies

Both systems are hierarchical. They are not the same hierarchy.

| | SVM system | XGBoost system |
|---|---|---|
| Level 1 | 4-way routing: economic crops / water / forest / others | binary water rejection, P(water) >= 0.56 |
| Level 2 | 4-way routing inside economic crops: field / plantation / orchards / sink | binary building rejection, P(building) >= 0.56 |
| Level 3 | one specialist model per crop group | one flat 14-label crop head on the survivors |
| Idea | semantic **routing** | sequential **rejection** |

Under rule (A) as literally stated, one side would have to abandon its architecture. Both
one-sided options are bad:

- **The SVM flattens to match XGBoost.** This removes the SVM study's only structural
  contribution, and it ignores a real asymmetry: a flat RBF-SVM over 24.3 million rows is not
  affordable the way a flat gradient-boosted tree is. Part of the reason the cascade exists is
  that constraint.
- **XGBoost adopts the SVM cascade.** This is clean and gives each author a distinct
  contribution, but it silently assumes the routing cascade is the better architecture, which is
  exactly the question worth asking.

## 3. The resolution: architecture as a second factor

Architecture becomes a second controlled factor. **The two levels of that factor are two
cascades, not "flat versus cascade".** This matters and an earlier draft of this document got it
wrong: the collaborator's end-to-end system is not a flat classifier, it is a rejection cascade
whose last stage happens to be flat. Comparing it against a genuinely flat SVM would confound
architecture with the presence of the upstream filters.

**Factor 1, algorithm:** RBF-SVM against XGBoost.
**Factor 2, architecture:**
- **Semantic routing cascade** — the SVM design: assign every pixel to a branch, then let a
  specialist inside that branch decide.
- **Sequential rejection cascade** — the collaborator's design: reject water, then reject
  buildings, then classify the survivors with one flat 14-label head.

Both architectures get implemented with both algorithms. Every cell emits the **same final label
set**: the 13 target crops plus `others`.

| | RBF-SVM | XGBoost |
|---|---|---|
| Semantic routing cascade | cell A — **to be run** | cell C — **to be run** |
| Sequential rejection cascade | cell B — **to be run** | cell D — **to be run** |

### All four cells are pending, and that is deliberate

No existing result is a factorial cell. This is the single biggest change from the first draft,
which wrongly marked two cells as already done.

- **The SVM's E7 result is not cell A.** It uses the SVM's own 30-feature Sentinel-2 set, not the
  agreed common features, and it was produced before these controls existed.
- **The collaborator's repaired-pipeline result is not cell D either.** A result from the
  repaired pipeline is now available (see §6a). It fixes the row-deletion problem and removes the
  sampling cap, which is real progress, but it is scored on a crop-only population with no
  held-out partition, so it is not yet a cell. The 2 September numbers from the *old* pipeline are
  superseded and should not be quoted at all.

E7 and the 2 September result are **motivation and baselines**. They belong in the introduction
and in a "where each study stood before alignment" table. They must not appear in the factorial
table or contribute to any estimated effect.

### Four cells are required for factorial language

An earlier draft said one cell could be dropped while keeping the design. That was wrong. With
three cells the algorithm effect, the architecture effect, and the interaction cannot all be
identified. So:

- **Four aligned cells** are required before the paper uses the words *main effect* or
  *interaction*.
- If only three are achievable, the study is renamed to **controlled pairwise comparisons**, and
  the claims shrink to the specific pairs that were actually run.

## 4. Proposed paper angle

> **Routing or rejection? A parcel-disjoint comparison of two hierarchical designs for crop
> mapping under extreme class imbalance in Rayong Province.**

Four contributions:

1. **A shared parcel-disjoint evaluation**, defined in section 5. Both systems scored on the same
   frozen pixel identities, the same parcel split, the same label definitions, and the same
   denominator. A pixel-split result may be reported as a protocol sensitivity experiment, but
   never as unseen-parcel performance.
2. **Routing against rejection.** Where each hierarchy loses information, traced stage by stage,
   instead of treating the final F1 as a black box.
3. **A rare-class support analysis.** Pixel count is not the same as independent information.
   Langsat has 1,639 training pixels from 10 parcels, and one parcel holds 1,310 of them.
4. **A disagreement map.** Where both systems agree with high confidence the prediction is a
   stronger candidate; where they disagree, those parcels become targets for field checking or
   the next LDD survey. This is useful to LDD regardless of which system scores higher.

A formal ensemble stays **optional**, for the reasons in section 9.

## 5. The eight controls to freeze

In descending order of how much damage each one does if left unfixed.

| # | Control | Proposed value | Why it matters |
|---|---|---|---|
| 1 | **Evaluation population** | the natural tile, uncapped | Population alone is worth **0.131 macro F1** on unchanged SVM predictions (0.2654 natural against 0.3965 capped at 200,000 per class), and accuracy and weighted F1 move the *opposite* way. Larger than any modelling difference either side has measured. |
| 2 | **Frozen evaluation pixel identities** | one shared raster grid, one linear pixel ID per cell, one evaluation-row mask computed over the **common** features only, saved and hashed as an artifact | This is the control the collaborator's own report shows is missing. If each model deletes its own rows, the four cells score different populations and nothing is comparable. Every cell must be provable to have predicted the same ordered list of test pixel IDs. |
| 3 | **Missing-value policy** | declared once and applied identically: which values count as missing, whether a row is dropped or imputed, and over which feature subset eligibility is decided | A row must not be eligible in one cell and ineligible in another. Eligibility is decided over the common feature set, never over features only one arm uses. |
| 4 | **Split** | the SVM side's parcel-grouped split, shared as `parcel_id_row.npy` and `split_assign.npy`, fixed seed | The SVM side already runs parcel-disjoint and E7 demonstrates the machinery. The XGBoost side currently splits at pixel level and needs migrating to this split. Earlier SVM experiments had the same defect, so this is a repair the SVM side has already made, not a standard invented for the other side. |
| 5 | **Training population** | corresponding cells receive the **same ordered training pixel IDs**, seeded, including any per-class cap | Otherwise the algorithm comparison is confounded with training volume and sample composition. If a cap is computationally necessary for the SVM, the same capped rows go to XGBoost. An additional uncapped-XGBoost arm may be reported as a **data-volume ablation**, clearly labelled, never as the algorithm cell. Evaluation stays natural and uncapped regardless. |
| 6 | **Labels and class list** | `label/label_47PQQ_buffered.tif` — the 3-pixel-eroded raster — as the artifact of record, with `label/label_47PQQ.tif` named as its raw source; an explicit ordered label list of the 13 crops plus `others` | The eroded raster is what `align_indices_labels.py` actually consumes. It cannot be regenerated from script, because compound mixed-crop `LU_ID_L3` codes are not handled, so the raster itself is the reference. |
| 7 | **Scoring convention** | full population, non-crop truth mapped to `others`, the explicit ordered label list passed to the metric, no masking before the metric, and the macro denominator stated in every table | Masking to crop-truth rows first discards a model's own false positives on non-crop ground. On the SVM side this read **0.034 macro F1 too high** until it was fixed. See section 6 on naming. |
| 8 | **Reported partition** | one held-out partition, fixed in advance, with its prior exposure stated honestly | See the caveat below — neither side currently has a globally untouched test set. |

### Feature set

The natural common denominator is the collaborator's 15 columns: NDVI, EVI, MNDWI, MTCI and raw
B12, for each of October, November and December 2018. The SVM side's DEM and Sentinel-1 stack
then becomes a clearly separate ablation. If the feature sets differ between cells, the paper
compares two complete *systems* and must not attribute the difference to the algorithm.

### The test-partition caveat, stated plainly

The SVM's fold 2 was read once **within the E1–E7 plan**, which is what that plan's
predeclaration required. It is **not** globally untouched: the same partition produced M5 and
several earlier checkpoints. Two honest options, and the paper must pick one:

- **Preferred:** lock a genuinely new paper test partition before any joint run, and never read it
  until all four cells are finished.
- **Otherwise:** describe fold 2 as a previously observed partition and drop any "read once"
  phrasing that implies it was untouched across the whole project.

### Two fairness conditions

- **Each cell gets its own hyperparameter search at a comparable, declared budget.** The SVM
  cascade's capacity was retuned specifically for an RBF kernel (C = 30, gamma at half the
  previous rule, 1,200 components). Dropping an untuned XGBoost into those stages would measure
  tuning effort rather than algorithm, and the same applies in reverse.
- **Preprocessing belongs to the algorithm arm, not the shared controls.** The SVM needs a
  Nystrom or PCA map before the kernel; XGBoost neither needs nor wants one. Forcing identical
  preprocessing would handicap one arm for no scientific gain. Declare it per arm in the methods.

## 6a. The repaired-pipeline result, and what it is measuring

A result from the repaired pipeline is now available. **It resolves the population question and
raises a different one.** Read from a screenshot of the classification report, so the exact
figures should be confirmed against the CSV before either side quotes them.

| | value |
|---|---|
| Rows scored | **19,680,774** |
| Reported macro avg F1 | 0.43 over 14 labels |
| Macro F1 over the 13 target crops | **0.4577** |
| Weighted avg F1 | 0.72 |
| Accuracy | 0.67 |
| `others` | precision 0.00, recall 0.00, F1 0.00, **support 0** |

**Two structural facts, both established by arithmetic rather than by reading the code.**

**First, the evaluation population contains no non-crop pixels at all.** The 13 crop supports sum
to 19,680,774 exactly — the reported total. There is no remainder. So `others` is not merely
under-represented, it is absent, and crop precision cannot be charged for a single false positive
landing on non-crop ground. This is precisely the masking convention that control 7 forbids, and
that the SVM side measured on its own predictions as reading **0.034 macro F1 too high**. The
reported 0.43 also averages a zero-support `others` into 14 labels, which costs about 0.033
against the 13-crop figure of 0.4577 — so the *headline* number is depressed by a naming choice
while the *per-crop* numbers are inflated by the population choice. The two must not be allowed to
cancel in the reader's mind.

**Second, the score appears to be in-sample.** `infer_crops.py` reads the whole extracted CSV,
predicts on all of it, and calls `classification_report` on the result. There is no train/test
separation in that path. The crop model was fitted on a capped draw from the same tile, so for
every class with fewer than 200,000 eroded pixels — that is, every crop except rubber, and by a
wide margin for the rare orchards — the training draw is close to the whole class, and those
pixels are inside the 19.68 million being scored. The rare-crop figures (coconut 0.39, longan
0.47, longkong 0.37) are therefore measured substantially on pixels the model was fitted on, and
they are exactly the classes a macro average rewards most.

**What the SVM reads under the same crop-only convention.** Rescoring E7's unchanged predictions
on crop-truth rows only, which is the closest available match to that population:

| convention | rows | 13-crop macro F1 |
|---|---:|---:|
| SVM E7, strict, full population | 5,500,269 | 0.2429 |
| SVM E7, crop-truth rows only | 3,800,567 | **0.2662** |
| XGBoost repaired pipeline | 19,680,774 | **0.4577** |

The masking convention alone is worth +0.0233 to the SVM. The remaining gap is not yet
attributable to the algorithm: it also spans the in-sample scoring above, the erosion asymmetry
(the training extractor erodes, the evaluation extractor does not), and the absence of any parcel
grouping. Each of those has been measured on SVM data to be worth a large amount on exactly these
crops — the parcel-split probe alone moved rare crops from 0.5852 to 0.3945 overall, with Langsat
going from 0.6774 to 0.0000. One number in that table is a generalisation estimate and the other
is not, and no honest comparison can be drawn until section 5 is applied to both.

One incidental cross-check worth recording: rubber precision is 0.98 in his report and 0.980 in
the SVM's crop-only rescore. The two systems agree closely on the one class where support is not
the limiting factor.

## 6. Naming the macro average

Three different numbers can be called "the macro F1" of the collaborator's 2 September table, and
the paper must not let them drift:

| statistic | value | denominator |
|---|---|---|
| macro over 13 target crops, including zero-support Longkong | 0.3692 | 13 |
| macro over supported target crops, Longkong excluded | ~0.400 | 12 |
| macro over crops plus `others`, including Longkong | 0.38 | 14 |

**Rule:** freeze the ordered label list, report the denominator in every table, and if
zero-support targets are excluded, call the statistic *macro F1 over supported target crops* and
state separately that Longkong was unevaluable. A 12-class average must never be labelled a
13-crop macro without that qualification.

## 7. Execution manifest, required for every cell

Each of the four runs must emit a manifest before its result is quotable:

- data hash and feature names in order
- evaluation pixel-ID hash and training pixel-ID hash
- split hash
- label artifact name and hash
- missing-value policy applied
- hyperparameter search space and budget, and the selected settings
- thresholds and operating points, and how they were selected
- random seed
- final prediction hash

Two cells are comparable only if their evaluation pixel-ID hashes match.

## 8. Open questions for the collaborator

Raised as joint methodology to settle together, not as criticism of the model.

1. **Why is `others` support 0?** This is now the most important question, because it decides
   whether any joint table is possible. The 13 crop supports sum exactly to the reported total, so
   the scored population is entirely crop pixels. Did the rasterised label file change so that
   non-crop codes now go to nodata rather than to 9999, or are the water and building filters
   removing them? Either way, control 7 requires non-crop ground back in the denominator.
2. **Is the reported score in-sample?** `infer_crops.py` scores the whole extracted CSV with no
   train/test separation, and the crop model was fitted on a capped draw from the same tile. If so,
   the figures are a fit quality rather than a generalisation estimate, and the rare crops are the
   classes most affected. Confirming this costs one sentence and changes how every number is read.
3. **The support comparison in the report is not like-for-like, in two ways.** The standalone crop
   numbers come from the 20 percent cross-validation partition of the capped training sample
   (`train_crops.py` reports on `y_cv`), totalling about 304,081 rows, while the pipeline numbers
   come from the full capped inference sample, totalling 755,295. So the pipeline population is
   roughly 2.5 times **larger** overall even though nine classes fall and five rise, and the two
   also had different missing-value rules applied. The support chart therefore cannot carry a
   general "support collapsed" reading on its own.
4. **The NaN mechanism is one identified contributor, not an established sole cause.** The
   mechanism is real and the per-model fix is the right fix. But lower support does not
   mechanically lower F1, and point 3 means the two results also differ in partition, class
   distribution, and missing-value handling. Establishing causation needs an old-versus-repaired
   comparison **on the same frozen pixel IDs** — which control 2 makes possible, and which the
   repaired code now makes cheap to run.
5. **The survivor denominator.** Crop metrics are computed only on pixels that survived the water
   and building filters, so a rubber pixel wrongly classified as water is removed from the crop
   evaluation rather than charged as a missed rubber pixel. With water precision 0.955 and
   building precision 0.875 this is a real number of pixels, and it raises crop recall. The joint
   paper needs a full-population score that charges all three models' errors together.
6. **The erosion asymmetry.** `binary_erosion` is applied in the training extractor but not in the
   final-pipeline evaluation extractor, so the crop model is trained on parcel interiors and
   scored on all pixels including mixed edges.
7. **Training and deployment populations differ.** `train_crops.py` reads the older capped,
   eroded, un-gated CSV while inference reads the un-eroded gated population, so the deployed
   model was never fitted on the distribution it is scored on.
8. **Sampling reproducibility, in the training draw only.** The 200,000-per-class *training* draw
   in `extract_crops.py` uses an unseeded `np.random.choice`, so the fitted model cannot be
   reproduced exactly. The old *evaluation* draw in `extract_all.py` was seeded with
   `default_rng(42)`, and the splits are seeded at 42, so this applies to the training sample
   alone.

Two things were checked and are **clean**, recorded so they are not raised again: there is no
coordinate leakage, because `raw_crops.csv` carries no row or column fields and inference selects
features by name, and the band indexing and the MTCI formula are both correct.

## 9. Ensemble

Combining the two systems' probability outputs honestly would require the same final label set,
calibration on the same parcel-disjoint validation data, and a combining rule fitted without
reading the test partition. Until that exists, the consensus and disagreement analysis in
contribution 4 is the safe version of the same idea. If the errors prove complementary, an
ensemble becomes a justified follow-up rather than a predetermined story.

## 10. What each side provides

**SVM side:**

- `aligned_features/svm_dem_s1_s2_features_labels.npz` — 153 features, 24,323,769 rows
- `splits/parcel_id_row.npy` and `splits/split_assign.npy` — the parcel-grouped split
- `label/label_47PQQ_buffered.tif` — the eroded label raster of record
- the frozen evaluation-row mask and pixel-ID list from control 2, with hashes
- the strict scoring script, so both sides compute the metric with the same code

If the collaborator prefers to keep his own extraction, the SVM side can instead reproduce his 15
columns from its own Sentinel-2 composites, since the red-edge bands needed for MTCI are present.
That avoids a data handoff and removes the unseeded-sample problem from the shared path.

**Collaborator:**

- answers to section 8, especially question 1
- cell C and cell D, or cell D alone if the SVM side implements the rejection cascade
- his crop model re-scored on the agreed population, denominator and pixel IDs

## 11. Decision requested

1. Do we accept two cascade architectures as the two levels of the architecture factor, rather
   than "flat versus cascade"?
2. Do we accept that all four cells are pending, and that factorial language requires all four?
3. Are the eight controls in section 5 agreed as written, and if not, which are amended?
4. Do we lock a new paper test partition, or report fold 2 as previously observed?
5. Who runs which cells?

---

### Appendix: current numbers, and why they are not comparable

Neither column is a factorial cell. Both are prior results included as motivation, and the right
column is **superseded by its own author**: it comes from the old shared-dataset pipeline, not the
repaired one. The repaired pipeline has no reported score yet.

| | SVM cascade, run E7 | XGBoost, **old** pipeline, 2 Sept report |
|---|---|---|
| Population | 5,500,269 rows, natural parcel-disjoint fold 2 | 755,295 rows: 200,000-per-class cap, then shared-dataset row deletion, then post-filter survivors |
| Macro F1 over the 13 target crops | **0.2429** (13 evaluable) | **0.3692** (13, one of them zero-support) |
| Macro F1 including `others` | 0.2743, measured on the M5 configuration | 0.38 (14 labels) |
| Weighted F1 | 0.7949 | 0.58 |
| Accuracy | not reported for this run | 0.57 |

That 0.3692 should not be treated as the XGBoost system's performance at all — it is the score of
a configuration its author has already replaced. It is reproduced here only because it is the most
recent number on record.

The 13-crop gap of 0.126 is **mostly the population**, not the model. Rescoring the SVM's own
unchanged predictions under two proxy versions of the collaborator's protocol gives 0.2673 and
0.2757 on the 13 crops. Those two scenarios do **not** bracket the answer — they order in
opposite directions on the 14-label column, 0.3078 against 0.3060 — so what the SVM scores under
the collaborator's exact protocol is **still unknown**, and it stays unknown until the controls in
section 5 are agreed. That is the whole argument for settling section 5 before running section 3.
