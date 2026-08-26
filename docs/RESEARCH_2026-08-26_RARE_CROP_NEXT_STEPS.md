# Adversarial second opinion: rare crops in the S2-only SVM cascade

**Date:** 2026-08-26  
**Scope:** the root-level, three-date Sentinel-2 hierarchical SVM; validation artifacts only.  
**Safety:** no weighted-run fold-2 score was computed. No training or run artifact was changed.

## Bottom line

The next experiment should not be a full cap sweep or RUESVM ensemble. It should be a
**paired Stage-2-only subtype-mass experiment**. Stage 2 caps each of its four target groups to
200,000 rows, but it samples uniformly *inside* each group. The plantation candidate pool is
96.82% rubber, so its expected 200,000-row fit contains about 193,645 rubber pixels, 6,196 oil-palm
pixels and only 159 coconut pixels. Equal group totals did not remove rubber dominance from the
Stage-2 fit.

This is not a small bookkeeping issue. On M5's tune half, Stage-2 correct-group recall conditional
on passing Stage 1 is 85.39% for rubber, but 13.65% for coconut, 10.99% for mango, 37.78% for
rambutan and 28.57% for Langsat. With every existing model frozen, replacing the learned Stage-2
route by the true group raises strict tune macro F1 from 0.2294 to **0.3785**. A probability-product
soft route does **not** solve it: cal-selected soft routing scores 0.2279 on tune. The failure is in
the learned boundary, not merely the hard argmax.

The adversarial verdict is:

1. **C2 is false and is the most consequential error.** Rubber re-enters fitting at Stage 1 and,
   overwhelmingly, inside Stage 2's plantation group.
2. **C4 is false for the actual pipeline.** Duplication and `sample_weight` are equivalent only
   after freezing the imputer, scaler and Nystroem map.
3. **C3 is directionally right but literally overstated.** Parcel diversity is a binding concern;
   effective sample size is not automatically equal to parcel count.
4. **C1's categorical conclusion is false.** Caps implement one sampling policy; they do not make
   cost weighting irrelevant. Pre-cap, post-cap and parcel counts encode different objectives.
5. **C5 survives its local attack for Langsat.** At k=5, 97.90% of nearest-neighbour edges are
   within the same parcel. The predicted downstream effect of SMOTE is still unmeasured.

## Ranked findings

### 1. C2 misses a third—and dominant—place where rubber enters fitting

**Claim attacked:** rubber prevalence survives only in Platt calibration and the test metric.

**Evidence:** Stage 1 first caps each LU code and then caps the superclass
([`config.py:50`](../config.py), [`train_parcel_cascade.py:342`](../train_parcel_cascade.py)).
Replaying that seeded sampler gives the Stage-1 economic fit 208,027 rubber pixels (20.80%) but
only 871 Langsat pixels (0.087%). Rubber is not larger than every other cap-binding major crop,
but it is grossly over-represented relative to rare crops.

The larger miss is Stage 2. Cross-fitted Stage-1 routes create the candidate pool
([`train_parcel_cascade.py:390`](../train_parcel_cascade.py),
[`train_parcel_cascade.py:490`](../train_parcel_cascade.py)). Stage 2 then caps **only the group
label**, not crop subtype ([`train_parcel_cascade.py:534`](../train_parcel_cascade.py),
[`train_parcel_cascade.py:545`](../train_parcel_cascade.py)). Its plantation pool is:

| crop | routed candidates | share | expected rows in 200,000 cap |
|---|---:|---:|---:|
| Rubber | 6,822,897 | 96.822% | 193,645 |
| Oil palm | 218,319 | 3.098% | 6,196 |
| Coconut | 5,604 | 0.080% | 159 |

The last column is an expectation because the actual `fit2` row indices were not persisted. The
uniform sample is so large that this qualification does not rescue the claim.

Cross-fitting also removes crop pixels selectively before Stage 2: routed recall is 94.13% for
rubber, 67.16% for coconut, 67.30% for mango and 77.67% for rambutan. Those lost rows cannot be
recovered by a Stage-3 cap.

**What the handoff should say instead:** “Rubber is capped equal to other abundant crops in its
Stage-3 expert. It remains a large Stage-1 economic mode and dominates the Stage-2 plantation
training population because Stage 2 balances groups, not crop subtypes. Cross-fitted Stage-1
routing further changes subtype representation.”

### 2. C4's duplication-equals-weighting argument does not hold for this pipeline

**Claim attacked:** the tempered weighted run indirectly tested naive upsampling.

**Evidence:** the fitted pipeline is `SimpleImputer -> StandardScaler -> Nystroem -> LinearSVC`
([`train_parcel_cascade.py:201`](../train_parcel_cascade.py)). The code explicitly routes weights
only to `LinearSVC`, while the scaler declines them ([`train_parcel_cascade.py:206`](../train_parcel_cascade.py));
the imputer and Nystroem map receive none. Duplicating rows changes the imputer's learned median,
the scaler's moments and—most importantly—the probability that a row is selected as one of the
Nystroem landmarks. Scikit-learn documents Nystroem as constructing its map from a subset of the
training rows ([Nystroem](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html));
the fitted statistics and weight APIs are documented for
[SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html),
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
and [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).

The equivalence is exact only for integer duplication **conditional on a frozen transformed design
matrix** and the same regularization convention. The weighted run used square-root weights, not
full duplication weights. Its one loss does not establish monotonicity along the exponent, and the
operating-point alpha is a decision-time prior correction, not the same axis as fit-time resampling.

**What the handoff should say instead:** “The weighted run tested square-root loss weighting with
unchanged preprocessing and landmarks. It did not test row duplication. A frozen-map micro-test can
confirm the mathematical equivalence; a refitted-map arm is required to measure the pipeline effect.”

### 3. C3 identifies the right unit problem but makes an unjustified equality

**Claim attacked:** the effective sample size for a rare crop *is* its parcel count.

**Evidence:** in the current training fold, Langsat has 1,639 pixels from **10**, not roughly 27,
feature-valid training parcels. One parcel contributes 1,310 pixels (79.9%); the remaining parcel
pixel counts are 116, 94, 29, 29, 22, 22, 15, 1 and 1. Pixel count is therefore a badly inflated
description of independent support. However, `n_eff = number of parcels` is not an identity:
between-parcel similarity, within-parcel correlation, parcel size imbalance and spatial separation
all matter.

The literature supports grouped/spatial validation, not that equality. Karasiak et al. found
ordinary CV optimistic and spatial leave-one-out closest to independent-map performance in a
Sentinel-2 forest experiment ([Machine Learning, 2022](https://link.springer.com/article/10.1007/s10994-021-05972-1)).
The quoted “up to 28%” is from CNN tree-species segmentation over UAV acquisitions, not an S2 SVM
parcel study ([Kattenborn et al., 2022](https://doi.org/10.1016/j.ophoto.2022.100018)).

**What the handoff should say instead:** “The effective sample size is bounded by parcel diversity
and can be far below the pixel count. We have 10 Langsat training parcels with extreme parcel-size
imbalance. Parcel count is a measurable proxy, not the effective sample size itself.”

### 4. The isolated Langsat probe does not rebut C3 because its split leaks parcels

This is the reconciliation the handoff was missing. The probe balances pixels by crop and performs
a random pixel permutation split ([`probe_dry_season.py:68`](../probe_dry_season.py),
[`probe_dry_season.py:102`](../probe_dry_season.py)). Replaying that split with parcel IDs shows:

- Langsat train side: 14 parcels; test side: 16 parcels.
- Shared across both sides: 14 parcels—**87.5% of test parcels**.
- For the other rare crops, 89.5% to 95.7% of test parcels are also represented in training.

Langsat can therefore score highest if its observed parcels form compact, distinctive
spectral-temporal clusters. That result says “Langsat pixels from mostly known parcels are separable
under a uniform prior with five dates.” It does not say that ten training parcels cover unseen-parcel
variation. In fact, high random-pixel F1 and poor parcel-disjoint F1 are exactly what within-parcel
autocorrelation can produce.

The current analysis text should withdraw “the ceiling is not spectral” as a generalization claim
([`docs/S2_SVM_ANALYSIS.md:1160`](S2_SVM_ANALYSIS.md)). The defensible replacement is: “The observed
parcels contain separable signal under a pixel-leaky balanced probe; unseen-parcel separability is
unmeasured.” A parcel-grouped rerun is required.

### 5. C1 confuses three different target distributions

**Claim attacked:** wherever caps bind for all labels, future imbalance work must change caps rather
than `class_weight`.

**Evidence:** the implemented weights are explicitly calculated from post-cap label counts
([`train_parcel_cascade.py:220`](../train_parcel_cascade.py)). Their becoming 1.0 means only that
*post-cap inverse-frequency weighting* is degenerate. It does not make other cost objectives
meaningless.

The denominator is indeed part of the experiment, but it must be defined within the model that sees
the classes:

- In the orchards expert, square-root weights from raw training counts give Langsat about **10.5x**
  Durian, not 50x.
- A global rubber-to-Langsat square-root ratio is about 66.5x, but those crops are in different
  Stage-3 experts; using that global scale silently changes each expert's effective `C`.
- At Stage 2, the useful denominator is the **routed pre-cap crop subtype count inside each target
  group**, even though crop subtype is not the Stage-2 label.

Pre-cap weighting is defensible if it states a target cost distribution and is normalized within
each Stage-2 group so total group loss remains fixed. That normalization is what prevents
double-correcting the already equalized group totals. Class costs and SVM hyperparameters are not
generally separable; joint search is supported by cost-sensitive SVM work, though existing evidence
is mostly non-remote-sensing and binary
([Guido et al.](https://link.springer.com/article/10.1007/s00500-022-06768-8)).

**What the handoff should say instead:** “Caps and weights target different distributions. Post-cap
weights correct residual fit imbalance; routed pre-cap subtype weights can deliberately preserve
rare modes inside a capped Stage-2 group. The objective, normalization and model scope must be
predeclared.”

### 6. C5's local mechanism is supported, but its outcome prediction is not yet measured

**Claim attacked:** most Langsat nearest neighbours are in the same parcel.

**Evidence:** among current 30-feature training rows, after orchard-train median imputation and
standardization, nearest neighbours restricted to Langsat give:

| k | same-parcel neighbour edges | queries whose every neighbour is same parcel |
|---:|---:|---:|
| 1 | 98.96% | 98.96% |
| 3 | 98.31% | 97.13% |
| 5 | **97.90%** | **95.42%** |
| 10 | 96.45% | 89.99% |

So ordinary pixel-SMOTE would overwhelmingly interpolate within a parcel in this feature space. It
would not create new between-parcel support. What remains unproved is the stronger prediction that
CV must rise while a parcel-disjoint score stays flat.

K-Means SMOTE is useful precedent but not validation of that prediction: it used seven
hyperspectral pixel benchmarks, LR/KNN/RF, and stratified pixel CV—not SVM, parcels or S2 time
series ([Fonseca et al., 2021](https://www.mdpi.com/2078-2489/12/7/266)).

**What the handoff should say instead:** “For current Langsat features, naive SMOTE is almost
certainly within-parcel densification. Do not run it. Parcel-constrained interpolation remains a
later hypothesis, to be evaluated with parcel-disjoint validation.”

## Two contaminated premises in the handoff

These do not change the recommendation, but an adversarial handoff should not call them facts.

1. `0.2720 - 0.2294 = 0.0426` is the difference between two disjoint parcel halves at the
   cal-selected cell. It is not a measured optimism bias from selecting and reporting on the same
   half. The directly relevant M5 comparison is tune-selected maximum 0.2302 versus honest
   cal-selected transfer 0.2294, a difference of 0.0008.
2. “Weighted fold 2 unread” is too broad. The run has no weighted end-to-end fold-2 score, which is
   the decision-relevant fact. The training script nevertheless predicts Stage 1 for all rows before
   `SKIP_TEST` is applied downstream. Future text should say “no weighted end-to-end fold-2 score was
   computed or used.”

## Re-ranked next work

Ranking considers diagnostic value per wall-clock hour, transfer to this exact hierarchy, and the
need to keep fold 2 closed.

1. **Paired Stage-2 subtype-mass correction—first intervention.** Two Stage-2-only fits; freeze
   Stage 1, Stage 3 and all validation arrays. This directly attacks the 96.82% rubber plantation
   pool and the oracle-routing gap. Expected cost is roughly two Stage-2 fits, not two 3 h 20 m
   cascades.
2. **Parcel-grouped isolated probe, then parcel-uniform orchards loss.** First rerun the 3-date and
   5-date isolated probe with parcel groups; this is minutes-to-tens-of-minutes and settles the
   misleading Langsat counter-evidence. If signal survives, compare pixel loss with weights that
   give every parcel equal total mass within crop. A parcel-based S2/SVM precedent averaged pixels
   within parcels before SVM fitting ([Sitokonstantinou et al., 2018](https://www.mdpi.com/2072-4292/10/6/911)).
3. **Honest, component-specific parcel GroupKFold retune.** Tune `C`, Nystroem gamma/components and
   at most one imbalance parameter jointly, using macro F1. Do it first for Stage 2 or the orchards
   expert, not the full cascade. The present accuracy-optimized pixel-level parameters remain a
   larger validity defect than the choice between fashionable samplers.
4. **Three-date phenology/difference features, first in a parcel-grouped expert probe.** This stays
   cheap if limited to predeclared differences/slopes for a few indices. Multi-temporal S2 crop work
   supports phenological feature importance, but not the local Oct-to-Dec rubber hypothesis. The
   recent rare-crop paper likewise finds full-year temporal information important, but uses a deep
   segmentation model, not SVM ([Wang et al., 2026](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2026.1822070/full)).
5. **Small RUESVM/Waske ensemble only at the failing expert or Stage 2.** RUESVMs is genuine S2+SVM
   precedent and reported +4.95 percentage points OA over SVM-SMOTE
   ([Naboureh et al., 2020](https://doi.org/10.3390/rs12213484)), but it searched 100 sampling
   fractions, trained ten RBF SVMs per setting, used hard votes and did not evaluate macro F1,
   calibration, hierarchy or parcel-disjoint transfer. Aggregate member scores first, then fit one
   ensemble calibrator on natural-prior calibration parcels; do not vote separately calibrated
   operating points.
6. **Fixed-total cap/quota strategies.** Preserve 200,000 rows per Stage-2 group and 1,200-component
   Stage-3 capacity. Allocate subtype quotas proportional to a predeclared rule such as `sqrt(n)`;
   never equalize the whole orchards expert down to 1,639 per crop.
7. **Parcel-aware or parcel-cluster oversampling.** Only after the grouped probe and only with
   synthetic edges forced across distinct parcels. Ordinary and K-Means pixel-SMOTE remain low
   priority despite positive benchmark papers. A direct multitemporal S2/GWO-SVM paper reports OA
   gains for SMOTE variants, but its accuracy objective and sampling protocol are not sufficient
   evidence for this parcel-disjoint macro-F1 problem
   ([Zhang et al., 2022](https://www.mdpi.com/2072-4292/14/20/5259)).

### Delete or demote

- **Delete ordinary pixel-SMOTE.** The kNN mechanism test is already strongly against it.
- **Delete “equalize every orchard crop to 1,639” as an experimental arm.** It changes sample size,
  Nystroem capacity and parcel coverage simultaneously.
- **Demote a full-cascade cap sweep.** Replace it with fixed-total, component-only paired tests.
- **Demote RUESVMs from favourite to fifth.** It multiplies fits and introduces an ensemble
  calibration problem before the actual Stage-2 subtype defect has been tested.
- **Do not pursue soft probability-product routing further.** It was cheap to test and lost:
  0.2279 versus 0.2294 on tune.

### Missing from the original list

- Stage-2 crop-subtype weighting or quotas while keeping the four group totals fixed.
- Parcel-uniform loss within each crop (`each parcel contributes equal total weight`).
- A routing-loss decomposition: Stage-1 pass rate, Stage-2 correct-group rate and Stage-3 conditional
  crop F1 by crop.
- Aggregate-level calibration for any ensemble.
- Uncertainty/sensitivity reporting for rare tune classes: Langsat has only 9 tune pixels from two
  parcels, and only 4 calibration pixels from one parcel.

## Predeclaration-ready design for rank 1

### Name and hypothesis

**S2-M6 Stage-2 routed-subtype loss probe.** The M5 Stage-2 group SVM under-represents rare crop
modes inside each capped group. Increasing their loss mass while holding total mass per group fixed
will improve strict tune-half 13-crop macro F1.

### Fixed data and models

- Use exactly M5's NPZ, 30 valid columns, split assignment, parcel IDs, OOF Stage-1 routes,
  `val_cal_idx.npy` and `val_tune_idx.npy`.
- Never load or score weighted-run fold 2. Set `SKIP_TEST=1` and do not construct test candidate
  arrays in the probe.
- Freeze M5 Stage-1 candidate membership and all three M5 Stage-3 validation probability arrays.
- Keep Stage-2 parameters fixed: 600 Nystroem components, existing gamma, `C=10`, OvR LinearSVC,
  seed 42 and 200,000 selected rows per group.

### Paired arms

Create one deterministic 800,000-row Stage-2 fit set using a new predeclared seed and the existing
200,000-per-group sampler. Both arms use these exact rows and identical preprocessing/Nystroem
landmarks.

- **A—paired control:** every row weight is 1.
- **B—routed-subtype treatment:** for a crop row of subtype `c` inside group `g`, start with

  `u(c,g) = sqrt(max_j n(g,j) / n(g,c))`,

  where `n(g,c)` is the crop count in the complete OOF-routed Stage-2 training candidate pool before
  the 200,000 group cap. Set sink rows to 1. Normalize the treatment weights separately within each
  Stage-2 target group so their mean is exactly 1. This preserves equal total loss mass across the
  four Stage-2 labels and avoids double-correcting the group cap.

Weights go only to LinearSVC. This is intentional: the paired test isolates loss mass and keeps the
imputer, scaler and landmark map fixed. Resampling/landmark effects are a later experiment.

### Calibration and composition

- Fit each Stage-2 base on its arm's fold-0 fit rows.
- Fit each arm's Platt sigmoids unweighted on the same natural-prior M5 calibration-half candidates.
- Predict only the complete M5 validation candidate set.
- Compose with frozen M5 Stage-3 validation probabilities using the existing hard hierarchy.
- For each arm separately, sweep the same 13 x 13 alpha grid (`0.0..1.2`, step `0.1`), select on the
  calibration half, and report the selected cell exactly once on the tune half with the strict
  full-population scorer.

### Outcomes and gate

**Primary:** paired difference `B - A` in strict tune-half 13-crop macro F1.

**Success:** `B - A >= +0.0020` absolute and the number of crops with F1 >= 0.01 does not fall.
This earns a later confirmatory full-cascade run; it is not permission to open fold 2.

**Failure:** the difference is below +0.0020, the alive-crop guard falls, or a gain appears only on
the calibration half and does not transfer. Close this weighting rule; do not tune its exponent on
the tune half.

**Mandatory secondary diagnostics (not alternative success criteria):** per-crop Stage-2
correct-group recall conditional on Stage-1 passage; end-to-end per-crop F1 and support; rubber F1;
calibration curves/Brier score by Stage-2 group; and paired parcel-level sensitivity because the
Langsat tune support is only two parcels.

**Sanity check:** report A against cached M5, but make the causal comparison B versus A. The original
M5 Stage-2 sampled indices were not saved, so B-versus-cached-M5 would confound treatment with a new
random cap sample.

### Capacity and compute

No class is reduced to Langsat's size, total Stage-2 rows stay 800,000 and Nystroem stays at 600
components. M5's Stage-2 fit/calibration took about 48 minutes before test prediction; with test
skipped, two serial arms should be far cheaper than two full cascades. Abort before Stage 3 because
its probabilities are frozen.

## Re-derivation recipes

All local audits used `C:\Conda_environment\envs\svm_env\python.exe` and validation/train arrays
only. The essential recipes are below; they deliberately do not mention a weighted fold-2 array.

### Crop pixels and parcels by split

```powershell
$code = @'
import json, numpy as np
R='runs/s2_2018_3date_parcel_m5'
y=np.load(json.load(open(R+'/manifest.json'))['npz'],allow_pickle=True)['y']
p=np.load('splits/parcel_id_row.npy'); a=np.load('splits/split_assign.npy')
cal=np.load(R+'/val_cal_idx.npy'); tune=np.load(R+'/val_tune_idx.npy')
for c in [2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420]:
    print(c, *[(int((y[r]==c).sum()), int(np.unique(p[r[y[r]==c]]).size))
               for r in [np.flatnonzero(a==0),cal,tune]])
'@
& 'C:\Conda_environment\envs\svm_env\python.exe' -c $code
```

### Stage-2 within-group candidate composition

```powershell
$code = @'
import json, numpy as np
R='runs/s2_2018_3date_parcel_m5'
y=np.load(json.load(open(R+'/manifest.json'))['npz'],allow_pickle=True)['y']
tr=np.load(R+'/stage1_train_idx.npy'); route=np.load(R+'/stage1_route_oof_train.npy')
c=tr[route==1]
for lu in [2302,2303,2405,2403,2404,2407,2413,2416,2419,2420,2101,2204,2205]:
    print(lu, int((y[c]==lu).sum()))
'@
& 'C:\Conda_environment\envs\svm_env\python.exe' -c $code
```

### Probe parcel leakage

Reproduce `probe_dry_season.py:68-104` with seed 42, then replace its metric calculation with
`np.unique(parcel_id_row[rows[tr]])`, `np.unique(parcel_id_row[rows[te]])` and their intersection.
The important indexing detail is that `tr` and `te` are positions within `rows`, exactly as the
probe uses them.

### Langsat neighbour identity

Use training-fold orchard rows, median-impute and standardize the current 30 valid features, fit
`NearestNeighbors(n_neighbors=11)` on Langsat rows, discard the self-neighbour, and compare
`parcel_id_row[query]` with `parcel_id_row[neighbour]` for k in `{1,3,5,10}`. Restricting the
neighbour index to Langsat matches ordinary within-minority SMOTE.

## Literature conclusions, with transfer limits

- **RUESVMs:** direct S2 time-series/SVM precedent; its reported +4.95 pp is OA against SVM-SMOTE,
  after an extensive fraction search and with hard voting—not evidence for calibrated hierarchical
  macro F1 ([paper](https://doi.org/10.3390/rs12213484)).
- **Fruit-tree imbalance:** the quoted SVM 71% is real, but the best result came from an imbalanced
  60%-of-original training sample and used OA. It cuts against “more balanced is always better”
  ([Chabalala et al., 2023](https://doi.org/10.3390/geomatics3010004)).
- **K-Means SMOTE:** strong pixel-benchmark results but no SVM, parcels or S2 time series
  ([paper](https://www.mdpi.com/2078-2489/12/7/266)).
- **Spatial validation:** supports parcel/spatial grouping; does not identify parcel count with
  effective sample size ([Karasiak et al.](https://link.springer.com/article/10.1007/s10994-021-05972-1),
  [Kattenborn et al.](https://doi.org/10.1016/j.ophoto.2022.100018)).
- **Probabilistic hierarchies:** genuine method precedent, but the local probability-product replay
  lost, so it should not displace Stage-2 retraining
  ([Silva-Palacios et al., 2018](https://riunet.upv.es/entities/publication/d58a9818-8f4e-40c2-9f07-1ba49d5c7515)).
- **Effective-number weights:** a bounded alternative to inverse frequency, but developed for deep
  cross-entropy models; using pixel or parcel counts in an SVM would be a new project-specific arm
  ([Cui et al., 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html)).

## One thing to check next

Rebuild the isolated three-date/five-date probe with a **parcel-grouped split and per-parcel summary
scores**. The current “Langsat is highest” counterexample is much less independent than documented;
if that result collapses under parcel grouping, the project's strongest argument that rare crops are
“not spectrally limited” must be retracted or sharply narrowed. If it survives, Stage-2 routing and
parcel-weighted fitting become substantially stronger bets.
