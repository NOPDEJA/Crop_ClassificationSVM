# Running XGBoost inside the SVM side's cascade

A handoff document. It tells you what to run, what you may change, and what you must not,
so that the result is comparable with our E7 run rather than merely adjacent to it.

Written 2026-09-10. Everything below was verified against the working tree and the run
artifacts, not against intentions; where a claim could not be verified it says so.

---

## 1. What this is for

The professor's condition for the joint paper is that one axis is held fixed: **different
algorithm, same architecture and same controls.** We are executing that as a **crossover**.

| | runs on | runs it |
|---|---|---|
| XGBoost inside **our** routing cascade, on **our** dataset | your machine | you |
| SVM inside **your** rejection cascade, on **your** dataset | our machine | us |

Each pair is compared **only within itself**. We are not putting your number and our number
in one table across the two datasets, because they would not be measuring the same thing.

This document is the first half. Your comparator is our run **E7**: strict macro F1
**0.2429** over 5,500,269 fold-2 rows.

## 2. The one thing that decides whether this works

Everything in the cascade falls into exactly one of three buckets. Getting a knob into the
wrong bucket either handicaps XGBoost or lets the architecture drift, and either one destroys
the comparison.

**Bucket A — architecture. Identical in both arms. Do not change any of it.**

| | value | enforced by |
|---|---|---|
| Split | parcel-grouped, `splits/split_assign.npy` + `parcel_id_row.npy` | the script |
| Seed | 42 | `config.py` |
| Superclasses | econ (13 codes) / water (6) / forest (8) / others | the script |
| Stage-2 groups | orchards / plantation / field, plus a sink | the script |
| Stage-1 sampling | 400k per LU code, then econ 1.0M / water 500k / forest 600k / others 800k | `config.py` |
| Stage-2 / Stage-3 caps | 200k per group, 70k per LU code | `config.py` |
| Cross-fit routing | fold 0 split into 3 parcel-grouped parts | `CROSSFIT_S1=3` |
| Fold-1 halving | by parcel, stratified by parcel label | the script |
| Calibration | per-class Platt sigmoids, 300k random rows (not per-class, so priors survive) | the script |
| Stage-2 weights | the subtype-mass vector, shipped as a file | `CLASS_WEIGHT=subtype` |
| Stage-3 weights | none | `CLASS_WEIGHT=subtype` |
| Operating point | grid 0 to 1.2 in 13 steps, selected on fold 1's **calibration** half | `sweep_operating_point.py` |
| Missing values | median imputation, fitted on each stage's own fit rows | the script |
| Endpoint | **hard** routing is the headline number. Joint is secondary. | see §6 |
| Scoring | strict: full population, non-crop truth mapped to 0, explicit 13-label list, no masking | the script |
| Test discipline | `SKIP_TEST=1` until the single final read | see §5 |

**Bucket B — the algorithm. Yours.** The estimator and the preprocessing it needs:

- `StandardScaler` and the Nyström kernel map are **dropped**. They exist only so a linear
  SVC can approximate an RBF kernel.
- `OneVsRestClassifier` is **dropped**. Keeping it would compare our SVM against an ensemble
  of independent binary XGBoost models, which is not what XGBoost means. This is a genuine
  change to the learning rule, not a cosmetic one, and we will say so in the methods.
- All XGBoost hyperparameters, per stage, from your own search (§4).
- Your own `alpha2` / `alpha3` from the sweep. **Do not copy our 0.4 / 0.5.**

**Bucket C — neither.** `CHUNK` / `PRED_CHUNK_OVERRIDE` is a memory setting. Raise it freely;
it changes nothing scientific. Thread count and device likewise, but declare them (§7).

## 3. What you receive, and how to check it

Code comes from git so that "same architecture" is a commit hash. Data comes separately
because it is 3 GB.

| item | where | what it is |
|---|---|---|
| the pipeline | this repo, branch `dev` | `train_parcel_cascade.py`, `cascade_algo.py`, `xgb_search.py`, `sweep_operating_point.py` |
| `svm_s2_3date_m3_features_labels.npz` | Google Drive | 24,323,769 rows x 30 Sentinel-2 features, Oct/Nov/Dec 2018 |
| `splits/split_assign.npy`, `splits/parcel_id_row.npy` | Google Drive | the parcel-grouped split |
| `label/label_47PQQ_buffered.tif` | Google Drive | the 3-pixel-eroded label raster of record |
| `configs/e7_svm.json` | this repo | E7's exact per-stage settings, for reference |
| `configs/e7_xgb_template.json` | this repo | the template you replace with your search winners |
| `E7_EFFECTIVE_CONFIG.json` | this repo | every Tier-A value plus hashes, the contract |

Verify the data against `E7_EFFECTIVE_CONFIG.json` before running anything. If a hash does
not match, stop and tell us: a silently different NPZ would make every number incomparable
and nothing downstream would raise an error.

**The 30 columns**, in the order the models saw them, which is the order the NPZ stores and
is part of the model rather than a convenience:

| block | features | dates |
|---|---|---|
| 1–24 | BSI, EVI, MSAVI, NDBI, NDVI, NDWI, SWIR_NIR, SWIR_RATIO | 2018-10-31, 11-30, 12-31 |
| 25–30 | MTCI, B11, interleaved per date | same three dates |

Worth noticing before you assume the feature sets are unrelated: **MTCI is already in here**,
using the same `(B06−B05)/(B05−B04)` formula as yours, and so is a raw SWIR band. The
differences that remain are that our SWIR band is **B11 where yours is B12**, our water index
is NDWI `(green−NIR)/(green+NIR)` where yours is MNDWI `(green−SWIR)/(green+SWIR)`, and we
carry BSI, NDBI, MSAVI and SWIR_RATIO which you do not. So this is not a foreign feature set
to you; it is a superset of most of yours with two substitutions.

Environment: Python 3.13, numpy 1.26.4, scikit-learn 1.7.2, xgboost 3.2.0. Other versions
are probably fine; record what you used.

## 4. What to run, in order

**Step 1 — a template run, to produce the routes your search needs.**

```bash
ALGO=xgb PARAMS=configs/e7_xgb_template.json CLASS_WEIGHT=subtype SKIP_TEST=1 \
NPZ_OVERRIDE=./aligned_features/svm_s2_3date_m3_features_labels.npz \
ARM_OUT=./runs/xgb_template python train_parcel_cascade.py
```

This exists because of a **framework rule**: your Stage 2 trains on whatever *your own*
Stage 1 routed to it. Your Stage-1 model will call a slightly different set of pixels
economic than ours did, so your Stage-2 training rows will differ from ours. **That is
correct and intended.** The rule is fixed; the rows it produces are a consequence of the
algorithm. The alternative, forcing our routes onto your cascade, would mean your Stage 1
was never actually tested. The script saves and hashes the rows it used, so the difference
is recorded rather than hidden.

**Step 2 — the hyperparameter search.**

```bash
SEARCH_FROM=./runs/xgb_template OUT=./runs/xgb_search python xgb_search.py
```

24 candidates x 3 parcel-grouped folds, per stage, scored by macro F1, on fold 0 only.
That is E4's budget, which is what the cleanly-tuned part of our arm received. Read §7
before assuming this is symmetric, because it is not, and the asymmetry favours you.

Two deliberate quirks, both copied from E4 rather than fixed, because matching the
comparator's procedure matters more than improving one arm: the search runs **unweighted**
even for Stage 2, whose final refit carries weights; and Stage 2 searches on a 50,000-per-group
subsample while the Stage-3 experts search on their full capped set.

**No early stopping.** `n_estimators` is a grid axis instead. Early stopping would make its
evaluation rows part of model selection, and there is no partition left to host them.

**One thing to know about this search, because it will look wrong otherwise.** Under
parcel-grouped folds a rare crop can be entirely absent from a fold's training half — Langsat
has about ten parcels in the whole tile. Plain `XGBClassifier` refuses to fit in that case,
raising `Invalid classes inferred from unique values of y`, because it requires labels to be
exactly `0..K-1` and a missing *middle* class leaves a gap. The SVM comparator's search did
not hit this: `OneVsRestClassifier` drops the absent class and carries on silently. So
`xgb_search.py` wraps the estimator in `FoldSafeXGB`, which re-encodes per fit and maps
predictions back, and a class absent from a fold simply scores 0 there. Without it your search
would lose whole folds and pick a winner from the survivors. This was found by running the
script, not by reading it.

**Step 3 — the tuned run, fold 2 still unread.**

```bash
ALGO=xgb PARAMS=./runs/xgb_search/winners.json CLASS_WEIGHT=subtype SKIP_TEST=1 \
NPZ_OVERRIDE=./aligned_features/svm_s2_3date_m3_features_labels.npz \
ARM_OUT=./runs/xgb_e7 python train_parcel_cascade.py
```

**Step 4 — your operating point.**

```bash
RUN_DIR=./runs/xgb_e7 python sweep_operating_point.py
```

Selects on fold 1's **calibration** half and reports the selected cell once on the **tuning**
half. Report the tuning column, never the calibration one: on our data the calibration
column read about 0.047 too high, and that gap is a population difference between the two
parcel halves, not selection optimism.

**Step 5 — the single read of fold 2.**

Rerun step 3 with `SKIP_TEST` unset and `ALPHA2` / `ALPHA3` set to what step 4 chose. Read it
**once**. If something crashes before fold-2 data is touched, fixing it and relaunching is
fine, which is what happened to us in E7. Reading it, disliking the number, and adjusting is
not.

## 5. Why `SKIP_TEST` matters more than it looks

Fold 2 is not a fresh partition. Our M5 and several earlier checkpoints have read it, and E7
read it once more. `SKIP_TEST=1` cannot restore blindness that is already gone. What it does
is stop *your* arm from adding new exposure, so the paper can describe fold 2 honestly as a
previously observed partition rather than pretending otherwise. If you would rather we lock a
genuinely new partition before you start, say so now, because after step 5 it is too late.

## 6. Hard, not joint

The script emits both a hard cascade prediction, committing to the top branch at each stage,
and a joint one, multiplying probabilities along all paths. **Hard is the headline for both
arms.** E7 reported hard. Fixing this in advance is the point: otherwise either side could
pick whichever composition flattered it after seeing both.

## 7. What we are disclosing, and you should too

Three asymmetries that we are stating in the paper rather than hiding. None of them is
resolved by this handoff.

1. **Our arm is a tuning hybrid, and yours will be cleaner.** Only Stage 2 and the orchards
   expert were searched honestly under GroupKFold. Stage 1, plantation and field are still
   frozen at values from an earlier randomized search whose inner CV leaked. `xgb_search.py`
   searches all five of your stages. So your arm gets *more* clean tuning than ours, not
   less. We are running the matching clean SVM retune afterwards; until it lands, the
   comparison is "algorithm plus tuning regime", not "algorithm".
2. **The weight vector transfers; its effect does not.** Both arms receive the same
   observation-level weight vector. Its effect under XGBoost's native softmax loss is not
   identical to its effect across 13 independent binary SVM problems. The control is "the
   same weights are passed to each algorithm's native loss", never "the same cost-sensitive
   learning rule". If you have the budget, an unweighted XGBoost run would tell us how much
   of any difference is this interaction.
3. **XGBoost with `hist` and multiple threads is not bit-reproducible.** That is acceptable;
   declare it. Record device, thread count and xgboost version in your manifest, which the
   script does automatically.

## 7a. What has and has not been tested, before you spend days on it

Stated plainly so you can plan, and so nothing here is a surprise at hour six.

**Tested.** The SVM path is byte-identical before and after the seam was added, across every
saved artifact. The XGBoost path runs the whole cascade end to end. The subtype-mass weight
function reproduces E7's saved 800,000-row vector exactly. The search completes a full
24-candidate run and writes a config the trainer accepts.

**Not tested: any of it at full scale.** Every run above was `SMOKE=1`, which subsamples to
500,000 rows. No full-scale XGBoost run of this cascade exists anywhere yet, on either
machine. Two consequences:

- **Runtime is unmeasured.** What is known: at equal smoke scale XGBoost finished the whole
  cascade in about 20 seconds at 50 trees where the SVM took about 10 minutes, so the
  per-model cost is not the worry. The search is: E4's equivalent 72-fit search took roughly
  13 hours for the SVM at full size, and yours is 5 stages rather than 2. **Run step 1 first
  and time it** before committing to step 2 overnight.
- **Memory is unmeasured for your box.** The SVM's chunking constant exists to keep a Nyström
  block under control and is irrelevant to you; raise `PRED_CHUNK_OVERRIDE` freely. Stage 1
  fits on roughly 2.9 million rows by 30 features.

If step 1 reveals something we got wrong, tell us before working around it. A workaround on
your side becomes an uncontrolled difference between the arms.

## 8. What to send back

- The whole `runs/xgb_e7/` directory, or at minimum `manifest.json`, `report_hard.csv`,
  `pred_hard.npy`, and the `*_fit_idx.npy` / `*_cal_idx.npy` arrays with their hashes.
- `runs/xgb_search/winners.json` and the per-stage `search_*.csv`.
- The selected `alpha2` / `alpha3` and the sweep CSV.
- Your environment versions and hardware.

Two runs are comparable only if their evaluation row identities match. The manifest carries
the hashes that let us check that rather than assume it.

## 9. Open questions we still owe you an answer on

- Do you want a genuinely new test partition locked before step 1 (§5)?
- Do you want the native-NaN ablation (`XGB_NATIVE_NAN=1`) as a labelled extra run? The run
  of record imputes, so both arms see the same values; native handling is a real XGBoost
  capability and denying it is arguably a handicap, but it changes the estimand from "swap
  the estimator on identical inputs" to "each algorithm with its own preprocessing".
- Is the 24-candidate budget acceptable, or do you want a different declared budget applied
  symmetrically to both arms?
