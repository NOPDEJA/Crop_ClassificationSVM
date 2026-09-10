# What we can set for XGBoost inside our cascade

Discussion document, drafted 2026-09-10, for independent review before anything is packaged
or sent to the collaborator.

**Context.** The professor's decision (rule A) is executed as a crossover: the collaborator runs
**XGBoost inside our cascade on our dataset**, and we run **an SVM inside his rejection cascade on
his dataset**, with each pair compared only within itself. Our immediate work is the first half:
prepare our method so that he can run it with XGBoost.

**The question this document settles.** Every configuration knob in `train_parcel_cascade.py`
belongs in exactly one of three tiers: fixed architecture that both arms must share, algorithm-arm
settings that are his to choose, or a genuinely open decision. Getting a knob into the wrong tier
either handicaps his algorithm or lets the architecture drift, and both destroy rule (A).

All code references verified at the current working tree.

---

## 1. The seam

The algorithm lives in one function, `train_parcel_cascade.py:213`:

```python
def base_pipe(p):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("nyst",    Nystroem(kernel="rbf", n_components=p["n_components"], gamma=p["gamma"])),
        ("svc",     LinearSVC(C=p["C"], class_weight=None, max_iter=5000)),
    ])
```

Nothing else in the file's 701 lines constructs an estimator. Everything else is routing, capping,
cross-fitting, calibration and scoring — i.e. the architecture.

**Two complications the seam does not cover.**

1. **`PlattCalibrated` (`train_parcel_cascade.py:256`) is SVM-specific.** It calls
   `base.decision_function(X)` and fits a per-class sigmoid on fold 1. XGBoost has native
   `predict_proba` and no `decision_function`, so it cannot pass through this class unmodified.
2. **`OneVsRestClassifier` is load-bearing for weights, not for accuracy.** The comment at
   `train_parcel_cascade.py:135-146` records why: `LabelBinarizer` turns each sub-problem into
   0/1, so a `class_weight` dict keyed by LU code cannot reach `LinearSVC`, which is why the
   tempered weights are delivered as a per-row `sample_weight`. XGBoost does native multiclass and
   accepts `sample_weight` directly, so the wrapper is pure overhead there.

---

## 2. Tier A — fixed architecture, identical in both arms

He must not change any of these. They are the "same everything else" half of rule (A).

| Knob | Value | Where |
|---|---|---|
| Split | parcel-grouped `split_assign.npy` + `parcel_id_row.npy` | `:207-208` |
| Seed | `RANDOM_STATE = 42` | `config.py:37` |
| Superclasses | ECON (13 codes) / WATER (6) / FOREST (8) / others | `:151-153` |
| Stage-2 groups | orchards / plantation / field, plus `SINK = 4` | `:178-186` |
| `MERGE_TREE` | must be declared and matched (default `0`) | `:154-177` |
| Stage-1 sampling | `SAMPLES_PER_LU = 400_000`, then per-superclass caps: econ 1.0M, water 500k, forest 600k, others 800k | `config.py:50-55`, `:354` |
| Stage-2/3 caps | `PER_GROUP_CAP = 200_000`, `PER_LU_CAP = 70_000` | `config.py:59,63` |
| Cross-fit routing | `CROSSFIT_S1 = 3` parcel-grouped parts | `:220`, `:391` |
| Fold-1 halving | `halve_by_parcel`, stratified by parcel label | `:364` |
| Calibration rows | `CALIB_MAX = 300_000`, **random not per-class**, to keep natural priors | `:219`, `:311` |
| Calibration scheme | per-class Platt on the fold-1 calibration half, base never refitted | `:256` |
| Cost sensitivity | `w_c = sqrt(n_max / n_c)` on post-cap counts, Stage 2 + Stage-3 experts only, **Stage 1 unweighted** | `:232` |
| Operating point | `ALPHA2` / `ALPHA3` swept on the fold-1 tuning half | `:216-217`, `:623` |
| Missing values | `SimpleImputer(strategy="median")` — see open decision O2 | `:223` |
| Scoring | strict: full population, non-crop truth mapped to 0, explicit ordered label list, no pre-masking | `evaluate_end_to_end.py:97` |
| Test discipline | `SKIP_TEST=1` until the final read | `:221` |

The tempered `sample_weight` vector transfers to XGBoost **exactly** — same formula, same rows,
same values — because `XGBClassifier.fit` takes `sample_weight` natively. This is the one piece of
SVM-specific plumbing that turns out to be algorithm-neutral in substance.

---

## 3. Tier B — his to set, because it belongs to the algorithm arm

Per the protocol's fairness condition (b), preprocessing belongs to the algorithm, not the
controls.

| Component | Disposition for the XGB arm | Reason |
|---|---|---|
| `StandardScaler` | **drop** | trees are scale-invariant; keeping it does nothing |
| `Nystroem(rbf)` | **drop** | it exists only to approximate an RBF kernel |
| `OneVsRestClassifier` | **drop** | native multiclass; OvR would handicap him and is only there for weight routing |
| `LinearSVC(C, max_iter)` | replaced by `XGBClassifier(...)` | the swap itself |
| `CHUNK = 400_000` | raise freely | it is a Nyström memory limit (`:322-329`), not a protocol constant |
| Hyperparameters | his, per stage: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`, `tree_method` | see §5 |

**Determinism caveat to declare:** XGBoost with `tree_method="hist"` and `n_jobs > 1` is not
bit-reproducible. He should set `random_state=42` and state the thread count in the manifest, or
pin `n_jobs=1` for the run of record.

---

## 4. Tier C — five genuinely open decisions

### O1. Does XGBoost go through `PlattCalibrated`?

The cascade's decision rule is an argmax over **calibrated** probabilities, and the operating-point
sweep divides by the calibration prior. If the XGB arm emits raw `predict_proba` while the SVM arm
emits Platt-scaled probabilities, the two arms use different decision rules and `ALPHA2`/`ALPHA3`
are not comparable across them.

- **(a) Keep Platt for both.** `_scores` gains a `predict_proba` fallback (roughly five lines:
  use `log(p)` as the score when `decision_function` is absent). Slightly unusual — calibrating an
  already-probabilistic model — but harmless, and it holds the decision rule fixed.
- **(b) Drop Platt for XGB.** Simpler, and arguably fairer to a model that is already calibrated,
  but the arms then differ in more than the estimator.

**Recommendation: (a).** Calibration is architecture here, not algorithm, because a downstream
architectural component consumes its output.

### O2. Imputer, or XGBoost's native missing-value handling?

Genuinely two-sided, and it is the one decision where "fair to the algorithm" and "hold the
controls fixed" point in opposite directions.

- **(a) Keep `SimpleImputer(median)` for both.** Control 3 says the missing-value policy is
  declared once and applied identically. Both arms then see byte-identical inputs.
- **(b) Let XGBoost use native NaN handling.** Learning a default direction per split is a real
  XGBoost capability; denying it is a handicap of the same kind as forcing Nyström on it.

Note that no rows are dropped under either option, so the *population* is identical regardless —
only the values differ. That weakens the control-3 objection considerably.

**Recommendation: (a) for the run of record, (b) as a cheap labelled ablation** if he wants it.
Reason: median imputation on this feature set touches few cells, so (b) is unlikely to be worth
much, and (a) removes an argument later.

### O3. Per-stage or global hyperparameters?

Our arm uses three distinct parameter sets — Stage 1 at `n_components=250, C=1.0`, Stage 2/3 at
`600, C=10`, and Stage-3 experts at `1200` in M5 (`:111-113`, M5 manifest). The stages differ in
row count by more than an order of magnitude, so per-stage capacity is part of our architecture.

**Recommendation: per-stage for him too**, with the per-stage budget declared. A single global
XGBoost configuration across three very differently-sized problems would be a self-inflicted
handicap.

### O4. Which population does he tune on?

`halve_by_parcel` (`:364`) exists precisely because fold 1 does double duty — Platt sigmoids on
one half, operating point on the other. Any hyperparameter search must use the same discipline.

**Recommendation:** hyperparameter search runs on **fold 0 with `GroupKFold` on `parcel_id_row`**,
exactly as E4 did (`e4_retune_stage2_orchards.py:22`). Fold 1 is reserved for calibration and the
operating point; fold 2 is never touched until the final read.

### O5. What tuning budget makes the arms fair? — **the hard one, see §5**

---

## 5. The tuning-budget problem

**Our arm is not tuned to one standard. It is a hybrid, and this has to be disclosed.**

- `P1` (Stage 1) and `P23` are **frozen constants** whose origin is the v2 randomized searches.
  The code comment is explicit that this origin is **a leaky inner CV**
  (`train_parcel_cascade.py:106-110`): "Their origin is a leaky inner CV, which may make them
  suboptimal — but as externally fixed constants they cannot bias a genuinely untouched test
  score."
- Stage 2 and the orchards expert were **cleanly re-searched in E4**: `GroupKFold(3)` on
  `parcel_id_row`, `scoring='f1_macro'`, grid `C ∈ {1, 10, 30}` × `gamma ∈ {0.25, 0.5, 1, 2} ×
  (1/n_features)` × `n_components ∈ {600, 1200}` (stage 2) or `{800, 1200}` (orchards) — **24
  candidates × 3 folds = 72 fits per stage**.

So there is no single answer to "match our budget." Three ways to make it fair:

| Option | What it means | Cost |
|---|---|---|
| **(i) Match E4's budget per stage** | He gets 24 candidates × 3 GroupKFold folds for each stage he tunes | cheap, symmetric with the *cleanly tuned* half of our arm only |
| **(ii) Retune the whole SVM arm cleanly** | We re-search Stage 1 and the remaining Stage-3 experts under GroupKFold at the same budget, so both arms are wholly clean | expensive — this is a multi-day chain — but it is the only option where neither arm carries a leaky-selection asterisk |
| **(iii) Both arms frozen at prior searches** | He freezes at his existing `n_estimators=800, lr=0.1, max_depth=10, subsample=0.6, colsample_bytree=0.8` | cheapest, but his values were selected on a *pixel* split against a *different* architecture, so they are not "his tuned values" for this cascade in any meaningful sense |

**One more fact that cuts in his favour and must be stated in the paper.** Both E4 winners landed
at the **top of every grid axis** — `n_components=1200` (max), `C=30` (max), for both Stage 2 and
the orchards expert (`runs/retune_stage2_orchards/winners.json`). The search hit its capacity
ceiling, so our tuned stages are **under-tuned, not over-tuned**. Any claim that the SVM arm was
given a tuning advantage is contradicted by its own search log.

**Recommendation: (i) now, and declare it.** Give him E4's exact budget per stage, report our arm
honestly as a hybrid (Stage 2 + orchards cleanly searched, the rest frozen from a leaky search),
and note the grid-ceiling finding. Option (ii) is the correct answer if the timeline allows it,
and it should be costed before it is dismissed.

---

## 6. A gap we have to close before packaging

**E7's effective configuration is not written down anywhere.** `e7_final_cascade_fold2_read.py`
composes artifacts from two runs — `runs/s2_2018_3date_parcel_m5` and
`runs/retune_stage2_orchards` — rather than running the cascade end to end. M5's manifest records
`class_weight: null` and `params_stage3: {n_components: 1200}`; E4's directory has **no
`manifest.json` at all**, only `winners.json`. So the configuration the collaborator is being asked
to match exists only as the union of two runs plus the composition script.

**Action:** emit a single consolidated `E7_EFFECTIVE_CONFIG.json` — every Tier A value, both
stages' selected hyperparameters, the operating point (`alpha2=0.4, alpha3=0.5`), the npz name and
hash, and the split hashes — and ship that as the contract. Nothing should be sent to him until
this file exists, because without it "same architecture" is not checkable.

---

## 7. Questions for review

1. **O2** is the decision I am least sure of. Does control 3 (identical missing-value policy)
   really bind when no rows are dropped either way and only the imputed *values* differ? If it
   does not, native NaN handling is the fairer default and (b) should be the run of record.
2. **O1**: is Platt-scaling an already-calibrated XGBoost defensible in a methods section, or does
   it invite exactly the criticism it is meant to prevent?
3. **§5**: is option (i) defensible given our arm is a hybrid, or does the leaky origin of `P1`
   contaminate the comparison badly enough that (ii) is mandatory? Specifically — `P1` governs
   Stage 1, and Stage 1 is where the largest single share of end-to-end loss occurs.
4. Is dropping `OneVsRestClassifier` for XGB definitely right, given the tempered weights were
   designed around its behaviour? The weight *vector* is unchanged, but a per-row weight under
   native softmax is not identical in effect to the same weight under 13 independent binary
   problems.
5. Is anything in Tier A actually algorithm-specific and mis-filed? `CALIB_MAX = 300_000` and
   `CROSSFIT_S1 = 3` are the two I would look at hardest.

## 8. Questions for the collaborator

1. Can he accept per-stage hyperparameters and the E4 search budget (24 candidates × 3
   parcel-grouped folds per tuned stage)?
2. Does he want native NaN handling (O2b) as a labelled ablation?
3. Thread count and `tree_method` for the run of record, so determinism is declared up front.
