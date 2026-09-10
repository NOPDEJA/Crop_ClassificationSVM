"""xgb_search.py

The XGBoost arm's hyperparameter search, at the SVM comparator's budget.

WHY THIS SCRIPT EXISTS RATHER THAN A PARAGRAPH IN A README
The fairness of the whole comparison rests on the two arms getting comparable
tuning effort. A sentence saying "24 candidates, three parcel-grouped folds" is
not auditable; a script that enforces it is. Everything the budget consists of
is a constant at the top of this file, and the search writes the grid it
actually ran into its output.

THE BUDGET, and where it comes from
E4 (e4_retune_stage2_orchards.py) is the only cleanly-searched part of the SVM
comparator: GroupKFold(3) on parcel_id_row, over fold 0 only, scoring f1_macro,
24 candidates per stage, refit=False. This script mirrors it exactly, including
two details that are quirks rather than best practice, because matching the
comparator's procedure matters more than improving one arm:

  1. THE SEARCH IS UNWEIGHTED even for Stage 2, whose final refit carries the
     subtype-mass weights. E4 did this (gs.fit without sample_weight, then a
     weighted refit), so the XGBoost search does it too. Fixing it in one arm
     only would make the tuning protocols differ.
  2. STAGE 2 SEARCHES ON A SUBSAMPLE (50,000 per group) while the Stage-3
     experts search on their full capped fit set. That is E4's split too.

WHAT HAS NO PRECEDENT, and is therefore a declared choice
E4 never searched Stage 1 or the plantation and field experts -- in the SVM arm
those are still frozen at the v2 randomized searches, whose origin is a leaky
inner CV. This script searches all five, which means the XGBoost arm will be
MORE cleanly tuned than the SVM arm, not less. That asymmetry is real, it
favours XGBoost, and it must be stated in the paper rather than buried: the
SVM arm is a hybrid until its own clean retune is run. Stage 1's search
population follows Stage 2's rule (50,000 per class) because it has no rule of
its own and 2.9M rows x 72 fits is not affordable.

NO EARLY STOPPING. n_estimators is a grid axis instead. Early stopping would
make its evaluation rows part of model selection, and every partition is
already spoken for: fold 1 is calibration plus operating point, fold 2 is the
single read.

INPUT: a completed SKIP_TEST=1 run of train_parcel_cascade.py, which supplies
the saved fit-row indices. Stage 2's rows depend on Stage 1's routes and are
therefore algorithm-dependent by design -- that is the framework rule, "your
Stage 2 trains on whatever your own Stage 1 routed", and it is why this script
reads them from a run rather than recomputing them.

Env:
  SEARCH_FROM=<dir>  the SKIP_TEST=1 run to take fit rows from (required)
  OUT=<dir>          output (default ./runs/xgb_search)
  NPZ_OVERRIDE       feature matrix; must match the run's manifest
  STAGES=a,b,c       subset of stage1,stage2,orchards,plantation,field
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, GroupKFold
from xgboost import XGBClassifier

from cascade_algo import XGB_DEVICE, XGB_NTHREAD
from config import RANDOM_STATE

SEARCH_FROM = os.environ.get("SEARCH_FROM", "")
OUT = os.environ.get("OUT", "./runs/xgb_search")
PARCEL_ID = "./splits/parcel_id_row.npy"
SPLIT_ASSIGN = "./splits/split_assign.npy"

SEARCH_SEED = 777             # E4's, independent of the trainer's seed-42 sequence
SEARCH_PER_CLASS = 50_000     # E4's STAGE2_SEARCH_PER_GROUP, applied to Stage 1 too
N_SPLITS = 3
SCORING = "f1_macro"

# 3 x 2 x 2 x 2 = 24 candidates, matching E4's 24 exactly.
#
# E4 searched three axes (C, gamma, n_components) and left everything else at
# library defaults. This searches four and leaves the rest fixed, which is the
# same shape of decision, but it raises the question of WHAT the unsearched knobs
# should be held at -- and the answer must not be an arbitrary choice of ours,
# because a value we picked would be a hyperparameter of the XGBoost arm selected
# by the SVM side.
#
# So the unsearched knobs are held at the COLLABORATOR'S OWN published values
# (train_crops.py at f5f1a6a): subsample 0.6, colsample_bytree 0.8, and
# reg_lambda at XGBoost's own default of 1.0 since he does not set it. The parts
# of this configuration we did not search are therefore his, not ours.
GRID = {"max_depth": [4, 8, 12],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [400, 800],
        "min_child_weight": [1, 10]}
FIXED = dict(subsample=0.6, colsample_bytree=0.8, reg_lambda=1.0,
             tree_method="hist", random_state=RANDOM_STATE, device=XGB_DEVICE)

ECON = {2101, 2204, 2205, 2302, 2303, 2403, 2404, 2405, 2407, 2413, 2416, 2419, 2420}
WATER = {4101, 4102, 4103, 4201, 4202, 4203}
FOREST = {3100, 3101, 3200, 3201, 3300, 3301, 3401, 3501}
GROUPS = {1: {2403, 2404, 2407, 2413, 2416, 2419, 2420},
          2: {2302, 2303, 2405},
          3: {2101, 2204, 2205}}
GNAME = {1: "orchards", 2: "plantation", 3: "field", 4: "sink"}
SINK = 4


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class FoldSafeXGB(BaseEstimator, ClassifierMixin):
    """XGBoost that survives a CV fold whose training half is missing a class.

    THE PROBLEM THIS EXISTS FOR, found by running the search rather than by
    reading it. XGBClassifier requires the labels it is fitted on to be exactly
    0..K-1 and raises `ValueError: Invalid classes inferred from unique values
    of y` otherwise. Encoding the whole search population once and then letting
    GroupKFold split it does NOT satisfy that: parcels are the grouping unit and
    the rare crops have very few parcels -- Langsat has about 10 in the entire
    tile -- so a fold's training half can legitimately contain none of a class.
    Every fit in that fold then fails, and GridSearchCV reports a winner chosen
    from whatever survived.

    WHY THE FIX IS THIS ONE. The SVM comparator's search (E4) used
    OneVsRestClassifier, which drops an absent class and carries on without
    comment. If XGBoost instead crashed on those folds, the two arms would not
    be facing the same cross-validation, and the "same budget" claim would be
    false in a way that is invisible in the output. So this re-encodes per fit,
    to whatever classes that fit actually sees, and maps predictions back to the
    caller's label space so the macro-F1 scorer still compares like with like.
    A class absent from a fold's training half simply scores 0 on that fold,
    which is the correct thing for a macro average to record.

    Parameters are named explicitly rather than collected in a dict because
    sklearn's clone() reads them off the __init__ signature, and because an
    auditable search should not hide its search space behind **kwargs.
    """

    def __init__(self, max_depth=6, learning_rate=0.1, n_estimators=400,
                 min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=1.0, tree_method="hist", random_state=RANDOM_STATE,
                 device="cpu", n_jobs=None):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.tree_method = tree_method
        self.random_state = random_state
        self.device = device
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        enc = np.searchsorted(self.classes_, y)
        p = {k: v for k, v in self.get_params().items() if v is not None}
        # objective is deliberately NOT forced: with enc guaranteed 0..k-1,
        # XGBoost infers binary or multiclass correctly for whatever this fold
        # actually holds.
        self.model_ = XGBClassifier(**p)
        self.model_.fit(X, enc, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.classes_[self.model_.predict(X)]


def subsample(idx, labels, per_class, gen):
    """E4's cap(), with its own generator so the trainer's sequence is untouched."""
    parts = []
    for c in np.unique(labels):
        w = idx[labels == c]
        parts.append(w if w.size <= per_class else gen.choice(w, per_class, replace=False))
    return np.sort(np.concatenate(parts))


def run_search(X_pop, y_pop, groups_pop, tag):
    n_cand = int(np.prod([len(v) for v in GRID.values()]))
    log(f"--- {tag}: {X_pop.shape[0]:,} rows, {n_cand} candidates x {N_SPLITS} folds")
    assert n_cand == 24, f"budget is 24 candidates, this grid has {n_cand}"
    t0 = time.time()

    # Labels stay in their own space (LU codes, or superclass codes); FoldSafeXGB
    # re-encodes per fit to whatever classes that fold actually holds. Encoding
    # once here instead would crash every fold whose training half is missing a
    # rare crop -- see the FoldSafeXGB docstring.
    splits = list(GroupKFold(n_splits=N_SPLITS).split(X_pop, y_pop, groups=groups_pop))
    for i, (a, b) in enumerate(splits):
        missing = set(np.unique(y_pop).tolist()) - set(np.unique(y_pop[a]).tolist())
        note = f"  (train half missing {sorted(missing)})" if missing else ""
        log(f"    fold {i}: train {a.size:,} test {b.size:,}{note}")

    est = FoldSafeXGB(**FIXED, **({"n_jobs": XGB_NTHREAD} if XGB_NTHREAD else {}))
    gs = GridSearchCV(est, GRID, scoring=SCORING, cv=splits, n_jobs=1,
                      refit=False, verbose=1)
    # Deliberately unweighted -- see the module docstring, point 1.
    gs.fit(X_pop, y_pop)
    df = pd.DataFrame(gs.cv_results_)
    df.to_csv(f"{OUT}/search_{tag}.csv", index=False, encoding="utf-8-sig")
    best = df.loc[df["mean_test_score"].idxmax()]
    winner = {k: (int(best[f"param_{k}"]) if k != "learning_rate"
                  else float(best[f"param_{k}"])) for k in GRID}
    winner.update({k: v for k, v in FIXED.items()
                   if k in ("subsample", "colsample_bytree", "reg_lambda")})
    log(f"    {tag} winner {winner}  mean {best['mean_test_score']:.4f} "
        f"({(time.time() - t0) / 60:.1f} min)")
    return winner, float(best["mean_test_score"])


if __name__ == "__main__":
    if not SEARCH_FROM:
        raise SystemExit("SEARCH_FROM=<a SKIP_TEST=1 run directory> is required")
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(f"{SEARCH_FROM}/manifest.json"))
    if meta.get("smoke"):
        raise SystemExit(
            f"{SEARCH_FROM} is a SMOKE run. Its saved row indices address the "
            "500,000-row subsample, not the full matrix, so searching against it "
            "would train on silently wrong pixels without raising. Point "
            "SEARCH_FROM at a full SKIP_TEST=1 run.")
    if meta.get("algo") != "xgb":
        log(f"NOTE: {SEARCH_FROM} was run with algo={meta.get('algo')!r}. Stage-2 "
            "and Stage-3 rows are algorithm-dependent by design, so searching "
            "XGBoost against another algorithm's routes is not the framework rule.")
    npz = os.environ.get("NPZ_OVERRIDE") or meta["npz"]
    log(f"=== xgb search ===  from {SEARCH_FROM}  npz {npz}")

    d = np.load(npz, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int32)
    parcels = np.load(PARCEL_ID)
    valid_cols = np.load(f"{SEARCH_FROM}/valid_cols.npy")
    if valid_cols.size != X.shape[1]:
        X = X[:, valid_cols]
    assert X.shape[1] == meta["n_features"], "feature count does not match the run"

    sup = np.where(np.isin(y, list(ECON)), 1,
          np.where(np.isin(y, list(WATER)), 2,
          np.where(np.isin(y, list(FOREST)), 4, 3))).astype(np.int32)
    g_of = np.zeros(y.size, dtype=np.int32)
    for g, codes in GROUPS.items():
        g_of[np.isin(y, list(codes))] = g
    g_of[g_of == 0] = SINK

    gen = np.random.default_rng(SEARCH_SEED)
    want = os.environ.get("STAGES", "stage1,stage2,orchards,plantation,field").split(",")
    winners, scores = {}, {}

    for tag in want:
        tag = tag.strip()
        if tag == "stage1":
            fit = np.load(f"{SEARCH_FROM}/stage1_fit_idx.npy")
            pop = subsample(fit, sup[fit], SEARCH_PER_CLASS, gen)
            lab = sup[pop]
        elif tag == "stage2":
            fit = np.load(f"{SEARCH_FROM}/stage2_fit_idx.npy")
            pop = subsample(fit, g_of[fit], SEARCH_PER_CLASS, gen)
            lab = g_of[pop]
        else:
            # experts search on their FULL capped fit set, as E4's orchards did
            pop = np.load(f"{SEARCH_FROM}/stage3_{tag}_fit_idx.npy")
            lab = y[pop]
        log(f"{tag} population {pop.size:,} "
            f"{dict(zip(*[a.tolist() for a in np.unique(lab, return_counts=True)]))}")
        winners[tag], scores[tag] = run_search(X[pop], lab, parcels[pop], tag)

    cfg = {"algo": "xgb",
           "_comment": [
               f"Selected by xgb_search.py from {SEARCH_FROM} on "
               f"{time.strftime('%Y-%m-%d')}.",
               f"Budget: 24 candidates x GroupKFold({N_SPLITS}) on parcel_id_row, "
               f"fold 0 only, scoring {SCORING}, unweighted search + weighted refit.",
               "Grid: " + json.dumps(GRID),
               "Mean CV scores: " + json.dumps({k: round(v, 4) for k, v in scores.items()})],
           "stage1": winners.get("stage1", {}),
           "stage2": winners.get("stage2", {}),
           "stage3": {g: winners[g] for g in ("orchards", "plantation", "field")
                      if g in winners}}
    with open(f"{OUT}/winners.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(f"{OUT}/search_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"search_from": SEARCH_FROM, "npz": npz, "grid": GRID,
                   "fixed": {k: str(v) for k, v in FIXED.items()},
                   "budget_candidates": 24, "n_splits": N_SPLITS, "scoring": SCORING,
                   "search_seed": SEARCH_SEED, "per_class": SEARCH_PER_CLASS,
                   "early_stopping": False, "weighted_search": False,
                   "mean_cv_scores": scores,
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    log("wrote", f"{OUT}/winners.json")
    log("DONE")
