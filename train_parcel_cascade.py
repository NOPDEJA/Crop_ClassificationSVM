"""train_parcel_cascade.py

W5 of PLAN.md: the three-stage SVM cascade trained and scored under the
parcel-disjoint split, with the balancing collapsed to a single mechanism.

This is a NEW script rather than an edit of stage1_weight_scale.py /
stage2_weighted.py / stage3_new_weight.py, for two reasons. First, a pre-flight
review found roughly ten separate places where those scripts would still leak
after the obvious edits -- all three sample and cap *before* splitting, valid_cols
is derived from the combined sample, Stage 2's true-economic filter deletes the
sink, Stage 3 selects on upstream success, Stage 2's optional full retrain spans
folds, and removing the searches leaves dangling `rsearch.best_params_` references
that crash at the very end of a multi-hour run. Second, leaving those files
untouched keeps the published s2_2018_3date_v2 arm reproducible.

WHAT IS DIFFERENT FROM THE PUBLISHED ARM
  split          parcel-atomic (splits/split_assign.npy), never pixel-level
  fitting        base pipeline sees fold 0 only; caps applied to fold 0 only
  calibration    per-class Platt sigmoids fitted on a random subsample of fold 1,
                 on top of a base fitted once on fold 0 and never refitted; the
                 subsample is random rather than per-class so natural priors survive
  balancing      caps only. No class_weight, no upsampling, no SMOTE
  Stage 2        four-way: the 13 crops map to orchards/plantation/field, and
                 EVERYTHING else Stage 1 called economic goes to the sink (4)
  Stage 3        experts selected by TRUE LU membership, not by upstream success
  scoring        fold 2 only, every test row retained, non-crop truth mapped to 0
                 so false positives count against precision
  hyperparams    frozen at the values the v2 searches chose, so no inner CV runs
                 and nothing is selected on pixel-level folds

Per-class decision thresholds are deliberately NOT tuned here. Tuning them on the
same fold used for calibration would make them in-sample to it, and several
classes lack the validation support to tune against at all (Langsat has 13 rows).
The decision rule is plain argmax; thresholds are deferred work.

Env:
  SMOKE=1        subsample rows for a fast end-to-end check
  ARM_OUT=<dir>  output directory (default ./runs/s2_2018_3date_parcel)
"""
import json
import os
import time

import joblib
import numpy as np
from sklearn.calibration import _SigmoidCalibration
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from config import (NPZ, RANDOM_STATE, SAMPLES_PER_LU,
                    CAP_ECON, CAP_WATER, CAP_FOREST, CAP_OTHERS,
                    PER_GROUP_CAP, PER_LU_CAP)

# -----------------------
# Config
# -----------------------
SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT = os.environ.get("ARM_OUT", "./runs/s2_2018_3date_parcel" + ("_smoke" if SMOKE else ""))
SPLIT_ASSIGN = "./splits/split_assign.npy"
PARCEL_ID = "./splits/parcel_id_row.npy"

# Frozen at the values the v2 randomized searches selected. Their origin is a
# leaky inner CV, which may make them suboptimal -- but as externally fixed
# constants they cannot bias a genuinely untouched test score, and freezing is
# what removes the pixel-level inner CV entirely.
P1 = dict(n_components=250, gamma=0.02, C=1.0)
P23 = dict(n_components=600, gamma=None, C=10.0)

CHUNK = int(os.environ.get("PRED_CHUNK_OVERRIDE", 400_000))
CALIB_MAX = 300_000       # random (NOT per-class) subsample -> natural priors kept
ECON = {2101, 2204, 2205, 2302, 2303, 2403, 2404, 2405, 2407, 2413, 2416, 2419, 2420}
WATER = {4101, 4102, 4103, 4201, 4202, 4203}
FOREST = {3100, 3101, 3200, 3201, 3300, 3301, 3401, 3501}
GROUPS = {1: {2403, 2404, 2407, 2413, 2416, 2419, 2420},   # orchards
          2: {2302, 2303, 2405},                            # plantation
          3: {2101, 2204, 2205}}                            # field
SINK = 4
GNAME = {1: "orchards", 2: "plantation", 3: "field", 4: "sink"}
CROPS = sorted(ECON)

os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(RANDOM_STATE)
manifest = {"smoke": SMOKE, "params_stage1": P1, "params_stage23": P23,
            "pca": False, "class_weight": None, "upsampling": False,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def base_pipe(p):
    """No PCA. No class_weight. Nystroem gamma=None means 1/n_features."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("nyst", Nystroem(kernel="rbf", n_components=p["n_components"],
                          gamma=p["gamma"], random_state=RANDOM_STATE)),
        ("svc", LinearSVC(C=p["C"], class_weight=None, max_iter=5000,
                          random_state=RANDOM_STATE)),
    ])


class PlattCalibrated:
    """Per-class Platt scaling on top of a base fitted ONCE on fold 0.

    CalibratedClassifierCV is deliberately not used. Even wrapped in
    FrozenEstimator it takes the cross_val_predict branch
    (`calibration.py:325`), splitting the calibration set into internal folds;
    when a fold happens to lack a class it indexes predictions by the raw LU
    code and crashes. Fold 1 IS the held-out calibration set, so cross-validating
    inside it buys nothing and costs robustness -- several crops have only tens
    of validation rows.

    The base estimator is never refitted here, which is the property that keeps
    the test fold untouched.
    """

    def __init__(self, base):
        self.base = base
        self.classes_ = base.classes_

    def _scores(self, X):
        s = self.base.decision_function(X)
        return s.reshape(-1, 1) if s.ndim == 1 else s

    def fit(self, X, y):
        S = self._scores(X)
        self.cal_ = []
        for k, c in enumerate(self.classes_):
            yk = (y == c).astype(int)
            if yk.sum() < 2 or yk.sum() == yk.size:
                # too few (or no) positives to fit a sigmoid: fall back to the
                # raw logistic of the margin rather than inventing a mapping
                self.cal_.append(None)
                log(f"    calibration fallback for class {c} "
                    f"({int(yk.sum())} positives of {yk.size})")
                continue
            sig = _SigmoidCalibration()
            sig.fit(S[:, k], yk)
            self.cal_.append(sig)
        return self

    def predict_proba(self, X):
        S = self._scores(X)
        P = np.empty_like(S, dtype=np.float64)
        for k, sig in enumerate(self.cal_):
            P[:, k] = 1.0 / (1.0 + np.exp(-S[:, k])) if sig is None else sig.predict(S[:, k])
        tot = P.sum(1, keepdims=True)
        tot[tot == 0] = 1.0
        return P / tot

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def fit_calibrated(X, y, fit_idx, cal_idx, p, tag):
    """Fit the base on fit_idx (fold 0), then only the sigmoids on cal_idx (fold 1)."""
    log(f"  {tag}: base fit on {fit_idx.size:,} rows (fold 0)")
    base = OneVsRestClassifier(base_pipe(p))
    base.fit(X[fit_idx], y[fit_idx])

    if cal_idx.size > CALIB_MAX:      # random, not per-class: preserves priors
        cal_idx = np.sort(rng.choice(cal_idx, CALIB_MAX, replace=False))
    missing = set(base.classes_.tolist()) - set(np.unique(y[cal_idx]).tolist())
    if missing:
        log(f"    WARNING {tag}: classes absent from calibration rows: {sorted(missing)}")
    log(f"  {tag}: sigmoid calibration on {cal_idx.size:,} rows (fold 1, natural priors)")
    return PlattCalibrated(base).fit(X[cal_idx], y[cal_idx])


def chunked_proba(model, X, idx):
    """Chunk size is deliberately NOT config.PRED_CHUNK.

    Nystroem materialises an (n_rows x n_components) float64 block per transform,
    and OneVsRest calls it once per class. At the inherited 2,000,000-row chunk
    with 600 components that is 9.6 GB per transform -- enough to swap or OOM on a
    32 GB box hours into the run. 400,000 keeps it under 2 GB. A 500k-row smoke
    test cannot surface this, because it never builds a chunk that large.
    """
    out = np.zeros((idx.size, len(model.classes_)), dtype=np.float32)
    for s in range(0, idx.size, CHUNK):
        e = min(idx.size, s + CHUNK)
        out[s:e] = model.predict_proba(X[idx[s:e]]).astype(np.float32)
        log(f"    {e:,}/{idx.size:,}")
    return out


def cap(idx, labels, per_class, limit_total=None):
    """Cap per class. Applied to TRAIN rows only, never to val or test."""
    parts = []
    for c in np.unique(labels):
        w = idx[labels == c]
        parts.append(w if w.size <= per_class else rng.choice(w, per_class, replace=False))
    out = np.concatenate(parts)
    if limit_total and out.size > limit_total:
        out = rng.choice(out, limit_total, replace=False)
    return np.sort(out)


if __name__ == "__main__":
    log("=== parcel-disjoint cascade ===  OUT =", OUT)
    d = np.load(NPZ, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int32)
    asg = np.load(SPLIT_ASSIGN)
    parcels = np.load(PARCEL_ID)
    assert X.shape[0] == y.size == asg.size == parcels.size

    if SMOKE:   # keep every fold represented; small enough to finish in minutes
        keep = rng.choice(y.size, 500_000, replace=False)
        keep = np.sort(keep)
        X, y, asg, parcels = X[keep], y[keep], asg[keep], parcels[keep]
        log(f"SMOKE: subsampled to {y.size:,} rows")

    tr = np.flatnonzero(asg == 0)
    va = np.flatnonzero(asg == 1)
    te = np.flatnonzero(asg == 2)
    log(f"folds  train {tr.size:,}  val {va.size:,}  test {te.size:,}")
    assert tr.size and va.size and te.size
    # a parcel must not straddle folds
    assert np.intersect1d(parcels[tr], parcels[te]).size == 0, "train/test share a parcel"
    assert np.intersect1d(parcels[va], parcels[te]).size == 0, "val/test share a parcel"
    manifest["folds"] = {"train": int(tr.size), "val": int(va.size), "test": int(te.size)}

    # valid_cols from TRAIN ONLY -- deriving it from the combined sample would let
    # the test fold influence which features exist.
    nan_cols = np.all(np.isnan(X[tr]), axis=0)
    valid_cols = np.flatnonzero(~nan_cols)
    if nan_cols.any():
        log(f"dropping {int(nan_cols.sum())} all-NaN columns (train-derived)")
        X = X[:, valid_cols]
    np.save(f"{OUT}/valid_cols.npy", valid_cols)
    manifest["n_features"] = int(X.shape[1])

    # ---------------- Stage 1 ----------------
    log("STAGE 1: superclass")
    sup = np.where(np.isin(y, list(ECON)), 1,
          np.where(np.isin(y, list(WATER)), 2,
          np.where(np.isin(y, list(FOREST)), 4, 3))).astype(np.int32)

    tr_lu = cap(tr, y[tr], SAMPLES_PER_LU)             # per-LU cap, train only
    caps1 = {1: CAP_ECON, 2: CAP_WATER, 4: CAP_FOREST, 3: CAP_OTHERS}
    parts = []
    for s, lim in caps1.items():
        w = tr_lu[sup[tr_lu] == s]
        parts.append(w if w.size <= lim else rng.choice(w, lim, replace=False))
    fit1 = np.sort(np.concatenate(parts))
    log(f"  train rows after caps: {fit1.size:,}  dist "
        f"{dict(zip(*np.unique(sup[fit1], return_counts=True)))}")
    manifest["stage1_train"] = {int(k): int(v) for k, v in
                                zip(*np.unique(sup[fit1], return_counts=True))}

    m1 = fit_calibrated(X, sup, fit1, va, P1, "stage1")
    joblib.dump(m1, f"{OUT}/stage1_model.joblib")

    # Stage 1 predictions are needed on ALL folds: Stage 2 builds its training
    # population from fold 0 predictions and calibrates on fold 1 predictions.
    log("  predicting stage 1 on all rows")
    all_idx = np.arange(y.size)
    p1 = chunked_proba(m1, X, all_idx)
    s1_pred = m1.classes_[p1.argmax(1)].astype(np.int32)
    np.save(f"{OUT}/stage1_pred.npy", s1_pred)
    np.save(f"{OUT}/stage1_prob_test.npy", p1[te])
    econ_p_all = p1[:, list(m1.classes_).index(1)]
    del p1

    # ---------------- Stage 2 ----------------
    log("STAGE 2: subclass + sink")
    cand = np.flatnonzero(s1_pred == 1)          # everything Stage 1 called economic
    g_of = np.zeros(y.size, dtype=np.int32)
    for g, codes in GROUPS.items():
        g_of[np.isin(y, list(codes))] = g
    g_of[g_of == 0] = SINK                       # Q11-C: anything not one of the 13
    log(f"  candidates {cand.size:,}; sink share "
        f"{(g_of[cand] == SINK).mean():.3%}")

    c_tr = np.intersect1d(cand, tr, assume_unique=False)
    c_va = np.intersect1d(cand, va)
    c_te = np.intersect1d(cand, te)
    for nm, a in (("train", c_tr), ("val", c_va), ("test", c_te)):
        cnt = dict(zip(*np.unique(g_of[a], return_counts=True)))
        log(f"  {nm} candidates {a.size:,}  groups { {GNAME[int(k)]: int(v) for k, v in cnt.items()} }")
        assert a.size > 0, f"no stage-2 candidates in {nm}"
    assert (g_of[c_tr] == SINK).sum() > 0, "sink has no training rows"
    manifest["stage2_sink_train"] = int((g_of[c_tr] == SINK).sum())

    fit2 = cap(c_tr, g_of[c_tr], PER_GROUP_CAP)
    m2 = fit_calibrated(X, g_of, fit2, c_va, P23, "stage2")
    joblib.dump(m2, f"{OUT}/stage2_model.joblib")
    log("  predicting stage 2 on test candidates")
    p2 = chunked_proba(m2, X, c_te)
    np.save(f"{OUT}/stage2_prob_test.npy", p2)
    np.save(f"{OUT}/stage2_test_idx.npy", c_te)

    # ---------------- Stage 3 ----------------
    log("STAGE 3: crop within group (experts selected by TRUE membership)")
    experts, e_classes = {}, {}
    for g, codes in GROUPS.items():
        g_tr = np.intersect1d(tr, np.flatnonzero(np.isin(y, list(codes))))
        g_va = np.intersect1d(va, np.flatnonzero(np.isin(y, list(codes))))
        log(f"  {GNAME[g]}: train {g_tr.size:,} val {g_va.size:,} "
            f"classes {sorted(set(y[g_tr].tolist()))}")
        assert g_tr.size and g_va.size
        assert len(set(y[g_tr].tolist())) == len(codes), f"{GNAME[g]} lost a class in train"
        fit3 = cap(g_tr, y[g_tr], PER_LU_CAP)        # cap only; NO upsampling
        m3 = fit_calibrated(X, y, fit3, g_va, P23, f"stage3-{GNAME[g]}")
        joblib.dump(m3, f"{OUT}/stage3_{GNAME[g]}_model.joblib")
        experts[g], e_classes[g] = m3, m3.classes_

    # every expert scores every test candidate, which is what the joint rule needs
    p3 = {}
    for g in GROUPS:
        log(f"  predicting {GNAME[g]} on all test candidates")
        p3[g] = chunked_proba(experts[g], X, c_te)
        np.save(f"{OUT}/stage3_{GNAME[g]}_prob_test.npy", p3[g])

    # ---------------- compose on the TEST FOLD ONLY ----------------
    log("composing cascade on fold 2")
    hard = np.zeros(y.size, dtype=np.int32)
    g_hat = m2.classes_[p2.argmax(1)]
    for g in GROUPS:
        sel = g_hat == g
        if sel.any():
            hard[c_te[sel]] = e_classes[g][p3[g][sel].argmax(1)]
    # sink and non-economic stay 0

    # joint: P(crop) = P(econ) * P(group|econ) * P(crop|group), argmax over 13,
    # with an explicit not-a-crop mass so the rule can decline like hard routing.
    joint = np.zeros(y.size, dtype=np.int32)
    pe = econ_p_all[c_te]
    g_col = {g: list(m2.classes_).index(g) for g in m2.classes_}
    scores = np.zeros((c_te.size, len(CROPS)), dtype=np.float32)
    for j, code in enumerate(CROPS):
        g = next(g for g, cs in GROUPS.items() if code in cs)
        k = list(e_classes[g]).index(code)
        scores[:, j] = pe * p2[:, g_col[g]] * p3[g][:, k]
    not_crop = (1.0 - pe) + pe * p2[:, g_col[SINK]]
    best = scores.argmax(1)
    take = scores[np.arange(c_te.size), best] > not_crop
    joint[c_te[take]] = np.array(CROPS)[best[take]]

    np.save(f"{OUT}/pred_hard.npy", hard)
    np.save(f"{OUT}/pred_joint.npy", joint)

    # ---------------- score ----------------
    y_eval = np.where(np.isin(y, CROPS), y, 0).astype(np.int32)
    results = {}
    for nm, pred in (("hard", hard), ("joint", joint)):
        rep = classification_report(y_eval[te], pred[te], labels=CROPS,
                                    output_dict=True, zero_division=0)
        mf = rep["macro avg"]["f1-score"]
        wf = rep["weighted avg"]["f1-score"]
        log(f"  {nm}: rows {te.size:,}  macro F1 {mf:.4f}  weighted F1 {wf:.4f}")
        results[nm] = {"rows": int(te.size), "macro_f1": round(mf, 4),
                       "weighted_f1": round(wf, 4)}
        import csv
        with open(f"{OUT}/report_{nm}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["lu_code", "precision", "recall", "f1", "support"])
            for c in CROPS:
                r = rep[str(c)]
                w.writerow([c, round(r["precision"], 4), round(r["recall"], 4),
                            round(r["f1-score"], 4), int(r["support"])])

    # assertions that make the number trustworthy
    assert np.all(hard[tr] == 0) and np.all(hard[va] == 0), "predictions leaked outside fold 2"
    manifest["results"] = results
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log("wrote", f"{OUT}/manifest.json")
    log("DONE")
