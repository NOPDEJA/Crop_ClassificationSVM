"""m3_ablation_orchards.py

Cheap go/no-go on the M3 features before committing to M5's ~6 h retrain.

Trains the Stage-3 ORCHARDS expert alone -- 24 columns vs 30 -- and scores it on
the tuning half of fold 1. Orchards is the right probe: it holds six of the seven
crops that sit near zero end to end, and red-edge chlorophyll (MTCI) is exactly
the signal that should separate tree crops if anything does. Two fits of ~15 min
against a full cascade retrain.

This measures the EXPERT in isolation, on rows whose true crop is in the group.
It is not an end-to-end number and must not be compared with one: no Stage-1 or
Stage-2 error reaches it. It answers only "do these six columns carry usable
signal for the orchard species", which is the question that gates M5.

Note a deliberate confound: P23 sets gamma=None, which sklearn reads as
1/n_features, so 24 -> 30 columns also moves gamma from 0.0417 to 0.0333. That is
not isolated here because it is not isolated in M5 either -- the same substitution
happens there -- so this reproduces the decision actually being made. M4 is where
capacity and gamma get examined on their own.

Env:
  RUN_DIR=<dir>   where val_cal_idx.npy / val_tune_idx.npy live
"""
import os
import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from config import RANDOM_STATE, PER_LU_CAP

M3_NPZ = "./aligned_features/svm_s2_3date_m3_features_labels.npz"
SPLIT_ASSIGN = "./splits/split_assign.npy"
RUN = os.environ.get("RUN_DIR", "./runs/s2_2018_3date_parcel_m0")
ORCHARDS = [2403, 2404, 2407, 2413, 2416, 2419, 2420]
NM = {2403: "Durian", 2404: "Rambutan", 2407: "Mango", 2413: "Longan",
      2416: "Jackfruit", 2419: "Mangosteen", 2420: "Langsat"}
P23 = dict(n_components=600, gamma=None, C=10.0)
CALIB_MAX = 300_000

rng = np.random.default_rng(RANDOM_STATE)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def fit_and_score(X, y, fit_idx, tune_idx, tag):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("nyst", Nystroem(kernel="rbf", n_components=P23["n_components"],
                          gamma=P23["gamma"], random_state=RANDOM_STATE)),
        ("svc", LinearSVC(C=P23["C"], class_weight=None, max_iter=5000,
                          random_state=RANDOM_STATE)),
    ])
    m = OneVsRestClassifier(pipe)
    log(f"  {tag}: fitting on {fit_idx.size:,} x {X.shape[1]} ...")
    t0 = time.time()
    m.fit(X[fit_idx], y[fit_idx])
    log(f"  {tag}: fitted in {(time.time() - t0) / 60:.1f} min")
    pred = m.classes_[m.decision_function(X[tune_idx]).argmax(1)]
    f = f1_score(y[tune_idx], pred, labels=ORCHARDS, average=None, zero_division=0)
    return f


if __name__ == "__main__":
    log("M3 ablation: orchards expert, 24 vs 30 columns")
    d = np.load(M3_NPZ, allow_pickle=True)
    X30 = d["X"]
    y = d["y"].astype(np.int32)
    names = list(d["feature_names"].astype(str))
    assert X30.shape[1] == 30, X30.shape
    log(f"loaded {X30.shape}; new columns: {names[24:]}")

    asg = np.load(SPLIT_ASSIGN)
    tr = np.flatnonzero(asg == 0)
    cal = np.load(f"{RUN}/val_cal_idx.npy")
    tune = np.load(f"{RUN}/val_tune_idx.npy")

    own = np.isin(y, ORCHARDS)
    g_tr = np.intersect1d(tr, np.flatnonzero(own))
    g_cal = np.intersect1d(cal, np.flatnonzero(own))
    g_tune = np.intersect1d(tune, np.flatnonzero(own))

    parts = []
    for c in ORCHARDS:
        w = g_tr[y[g_tr] == c]
        parts.append(w if w.size <= PER_LU_CAP else rng.choice(w, PER_LU_CAP, replace=False))
    fit_idx = np.sort(np.concatenate(parts))
    log(f"fit {fit_idx.size:,} | cal {g_cal.size:,} | tune {g_tune.size:,}")
    assert fit_idx.size == 172_371, f"fit set changed: {fit_idx.size}"

    res = {}
    # scored on raw decision functions: no calibrator between the features
    # and the verdict, so this measures the columns and nothing else
    res[24] = fit_and_score(X30[:, :24], y, fit_idx, g_tune, "24-col")
    res[30] = fit_and_score(X30, y, fit_idx, g_tune, "30-col")

    print(f"\n{'crop':<12}{'24 col':>9}{'30 col':>9}{'delta':>9}   tune support")
    for i, c in enumerate(ORCHARDS):
        n = int((y[g_tune] == c).sum())
        print(f"{NM[c]:<12}{res[24][i]:>9.4f}{res[30][i]:>9.4f}"
              f"{res[30][i] - res[24][i]:>+9.4f}   {n:,}")
    print(f"{'MACRO-7':<12}{res[24].mean():>9.4f}{res[30].mean():>9.4f}"
          f"{res[30].mean() - res[24].mean():>+9.4f}")
    verdict = "PROCEED to M5 with 30 columns" if res[30].mean() > res[24].mean() \
        else "the six columns do not help this expert -- reconsider before M5"
    print(f"\nverdict: {verdict}")
