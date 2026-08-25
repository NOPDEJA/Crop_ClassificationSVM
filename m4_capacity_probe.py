"""m4_capacity_probe.py

M4 of docs/PLAN_S2_3DATE_COMPETITIVE.md: is kernel capacity the binding constraint?

Every hyperparameter search this project has run selected the MAXIMUM n_components
it was offered -- 250 at Stage 1, 600 at Stages 2 and 3, from grids topping out
there. A parameter pinned at its ceiling is not a converged choice; it is a grid
that stopped too early. This probes above the ceiling.

Bounded exactly as the plan requires: one expert at a time, at most two capacity
points above 600, no full cascade. Capacity is isolated cleanly here because
`gamma=None` resolves to 1/n_features, which depends on the COLUMN count and not
on n_components -- so unlike the M3 probe, nothing else moves.

Scored on the tuning half by raw decision function, at fixed 24 columns. This is
an expert-level number and is not comparable with an end-to-end one.

The plan also asks for a memory benchmark before an overnight run: a Nystroem
block is n_rows x n_components float64, materialised once per class by OneVsRest,
so the peak for the largest point is reported below in GB.

Env:
  GROUP=orchards|plantation|field   (default orchards -- where the dead crops are)
  POINTS=600,1200,1800              n_components to try, in order
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

from config import NPZ, RANDOM_STATE, PER_LU_CAP

SPLIT_ASSIGN = "./splits/split_assign.npy"
RUN = os.environ.get("RUN_DIR", "./runs/s2_2018_3date_parcel_m0")
GROUPS = {"orchards": [2403, 2404, 2407, 2413, 2416, 2419, 2420],
          "plantation": [2302, 2303, 2405],
          "field": [2101, 2204, 2205]}
EXPECTED_FIT = {"orchards": 172_371, "plantation": 148_344, "field": 210_000}
NM = {2403: "Durian", 2404: "Rambutan", 2407: "Mango", 2413: "Longan",
      2416: "Jackfruit", 2419: "Mangosteen", 2420: "Langsat",
      2302: "Rubber", 2303: "OilPalm", 2405: "Coconut",
      2101: "Rice", 2204: "Cassava", 2205: "Pineapple"}
GROUP = os.environ.get("GROUP", "orchards")
CODES = GROUPS[GROUP]
POINTS = [int(v) for v in os.environ.get("POINTS", "600,1200,1800").split(",")]
C_FIXED = 10.0

rng = np.random.default_rng(RANDOM_STATE)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def fit_score(X, y, fit_idx, tune_idx, n_comp):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("nyst", Nystroem(kernel="rbf", n_components=n_comp, gamma=None,
                          random_state=RANDOM_STATE)),
        ("svc", LinearSVC(C=C_FIXED, class_weight=None, max_iter=5000,
                          random_state=RANDOM_STATE)),
    ])
    m = OneVsRestClassifier(pipe)
    gb = fit_idx.size * n_comp * 8 / 1e9
    log(f"  n_components={n_comp}: fitting on {fit_idx.size:,} rows "
        f"(Nystroem block {gb:.2f} GB) ...")
    t0 = time.time()
    m.fit(X[fit_idx], y[fit_idx])
    mins = (time.time() - t0) / 60
    pred = m.classes_[m.decision_function(X[tune_idx]).argmax(1)]
    f = f1_score(y[tune_idx], pred, labels=CODES, average=None, zero_division=0)
    log(f"  n_components={n_comp}: {mins:.1f} min, macro {f.mean():.4f}")
    return f, mins, gb


if __name__ == "__main__":
    log(f"M4 capacity probe: {GROUP} expert, n_components {POINTS}, 24 columns")
    d = np.load(NPZ, allow_pickle=True)
    X = d["X"]
    y = d["y"].astype(np.int32)
    assert X.shape[1] == 24, f"expected the 24-column matrix, got {X.shape}"

    asg = np.load(SPLIT_ASSIGN)
    tr = np.flatnonzero(asg == 0)
    tune = np.load(f"{RUN}/val_tune_idx.npy")
    own = np.flatnonzero(np.isin(y, CODES))
    g_tr = np.intersect1d(tr, own)
    g_tune = np.intersect1d(tune, own)

    parts = []
    for c in CODES:
        w = g_tr[y[g_tr] == c]
        parts.append(w if w.size <= PER_LU_CAP else rng.choice(w, PER_LU_CAP, replace=False))
    fit_idx = np.sort(np.concatenate(parts))
    assert fit_idx.size == EXPECTED_FIT[GROUP], f"fit set changed: {fit_idx.size}"
    log(f"fit {fit_idx.size:,} | tune {g_tune.size:,}")

    res = {}
    for n_comp in POINTS:
        res[n_comp] = fit_score(X, y, fit_idx, g_tune, n_comp)

    print(f"\n{'crop':<12}" + "".join(f"{n:>10}" for n in POINTS) + "   tune support")
    for i, c in enumerate(CODES):
        n_px = int((y[g_tune] == c).sum())
        print(f"{NM[c]:<12}" + "".join(f"{res[n][0][i]:>10.4f}" for n in POINTS)
              + f"   {n_px:,}")
    print(f"{'MACRO':<12}" + "".join(f"{res[n][0].mean():>10.4f}" for n in POINTS))
    print(f"{'minutes':<12}" + "".join(f"{res[n][1]:>10.1f}" for n in POINTS))
    print(f"{'block GB':<12}" + "".join(f"{res[n][2]:>10.2f}" for n in POINTS))

    base = res[POINTS[0]][0].mean()
    best_n = max(POINTS, key=lambda n: res[n][0].mean())
    print(f"\n  best {best_n} at {res[best_n][0].mean():.4f} "
          f"({res[best_n][0].mean() - base:+.4f} vs {POINTS[0]})")
    if best_n == POINTS[-1]:
        print("  STILL AT THE CEILING -- report the bound, do not escalate further "
              "(the plan's instruction, and the cost grows superlinearly)")
    else:
        print("  interior optimum: capacity is no longer the binding constraint")
