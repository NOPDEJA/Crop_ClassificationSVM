"""Separate the M3 feature effect from the gamma side-effect.

P23 sets gamma=None -> 1/n_features, so 24 -> 30 columns silently moved gamma
0.041667 -> 0.033333. The plantation ablation showed +0.0046 macro. This asks
which change produced it, by fitting 30 columns with gamma PINNED at the
24-column value.

  30 col, gamma=1/30 (as run)  -> 0.4591
  30 col, gamma=1/24 (pinned)  -> ?
  24 col, gamma=1/24 (as run)  -> 0.4545

If the pinned run lands near 0.4591 the features did it; near 0.4545 and gamma
did. Plantation is the probe because its three classes carry 695-732,519 px, so
the differences are not noise.
"""
import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

NPZ30 = "./aligned_features/svm_s2_3date_m3_features_labels.npz"
RUN = "./runs/s2_2018_3date_parcel_m0"
CODES = [2302, 2303, 2405]
NM = {2302: "Rubber", 2303: "OilPalm", 2405: "Coconut"}
PER_LU_CAP = 70_000
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


d = np.load(NPZ30, allow_pickle=True)
X = d["X"]
y = d["y"].astype(np.int32)
asg = np.load("./splits/split_assign.npy")
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
assert fit_idx.size == 148_344, fit_idx.size
log(f"fit {fit_idx.size:,} | tune {g_tune.size:,}")

for tag, gamma in (("30col gamma=1/24 (pinned)", 1.0 / 24), ("30col gamma=1/30 (as run)", None)):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("nyst", Nystroem(kernel="rbf", n_components=600, gamma=gamma,
                          random_state=RANDOM_STATE)),
        ("svc", LinearSVC(C=10.0, class_weight=None, max_iter=5000,
                          random_state=RANDOM_STATE)),
    ])
    m = OneVsRestClassifier(pipe)
    t0 = time.time()
    m.fit(X[fit_idx], y[fit_idx])
    pred = m.classes_[m.decision_function(X[g_tune]).argmax(1)]
    f = f1_score(y[g_tune], pred, labels=CODES, average=None, zero_division=0)
    log(f"{tag}: {(time.time() - t0) / 60:.1f} min  macro {f.mean():.4f}  "
        + "  ".join(f"{NM[c]} {v:.4f}" for c, v in zip(CODES, f)))

print("\nreference: 24col gamma=1/24 -> 0.4545 ; 30col gamma=1/30 -> 0.4591")
