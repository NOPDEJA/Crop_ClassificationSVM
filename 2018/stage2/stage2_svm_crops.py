# stage2_svm_crops.py
"""
Stage-2 crop classifier (within econ):
- Load aligned_features/svm_features_labels.npz (X, y)
- Keep only economic crop LU_CODEs (use economic_crops set)
- Optionally compute per-indicator mean/median per crop (saved CSV)
- Sample/balance per crop
- Train Nystroem->LinearSVC (OneVsRest) and calibrate
- Save model and evaluation CSV
"""

import os, json
import numpy as np
import joblib
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------
# Config -- edit as needed
# ------------------------
NPZ = "./aligned_features/svm_features_labels.npz"
RANDOM_STATE = 42

# Which LU_CODEs are economic crops (same as your list)
economic_crops = {2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420}

# Sampling / training params
MIN_PIXELS_PER_CROP = 100     # minimum pixels to keep crop for training
SAMPLES_PER_CROP = 2000       # sample per crop for training (if available)
UPSAMPLE_SMALL = True         # if True, upsample crops with <SAMPLES_PER_CROP with replacement
USE_PCA = False               # toggle PCA before Nystroem
PCA_NCOMP = 10

# RandomizedSearch params
NYST_COMPONENTS = [200, 400, 600]   # try 600 if you have RAM
NYST_GAMMA = [0.5, 1.0, 2.0]
SVC_C = [0.1, 1.0, 10.0]
N_ITER_SEARCH = 8
N_JOBS = 1

OUT_MODEL = "stage2_svm_crops.joblib"
OUT_REPORT = "stage2_eval.csv"
OUT_STATS = "stage2_indicator_stats.csv"

# ------------------------
# Helper functions
# ------------------------
def load_Xy(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    print("Loaded X", X.shape, "y", y.shape)
    return X, y

def filter_to_econ(X, y, econ_set):
    mask = np.isin(y, list(econ_set))
    X_e = X[mask]
    y_e = y[mask]
    print("Filtered econ pixels:", X_e.shape[0])
    return X_e, y_e

def compute_indicator_stats_per_crop(X, y, crop_codes, out_csv):
    # X shape (n, n_features), y shape (n,)
    rows = []
    for c in sorted(crop_codes):
        mask = y == c
        if mask.sum() == 0:
            continue
        mean = X[mask].mean(axis=0)
        med = np.median(X[mask], axis=0)
        row = {"LU_CODE": int(c), "count": int(mask.sum())}
        # add feature columns: F0_mean, F0_median, F1_mean...
        for i in range(X.shape[1]):
            row[f"F{i}_mean"] = float(mean[i])
            row[f"F{i}_median"] = float(med[i])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved indicator stats to", out_csv)

def sample_per_crop(X, y, samples_per_crop=SAMPLES_PER_CROP, upsample_small=UPSAMPLE_SMALL):
    rng = np.random.default_rng(RANDOM_STATE)
    crops = np.unique(y)
    parts_X = []
    parts_y = []
    for c in crops:
        idxs = np.flatnonzero(y == c)
        n_avail = idxs.size
        if n_avail == 0:
            continue
        if n_avail >= samples_per_crop:
            chosen = rng.choice(idxs, size=samples_per_crop, replace=False)
        else:
            if upsample_small:
                chosen = rng.choice(idxs, size=samples_per_crop, replace=True)
            else:
                chosen = idxs  # keep all if not upsampling
        parts_X.append(X[chosen])
        parts_y.append(y[chosen])
        print(f"Crop {int(c)}: available={n_avail}, sampled={chosen.size}")
    Xs = np.vstack(parts_X)
    ys = np.hstack(parts_y)
    # shuffle
    perm = rng.permutation(Xs.shape[0])
    return Xs[perm], ys[perm]

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    X, y = load_Xy(NPZ)

    # Filter to econ LU codes
    X_econ, y_econ = filter_to_econ(X, y, economic_crops)

    # Optionally compute indicator mean/median per crop for diagnostics
    compute_indicator_stats_per_crop(X_econ, y_econ, sorted(economic_crops), OUT_STATS)

    # Filter out crops with too few pixels
    uniques, counts = np.unique(y_econ, return_counts=True)
    keep = uniques[counts >= MIN_PIXELS_PER_CROP]
    print("Crops kept for training (min pixels):", keep)
    mask_keep = np.isin(y_econ, keep)
    X_train_all = X_econ[mask_keep]
    y_train_all = y_econ[mask_keep]

    # Sample/Balance
    Xs, ys = sample_per_crop(X_train_all, y_train_all, samples_per_crop=SAMPLES_PER_CROP, upsample_small=UPSAMPLE_SMALL)
    print("Sampled training shape:", Xs.shape, ys.shape, Counter(ys))

    # Train/test split (stratify by crop LU_CODE)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, ys, test_size=0.3, stratify=ys, random_state=RANDOM_STATE)
    print("Train/test sizes:", X_tr.shape[0], X_te.shape[0])

    # Build pipeline: scaler -> optional PCA -> nystroem -> linearSVC
    steps = [('scaler', StandardScaler())]
    if USE_PCA:
        steps.append(('pca', PCA(n_components=PCA_NCOMP, random_state=RANDOM_STATE)))
    steps.append(('nyst', Nystroem(kernel='rbf', random_state=RANDOM_STATE)))
    steps.append(('svc', LinearSVC(class_weight='balanced', max_iter=20000, random_state=RANDOM_STATE)))
    pipe = Pipeline(steps)

    ovr = OneVsRestClassifier(pipe)
    calibrated = CalibratedClassifierCV(estimator=ovr, cv=3, method='sigmoid')

    param_dist = {
        'estimator__estimator__nyst__n_components': NYST_COMPONENTS,
        'estimator__estimator__nyst__gamma': NYST_GAMMA,
        'estimator__estimator__svc__C': SVC_C
    }

    rsearch = RandomizedSearchCV(calibrated, param_distributions=param_dist, n_iter=N_ITER_SEARCH, cv=3, random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=2)
    print("Starting RandomizedSearchCV for stage-2 (this may take a while)...")
    rsearch.fit(X_tr, y_tr)
    print("Best params:", rsearch.best_params_)

    best_clf = rsearch.best_estimator_

    # Evaluate
    y_pred = best_clf.predict(X_te)
    report = classification_report(y_te, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(OUT_REPORT, index=True, encoding="utf-8-sig")
    print("Saved stage-2 eval to", OUT_REPORT)
    print("Confusion matrix:")
    print(confusion_matrix(y_te, y_pred))

    # Save model
    joblib.dump(best_clf, OUT_MODEL)
    print("Saved Stage-2 model to", OUT_MODEL)
