# stage1_svm_increase_econ2.py
"""
Stage-1 SVM improvement with increased ECON support.
- Same pipeline as stage1_svm_improved.py but econ class will be upsampled
  (with replacement) if available econ samples < TARGET_ECON_SUPPORT.
"""

import os
import json
import numpy as np
import joblib
import csv
from collections import Counter

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pandas as pd

# -----------------------
# User-configurable settings
# -----------------------
NPZ = "./aligned_features/svm_features_labels.npz"
RANDOM_STATE = 42

# sampling defaults (per-LU code)
SAMPLES_PER_LU = 5000
MIN_CLASS_PIXELS = 200

# rebalance caps AFTER mapping to super-classes
# econ will be upsampled to TARGET_ECON_SUPPORT if fewer examples available
TARGET_ECON_SUPPORT = 50000  # <-- increase this to increase econ support
CAP_WATER = 15000             # cap for water (None to keep all)
CAP_OTHERS = 50000            # cap for others (None to keep all)

# pipeline options
USE_PCA = True
PCA_NCOMP = 10
N_ITER_SEARCH = 10
N_JOBS = 1

OUT_MODEL = "stage1_svm_change_econ_sample3.joblib"
OUT_REPORT_CSV = "stage1_svm_change_econ_sample_report3.csv"
OUT_THRESH_JSON = "stage1_change_econ_sample_thresholds3.json"

# nystroem/grid options
NYST_COMPONENTS_CANDIDATES = [400, 600, 800]
NYST_GAMMA_CANDIDATES = [0.5, 1.0, 2.0]
SVC_C_CANDIDATES = [0.1, 1.0, 10.0]

# mapping sets
economic_crops = {2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420}
water_code = {4101,4102,4103,4201,4202,4203}

# -----------------------
# Helpers (mostly same as before)
# -----------------------
def load_and_sample_per_lu(npz_path, samples_per_lu=SAMPLES_PER_LU, min_pixels=MIN_CLASS_PIXELS):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    print(f"Loaded X {X.shape}, y {y.shape}")

    # remove nodata
    mask_valid = (y != 0) & (y != 32767)
    Xv, yv = X[mask_valid], y[mask_valid]
    print("Valid labeled pixels:", Xv.shape[0])

    # filter rare LU_CODEs
    uniques, counts = np.unique(yv, return_counts=True)
    keep_codes = uniques[counts >= min_pixels]
    print(f"Keeping {len(keep_codes)} LU_CODEs (>= {min_pixels} pixels)")

    keep_mask = np.isin(yv, keep_codes)
    Xf, yf = Xv[keep_mask], yv[keep_mask]

    rng = np.random.default_rng(RANDOM_STATE)
    sampled_X_parts = []
    sampled_y_parts = []
    for code in keep_codes:
        idxs = np.flatnonzero(yf == code)
        n_take = min(samples_per_lu, idxs.size)
        chosen = rng.choice(idxs, size=n_take, replace=False)
        sampled_X_parts.append(Xf[chosen])
        sampled_y_parts.append(yf[chosen])
        print(f"LU {int(code)}: available={idxs.size}, sampled={n_take}")

    Xs = np.vstack(sampled_X_parts)
    ys_codes = np.hstack(sampled_y_parts)

    # map to super-class (1 econ, 2 water, 3 others)
    ys_super = np.array([1 if c in economic_crops else (2 if c in water_code else 3) for c in ys_codes], dtype=np.int32)
    return Xs, ys_codes, ys_super

def rebalance_and_upsample_econ(Xs, ys_codes, ys_super,
                                target_econ=TARGET_ECON_SUPPORT,
                                cap_water=CAP_WATER,
                                cap_others=CAP_OTHERS):
    """
    Rebalance by:
      - capping water/others to caps if provided
      - upsampling econ (with replacement) to reach target_econ if necessary
    Returns balanced X, y_codes, y_super
    """
    rng = np.random.default_rng(RANDOM_STATE)
    idx_econ = np.flatnonzero(ys_super == 1)
    idx_water = np.flatnonzero(ys_super == 2)
    idx_others = np.flatnonzero(ys_super == 3)

    print("Pre-rebalance counts:", "econ", len(idx_econ), "water", len(idx_water), "others", len(idx_others))

    # cap water/others
    def cap_choice(idxs, cap):
        if cap is None:
            return idxs
        if len(idxs) <= cap:
            return idxs
        return rng.choice(idxs, size=cap, replace=False)

    idx_water2 = cap_choice(idx_water, cap_water)
    idx_others2 = cap_choice(idx_others, cap_others)

    # upsample econ to target (with replacement) if needed
    if target_econ is None:
        idx_econ2 = idx_econ
    else:
        if len(idx_econ) >= target_econ:
            idx_econ2 = rng.choice(idx_econ, size=target_econ, replace=False)
        else:
            # upsample with replacement to reach target
            extras = rng.choice(idx_econ, size=(target_econ - len(idx_econ)), replace=True)
            idx_econ2 = np.concatenate([idx_econ, extras])

    chosen = np.concatenate([idx_econ2, idx_water2, idx_others2])
    rng.shuffle(chosen)

    Xb = Xs[chosen]
    y_codes_b = ys_codes[chosen]
    y_super_b = ys_super[chosen]

    print("Post-rebalance counts:", Counter(y_super_b))
    return Xb, y_codes_b, y_super_b

def save_report_dict_to_csv(report_dict, csv_path):
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(csv_path, index=True, encoding="utf-8-sig")
    print("Saved report:", csv_path)

def apply_thresholds_from_probs(probs, classes, thresh_map):
    idx_map = {c: i for i, c in enumerate(classes)}
    default_labels = classes[np.argmax(probs, axis=1)]
    labels = default_labels.copy()
    candidate_mask = np.zeros_like(probs, dtype=bool)
    for cls_val, t in thresh_map.items():
        if t is None:
            continue
        i = idx_map.get(cls_val)
        if i is None:
            continue
        candidate_mask[:, i] = probs[:, i] > t
    any_candidate = candidate_mask.any(axis=1)
    if any_candidate.any():
        cand_probs = np.where(candidate_mask, probs, -1.0)
        best_idx = np.argmax(cand_probs, axis=1)
        for r in np.where(any_candidate)[0]:
            labels[r] = classes[best_idx[r]]
    return labels

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    print("=== Stage-1 SVM (change econ sample) ===")
    Xs, ys_codes, ys_super = load_and_sample_per_lu(NPZ)

    # rebalance & upsample econ
    Xb, y_codes_b, y_super_b = rebalance_and_upsample_econ(Xs, ys_codes, ys_super,
                                                           target_econ=TARGET_ECON_SUPPORT,
                                                           cap_water=CAP_WATER,
                                                           cap_others=CAP_OTHERS)

    # train/val/test split
    X_train, X_rest, y_train, y_rest = train_test_split(Xb, y_super_b, test_size=0.3, stratify=y_super_b, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE)
    print("Split sizes: train", X_train.shape[0], "val", X_val.shape[0], "test", X_test.shape[0],
          "class dist (train):", Counter(y_train))

    # pipeline
    steps = []
    steps.append(('scaler', StandardScaler()))
    if USE_PCA:
        steps.append(('pca', PCA(n_components=PCA_NCOMP, random_state=RANDOM_STATE)))
    nyst = Nystroem(kernel='rbf', random_state=RANDOM_STATE)
    svc_lin = LinearSVC(class_weight='balanced', max_iter=20000, random_state=RANDOM_STATE)
    steps.append(('nyst', nyst))
    steps.append(('svc', svc_lin))
    pipe = Pipeline(steps)

    ovr = OneVsRestClassifier(pipe)
    calibrated = CalibratedClassifierCV(estimator=ovr, cv=3, method='sigmoid')

    param_dist = {
        'estimator__estimator__nyst__n_components': NYST_COMPONENTS_CANDIDATES,
        'estimator__estimator__nyst__gamma': NYST_GAMMA_CANDIDATES,
        'estimator__estimator__svc__C': SVC_C_CANDIDATES
    }

    print("Starting RandomizedSearchCV...")
    rsearch = RandomizedSearchCV(calibrated, param_distributions=param_dist, n_iter=N_ITER_SEARCH,
                                 cv=3, random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=2)
    rsearch.fit(X_train, y_train)
    print("Randomized search done. Best params:", rsearch.best_params_)

    best_clf = rsearch.best_estimator_

    # validation probs and threshold search
    val_probs = best_clf.predict_proba(X_val)
    classes = best_clf.classes_
    print("Model classes order:", classes)

    thresh_grid = np.linspace(0.2, 0.95, 76)   # wider search (start at 0.2)
    best_thresh = {int(classes[0]): None, int(classes[1]): None, int(classes[2]): None}
    for cls_val in [1, 2]:
        best_f1 = -1.0
        best_t = None
        for t in thresh_grid:
            thr_map = {1: None, 2: None, 3: None}
            thr_map[cls_val] = t
            y_val_pred = apply_thresholds_from_probs(val_probs, classes, thr_map)
            f1 = f1_score(y_val, y_val_pred, labels=[cls_val], average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresh[cls_val] = best_t
        print(f"Best threshold for class {cls_val} = {best_t} (val f1={best_f1:.3f})")

    # save thresholds
    with open(OUT_THRESH_JSON, 'w') as fj:
        json.dump(best_thresh, fj)
    print("Saved thresholds to", OUT_THRESH_JSON)

    # Apply thresholds + evaluate on test set
    test_probs = best_clf.predict_proba(X_test)
    y_test_pred_thresh = apply_thresholds_from_probs(test_probs, classes, best_thresh)

    report = classification_report(y_test, y_test_pred_thresh, target_names=["econ", "water", "others"], output_dict=True)
    print("Final classification report on test set (after thresholding):")
    print(classification_report(y_test, y_test_pred_thresh, target_names=["econ", "water", "others"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred_thresh))

    # save CSV report
    save_report_dict_to_csv(report, OUT_REPORT_CSV)

    # Save model
    joblib.dump(best_clf, OUT_MODEL)
    print("Saved model to", OUT_MODEL)
