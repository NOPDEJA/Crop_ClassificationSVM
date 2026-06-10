# stage1_svm_improved.py
"""
Improved Stage-1 SVM pipeline (Kernel-approx + LinearSVC) with:
 - post-sampling rebalancing (caps per super-class)
 - optional PCA
 - RandomizedSearchCV for hyperparams
 - validation-based probability threshold tuning (econ, water)
 - evaluation saved to CSV, model saved to disk
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
NPZ = "./aligned_features/svm_new_features_labels.npz"
RANDOM_STATE = 42

# sampling defaults (per-LU code)
SAMPLES_PER_LU = 5000
MIN_CLASS_PIXELS = 200

# rebalance caps AFTER mapping to super-classes
CAP_ECON = 50000     # maximum econ samples to keep (set None to keep all)
CAP_WATER = 15000     # maximum water samples
CAP_OTHERS = 50000    # maximum others samples

# pipeline options
USE_PCA = True
PCA_NCOMP = 10        # if USE_PCA: target components (or None to use variance)
N_ITER_SEARCH = 10    # randomized search iterations (increase for more tuning)
N_JOBS = 1

OUT_MODEL = "stage1_svm_improved_fixed2.joblib"
OUT_REPORT_CSV = "stage1_svm_improved_fixed_report2.csv"
OUT_THRESH_JSON = "stage1_thresholds_fixed2.json"

# nystroem/grid options
NYST_COMPONENTS_CANDIDATES = [200, 400, 600]   # add 800 if you have RAM/time
NYST_GAMMA_CANDIDATES = [0.5, 1.0, 2.0]
SVC_C_CANDIDATES = [0.1, 1.0, 10.0]

# mapping sets
economic_crops = {2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420}
water_code = {4101,4102,4103,4201,4202,4203}

# -----------------------
# Helpers
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

def rebalance_by_superclass(Xs, ys_codes, ys_super, cap_econ=CAP_ECON, cap_water=CAP_WATER, cap_others=CAP_OTHERS):
    rng = np.random.default_rng(RANDOM_STATE)
    idx_econ = np.flatnonzero(ys_super == 1)
    idx_water = np.flatnonzero(ys_super == 2)
    idx_others = np.flatnonzero(ys_super == 3)

    print("Pre-rebalance counts:", "econ", len(idx_econ), "water", len(idx_water), "others", len(idx_others))

    def cap_indices(idxs, cap):
        if cap is None:
            return idxs
        cap_n = min(len(idxs), cap)
        return rng.choice(idxs, size=cap_n, replace=False)

    idx_econ2 = cap_indices(idx_econ, cap_econ)
    idx_water2 = cap_indices(idx_water, cap_water)
    idx_others2 = cap_indices(idx_others, cap_others)

    chosen = np.concatenate([idx_econ2, idx_water2, idx_others2])
    rng.shuffle(chosen)

    Xb = Xs[chosen]
    y_codes_b = ys_codes[chosen]
    y_super_b = ys_super[chosen]

    print("Post-rebalance counts:", Counter(y_super_b))
    return Xb, y_codes_b, y_super_b

def save_report_dict_to_csv(report_dict, csv_path):
    # report_dict: sklearn classification_report(..., output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(csv_path, index=True, encoding="utf-8-sig")
    print("Saved report:", csv_path)

def apply_thresholds_from_probs(probs, classes, thresh_map):
    """
    probs: (n_samples, n_classes) in same order as classes
    thresh_map: dict mapping class_value (1,2,3) -> threshold (0..1) or None
    logic:
      - by default label = argmax(probs) (map to classes)
      - then for each class with threshold t: if probs[:,idx] > t -> label = class_value
      - if multiple thresholds true, pick class with highest prob among them.
    """
    idx_map = {c: i for i, c in enumerate(classes)}
    default_labels = classes[np.argmax(probs, axis=1)]
    labels = default_labels.copy()
    # check thresholds
    candidate_mask = np.zeros_like(probs, dtype=bool)
    for cls_val, t in thresh_map.items():
        if t is None:
            continue
        i = idx_map.get(cls_val)
        if i is None:
            continue
        candidate_mask[:, i] = probs[:, i] > t
    # resolve candidates
    any_candidate = candidate_mask.any(axis=1)
    if any_candidate.any():
        # choose candidate with highest prob among candidate columns
        cand_probs = np.where(candidate_mask, probs, -1.0)  # -1 where not candidate
        best_idx = np.argmax(cand_probs, axis=1)
        # only update rows where any_candidate True
        for r in np.where(any_candidate)[0]:
            labels[r] = classes[best_idx[r]]
    return labels

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    print("=== Stage-1 improved SVM pipeline ===")
    Xs, ys_codes, ys_super = load_and_sample_per_lu(NPZ)

    # rebalance at super-class level
    Xb, y_codes_b, y_super_b = rebalance_by_superclass(Xs, ys_codes, ys_super,
                                                       cap_econ=CAP_ECON, cap_water=CAP_WATER, cap_others=CAP_OTHERS)

    # Split into train / val / test (stratified by super-class)
    # First split train vs rest (train 70%, rest 30%), then split rest into val/test 50/50 => val 15%, test 15%
    X_train, X_rest, y_train, y_rest = train_test_split(Xb, y_super_b, test_size=0.3, stratify=y_super_b, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE)
    print("Split sizes: train", X_train.shape[0], "val", X_val.shape[0], "test", X_test.shape[0], "class dist (train):", Counter(y_train))

    # build pipeline steps list
    steps = []
    steps.append(('scaler', StandardScaler()))
    if USE_PCA:
        steps.append(('pca', PCA(n_components=PCA_NCOMP, random_state=RANDOM_STATE)))
    # add placeholder Nystroem + SVC in pipeline via param grid
    nyst = Nystroem(kernel='rbf', random_state=RANDOM_STATE)
    svc_lin = LinearSVC(class_weight='balanced', max_iter=20000, random_state=RANDOM_STATE)
    steps.append(('nyst', nyst))
    steps.append(('svc', svc_lin))
    pipe = Pipeline(steps)

    ovr = OneVsRestClassifier(pipe)
    calibrated = CalibratedClassifierCV(estimator=ovr, cv=3, method='sigmoid')

    # param grid: note nested param names: estimator__estimator__<step>__param
    param_dist = {
        'estimator__estimator__nyst__n_components': NYST_COMPONENTS_CANDIDATES,
        'estimator__estimator__nyst__gamma': NYST_GAMMA_CANDIDATES,
        'estimator__estimator__svc__C': SVC_C_CANDIDATES
    }

    print("Starting RandomizedSearchCV...")
    rsearch = RandomizedSearchCV(
        calibrated,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=2
    )
    rsearch.fit(X_train, y_train)
    print("Randomized search done. Best params:", rsearch.best_params_)

    best_clf = rsearch.best_estimator_

    # Evaluate on validation set and tune thresholds for econ (1) and water (2)
    val_probs = best_clf.predict_proba(X_val)   # shape (n_val, n_classes)
    classes = best_clf.classes_                 # e.g. array([1,2,3])
    print("Model classes order:", classes)

    # search thresholds grid for econ and water
    thresh_grid = np.linspace(0.4, 0.95, 56)  # 0.4..0.95 step ~0.01
    best_thresh = {int(classes[0]): None, int(classes[1]): None, int(classes[2]): None}
    # Find best threshold for econ class (value 1) and water (2) independently maximizing F1 for that class
    # We'll do independent search: fix others as None
    for cls_val in [1, 2]:
        best_f1 = -1.0
        best_t = None
        for t in thresh_grid:
            thr_map = {1: None, 2: None, 3: None}
            thr_map[cls_val] = t
            y_val_pred = apply_thresholds_from_probs(val_probs, classes, thr_map)
            # compute f1 for the target class
            f1 = f1_score(y_val, y_val_pred, labels=[cls_val], average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresh[cls_val] = best_t
        print(f"Best threshold for class {cls_val} = {best_t} (val f1={best_f1:.3f})")

    # Save thresholds
    with open(OUT_THRESH_JSON, 'w') as fj:
        json.dump(best_thresh, fj)
    print("Saved thresholds to", OUT_THRESH_JSON)

    # Apply thresholds to test set and evaluate
    test_probs = best_clf.predict_proba(X_test)
    y_test_pred_thresh = apply_thresholds_from_probs(test_probs, classes, best_thresh)

    # Final report
    report = classification_report(y_test, y_test_pred_thresh, target_names=["econ", "water", "others"], output_dict=True)
    print("Final classification report on test set (after thresholding):")
    print(classification_report(y_test, y_test_pred_thresh, target_names=["econ", "water", "others"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred_thresh))

    # save CSV report
    save_report_dict_to_csv(report, OUT_REPORT_CSV)

    # Save best calibrated pipeline
    joblib.dump(best_clf, OUT_MODEL)
    print("Saved model to", OUT_MODEL)
