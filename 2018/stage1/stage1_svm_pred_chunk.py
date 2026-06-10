"""stage1_svm_pred_chunk.py

Stage-1 3-class SVM pipeline integrating improved sampling logic:
 - per-LU sampling (limit per LU_CODE via SAMPLES_PER_LU)
 - super-class rebalance (caps per super-class before split)
 - no post-split up/down sampling (natural class distribution retained)
 - kernel approximation (Nystroem) + LinearSVC wrapped in OneVsRest + calibration
 - randomized hyperparameter search
 - validation threshold tuning (econ=1, water=2) with simple independent search
 - test evaluation + extended train/test support stats in CSV
 - chunked full-dataset prediction to .npy (memory safe)

Configuration NOTE: TARGET_SUPERCLASS_SUPPORT removed (no longer used); training now uses natural counts after rebalance.
"""
"""
Stage-1: 3-class SVM with:
 - per-LU sampling
 - caps per super-class (pre-split)
 - train-only upsampling to TARGET_SUPERCLASS_SUPPORT (default 6000)
 - RandomizedSearchCV + CalibratedClassifierCV
 - validation threshold tuning (econ, water)
 - chunked full-data prediction saved to .npy
 - CSV report that includes train_unique_support & train_final_support & test_support
"""
import os, json
import numpy as np
import joblib
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
# Config - edit if needed
# -----------------------
NPZ = "./aligned_features/svm_new_features_labels.npz"
RANDOM_STATE = 42

# per-LU sampling before mapping to super-classes
SAMPLES_PER_LU = 5000
MIN_CLASS_PIXELS = 200

# Caps (pre-split unique samples per superclass)
CAP_ECON = 50000      # keep up to this many unique econ samples before split
CAP_WATER = 15000
CAP_OTHERS = 50000

## (Removed) TARGET_SUPERCLASS_SUPPORT: previous upsample target; kept commented for reference.
# TARGET_SUPERCLASS_SUPPORT = 6000

# Pipeline / search settings
USE_PCA = True
PCA_NCOMP = 10
NYST_COMPONENTS_CANDIDATES = [200, 400, 600]
NYST_GAMMA_CANDIDATES = [0.5, 1.0, 2.0]
SVC_C_CANDIDATES = [0.1, 1.0, 10.0]
N_ITER_SEARCH = 8
N_JOBS = 4

# Prediction chunk
PRED_CHUNK = 1_000_000

# Outputs
OUT_MODEL = "stage1_svm_pred_chunk_new_features.joblib"
OUT_THRESH = "stage1_thresholds_pred_chunk_new_features.json"
OUT_REPORT = "stage1_report_pred_chunk_new_features.csv"
OUT_PRED_NPY = "stage1_pred_new_features.npy"
OUT_PROB_NPY = "stage1_prob_new_features.npy"

# mapping sets

economic_crops = {2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420}
water_code = {4101,4102,4103,4201,4202,4203}

# Rebalance function from improved script
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


# -----------------------
# Helpers
# -----------------------
def load_and_sample_per_lu(npz_path, samples_per_lu=SAMPLES_PER_LU, min_pixels=MIN_CLASS_PIXELS):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int32)
    print(f"Loaded X {X.shape}, y {y.shape}")

    mask_valid = (y != 0) & (y != 32767)
    Xv, yv = X[mask_valid], y[mask_valid]
    print("Valid labeled pixels:", Xv.shape[0])

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
    ys_super = np.array([1 if c in economic_crops else (2 if c in water_code else 3) for c in ys_codes], dtype=np.int32)
    return Xs, ys_codes, ys_super

## Removed legacy functions: cap_indices_pre_split, upsample_train_superclass (no longer needed with direct rebalance)

def save_report_with_traininfo(report_dict, train_unique_counts, train_final_counts, test_counts, out_csv):
    # report_dict from classification_report(output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    # add train/test support columns where index are class strings 'econ' etc. We'll map numeric classes to names
    name_map = {"econ":1, "water":2, "others":3}
    train_unique = []
    train_final = []
    test_supp = []
    for idx in df.index:
        if idx in name_map:
            cls = name_map[idx]
            train_unique.append(train_unique_counts.get(cls, 0))
            train_final.append(train_final_counts.get(cls, 0))
            test_supp.append(test_counts.get(cls, 0))
        else:
            train_unique.append("")
            train_final.append("")
            test_supp.append("")
    df["train_unique_support"] = train_unique
    df["train_final_support"] = train_final
    df["test_support"] = test_supp
    df.to_csv(out_csv, index=True, encoding="utf-8-sig")
    print("Saved report:", out_csv)

def chunked_predict_and_save(model, npz_path, out_pred=OUT_PRED_NPY, out_prob=OUT_PROB_NPY, chunk_size=PRED_CHUNK):
    data = np.load(npz_path, allow_pickle=True)
    X_all = data["X"].astype(np.float32)
    n = X_all.shape[0]
    preds = np.zeros(n, dtype=np.uint8)
    probs = np.zeros((n, model.classes_.shape[0]), dtype=np.float32)
    print(f"Starting chunked prediction with chunk_size={chunk_size}. Monitor Task Manager (Memory).")
    for s in range(0, n, chunk_size):
        e = min(n, s+chunk_size)
        Xc = X_all[s:e]
        preds[s:e] = model.predict(Xc).astype(np.uint8)
        probs[s:e, :] = model.predict_proba(Xc).astype(np.float32)
        print(f"  predicted chunk {s}:{e} ({e-s} rows)")
    np.save(out_pred, preds)
    np.save(out_prob, probs)
    print("Saved stage1 pred/prob:", out_pred, out_prob)


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    print("=== Stage-1 (per-LU sample + superclass rebalance + calibrated SVM) ===")

    Xs, ys_codes, ys_super = load_and_sample_per_lu(NPZ)
    # rebalance at super-class level (after per-LU sampling)
    Xc, y_codes_c, y_super_c = rebalance_by_superclass(Xs, ys_codes, ys_super,
                                                      cap_econ=CAP_ECON, cap_water=CAP_WATER, cap_others=CAP_OTHERS)
    print("After rebalance dataset size:", Xc.shape)

    # Split (train 70% / val 15% / test 15%)
    X_train, X_rest, y_train, y_rest = train_test_split(Xc, y_super_c, test_size=0.3, stratify=y_super_c, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE)
    print("Split sizes (train,val,test):", X_train.shape[0], X_val.shape[0], X_test.shape[0], "train dist:", Counter(y_train))

    # record pre-upsample train unique counts
    train_unique_counts = dict(Counter(y_train))

    # Natural train distribution (post-rebalance) retained
    X_train_bal, y_train_bal = X_train, y_train
    train_final_counts = dict(Counter(y_train_bal))
    print("Train distribution (final):", train_final_counts)

    # build pipeline
    steps = [('scaler', StandardScaler())]
    if USE_PCA:
        steps.append(('pca', PCA(n_components=PCA_NCOMP, random_state=RANDOM_STATE)))
    steps.append(('nyst', Nystroem(kernel='rbf', random_state=RANDOM_STATE)))
    steps.append(('svc', LinearSVC(class_weight='balanced', max_iter=20000, random_state=RANDOM_STATE)))
    pipe = Pipeline(steps)

    ovr = OneVsRestClassifier(pipe)
    calibrated = CalibratedClassifierCV(estimator=ovr, cv=3, method='sigmoid')

    param_dist = {
        'estimator__estimator__nyst__n_components': NYST_COMPONENTS_CANDIDATES,
        'estimator__estimator__nyst__gamma': NYST_GAMMA_CANDIDATES,
        'estimator__estimator__svc__C': SVC_C_CANDIDATES
    }

    print("Starting RandomizedSearchCV (stage1)...")
    rsearch = RandomizedSearchCV(calibrated, param_distributions=param_dist, n_iter=N_ITER_SEARCH, cv=3,
                                 random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=2)
    rsearch.fit(X_train_bal, y_train_bal)
    print("RandomizedSearchCV done. Best params:", rsearch.best_params_)

    best_clf = rsearch.best_estimator_
    joblib.dump(best_clf, OUT_MODEL)
    print("Saved model to", OUT_MODEL)

    # tune thresholds on validation set (search for econ and water)
    val_probs = best_clf.predict_proba(X_val)
    classes = best_clf.classes_
    print("Model classes order:", classes)
    # threshold grid
    thresh_grid = np.linspace(0.4, 0.95, 56)
    best_thresh = {int(c): None for c in classes}
    idx_map = {c:i for i,c in enumerate(classes)}
    base_val_labels = classes[np.argmax(val_probs, axis=1)]
    for cls_val in [1, 2]:
        best_f1 = -1.0
        best_t = None
        col = idx_map[cls_val]
        for t in thresh_grid:
            labels = base_val_labels.copy()
            labels[val_probs[:, col] > t] = cls_val
            f1 = f1_score(y_val, labels, labels=[cls_val], average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresh[cls_val] = float(best_t) if best_t is not None else None
        print(f"Best threshold class {cls_val}: t={best_t} f1={best_f1:.3f}")

    with open(OUT_THRESH, 'w') as fj:
        json.dump(best_thresh, fj)
    print("Saved thresholds to", OUT_THRESH)

    # evaluate on test set with thresholds
    test_probs = best_clf.predict_proba(X_test)
    idx_map = {c:i for i,c in enumerate(classes)}
    default_test = classes[np.argmax(test_probs, axis=1)]
    test_labels = default_test.copy()
    for cls_val, t in best_thresh.items():
        if t is None: continue
        mask = test_probs[:, idx_map[int(cls_val)]] > float(t)
        test_labels[mask] = int(cls_val)

    report = classification_report(y_test, test_labels, target_names=["econ", "water", "others"], output_dict=True)
    print("Stage-1 test report:")
    print(classification_report(y_test, test_labels, target_names=["econ", "water", "others"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_labels))

    test_counts = dict(Counter(y_test))
    save_report_with_traininfo(report, train_unique_counts, train_final_counts, test_counts, OUT_REPORT)

    # chunked predict whole dataset & save
    chunked_predict_and_save(best_clf, NPZ, out_pred=OUT_PRED_NPY, out_prob=OUT_PROB_NPY, chunk_size=PRED_CHUNK)
    print("Stage-1 complete.")
