# Updated stage2_svm_weighted.py
import os
import json
import numpy as np
import joblib
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# -----------------------
# Config
# -----------------------
NPZ = "./aligned_features/svm_add_data_features_labels.npz"
STAGE1_PRED = "stage1_svm_weight_scale_increased.npy"
STAGE1_MODEL = "stage1_svm_weight_scale_increased.joblib"
STAGE1_CHUNK = 2_000_000
RANDOM_STATE = 42

economic_crops = {2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420}
orchards_codes = {2403, 2404, 2407, 2420, 2416, 2419}
plantation_codes = {2302, 2303, 2405}
field_codes = {2101, 2204, 2205}
SUBCLASS_LABELS = { "orchards": 1, "plantation": 2, "field": 3, "other_econ": 4 }

MIN_PIXELS_PER_GROUP = 100
PER_GROUP_CAP = 200000

NYST_COMPONENTS = [400, 600]
NYST_GAMMA = [0.5, 1.0]
SVC_C = [0.1, 1.0, 10.0]
N_ITER_SEARCH = 8
N_JOBS = 4

TUNE_THRESHOLDS = False
THRESH_GRID = np.linspace(0.2, 0.95, 40)

# Outputs
OUT_MODEL = "stage2_weighted_increased_model.joblib"
OUT_MODEL_FULL = "stage2_weighted_increased_model_fulldata.joblib"  # optional final retrain
OUT_REPORT = "stage2_weighted_increased_report.csv"
OUT_STATS_LU = "stage2_weighted_increased_indicator_stats_per_lu.csv"
OUT_STATS_GROUP = "stage2_weighted_increased_indicator_stats_per_group.csv"
OUT_TEST_PROB = "stage2_weighted_increased_test_prob.npy"
OUT_TEST_PRED = "stage2_weighted_increased_test_pred.npy"
OUT_CONF_CSV = "stage2_weighted_increased_confusion_matrix.csv"
OUT_META_JSON = "stage2_weighted_increased_meta.json"

# full-dataset predictions filename (for Stage-3)
STAGE2_PRED = "stage2_weighted_increased.npy"

# controls
RETRAIN_ON_FULL = False   # set True to retrain final model on uncapped full data (may be heavy)
SAVE_FULL_PRED = True     # save full-length stage2_subclass_pred.npy for Stage-3
PRED_CHUNK = 2_000_000    # chunk size for full-dataset predictions

# -----------------------
# Helper Functions
# -----------------------
def ensure_stage1_pred(npz_path, pred_npy=STAGE1_PRED, model_path=STAGE1_MODEL, chunk_size=STAGE1_CHUNK):
    if os.path.exists(pred_npy):
        print("Found Stage-1 predictions:", pred_npy)
        return np.load(pred_npy)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Neither stage1_pred nor stage1 model found. Run Stage-1 first.")
    print("Generating Stage-1 predictions using model:", model_path)
    model = joblib.load(model_path)
    data = np.load(npz_path, allow_pickle=True)
    X_all = data["X"].astype(np.float32)
    n = X_all.shape[0]
    preds = np.zeros(n, dtype=np.uint8)
    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        preds[s:e] = model.predict(X_all[s:e]).astype(np.uint8)
        print(f"  chunk {s}:{e}")
    np.save(pred_npy, preds)
    print("Saved Stage-1 predictions to", pred_npy)
    return preds

def load_Xy(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.int32)

def map_econ_to_subclass_array(y_lu_codes):
    mapped = np.full_like(y_lu_codes, fill_value=SUBCLASS_LABELS["other_econ"], dtype=np.int32)
    for i, c in enumerate(y_lu_codes):
        if c in orchards_codes:
            mapped[i] = SUBCLASS_LABELS["orchards"]
        elif c in plantation_codes:
            mapped[i] = SUBCLASS_LABELS["plantation"]
        elif c in field_codes:
            mapped[i] = SUBCLASS_LABELS["field"]
    return mapped

def compute_indicator_stats_per_lu(X, y, crop_codes, out_csv):
    rows = []
    for c in sorted(crop_codes):
        mask = (y == c)
        if mask.sum() == 0:
            continue
        mean = X[mask].mean(axis=0)
        med = np.median(X[mask], axis=0)
        r = {"LU_CODE": int(c), "count": int(mask.sum())}
        for i in range(X.shape[1]):
            r[f"F{i}_mean"] = float(mean[i])
            r[f"F{i}_median"] = float(med[i])
        rows.append(r)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved per-LU indicator stats to", out_csv)

def compute_indicator_stats_per_group(X, y_group, out_csv):
    rows = []
    for g in sorted(np.unique(y_group)):
        mask = (y_group == g)
        mean = X[mask].mean(axis=0)
        med = np.median(X[mask], axis=0)
        r = {"SUBCLASS_LABEL": int(g), "count": int(mask.sum())}
        for i in range(X.shape[1]):
            r[f"F{i}_mean"] = float(mean[i])
            r[f"F{i}_median"] = float(med[i])
        rows.append(r)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved per-group indicator stats to", out_csv)

def cap_per_group(X, y_group, cap=PER_GROUP_CAP):
    rng = np.random.default_rng(RANDOM_STATE)
    partsX, partsy = [], []
    for c in np.unique(y_group):
        idxs = np.flatnonzero(y_group == c)
        if len(idxs) > cap:
            chosen = rng.choice(idxs, size=cap, replace=False)
        else:
            chosen = idxs
        partsX.append(X[chosen]); partsy.append(y_group[chosen])
        print(f"Group {int(c)}: available={len(idxs)}, kept={len(chosen)}")
    Xc = np.vstack(partsX); yc = np.hstack(partsy)
    p = rng.permutation(Xc.shape[0])
    return Xc[p], yc[p]

# JSON safety helper
def to_py(o):
    import numpy as _np
    if isinstance(o, _np.integer):
        return int(o)
    if isinstance(o, _np.floating):
        return float(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    return o

def save_confusion_csv(cm, classes, path):
    # classes might be ndarray of ints - convert to python ints for readable CSV
    idx = [int(x) for x in classes]
    cm_df = pd.DataFrame(cm, index=idx, columns=idx)
    cm_df.to_csv(path, encoding="utf-8-sig")
    print("Saved confusion matrix:", path)

def save_full_stage2_predictions(model, npz_path, stage1_pred, out_npy=STAGE2_PRED, chunk_size=PRED_CHUNK):
    """Predict stage-2 subclass for entire dataset aligned to NPZ rows and save as .npy."""
    data = np.load(npz_path, allow_pickle=True)
    X_all = data["X"].astype(np.float32)
    n = X_all.shape[0]
    preds = np.zeros(n, dtype=np.int32)  # default 0 for non-econ
    econ_idxs = np.flatnonzero(stage1_pred == 1)
    print(f"Generating full Stage-2 predictions for {econ_idxs.size} econ pixels in chunks...")
    for s in range(0, econ_idxs.size, chunk_size):
        e = min(econ_idxs.size, s + chunk_size)
        sel = econ_idxs[s:e]
        preds[sel] = model.predict(X_all[sel]).astype(np.int32)
        print(f"  chunk {s}:{e}")
    np.save(out_npy, preds)
    print("Saved full Stage-2 predictions to", out_npy)
    return preds

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    print("=== Stage-2 (weighted subclass classifier) ===")
    stage1_pred = ensure_stage1_pred(NPZ, STAGE1_PRED, STAGE1_MODEL)
    X_all, y_all = load_Xy(NPZ)

    # Filter Stage-1 economic
    mask_stage1 = (stage1_pred == 1)
    X_stage1, y_stage1 = X_all[mask_stage1], y_all[mask_stage1]
    print("Pixels predicted econ by Stage-1:", X_stage1.shape[0])

    # Keep only real economic LU codes
    mask_true_econ = np.isin(y_stage1, list(economic_crops))
    X_econ, y_econ = X_stage1[mask_true_econ], y_stage1[mask_true_econ]
    print("Pixels both predicted & true econ:", X_econ.shape[0])

    compute_indicator_stats_per_lu(X_econ, y_econ, sorted(economic_crops), OUT_STATS_LU)

    y_sub = map_econ_to_subclass_array(y_econ)
    print("Subclass distribution:", Counter(y_sub))
    compute_indicator_stats_per_group(X_econ, y_sub, OUT_STATS_GROUP)

    # Filter small groups & cap
    uniques, counts = np.unique(y_sub, return_counts=True)
    keep_groups = uniques[counts >= MIN_PIXELS_PER_GROUP]
    mask_keep = np.isin(y_sub, keep_groups)
    X_keep, y_keep = X_econ[mask_keep], y_sub[mask_keep]

    # Save uncapped copy (useful for optional final retrain)
    X_keep_uncapped, y_keep_uncapped = X_keep.copy(), y_keep.copy()

    Xc, yc = cap_per_group(X_keep, y_keep)

    # Split data
    X_tr, X_rest, y_tr, y_rest = train_test_split(Xc, yc, test_size=0.3, stratify=yc, random_state=RANDOM_STATE)
    X_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE)
    print("Split sizes:", X_tr.shape[0], X_val.shape[0], X_te.shape[0])
    print("Train dist:", Counter(y_tr))

    # --- Weighted SVM pipeline ---
    steps = [
        ('scaler', StandardScaler()),
        ('nyst', Nystroem(kernel='rbf', random_state=RANDOM_STATE)),
        ('svc', LinearSVC(class_weight='balanced', max_iter=20000, random_state=RANDOM_STATE))
    ]
    pipe = Pipeline(steps)
    ovr = OneVsRestClassifier(pipe)
    clf = CalibratedClassifierCV(estimator=ovr, cv=3, method='sigmoid')

    param_dist = {
        'estimator__estimator__nyst__n_components': NYST_COMPONENTS,
        'estimator__estimator__nyst__gamma': NYST_GAMMA,
        'estimator__estimator__svc__C': SVC_C
    }

    print("Starting RandomizedSearchCV (weighted subclass)...")
    rsearch = RandomizedSearchCV(clf, param_distributions=param_dist,
                                 n_iter=N_ITER_SEARCH, cv=3, n_jobs=N_JOBS,
                                 random_state=RANDOM_STATE, verbose=2)
    rsearch.fit(X_tr, y_tr)
    print("Best params:", rsearch.best_params_)

    best_clf = rsearch.best_estimator_
    joblib.dump(best_clf, OUT_MODEL)
    print("Saved model:", OUT_MODEL)

    # Optionally save full dataset predictions (for Stage-3)
    if SAVE_FULL_PRED:
        try:
            save_full_stage2_predictions(best_clf, NPZ, stage1_pred, out_npy=STAGE2_PRED, chunk_size=PRED_CHUNK)
        except Exception as e:
            print("Warning: failed to save full Stage-2 predictions:", e)

    # Evaluate on test set
    test_probs = best_clf.predict_proba(X_te)
    np.save(OUT_TEST_PROB, test_probs.astype(np.float32))
    test_pred = best_clf.classes_[np.argmax(test_probs, axis=1)]
    np.save(OUT_TEST_PRED, test_pred.astype(np.int32))

    report = classification_report(y_te, test_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(OUT_REPORT, encoding="utf-8-sig")
    print("Saved report:", OUT_REPORT)

    # confusion matrix using model classes ordering
    print("Confusion matrix:")
    cm = confusion_matrix(y_te, test_pred, labels=best_clf.classes_)
    print(cm)
    save_confusion_csv(cm, best_clf.classes_, OUT_CONF_CSV)

    # Build meta dict with JSON-safe types
    params_py = {k: to_py(v) for k, v in rsearch.best_params_.items()}
    counts_native = {int(k): int(v) for k, v in Counter(y_keep).items()}

    meta = {
        "stage": 2,
        "strategy": "class_weight_balanced (no SMOTE)",
        "subclasses": {
            "orchards": sorted(orchards_codes),
            "plantation": sorted(plantation_codes),
            "field_non_orchard": sorted(field_codes),
            "other_econ": "remaining econ LU codes"
        },
        "params": params_py,
        "counts": counts_native
    }

    # Optionally retrain on uncapped full data (useful if you want final model trained on all samples)
    if RETRAIN_ON_FULL:
        print("Retraining final model on uncapped full per-group data (this may be slow)...")
        # rebuild same estimator and set params
        pipe_full = Pipeline([
            ('scaler', StandardScaler()),
            ('nyst', Nystroem(kernel='rbf', random_state=RANDOM_STATE)),
            ('svc', LinearSVC(class_weight='balanced', max_iter=20000, random_state=RANDOM_STATE))
        ])
        ovr_full = OneVsRestClassifier(pipe_full)
        clf_full = CalibratedClassifierCV(estimator=ovr_full, cv=3, method='sigmoid')
        clf_full.set_params(**params_py)
        # fit on uncapped X_keep_uncapped / y_keep_uncapped
        clf_full.fit(X_keep_uncapped, y_keep_uncapped)
        joblib.dump(clf_full, OUT_MODEL_FULL)
        print("Saved full-data model:", OUT_MODEL_FULL)
        meta['model_full'] = OUT_MODEL_FULL

    with open(OUT_META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Saved metadata:", OUT_META_JSON)
