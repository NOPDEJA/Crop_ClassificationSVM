"""
stage3_weightscale_fixed.py

Stage-3: Per-subclass fine-grained LU_CODE classifiers using weight-scaling
(instead of SMOTETomek). Ensures each subclass training set has at least
MIN_TOTAL_SAMPLES (default 10000) by upsampling with replacement if needed.

Notes:
 - If a feature weight vector file exists (WEIGHT_SCALE_FILE), it is applied as
   a per-feature multiplicative scale before fitting/predicting.
 - The pipeline still uses class_weight='balanced' in LinearSVC to handle
   residual class imbalance during optimization.
 - Confusion matrices are saved to CSV (one per subclass).
 - WARNING printed for "other_econ" subclass: since Stage-3 operates only on
   pixels that Stage-2 labeled as "other_econ", many true 'other' LU codes
   may be filtered out by Stage-2 predictions (as you warned).

Outputs per subclass (format placeholders):
 - stage3_{grp}_weightscale_model.joblib
 - stage3_{grp}_weightscale_report.csv
 - stage3_{grp}_weightscale_test_prob.npy
 - stage3_{grp}_weightscale_test_pred.npy
 - stage3_{grp}_weightscale_confusion_matrix.csv
 - stage3_{grp}_weightscale_meta.json
 - top-level: stage3_weightscale_meta.json

"""

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
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------
# Config
# -----------------------
NPZ = "./aligned_features/svm_drop_features_labels.npz"
STAGE1_PRED = "stage1_svm_drop_feature_weight_scale.npy"
STAGE1_MODEL = "stage1_svm_drop_feature_weight_scale.joblib"
STAGE2_PRED = "stage2_subclass_weighted_pred.npy"
STAGE2_MODEL = "stage2_weighted_model2.joblib"
WEIGHT_SCALE_FILE = "stage1_feature_weight_scale.npy"  # optional; if present applied to features
RANDOM_STATE = 42

economic_crops = {2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420}
orchards_codes = {2403, 2404, 2407, 2420, 2416, 2419}
plantation_codes = {2302, 2303, 2405}
field_codes = {2101, 2204, 2205}

SUBCLASS_LABELS = {"orchards": 1, "plantation": 2, "field": 3, "other_econ": 4}
SUBCLASS_TO_CODES = {
    1: sorted(list(orchards_codes)),
    2: sorted(list(plantation_codes)),
    3: sorted(list(field_codes)),
    4: sorted([c for c in economic_crops if c not in (orchards_codes | plantation_codes | field_codes)])
}

MIN_PIXELS_PER_LU = 100
PER_LU_CAP = 70000
TARGET_PER_LU = 20000
MIN_TOTAL_SAMPLES = 10000
UPSAMPLE_SMALL = True
CONDITIONAL_UPSAMPLE = True
MIN_UPSAMPLE_TRIGGER = 0.67

NYST_COMPONENTS = [400, 600]
NYST_GAMMA = [0.5, 1.0]
SVC_C = [0.1, 1.0, 10.0]
N_ITER_SEARCH = 8
N_JOBS = 4

OUT_MODEL_TPL = "stage3_{grp}_weightscale_model.joblib"
OUT_REPORT_TPL = "stage3_{grp}_weightscale_report.csv"
OUT_TEST_PROB_TPL = "stage3_{grp}_weightscale_test_prob.npy"
OUT_TEST_PRED_TPL = "stage3_{grp}_weightscale_test_pred.npy"
OUT_META_TPL = "stage3_{grp}_weightscale_meta.json"
OUT_CONF_TPL = "stage3_{grp}_weightscale_confusion_matrix.csv"
OUT_TOP_META = "stage3_weightscale_meta.json"

# -----------------------
# Helpers
# -----------------------

def ensure_stage1_pred(npz_path):
    if os.path.exists(STAGE1_PRED):
        print("Loading Stage-1 predictions:", STAGE1_PRED)
        return np.load(STAGE1_PRED)
    if not os.path.exists(STAGE1_MODEL):
        raise FileNotFoundError("Stage-1 model not found and Stage-1 predictions missing.")
    print("Generating Stage-1 predictions using model:", STAGE1_MODEL)
    model = joblib.load(STAGE1_MODEL)
    d = np.load(npz_path, allow_pickle=True)
    X_all = d["X"].astype(np.float32)
    preds = model.predict(X_all).astype(np.uint8)
    np.save(STAGE1_PRED, preds)
    return preds


def ensure_stage2_pred(npz_path, stage1_pred):
    if os.path.exists(STAGE2_PRED):
        print("Loading Stage-2 predictions:", STAGE2_PRED)
        return np.load(STAGE2_PRED)
    if not os.path.exists(STAGE2_MODEL):
        raise FileNotFoundError("Stage-2 model not found and Stage-2 predictions missing.")
    print("Generating Stage-2 predictions using model:", STAGE2_MODEL)
    model = joblib.load(STAGE2_MODEL)
    d = np.load(npz_path, allow_pickle=True)
    X_all = d["X"].astype(np.float32)
    preds = np.zeros(X_all.shape[0], dtype=np.int32)
    mask = (stage1_pred == 1)
    idxs = np.flatnonzero(mask)
    for s in range(0, idxs.size, 500_000):
        e = min(idxs.size, s + 500_000)
        sel = idxs[s:e]
        preds[sel] = model.predict(X_all[sel]).astype(np.int32)
        print(f"  Stage-2 predict chunk {s}:{e}")
    np.save(STAGE2_PRED, preds)
    return preds


def load_Xy(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.int32)


def cap_per_lu(X, y, cap=PER_LU_CAP):
    rng = np.random.default_rng(RANDOM_STATE)
    partsX, partsy = [], []
    for c in np.unique(y):
        idxs = np.flatnonzero(y == c)   
        keep_n = min(len(idxs), cap)
        chosen = rng.choice(idxs, size=keep_n, replace=False)
        partsX.append(X[chosen])
        partsy.append(y[chosen])
        print(f"  LU {int(c)}: available={len(idxs)}, kept={keep_n}")
    Xc = np.vstack(partsX)
    yc = np.hstack(partsy)
    p = rng.permutation(Xc.shape[0])
    return Xc[p], yc[p]


def sample_per_lu_train_only(X_train, y_train, target_per_lu=TARGET_PER_LU,
                             upsample=UPSAMPLE_SMALL, conditional=CONDITIONAL_UPSAMPLE,
                             min_trigger=MIN_UPSAMPLE_TRIGGER):
    rng = np.random.default_rng(RANDOM_STATE)
    lus = np.unique(y_train)
    partsX, partsy = [], []
    for c in lus:
        idxs = np.flatnonzero(y_train == c)
        n = idxs.size
        tgt = int(target_per_lu)
        if conditional:
            effective_target = n if n >= tgt else tgt
        else:
            effective_target = tgt
        if n >= effective_target:
            chosen = rng.choice(idxs, size=effective_target, replace=False)
        else:
            chosen = rng.choice(idxs, size=effective_target, replace=True) if upsample else idxs
        partsX.append(X_train[chosen])
        partsy.append(y_train[chosen])
        print(f"  LU {int(c)}: train n={n} -> final={len(chosen)} (target={effective_target})")
    Xb = np.vstack(partsX)
    yb = np.hstack(partsy)
    perm = rng.permutation(Xb.shape[0])
    return Xb[perm], yb[perm]


def enforce_min_total_samples(X, y, min_total=MIN_TOTAL_SAMPLES):
    if X.shape[0] >= min_total:
        return X, y
    print(f"  Total samples={X.shape[0]} < {min_total}, upsampling globally to reach min_total")
    rng = np.random.default_rng(RANDOM_STATE)
    # sample indices with replacement, preserving class distribution roughly
    idxs = rng.choice(np.arange(X.shape[0]), size=min_total, replace=True)
    return X[idxs], y[idxs]


def load_weight_scale():
    if os.path.exists(WEIGHT_SCALE_FILE):
        print(f"Loading weight scale: {WEIGHT_SCALE_FILE}")
        return np.load(WEIGHT_SCALE_FILE)
    print("No weight scale file found — proceeding without per-feature weight scaling.")
    return None


def apply_weight_scale(X, scale_vec):
    if scale_vec is None:
        return X
    if scale_vec.shape[0] != X.shape[1]:
        print("Warning: weight scale vector length mismatch, skipping scale application.")
        return X
    return X * scale_vec


def save_report(report_dict, train_counts, final_counts, test_counts, out_csv):
    df = pd.DataFrame(report_dict).transpose()
    train_unique_col = []
    train_final_col = []
    test_col = []
    for idx in df.index:
        try:
            cls = int(idx)
            train_unique_col.append(train_counts.get(cls, 0))
            train_final_col.append(final_counts.get(cls, 0))
            test_col.append(test_counts.get(cls, 0))
        except Exception:
            train_unique_col.append("")
            train_final_col.append("")
            test_col.append("")
    df["train_unique_support"] = train_unique_col
    df["train_final_support"] = train_final_col
    df["test_support"] = test_col
    df.to_csv(out_csv, index=True, encoding="utf-8-sig")
    print("Saved report:", out_csv)

# -----------------------
# Main Stage-3 loop
# -----------------------
if __name__ == "__main__":
    print("=== Stage-3 with weight-scaling (no SMOTE) : start ===")
    stage1_pred = ensure_stage1_pred(NPZ)
    stage2_pred = ensure_stage2_pred(NPZ, stage1_pred)
    X_all, y_all = load_Xy(NPZ)
    econ_mask = (stage1_pred == 1)
    scale_vec = load_weight_scale()

    top_meta = {"stage": 3, "trained_subclasses": {}, "config": {
        "MIN_PIXELS_PER_LU": MIN_PIXELS_PER_LU,
        "PER_LU_CAP": PER_LU_CAP,
        "TARGET_PER_LU": TARGET_PER_LU,
        "MIN_TOTAL_SAMPLES": MIN_TOTAL_SAMPLES,
        "UPSAMPLE_SMALL": UPSAMPLE_SMALL,
        "CONDITIONAL_UPSAMPLE": CONDITIONAL_UPSAMPLE
    }}

    for subclass_label, lu_list in SUBCLASS_TO_CODES.items():
        name_list = [k for k, v in SUBCLASS_LABELS.items() if v == subclass_label]
        name = name_list[0] if name_list else f"subclass_{subclass_label}"
        print(f"\n--- Processing subclass {name} (label={subclass_label}) with LU codes: {lu_list} ---")

        if subclass_label == SUBCLASS_LABELS.get("other_econ"):
            print("WARNING: 'other_econ' subclass may be heavily filtered by Stage-2 predictions; many true 'other' LU codes could be absent.")

        mask = econ_mask & (stage2_pred == subclass_label) & np.isin(y_all, lu_list)
        idxs = np.flatnonzero(mask)
        print("  pixels matching subclass & LU_list:", idxs.size)
        if idxs.size == 0:
            print("  skipping (no samples).")
            continue

        X_sub = X_all[idxs]
        y_sub = y_all[idxs]

        uniques, counts = np.unique(y_sub, return_counts=True)
        keep_mask = counts >= MIN_PIXELS_PER_LU
        keep_lu = uniques[keep_mask]
        if keep_lu.size == 0:
            print(f"  no LU codes pass MIN_PIXELS_PER_LU={MIN_PIXELS_PER_LU}: skipping.")
            continue
        keep_selector = np.isin(y_sub, keep_lu)
        X_keep = X_sub[keep_selector]
        y_keep = y_sub[keep_selector]
        print(f"  after dropping tiny LUs: samples={X_keep.shape[0]}, LU_count={len(keep_lu)}")

        Xc, yc = cap_per_lu(X_keep, y_keep, cap=PER_LU_CAP)

        X_tr, X_rest, y_tr, y_rest = train_test_split(Xc, yc, test_size=0.3, stratify=yc, random_state=RANDOM_STATE)
        X_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE)
        print("  split sizes (train,val,test):", X_tr.shape[0], X_val.shape[0], X_te.shape[0])

        train_unique_counts = dict(Counter(y_tr))

        print("  Step 1: manual per-LU upsample to reach TARGET_PER_LU (if configured)")
        X_tr_pre, y_tr_pre = sample_per_lu_train_only(X_tr, y_tr, target_per_lu=TARGET_PER_LU,
                                                     upsample=UPSAMPLE_SMALL, conditional=CONDITIONAL_UPSAMPLE,
                                                     min_trigger=MIN_UPSAMPLE_TRIGGER)
        print("  after manual upsample dist:", dict(Counter(y_tr_pre)))

        print("  Step 2: ensure minimum total samples (no SMOTE).")
        X_tr_bal, y_tr_bal = enforce_min_total_samples(X_tr_pre, y_tr_pre, min_total=MIN_TOTAL_SAMPLES)
        print("  after enforce_min_total dist:", dict(Counter(y_tr_bal)))

        train_final_counts = dict(Counter(y_tr_bal))
        too_small = [c for c, n in train_final_counts.items() if n < 3]
        if too_small:
            print("  Some LU codes too small for 3-fold CV, skipping this subclass:", too_small)
            continue

        # Apply per-feature weight scale if available (no synthetic samples)
        X_tr_bal = apply_weight_scale(X_tr_bal, scale_vec)
        X_val_scaled = apply_weight_scale(X_val, scale_vec)
        X_te_scaled = apply_weight_scale(X_te, scale_vec)

        # Build pipeline + RandomizedSearch
        steps = [
            ('scaler', StandardScaler()),
            ('nyst', Nystroem(kernel='rbf', random_state=RANDOM_STATE)),
            ('svc', LinearSVC(class_weight='balanced', max_iter=20000, random_state=RANDOM_STATE))
        ]
        pipe = Pipeline(steps)
        ovr = OneVsRestClassifier(pipe)
        calibrated = CalibratedClassifierCV(estimator=ovr, cv=3, method='sigmoid')

        param_dist = {
            'estimator__estimator__nyst__n_components': NYST_COMPONENTS,
            'estimator__estimator__nyst__gamma': NYST_GAMMA,
            'estimator__estimator__svc__C': SVC_C
        }

        print("  Running RandomizedSearchCV ... (this may take a while)")
        rsearch = RandomizedSearchCV(calibrated, param_distributions=param_dist, n_iter=N_ITER_SEARCH,
                                     cv=3, random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=2)
        rsearch.fit(X_tr_bal, y_tr_bal)
        print("  Best params:", rsearch.best_params_)

        best_clf = rsearch.best_estimator_
        lu_classes = best_clf.classes_
        print("  LU classes (model):", lu_classes)

        # Evaluate on test set
        test_probs = best_clf.predict_proba(X_te_scaled)
        base_test = lu_classes[np.argmax(test_probs, axis=1)]
        test_pred = base_test.copy()

        np.save(OUT_TEST_PROB_TPL.format(grp=name), test_probs.astype(np.float32))
        np.save(OUT_TEST_PRED_TPL.format(grp=name), test_pred.astype(np.int32))
        print("  saved test prob/pred arrays")

        report = classification_report(y_te, test_pred, output_dict=True)
        test_counts = dict(Counter(y_te))
        save_report(report, train_unique_counts, train_final_counts, test_counts, OUT_REPORT_TPL.format(grp=name))

        print("  confusion matrix for", name)
        cm = confusion_matrix(y_te, test_pred, labels=lu_classes)
        print(cm)
        cm_df = pd.DataFrame(cm, index=[int(x) for x in lu_classes], columns=[int(x) for x in lu_classes])
        cm_df.to_csv(OUT_CONF_TPL.format(grp=name), encoding="utf-8-sig")
        print("  saved confusion matrix:", OUT_CONF_TPL.format(grp=name))

        # save model & meta
        joblib.dump(best_clf, OUT_MODEL_TPL.format(grp=name))
        meta = {
            "subclass": name,
            "subclass_label": int(subclass_label),
            "lu_codes_trained": [int(x) for x in lu_classes],
            "counts": {
                "train_unique": {int(k): int(v) for k, v in train_unique_counts.items()},
                "train_final": {int(k): int(v) for k, v in train_final_counts.items()},
                "test": {int(k): int(v) for k, v in test_counts.items()}
            },
            "best_params": rsearch.best_params_
        }
        with open(OUT_META_TPL.format(grp=name), 'w') as fm:
            json.dump(meta, fm, indent=2)
        print("  saved model & meta for", name)

        top_meta["trained_subclasses"][name] = {
            "label": int(subclass_label),
            "lu_codes_trained": [int(x) for x in lu_classes],
            "model": OUT_MODEL_TPL.format(grp=name),
            "report": OUT_REPORT_TPL.format(grp=name),
            "test_prob": OUT_TEST_PROB_TPL.format(grp=name),
            "test_pred": OUT_TEST_PRED_TPL.format(grp=name),
            "confusion_matrix": OUT_CONF_TPL.format(grp=name),
            "meta": OUT_META_TPL.format(grp=name)
        }

    # top-level meta
    with open(OUT_TOP_META, 'w') as ftop:
        json.dump(top_meta, ftop, indent=2)
    print("Stage-3 with weight-scaling complete.")
