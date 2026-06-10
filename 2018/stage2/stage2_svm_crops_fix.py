# stage2_svm_crops_fix.py (enhanced)
"""Stage-2: Fine-grained crop classifier (hierarchical after Stage-1)

Steps:
 1. Ensure Stage-1 predictions (or generate via Stage-1 model).
 2. Keep pixels predicted econ (class=1) AND whose true LU_CODE is in economic_crops.
 3. Compute indicator stats (mean/median per feature per crop).
 4. Filter tiny crops (< MIN_PIXELS_PER_CROP) then cap each crop to PER_CROP_CAP.
 5. Split (train 70%, val 15%, test 15%).
 6. Conditional per-crop balancing (upsample only small crops) to TARGET_PER_CROP.
 7. RandomizedSearchCV + calibrated One-vs-Rest (Nystroem + LinearSVC).
 8. (Optional) per-crop probability threshold tuning (disabled by default).
 9. Evaluate on test set and save model, report, metadata, probability & prediction arrays.

Enhancements to add next in subsequent patch portions.
"""
import os, json
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
# Config - edit if needed
# -----------------------
NPZ = "./aligned_features/svm_features_labels.npz"
STAGE1_PRED = "stage1_pred.npy"
STAGE1_MODEL = "stage1_model.joblib"  # expected Stage-1 model name
STAGE1_CHUNK = 1_000_000

RANDOM_STATE = 42
economic_crops = {2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420}

# per-crop caps & sampling
MIN_PIXELS_PER_CROP = 100
PER_CROP_CAP = 24000      # unique samples to keep per crop before split
TARGET_PER_CROP = 6000    # desired minimum train size (only enforced for small crops)
UPSAMPLE_SMALL = True
CONDITIONAL_UPSAMPLE = True
MIN_UPSAMPLE_TRIGGER = 0.67  # fraction of target above which we lightly upsample (else full)
TUNE_THRESHOLDS = False
THRESH_GRID = np.linspace(0.2, 0.95, 40)

# Search / pipelines
NYST_COMPONENTS = [400,600]
NYST_GAMMA = [0.5,1.0]
SVC_C = [0.1,1.0,10.0]
N_ITER_SEARCH = 8
N_JOBS = 4

OUT_MODEL = "stage2_model.joblib"
OUT_REPORT = "stage2_report.csv"
OUT_STATS = "stage2_indicator_stats.csv"
OUT_TEST_PROB = "stage2_test_prob.npy"
OUT_TEST_PRED = "stage2_test_pred.npy"
OUT_META_JSON = "stage2_meta.json"

# -----------------------
# Helpers
# -----------------------
def ensure_stage1_pred(npz_path, pred_npy=STAGE1_PRED, model_path=STAGE1_MODEL, chunk_size=STAGE1_CHUNK):
    if os.path.exists(pred_npy):
        print("Found Stage-1 predictions:", pred_npy)
        return np.load(pred_npy)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Neither stage1_pred.npy nor stage1 model found. Run Stage-1 first.")
    print("Generating Stage-1 predictions using model:", model_path)
    model = joblib.load(model_path)
    data = np.load(npz_path, allow_pickle=True)
    X_all = data["X"].astype(np.float32)
    n = X_all.shape[0]
    preds = np.zeros(n, dtype=np.uint8)
    for s in range(0, n, chunk_size):
        e = min(n, s+chunk_size)
        preds[s:e] = model.predict(X_all[s:e]).astype(np.uint8)
        print(f"  chunk {s}:{e}")
    np.save(pred_npy, preds)
    print("Saved Stage-1 predictions to", pred_npy)
    return preds

def load_Xy(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.int32)

def compute_indicator_stats_per_crop(X, y, crop_codes, out_csv):
    rows = []
    for c in sorted(crop_codes):
        mask = (y == c)
        if mask.sum()==0: continue
        mean = X[mask].mean(axis=0)
        med = np.median(X[mask], axis=0)
        r = {"LU_CODE": int(c), "count": int(mask.sum())}
        for i in range(X.shape[1]):
            r[f"F{i}_mean"] = float(mean[i])
            r[f"F{i}_median"] = float(med[i])
        rows.append(r)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved indicator stats to", out_csv)

def cap_per_crop(X, y, cap=PER_CROP_CAP):
    rng = np.random.default_rng(RANDOM_STATE)
    partsX, partsy = [], []
    for c in np.unique(y):
        idxs = np.flatnonzero(y==c)
        if len(idxs) > cap:
            chosen = rng.choice(idxs, size=cap, replace=False)
        else:
            chosen = idxs
        partsX.append(X[chosen]); partsy.append(y[chosen])
        print(f"Crop {int(c)}: available={len(idxs)}, kept={len(chosen)}")
    Xc = np.vstack(partsX); yc = np.hstack(partsy)
    p = rng.permutation(Xc.shape[0])
    return Xc[p], yc[p]

def sample_per_crop_train_only(X_train, y_train, target_per_crop=TARGET_PER_CROP, upsample=UPSAMPLE_SMALL,
                               conditional=CONDITIONAL_UPSAMPLE, min_trigger=MIN_UPSAMPLE_TRIGGER):
    rng = np.random.default_rng(RANDOM_STATE)
    crops = np.unique(y_train)
    partsX, partsy = [], []
    for c in crops:
        idxs = np.flatnonzero(y_train==c)
        n = idxs.size
        tgt = int(target_per_crop)
        effective_target = tgt
        if conditional:
            if n >= tgt:
                effective_target = n  # keep full data for large crop
            elif n >= min_trigger * tgt:
                effective_target = tgt  # modest upsample to target
            else:
                effective_target = tgt  # very small -> full upsample
        else:
            effective_target = tgt
        if n >= effective_target:
            chosen = rng.choice(idxs, size=effective_target, replace=False)
        else:
            chosen = rng.choice(idxs, size=effective_target, replace=True) if upsample else idxs
        partsX.append(X_train[chosen])
        partsy.append(y_train[chosen])
        print(f"Crop {int(c)}: train n={n} -> final={len(chosen)} target_policy={effective_target}")
    Xb = np.vstack(partsX); yb = np.hstack(partsy)
    perm = rng.permutation(Xb.shape[0])
    return Xb[perm], yb[perm]

def save_report_with_traininfo(report_dict, train_unique_counts, train_final_counts, test_counts, out_csv):
    df = pd.DataFrame(report_dict).transpose()
    # for stage2 index will be stringified LU_CODEs
    train_unique = []
    train_final = []
    test_supp = []
    for idx in df.index:
        try:
            cls = int(idx)
            train_unique.append(train_unique_counts.get(cls, 0))
            train_final.append(train_final_counts.get(cls, 0))
            test_supp.append(test_counts.get(cls, 0))
        except:
            train_unique.append("")
            train_final.append("")
            test_supp.append("")
    df["train_unique_support"] = train_unique
    df["train_final_support"] = train_final
    df["test_support"] = test_supp
    df.to_csv(out_csv, index=True, encoding="utf-8-sig")
    print("Saved report:", out_csv)


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    print("=== Stage-2 (enhanced crop classifier) ===")
    stage1_pred = ensure_stage1_pred(NPZ, STAGE1_PRED, STAGE1_MODEL, chunk_size=STAGE1_CHUNK)

    X_all, y_all = load_Xy(NPZ)
    print("Loaded X,y:", X_all.shape, y_all.shape)

    mask_stage1 = (stage1_pred == 1)
    X_stage1 = X_all[mask_stage1]
    y_stage1 = y_all[mask_stage1]
    print("Pixels predicted econ by Stage-1:", X_stage1.shape[0])

    mask_true_econ = np.isin(y_stage1, list(economic_crops))
    X_econ = X_stage1[mask_true_econ]
    y_econ = y_stage1[mask_true_econ]
    print("Pixels both predicted econ and true econ LU codes:", X_econ.shape[0])

    # indicator stats per crop
    compute_indicator_stats_per_crop(X_econ, y_econ, sorted(economic_crops), OUT_STATS)

    # drop tiny crops and cap per-crop before split
    uniques, counts = np.unique(y_econ, return_counts=True)
    keep = uniques[counts >= MIN_PIXELS_PER_CROP]
    print("Crops with >= min pixels:", keep)
    mask_keep = np.isin(y_econ, keep)
    X_keep = X_econ[mask_keep]; y_keep = y_econ[mask_keep]
    print("After filtering small crops:", X_keep.shape)

    # cap per crop (keeps diversity)
    Xc, yc = cap_per_crop(X_keep, y_keep, cap=PER_CROP_CAP)

    # split train / val / test: 70 / 15 / 15
    X_tr, X_rest, y_tr, y_rest = train_test_split(Xc, yc, test_size=0.3, stratify=yc, random_state=RANDOM_STATE)
    X_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE)
    print("Split sizes (train, val, test):", X_tr.shape[0], X_val.shape[0], X_te.shape[0])
    print("Train dist (pre-balance):", Counter(y_tr))

    train_unique_counts = dict(Counter(y_tr))
    # conditional upsample train-only per crop
    X_tr_bal, y_tr_bal = sample_per_crop_train_only(
        X_tr, y_tr,
        target_per_crop=TARGET_PER_CROP,
        upsample=UPSAMPLE_SMALL,
        conditional=CONDITIONAL_UPSAMPLE,
        min_trigger=MIN_UPSAMPLE_TRIGGER
    )
    train_final_counts = dict(Counter(y_tr_bal))
    print("Train dist (final balanced):", train_final_counts)
    too_small = [c for c,n in train_final_counts.items() if n < 3]
    if too_small:
        raise RuntimeError(f"Classes too small for 3-fold CV: {too_small}")

    # pipeline
    steps = [('scaler', StandardScaler()), ('nyst', Nystroem(kernel='rbf', random_state=RANDOM_STATE)),
             ('svc', LinearSVC(class_weight='balanced', max_iter=20000, random_state=RANDOM_STATE))]
    pipe = Pipeline(steps)
    ovr = OneVsRestClassifier(pipe)
    calibrated = CalibratedClassifierCV(estimator=ovr, cv=3, method='sigmoid')

    param_dist = {
        'estimator__estimator__nyst__n_components': NYST_COMPONENTS,
        'estimator__estimator__nyst__gamma': NYST_GAMMA,
        'estimator__estimator__svc__C': SVC_C
    }

    print("Starting RandomizedSearchCV (stage2)...")
    rsearch = RandomizedSearchCV(calibrated, param_distributions=param_dist, n_iter=N_ITER_SEARCH, cv=3, random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=2)
    rsearch.fit(X_tr_bal, y_tr_bal)
    print("Best params:", rsearch.best_params_)

    best_clf = rsearch.best_estimator_
    crop_classes = best_clf.classes_

    # Optional per-crop threshold tuning on validation set
    thresholds = {int(c): None for c in crop_classes}
    if TUNE_THRESHOLDS:
        print("Threshold tuning enabled ...")
        val_probs = best_clf.predict_proba(X_val)
        base_val = crop_classes[np.argmax(val_probs, axis=1)]
        idx_map = {c:i for i,c in enumerate(crop_classes)}
        for cls in crop_classes:
            best_f1 = -1.0; best_t = None
            col = idx_map[cls]
            for t in THRESH_GRID:
                labels = base_val.copy()
                labels[val_probs[:, col] > t] = cls
                f1 = f1_score(y_val, labels, labels=[cls], average='macro')
                if f1 > best_f1:
                    best_f1 = f1; best_t = t
            thresholds[int(cls)] = float(best_t) if best_t is not None else None
        print("Thresholds:", thresholds)
    else:
        print("Threshold tuning disabled (TUNE_THRESHOLDS=False)")

    # Test evaluation
    test_probs = best_clf.predict_proba(X_te)
    np.save(OUT_TEST_PROB, test_probs.astype(np.float32))
    if TUNE_THRESHOLDS and any(v is not None for v in thresholds.values()):
        base_test = crop_classes[np.argmax(test_probs, axis=1)]
        idx_map = {c:i for i,c in enumerate(crop_classes)}
        test_pred = base_test.copy()
        for cls, t in thresholds.items():
            if t is None: continue
            col = idx_map[cls]
            mask = test_probs[:, col] > t
            test_pred[mask] = cls
    else:
        test_pred = crop_classes[np.argmax(test_probs, axis=1)]
    np.save(OUT_TEST_PRED, test_pred.astype(np.int32))
    print("Saved test prob/pred arrays.")

    report = classification_report(y_te, test_pred, output_dict=True)
    test_counts = dict(Counter(y_te))
    save_report_with_traininfo(report, train_unique_counts, train_final_counts, test_counts, OUT_REPORT)

    print("Confusion matrix:")
    print(confusion_matrix(y_te, test_pred))

    joblib.dump(best_clf, OUT_MODEL)
    print("Saved Stage-2 model to", OUT_MODEL)

    # Metadata JSON
    meta = {
        "stage": 2,
        "economic_crops": sorted(int(c) for c in economic_crops),
        "config": {
            "PER_CROP_CAP": PER_CROP_CAP,
            "TARGET_PER_CROP": TARGET_PER_CROP,
            "CONDITIONAL_UPSAMPLE": CONDITIONAL_UPSAMPLE,
            "MIN_UPSAMPLE_TRIGGER": MIN_UPSAMPLE_TRIGGER,
            "UPSAMPLE_SMALL": UPSAMPLE_SMALL,
            "TUNE_THRESHOLDS": TUNE_THRESHOLDS,
            "NYST_COMPONENTS": NYST_COMPONENTS,
            "NYST_GAMMA": NYST_GAMMA,
            "SVC_C": SVC_C
        },
        "counts": {
            "train_unique": {int(k): int(v) for k,v in train_unique_counts.items()},
            "train_final": {int(k): int(v) for k,v in train_final_counts.items()},
            "test": {int(k): int(v) for k,v in test_counts.items()}
        },
        "best_params": rsearch.best_params_,
        "thresholds": thresholds
    }
    with open(OUT_META_JSON, 'w') as fmeta:
        json.dump(meta, fmeta, indent=2)
    print("Saved metadata JSON:", OUT_META_JSON)
