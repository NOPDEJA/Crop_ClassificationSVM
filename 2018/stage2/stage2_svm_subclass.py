# stage2_svm_subclass.py  (updated for subclass/hierarchical Stage-2)
"""
Stage-2 (hierarchical subclass classifier)
 - Input: Stage-1 predictions + X,y features
 - Task: map economic pixels (stage1_pred == 1) to one of:
     1 = orchards (permanent fruit-tree crops)
     2 = plantation_tree_crops (managed plantations)
     3 = field_non_orchard (annual/field crops)
     4 = other_econ (fallback for econ LU codes not in above groups)
 - Output: model, report, per-group indicator stats, test probs/preds   , metadata

Notes:
 - Stage-3 (not implemented here) should do fine-grained crop classification within each subclass.
 - Defaults for STAGE1_PRED / STAGE1_MODEL align with your Stage-1 script outputs.
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# -----------------------
# Config - edit if needed
# -----------------------
# Use same NPZ that Stage-1 used (updated to match your Stage-1 script)
NPZ = "./aligned_features/svm_new_features_labels.npz"

# defaults aligned with your Stage-1 outputs (from the script you ran earlier)
STAGE1_PRED = "stage1_pred_new_features.npy"
STAGE1_MODEL = "stage1_svm_pred_chunk_new_superclass.joblib"
STAGE1_CHUNK = 2_000_000

RANDOM_STATE = 42

# Economic LU codes set (from your previous lists)
economic_crops = {2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420}

# ---------- Subclass grouping (edit if you want to change membership) ----------
# Orchards (permanent fruit-tree crops)
orchards_codes = {2403, 2404, 2407, 2420, 2416, 2419}  # Durian(2403), Rambutan(2404), Mango(2407), Longan/Langsat(2420), Jackfruit(2416), Mangosteen(2419)

# Plantation tree crops (managed plantations)
plantation_codes = {2302, 2303, 2405}  # Para rubber(2302), Oil palm(2303), Coconut(2405)

# Field / non-orchard crops
field_codes = {2101, 2204, 2205}  # Active paddy (2101), Cassava(2204), Pineapple(2205)

# Label mapping for subclasses: 1=orchards, 2=plantation, 3=field, 4=other_econ
SUBCLASS_LABELS = { "orchards": 1, "plantation": 2, "field": 3, "other_econ": 4 }

# per-group caps & sampling (rename semantics: per-subclass/group)
MIN_PIXELS_PER_GROUP = 100
PER_GROUP_CAP = 24000      # unique samples to keep per group before split
TARGET_PER_GROUP = 6000    # desired minimum train size (only enforced for small groups)
UPSAMPLE_SMALL = True
CONDITIONAL_UPSAMPLE = True
MIN_UPSAMPLE_TRIGGER = 0.67  # fraction of target above which we lightly upsample (else full)
TUNE_THRESHOLDS = False
THRESH_GRID = np.linspace(0.2, 0.95, 40)

# Search / pipelines
NYST_COMPONENTS = [400, 600]
NYST_GAMMA = [0.5, 1.0]
SVC_C = [0.1, 1.0, 10.0]
N_ITER_SEARCH = 8
N_JOBS = 4

# Outputs
OUT_MODEL = "stage2_subclass_model.joblib"
OUT_REPORT = "stage2_subclass_report.csv"
OUT_STATS_LU = "stage2_indicator_stats_per_lu.csv"
OUT_STATS_GROUP = "stage2_indicator_stats_per_group.csv"
OUT_TEST_PROB = "stage2_test_prob.npy"
OUT_TEST_PRED = "stage2_test_pred.npy"
OUT_META_JSON = "stage2_subclass_meta.json"

# -----------------------
# Helpers
# -----------------------
def ensure_stage1_pred(npz_path, pred_npy=STAGE1_PRED, model_path=STAGE1_MODEL, chunk_size=STAGE1_CHUNK):
    """
    Return Stage-1 predictions array. If pred_npy exists, load it.
    Else, if model_path exists, generate predictions in chunks and save to pred_npy.
    """
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
        e = min(n, s+chunk_size)
        preds[s:e] = model.predict(X_all[s:e]).astype(np.uint8)
        print(f"  chunk {s}:{e}")
    np.save(pred_npy, preds)
    print("Saved Stage-1 predictions to", pred_npy)
    return preds

def load_Xy(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.int32)

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
    # y_group are subclass labels (1..4)
    rows = []
    groups = np.unique(y_group)
    for g in sorted(groups):
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

def map_econ_to_subclass_array(y_lu_codes):
    """
    Map LU codes (only economic_crops passed here) to subclass labels:
     - orchards -> 1
     - plantation -> 2
     - field -> 3
     - other_econ -> 4
    """
    mapped = np.full_like(y_lu_codes, fill_value=SUBCLASS_LABELS["other_econ"], dtype=np.int32)
    for i, c in enumerate(y_lu_codes):
        if c in orchards_codes:
            mapped[i] = SUBCLASS_LABELS["orchards"]
        elif c in plantation_codes:
            mapped[i] = SUBCLASS_LABELS["plantation"]
        elif c in field_codes:
            mapped[i] = SUBCLASS_LABELS["field"]
        else:
            mapped[i] = SUBCLASS_LABELS["other_econ"]
    return mapped

def cap_per_group(X, y_group, cap=PER_GROUP_CAP):
    """
    Cap number of samples per subclass/group to `cap`.
    """
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

def sample_per_group_train_only(X_train, y_train, target_per_group=TARGET_PER_GROUP, upsample=UPSAMPLE_SMALL,
                                conditional=CONDITIONAL_UPSAMPLE, min_trigger=MIN_UPSAMPLE_TRIGGER):
    """
    Same logic as your previous sample_per_crop_train_only but applied to subclass labels.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    groups = np.unique(y_train)
    partsX, partsy = [], []
    for c in groups:
        idxs = np.flatnonzero(y_train == c)
        n = idxs.size
        tgt = int(target_per_group)
        effective_target = tgt
        if conditional:
            if n >= tgt:
                effective_target = n  # keep full data for large group
            elif n >= min_trigger * tgt:
                effective_target = tgt
            else:
                effective_target = tgt
        else:
            effective_target = tgt
        if n >= effective_target:
            chosen = rng.choice(idxs, size=effective_target, replace=False)
        else:
            chosen = rng.choice(idxs, size=effective_target, replace=True) if upsample else idxs
        partsX.append(X_train[chosen])
        partsy.append(y_train[chosen])
        print(f"Group {int(c)}: train n={n} -> final={len(chosen)} target_policy={effective_target}")
    Xb = np.vstack(partsX); yb = np.hstack(partsy)
    perm = rng.permutation(Xb.shape[0])
    return Xb[perm], yb[perm]

def save_report_with_traininfo(report_dict, train_unique_counts, train_final_counts, test_counts, out_csv):
    """
    report_dict from classification_report(output_dict=True)
    For Stage-2 subclass classification, indices are subclass labels (strings) or summary rows.
    Attempt to convert index to int and attach counts.
    """
    df = pd.DataFrame(report_dict).transpose()
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

def chunked_predict_and_save(model, npz_path, out_prob=OUT_TEST_PROB, out_pred=OUT_TEST_PRED, chunk_size=STAGE1_CHUNK):
    """
    Predict probabilities/predictions on the economic subset X (not the full dataset).
    This helper is left here should you want to predict entire NPZ in chunks with model.predict_proba.
    (Not used directly in main flow because Stage-2 works on filtered econ subset.)
    """
    data = np.load(npz_path, allow_pickle=True)
    X_all = data["X"].astype(np.float32)
    n = X_all.shape[0]
    probs = np.zeros((n, model.classes_.shape[0]), dtype=np.float32)
    preds = np.zeros(n, dtype=np.int32)
    print(f"Starting chunked prediction with chunk_size={chunk_size}.")
    for s in range(0, n, chunk_size):
        e = min(n, s+chunk_size)
        Xc = X_all[s:e]
        preds[s:e] = model.predict(Xc).astype(np.int32)
        probs[s:e, :] = model.predict_proba(Xc).astype(np.float32)
        print(f"  predicted chunk {s}:{e} ({e-s} rows)")
    np.save(out_pred, preds)
    np.save(out_prob, probs)
    print("Saved preds/probs:", out_pred, out_prob)

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    print("=== Stage-2 (subclass classifier) ===")
    # get stage-1 predictions (loads or generates)
    stage1_pred = ensure_stage1_pred(NPZ, STAGE1_PRED, STAGE1_MODEL, chunk_size=STAGE1_CHUNK)

    X_all, y_all = load_Xy(NPZ)
    print("Loaded X,y:", X_all.shape, y_all.shape)

    # Filter to pixels predicted econ by Stage-1 (class=1). This excludes Stage-1 forest (4).
    mask_stage1 = (stage1_pred == 1)
    X_stage1 = X_all[mask_stage1]
    y_stage1 = y_all[mask_stage1]
    print("Pixels predicted econ by Stage-1:", X_stage1.shape[0])

    # Keep only those with true LU_CODE in your economic_crops set
    mask_true_econ = np.isin(y_stage1, list(economic_crops))
    X_econ = X_stage1[mask_true_econ]
    y_econ = y_stage1[mask_true_econ]
    print("Pixels both predicted econ and true econ LU codes:", X_econ.shape[0])

    # Save per-LU indicator stats (useful later for per-crop Stage-3)
    compute_indicator_stats_per_lu(X_econ, y_econ, sorted(economic_crops), OUT_STATS_LU)

    # Map economic LU codes -> subclass label (1..4)
    y_sub = map_econ_to_subclass_array(y_econ)
    print("Subclass distribution (raw):", Counter(y_sub))

    # Save per-subclass indicator stats
    compute_indicator_stats_per_group(X_econ, y_sub, OUT_STATS_GROUP)

    # Filter groups with too few pixels (optional)
    uniques, counts = np.unique(y_sub, return_counts=True)
    keep_groups = uniques[counts >= MIN_PIXELS_PER_GROUP]
    print("Groups with >= min pixels:", keep_groups)
    mask_keep = np.isin(y_sub, keep_groups)
    X_keep = X_econ[mask_keep]
    y_keep = y_sub[mask_keep]
    print("After filtering small groups:", X_keep.shape, Counter(y_keep))

    # cap per-group (keeps diversity)
    Xc, yc = cap_per_group(X_keep, y_keep, cap=PER_GROUP_CAP)

    # split train / val / test: 70 / 15 / 15
    X_tr, X_rest, y_tr, y_rest = train_test_split(Xc, yc, test_size=0.3, stratify=yc, random_state=RANDOM_STATE)
    X_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE)
    print("Split sizes (train, val, test):", X_tr.shape[0], X_val.shape[0], X_te.shape[0])
    print("Train dist (pre-balance):", Counter(y_tr))

    train_unique_counts = dict(Counter(y_tr))
    # conditional upsample train-only per group
    X_tr_bal, y_tr_bal = sample_per_group_train_only(
        X_tr, y_tr,
        target_per_group=TARGET_PER_GROUP,
        upsample=UPSAMPLE_SMALL,
        conditional=CONDITIONAL_UPSAMPLE,
        min_trigger=MIN_UPSAMPLE_TRIGGER
    )
    train_final_counts = dict(Counter(y_tr_bal))
    print("Train dist (final balanced):", train_final_counts)
    too_small = [c for c, n in train_final_counts.items() if n < 3]
    if too_small:
        raise RuntimeError(f"Classes too small for 3-fold CV: {too_small}")

    # pipeline (same pattern as Stage-1)
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

    print("Starting RandomizedSearchCV (stage2 subclass)...")
    rsearch = RandomizedSearchCV(calibrated, param_distributions=param_dist, n_iter=N_ITER_SEARCH, cv=3,
                                 random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=2)
    rsearch.fit(X_tr_bal, y_tr_bal)
    print("Best params:", rsearch.best_params_)

    best_clf = rsearch.best_estimator_
    group_classes = best_clf.classes_
    print("Group classes (model):", group_classes)

    # Optional per-group threshold tuning on validation set
    thresholds = {int(c): None for c in group_classes}
    if TUNE_THRESHOLDS:
        print("Threshold tuning enabled ...")
        val_probs = best_clf.predict_proba(X_val)
        base_val = group_classes[np.argmax(val_probs, axis=1)]
        idx_map = {c:i for i, c in enumerate(group_classes)}
        for cls in group_classes:
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
        base_test = group_classes[np.argmax(test_probs, axis=1)]
        idx_map = {c:i for i, c in enumerate(group_classes)}
        test_pred = base_test.copy()
        for cls, t in thresholds.items():
            if t is None: continue
            col = idx_map[cls]
            mask = test_probs[:, col] > t
            test_pred[mask] = cls
    else:
        test_pred = group_classes[np.argmax(test_probs, axis=1)]
    np.save(OUT_TEST_PRED, test_pred.astype(np.int32))
    print("Saved test prob/pred arrays.")

    report = classification_report(y_te, test_pred, output_dict=True)
    test_counts = dict(Counter(y_te))
    save_report_with_traininfo(report, train_unique_counts, train_final_counts, test_counts, OUT_REPORT)

    print("Confusion matrix:")
    print(confusion_matrix(y_te, test_pred))

    joblib.dump(best_clf, OUT_MODEL)
    print("Saved Stage-2 subclass model to", OUT_MODEL)

    # Metadata JSON (includes grouping used)
    meta = {
        "stage": 2,
        "subclasses": {
            "orchards": sorted([int(x) for x in orchards_codes]),
            "plantation": sorted([int(x) for x in plantation_codes]),
            "field_non_orchard": sorted([int(x) for x in field_codes]),
            "other_econ": "all remaining economic_crops not in above sets"
        },
        "config": {
            "PER_GROUP_CAP": PER_GROUP_CAP,
            "TARGET_PER_GROUP": TARGET_PER_GROUP,
            "CONDITIONAL_UPSAMPLE": CONDITIONAL_UPSAMPLE,
            "MIN_UPSAMPLE_TRIGGER": MIN_UPSAMPLE_TRIGGER,
            "UPSAMPLE_SMALL": UPSAMPLE_SMALL,
            "TUNE_THRESHOLDS": TUNE_THRESHOLDS,
            "NYST_COMPONENTS": NYST_COMPONENTS,
            "NYST_GAMMA": NYST_GAMMA,
            "SVC_C": SVC_C
        },
        "counts": {
            "train_unique": {int(k): int(v) for k, v in train_unique_counts.items()},
            "train_final": {int(k): int(v) for k, v in train_final_counts.items()},
            "test": {int(k): int(v) for k, v in test_counts.items()}
        },
        "best_params": rsearch.best_params_,
        "thresholds": thresholds
    }
    with open(OUT_META_JSON, 'w') as fmeta:
        json.dump(meta, fmeta, indent=2)
    print("Saved metadata JSON:", OUT_META_JSON)

    print("Stage-2 (subclass) complete.")
