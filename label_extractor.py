# train_hierarchical_svm.py
import os
import numpy as np
import pandas as pd
import rasterio
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib
import sys
from collections import Counter

# -----------------------------
# ========== SETTINGS =========
# -----------------------------
# Filepaths (edit if needed)
NPZ_PATH = "./aligned_features/svm_features_labels.npz"   # preferred (X,y)
LABEL_RASTER = "./label/label_47PQQ_buffered.tif"        # fallback
INDICES_FOLDER = "./indices"                             # fallback: .tif index files
CSV_REPORT = "svm_report.csv"
STATS_CSV = "class_indicator_stats.csv"
MODEL_STAGE1_PATH = "stage1_model.joblib"
SCALER_STAGE1_PATH = "stage1_scaler.joblib"
MODEL_STAGE2_PATH = "stage2_model.joblib"
SCALER_STAGE2_PATH = "stage2_scaler.joblib"

# sampling and class filtering
MIN_CLASS_PIXELS = 200        # drop LU_CODE classes with fewer pixels than this (before remap)
SAMPLES_PER_CLASS = 2000      # max samples per class for training
RANDOM_STATE = 42

# Try Linear SVM first; if you prefer RF set USE_RF=True
USE_RANDOM_FOREST_AS_FALLBACK = True

# ===============================
# ========== MAPPINGS ===========
# ===============================
economic_crops = {
    2101: "ข้าว",
    2204: "มันสำปะหลัง",
    2205: "สับปะรด",
    2302: "ยางพารา",
    2303: "ปาล์มน้ำมัน",
    2403: "ทุเรียน",
    2404: "เงาะ",
    2405: "มะพร้าว",
    2407: "มะม่วง",
    2413: "ลำไย",
    2416: "ขนุน",
    2419: "มังคุด",
    2420: "ลางสาด/ลองกอง",
}
water_code = {4101, 4102, 4103, 4201, 4202, 4203}

# -----------------------------
# ======= Helper funcs ========
# -----------------------------
def log(msg):
    print(msg)
    sys.stdout.flush()

def remap_to_superclass(y_array):
    """
    Map LU_CODEs into super-classes:
      1 = economic crops
      2 = water
      3 = others
    Returns integer array same shape as y_array.
    """
    y_remap = np.zeros_like(y_array, dtype=np.uint8)
    # vectorized approach: create set for faster checks
    econ_set = set(economic_crops.keys())
    water_set = set(water_code)
    # flatten and map
    flat = y_array.flatten()
    out = np.zeros(flat.shape, dtype=np.uint8)
    econ_mask = np.isin(flat, list(econ_set))
    water_mask = np.isin(flat, list(water_set))
    out[econ_mask] = 1
    out[water_mask] = 2
    out[(~econ_mask) & (~water_mask) & (flat != 0) & (flat != 32767)] = 3
    return out.reshape(y_array.shape)

def load_npz_or_build(npz_path=NPZ_PATH, label_raster=LABEL_RASTER, indices_folder=INDICES_FOLDER):
    """
    Preferred: load .npz with X (n_samples, n_features) and y (n_samples,)
    Fallback: try to load indices TIFFs and label raster and align them like your align script.
    """
    if os.path.exists(npz_path):
        log(f"Loading features/labels from {npz_path}")
        data = np.load(npz_path)
        X = data["X"]
        y = data["y"]
        log(f"Loaded X shape {X.shape}, y shape {y.shape}")
        return X, y
    log("NPZ not found. Attempting to build from indices folder + label raster...")

    # read label raster
    if not os.path.exists(label_raster):
        raise FileNotFoundError("Neither npz nor label raster found. Provide svm_features_labels.npz or label raster.")
    with rasterio.open(label_raster) as src:
        labels = src.read(1).astype(np.int32)
        profile = src.profile

    # read all index tifs and align by assuming same extent/proj/resolution.
    idx_files = sorted(glob(os.path.join(indices_folder, "*.tif")))
    if len(idx_files) == 0:
        raise FileNotFoundError("No index tifs found in indices folder.")

    feats = []
    for f in idx_files:
        with rasterio.open(f) as src:
            arr = src.read(1).astype(np.float32)
            # if shapes mismatch try resize with numpy (warning: crude)
            if arr.shape != labels.shape:
                log(f"Warning: index {f} shape {arr.shape} != label shape {labels.shape}. Attempting resize (may distort).")
                arr = np.resize(arr, labels.shape)
            feats.append(arr)
    stack = np.stack(feats, axis=-1)   # (H, W, n_features)
    X = stack.reshape(-1, stack.shape[-1])
    y = labels.flatten()
    # mask out no data? keep y==0 for now; later we will remove
    log(f"Built X shape {X.shape}, y shape {y.shape}")
    return X, y

def filter_and_sample(X, y, min_class_pixels=MIN_CLASS_PIXELS, samples_per_class=SAMPLES_PER_CLASS, random_state=RANDOM_STATE):
    """Filter rare classes, remove nodata, sample balanced sets per class"""
    # remove nodata
    mask_valid = (y != 0) & (y != 32767)
    Xv, yv = X[mask_valid], y[mask_valid]
    # count per LU_CODE
    uniques, counts = np.unique(yv, return_counts=True)
    # keep LU_CODEs with counts >= min_class_pixels
    keep_codes = uniques[counts >= min_class_pixels]
    keep_mask = np.isin(yv, keep_codes)
    Xf, yf = Xv[keep_mask], yv[keep_mask]
    log(f"Classes kept (LU_CODE count): {len(keep_codes)} (min pixels {min_class_pixels})")
    # sample per class
    sampled_X_parts = []
    sampled_y_parts = []
    rng = np.random.default_rng(random_state)
    for code in keep_codes:
        idxs = np.flatnonzero(yf == code)
        n_available = len(idxs)
        n_take = min(samples_per_class, n_available)
        chosen = rng.choice(idxs, size=n_take, replace=False)
        sampled_X_parts.append(Xf[chosen])
        sampled_y_parts.append(yf[chosen])
        log(f"LU {code}: available={n_available}, sampled={n_take}")
    Xs = np.vstack(sampled_X_parts)
    ys = np.hstack(sampled_y_parts)
    # shuffle
    perm = rng.permutation(len(ys))
    return Xs[perm], ys[perm]

def compute_indicator_stats_and_save(X_full, y_full, out_csv=STATS_CSV, feature_names=None):
    """
    Compute per-LU_CODE and per-superclass statistics (count, mean, median) for each feature index.
    Saves as CSV for inspection.
    """
    log("Computing indicator stats per LU_CODE and per super-class...")
    mask_valid = (y_full != 0) & (y_full != 32767)
    Xv, yv = X_full[mask_valid], y_full[mask_valid]
    # per LU_CODE
    rows = []
    uniques = np.unique(yv)
    for code in uniques:
        idxs = np.where(yv == code)[0]
        sub = Xv[idxs]
        row = {"LU_CODE": int(code), "Category": ("economic" if code in economic_crops else ("water" if code in water_code else "other")), "Pixel_Count": len(idxs)}
        for fi in range(sub.shape[1]):
            colname_mean = f"f{fi}_mean" if feature_names is None else f"{feature_names[fi]}_mean"
            colname_med = f"f{fi}_median" if feature_names is None else f"{feature_names[fi]}_median"
            row[colname_mean] = float(np.nanmean(sub[:, fi]))
            row[colname_med] = float(np.nanmedian(sub[:, fi]))
        rows.append(row)
    df = pd.DataFrame(rows)
    # per super-class aggregated
    agg_rows = []
    for scode, name in [(1, "economic_crops"), (2, "water"), (3, "others")]:
        if scode == 1:
            mask_sc = np.isin(yv, list(economic_crops.keys()))
        elif scode == 2:
            mask_sc = np.isin(yv, list(water_code))
        else:
            # others = everything else (excluding nodata)
            mask_sc = (~np.isin(yv, list(economic_crops.keys()))) & (~np.isin(yv, list(water_code)))
        if np.any(mask_sc):
            sub = Xv[mask_sc]
            row = {"LU_CODE": scode, "Category": name, "Pixel_Count": int(sub.shape[0])}
            for fi in range(sub.shape[1]):
                colname_mean = f"f{fi}_mean" if feature_names is None else f"{feature_names[fi]}_mean"
                colname_med = f"f{fi}_median" if feature_names is None else f"{feature_names[fi]}_median"
                row[colname_mean] = float(np.nanmean(sub[:, fi]))
                row[colname_med] = float(np.nanmedian(sub[:, fi]))
            agg_rows.append(row)
    df2 = pd.DataFrame(agg_rows)
    df_all = pd.concat([df, df2], ignore_index=True, sort=False)
    df_all.to_csv(out_csv, index=False, encoding="utf-8-sig")
    log(f"Indicator stats saved to {out_csv}")

def save_classification_csv(y_true, y_pred, class_names, csv_path, stage_label):
    """Saves classification_report as CSV rows (one row per class)"""
    cr_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df = pd.DataFrame(cr_dict).transpose()
    df.insert(0, "Stage", stage_label)
    # Append mode; write header only if file doesn't exist
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=True, encoding="utf-8-sig")
    log(f"{stage_label}: metrics appended to {csv_path}")

# -----------------------------
# ======= TRAINING STEP 1 =====
# -----------------------------
def train_stage1(X_full, y_full, csv_path=CSV_REPORT):
    log("=== Stage 1: Broad-class training (economic_crops, water, others) ===")
    # compute indicator stats first
    compute_indicator_stats_and_save(X_full, y_full, out_csv=STATS_CSV)

    # remap
    # We will remap y_full (flat labels) into 1/2/3
    # If y_full is flat (n_samples,), convert to 2D dummy for remap function compatibility
    # (we only need vector)
    y_flat = y_full.copy()
    mask = (y_flat != 0) & (y_flat != 32767)
    if not np.any(mask):
        raise RuntimeError("No valid labeled pixels found after removing nodata.")
    # For training we will sample by LU_CODE first (original LU codes) to preserve representativeness,
    # then map sampled labels to 3-class labels.
    Xs, ys = filter_and_sample(X_full, y_flat)
    # Map sampled ys to superclasses
    ys_super = np.array([1 if v in economic_crops else (2 if v in water_code else 3) for v in ys])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, ys_super, test_size=0.3, random_state=RANDOM_STATE, stratify=ys_super
    )
    log(f"Stage1 sample sizes: train={len(y_train)}, test={len(y_test)}; class dist train={Counter(y_train)}")

    # pipeline: scaler + classifier
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Linear SVC with balanced weights
    clf = OneVsRestClassifier(LinearSVC(max_iter=10000, class_weight='balanced', random_state=RANDOM_STATE))
    try:
        log("Training LinearSVC (Stage 1)...")
        clf.fit(X_train_s, y_train)
    except Exception as e:
        log(f"LinearSVC training failed: {e}")
        if USE_RANDOM_FOREST_AS_FALLBACK:
            log("Falling back to RandomForestClassifier for Stage 1.")
            rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE, class_weight='balanced'))
            rf.fit(X_train_s, y_train)
            clf = rf
        else:
            raise

    # Evaluate and save CSV
    y_pred = clf.predict(X_test_s)
    class_names = ["economic_crops", "water", "others"]
    save_classification_csv(y_test, y_pred, class_names, csv_path, "Stage 1")

    # Save model + scaler
    joblib.dump(clf, MODEL_STAGE1_PATH)
    joblib.dump(scaler, SCALER_STAGE1_PATH)
    log(f"Stage1 model saved to {MODEL_STAGE1_PATH} and scaler to {SCALER_STAGE1_PATH}")

    return clf, scaler

# -----------------------------
# ======= TRAINING STEP 2 =====
# -----------------------------
def train_stage2(X_full, y_full, stage1_clf, stage1_scaler, csv_path=CSV_REPORT):
    log("=== Stage 2: Crop-type training inside predicted economic_crops ===")
    # remove nodata first
    mask_valid = (y_full != 0) & (y_full != 32767)
    Xv, yv = X_full[mask_valid], y_full[mask_valid]

    # Predict Stage1 classes for all valid pixels
    Xv_scaled = stage1_scaler.transform(Xv)
    y_stage1_pred = stage1_clf.predict(Xv_scaled)

    # keep only pixels where stage1_pred == economic_crops (1)
    mask_crops_pred = (y_stage1_pred == 1)
    X_crops_pred = Xv[mask_crops_pred]
    y_crops_true = yv[mask_crops_pred]   # these are LU_CODEs (original codes)

    # keep only LU_CODES that are in your economic_crops mapping
    econ_codes = np.array(list(economic_crops.keys()))
    keep_mask = np.isin(y_crops_true, econ_codes)
    X_crops = X_crops_pred[keep_mask]
    y_crops = y_crops_true[keep_mask]

    if len(y_crops) == 0:
        log("Stage 2: no crop pixels after stage1 prediction. Skipping stage2.")
        return None, None

    log(f"Stage2: Found {len(y_crops)} pixels predicted as economic crops (before sampling). Unique crop codes: {np.unique(y_crops)}")

    # Filter and sample per LU_CODE for stage2 (use same helper but tweak threshold)
    Xs_parts, ys_parts = [], []
    rng = np.random.default_rng(RANDOM_STATE)
    unique_codes = np.unique(y_crops)
    for code in unique_codes:
        idxs = np.flatnonzero(y_crops == code)
        n_available = len(idxs)
        n_take = min(SAMPLES_PER_CLASS, n_available)
        chosen = rng.choice(idxs, size=n_take, replace=False)
        Xs_parts.append(X_crops[chosen])
        ys_parts.append(y_crops[chosen])
        log(f"Stage2 LU {code}: available={n_available}, sampled={n_take}")
    Xs2 = np.vstack(Xs_parts)
    ys2 = np.hstack(ys_parts)
    # optionally map ys2 to themselves (they are LU_CODE ints). We'll train multiclass on LU_CODEs.

    # train/test split (stratify by LU_CODE)
    X_train, X_test, y_train, y_test = train_test_split(
        Xs2, ys2, test_size=0.3, random_state=RANDOM_STATE, stratify=ys2
    )

    log(f"Stage2 train/test sizes: {len(y_train)}, {len(y_test)}")

    # scaler and classifier
    scaler2 = StandardScaler()
    X_train_s = scaler2.fit_transform(X_train)
    X_test_s = scaler2.transform(X_test)

    clf2 = OneVsRestClassifier(LinearSVC(max_iter=10000, class_weight='balanced', random_state=RANDOM_STATE))
    try:
        log("Training LinearSVC (Stage2)...")
        clf2.fit(X_train_s, y_train)
    except Exception as e:
        log(f"Stage2 LinearSVC failed: {e}")
        if USE_RANDOM_FOREST_AS_FALLBACK:
            log("Falling back to RandomForestClassifier for Stage2.")
            rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE, class_weight='balanced'))
            rf.fit(X_train_s, y_train)
            clf2 = rf
        else:
            raise

    # evaluate & CSV
    # class names: use the Thai crop names (preserve order)
    unique_test_codes = np.unique(y_test)
    class_names = [economic_crops.get(int(c), str(c)) for c in unique_test_codes]
    save_classification_csv(y_test, clf2.predict(X_test_s), class_names, csv_path, "Stage 2")

    # save stage2 model/scaler
    joblib.dump(clf2, MODEL_STAGE2_PATH)
    joblib.dump(scaler2, SCALER_STAGE2_PATH)
    log(f"Stage2 model/scaler saved to {MODEL_STAGE2_PATH} / {SCALER_STAGE2_PATH}")

    return clf2, scaler2

# -----------------------------
# ====== PREDICT RASTER =======
# -----------------------------
def predict_raster(input_multi_band_tif, model, scaler, output_tif="classified_stage1.tif", block_size=1024):
    """
    Classify input multi-band raster into output raster using model+scaler.
    Works in blocks to avoid memory explosion.
    model expects features in shape (n_pixels, n_bands), scaler is StandardScaler.
    """
    with rasterio.open(input_multi_band_tif) as src:
        profile = src.profile.copy()
        n_bands = src.count
        profile.update(count=1, dtype=rasterio.uint8, compress='lzw')
        width, height = src.width, src.height

        with rasterio.open(output_tif, "w", **profile) as dst:
            # iterate by windows
            for y_off in range(0, height, block_size):
                h = min(block_size, height - y_off)
                for x_off in range(0, width, block_size):
                    w = min(block_size, width - x_off)
                    window = rasterio.windows.Window(x_off, y_off, w, h)
                    data = src.read(window=window)  # shape (bands, h, w)
                    # reshape to (h*w, bands)
                    arr = np.moveaxis(data, 0, -1).reshape(-1, n_bands)
                    # handle nodata / mask if needed
                    if scaler is not None:
                        arr_s = scaler.transform(arr)
                    else:
                        arr_s = arr
                    preds = model.predict(arr_s)
                    out = preds.reshape(h, w).astype(np.uint8)
                    dst.write(out, 1, window=window)
    log(f"Raster prediction saved to {output_tif}")

# -----------------------------
# ======= MAIN ENTRY ==========
# -----------------------------
if __name__ == "__main__":
    log("Starting hierarchical SVM training pipeline...")
    X_all, y_all = load_npz_or_build()
    # Train stage1
    stage1_clf, stage1_scaler = train_stage1(X_all, y_all, csv_path=CSV_REPORT)
    # Train stage2
    stage2_clf, stage2_scaler = train_stage2(X_all, y_all, stage1_clf, stage1_scaler, csv_path=CSV_REPORT)

    # Optional: predict a full tile (example using one of your S2 multi-band tiles)
    example_tile = "./S2_data/47PQQ_2018-10-31.tif"
    if os.path.exists(example_tile):
        predict_raster(example_tile, stage1_clf, stage1_scaler, output_tif="stage1_predicted.tif")
    log("Pipeline finished.")
