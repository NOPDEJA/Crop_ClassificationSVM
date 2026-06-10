"""
Kernel-approx SVM for Stage-1 (economic_crops / water / others)
- Uses Nystroem (RBF approx) + LinearSVC inside OneVsRestClassifier
- Calibrated with CalibratedClassifierCV to produce probabilities
- RandomizedSearchCV tunes n_components, gamma, and C (small search)
- Rebalanced classes to reduce others-class dominance
- Saves classification report to CSV
"""

import os
import numpy as np
import joblib
import csv
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
# Settings
# -----------------------
NPZ = "./aligned_features/svm_features_labels.npz"
RANDOM_STATE = 42
SAMPLES_PER_CLASS = 2000      # sample per LU_CODE before mapping to super-classes
MIN_CLASS_PIXELS = 200        # drop rare LU_CODEs before sampling
OUT_MODEL = "stage1_svm_nystroem.joblib"
OUT_REPORT = "stage1_svm_classification_report.csv"
N_ITER_SEARCH = 8             # number of random search iterations (small)
N_JOBS = 1                    # set to 1 to avoid nested parallelism issues

# mappings
economic_crops = {2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420}
water_code = {4101,4102,4103,4201,4202,4203}

# -----------------------
# Helper functions
# -----------------------
def load_and_sample(npz_path, samples_per_class=SAMPLES_PER_CLASS, min_pixels=MIN_CLASS_PIXELS):
    """
    Load X,y from npz, remove nodata (0,32767), filter LU codes with < min_pixels,
    sample up to samples_per_class per LU_CODE, then map sampled LU_CODEs to super-classes:
      1 = economic_crops, 2 = water, 3 = others
    Returns: Xs (n_samples, n_features), ys_super (n_samples,)
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    print(f"Loaded X shape {X.shape}, y shape {y.shape}")

    # Remove nodata
    mask_valid = (y != 0) & (y != 32767)
    Xv, yv = X[mask_valid], y[mask_valid]
    print(f"Valid labeled pixels: {Xv.shape[0]}")

    # Filter LU_CODE classes that are too rare
    uniques, counts = np.unique(yv, return_counts=True)
    keep_codes = uniques[counts >= min_pixels]
    print(f"Keeping {len(keep_codes)} LU_CODEs (>= {min_pixels} pixels)")

    keep_mask = np.isin(yv, keep_codes)
    Xf, yf = Xv[keep_mask], yv[keep_mask]

    # Sample per LU_CODE to keep class balance across many LU_CODEs
    rng = np.random.default_rng(RANDOM_STATE)
    sampled_X_parts = []
    sampled_y_parts = []
    for code in keep_codes:
        idxs = np.flatnonzero(yf == code)
        n_avail = idxs.size
        n_take = min(samples_per_class, n_avail)
        chosen = rng.choice(idxs, size=n_take, replace=False)
        sampled_X_parts.append(Xf[chosen])
        sampled_y_parts.append(yf[chosen])
        print(f"LU {int(code)}: available={n_avail}, sampled={n_take}")

    Xs = np.vstack(sampled_X_parts)
    ys_codes = np.hstack(sampled_y_parts)

    # Map LU_CODE -> super-class
    ys_super = np.array([1 if c in economic_crops else (2 if c in water_code else 3) for c in ys_codes], dtype=np.int32)

    return Xs, ys_super

def save_classification_report_csv(report, csv_path):
    """
    Save sklearn classification_report dict to CSV
    """
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'precision', 'recall', 'f1-score', 'support'])
        for label, metrics in report.items():
            if label not in ('accuracy', 'macro avg', 'weighted avg'):
                writer.writerow([label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])
        # write accuracy row
        if 'accuracy' in report:
            writer.writerow(['accuracy', report['accuracy'], report['accuracy'], report['accuracy'], ''])
        # write macro avg
        if 'macro avg' in report:
            mac = report['macro avg']
            writer.writerow(['macro avg', mac['precision'], mac['recall'], mac['f1-score'], mac['support']])
        # write weighted avg
        if 'weighted avg' in report:
            wgt = report['weighted avg']
            writer.writerow(['weighted avg', wgt['precision'], wgt['recall'], wgt['f1-score'], wgt['support']])

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    print("Loading and sampling...")
    Xs, ys = load_and_sample(NPZ)
    print("Sampled shape:", Xs.shape, ys.shape, Counter(ys))

    # ---------------------------
    # Extra balancing step (cap "others" class to max 30k)
    # ---------------------------
    rng = np.random.default_rng(RANDOM_STATE)
    idx_econ = np.where(ys == 1)[0]
    idx_water = np.where(ys == 2)[0]
    idx_others = np.where(ys == 3)[0]

    idx_others = rng.choice(idx_others, size=min(len(idx_others), 30000), replace=False)

    balanced_idx = np.concatenate([idx_econ, idx_water, idx_others])
    rng.shuffle(balanced_idx)

    Xs, ys = Xs[balanced_idx], ys[balanced_idx]
    print("After rebalancing:", Counter(ys))

    # ---------------------------
    # Train/test split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, ys, test_size=0.3, random_state=RANDOM_STATE, stratify=ys
    )
    print("Train/test sizes:", X_train.shape[0], X_test.shape[0])

    # ---------------------------
    # Pipeline: scaler -> nystroem -> linear SVC (wrapped in OneVsRest)
    # ---------------------------
    scaler = StandardScaler()
    nyst = Nystroem(kernel='rbf', random_state=RANDOM_STATE)
    svc_lin = LinearSVC(class_weight='balanced', max_iter=20000, random_state=RANDOM_STATE)
    pipe = Pipeline([('scaler', scaler), ('nyst', nyst), ('svc', svc_lin)])

    ovr = OneVsRestClassifier(pipe)

    # Calibrated classifier to produce probabilities
    calibrated = CalibratedClassifierCV(estimator=ovr, cv=3, method='sigmoid')

    # Randomized search
    param_dist = {
        'estimator__estimator__nyst__n_components': [200, 400, 600],
        'estimator__estimator__nyst__gamma': [0.5, 1.0, 2.0, 5.0],
        'estimator__estimator__svc__C': [0.01, 0.1, 1.0, 10.0]
    }

    print("Starting RandomizedSearchCV (this may take a while)...")
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

    print("RandomizedSearchCV done.")
    print("Best params:")
    print(rsearch.best_params_)

    best_clf = rsearch.best_estimator_

    # ---------------------------
    # Evaluate on holdout test set
    # ---------------------------
    print("Evaluating on holdout test set...")
    y_pred = best_clf.predict(X_test)

    class_report = classification_report(y_test, y_pred, target_names=["econ", "water", "others"], output_dict=True)
    print("\nClassification report (SVM Nystroem):")
    for k,v in class_report.items():
        print(k, v)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save report to CSV
    save_classification_report_csv(class_report, OUT_REPORT)
    print(f"Saved classification report to CSV: {OUT_REPORT}")

    # Save the full calibrated pipeline
    joblib.dump(best_clf, OUT_MODEL)
    print(f"Saved best calibrated SVM pipeline to: {OUT_MODEL}")
