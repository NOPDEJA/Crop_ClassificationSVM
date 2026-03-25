# features_importance_check.py
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split
from collections import Counter
import joblib
import json

MODEL_FILE = "stage1_weight_scale.joblib"
NPZ_FILE = "./aligned_features/svm_new_features_labels.npz"
OUT_PI_JSON = "stage1_features_importance_balanced.json"

# === Load model and data ===
print("Loading model and data...")
clf = joblib.load(MODEL_FILE)
data = np.load(NPZ_FILE)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.int32)
print(f"Loaded: X={X.shape}, y={y.shape}")

# === Filter out invalid labels ===
mask_valid = (y != 0) & (y != 32767)
X, y = X[mask_valid], y[mask_valid]
print(f"After filtering invalid labels: {X.shape}")

# === Filter out classes with too few samples ===
min_samples = 20
counts = Counter(y)
valid_classes = [cls for cls, c in counts.items() if c >= min_samples]
mask_valid_class = np.isin(y, valid_classes)
X, y = X[mask_valid_class], y[mask_valid_class]
print(f"After removing rare classes (<{min_samples}): {X.shape}, {len(valid_classes)} valid classes")

# === Cap large classes to 30,000 samples each ===
cap_per_class = 30000
X_bal, y_bal = [], []
rng = np.random.default_rng(42)
for cls in valid_classes:
    idx = np.where(y == cls)[0]
    if len(idx) > cap_per_class:
        idx = rng.choice(idx, cap_per_class, replace=False)
    X_bal.append(X[idx])
    y_bal.append(np.full(len(idx), cls, dtype=np.int32))

X_bal = np.concatenate(X_bal)
y_bal = np.concatenate(y_bal)
print(f"Balanced dataset: {X_bal.shape[0]} samples across {len(valid_classes)} LU_CODEs")

# === Map to superclasses ===
def map_gt_to_super_local(y_codes):
    economic_crops = {2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420}
    water_code = {4101,4102,4103,4201,4202,4203}
    forest_code = {3100,3101,3200,3201,3300,3301,3401,3501}
    mapped = np.array([1 if c in economic_crops else (2 if c in water_code else (3 if c in forest_code else 4))
                       for c in y_codes], dtype=np.int32)
    return mapped

y_super = map_gt_to_super_local(y_bal)

# === Create stratified validation sample ===
Xv, _, yv, _ = train_test_split(
    X_bal, y_super, test_size=0.9, stratify=y_super, random_state=42
)
print(f"Validation sample for permutation importance: {Xv.shape}")

# === Select smaller subset to speed up importance ===
n_val = min(50000, Xv.shape[0])
idx = rng.choice(Xv.shape[0], size=n_val, replace=False)
X_val = Xv[idx]
y_val = yv[idx]
print(f"Subsampled to {X_val.shape} for permutation importance")

# === Run permutation importance ===
scorer = make_scorer(f1_score, average="macro", labels=[1,2,3,4])
print("Computing permutation importance (this may take ~10–20 minutes)...")
res = permutation_importance(
    clf, X_val, y_val, scoring=scorer, n_repeats=8, random_state=42, n_jobs=4
)

# === Save results ===
out = {
    "importances_mean": res.importances_mean.tolist(),
    "importances_std": res.importances_std.tolist(),
    "n_features": X_val.shape[1]
}
with open(OUT_PI_JSON, "w") as f:
    json.dump(out, f, indent=2)

# === Print top-ranked features ===
feat_idx_sorted = np.argsort(res.importances_mean)[::-1]
print("\nTop features (index, mean, std):")
for i in feat_idx_sorted[:20]:
    print(f"Feature {i}: mean={res.importances_mean[i]:.6f}, std={res.importances_std[i]:.6f}")

print(f"\nSaved permutation importance to {OUT_PI_JSON}")
